"""Pareto front of retrofit packages (LLD 4.2): which packages are not beaten on cost, carbon, bills and score.

1. Candidates: feasible / with-conditions retrofits that change energy and have a sourced cost (unsourced costs are
   excluded from cost ranking, compliance fix 2). Subsets respect alternatives (one glazing, one main heating
   system, catalogue `excludes`). At most 256 packages: if there are more, the items with the weakest median
   lifetime net benefit are left out first (R-14).
2. Every package is ONE twin run (batched), anchored to the meters like the ladder.
3. Objectives per posterior sample: net cost after confirmed grants (min), carbon saving (max), bill saving (max),
   score gain (max). Cost samples are paired with twin samples by index (independent draws).
   Score gain uses the projected score's point value per package (the scorer returns a distribution, not samples),
   so it is fixed across samples in the dominance check (R-15).
4. A non-dominated sort per sample gives P(on front). The knee (R-16) is read off the 2-D median trade-off of
   net cost vs the goal's benefit (bill or carbon saving): of the packages not beaten on those two, the one
   furthest above the straight line from the cheapest to the biggest-benefit package (both axes scaled to
   [0, 1]), i.e. where extra spend stops buying much. With 4 objectives almost everything is on the front, so a
   4-D distance-to-ideal knee just picked the largest package.
Value is never an objective. Baselines for evaluation: greedy kWh-per-GBP and a CO2-only optimiser under the same
budget, compared by hypervolume (R-25). Modes: budget filter, reference point with hard targets (R-23), and a
memory-learned re-ranking that orders front packages without adding or removing any (R-24).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from contracts.models import AreaEPCStats, FeasibilityReport, ParetoPoint, Property, RetrofitAction
from contracts.summarise import summarise
from greentwin.interfaces import PosteriorSamples
from greentwin.retrofit import catalogue as cat
from greentwin.retrofit.ladder import Goal, _exclusive, select
from greentwin.retrofit.property_edits import property_after
from greentwin.retrofit.rating import rating_from_runs
from greentwin.retrofit.runner import Meter, anchored, breakdown, fuel_supported, run_packages

MAX_PACKAGES = 256


def _valid(pkg: Sequence[str]) -> bool:
    return all(not _exclusive(a, b) for a, b in combinations(pkg, 2))


def enumerate_packages(items: Sequence[str], limit: int = MAX_PACKAGES) -> list[tuple[str, ...]]:
    """All valid subsets (including the empty package), in ladder order inside each package."""
    items = sorted(items, key=cat.sequence_position)
    out: list[tuple[str, ...]] = []
    for k in range(len(items) + 1):
        for c in combinations(items, k):
            if _valid(c):
                out.append(c)
                if len(out) > limit:
                    raise OverflowError
    return out


def non_dominated(F: np.ndarray) -> np.ndarray:
    """F: (packages, objectives), all minimised. Boolean mask of the non-dominated rows."""
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dominated = (le & lt).any(0)  # row j is dominated if some i <= j everywhere and < somewhere
    return ~dominated


@dataclass
class ParetoResult:
    points: list[ParetoPoint]
    packages: list[tuple[str, ...]]
    medians: np.ndarray                       # (packages, 4): net cost, carbon saving, bill saving, score gain
    p_on_front: np.ndarray
    excluded: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    knees: dict[str, Optional[tuple[str, ...]]] = field(default_factory=dict)  # goal -> knee package
    samples: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # (cost, carbon, bill), each (packages, n)


def pareto(prop: Property, samples: PosteriorSamples, feasibility: FeasibilityReport, meter: Optional[Meter] = None,
           scorer=None, peer: Optional[AreaEPCStats] = None, budget_gbp: Optional[float] = None,
           goal: Goal = "bills", seed: int = 42) -> ParetoResult:
    if not fuel_supported(prop):
        return ParetoResult([], [], np.zeros((0, 4)), np.zeros(0),
                            warnings=[f"Main fuel {prop.heating.main_fuel}: no sourced price or factor; no Pareto front"])
    sel = select(prop, samples, feasibility, goal, meter, seed)
    n, off_gas = samples.n_samples, prop.constraints.off_gas_grid
    excluded: dict[str, str] = {}
    pool = []
    ok = {f.retrofit_id for f in feasibility.items if f.status in ("feasible", "feasible_with_conditions")}
    for rid in sorted(ok):
        item = cat.get(rid)
        if not item.changes_kwh:
            excluded[rid] = "Changes bills, not energy"
        elif not cat.cost_samples(rid, 8, seed).available:
            excluded[rid] = "No sourced cost: excluded from cost ranking"
        else:
            pool.append(rid)
    # weakest first out until the package count fits (R-14)
    life_key = lambda r: sel.single_saving.get(r, 0.0) * (cat.get(r).lifetime_years or 1) - float(  # noqa: E731
        np.median(cat.cost_samples(r, 256, seed, off_gas_grid=off_gas).net_gbp))
    while True:
        try:
            packages = enumerate_packages(pool)
            break
        except OverflowError:
            weakest = min(pool, key=life_key)
            pool.remove(weakest)
            excluded[weakest] = f"Left out to keep the search under {MAX_PACKAGES} packages (weakest net benefit)"

    acts = [[RetrofitAction(retrofit_id=r) for r in p] for p in packages]
    runs = run_packages(samples, acts)
    energies = [anchored(r, runs[0], meter) for r in runs]
    base = energies[0]
    costs = {r: cat.cost_samples(r, n, seed, off_gas_grid=off_gas).net_gbp for r in pool}

    cost = np.stack([sum((costs[r] for r in p), np.zeros(n)) for p in packages])
    carbon = np.stack([base.carbon() - e.carbon() for e in energies])
    bill = np.stack([base.bill() - e.bill() for e in energies])

    score_gain = np.zeros(len(packages))
    score_dists = [None] * len(packages)
    if scorer is not None:
        std = run_packages(samples, acts, mode="standard")
        s0 = scorer.score(prop, breakdown(base, prop.floor_area_m2, meter is not None), None, peer).score
        for i, p in enumerate(packages):
            if not p:
                continue
            rating = rating_from_runs(prop, std[0], std[i]).estimate
            res = scorer.score(property_after(prop, acts[i]), breakdown(energies[i], prop.floor_area_m2,
                                                                       meter is not None), rating, peer)
            score_gain[i] = res.score - s0
            d = res.score_distribution
            score_dists[i] = (d, s0)

    keep = np.ones(len(packages), bool)
    if budget_gbp is not None:
        keep = np.median(cost, axis=1) <= budget_gbp
    idx = np.flatnonzero(keep)

    on = np.zeros(len(packages))
    for s in range(n):
        F = np.column_stack([cost[idx, s], -carbon[idx, s], -bill[idx, s], -score_gain[idx]])
        on[idx] += non_dominated(F)
    p_on = on / n

    med = np.column_stack([np.median(cost, 1), np.median(carbon, 1), np.median(bill, 1), score_gain])
    Fm = np.column_stack([med[idx, 0], -med[idx, 1], -med[idx, 2], -med[idx, 3]])
    front = idx[non_dominated(Fm)]
    knee = _knee(med, idx, goal)
    knees = {g: (packages[k] if (k := _knee(med, idx, g)) is not None else None) for g in ("bills", "carbon")}

    points = []
    for i in sorted(set(np.flatnonzero(p_on > 0)) | set(front.tolist()), key=lambda i: med[i, 0]):
        if score_dists[i] is not None:
            d, s0 = score_dists[i]
            sg = summarise(np.asarray(_shifted(d, s0, n, seed)), "points")
        else:
            sg = summarise(np.full(n, score_gain[i]), "points")
        points.append(ParetoPoint(
            package=list(packages[i]),
            net_cost_gbp=summarise(cost[i], "GBP", worst_tail="high"),
            carbon_saving_kgco2e=summarise(carbon[i], "kgCO2e/yr"),
            bill_saving_gbp=summarise(bill[i], "GBP/yr", tails=[("bill rises", 0.0, "below")]),
            score_gain=sg, p_on_front=round(float(p_on[i]), 4), is_knee=(i == knee)))
    warnings = [f"{cat.get(r).name}: {why}" for r, why in excluded.items()]
    return ParetoResult(points, packages, med, p_on, excluded, warnings, knees, (cost, carbon, bill))


def _knee(med: np.ndarray, idx: np.ndarray, goal: Goal) -> Optional[int]:
    col = 2 if goal == "bills" else 1
    F2 = np.column_stack([med[idx, 0], -med[idx, col]])
    front = idx[non_dominated(F2)]
    if len(front) < 3:
        return None
    x, y = med[front, 0], med[front, col]
    xs = (x - x.min()) / max(np.ptp(x), 1e-9)
    ys = (y - y.min()) / max(np.ptp(y), 1e-9)
    a, b = np.argmin(xs), np.argmax(ys)
    dx, dy = xs[b] - xs[a], ys[b] - ys[a]
    dist = (dx * (ys - ys[a]) - dy * (xs - xs[a])) / max(np.hypot(dx, dy), 1e-9)  # > 0: above the chord
    k = int(np.argmax(dist))
    return int(front[k]) if dist[k] > 0 else None


def _shifted(d, s0: float, n: int, seed: int) -> np.ndarray:
    """Score-gain samples: resampled projected score minus the current point score."""
    from greentwin.score.sampling import from_distribution

    return from_distribution(d, n, np.random.default_rng(seed)) - s0


def greedy_baseline(result: ParetoResult, budget_gbp: float) -> tuple[tuple[str, ...], bool]:
    """Greedy kWh-per-GBP stand-in using the same runs: of the single-item packages, add items by median bill saving
    per GBP while the package stays valid and within budget; then report whether the median front dominates it."""
    singles = {p[0]: i for i, p in enumerate(result.packages) if len(p) == 1}
    ratio = {r: result.medians[i, 2] / max(result.medians[i, 0], 1.0) for r, i in singles.items()}
    chosen: list[str] = []
    for r in sorted(ratio, key=ratio.get, reverse=True):
        trial = tuple(sorted(chosen + [r], key=cat.sequence_position))
        if trial in result.packages and result.medians[result.packages.index(trial), 0] <= budget_gbp:
            chosen = list(trial)
    g = result.medians[result.packages.index(tuple(chosen))] if tuple(chosen) in result.packages else None
    if g is None:
        return tuple(chosen), False
    others = np.delete(result.medians, result.packages.index(tuple(chosen)), axis=0)
    within = others[others[:, 0] <= budget_gbp]
    dominated = bool(np.any((within[:, 0] <= g[0]) & (within[:, 1:] >= g[1:]).all(1)
                            & ((within[:, 0] < g[0]) | (within[:, 1:] > g[1:]).any(1))))
    return tuple(chosen), dominated


# --------------------------------------------------------------------------- modes and baselines (R-23 .. R-25)
OBJECTIVES = ("net_cost_gbp", "carbon_saving_kgco2e", "bill_saving_gbp", "score_gain")


def reference_point(result: ParetoResult, min_carbon_kg: Optional[float] = None, max_cost_gbp: Optional[float] = None,
                    min_bill_gbp: Optional[float] = None, min_score_gain: Optional[float] = None,
                    goal: Goal = "bills", p_min: float = 0.5) -> dict:
    """Reference-point mode, e.g. ">= 2 t CO2, <= GBP 10k". A package qualifies when P(meets every target) >= p_min
    across posterior samples (cost and savings from the same paired samples); of those, the one with the best median
    goal saving, then the cheapest. Targets are hard: nothing is relaxed. Needs `result.samples`."""
    if result.samples is None or not result.packages:
        return {"package": None, "p_meets": None, "reason": "No package samples"}
    cost, carbon, bill = result.samples
    ok = np.ones_like(cost, bool)
    if max_cost_gbp is not None:
        ok &= cost <= max_cost_gbp
    if min_carbon_kg is not None:
        ok &= carbon >= min_carbon_kg
    if min_bill_gbp is not None:
        ok &= bill >= min_bill_gbp
    if min_score_gain is not None:
        ok &= (result.medians[:, 3] >= min_score_gain)[:, None]
    p = ok.mean(1)
    cand = np.flatnonzero(p >= p_min)
    targets = {"min_carbon_kg": min_carbon_kg, "max_cost_gbp": max_cost_gbp, "min_bill_gbp": min_bill_gbp,
               "min_score_gain": min_score_gain, "p_min": p_min}
    if len(cand) == 0:
        best = int(np.argmax(p))
        return {"package": None, "p_meets": None, "targets": targets,
                "closest": {"package": list(result.packages[best]), "p_meets": round(float(p[best]), 4)},
                "reason": f"No package meets every target with probability >= {p_min:.0%}"}
    col = 2 if goal == "bills" else 1
    k = int(min(cand, key=lambda i: (-result.medians[i, col], result.medians[i, 0])))
    return {"package": list(result.packages[k]), "p_meets": round(float(p[k]), 4), "targets": targets,
            "n_qualifying": int(len(cand)), "medians": dict(zip(OBJECTIVES, map(float, result.medians[k])))}


def preference_rerank(result: ParetoResult, weights: dict[str, float],
                      acceptance: Optional[dict[str, float]] = None, lam: float = 0.3) -> list[dict]:
    """Memory-learned re-ranking of the packages that can be on the front (P(on front) > 0). It only ORDERS them:
    no package is added or removed, budget and feasibility are untouched. utility = sum_k w_k * normalised objective
    (cost inverted) + lam * (mean acceptance of the package's items - 0.5); acceptance is the memory score (0-1)."""
    idx = [i for i in range(len(result.packages)) if result.p_on_front[i] > 0 and result.packages[i]]
    if not idx:
        return []
    M = result.medians[idx].astype(float)
    lo, span = M.min(0), np.maximum(np.ptp(M, 0), 1e-9)
    Z = (M - lo) / span
    Z[:, 0] = 1 - Z[:, 0]
    w = np.array([max(0.0, float(weights.get(k, 0.0))) for k in OBJECTIVES])
    w = w / w.sum() if w.sum() > 0 else np.full(4, 0.25)
    rows = []
    for j, i in enumerate(idx):
        acc = [acceptance.get(r, 0.5) for r in result.packages[i]] if acceptance else []
        m = float(np.mean(acc)) if acc else 0.5
        u = float(Z[j] @ w) + lam * (m - 0.5)
        rows.append({"package": list(result.packages[i]), "utility": round(u, 4), "memory_score": round(m, 3),
                     "p_on_front": round(float(result.p_on_front[i]), 4),
                     "medians": dict(zip(OBJECTIVES, map(lambda x: round(float(x), 2), M[j])))})
    rows.sort(key=lambda r: -r["utility"])
    for k, r in enumerate(rows, 1):
        r["rank"] = k
    return rows


def weights_from_memory(confirmed: Sequence[dict]) -> dict[str, float]:
    """Goal items in customer memory -> objective weights. Unknown goals leave the even default."""
    w = {k: 1.0 for k in OBJECTIVES}
    for it in confirmed:
        if it.get("kind") != "goal":
            continue
        v = f"{it.get('key', '')} {it.get('value', '')}".lower()
        if "carbon" in v or "co2" in v or "climate" in v:
            w["carbon_saving_kgco2e"] += 2
        if "bill" in v or "money" in v or "cost" in v:
            w["bill_saving_gbp"] += 2
        if "cheap" in v or "budget" in v or "low cost" in v:
            w["net_cost_gbp"] += 2
        if "score" in v or "rating" in v or "epc" in v or "value" in v:
            w["score_gain"] += 2
    return w


def co2_only_baseline(result: ParetoResult, budget_gbp: float) -> tuple[tuple[str, ...], bool]:
    """CO2-only optimiser: the package with the largest median carbon saving within budget (ignores bills/score).
    Returns it and whether another in-budget package dominates it on all four medians."""
    within = [i for i in range(len(result.packages)) if result.medians[i, 0] <= budget_gbp]
    if not within:
        return (), False
    k = max(within, key=lambda i: (result.medians[i, 1], -result.medians[i, 0]))
    return result.packages[k], _dominated(result, k, budget_gbp)


def _dominated(result: ParetoResult, k: int, budget_gbp: float) -> bool:
    g = result.medians[k]
    others = np.delete(result.medians, k, axis=0)
    within = others[others[:, 0] <= budget_gbp]
    return bool(np.any((within[:, 0] <= g[0]) & (within[:, 1:] >= g[1:]).all(1)
                       & ((within[:, 0] < g[0]) | (within[:, 1:] > g[1:]).any(1))))


def hypervolume(result: ParetoResult, rows: Sequence[int], budget_gbp: Optional[float] = None,
                n: int = 200_000, seed: int = 42) -> float:
    """Share of the normalised 4-D objective box dominated by the given packages' medians (Monte Carlo, fixed seed).
    Box: cost in [0, budget or max cost], each saving/gain in [0, its max over all packages]. 0 = nothing, 1 = ideal."""
    M = result.medians
    hi_cost = budget_gbp if budget_gbp is not None else max(float(M[:, 0].max()), 1.0)
    hi = np.maximum(M[:, 1:].max(0), 1e-9)
    P = np.column_stack([1 - M[rows, 0] / hi_cost, np.clip(M[rows, 1:], 0, None) / hi])  # all maximised in [0, 1]
    P = P[(M[rows, 0] <= hi_cost)]
    if len(P) == 0:
        return 0.0
    U = np.random.default_rng(seed).random((n, 4))
    dom = np.zeros(n, bool)
    for p in P:
        dom |= (U <= p).all(1)
    return float(dom.mean())
