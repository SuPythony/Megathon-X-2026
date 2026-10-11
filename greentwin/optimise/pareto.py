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
Value is never an objective. Baseline for evaluation: greedy kWh-per-GBP under the same budget.
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
    return ParetoResult(points, packages, med, p_on, excluded, warnings, knees)


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
