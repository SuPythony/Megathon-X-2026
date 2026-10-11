"""Improvement ladder (LLD 4.2): feasible retrofits in fabric-first order, each step's MARGINAL effect on every sample.

Step k's effect = run(steps 1..k-1) - run(steps 1..k), per fuel and per sample, so the steps add up exactly to the
package and interactions come from the physics. A different combination is a new run, never "package minus a row".

Selection (R-12): the ladder is one coherent package, so of mutually exclusive alternatives (catalogue `excludes`,
and any two main heating systems) it keeps one, judged across feasible and needs-survey items together: items with
a sourced cost first, then the highest median lifetime net benefit under the goal. A needs-survey winner pushes its
feasible rival out (the reason names it as the fallback if the survey fails). Order (R-13): ordering group from feasibility_rules.json, then the item's own median
goal saving when run alone. `needs_survey` items follow the feasible ones (verdict `needs_survey`), so the feasible
marginals never depend on a survey outcome. Tariff switches change bills, not kWh, and need a sourced time-of-use
tariff: left out with a warning.

Verdicts (R-09): p_gain = P(step's goal saving >= 3% of the baseline goal metric).
recommended: p_suitable >= 0.8 · optional: p_suitable >= 0.5, or p_gain >= 0.5 (a big step that doesn't pay back on
the goal alone) · else not_worth_adding. No sourced cost -> at most optional, since payback can't be judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

from contracts.models import (
    AreaEPCStats, Distribution, Feasibility, FeasibilityReport, Property, RetrofitAction, RetrofitEffect,
)
from contracts.summarise import summarise
from greentwin.interfaces import PosteriorSamples

from . import catalogue as cat
from .property_edits import property_after
from .rating import rating_from_runs
from .runner import Energy, Meter, anchored, breakdown, fuel_supported, run_packages, saving

Goal = Literal["bills", "carbon"]
GAIN_THRESHOLD = 0.03          # R-09: a step must cut the goal metric by >= 3% of the baseline
P_GAIN_MIN = 0.5
SUITABILITY = ((0.8, "High"), (0.5, "Medium"))
OK_STATUS = ("feasible", "feasible_with_conditions")


def suitability(p: float) -> str:
    for cut, label in SUITABILITY:
        if p >= cut:
            return label
    return "Low"


def goal_metric(e: Energy, goal: Goal) -> np.ndarray:
    return e.bill() if goal == "bills" else e.carbon()


def _exclusive(a: str, b: str) -> bool:
    ia, ib = cat.get(a), cat.get(b)
    return b in ia.excludes or a in ib.excludes or (ia.category == ib.category == "heating")


@dataclass
class Selection:
    feasible: list[str]
    survey: list[str]
    dropped: dict[str, str] = field(default_factory=dict)      # retrofit_id -> why it is not in the ladder
    single_saving: dict[str, float] = field(default_factory=dict)  # median goal saving, run alone
    warnings: list[str] = field(default_factory=list)


def _net_benefit(rid: str, saving_med: float, n: int, seed: int, off_gas: bool) -> tuple[int, float]:
    c = cat.cost_samples(rid, n, seed, off_gas_grid=off_gas)
    life = cat.get(rid).lifetime_years or 1
    if not c.available:
        return 0, saving_med * life
    return 1, saving_med * life - float(np.median(c.net_gbp))


def select(prop: Property, samples: PosteriorSamples, feasibility: FeasibilityReport, goal: Goal,
           meter: Optional[Meter], seed: int = 42) -> Selection:
    by_id = {f.retrofit_id: f for f in feasibility.items}
    warnings, dropped = [], {}
    feas, survey = [], []
    for f in feasibility.items:
        if f.retrofit_id not in cat.items():
            continue
        if not cat.get(f.retrofit_id).changes_kwh:
            if f.status != "infeasible":
                dropped[f.retrofit_id] = "Changes bills, not energy; no sourced time-of-use tariff yet"
            continue
        if f.status in OK_STATUS:
            feas.append(f.retrofit_id)
        elif f.status == "needs_survey":
            survey.append(f.retrofit_id)

    cands = feas + survey
    runs = run_packages(samples, [[]] + [[RetrofitAction(retrofit_id=r)] for r in cands])
    base = anchored(runs[0], runs[0], meter)
    single = {r: float(np.median(goal_metric(base, goal) - goal_metric(anchored(run, runs[0], meter), goal)))
              for r, run in zip(cands, runs[1:])}

    off_gas = prop.constraints.off_gas_grid
    key = {r: _net_benefit(r, single[r], samples.n_samples, seed, off_gas) for r in cands}

    def resolve(pool: list[str], keep: list[str]) -> list[str]:
        chosen: list[str] = []
        for r in sorted(pool, key=lambda r: key[r], reverse=True):
            rival = next((c for c in keep + chosen if _exclusive(r, c)), None)
            if rival is None:
                chosen.append(r)
            else:
                dropped[r] = f"Alternative to {cat.get(rival).name.lower()}, which is chosen"
        return chosen

    chosen = resolve(cands, [])
    chosen_feas = [r for r in feas if r in chosen]
    chosen_survey = [r for r in survey if r in chosen]
    for r in survey:
        if r in chosen:
            for d, why in list(dropped.items()):
                if why.endswith(f"{cat.get(r).name.lower()}, which is chosen"):
                    dropped[d] = f"Alternative to {cat.get(r).name.lower()} (needs a survey); fallback if the survey fails"
    order = lambda r: (cat.sequence_group(r), -single[r])  # noqa: E731
    return Selection(sorted(chosen_feas, key=order), sorted(chosen_survey, key=order), dropped, single, warnings)


@dataclass
class LadderResult:
    effects: list[RetrofitEffect]
    selection: Optional[Selection]
    baseline: Optional[Energy] = None
    cumulative: list[Energy] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _reason(rid: str, f: Feasibility, step: int, goal: Goal, p_net: float, cost_note: Optional[str],
            verdict: str) -> str:
    item = cat.get(rid)
    group = ["fabric", "glazing", "controls", "heating system", "generation", "storage", "tariff"]
    g = cat.sequence_group(rid)
    parts = [f"Step {step}: {item.name} ({group[g] if g < len(group) else 'other'})."]
    if f.status == "feasible_with_conditions" or f.status == "needs_survey":
        parts.append("Conditions: " + "; ".join(f.reasons) + ".")
    what = "lower bills" if goal == "bills" else "lower carbon"
    life = f" over its {item.lifetime_years}-year life" if goal == "bills" and not cost_note else ""
    parts.append(f"Chance it pays back in {what}{life}: {p_net:.0%}.")
    if cost_note:
        parts.append(cost_note)
    if verdict == "optional" and p_net < 0.5:
        parts.append(f"A big step, but unlikely to pay back on {what} alone.")
    if verdict == "not_worth_adding":
        parts.append("Unlikely to pay back and adds little after the earlier steps.")
    return " ".join(parts)


def build_ladder(prop: Property, samples: PosteriorSamples, feasibility: FeasibilityReport, goal: Goal = "bills",
                 meter: Optional[Meter] = None, scorer=None, peer: Optional[AreaEPCStats] = None,
                 seed: int = 42) -> LadderResult:
    if not fuel_supported(prop):
        return LadderResult([], None, warnings=[f"Main fuel {prop.heating.main_fuel} has no sourced price or carbon "
                                                "factor: ladder unavailable"])
    sel = select(prop, samples, feasibility, goal, meter, seed)
    steps = sel.feasible + sel.survey
    warnings = list(sel.warnings) + [f"{cat.get(r).name}: {why}" for r, why in sel.dropped.items()]
    if meter is None:
        warnings.append("No meter readings: energy is the twin's prior, not anchored to bills")
    actions = [RetrofitAction(retrofit_id=r) for r in steps]
    packages = [actions[:k] for k in range(len(steps) + 1)]
    runs = run_packages(samples, packages)
    std = run_packages(samples, packages, mode="standard") if scorer is not None else None
    energies = [anchored(r, runs[0], meter) for r in runs]
    base = energies[0]
    base_goal = goal_metric(base, goal)
    base_purchased = base.purchased
    by_id = {f.retrofit_id: f for f in feasibility.items}
    n = samples.n_samples
    off_gas = prop.constraints.off_gas_grid

    effects = []
    for k, rid in enumerate(steps, 1):
        prev, cur = energies[k - 1], energies[k]
        f = by_id[rid]
        d_goal = goal_metric(prev, goal) - goal_metric(cur, goal)
        bill = prev.bill() - cur.bill()
        carbon = prev.carbon() - cur.carbon()

        c = cat.cost_samples(rid, n, seed, actions[k - 1], off_gas)
        life = cat.get(rid).lifetime_years or 1
        cost_note = None
        if goal == "bills" and c.available:
            p_net = float(np.mean(bill * life - c.net_gbp > 0))
        elif goal == "bills":
            p_net = float(np.mean(bill > 0))
            cost_note = "No sourced cost, so the chance ignores the installation cost."
        else:
            p_net = float(np.mean(carbon > 0))
        p_suit = f.p_feasible * p_net
        p_gain = float(np.mean(d_goal >= GAIN_THRESHOLD * base_goal))
        if f.status == "needs_survey":
            verdict = "needs_survey"
        elif p_suit >= 0.8 and c.available:
            verdict = "recommended"
        elif p_suit >= 0.5 or p_gain >= P_GAIN_MIN:
            verdict = "optional"
        else:
            verdict = "not_worth_adding"

        payback = None
        if c.available and np.all(bill > 0):
            payback = summarise(c.net_gbp / bill, "years", worst_tail="high")
        pot = None
        if np.all(base_purchased > 0):
            pot = summarise(100 * (base_purchased - cur.purchased) / base_purchased, "%")

        score_after = None
        if scorer is not None:
            rating = rating_from_runs(prop, std[0], std[k]).estimate
            prop_k = property_after(prop, actions[:k])
            score_after = scorer.score(prop_k, breakdown(cur, prop.floor_area_m2, meter is not None), rating,
                                       peer).score_distribution

        effects.append(RetrofitEffect(
            retrofit_id=rid, order=k, feasibility=f,
            electricity_saving_kwh=saving(prev.electricity, cur.electricity, "kWh/yr", "electricity use"),
            gas_saving_kwh=saving(prev.gas, cur.gas, "kWh/yr", "gas use"),
            carbon_saving_kgco2e=saving(prev.carbon(), cur.carbon(), "kgCO2e/yr", "carbon"),
            bill_saving_gbp=saving(prev.bill(), cur.bill(), "GBP/yr", "bill"),
            cost_gbp=c.distribution(), cost_unavailable_reason=c.unavailable_reason,
            purchased_energy_saving_kwh=saving(prev.purchased, cur.purchased, "kWh/yr", "purchased energy"),
            score_after=score_after, improvement_potential_cumulative_pct=pot, payback_years=payback,
            p_net_benefit=round(p_net, 4), p_suitable=round(p_suit, 4), suitability=suitability(p_suit),
            p_gain_above_threshold=round(p_gain, 4), verdict=verdict,
            reason=_reason(rid, f, k, goal, p_net, cost_note, verdict),
        ))
        warnings += [w for w in c.warnings if w not in warnings]
    return LadderResult(effects, sel, base, energies, warnings)
