"""Stage a retrofit package over years by dependencies and cash flow (plan 4c.4, decision R-22).

Input: the ladder's steps (fabric-first order, marginal effects from the twin). The household puts aside a fixed
budget each year; bill savings from finished work are added to the pot when `reinvest` is on. Each year a set of
not-yet-done steps is installed if the pot covers their median net cost and every `depends_on` step that is part of
the package is already done or installed that year (heat pump after the fabric, battery after PV).

A small dynamic programme over (done-set) states, year by year, minimises the savings lost while waiting:
    lost = sum over years of the annual saving of the steps not yet installed,
with leftover cash as the tie-break. States keep only non-dominated (lost, cash) labels, so the search is exact for
the stated objective. The savings of a set of steps is the sum of their ladder marginals: exact when the stage
order follows the ladder, an approximation otherwise (stated in every result). Steps without a sourced cost can't
be cash-planned and are listed separately with the reason; `needs_survey` and `not_worth_adding` steps are left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from contracts.models import RetrofitEffect
from greentwin.retrofit import catalogue as cat

Goal = Literal["bills", "carbon"]
MAX_ITEMS = 12
APPROX_NOTE = ("Savings of a stage are the sum of the ladder's marginal savings: exact in ladder order, an "
               "approximation when the stages reorder steps. Every package itself is still one twin run.")


@dataclass
class Stage:
    year: int
    retrofit_ids: list[str]
    cost_gbp: float               # median net cost after confirmed grants
    cash_before_gbp: float        # pot at the start of the year, incl. this year's budget
    cash_after_gbp: float
    annual_bill_saving_gbp: float  # once this stage is done (sum of marginals of everything done so far)


@dataclass
class StagingPlan:
    stages: list[Stage]
    annual_budget_gbp: float
    reinvest_savings: bool
    lost_saving_gbp: float
    not_staged: dict[str, str] = field(default_factory=dict)  # retrofit_id -> reason
    notes: list[str] = field(default_factory=lambda: [APPROX_NOTE])

    def as_dict(self) -> dict:
        return {"annual_budget_gbp": self.annual_budget_gbp, "reinvest_savings": self.reinvest_savings,
                "lost_saving_gbp": round(self.lost_saving_gbp, 0),
                "stages": [{"year": s.year, "retrofit_ids": s.retrofit_ids, "cost_gbp": round(s.cost_gbp),
                            "cash_before_gbp": round(s.cash_before_gbp), "cash_after_gbp": round(s.cash_after_gbp),
                            "annual_bill_saving_gbp": round(s.annual_bill_saving_gbp)} for s in self.stages],
                "not_staged": self.not_staged, "notes": self.notes}


def _submasks(m: int):
    s = m
    while s:
        yield s
        s = (s - 1) & m


def stage(steps: Sequence[RetrofitEffect], annual_budget_gbp: float, max_years: int = 10,
          reinvest: bool = True, upfront_gbp: float = 0.0) -> StagingPlan:
    not_staged: dict[str, str] = {}
    ids, cost, save = [], [], []
    for e in steps:
        if e.verdict in ("needs_survey", "not_worth_adding"):
            not_staged[e.retrofit_id] = ("Needs a survey first" if e.verdict == "needs_survey"
                                         else "Not worth adding after the earlier steps")
        elif e.cost_gbp is None:
            not_staged[e.retrofit_id] = e.cost_unavailable_reason or "No sourced cost: can't be cash-planned"
        else:
            ids.append(e.retrofit_id)
            cost.append(max(0.0, e.cost_gbp.median))
            save.append(e.bill_saving_gbp.median)
    if len(ids) > MAX_ITEMS:
        for r in ids[MAX_ITEMS:]:
            not_staged[r] = f"Beyond the first {MAX_ITEMS} steps of the ladder"
        ids, cost, save = ids[:MAX_ITEMS], cost[:MAX_ITEMS], save[:MAX_ITEMS]
    n = len(ids)
    full = (1 << n) - 1
    pos = {r: i for i, r in enumerate(ids)}
    dep = [sum(1 << pos[d] for d in cat.get(r).depends_on if d in pos) for r in ids]
    cost_of = [sum(cost[i] for i in range(n) if m >> i & 1) for m in range(1 << n)]
    save_of = [sum(save[i] for i in range(n) if m >> i & 1) for m in range(1 << n)]
    total_save = save_of[full]

    def ok(done: int, add: int) -> bool:
        both = done | add
        return all((dep[i] & ~both) == 0 for i in range(n) if add >> i & 1)

    # labels: mask -> list of (lost, cash, path) non-dominated
    labels: dict[int, list[tuple[float, float, list[int]]]] = {0: [(0.0, upfront_gbp, [])]}
    best: Optional[tuple[float, float, list[int]]] = None
    for _year in range(max_years):
        nxt: dict[int, list[tuple[float, float, list[int]]]] = {}
        for done, labs in labels.items():
            rest = full & ~done
            for lost, cash, path in labs:
                pot = cash + annual_budget_gbp
                for add in list(_submasks(rest)) + [0]:
                    if cost_of[add] > pot + 1e-6 or not ok(done, add):
                        continue
                    nd = done | add
                    # this year's saving accrues on everything done by its end; lost = what's still missing
                    nl = lost + (total_save - save_of[nd])
                    nc = pot - cost_of[add] + (max(0.0, save_of[nd]) if reinvest else 0.0)
                    cand = (nl, nc, path + [add])
                    keep = [x for x in nxt.get(nd, []) if not (nl <= x[0] and nc >= x[1])]
                    if not any(x[0] <= nl and x[1] >= nc for x in keep):
                        keep.append(cand)
                    nxt[nd] = keep
        labels = nxt
        if full in labels:
            cands = labels[full]
            top = min(cands, key=lambda x: (x[0], -x[1]))
            if best is None or (top[0], -top[1]) < (best[0], -best[1]):
                best = top
            break
    if best is None:  # could not finish within max_years: take the state with the least lost saving
        allc = [(lab, m) for m, labs in labels.items() for lab in labs]
        lab, m = min(allc, key=lambda x: (x[0][0] + (total_save - save_of[x[1]]) * max_years, -x[0][1]))
        best = lab
        for i in range(n):
            if not (m >> i & 1):
                not_staged[ids[i]] = f"Not affordable within {max_years} years at this budget"

    stages, cash, done = [], upfront_gbp, 0
    for y, add in enumerate(best[2], 1):
        pot = cash + annual_budget_gbp
        done |= add
        after = pot - cost_of[add]
        cash = after + (max(0.0, save_of[done]) if reinvest else 0.0)
        if add:
            stages.append(Stage(y, [ids[i] for i in range(n) if add >> i & 1], cost_of[add], pot, after,
                                save_of[done]))
    return StagingPlan(stages, annual_budget_gbp, reinvest, best[0], not_staged)
