"""Improvement potential (PS section 3): 100 x (E_base - E_after) / E_base on purchased energy (grid import + gas),
with ALL feasible retrofits run together as ONE package. Null + reason when the baseline is 0 or unavailable.

The package is the ladder's selection (one of each set of alternatives, R-12). `pct_if_surveys_pass` adds the
needs-survey items. `p_top_decile` needs the whole population, so it is filled in batch, not here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from contracts.models import FeasibilityReport, ImprovementPotential, Property, RetrofitAction
from contracts.summarise import summarise
from greentwin.interfaces import PosteriorSamples

from .ladder import Goal, select
from .runner import Energy, Meter, anchored, fuel_supported, run_packages, saving


def _pct(base: Energy, after: Energy) -> tuple[Optional[np.ndarray], Optional[str]]:
    b = base.purchased
    if not np.all(np.isfinite(b)) or np.any(b <= 0):
        return None, "Baseline purchased energy is 0 or unavailable, so improvement potential is undefined"
    return 100 * (b - after.purchased) / b, None


def improvement_potential(prop: Property, samples: PosteriorSamples, feasibility: FeasibilityReport,
                          meter: Optional[Meter] = None, goal: Goal = "bills", seed: int = 42) -> ImprovementPotential:
    if not fuel_supported(prop):
        z = summarise(np.zeros(samples.n_samples), "kWh/yr")
        return ImprovementPotential(unavailable_reason=f"Main fuel {prop.heating.main_fuel}: purchased energy of that "
                                    "fuel isn't modelled yet", electricity_change_kwh=z, gas_change_kwh=z)
    sel = select(prop, samples, feasibility, goal, meter, seed)
    acts = [RetrofitAction(retrofit_id=r) for r in sel.feasible]
    acts_s = acts + [RetrofitAction(retrofit_id=r) for r in sel.survey]
    pk = [[], acts] + ([acts_s] if sel.survey else [])
    runs = run_packages(samples, pk)
    base, after = anchored(runs[0], runs[0], meter), anchored(runs[1], runs[0], meter)
    pct, reason = _pct(base, after)
    pct_s = None
    if sel.survey and reason is None:
        x, _ = _pct(base, anchored(runs[2], runs[0], meter))
        pct_s = summarise(x, "%")
    return ImprovementPotential(
        pct=None if pct is None else summarise(pct, "%", tails=[("potential below 20%", 20.0, "below")]),
        unavailable_reason=reason,
        electricity_change_kwh=saving(base.electricity, after.electricity, "kWh/yr", "electricity use"),
        gas_change_kwh=saving(base.gas, after.gas, "kWh/yr", "gas use"),
        included_retrofits=sel.feasible, excluded_needs_survey=sel.survey, pct_if_surveys_pass=pct_s,
    )
