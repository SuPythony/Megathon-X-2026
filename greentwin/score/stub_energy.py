"""STUB energy input for the score until Part 2's ``EnergyRetrofitEngine.baseline`` lands.

Builds the brief's intensity: annualised METERED (grid import + gas) / floor area. Missing months are never
zero: gas is filled from a per-home degree-day fit (gas = a + b * HDD) on the observed months, electricity
from the observed monthly mean. Uncertainty is a bootstrap over observed months plus meter noise, so more
missing months -> wider interval. Not weather-normalised (that is a sensitivity only).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from contracts.models import EnergyBreakdown, EnergyUsageMonth, Property
from contracts.summarise import summarise

# SAP 10.2 Table U1, UK average mean external temperature (deg C), Jan..Dec
UK_MEAN_T = np.array([4.3, 4.9, 6.5, 8.9, 11.7, 14.6, 16.6, 16.4, 14.1, 10.6, 7.1, 4.2])
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
HDD = DAYS * np.maximum(0.0, 15.5 - UK_MEAN_T)  # monthly heating degree days, base 15.5 C
METER_NOISE = 0.03


def _annual(vals: dict[int, float], kind: str, idx: np.ndarray, rng) -> float:
    obs = np.array(sorted(vals))
    sel = obs[idx]
    y = np.array([vals[i] for i in sel])
    missing = [i for i in range(12) if i not in vals]
    if kind == "gas" and len(set(HDD[sel])) > 1 and y.sum() > 0:
        b, a = np.polyfit(HDD[sel], y, 1)
        b, a = max(b, 0.0), max(a, 0.0)
        fill = sum(a + b * HDD[i] for i in missing)
    else:
        fill = y.mean() * len(missing)
    return sum(vals.values()) + fill


def energy_from_usage(prop: Property, usage: Sequence[EnergyUsageMonth], n: int = 2000,
                      seed: int = 42) -> EnergyBreakdown:
    rng = np.random.default_rng(seed)
    elec = {int(u.month[5:7]) - 1: u.electricity_import_kwh for u in usage if u.electricity_import_kwh is not None}
    gas = {int(u.month[5:7]) - 1: u.gas_kwh for u in usage if u.gas_kwh is not None}
    if not elec:
        raise ValueError("no metered electricity months: intensity unavailable (use the twin prior instead)")
    no_gas_supply = not gas and prop.heating.main_fuel != "mains_gas"

    es, gs = np.empty(n), np.empty(n)
    for s in range(n):
        ie = rng.integers(0, len(elec), len(elec))
        es[s] = _annual(elec, "elec", ie, rng) * (1 + METER_NOISE * rng.standard_normal())
        if gas:
            ig = rng.integers(0, len(gas), len(gas))
            gs[s] = _annual(gas, "gas", ig, rng) * (1 + METER_NOISE * rng.standard_normal())
        else:
            gs[s] = 0.0 if no_gas_supply else np.nan
    if np.isnan(gs).any():
        raise ValueError("gas readings missing for a gas-heated home: intensity unavailable")
    tot = es + gs
    return EnergyBreakdown(
        electricity_import_kwh=summarise(es, "kWh/yr"),
        gas_kwh=summarise(gs, "kWh/yr"),
        purchased_total_kwh=summarise(tot, "kWh/yr"),
        intensity_kwh_m2=summarise(tot / prop.floor_area_m2, "kWh/m2/yr", worst_tail="high"),
    )
