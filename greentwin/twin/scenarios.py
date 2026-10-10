"""Retrofit packages for the viewer: re-simulate the whole package (never sum) and report before / after / savings
for the house plus per-room numbers after the package. Used by the dev twin server and the example generator.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from contracts.models import RetrofitAction
from contracts.summarise import summarise
from greentwin.interfaces import PosteriorSamples, TwinRun
from greentwin.twin.appliances import summarise_appliances
from greentwin.twin.archetypes import TwinLayout
from greentwin.twin.rooms import _range, allocate_rooms, energy_overlay, unit_rates_gbp
from greentwin.twin.scene import AppliancePlacement, HouseNumbers, ScenarioView, scenario_key
from greentwin.twin.twin import GreenDigitalTwin, retrofit_samples

FACTORS = Path(__file__).resolve().parents[2] / "contracts" / "emission_factors.json"


@lru_cache(maxsize=1)
def carbon_factors() -> dict[str, float]:
    f = json.loads(FACTORS.read_text(encoding="utf-8"))["factors"]
    return {"electricity": f["electricity"]["value"], "mains_gas": f["gas"]["value"]}


def _house(run: TwinRun, other_fuel: bool) -> dict[str, np.ndarray | None]:
    gas = np.asarray(run.gas_kwh, dtype=float)
    elec = np.asarray(run.electricity_import_kwh, dtype=float)
    rates, cf = unit_rates_gbp(), carbon_factors()
    cost = None if other_fuel else gas * rates["mains_gas"] + elec * rates["electricity"]
    carbon = None if other_fuel else gas * cf["mains_gas"] + elec * cf["electricity"]
    return {"gas_kwh": gas, "electricity_kwh": elec, "cost_gbp": cost, "carbon_kg": carbon}


def _numbers(values: dict, worst: str) -> HouseNumbers:
    units = {"gas_kwh": "kWh/yr", "electricity_kwh": "kWh/yr", "cost_gbp": "GBP/yr", "carbon_kg": "kgCO2e/yr"}
    return HouseNumbers(**{k: (_range(summarise(v, units[k], worst_tail=worst)) if v is not None else None)
                           for k, v in values.items()})


def run_scenario(twin: GreenDigitalTwin, layout: TwinLayout, samples: PosteriorSamples, retrofit_ids: Sequence[str],
                 appliances: Optional[Sequence[AppliancePlacement]] = None, weather_source: str = "synthetic",
                 calibrated: bool = False) -> ScenarioView:
    actions = [RetrofitAction(retrofit_id=r) for r in retrofit_ids]
    before = twin.simulate(samples, [])
    after = twin.simulate(samples, actions)
    other_fuel = samples.base.heating_fuel not in ("mains_gas", "electricity")
    b, a = _house(before, other_fuel), _house(after, other_fuel)
    savings = {k: (b[k] - a[k]) if b[k] is not None else None for k in b}

    summary = summarise_appliances(appliances, getattr(samples.base, "occupants", 2.4)) if appliances else None
    alloc = allocate_rooms(layout, after, retrofit_samples(samples, actions), summary, engine=twin.last_engine)
    warnings = []
    if other_fuel:
        warnings.append(f"{samples.base.heating_fuel}: no sourced price or carbon factor yet, so cost and carbon are null")
    if "time_of_use_tariff" in retrofit_ids:
        warnings.append("Time-of-use tariff changes bills, not kWh; no sourced ToU tariff yet, so no bill effect is shown")
    return ScenarioView(
        key=scenario_key(retrofit_ids), retrofit_ids=sorted(retrofit_ids), engine=twin.last_engine,
        calibrated=calibrated, weather_source=weather_source,
        before=_numbers(b, "high"), after=_numbers(a, "high"), savings=_numbers(savings, "low"),
        rooms_after=energy_overlay(alloc, calibrated=calibrated, weather_source=weather_source),
        warnings=warnings,
    )
