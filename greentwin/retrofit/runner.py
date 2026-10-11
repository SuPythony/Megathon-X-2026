"""Run the twin on many retrofit packages at once, then price, carbon-count and meter-anchor the results.

Every package is ONE twin run on the whole package (never a sum). For speed, all packages that share a fuel
configuration are stacked into a single ``jax.vmap`` call with a cached ``jit`` (R-08): the ladder, improvement
potential and Pareto front then cost a few compiled calls instead of one compile per package.

Meter anchoring (R-07). Until the Bayesian calibration lands on ``main``, twin samples come from the prior and can
sit far from the home's bills (SH1 gas: twin median ~25,000 vs metered ~15,000 kWh). When metered annual energy is
available, each fuel's "after" is anchored to it: after_f = metered_f x twin_after_f / twin_base_f, per sample
(electricity added by a fuel switch takes the heating fuel's ratio, see ``anchored``).
The physics still decides the relative change, interactions included; the meter decides the level. With a
calibrated posterior the ratio metered / twin_base is ~1 and the anchor changes little.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from contracts.models import EnergyBreakdown, Property, RetrofitAction
from contracts.summarise import summarise
from greentwin.interfaces import PosteriorSamples, TwinRun
from greentwin.twin.params import ExtendedTwinParams, fuel_config
from greentwin.twin.physics_api import BEHAVIOUR_KEYS, ELEMENTS, MONTHLY_KEYS, TOTAL_KEYS, FuelConfig, get_run_physics
from greentwin.twin.simulate import _input_keys
from greentwin.twin.twin import retrofit_samples
from greentwin.twin.weather import resolve_weather

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_FUELS = ("mains_gas", "electricity")


# --------------------------------------------------------------------------- sourced factors
@lru_cache(maxsize=1)
def emission_factors() -> dict[str, float]:
    f = json.loads((ROOT / "contracts" / "emission_factors.json").read_text(encoding="utf-8"))["factors"]
    return {"electricity": f["electricity"]["value"], "gas": f["gas"]["value"]}


@lru_cache(maxsize=1)
def emission_factor_source() -> str:
    s = json.loads((ROOT / "contracts" / "emission_factors.json").read_text(encoding="utf-8"))["source"]
    return f"{s['publisher']}: {s['title']} ({s['url']})"


@lru_cache(maxsize=1)
def tariffs() -> dict:
    return json.loads((ROOT / "contracts" / "tariffs.json").read_text(encoding="utf-8"))


def unit_rates() -> dict[str, float]:
    t = tariffs()
    return {"electricity": t["electricity"]["unit_rate_p_per_kwh"] / 100, "gas": t["gas"]["unit_rate_p_per_kwh"] / 100}


def standing_charges() -> dict[str, float]:
    """GBP per year per supplied fuel."""
    t = tariffs()
    return {"electricity": t["electricity"]["standing_charge_p_per_day"] * 3.65,
            "gas": t["gas"]["standing_charge_p_per_day"] * 3.65}


def fuel_supported(prop: Property) -> bool:
    return prop.heating.main_fuel in SUPPORTED_FUELS


# --------------------------------------------------------------------------- batched twin runs
@lru_cache(maxsize=None)
def _compiled(mode: str, fuel: FuelConfig):
    fn, _ = get_run_physics()
    return jax.jit(jax.vmap(lambda p, w: fn(p, w, mode, fuel), in_axes=(0, None)))


def engine_name() -> str:
    return get_run_physics()[1]


def _fuel(base) -> FuelConfig:
    if isinstance(base, ExtendedTwinParams):
        return fuel_config(base)
    return FuelConfig(heating_fuel=base.heating_fuel, hot_water_fuel=base.heating_fuel)


def _package_params(samples: PosteriorSamples, mode: str) -> dict[str, np.ndarray]:
    """Full parameter arrays for one (already retrofitted) sample set, honouring standard-mode behaviour."""
    out = {}
    for key in _input_keys(samples.base):
        if (mode == "standard" and key in BEHAVIOUR_KEYS) or key not in samples.params:
            out[key] = np.full(samples.n_samples, float(getattr(samples.base, key)), dtype=np.float32)
        else:
            out[key] = np.asarray(samples.params[key], dtype=np.float32)
    return out


def run_packages(samples: PosteriorSamples, packages: Sequence[Sequence[RetrofitAction]],
                 mode: str = "calibrated") -> list[TwinRun]:
    """One TwinRun per package, each the whole package re-simulated. Packages are batched by fuel config."""
    weather, _ = resolve_weather(getattr(samples.base, "postcode_district", "UNKNOWN"))
    w = {k: jnp.asarray(v) for k, v in weather.items()}
    n = samples.n_samples
    prepared = [retrofit_samples(samples, list(p)) for p in packages]
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(prepared):
        keys = tuple(_input_keys(s.base))
        groups.setdefault((_fuel(s.base), keys), []).append(i)

    runs: list[Optional[TwinRun]] = [None] * len(packages)
    for (fuel, keys), idx in groups.items():
        per = [_package_params(prepared[i], mode) for i in idx]
        batch = {k: jnp.asarray(np.concatenate([p[k] for p in per])) for k in keys}
        out = _compiled(mode, fuel)(batch, w)
        out = {k: np.asarray(v) for k, v in out.items() if k != "element_heat_loss_w_per_k"} | {
            "element_heat_loss_w_per_k": {k: np.asarray(v) for k, v in out.get("element_heat_loss_w_per_k", {}).items()}}
        for j, i in enumerate(idx):
            sl = slice(j * n, (j + 1) * n)
            runs[i] = TwinRun(
                **{k: out[k][sl] for k in TOTAL_KEYS},
                monthly_electricity_kwh=out[MONTHLY_KEYS[0]][sl], monthly_gas_kwh=out[MONTHLY_KEYS[1]][sl],
                element_heat_loss_w_per_k={e: v[sl] for e, v in out["element_heat_loss_w_per_k"].items() if e in ELEMENTS},
            )
    return runs  # type: ignore[return-value]


# --------------------------------------------------------------------------- anchored per-fuel energy
@dataclass
class Energy:
    """Per-fuel annual energy samples for one package, shape (n,). Anchored to meters when available."""

    electricity: np.ndarray
    gas: np.ndarray
    pv: Optional[np.ndarray] = None
    space_heating: Optional[np.ndarray] = None
    hot_water: Optional[np.ndarray] = None

    @property
    def purchased(self) -> np.ndarray:
        return self.electricity + self.gas

    def bill(self) -> np.ndarray:
        r = unit_rates()
        return self.electricity * r["electricity"] + self.gas * r["gas"]

    def carbon(self) -> np.ndarray:
        f = emission_factors()
        return self.electricity * f["electricity"] + self.gas * f["gas"]


@dataclass
class Meter:
    """Metered annual energy samples (from the home's bills), the anchor for the twin's level."""

    electricity: np.ndarray
    gas: np.ndarray
    breakdown: EnergyBreakdown
    warnings: list[str] = field(default_factory=list)


def _k(metered: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.where(base > 0, metered / np.where(base > 0, base, 1.0), 1.0)


def anchored(run: TwinRun, base_run: TwinRun, meter: Optional[Meter]) -> Energy:
    """Per-sample meter anchor (R-07). Gas: after x (metered / twin baseline). Electricity: the part up to the
    twin's baseline keeps the electricity ratio; any increase above it is new heating demand from a fuel switch
    (heat pump, storage heaters) and takes the heating fuel's ratio, because the gas meter is what measured that
    home's heat demand. For an all-electric home both ratios are the electricity ratio."""
    e = np.asarray(run.electricity_import_kwh, dtype=float)
    g = np.asarray(run.gas_kwh, dtype=float)
    if meter is not None:
        be = np.asarray(base_run.electricity_import_kwh, dtype=float)
        bg = np.asarray(base_run.gas_kwh, dtype=float)
        k_e, k_g = _k(meter.electricity, be), _k(meter.gas, bg)
        k_heat = np.where(bg > 0, k_g, k_e)
        e = k_e * np.minimum(e, be) + k_heat * np.maximum(e - be, 0.0)
        g = np.where(bg > 0, k_g * g, g)
    return Energy(electricity=np.maximum(e, 0.0), gas=np.maximum(g, 0.0),
                  pv=np.asarray(run.pv_generation_kwh, dtype=float),
                  space_heating=np.asarray(run.space_heating_kwh, dtype=float),
                  hot_water=np.asarray(run.hot_water_kwh, dtype=float))


def breakdown(energy: Energy, floor_area_m2: float, metered: bool) -> EnergyBreakdown:
    """EnergyBreakdown via summarise. Intensity is (grid import + gas) / floor area, not weather-normalised."""
    s = lambda x: summarise(x, "kWh/yr", worst_tail="high")  # noqa: E731
    return EnergyBreakdown(
        electricity_import_kwh=s(energy.electricity), gas_kwh=s(energy.gas), purchased_total_kwh=s(energy.purchased),
        space_heating_kwh=s(energy.space_heating) if energy.space_heating is not None else None,
        hot_water_kwh=s(energy.hot_water) if energy.hot_water is not None else None,
        pv_generation_kwh=s(energy.pv) if energy.pv is not None and np.any(energy.pv > 0) else None,
        intensity_kwh_m2=summarise(energy.purchased / floor_area_m2, "kWh/m2/yr", worst_tail="high"),
    )


def saving(before: np.ndarray, after: np.ndarray, unit: str, what: str) -> "Distribution":  # noqa: F821
    """baseline - after; negative = increase. Tail probability of an increase is always reported."""
    return summarise(before - after, unit, tails=[(f"{what} rises", 0.0, "below")], worst_tail="low")


def meter_from_usage(prop: Property, usage: Sequence, n: int, seed: int = 42) -> Optional[Meter]:
    """Annualised metered energy from monthly readings (missing months filled by a degree-day fit and flagged,
    never 0), resampled to n samples. None when the readings can't support it (then the twin prior is used)."""
    from greentwin.score.sampling import from_distribution
    from greentwin.score.stub_energy import energy_from_usage

    try:
        eb = energy_from_usage(prop, usage, n=max(n, 500), seed=seed)
    except ValueError:
        return None
    rng = np.random.default_rng(seed)
    elec = np.maximum(from_distribution(eb.electricity_import_kwh, n, rng), 0.0)
    gas = np.maximum(from_distribution(eb.gas_kwh, n, rng), 0.0)
    missing = 12 - len({u.month for u in usage if u.electricity_import_kwh is not None})
    warnings = [f"{missing} month(s) of electricity readings missing: filled and the range widened"] if missing > 0 else []
    return Meter(electricity=elec, gas=gas, breakdown=eb, warnings=warnings)
