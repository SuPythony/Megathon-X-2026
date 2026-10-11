"""Run the physics on every posterior sample of a home: ``PosteriorSamples`` -> ``TwinRun`` (``jax.vmap``).

``simulate`` is the wrapper Parts 2-4 call (through ``DigitalTwin.simulate``). It doesn't care which physics runs:
``physics_api.get_run_physics()`` picks the teammate's model if present, else the stub. Retrofit actions are applied
by ``apply_retrofits`` before this (not built yet; blocked on the catalogue key mismatch, see the handoff doc).

``prior_samples`` gives samples from the prior when a home has no calibration yet (``CalibrationResult.method =
"prior_only"``): the four calibrated parameters get prior spread, everything else stays at its RdSAP/SAP value.
"""

from __future__ import annotations

from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

from contracts.models import TwinParams
from greentwin.interfaces import PosteriorSamples, TwinRun
from greentwin.twin.physics_api import (
    BEHAVIOUR_KEYS,
    ELEMENTS,
    EXTRA_PARAM_KEYS,
    MONTHLY_KEYS,
    PHYSICS_PARAM_KEYS,
    TOTAL_KEYS,
    FuelConfig,
    RunPhysics,
    get_run_physics,
)

# Prior spread for the calibrated parameters (CalibrationResult.parameters). Working priors, to be replaced by the
# EBM energy prior and the hierarchy once calibration is built.
PRIOR = {
    "heat_loss_multiplier": ("lognormal", 0.0, 0.25),   # x1, ~ +/-25%
    "setpoint_c": ("normal_add", 0.0, 1.0),             # +/- 1 K around the default
    "baseload_kwh_per_day": ("lognormal", 0.0, 0.30),   # x1, ~ +/-30%
    "heating_efficiency": ("lognormal", 0.0, 0.07),     # x1, ~ +/-7%
}


def _input_keys(base: TwinParams) -> list[str]:
    """TwinParams physics keys, plus the extra keys an ExtendedTwinParams carries."""
    return [*PHYSICS_PARAM_KEYS, *(k for k in EXTRA_PARAM_KEYS if getattr(base, k, None) is not None)]


def prior_samples(base: TwinParams, n_samples: int = 1000, seed: int = 42,
                  priors: Optional[dict[str, tuple]] = None) -> PosteriorSamples:
    """Samples from the prior. ``priors`` maps a key to (kind, a, b) or (kind, a, b, lower, upper): 'lognormal'
    multiplies the base value by lognormal(a, b); 'normal_add' adds normal(a, b). Bounds (None = open) are enforced
    by clipping, so no sample leaves the physical range (e.g. a gas boiler above 100%). Defaults to PRIOR."""
    rng = np.random.default_rng(seed)
    table = PRIOR if priors is None else priors
    params: dict[str, np.ndarray] = {}
    for key in _input_keys(base):
        v = float(getattr(base, key))
        spec = table.get(key)
        if spec is None:
            params[key] = np.full(n_samples, v, dtype=np.float32)
            continue
        if spec[0] == "lognormal":
            x = v * rng.lognormal(spec[1], spec[2], n_samples)
        else:
            x = v + rng.normal(spec[1], spec[2], n_samples)
        lower, upper = (spec[3], spec[4]) if len(spec) >= 5 else (None, None)
        if lower is not None or upper is not None:
            x = np.clip(x, -np.inf if lower is None else lower, np.inf if upper is None else upper)
        if key == "heating_efficiency" and getattr(base, "heating_fuel", "electricity") != "electricity":
            x = np.minimum(x, 0.99)  # combustion: never above 100%, whichever prior table is used
        params[key] = x.astype(np.float32)
    return PosteriorSamples(property_id=base.property_id, n_samples=n_samples, params=params, base=base, seed=seed)


def _batched_params(samples: PosteriorSamples, mode: str) -> dict[str, jnp.ndarray]:
    out = {}
    keys = _input_keys(samples.base)
    if samples.inputs is not None:  # calibrated by P2: only what P2's fit used, no twin-only defaults (thermal mass)
        keys = [k for k in keys if k in PHYSICS_PARAM_KEYS or k in samples.params]
    for key in keys:
        if mode == "standard" and key in BEHAVIOUR_KEYS:
            values = np.full(samples.n_samples, float(getattr(samples.base, key)), dtype=np.float32)
        elif key in samples.params:
            values = np.asarray(samples.params[key], dtype=np.float32)
        else:  # not calibrated: the prior value repeated
            values = np.full(samples.n_samples, float(getattr(samples.base, key)), dtype=np.float32)
        out[key] = jnp.asarray(values)
    return out


# Standard occupancy for the rating when the physics has no "standard" mode of its own (P2's physics_l1): the wrapper
# fixes the behaviour (same values the stub uses) and runs it in its normal mode.
STANDARD_SETPOINT_C, STANDARD_HEATING_HOURS = 19.0, 11.0


def _months_from(moy: np.ndarray, start_year: int = 2025) -> list[str]:
    """Chronological YYYY-MM for a run of month-of-year values (the year rolls over when the month goes back)."""
    out, year, prev = [], start_year, 0
    for m in (int(x) for x in moy):
        if m <= prev:
            year += 1
        out.append(f"{year}-{m:02d}")
        prev = m
    return out


def _needs_electricity_layer(samples: PosteriorSamples) -> bool:
    for key in ("pv_kwp", "battery_kwh"):
        if float(getattr(samples.base, key, 0.0) or 0.0) > 0 or np.any(np.asarray(samples.params.get(key, 0.0)) > 0):
            return True
    return False


def electricity_layer(base: TwinParams, weather: dict[str, np.ndarray]) -> dict:
    """P2's half-hourly PV/battery inputs, generated in development mode from the monthly weather and the home's PV
    angles (labelled synthetic, not PVGIS). Prepared once per home and shared by every sample."""
    from contracts.electricity import ElectricityIntake
    from contracts.models import WeatherMonth
    from greentwin.twin.electricity_inputs import prepare_electricity_inputs

    months = _months_from(np.asarray(weather["month_of_year"]))
    area = getattr(base, "postcode_district", None) or "UNKNOWN"
    wm = [WeatherMonth(area_code=area, month=m, hdd_15_5=float(h), mean_temp_c=float(t), ghi_kwh_m2=float(g),
                       source="synthetic")
          for m, h, t, g in zip(months, weather["hdd_15_5"], weather["mean_temp_c"], weather["ghi_kwh_m2"])]
    # The twin's battery round trip (generated, never asked) split evenly between charging and discharging.
    leg = float(getattr(base, "battery_round_trip_efficiency", 0.9)) ** 0.5
    intake = ElectricityIntake(battery_charge_efficiency=leg, battery_discharge_efficiency=leg)
    return prepare_electricity_inputs(base, wm, development=True, intake=intake).numeric_arrays()


def simulate(samples: PosteriorSamples, weather: dict[str, np.ndarray], fuel: FuelConfig,
             mode: Literal["calibrated", "standard"] = "calibrated",
             run_physics: Optional[RunPhysics] = None) -> tuple[TwinRun, str]:
    """Physics on every sample. Returns the ``TwinRun`` and the engine name ('physics_l1', 'physics_stub' or
    'custom')."""
    fn, engine = (run_physics, "custom") if run_physics is not None else get_run_physics()
    params = _batched_params(samples, mode)
    per_sample = samples.inputs is not None
    if per_sample:  # each draw's own monthly weather and schedules (P2 posterior); replaces the shared weather
        weather = {k: np.asarray(v).mean(axis=0) for k, v in samples.inputs.items()}
        w_each = {k: jnp.asarray(v) for k, v in samples.inputs.items()}
    w = {k: jnp.asarray(v) for k, v in weather.items()}
    run_mode = mode
    if engine not in ("physics_stub", "custom"):
        if mode == "standard":
            n = samples.n_samples
            params["setpoint_c"] = jnp.full(n, STANDARD_SETPOINT_C, dtype=jnp.float32)
            params["heating_hours_per_day"] = jnp.full(n, STANDARD_HEATING_HOURS, dtype=jnp.float32)
            run_mode = "calibrated"  # P2 alias for "run these parameters as given"
        if _needs_electricity_layer(samples):
            w["electricity_inputs"] = jax.tree_util.tree_map(jnp.asarray, electricity_layer(samples.base, weather))
    if per_sample:
        shared = {k: v for k, v in w.items() if k not in w_each}  # e.g. the electricity layer
        batched = jax.jit(jax.vmap(lambda p, wi: fn(p, {**shared, **wi}, run_mode, fuel)))
        out = batched(params, w_each)
    else:
        batched = jax.jit(jax.vmap(lambda p: fn(p, w, run_mode, fuel)))
        out = batched(params)

    missing = [k for k in (*TOTAL_KEYS, *MONTHLY_KEYS) if k not in out]
    if missing:
        raise KeyError(f"run_physics output is missing {missing} (see docs/digitaltwinvariables.md section 5)")
    elements = out.get("element_heat_loss_w_per_k", {})
    run = TwinRun(
        **{k: np.asarray(out[k]) for k in TOTAL_KEYS},
        monthly_electricity_kwh=np.asarray(out["monthly_electricity_kwh"]),
        monthly_gas_kwh=np.asarray(out["monthly_gas_kwh"]),
        element_heat_loss_w_per_k={k: np.asarray(elements[k]) for k in ELEMENTS if k in elements},
    )
    return run, engine


def simulate_one(params: TwinParams, weather: dict[str, np.ndarray], fuel: FuelConfig,
                 mode: Literal["calibrated", "standard"] = "calibrated") -> dict[str, float]:
    """Single deterministic run (no uncertainty), handy for debugging and quick checks."""
    fn, _ = get_run_physics()
    p = {k: jnp.asarray(float(getattr(params, k))) for k in _input_keys(params)}
    out = fn(p, {k: jnp.asarray(v) for k, v in weather.items()}, mode, fuel)
    return {k: float(out[k]) for k in TOTAL_KEYS}
