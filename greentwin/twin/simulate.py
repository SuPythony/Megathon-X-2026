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
                  priors: Optional[dict[str, tuple[str, float, float]]] = None) -> PosteriorSamples:
    """Samples from the prior. ``priors`` maps a key to (kind, a, b): 'lognormal' multiplies the base value by
    lognormal(a, b); 'normal_add' adds normal(a, b). Defaults to PRIOR."""
    rng = np.random.default_rng(seed)
    table = PRIOR if priors is None else priors
    params: dict[str, np.ndarray] = {}
    for key in _input_keys(base):
        v = float(getattr(base, key))
        spec = table.get(key)
        if spec is None:
            params[key] = np.full(n_samples, v, dtype=np.float32)
        elif spec[0] == "lognormal":
            params[key] = (v * rng.lognormal(spec[1], spec[2], n_samples)).astype(np.float32)
        else:
            params[key] = (v + rng.normal(spec[1], spec[2], n_samples)).astype(np.float32)
    return PosteriorSamples(property_id=base.property_id, n_samples=n_samples, params=params, base=base, seed=seed)


def _batched_params(samples: PosteriorSamples, mode: str) -> dict[str, jnp.ndarray]:
    out = {}
    for key in _input_keys(samples.base):
        if mode == "standard" and key in BEHAVIOUR_KEYS:
            values = np.full(samples.n_samples, float(getattr(samples.base, key)), dtype=np.float32)
        elif key in samples.params:
            values = np.asarray(samples.params[key], dtype=np.float32)
        else:  # not calibrated: the prior value repeated
            values = np.full(samples.n_samples, float(getattr(samples.base, key)), dtype=np.float32)
        out[key] = jnp.asarray(values)
    return out


def simulate(samples: PosteriorSamples, weather: dict[str, np.ndarray], fuel: FuelConfig,
             mode: Literal["calibrated", "standard"] = "calibrated",
             run_physics: Optional[RunPhysics] = None) -> tuple[TwinRun, str]:
    """Physics on every sample. Returns the ``TwinRun`` and the engine name ('physics' or 'physics_stub')."""
    fn, engine = (run_physics, "custom") if run_physics is not None else get_run_physics()
    w = {k: jnp.asarray(v) for k, v in weather.items()}
    batched = jax.jit(jax.vmap(lambda p: fn(p, w, mode, fuel)))
    out = batched(_batched_params(samples, mode))

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
