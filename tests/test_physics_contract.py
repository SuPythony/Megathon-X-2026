"""The run_physics contract (docs/digitaltwinvariables.md section 9), checked against whichever physics is installed.

Physics owner: drop your ``greentwin/twin/physics.py`` in and run ``pytest tests/test_physics_contract.py``. The same
tests run against the stub, so the pipeline and your model are held to one contract.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from contracts.models import EnergyUsageMonth, Property
from greentwin.twin.params import fuel_config, params_with_sources
from greentwin.twin.physics_api import (
    ELEMENTS,
    MONTHLY_KEYS,
    PHYSICS_PARAM_KEYS,
    TOTAL_KEYS,
    FuelConfig,
    get_run_physics,
)
from greentwin.twin.physics_stub import run_physics as stub_run
from greentwin.twin.simulate import prior_samples, simulate
from greentwin.twin.weather import typical_year, weather_arrays

EX = Path(__file__).resolve().parents[1] / "contracts" / "examples"
INSTALLED, ENGINE = get_run_physics()
IMPLEMENTATIONS = {"stub": stub_run} | ({"physics": INSTALLED} if ENGINE == "physics" else {})
GAS = FuelConfig("mains_gas", "mains_gas")
ELEC = FuelConfig("electricity", "electricity")


def _prop(pid):
    return Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))


def _inputs(pid="SH2"):
    params, _ = params_with_sources(_prop(pid))
    p = {k: jnp.asarray(float(getattr(params, k))) for k in PHYSICS_PARAM_KEYS}
    w = {k: jnp.asarray(v) for k, v in weather_arrays(typical_year("M14")).items()}
    return p, w, fuel_config(_prop(pid))


@pytest.fixture(params=sorted(IMPLEMENTATIONS))
def run(request):
    return IMPLEMENTATIONS[request.param]


def test_outputs_complete_and_consistent(run):
    p, w, fuel = _inputs()
    out = run(p, w, "calibrated", fuel)
    for k in (*TOTAL_KEYS, *MONTHLY_KEYS):
        assert k in out, k
    assert set(out["element_heat_loss_w_per_k"]) == set(ELEMENTS)
    assert out["monthly_gas_kwh"].shape == (12,)
    assert float(out["electricity_import_kwh"]) == pytest.approx(float(out["monthly_electricity_kwh"].sum()), rel=1e-5)
    assert float(out["gas_kwh"]) == pytest.approx(float(out["monthly_gas_kwh"].sum()), rel=1e-5)
    for k in TOTAL_KEYS:
        assert float(out[k]) >= 0, k
    assert float(out["export_kwh"]) <= float(out["pv_generation_kwh"]) + 1e-6


def test_zero_cases(run):
    p, w, _ = _inputs()
    out = run(p, w, "calibrated", GAS)
    assert float(out["pv_generation_kwh"]) == 0 and float(out["export_kwh"]) == 0
    assert float(run(p, w, "calibrated", ELEC)["gas_kwh"]) == 0


@pytest.mark.parametrize("key,direction", [
    ("wall_u", -1), ("roof_u", -1), ("window_u", -1), ("air_changes_per_hour", -1), ("setpoint_c", -1),
    ("heating_efficiency", +1),
])
def test_monotonic(run, key, direction):
    p, w, _ = _inputs()
    base = float(run(p, w, "calibrated", GAS)["gas_kwh"])
    changed = dict(p)
    changed[key] = p[key] * (0.8 if direction < 0 else 1.25) if key != "setpoint_c" else p[key] - 1.0
    assert float(run(changed, w, "calibrated", GAS)["gas_kwh"]) <= base + 1e-6


def test_pv_reduces_import(run):
    p, w, _ = _inputs()
    with_pv = dict(p, pv_kwp=jnp.asarray(3.5))
    assert float(run(with_pv, w, "calibrated", GAS)["electricity_import_kwh"]) < float(run(p, w, "calibrated", GAS)["electricity_import_kwh"])


def test_jit_vmap_grad(run):
    p, w, fuel = _inputs()
    batched = {k: jnp.full((1000,), v) for k, v in p.items()}
    out = jax.jit(jax.vmap(lambda q: run(q, w, "calibrated", fuel)))(batched)
    assert out["gas_kwh"].shape == (1000,)
    for key in ("heat_loss_multiplier", "setpoint_c", "heating_efficiency", "baseload_kwh_per_day"):
        g = jax.grad(lambda x: run(dict(p, **{key: x}), w, "calibrated", fuel)["gas_kwh"]
                     + run(dict(p, **{key: x}), w, "calibrated", fuel)["electricity_import_kwh"])(p[key])
        assert np.isfinite(float(g)), key


def test_standard_mode_ignores_calibrated_behaviour():
    params, _ = params_with_sources(_prop("SH2"))
    samples = prior_samples(params, n_samples=50)
    for k in ("heat_loss_multiplier", "setpoint_c", "heating_efficiency"):
        samples.params[k] = np.full(50, float(getattr(params, k)), dtype=np.float32)  # only baseload varies
    w = weather_arrays(typical_year("BS16"))
    calibrated, _ = simulate(samples, w, GAS, "calibrated")
    standard, _ = simulate(samples, w, GAS, "standard")
    assert np.std(calibrated.electricity_import_kwh) > 1
    assert np.std(standard.electricity_import_kwh) < 1e-3


@pytest.mark.skipif(ENGINE != "physics", reason="plausibility is the real physics model's 'done when', not the stub's")
@pytest.mark.parametrize("pid", ["SH1", "SH2", "SH3"])
def test_showcase_within_15pct_of_metered(pid):
    import json

    params, _ = params_with_sources(_prop(pid))
    rows = [EnergyUsageMonth.model_validate(r) for r in json.loads((EX / f"{pid}_usage_2025.json").read_text())]
    w = weather_arrays(typical_year(_prop(pid).location.postcode_district))
    run, _ = simulate(prior_samples(params, n_samples=1), w, fuel_config(_prop(pid)))
    metered = sum((r.gas_kwh or 0) + (r.electricity_import_kwh or 0) for r in rows)
    modelled = float(run.gas_kwh[0] + run.electricity_import_kwh[0])
    assert abs(modelled - metered) / metered <= 0.15
