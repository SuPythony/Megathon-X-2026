"""The twin on P2's physics and calibration: thermal mass, heater scale and battery efficiency reach the physics;
metabolic gains follow SAP; calibrate() runs P2's Bayesian calibration and keeps each draw's own inputs."""

import jax.numpy as jnp
import numpy as np
import pytest

from greentwin.twin.household import occupancy
from greentwin.twin.inputs import build_physics_inputs
from greentwin.twin.physics import run_physics
from greentwin.twin.showcase import load_showcase
from greentwin.twin.simulate import prior_samples, simulate


@pytest.fixture(scope="module")
def sh2():
    sc = load_showcase("SH2")
    return sc, build_physics_inputs(sc.prop, layout=sc.layout, usage=sc.usage)


def _gas(pkg, **extra):
    p = {k: jnp.asarray(v) for k, v in pkg.physics_params().items()}
    p.pop("heat_capacity_kj_per_k", None)
    p.update({k: jnp.asarray(v) for k, v in extra.items()})
    w = {k: jnp.asarray(v) for k, v in pkg.physics_timeseries().items()}
    return float(run_physics(p, w, "calibrated", pkg.fuel_config())["gas_kwh"])


def test_thermal_mass_is_opt_in_and_heavier_holds_heat(sh2):
    _, pkg = sh2
    steady = _gas(pkg)
    light, heavy = _gas(pkg, heat_capacity_kj_per_k=5_000.0), _gas(pkg, heat_capacity_kj_per_k=60_000.0)
    assert steady > 0 and light != steady  # without the key the steady-state model is unchanged
    assert heavy > light  # intermittent heating saves less in a heavy house: it cools less while off


def test_heater_scale_and_battery_efficiency_reach_the_physics(sh2):
    sc, pkg = sh2
    sec = dict(pkg.physics_timeseries(), secondary_heating_kwh_monthly=np.full(12, 100.0, dtype=np.float32))
    p = {k: jnp.asarray(v) for k, v in pkg.physics_params().items()}
    elec = lambda scale: float(run_physics(dict(p, secondary_heating_scale=jnp.asarray(scale)),  # noqa: E731
                                           {k: jnp.asarray(v) for k, v in sec.items()}, "calibrated",
                                           pkg.fuel_config())["electricity_import_kwh"])
    assert elec(2.0) > elec(0.5)
    base = pkg.params.model_copy(update={"pv_kwp": 4.0, "battery_kwh": 10.0})
    run = lambda rte: simulate(prior_samples(base.model_copy(update={"battery_round_trip_efficiency": rte}), 1,  # noqa: E731
                                             priors={}), pkg.physics_timeseries(), pkg.fuel_config())
    (r1, engine), (r2, _) = run(1.0), run(0.7)
    assert engine == "physics_l1" and r2.electricity_import_kwh.sum() > r1.electricity_import_kwh.sum()


def test_metabolic_gains_follow_sap_for_a_typical_household():
    sc = load_showcase("SH2")
    occ = occupancy(sc.prop, 3, "part_day", 3)
    assert occ.metabolic_gains_w - 40 * 3 == pytest.approx(20 * 3)  # SAP / P2: net 20 W per person
    assert occupancy(sc.prop, 3, "home_all_day", 3).metabolic_gains_w > occ.metabolic_gains_w


def test_calibrate_runs_p2_and_reproduces_its_posterior(sh2):
    from contracts.calibration import CalibrationConfig
    from greentwin.twin.twin import GreenDigitalTwin

    sc, _ = sh2
    twin = GreenDigitalTwin(n_samples=64, calibration_config=CalibrationConfig(
        method="svi", nuisance_samples=4, svi_steps=600, samples=100, chains=1))
    result, samples = twin.calibrate(sc.prop, sc.usage, [])
    assert result.method == "svi" and result.months_used == 12 and result.provenance.engine == "physics_l1"
    assert samples.inputs is not None and samples.inputs["mean_temp_c"].shape == (64, 12)
    run = twin.simulate(samples, [])
    gas, elec = np.asarray(run.gas_kwh), np.asarray(run.electricity_import_kwh)
    assert np.isfinite(gas).all() and np.isfinite(elec).all()
    metered_gas = sum(r.gas_kwh for r in sc.usage)
    assert np.percentile(gas, 5) <= metered_gas <= np.percentile(gas, 95)
    # A retrofit reruns the calibrated draws with their own inputs
    from contracts.models import RetrofitAction
    loft = twin.simulate(samples, [RetrofitAction(retrofit_id="loft_insulation")])
    assert np.median(loft.gas_kwh) < np.median(gas)


def test_no_bills_stays_prior_only():
    from greentwin.twin.twin import GreenDigitalTwin

    sc = load_showcase("SH2")
    result, samples = GreenDigitalTwin(n_samples=32).calibrate(sc.prop, [], [])
    assert result.method == "prior_only" and samples.inputs is None
