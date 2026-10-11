"""Calibration correctness: held-out isolation, exact facts, dependency and replay."""

import copy

import jax
import numpy as np
import pytest

from contracts.calibration import CalibrationConfig, MeterObservation
from greentwin.bayes.calibration import (
    CalibrationFit,
    CalibrationModel,
    _apply_constants,
    _parameter_specs,
    calibrate_household,
    observations_from_package,
    predictive_inputs,
    scenario_request,
)
from greentwin.bayes.io import load_calibration, save_calibration
from greentwin.twin.ensemble import paired_differences, simulate_ensemble
from greentwin.twin.missing_inputs import generate_missing_inputs
from greentwin.twin.physics import run_physics
from greentwin.twin.physics_api import FuelConfig
from scripts.evaluate_bayesian_calibration import experiment_request


@pytest.fixture(scope="module")
def problem():
    request, prior = experiment_request()
    bank = generate_missing_inputs(request, 4, prior=prior)
    params = {k: a[0] for k, a in bank.parameters.items()} | {"wall_u": 0.85, "hot_water_efficiency": 0.8}
    truth = jax.device_get(run_physics(params, {k: a[0] for k, a in bank.weather.items()},
                                     "uncalibrated", FuelConfig("mains_gas", "mains_gas")))
    meters = [MeterObservation(month=month, gas_kwh=float(truth["monthly_gas_kwh"][i]),
                              electricity_import_kwh=float(truth["monthly_electricity_kwh"][i]),
                              use_for_calibration=i not in (2, 5, 8, 11))
              for i, month in enumerate(request.months)]
    return request, prior, bank, meters


@pytest.fixture(scope="module")
def fitted(problem):
    request, prior, _, meters = problem
    return calibrate_household(request, meters, prior=prior, config=CalibrationConfig(
        parameters=["wall_u", "hot_water_efficiency"], nuisance_samples=4, warmup=150,
        samples=200, chains=2, max_r_hat=1.2, min_effective_samples=10,
    ))


def test_held_out_values_and_nulls_never_enter_likelihood(problem):
    _, _, bank, rows = problem
    config = CalibrationConfig(parameters=["wall_u"], nuisance_samples=4)
    specs, _ = _parameter_specs(bank, config)
    before = CalibrationModel(bank, rows, config, specs)
    edited = [row.model_copy(update={"gas_kwh": 1e8, "electricity_import_kwh": None})
              if not row.use_for_calibration else row for row in rows]
    after = CalibrationModel(bank, edited, config, specs)
    np.testing.assert_array_equal(before.log_likelihood({"wall_u": 0.85}),
                                  after.log_likelihood({"wall_u": 0.85}))
    assert len(before.entries) == 16
    missing = CalibrationModel(bank, [MeterObservation(month=bank.request.months[0], gas_kwh=0)], config, specs)
    assert len(missing.entries) == 1 and missing.values[0] == 0
    with pytest.raises(ValueError, match="nonmissing"):
        CalibrationModel(bank, [MeterObservation(month=bank.request.months[0])], config, specs)


def test_exact_answers_not_fitted_and_observations_aligned(problem):
    _, _, bank, meters = problem
    config = CalibrationConfig(parameters=["wall_u", "heating_efficiency", "setpoint_c"], nuisance_samples=4)
    specs, fixed = _parameter_specs(bank, config)
    assert set(specs) == {"wall_u"}
    assert set(fixed) == {"heating_efficiency", "setpoint_c"}
    with pytest.raises(ValueError, match="exact"):
        _parameter_specs(bank, CalibrationConfig(parameters=["heating_efficiency"]))
    with pytest.raises(ValueError, match="Duplicate"):
        CalibrationModel(bank, [meters[0], meters[0]], config, specs)
    with pytest.raises(ValueError, match="aligned"):
        CalibrationModel(bank, [MeterObservation(month="2024-01", gas_kwh=10)], config, specs)


def test_demand_constants_keep_internal_heat_coupled():
    p = {"baseload_kwh_per_day": 10., "retained_electric_heat_fraction": .8, "internal_gains_w": 400.}
    changed, _ = _apply_constants(p, {}, {}, {"baseload_kwh_per_day": 5.}, {})
    assert changed["internal_gains_w"] == pytest.approx(400 - .8 * 5 * 1000 / 24)
    exact, _ = _apply_constants(p, {}, {}, {"baseload_kwh_per_day": 5.}, {"internal_gains_w": 400})
    assert exact["internal_gains_w"] == 400


def test_nuts_recovers_constants_and_preserves_user_facts(fitted):
    assert fitted.result.status == "converged"
    assert fitted.result.diagnostics["divergences"] == 0
    assert fitted.result.observation_count == 16
    assert abs(fitted.result.posterior_parameters["wall_u"].mean - .85) < .08
    assert abs(fitted.result.posterior_parameters["hot_water_efficiency"].mean - .8) < .05
    for k in fitted.result.fitted_parameters:
        assert fitted.result.posterior_parameters[k].std < fitted.result.prior_parameters[k].std
    inputs = predictive_inputs(fitted, n_samples=8)
    np.testing.assert_array_equal(inputs.parameters["setpoint_c"], 20)
    np.testing.assert_array_equal(inputs.parameters["heating_efficiency"], .9)
    assert inputs.summary.calibration_status == "performed"
    repeat = predictive_inputs(fitted, n_samples=8)
    for k in inputs.parameters:
        np.testing.assert_array_equal(inputs.parameters[k], repeat.parameters[k])


def test_posterior_replay_and_hash_verification(fitted, tmp_path):
    path = save_calibration(fitted, tmp_path / "calibration")
    loaded = load_calibration(path)
    a, b = predictive_inputs(fitted, n_samples=4), predictive_inputs(loaded, n_samples=4)
    for key in a.parameters:
        np.testing.assert_array_equal(a.parameters[key], b.parameters[key])
    (path / "calibration.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_calibration(path)


def test_full_posterior_physics_and_retrofit_use_same_draws(fitted):
    before_inputs = predictive_inputs(fitted, n_samples=4)
    request = fitted.bank.request.model_copy(update={"user_inputs": fitted.bank.request.user_inputs | {"wall_u": .3}})
    after_inputs = predictive_inputs(fitted, request=request, n_samples=4)
    np.testing.assert_array_equal(after_inputs.parameters["wall_u"], .3)
    np.testing.assert_array_equal(before_inputs.parameters["hot_water_efficiency"],
                                  after_inputs.parameters["hot_water_efficiency"])
    before, after = simulate_ensemble(before_inputs, batch_size=4), simulate_ensemble(after_inputs, batch_size=4)
    assert before.result.uncertainty_basis == "posterior_predictive"
    assert before.result.calibration_id == fitted.result.calibration_id
    assert paired_differences(before, after)["gas_kwh"].mean > 0
    assert max(before.result.max_conservation_error_kwh.values()) < .01
    prior = simulate_ensemble(predictive_inputs(fitted, n_samples=4, posterior=False), batch_size=4)
    with pytest.raises(ValueError, match="posterior sample population"):
        paired_differences(prior, before)


def test_bad_diagnostics_require_explicit_override(fitted):
    status = fitted.result.status
    fitted.result.status = "needs_review"
    try:
        with pytest.raises(ValueError, match="diagnostics"):
            predictive_inputs(fitted, n_samples=4)
        assert predictive_inputs(fitted, n_samples=4, allow_needs_review=True).summary.calibration_status == "performed"
    finally:
        fitted.result.status = status


def test_handoff_missing_is_not_zero():
    rows = observations_from_package({"meter": {"months": ["2025-01", "2025-02"],
        "electricity_kwh": [100, None], "gas_kwh": [None, 0], "estimated": [False, True]}})
    assert rows[0].gas_kwh is None and rows[1].gas_kwh == 0
    assert rows[1].electricity_import_kwh is None and rows[1].estimated


def test_svi_runs_and_returns_joint_draws(problem):
    request, prior, _, meters = problem
    fit = calibrate_household(request, meters, prior=prior, config=CalibrationConfig(
        method="svi", parameters=["wall_u"], nuisance_samples=4, samples=20, chains=1, svi_steps=100,
    ))
    assert fit.result.status == "approximate"
    assert fit.samples["wall_u"].shape == (20,)
    assert np.isfinite(fit.samples["wall_u"]).all()


def test_nuisance_indices_are_conditioned_on_meters(fitted):
    bank = copy.deepcopy(fitted.bank)
    bank.weather["mean_temp_c"][1:] += 10
    model = CalibrationModel(bank, fitted.result.observations, fitted.result.config, fitted.result.prior_specs)
    conditional = CalibrationFit(fitted.result, bank, fitted.samples, model)
    inputs = predictive_inputs(conditional, n_samples=32)
    # The alternate weather trajectories predict very different winter meters.
    assert np.all(inputs.latent["interval_seed_index"] == 0)
    np.testing.assert_array_equal(inputs.weather["mean_temp_c"],
                                  np.tile(bank.weather["mean_temp_c"][0], (32, 1)))


def test_equipment_scenario_discards_old_efficiency_and_preserves_house_posterior(fitted):
    request = scenario_request(fitted, {"heating_system": "air_source_heat_pump"})
    assert "heating_efficiency" not in request.user_inputs
    inputs = predictive_inputs(fitted, request=request, n_samples=8)
    baseline = predictive_inputs(fitted, n_samples=8)
    np.testing.assert_array_equal(inputs.parameters["wall_u"], baseline.parameters["wall_u"])
    assert np.all(inputs.parameters["heating_efficiency"] >= 1)
    assert "hot_water_efficiency" not in inputs.summary.calibration_diagnostics["inherited_constants"]
    # Replacing space heating must not discard an independent DHW device's fact.
    independent_bank = copy.deepcopy(fitted.bank)
    independent_bank.prior.equipment["hot_water_source"] = "heat_pump_cylinder"
    independent_bank.request.user_inputs.update({"hot_water_source": "heat_pump_cylinder",
                                               "hot_water_efficiency": 2.2})
    independent_fit = CalibrationFit(fitted.result, independent_bank, fitted.samples, fitted.model)
    changed = scenario_request(independent_fit, {"heating_system": "air_source_heat_pump"})
    assert changed.user_inputs["hot_water_efficiency"] == 2.2


def test_pv_battery_likelihood_is_differentiable_and_uses_grid_import():
    from tests.test_uncertain_twin import request

    req = request(occupants=3, thermostat_c=20, pv_kwp=2, battery_kwh=2)
    req = req.model_copy(update={"months": ["2025-01"]})
    bank = generate_missing_inputs(req, 2)
    config = CalibrationConfig(parameters=["wall_u"], nuisance_samples=2)
    specs, _ = _parameter_specs(bank, config)
    model = CalibrationModel(bank, [MeterObservation(month="2025-01", electricity_import_kwh=200)], config, specs)
    assert model.interval_likelihood
    prediction = model.predict({"wall_u": 1.2})
    assert np.all(np.asarray(prediction["monthly_electricity_kwh"]) <=
                  np.asarray(prediction["monthly_electricity_demand_kwh"]) + .001)
    gradient = jax.grad(lambda u: jax.scipy.special.logsumexp(model.log_likelihood({"wall_u": u})))(1.2)
    assert np.isfinite(gradient)
