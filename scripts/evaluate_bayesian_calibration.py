"""Controlled prior/posterior recovery with frozen held-out months and known truth."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from contracts.calibration import CalibrationConfig, MeterObservation
from contracts.models import Property
from contracts.uncertain_twin import HouseholdRequest
from greentwin.bayes.calibration import calibrate_household, predictive_inputs
from greentwin.bayes.io import save_calibration
from greentwin.twin.ensemble import paired_differences, simulate_ensemble
from greentwin.twin.ensemble_io import new_output, save_inputs, save_result
from greentwin.twin.missing_inputs import generate_missing_inputs
from greentwin.twin.physics import run_physics
from greentwin.twin.physics_api import FuelConfig
from greentwin.twin.priors import build_priors


def experiment_request():
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH2_property.json").read_text())
    request = HouseholdRequest(
        property_id="SH2", property=prop, allow_development_weather=True,
        data_version="bayesian-controlled-example/0.1",
        user_inputs={
            "occupants": 4, "daytime_occupancy": "part_day", "thermostat_c": 20,
            "heating_hours_per_day": 8, "heating_efficiency": 0.9,
            "baseload_kwh_per_day": 7, "hot_water_kwh_per_day": 6,
            "cooking_fuel": "mains_gas", "gas_cooking_kwh_per_day": 1,
            "cylinder_present": False, "retained_electric_heat_fraction": 0.85,
            "baseload_multiplier": [1] * 12, "hot_water_multiplier": [1] * 12,
            "monthly_temperature_c": [4, 5, 7, 10, 13, 17, 19, 18, 15, 11, 7, 4],
            "monthly_ghi_kwh_m2": [18, 30, 65, 100, 135, 150, 145, 120, 85, 45, 22, 15],
        },
    )
    prior = build_priors(request)
    # This controlled survey scenario isolates two unknown physical constants.
    # These are deliberate synthetic experiment conditions, not real surveys.
    for key, spec in list(prior.marginals.items()):
        if key not in ("wall_u", "hot_water_efficiency") and spec.family != "categorical":
            prior.marginals[key] = spec.model_copy(update={
                "family": "exact", "scale": 0,
                "source": "Controlled synthetic experiment: other physical inputs held fixed",
            })
    return request, prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=400, help="Retained NUTS draws per chain")
    parser.add_argument("--predictive-samples", type=int, default=128)
    args = parser.parse_args()
    output = new_output(args.output); output.mkdir(parents=True)
    request, prior = experiment_request()
    truth_constants = {"wall_u": 0.85, "hot_water_efficiency": 0.80}
    truth_inputs = generate_missing_inputs(request, 2, prior=prior)
    params = {k: a[0] for k, a in truth_inputs.parameters.items()} | truth_constants
    weather = {k: a[0] for k, a in truth_inputs.weather.items()}
    truth = jax.device_get(run_physics(params, weather, "uncalibrated", FuelConfig("mains_gas", "mains_gas")))
    rng = np.random.default_rng(731)
    held_out = {"2025-03", "2025-06", "2025-09", "2025-12"}
    meters = [
        MeterObservation(month=month,
                         gas_kwh=float(truth["monthly_gas_kwh"][i] * np.exp(rng.normal(0, 0.02))),
                         electricity_import_kwh=float(truth["monthly_electricity_kwh"][i]
                                                      * np.exp(rng.normal(0, 0.02))),
                         use_for_calibration=month not in held_out,
                         source="Controlled synthetic meters: known JAX truth plus seeded 2% log noise")
        for i, month in enumerate(request.months)
    ]
    (output / "request.json").write_text(request.model_dump_json(indent=2) + "\n")
    (output / "meters.json").write_text(json.dumps([o.model_dump() for o in meters], indent=2) + "\n")
    fit = calibrate_household(request, meters, prior=prior, config=CalibrationConfig(
        parameters=["wall_u", "hot_water_efficiency"], nuisance_samples=4,
        warmup=args.warmup, samples=args.samples, chains=2,
    ))
    save_calibration(fit, output / "calibration")
    print(f"Fit: {fit.result.status}; {fit.result.elapsed_seconds:.2f}s; {fit.result.diagnostics}", flush=True)
    runs = {}
    for name, posterior in (("prior", False), ("posterior", True)):
        inputs = predictive_inputs(fit, n_samples=args.predictive_samples, posterior=posterior)
        save_inputs(inputs, output / f"{name}_inputs")
        runs[name] = simulate_ensemble(inputs, batch_size=8)
        save_result(runs[name], output / name)
        print(f"{name}: gas mean {runs[name].result.totals['gas_kwh'].mean:.2f} kWh", flush=True)
    scenario = request.model_copy(update={"user_inputs": request.user_inputs | {
        "wall_u": 0.3, "roof_u": 0.16, "window_u": 1.4,
    }})
    scenario_inputs = predictive_inputs(fit, request=scenario, n_samples=args.predictive_samples)
    scenario_run = simulate_ensemble(scenario_inputs, batch_size=8)
    save_inputs(scenario_inputs, output / "retrofit_inputs")
    save_result(scenario_run, output / "retrofit")
    evidence = {
        "experiment": "controlled synthetic parameter recovery and held-out month prediction",
        "noise_seed": 731, "synthetic_meter_log_noise_sd": 0.02,
        "n_predictive_samples": args.predictive_samples, "truth_constants": truth_constants,
        "truth_gas_annual_kwh": float(truth["gas_kwh"]),
        "truth_space_heating_annual_kwh": float(truth["space_heating_kwh"]),
        "truth_heat_loss_coefficient_w_per_k": float(truth["heat_loss_coefficient_w_per_k"]),
        "fit_status": fit.result.status, "diagnostics": fit.result.diagnostics,
        "fit_seconds": fit.result.elapsed_seconds, "training_months": fit.result.training_months,
        "held_out_months": fit.result.held_out_months,
        "constants": {k: {"truth": truth_constants[k],
                          "prior": fit.result.prior_parameters[k].model_dump(),
                          "posterior": fit.result.posterior_parameters[k].model_dump()}
                      for k in truth_constants},
        "outputs": {name: {k: d.model_dump() for k, d in run.result.totals.items()}
                    for name, run in runs.items()},
        "monthly": [], "accuracy": {},
        "retrofit_paired_savings": {k: d.model_dump() for k, d in
                                   paired_differences(runs["posterior"], scenario_run).items()},
        "max_conservation_error_kwh": {name: run.result.max_conservation_error_kwh
                                       for name, run in (runs | {"retrofit": scenario_run}).items()},
    }
    for i, month in enumerate(request.months):
        evidence["monthly"].append({
            "month": month, "split": "held_out" if month in held_out else "training",
            "true_gas_kwh": float(truth["monthly_gas_kwh"][i]), "observed_gas_kwh": meters[i].gas_kwh,
            **{name: run.result.monthly["monthly_gas_kwh"][i].model_dump() for name, run in runs.items()},
        })
    for split in ("training", "held_out", "all"):
        rows = [r for r in evidence["monthly"] if split == "all" or r["split"] == split]
        measured = np.array([r["observed_gas_kwh"] for r in rows])
        truth_array = np.array([r["true_gas_kwh"] for r in rows])
        evidence["accuracy"][split] = {}
        for name in runs:
            pred = np.array([r[name]["mean"] for r in rows])
            evidence["accuracy"][split][name] = {
                "mae_vs_meter_kwh": float(np.abs(pred - measured).mean()),
                "rmse_vs_meter_kwh": float(np.sqrt(np.mean((pred - measured)**2))),
                "mape_vs_meter_pct": float(100 * np.mean(np.abs(pred - measured) / measured)),
                "mae_vs_truth_kwh": float(np.abs(pred - truth_array).mean()),
                "physical_range_90_coverage_truth_pct": float(100 * np.mean([
                    r[name]["percentiles"]["p5"] <= r["true_gas_kwh"] <= r[name]["percentiles"]["p95"]
                    for r in rows
                ])),
            }
    # Meter-observation predictive intervals add the declared observation model;
    # downstream physics outputs retain physical uncertainty without bill noise.
    predictive_rng = np.random.default_rng(884)
    for name, run in runs.items():
        energy = run.samples["monthly_gas_kwh"]
        scale = np.sqrt(5**2 + (0.05 * energy)**2)
        observation_draws = energy + scale * predictive_rng.standard_t(4, size=energy.shape)
        evidence.setdefault("meter_predictive_90", {})[name] = {
            month: [float(x) for x in np.percentile(observation_draws[:, i], [5, 95])]
            for i, month in enumerate(request.months)
        }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence["accuracy"], indent=2), flush=True)


if __name__ == "__main__":
    main()
