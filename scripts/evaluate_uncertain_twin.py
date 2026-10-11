"""Reproducible development evidence, not an empirical household accuracy claim."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from contracts.models import Property
from contracts.uncertain_twin import HouseholdRequest
from greentwin.twin.ensemble import paired_differences, simulate_ensemble
from greentwin.twin.ensemble_io import new_output, save_inputs, save_result
from greentwin.twin.missing_inputs import generate_missing_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    output = new_output(args.output)
    output.mkdir(parents=True)
    prop = Property.model_validate_json((ROOT / "contracts/examples/SH2_property.json").read_text())
    req = HouseholdRequest(
        property_id="SH2",
        property=prop,
        allow_development_weather=True,
        user_inputs={
            "occupants": 4,
            "thermostat_c": 20,
            "gas_cooking_kwh_per_day": 1,
            "cooking_fuel": "mains_gas",
            "cylinder_present": False,
        },
    )
    inputs = generate_missing_inputs(req, args.samples)
    save_inputs(inputs, output / "prepared")
    scenarios = {
        "baseline": {},
        "fabric": {"wall_u": 0.3, "roof_u": 0.16, "window_u": 1.4},
        "heat_pump": {"heating_system": "air_source_heat_pump"},
        "fabric_heat_pump_pv_battery": {
            "wall_u": 0.3,
            "roof_u": 0.16,
            "window_u": 1.4,
            "heating_system": "air_source_heat_pump",
            "pv_kwp": 3.5,
            "battery_kwh": 5,
        },
    }
    runs = {}
    for name, changes in scenarios.items():
        scenario = req.model_copy(update={"user_inputs": req.user_inputs | changes})
        samples = inputs if name == "baseline" else generate_missing_inputs(scenario, args.samples)
        run = simulate_ensemble(samples, batch_size=8)
        save_result(run, output / name)
        runs[name] = run
        print(
            f"{name}: {run.elapsed_seconds:.2f}s; all-fuel mean {run.result.totals['all_fuel_purchased_kwh'].mean:.1f} kWh",
            flush=True,
        )
    baseline = runs["baseline"]
    evidence = {
        "n_samples": args.samples,
        "seed": req.seed,
        "months": req.months,
        "calibration_status": "not_performed",
        "empirical_accuracy_evaluated": False,
        "scenarios": {
            k: {
                "runtime_seconds": v.elapsed_seconds,
                "totals": {name: d.model_dump() for name, d in v.result.totals.items()},
                "metrics": {name: d.model_dump() if d else None for name, d in v.result.metrics.items()},
                "max_conservation_error_kwh": v.result.max_conservation_error_kwh,
            }
            for k, v in runs.items()
        },
        "paired_changes": {
            k: {name: d.model_dump() for name, d in paired_differences(baseline, v).items()}
            for k, v in runs.items()
            if k != "baseline"
        },
        "monotonic_fabric_draws": int(
            np.sum(runs["fabric"].samples["space_heating_kwh"] <= baseline.samples["space_heating_kwh"])
        ),
        "mean_convergence": {
            str(n): {
                "gas_mean_kwh": float(baseline.samples["gas_kwh"][:n].mean()),
                "gas_mcse_kwh": float(baseline.samples["gas_kwh"][:n].std(ddof=1) / np.sqrt(n)),
            }
            for n in (16, 32, 64, 128, 256)
            if n <= args.samples
        },
    }
    # Same seed replay uses regenerated inputs, not stored outputs.
    replay_inputs = generate_missing_inputs(req, args.samples)
    evidence["identical_regenerated_parameter_arrays"] = all(
        np.array_equal(a, replay_inputs.parameters[k]) for k, a in inputs.parameters.items()
    )
    # Measure how supplied envelope evidence narrows H, without assuming totals always narrow.
    measured = {
        k: float(inputs.prior.base_parameters[k])
        for k in (
            "wall_u",
            "roof_u",
            "floor_u",
            "window_u",
            "door_u",
            "air_changes_per_hour",
            "thermal_bridging_y",
        )
    }
    better = simulate_ensemble(
        generate_missing_inputs(
            req.model_copy(update={"user_inputs": req.user_inputs | measured}), args.samples
        ),
        batch_size=8,
    )
    save_result(better, output / "known_envelope")
    evidence["information_gain"] = {
        "baseline_h_std_w_per_k": baseline.result.metrics["heat_loss_coefficient_w_per_k"].std,
        "known_envelope_h_std_w_per_k": better.result.metrics["heat_loss_coefficient_w_per_k"].std,
        "std_reduction_pct": 100
        * (
            1
            - better.result.metrics["heat_loss_coefficient_w_per_k"].std
            / baseline.result.metrics["heat_loss_coefficient_w_per_k"].std
        ),
    }
    # Independent per-fuel conversion checks in January, including retained gas cooking.
    fuel_checks = {}
    for system, fuel in [
        ("gas_boiler_condensing", "mains_gas"),
        ("oil_boiler", "oil"),
        ("lpg_boiler", "lpg"),
        ("solid_fuel", "solid"),
        ("district_heating", "district"),
        ("air_source_heat_pump", "electricity"),
    ]:
        short = req.model_copy(
            update={"months": ["2025-01"], "user_inputs": req.user_inputs | {"heating_system": system}}
        )
        s = generate_missing_inputs(short, 8)
        r = simulate_ensemble(s, batch_size=8)
        expected = r.samples["space_heating_kwh"] / s.parameters["heating_efficiency"]
        actual = r.samples["monthly_heating_fuel_kwh"].sum(axis=1)
        fuel_checks[fuel] = {
            "mean_purchased_kwh": r.result.totals["all_fuel_purchased_kwh"].mean,
            "max_conversion_error_kwh": float(np.max(np.abs(expected - actual))),
            "max_conservation_error_kwh": r.result.max_conservation_error_kwh,
            "gas_kwh_mean": r.result.totals["gas_kwh"].mean,
        }
    evidence["fuel_checks"] = fuel_checks
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Evidence saved: {output / 'evidence.json'}", flush=True)


if __name__ == "__main__":
    main()
