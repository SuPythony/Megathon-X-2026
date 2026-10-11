"""Fit monthly meters, then run the full twin on joint posterior realizations."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.calibration import CalibrationConfig, MeterObservation
from contracts.uncertain_twin import HouseholdRequest
from greentwin.bayes.calibration import calibrate_household, predictive_inputs, scenario_request
from greentwin.bayes.io import load_calibration, save_calibration
from greentwin.twin.application_history import ApplicationHistory
from greentwin.twin.ensemble import simulate_ensemble
from greentwin.twin.ensemble_io import new_output, save_inputs, save_result
from greentwin.twin.priors import build_priors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "predict"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--meters", type=Path, help="JSON list of nullable MeterObservation rows")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--history", type=Path, help="Optional read-only application-history prior source")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--changes", type=Path, help="Physical scenario edits; never recalibrates hypothetical meters")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-intervals", action="store_true")
    parser.add_argument("--allow-needs-review", action="store_true", help="Explicit diagnostic-only posterior prediction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = new_output(args.output)
        if args.mode == "calibrate":
            if not args.request or args.calibration or args.changes:
                raise ValueError("calibrate requires --request; scenarios use predict --changes")
            request = HouseholdRequest.model_validate_json(args.request.read_text())
            observations = None if not args.meters else [
                MeterObservation.model_validate(row) for row in json.loads(args.meters.read_text())
            ]
            config = CalibrationConfig() if not args.config else CalibrationConfig.model_validate_json(
                args.config.read_text()
            )
            prior = build_priors(request, history=ApplicationHistory(args.history)) if args.history else None
            fit = calibrate_household(request, observations, config=config, prior=prior)
            save_calibration(fit, output)
            print(f"Saved {fit.result.method} calibration: {fit.result.status}; {output}")
            print(f"Constants: {', '.join(fit.result.fitted_parameters)}; meter values: {fit.result.observation_count}")
        else:
            if not args.calibration or args.meters or args.config or args.history:
                raise ValueError("predict requires --calibration; no new meter likelihood is fitted")
            fit = load_calibration(args.calibration)
            request = fit.bank.request if not args.request else HouseholdRequest.model_validate_json(
                args.request.read_text()
            )
            if args.changes:
                if args.request:
                    raise ValueError("Use --changes from the calibrated baseline or --request for a complete proposed state")
                request = scenario_request(fit, json.loads(args.changes.read_text()))
            inputs = predictive_inputs(fit, request=request, n_samples=args.samples,
                                       allow_needs_review=args.allow_needs_review)
            output.mkdir(parents=True)
            save_inputs(inputs, output / "prepared")
            run = simulate_ensemble(inputs, batch_size=args.batch_size, include_intervals=args.include_intervals)
            save_result(run, output / "result")
            print(f"Saved {args.samples}-draw posterior physics output: {output}")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Calibrated twin failed: {exc}\n")


if __name__ == "__main__":
    main()
