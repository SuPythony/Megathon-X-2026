"""Prepare missing-input distributions separately, then run a prior-predictive twin."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.models import Property
from contracts.uncertain_twin import HouseholdRequest
from greentwin.twin.application_history import DEFAULT_HISTORY, ApplicationHistory, prepare_application_inputs
from greentwin.twin.ensemble_io import load_inputs, new_output, save_inputs, save_result
from greentwin.twin.missing_inputs import generate_missing_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "simulate", "run"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path, help="HouseholdRequest JSON")
    source.add_argument("--demo", choices=("SH1", "SH2", "SH3"))
    source.add_argument("--prepared", type=Path, help="Prior prepared artifact directory")
    parser.add_argument("--changes", type=Path, help="Simple user_inputs JSON overrides, preserving seed")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-intervals", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--history", type=Path, default=DEFAULT_HISTORY, help="Application-answer history SQLite file"
    )
    parser.add_argument(
        "--no-history", action="store_true", help="Ignore history and do not save an application observation"
    )
    parser.add_argument(
        "--what-if", action="store_true", help="Preview only: do not add modified facts to population history"
    )
    args = parser.parse_args(argv)
    try:
        output = new_output(args.output)
        if args.mode == "simulate":
            if not args.prepared or args.changes or args.seed is not None:
                raise ValueError("simulate requires --prepared and uses its exact stored inputs/seed")
            inputs = load_inputs(args.prepared)
        else:
            if args.prepared:
                raise ValueError("Use simulate to replay a prepared artifact")
            if args.demo:
                prop = Property.model_validate_json(
                    (ROOT / f"contracts/examples/{args.demo}_property.json").read_text()
                )
                request = HouseholdRequest(
                    property_id=args.demo, property=prop, allow_development_weather=True
                )
            else:
                request = HouseholdRequest.model_validate_json(args.request.read_text())
            if args.seed is not None:
                request = request.model_copy(update={"seed": args.seed})
            if args.changes:
                request = request.model_copy(
                    update={"user_inputs": request.user_inputs | json.loads(args.changes.read_text())}
                )
            request = HouseholdRequest.model_validate(request.model_dump())
            if args.no_history or args.demo:
                inputs = generate_missing_inputs(request, args.samples)
            else:
                inputs = prepare_application_inputs(
                    request,
                    args.samples,
                    history=ApplicationHistory(args.history),
                    kind="what_if" if args.what_if or args.changes else "baseline",
                )
        if args.mode == "prepare":
            save_inputs(inputs, output)
            print(
                f"Prepared {inputs.summary.n_samples} joint input draws; no simulator or calibration called: {output}"
            )
        else:
            # JAX is only imported by the simulation stage, never by prepare.
            from greentwin.twin.ensemble import simulate_ensemble

            run = simulate_ensemble(
                inputs, batch_size=args.batch_size, include_intervals=args.include_intervals
            )
            save_result(run, output)
            print(f"Saved {run.result.n_samples}-draw {run.result.uncertainty_basis} result: {output}")
            print(f"Run seconds: {run.elapsed_seconds:.3f}; Bayesian calibration: {run.result.calibration_status}")
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.exit(2, f"Uncertain twin failed: {exc}\n")


if __name__ == "__main__":
    main()
