"""Generate inputs now; connect a versioned physics simulator to generate meters later."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.models import AreaEPCStats, Property, WeatherMonth
from greentwin.twin.generate import (
    GeneratorConfig,
    Simulator,
    generate_dataset,
    write_dataset,
)


def load_records(path, model):
    """Read mapped contract records as a JSON list or JSONL, not arbitrary raw EPC CSV."""
    content = path.read_text(encoding="utf-8")
    rows = (
        json.loads(content)
        if content.lstrip().startswith("[")
        else [json.loads(line) for line in content.splitlines() if line.strip()]
    )
    return [model.model_validate(row) for row in rows]


def load_callable(spec):
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise ValueError("Use module:function for simulator/scenario factory")
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--properties", type=Path, help="Mapped real EPC Property JSON/JSONL")
    source.add_argument("--demo", action="store_true", help="Explicit development mode using SH1-SH3")
    parser.add_argument("--weather", type=Path, help="WeatherMonth JSON/JSONL, matching location and period")
    parser.add_argument(
        "--area-stats", type=Path, help="Comparable LA/period AreaEPCStats for optional weighting"
    )
    parser.add_argument("--homes", type=int, default=100)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-version", default="development-v1")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inputs-only", action="store_true", help="Generate no fabricated energy readings")
    mode.add_argument("--simulator", help="module:run_physics; must supply conservation diagnostics")
    parser.add_argument("--simulator-version", help="Required with --simulator")
    parser.add_argument(
        "--electricity-development",
        action="store_true",
        help="Prepare labelled half-hour electricity inputs from synthetic household/weather",
    )
    parser.add_argument(
        "--scenario-factory", help="module:function; returns validated RetrofitScenario packages"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/synthetic/development-v1")
    args = parser.parse_args(argv)
    if args.simulator and not args.simulator_version:
        parser.error("--simulator-version is required for traceable energy generation")
    if args.scenario_factory and args.inputs_only:
        parser.error("Retrofit labels require a simulator")
    if args.electricity_development and (not args.demo or args.inputs_only):
        parser.error("--electricity-development requires --demo and --simulator")
    try:
        properties = (
            [
                Property.model_validate_json(p.read_text())
                for p in sorted((ROOT / "contracts/examples").glob("SH*_property.json"))
            ]
            if args.demo
            else load_records(args.properties, Property)
        )
        config = GeneratorConfig(
            n_homes=args.homes,
            n_months=args.months,
            start_month=args.start_month,
            seed=args.seed,
            development=args.demo,
            data_version=args.data_version,
        )
        simulator = (
            Simulator(load_callable(args.simulator), args.simulator_version) if args.simulator else None
        )
        factory = load_callable(args.scenario_factory) if args.scenario_factory else None
        electricity_factory = None
        if args.electricity_development:
            from greentwin.twin.electricity_inputs import development_electricity_factory

            electricity_factory = development_electricity_factory
        dataset = generate_dataset(
            properties,
            config,
            simulator=simulator,
            weather_records=load_records(args.weather, WeatherMonth) if args.weather else (),
            area_stats=load_records(args.area_stats, AreaEPCStats) if args.area_stats else (),
            scenario_factory=factory,
            electricity_factory=electricity_factory,
        )
        write_dataset(dataset, args.output)
    except (ValueError, TypeError, OSError, ImportError, AttributeError) as error:
        parser.exit(2, f"Generation failed: {error}\n")
    print(f"Generated {len(dataset.homes)} homes ({dataset.metadata['mode']}) in {args.output.resolve()}")
    if args.demo:
        print(
            "Development inputs: showcase homes, synthetic fallback weather and provisional priors/defaults."
        )
    if args.inputs_only:
        print("No energy readings or retrofit labels generated; supply your physics simulator when ready.")


if __name__ == "__main__":
    main()
