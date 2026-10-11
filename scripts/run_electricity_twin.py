"""Simulate an already prepared electricity input package; no implicit input generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.electricity import PreparedElectricityInputs
from contracts.models import TwinParams, WeatherMonth
from greentwin.twin.electricity_simulation import simulate_electricity
from greentwin.twin.physics_api import FuelConfig
from greentwin.twin.satyarth_inputs import adapt_satyarth_package


def load_prepared(path):
    package = json.loads(path.read_text())
    values = package.get("electricity_inputs", package.get("inputs"))
    if values is None:
        raise ValueError("Expected a prepared electricity package")
    if "interval_file" in package:
        interval_path = (path.parent / package["interval_file"]).resolve()
        if interval_path.parent != path.parent.resolve():
            raise ValueError("Interval file must be alongside its input package")
        with np.load(interval_path, allow_pickle=False) as arrays:
            values = values | {key: arrays[key].tolist() for key in arrays.files}
    return package, PreparedElectricityInputs.model_validate(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--satyarth-package", type=Path, help="Optional latest physics_inputs/0.1 handoff")
    parser.add_argument("--include-intervals", action="store_true")
    parser.add_argument("--output", type=Path, help="New result JSON inside this worktree; otherwise stdout")
    args = parser.parse_args(argv)
    try:
        package, inputs = load_prepared(args.prepared)
        params = TwinParams.model_validate(package["parameters"])
        weather = [WeatherMonth.model_validate(w) for w in package["weather"]]
        profiles, extras = {}, {}
        if args.satyarth_package:
            adapted = adapt_satyarth_package(json.loads(args.satyarth_package.read_text()))
            params, weather, fuel = adapted.params, adapted.weather, adapted.fuel
            profiles, extras = adapted.profiles, adapted.extra_params
        else:
            equipment = package.get("equipment")
            if equipment is None:
                raise ValueError("Prepared package must include equipment or --satyarth-package")
            fuel = FuelConfig(**equipment)
            if "latent" in package:
                profiles = {
                    key: package["latent"][key] for key in ("baseload_multiplier", "hot_water_multiplier")
                }
        result = simulate_electricity(
            params,
            weather,
            fuel,
            inputs,
            profiles=profiles,
            extra_params=extras,
            include_intervals=args.include_intervals,
        )
        content = result.model_dump_json(indent=2) + "\n"
        if args.output:
            output = args.output.resolve()
            if not output.is_relative_to(ROOT):
                raise ValueError("Output must stay inside this worktree")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(content)
            print(f"Saved uncalibrated coupled simulation to {output}")
        else:
            print(content, end="")
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"Electricity simulation failed: {error}\n")


if __name__ == "__main__":
    main()
