"""Run the uncalibrated monthly thermal twin; print a contract-validated JSON result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.models import Property, WeatherMonth
from greentwin.twin.params import fuel_config, params_from_property
from greentwin.twin.physics import ThermalAssumptions, simulate_thermal
from greentwin.twin.weather import typical_year
from scripts.generate_synthetic import load_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--property", type=Path, help="One mapped Property JSON record")
    source.add_argument(
        "--demo", choices=("SH1", "SH2", "SH3"), help="Explicit showcase/synthetic-weather run"
    )
    parser.add_argument("--weather", type=Path, help="WeatherMonth JSON list or JSONL")
    parser.add_argument("--year", type=int, default=2025, help="Synthetic weather year in demo mode")
    parser.add_argument("--data-version", default="development-thermal-v1")
    parser.add_argument("--setback-delta-c", type=float, default=3.0)
    parser.add_argument("--solar-aperture-factor", type=float, default=0.45)
    parser.add_argument("--party-wall-u", type=float, default=0.0)
    parser.add_argument(
        "--hot-water-efficiency", type=float, help="Explicit efficiency/COP for independent DHW"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional new JSON file inside this worktree; default stdout"
    )
    args = parser.parse_args(argv)
    if not args.demo and args.weather is None:
        parser.error("--weather is required with --property")
    try:
        path = ROOT / f"contracts/examples/{args.demo}_property.json" if args.demo else args.property
        prop = Property.model_validate_json(path.read_text())
        records = (
            load_records(args.weather, WeatherMonth)
            if args.weather
            else typical_year(prop.location.postcode_district, args.year)
        )
        if any(record.area_code != prop.location.postcode_district for record in records):
            raise ValueError("Weather area must match the property's postcode district")
        # The schema has no solar-thermal contribution or DHW storage model yet.
        if prop.renewables.solar_thermal:
            raise ValueError("Solar-thermal DHW contribution is deferred; cannot model this home yet")
        result = simulate_thermal(
            params_from_property(prop),
            records,
            fuel_config(prop, hot_water_efficiency=args.hot_water_efficiency),
            assumptions=ThermalAssumptions(
                args.setback_delta_c, args.solar_aperture_factor, args.party_wall_u
            ),
            data_version=args.data_version,
        )
        content = result.model_dump_json(indent=2) + "\n"
        if args.output:
            output = args.output.resolve()
            if not output.is_relative_to(ROOT):
                raise ValueError("Output must stay inside the Physics Model and Bayesian worktree")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(content)
            print(f"Saved uncalibrated thermal result to {output}")
        else:
            print(content, end="")
    except (ValueError, TypeError, OSError) as error:
        parser.exit(2, f"Thermal simulation failed: {error}\n")


if __name__ == "__main__":
    main()
