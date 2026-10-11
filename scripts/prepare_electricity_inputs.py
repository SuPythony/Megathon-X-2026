"""Prepare and validate inputs BEFORE electricity simulation, entirely inside this worktree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.electricity import ElectricityIntake
from contracts.models import TwinParams, WeatherMonth
from greentwin.twin.electricity_inputs import prepare_electricity_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-truth",
        type=Path,
        default=ROOT / "data/processed/synthetic/thermal-development-v1/truth.jsonl",
    )
    parser.add_argument("--homes", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/synthetic/electricity-inputs-v1"
    )
    args = parser.parse_args(argv)
    if args.homes <= 0:
        parser.error("--homes must be positive")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("Choose a new output directory inside this worktree")
    rows = [json.loads(line) for line in args.synthetic_truth.read_text().splitlines() if line.strip()][
        : args.homes
    ]
    if not rows:
        parser.error("No synthetic homes found")
    # Complete every preparation/validation before creating the output package.
    prepared = []
    for i, home in enumerate(rows):
        params = TwinParams.model_validate(home["true_params"])
        records = [WeatherMonth.model_validate(w) for w in home["weather"]]
        location = home["true_property"]["location"]
        presence = {"mostly_away": "out_all_day", "evenings": "part_day", "mostly_home": "home_all_day"}[
            home["latent"]["behaviour"]
        ]
        inputs = prepare_electricity_inputs(
            params,
            records,
            seed=42 + i,
            development=True,
            latitude=location["lat"],
            longitude=location["lon"],
            intake=ElectricityIntake(presence=presence),
        )
        prepared.append((home, inputs))
    output.mkdir(parents=True)
    interval_names = (
        "timestamps_utc",
        "month_index",
        "local_hour",
        "duration_hours",
        "temperature_c",
        "pv_kwh_per_kwp",
        "baseload_weights",
        "lighting_weights",
        "heating_on_fraction",
        "hot_water_weights",
        "extra_weights",
    )
    summary = []
    for i, (home, inputs) in enumerate(prepared):
        stem = f"home-{i:04d}"
        np.savez_compressed(
            output / f"{stem}.npz", **{key: np.asarray(getattr(inputs, key)) for key in interval_names}
        )
        metadata = inputs.model_dump(exclude=set(interval_names))
        package = {
            "inputs": metadata,
            "parameters": home["true_params"],
            "weather": home["weather"],
            "equipment": home["equipment"],
            "latent": home["latent"],
            "location": home["true_property"]["location"],
            "interval_file": f"{stem}.npz",
        }
        (output / f"{stem}.json").write_text(json.dumps(package, indent=2, allow_nan=False) + "\n")
        summary.append(
            {
                "property_id": inputs.property_id,
                "intervals": len(inputs.timestamps_utc),
                "months": len(inputs.months),
                "package": f"{stem}.json",
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "version": "electricity-inputs/0.1.0",
                "seed": 42,
                "data_version": "electricity-inputs-v1",
                "source": str(args.synthetic_truth.resolve()),
                "input_validation": "passed",
                "mode": "development",
                "homes": summary,
                "limitations": [
                    "Synthetic conditional profiles, not actual smart-meter or PVGIS observations",
                    "No empirical calibration",
                    "No invented tariff or export prices",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Validated and saved {len(prepared)} electricity input packages in {output}")


if __name__ == "__main__":
    main()
