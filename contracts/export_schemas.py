"""Write JSON Schema for every top-level contract into contracts/schema/."""
import json
from pathlib import Path

from . import models

TOP_LEVEL = [
    models.Property, models.EnergyUsageMonth, models.SaleRecord, models.WeatherMonth,
    models.RetrofitItem, models.ScenarioSpec, models.ScenarioResult, models.WhatIfRoute,
    models.ExplanationCheck, models.CustomerMemory,
]

def main() -> None:
    out = Path(__file__).parent / "schema"
    out.mkdir(exist_ok=True)
    for m in TOP_LEVEL:
        (out / f"{m.__name__}.schema.json").write_text(
            json.dumps(m.model_json_schema(), indent=2), encoding="utf-8"
        )
    print(f"wrote {len(TOP_LEVEL)} schemas to {out}")

if __name__ == "__main__":
    main()
