"""Write JSON Schema for every top-level contract into contracts/schema/."""
import json
from pathlib import Path

from . import models

TOP_LEVEL = [
    # Part 0 data layer
    models.Property, models.EnergyUsageMonth, models.SaleRecord, models.WeatherMonth, models.DatasetMeta,
    models.EPCRecommendation, models.AreaEPCStats, models.LSOAConsumption, models.LoadProfileArchetype,
    models.ReferenceLabel, models.RetrofitItem,
    # Part 1 digital twin
    models.TwinParams, models.CalibrationResult, models.TwinScene, models.CaptureResult,
    # Part 2 energy & retrofit
    models.EnergyPrior, models.ImprovementPotential, models.RatingEstimate,
    # Parts 3-4
    models.ScoreResult, models.ValueImpact, models.GreenTwin,
    # Part 5 feasibility & LLM
    models.FeasibilityReport, models.WhatIfRoute, models.ExplanationCheck, models.ChatRequest, models.ChatResponse,
    models.CustomerMemory,
    # Orchestration
    models.ScenarioSpec, models.ScenarioResult, models.DashboardArea,
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
