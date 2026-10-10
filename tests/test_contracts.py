"""Contract tests: examples validate, score mapping is well-formed and monotonic."""

import json
from pathlib import Path

import pytest

from contracts.models import EnergyUsageMonth, Property, ScenarioResult, ScenarioSpec

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "contracts" / "examples"
MAPPING = json.loads((ROOT / "contracts" / "score_mapping.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("pid", ["SH1", "SH2", "SH3"])
def test_showcase_property_and_usage_validate(pid):
    Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))
    rows = json.loads((EX / f"{pid}_usage_2025.json").read_text(encoding="utf-8"))
    assert len(rows) == 12
    for r in rows:
        EnergyUsageMonth.model_validate(r)


def test_scenario_examples_validate():
    ScenarioSpec.model_validate_json((EX / "SH2_scenario_spec.json").read_text(encoding="utf-8"))
    res = ScenarioResult.model_validate_json((EX / "SH2_scenario_result_MOCK.json").read_text(encoding="utf-8"))
    assert res.provenance.engine == "mock"


def test_weights_sum_to_one():
    assert abs(sum(MAPPING["weights"].values()) - 1.0) < 1e-9


def _non_decreasing(points):
    ys = [y for _, y in points]
    return all(b >= a for a, b in zip(ys, ys[1:]))


def test_piecewise_mappings_are_monotonic():
    c = MAPPING["components"]
    # intensity: lower kWh/m2 is better -> sub-score non-increasing in x
    ys = [y for _, y in c["energy_intensity"]["points"]]
    assert all(b <= a for a, b in zip(ys, ys[1:]))
    assert _non_decreasing(c["insulation"]["roof_loft_mm_points"])
    assert _non_decreasing(c["renewable_energy"]["pv_kwp_points"])


def test_glazing_order_is_monotonic():
    m = MAPPING["components"]["windows_glazing"]["map"]
    order = ["single", "secondary", "double_pre_2002", "double_post_2002", "triple"]
    vals = [m[k] for k in order]
    assert vals == sorted(vals)


def test_classification_covers_0_to_100():
    covered = set()
    for b in MAPPING["classification"]:
        covered.update(range(b["min"], b["max"] + 1))
    assert covered == set(range(0, 101))


# --------------------------------------------------------------------------- compliance checks
from contracts.models import HeatingSystem, RetrofitItem  # noqa: E402

CONTRACTS = ROOT / "contracts"


def _load(name):
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def test_heating_mapping_covers_every_heating_system():
    mapped = set(MAPPING["components"]["heating_system"]["map"])
    assert {h.value for h in HeatingSystem} <= mapped


def test_emission_factors_are_sourced_and_consistent():
    ef = _load("emission_factors.json")
    assert ef["source"]["url"].startswith("https://assets.publishing.service.gov.uk/")
    elec = ef["factors"]["electricity"]
    comp = sum(c["value"] for c in elec["components"].values())
    assert abs(comp - elec["value"]) < 1e-9
    assert ef["factors"]["gas"]["basis"] == "kWh (Gross CV)"


def test_catalogue_costs_are_gated_by_status():
    cat = _load("retrofit_catalogue.json")
    for item in cat["items"]:
        it = RetrofitItem.model_validate(item)
        has_cost = any(v is not None for v in (it.cost_gbp_low, it.cost_gbp_central, it.cost_gbp_high))
        if it.cost_status == "unsourced":
            assert not has_cost, f"{it.id}: unsourced item must not carry a cost"
        else:
            assert has_cost and it.source and "TO_SOURCE" not in it.source, it.id


def test_intensity_definition_is_metered_not_weather_normalised():
    c = MAPPING["components"]["energy_intensity"]
    assert "NOT weather-normalised" in c["input"]
    assert "overlap_with_renewables" in c


def test_sap_estimator_documented():
    sap = _load("sap_estimator.json")
    assert sap["label"].startswith("Estimated rating, not an EPC")
    assert set(sap["band_thresholds"]) == set("ABCDEFG")


def test_green_profits_is_scenario_not_trained():
    gp = _load("green_profits_assumptions.json")
    assert all(v["status"] == "assumption" for v in gp["take_up_rule"]["coefficients"].values())
