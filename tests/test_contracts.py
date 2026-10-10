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


# --------------------------------------------------------------------------- 0.4.0: parts, mocks, generated TS
from contracts import export_ts  # noqa: E402
from contracts.models import (  # noqa: E402
    AreaEPCStats, ChatResponse, LoadProfileArchetype, TwinScene,
)


def test_scene_and_chat_mocks_validate():
    scene = TwinScene.model_validate_json((EX / "SH2_twin_scene_MOCK.json").read_text(encoding="utf-8"))
    assert all(s.estimated for s in scene.surfaces)  # heat-loss split is never presented as measured
    chat = ChatResponse.model_validate_json((EX / "SH2_chat_response_MOCK.json").read_text(encoding="utf-8"))
    assert chat.check.passed and not chat.check.unsupported_numbers


def test_mock_result_uses_structured_improvement_potential_and_issues():
    res = ScenarioResult.model_validate_json((EX / "SH2_scenario_result_MOCK.json").read_text(encoding="utf-8"))
    assert res.improvement_potential.pct is not None
    assert res.score_before.key_issues and res.score_before.key_issues[0].code
    assert res.score_after.projected and res.score_after.rating_source == "estimated"


def test_types_ts_is_generated_from_models():
    assert (ROOT / "contracts" / "types.ts").read_text(encoding="utf-8") == export_ts.render(), \
        "contracts/types.ts is stale: run `python -m contracts.export_ts`"


def test_load_profile_must_have_48_shares_summing_to_one():
    LoadProfileArchetype(archetype_id="a", season="winter", day_type="weekday",
                         half_hourly_share=[1 / 48] * 48, n_households=10)
    with pytest.raises(ValueError):
        LoadProfileArchetype(archetype_id="a", season="winter", day_type="weekday",
                             half_hourly_share=[0.1] * 48, n_households=10)


def test_area_epc_stats_row_from_live_table_d1():
    row = AreaEPCStats(area_code="E07000223", area_type="local_authority", period="2008/4", lodgements=522,
                       total_floor_area_m2=36714.1, band_counts={"A": 1, "B": 52, "C": 141, "D": 180, "E": 105,
                                                                 "F": 33, "G": 10}, source_tables=["D1"],
                       release="data to June 2026")
    assert sum(row.band_counts.values()) == row.lodgements


def test_interfaces_import_and_cover_every_part():
    from greentwin import interfaces as it
    for name in ["DataLayer", "DigitalTwin", "EnergyRetrofitEngine", "GreenHomeScorer", "ValueModel",
                 "FeasibilityEngine", "Router", "Explainer", "Orchestrator"]:
        assert hasattr(it, name), name


def test_tariff_switch_changes_bills_not_kwh():
    cat = _load("retrofit_catalogue.json")
    tou = next(i for i in cat["items"] if i["id"] == "time_of_use_tariff")
    assert RetrofitItem.model_validate(tou).changes_kwh is False
