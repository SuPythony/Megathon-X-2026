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
