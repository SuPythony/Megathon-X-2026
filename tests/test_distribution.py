"""Distribution contract + summarise(): every uncertain output carries the full summary."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from contracts.models import Distribution, ScenarioResult
from contracts.summarise import category_probabilities, exact, summarise

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)


def test_normal_samples_summary_is_consistent():
    x = RNG.normal(1000, 100, 20_000)
    d = summarise(x, "kWh/yr")
    p = d.percentiles
    assert abs(d.mean - 1000) < 5 and abs(d.median - 1000) < 5 and abs(d.mode - 1000) < 15
    assert abs(d.std - 100) < 3
    assert [p.p1, p.p5, p.p10, p.p25, p.p50, p.p75, p.p90, p.p95, p.p99] == sorted(
        [p.p1, p.p5, p.p10, p.p25, p.p50, p.p75, p.p90, p.p95, p.p99]
    )
    assert d.interval(0.9).lo == p.p5 and d.interval(0.9).hi == p.p95
    assert d.interval(0.5).lo == p.p25 and d.interval(0.5).hi == p.p75
    assert sum(d.histogram.counts) == d.n_samples
    assert d.headline == "mean"


def test_skewed_samples_show_mean_median_mode_apart_and_headline_median():
    x = RNG.lognormal(0, 0.8, 20_000)
    d = summarise(x, "%")
    assert d.mean > d.median > d.mode
    assert d.skewness > 0.5 and d.headline == "median"


def test_tail_probabilities_and_worst_tail():
    x = RNG.normal(-35, 88, 50_000)  # e.g. heat-pump bill change, GBP/yr
    d = summarise(
        x, "GBP/yr",
        tails=[("bill saving is positive", 0.0, "above"), ("bill rises by more than £100/yr", -100.0, "below")],
    )
    pos, big = d.tail_probabilities
    assert abs(pos.probability - np.mean(x > 0)) < 1e-12
    assert abs(big.probability - np.mean(x < -100)) < 1e-12
    assert d.worst_5pct_mean < d.percentiles.p5  # mean of the worst 5% lies beyond P5


def test_worst_tail_high_for_costs():
    x = RNG.normal(10_000, 1_500, 10_000)
    d = summarise(x, "GBP", worst_tail="high")
    assert d.worst_5pct_mean > d.percentiles.p95


def test_exact_value_is_degenerate():
    d = exact(0.0, "kWh/yr", 500)
    assert d.std == 0 and d.mean == d.median == d.mode == 0.0
    assert d.interval(0.9).lo == d.interval(0.9).hi == 0.0


def test_missing_values_are_rejected_not_zeroed():
    with pytest.raises(ValueError):
        summarise([1.0, float("nan"), 3.0], "kWh/yr")
    with pytest.raises(ValueError):
        summarise([], "kWh/yr")


def test_validator_rejects_inconsistent_summaries():
    good = summarise(RNG.normal(0, 1, 1000), "x").model_dump()
    bad = json.loads(json.dumps(good))
    bad["percentiles"]["p95"] = bad["percentiles"]["p5"] - 1
    with pytest.raises(ValidationError):
        Distribution.model_validate(bad)
    bad = json.loads(json.dumps(good))
    bad["credible_intervals"] = [ci for ci in bad["credible_intervals"] if ci["level"] != 0.5]
    with pytest.raises(ValidationError):
        Distribution.model_validate(bad)


def test_category_probabilities_sum_to_one_and_mode():
    mode, probs = category_probabilities(["Poor"] * 81 + ["Moderate"] * 19, ["Very poor", "Poor", "Moderate"])
    assert mode == "Poor" and abs(sum(probs.values()) - 1) < 1e-12 and probs["Very poor"] == 0


def test_mock_result_carries_full_distributions():
    res = ScenarioResult.model_validate_json(
        (ROOT / "contracts" / "examples" / "SH2_scenario_result_MOCK.json").read_text(encoding="utf-8")
    )
    for d in (res.gas_saving_kwh, res.electricity_saving_kwh, res.bill_saving_gbp, res.score_before.score_distribution):
        assert isinstance(d, Distribution) and d.n_samples > 1 and d.histogram is not None
    assert abs(sum(res.score_before.band_probabilities.values()) - 1) < 1e-9
    hp = res.retrofits[0]
    assert hp.electricity_saving_kwh.mean < 0  # electricity rises after a heat pump
    assert hp.bill_saving_gbp.tail_probabilities, "money outputs report tail probabilities"
