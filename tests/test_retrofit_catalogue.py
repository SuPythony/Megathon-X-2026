"""Retrofit catalogue: validation, ordering, sourced-cost rule, grants."""

import numpy as np
import pytest

from contracts.models import RetrofitAction, TwinParams
from greentwin.retrofit import catalogue as cat
from greentwin.twin.retrofits import KEY_MAP

N = 4000


def test_catalogue_validates_and_references_exist():
    items = cat.items()
    assert len(items) == 16
    assert "solar_pv" in items["battery"].depends_on


def test_every_twin_edit_is_mapped():
    fields = set(TwinParams.model_fields)
    for it in cat.items().values():
        for key in it.twin_edits:
            assert key in KEY_MAP or key in fields or key == "tariff", (it.id, key)


def test_every_item_has_an_ordering_group():
    n_groups = len(cat.ordering_groups())
    for rid in cat.items():
        assert cat.sequence_group(rid) < n_groups, rid
    assert cat.sequence_group("loft_insulation") < cat.sequence_group("air_source_heat_pump") < cat.sequence_group("solar_pv")


def test_unsourced_items_return_no_cost_with_reason():
    for rid, it in cat.items().items():
        c = cat.cost_samples(rid, N)
        if it.cost_status == "unsourced":
            assert c.gross_gbp is None and c.distribution() is None
            assert "excluded from cost ranking" in c.unavailable_reason
        else:
            assert c.available and np.all(c.gross_gbp >= 0) and np.all(np.isfinite(c.gross_gbp))


def test_ranged_cost_matches_source_percentiles():
    it = cat.get("loft_insulation")
    x = cat.cost_samples("loft_insulation", 200_000).gross_gbp
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    assert p50 == pytest.approx(it.cost_gbp_central, rel=0.02)
    assert p5 == pytest.approx(it.cost_gbp_low, rel=0.03)
    assert p95 == pytest.approx(it.cost_gbp_high, rel=0.03)


def test_central_only_cost_is_flagged():
    c = cat.cost_samples("smart_thermostat", N)
    assert np.median(c.gross_gbp) == pytest.approx(620, rel=0.03)
    assert any("assumption" in w for w in c.warnings)


def test_per_kwp_scales_with_size():
    default = np.median(cat.cost_samples("solar_pv", N).gross_gbp)
    big = np.median(cat.cost_samples("solar_pv", N, action=RetrofitAction(retrofit_id="solar_pv", size=5.0)).gross_gbp)
    assert default == pytest.approx(1876 * 3.5, rel=0.03)
    assert big / default == pytest.approx(5.0 / 3.5, rel=1e-6)


def test_heat_pump_grant_confirmed_only():
    c = cat.cost_samples("air_source_heat_pump", N)
    assert c.grants_applied == ["BUS_7500"] and c.grant_gbp == 7500
    assert np.all(c.net_gbp >= 0) and np.all(c.net_gbp <= c.gross_gbp)
    off = cat.cost_samples("air_source_heat_pump", N, off_gas_grid=True)
    assert off.grant_gbp == 7500  # BUS £9,000 off-gas rate not confirmed yet
    assert any(g.startswith("BUS_9000_off_gas") for g in off.grants_not_applied)
    loft = cat.cost_samples("loft_insulation", N)
    assert loft.grant_gbp == 0 and len(loft.grants_not_applied) == 2  # ECO4 / GBIS unconfirmed


def test_tariff_switch_costs_zero():
    c = cat.cost_samples("time_of_use_tariff", N)
    assert np.all(c.gross_gbp == 0)


def test_costs_are_reproducible():
    a = cat.cost_samples("cavity_wall_insulation", N, seed=7).gross_gbp
    b = cat.cost_samples("cavity_wall_insulation", N, seed=7).gross_gbp
    assert np.array_equal(a, b)


def test_package_cost_sums_and_null_when_any_unsourced():
    acts = [RetrofitAction(retrofit_id=r) for r in ("loft_insulation", "cavity_wall_insulation")]
    total, parts = cat.package_cost(acts, N)
    assert np.allclose(total, parts[0].net_gbp + parts[1].net_gbp)
    total, _ = cat.package_cost(acts + [RetrofitAction(retrofit_id="triple_glazing")], N)
    assert total is None
    empty, _ = cat.package_cost([], N)
    assert np.all(empty == 0)


def test_exclusions():
    assert cat.conflicts(["internal_wall_insulation", "external_wall_insulation", "loft_insulation"]) == [
        ("external_wall_insulation", "internal_wall_insulation")]
    assert cat.valid_package(["loft_insulation", "double_glazing"])
    assert not cat.valid_package(["double_glazing", "triple_glazing"])
