"""Demo city: deterministic synthetic homes, batched physics equals per-home runs, insight maths, area roll-ups."""

import numpy as np
import pytest

from contracts.models import DashboardArea
from greentwin.twin.archetypes import build_layout
from greentwin.twin.city import (
    aggregate,
    build_city,
    monthly_repayment,
    package_cost,
    simulate_many,
    synthetic_homes,
)
from greentwin.twin.inputs import build_physics_inputs
from greentwin.twin.params import fuel_config
from greentwin.twin.simulate import prior_samples, simulate
from greentwin.twin.weather import resolve_weather


@pytest.fixture(scope="module")
def city():
    return build_city(n=60, n_samples=60, seed=7)


def test_synthetic_homes_are_deterministic_and_labelled():
    a, b = synthetic_homes(30, seed=3), synthetic_homes(30, seed=3)
    assert [h.prop.model_dump() for h in a] == [h.prop.model_dump() for h in b]
    assert all(h.prop.source == "synthetic" for h in a)
    assert all("-" in h.small_area and h.small_area.split("-")[0] == h.prop.location.postcode_district for h in a)
    assert not any(h.small_area.startswith("E0") for h in a)  # never a real LSOA code


def test_batched_physics_matches_single_home_runs():
    homes = synthetic_homes(6, seed=11)
    weather, _ = resolve_weather("M14")
    samples = []
    for h in homes:
        layout = build_layout(h.archetype_id, total_floor_area_m2=h.prop.floor_area_m2, built_form=h.prop.built_form)
        pkg = build_physics_inputs(h.prop, layout=layout)
        samples.append(prior_samples(pkg.params, 20, seed=1, priors=pkg.prior_table()))
    batched, _ = simulate_many(samples, weather)
    for s, b in zip(samples, batched):
        single, _ = simulate(s, weather, fuel_config(s.base))
        np.testing.assert_allclose(b["gas"], single.gas_kwh, rtol=1e-4, atol=1e-2)
        np.testing.assert_allclose(b["elec"], single.electricity_import_kwh, rtol=1e-4, atol=1e-2)


def test_loan_and_cost_maths():
    assert monthly_repayment(12000, 0.0, 10) == pytest.approx(100)
    assert monthly_repayment(10000, 0.06, 10) == pytest.approx(111.02, abs=0.01)
    cost, note = package_cost(["loft_insulation", "draught_proofing"], 3.5)
    assert cost == 620 + 250 and "grants not applied" in note
    assert package_cost(["triple_glazing"], 3.5)[0] is None  # unsourced cost -> no figure
    pv, _ = package_cost(["solar_pv"], 3.5)
    assert pv == pytest.approx(1876 * 3.5, abs=1)


def test_city_outputs(city):
    homes, areas, config = city
    assert len(homes) == 60 and config["label"].startswith("Synthetic")
    for h in homes:
        for name, p in h.packages.items():
            if p is None:
                continue
            assert 0 <= p.p_bill_falls <= 1
            if p.cost_gbp:
                assert p.monthly_repayment_gbp > 0 and 0 <= p.p_self_funding <= 1
        if h.best_package:
            assert h.packages[h.best_package].p_self_funding is not None
        if h.is_customer:
            assert h.confidence_tier in ("A", "B", "C")
        else:
            assert h.confidence_tier is None
        hp = h.packages.get("heat_pump")
        if hp:
            assert hp.gas_saving_kwh.headline >= 0  # gas goes, electricity may rise
    stress = [h.stress["price_cap_up_30pct"] for h in homes]
    assert all(s.p_over_before >= h.stress["base"].p_over_before for s, h in zip(stress, homes))


def test_areas_roll_up_and_fit_the_dashboard_contract(city):
    homes, areas, _ = city
    districts = [a for a in areas if a.area_type == "postcode_district"]
    assert sum(a.n_homes_modelled for a in districts) == len(homes)
    for a in areas:
        DashboardArea.model_validate(a.model_dump(include=set(DashboardArea.model_fields)))
        if a.self_funding_share is not None:
            assert 0 <= a.self_funding_share <= 1
        assert set(a.tier_share) <= {"A", "B", "C"}
    assert aggregate(homes)[0].area_code == areas[0].area_code  # deterministic order
