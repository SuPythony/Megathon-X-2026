"""Retrofit engine: rerun-never-sum, per-fuel signs, showcase checks, rating, potential, Pareto."""

import json
from functools import lru_cache

import numpy as np
import pytest

from contracts.models import EnergyUsageMonth, Property, RetrofitAction
from greentwin.interfaces import EnergyRetrofitEngine
from greentwin.optimise.pareto import non_dominated
from greentwin.retrofit.engine import RetrofitEngine
from greentwin.retrofit.ladder import build_ladder
from greentwin.retrofit.potential import _pct
from greentwin.retrofit.property_edits import property_after
from greentwin.retrofit.rating import band_of, estimate_rating, sap_from_cost
from greentwin.retrofit.rules import assess
from greentwin.retrofit.runner import Energy, run_packages
from greentwin.score.score import GreenHomeScorer
from greentwin.twin.simulate import prior_samples
from greentwin.twin.twin import GreenDigitalTwin

N = 200
A = lambda *ids: [RetrofitAction(retrofit_id=i) for i in ids]  # noqa: E731


@lru_cache(maxsize=None)
def home(h: str):
    p = Property.model_validate_json(open(f"contracts/examples/{h}_property.json").read())
    usage = [EnergyUsageMonth(**d) for d in json.load(open(f"contracts/examples/{h}_usage_2025.json"))]
    s = prior_samples(GreenDigitalTwin().params_from_property(p), N)
    return p, usage, s


@lru_cache(maxsize=None)
def engine() -> RetrofitEngine:
    usage = {h: home(h)[1] for h in ("SH1", "SH2", "SH3")}
    return RetrofitEngine(usage=usage, scorer=GreenHomeScorer(n_samples=300, learned_weights_dir=None))


@lru_cache(maxsize=None)
def ladder(h: str):
    p, _, s = home(h)
    return engine().ladder(p, s, assess(p), "bills")


def test_implements_protocol():
    assert isinstance(engine(), EnergyRetrofitEngine)


def test_stub_physics_is_marked_mock():
    prov = engine().provenance(N)
    from greentwin.retrofit.runner import engine_name
    assert prov.engine == ("mock" if engine_name() == "physics_stub" else "physics_l1")


def test_package_is_one_run_not_a_sum():
    """Heat pump + cavity fill: summing the single gas savings double-counts (no gas is left to save)."""
    p, _, s = home("SH2")
    e = engine()
    base = e.baseline(p, s)
    single = lambda ids: base.gas_kwh.median - e.package(p, s, A(*ids)).gas_kwh.median  # noqa: E731
    both = single(["cavity_wall_insulation", "air_source_heat_pump"])
    assert single(["cavity_wall_insulation"]) + single(["air_source_heat_pump"]) > both * 1.05


def test_ladder_steps_add_up_to_package():
    p, _, s = home("SH2")
    steps = ladder("SH2")
    ids = [x.retrofit_id for x in steps if x.verdict != "needs_survey"]
    total = sum(x.gas_saving_kwh.mean for x in steps if x.verdict != "needs_survey")
    e = engine()
    direct = e.baseline(p, s).gas_kwh.mean - e.package(p, s, A(*ids)).gas_kwh.mean
    assert total == pytest.approx(direct, rel=1e-4)


def test_heat_pump_raises_electricity_and_cuts_gas():
    hp = next(x for x in ladder("SH2") if x.retrofit_id == "air_source_heat_pump")
    assert hp.gas_saving_kwh.median > 0
    assert hp.electricity_saving_kwh.median < 0  # electricity rises: negative saving
    assert hp.electricity_saving_kwh.tail_probabilities[0].probability > 0.9


def test_showcase_feasibility_reaches_the_ladder():
    sh1 = {x.retrofit_id for x in ladder("SH1")}
    assert "external_wall_insulation" not in sh1 and "internal_wall_insulation" in sh1
    sh3 = {x.retrofit_id: x for x in ladder("SH3")}
    assert "loft_insulation" not in sh3 and "solar_pv" not in sh3
    assert sh3["air_source_heat_pump"].verdict == "needs_survey"


def test_ladder_effects_are_well_formed():
    for h in ("SH1", "SH2", "SH3"):
        steps = ladder(h)
        assert [x.order for x in steps] == list(range(1, len(steps) + 1))
        for x in steps:
            assert 0 <= x.p_suitable <= x.p_net_benefit + 1e-9 <= 1 + 1e-9
            assert (x.cost_gbp is None) == (x.cost_unavailable_reason is not None)
            if x.cost_gbp is None:
                assert x.verdict != "recommended"  # payback can't be judged without a sourced cost
            assert x.reason and x.score_after is not None


def test_unsourced_cost_is_null_with_reason():
    led = next(x for x in ladder("SH1") if x.retrofit_id == "led_lighting")
    assert led.cost_gbp is None and "No sourced cost" in led.cost_unavailable_reason


def test_potential_null_when_baseline_zero():
    z = np.zeros(10)
    pct, reason = _pct(Energy(electricity=z, gas=z), Energy(electricity=z, gas=z))
    assert pct is None and "undefined" in reason


def test_improvement_potential():
    p, _, s = home("SH3")
    ip = engine().improvement_potential(p, s, assess(p))
    assert ip.pct is not None and 0 < ip.pct.median < 100
    assert ip.excluded_needs_survey == ["air_source_heat_pump"]
    assert ip.pct_if_surveys_pass.median > ip.pct.median


def test_rating_formula_and_bands():
    assert sap_from_cost(np.array([0.0]), 100)[0] == pytest.approx(100)
    assert list(band_of([92, 91.9, 69, 54.9, 1])) == ["A", "B", "C", "E", "G"]
    costs = np.linspace(200, 6000, 50)
    assert np.all(np.diff(sap_from_cost(costs, 90, clamp=False)) < 0)  # dearer to run -> lower SAP


def test_rating_after_fabric_is_not_lower():
    p, _, s = home("SH2")
    r = estimate_rating(p, s, A("cavity_wall_insulation", "loft_insulation"))
    assert r.estimate.anchored_to_recorded_sap
    assert r.estimate.sap.percentiles.p1 >= p.epc.sap_score - 1e-9
    assert r.estimate.label == "estimated rating, not an EPC"
    none = estimate_rating(p, s, [])
    assert none.estimate.sap.median == pytest.approx(p.epc.sap_score)


def test_rating_without_recorded_sap_and_unsupported_fuel():
    p, _, s = home("SH2")
    no_epc = p.model_copy(update={"epc": p.epc.model_copy(update={"sap_score": None})})
    assert not estimate_rating(no_epc, s, A("loft_insulation")).estimate.anchored_to_recorded_sap
    oil = p.model_copy(update={"heating": p.heating.model_copy(update={"main_fuel": "oil"})})
    r = estimate_rating(oil, s, A("loft_insulation"))
    assert r.estimate is None and "oil" in r.unavailable_reason


def test_property_after_and_projected_score_not_lower():
    p, _, s = home("SH2")
    pkg = A("cavity_wall_insulation", "loft_insulation", "double_glazing", "smart_thermostat", "solar_pv")
    after = property_after(p, pkg)
    assert after.walls.insulation.value == "cavity_filled" and after.roof.loft_insulation_mm == 270
    assert after.renewables.pv_kwp == pytest.approx(3.5) and "smart_thermostat" in [c.value for c in after.controls]
    e = engine()
    sc = e.scorer
    now = sc.score(p, e.baseline(p, s)).score
    proj = sc.score(after, e.package(p, s, pkg), e.estimate_rating(p, s, pkg)).score
    assert proj >= now


def test_non_dominated():
    F = np.array([[0, 0], [1, -1], [1, 0], [2, -1]])
    assert list(non_dominated(F)) == [True, True, False, False]


def test_pareto_small_home():
    p, _, s = home("SH3")
    pts = engine().pareto(p, s, assess(p))
    assert any(pt.package == [] for pt in pts)
    assert all(0 <= pt.p_on_front <= 1 for pt in pts)
    assert all(pt.net_cost_gbp is not None for pt in pts)
    assert sum(pt.is_knee for pt in pts) <= 1
    cheap = engine().pareto(p, s, assess(p), budget_gbp=300)
    assert all(pt.net_cost_gbp.median <= 300 for pt in cheap)


def test_batched_runs_match_single_runs():
    p, _, s = home("SH2")
    batched = run_packages(s, [[], A("loft_insulation"), A("air_source_heat_pump")])
    twin = GreenDigitalTwin()
    for run, acts in zip(batched, [[], A("loft_insulation"), A("air_source_heat_pump")]):
        ref = twin.simulate(s, acts)
        assert np.allclose(run.gas_kwh, ref.gas_kwh, rtol=1e-5)
        assert np.allclose(run.electricity_import_kwh, ref.electricity_import_kwh, rtol=1e-5)
