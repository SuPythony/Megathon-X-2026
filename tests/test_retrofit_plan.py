"""Staging by dependencies and cash flow; Pareto reference point, memory re-rank, CO2-only baseline, hypervolume."""

import numpy as np

from contracts.models import Feasibility, RetrofitEffect
from contracts.summarise import exact
from greentwin.optimise.pareto import (
    ParetoResult, co2_only_baseline, hypervolume, preference_rerank, reference_point, weights_from_memory,
)
from greentwin.optimise.staging import stage


def _step(rid, cost, saving, verdict="recommended", order=1):
    f = Feasibility(retrofit_id=rid, status="feasible", p_feasible=1.0, decided_by="rules")
    z = exact(0.0, "kWh/yr")
    return RetrofitEffect(
        retrofit_id=rid, order=order, feasibility=f, electricity_saving_kwh=z, gas_saving_kwh=z,
        carbon_saving_kgco2e=exact(0.0, "kgCO2e/yr"), bill_saving_gbp=exact(float(saving), "GBP/yr"),
        cost_gbp=exact(float(cost), "GBP") if cost is not None else None,
        cost_unavailable_reason=None if cost is not None else "unsourced",
        purchased_energy_saving_kwh=z, p_net_benefit=1.0, p_suitable=1.0, suitability="High",
        p_gain_above_threshold=1.0, verdict=verdict, reason="test")


def test_staging_respects_budget_and_dependencies():
    steps = [_step("loft_insulation", 900, 200, order=1), _step("cavity_wall_insulation", 2000, 250, order=2),
             _step("air_source_heat_pump", 6000, 300, order=3), _step("solar_pv", 5000, 400, order=4),
             _step("battery", None, 100, order=5), _step("smart_thermostat", 300, 50, verdict="needs_survey", order=6)]
    plan = stage(steps, annual_budget_gbp=3000, reinvest=False)
    year = {r: s.year for s in plan.stages for r in s.retrofit_ids}
    assert all(s.cost_gbp <= s.cash_before_gbp + 1e-6 for s in plan.stages)
    assert year["air_source_heat_pump"] >= max(year["loft_insulation"], year["cavity_wall_insulation"])
    assert "battery" in plan.not_staged and "smart_thermostat" in plan.not_staged
    assert set(year) == {"loft_insulation", "cavity_wall_insulation", "air_source_heat_pump", "solar_pv"}


def test_staging_reinvest_is_never_slower():
    steps = [_step("loft_insulation", 900, 300), _step("cavity_wall_insulation", 2000, 300),
             _step("air_source_heat_pump", 6000, 400), _step("solar_pv", 5000, 500)]
    a = stage(steps, 2000, reinvest=False)
    b = stage(steps, 2000, reinvest=True)
    assert b.lost_saving_gbp <= a.lost_saving_gbp
    assert max(s.year for s in b.stages) <= max(s.year for s in a.stages)


def _result():
    pk = [(), ("a",), ("b",), ("a", "b")]
    rng = np.random.default_rng(0)
    n = 400
    cost = np.stack([np.zeros(n), 1000 + 50 * rng.standard_normal(n), 4000 + 100 * rng.standard_normal(n),
                     5000 + 100 * rng.standard_normal(n)])
    carbon = np.stack([np.zeros(n), 300 + 30 * rng.standard_normal(n), 1800 + 100 * rng.standard_normal(n),
                       2100 + 100 * rng.standard_normal(n)])
    bill = np.stack([np.zeros(n), 150 + 10 * rng.standard_normal(n), 100 + 10 * rng.standard_normal(n),
                     260 + 10 * rng.standard_normal(n)])
    med = np.column_stack([np.median(cost, 1), np.median(carbon, 1), np.median(bill, 1), [0, 2, 5, 7]])
    return ParetoResult([], pk, med, np.array([1.0, 1.0, 0.6, 1.0]), samples=(cost, carbon, bill))


def test_reference_point_hard_targets():
    r = _result()
    ok = reference_point(r, min_carbon_kg=2000, max_cost_gbp=6000)
    assert ok["package"] == ["a", "b"] and ok["p_meets"] > 0.5
    none = reference_point(r, min_carbon_kg=2000, max_cost_gbp=3000)
    assert none["package"] is None and "closest" in none


def test_preference_rerank_orders_only():
    r = _result()
    carbon = preference_rerank(r, {"carbon_saving_kgco2e": 1})
    cheap = preference_rerank(r, {"net_cost_gbp": 1})
    assert {tuple(x["package"]) for x in carbon} == {tuple(x["package"]) for x in cheap}  # same set, other order
    assert carbon[0]["package"] == ["a", "b"] and cheap[0]["package"] == ["a"]
    w = weights_from_memory([{"kind": "goal", "key": "goal", "value": "carbon"}])
    assert w["carbon_saving_kgco2e"] > w["bill_saving_gbp"]


def test_co2_only_and_hypervolume():
    r = _result()
    pkg, _ = co2_only_baseline(r, 4500)
    assert pkg == ("b",)
    full = hypervolume(r, [1, 2, 3], 6000)
    single = hypervolume(r, [1], 6000)
    assert 0 <= single <= full <= 1


# --------------------------------------------------------------------------- orchestrator (glue), small home
def test_orchestrator_scenario_and_plan_sh3():
    from contracts.models import ScenarioResult
    from greentwin.api.orchestrator import GreenTwinOrchestrator, spec_for

    o = GreenTwinOrchestrator(n_samples=64, disk_cache=False)
    r = o.run_scenario(spec_for("SH3", ["draught_proofing"], include_value=False, include_element_heat_loss=False))
    assert isinstance(r, ScenarioResult) and r.provenance.engine == "physics_l1"
    assert r.calibration is not None and r.calibration.method != "prior_only"
    # per fuel, baseline - after; an all-electric flat has no gas to save
    assert r.gas_saving_kwh.median == 0 and r.electricity_saving_kwh.median > 0
    assert r.rating_after is not None and r.score_after.score >= r.score_before.score
    assert o.run_scenario(spec_for("SH3", ["draught_proofing"], include_value=False,
                                   include_element_heat_loss=False)) is r  # cached
    p = o.plan("SH3", annual_budget_gbp=1000)
    assert p["staging"]["stages"] and p["baselines"]["hypervolume"]["front"] >= p["baselines"]["hypervolume"]["greedy"]
    assert "air_source_heat_pump" in p["staging"]["not_staged"]  # needs a survey on SH3
