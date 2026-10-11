"""Orchestrator: ``run_scenario(ScenarioSpec) -> ScenarioResult`` (``greentwin.interfaces.Orchestrator``).

    orch = GreenTwinOrchestrator()
    res  = orch.run_scenario(ScenarioSpec(scenario_id="SH2-hp", property_id="SH2",
                                          retrofits=[RetrofitAction(retrofit_id="air_source_heat_pump")]))

calibrate (cached per home) -> feasibility (rules, layer 1) -> baseline + package (ONE twin run, anchored to the
meters) -> ladder -> improvement potential -> rating after (sap_estimator) -> score before/after -> value ->
element heat loss -> Pareto (+ staging and the reference-point/preference modes on request).
Results are cached in memory and on disk (``cache/scenarios``) by hash(spec + model versions), so the showcase homes
replay offline. Every number comes from the parts' own models; this module only wires them together.
"""

from __future__ import annotations

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from contracts.models import (
    EnergyUsageMonth, FeasibilityReport, Property, RetrofitAction, ScenarioResult, ScenarioSpec,
)
from greentwin.retrofit import catalogue as cat
from greentwin.retrofit.engine import MODEL_VERSION, RetrofitEngine
from greentwin.retrofit.property_edits import property_after
from greentwin.retrofit.rules import assess
from greentwin.retrofit.runner import (
    anchored, emission_factor_source, emission_factors, fuel_supported, run_packages, saving,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "contracts" / "examples"
CACHE = ROOT / "cache" / "scenarios"
CACHE_VERSION = "orchestrator/0.1"


# --------------------------------------------------------------------------- data access (Part 0 stand-in)
def load_property(pid: str) -> Property:
    f = EXAMPLES / f"{pid}_property.json"
    if f.exists():
        return Property.model_validate_json(f.read_text(encoding="utf-8"))
    from greentwin.llm.tools import _prop  # knowledge-base store (DuckDB), built by greentwin.kb.build

    return _prop(pid)


def load_usage(pid: str) -> list[EnergyUsageMonth]:
    f = EXAMPLES / f"{pid}_usage_2025.json"
    if f.exists():
        return [EnergyUsageMonth.model_validate(r) for r in json.loads(f.read_text(encoding="utf-8"))]
    try:
        from greentwin.kb.store import q

        rows = q("SELECT * FROM usage_monthly WHERE property_id = ? ORDER BY month", [pid])
        return [EnergyUsageMonth(**r) for r in rows]
    except Exception:  # noqa: BLE001 - no store: the twin falls back to its prior and says so
        return []


class GreenTwinOrchestrator:
    """Implements ``greentwin.interfaces.Orchestrator.run_scenario`` (dashboard_areas lives in the dashboard)."""

    def __init__(self, n_samples: int = 256, seed: int = 42, disk_cache: bool = True):
        from greentwin.score import GreenHomeScorer
        from greentwin.twin.twin import GreenDigitalTwin

        self.n_samples = n_samples
        self.seed = seed
        self.disk_cache = disk_cache
        self.twin = GreenDigitalTwin(n_samples=n_samples, seed=seed)
        self.scorer = GreenHomeScorer(n_samples=1000, seed=seed)
        self.engine = RetrofitEngine(scorer=self.scorer, seed=seed)
        self._calib: dict[str, tuple] = {}
        self._results: dict[str, ScenarioResult] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- shared steps
    def home(self, pid: str) -> tuple[Property, list[EnergyUsageMonth]]:
        prop, usage = load_property(pid), load_usage(pid)
        if usage and pid not in self.engine.usage:
            self.engine.usage[pid] = usage
        return prop, usage

    def calibrate(self, pid: str):
        """(CalibrationResult, PosteriorSamples), cached per home: SVI takes seconds, the demo calls it often."""
        with self._lock:
            if pid not in self._calib:
                prop, usage = self.home(pid)
                self._calib[pid] = self.twin.calibrate(prop, usage, [])
            return self._calib[pid]

    def feasibility(self, prop: Property, constraints: Optional[dict] = None) -> FeasibilityReport:
        """Layer 1 rules, then layer 2 (OpenAI screen on survivors; rules-only without a key or if disabled)."""
        import os

        if os.environ.get("GREENTWIN_FEASIBILITY_LAYER2", "1") == "0":
            return assess(prop, constraints)
        from greentwin.retrofit.feasibility_llm import assess_layer2

        return assess_layer2(prop, constraints)

    def value_model(self):
        return _value_model()

    # ---------------------------------------------------------------- Protocol
    def run_scenario(self, spec: ScenarioSpec) -> ScenarioResult:
        key = self._key(spec)
        if key in self._results:
            return self._results[key]
        f = CACHE / f"{key}.json"
        if self.disk_cache and f.exists():
            res = ScenarioResult.model_validate_json(f.read_text(encoding="utf-8"))
            self._results[key] = res
            return res
        res = self._run(spec)
        self._results[key] = res
        if self.disk_cache and res.provenance.engine != "mock":
            CACHE.mkdir(parents=True, exist_ok=True)
            f.write_text(res.model_dump_json(), encoding="utf-8")
        return res

    def _key(self, spec: ScenarioSpec) -> str:
        body = spec.model_dump(mode="json", exclude={"scenario_id", "requested_by"})
        body["versions"] = [CACHE_VERSION, MODEL_VERSION, cat.version(), self.n_samples]
        return f"{spec.property_id}-{hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]}"

    def _run(self, spec: ScenarioSpec) -> ScenarioResult:
        if spec.weather != "typical":
            raise NotImplementedError(f"weather {spec.weather!r}: only 'typical' runs today")
        prop, usage = self.home(spec.property_id)
        calib, samples = self.calibrate(spec.property_id)
        n = samples.n_samples
        eng = self.engine
        feas = self.feasibility(prop)
        actions = list(spec.retrofits)
        warnings: list[str] = []
        unknown = [a.retrofit_id for a in actions if a.retrofit_id not in cat.items()]
        if unknown:
            raise ValueError(f"unknown retrofit_id(s): {unknown}")
        status = {f.retrofit_id: f for f in feas.items}
        for a in actions:
            f = status.get(a.retrofit_id)
            if f is not None and f.status == "infeasible":
                warnings.append(f"{cat.get(a.retrofit_id).name} is ruled out for this home ({'; '.join(f.reasons)}); "
                                "the run shows what the physics would do, not a recommendation")
        if (c := cat.conflicts([a.retrofit_id for a in actions])):
            warnings.append(f"Mutually exclusive retrofits in one package: {c}")

        meter = eng.meter(prop, n)
        base_run, after_run = run_packages(samples, [[], actions])
        base, after = anchored(base_run, base_run, meter), anchored(after_run, base_run, meter)
        baseline = eng.baseline(prop, samples)
        after_bd = eng.package(prop, samples, actions) if actions else baseline

        carbon, carbon_reason = None, None
        if fuel_supported(prop):
            carbon = saving(base.carbon(), after.carbon(), "kgCO2e/yr", "carbon")
        else:
            carbon_reason = f"Main fuel {prop.heating.main_fuel}: no sourced emission factor"

        ladder = eng.ladder(prop, samples, feas, spec.goal)
        potential = eng.improvement_potential(prop, samples, feas)
        rating = eng.estimate_rating(prop, samples, actions) if actions else None
        peer = eng._peer(prop)
        score_before = self.scorer.score(prop, baseline, None, peer)
        score_after = (self.scorer.score(property_after(prop, actions), after_bd, rating, peer) if actions
                       else score_before)

        if spec.include_value:
            value = self.value_model().value_impact(prop, rating)
        else:
            from contracts.models import ValueImpact

            value = ValueImpact(current_value_gbp=prop.current_value_estimate_gbp, band_before=prop.epc.current_band,
                                method="unavailable", unavailable_reason="Value not requested for this scenario")

        heat_before, heat_after = [], []
        if spec.include_element_heat_loss:
            try:
                heat_before = self.twin.element_heat_loss(samples)
                heat_after = self.twin.element_heat_loss(samples, actions) if actions else heat_before
            except Exception as e:  # noqa: BLE001 - visual extra; the numbers above don't depend on it
                warnings.append(f"Element heat loss unavailable: {e}")

        pareto_points = eng.pareto(prop, samples, feas, None, spec.goal) if spec.include_pareto else []

        prov = eng.provenance(n, data_version=f"calibration {calib.method}")
        return ScenarioResult(
            scenario_id=spec.scenario_id, property_id=prop.property_id, provenance=prov, goal=spec.goal,
            calibration=calib, baseline=baseline, after=after_bd,
            electricity_saving_kwh=saving(base.electricity, after.electricity, "kWh/yr", "electricity use"),
            gas_saving_kwh=saving(base.gas, after.gas, "kWh/yr", "gas use"),
            purchased_energy_saving_kwh=saving(base.purchased, after.purchased, "kWh/yr", "purchased energy"),
            improvement_potential=potential, carbon_saving_kgco2e=carbon, carbon_unavailable_reason=carbon_reason,
            emission_factors=emission_factors(), emission_factor_source=emission_factor_source(),
            bill_saving_gbp=saving(base.bill(), after.bill(), "GBP/yr", "bill"),
            score_before=score_before, score_after=score_after, rating_after=rating, feasibility=feas,
            retrofits=ladder, value=value, element_heat_loss_before=heat_before, element_heat_loss_after=heat_after,
            pareto=pareto_points, data_confidence=calib.data_confidence,
            warnings=warnings + eng.warnings.get(prop.property_id, []),
        )

    # ---------------------------------------------------------------- Part 2 extras for the API / UI
    def plan(self, pid: str, goal: str = "bills", budget_gbp: Optional[float] = None,
             annual_budget_gbp: float = 5_000, reinvest: bool = True, customer_id: Optional[str] = None,
             min_carbon_kg: Optional[float] = None, max_cost_gbp: Optional[float] = None) -> dict:
        """Ladder + staging + Pareto (knee, P(on front), budget, reference point, memory re-rank, baselines)."""
        from greentwin.optimise.pareto import (
            co2_only_baseline, greedy_baseline, hypervolume, non_dominated, pareto, preference_rerank,
            reference_point, weights_from_memory,
        )
        from greentwin.optimise.staging import stage

        prop, _ = self.home(pid)
        _calib, samples = self.calibrate(pid)
        eng = self.engine
        feas = self.feasibility(prop)
        steps = eng.ladder(prop, samples, feas, goal)
        res = pareto(prop, samples, feas, eng.meter(prop, samples.n_samples), self.scorer, eng._peer(prop),
                     None, goal, self.seed)
        acceptance, weights, memory_note = None, {k: 1.0 for k in ("net_cost_gbp", "carbon_saving_kgco2e",
                                                                   "bill_saving_gbp", "score_gain")}, None
        if customer_id:
            try:
                from greentwin.memory import service as mem

                acceptance = {r: v["memory_score"] for r, v in mem.scores(customer_id, pid).items()}
                weights = weights_from_memory(mem.memory(customer_id)["confirmed"])
            except Exception as e:  # noqa: BLE001
                memory_note = f"Customer memory unavailable: {e}"
        cap = budget_gbp if budget_gbp is not None else 10_000
        g_pkg, g_dom = greedy_baseline(res, cap)
        c_pkg, c_dom = co2_only_baseline(res, cap)
        within = [i for i in range(len(res.packages)) if res.medians[i, 0] <= cap]
        front = [within[j] for j in np.flatnonzero(non_dominated(np.column_stack(
            [res.medians[within, 0], -res.medians[within, 1], -res.medians[within, 2], -res.medians[within, 3]])))]
        idx = {p: i for i, p in enumerate(res.packages)}
        hv = {"front": hypervolume(res, front, cap),
              "greedy": hypervolume(res, [idx[g_pkg]], cap) if g_pkg in idx else None,
              "co2_only": hypervolume(res, [idx[c_pkg]], cap) if c_pkg in idx else None}
        from greentwin.models.relevance import load as load_relevance

        rel_model = load_relevance()
        ok_ids = [f.retrofit_id for f in feas.items if f.status != "infeasible"]
        relevance = (rel_model.for_property(prop, ok_ids) if rel_model is not None
                     else {"unavailable": "Relevance model not trained (scripts/train_relevance.py)"})
        budget_points = [p.model_dump(mode="json") for p in res.points
                         if budget_gbp is None or p.net_cost_gbp.median <= budget_gbp]
        return {
            "property_id": pid, "goal": goal, "provenance": eng.provenance(samples.n_samples).model_dump(mode="json"),
            "ladder": [e.model_dump(mode="json") for e in steps],
            "staging": stage(steps, annual_budget_gbp, reinvest=reinvest).as_dict(),
            "pareto": {"points": budget_points, "knee_by_goal": {g: list(k) if k else None for g, k in res.knees.items()},
                       "excluded": res.excluded, "budget_gbp": budget_gbp},
            "reference_point": (reference_point(res, min_carbon_kg, max_cost_gbp, goal=goal)
                                if (min_carbon_kg is not None or max_cost_gbp is not None) else None),
            "preference_ranking": preference_rerank(res, weights, acceptance)[:10],
            "preference_weights": weights, "memory_note": memory_note,
            "baselines": {"budget_gbp": cap, "greedy": {"package": list(g_pkg), "dominated": g_dom},
                          "co2_only": {"package": list(c_pkg), "dominated": c_dom}, "hypervolume": hv},
            "assessor_relevance": relevance,
            "warnings": eng.warnings.get(pid, []),
        }


@lru_cache(maxsize=1)
def _value_model():
    from greentwin.value.model import HedonicValueModel

    return HedonicValueModel()


@lru_cache(maxsize=1)
def default() -> GreenTwinOrchestrator:
    return GreenTwinOrchestrator()


def spec_for(pid: str, retrofit_ids: Sequence[str], **kw) -> ScenarioSpec:
    ids = sorted(retrofit_ids, key=cat.sequence_position)
    sid = f"{pid}-" + ("+".join(ids) if ids else "baseline")
    return ScenarioSpec(scenario_id=sid, property_id=pid, retrofits=[RetrofitAction(retrofit_id=r) for r in ids], **kw)

