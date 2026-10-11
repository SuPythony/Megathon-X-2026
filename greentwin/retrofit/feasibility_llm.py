"""Feasibility layer 2 (OpenAI fallback for Jev): a second look at the retrofits that PASS the rules.

Rules (layer 1) run first and always win: the model only sees survivors and can only keep or DOWNGRADE them
(feasible -> feasible_with_conditions -> needs_survey). It never lifts a veto, never marks anything infeasible and
never produces a number. Shown reasons are fixed texts keyed by a code the model picks (Feasibility.reasons is never
LLM wording). Confidence below TAU -> needs_survey. Structured output on `gpt-6-luna`; every call is cached in
cache/llm so runs replay without a key. No key or any error -> the rules report unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from contracts.models import Feasibility, FeasibilityReport, Property

from .rules import ITEMS, assess

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "llm" / "feasibility"
PROMPT_VERSION = "feas-l2/1"
TAU = 0.6
RANK = {"feasible": 0, "feasible_with_conditions": 1, "needs_survey": 2}
REASONS = {
    "NONE": None,
    "ACCESS": "Access for installation needs checking (scaffold, loft hatch, roof or floor void).",
    "STRUCTURE": "Structure or fabric condition should be checked before installing.",
    "MOISTURE": "Damp or condensation risk: needs a moisture and ventilation check.",
    "SPACE": "Space for the equipment (indoor or outdoor unit, cylinder, battery) needs confirming.",
    "PLANNING": "May need planning or freeholder/landlord consent.",
    "ELECTRICAL": "Electrical supply or consumer unit capacity needs checking.",
    "HEAT_EMITTERS": "Radiators and pipework may need upsizing for low-temperature heating.",
    "LOW_CONFIDENCE": "Not enough information to be sure: a survey is needed.",
}
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["retrofit_id", "status", "condition_code", "confidence"],
        "properties": {
            "retrofit_id": {"type": "string"},
            "status": {"type": "string", "enum": list(RANK)},
            "condition_code": {"type": "string", "enum": list(REASONS)},
            "confidence": {"type": "number"},
        }}}},
}
SYSTEM = ("You screen home retrofit feasibility for a UK home AFTER hard rules have run. For each candidate retrofit "
          "return status feasible, feasible_with_conditions or needs_survey, one condition_code, and your confidence "
          "(0-1) that the status is right. Be conservative: if a practical issue is plausible from the facts, pick "
          "feasible_with_conditions or needs_survey. Use only the facts given. Do not estimate costs or savings.")


def _facts(prop: Property, memory: Optional[dict]) -> dict:
    d = prop.model_dump(mode="json", exclude={"customer_id", "current_value_estimate_gbp", "split"})
    if memory:
        d["customer_memory"] = memory
    return d


def _call(payload: dict) -> Optional[dict]:
    from greentwin.llm import client as llm

    key = hashlib.sha256(json.dumps({"v": PROMPT_VERSION, "m": llm.MODEL, **payload}, sort_keys=True,
                                    default=str).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if not llm.has_key():
        return None
    r = llm.client().responses.create(
        model=llm.MODEL, instructions=SYSTEM, input=json.dumps(payload, default=str),
        text={"format": {"type": "json_schema", "name": "feasibility", "schema": SCHEMA, "strict": True}})
    out = json.loads(r.output_text)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


def assess_layer2(prop: Property, constraints: Optional[dict] = None,
                  memory: Optional[dict] = None) -> FeasibilityReport:
    """Rules, then the OpenAI screen on the survivors. Falls back to the rules report on no key or any error."""
    report = assess(prop, constraints)
    survivors = [f for f in report.items if f.status in ("feasible", "feasible_with_conditions")]
    if not survivors:
        return report
    payload = {"home": _facts(prop, memory), "candidates": [
        {"retrofit_id": f.retrofit_id, "name": ITEMS[f.retrofit_id]["name"], "rule_status": f.status,
         "rule_conditions": f.reasons} for f in survivors]}
    try:
        out = _call(payload)
    except Exception:  # noqa: BLE001 - layer 2 is advisory; rules stand
        out = None
    if not out:
        return report
    by_id = {x["retrofit_id"]: x for x in out.get("items", []) if x.get("retrofit_id") in ITEMS}
    items = []
    for f in report.items:
        x = by_id.get(f.retrofit_id)
        if f.status not in ("feasible", "feasible_with_conditions") or x is None:
            items.append(f)
            continue
        conf = max(0.0, min(1.0, float(x.get("confidence", 0))))
        status, code = x.get("status"), x.get("condition_code", "NONE")
        if status not in RANK:
            items.append(f)
            continue
        if conf < TAU:
            status, code = "needs_survey", "LOW_CONFIDENCE"
        if RANK[status] <= RANK[f.status]:  # never upgrade past the rules
            items.append(f.model_copy(update={"p_feasible": round(conf, 3)}) if status == f.status else f)
            continue
        why = REASONS.get(code) or REASONS["LOW_CONFIDENCE"]
        items.append(Feasibility(retrofit_id=f.retrofit_id, status=status, decided_by="llm_fallback",
                                 reason_codes=f.reason_codes + [f"L2_{code}"], reasons=f.reasons + [why],
                                 p_feasible=round(conf, 3)))
    return report.model_copy(update={"items": items, "router": "llm_fallback"})
