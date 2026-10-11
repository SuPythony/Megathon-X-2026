"""Feasibility layer 2 (Jev System-1): a second look at the retrofits that PASS the rules.

Rules (layer 1) run first and always win: the model only sees survivors and can only keep or DOWNGRADE them
(feasible -> feasible_with_conditions -> needs_survey). It never lifts a veto, never marks anything infeasible and
never produces a number. Shown reasons are fixed texts keyed by a code Jev picks (Feasibility.reasons is never
model wording). Jev changes a status only when its choice has probability >= 0.5 and is stricter; otherwise
the rules status stands and p_feasible = 1 - P(needs_survey). Calls go through greentwin.jev (cached in
cache/jev). Jev disabled, no key or any error -> the rules report unchanged.
"""
from __future__ import annotations

from typing import Optional

from contracts.models import Feasibility, FeasibilityReport, Property

from .rules import ITEMS, assess

PROMPT_VERSION = "feas-l2/2"
P_ACT = 0.5  # Jev's chosen status must have probability >= 0.5 to change anything
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
STATUS_CRITERIA = {
    "feasible": "Can be installed as normal in this home; no practical issue is apparent from the facts.",
    "feasible_with_conditions": "Can be installed, but a specific practical condition must be handled first.",
    "needs_survey": "Too uncertain from the facts: an on-site survey is needed before recommending it.",
}
SYSTEM = ("UK home retrofit feasibility screen, after hard rules have already passed this retrofit: {name}. "
          "Using only the home facts given, can it be installed as normal? Be conservative: if a practical issue is "
          "plausible, choose feasible_with_conditions or needs_survey.")
CONDITION = "Which practical issue most affects installing {name} in this home? NONE if there is none."


def _facts(prop: Property, memory: Optional[dict]) -> dict:
    d = prop.model_dump(mode="json", exclude={"customer_id", "current_value_estimate_gbp", "split"})
    if memory:
        d["customer_memory"] = memory
    return d


def _call(payload: dict) -> Optional[dict]:
    """One Jev request for all survivors: per retrofit a status choice and a condition-code choice.
    Returns {"items": [{retrofit_id, status, condition_code, confidence}]} or None."""
    from greentwin import jev

    qs = {}
    for c in payload["candidates"]:
        rid, name = c["retrofit_id"], c["name"]
        qs[f"{rid}__status"] = {"type": "choice", "instructions": SYSTEM.format(name=name), "criteria": STATUS_CRITERIA}
        qs[f"{rid}__condition"] = {"type": "choice", "instructions": CONDITION.format(name=name),
                                   "criteria": {k: (v or "No particular issue.") for k, v in REASONS.items()
                                                if k != "LOW_CONFIDENCE"}}
    ans = jev.systemone("feasibility", {"v": PROMPT_VERSION, "home": payload["home"]}, qs)
    if not ans:
        return None
    items = []
    for c in payload["candidates"]:
        st, co = ans.get(f"{c['retrofit_id']}__status"), ans.get(f"{c['retrofit_id']}__condition")
        if not st or "choice" not in st:
            continue
        items.append({"retrofit_id": c["retrofit_id"], "status": st["choice"],
                      "condition_code": (co or {}).get("choice", "NONE"), "confidence": st.get("confidence", 0.0),
                      "probabilities": st.get("probabilities", {})})
    return {"items": items}


def assess_layer2(prop: Property, constraints: Optional[dict] = None,
                  memory: Optional[dict] = None) -> FeasibilityReport:
    """Rules, then Jev on the survivors. Falls back to the rules report when Jev is off, keyless or fails."""
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
        status, code = x.get("status"), x.get("condition_code", "NONE")
        if status not in RANK:
            items.append(f)
            continue
        probs = {k: float(v) for k, v in (x.get("probabilities") or {status: x.get("confidence", 0.0)}).items()}
        p_go = round(max(0.0, min(1.0, 1.0 - probs.get("needs_survey", 0.0))), 3)  # P(can go ahead)
        if probs.get(status, 0.0) < P_ACT or RANK[status] <= RANK[f.status]:
            # Jev unsure or not stricter: the rules status stands, with Jev's P(can go ahead) as p_feasible
            items.append(f.model_copy(update={"p_feasible": p_go, "decided_by": "jev"}))
            continue
        if status == "feasible_with_conditions" and code in ("NONE", "LOW_CONFIDENCE"):
            items.append(f.model_copy(update={"p_feasible": p_go, "decided_by": "jev"}))  # no named condition
            continue
        if not REASONS.get(code):
            code = "LOW_CONFIDENCE"
        why = REASONS[code]
        items.append(Feasibility(retrofit_id=f.retrofit_id, status=status, decided_by="jev",
                                 reason_codes=f.reason_codes + [f"L2_{code}"], reasons=f.reasons + [why],
                                 p_feasible=p_go))
    return report.model_copy(update={"items": items, "router": "jev", "jev_used": True})
