"""Customer memory: governed items (proposed -> confirmed | retracted), events, Beta acceptance score, ranking.

Memory changes the ORDER and framing of feasible options only; it never changes a computed number, never lifts a
rule veto and never hides the best base option. Every move carries a reason.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Optional

from greentwin.kb.store import execute, new_id, now, q
from greentwin.retrofit.rules import ITEMS, ORDER, RULES

GROUP = {rid: gi for gi, group in enumerate(RULES["ordering"]) for rid in group}
N_GROUPS = len(RULES["ordering"])

KINDS = ("constraint", "goal", "preference", "fact", "note")
CONSTRAINT_KEYS = {"tenure": ["renting", "owner"], "outdoor_space": ["none", "yes"], "roof": ["not_available", "available"],
                   "budget_gbp": None}
EVENT_WEIGHTS = {"accepted": (3.0, 0.0), "completed": (3.0, 0.0), "saved": (1.0, 0.0), "viewed": (0.2, 0.0),
                 "rejected": (0.0, 3.0), "dismissed": (0.0, 1.0)}
SIMILAR = 0.3        # events on another retrofit in the same category count at 30%
HALF_LIFE_DAYS = 180
LAMBDA = 0.3         # memory moves a score by at most +/-15%
LOW_DISRUPTION = {"draught_proofing", "loft_insulation", "cavity_wall_insulation", "smart_thermostat", "led_lighting",
                  "time_of_use_tariff", "secondary_glazing"}

PII = [re.compile(p, re.I) for p in (r"[\w.+-]+@[\w-]+\.[\w.]+",                  # email
                                    r"(?:\+44|0)\s?\d{3,4}\s?\d{3}\s?\d{3,4}",    # UK phone
                                    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",    # full postcode
                                    r"\b\d{1,4}\s+\w+\s+(street|road|lane|avenue|close|drive)\b")]  # address


def screen(text: str) -> Optional[str]:
    """Reason to refuse storing this text, or None if it is safe (no personal data)."""
    for p in PII:
        if p.search(text or ""):
            return "Looks like personal data (contact details, full postcode or address); not stored."
    return None


# --------------------------------------------------------------------------- items
def add_item(customer_id: str, kind: str, key: str, value: str, source: str = "user", evidence: str = "",
             status: Optional[str] = None) -> dict:
    if kind not in KINDS:
        return {"ok": False, "error": f"kind must be one of {KINDS}"}
    if not re.fullmatch(r"[a-z_]{1,40}", key or ""):
        return {"ok": False, "error": "key must be 1-40 lowercase letters or underscores"}
    value, evidence = str(value)[:200], str(evidence)[:300]
    why = screen(f"{key} {value} {evidence}")
    if why:
        return {"ok": False, "error": why}
    if kind == "constraint" and key not in CONSTRAINT_KEYS:
        return {"ok": False, "error": f"constraint key must be one of {list(CONSTRAINT_KEYS)}"}
    allowed = CONSTRAINT_KEYS.get(key) if kind == "constraint" else None
    if allowed and value not in allowed:
        return {"ok": False, "error": f"{key} must be one of {allowed}"}
    # LLM proposals start as 'proposed'; user/agent entries are confirmed directly.
    status = status or ("proposed" if source == "llm" else "confirmed")
    dup = q("SELECT item_id, status FROM memory_items WHERE customer_id = ? AND kind = ? AND key = ? AND value = ? "
            "AND status IN ('proposed', 'confirmed') ORDER BY created_at DESC LIMIT 1", [customer_id, kind, key, str(value)])
    if dup:  # already known: do not store twice (a user entry upgrades an open proposal)
        if dup[0]["status"] == "proposed" and status == "confirmed":
            set_status(dup[0]["item_id"], "confirmed")
        return {"ok": True, "item_id": dup[0]["item_id"], "status": "confirmed" if status == "confirmed" else dup[0]["status"],
                "note": "already remembered"}
    item_id = new_id("mem")
    execute("INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [item_id, customer_id, kind, key, str(value), status, source, evidence, now(), now()])
    return {"ok": True, "item_id": item_id, "status": status}


def set_status(item_id: str, status: str) -> dict:
    if status not in ("confirmed", "retracted"):
        return {"ok": False, "error": "status must be confirmed or retracted"}
    execute("UPDATE memory_items SET status = ?, updated_at = ? WHERE item_id = ?", [status, now(), item_id])
    return {"ok": True}


def add_event(customer_id: str, property_id: str, retrofit_id: str, type_: str, reason: str = "") -> dict:
    if type_ not in EVENT_WEIGHTS and type_ != "asked_about":
        return {"ok": False, "error": f"type must be one of {list(EVENT_WEIGHTS) + ['asked_about']}"}
    if screen(reason):
        reason = ""
    execute("INSERT INTO memory_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            [new_id("evt"), customer_id, property_id, retrofit_id, type_, reason, now()])
    return {"ok": True}


def memory(customer_id: str) -> dict:
    items = q("SELECT * FROM memory_items WHERE customer_id = ? ORDER BY created_at", [customer_id])
    latest = {}
    for it in items:  # latest confirmed value per (kind, key)
        if it["status"] == "confirmed":
            latest[(it["kind"], it["key"])] = it
    events = q("SELECT * FROM memory_events WHERE customer_id = ? ORDER BY created_at DESC LIMIT 30", [customer_id])
    cust = q("SELECT * FROM customers WHERE customer_id = ?", [customer_id])
    return {"customer_id": customer_id, "label": cust[0]["label"] if cust else None,
            "confirmed": list(latest.values()),
            "proposed": [i for i in items if i["status"] == "proposed"],
            "events": events}


def constraints(customer_id: Optional[str]) -> dict:
    if not customer_id:
        return {}
    return {it["key"]: it["value"] for it in memory(customer_id)["confirmed"] if it["kind"] == "constraint"}


# --------------------------------------------------------------------------- score
def _decay(ts: datetime) -> float:
    age = max(0.0, (now() - ts).total_seconds() / 86400)
    return 0.5 ** (age / HALF_LIFE_DAYS)


def scores(customer_id: Optional[str], property_id: str) -> dict:
    """Beta(alpha, beta) acceptance posterior + preference fit per retrofit -> memory score m with 90% interval."""
    from scipy.stats import beta as beta_dist
    events = q("SELECT * FROM memory_events WHERE customer_id = ? AND property_id = ?", [customer_id, property_id]) \
        if customer_id else []
    prefs = [it for it in memory(customer_id)["confirmed"] if it["kind"] == "preference"] if customer_id else []
    out = {}
    for rid, item in ITEMS.items():
        a, b, why = 1.0, 1.0, []
        for e in events:
            if e["type"] not in EVENT_WEIGHTS:
                continue
            same = e["retrofit_id"] == rid
            similar = not same and ITEMS.get(e["retrofit_id"], {}).get("category") == item["category"]
            if not (same or similar):
                continue
            w = (1.0 if same else SIMILAR) * _decay(e["created_at"])
            da, db = EVENT_WEIGHTS[e["type"]]
            a, b = a + w * da, b + w * db
            when = e["created_at"].strftime("%d %b")
            label = ITEMS.get(e["retrofit_id"], {}).get("name", e["retrofit_id"]).lower()
            why.append(f"you {e['type']} {'this' if same else 'a similar option (' + label + ')'} on {when}"
                       + (f" (reason: {e['reason']})" if e["reason"] else ""))
        f = 0.0
        for p in prefs:
            k, v = p["key"], p["value"].lower()
            if k == "low_disruption" and v in ("yes", "true"):
                d = 0.5 if rid in LOW_DISRUPTION else -0.5
                f += d
                why.append("you prefer low-disruption work" + ("" if d > 0 else " (this is disruptive)"))
            elif k == "avoid" and v in (rid, item["category"]):
                f -= 1.0
                why.append(f"you said you want to avoid {v.replace('_', ' ')}")
            elif k == "interested_in" and v in (rid, item["category"]):
                f += 1.0
                why.append(f"you said you are interested in {v.replace('_', ' ')}")
        f = max(-1.0, min(1.0, f))
        mean = a / (a + b)
        m = max(0.0, min(1.0, mean + 0.15 * f))
        lo, hi = beta_dist.ppf([0.05, 0.95], a, b)
        out[rid] = {"alpha": round(a, 2), "beta": round(b, 2), "p_accept": round(mean, 3),
                    "interval_90": [round(float(lo), 3), round(float(hi), 3)], "preference_fit": round(f, 2),
                    "memory_score": round(m, 3), "reasons": why}
    return out


def rank(prop, customer_id: Optional[str]) -> dict:
    """Feasible options in base order (fabric first) with a bounded memory nudge; infeasible listed with reasons."""
    from greentwin.retrofit.rules import assess
    cons = constraints(customer_id)
    report = assess(prop, cons)
    sc = scores(customer_id, prop.property_id)
    feasible = [f for f in report.items if f.status != "infeasible"]
    rows = []
    for f in feasible:
        # Base utility from the fabric-first GROUP order. Within a group the config says "order by expected net
        # benefit", which needs physics we do not have yet, so items in a group tie and memory breaks the tie.
        base = 1.0 - GROUP.get(f.retrofit_id, N_GROUPS) / N_GROUPS
        m = sc[f.retrofit_id]["memory_score"]
        rows.append({"retrofit_id": f.retrofit_id, "name": ITEMS[f.retrofit_id]["name"],
                     "category": ITEMS[f.retrofit_id]["category"], "status": f.status, "conditions": f.reasons,
                     "base_score": round(base, 3), "memory": sc[f.retrofit_id],
                     "rank_score": round(base * (1 + LAMBDA * (m - 0.5)), 4),
                     "cost_central_gbp": ITEMS[f.retrofit_id].get("cost_gbp_central")
                     if ITEMS[f.retrofit_id]["cost_status"] != "unsourced" else None,
                     "cost_basis": ITEMS[f.retrofit_id].get("cost_basis"),
                     "cost_status": ITEMS[f.retrofit_id]["cost_status"]})
    pos = {rid: i for i, rid in enumerate(ORDER)}
    base_order = [r["retrofit_id"] for r in sorted(rows, key=lambda r: (-r["base_score"], pos.get(r["retrofit_id"], 99)))]
    rows.sort(key=lambda r: (-r["rank_score"], pos.get(r["retrofit_id"], 99)))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["base_rank"] = base_order.index(r["retrofit_id"]) + 1
        r["moved"] = r["base_rank"] - r["rank"]  # >0 moved up
        r["is_best_base"] = r["base_rank"] == 1
        if r["moved"]:
            direction = "up" if r["moved"] > 0 else "down"
            r["move_reason"] = f"Moved {direction} {abs(r['moved'])}: " + ("; ".join(r["memory"]["reasons"]) or
                                                                         "memory score")
    return {"property_id": prop.property_id, "customer_id": customer_id, "constraints_applied": cons,
            "base_order_source": "Fabric-first group order from contracts/feasibility_rules.json (fabric, glazing, "
                                 "controls, heating, generation, storage, tariff). Within a group, options tie until "
                                 "physics savings are available (scenario engine pending); no savings figures are used.",
            "memory_rule": "Memory can move an option by at most ±15% of its score; it never changes numbers, "
                           "never lifts a rule veto, and the best base option is always shown.",
            "recommendations": rows,
            "not_feasible": [{"retrofit_id": f.retrofit_id, "name": ITEMS[f.retrofit_id]["name"],
                              "reasons": f.reasons, "codes": f.reason_codes}
                             for f in report.items if f.status == "infeasible"]}
