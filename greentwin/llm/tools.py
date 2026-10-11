"""Function tools for the grounded chat (strict JSON schemas) and their handlers.

Tools return compact JSON with units and 90% ranges already attached, so the model quotes and never derives.
Home-specific numbers come only from these tools.
"""
from __future__ import annotations

import json
from functools import lru_cache

from contracts.models import EnergyUsageMonth, Property
from greentwin.kb.store import q
from greentwin.memory import service as mem
from greentwin.retrofit.rules import ITEMS, assess


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props), "additionalProperties": False}


S = {"type": "string"}
NS = {"type": ["string", "null"]}
TOOLS = [
    {"type": "function", "name": "get_home_facts", "strict": True,
     "description": "Facts about one home: type, age, size, walls, roof, glazing, heating, EPC, constraints, data gaps.",
     "parameters": _obj({"property_id": S})},
    {"type": "function", "name": "get_energy_prior", "strict": True,
     "description": "Estimated electricity and gas intensity (kWh/m2/yr) for the home from the M04 model, with 90% "
                    "ranges and the main factors behind it. Set include_meter_update to also get the estimate "
                    "updated with the home's own 12 months of meter readings.",
     "parameters": _obj({"property_id": S, "include_meter_update": {"type": "boolean"}})},
    {"type": "function", "name": "run_scenario", "strict": True,
     "description": "Rerun the home's physics model with a set of retrofits to get savings, bills and carbon. Use for "
                    "every what-if question.",
     "parameters": _obj({"property_id": S, "retrofit_ids": {"type": "array", "items": {"type": "string", "enum": list(ITEMS)}}})},
    {"type": "function", "name": "rank_recommendations", "strict": True,
     "description": "Feasible retrofits for the home in recommended order, personalised by the customer's memory "
                    "(with the reason for every move), plus options that are not feasible and why.",
     "parameters": _obj({"property_id": S})},
    {"type": "function", "name": "search_docs", "strict": True,
     "description": "Search the knowledge base (retrofit guides, feasibility rules, grants, methods, wording rules, "
                    "datasets). Returns cited passages. kind filters: retrofit, rule, grant, method, faq, dataset.",
     "parameters": _obj({"query": S, "kind": NS})},
    {"type": "function", "name": "query_graph", "strict": True,
     "description": "Fixed graph queries. template: blocked_retrofits(property_id), conditions(property_id), "
                    "dependencies(retrofit_id), grants(retrofit_id), excludes(retrofit_id).",
     "parameters": _obj({"template": {"type": "string", "enum": ["blocked_retrofits", "conditions", "dependencies",
                                                                 "grants", "excludes"]},
                         "property_id": NS, "retrofit_id": {"type": ["string", "null"], "enum": list(ITEMS) + [None]}})},
    {"type": "function", "name": "get_customer_memory", "strict": True,
     "description": "What we remember about the current customer (the signed-in session's customer only): "
                    "confirmed constraints, goals, preferences, facts, unconfirmed proposals and recent actions.",
     "parameters": _obj({"topic": NS})},
    {"type": "function", "name": "propose_memory", "strict": True,
     "description": "Propose remembering something the customer said (stored as unconfirmed until they confirm). "
                    "kind: constraint (keys: tenure=renting|owner, outdoor_space=none|yes, roof=not_available|"
                    "available, budget_gbp=<number they stated>), goal (key goal: bills|carbon|comfort), preference "
                    "(keys: low_disruption=yes, avoid=<retrofit_id or category>, interested_in=<retrofit_id or "
                    "category>), fact, note. evidence must quote the customer's own words.",
     "parameters": _obj({"kind": {"type": "string", "enum": list(mem.KINDS)}, "key": S, "value": S, "evidence": S})},
]


@lru_cache(maxsize=8)
def _prop(pid: str) -> Property:
    rows = q("SELECT data FROM properties WHERE property_id = ?", [pid])
    if not rows:
        raise KeyError(f"unknown property_id {pid}; known: {[r['property_id'] for r in q('SELECT property_id FROM properties')]}")
    return Property.model_validate(rows[0]["data"])


def _dist(d: dict) -> dict:
    ci = next(c for c in d["credible_intervals"] if c["level"] == 0.9)
    out = {"unit": d["unit"], "median": round(d["median"], 1), "mean": round(d["mean"], 1),
           "range_90": [round(ci["lo"], 1), round(ci["hi"], 1)], "std": round(d["std"], 1)}
    if d.get("tail_probabilities"):
        t = d["tail_probabilities"][0]
        out["p_above_peers"] = {"label": t["label"], "peer_median": round(t["threshold"], 1),
                                "probability": round(t["probability"], 2)}
    return out


def _v(x):
    return getattr(x, "value", x)


def get_home_facts(property_id: str) -> dict:
    p = _prop(property_id)
    lbl = q("SELECT label FROM properties WHERE property_id = ?", [property_id])[0]["label"]
    usage = q("SELECT count(*) n, count(gas_kwh) g, count(electricity_import_kwh) e FROM usage_monthly WHERE property_id = ?",
              [property_id])[0]
    return {"property_id": property_id, "label": lbl, "district": p.location.postcode_district,
            "region": p.location.region, "type": f"{_v(p.property_type)} ({_v(p.built_form).replace('_', ' ')})",
            "age_band": p.construction_age_band, "floor_area_m2": p.floor_area_m2,
            "walls": f"{_v(p.walls.type)} wall, insulation {_v(p.walls.insulation)}",
            "roof": f"{_v(p.roof.type)}, " + (f"loft insulation {p.roof.loft_insulation_mm} mm"
                                             if p.roof.loft_insulation_mm is not None else "no loft"),
            "glazing": [f"{g.share_pct:g}% {_v(g.type)}" for g in p.glazing],
            "heating": f"{_v(p.heating.main_system)} ({p.heating.main_fuel})",
            "epc": {"band": _v(p.epc.current_band) if p.epc.current_band else None, "sap": p.epc.sap_score},
            "constraints": {k: v for k, v in p.constraints.model_dump().items() if v not in (None, [], False)},
            "meter_months": {"total": usage["n"], "electricity_present": usage["e"], "gas_present": usage["g"]}}


def get_energy_prior(property_id: str, include_meter_update: bool) -> dict:
    from greentwin.models.energy_prior import load_default
    p = _prop(property_id)
    model = load_default()
    prior = model.energy_prior(p)
    out = {"property_id": property_id, "model": prior.provenance.model_version, "engine": prior.provenance.engine,
           "label": "estimate from home type (M04), before meter readings", "electricity": _dist(prior.electricity_kwh_m2.model_dump()),
           "gas": _dist(prior.gas_kwh_m2.model_dump()) if prior.gas_kwh_m2.std > 0 else
           {"unit": "kWh/m2/yr", "median": 0, "note": "No mains gas supply: gas is structurally 0 (not missing data)"},
           "top_factors_electricity": [{"factor": c.feature, "value": c.value, "kwh_m2": c.contribution}
                                       for c in prior.contributions_electricity[:5]],
           "top_factors_gas": [{"factor": c.feature, "value": c.value, "kwh_m2": c.contribution}
                               for c in prior.contributions_gas[:5]]}
    if include_meter_update:
        from greentwin.models.prior_update import update_with_meters
        usage = [EnergyUsageMonth(**{k: v for k, v in r.items() if k != "property_id"}, property_id=property_id)
                 for r in q("SELECT * FROM usage_monthly WHERE property_id = ? ORDER BY month", [property_id])]
        upd = update_with_meters(model, p, usage, n=8000)
        out["after_meter_readings"] = {}
        for fuel, v in upd["fuels"].items():
            if v.get("posterior"):
                out["after_meter_readings"][fuel] = {
                    "metered_intensity_kwh_m2": v["observed"]["intensity_kwh_m2"],
                    "metered_annual_kwh": v["observed"]["annual_kwh"],
                    "imputed_months": v["observed"]["imputed_months"],
                    "updated_estimate": _dist(v["posterior"]), "spread_reduction_pct": v["sd_reduction_pct"]}
            else:
                out["after_meter_readings"][fuel] = {"note": v.get("note")}
    return out


def run_scenario(property_id: str, retrofit_ids: list[str]) -> dict:
    p = _prop(property_id)
    feas = {f.retrofit_id: f for f in assess(p).items}
    return {"status": "unavailable",
            "reason": "The physics scenario engine (digital twin) is not connected yet, so savings, bills and carbon "
                      "for this package cannot be computed. No estimate is given instead.",
            "feasibility": {r: {"status": feas[r].status, "reasons": feas[r].reasons} if r in feas
                            else {"status": "unknown retrofit_id"} for r in retrofit_ids}}


def rank_recommendations(property_id: str, customer_id: str | None) -> dict:
    r = mem.rank(_prop(property_id), customer_id)
    return {"base_order_source": r["base_order_source"], "memory_rule": r["memory_rule"],
            "constraints_applied": r["constraints_applied"],
            "recommendations": [{"rank": x["rank"], "retrofit_id": x["retrofit_id"], "name": x["name"],
                                 "status": x["status"], "conditions": x["conditions"], "moved": x["moved"],
                                 "move_reason": x.get("move_reason"), "is_best_base": x["is_best_base"],
                                 "p_accept": x["memory"]["p_accept"],
                                 "typical_cost_gbp": x["cost_central_gbp"], "cost_basis": x["cost_basis"],
                                 "cost_status": x["cost_status"]} for x in r["recommendations"]],
            "not_feasible": r["not_feasible"]}


def search_docs(query: str, kind: str | None) -> dict:
    rows = q("""SELECT doc_id, kind, title, text, source_url, verified,
                       fts_main_documents.match_bm25(doc_id, ?) AS score
                FROM documents WHERE score IS NOT NULL AND (? IS NULL OR kind = ?)
                ORDER BY score DESC LIMIT 5""", [query, kind, kind])
    return {"query": query, "results": [{"doc_id": r["doc_id"], "kind": r["kind"], "title": r["title"],
                                         "snippet": r["text"][:700], "source_url": r["source_url"],
                                         "verified": r["verified"]} for r in rows]}


def query_graph(template: str, property_id: str | None, retrofit_id: str | None) -> dict:
    if template in ("blocked_retrofits", "conditions"):
        if not property_id:
            return {"error": "property_id required"}
        want = ("infeasible",) if template == "blocked_retrofits" else ("feasible_with_conditions", "needs_survey")
        return {"template": template, "rows": [{"retrofit_id": f.retrofit_id, "name": ITEMS[f.retrofit_id]["name"],
                                                 "status": f.status, "rules": f.reason_codes, "reasons": f.reasons}
                                                for f in assess(_prop(property_id)).items if f.status in want]}
    if not retrofit_id:
        return {"error": "retrofit_id required"}
    rel = {"dependencies": ("REQUIRES_AFTER", "src"), "excludes": ("EXCLUDES", "src"), "grants": ("GRANT_APPLIES", "dst")}[template]
    col, other = ("src", "dst") if rel[1] == "src" else ("dst", "src")
    rows = q(f"SELECT {other} AS id FROM edges WHERE rel = ? AND {col} = ?", [rel[0], retrofit_id])
    if template == "grants":
        docs = {d["doc_id"]: d for d in q("SELECT doc_id, text, verified FROM documents WHERE kind = 'grant'")}
        return {"template": template, "rows": [{"grant": r["id"], "doc_id": f"grant:{r['id']}",
                                                 "text": docs.get(f"grant:{r['id']}", {}).get("text"),
                                                 "verified": docs.get(f"grant:{r['id']}", {}).get("verified")} for r in rows]}
    return {"template": template, "rows": [{"retrofit_id": r["id"], "name": ITEMS.get(r["id"], {}).get("name")} for r in rows]}


def get_customer_memory(customer_id: str | None, topic: str | None = None) -> dict:
    if not customer_id:
        return {"note": "Anonymous session: nothing is remembered."}
    m = mem.memory(customer_id)
    short = lambda it: {"kind": it["kind"], "key": it["key"], "value": it["value"], "source": it["source"]}  # noqa: E731
    return {"customer_id": customer_id, "label": m["label"], "confirmed": [short(i) for i in m["confirmed"]],
            "unconfirmed": [short(i) for i in m["proposed"]],
            "recent_actions": [{"retrofit_id": e["retrofit_id"], "action": e["type"], "reason": e["reason"],
                                "date": e["created_at"].strftime("%d %b %Y")} for e in m["events"][:10]]}


def propose_memory(customer_id: str | None, kind: str, key: str, value: str, evidence: str) -> dict:
    if not customer_id:
        return {"ok": False, "error": "Anonymous session: memory is not stored."}
    return mem.add_item(customer_id, kind, key, value, source="llm", evidence=evidence)


HANDLERS = {"get_home_facts": get_home_facts, "get_energy_prior": get_energy_prior, "run_scenario": run_scenario,
            "rank_recommendations": rank_recommendations, "search_docs": search_docs, "query_graph": query_graph,
            "get_customer_memory": get_customer_memory, "propose_memory": propose_memory}


SESSION_SCOPED = {"rank_recommendations", "get_customer_memory", "propose_memory"}


def call(name: str, args: dict, customer_id: str | None = None) -> dict:
    """Run a tool. customer_id always comes from the server-side session, never from model arguments, so the
    model cannot read or write another customer's memory (prompt-injection safe)."""
    try:
        args = {k: v for k, v in args.items() if k != "customer_id"}
        if name in SESSION_SCOPED:
            args["customer_id"] = customer_id
        return HANDLERS[name](**args)
    except Exception as e:  # tool errors go back to the model as data, never dropped
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    for n, a in [("get_energy_prior", {"property_id": "SH2", "include_meter_update": True}),
                 ("search_docs", {"query": "cavity wall grant", "kind": None}),
                 ("query_graph", {"template": "blocked_retrofits", "property_id": "SH3", "retrofit_id": None})]:
        print(n, json.dumps(call(n, a))[:600])
