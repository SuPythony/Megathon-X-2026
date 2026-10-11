"""Build the knowledge base (idempotent):  uv run python -m greentwin.kb.build

Loads homes, usage and configs; renders the document corpus (retrofits, rules, grants, factors, tariffs, M04
method, wording rules, datasets); builds config-graph edges and the BM25 full-text index; stores M04 priors;
seeds two synthetic demo customers.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

from contracts.models import EnergyUsageMonth, Property
from greentwin.kb.store import ROOT, con, execute, now, q

CFG = ROOT / "contracts"
LABELS = {"SH1": "Victorian terrace, M14 (conservation area)", "SH2": "1970s cavity semi, BS16",
          "SH3": "All-electric 3rd-floor flat, CR0"}
DEMO_CUSTOMERS = [("C-001", "Demo customer A (owner, SH2)", "SH2"), ("C-002", "Demo customer B (SH1)", "SH1")]


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _doc(doc_id, kind, title, text, applies_to=None, url=None, verified=True):
    return {"doc_id": doc_id, "kind": kind, "title": title, "text": text.strip(), "applies_to": applies_to or {},
            "source_url": url, "verified": verified}


def render_documents() -> list[dict]:
    cat = json.loads((CFG / "retrofit_catalogue.json").read_text())
    rules = json.loads((CFG / "feasibility_rules.json").read_text())
    docs = []
    blocked_by = {}
    for r in rules["rules"]:
        for rid in r.get("blocks", []) + list(r.get("downgrade", {})):
            blocked_by.setdefault(rid, []).append(r["id"])
    for it in cat["items"]:
        cost = ("Cost: not sourced, so no cost figure is shown." if it["cost_status"] == "unsourced" else
                f"Typical cost (central): £{it['cost_gbp_central']:,.0f} {it['cost_basis'].replace('_', ' ')}"
                f" (range £{it.get('cost_gbp_low') or 0:,.0f}–£{it.get('cost_gbp_high') or 0:,.0f}),"
                f" price year {it.get('cost_price_year')}, status {it['cost_status'].replace('_', ' ')}.")
        grants = ", ".join(it.get("grants") or []) or "none listed"
        text = f"""
{it['name']} ({it['category']}).
What it changes in the home model: {', '.join(f'{k} {v}' for k, v in it['twin_edits'].items())}.
{cost}
Expected lifetime: {it.get('lifetime_years') or 'not stated'} years. Grants: {grants}.
Should come after: {', '.join(it.get('depends_on') or []) or 'nothing'}. Cannot be combined with: {', '.join(it.get('excludes') or []) or 'nothing'}.
Feasibility rules that can block or condition it: {', '.join(blocked_by.get(it['id'], [])) or 'none'}.
Source: {it['source']}"""
        docs.append(_doc(f"retrofit:{it['id']}", "retrofit", it["name"], text, {"retrofit_id": it["id"]}))
    for r in rules["rules"]:
        acts = ", ".join(r.get("blocks", [])) or "nothing (conditions only)"
        down = "; ".join(f"{k} becomes {v.replace('_', ' ')}" for k, v in r.get("downgrade", {}).items())
        docs.append(_doc(f"rule:{r['id']}", "rule", f"Rule {r['id']}: {r['reason']}",
                         f"Feasibility rule {r['id']}. Reason shown to customers: {r['reason']}. Blocks: {acts}. "
                         f"{('Conditions: ' + down + '.') if down else ''} Condition checked: {r['when']}. "
                         f"Rule vetoes always win; they cannot be overridden by the LLM, Jev or customer memory.",
                         {"retrofit_ids": r.get("blocks", []) + list(r.get("downgrade", {}))}))
    for gid, text in cat["grants"].items():
        unverified = "confirm" in text.lower()
        url = (re.findall(r"https?://\S+?(?=[)\s]|$)", text) or [None])[0]
        docs.append(_doc(f"grant:{gid}", "grant", gid.replace("_", " "),
                         text + (" Status: NOT VERIFIED, do not quote amounts as confirmed." if unverified else ""),
                         {"grant_id": gid}, url, not unverified))
    ef = json.loads((CFG / "emission_factors.json").read_text())
    docs.append(_doc("method:emission_factors", "method", "Carbon emission factors",
                     "Carbon = sum over fuels of energy change times the emission factor. " +
                     json.dumps({k: v for k, v in ef.items() if not k.startswith("_")})[:1500],
                     url="https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting"))
    tf = json.loads((CFG / "tariffs.json").read_text())
    docs.append(_doc("method:tariffs", "method", "Energy prices used for bills",
                     "Bills use the Ofgem price cap for the current quarter. " +
                     json.dumps({k: v for k, v in tf.items() if not k.startswith("_")})[:1500],
                     url="https://www.ofgem.gov.uk/energy-price-cap"))
    card = (ROOT / "docs/M04_ENERGY_PRIOR.md")
    if card.exists():
        for sec in re.split(r"\n(?=## )", card.read_text()):
            title = sec.splitlines()[0].lstrip("# ").strip()
            if len(sec) > 80:
                docs.append(_doc(f"method:m04:{re.sub(r'[^a-z0-9]+', '_', title.lower())[:40]}", "method",
                                 f"Energy prior model (M04): {title}", sec[:3500]))
    agents = (ROOT / "AGENTS.md").read_text()
    for name in ("Core philosophies", "Core rules", "Compliance fixes"):
        m = re.search(rf"## {name}.*?(?=\n## )", agents, re.S)
        if m:
            docs.append(_doc(f"faq:{name.lower().replace(' ', '_')}", "faq", f"GreenTwin {name.lower()}", m.group(0)[:4000]))
    sys.path.insert(0, str(ROOT / "deliverables" / "src"))
    try:
        from datasets import DATASETS  # noqa: PLC0415
        for d in DATASETS:
            docs.append(_doc(f"dataset:{re.sub(r'[^a-z0-9]+', '_', d['name'].lower())[:40]}", "dataset", d["name"],
                             f"{d['name']} ({d.get('publisher', '')}). Used for: {d.get('use', '')}. Licence: "
                             f"{d.get('licence', '')}. Coverage: {d.get('coverage', '')}. Notes: {d.get('gotchas', '')}",
                             url=d.get("url")))
    except Exception as e:  # datasets module is optional for the KB
        print("datasets skipped:", e)
    return docs


def build() -> dict:
    con()
    for t in ("properties", "usage_monthly", "configs", "documents", "edges", "energy_priors"):
        execute(f"DELETE FROM {t}")
    # Homes + usage
    props = {}
    for sh in LABELS:
        p = Property.model_validate_json((CFG / f"examples/{sh}_property.json").read_text())
        props[sh] = p
        execute("INSERT INTO properties VALUES (?, ?, ?, 'showcase')", [sh, LABELS[sh], p.model_dump_json()])
        for u in json.loads((CFG / f"examples/{sh}_usage_2025.json").read_text()):
            u = EnergyUsageMonth.model_validate(u)
            execute("INSERT INTO usage_monthly VALUES (?, ?, ?, ?, ?)",
                    [sh, u.month, u.electricity_import_kwh, u.gas_kwh, u.estimated_read])
    # Configs registry
    for f in sorted(CFG.glob("*.json")):
        d = json.loads(f.read_text())
        execute("INSERT INTO configs VALUES (?, ?, ?, ?)", [f.stem, str(d.get("version", "")), _h(d), json.dumps(d)])
    # Documents + full-text index
    docs = render_documents()
    for d in docs:
        execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [d["doc_id"], d["kind"], d["title"], d["text"], json.dumps(d["applies_to"]), d["source_url"],
                 d["verified"], _h(d)])
    execute("PRAGMA create_fts_index('documents', 'doc_id', 'title', 'text', overwrite=1)")
    # Config graph
    cat = json.loads((CFG / "retrofit_catalogue.json").read_text())
    rules = json.loads((CFG / "feasibility_rules.json").read_text())
    for it in cat["items"]:
        for dep in it.get("depends_on") or []:
            execute("INSERT INTO edges VALUES (?, 'REQUIRES_AFTER', ?, '{}')", [it["id"], dep])
        for ex in it.get("excludes") or []:
            execute("INSERT INTO edges VALUES (?, 'EXCLUDES', ?, '{}')", [it["id"], ex])
        for g in it.get("grants") or []:
            execute("INSERT INTO edges VALUES (?, 'GRANT_APPLIES', ?, '{}')", [g, it["id"]])
    for r in rules["rules"]:
        for rid in r.get("blocks", []):
            execute("INSERT INTO edges VALUES (?, 'BLOCKS', ?, ?)", [r["id"], rid, json.dumps({"reason": r["reason"]})])
        for rid, st in r.get("downgrade", {}).items():
            execute("INSERT INTO edges VALUES (?, 'CONDITIONS', ?, ?)", [r["id"], rid, json.dumps({"reason": r["reason"], "status": st})])
    # M04 priors (real model)
    try:
        from greentwin.models.energy_prior import load_default
        m = load_default()
        for sh, p in props.items():
            pr = m.energy_prior(p)
            execute("INSERT INTO energy_priors VALUES (?, ?, ?, ?)",
                    [sh, pr.provenance.model_version, pr.model_dump_json(), now()])
    except Exception as e:
        print("energy priors skipped:", e)
    # Demo customers (synthetic; memory is kept across rebuilds)
    for cid, label, pid in DEMO_CUSTOMERS:
        if not q("SELECT 1 FROM customers WHERE customer_id = ?", [cid]):
            execute("INSERT INTO customers VALUES (?, ?, ?, ?)", [cid, label, pid, now()])
    out = {t: q(f"SELECT count(*) n FROM {t}")[0]["n"] for t in
           ("properties", "usage_monthly", "configs", "documents", "edges", "energy_priors", "customers")}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    build()
