"""GreenTwin assistant: chat + knowledge base + memory + visual panels in one app.

    uv run python -m greentwin.kb.build                 # once (and after config changes)
    uv run uvicorn greentwin.api.app:app --port 8050     # then open http://127.0.0.1:8050
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import hmac
import os

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from contracts.models import ChatRequest, ChatTurn, EnergyUsageMonth, ScenarioSpec
from greentwin.api.energy_prior import router as m04_router
from greentwin.kb.store import q
from greentwin.llm import client as llm
from greentwin.llm.chat import chat
from greentwin.llm.tools import _prop, get_home_facts, search_docs
from greentwin.memory import service as mem

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="GreenTwin assistant")
app.include_router(m04_router)


def _include_twin_routes() -> None:
    """The 3D viewer's live re-simulation endpoints (/api/twin/*, scripts/twin_server.py) on this same origin."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("twin_server", Path(__file__).resolve().parents[2] / "scripts" / "twin_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["twin_server"] = mod  # pydantic resolves the module's postponed annotations through sys.modules
    spec.loader.exec_module(mod)
    app.router.routes.extend(r for r in mod.app.router.routes if getattr(r, "path", "").startswith("/api/twin"))


_include_twin_routes()
if (STATIC / "twin").exists():  # built by `cd web/twin-viewer && npm run build`
    app.mount("/twin", StaticFiles(directory=STATIC / "twin", html=True), name="twin")
API_TOKEN = os.environ.get("GREENTWIN_API_TOKEN")  # optional; demo runs on localhost without it


@app.middleware("http")
async def require_token(request: Request, call_next):
    """If GREENTWIN_API_TOKEN is set, every /api call must send it as X-API-Key."""
    if API_TOKEN and request.url.path.startswith("/api/"):
        if not hmac.compare_digest(request.headers.get("x-api-key", ""), API_TOKEN):
            return JSONResponse({"detail": "missing or wrong X-API-Key"}, status_code=401)
    return await call_next(request)


def _customer(cid: Optional[str]) -> Optional[str]:
    """Only known (synthetic) customers; unknown ids are rejected rather than silently created."""
    if not cid:
        return None
    if not q("SELECT 1 FROM customers WHERE customer_id = ?", [cid]):
        raise HTTPException(404, "unknown customer")
    return cid


def _home(pid: str):
    try:
        return _prop(pid)
    except KeyError as e:
        raise HTTPException(404, "unknown property") from e


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    counts = {t: q(f"SELECT count(*) n FROM {t}")[0]["n"] for t in ("properties", "documents", "edges", "customers")}
    return {"llm": {"provider": "openai", "model": llm.MODEL, "key": llm.has_key()}, "kb": counts}


@app.get("/api/homes")
def homes():
    return {"homes": q("SELECT property_id, label FROM properties ORDER BY property_id"),
            "customers": q("SELECT customer_id, label, property_id FROM customers ORDER BY customer_id")}


@app.get("/api/homes/{pid}")
def home(pid: str):
    from greentwin.models.energy_prior import load_default
    from greentwin.models.prior_update import update_with_meters
    p = _home(pid)
    usage_rows = q("SELECT * FROM usage_monthly WHERE property_id = ? ORDER BY month", [pid])
    usage = [EnergyUsageMonth(**{k: v for k, v in r.items()}) for r in usage_rows]
    model = load_default()
    prior = model.energy_prior(p)
    upd = update_with_meters(model, p, usage, n=8000)
    return {"facts": get_home_facts(pid), "prior": prior.model_dump(mode="json"), "update": upd,
            "usage": [{"month": r["month"], "electricity_kwh": r["electricity_import_kwh"], "gas_kwh": r["gas_kwh"]}
                      for r in usage_rows]}


@app.get("/api/recommendations")
def recommendations(property_id: str, customer_id: Optional[str] = None):
    return mem.rank(_home(property_id), _customer(customer_id))


def _known(pid: str) -> None:
    """Showcase homes from contracts/examples, others from the knowledge base."""
    from greentwin.api.orchestrator import load_property
    try:
        load_property(pid)
    except Exception as e:  # noqa: BLE001 - unknown id or no store
        raise HTTPException(404, "unknown property") from e


@app.post("/api/scenario")
def scenario(spec: ScenarioSpec):
    """run_scenario: calibrated twin -> package (one run) -> ladder, potential, rating, score, value (ScenarioResult)."""
    from greentwin.api.orchestrator import default
    _known(spec.property_id)
    try:
        return default().run_scenario(spec).model_dump(mode="json")
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(422, str(e)) from e


class SummaryIn(BaseModel):
    property_id: str
    retrofit_ids: list[str] = Field(default_factory=list)


@app.post("/api/scenario/summary")
def scenario_summary(body: SummaryIn):
    """Calibrated whole-house numbers for the 3D viewer: per fuel, full bill, carbon, net cost, score, rating, value."""
    from greentwin.api.orchestrator import default, house_summary
    _known(body.property_id)
    try:
        return house_summary(default(), body.property_id, body.retrofit_ids)
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/plan/{pid}")
def plan(pid: str, goal: str = "bills", budget_gbp: Optional[float] = None, annual_budget_gbp: float = 5000,
         reinvest: bool = True, customer_id: Optional[str] = None, min_carbon_kg: Optional[float] = None,
         max_cost_gbp: Optional[float] = None):
    """Ladder + staging by year + Pareto (knee, P(on front), budget, reference point, memory re-rank, baselines)."""
    from greentwin.api.orchestrator import default
    if goal not in ("bills", "carbon"):
        raise HTTPException(422, "goal must be bills or carbon")
    _known(pid)
    return default().plan(pid, goal, budget_gbp, annual_budget_gbp, reinvest, _customer(customer_id),
                          min_carbon_kg, max_cost_gbp)


@app.get("/api/memory/{cid}")
def get_memory(cid: str):
    m = mem.memory(_customer(cid))
    for k in ("confirmed", "proposed", "events"):
        for it in m[k]:
            for f in ("created_at", "updated_at"):
                if f in it and it[f] is not None:
                    it[f] = it[f].strftime("%d %b %Y %H:%M")
    return m


class ItemIn(BaseModel):
    kind: str
    key: str
    value: str
    evidence: str = ""


@app.post("/api/memory/{cid}/items")
def add_item(cid: str, body: ItemIn):
    return mem.add_item(_customer(cid), body.kind, body.key, body.value, source="user", evidence=body.evidence)


class StatusIn(BaseModel):
    status: str


@app.post("/api/memory/{cid}/items/{item_id}/status")
def set_status(cid: str, item_id: str, body: StatusIn):
    if not q("SELECT 1 FROM memory_items WHERE item_id = ? AND customer_id = ?", [item_id, _customer(cid)]):
        raise HTTPException(404, "unknown memory item for this customer")
    return mem.set_status(item_id, body.status)


@app.post("/api/memory/{cid}/reset")
def reset_memory(cid: str):
    """Demo helper: retract every item (kept for audit) and clear this customer's actions."""
    from greentwin.kb.store import execute
    cid = _customer(cid)
    execute("UPDATE memory_items SET status = 'retracted' WHERE customer_id = ? AND status <> 'retracted'", [cid])
    execute("DELETE FROM memory_events WHERE customer_id = ?", [cid])
    return {"ok": True}


class EventIn(BaseModel):
    property_id: str
    retrofit_id: str
    type: str
    reason: str = ""


@app.post("/api/memory/{cid}/events")
def add_event(cid: str, body: EventIn):
    _home(body.property_id)
    return mem.add_event(_customer(cid), body.property_id, body.retrofit_id, body.type, body.reason)


class ChatIn(BaseModel):
    property_id: str
    customer_id: Optional[str] = None
    message: str
    history: list[ChatTurn] = Field(default_factory=list)
    session_id: Optional[str] = None


@app.post("/api/chat")
def chat_endpoint(body: ChatIn):
    _home(body.property_id)
    req = ChatRequest(property_id=body.property_id, customer_id=_customer(body.customer_id), message=body.message[:4000],
                      history=body.history[-8:])
    return chat(req, body.session_id)


class CaptureIn(BaseModel):
    images_b64: list[str] = Field(default_factory=list, max_length=4)
    consent: bool = False


@app.post("/api/capture/{pid}")
def capture_photos(pid: str, body: CaptureIn):
    from greentwin.twin.capture import capture
    _home(pid)
    return capture(pid, body.images_b64, body.consent)


class ApplyIn(BaseModel):
    capture_id: str
    accepted: list[int] = Field(default_factory=list)


@app.post("/api/capture/{pid}/apply")
def capture_apply(pid: str, body: ApplyIn):
    from greentwin.twin.capture import apply
    _home(pid)
    return apply(pid, body.capture_id, body.accepted)


@app.get("/api/kb/search")
def kb_search(query: str = Query("", alias="q"), kind: Optional[str] = None):
    return search_docs(query, kind or None)


@app.get("/api/kb/doc/{doc_id:path}")
def kb_doc(doc_id: str):
    rows = q("SELECT doc_id, kind, title, text, source_url, verified FROM documents WHERE doc_id = ?", [doc_id])
    if not rows:
        raise HTTPException(404, "unknown doc")
    return rows[0]
