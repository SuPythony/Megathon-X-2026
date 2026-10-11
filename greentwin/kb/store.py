"""DuckDB knowledge base: one file, one process-wide connection (data/greentwin.duckdb, git-ignored)."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "greentwin.duckdb"
_LOCK = threading.RLock()
_CON = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS properties (property_id VARCHAR PRIMARY KEY, label VARCHAR, data JSON, source VARCHAR);
CREATE TABLE IF NOT EXISTS usage_monthly (property_id VARCHAR, month VARCHAR, electricity_import_kwh DOUBLE,
    gas_kwh DOUBLE, estimated_read BOOLEAN);
CREATE TABLE IF NOT EXISTS energy_priors (property_id VARCHAR PRIMARY KEY, model_version VARCHAR, prior JSON,
    created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS configs (config_id VARCHAR PRIMARY KEY, version VARCHAR, content_hash VARCHAR, data JSON);
CREATE TABLE IF NOT EXISTS documents (doc_id VARCHAR PRIMARY KEY, kind VARCHAR, title VARCHAR, text VARCHAR,
    applies_to JSON, source_url VARCHAR, verified BOOLEAN, content_hash VARCHAR);
CREATE TABLE IF NOT EXISTS edges (src VARCHAR, rel VARCHAR, dst VARCHAR, props JSON);
CREATE TABLE IF NOT EXISTS customers (customer_id VARCHAR PRIMARY KEY, label VARCHAR, property_id VARCHAR,
    created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS memory_items (item_id VARCHAR PRIMARY KEY, customer_id VARCHAR, kind VARCHAR,
    key VARCHAR, value VARCHAR, status VARCHAR, source VARCHAR, evidence VARCHAR, created_at TIMESTAMP,
    updated_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS memory_events (event_id VARCHAR PRIMARY KEY, customer_id VARCHAR, property_id VARCHAR,
    retrofit_id VARCHAR, type VARCHAR, reason VARCHAR, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS chat_turns (turn_id VARCHAR PRIMARY KEY, session_id VARCHAR, customer_id VARCHAR,
    property_id VARCHAR, role VARCHAR, text VARCHAR, meta JSON, created_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS tool_calls (call_id VARCHAR, turn_id VARCHAR, name VARCHAR, input JSON, output JSON,
    ms DOUBLE, created_at TIMESTAMP);
"""


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def con() -> duckdb.DuckDBPyConnection:
    global _CON
    with _LOCK:
        if _CON is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CON = duckdb.connect(str(DB_PATH))
            _CON.execute(SCHEMA)
            _CON.execute("INSTALL fts; LOAD fts;")
        return _CON


def q(sql: str, params: list | tuple = ()) -> list[dict]:
    """Run a query, return rows as dicts (JSON columns parsed)."""
    with _LOCK:
        cur = con().execute(sql, params)
        if cur.description is None:
            return []
        cols = [d[0] for d in cur.description]
        rows = []
        for r in cur.fetchall():
            row = {}
            for c, v in zip(cols, r):
                if isinstance(v, str) and v[:1] in "[{" and c in ("data", "prior", "applies_to", "props", "meta",
                                                                   "input", "output"):
                    try:
                        v = json.loads(v)
                    except ValueError:
                        pass
                row[c] = v
            rows.append(row)
        return rows


def execute(sql: str, params: list | tuple = ()) -> None:
    with _LOCK:
        con().execute(sql, params)
