"""Jev System-1 client (TypeSafe): structured judgements for routing, feasibility layer 2 and explanation checks.

Settings from .env: JEV_ENABLED (true/false), JEV_API_KEY, JEV_BASE_URL, JEV_MODEL, JEV_CONFIDENCE_TAU, JEV_CACHE_DIR.
Every response is cached under JEV_CACHE_DIR by hash(request), so runs replay without a key.
Disabled, no key or any error -> `judge` returns None and the caller falls back to its rules.

HACK: the request/response shape below (POST {JEV_BASE_URL}/judge with {model, task, instructions, input, schema};
JSON reply matching the schema, optionally wrapped in {"output": ...}) is our assumption. Confirm it against
TypeSafe's API docs once the key arrives; only `_post` should need to change.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from greentwin.llm.client import _load_env  # noqa: F401 - loads .env into os.environ on import

ROOT = Path(__file__).resolve().parents[2]


def _env(k: str, default: str = "") -> str:
    return os.environ.get(k, default).split("#")[0].strip()


BASE_URL = _env("JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone").rstrip("/")
MODEL = _env("JEV_MODEL", "jev-latest")
TAU = float(_env("JEV_CONFIDENCE_TAU", "0.85") or 0.85)
CACHE = ROOT / (_env("JEV_CACHE_DIR", "cache/jev") or "cache/jev")


def enabled() -> bool:
    key = _env("JEV_API_KEY")
    return _env("JEV_ENABLED").lower() == "true" and bool(key) and not key.startswith("__ASK_USER__")


def _post(task: str, instructions: str, payload: dict, schema: dict) -> dict:
    import httpx

    r = httpx.post(f"{BASE_URL}/judge", timeout=60,
                   headers={"Authorization": f"Bearer {_env('JEV_API_KEY')}", "Content-Type": "application/json"},
                   json={"model": MODEL, "task": task, "instructions": instructions, "input": payload,
                         "schema": schema})
    r.raise_for_status()
    body = r.json()
    return body.get("output", body) if isinstance(body, dict) else body


def judge(task: str, instructions: str, payload: dict, schema: dict) -> Optional[dict]:
    """Cached structured judgement, or None (disabled / no key / error)."""
    key = hashlib.sha256(json.dumps({"task": task, "m": MODEL, "i": instructions, "p": payload}, sort_keys=True,
                                    default=str).encode()).hexdigest()
    f = CACHE / task / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if not enabled():
        return None
    try:
        out = _post(task, instructions, payload, schema)
    except Exception:  # noqa: BLE001 - Jev is advisory; callers fall back to rules
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out), encoding="utf-8")
    return out
