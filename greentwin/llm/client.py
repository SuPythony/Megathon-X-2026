"""OpenAI client + settings. Reads .env; tolerates a bare `sk-...` line (treated as OPENAI_API_KEY)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    for env in (ROOT / ".env", ROOT.parents[1] / ".env"):  # worktree, then the main checkout
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))
            elif line.startswith("sk-"):
                os.environ.setdefault("OPENAI_API_KEY", line)
        break


_load_env()
MODEL = os.environ.get("LLM_MODEL", "gpt-6-luna")
if not MODEL.startswith("gpt"):  # .env.example still carries the old Anthropic default
    MODEL = "gpt-6-luna"


def has_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def client():
    from openai import OpenAI
    return OpenAI()
