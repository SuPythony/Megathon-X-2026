"""Download EPC domestic certificates via the Get Energy Performance Data API.

Usage:
    uv run python scripts/download_epc.py info   [--env-file PATH]
    uv run python scripts/download_epc.py bulk   [--env-file PATH] [--out data/raw/epc]
    uv run python scripts/download_epc.py search --council Manchester --date-start 2024-01-01 --date-end 2024-12-31

The bearer token is read from EPC_API_TOKEN (environment or .env file). It is never printed or logged.
API docs: https://get-energy-performance-data.communities.gov.uk/api-technical-documentation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE = "https://api.get-energy-performance-data.communities.gov.uk"


def _token(env_file: str | None) -> str:
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)
    tok = os.environ.get("EPC_API_TOKEN", "")
    if not tok or tok.startswith("__ASK_USER__"):
        sys.exit("EPC_API_TOKEN is not set (ask the team lead).")
    return tok


def _headers(tok: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {tok}", "Accept": "application/json"}


def info(tok: str) -> None:
    r = httpx.get(f"{BASE}/api/files/domestic/csv/info", headers=_headers(tok), timeout=60)
    print(r.status_code, r.text[:2000])


def bulk(tok: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = httpx.get(f"{BASE}/api/files/domestic/csv", headers=_headers(tok), timeout=60, follow_redirects=False)
    if r.status_code != 302:
        sys.exit(f"expected 302 redirect, got {r.status_code}: {r.text[:500]}")
    url = r.headers["location"]  # pre-signed S3 URL; not logged
    dest = out / "domestic-csv.zip"
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", url, timeout=None) as s:
        s.raise_for_status()
        total = int(s.headers.get("content-length", 0))
        done, last = 0, -1
        with open(tmp, "wb") as f:
            for chunk in s.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                pct = int(100 * done / total) if total else -1
                if pct != last and pct % 5 == 0:
                    print(f"{done / 1e9:.2f} / {total / 1e9:.2f} GB ({pct}%)", flush=True)
                    last = pct
    tmp.replace(dest)
    print("saved", dest)


def search(tok: str, council: str, date_start: str, date_end: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows, page = [], 1
    while True:
        params = {"council[]": council, "date_start": date_start, "date_end": date_end,
                  "current_page": page, "page_size": 5000}
        r = httpx.get(f"{BASE}/api/domestic/search", headers=_headers(tok), params=params, timeout=120)
        if r.status_code == 404:
            break
        r.raise_for_status()
        j = r.json()
        rows += j["data"]
        if not j["pagination"]["nextPage"]:
            break
        page += 1
    dest = out / f"search_{council.replace(' ', '_')}_{date_start}_{date_end}.json"
    dest.write_text(json.dumps(rows))
    print(len(rows), "rows ->", dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["info", "bulk", "search"])
    ap.add_argument("--env-file")
    ap.add_argument("--out", default="data/raw/epc")
    ap.add_argument("--council")
    ap.add_argument("--date-start")
    ap.add_argument("--date-end")
    a = ap.parse_args()
    tok = _token(a.env_file)
    if a.cmd == "info":
        info(tok)
    elif a.cmd == "bulk":
        bulk(tok, Path(a.out))
    else:
        search(tok, a.council, a.date_start, a.date_end, Path(a.out))


if __name__ == "__main__":
    main()
