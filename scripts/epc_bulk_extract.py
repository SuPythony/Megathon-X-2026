"""Stream the EPC domestic bulk zip over HTTP Range requests and keep a deterministic sample.

The full zip is ~8.3 GB (yearly CSVs). Instead of downloading it, we open it as a seekable remote file
(the pre-signed S3 URL supports Range), stream each CSV member in chunks, keep only the columns the
score/value work needs, and keep a deterministic sample of properties: hash(UPRN or address key) mod 100 < pct.
Sampling by property keeps every certificate of a sampled home together (needed for history + splits).

Usage:
    uv run python scripts/epc_bulk_extract.py --env-file ../.env --list
    uv run python scripts/epc_bulk_extract.py --env-file ../.env --pct 10 --out data/interim/epc
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

BASE = "https://api.get-energy-performance-data.communities.gov.uk"

KEEP = [
    "CERTIFICATE_NUMBER", "UPRN", "POSTCODE", "LOCAL_AUTHORITY", "CONSTITUENCY", "LODGEMENT_DATE",
    "INSPECTION_DATE", "TRANSACTION_TYPE", "REPORT_TYPE", "TENURE",
    "PROPERTY_TYPE", "BUILT_FORM", "CONSTRUCTION_AGE_BAND", "TOTAL_FLOOR_AREA", "FLOOR_HEIGHT",
    "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS", "FLOOR_LEVEL", "FLAT_TOP_STOREY", "EXTENSION_COUNT",
    "CURRENT_ENERGY_RATING", "CURRENT_ENERGY_EFFICIENCY", "POTENTIAL_ENERGY_RATING", "POTENTIAL_ENERGY_EFFICIENCY",
    "ENVIRONMENT_IMPACT_CURRENT", "ENERGY_CONSUMPTION_CURRENT", "CO2_EMISSIONS_CURRENT",
    "CO2_EMISS_CURR_PER_FLOOR_AREA", "LIGHTING_COST_CURRENT", "HEATING_COST_CURRENT", "HOT_WATER_COST_CURRENT",
    "WALLS_DESCRIPTION", "WALLS_ENERGY_EFF", "WALLS_ENV_EFF",
    "ROOF_DESCRIPTION", "ROOF_ENERGY_EFF", "ROOF_ENV_EFF",
    "FLOOR_DESCRIPTION", "FLOOR_ENERGY_EFF", "FLOOR_ENV_EFF",
    "WINDOWS_DESCRIPTION", "WINDOWS_ENERGY_EFF", "WINDOWS_ENV_EFF",
    "MULTI_GLAZE_PROPORTION", "GLAZED_TYPE", "GLAZED_AREA",
    "MAINHEAT_DESCRIPTION", "MAINHEAT_ENERGY_EFF", "MAINHEAT_ENV_EFF", "MAIN_FUEL", "MAINS_GAS_FLAG",
    "MAINHEATCONT_DESCRIPTION", "MAINHEATC_ENERGY_EFF", "MAINHEATC_ENV_EFF", "MAIN_HEATING_CONTROLS",
    "HOTWATER_DESCRIPTION", "HOT_WATER_ENERGY_EFF", "HOT_WATER_ENV_EFF",
    "SECONDHEAT_DESCRIPTION", "SHEATING_ENERGY_EFF",
    "LIGHTING_DESCRIPTION", "LIGHTING_ENERGY_EFF", "LOW_ENERGY_LIGHTING",
    "LOW_ENERGY_FIXED_LIGHT_COUNT", "FIXED_LIGHTING_OUTLETS_COUNT",
    "NUMBER_OPEN_FIREPLACES", "MECHANICAL_VENTILATION", "PHOTO_SUPPLY", "SOLAR_WATER_HEATING_FLAG",
    "WIND_TURBINE_COUNT", "ENERGY_TARIFF",
]


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range; refreshes the pre-signed URL when it expires."""

    def __init__(self, get_url, block: int = 8 << 20):
        self._get_url = get_url
        self.url = get_url()
        h = httpx.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60)
        h.raise_for_status()
        self.size = int(h.headers["content-range"].split("/")[1])
        self.pos = 0
        self.block = block
        self._cache_start, self._cache = -1, b""
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(6):
            try:
                r = httpx.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (400, 403):  # expired signature
                    self.url = self._get_url()
                    continue
                r.raise_for_status()
                self.bytes_fetched += len(r.content)
                return r.content
            except httpx.HTTPError:
                time.sleep(2 ** attempt)
        raise RuntimeError("range fetch failed")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs, c = self._cache_start, self._cache
        if cs <= self.pos and self.pos + n <= cs + len(c):
            out = c[self.pos - cs: self.pos - cs + n]
        else:
            want = max(n, self.block)
            end = min(self.size, self.pos + want) - 1
            data = self._fetch(self.pos, end)
            self._cache_start, self._cache = self.pos, data
            out = data[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def signed_url_factory(tok: str):
    def get():
        r = httpx.get(f"{BASE}/api/files/domestic/csv",
                      headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"},
                      timeout=60, follow_redirects=False)
        if r.status_code != 302:
            sys.exit(f"bulk endpoint returned {r.status_code}")
        return r.headers["location"]
    return get


def keep_mask(keys: pd.Series, pct: int) -> pd.Series:
    def h(k: str) -> int:
        return int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 100
    return keys.map(h) < pct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pct", type=int, default=10)
    ap.add_argument("--out", default="data/interim/epc")
    ap.add_argument("--only", nargs="*", help="member names to process (default: all certificate CSVs)")
    ap.add_argument("--zip", help="local copy of domestic-csv.zip (skips the remote Range reads)")
    a = ap.parse_args()
    if a.zip:
        rf = None
        z = zipfile.ZipFile(a.zip)
    else:
        load_dotenv(a.env_file) if a.env_file else load_dotenv()
        tok = os.environ.get("EPC_API_TOKEN", "")
        if not tok or tok.startswith("__ASK_USER__"):
            sys.exit("EPC_API_TOKEN not set")
        rf = RemoteFile(signed_url_factory(tok))
        z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    members = [i for i in z.infolist() if not i.is_dir()]
    if a.list:
        for i in members:
            print(f"{i.filename:60s} {i.file_size / 1e9:7.2f} GB unz  {i.compress_size / 1e9:6.2f} GB zip")
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = [i for i in members if i.filename.lower().endswith(".csv")
               and "recommendation" not in i.filename.lower()]
    if a.only:
        targets = [i for i in targets if i.filename in a.only]
    for info in targets:
        dest = out / (Path(info.filename).stem + f"_s{a.pct}.parquet")
        if dest.exists():
            print("skip", dest)
            continue
        t0, kept, seen, parts = time.time(), 0, 0, []
        with z.open(info) as fh:
            reader = pd.read_csv(fh, chunksize=200_000, dtype=str, low_memory=False)
            for chunk in reader:
                chunk.columns = [c.upper() for c in chunk.columns]
                cols = [c for c in KEEP if c in chunk.columns]
                key = chunk["UPRN"].fillna("") if "UPRN" in chunk else pd.Series("", index=chunk.index)
                key = key.where(key != "", chunk.get("POSTCODE", "").fillna("") + "|" + chunk.get("ADDRESS1", chunk.get("ADDRESS", "")).fillna(""))
                m = keep_mask(key, a.pct)
                parts.append(chunk.loc[m, cols])
                seen += len(chunk)
                kept += int(m.sum())
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEEP)
        df.to_parquet(dest, index=False)
        fetched = f"; fetched {rf.bytes_fetched / 1e9:.2f} GB so far" if rf else ""
        print(f"{info.filename}: kept {kept:,}/{seen:,} rows in {time.time() - t0:.0f}s{fetched} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
