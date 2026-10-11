"""Sample one year of EPC certificates AND their assessor recommendations, keyed by certificate number.

Labels for the per-measure relevance model (plan 4c.3): P(an assessor would recommend X | home features).
Streams the bulk zip over HTTP Range (see epc_bulk_extract.py) and keeps hash(CERTIFICATE_NUMBER) mod 1000 < permille
from both files, so every kept certificate has all of its recommendations.

    uv run python scripts/epc_recs_extract.py --head                      # print the recommendations header
    uv run python scripts/epc_recs_extract.py --year 2025 --permille 20   # ~2% of 2025 -> data/interim/epc_recs
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

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epc_bulk_extract import KEEP, RemoteFile, signed_url_factory  # noqa: E402


def _mask(keys: pd.Series, permille: int) -> pd.Series:
    return keys.fillna("").map(lambda k: int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 1000) < permille


def _key_col(cols) -> str:
    for c in ("CERTIFICATE_NUMBER", "LMK_KEY"):
        if c in cols:
            return c
    raise KeyError(f"no certificate key in {list(cols)[:10]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2025")
    ap.add_argument("--permille", type=int, default=20)
    ap.add_argument("--out", default="data/interim/epc_recs")
    ap.add_argument("--head", action="store_true")
    a = ap.parse_args()
    load_dotenv()
    tok = os.environ.get("EPC_API_TOKEN", "")
    if not tok or tok.startswith("__ASK_USER__"):
        sys.exit("EPC_API_TOKEN not set")
    rf = RemoteFile(signed_url_factory(tok))
    z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    if a.head:
        with z.open(f"recommendations-{a.year}.csv") as fh:
            print(pd.read_csv(fh, nrows=5, dtype=str).to_string())
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("certificates", "recommendations"):
        dest = out / f"{kind}-{a.year}_m{a.permille}.parquet"
        if dest.exists():
            print("skip", dest)
            continue
        t0, parts, seen = time.time(), [], 0
        with z.open(f"{kind}-{a.year}.csv") as fh:
            for chunk in pd.read_csv(fh, chunksize=200_000, dtype=str, low_memory=False):
                chunk.columns = [c.upper() for c in chunk.columns]
                k = _key_col(chunk.columns)
                keep = chunk.loc[_mask(chunk[k], a.permille)]
                if kind == "certificates":
                    keep = keep[[c for c in [k] + KEEP if c in keep.columns]].loc[:, lambda d: ~d.columns.duplicated()]
                parts.append(keep)
                seen += len(chunk)
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(dest, index=False)
        print(f"{kind}-{a.year}: kept {len(df):,}/{seen:,} rows in {time.time() - t0:.0f}s; "
              f"fetched {rf.bytes_fetched / 1e9:.2f} GB -> {dest}", flush=True)


if __name__ == "__main__":
    main()
