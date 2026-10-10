"""Stream the UCL linked Price Paid + EPC dataset (ReShare 857911, CC BY 4.0) without downloading 7 GB.

The two bundles are zips served with HTTP Range support, so members are read remotely (same RemoteFile reader as
the EPC extractor) and only the needed columns of the linked table are written to Parquet.

    uv run python scripts/ucl_sales_extract.py --list
    uv run python scripts/ucl_sales_extract.py --peek <member>          # header + 5 rows
    uv run python scripts/ucl_sales_extract.py --member <member> --pct 10 --out data/interim/sales

Attribution: Chi, B., Dennett, A., Oleron-Evans, T. and Morphet, R. (UCL), Land Registry Price Paid Data linked to
Domestic EPCs, UK Data Service ReShare 857911, CC BY 4.0. Contains HM Land Registry data (c) Crown copyright and
database right, OGL v3.0.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from epc_bulk_extract import RemoteFile  # noqa: E402

ADDRESS_COLS = ["postcode", "paon", "saon", "street", "locality"]
SALE_COLS = [
    "postcode", "transactionid", "price", "dateoftransfer", "propertytype", "oldnew", "duration", "categorytype",
    "recordstatus", "classt", "lad23cd", "UPRN", "priceper",
    "CURRENT_ENERGY_RATING", "CURRENT_ENERGY_EFFICIENCY", "ENVIRONMENT_IMPACT_CURRENT", "LODGEMENT_DATE",
    "PROPERTY_TYPE", "BUILT_FORM", "CONSTRUCTION_AGE_BAND", "TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS", "TENURE",
    "MAINS_GAS_FLAG", "FLOOR_LEVEL", "EXTENSION_COUNT",
    # element descriptions, so the Green Home Score sub-scores can be computed at sale (green twins, D-xx)
    "WALLS_DESCRIPTION", "ROOF_DESCRIPTION", "FLOOR_DESCRIPTION", "WINDOWS_DESCRIPTION", "GLAZED_TYPE",
    "MULTI_GLAZE_PROPORTION", "MAINHEAT_DESCRIPTION", "MAIN_FUEL", "MAINHEAT_ENERGY_EFF", "MAINHEATCONT_DESCRIPTION",
    "LOW_ENERGY_LIGHTING", "PHOTO_SUPPLY", "SOLAR_WATER_HEATING_FLAG", "CO2_EMISS_CURR_PER_FLOOR_AREA",
]

BUNDLES = {
    "bundle_1": "https://reshare.ukdataservice.ac.uk/857911/3/857911_bundle_1.zip",
    "bundle_2": "https://reshare.ukdataservice.ac.uk/857911/2/857911_bundle_2.zip",
}


def open_zip(url: str) -> tuple[zipfile.ZipFile, RemoteFile]:
    rf = RemoteFile(lambda: url)
    return zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20)), rf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--peek")
    ap.add_argument("--member")
    ap.add_argument("--bundle", default=None, help="bundle_1 or bundle_2 (default: search both)")
    ap.add_argument("--pct", type=int, default=100, help="keep this %% of properties (hash of the property key)")
    ap.add_argument("--key", default="uprn", help="column used for the property-level sample")
    ap.add_argument("--usecols", nargs="*")
    ap.add_argument("--sales-cols", action="store_true", help="use the value model's SALE_COLS")
    ap.add_argument("--out", default="data/interim/sales")
    a = ap.parse_args()
    if a.sales_cols:
        a.usecols = SALE_COLS

    bundles = [a.bundle] if a.bundle else list(BUNDLES)
    for b in bundles:
        z, rf = open_zip(BUNDLES[b])
        names = {i.filename: i for i in z.infolist() if not i.is_dir()}
        if a.list:
            print(f"== {b}")
            for n, i in names.items():
                print(f"  {n:70s} {i.file_size / 1e9:7.2f} GB unz {i.compress_size / 1e9:6.2f} GB zip")
            continue
        target = a.peek or a.member
        # members of nested zips are addressed as "outer.zip::inner.csv"
        outer, _, inner = target.partition("::")
        if outer not in names:
            continue
        if outer.endswith(".zip"):
            zi = zipfile.ZipFile(z.open(names[outer]))
            if not inner:
                for i in zi.infolist():
                    print(f"  {outer}::{i.filename:50s} {i.file_size / 1e9:7.2f} GB unz {i.compress_size / 1e9:6.2f} GB zip")
                return
            src = zi.open(inner)
        else:
            src = z.open(names[outer])
        with src as fh:
            if a.peek:
                df = pd.read_csv(fh, nrows=5, low_memory=False)
                print(f"{target}: {df.shape[1]} columns")
                print(df.T.to_string(max_colwidth=60))
                return
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            t0, parts, seen = time.time(), [], 0
            usecols = a.usecols
            if a.key == "address" and usecols:
                usecols = list(dict.fromkeys(usecols + ADDRESS_COLS))
            for ch in pd.read_csv(fh, chunksize=250_000, usecols=usecols, low_memory=False, dtype=str):
                if a.key == "address":
                    # property key = hash of the full address; the address itself is never stored (privacy)
                    addr = ch[ADDRESS_COLS].fillna("").agg("|".join, axis=1).str.upper().str.strip()
                    ch["prop_key"] = addr.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:16])
                    ch = ch.drop(columns=[c for c in ("paon", "saon", "street", "locality") if c in ch])
                    k = ch["prop_key"]
                elif a.pct < 100:
                    k = ch[a.key].astype(str)
                if a.pct < 100:
                    keep = k.map(lambda s: int(hashlib.md5(f"s:{s}".encode()).hexdigest()[:8], 16) % 100 < a.pct)
                    ch = ch[keep]
                parts.append(ch)
                seen += 250_000
                if seen % 2_000_000 == 0:
                    print(f"  {seen:,} rows read, {sum(len(p) for p in parts):,} kept, "
                          f"{rf.bytes_fetched / 1e9:.2f} GB fetched, {time.time() - t0:.0f}s", flush=True)
            df = pd.concat(parts, ignore_index=True)
            stem = Path(inner or outer).stem  # never put "::" in a filename (invalid on Windows)
            dest = out / (stem + (f"_s{a.pct}" if a.pct < 100 else "") + ".parquet")
            df.to_parquet(dest, index=False)
            print(f"wrote {dest}: {len(df):,} rows, {df.shape[1]} cols, {time.time() - t0:.0f}s")
            return


if __name__ == "__main__":
    main()
