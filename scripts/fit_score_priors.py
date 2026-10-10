"""Fit the score's priors for missing inputs from real data -> greentwin/score/priors.json.

    uv run python scripts/fit_score_priors.py --epc-dir data/interim/epc25 --need data/raw/need/anon2025_4million.zip

* national_band_mix: EPC band shares, latest certificate per property, lodged in the last 10 years (valid EPCs).
* loft_mm_by_age: recorded loft depths of pitched-loft roofs by RdSAP age band ("insulated (assumed)" excluded).
* intensity_peer_p50: NEED median metered kWh/m2 by property type x floor-area band (peer for key issues).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greentwin.score import need_features  # noqa: E402
from greentwin.score.mapping import age_band_start  # noqa: E402

AGE_KEYS = [(1800, 1899), (1900, 1929), (1930, 1949), (1950, 1966), (1967, 1975), (1976, 1982), (1983, 1990),
            (1991, 1995), (1996, 2002), (2003, 2006), (2007, 2011), (2012, 2100)]
DEPTHS = [0, 12, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 270, 300, 350, 400]
COLS = ["UPRN", "CERTIFICATE_NUMBER", "LODGEMENT_DATE", "CURRENT_ENERGY_RATING", "ROOF_DESCRIPTION",
        "CONSTRUCTION_AGE_BAND"]


def age_key(band) -> str | None:
    if not isinstance(band, str):
        return None
    s = age_band_start(band.replace("England and Wales: ", ""))
    if s is None:
        return None
    for a, b in AGE_KEYS:
        if a <= s <= b:
            return f"{a}-{b}"
    return None


def loft_mm(desc) -> float:
    if not isinstance(desc, str) or not desc.lower().startswith("pitched"):
        return np.nan
    d = desc.lower()
    g = re.search(r"(\d+)\+?\s*mm", d)
    if g:
        v = int(g.group(1))
        return float(min(DEPTHS, key=lambda x: abs(x - v)))
    if "no insulation" in d:
        return 0.0
    return np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epc-dir", default="data/interim/epc25")
    ap.add_argument("--need", default="data/raw/need/anon2025_4million.zip")
    ap.add_argument("--asof", default="2026-10-11")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.epc_dir}/certificates-*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    d["date"] = pd.to_datetime(d["LODGEMENT_DATE"], errors="coerce")
    d["key"] = d["UPRN"].fillna("cert:" + d["CERTIFICATE_NUMBER"].astype(str))
    latest = d.sort_values("date").groupby("key").tail(1)
    valid = latest[latest["date"] >= pd.Timestamp(a.asof) - pd.DateOffset(years=10)]
    bands = valid["CURRENT_ENERGY_RATING"].where(valid["CURRENT_ENERGY_RATING"].isin(list("ABCDEFG")))
    mix = bands.value_counts(normalize=True).reindex(list("ABCDEFG"), fill_value=0.0)

    latest = latest.assign(mm=latest["ROOF_DESCRIPTION"].map(loft_mm),
                           age=latest["CONSTRUCTION_AGE_BAND"].map(age_key)).dropna(subset=["mm"])
    loft = {"all": {"source": f"EPC register 25% property sample, latest certificate, n={len(latest):,}",
                    "probs": {str(int(k)): round(float(v), 4) for k, v in latest["mm"].value_counts(normalize=True).sort_index().items()}}}
    for k, g in latest.dropna(subset=["age"]).groupby("age"):
        loft[k] = {"source": f"EPC register 25% sample, age band {k}, n={len(g):,}",
                   "probs": {str(int(m)): round(float(v), 4) for m, v in g["mm"].value_counts(normalize=True).sort_index().items()}}

    nd = need_features.load(a.need).dropna(subset=["intensity_kwh_m2"])
    peers = nd.groupby(["property_type", "area_band"])["intensity_kwh_m2"].agg(["median", "size"])
    peer = {f"{t}|{int(b)}": {"p50": round(float(r["median"]), 1), "n": int(r["size"])}
            for (t, b), r in peers.iterrows() if r["size"] >= 100}

    pri = json.loads((ROOT / "greentwin" / "score" / "priors.json").read_text(encoding="utf-8"))
    pri["_doc"] = ("Priors for missing score inputs, fitted by scripts/fit_score_priors.py from the EPC register "
                   "(25% property sample) and NEED 2025 (4M sample). Each entry records its source.")
    pri["national_band_mix"] = {
        "source": f"EPC register 25% property sample, latest certificate lodged since "
                  f"{(pd.Timestamp(a.asof) - pd.DateOffset(years=10)).date()}, n={bands.notna().sum():,}",
        "probs": {b: round(float(v), 4) for b, v in mix.items()}}
    pri["loft_mm_by_age"] = loft
    pri["intensity_peer_p50"] = {"source": "NEED 2025 4M sample, valid 2023 meter reads; median (gas+elec)/floor-area "
                                           "band midpoint; key = NEED property type|floor-area band", "cells": peer}
    (ROOT / "greentwin" / "score" / "priors.json").write_text(json.dumps(pri, indent=2), encoding="utf-8")
    print("band mix:", pri["national_band_mix"])
    print("loft all:", loft["all"])
    print("peer cells:", len(peer))


if __name__ == "__main__":
    main()
