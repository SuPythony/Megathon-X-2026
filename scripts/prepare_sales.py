"""Run only the sales preparation step and print its log (sanity check before fitting).

    uv run python scripts/prepare_sales.py [--max-epc-age 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greentwin.value.prepare import PrepConfig, load_hpi, prepare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", default="data/interim/sales/tranall_link_26122024_s10.parquet")
    ap.add_argument("--max-epc-age", type=float, default=10.0)
    a = ap.parse_args()
    s = pd.read_parquet(a.sales)
    print("raw rows", len(s), "| years", s["dateoftransfer"].str[:4].value_counts().sort_index().to_dict())
    df, log = prepare(s, pd.read_parquet("data/interim/geo/NSPL21_NOV_2024_UK.parquet"),
                      load_hpi("data/raw/hpi/UK-HPI-full-file-2026-07.csv"), PrepConfig(max_epc_age_years=a.max_epc_age))
    print(json.dumps(log, indent=1, default=str))
    print("years kept", df["year"].value_counts().sort_index().to_dict())
    print(df[["price", "price_adj", "sap", "tfa", "epc_age_years"]].describe().round(1).to_string())
    print("ptype", df["ptype"].value_counts().to_dict(), "| regions", df["rgn"].nunique(), "| LSOAs", df["lsoa21"].nunique())


if __name__ == "__main__":
    main()
