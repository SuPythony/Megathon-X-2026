"""Add the green score at sale to the prepared sales and precompute green twins for the showcase homes (V-10).

    uv run python scripts/build_green_twins.py --version v1

Writes data/processed/sales_scored_<version>.parquet (git-ignored) and
artifacts/value_model/<version>/green_twins_showcase.json (ranges only, no addresses).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from contracts.models import Property  # noqa: E402
from greentwin.value.green_twins import find, home_score, sale_scores  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    s = pd.read_parquet(ROOT / "data" / "processed" / f"sales_prepared_{a.version}.parquet")
    s["green_score"] = sale_scores(s)
    print(f"green score at sale: {s['green_score'].notna().mean():.1%} of {len(s):,} sales scored; "
          f"quantiles {s['green_score'].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(1).tolist()}")
    s.to_parquet(ROOT / "data" / "processed" / f"sales_scored_{a.version}.parquet", index=False)
    out = {}
    for pid in ("SH1", "SH2", "SH3"):
        p = Property.model_validate_json((ROOT / "contracts" / "examples" / f"{pid}_property.json").read_text("utf-8"))
        twins = find(p, s.dropna(subset=["green_score"]))
        out[pid] = {"home_score_no_intensity": round(float(home_score(p)), 1),
                    "twins": [t.model_dump() for t in twins]}
        print(pid, "own", out[pid]["home_score_no_intensity"], "->", [
            (t.score_gap, t.epc_band, round(t.price_per_m2_gbp.median), t.same[-2:]) for t in twins])
    (ROOT / "artifacts" / "value_model" / a.version / "green_twins_showcase.json").write_text(
        json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
