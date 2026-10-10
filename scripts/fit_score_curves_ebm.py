"""M5: learn monotone shape functions with an EBM and compare them with the documented sub-score curves.

    uv run python scripts/fit_score_curves_ebm.py --epc-dir data/interim/epc25 --n 300000

EBM (InterpretML) on raw EPC parameters -> SAP. Each term's shape function is the data's own "sub-score curve".
We report (a) Spearman between each documented sub-score and the EBM term score per home (does our mapping rank
states the way the data does?) and (b) the EBM term importances. Output: artifacts/score_curves/v1/.
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
from greentwin.score import epc_features as E  # noqa: E402
from greentwin.score.learn import split_of  # noqa: E402

RAW = ["loft_mm", "s_walls", "s_floor", "s_windows_glazing", "heating_system", "s_smart_energy_management",
       "pv_kwp_est", "low_energy_lighting_pct"]


def loft_mm(desc) -> float:
    if not isinstance(desc, str) or not desc.lower().startswith("pitched"):
        return np.nan
    g = re.search(r"(\d+)\+?\s*mm", desc.lower())
    return float(g.group(1)) if g else (0.0 if "no insulation" in desc.lower() else np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epc-dir", default="data/interim/epc25")
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from interpret.glassbox import ExplainableBoostingRegressor

    files = sorted(glob.glob(f"{a.epc_dir}/certificates-*.parquet"), reverse=True)[:4]  # recent years
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    f = E.latest_per_property(E.build(raw))
    f["loft_mm"] = raw.loc[f.index, "ROOF_DESCRIPTION"].map(loft_mm).values
    f = f.dropna(subset=["sap", "s_walls", "s_windows_glazing", "heating_system", "s_smart_energy_management"])
    f = f.sample(min(a.n, len(f)), random_state=a.seed)
    f["split"] = split_of(f["uprn"].fillna(f["certificate"]))
    tr, te = f[f.split == "train"], f[f.split == "test"]
    X = lambda d: d[RAW].assign(heating_system=d["heating_system"].astype(str))  # noqa: E731
    mono = {"loft_mm": 1, "s_walls": 1, "s_floor": 1, "s_windows_glazing": 1, "s_smart_energy_management": 1,
            "pv_kwp_est": 1, "low_energy_lighting_pct": 1}
    ebm = ExplainableBoostingRegressor(random_state=a.seed, interactions=0,
                                       monotone_constraints=[mono.get(c, 0) for c in RAW])
    ebm.fit(X(tr), tr["sap"])
    pred = ebm.predict(X(te))
    r2 = 1 - np.sum((te["sap"] - pred) ** 2) / np.sum((te["sap"] - te["sap"].mean()) ** 2)

    terms = ebm.eval_terms(X(te))  # per-home additive contribution of each term
    contrib = pd.DataFrame(terms, columns=ebm.term_names_, index=te.index)
    imp = dict(zip(ebm.term_names_, np.round(ebm.term_importances(), 3)))
    ours = {"loft_mm": "s_roof", "s_walls": "s_walls", "s_floor": "s_floor", "s_windows_glazing": "s_windows_glazing",
            "heating_system": "s_heating_system", "s_smart_energy_management": "s_smart_energy_management",
            "pv_kwp_est": "s_renewable_energy"}
    agree = {}
    for term, col in ours.items():
        d = pd.DataFrame({"ebm": contrib[term], "ours": te[col]}).dropna()
        agree[term] = round(float(d["ebm"].corr(d["ours"], method="spearman")), 3) if len(d) > 100 else None
    heat = (pd.DataFrame({"h": te["heating_system"], "ebm": contrib["heating_system"], "ours": te["s_heating_system"]})
            .groupby("h").agg(ebm_points=("ebm", "mean"), our_sub_score=("ours", "mean"), n=("ebm", "size"))
            .sort_values("ebm_points").round(2))
    out = ROOT / "artifacts" / "score_curves" / "v1"
    out.mkdir(parents=True, exist_ok=True)
    res = {"n_train": len(tr), "n_test": len(te), "r2_test_sap": round(float(r2), 4), "term_importance": imp,
           "spearman_ebm_term_vs_our_subscore": agree, "heating_system_ebm_vs_ours": heat.reset_index().to_dict("records"),
           "seed": a.seed}
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
