"""Build the M04 evaluation report (interactive HTML) from artifacts/m04_energy_prior/v2/results.json.

    uv run python scripts/report_energy_prior.py   ->  eval/reports/m04/report.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "age_band": {1: "pre-1930", 2: "1930-72", 3: "1973-99", 4: "2000+"},
    "floor_band": {1: "<=50 m²", 2: "51-100", 3: "101-150", 4: "151-200", 5: ">200 m²"},
    "epc_band": {1: "A/B", 2: "C", 3: "D", 4: "E", 5: "F/G"},
    "imd_quintile": {1: "1 most deprived", 2: "2", 3: "3", 4: "4", 5: "5 least"},
    "li_flag": {0: "no", 1: "yes"}, "cwi_flag": {0: "no", 1: "yes"}, "pv_flag": {0: "no", 1: "yes"},
    "conservatory": {0: "no", 1: "yes"},
}
FEATURE_NAMES = {
    "prop_type": "Property type", "age_band": "Age band", "floor_band": "Floor-area band", "region": "Region",
    "epc_band": "EPC band", "imd_quintile": "Deprivation (IMD quintile)", "conservatory": "Conservatory",
    "li_flag": "Loft insulation (scheme)", "cwi_flag": "Cavity insulation (scheme)", "pv_flag": "Solar PV",
    "main_heat_fuel": "Main heating fuel",
}


def finite(o):
    """NaN/inf -> None so the embedded JSON is valid (JSON has no NaN)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def term_label(term: str) -> str:
    return " × ".join(FEATURE_NAMES.get(p, p) for p in term.split(" & "))


def relabel_bins(term: str, bins: list[str]) -> list[str]:
    """EBM continuous bins come as 'lo-hi' edges; map ordinal codes back to readable labels."""
    if term not in LABELS:
        return bins
    out = []
    for b in bins:
        if b == "unknown":
            out.append("unknown")
            continue
        try:
            lo, hi = (float(x) for x in b.replace("--", "-").split("-", 1)) if b.count("-") else (float(b),) * 2
            out.append(LABELS[term].get(int(round((lo + hi) / 2)), b))
        except ValueError:
            out.append(b)
    return out


def hist2d(y, p, hi):
    edges = np.linspace(0, hi, 61)
    h, _, _ = np.histogram2d(np.clip(p, 0, hi - 1e-9), np.clip(y, 0, hi - 1e-9), bins=[edges, edges])
    return {"z": h.T.astype(int).tolist(), "edges": edges.round(2).tolist()}


def fuel_payload(name: str, r: dict) -> dict:
    s = r["scatter_sample"]
    y = np.array(s["y"])
    hi = float(np.percentile(y, 99))
    shapes = []
    for t in r["ebm_global"]["terms"]:
        if "bins" in t:
            shapes.append({"term": t["term"], "label": term_label(t["term"]),
                           "bins": relabel_bins(t["term"], t["bins"]), "scores": t["scores"],
                           "lower": t["lower"], "upper": t["upper"], "importance": t["importance"]})
    def value_text(feat, v):
        if v is None:
            return "unknown"
        if isinstance(v, str):
            return v.replace("_", " ")
        return LABELS.get(feat, {}).get(int(v), f"{v:g}")

    ex = []
    for e in r["exemplars"]:
        inp = e.get("inputs", {})
        terms = [(term_label(k) + ": " + ", ".join(value_text(p, inp.get(p)) for p in k.split(" & ")), v)
                 for k, v in e["terms"]]
        ex.append({"title": e["title"][0].upper() + e["title"][1:], "intercept": e["intercept"], "prediction": e["prediction"],
                   "actual": e["actual"], "terms": terms})
    return {
        "fuel": name, "target": r["target"], "n": r["n"], "test": r["test"], "bootstrap": r["bootstrap"],
        "intervals": {k: {kk: v[kk] for kk in ("coverage", "mean_width", "median_width")}
                      for k, v in r["intervals"].items()},
        "ebm_strata_coverage": r["intervals"]["ebm"]["coverage_by_stratum"],
        "ebm_imd_unknown": r["ebm_imd_unknown"], "ceiling": r["ceiling"],
        "archetype": {k: {kk: v[kk] for kk in ("cells", "homes", "r2", "mae", "mape_pct")}
                      for k, v in r["archetype"].items()},
        "archetype_points": r["archetype"]["ebm"]["points"],
        "segments": r["segments"], "calibration": r["calibration_deciles"],
        "hist": {"ebm": hist2d(y, np.array(s["ebm"]), hi), "ols_onehot": hist2d(y, np.array(s["ols_onehot"]), hi)},
        "importance": [{"term": term_label(t["term"]), "ebm": t["importance"]} for t in r["ebm_global"]["terms"]],
        "shap": {FEATURE_NAMES.get(k, k): v for k, v in r["lgb_shap"]["importance"].items()},
        "shap_unit": r["lgb_shap"]["unit"],
        "shapes": shapes, "intercept": r["ebm_global"]["intercept"], "exemplars": ex,
        "monotonicity": r["monotonicity"], "monotonicity_poly2": r.get("monotonicity_poly2_ridge"),
        "linear_search": r.get("linear_search"), "additivity": r["local_additivity_max_abs_err"],
        "best_params": r["best_params"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "artifacts/m04_energy_prior/v2"))
    ap.add_argument("--out", default=str(ROOT / "eval/reports/m04/report.html"))
    a = ap.parse_args()
    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    trials = pd.read_csv(run / "trials.csv")
    showcase = json.loads((run / "showcase.json").read_text())
    config = json.loads((run / "config.json").read_text())
    data = {
        "model_names": res["model_names"],
        "fuels": {f: fuel_payload(f, res[f]) for f in ("electricity", "gas") if f in res},
        "trials": trials[["fuel", "model", "trial", "tune_mae", "seconds"]].round(3).to_dict("records"),
        "showcase": {sh: {"elec": p["electricity_kwh_m2"], "gas": p["gas_kwh_m2"],
                          "contrib_e": p["contributions_electricity"], "contrib_g": p["contributions_gas"]}
                     for sh, p in showcase.items()},
        "config": {k: config[k] for k in ("split", "monotone", "imd_mask_frac", "floor_band_area_m2", "source")},
    }
    for name in ("bayes.json", "showcase_bayes.json", "diagnostics.json"):  # Bayesian layer (scripts/bayes_energy_prior.py), optional
        f = run / name
        data[name.split(".")[0]] = json.loads(f.read_text()) if f.exists() else None
    template = (ROOT / "scripts/report_energy_prior.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(finite(data), separators=(",", ":"), allow_nan=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
