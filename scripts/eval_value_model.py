"""Evaluate a fitted value model from its saved posterior (no refit), plus baselines and confounding checks.

    uv run python scripts/eval_value_model.py --version v1

* Bayesian predictive intervals on held-out and out-of-time sales. Location effects are redrawn from their saved
  mean/std (the SVI guide is mean-field normal, so this is the same distribution).
* Baselines: OLS without / with SAP (LA + type/form + age + year fixed effects) solved by normal equations
  (a 420 x 420 system, no large-matrix memory), and LightGBM quantile.
* Confounding checks for the rating effect (V-11):
  - within-cell: SAP slope after removing LSOA x type/form x age band x sale-year means (homes compared only with
    near-identical neighbours sold the same year);
  - repeat sales: same property sold twice with a different linked SAP; change in log price vs change in SAP.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from greentwin.value import hedonic as H  # noqa: E402
from train_value_model import interval_metrics, ols_design  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_posterior(d: Path, n: int, rng) -> dict:
    z = dict(np.load(d / "posterior.npz"))
    post = {k: v[:n] for k, v in z.items() if not k.endswith(("_mean", "_std"))}
    for k in ("a_lsoa", "a_la"):
        post[k] = rng.normal(z[f"{k}_mean"], z[f"{k}_std"], size=(n, len(z[f"{k}_mean"])))
    return post


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    XtX = X.T.astype(np.float64) @ X.astype(np.float64)
    return np.linalg.solve(XtX + 1e-6 * np.eye(len(XtX)), X.T.astype(np.float64) @ y)


def demean(df: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[cols]
    return df[cols] - g.transform("mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--eval-n", type=int, default=50000)
    ap.add_argument("--ols-n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    art = ROOT / "artifacts" / "value_model" / a.version
    df = pd.read_parquet(ROOT / "data" / "processed" / f"sales_prepared_{a.version}.parquet")
    in_time = ~df["oot"]
    tr = df[(df.split == "train") & in_time]
    te = df[(df.split == "test") & in_time].sample(a.eval_n, random_state=a.seed)
    oot = df[df["oot"] & (df.split != "train")]
    oot = oot.sample(min(a.eval_n, len(oot)), random_state=a.seed)
    metrics: dict = {"version": a.version, "seed": a.seed, "n_test_eval": len(te), "n_oot_eval": len(oot)}

    # ------------------------------------------------------------------ Bayesian predictive
    enc = H.Encoding(json.loads((art / "encoding.json").read_text()))
    fit = H.HedonicFit(enc, {k: len(v) for k, v in enc.levels.items()}, None, None, None)
    post = load_posterior(art, 300, rng)

    def bayes_eval(d):
        chunks = [d.iloc[i] for i in np.array_split(np.arange(len(d)), 10)]
        ys = np.concatenate([fit.predict_samples(c, post, rng) for c in chunks], axis=1)
        lo, med, hi = np.percentile(ys, [5, 50, 95], axis=0)
        return interval_metrics(d["log_price_adj"].to_numpy(), lo, med, hi)

    metrics["bayes"] = {"test": bayes_eval(te), "out_of_time": bayes_eval(oot)}
    log("bayes", json.dumps(metrics["bayes"]))

    # ------------------------------------------------------------------ OLS baselines (normal equations)
    cats = ["laua", "typeform", "age_band", "year"]
    trd = H.add_design_cols(tr.sample(min(a.ols_n, len(tr)), random_state=a.seed))
    ted, ootd = H.add_design_cols(te), H.add_design_cols(oot)
    for with_sap in (False, True):
        Xtr, lv = ols_design(trd, cats, with_sap)
        y = trd["log_price_adj"].to_numpy()
        coef = ols_fit(Xtr, y)
        q5, q95 = np.percentile(y - Xtr @ coef, [5, 95])
        res = {}
        for name, d in (("test", ted), ("out_of_time", ootd)):
            Xd, _ = ols_design(d, cats, with_sap, lv)
            med = Xd @ coef
            res[name] = interval_metrics(d["log_price_adj"].to_numpy(), med + q5, med, med + q95)
        if with_sap:
            res["sap_coef_pct_per_point"] = round(float((np.exp(coef[-1]) - 1) * 100), 4)
            resid = y - Xtr @ coef
            s2 = float(resid @ resid) / (len(y) - Xtr.shape[1])
            XtX = Xtr.T.astype(np.float64) @ Xtr.astype(np.float64)
            se = float(np.sqrt(s2 * np.linalg.inv(XtX + 1e-6 * np.eye(len(XtX)))[-1, -1]))
            res["sap_coef_log_per_point"] = float(coef[-1])
            res["sap_coef_log_se"] = se
        metrics[f"ols_{'with' if with_sap else 'without'}_sap"] = res
        log(f"OLS {'with' if with_sap else 'without'} SAP", json.dumps(res))
        del Xtr

    # ------------------------------------------------------------------ LightGBM quantile
    import lightgbm as lgb
    def lgb_X(d, ref):
        x = pd.DataFrame({"log_tfa": d["log_tfa"], "rooms": d["rooms"], "sap": d["sap"],
                          "epc_age_years": d["epc_age_years"], "lat": d["lat"], "long": d["long"],
                          "year_num": d["year"].astype(int), "leasehold": d["leasehold"].astype(int),
                          "new_build": d["new_build"].astype(int)})
        for c, src in (("ptype_c", "ptype"), ("typeform_c", "typeform"), ("age_c", "age_band")):
            cats_ = sorted(ref[src].astype(object).fillna("unknown").astype(str).unique())
            x[c] = pd.Categorical(d[src].astype(object).fillna("unknown").astype(str), categories=cats_)
        return x
    Xl = lgb_X(trd, trd)
    qs = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=600, learning_rate=0.05,
                               num_leaves=127, min_child_samples=50, random_state=a.seed, verbose=-1
                               ).fit(Xl, trd["log_price_adj"]) for q in (0.05, 0.5, 0.95)}
    metrics["lgbm_quantile"] = {
        name: interval_metrics(d["log_price_adj"].to_numpy(), *(qs[q].predict(lgb_X(d, trd)) for q in (0.05, 0.5, 0.95)))
        for name, d in (("test", ted), ("out_of_time", ootd))}
    log("LightGBM", json.dumps(metrics["lgbm_quantile"]))

    # ------------------------------------------------------------------ confounding checks (V-11)
    w = H.add_design_cols(df[in_time])
    w = w.assign(rooms_f=w["rooms"].fillna(5).clip(1, 12), lease_f=w["leasehold"].astype(float),
                 new_f=w["new_build"].astype(float))
    cols = ["log_price_adj", "sap", "log_tfa", "rooms_f", "lease_f", "new_f", "epc_age_years"]
    cell = ["lsoa21", "typeform", "age_band", "year"]
    w = w.dropna(subset=cell)
    size = w.groupby(cell, observed=True)["sap"].transform("size")
    wc = w[size >= 2]
    dm = demean(wc, cols, cell)
    X = dm[cols[1:]].to_numpy()
    b = ols_fit(X, dm["log_price_adj"].to_numpy())
    within = (np.exp(b[0]) - 1) * 100
    # bootstrap over cells for an interval
    keys = wc[cell].astype(str).agg("|".join, axis=1).to_numpy()
    uk, inv = np.unique(keys, return_inverse=True)
    boots = []
    for _ in range(100):
        pick = rng.integers(0, len(uk), len(uk))
        wts = np.bincount(pick, minlength=len(uk))[inv].astype(float)
        sw = np.sqrt(wts)[:, None]
        bb = ols_fit(X * sw, dm["log_price_adj"].to_numpy() * sw[:, 0])
        boots.append((np.exp(bb[0]) - 1) * 100)
    within_log_draws = np.log1p(np.array(boots) / 100)  # log points per SAP point
    metrics["within_cell"] = {"cells": "LSOA x type/form x age band x sale year, >= 2 sales",
                              "n_sales": int(len(wc)), "n_cells": int(len(uk)),
                              "sap_pct_per_point": round(float(within), 4),
                              "sap_pct_per_point_boot_p5_p95": np.percentile(boots, [5, 95]).round(4).tolist(),
                              "implied_AB_vs_D_pct": round(float((np.exp(b[0] * 29.5) - 1) * 100), 2)}
    log("within-cell", json.dumps(metrics["within_cell"]))

    multi = df[df.duplicated("prop_key", keep=False)].sort_values(["prop_key", "date"])
    first = multi.groupby("prop_key").head(1).set_index("prop_key")
    last = multi.groupby("prop_key").tail(1).set_index("prop_key")
    pair = first.join(last, lsuffix="_1", rsuffix="_2")
    pair = pair[pair["date_2"] > pair["date_1"]]
    pair["d_sap"] = pair["sap_2"] - pair["sap_1"]
    pair["d_lp"] = pair["log_price_adj_2"] - pair["log_price_adj_1"]
    pair["d_tfa"] = pair["log_tfa_2"] - pair["log_tfa_1"]
    ch = pair[(pair["d_sap"] != 0) & (pair["d_tfa"].abs() < 0.05)]  # same floor area: no extension in between
    if len(ch) > 100:
        Xr = np.column_stack([np.ones(len(ch)), ch["d_sap"].to_numpy(float)])
        br = ols_fit(Xr, ch["d_lp"].to_numpy())
        metrics["repeat_sales"] = {"n_pairs_total": int(len(pair)), "n_pairs_sap_changed_same_area": int(len(ch)),
                                   "sap_pct_per_point": round(float((np.exp(br[1]) - 1) * 100), 4),
                                   "median_d_sap": float(ch["d_sap"].median())}
    else:
        metrics["repeat_sales"] = {"n_pairs_total": int(len(pair)), "n_pairs_sap_changed_same_area": int(len(ch)),
                                   "note": "too few repeat sales with a different linked SAP"}
    log("repeat sales", json.dumps(metrics["repeat_sales"]))
    # ------------------------------------------------------------------ V-13 triangulated rating effect
    ols = metrics["ols_with_sap"]
    effects = {
        "doc": "Rating effect per SAP point (log points) from three designs; the product mixes them with equal "
               "weight (decision V-13). bayesian = per region x type posterior draws (beta_rt / 10).",
        "designs": {
            "ols_la_fixed_effects": {"type": "normal", "mean": ols["sap_coef_log_per_point"], "sd": ols["sap_coef_log_se"]},
            "within_neighbourhood_cells": {"type": "normal", "mean": float(within_log_draws.mean()),
                                           "sd": float(within_log_draws.std(ddof=1))},
            "bayesian_hierarchical": {"type": "posterior", "source": "posterior.npz:beta_rt / 10"},
        },
        "weights": {"ols_la_fixed_effects": 1 / 3, "within_neighbourhood_cells": 1 / 3, "bayesian_hierarchical": 1 / 3},
    }
    (art / "rating_effects.json").write_text(json.dumps(effects, indent=2))
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log("wrote", art / "metrics.json")


if __name__ == "__main__":
    main()
