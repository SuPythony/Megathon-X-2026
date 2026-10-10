"""Train and evaluate the property-value model (Part 4). Decisions: docs/value/DECISIONS.md.

    uv run python scripts/train_value_model.py --sales data/interim/sales/tranall_link_26122024_s10.parquet

Outputs artifacts/value_model/<version>/: metrics.json, config.json, posterior.npz (small parameters as samples,
location effects as mean/std), encoding.json.
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
from greentwin.value import hedonic as H  # noqa: E402
from greentwin.value.prepare import PrepConfig, load_hpi, prepare  # noqa: E402

BAND_SAP_MID = {"A/B": 91.0, "C": 74.5, "D": 61.5, "E": 46.5, "F/G": 20.0}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def interval_metrics(y: np.ndarray, lo: np.ndarray, med: np.ndarray, hi: np.ndarray) -> dict:
    """90% coverage, mean pinball loss at 5/50/95%, MAE and median APE of the median (log and price scale)."""
    def pinball(q, pred):
        e = y - pred
        return float(np.mean(np.maximum(q * e, (q - 1) * e)))
    return {"coverage_90": round(float(np.mean((y >= lo) & (y <= hi))), 4),
            "mean_width_90_log": round(float(np.mean(hi - lo)), 4),
            "pinball_mean": round((pinball(0.05, lo) + pinball(0.5, med) + pinball(0.95, hi)) / 3, 5),
            "mae_log": round(float(np.mean(np.abs(y - med))), 4),
            "median_ape_price": round(float(np.median(np.abs(np.exp(med) - np.exp(y)) / np.exp(y))), 4),
            "n": int(len(y))}


def ols_design(d: pd.DataFrame, cols_cat: list[str], with_sap: bool, levels: dict | None = None):
    X = [np.ones(len(d)), d["log_tfa"].to_numpy(float), d["rooms"].fillna(5).clip(1, 12).to_numpy(float),
         d["leasehold"].astype(float).to_numpy(), d["new_build"].astype(float).to_numpy(),
         d["epc_age_years"].to_numpy(float)]
    levels = levels or {c: sorted(d[c].astype(str).unique())[1:] for c in cols_cat}
    for c in cols_cat:
        v = d[c].astype(str).to_numpy()
        X += [(v == lv).astype(float) for lv in levels[c]]
    if with_sap:
        X.append(d["sap"].to_numpy(float))
    return np.column_stack(X), levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", default="data/interim/sales/tranall_link_26122024_s10.parquet")
    ap.add_argument("--nspl", default="data/interim/geo/NSPL21_NOV_2024_UK.parquet")
    ap.add_argument("--hpi", default="data/raw/hpi/UK-HPI-full-file-2026-07.csv")
    ap.add_argument("--max-epc-age", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--eval-n", type=int, default=50000)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = ROOT / "artifacts" / "value_model" / a.version
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ------------------------------------------------------------------ data
    sales = pd.read_parquet(a.sales)
    nspl = pd.read_parquet(a.nspl)
    hpi = load_hpi(a.hpi)
    df, prep_log = prepare(sales, nspl, hpi, PrepConfig(max_epc_age_years=a.max_epc_age, seed=a.seed))
    log("prepared", json.dumps(prep_log, default=str))
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc / f"sales_prepared_{a.version}.parquet", index=False)
    in_time = ~df["oot"]
    tr = df[(df.split == "train") & in_time]
    te = df[(df.split == "test") & in_time]
    oot = df[df["oot"] & (df.split != "train")]  # unseen properties AND unseen period
    log(f"train {len(tr):,}  test {len(te):,}  out-of-time {len(oot):,}")

    # ------------------------------------------------------------------ Bayesian hierarchical hedonic
    log("fitting Bayesian hedonic (SVI) ...")
    fit = H.fit(tr, steps=a.steps, batch=a.batch, seed=a.seed)
    post = fit.posterior(1000, a.seed)
    log(f"ELBO loss start {fit.losses[:100].mean():.0f} end {fit.losses[-100:].mean():.0f}")

    def bayes_eval(d):
        d = d.sample(min(a.eval_n, len(d)), random_state=a.seed)
        ys = fit.predict_samples(d, post, rng)
        lo, med, hi = np.percentile(ys, [5, 50, 95], axis=0)
        return interval_metrics(d["log_price_adj"].to_numpy(), lo, med, hi)

    metrics = {"seed": a.seed, "prep": prep_log, "n_train": len(tr), "n_test": len(te), "n_oot": len(oot),
               "bayes": {"test": bayes_eval(te), "out_of_time": bayes_eval(oot)}}

    # rating effect: beta per 10 SAP points -> % per SAP point and implied band premiums vs D
    beta = post["beta_rt"][:, :-1]  # drop the 'unknown' slot
    rt_levels = fit.enc.levels["rt"]
    pct_per_pt = (np.exp(beta / H.SAP_SCALE) - 1) * 100
    metrics["rating_effect"] = {
        "beta0_pct_per_sap_point": np.percentile((np.exp(post["beta0"] / H.SAP_SCALE) - 1) * 100,
                                                 [5, 50, 95]).round(4).tolist(),
        "by_region_type_pct_per_point_p50": dict(zip(rt_levels, np.median(pct_per_pt, 0).round(4))),
        "by_region_type_pct_per_point_p5": dict(zip(rt_levels, np.percentile(pct_per_pt, 5, 0).round(4))),
        "by_region_type_pct_per_point_p95": dict(zip(rt_levels, np.percentile(pct_per_pt, 95, 0).round(4))),
        "tau_beta": np.percentile(post["tau_beta"], [5, 50, 95]).round(4).tolist(),
    }
    b0 = post["beta0"]
    metrics["implied_premium_vs_D_pct"] = {
        band: np.percentile((np.exp(b0 * (s - BAND_SAP_MID["D"]) / H.SAP_SCALE) - 1) * 100, [5, 50, 95]).round(2).tolist()
        for band, s in BAND_SAP_MID.items() if band != "D"}
    log("rating effect", json.dumps(metrics["rating_effect"]["beta0_pct_per_sap_point"]),
        json.dumps(metrics["implied_premium_vs_D_pct"]))

    # ------------------------------------------------------------------ baselines
    cats = ["laua", "typeform", "age_band", "year"]
    trd, ted, ootd = (H.add_design_cols(x) for x in (tr, te.sample(min(a.eval_n, len(te)), random_state=a.seed),
                                                    oot.sample(min(a.eval_n, len(oot)), random_state=a.seed)))
    for with_sap in (False, True):
        Xtr, lv = ols_design(trd, cats, with_sap)
        coef, *_ = np.linalg.lstsq(Xtr, trd["log_price_adj"].to_numpy(), rcond=None)
        resid = trd["log_price_adj"].to_numpy() - Xtr @ coef
        q5, q95 = np.percentile(resid, [5, 95])
        res = {}
        for name, d in (("test", ted), ("out_of_time", ootd)):
            Xd, _ = ols_design(d, cats, with_sap, lv)
            med = Xd @ coef
            res[name] = interval_metrics(d["log_price_adj"].to_numpy(), med + q5, med, med + q95)
        if with_sap:
            res["sap_coef_pct_per_point"] = round(float((np.exp(coef[-1]) - 1) * 100), 4)
        metrics[f"ols_{'with' if with_sap else 'without'}_sap"] = res
        log(f"OLS {'with' if with_sap else 'without'} SAP", json.dumps(res))

    import lightgbm as lgb
    feats = ["log_tfa", "rooms", "sap", "epc_age_years", "lat", "long", "year_num", "leasehold", "new_build",
             "ptype_c", "typeform_c", "age_c"]
    def lgb_X(d):
        x = pd.DataFrame({"log_tfa": d["log_tfa"], "rooms": d["rooms"], "sap": d["sap"],
                          "epc_age_years": d["epc_age_years"], "lat": d["lat"], "long": d["long"],
                          "year_num": d["year"].astype(int), "leasehold": d["leasehold"].astype(int),
                          "new_build": d["new_build"].astype(int)})
        for c, src in (("ptype_c", "ptype"), ("typeform_c", "typeform"), ("age_c", "age_band")):
            x[c] = pd.Categorical(d[src].astype(str), categories=sorted(trd[src].astype(str).unique()))
        return x[feats]
    Xl = lgb_X(trd.sample(min(500_000, len(trd)), random_state=a.seed))
    yl = trd.loc[Xl.index, "log_price_adj"]
    qs = {}
    for q in (0.05, 0.5, 0.95):
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=600, learning_rate=0.05, num_leaves=127,
                              min_child_samples=50, random_state=a.seed, verbose=-1)
        qs[q] = m.fit(Xl, yl)
    res = {}
    for name, d in (("test", ted), ("out_of_time", ootd)):
        Xd = lgb_X(d)
        res[name] = interval_metrics(d["log_price_adj"].to_numpy(), qs[0.05].predict(Xd), qs[0.5].predict(Xd),
                                     qs[0.95].predict(Xd))
    metrics["lgbm_quantile"] = res
    log("LightGBM quantile", json.dumps(res))

    # ------------------------------------------------------------------ save
    keep_samples = {k: v for k, v in post.items() if k not in ("a_lsoa", "a_la")}
    np.savez_compressed(out / "posterior.npz", **keep_samples,
                        a_lsoa_mean=post["a_lsoa"].mean(0), a_lsoa_std=post["a_lsoa"].std(0),
                        a_la_mean=post["a_la"].mean(0), a_la_std=post["a_la"].std(0))
    (out / "encoding.json").write_text(json.dumps(fit.enc.levels))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    log("wrote", out)


if __name__ == "__main__":
    main()
