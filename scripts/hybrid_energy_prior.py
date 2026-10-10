"""Hybrid M04: served EBM + a monotone LightGBM correction fitted on the EBM's residuals.

    uv run python scripts/hybrid_energy_prior.py            # experiment -> hybrid.json + model_hybrid.pkl
    uv run python scripts/hybrid_energy_prior.py --serve    # also make it the served model if it passes

Prediction = EBM terms + one "higher-order interactions" term. Both parts use the same monotone constraints,
so the sum is monotone too; the waterfall still sums exactly. Same splits, cells and proof battery as the
main pipeline: tune for early stopping and selection, cal for conformal, test for every reported number.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import ParameterSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from greentwin.models import m04_eval as ev  # noqa: E402
from greentwin.models.energy_prior import (  # noqa: E402
    FEATURES, MONOTONE, SEED, EnergyPriorModel, prepare_frame, property_to_features, tree_frame,
)
from train_energy_prior import IMD_MASK_FRAC, cells, fit_lgb, load, monotone_sweep, pred_lgb  # noqa: E402
from diagnostics_energy_prior import pit_values  # noqa: E402

RUN = ROOT / "artifacts/m04_energy_prior/v2"
SPACE = {"num_leaves": [31, 63, 127, 255], "learning_rate": [0.03, 0.05], "min_sum_hessian_in_leaf": [10, 50, 200],
         "lambda_l2": [10, 100], "feature_fraction": [0.7, 0.9]}


def fit_correction(tr_c, tu_c, ebm, fuel, p):
    r_tr = tr_c.y - ebm.predict(prepare_frame(tr_c, fuel))
    r_tu = tu_c.y - ebm.predict(prepare_frame(tu_c, fuel))
    params = {"objective": "regression", "learning_rate": p["learning_rate"], "num_leaves": p["num_leaves"],
              "min_sum_hessian_in_leaf": p["min_sum_hessian_in_leaf"], "min_child_samples": 1,
              "lambda_l2": p["lambda_l2"], "feature_fraction": p["feature_fraction"], "cat_smooth": 10,
              "monotone_constraints": [MONOTONE[fuel].get(f, 0) for f in FEATURES[fuel]],
              "monotone_constraints_method": "advanced", "verbose": -1, "seed": SEED, "num_threads": -1}
    dtr = lgb.Dataset(tree_frame(tr_c, fuel), r_tr, weight=tr_c.w)
    dtu = lgb.Dataset(tree_frame(tu_c, fuel), r_tu, weight=tu_c.w, reference=dtr)
    booster = lgb.train(params, dtr, 5000, valid_sets=[dtu], callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = ebm.predict(prepare_frame(tu_c, fuel)) + booster.predict(tree_frame(tu_c, fuel), num_iteration=booster.best_iteration)
    return booster, float(np.average(np.abs(tu_c.y - pred), weights=tu_c.w))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--serve", action="store_true", help="replace the served model if the hybrid passes every gate")
    a = ap.parse_args()
    served = EnergyPriorModel.load(RUN / "model.pkl")
    cfg = json.loads((RUN / "config.json").read_text())
    rng = np.random.default_rng(SEED)
    hybrid = EnergyPriorModel(models=dict(served.models), residual_quantiles={}, metadata=dict(served.metadata),
                              corrections={})
    report = {"seed": SEED, "search_space": SPACE}
    for fuel in ("electricity", "gas"):
        t0 = time.time()
        df = load(ROOT / "data/processed/need_intensity.parquet", fuel)
        tr, tu, ca, te = (df[df.split == s].copy() for s in ("train", "tune", "cal", "test"))
        tr.loc[rng.random(len(tr)) < IMD_MASK_FRAC, "imd_quintile"] = np.nan
        tr_c, tu_c = cells(tr, fuel), cells(tu, fuel)
        ebm = served.models[fuel]
        trials, best = [], (np.inf, None, None)
        for i, p in enumerate(ParameterSampler(SPACE, n_iter=a.trials, random_state=SEED)):
            booster, score = fit_correction(tr_c, tu_c, ebm, fuel, p)
            trials.append({**p, "tune_mae": score, "rounds": booster.best_iteration})
            print(f"  [{fuel}] trial {i} tune MAE {score:.3f} rounds {booster.best_iteration} {p}", flush=True)
            if score < best[0]:
                best = (score, p, booster)
        hybrid.corrections[fuel] = best[2]
        # Reference models on the same test homes.
        lgbm = fit_lgb(tr_c, tu_c, fuel, cfg["best_params"][fuel]["lightgbm"])
        y = te.y.to_numpy()
        preds = {"ebm": served.predict_point(te, fuel), "hybrid": hybrid.predict_point(te, fuel),
                 "lightgbm": pred_lgb(lgbm, te, fuel)}
        test = {k: ev.point_metrics(y, v) for k, v in preds.items()}
        boot = ev.paired_bootstrap(y, preds, reference="hybrid", n_boot=a.n_boot)
        iv, rq = ev.conformal_eval(ca, hybrid.predict_point(ca, fuel), te, preds["hybrid"], fuel)
        hybrid.residual_quantiles[fuel] = rq
        pit = pit_values(hybrid, te, fuel)
        corr = hybrid.correction(te, fuel)
        # Additivity of the hybrid waterfall on a sample of test homes (before the floor at 1 kWh/m2).
        s = te.sample(300, random_state=SEED)
        add_err = max(abs(sum(c.contribution for c in hybrid.contributions(s.iloc[[i]], fuel))
                          - float(ebm.predict(prepare_frame(s.iloc[[i]], fuel))[0] + hybrid.correction(s.iloc[[i]], fuel)[0]))
                      for i in range(len(s)))
        # Category mapping: one home scored alone must equal the same home scored in a batch.
        single = np.array([hybrid.predict_point(s.iloc[[i]], fuel)[0] for i in range(50)])
        batch = hybrid.predict_point(s.iloc[:50], fuel)
        report[fuel] = {
            "best_params": best[1], "trials": trials, "test": test,
            "hybrid_vs": boot["delta_vs_reference"], "ci": boot["models"],
            "interval_90": {k: iv[k] for k in ("coverage", "mean_width", "median_width")},
            "pit_ks": float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, len(pit))))),
            "reliability": {lv: float(np.mean(np.abs(pit - 0.5) <= lv / 2)) for lv in (0.5, 0.8, 0.9, 0.95)},
            "monotonicity": monotone_sweep(hybrid, te, fuel),
            "correction": {"mean_abs": float(np.mean(np.abs(corr))), "p95_abs": float(np.percentile(np.abs(corr), 95)),
                           "share_of_prediction_pct": float(100 * np.mean(np.abs(corr)) / np.mean(preds["hybrid"]))},
            "additivity_max_abs_err": float(add_err),
            "single_vs_batch_max_diff": float(np.max(np.abs(single - batch))),
            "ceiling_omega2": json.loads((RUN / "results.json").read_text())[fuel]["ceiling"]["all_features"]["omega2_unbiased"],
            "seconds": round(time.time() - t0, 1),
        }
        h, e, l = test["hybrid"], test["ebm"], test["lightgbm"]
        print(f"[{fuel}] R2 EBM {e['r2']:.4f}  hybrid {h['r2']:.4f}  LightGBM {l['r2']:.4f} | "
              f"MAE {e['mae']:.3f} / {h['mae']:.3f} / {l['mae']:.3f} | cover {iv['coverage']:.3f} | "
              f"PIT KS {report[fuel]['pit_ks']:.4f} | mono {[v['violations'] for v in report[fuel]['monotonicity'].values()]} | "
              f"|corr| {report[fuel]['correction']['mean_abs']:.2f} | add err {add_err:.3f} | single/batch {report[fuel]['single_vs_batch_max_diff']:.2e}",
              flush=True)
    # Gates: every one must pass before the hybrid may replace the served EBM.
    gates = {}
    for fuel in ("electricity", "gas"):
        r = report[fuel]
        gates[fuel] = {
            "beats_ebm_mae_ci": r["hybrid_vs"]["ebm"]["mae_gain_ci95"][0] > 0,
            "zero_monotone_violations": all(v["violations"] == 0 for v in r["monotonicity"].values()),
            "coverage_88_92": 0.88 <= r["interval_90"]["coverage"] <= 0.92,
            "pit_ks_below_0.01": r["pit_ks"] < 0.01,
            "additive_waterfall": r["additivity_max_abs_err"] < 0.2,
            "single_equals_batch": r["single_vs_batch_max_diff"] < 1e-6,
        }
    report["gates"] = gates
    report["passed"] = all(all(g.values()) for g in gates.values())
    hybrid.metadata["hybrid"] = {"kind": "EBM + monotone LightGBM residual correction",
                                 "params": {f: report[f]["best_params"] for f in ("electricity", "gas")}}
    hybrid.save(RUN / "model_hybrid.pkl")
    (RUN / "hybrid.json").write_text(json.dumps(report, indent=2, default=str))
    print("gates:", json.dumps(gates), "-> PASSED" if report["passed"] else "-> NOT PASSED")
    if a.serve and report["passed"]:
        shutil.copy(RUN / "model.pkl", RUN / "model_ebm_only.pkl")
        shutil.copy(RUN / "model_hybrid.pkl", RUN / "model.pkl")
        print("served model is now the hybrid (EBM-only kept as model_ebm_only.pkl)")


if __name__ == "__main__":
    main()
