"""M04 training diagnostics: generalisation gap, learning curve, probabilistic calibration (PIT, reliability).

    uv run python scripts/diagnostics_energy_prior.py      (after scripts/train_energy_prior.py)

Writes artifacts/m04_energy_prior/v2/diagnostics.json (read by the report and the model card).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greentwin.models.energy_prior import (  # noqa: E402
    MODEL_ID, QUANTILE_GRID, SEED, STRATA, EnergyPriorModel, prepare_frame,
)
from greentwin.models.m04_eval import point_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_energy_prior import cells, fit_ebm, load  # noqa: E402


def pit_values(model: EnergyPriorModel, df: pd.DataFrame, fuel: str) -> np.ndarray:
    """Probability integral transform: F_home(y_observed). Uniform on [0, 1] if the predictive is calibrated."""
    yhat = model.predict_point(df, fuel)
    rel = df.y.to_numpy() / yhat - 1  # observed relative residual
    rq = model.residual_quantiles[fuel]
    keys = df[STRATA[fuel]].astype(str).agg("|".join, axis=1).to_numpy()
    pit = np.empty(len(df))
    for k in np.unique(keys):
        m = keys == k
        q = rq.get(k, rq["__all__"])
        pit[m] = np.interp(rel[m], q, QUANTILE_GRID, left=0.0, right=1.0)
    return pit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(ROOT / "artifacts" / MODEL_ID / "v2"))
    ap.add_argument("--parquet", default=str(ROOT / "data/processed/need_intensity.parquet"))
    a = ap.parse_args()
    run = Path(a.run)
    model = EnergyPriorModel.load(run / "model.pkl")
    cfg = json.loads((run / "config.json").read_text())
    out = {"seed": SEED}
    for fuel in ("electricity", "gas"):
        t0 = time.time()
        df = load(Path(a.parquet), fuel)
        res = {}
        # 1. Generalisation gap: the served EBM on every split (train includes IMD as recorded, unmasked).
        res["by_split"] = {s: {k: v for k, v in point_metrics(d.y.to_numpy(), model.predict_point(d, fuel)).items()
                               if k in ("r2", "mae", "rmse", "bias")} | {"n": int(len(d))}
                           for s in ("train", "tune", "cal", "test") for d in [df[df.split == s]]}
        # 2. Probabilistic calibration on test: PIT histogram + reliability at several nominal levels.
        te = df[df.split == "test"]
        pit = pit_values(model, te, fuel)
        hist, _ = np.histogram(pit, bins=20, range=(0, 1))
        levels = [0.5, 0.8, 0.9, 0.95]
        res["pit"] = {"bins": 20, "density": (hist / hist.sum() * 20).round(4).tolist(),
                      "ks_distance_from_uniform": float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, len(pit))))),
                      "reliability": [{"nominal": lv, "empirical": float(np.mean(np.abs(pit - 0.5) <= lv / 2))}
                                      for lv in levels]}
        # 3. Learning curve: EBM with the chosen params, trained on growing samples of homes, scored on test.
        params = {**cfg["best_params"][fuel]["ebm"], "outer_bags": 2}
        tr = df[df.split == "train"]
        rng = np.random.default_rng(SEED)
        curve = []
        for n in (5_000, 20_000, 100_000, 400_000, len(tr)):
            s = tr.sample(min(n, len(tr)), random_state=SEED).copy()
            s.loc[rng.random(len(s)) < cfg["imd_mask_frac"], "imd_quintile"] = np.nan
            c = cells(s, fuel)
            t1 = time.time()
            ebm = fit_ebm(c, fuel, params)
            p = np.maximum(ebm.predict(prepare_frame(te, fuel)), 1.0)
            m = point_metrics(te.y.to_numpy(), p)
            curve.append({"homes": int(len(s)), "cells": int(len(c)), "test_r2": m["r2"], "test_mae": m["mae"],
                          "fit_seconds": round(time.time() - t1, 1)})
            print(f"[{fuel}] learning curve {len(s):>9,} homes ({len(c):,} cells): R2 {m['r2']:.4f} MAE {m['mae']:.3f}",
                  flush=True)
        res["learning_curve"] = curve
        res["seconds"] = round(time.time() - t0, 1)
        out[fuel] = res
        print(f"[{fuel}] by split: " + ", ".join(f"{k} R2 {v['r2']:.4f}" for k, v in res["by_split"].items())
              + f" | PIT KS {res['pit']['ks_distance_from_uniform']:.4f} | "
              + ", ".join(f"{r['nominal']:.0%}->{r['empirical']:.3f}" for r in res["pit"]["reliability"]), flush=True)
    (run / "diagnostics.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {run / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
