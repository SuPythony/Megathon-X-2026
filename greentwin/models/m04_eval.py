"""M04 evaluation helpers: point metrics, paired bootstrap, conformal intervals, ceiling, archetype accuracy.

All metrics are computed on raw held-out homes (never on aggregated cells).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from greentwin.models.energy_prior import STRATA, fit_residual_quantiles, interval_from_quantiles


def point_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    e = y - yhat
    ape = np.abs(e) / y
    sst = np.sum((y - y.mean()) ** 2)
    return {
        "r2": float(1 - np.sum(e ** 2) / sst),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mape_pct": float(100 * np.mean(ape)),
        "mdape_pct": float(100 * np.median(ape)),
        "bias": float(np.mean(-e)),
        "spearman": float(spearmanr(y, yhat).statistic) if np.ptp(yhat) > 0 else None,  # undefined if constant
    }


def paired_bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], reference: str, n_boot: int = 1000,
                     seed: int = 42) -> dict:
    """95% CIs for each model's R2/MAE and for (reference - model) deltas, resampling the SAME homes per draw.

    Pairing removes between-home noise from the comparison, so a CI on dMAE that excludes 0 is a real win.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(preds)
    abs_e = {k: np.abs(y - p) for k, p in preds.items()}
    sq_e = {k: (y - p) ** 2 for k, p in preds.items()}
    r2 = {k: np.empty(n_boot) for k in names}
    mae = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        sst = np.sum((yb - yb.mean()) ** 2)
        for k in names:
            r2[k][b] = 1 - sq_e[k][idx].sum() / sst
            mae[k][b] = abs_e[k][idx].mean()

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    out = {"n_boot": n_boot, "reference": reference, "models": {}, "delta_vs_reference": {}}
    for k in names:
        out["models"][k] = {"r2_ci95": ci(r2[k]), "mae_ci95": ci(mae[k])}
        if k != reference:
            d_mae = mae[k] - mae[reference]  # > 0 => reference better
            d_r2 = r2[reference] - r2[k]  # > 0 => reference better
            out["delta_vs_reference"][k] = {
                "mae_gain": float(np.mean(d_mae)), "mae_gain_ci95": ci(d_mae),
                "mae_gain_pct": float(100 * np.mean(d_mae / mae[k])),
                "r2_gain": float(np.mean(d_r2)), "r2_gain_ci95": ci(d_r2),
                "p_reference_not_better_mae": float(np.mean(d_mae <= 0)),
            }
    return out


def conformal_eval(cal: pd.DataFrame, yhat_cal: np.ndarray, test: pd.DataFrame, yhat_test: np.ndarray,
                   fuel: str, level: float = 0.90) -> tuple[dict, dict]:
    """Split-conformal relative-residual intervals per stratum; returns (summary, residual quantiles)."""
    rq = fit_residual_quantiles(cal, cal.y.to_numpy(), np.maximum(yhat_cal, 1.0), fuel)
    keys = test[STRATA[fuel]].astype(str).agg("|".join, axis=1).to_numpy()
    yh = np.maximum(yhat_test, 1.0)
    lo, hi = interval_from_quantiles(yh, keys, rq, level)
    y = test.y.to_numpy()
    inside = (y >= lo) & (y <= hi)
    return {"level": level, "coverage": float(inside.mean()), "mean_width": float(np.mean(hi - lo)),
            "median_width": float(np.median(hi - lo)),
            "coverage_by_stratum": {k: float(inside[keys == k].mean()) for k in np.unique(keys)}}, rq


def variance_ceiling(df: pd.DataFrame, features: list[str]) -> dict:
    """Share of variance BETWEEN feature cells = the most any model of these features could explain.

    eta2 is in-sample (optimistic: singleton cells have no within variance). omega2 removes that bias:
    (SSB - (k-1) MSW) / (SST + MSW). Missing values form their own level (as they do in the models).
    """
    keys = df.groupby(features, dropna=False, sort=False).ngroup().to_numpy()  # NaN = its own level
    y = df.y.to_numpy()
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k").y.agg(["count", "mean", "var"])
    n, k = len(y), len(g)
    sst = float(np.sum((y - y.mean()) ** 2))
    ssw = float(np.nansum(g["var"].fillna(0) * (g["count"] - 1)))
    ssb = sst - ssw
    msw = ssw / (n - k)
    return {"n": n, "cells": k, "eta2_in_sample": ssb / sst,
            "omega2_unbiased": (ssb - (k - 1) * msw) / (sst + msw),
            "median_rows_per_cell": float(g["count"].median())}


def archetype_r2(test: pd.DataFrame, yhat: np.ndarray, by=("prop_type", "age_band", "floor_band", "region"),
                 min_n: int = 50) -> dict:
    """Accuracy at archetype level: mean predicted vs mean metered per cell (cells with >= min_n test homes)."""
    d = test[list(by)].copy()
    d["y"], d["p"] = test.y.to_numpy(), yhat
    g = d.groupby(list(by)).agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean")).reset_index()
    g = g[g.n >= min_n]
    sst = np.sum((g.y - g.y.mean()) ** 2)
    return {"cells": int(len(g)), "homes": int(g.n.sum()), "min_n": min_n,
            "r2": float(1 - np.sum((g.y - g.p) ** 2) / sst),
            "mae": float(np.mean(np.abs(g.y - g.p))),
            "mape_pct": float(100 * np.mean(np.abs(g.y - g.p) / g.y)),
            "points": g[["y", "p", "n"]].round(2).to_dict("list")}


def segment_table(test: pd.DataFrame, yhat: np.ndarray, col: str) -> list[dict]:
    rows = []
    y = test.y.to_numpy()
    labels = test[col].astype("string").fillna("unknown").astype(str).to_numpy()
    for v in sorted(np.unique(labels)):
        m = labels == v
        if m.sum() < 100:
            continue
        pm = point_metrics(y[m], yhat[m])
        rows.append({"segment": v, "n": int(m.sum()), "mean_y": float(y[m].mean()),
                     **{k: pm[k] for k in ("r2", "mae", "mape_pct", "bias")}})
    return rows


def decile_calibration(y: np.ndarray, yhat: np.ndarray, q: int = 10) -> dict:
    """Mean metered vs mean predicted per decile of the prediction (reliability of the point forecast)."""
    bins = pd.qcut(yhat, q, labels=False, duplicates="drop")
    d = pd.DataFrame({"b": bins, "y": y, "p": yhat}).groupby("b").agg(y=("y", "mean"), p=("p", "mean"))
    return {"pred_mean": d.p.round(2).tolist(), "actual_mean": d.y.round(2).tolist()}
