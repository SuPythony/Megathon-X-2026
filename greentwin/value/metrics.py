"""Evaluation metrics for predictive DISTRIBUTIONS (V-15), shared by every value model and baseline.

Every model is scored from predictive samples (draws x observations) on log price, so point and distributional
metrics are computed the same way for all models:

* point (from the predictive mean on the log scale, and from exp(median) on the price scale):
  MSE, RMSE, MAE, R2 (log); RMSE, MAE (GBP), R2 (price), MAPE, median APE, Spearman rank correlation;
* distributional: CRPS (continuous ranked probability score, a proper scoring rule; lower is better),
  coverage of central 50/80/90/95% intervals and their mean widths, mean pinball loss at 5/50/95%,
  PIT (probability integral transform) KS distance from uniform (0 = perfectly calibrated), and the mean log
  predictive density from a Gaussian fitted to each observation's samples (a simple log score).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

LEVELS = (0.5, 0.8, 0.9, 0.95)


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-observation CRPS from draws (m x n): E|X - y| - 0.5 E|X - X'| (exact for the empirical distribution)."""
    x = np.sort(samples, axis=0)
    m = x.shape[0]
    term1 = np.mean(np.abs(x - y[None, :]), axis=0)
    w = (2 * np.arange(1, m + 1) - m - 1)[:, None]
    term2 = np.sum(w * x, axis=0) / (m * m)  # = 0.5 E|X - X'|
    return term1 - term2


def distribution_metrics(y_log: np.ndarray, samples_log: np.ndarray) -> dict:
    y = np.asarray(y_log, float)
    s = np.asarray(samples_log, float)
    mean = s.mean(axis=0)
    med = np.median(s, axis=0)
    resid = y - mean
    mse = float(np.mean(resid ** 2))
    price, p_hat = np.exp(y), np.exp(med)
    perr = p_hat - price
    out = {
        "n": int(len(y)), "draws": int(s.shape[0]),
        "log": {"mse": round(mse, 5), "rmse": round(float(np.sqrt(mse)), 5),
                "mae": round(float(np.mean(np.abs(resid))), 5),
                "r2": round(float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)), 4)},
        "price_gbp": {"rmse": round(float(np.sqrt(np.mean(perr ** 2))), 0),
                      "mae": round(float(np.mean(np.abs(perr))), 0),
                      "r2": round(float(1 - np.sum(perr ** 2) / np.sum((price - price.mean()) ** 2)), 4),
                      "mape": round(float(np.mean(np.abs(perr) / price)), 4),
                      "median_ape": round(float(np.median(np.abs(perr) / price)), 4),
                      "spearman": round(float(stats.spearmanr(p_hat, price).statistic), 4)},
    }
    cov = {}
    for lv in LEVELS:
        lo, hi = np.percentile(s, [50 * (1 - lv), 50 * (1 + lv)], axis=0)
        cov[f"{int(lv * 100)}"] = {"coverage": round(float(np.mean((y >= lo) & (y <= hi))), 4),
                                   "mean_width_log": round(float(np.mean(hi - lo)), 4)}
    q5, q50, q95 = np.percentile(s, [5, 50, 95], axis=0)

    def pin(q, p):
        e = y - p
        return np.mean(np.maximum(q * e, (q - 1) * e))

    pit = np.mean(s <= y[None, :], axis=0)
    sd = s.std(axis=0, ddof=1) + 1e-9
    out["distribution"] = {
        "crps_log": round(float(np.mean(crps_samples(s, y))), 5),
        "intervals": cov,
        "pinball_mean_5_50_95": round(float((pin(0.05, q5) + pin(0.5, q50) + pin(0.95, q95)) / 3), 5),
        "pit_ks_distance": round(float(stats.kstest(pit, "uniform").statistic), 4),
        "log_score_gaussian": round(float(np.mean(stats.norm.logpdf(y, mean, sd))), 4),
    }
    return out


def samples_from_quantiles(qs: dict[float, np.ndarray], m: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate predictive draws from a few predicted quantiles (piecewise-linear inverse CDF, linear tails),
    for quantile models such as LightGBM. Marked 'approximate' wherever used."""
    levels = np.array(sorted(qs))
    vals = np.vstack([np.sort(np.vstack([qs[q] for q in levels]), axis=0)])  # enforce non-crossing
    lo = vals[0] - (vals[1] - vals[0]) * levels[0] / (levels[1] - levels[0])
    hi = vals[-1] + (vals[-1] - vals[-2]) * (1 - levels[-1]) / (levels[-1] - levels[-2])
    grid = np.concatenate([[0.0], levels, [1.0]])
    allv = np.vstack([lo, vals, hi])
    u = rng.uniform(size=(m, vals.shape[1]))
    out = np.empty_like(u)
    for j in range(vals.shape[1]):
        out[:, j] = np.interp(u[:, j], grid, allv[:, j])
    return out
