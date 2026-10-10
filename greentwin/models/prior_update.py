"""Bayesian update of the M04 prior with a home's own annual meter readings.

theta = the home's long-run annual intensity (kWh/m2/yr).
  Prior:      theta ~ M04 predictive distribution (EBM point x conformal relative residuals), sampled by
              inverse-CDF interpolation of its quantile grid in log space.
  Likelihood: log(y_year) = log(theta) + eps,  eps ~ Student-t(nu, 0, s), with s and nu ESTIMATED from NEED
              year-on-year changes of the same homes (heavy tails: moves, new occupants, home working).
              Extra variance is added for imputed months and, for raw (not weather-corrected) gas, weather.
  Posterior:  importance weights on the prior samples (self-normalised); ESS is reported.
  Predictive: next year's reading = theta * exp(eps).

This is the per-home closed loop for M04 only. The full physics calibration (M09, NumPyro) uses this prior as
its centre; it does not replace it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from greentwin.models.energy_prior import QUANTILE_GRID


@dataclass
class NoiseModel:
    """Year-to-year noise of one home's annual reading around its long-run level (log scale)."""

    scale: float  # Student-t scale of log(y_year / theta)
    nu: float  # degrees of freedom (heavy tails)
    n_pairs: int = 0

    @classmethod
    def fit(cls, y_a: np.ndarray, y_b: np.ndarray, seed: int = 42, max_n: int = 400_000) -> "NoiseModel":
        """Fit from pairs of valid readings of the SAME homes in two years.

        log(y_b / y_a) = eps_b - eps_a. For t-distributed eps the difference is close to t with scale*sqrt(2);
        we fit a location-free t to the differences and divide the scale by sqrt(2).
        """
        r = np.log(y_b / y_a)
        r = r - np.median(r)  # remove the common year-on-year drift (prices, weather residue)
        rng = np.random.default_rng(seed)
        if len(r) > max_n:
            r = rng.choice(r, max_n, replace=False)
        nu, _, sc = stats.t.fit(r, floc=0.0)
        return cls(scale=float(sc / np.sqrt(2.0)), nu=float(nu), n_pairs=int(len(y_a)))

    def logpdf(self, resid: np.ndarray, extra_var: np.ndarray | float = 0.0) -> np.ndarray:
        s = np.sqrt(self.scale ** 2 + extra_var)
        return stats.t.logpdf(resid / s, df=self.nu) - np.log(s)

    # The fitted nu can be < 2 (infinite variance): a few homes halve or double use year on year. NEED itself
    # suppresses readings outside fixed bounds, so changes beyond x4 cannot occur in the data; truncating draws
    # there keeps exp(eps) with a finite mean. The likelihood stays heavy-tailed (robust to outlying readings).
    MAX_LOG_CHANGE = float(np.log(4.0))

    def sample(self, size, rng) -> np.ndarray:
        eps = self.scale * rng.standard_t(self.nu, size=size)
        return np.clip(eps, -self.MAX_LOG_CHANGE, self.MAX_LOG_CHANGE)


def prior_samples(yhat: float | np.ndarray, rel_quantiles: np.ndarray, n: int, rng, factor: float = 1.0) -> np.ndarray:
    """Draw n prior samples of theta: inverse-CDF on the conformal quantile grid, interpolated in log space.

    yhat may be a vector (one row per home) -> returns (homes, n). ``factor`` rescales (within-band area).
    """
    u = (np.arange(n) + rng.random(n)) / n  # stratified uniforms: low-variance inverse-CDF sampling
    u = np.clip(u, QUANTILE_GRID[0], QUANTILE_GRID[-1])
    logq = np.log(np.maximum(1 + rel_quantiles, 1e-3))  # relative residual quantiles -> log multipliers
    mult = np.exp(np.interp(u, QUANTILE_GRID, logq))
    return np.multiply.outer(np.atleast_1d(yhat) * factor, mult).squeeze()


def posterior_weights(theta: np.ndarray, obs: list[float], noise: NoiseModel,
                      extra_var: list[float] | None = None) -> tuple[np.ndarray, float]:
    """Self-normalised importance weights of prior samples theta given observed annual intensities."""
    logw = np.zeros_like(theta, dtype=float)
    for i, y in enumerate(obs):
        ev = 0.0 if extra_var is None else extra_var[i]
        logw += noise.logpdf(np.log(y) - np.log(theta), ev)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    return w, ess


def resample(theta: np.ndarray, w: np.ndarray, n: int, rng) -> np.ndarray:
    """Systematic resampling (lower variance than multinomial)."""
    pos = (np.arange(n) + rng.random()) / n
    idx = np.minimum(np.searchsorted(np.cumsum(w), pos), len(theta) - 1)
    return theta[idx]


def crps_samples(x: np.ndarray, y: float) -> float:
    """CRPS from samples (energy form): E|X - y| - 0.5 E|X - X'|. Lower is better; units of y."""
    x = np.sort(x)
    n = len(x)
    term1 = np.mean(np.abs(x - y))
    # E|X - X'| for sorted samples: (2 / n^2) * sum_i (2i - n - 1) x_i
    term2 = 2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2
    return float(term1 - 0.5 * term2)


def annual_from_months(values: list[float | None], seasonal_cv_missing: float = 0.5) -> tuple[float, float, list[int]]:
    """Annual total from 12 monthly readings. Missing months are IMPUTED (mean of the neighbouring months,
    wrapping round the year), never set to 0. Returns (annual, extra log-variance from imputation, imputed idx).
    """
    v = list(values)
    if len(v) != 12:
        raise ValueError("need 12 monthly readings (use None for missing)")
    missing = [i for i, x in enumerate(v) if x is None]
    if len(missing) > 6:
        raise ValueError("more than 6 months missing: too little data for an annual reading")
    filled = v[:]
    for i in missing:
        nb = [v[(i + d) % 12] for d in (-1, 1) if v[(i + d) % 12] is not None]
        filled[i] = float(np.mean(nb)) if nb else float(np.mean([x for x in v if x is not None]))
    annual = float(np.sum(filled))
    # Each imputed month m_i has an independent relative error of sd seasonal_cv_missing (ASSUMPTION, 50%):
    # var(log annual) ~= sum (c * m_i)^2 / annual^2 (delta method).
    extra = float(sum((seasonal_cv_missing * filled[i]) ** 2 for i in missing) / annual ** 2) if annual > 0 else 0.0
    return annual, extra, missing
