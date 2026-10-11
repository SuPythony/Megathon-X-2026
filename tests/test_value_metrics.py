"""Distribution metrics (V-15): checked against analytic answers."""

import numpy as np
from scipy import stats

from greentwin.value.metrics import crps_samples, distribution_metrics, samples_from_quantiles


def test_crps_matches_normal_closed_form():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, size=(20000, 1))
    for y in (0.0, 1.3, -2.0):
        z = y
        exact = z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
        assert abs(crps_samples(s, np.array([y]))[0] - exact) < 0.01


def test_calibrated_forecast_scores_as_expected():
    rng = np.random.default_rng(1)
    n, m = 4000, 400
    mu = rng.normal(12, 0.5, n)                 # true conditional means (log price)
    y = mu + rng.normal(0, 0.2, n)
    s = mu[None, :] + rng.normal(0, 0.2, (m, n))  # perfectly calibrated predictive draws
    r = distribution_metrics(y, s)
    for lv in ("50", "80", "90", "95"):
        assert abs(r["distribution"]["intervals"][lv]["coverage"] - int(lv) / 100) < 0.03
    assert r["distribution"]["pit_ks_distance"] < 0.05
    assert abs(r["distribution"]["crps_log"] - 0.2 / np.sqrt(np.pi)) < 0.005  # E[CRPS] = sigma / sqrt(pi)
    assert 0.8 < r["log"]["r2"] < 0.9          # var(mu)=0.25 vs noise 0.04 -> R2 ~ 0.86
    assert abs(r["log"]["mse"] - 0.04) < 0.005


def test_overconfident_forecast_is_penalised():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 3000)
    good = distribution_metrics(y, rng.normal(0, 1, (300, 3000)))
    tight = distribution_metrics(y, rng.normal(0, 0.3, (300, 3000)))
    assert tight["distribution"]["intervals"]["90"]["coverage"] < 0.5
    assert tight["distribution"]["crps_log"] > good["distribution"]["crps_log"]
    assert tight["distribution"]["pit_ks_distance"] > good["distribution"]["pit_ks_distance"]


def test_samples_from_quantiles_reproduce_quantiles():
    rng = np.random.default_rng(3)
    qs = {0.05: np.array([-1.645]), 0.25: np.array([-0.674]), 0.5: np.array([0.0]),
          0.75: np.array([0.674]), 0.95: np.array([1.645])}
    s = samples_from_quantiles(qs, 20000, rng)[:, 0]
    assert abs(np.percentile(s, 50)) < 0.03 and abs(np.percentile(s, 95) - 1.645) < 0.05
