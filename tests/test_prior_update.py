"""Bayesian update of the M04 prior: missing months never become 0, data narrows the posterior, sanity of maths."""
import numpy as np
import pytest

from greentwin.models.energy_prior import QUANTILE_GRID
from greentwin.models.prior_update import (
    NoiseModel, annual_from_months, crps_samples, posterior_weights, prior_samples, resample,
)


def test_missing_month_is_imputed_not_zero():
    months = [100.0] * 12
    months[7] = None
    annual, extra, missing = annual_from_months(months)
    assert missing == [7] and annual == pytest.approx(1200.0)  # neighbours' mean, not 0
    assert extra > 0  # imputation widens the observation noise
    assert annual_from_months([100.0] * 12)[1] == 0.0


def test_too_many_missing_months_rejected():
    with pytest.raises(ValueError):
        annual_from_months([None] * 7 + [1.0] * 5)


def test_noise_fit_recovers_known_scale():
    rng = np.random.default_rng(0)
    theta = rng.lognormal(4, 0.5, 200_000)
    eps = 0.15 * rng.standard_t(5, size=(2, theta.size))
    nm = NoiseModel.fit(theta * np.exp(eps[0]), theta * np.exp(eps[1]))
    assert nm.scale == pytest.approx(0.15, rel=0.1)


def test_posterior_moves_toward_data_and_narrows():
    rng = np.random.default_rng(42)
    rel_q = np.quantile(rng.normal(0, 0.4, 50_000), QUANTILE_GRID)  # prior: relative residual quantiles
    theta = prior_samples(100.0, rel_q, 20_000, rng)
    nm = NoiseModel(scale=0.12, nu=4.0)
    w, ess = posterior_weights(theta, [150.0], nm)
    post = resample(theta, w, 20_000, rng)
    assert np.median(theta) < np.median(post) < 150.0  # shrinks toward the reading, from the prior
    assert np.std(post) < 0.6 * np.std(theta)
    assert 100 < ess <= len(theta)


def test_predictive_draws_are_bounded():
    nm = NoiseModel(scale=0.1, nu=1.2)  # infinite-variance fit
    eps = nm.sample(1_000_000, np.random.default_rng(1))
    assert np.max(np.abs(eps)) <= nm.MAX_LOG_CHANGE + 1e-12


def test_crps_of_point_mass_is_abs_error():
    assert crps_samples(np.full(100, 5.0), 7.0) == pytest.approx(2.0)
