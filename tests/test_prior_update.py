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


# ----------------------------------------------------------------- system entry point (needs the trained model)
import json  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/m04_energy_prior/v2/model.pkl"


@pytest.mark.skipif(not MODEL.exists(), reason="train first")
def test_update_with_meters_end_to_end():
    from contracts.models import EnergyUsageMonth, Property
    from greentwin.models.energy_prior import energy_prior, load_default
    from greentwin.models.prior_update import update_with_meters

    prop = Property.model_validate_json((ROOT / "contracts/examples/SH1_property.json").read_text())
    usage = [EnergyUsageMonth.model_validate(u) for u in
             json.loads((ROOT / "contracts/examples/SH1_usage_2025.json").read_text())]
    assert energy_prior(prop).provenance.engine == "ebm"
    r = update_with_meters(load_default(), prop, usage, n=5000)
    gas = r["fuels"]["gas"]
    assert gas["observed"]["imputed_months"] == ["2025-08"]  # missing August imputed, not zero
    assert gas["posterior"]["std"] < gas["prior"]["std"]
    # Too little data -> prior only, with a reason; no readings -> prior only.
    assert update_with_meters(load_default(), prop, usage[:4], n=2000)["fuels"]["gas"]["posterior"] is None
    assert "no meter readings" in update_with_meters(load_default(), prop, [], n=2000)["fuels"]["electricity"]["note"]
