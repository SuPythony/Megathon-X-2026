"""Learn Green Home Score weights from data (the learned-weights companion score, decision D-05/D-06).

Model family: target ~ a + c * sum_k w_k s_k, with w on the simplex (w_k >= 0, sum w = 1) and c > 0, i.e. the
same functional form as the score. Fitted as non-negative least squares on centred data; w = beta / sum(beta).
Robustness: standardised OLS, Shapley (LMG) R^2 shares, monotone LightGBM + grouped SHAP, Bayesian Dirichlet.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def split_of(keys: pd.Series, seed: int = 42) -> pd.Series:
    """70/15/15 by a stable hash of the property key (assigned once; no key spans two splits)."""
    def f(k) -> str:
        h = int(hashlib.md5(f"{seed}:{k}".encode()).hexdigest()[:8], 16) % 100
        return "train" if h < 70 else "val" if h < 85 else "test"
    return keys.astype(str).map(f)


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


@dataclass
class SimplexFit:
    weights: dict[str, float]
    scale: float        # c: target units per score point
    intercept: float
    r2_train: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        s = sum(self.weights[k] * X[k].to_numpy(float) for k in self.weights)
        return self.intercept + self.scale * s


def simplex_nnls(X: pd.DataFrame, y: pd.Series, sample_weight=None) -> SimplexFit:
    A = X.to_numpy(float)
    b = y.to_numpy(float)
    sw = np.ones(len(b)) if sample_weight is None else np.asarray(sample_weight, float)
    Am, bm = np.average(A, axis=0, weights=sw), np.average(b, weights=sw)
    r = np.sqrt(sw)[:, None]
    beta, _ = nnls((A - Am) * r, (b - bm) * r[:, 0])
    c = beta.sum()
    if c <= 0:
        raise ValueError("no component has a positive association with the target")
    w = beta / c
    a = bm - Am @ beta
    return SimplexFit(dict(zip(X.columns, w.round(6))), float(c), float(a), r2(b, a + A @ beta))


def standardised_ols(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    A = (X - X.mean()) / X.std(ddof=0)
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(A)), A.to_numpy()], y.to_numpy(float), rcond=None)
    return dict(zip(X.columns, coef[1:].round(4)))


def shapley_r2(X: pd.DataFrame, y: pd.Series, sample_weight=None, centred: bool = False) -> dict[str, float]:
    """LMG: each component's average marginal R^2 over all orderings (exact over subsets).

    ``sample_weight`` gives weighted least squares; ``centred=True`` means X and y are already (weighted)
    demeaned, e.g. within fixed-effect groups, so no intercept is fitted.
    """
    cols = list(X.columns)
    p = len(cols)
    sw = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
    rw = np.sqrt(sw)
    yv = y.to_numpy(float)
    if not centred:
        yv = yv - np.average(yv, weights=sw)
    sst = float(np.sum(sw * yv ** 2))
    cache: dict[frozenset, float] = {frozenset(): 0.0}

    def fit(sub: frozenset) -> float:
        if sub not in cache:
            A = X[list(sub)].to_numpy(float)
            if not centred:
                A = A - np.average(A, axis=0, weights=sw)
            coef, *_ = np.linalg.lstsq(A * rw[:, None], yv * rw, rcond=None)
            cache[sub] = 1 - float(np.sum(sw * (yv - A @ coef) ** 2)) / sst
        return cache[sub]

    from math import factorial
    out = {}
    for k in cols:
        others = [c for c in cols if c != k]
        tot = 0.0
        for r in range(p):
            for sub in itertools.combinations(others, r):
                s = frozenset(sub)
                wgt = factorial(r) * factorial(p - r - 1) / factorial(p)
                tot += wgt * (fit(s | {k}) - fit(s))
        out[k] = tot
    return out


def lgbm_shap(X: pd.DataFrame, y: pd.Series, Xv: pd.DataFrame, yv: pd.Series, seed: int = 42,
              monotone: dict[str, int] | None = None, groups: dict[str, list[str]] | None = None):
    """Monotone LightGBM: non-linear ceiling (val R^2) and grouped mean |SHAP| shares."""
    import lightgbm as lgb
    mono = [int((monotone or {}).get(c, 0)) for c in X.columns]
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                              monotone_constraints=mono, random_state=seed, verbose=-1)
    model.fit(X, y)
    sv = model.predict(Xv.sample(min(20000, len(Xv)), random_state=seed), pred_contrib=True)[:, :-1]
    imp = dict(zip(X.columns, np.abs(sv).mean(0)))
    if groups:
        imp = {g: sum(imp[c] for c in cs) for g, cs in groups.items()}
    tot = sum(imp.values())
    return model, r2(yv, model.predict(Xv)), {k: v / tot for k, v in imp.items()}


def bayes_dirichlet(X: pd.DataFrame, y: pd.Series, prior_w: dict[str, float], concentration: float = 10.0,
                    n: int = 5000, seed: int = 42, samples: int = 1000) -> np.ndarray:
    """Posterior over w ~ Dirichlet(concentration * prior_w); y ~ Normal(a + c * X w, sigma). NUTS (numpyro)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    A = jnp.asarray(X.to_numpy(float)[idx] / 100.0)
    b = y.to_numpy(float)[idx]
    bm, bs = b.mean(), b.std()
    b = jnp.asarray((b - bm) / bs)
    alpha = jnp.asarray([concentration * prior_w[c] for c in X.columns])

    def model():
        w = numpyro.sample("w", dist.Dirichlet(alpha))
        c = numpyro.sample("c", dist.HalfNormal(10.0))
        a = numpyro.sample("a", dist.Normal(0.0, 5.0))
        s = numpyro.sample("sigma", dist.HalfNormal(1.0))
        numpyro.sample("y", dist.Normal(a + c * (A @ w), s), obs=b)

    mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=samples, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    return np.asarray(mcmc.get_samples()["w"])
