"""Bayesian hierarchical hedonic price model (Part 4, decision V-02).

log(price adjusted to the reference month by UK HPI) ~ StudentT(nu=5, mu, sigma_type)
mu = a_region + a_LA + a_LSOA                (nested, partially pooled location)
   + b_typeform + b_ageband + b_year + b_lease + b_new          (fixed effects, weak priors)
   + c1 log(floor area) + c2 rooms + c3 EPC age
   + beta[region x type] * (SAP - 60) / 10                        (rating effect, partially pooled; V-03)

beta has a weak prior N(0, 0.1) per 10 SAP points (i.e. +-10% per 10 points), so the comparison with the published
premiums is not driven by the prior. Fitted with minibatch SVI (AutoNormal), seed 42.
uplift for a change dSAP: exp(beta * dSAP / 10) - 1, per posterior sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATS = ["typeform", "age_band", "year"]
SAP_CENTRE, SAP_SCALE = 60.0, 10.0


@dataclass
class Encoding:
    levels: dict[str, list] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "Encoding":
        for c in cols:
            self.levels[c] = sorted(df[c].dropna().astype(str).unique().tolist())
        return self

    def codes(self, s: pd.Series, col: str) -> np.ndarray:
        """Integer codes; unseen or missing -> len(levels) (an extra 'unknown' slot with effect 0)."""
        lut = {v: i for i, v in enumerate(self.levels[col])}
        return s.astype(str).map(lut).fillna(len(lut)).astype(int).to_numpy()


def add_design_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["typeform"] = d["ptype"].astype(str) + "|" + d["BUILT_FORM"].fillna("unknown").astype(str)
    d["year"] = d["year"].astype(str)
    d["rt"] = d["rgn"].astype(str) + "|" + d["ptype"].astype(str)
    return d


def arrays(d: pd.DataFrame, enc: Encoding) -> dict[str, np.ndarray]:
    la_of_lsoa = None  # filled by the fitter
    out = {k: enc.codes(d[k], k) for k in ["rgn", "laua", "lsoa21", "rt", *CATS]}
    # V-12: a sale year after the training data (e.g. 2024, or 'today' in the product) takes the LATEST fitted
    # year's effect: prices are HPI-adjusted to the reference month, so the newest year is the closest market.
    yl = enc.levels["year"]
    late = d["year"].astype(str).to_numpy() > yl[-1]
    out["year"] = np.where(late, len(yl) - 1, out["year"])
    out["lease"] = d["leasehold"].astype(float).to_numpy()
    out["new"] = d["new_build"].astype(float).to_numpy()
    out["log_tfa"] = (d["log_tfa"] - 4.5).to_numpy(float)
    out["rooms"] = (d["rooms"].fillna(d["rooms"].median()).clip(1, 12) - 5).to_numpy(float)
    out["epc_age"] = d["epc_age_years"].to_numpy(float) - 3
    out["sap"] = ((d["sap"] - SAP_CENTRE) / SAP_SCALE).to_numpy(float)
    out["ptype"] = enc.codes(d["ptype"], "ptype")
    return out


def model(data, y=None, n_obs=None, sizes=None, batch=None):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    S = sizes
    s_rg = numpyro.sample("s_region", dist.HalfNormal(0.5))
    s_la = numpyro.sample("s_la", dist.HalfNormal(0.5))
    s_ls = numpyro.sample("s_lsoa", dist.HalfNormal(0.5))
    a0 = numpyro.sample("a0", dist.Normal(12.5, 2.0))
    a_rg = numpyro.sample("a_region", dist.Normal(0, s_rg).expand([S["rgn"] + 1]).to_event(1))
    a_la = numpyro.sample("a_la", dist.Normal(0, s_la).expand([S["laua"] + 1]).to_event(1))
    a_ls = numpyro.sample("a_lsoa", dist.Normal(0, s_ls).expand([S["lsoa21"] + 1]).to_event(1))
    b = {c: numpyro.sample(f"b_{c}", dist.Normal(0, 1.0).expand([S[c] + 1]).to_event(1)) for c in CATS}
    c_lease = numpyro.sample("c_lease", dist.Normal(0, 1.0))
    c_new = numpyro.sample("c_new", dist.Normal(0, 1.0))
    c_tfa = numpyro.sample("c_log_tfa", dist.Normal(0.7, 0.5))
    c_rooms = numpyro.sample("c_rooms", dist.Normal(0, 0.5))
    c_age = numpyro.sample("c_epc_age", dist.Normal(0, 0.1))
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 0.1))
    tau_b = numpyro.sample("tau_beta", dist.HalfNormal(0.05))
    beta = numpyro.sample("beta_rt", dist.Normal(beta0, tau_b).expand([S["rt"] + 1]).to_event(1))
    sigma = numpyro.sample("sigma_type", dist.HalfNormal(0.5).expand([S["ptype"] + 1]).to_event(1))

    with numpyro.plate("obs", n_obs, subsample_size=batch) as idx:
        g = {k: v[idx] for k, v in data.items()}
        mu = (a0 + a_rg[g["rgn"]] + a_la[g["laua"]] + a_ls[g["lsoa21"]]
              + sum(b[c][g[c]] for c in CATS)
              + c_lease * g["lease"] + c_new * g["new"] + c_tfa * g["log_tfa"] + c_rooms * g["rooms"]
              + c_age * g["epc_age"] + beta[g["rt"]] * g["sap"])
        obs = None if y is None else y[idx]
        numpyro.sample("y", dist.StudentT(5.0, mu, sigma[g["ptype"]]), obs=obs)


@dataclass
class HedonicFit:
    enc: Encoding
    sizes: dict[str, int]
    params: dict
    guide: object
    losses: np.ndarray

    def posterior(self, n: int = 1000, seed: int = 42) -> dict[str, np.ndarray]:
        import jax
        return {k: np.asarray(v) for k, v in
                self.guide.sample_posterior(jax.random.PRNGKey(seed), self.params, sample_shape=(n,)).items()}

    def predict_samples(self, d: pd.DataFrame, post: dict, rng: np.random.Generator) -> np.ndarray:
        """Posterior predictive draws of log price (samples x rows), incl. StudentT observation noise."""
        g = arrays(add_design_cols(d), self.enc)
        mu = (post["a0"][:, None] + post["a_region"][:, g["rgn"]] + post["a_la"][:, g["laua"]]
              + post["a_lsoa"][:, g["lsoa21"]] + sum(post[f"b_{c}"][:, g[c]] for c in CATS)
              + post["c_lease"][:, None] * g["lease"] + post["c_new"][:, None] * g["new"]
              + post["c_log_tfa"][:, None] * g["log_tfa"] + post["c_rooms"][:, None] * g["rooms"]
              + post["c_epc_age"][:, None] * g["epc_age"] + post["beta_rt"][:, g["rt"]] * g["sap"])
        sig = post["sigma_type"][:, g["ptype"]]
        return mu + sig * rng.standard_t(5.0, size=mu.shape)


def fit(train: pd.DataFrame, steps: int = 8000, batch: int = 20000, lr: float = 0.01, seed: int = 42) -> HedonicFit:
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam

    d = add_design_cols(train)
    enc = Encoding().fit(d, ["rgn", "laua", "lsoa21", "rt", "ptype", *CATS])
    sizes = {k: len(v) for k, v in enc.levels.items()}
    data = {k: jnp.asarray(v) for k, v in arrays(d, enc).items()}
    y = jnp.asarray(d["log_price_adj"].to_numpy(float))
    guide = AutoNormal(model, init_scale=0.05)
    svi = SVI(model, guide, Adam(lr), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), steps, data, y=y, n_obs=len(d), sizes=sizes, batch=batch,
                  progress_bar=False)
    return HedonicFit(enc, sizes, res.params, guide, np.asarray(res.losses))
