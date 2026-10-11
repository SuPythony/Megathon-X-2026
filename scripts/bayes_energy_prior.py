"""M04 Bayesian layer: estimate noise + area elasticity from NEED, test the prior->posterior update, run showcase.

    uv run python scripts/bayes_energy_prior.py      (after scripts/train_energy_prior.py)

1. Year-to-year noise per fuel: Student-t fitted to log(y2023 / y2022) of the same homes (train split).
2. Area elasticity beta: OLS of log annual kWh on log band area + the other features (train split).
3. Peer medians (intensity by property type x age band) -> tail probability "above peers" in EnergyPrior.
4. Held-out test of the Bayesian update: observe 2021-2022 -> predict the unseen 2023 reading, compared with
   prior-only and naive persistence (2023 = 2022). Metrics: MAE, 90% coverage, width, CRPS.
5. Showcase homes SH1-SH3: prior, posterior and next-year predictive Distributions from their 2025 meters.
Writes artifacts/m04_energy_prior/v2/{bayes.json, showcase_bayes.json} and updates model.pkl metadata.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts.models import EnergyUsageMonth, Property  # noqa: E402
from contracts.summarise import summarise  # noqa: E402
from greentwin.models.energy_prior import (  # noqa: E402
    MODEL_ID, SEED, STRATA, UNIT, EnergyPriorModel, _stratum_key, property_to_features,
)
from greentwin.models.prior_update import (  # noqa: E402
    GAS_WEATHER_CV, NoiseModel, crps_samples, posterior_weights, prior_samples, resample, update_with_meters,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / MODEL_ID / "v2"
COL = {"electricity": "elec", "gas": "gas"}
N_PRIOR = 4000


def load(fuel: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/processed/need_intensity.parquet")
    c = COL[fuel]
    if fuel == "gas":
        df = df[(df.main_heat_fuel == "gas") & df.gas_kwh.notna()]
    else:
        df = df[df.elec_kwh.notna()]
    return df.assign(y=df[f"{c}_kwh"] / df.floor_area_rep_m2).reset_index(drop=True)


def area_elasticity(tr: pd.DataFrame, fuel: str, n: int = 600_000) -> dict:
    """beta in log kWh = beta * log(area) + controls. Band-level variation identifies it (NEED has bands only)."""
    s = tr.sample(min(n, len(tr)), random_state=SEED)
    ctrl = ["prop_type", "age_band", "region", "epc_band", "imd_quintile", "li_flag", "cwi_flag", "pv_flag"]
    if fuel == "electricity":
        ctrl.append("main_heat_fuel")
    enc = OneHotEncoder(handle_unknown="ignore", drop="first").fit(s[ctrl].astype("string").fillna("NA").astype(str))
    from scipy.sparse import hstack, csr_matrix
    X = hstack([csr_matrix(np.log(s.floor_area_rep_m2.to_numpy())[:, None]),
                enc.transform(s[ctrl].astype("string").fillna("NA").astype(str))]).tocsr()
    y = np.log(s[f"{COL[fuel]}_kwh"].to_numpy())
    m = LinearRegression().fit(X, y)
    beta = float(m.coef_[0])
    # Bootstrap CI for beta (200 resamples of 100k rows) so the estimate carries its own uncertainty.
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(200):
        i = rng.integers(0, X.shape[0], 100_000)
        bs.append(LinearRegression().fit(X[i], y[i]).coef_[0])
    return {"beta": beta, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(s))}


def predictive(theta: np.ndarray, noise: NoiseModel, rng, extra_scale: float = 0.0) -> np.ndarray:
    eps = noise.sample(theta.shape, rng)
    if extra_scale:
        eps = eps + rng.normal(0, extra_scale, theta.shape)
    return theta * np.exp(eps)


def evaluate_update(model: EnergyPriorModel, te: pd.DataFrame, fuel: str, noise: NoiseModel,
                    n_homes: int = 20000) -> dict:
    """Observe 2021 and/or 2022, predict the unseen 2023 reading of held-out homes."""
    c = COL[fuel]
    ok = te[f"{c}_2021"].notna() & te[f"{c}_2022"].notna() & te[f"{c}_2023"].notna()
    s = te[ok].sample(min(n_homes, int(ok.sum())), random_state=SEED).reset_index(drop=True)
    a = s.floor_area_rep_m2.to_numpy()
    y21, y22, y23 = (s[f"{c}_{y}"].to_numpy() / a for y in (2021, 2022, 2023))
    yhat = model.predict_point(s, fuel)
    rq = model.residual_quantiles[fuel]
    rng = np.random.default_rng(SEED)
    # Log-ratio quantiles for the persistence baseline come from TRAIN-free data: the cal split is not used here,
    # so take them from this sample's 2021->2022 change (years before the target year: no peeking at 2023).
    lr = np.log(y22 / y21)
    lo_r, hi_r = np.percentile(lr - np.median(lr), [5, 95])
    rows = {k: {"ae": [], "inside": [], "width": [], "crps": []} for k in
            ("prior_only", "posterior_1yr", "posterior_2yr", "persistence")}
    ess1, ess2, shrink = [], [], []
    for i in range(len(s)):
        q = rq.get(_stratum_key(s.iloc[i], fuel), rq["__all__"])
        th = prior_samples(yhat[i], q, N_PRIOR, rng)
        cands = {"prior_only": th}
        w1, e1 = posterior_weights(th, [y22[i]], noise)
        w2, e2 = posterior_weights(th, [y21[i], y22[i]], noise)
        cands["posterior_1yr"], cands["posterior_2yr"] = resample(th, w1, N_PRIOR, rng), resample(th, w2, N_PRIOR, rng)
        ess1.append(e1)
        ess2.append(e2)
        shrink.append(np.std(np.log(cands["posterior_2yr"])) / np.std(np.log(th)))
        for k, theta in cands.items():
            pred = predictive(theta, noise, rng)
            lo, med, hi = np.percentile(pred, [5, 50, 95])
            r = rows[k]
            r["ae"].append(abs(med - y23[i]))
            r["inside"].append(lo <= y23[i] <= hi)
            r["width"].append(hi - lo)
            r["crps"].append(crps_samples(pred[::4], y23[i]))
        # Persistence: 2023 = 2022, interval from the empirical year-on-year change.
        r = rows["persistence"]
        r["ae"].append(abs(y22[i] - y23[i]))
        lo, hi = y22[i] * np.exp(lo_r), y22[i] * np.exp(hi_r)
        r["inside"].append(lo <= y23[i] <= hi)
        r["width"].append(hi - lo)
        r["crps"].append(np.nan)
    out = {k: {"mae": float(np.mean(v["ae"])), "mdae": float(np.median(v["ae"])),
               "coverage_90": float(np.mean(v["inside"])), "mean_width_90": float(np.mean(v["width"])),
               "crps": None if np.all(np.isnan(v["crps"])) else float(np.nanmean(v["crps"]))}
           for k, v in rows.items()}
    return {"n_homes": int(len(s)), "target": f"{c} intensity in 2023 (unseen)", "results": out,
            "median_ess_1yr": float(np.median(ess1)), "median_ess_2yr": float(np.median(ess2)),
            "posterior_sd_ratio_2yr": float(np.median(shrink))}


def dist_json(samples: np.ndarray, unit: str, tails=()):
    return json.loads(summarise(samples, unit, tails=list(tails), worst_tail="high").model_dump_json())


def showcase(model: EnergyPriorModel) -> dict:
    """SH1-SH3 through the same entry point the system uses (prior_update.update_with_meters)."""
    out = {}
    for sh in ("SH1", "SH2", "SH3"):
        prop = Property.model_validate_json((ROOT / f"contracts/examples/{sh}_property.json").read_text())
        usage = [EnergyUsageMonth.model_validate(u) for u in
                 json.loads((ROOT / f"contracts/examples/{sh}_usage_2025.json").read_text())]
        out[sh] = update_with_meters(model, prop, usage)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--n-homes", type=int, default=20000, help="held-out homes for the update test")
    a = ap.parse_args()
    run = Path(a.run)
    t0 = time.time()
    model = EnergyPriorModel.load(run / "model.pkl")
    out = {"seed": SEED, "assumptions": {
        "gas_weather_cv": {"value": GAS_WEATHER_CV, "status": "assumption",
                           "why": "raw metered gas includes weather; NEED gas is weather-corrected"},
        "imputed_month_relative_sd": {"value": 0.5, "status": "assumption"}}}
    noise, beta, peers = {}, {}, {}
    for fuel in ("electricity", "gas"):
        df = load(fuel)
        tr, te = df[df.split == "train"], df[df.split == "test"]
        c = COL[fuel]
        pair = tr[f"{c}_2022"].notna() & tr[f"{c}_2023"].notna()
        noise[fuel] = NoiseModel.fit(tr.loc[pair, f"{c}_2022"].to_numpy(), tr.loc[pair, f"{c}_2023"].to_numpy())
        beta[fuel] = area_elasticity(tr, fuel)
        peers[fuel] = {f"{k[0]}|{int(k[1])}": float(v) for k, v in tr.groupby(["prop_type", "age_band"]).y.median().items()}
        print(f"[{fuel}] noise t(nu={noise[fuel].nu:.2f}, scale={noise[fuel].scale:.3f}) from {noise[fuel].n_pairs:,} "
              f"pairs; area elasticity {beta[fuel]['beta']:.3f} {beta[fuel]['ci95']}", flush=True)
        model.metadata.setdefault("area_elasticity", {})[fuel] = beta[fuel]["beta"]
        model.metadata.setdefault("peer_median_kwh_m2", {})[fuel] = peers[fuel]
        out[fuel] = {"noise": {"nu": noise[fuel].nu, "scale": noise[fuel].scale, "n_pairs": noise[fuel].n_pairs,
                               "years": [2022, 2023]},
                     "area_elasticity": beta[fuel], "peer_median_kwh_m2": peers[fuel]}
        out[fuel]["update_test"] = evaluate_update(model, te, fuel, noise[fuel], a.n_homes)
        print(json.dumps(out[fuel]["update_test"], indent=1), flush=True)
    model.metadata["noise"] = {f: {"nu": n.nu, "scale": n.scale} for f, n in noise.items()}
    model.save(run / "model.pkl")
    out["showcase"] = showcase(model)
    (run / "bayes.json").write_text(json.dumps({k: v for k, v in out.items() if k != "showcase"}, indent=2))
    (run / "showcase_bayes.json").write_text(json.dumps(out["showcase"], indent=2))
    # Refresh the contract examples with the area-adjusted, tail-annotated priors.
    for sh in ("SH1", "SH2", "SH3"):
        prop = Property.model_validate_json((ROOT / f"contracts/examples/{sh}_property.json").read_text())
        (ROOT / f"contracts/examples/{sh}_energy_prior.json").write_text(model.energy_prior(prop).model_dump_json(indent=2))
    for sh, r in out["showcase"].items():
        for f, v in r["fuels"].items():
            if "prior" in v:
                p, q = v["prior"], v["posterior"]
                print(f"{sh} {f}: observed {v['observed']['intensity_kwh_m2']}  prior median {p['median']:.1f} "
                      f"sd {p['std']:.1f} -> posterior median {q['median']:.1f} sd {q['std']:.1f}  ESS {v['ess']}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
