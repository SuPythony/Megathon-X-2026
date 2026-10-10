"""M04 energy-intensity prior: tune, fit the model ladder, evaluate on test, log explanations.

    uv run python -m greentwin.data.need                 # once: NEED CSV -> data/processed/need_intensity.parquet
    uv run python scripts/train_energy_prior.py          # this script (see docs/M04_PLAN.md)
    uv run python scripts/report_energy_prior.py         # figures + HTML report from results.json

Splits (stratified by property, assigned once in greentwin/data/need.py): train 60 / tune 10 / cal 10 / test 20.
Training uses unique-feature cells weighted by count (identical optimum for squared loss, ~30x faster).
Writes artifacts/m04_energy_prior/v2/: model.pkl (served EBMs + conformal), config.json, metrics.json,
results.json (everything the report needs), trials.csv, test_explanations_<fuel>.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts.models import Property  # noqa: E402
from greentwin.data.need import DATA_VERSION, FLOOR_BAND_AREA_M2, SPLIT_CUTS, SPLIT_STRATA, TARGET_YEARS  # noqa: E402
from greentwin.models import m04_eval as ev  # noqa: E402
from greentwin.models.energy_prior import (  # noqa: E402
    FEATURE_LABELS, FEATURES, MODEL_ID, MONOTONE, NOMINAL, SEED, STRATA, EnergyPriorModel, prepare_frame,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v2"
IMD_MASK_FRAC = 0.20  # Property rarely carries IMD: teach the models an honest "unknown" level
MODEL_NAMES = {
    "global_mean": "Global mean", "need_median": "NEED median (type x age x region)",
    "ols_linear": "OLS, ordinals as numbers", "ols_onehot": "OLS, one-hot", "ols_log": "OLS, one-hot, log target",
    "ridge": "Ridge (L2), one-hot", "lasso": "Lasso (L1), one-hot", "poly2_ridge": "Polynomial deg-2 + Ridge",
    "lightgbm": "LightGBM (monotone)", "xgboost": "XGBoost (monotone)", "ebm": "EBM (monotone)",
}


# =============================================================================== data
def load(parquet: Path, fuel: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    if fuel == "gas":
        df = df[(df.main_heat_fuel == "gas") & df.gas_kwh.notna()].assign(y=lambda d: d.gas_kwh / d.floor_area_rep_m2)
    else:
        df = df[df.elec_kwh.notna()].assign(y=lambda d: d.elec_kwh / d.floor_area_rep_m2)
    return df.reset_index(drop=True)


def cells(df: pd.DataFrame, fuel: str) -> pd.DataFrame:
    """Collapse rows to unique feature combinations: mean y, mean log y, count (= sample weight)."""
    f = FEATURES[fuel]
    key = df[f].astype("string").fillna("<NA>")
    g = (df.assign(_ly=np.log(df.y)).assign(**{f"_k_{c}": key[c] for c in f})
         .groupby([f"_k_{c}" for c in f], sort=False)
         .agg(**{c: (c, "first") for c in f}, y=("y", "mean"), ly=("_ly", "mean"), w=("y", "size")))
    return g.reset_index(drop=True)


# =============================================================================== models
def lgb_frame(df, fuel):
    X = prepare_frame(df, fuel)
    for c in NOMINAL:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def smear(c: pd.DataFrame, pred_log: np.ndarray) -> float:
    """Exact Duan smearing factor over ROWS from cells: mean_i exp(log y_i - pred) = sum_c n_c ybar_c e^-pred_c / N."""
    return float(np.sum(c.w * c.y * np.exp(-pred_log)) / np.sum(c.w))


class OLS:
    """Weighted least squares on cells == OLS on rows. Variants: one-hot / ordinals-as-numbers / log target."""

    def __init__(self, fuel, kind):
        self.fuel, self.kind = fuel, kind

    def _X(self, df, fit=False):
        X = prepare_frame(df, self.fuel)
        if self.kind == "linear":  # naive: ordinals as numbers, NaN -> indicator + 0
            num = [c for c in X if c not in NOMINAL]
            parts = [X[num].fillna(0).to_numpy(), X[num].isna().to_numpy().astype(float)]
            cat = X[[c for c in X if c in NOMINAL]]
            if fit:
                self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cat)
            return np.hstack(parts + [self.enc.transform(cat)])
        Xs = X.astype("string").fillna("<NA>").astype(str)
        if fit:
            self.enc = OneHotEncoder(handle_unknown="ignore").fit(Xs)
        return self.enc.transform(Xs)

    def fit(self, c):
        target = c.ly if self.kind == "log" else c.y
        self.m = LinearRegression().fit(self._X(c, fit=True), target, sample_weight=c.w)
        if self.kind == "log":
            self.smear = smear(c, self.m.predict(self._X(c)))
        return self

    def predict(self, df):
        p = self.m.predict(self._X(df))
        return np.exp(p) * self.smear if self.kind == "log" else p


class Penalised:
    """Ridge / Lasso on one-hot features, or Ridge on all pairwise one-hot interactions (degree-2 polynomial).

    alpha is chosen on the tune cells by count-weighted MSE, which equals row-level MSE up to a constant.
    """

    ALPHAS = {"ridge": [1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 1e4], "lasso": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2],
              "poly2_ridge": [1, 10, 100, 1000, 1e4, 1e5]}

    def __init__(self, fuel, kind):
        self.fuel, self.kind = fuel, kind

    def _X(self, df, fit=False):
        Xs = prepare_frame(df, self.fuel).astype("string").fillna("<NA>").astype(str)
        if fit:
            self.enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit(Xs)
        X = self.enc.transform(Xs).tocsr()
        if self.kind == "poly2_ridge":
            if fit:
                self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(X)
            X = self.poly.transform(X).tocsr()
        return X

    def _model(self, alpha):
        if self.kind == "lasso":
            return Lasso(alpha=alpha, max_iter=5000, tol=1e-4, selection="cyclic")
        return Ridge(alpha=alpha, solver="sparse_cg" if self.kind == "poly2_ridge" else "auto")

    def fit(self, tr, tu):
        Xtr, Xtu = self._X(tr, fit=True), self._X(tu)
        self.trace = []
        best = (np.inf, None, None)
        for a in self.ALPHAS[self.kind]:
            m = self._model(a).fit(Xtr, tr.y, sample_weight=tr.w)
            mse = float(np.average((tu.y - m.predict(Xtu)) ** 2, weights=tu.w))
            nz = int(np.sum(np.abs(m.coef_) > 1e-9))
            self.trace.append({"alpha": a, "tune_wmse": mse, "nonzero_coefs": nz, "n_coefs": int(m.coef_.size)})
            if mse < best[0]:
                best = (mse, a, m)
        _, self.alpha, self.m = best
        return self

    def predict(self, df):
        return self.m.predict(self._X(df))


def fit_lgb(tr, tu, fuel, p):
    log = p.get("log_target", False)
    params = {"objective": "regression", "learning_rate": p["learning_rate"], "num_leaves": p["num_leaves"],
              "min_sum_hessian_in_leaf": p["min_sum_hessian_in_leaf"], "min_child_samples": 1,
              "feature_fraction": p["feature_fraction"], "lambda_l2": p["lambda_l2"], "cat_smooth": 10,
              "monotone_constraints": [MONOTONE[fuel].get(f, 0) for f in FEATURES[fuel]],
              "monotone_constraints_method": "advanced", "verbose": -1, "seed": SEED, "num_threads": -1}
    dtr = lgb.Dataset(lgb_frame(tr, fuel), tr.ly if log else tr.y, weight=tr.w)
    dtu = lgb.Dataset(lgb_frame(tu, fuel), tu.ly if log else tu.y, weight=tu.w, reference=dtr)
    m = lgb.train(params, dtr, 5000, valid_sets=[dtu], callbacks=[lgb.early_stopping(100, verbose=False)])
    return {"model": m, "log": log, "smear": smear(tr, m.predict(lgb_frame(tr, fuel))) if log else 1.0}


def pred_lgb(b, df, fuel):
    p = b["model"].predict(lgb_frame(df, fuel), num_iteration=b["model"].best_iteration)
    return np.exp(p) * b["smear"] if b["log"] else p


def fit_xgb(tr, tu, fuel, p):
    m = xgb.XGBRegressor(n_estimators=5000, learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                         min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                         reg_lambda=p["reg_lambda"], colsample_bytree=0.9, tree_method="hist",
                         enable_categorical=True, max_cat_to_onehot=1, early_stopping_rounds=100,
                         monotone_constraints=tuple(MONOTONE[fuel].get(f, 0) for f in FEATURES[fuel]),
                         n_jobs=-1, random_state=SEED)
    m.fit(lgb_frame(tr, fuel), tr.y, sample_weight=tr.w, eval_set=[(lgb_frame(tu, fuel), tu.y)],
          sample_weight_eval_set=[tu.w], verbose=False)
    return m


def fit_ebm(tr, fuel, p, n_jobs=-2):
    feats = FEATURES[fuel]
    mono = MONOTONE[fuel]
    ebm = ExplainableBoostingRegressor(
        feature_names=feats, feature_types=["nominal" if f in NOMINAL else "continuous" for f in feats],
        interactions=p["interactions"],
        # Monotone features stay out of pairs, so the monotone guarantee holds for the whole model.
        exclude=[(a, b) for i, a in enumerate(feats) for b in feats[i + 1:] if a in mono or b in mono],
        monotone_constraints=[mono.get(f, 0) for f in feats],
        learning_rate=p["learning_rate"], max_leaves=p["max_leaves"], min_samples_leaf=p["min_samples_leaf"],
        min_hessian=p["min_hessian"], smoothing_rounds=p["smoothing_rounds"], max_bins=1024,
        outer_bags=p.get("outer_bags", 8), max_rounds=10000, early_stopping_rounds=100,
        n_jobs=n_jobs, random_state=SEED)
    ebm.fit(prepare_frame(tr, fuel), tr.y, sample_weight=tr.w)
    return ebm


def wmae(c, p):
    return float(np.average(np.abs(c.y - p), weights=c.w))  # cell-weighted MAE of cell means (tuning score)


SPACES = {
    "lightgbm": {"learning_rate": [0.02, 0.05, 0.1], "num_leaves": [15, 31, 63, 127, 255],
                 "min_sum_hessian_in_leaf": [10, 50, 200, 1000], "feature_fraction": [0.7, 0.85, 1.0],
                 "lambda_l2": [0, 1, 10, 100], "log_target": [False, True]},
    "xgboost": {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [4, 6, 8, 10],
                "min_child_weight": [10, 50, 200, 1000], "subsample": [0.7, 0.85, 1.0], "reg_lambda": [0, 1, 10, 100]},
    "ebm": {"learning_rate": [0.01, 0.02, 0.05], "max_leaves": [2, 3], "min_samples_leaf": [2, 10, 50],
            "min_hessian": [1e-4, 1.0, 50.0], "smoothing_rounds": [0, 200, 500], "interactions": [0, 5, 10, 20]},
}
# Second-round EBM space: the first search picked interactions=20 and max_leaves=3, both at the grid edge.
# 45 = every pair of the 10 non-monotone electricity features (EBM caps at the pairs available, 21 for gas).
SPACES["ebm_wide"] = {"learning_rate": [0.02, 0.05], "max_leaves": [3, 4, 6], "min_samples_leaf": [2, 10],
                      "min_hessian": [1e-4, 1.0], "smoothing_rounds": [200, 500, 1000],
                      "interactions": [20, 30, 45]}


def tune(name, tr_c, tu_c, fuel, n_trials, log_rows, space=None):
    best, best_score = None, np.inf
    for i, p in enumerate(ParameterSampler(SPACES[space or name], n_iter=n_trials, random_state=SEED)):
        t0 = time.time()
        if name == "lightgbm":
            b = fit_lgb(tr_c, tu_c, fuel, p)
            score, extra = wmae(tu_c, pred_lgb(b, tu_c, fuel)), {"rounds": b["model"].best_iteration}
        elif name == "xgboost":
            m = fit_xgb(tr_c, tu_c, fuel, p)
            score, extra = wmae(tu_c, m.predict(lgb_frame(tu_c, fuel))), {"rounds": m.best_iteration}
        else:
            m = fit_ebm(tr_c, fuel, {**p, "outer_bags": 4})  # 4 bags while searching; 8 for the final fit
            score, extra = wmae(tu_c, m.predict(prepare_frame(tu_c, fuel))), {"terms": len(m.term_names_)}
        log_rows.append({"fuel": fuel, "model": name, "space": space or name, "trial": i, "tune_mae": score,
                         "seconds": round(time.time() - t0, 1), **{k: str(v) for k, v in p.items()}, **extra})
        print(f"  [{fuel}] {name} trial {i:02d} tune MAE {score:.3f}  {p}  ({time.time() - t0:.0f}s)", flush=True)
        if score < best_score:
            best, best_score = p, score
    return best, best_score


# =============================================================================== explanations
def ebm_global(ebm, fuel) -> dict:
    """Shape function (bin -> kWh/m2 contribution, +/- 1 sd over bags) for every term + importances."""
    exp = ebm.explain_global()
    terms = []
    for i, name in enumerate(ebm.term_names_):
        d = exp.data(i)
        t = {"term": name, "label": " x ".join(FEATURE_LABELS.get(p, p) for p in name.split(" & ")),
             "importance": float(ebm.term_importances()[i])}
        if " & " not in name:
            names = [str(x) for x in d["names"]]
            scores = list(map(float, d["scores"]))
            lo = list(map(float, d.get("lower_bounds", scores)))
            hi = list(map(float, d.get("upper_bounds", scores)))
            if len(names) == len(scores) + 1:  # continuous: names are bin edges
                names = [f"{float(a):g}-{float(b):g}" for a, b in zip(names[:-1], names[1:])]
            # Missing-value bin (index 0 of term_scores_) is not in explain_global: append it explicitly.
            miss = float(ebm.term_scores_[i][0])
            if ebm.feature_types_in_[ebm.feature_names_in_.index(name)] == "continuous" and miss != 0:
                names, scores, lo, hi = names + ["unknown"], scores + [miss], lo + [miss], hi + [miss]
            t.update(bins=names, scores=scores, lower=lo, upper=hi)
        terms.append(t)
    return {"intercept": float(ebm.intercept_), "terms": sorted(terms, key=lambda t: -t["importance"])}


def ebm_local(ebm, df, fuel) -> pd.DataFrame:
    """Per-home additive contributions (kWh/m2/yr) for every test home; columns = terms + intercept + pred."""
    X = prepare_frame(df, fuel)
    S = ebm.eval_terms(X).astype(np.float32)
    out = pd.DataFrame(S, columns=ebm.term_names_)
    out.insert(0, "intercept", np.float32(ebm.intercept_))
    out["prediction"] = ebm.predict(X).astype(np.float32)
    out.insert(0, "property_id", df.property_id.to_numpy())
    return out


def lgb_shap_importance(b, df, fuel, n=50000) -> dict:
    s = df.sample(min(n, len(df)), random_state=SEED)
    contrib = b["model"].predict(lgb_frame(s, fuel), pred_contrib=True, num_iteration=b["model"].best_iteration)
    imp = np.abs(contrib[:, :-1]).mean(axis=0)
    unit = "log(kWh/m2)" if b["log"] else "kWh/m2/yr"
    return {"unit": unit, "importance": dict(zip(FEATURES[fuel], map(float, imp)))}


def exemplar(ebm, row: pd.DataFrame, fuel, title) -> dict:
    X = prepare_frame(row, fuel)
    s = ebm.eval_terms(X)[0]
    terms = sorted(zip(ebm.term_names_, map(float, s)), key=lambda kv: -abs(kv[1]))
    return {"title": title, "intercept": float(ebm.intercept_), "terms": terms,
            "prediction": float(ebm.predict(X)[0]),
            "actual": float(row.y.iloc[0]) if "y" in row else None,
            "inputs": {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
                       for k, v in X.iloc[0].items()}}


# =============================================================================== main
def run_fuel(fuel, a, rng, trials_log):
    t0 = time.time()
    df = load(Path(a.parquet), fuel)
    tr, tu, ca, te = (df[df.split == s].copy() for s in ("train", "tune", "cal", "test"))
    tr.loc[rng.random(len(tr)) < IMD_MASK_FRAC, "imd_quintile"] = np.nan
    tr_c, tu_c = cells(tr, fuel), cells(tu, fuel)
    print(f"[{fuel}] rows train {len(tr):,} tune {len(tu):,} cal {len(ca):,} test {len(te):,}; "
          f"cells train {len(tr_c):,} tune {len(tu_c):,}", flush=True)

    # ---- hyperparameter search (scored on tune; test untouched)
    best = {}
    fixed = json.loads(Path(a.fixed_params).read_text())["best_params"].get(fuel, {}) if a.fixed_params else {}
    for name, n in (("lightgbm", a.trials_lgb), ("xgboost", a.trials_xgb), ("ebm", a.trials_ebm)):
        if name in fixed and name not in a.retune.split(","):
            best[name] = fixed[name]  # reuse a previous run's search (its trials are carried into trials.csv)
            print(f"[{fuel}] {name}: reusing tuned params {best[name]}", flush=True)
            continue
        space = "ebm_wide" if (name == "ebm" and a.ebm_space == "wide") else None
        best[name], score = tune(name, tr_c, tu_c, fuel, n, trials_log, space)
        print(f"[{fuel}] best {name}: tune MAE {score:.3f} {best[name]}", flush=True)

    # ---- final fits on full train (cells)
    models, preds_te, preds_ca = {}, {}, {}
    gm = float(np.average(tr_c.y, weights=tr_c.w))
    preds_te["global_mean"], preds_ca["global_mean"] = np.full(len(te), gm), np.full(len(ca), gm)
    med = [(c, tr.groupby(c).y.median()) for c in (["prop_type", "age_band", "region"], ["prop_type", "age_band"],
                                                   ["prop_type"])]

    def need_median(d):
        out = pd.Series(np.nan, index=d.index)
        for cols, tab in med:
            j = d[cols].merge(tab.rename("m").reset_index(), on=cols, how="left")["m"].to_numpy()
            out = out.fillna(pd.Series(j, index=d.index))
        return out.fillna(tr.y.median()).to_numpy()
    preds_te["need_median"], preds_ca["need_median"] = need_median(te), need_median(ca)

    for kind in ("linear", "onehot", "log"):
        m = OLS(fuel, kind).fit(tr_c)
        preds_te[f"ols_{kind}"], preds_ca[f"ols_{kind}"] = m.predict(te), m.predict(ca)
    linear_search, penalised = {}, {}
    for kind in ("ridge", "lasso", "poly2_ridge"):
        t1 = time.time()
        m = Penalised(fuel, kind).fit(tr_c, tu_c)
        preds_te[kind], preds_ca[kind] = m.predict(te), m.predict(ca)
        linear_search[kind] = {"chosen_alpha": m.alpha, "trace": m.trace}
        penalised[kind] = m
        print(f"[{fuel}] {kind}: alpha {m.alpha}  ({time.time() - t1:.0f}s)", flush=True)
    b = fit_lgb(tr_c, tu_c, fuel, best["lightgbm"])
    models["lightgbm"] = b
    preds_te["lightgbm"], preds_ca["lightgbm"] = pred_lgb(b, te, fuel), pred_lgb(b, ca, fuel)
    m = fit_xgb(tr_c, tu_c, fuel, best["xgboost"])
    preds_te["xgboost"], preds_ca["xgboost"] = m.predict(lgb_frame(te, fuel)), m.predict(lgb_frame(ca, fuel))
    ebm = fit_ebm(tr_c, fuel, {**best["ebm"], "outer_bags": 8})
    models["ebm"] = ebm
    serve = EnergyPriorModel(models={fuel: ebm}, residual_quantiles={})
    preds_te["ebm"], preds_ca["ebm"] = serve.predict_point(te, fuel), serve.predict_point(ca, fuel)
    print(f"[{fuel}] final fits done {time.time() - t0:.0f}s", flush=True)

    # ---- evaluation on test
    y = te.y.to_numpy()
    res = {"target": f"mean valid annual {fuel} kWh {TARGET_YEARS[0]}-{TARGET_YEARS[-1]} / band floor area"
                     + (" (gas-heated homes; DESNZ weather-corrected)" if fuel == "gas" else " (not weather-corrected)"),
           "n": {"train": len(tr), "train_cells": len(tr_c), "tune": len(tu), "cal": len(ca), "test": len(te)},
           "best_params": {k: {kk: (vv if not isinstance(vv, np.generic) else vv.item()) for kk, vv in v.items()}
                           for k, v in best.items()},
           "test": {k: ev.point_metrics(y, p) for k, p in preds_te.items()}}
    res["bootstrap"] = ev.paired_bootstrap(y, preds_te, reference="ebm", n_boot=a.n_boot)
    res["intervals"], rq = {}, {}
    for k in preds_te:
        res["intervals"][k], rq[k] = ev.conformal_eval(ca, preds_ca[k], te, preds_te[k], fuel)
    # IMD is usually unknown for a real Property: also score test with IMD masked.
    te_noimd = te.assign(imd_quintile=np.nan)
    p_noimd = serve.predict_point(te_noimd, fuel)
    res["ebm_imd_unknown"] = ev.point_metrics(y, p_noimd)
    feats_all = FEATURES[fuel]
    res["ceiling"] = {"all_features": ev.variance_ceiling(df, feats_all),
                      "without_imd_conservatory": ev.variance_ceiling(
                          df, [f for f in feats_all if f not in ("imd_quintile", "conservatory")])}
    res["archetype"] = {k: ev.archetype_r2(te, preds_te[k]) for k in ("need_median", "ols_onehot", "poly2_ridge", "ebm")}
    res["linear_search"] = linear_search
    res["segments"] = {c: ev.segment_table(te, preds_te["ebm"], c)
                       for c in ("prop_type", "age_band", "floor_band", "region", "epc_band")}
    res["calibration_deciles"] = {k: ev.decile_calibration(y, preds_te[k]) for k in ("ols_onehot", "ebm")}
    # Hexbin data for predicted-vs-actual plots (sample keeps the JSON small).
    s = np.random.default_rng(SEED).choice(len(te), min(60000, len(te)), replace=False)
    res["scatter_sample"] = {"y": y[s].round(2).tolist(), "ebm": preds_te["ebm"][s].round(2).tolist(),
                             "ols_onehot": preds_te["ols_onehot"][s].round(2).tolist()}

    # ---- explainability (test set)
    res["ebm_global"] = ebm_global(ebm, fuel)
    res["lgb_shap"] = lgb_shap_importance(b, te, fuel)
    local = ebm_local(ebm, te, fuel)
    term_cols = [c for c in local.columns if c not in ("property_id", "prediction")]
    res["local_additivity_max_abs_err"] = float(np.max(np.abs(local[term_cols].sum(axis=1) - local.prediction)))
    res["test_mean_abs_contribution"] = {c: float(local[c].abs().mean()) for c in ebm.term_names_}
    out_dir = Path(a.out)
    local.to_parquet(out_dir / f"test_explanations_{fuel}.parquet", index=False)
    order = np.argsort(preds_te["ebm"])
    picks = {"low-intensity test home": order[len(order) // 50], "typical test home": order[len(order) // 2],
             "high-intensity test home": order[-len(order) // 50]}
    res["exemplars"] = [exemplar(ebm, te.iloc[[i]], fuel, t) for t, i in picks.items()]
    res["monotonicity"] = monotone_sweep(serve, te, fuel)
    res["monotonicity_poly2_ridge"] = monotone_sweep(None, te, fuel, predict=penalised["poly2_ridge"].predict)
    res["seconds"] = round(time.time() - t0, 1)
    return res, ebm, rq["ebm"]


def monotone_sweep(model, df, fuel, n=50000, predict=None):
    """Set each monotone feature to every value on real test homes; count predictions moving the wrong way."""
    predict = predict or (lambda d: model.predict_point(d, fuel))
    s = df.sample(min(n, len(df)), random_state=SEED)
    out = {}
    for feat, sign in MONOTONE[fuel].items():
        vals = sorted(df[feat].dropna().unique())
        preds = [predict(s.assign(**{feat: v})) for v in vals]
        bad = sum(int(np.sum(sign * (b - a) < -1e-9)) for a, b in zip(preds, preds[1:]))
        out[feat] = {"direction": sign, "checked": len(s) * (len(vals) - 1), "violations": bad}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", default=str(ROOT / "data/processed/need_intensity.parquet"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / MODEL_ID / VERSION))
    ap.add_argument("--trials-lgb", type=int, default=30)
    ap.add_argument("--trials-xgb", type=int, default=20)
    ap.add_argument("--trials-ebm", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--fuels", default="electricity,gas")
    ap.add_argument("--fixed-params", default=None, help="config.json of a previous run: reuse its tuned params")
    ap.add_argument("--retune", default="", help="comma list of models to search again despite --fixed-params")
    ap.add_argument("--ebm-space", choices=["base", "wide"], default="base")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    trials, results, ebms, rqs = [], {}, {}, {}
    prev = []
    if a.fixed_params:  # carry forward the earlier search history of the models we did not re-search
        pt = Path(a.fixed_params).with_name("trials.csv")
        if pt.exists():
            prev = pd.read_csv(pt).to_dict("records")
    for fuel in a.fuels.split(","):
        results[fuel], ebms[fuel], rqs[fuel] = run_fuel(fuel, a, rng, trials)
        pd.DataFrame(prev + trials).to_csv(out / "trials.csv", index=False)

    model = EnergyPriorModel(models=ebms, residual_quantiles=rqs,
                             metadata={"model_id": MODEL_ID, "version": VERSION, "data_version": DATA_VERSION})
    model.save(out / "model.pkl")
    config = {
        "model_id": MODEL_ID, "version": VERSION, "seed": SEED, "data_version": DATA_VERSION,
        "source": "DESNZ NEED anonymised data 2025, 4M-row sample (OGL v3)",
        "target_years": list(TARGET_YEARS), "floor_band_area_m2": FLOOR_BAND_AREA_M2,
        "split": {"strata": list(SPLIT_STRATA), "cumulative_cuts": dict(SPLIT_CUTS), "unit": "property"},
        "features": FEATURES, "monotone": MONOTONE, "conformal_strata": STRATA, "imd_mask_frac": IMD_MASK_FRAC,
        "search_spaces": SPACES, "best_params": {f: r["best_params"] for f, r in results.items()},
        "excluded_features": {"COUNCIL_TAX_BAND": "not in the Property contract (unavailable at prediction time)"},
        "training": "unique feature cells weighted by count (same optimum as rows for squared loss)",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    summary = {"seed": SEED, "data_version": DATA_VERSION, "model_names": MODEL_NAMES}
    for f, r in results.items():
        summary[f] = {"test": r["test"], "interval_90": {k: {kk: v[kk] for kk in ("coverage", "mean_width")}
                                                        for k, v in r["intervals"].items()},
                      "ebm_vs": r["bootstrap"]["delta_vs_reference"], "ceiling": r["ceiling"],
                      "archetype_r2": {k: v["r2"] for k, v in r["archetype"].items()},
                      "ebm_imd_unknown": r["ebm_imd_unknown"], "monotonicity": r["monotonicity"],
                      "best_params": r["best_params"]}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out / "results.json").write_text(json.dumps({"model_names": MODEL_NAMES, **results}, default=str))

    showcase = {}
    for sh in ("SH1", "SH2", "SH3"):
        prop = Property.model_validate_json((ROOT / f"contracts/examples/{sh}_property.json").read_text())
        prior = model.energy_prior(prop)
        (ROOT / f"contracts/examples/{sh}_energy_prior.json").write_text(prior.model_dump_json(indent=2))
        showcase[sh] = json.loads(prior.model_dump_json())
        print(f"{sh}: elec {prior.electricity_kwh_m2.median:.1f} gas {prior.gas_kwh_m2.median:.1f} kWh/m2/yr")
    (out / "showcase.json").write_text(json.dumps(showcase))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
