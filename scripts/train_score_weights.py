"""Learn Green Home Score weights (learned-weights companion score; decisions D-05, D-06 cross-target).

    uv run python scripts/train_score_weights.py --epc-dir data/interim/epc --need data/raw/need/anon2025_50k.csv

Cross-target rule (D-06):
  * EPC register (modelled targets SAP, EIR, -CO2/m2, -primary energy/m2): fit the five ATTRIBUTE components.
    Energy efficiency is excluded (it IS the SAP rating); intensity is not available on the EPC.
  * NEED (real meters): w_EE = share of energy efficiency when predicting metered -kWh/m2;
    w_INT = share of metered intensity when predicting the EPC rating.
  * Final: w_EE, w_INT from NEED; the five attributes share 1 - w_EE - w_INT in proportion to their mean EPC weights.
Outputs artifacts/score_weights/<version>/{weights.json, metrics.json, config.json}.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greentwin.score import epc_features, need_features  # noqa: E402
from greentwin.score import learn as L  # noqa: E402
from greentwin.score.mapping import weights as brief_weights  # noqa: E402

ATTR = ["s_insulation", "s_heating_system", "s_windows_glazing", "s_renewable_energy", "s_smart_energy_management"]
EXTRAS = ["low_energy_lighting_pct", "open_fireplaces", "mech_vent", "dual_tariff"]
TARGETS = {"sap": 1, "eir": 1, "co2_m2": -1, "pe_m2": -1}  # +1 higher is greener, -1 lower is greener
NEED_X_EE = ["s_energy_efficiency", "s_heating_proxy", "s_renewables_proxy", "loft_installed", "cavity_installed"]
NEED_X_INT = ["s_energy_intensity", "s_heating_proxy", "s_renewables_proxy", "loft_installed", "cavity_installed"]
COMP = {"s_energy_efficiency": "energy_efficiency", "s_energy_intensity": "energy_intensity",
        "s_insulation": "insulation", "s_heating_system": "heating_system", "s_windows_glazing": "windows_glazing",
        "s_renewable_energy": "renewable_energy", "s_smart_energy_management": "smart_energy_management"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_epc(epc_dir: str, limit_files: int | None) -> pd.DataFrame:
    # newest year first; keep only the latest certificate per property as we go (memory-safe on millions of rows)
    files = sorted(glob.glob(f"{epc_dir}/certificates-*.parquet"), reverse=True)[: limit_files or None]
    log(f"EPC files: {[Path(f).name for f in files]}")
    keep_cols = ["uprn", "certificate", "lodgement_date", "property_type", "built_form", "age_band", "tfa",
                 "sap", "eir", "co2_m2", "pe_m2", "s_energy_efficiency"] + ATTR + EXTRAS
    seen: set = set()
    feats = []
    for fp in files:
        b = epc_features.latest_per_property(epc_features.build(pd.read_parquet(fp)))[keep_cols]
        key = b["uprn"].fillna("cert:" + b["certificate"].astype(str))
        b = b[~key.isin(seen)]
        seen.update(key[key.index.isin(b.index)])
        for c in ["sap", "eir", "co2_m2", "pe_m2", "tfa", "s_energy_efficiency"] + ATTR + EXTRAS:
            b[c] = b[c].astype("float32")
        feats.append(b)
        log(f"  {Path(fp).name}: +{len(b):,} properties")
    f = pd.concat(feats, ignore_index=True)
    f = f[(f["tfa"].between(15, 1000)) & (f["sap"] >= 1)]
    f = f.dropna(subset=ATTR + list(TARGETS))
    f["key"] = f["uprn"].fillna("cert:" + f["certificate"].astype(str))
    f["split"] = L.split_of(f["key"])
    for t, sign in TARGETS.items():  # orient so higher = greener; winsorise 1-99% (modelled outliers)
        lo, hi = f[t].quantile([0.01, 0.99])
        f[f"y_{t}"] = sign * f[t].clip(lo, hi)
    return f


def nnls_boot(X, y, B, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        i = rng.integers(0, len(X), len(X))
        out.append(list(L.simplex_nnls(X.iloc[i], y.iloc[i]).weights.values()))
    return np.array(out)


def fixed_weight_r2(w: dict[str, float], Xtr, ytr, Xte, yte) -> float:
    """R^2 of target ~ a + c * score(w): only a and c are fitted (weights fixed)."""
    s_tr = sum(w[k] * Xtr[k] for k in w)
    s_te = sum(w[k] * Xte[k] for k in w)
    c, a = np.polyfit(s_tr, ytr, 1)
    return L.r2(yte, a + c * s_te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epc-dir", default="data/interim/epc")
    ap.add_argument("--need", default="data/raw/need/anon2025_4million.zip")
    ap.add_argument("--limit-files", type=int)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cell", type=int, default=30)
    a = ap.parse_args()
    bw = brief_weights()
    metrics: dict = {"seed": a.seed}

    # ------------------------------------------------------------------ EPC: attribute weights
    f = load_epc(a.epc_dir, a.limit_files)
    tr, va, te = (f[f.split == s] for s in ("train", "val", "test"))
    log(f"EPC properties: {len(f):,} (train {len(tr):,} / val {len(va):,} / test {len(te):,})")
    metrics["epc"] = {"n": len(f), "n_train": len(tr), "n_val": len(va), "n_test": len(te), "targets": {}}
    brief_attr = {k: bw[COMP[k]] for k in ATTR}
    tot = sum(brief_attr.values())
    brief_attr = {k: v / tot for k, v in brief_attr.items()}
    equal = {k: 1 / len(ATTR) for k in ATTR}
    attr_w = []
    for t in TARGETS:
        y = f"y_{t}"
        fit = L.simplex_nnls(tr[ATTR], tr[y])
        boot = nnls_boot(tr[ATTR].sample(min(200_000, len(tr)), random_state=a.seed),
                         tr[y].sample(min(200_000, len(tr)), random_state=a.seed), a.boot, a.seed)
        samp = tr.sample(min(200_000, len(tr)), random_state=a.seed)
        shap_r2 = L.shapley_r2(samp[ATTR], samp[y])
        _, gbm_r2, gbm_imp = L.lgbm_shap(
            samp[ATTR + EXTRAS].fillna(0), samp[y], te[ATTR + EXTRAS].fillna(0), te[y], a.seed,
            monotone={k: 1 for k in ATTR})
        res = {
            "nnls_weights": fit.weights,
            "nnls_weights_boot_p5": dict(zip(ATTR, np.percentile(boot, 5, 0).round(4))),
            "nnls_weights_boot_p95": dict(zip(ATTR, np.percentile(boot, 95, 0).round(4))),
            "scale_target_units_per_point": round(fit.scale, 4),
            "r2_test_learned": round(L.r2(te[y], fit.predict(te[ATTR])), 4),
            "r2_test_brief_weights": round(fixed_weight_r2(brief_attr, tr[ATTR], tr[y], te[ATTR], te[y]), 4),
            "r2_test_equal_weights": round(fixed_weight_r2(equal, tr[ATTR], tr[y], te[ATTR], te[y]), 4),
            "r2_test_lgbm_with_extras": round(gbm_r2, 4),
            "standardised_ols": L.standardised_ols(samp[ATTR], samp[y]),
            "shapley_r2": {k: round(v, 4) for k, v in shap_r2.items()},
            "shapley_r2_total": round(sum(shap_r2.values()), 4),
            "lgbm_shap_share": {k: round(v, 4) for k, v in gbm_imp.items()},
        }
        metrics["epc"]["targets"][t] = res
        attr_w.append(list(fit.weights.values()))
        log(t, json.dumps({k: res[k] for k in ("nnls_weights", "r2_test_learned", "r2_test_brief_weights",
                                                 "r2_test_equal_weights", "r2_test_lgbm_with_extras")}))
    attr_mean = np.mean(attr_w, 0)

    # Bayesian check on the SAP target: how far does the data move the brief's (renormalised) weights?
    log("Bayesian Dirichlet (NUTS) on SAP target ...")
    post = L.bayes_dirichlet(tr[ATTR], tr["y_sap"], brief_attr, seed=a.seed)
    metrics["epc"]["bayes_sap"] = {
        "prior": "Dirichlet(10 x brief weights renormalised over the 5 attributes)",
        "mean": dict(zip(ATTR, post.mean(0).round(4))),
        "p5": dict(zip(ATTR, np.percentile(post, 5, 0).round(4))),
        "p95": dict(zip(ATTR, np.percentile(post, 95, 0).round(4))),
    }

    # ------------------------------------------------------------------ NEED per home (diagnostic only)
    nd_all = need_features.load(a.need)
    log(f"NEED rows: {len(nd_all):,}")
    nd = nd_all.dropna(subset=["s_energy_intensity", "s_energy_efficiency"])
    for c in ("loft_installed", "cavity_installed"):
        nd[c] = nd[c] * 100.0  # 0/100 so all inputs share the 0-100 scale
    nd["split"] = L.split_of(nd["row_id"])
    lo, hi = nd["intensity_kwh_m2"].quantile([0.01, 0.99])
    nd["y_neg_intensity"] = -nd["intensity_kwh_m2"].clip(lo, hi)
    ntr, nte = nd[nd.split == "train"], nd[nd.split == "test"]
    fit_ee = L.simplex_nnls(ntr[NEED_X_EE], ntr["y_neg_intensity"])
    fit_int = L.simplex_nnls(ntr[NEED_X_INT], ntr["sap_mid"])
    b_ee = nnls_boot(ntr[NEED_X_EE], ntr["y_neg_intensity"], a.boot, a.seed)[:, 0]
    b_int = nnls_boot(ntr[NEED_X_INT], ntr["sap_mid"], a.boot, a.seed)[:, 0]
    metrics["need_per_home_diagnostic"] = {
        "note": "Per-home NEED fits: attributes are coarse flags and behaviour dominates, so these are NOT used "
                "for the weights (D-18); kept to show how little single-home meter data is explained.",
        "n": len(nd), "n_train": len(ntr), "n_test": len(nte),
        "ee_on_metered": {"weights": fit_ee.weights, "r2_test": round(L.r2(nte["y_neg_intensity"], fit_ee.predict(nte[NEED_X_EE])), 4)},
        "int_on_sap": {"weights": fit_int.weights, "r2_test": round(L.r2(nte["sap_mid"], fit_int.predict(nte[NEED_X_INT])), 4)},
    }
    log("NEED per home", json.dumps({k: metrics["need_per_home_diagnostic"][k]["r2_test"] for k in ("ee_on_metered", "int_on_sap")}))

    # ------------------------------------------------------------------ archetype cells: EE and INT vs attributes (D-18)
    from greentwin.score import archetypes as AR
    from greentwin.score.mapping import intensity_sub
    ec = AR.epc_cells(f, ATTR + ["s_energy_efficiency"])
    nc = AR.need_cells(nd_all)
    cells = ec.join(nc, how="inner").reset_index()
    cells = cells[(cells.n_epc >= a.min_cell) & (cells.n_need >= a.min_cell)].copy()
    cells["s_energy_intensity"] = intensity_sub(cells["metered_kwh_m2_median"])
    cells["y_neg_metered"] = -cells["metered_kwh_m2_median"]
    cells["y_sap"] = cells["s_energy_efficiency"]
    cw = np.minimum(cells.n_epc, cells.n_need)
    log(f"archetype cells: {len(cells)} (min {a.min_cell} homes in each source)")

    def cell_fit(xcols, ycol, idx=None):
        """Weighted simplex NNLS with floor-area-band fixed effects: kWh/m2 falls with home size and NEED's
        area is banded, so comparisons are made WITHIN each area band (D-18)."""
        c = cells if idx is None else cells.iloc[idx]
        w = np.minimum(c.n_epc, c.n_need).astype(float)
        cols = xcols + [ycol]
        means = (c[cols].mul(w, axis=0).groupby(c["cell_area"]).transform("sum")
                 .div(w.groupby(c["cell_area"]).transform("sum"), axis=0))
        d = c[cols] - means
        return L.simplex_nnls(d[xcols], d[ycol], sample_weight=w)

    def within_r2(fit, xcols, ycol):
        w = np.minimum(cells.n_epc, cells.n_need).astype(float)
        cols = xcols + [ycol]
        means = (cells[cols].mul(w, axis=0).groupby(cells["cell_area"]).transform("sum")
                 .div(w.groupby(cells["cell_area"]).transform("sum"), axis=0))
        d = cells[cols] - means
        pred = fit.scale * sum(fit.weights[k] * d[k] for k in xcols)
        return round(1 - float(np.sum(w * (d[ycol] - pred) ** 2) / np.sum(w * d[ycol] ** 2)), 4)

    def lmg_ratio(head, ycol, idx=None):
        """Shapley/LMG R^2 share of the outcome component vs the whole attribute block (within area band)."""
        c = cells if idx is None else cells.iloc[idx].reset_index(drop=True)
        w = np.minimum(c.n_epc, c.n_need).astype(float)
        cols = [head] + ATTR + [ycol]
        means = (c[cols].mul(w, axis=0).groupby(c["cell_area"]).transform("sum")
                 .div(w.groupby(c["cell_area"]).transform("sum"), axis=0))
        d = c[cols] - means
        sh = L.shapley_r2(d[[head] + ATTR], d[ycol], sample_weight=w, centred=True)
        return sh[head] / max(sum(sh[k] for k in ATTR), 1e-9), sh

    fa = cell_fit(["s_energy_efficiency"] + ATTR, "y_neg_metered")
    fb = cell_fit(["s_energy_intensity"] + ATTR, "y_sap")
    r_ee, sh_a = lmg_ratio("s_energy_efficiency", "y_neg_metered")
    r_int, sh_b = lmg_ratio("s_energy_intensity", "y_sap")
    rng = np.random.default_rng(a.seed)
    boot_ratio = []
    for _ in range(a.boot * 5):
        i = rng.integers(0, len(cells), len(cells))
        boot_ratio.append((lmg_ratio("s_energy_efficiency", "y_neg_metered", i)[0],
                           lmg_ratio("s_energy_intensity", "y_sap", i)[0]))
    boot_ratio = np.array(boot_ratio)
    metrics["cells"] = {
        "n_cells": len(cells), "homes_epc": int(cells.n_epc.sum()), "homes_need": int(cells.n_need.sum()),
        "fit_ee_on_metered": {"nnls_weights": fa.weights, "lmg_r2": {k: round(v, 4) for k, v in sh_a.items()},
                              "r2_within_area_band": within_r2(fa, ["s_energy_efficiency"] + ATTR, "y_neg_metered")},
        "fit_int_on_sap": {"nnls_weights": fb.weights, "lmg_r2": {k: round(v, 4) for k, v in sh_b.items()},
                           "r2_within_area_band": within_r2(fb, ["s_energy_intensity"] + ATTR, "y_sap")},
        "ratio_rule": "LMG R^2 of the outcome component / summed LMG R^2 of the five attributes (D-18)",
        "ratio_ee_to_attributes": round(r_ee, 4), "ratio_int_to_attributes": round(r_int, 4),
        "ratio_ee_boot_p5_p95": np.percentile(boot_ratio[:, 0], [5, 95]).round(4).tolist(),
        "ratio_int_boot_p5_p95": np.percentile(boot_ratio[:, 1], [5, 95]).round(4).tolist(),
        "corr_metered_vs_epc_sap": round(float(np.corrcoef(cells.metered_kwh_m2_median, cells.s_energy_efficiency)[0, 1]), 4),
    }
    cells.to_csv(ROOT / "artifacts" / "score_cells.csv", index=False)
    log("cells", json.dumps({k: metrics["cells"][k] for k in ("fit_ee_on_metered", "fit_int_on_sap",
                                                               "ratio_ee_to_attributes", "ratio_int_to_attributes")}))

    # ------------------------------------------------------------------ combine (D-06 + D-18)
    # each outcome component is weighted relative to the attribute block: w_EE = r_EE * T, w_INT = r_INT * T,
    # attributes share T in proportion to their per-home EPC weights; T = 1 / (1 + r_EE + r_INT)
    T = 1 / (1 + r_ee + r_int)
    final = {"energy_efficiency": r_ee * T, "energy_intensity": r_int * T}
    final.update({COMP[k]: float(T * v / attr_mean.sum()) for k, v in zip(ATTR, attr_mean)})
    bt = 1 / (1 + boot_ratio[:, 0] + boot_ratio[:, 1])
    metrics["final_weights_boot_p5_p95"] = {
        "energy_efficiency": np.percentile(boot_ratio[:, 0] * bt, [5, 95]).round(4).tolist(),
        "energy_intensity": np.percentile(boot_ratio[:, 1] * bt, [5, 95]).round(4).tolist(),
        "attribute_block": np.percentile(bt, [5, 95]).round(4).tolist(),
    }
    for k, v in zip(ATTR, attr_mean):  # block uncertainty x the attribute's share (stage-1 spread is reported separately)
        metrics["final_weights_boot_p5_p95"][COMP[k]] = np.percentile(bt * v / attr_mean.sum(), [5, 95]).round(4).tolist()
    final = {k: round(v, 4) for k, v in final.items()}
    drift = round(1 - sum(final.values()), 4)
    final["energy_efficiency"] = round(final["energy_efficiency"] + drift, 4)  # exact sum to 1 after rounding
    metrics["final_weights"] = final
    metrics["brief_weights"] = bw
    log("FINAL learned weights:", final)

    # ------------------------------------------------------------------ how different are the two scores? (EPC test homes)
    # intensity is not on the EPC: use the archetype cell's median METERED intensity (labelled imputation)
    from greentwin.score.mapping import classify
    tt = te.assign(cell_type=[AR.epc_need_type(p, b) for p, b in zip(te["property_type"], te["built_form"])],
                   cell_age=te["age_band"].map(AR.epc_need_age), cell_area=AR.area_band(te["tfa"]))
    tt = tt.merge(cells[["cell_type", "cell_age", "cell_area", "s_energy_intensity"]],
                  on=["cell_type", "cell_age", "cell_area"], how="inner")
    sub = {COMP[k]: tt[k] for k in ["s_energy_efficiency", "s_energy_intensity"] + ATTR}
    sb = np.floor(sum(bw[k] * sub[k] for k in bw) + 0.5)
    sl = np.floor(sum(final[k] * sub[k] for k in final) + 0.5)
    bb, bl = sb.astype(int).map(classify), sl.astype(int).map(classify)
    metrics["brief_vs_learned_on_epc_test"] = {
        "n": len(tt), "intensity": "archetype-cell median metered (imputed)",
        "spearman": round(float(pd.Series(sb).corr(pd.Series(sl), method="spearman")), 4),
        "band_agreement": round(float((bb == bl).mean()), 4),
        "mean_abs_diff_points": round(float(np.abs(sb - sl).mean()), 2),
        "mean_brief": round(float(sb.mean()), 2), "mean_learned": round(float(sl.mean()), 2),
        "share_poor_or_very_poor_brief": round(float(bb.isin(["Poor", "Very poor"]).mean()), 4),
        "share_poor_or_very_poor_learned": round(float(bl.isin(["Poor", "Very poor"]).mean()), 4),
        "band_crosstab": pd.crosstab(bb, bl).to_dict(),
    }
    log("brief vs learned", json.dumps({k: v for k, v in metrics["brief_vs_learned_on_epc_test"].items()
                                        if k != "band_crosstab"}))

    out = ROOT / "artifacts" / "score_weights" / a.version
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.json").write_text(json.dumps({"weights": final, "method": "cross-target simplex NNLS (D-06)",
                                                  "version": a.version}, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    log("wrote", out)


if __name__ == "__main__":
    main()
