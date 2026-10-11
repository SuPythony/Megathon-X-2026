"""Train the per-measure relevance EBMs on real EPC certificates + assessor recommendations (plan 4c.3).

    uv run python scripts/epc_recs_extract.py --year 2025 --permille 20     # data (EPC_API_TOKEN)
    uv run python scripts/train_relevance.py [--dir data/interim/epc_recs] [--version v1]

Label: measure X appears in the certificate's recommendations. Split by property hash (70/10/20), seed 42.
Metrics on the held-out test properties, EBM vs the rules-only baseline: AUC per measure, precision@3 per home
(of each home's 3 highest-ranked measures, the share the assessor recommended), and Brier score.
Writes artifacts/relevance/<version>/{model.pkl, metrics.json, config.json}.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interpret.glassbox import ExplainableBoostingClassifier  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from greentwin.models.relevance import (  # noqa: E402
    CAT, FEATURES, MEASURES, TARGETS, catalogue_ids, features_from_epc, rules_baseline, split_of,
)
from greentwin.retrofit import catalogue as cat  # noqa: E402
from greentwin.score import epc_features  # noqa: E402

MIN_POS = 40


def precision_at_k(P: pd.DataFrame, Y: pd.DataFrame, k: int = 3) -> float:
    order = [cat.sequence_position(t) for t in P.columns]  # ties (rules 0/1) broken fabric-first
    tie = np.array([o[0] * 10 + o[1] for o in order], float) * 1e-6
    S = P.to_numpy() - tie
    top = np.argsort(-S, axis=1)[:, :k]
    hits = np.take_along_axis(Y.to_numpy(), top, axis=1)
    has = Y.to_numpy().sum(1) > 0
    return float(hits[has].mean()) if has.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/interim/epc_recs")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0 = time.time()
    certs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.dir}/certificates-*.parquet"))], ignore_index=True)
    recs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.dir}/recommendations-*.parquet"))], ignore_index=True)
    recs.columns = [c.upper() for c in recs.columns]
    key = "CERTIFICATE_NUMBER" if "CERTIFICATE_NUMBER" in recs else "LMK_KEY"
    recs["ids"] = recs["IMPROVEMENT_SUMMARY_TEXT"].map(catalogue_ids)
    unmapped = recs.loc[recs["ids"].map(len) == 0, "IMPROVEMENT_SUMMARY_TEXT"].value_counts().head(15).to_dict()
    long = recs.explode("ids").dropna(subset=["ids"])
    long = long[[key, "ids"]].drop_duplicates().reset_index(drop=True)
    Ylong = pd.crosstab(long[key], long["ids"]).clip(upper=1)

    for c in ("UPRN", "CERTIFICATE_NUMBER", "LOCAL_AUTHORITY", "TENURE", "REPORT_TYPE"):
        if c not in certs:
            certs[c] = None
    feat = epc_features.build(certs)
    X = features_from_epc(feat)
    X.index = certs["CERTIFICATE_NUMBER"].values
    X = X[~X.index.duplicated()]
    # only certificates that have a recommendations row at all (a cert with none is a valid all-negative row only
    # if the recommendations file covers it; sampling keeps both files on the same certificate hash, so it does)
    Y = Ylong.reindex(X.index).fillna(0).astype(int)
    for t in TARGETS:
        if t not in Y:
            Y[t] = 0
    Y = Y[TARGETS]
    prop_key = pd.Series(certs.drop_duplicates("CERTIFICATE_NUMBER").set_index("CERTIFICATE_NUMBER")["UPRN"]).reindex(X.index)
    prop_key = prop_key.fillna(pd.Series(X.index, index=X.index))
    split = prop_key.astype(str).map(lambda k: split_of(k, a.seed))
    for c in CAT:
        X[c] = X[c].astype(str)

    models, per = {}, {}
    tr, te = split == "train", split == "test"
    base = rules_baseline(X)
    P_ebm = pd.DataFrame(index=X.index[te])
    for t in TARGETS:
        y = Y[t]
        pos = int(y[tr].sum())
        if pos < MIN_POS or y[te].nunique() < 2:
            per[t] = {"n_pos_train": pos, "trained": False, "reason": f"fewer than {MIN_POS} positives or one class in test"}
            continue
        mdl = ExplainableBoostingClassifier(feature_names=FEATURES, interactions=0, max_bins=64, outer_bags=4,
                                            random_state=a.seed, n_jobs=1)
        mdl.fit(X.loc[tr, FEATURES], y[tr])
        p = mdl.predict_proba(X.loc[te, FEATURES])[:, 1]
        P_ebm[t] = p
        b = base.loc[te, t]
        per[t] = {"n_pos_train": pos, "prevalence_test": round(float(y[te].mean()), 4), "trained": True,
                  "ebm": {"auc": round(float(roc_auc_score(y[te], p)), 4), "brier": round(float(brier_score_loss(y[te], p)), 4)},
                  "rules": {"auc": round(float(roc_auc_score(y[te], b)), 4), "brier": round(float(brier_score_loss(y[te], b)), 4)}}
        models[t] = mdl
        print(f"{t}: EBM AUC {per[t]['ebm']['auc']} vs rules {per[t]['rules']['auc']} (pos {pos})", flush=True)

    trained = list(P_ebm.columns)
    metrics = {
        "seed": a.seed, "data_version": f"EPC domestic bulk ({a.dir}), certificates + recommendations, hashed sample",
        "n_certificates": int(len(X)), "split_counts": split.value_counts().to_dict(),
        "per_measure": per,
        "precision_at_3": {"ebm": precision_at_k(P_ebm[trained], Y.loc[te, trained]),
                           "rules": precision_at_k(base.loc[te, trained], Y.loc[te, trained])},
        "mean_auc": {"ebm": float(np.mean([per[t]["ebm"]["auc"] for t in trained])),
                     "rules": float(np.mean([per[t]["rules"]["auc"] for t in trained]))},
        "unmapped_summaries_top": unmapped,
        "mapping": [{"pattern": p, "ids": list(ids)} for p, ids in MEASURES],
        "seconds": round(time.time() - t0, 1),
        "notes": ["Rules-only baseline is 0/1, so its AUC is the balanced accuracy of the rule.",
                  "The model ranks feasible options only; rule vetoes always win (zero infeasible items can be "
                  "recommended because the engine filters on feasibility first)."],
    }
    out = ROOT / "artifacts" / "relevance" / a.version
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(models, fh)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"features": FEATURES, "targets": TARGETS, "min_pos": MIN_POS,
                                                 "ebm": {"interactions": 0, "max_bins": 64, "outer_bags": 4}}, indent=1))
    print(json.dumps({k: metrics[k] for k in ("n_certificates", "precision_at_3", "mean_auc")}, indent=1))


if __name__ == "__main__":
    main()
