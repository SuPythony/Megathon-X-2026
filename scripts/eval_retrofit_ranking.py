"""Ranking check for improvement potential (plan 4c "Done when": Spearman / NDCG vs the generator's labels).

    uv run python scripts/eval_retrofit_ranking.py [--homes 40] [--n 128]

Per synthetic city home (greentwin.twin.city.synthetic_homes, seeded):
1. Truth: one draw of the twin's prior is the home's hidden physics. Its 2025 monthly meter readings are that draw's
   simulated months plus 2.5% read noise (the generator's meter_noise_relative).
2. Label (the generator's rule ``all_feasible_one_run_null_if_zero_or_unsupported``): 100 x (E_base - E_after) / E_base
   on purchased energy, the truth draw re-run with the engine's all-feasible package as ONE run.
3. Prediction: calibrate on the noisy meters (P2's SVI), then the engine's improvement potential (posterior median).
4. Baseline ranker: rank by total metered kWh (the plan's "rank by total kWh"), and by metered kWh/m2.

Metrics: Spearman rho and NDCG@k (relevance = true potential) for each ranker; P(top decile) per home from the
posterior (threshold = 90th percentile of the predicted medians) with its Brier score vs true top-decile membership.
Caveat (stated in the report): truth and prediction share the same physics, so this measures how well the meters
identify each home's potential, not the physics' error against real homes.
Writes eval/reports/retrofit_ranking/{metrics.json, report.md, homes.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scipy.stats import spearmanr  # noqa: E402

from contracts.models import EnergyUsageMonth, RetrofitAction  # noqa: E402
from greentwin.retrofit.potential import improvement_potential  # noqa: E402
from greentwin.retrofit.rules import assess  # noqa: E402
from greentwin.retrofit.runner import anchored, meter_from_usage, run_packages  # noqa: E402
from greentwin.twin.city import synthetic_homes  # noqa: E402
from greentwin.twin.simulate import prior_samples  # noqa: E402
from greentwin.twin.twin import GreenDigitalTwin  # noqa: E402

OUT = ROOT / "eval" / "reports" / "retrofit_ranking"
NOISE = 0.025


def ndcg(order: np.ndarray, rel: np.ndarray, k: int) -> float:
    rel = np.clip(rel, 0, None)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = float((rel[order[:k]] * disc).sum())
    ideal = float((np.sort(rel)[::-1][:k] * disc).sum())
    return dcg / ideal if ideal > 0 else float("nan")


def meters(pid: str, run, rng) -> list[EnergyUsageMonth]:
    me, mg = np.asarray(run.monthly_electricity_kwh)[0], np.asarray(run.monthly_gas_kwh)[0]
    return [EnergyUsageMonth(property_id=pid, month=f"2025-{m + 1:02d}",
                             electricity_import_kwh=round(float(me[m] * (1 + NOISE * rng.standard_normal())), 1),
                             gas_kwh=round(float(max(mg[m], 0) * (1 + NOISE * rng.standard_normal())), 1))
            for m in range(12)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--homes", type=int, default=40)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    twin = GreenDigitalTwin(n_samples=a.n, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    rows, pct_samples = [], []
    t_all = time.time()
    for i, h in enumerate(synthetic_homes(a.homes, a.seed)):
        prop = h.prop
        t0 = time.time()
        try:
            truth = prior_samples(twin.params_from_property(prop), 1, seed=10_000 + i)
            usage = meters(prop.property_id, run_packages(truth, [[]])[0], rng)
            calib, post = twin.calibrate(prop, usage, [])
            feas = assess(prop)
            meter = meter_from_usage(prop, usage, post.n_samples, a.seed)
            pot = improvement_potential(prop, post, feas, meter, seed=a.seed)
            if pot.pct is None:
                print(f"{prop.property_id}: skipped ({pot.unavailable_reason})", flush=True)
                continue
            pkg = [RetrofitAction(retrofit_id=r) for r in pot.included_retrofits]
            tb, ta = run_packages(truth, [[], pkg])
            b, af = anchored(tb, tb, None), anchored(ta, tb, None)
            true_pct = float(100 * (b.purchased[0] - af.purchased[0]) / b.purchased[0])
            # posterior samples of the potential, for P(top decile)
            pb, pa = run_packages(post, [[], pkg])
            eb, ea = anchored(pb, pb, meter), anchored(pa, pb, meter)
            s = 100 * (eb.purchased - ea.purchased) / eb.purchased
        except Exception as e:  # noqa: BLE001 - one bad home must not stop the batch; it is reported
            print(f"{prop.property_id}: failed ({type(e).__name__}: {e})", flush=True)
            continue
        total = sum((u.electricity_import_kwh or 0) + (u.gas_kwh or 0) for u in usage)
        rows.append({"property_id": prop.property_id, "archetype": h.archetype_id,
                     "fuel": str(getattr(prop.heating.main_fuel, "value", prop.heating.main_fuel)),
                     "floor_area_m2": prop.floor_area_m2, "calibration": calib.method,
                     "pred_median_pct": pot.pct.median, "pred_ci90": [pot.pct.interval(0.9).lo, pot.pct.interval(0.9).hi],
                     "true_pct": true_pct, "metered_kwh": total, "metered_kwh_m2": total / prop.floor_area_m2,
                     "package": pot.included_retrofits, "seconds": round(time.time() - t0, 1)})
        pct_samples.append(np.asarray(s, float))
        print(f"{prop.property_id}: pred {pot.pct.median:.1f}% true {true_pct:.1f}% ({rows[-1]['seconds']}s)", flush=True)

    pred = np.array([r["pred_median_pct"] for r in rows])
    true = np.array([r["true_pct"] for r in rows])
    kwh = np.array([r["metered_kwh"] for r in rows])
    kwh_m2 = np.array([r["metered_kwh_m2"] for r in rows])
    inside = np.mean([r["pred_ci90"][0] <= r["true_pct"] <= r["pred_ci90"][1] for r in rows])
    k = max(1, len(rows) // 10)
    thr = float(np.quantile(pred, 0.9))
    p_top = np.array([float(np.mean(s >= thr)) for s in pct_samples])
    is_top = true >= np.quantile(true, 0.9)
    for r, p in zip(rows, p_top):
        r["p_top_decile"] = round(float(p), 4)

    def ranker(score):
        order = np.argsort(-score)
        return {"spearman": float(spearmanr(score, true).statistic), f"ndcg@{k}": ndcg(order, true, k),
                "ndcg@all": ndcg(order, true, len(rows))}

    metrics = {
        "seed": a.seed, "data_version": "synthetic city homes (greentwin.twin.city) + prior-draw truth + 2.5% meter noise",
        "n_homes": len(rows), "n_samples": a.n, "seconds": round(time.time() - t_all, 1),
        "engine_potential": ranker(pred), "baseline_total_kwh": ranker(kwh), "baseline_kwh_m2": ranker(kwh_m2),
        "potential_mae_pct_points": float(np.mean(np.abs(pred - true))),
        "coverage_90": float(inside),
        "top_decile": {"threshold_pct": thr, "brier": float(np.mean((p_top - is_top) ** 2)),
                       "base_rate": float(is_top.mean())},
        "caveat": "Truth and prediction share the twin's physics: this measures identifiability from meters, not "
                  "physics error against real homes.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (OUT / "homes.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = ["# Improvement-potential ranking vs generator-style labels", "",
          f"{len(rows)} synthetic homes, seed {a.seed}, {a.n} posterior draws. {metrics['caveat']}", "",
          f"| Ranker | Spearman | NDCG@{k} | NDCG@all |", "| --- | --- | --- | --- |"]
    for name, key in (("Engine improvement potential", "engine_potential"), ("Total metered kWh", "baseline_total_kwh"),
                      ("Metered kWh/m2", "baseline_kwh_m2")):
        m = metrics[key]
        md.append(f"| {name} | {m['spearman']:.3f} | {m[f'ndcg@{k}']:.3f} | {m['ndcg@all']:.3f} |")
    md += ["", f"Potential MAE {metrics['potential_mae_pct_points']:.1f} points; 90% interval coverage "
               f"{metrics['coverage_90']:.0%}. P(top decile) Brier {metrics['top_decile']['brier']:.3f} "
               f"(base rate {metrics['top_decile']['base_rate']:.0%})."]
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
