"""Precompute the retrofit engine's outputs for SH1-SH3 and print them as Markdown.

    uv run python scripts/retrofit_showcase.py [--n 500]

Writes artifacts/retrofit/v1/<SH>_retrofit.json (feasibility, ladder, improvement potential, rating after the
recommended package, Pareto front with the knee for each goal) and metrics.json (seed, data version, Pareto vs greedy baseline).
Samples are each home's Bayesian posterior (P2's SVI calibration on its 2025 meter readings, development weather).
Engine "mock" in the provenance would mean the stand-in physics: never for the demo.
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
from contracts.models import EnergyUsageMonth, Property, RetrofitAction  # noqa: E402
from greentwin.optimise.pareto import (  # noqa: E402
    co2_only_baseline, greedy_baseline, hypervolume, non_dominated, pareto, reference_point,
)
from greentwin.optimise.staging import stage  # noqa: E402
from greentwin.retrofit.engine import RetrofitEngine  # noqa: E402
from greentwin.retrofit.rules import assess  # noqa: E402
from greentwin.score import GreenHomeScorer  # noqa: E402
from greentwin.twin.twin import GreenDigitalTwin  # noqa: E402

EX = ROOT / "contracts" / "examples"
OUT = ROOT / "artifacts" / "retrofit" / "v1"
BUDGET = 10_000
ANNUAL_BUDGET = 3_000  # staging: GBP set aside per year, bill savings reinvested


def rng(d) -> str:
    ci = d.interval(0.9)
    return f"{d.median:,.0f} ({ci.lo:,.0f} to {ci.hi:,.0f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="posterior draws (the twin caps calibrated draws at 256)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    homes = {}
    for pid in ("SH1", "SH2", "SH3"):
        prop = Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))
        usage = [EnergyUsageMonth.model_validate(r) for r in json.loads((EX / f"{pid}_usage_2025.json").read_text())]
        homes[pid] = (prop, usage)
    engine = RetrofitEngine(usage={k: v[1] for k, v in homes.items()}, scorer=GreenHomeScorer(n_samples=1000))
    twin = GreenDigitalTwin(n_samples=args.n, seed=42)
    metrics = {"seed": engine.seed, "n_samples": args.n, "data_version": "showcase examples + synthetic weather",
               "posterior": "per-home SVI calibration (GreenDigitalTwin.calibrate)", "homes": {}}

    for pid, (prop, usage) in homes.items():
        t0 = time.time()
        calib, s = twin.calibrate(prop, usage, [])
        feas = assess(prop)
        base = engine.baseline(prop, s)
        steps = engine.ladder(prop, s, feas, "bills")
        pot = engine.improvement_potential(prop, s, feas)
        rec = [RetrofitAction(retrofit_id=x.retrofit_id) for x in steps if x.verdict in ("recommended", "optional")]
        rating = engine.estimate_rating(prop, s, rec)
        res = pareto(prop, s, feas, engine.meter(prop, args.n), engine.scorer, None, None, "bills", engine.seed)
        pkg, dominated = greedy_baseline(res, BUDGET)
        greedy = {"budget_gbp": BUDGET, "greedy_package": list(pkg), "dominated_by_another_package": dominated,
                  "n_packages": len(res.packages), "n_points_p_on_front_gt_0": len(res.points)}
        c_pkg, c_dom = co2_only_baseline(res, BUDGET)
        within = [i for i in range(len(res.packages)) if res.medians[i, 0] <= BUDGET]
        F = np.column_stack([res.medians[within, 0], -res.medians[within, 1], -res.medians[within, 2], -res.medians[within, 3]])
        front_idx = [within[j] for j in np.flatnonzero(non_dominated(F))]
        idx = {q: i for i, q in enumerate(res.packages)}
        hv = {"front": hypervolume(res, front_idx, BUDGET),
              "greedy": hypervolume(res, [idx[pkg]], BUDGET) if pkg in idx else None,
              "co2_only": hypervolume(res, [idx[c_pkg]], BUDGET) if c_pkg in idx else None}
        baselines = {"co2_only_package": list(c_pkg), "co2_only_dominated": c_dom, "hypervolume": hv,
                     "n_front_in_budget": len(front_idx)}
        refpt = reference_point(res, min_carbon_kg=2000, max_cost_gbp=BUDGET, goal="carbon")
        plan = stage(steps, ANNUAL_BUDGET).as_dict()
        front = [pt.model_dump(mode="json") for pt in res.points if pt.p_on_front >= 0.5 or pt.is_knee]
        knees = {g: list(k) if k is not None else None for g, k in res.knees.items()}
        out = {
            "property_id": pid, "calibration": calib.model_dump(mode="json"), "provenance": engine.provenance(args.n).model_dump(mode="json"),
            "feasibility": feas.model_dump(mode="json"), "baseline": base.model_dump(mode="json"),
            "ladder": [x.model_dump(mode="json") for x in steps],
            "improvement_potential": pot.model_dump(mode="json"),
            "recommended_package": [a.retrofit_id for a in rec],
            "rating_after_recommended": rating.model_dump(mode="json") if rating else None,
            "pareto": {"points": front, "knee_by_goal": knees, "excluded": res.excluded, "baselines": baselines,
                       "reference_point_2t_10k": refpt},
            "staging": plan, "warnings": engine.warnings.get(pid, []),
        }
        (OUT / f"{pid}_retrofit.json").write_text(json.dumps(out), encoding="utf-8")
        metrics["homes"][pid] = {"calibration": calib.method, "n_samples": s.n_samples, "seconds": round(time.time() - t0, 1), "pareto_vs_greedy": greedy,
                                 "pareto_baselines": baselines, "reference_point_2t_10k": refpt.get("package"),
                                 "staging_years": len(plan["stages"]),
                                 "improvement_potential_median_pct": pot.pct.median if pot.pct else None}

        print(f"\n### {pid}: {prop.property_type.value}, {prop.construction_age_band}, SAP {prop.epc.sap_score}\n")
        print(f"Baseline (metered): gas {rng(base.gas_kwh)} kWh/yr, electricity {rng(base.electricity_import_kwh)} kWh/yr\n")
        print("| # | Retrofit | Gas saving kWh/yr | Electricity saving kWh/yr | Bill saving £/yr | CO2e saving kg/yr "
              "| Net cost £ | P(net benefit) | Verdict | Score after |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for x in steps:
            cost = rng(x.cost_gbp) if x.cost_gbp else "no sourced cost"
            print(f"| {x.order} | {x.retrofit_id} | {rng(x.gas_saving_kwh)} | {rng(x.electricity_saving_kwh)} "
                  f"| {rng(x.bill_saving_gbp)} | {rng(x.carbon_saving_kgco2e)} | {cost} | {x.p_net_benefit:.2f} "
                  f"| {x.verdict} | {x.score_after.median:.0f} |")
        ip = rng(pot.pct) if pot.pct else pot.unavailable_reason
        extra = f"; if surveys pass {rng(pot.pct_if_surveys_pass)}" if pot.pct_if_surveys_pass else ""
        print(f"\nImprovement potential: {ip} %{extra}")
        if rating:
            print(f"Rating after recommended + optional ({', '.join(a.retrofit_id for a in rec)}): SAP "
                  f"{rng(rating.sap)}, band {rating.band} {({k: round(v, 2) for k, v in rating.band_probabilities.items() if v})}")
        for goal, k in knees.items():
            if k is not None:
                i = res.packages.index(tuple(k))
                c, co2, bill, sg = res.medians[i]
                print(f"Pareto knee ({goal}): {k}, net cost £{c:,.0f}, bill £{bill:,.0f}/yr, CO2e {co2:,.0f} kg/yr, "
                      f"score +{sg:.0f}, P(on front) {res.p_on_front[i]:.2f}")
        print(f"Greedy £/kWh at £{BUDGET:,}: {greedy['greedy_package']}, dominated: {greedy['dominated_by_another_package']}")
        print(f"CO2-only at £{BUDGET:,}: {list(c_pkg)}; hypervolume front {hv['front']:.3f} vs greedy {hv['greedy']} "
              f"vs CO2-only {hv['co2_only']}")
        print(f"Reference point >= 2 t CO2e, <= £{BUDGET:,}: {refpt.get('package')} (P {refpt.get('p_meets')})")
        print("Staging at £{:,}/yr: ".format(ANNUAL_BUDGET) + "; ".join(f"year {st['year']}: {', '.join(st['retrofit_ids'])}"
                                                                     for st in plan["stages"]))
        print(f"_{time.time() - t0:.1f} s; engine {out['provenance']['engine']}_")

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"n_samples": args.n, "seed": engine.seed, "budget_gbp": BUDGET},
                                                indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
