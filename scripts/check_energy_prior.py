"""M04 end-to-end health check (monitoring). Exit code 1 if any check fails.

    uv run python scripts/check_energy_prior.py           # human-readable table
    uv run python scripts/check_energy_prior.py --json    # machine-readable

Checks the chain artefact -> served prior -> Bayesian update -> metrics -> report, and that nothing is stale.
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
from contracts.models import EnergyPrior, EnergyUsageMonth, Property  # noqa: E402
from greentwin.models.energy_prior import EnergyPriorModel, property_to_features  # noqa: E402
from greentwin.models.prior_update import update_with_meters  # noqa: E402

RUN = ROOT / "artifacts/m04_energy_prior/v2"
REPORT = ROOT / "eval/reports/m04/report.html"
CARD = ROOT / "docs/M04_ENERGY_PRIOR.md"
SH = ("SH1", "SH2", "SH3")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    results = []

    def check(name, fn):
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as e:  # a crash is a failure, reported with its message
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append({"check": name, "ok": bool(ok), "detail": detail, "seconds": round(time.time() - t0, 2)})

    files = ["model.pkl", "config.json", "metrics.json", "results.json", "bayes.json", "diagnostics.json",
             "showcase_bayes.json", "trials.csv"]
    check("artefacts present", lambda: (all((RUN / f).exists() for f in files),
                                        ("missing: " + ", ".join(f for f in files if not (RUN / f).exists()))
                                        if any(not (RUN / f).exists() for f in files) else f"all {len(files)} present"))
    model = EnergyPriorModel.load(RUN / "model.pkl")
    meta_keys = ("noise", "area_elasticity", "peer_median_kwh_m2")
    check("model metadata (Bayes layer fitted)", lambda: (all(k in model.metadata for k in meta_keys),
                                                         {k: k in model.metadata for k in meta_keys}))
    props = {s: Property.model_validate_json((ROOT / f"contracts/examples/{s}_property.json").read_text()) for s in SH}
    priors = {s: model.energy_prior(p) for s, p in props.items()}

    def contract():
        bad = []
        for s, pr in priors.items():
            EnergyPrior.model_validate(pr.model_dump())
            if pr.provenance.engine != "ebm":
                bad.append(f"{s} engine {pr.provenance.engine}")
            for fuel in ("electricity_kwh_m2", "gas_kwh_m2"):
                d = getattr(pr, fuel)
                ci = d.interval(0.9)
                if not (ci.lo <= d.median <= ci.hi) or d.std < 0:
                    bad.append(f"{s} {fuel} interval")
        if priors["SH3"].gas_kwh_m2.std != 0 or priors["SH3"].gas_kwh_m2.mean != 0:
            bad.append("SH3 gas should be a structural 0")
        return not bad, bad or "3 homes valid, provenance ebm, SH3 gas = 0"
    check("served prior: contract + provenance", contract)

    def additivity():
        errs = []
        for s, p in props.items():
            f = property_to_features(p)
            for fuel in ("electricity", "gas"):
                if fuel == "gas" and f.iloc[0].main_heat_fuel != "gas":
                    continue
                total = sum(c.contribution for c in model._summary(p, f, fuel)[1])
                expect = float(model.predict_point(f, fuel)[0]) * model.area_factor(p, fuel)
                errs.append(abs(total - expect))
        return max(errs) < 0.1, f"max |sum - prediction| = {max(errs):.3f} kWh/m2/yr (rounding to 0.01 per term)"
    check("contributions sum to prediction", additivity)

    def monotone():
        viol = []
        for s in ("SH1", "SH2"):
            base = props[s]
            for fuel, edit in (("gas", lambda q: setattr(q.walls, "insulation", "cavity_filled")),
                               ("gas", lambda q: setattr(q.roof, "loft_insulation_mm", 300)),
                               ("gas", lambda q: setattr(q.epc, "current_band", "B")),
                               ("electricity", lambda q: setattr(q.renewables, "pv_kwp", 4.0))):
                q = base.model_copy(deep=True)
                edit(q)
                if model.predict_point(property_to_features(q), fuel)[0] > model.predict_point(property_to_features(base), fuel)[0] + 1e-9:
                    viol.append(f"{s} {fuel}")
        return not viol, viol or "8 physics edits, 0 violations"
    check("physics monotonicity (showcase edits)", monotone)

    def bayes():
        out = []
        for s, p in props.items():
            usage = [EnergyUsageMonth.model_validate(u) for u in
                     json.loads((ROOT / f"contracts/examples/{s}_usage_2025.json").read_text())]
            r = update_with_meters(model, p, usage, n=8000)
            for fuel, v in r["fuels"].items():
                if v.get("posterior"):
                    if not v["posterior"]["std"] < v["prior"]["std"] or v["ess"] < 200:
                        return False, f"{s} {fuel}: posterior sd {v['posterior']['std']:.1f} vs prior {v['prior']['std']:.1f}, ESS {v['ess']}"
                    out.append(f"{s} {fuel} sd -{v['sd_reduction_pct']:.0f}%")
        return True, ", ".join(out)
    check("Bayesian update narrows prior", bayes)

    m = json.loads((RUN / "metrics.json").read_text())
    dg = json.loads((RUN / "diagnostics.json").read_text())
    bj = json.loads((RUN / "bayes.json").read_text())

    def quality():
        bad = []
        for fuel in ("electricity", "gas"):
            t, iv = m[fuel]["test"], m[fuel]["interval_90"]
            if not t["ebm"]["r2"] > t["ols_onehot"]["r2"]:
                bad.append(f"{fuel}: EBM R2 not above OLS")
            if m[fuel]["ebm_vs"]["ols_onehot"]["mae_gain_ci95"][0] <= 0:
                bad.append(f"{fuel}: EBM vs OLS gain CI includes 0")
            if not 0.88 <= iv["ebm"]["coverage"] <= 0.92:
                bad.append(f"{fuel}: coverage {iv['ebm']['coverage']:.3f}")
            if any(v["violations"] for v in m[fuel]["monotonicity"].values()):
                bad.append(f"{fuel}: monotone violations")
            if dg[fuel]["pit"]["ks_distance_from_uniform"] > 0.01:
                bad.append(f"{fuel}: PIT KS {dg[fuel]['pit']['ks_distance_from_uniform']:.4f}")
            gap = dg[fuel]["by_split"]["train"]["r2"] - dg[fuel]["by_split"]["test"]["r2"]
            if gap > 0.02:
                bad.append(f"{fuel}: overfit gap {gap:.3f}")
            u = bj[fuel]["update_test"]["results"]
            if not u["posterior_1yr"]["mae"] < u["prior_only"]["mae"] or not 0.85 <= u["posterior_1yr"]["coverage_90"] <= 0.95:
                bad.append(f"{fuel}: Bayesian update test")
        detail = {f: f"R2 {m[f]['test']['ebm']['r2']:.3f} (OLS {m[f]['test']['ols_onehot']['r2']:.3f}), "
                     f"cover {m[f]['interval_90']['ebm']['coverage']:.3f}, PIT KS {dg[f]['pit']['ks_distance_from_uniform']:.4f}"
                  for f in ("electricity", "gas")}
        return not bad, bad or detail
    check("model quality thresholds", quality)

    def fresh():
        bad = []
        src = max((RUN / f).stat().st_mtime for f in ("results.json", "bayes.json", "diagnostics.json"))
        for f in (REPORT, CARD):
            if not f.exists() or f.stat().st_mtime < src:
                bad.append(f"{f.relative_to(ROOT)} older than results")
        html = REPORT.read_text() if REPORT.exists() else ""
        if "NaN" in html.split('id="report-data"')[-1][:3_000_000].split("</script>")[0]:
            bad.append("report data contains NaN")
        for s, pr in priors.items():
            ex = json.loads((ROOT / f"contracts/examples/{s}_energy_prior.json").read_text())
            if abs(ex["electricity_kwh_m2"]["median"] - pr.electricity_kwh_m2.median) > 0.01 * max(1, pr.electricity_kwh_m2.median):
                bad.append(f"{s}_energy_prior.json stale")
        return not bad, bad or "report, model card and contract examples match the current model"
    check("outputs up to date", fresh)

    n_fail = sum(not r["ok"] for r in results)
    if a.json:
        print(json.dumps({"ok": n_fail == 0, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, indent=2))
    else:
        for r in results:
            print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['check']:<40} {r['detail']}")
        print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
