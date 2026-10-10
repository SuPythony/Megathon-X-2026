"""Print the Green Home Score for SH1-SH3 as Markdown (worked examples for docs/score/DECISIONS.md).

    uv run python scripts/score_showcase.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contracts.models import EnergyUsageMonth, Property  # noqa: E402
from greentwin.score import GreenHomeScorer  # noqa: E402
from greentwin.score.stub_energy import energy_from_usage  # noqa: E402

EX = ROOT / "contracts" / "examples"


def main() -> None:
    scorer = GreenHomeScorer(n_samples=4000, seed=42)
    for pid in ("SH1", "SH2", "SH3"):
        prop = Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))
        usage = [EnergyUsageMonth.model_validate(r) for r in json.loads((EX / f"{pid}_usage_2025.json").read_text())]
        e = energy_from_usage(prop, usage)
        r = scorer.score(prop, e)
        d = r.score_distribution
        ci = d.interval(0.9)
        print(f"\n### {pid}: {prop.property_type.value}, {prop.built_form.value}, {prop.construction_age_band}, "
              f"{prop.floor_area_m2:.0f} m2, EPC {prop.epc.current_band} (SAP {prop.epc.sap_score}), "
              f"{len([u for u in usage if u.gas_kwh is not None or u.electricity_import_kwh is not None])} months metered\n")
        print(f"Intensity {e.intensity_kwh_m2.mean:.1f} kWh/m2/yr (90% {e.intensity_kwh_m2.interval(0.9).lo:.1f}-"
              f"{e.intensity_kwh_m2.interval(0.9).hi:.1f}).  "
              f"**Score {r.score}/100, {r.band}** (90% {ci.lo:.1f}-{ci.hi:.1f}; P(Poor or Very poor) = "
              f"{r.p_poor_or_very_poor:.2f})\n")
        print("| Component | Weight | Sub-score | Contribution | Inputs |\n| --- | --- | --- | --- | --- |")
        for c in r.components:
            inp = "; ".join(f"{k}={v}" for k, v in c.inputs_used.items())
            print(f"| {c.name} | {c.weight:.2f} | {c.sub_score:.1f} | {c.contribution:.2f} | {inp} |")
        print(f"| **Total** | 1.00 | | **{sum(c.contribution for c in r.components):.2f} -> {r.score}** | |")
        for alt in r.alternatives:
            aci = alt.score_distribution.interval(0.9)
            print(f"\nLearned-weights companion ({alt.weights_version}): **{alt.score}/100, {alt.band}** "
                  f"(90% {aci.lo:.1f}-{aci.hi:.1f}); contributions: " + ", ".join(
                      f"{c.name} {alt.weights[c.name] * c.sub_score:.2f}" for c in r.components))
        print(f"Peer context: {r.peer_context}")
        print("\nKey issues: " + "; ".join(f"{i.code} ({i.severity})" for i in r.key_issues))
        print("Band probabilities: " + ", ".join(f"{k} {v:.2f}" for k, v in r.band_probabilities.items() if v))


if __name__ == "__main__":
    main()
