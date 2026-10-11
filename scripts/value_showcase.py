"""Worked value examples for SH1-SH3 (docs/value/DECISIONS.md).

The post-retrofit rating comes from Part 2's sap_estimator when it lands; until then each home gets an ILLUSTRATIVE
RatingEstimate (stated below, sd 3 SAP points) so the value step can be shown end to end.

    uv run python scripts/value_showcase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contracts.models import Property, RatingEstimate  # noqa: E402
from contracts.summarise import summarise  # noqa: E402
from greentwin.value.model import HedonicValueModel  # noqa: E402

EX = ROOT / "contracts" / "examples"
ILLUSTRATIVE = {  # target SAP after a package (illustrative, not a sap_estimator output)
    "SH1": (69, "C", "E to C (the brief's own example)"),
    "SH2": (72, "C", "D to C after cavity fill + loft top-up"),
    "SH3": (80, "C", "C to high C after heat pump + controls"),
}


def main() -> None:
    vm = HedonicValueModel(n_samples=4000)
    rng = np.random.default_rng(42)
    for pid, (target, band, what) in ILLUSTRATIVE.items():
        p = Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))
        r = RatingEstimate(sap=summarise(rng.normal(target, 3, 4000), "SAP points"), band=band,
                           anchored_to_recorded_sap=True)
        v = vm.value_impact(p, r)
        u, va = v.uplift_pct, v.value_after_gbp
        print(f"\n### {pid}: {what}; SAP {p.epc.sap_score} -> ~{target}; current value GBP {p.current_value_estimate_gbp:,.0f}")
        print(f"Uplift median {u.median:.2f}% (90% {u.interval(0.9).lo:.2f}% to {u.interval(0.9).hi:.2f}%); "
              f"P(no increase) {u.tail_probabilities[0].probability:.3f}")
        print(f"Value after: GBP {va.interval(0.9).lo:,.0f} to {va.interval(0.9).hi:,.0f} (median {va.median:,.0f})")
        e = v.rating_effect_pct_per_sap_point
        print(f"Rating effect used: median {e.median:.3f}%/SAP point (90% {e.interval(0.9).lo:.3f} to {e.interval(0.9).hi:.3f})")
        print(f"Label: {v.label}. {v.wording}")


if __name__ == "__main__":
    main()
