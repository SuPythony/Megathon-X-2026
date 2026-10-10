"""Run the digital twin end to end on the showcase homes (or one intake) and print what it produced.

    uv run python scripts/run_twin.py              # SH1-SH3
    uv run python scripts/run_twin.py SH2 --n 2000

Path: Property (EPC) -> closest archetype layout -> params_from_property -> weather (synthetic typical year until
Part 0's load_weather is wired) -> prior samples (no calibration yet) -> simulate (physics or stub) -> Distributions.
Metered annual totals are printed alongside as a sanity check only; this run is NOT calibrated to them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts.models import EnergyUsageMonth, Property  # noqa: E402
from contracts.summarise import summarise  # noqa: E402
from greentwin.twin.params import choose_archetype, fuel_config, params_with_sources  # noqa: E402
from greentwin.twin.simulate import prior_samples, simulate  # noqa: E402
from greentwin.twin.weather import resolve_weather  # noqa: E402

EX = Path(__file__).resolve().parents[1] / "contracts" / "examples"


def run(pid: str, n: int) -> None:
    prop = Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))
    params, sources = params_with_sources(prop)
    weather, weather_src = resolve_weather(prop.location.postcode_district)
    run_, engine = simulate(prior_samples(params, n_samples=n), weather, fuel_config(prop))

    usage_path = EX / f"{pid}_usage_2025.json"
    metered = {}
    if usage_path.exists():
        rows = [EnergyUsageMonth.model_validate(r) for r in json.loads(usage_path.read_text(encoding="utf-8"))]
        metered = {"gas": sum(r.gas_kwh or 0 for r in rows), "electricity": sum(r.electricity_import_kwh or 0 for r in rows)}

    print(f"\n=== {pid}: {prop.property_type.value} {prop.built_form.value}, {prop.construction_age_band}, "
          f"{prop.floor_area_m2} m2, {prop.location.postcode_district}")
    print(f"archetype {choose_archetype(prop)} | engine {engine} | weather {weather_src} | {n} prior samples (uncalibrated)")
    print(f"U-values  wall {params.wall_u}  roof {params.roof_u}  floor {params.floor_u}  window {params.window_u}  "
          f"| ach {params.air_changes_per_hour} | eff {params.heating_efficiency} {params.heating_fuel}")
    for label, arr, key in (("gas", run_.gas_kwh, "gas"), ("electricity", run_.electricity_import_kwh, "electricity")):
        d = summarise(np.asarray(arr), "kWh/yr")
        ci90 = next(c for c in d.credible_intervals if abs(c.level - 0.9) < 1e-9)
        m = metered.get(key)
        vs = f" | metered {m:,.0f}" if m is not None else ""
        print(f"{label:<12} {d.headline} {getattr(d, d.headline):>8,.0f}  90% {ci90.lo:>8,.0f} - {ci90.hi:>8,.0f}{vs}")
    hl = {k: float(np.mean(v)) for k, v in run_.element_heat_loss_w_per_k.items()}
    total = sum(hl.values())
    print("heat loss " + "  ".join(f"{k} {v / total:.0%}" for k, v in sorted(hl.items(), key=lambda kv: -kv[1]) if v > 0))
    defaults = sorted(k for k, s in sources.items() if "to_verify" in s or "assumption" in s)
    print(f"from unverified defaults/assumptions: {', '.join(defaults)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("homes", nargs="*", default=["SH1", "SH2", "SH3"])
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    for pid in args.homes:
        run(pid, args.n)


if __name__ == "__main__":
    main()
