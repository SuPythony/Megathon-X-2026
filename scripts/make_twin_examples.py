"""Write the twin's example outputs for the showcase homes SH1-SH3:

- greentwin/twin/examples/physics_inputs_<SH>.json : the physics inputs package (for the physics/calibration team)
- greentwin/twin/examples/<SH>.scene.json          : room scene with per-room energy, retrofit options and
                                                     precomputed single-retrofit scenarios (viewer offline fallback)
- greentwin/twin/examples/physics_inputs.schema.json

    uv run python scripts/make_twin_examples.py

Household answers and appliances (greentwin/twin/showcase.py) are illustrative examples, not real customer data.
Numbers come from the stub physics until greentwin/twin/physics.py exists; scenes say so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from greentwin.twin.inputs import PhysicsInputs, build_physics_inputs  # noqa: E402
from greentwin.twin.rooms import energy_overlay  # noqa: E402
from greentwin.twin.scenarios import run_scenario  # noqa: E402
from greentwin.twin.scene import build_scene, retrofit_options  # noqa: E402
from greentwin.twin.showcase import HOMES, load_showcase  # noqa: E402
from greentwin.twin.simulate import prior_samples  # noqa: E402
from greentwin.twin.twin import GreenDigitalTwin  # noqa: E402

OUT = ROOT / "greentwin" / "twin" / "examples"
N = 1000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    twin = GreenDigitalTwin(n_samples=N)
    for pid in HOMES:
        sc = load_showcase(pid)
        pkg = build_physics_inputs(sc.prop, layout=sc.layout, appliances=sc.appliances, intake=sc.intake, usage=sc.usage)
        (OUT / f"physics_inputs_{pid}.json").write_text(pkg.model_dump_json(indent=2), encoding="utf-8")

        samples = prior_samples(pkg.params, N, priors=pkg.prior_table())
        alloc = twin.room_energy(sc.prop, sc.layout, samples, sc.appliances)
        scene = build_scene(sc.layout, pid, sc.appliances)
        scene.energy = energy_overlay(alloc, calibrated=False, weather_source=pkg.weather.source)
        scene.retrofits = retrofit_options(sc.layout, sc.prop)
        scene.geometry = pkg.geometry.model_dump(mode="json")
        scene.input_quality = pkg.provenance_summary.model_dump(mode="json")
        for option in [None, *[r for r in scene.retrofits if r.applicable]]:
            ids = [] if option is None else [option.retrofit_id]
            view = run_scenario(twin, sc.layout, samples, ids, sc.appliances, weather_source=pkg.weather.source)
            scene.scenarios[view.key] = view
        (OUT / f"{pid}.scene.json").write_text(scene.model_dump_json(indent=2), encoding="utf-8")
        print(f"{pid}: package ({pkg.calibration_readiness.method}), {len(scene.scenarios)} scenarios, engine {alloc.engine}")
    (OUT / "physics_inputs.schema.json").write_text(json.dumps(PhysicsInputs.model_json_schema(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
