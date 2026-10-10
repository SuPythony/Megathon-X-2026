"""Write the twin's example outputs for the showcase homes SH1-SH3:

- greentwin/twin/examples/physics_inputs_<SH>.json : the physics inputs package (for the physics/calibration team)
- greentwin/twin/examples/<SH>.scene.json          : room scene with per-room energy (for the 3D viewer)
- greentwin/twin/examples/physics_inputs.schema.json

    uv run python scripts/make_twin_examples.py

Household answers and appliances below are illustrative examples, not real customer data.
Numbers come from the stub physics until greentwin/twin/physics.py exists; scenes say so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contracts.models import EnergyUsageMonth, Property  # noqa: E402
from greentwin.twin.archetypes import build_layout  # noqa: E402
from greentwin.twin.inputs import PhysicsInputs, build_physics_inputs  # noqa: E402
from greentwin.twin.intake import TwinIntake  # noqa: E402
from greentwin.twin.params import choose_archetype  # noqa: E402
from greentwin.twin.rooms import energy_overlay  # noqa: E402
from greentwin.twin.scene import AppliancePlacement as P  # noqa: E402
from greentwin.twin.scene import build_scene  # noqa: E402
from greentwin.twin.simulate import prior_samples  # noqa: E402
from greentwin.twin.twin import GreenDigitalTwin  # noqa: E402

EX = ROOT / "contracts" / "examples"
OUT = ROOT / "greentwin" / "twin" / "examples"

HOMES = {
    "SH1": dict(occupants=3, daytime_occupancy="part_day", thermostat_c=20.0, heating_periods=[(6, 8), (17, 22)], appliances=[
        P(appliance_id="a1", type="fridge_freezer", room_id="kitchen", answers={"size": "medium", "age": "old"}),
        P(appliance_id="a2", type="washing_machine", room_id="kitchen", answers={"drum_kg": "8"}),
        P(appliance_id="a3", type="electric_oven", room_id="kitchen"),
        P(appliance_id="a4", type="hob", room_id="kitchen", answers={"type": "gas"}),
        P(appliance_id="a5", type="tv", room_id="living", answers={"inches": "55"}),
        P(appliance_id="a6", type="electric_shower", room_id="bath", answers={"rating_kw": "8.5"}),
    ]),
    "SH2": dict(occupants=4, daytime_occupancy="part_day", thermostat_c=21.0, heating_periods=[(6, 9), (16, 22)], appliances=[
        P(appliance_id="a1", type="fridge_freezer", room_id="kitchen", answers={"size": "large", "age": "mid"}),
        P(appliance_id="a2", type="tumble_dryer", room_id="kitchen", answers={"type": "condenser"}),
        P(appliance_id="a3", type="dishwasher", room_id="kitchen"),
        P(appliance_id="a4", type="tv", room_id="living", answers={"inches": "65"}),
        P(appliance_id="a5", type="desktop_pc", room_id="bed3", answers={"type": "gaming"}),
        P(appliance_id="a6", type="ev_charger", room_id="outside", answers={"miles_per_week": "150"}),
    ]),
    "SH3": dict(occupants=2, daytime_occupancy="home_all_day", thermostat_c=None, heating_periods=[], appliances=[
        P(appliance_id="a1", type="fridge_freezer", room_id="kitchen"),
        P(appliance_id="a2", type="washing_machine", room_id="kitchen", answers={"drum_kg": "6"}),
        P(appliance_id="a3", type="hob", room_id="kitchen", answers={"type": "ceramic"}),
        P(appliance_id="a4", type="immersion_heater", room_id="hall", answers={"jacket": "factory"}),
        P(appliance_id="a5", type="electric_heater", room_id="bed1", answers={"rating_kw": "2"}),
        P(appliance_id="a6", type="tv", room_id="living"),
    ]),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    twin = GreenDigitalTwin(n_samples=1000)
    for pid, h in HOMES.items():
        prop = Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))
        usage = [EnergyUsageMonth.model_validate(r) for r in json.loads((EX / f"{pid}_usage_2025.json").read_text())]
        intake = TwinIntake(intake_id=f"example-{pid}", postcode_district=prop.location.postcode_district,
                            occupants=h["occupants"], daytime_occupancy=h["daytime_occupancy"],
                            thermostat_c=h["thermostat_c"], heating_periods=h["heating_periods"])
        layout = build_layout(choose_archetype(prop), total_floor_area_m2=prop.floor_area_m2, built_form=prop.built_form)
        pkg = build_physics_inputs(prop, layout=layout, appliances=h["appliances"], intake=intake, usage=usage)
        (OUT / f"physics_inputs_{pid}.json").write_text(pkg.model_dump_json(indent=2), encoding="utf-8")

        samples = prior_samples(pkg.params, 1000, priors=pkg.prior_table())
        alloc = twin.room_energy(prop, layout, samples, h["appliances"])
        scene = build_scene(layout, pid, h["appliances"])
        scene.energy = energy_overlay(alloc, calibrated=False, weather_source=pkg.weather.source)
        (OUT / f"{pid}.scene.json").write_text(scene.model_dump_json(indent=2), encoding="utf-8")
        print(f"{pid}: package ({pkg.calibration_readiness.method}), scene with {len(alloc.rooms)} rooms, engine {alloc.engine}")
    (OUT / "physics_inputs.schema.json").write_text(json.dumps(PhysicsInputs.model_json_schema(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
