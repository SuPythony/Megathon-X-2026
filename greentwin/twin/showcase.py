"""Showcase homes SH1-SH3 wired for the twin: EPC record, usage, an example household and appliance list, layout.

Household answers and appliances are illustrative examples, not real customer data. Used by
``scripts/make_twin_examples.py`` and the dev twin server (``scripts/twin_server.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contracts.models import EnergyUsageMonth, Property
from greentwin.twin.archetypes import TwinLayout, build_layout
from greentwin.twin.intake import TwinIntake
from greentwin.twin.params import choose_archetype
from greentwin.twin.scene import AppliancePlacement as P

EX = Path(__file__).resolve().parents[2] / "contracts" / "examples"

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


@dataclass
class Showcase:
    prop: Property
    usage: list[EnergyUsageMonth]
    intake: TwinIntake
    layout: TwinLayout
    appliances: list[P]


def load_showcase(pid: str) -> Showcase:
    if pid not in HOMES:
        raise KeyError(f"unknown showcase home {pid!r}; known: {sorted(HOMES)}")
    h = HOMES[pid]
    prop = Property.model_validate_json((EX / f"{pid}_property.json").read_text(encoding="utf-8"))
    usage = [EnergyUsageMonth.model_validate(r) for r in json.loads((EX / f"{pid}_usage_2025.json").read_text())]
    intake = TwinIntake(intake_id=f"example-{pid}", postcode_district=prop.location.postcode_district,
                        occupants=h["occupants"], daytime_occupancy=h["daytime_occupancy"],
                        thermostat_c=h["thermostat_c"], heating_periods=h["heating_periods"])
    layout = build_layout(choose_archetype(prop), total_floor_area_m2=prop.floor_area_m2, built_form=prop.built_form)
    return Showcase(prop=prop, usage=usage, intake=intake, layout=layout, appliances=list(h["appliances"]))
