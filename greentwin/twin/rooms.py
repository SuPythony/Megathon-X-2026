"""Room allocation: split the calibrated whole-house results across rooms. Rooms always sum to the whole house.

Heat loss: each room's share of every element (exterior wall, window, door, roof, floor, ventilation by volume) from
the fitted layout's geometry, applied per posterior sample to the twin's element heat loss. Space heating follows the
room's heat-loss share. Appliances: the user's in-home electric appliances, scaled to the calibrated baseload with a
x1.5 cap; the rest is reported as 'unaccounted' at house level. Every room number is labelled estimated.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from contracts.models import Distribution
from contracts.summarise import summarise
from greentwin.interfaces import PosteriorSamples, TwinRun
from greentwin.twin.appliances import ApplianceSummary
from greentwin.twin.archetypes import TwinLayout

TARIFFS = Path(__file__).resolve().parents[2] / "contracts" / "tariffs.json"
SCALE_CAP = 1.5
BASIS = ("Estimated: each room's share of the calibrated whole-house heat loss comes from its outside walls, windows, "
         "roof and floor in the fitted layout; appliance energy from your appliance list, scaled to the whole house")


@lru_cache(maxsize=1)
def unit_rates_gbp() -> dict[str, float]:
    t = json.loads(TARIFFS.read_text(encoding="utf-8"))
    return {"electricity": t["electricity"]["unit_rate_p_per_kwh"] / 100, "mains_gas": t["gas"]["unit_rate_p_per_kwh"] / 100}


class RoomEnergy(BaseModel):
    room_id: str
    label: str
    heat_loss_w_per_k: Distribution
    heat_loss_share_pct: float
    space_heating_kwh: Distribution = Field(description="Useful heat delivered to this room, kWh/yr")
    appliance_kwh: Optional[Distribution] = Field(None, description="In-home electric appliances, scaled, kWh/yr")
    lighting_kwh: Distribution
    other_kwh: dict[str, float] = Field(default_factory=dict, description="Showers, plug-in heaters, gas cooking (kWh/yr)")
    running_cost_gbp: Optional[Distribution] = Field(None, description="Heating + appliances + lighting, unit rates only")
    estimated: bool = True
    basis: str = BASIS


class RoomAllocation(BaseModel):
    rooms: list[RoomEnergy]
    unaccounted_electricity_kwh: Optional[Distribution] = Field(
        None, description="Calibrated baseload not explained by the listed appliances (standby, small items)")
    engine: str


def _geometry_shares(layout: TwinLayout) -> dict[str, dict[str, float]]:
    """Per element, each room's share (0..1) of the house total, from the layout geometry."""
    rooms = {r.room_id: r for r in layout.rooms}
    top = max(r.storey for r in layout.rooms)
    opening_area: dict[str, dict[str, float]] = {}
    for o in layout.openings:
        kind = "doors" if o.kind == "door" else "windows"
        opening_area.setdefault(o.room_id, {}).setdefault(kind, 0.0)
        opening_area[o.room_id][kind] += o.width_m * o.height_m
    raw: dict[str, dict[str, float]] = {e: {} for e in ("walls", "windows", "doors", "roof", "floor", "ventilation", "party_walls")}
    for rid, room in rooms.items():
        walls = [w for w in layout.walls if w.room_id == rid]
        ext = sum(w.length_m * w.height_m for w in walls if w.kind == "exterior")
        ops = opening_area.get(rid, {})
        raw["windows"][rid] = ops.get("windows", 0.0)
        raw["doors"][rid] = ops.get("doors", 0.0)
        raw["walls"][rid] = max(ext - raw["windows"][rid] - raw["doors"][rid], 0.0)
        raw["party_walls"][rid] = sum(w.length_m * w.height_m for w in walls if w.kind == "party")
        raw["roof"][rid] = room.area_m2 if layout.roof_exposed and room.storey == top else 0.0
        raw["floor"][rid] = room.area_m2 if layout.ground_floor_exposed and room.storey == 0 else 0.0
        raw["ventilation"][rid] = room.area_m2 * (room.ceiling_height_m or layout.ceiling_height_m)
    shares = {}
    for element, by_room in raw.items():
        total = sum(by_room.values())
        shares[element] = {rid: (v / total if total > 0 else 0.0) for rid, v in by_room.items()}
    return shares


def allocate_rooms(layout: TwinLayout, run: TwinRun, samples: PosteriorSamples,
                   appliances: Optional[ApplianceSummary] = None, engine: str = "physics_stub") -> RoomAllocation:
    shares = _geometry_shares(layout)
    n = samples.n_samples
    element_samples = {e: np.asarray(v, dtype=float) for e, v in run.element_heat_loss_w_per_k.items()}
    total_hl = sum(element_samples.values())
    room_hl = {r.room_id: sum(element_samples[e] * shares[e][r.room_id] for e in element_samples) for r in layout.rooms}

    base = samples.base
    baseload = np.asarray(samples.params.get("baseload_kwh_per_day", np.full(n, base.baseload_kwh_per_day)), dtype=float)
    lighting_share = float(base.lighting_share_of_baseload)
    lighting_total = baseload * 365 * lighting_share
    non_lighting = baseload * 365 * (1 - lighting_share)
    floor_total = sum(r.area_m2 for r in layout.rooms)
    rates = unit_rates_gbp()
    heat_fuel = base.heating_fuel
    heat_rate = rates.get(heat_fuel)

    listed = {r.room_id: 0.0 for r in layout.rooms}
    other: dict[str, dict[str, float]] = {r.room_id: {} for r in layout.rooms}
    if appliances:
        for a in appliances.appliances:
            if a.room_id not in listed:
                continue
            if a.fuel == "electricity" and a.category in ("appliance", "cooking"):
                listed[a.room_id] += a.kwh_per_year
            elif a.kwh_per_year > 0:
                key = {"hot_water": "electric_shower", "secondary_heating": "plug_in_heating", "cooking": "gas_cooking"}.get(a.category, a.category)
                other[a.room_id][key] = round(other[a.room_id].get(key, 0.0) + a.kwh_per_year, 1)
    listed_total = sum(listed.values())
    scale = np.minimum(non_lighting / listed_total, SCALE_CAP) if listed_total > 0 else np.zeros(n)

    out = []
    for room in layout.rooms:
        rid = room.room_id
        share = room_hl[rid] / np.where(total_hl > 0, total_hl, 1.0)
        heating = np.asarray(run.space_heating_kwh, dtype=float) * share
        lighting = lighting_total * room.area_m2 / floor_total
        app = listed[rid] * scale if appliances else None
        elec_room = lighting + (app if app is not None else 0.0) + other[rid].get("electric_shower", 0.0) + other[rid].get("plug_in_heating", 0.0)
        cost = None
        if heat_rate is not None:
            cost = heating / float(base.heating_efficiency) * heat_rate + elec_room * rates["electricity"]
            cost = cost + other[rid].get("gas_cooking", 0.0) * rates["mains_gas"]
        out.append(RoomEnergy(
            room_id=rid, label=room.label,
            heat_loss_w_per_k=summarise(room_hl[rid], "W/K", worst_tail="high"),
            heat_loss_share_pct=round(float(np.mean(share)) * 100, 1),
            space_heating_kwh=summarise(heating, "kWh/yr", worst_tail="high"),
            appliance_kwh=summarise(app, "kWh/yr", worst_tail="high") if app is not None else None,
            lighting_kwh=summarise(lighting, "kWh/yr", worst_tail="high"),
            other_kwh=other[rid],
            running_cost_gbp=summarise(cost, "GBP/yr", worst_tail="high") if cost is not None else None,
        ))
    unaccounted = None
    if appliances:
        unaccounted = summarise(np.maximum(non_lighting - listed_total * scale, 0.0), "kWh/yr", worst_tail="high")
    return RoomAllocation(rooms=out, unaccounted_electricity_kwh=unaccounted, engine=engine)


def _range(d: Optional[Distribution]):
    from greentwin.twin.scene import Range

    if d is None:
        return None
    ci = next(c for c in d.credible_intervals if abs(c.level - 0.9) < 1e-9)
    return Range(headline=round(getattr(d, d.headline), 1), lo=round(ci.lo, 1), hi=round(ci.hi, 1), unit=d.unit)


def energy_overlay(alloc: RoomAllocation, calibrated: bool, weather_source: str):
    """Compact per-room numbers for the 3D viewer's scene JSON."""
    from greentwin.twin.scene import EnergyOverlay, RoomEnergyView

    rooms = [RoomEnergyView(room_id=r.room_id, heat_loss_w_per_k=_range(r.heat_loss_w_per_k),
                            heat_loss_share_pct=r.heat_loss_share_pct, space_heating_kwh=_range(r.space_heating_kwh),
                            appliance_kwh=_range(r.appliance_kwh), lighting_kwh=_range(r.lighting_kwh),
                            running_cost_gbp=_range(r.running_cost_gbp), other_kwh=r.other_kwh)
             for r in alloc.rooms]
    return EnergyOverlay(engine=alloc.engine, calibrated=calibrated, weather_source=weather_source, rooms=rooms,
                         unaccounted_electricity_kwh=_range(alloc.unaccounted_electricity_kwh), basis=BASIS)
