"""Appliances -> consumption schedules and matching internal heat gains.

Each appliance the user placed (type + size answers, from ``AppliancePlacement``) becomes:
- annual kWh and a fuel (electricity or gas),
- a monthly schedule (12 calendar months) and an average-day hourly profile (24 h),
- internal heat gains: the share of its energy released inside the heated envelope, as average watts per month.

Categories decide where the energy goes in the physics (``appliance_energy.json``):
appliance/cooking -> baseload (or gas cooking), hot_water -> electric shower, secondary_heating -> plug-in heaters,
outdoor -> EV charger / hot tub (no gains), hot_water_system -> describes the cylinder, no consumption of its own.

Figures are typical-product PRIOR means (status typical_estimate); calibration corrects the total baseload.
"""

from __future__ import annotations

import calendar
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from greentwin.twin.scene import AppliancePlacement

HERE = Path(__file__).resolve().parent
Category = Literal["appliance", "cooking", "hot_water", "secondary_heating", "outdoor", "hot_water_system"]
DAYS = [calendar.monthrange(2025, m)[1] for m in range(1, 13)]


@lru_cache(maxsize=1)
def catalogue() -> dict:
    return json.loads((HERE / "appliance_energy.json").read_text(encoding="utf-8"))


def _hours(*ranges: tuple[int, int, float]) -> list[float]:
    w = [0.0] * 24
    for start, end, weight in ranges:
        for h in range(start, end):
            w[h % 24] += weight
    s = sum(w)
    return [x / s for x in w]


# Average-day shape of use (fractions of the day's energy per hour). Working shapes, not measured load profiles.
DAILY_PROFILES: dict[str, list[float]] = {
    "flat": _hours((0, 24, 1.0)),
    "evening": _hours((17, 23, 1.0), (7, 9, 0.2)),
    "morning_evening": _hours((6, 9, 1.0), (17, 22, 0.8)),
    "cooking": _hours((7, 9, 0.4), (12, 14, 0.5), (17, 20, 1.0)),
    "laundry": _hours((9, 12, 1.0), (17, 21, 0.7)),
    "daytime": _hours((9, 18, 1.0), (19, 22, 0.5)),
    "overnight": _hours((0, 6, 1.0), (22, 24, 0.5)),
}

# Monthly weights by calendar month (Jan..Dec).
MONTHLY_PROFILES: dict[str, list[float]] = {
    "flat": [1.0] * 12,
    "winter_tilt": [1 + 0.15 * math.cos(2 * math.pi * m / 12) for m in range(12)],
    "heating_season": [1.0, 1.0, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.2, 1.0, 1.0, 1.0],
}


class ApplianceEnergy(BaseModel):
    appliance_id: str
    type: str
    room_id: Optional[str]
    category: Category
    fuel: Literal["electricity", "mains_gas"]
    answers: dict[str, str] = Field(description="User's answers, with defaults filled for unanswered questions")
    defaulted: list[str] = Field(default_factory=list, description="Questions answered by default")
    kwh_per_year: float
    monthly_kwh: list[float] = Field(description="Jan..Dec")
    hourly_profile_kw: list[float] = Field(description="Average day, kW per hour 0..23")
    heat_to_room: float
    gains_w_monthly: list[float] = Field(description="Average internal gain, W, Jan..Dec")
    standing_loss_kwh_per_day: Optional[float] = None
    source: str


class ApplianceSummary(BaseModel):
    """What the physics needs, aggregated over all appliances."""

    occupants: float
    baseload_kwh_per_day: float = Field(description="Electric appliances + electric cooking (excl. lighting)")
    gas_cooking_kwh_per_day: float
    shower_electric_kwh_per_day: float
    secondary_heating_kwh_monthly: list[float]
    outdoor_kwh_monthly: list[float]
    gains_w_mean: float = Field(description="Average internal gains from appliances, cooking and showers")
    appliance_month_factor: list[float] = Field(description="Monthly baseload shape, mean 1 (Jan..Dec)")
    gains_month_factor: list[float] = Field(description="Monthly appliance-gains shape, mean 1 (Jan..Dec)")
    hourly_profile_kw: list[float] = Field(description="Average day, all in-home electricity, kW 0..23")
    room_kwh_per_year: dict[str, float]
    has_electric_shower: bool
    has_immersion: bool
    cylinder_standing_loss_kwh_per_day: Optional[float]
    appliances: list[ApplianceEnergy]
    status: str


def _answers(spec: dict, given: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    out, defaulted = {}, []
    for q in spec.get("questions", []):
        valid = {o[0] for o in q["options"]}
        v = given.get(q["key"])
        if v is None or str(v) not in valid:
            out[q["key"]] = q["default"]
            defaulted.append(q["key"])
        else:
            out[q["key"]] = str(v)
    return out, defaulted


def _lookup(table: dict, answers: dict[str, str], default: float) -> float:
    for q, by in table.items():
        if q in answers and answers[q] in by:
            return float(by[answers[q]])
    return default


def appliance_energy(p: AppliancePlacement, occupants: float) -> ApplianceEnergy:
    cat = catalogue()
    if p.type not in cat["types"]:
        raise ValueError(f"no energy data for appliance type {p.type!r}")
    spec = cat["types"][p.type]
    answers, defaulted = _answers(spec, getattr(p, "answers", {}) or {})
    m = spec["model"]
    occ_scale = occupants / cat["occupant_reference"] if m.get("scale_with_occupants") else 1.0
    monthly_w = MONTHLY_PROFILES[spec["monthly_profile"]]
    weight_days = sum(d * w for d, w in zip(DAYS, monthly_w))

    if m["type"] == "annual":
        annual = _lookup(m.get("kwh_by", {}), answers, m.get("base_kwh", 0.0))
        for q, by in m.get("multiplier_by", {}).items():
            annual *= float(by.get(answers.get(q), 1.0))
        annual *= occ_scale
    elif m["type"] == "per_use":
        annual = _lookup(m["kwh_per_use_by"], answers, 1.0) * m["uses_per_week"] * 52.14 * occ_scale
    elif m["type"] == "power":
        watts = _lookup(m.get("watts_by", {}), answers, m.get("base_watts", 0.0))
        annual = watts * m["hours_per_day"] / 1000 * weight_days
    elif m["type"] == "power_kw":
        annual = float(answers[m["kw_question"]]) * m["hours_per_day"] * weight_days
    elif m["type"] == "shower":
        annual = float(answers["rating_kw"]) * m["minutes_per_person_per_day"] / 60 * occupants * 365
    elif m["type"] == "ev":
        annual = float(answers["miles_per_week"]) * 52.14 * m["kwh_per_mile"]
    else:  # "system": the hot-water cylinder itself; its energy is in the hot-water demand
        annual = 0.0

    fuel = "electricity"
    if "fuel_by" in spec:
        q, by = next(iter(spec["fuel_by"].items()))
        fuel = by.get(answers.get(q), "electricity")
    if "fuel_question" in spec:
        fuel = answers[spec["fuel_question"]]
    if fuel == "mains_gas" and "gas_multiplier" in spec:
        annual *= spec["gas_multiplier"]  # gas cooking is less efficient than electric

    heat = spec["heat_to_room"]
    for q, by in spec.get("heat_to_room_by", {}).items():
        heat = float(by.get(answers.get(q), heat))

    daily = annual / weight_days if weight_days else 0.0
    monthly = [daily * d * w for d, w in zip(DAYS, monthly_w)]
    profile = DAILY_PROFILES[spec["daily_profile"]]
    hourly_kw = [annual / 365 * f for f in profile]
    gains = [kwh * heat * 1000 / (24 * d) for kwh, d in zip(monthly, DAYS)]
    loss = None
    if "standing_loss_kwh_per_day_by" in spec:
        loss = _lookup(spec["standing_loss_kwh_per_day_by"], answers, 1.0)

    return ApplianceEnergy(
        appliance_id=p.appliance_id, type=p.type, room_id=p.room_id if p.room_id not in (None, "outside") else None,
        category=spec["category"], fuel=fuel, answers=answers, defaulted=defaulted,
        kwh_per_year=round(annual, 1), monthly_kwh=[round(x, 2) for x in monthly],
        hourly_profile_kw=[round(x, 4) for x in hourly_kw], heat_to_room=heat,
        gains_w_monthly=[round(x, 2) for x in gains], standing_loss_kwh_per_day=loss,
        source=f"appliance_energy.json ({cat['status']})",
    )


def summarise_appliances(placements: Sequence[AppliancePlacement], occupants: float) -> ApplianceSummary:
    items = [appliance_energy(p, occupants) for p in placements]
    zero = [0.0] * 12

    def monthly(pred) -> list[float]:
        out = list(zero)
        for a in items:
            if pred(a):
                out = [x + y for x, y in zip(out, a.monthly_kwh)]
        return out

    in_home_elec = lambda a: a.fuel == "electricity" and a.category in ("appliance", "cooking")  # noqa: E731
    base_monthly = monthly(in_home_elec)
    base_annual = sum(base_monthly)
    gains_monthly = [sum(a.gains_w_monthly[i] for a in items if a.category in ("appliance", "cooking", "hot_water"))
                     for i in range(12)]
    gains_mean = sum(g * d for g, d in zip(gains_monthly, DAYS)) / 365

    def shape(month_values: list[float], per_day: bool) -> list[float]:
        vals = [v / d for v, d in zip(month_values, DAYS)] if per_day else month_values
        mean = sum(v * d for v, d in zip(vals, DAYS)) / 365
        return [round(v / mean, 4) if mean > 0 else 1.0 for v in vals]

    rooms: dict[str, float] = {}
    for a in items:
        if a.room_id and a.category in ("appliance", "cooking", "hot_water", "secondary_heating"):
            rooms[a.room_id] = round(rooms.get(a.room_id, 0.0) + a.kwh_per_year, 1)
    hourly = [round(sum(a.hourly_profile_kw[h] for a in items if in_home_elec(a)), 4) for h in range(24)]
    cylinder = next((a for a in items if a.type == "immersion_heater"), None)

    return ApplianceSummary(
        occupants=occupants,
        baseload_kwh_per_day=round(base_annual / 365, 3),
        gas_cooking_kwh_per_day=round(sum(a.kwh_per_year for a in items if a.fuel == "mains_gas") / 365, 3),
        shower_electric_kwh_per_day=round(sum(a.kwh_per_year for a in items if a.category == "hot_water") / 365, 3),
        secondary_heating_kwh_monthly=[round(x, 2) for x in monthly(lambda a: a.category == "secondary_heating")],
        outdoor_kwh_monthly=[round(x, 2) for x in monthly(lambda a: a.category == "outdoor")],
        gains_w_mean=round(gains_mean, 1),
        appliance_month_factor=shape(base_monthly, per_day=True) if base_annual else [1.0] * 12,
        gains_month_factor=shape(gains_monthly, per_day=False) if gains_mean else [1.0] * 12,
        hourly_profile_kw=hourly,
        room_kwh_per_year=rooms,
        has_electric_shower=any(a.category == "hot_water" for a in items),
        has_immersion=cylinder is not None,
        cylinder_standing_loss_kwh_per_day=cylinder.standing_loss_kwh_per_day if cylinder else None,
        appliances=items,
        status=catalogue()["status"],
    )
