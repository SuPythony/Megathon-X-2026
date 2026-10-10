"""Household: occupancy, heating behaviour and the hot-water system, from the intake answers, the EPC and appliances.

Each comes with where its values came from (user, EPC, appliances, SAP default) and, for behaviour, an uncertainty
that becomes the calibration prior: a value the user told us gets a tighter prior than a SAP default.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from contracts.models import HeatingSystem, Property
from greentwin.twin.appliances import ApplianceSummary

Presence = Literal["out_all_day", "part_day", "home_all_day"]
# Share of the day someone is home, for metabolic gains. Working assumption.
PRESENCE_FRACTION: dict[str, float] = {"out_all_day": 0.55, "part_day": 0.7, "home_all_day": 0.85}


class Occupancy(BaseModel):
    occupants: float
    presence: Presence
    presence_fraction: float
    metabolic_gains_w: float = Field(description="60 W per person (SAP 2012 Table 5) x share of the day at home")
    source: str


class HeatingBehaviour(BaseModel):
    setpoint_c: float
    setpoint_sd: float = Field(description="Prior standard deviation, K")
    heating_hours_per_day: float
    hours_sd: float
    periods: list[tuple[int, int]] = Field(default_factory=list, description="Heating periods (start, end hour)")
    source: str


class HotWaterSystem(BaseModel):
    """Hot water, separate from space heating: its own equipment, fuel and efficiency."""

    system: Literal["combi_or_instant_from_main", "cylinder_from_main", "electric_immersion", "heat_pump_cylinder",
                    "instantaneous_electric"]
    fuel: Literal["mains_gas", "electricity", "oil", "lpg", "solid", "district"]
    efficiency: float = Field(description="Delivered hot water / fuel, after any summer-efficiency penalty")
    standing_loss_kwh_per_day: float = Field(description="Cylinder heat loss; 0 for combi/instantaneous")
    electric_shower: bool
    shower_electric_kwh_per_day: float
    main_useful_kwh_per_day: float = Field(description="Useful hot water from the main system (after showers)")
    sources: dict[str, str]


def occupancy(prop: Property, occupants: Optional[float], presence: Optional[Presence], sap_n: float) -> Occupancy:
    n = float(occupants) if occupants else sap_n
    pres = presence or "part_day"
    frac = PRESENCE_FRACTION[pres]
    src = ("user" if occupants else "sap_2012 table 1b (from floor area)") + (", presence user" if presence else ", presence assumed")
    return Occupancy(occupants=round(n, 2), presence=pres, presence_fraction=frac,
                     metabolic_gains_w=round(60 * n * frac, 1), source=src)


def heating_behaviour(thermostat_c: Optional[float], periods: Optional[Sequence[tuple[int, int]]],
                      presence: Presence) -> HeatingBehaviour:
    if periods:
        hours = float(sum((end - start) % 24 or 24 for start, end in periods))
        hours_sd, hsrc = 1.0, "user schedule"
    else:
        hours = {"out_all_day": 8.0, "part_day": 9.0, "home_all_day": 12.0}[presence]
        hours_sd, hsrc = 2.5, "sap-style default by presence"
    if thermostat_c is not None:
        sp, sp_sd, ssrc = thermostat_c - 1.0, 1.0, "user thermostat less 1 K (rooms away from the stat run cooler)"
    else:
        sp, sp_sd, ssrc = 19.5, 1.5, "default 19.5 C"
    return HeatingBehaviour(setpoint_c=sp, setpoint_sd=sp_sd, heating_hours_per_day=min(hours, 24.0), hours_sd=hours_sd,
                            periods=list(periods or []), source=f"setpoint: {ssrc}; hours: {hsrc}")


_BOILERS = {HeatingSystem.gas_boiler_condensing, HeatingSystem.gas_boiler_non_condensing, HeatingSystem.oil_boiler,
            HeatingSystem.lpg_boiler, HeatingSystem.communal_boiler, HeatingSystem.solid_fuel}
_HEAT_PUMPS = {HeatingSystem.air_source_heat_pump, HeatingSystem.ground_source_heat_pump}


def hot_water_system(prop: Property, heating_efficiency: float, sap_useful_kwh_per_day: float,
                     appliances: Optional[ApplianceSummary]) -> HotWaterSystem:
    """Hot-water equipment from the EPC (heating.hot_water, cylinder) plus what the user listed (electric shower,
    immersion cylinder). Efficiencies: boilers lose ~10% in summer-only DHW operation (assumption); immersion and
    instantaneous electric 1.0; heat-pump cylinder 2.0 (assumption)."""
    h = prop.heating
    src: dict[str, str] = {}
    has_immersion = bool(appliances and appliances.has_immersion)

    if h.hot_water == "electric_immersion" or (has_immersion and h.main_fuel == "electricity" and h.main_system not in _HEAT_PUMPS):
        system, fuel, eff = "electric_immersion", "electricity", 1.0
    elif h.hot_water == "heat_pump_cylinder" or h.main_system in _HEAT_PUMPS:
        system, fuel, eff = "heat_pump_cylinder", "electricity", 2.0
    elif h.hot_water == "instantaneous" and h.main_fuel == "electricity":
        system, fuel, eff = "instantaneous_electric", "electricity", 1.0
    elif h.cylinder_insulated is not None or has_immersion:
        system, fuel = "cylinder_from_main", h.main_fuel
        eff = heating_efficiency * (0.9 if h.main_system in _BOILERS else 1.0)
    else:
        system, fuel = "combi_or_instant_from_main", h.main_fuel
        eff = heating_efficiency * (0.9 if h.main_system in _BOILERS else 1.0)
    src["system"] = "epc heating.hot_water" + (" + appliance list (cylinder)" if has_immersion else "")
    src["efficiency"] = "assumption from main system (boiler summer penalty 0.9; heat pump cylinder 2.0; electric 1.0)"

    if system in ("combi_or_instant_from_main", "instantaneous_electric"):
        loss = 0.0
    elif appliances and appliances.cylinder_standing_loss_kwh_per_day is not None:
        loss = appliances.cylinder_standing_loss_kwh_per_day
        src["standing_loss"] = "appliance list (cylinder insulation answer)"
    else:
        loss = 1.0 if h.cylinder_insulated in (True, None) else 3.0
    src.setdefault("standing_loss", "epc cylinder_insulated (assumption: 1.0 / 3.0 kWh/day)")

    shower = appliances.shower_electric_kwh_per_day if appliances else 0.0
    # Electric showers supply part of the household's hot water directly; the main system makes the rest.
    main_useful = max(sap_useful_kwh_per_day - shower, 0.4 * sap_useful_kwh_per_day)
    src["main_useful"] = "sap_2012 appendix J demand less electric shower use (floor 40%)"

    return HotWaterSystem(system=system, fuel=fuel, efficiency=round(eff, 3), standing_loss_kwh_per_day=loss,
                          electric_shower=shower > 0, shower_electric_kwh_per_day=round(shower, 3),
                          main_useful_kwh_per_day=round(main_useful, 3), sources=src)
