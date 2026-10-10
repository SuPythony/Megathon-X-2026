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


# --------------------------------------------------------------------------- user-given heating (overrides the EPC)
UserHeating = Literal[
    "gas_boiler_combi", "gas_boiler_regular", "oil_boiler", "lpg_boiler", "air_source_heat_pump",
    "ground_source_heat_pump", "electric_storage", "electric_storage_hhr", "electric_radiators",
    "district_or_communal", "solid_fuel",
]
UserHotWater = Literal["from_boiler", "cylinder_from_boiler", "electric_immersion", "heat_pump_cylinder",
                       "instantaneous_electric"]

_USER_HEATING: dict[str, tuple[HeatingSystem, str]] = {
    "gas_boiler_combi": (HeatingSystem.gas_boiler_condensing, "mains_gas"),
    "gas_boiler_regular": (HeatingSystem.gas_boiler_condensing, "mains_gas"),
    "oil_boiler": (HeatingSystem.oil_boiler, "oil"),
    "lpg_boiler": (HeatingSystem.lpg_boiler, "lpg"),
    "air_source_heat_pump": (HeatingSystem.air_source_heat_pump, "electricity"),
    "ground_source_heat_pump": (HeatingSystem.ground_source_heat_pump, "electricity"),
    "electric_storage": (HeatingSystem.electric_storage, "electricity"),
    "electric_storage_hhr": (HeatingSystem.electric_storage_hhr, "electricity"),
    "electric_radiators": (HeatingSystem.electric_direct, "electricity"),
    "district_or_communal": (HeatingSystem.district_heating, "district"),
    "solid_fuel": (HeatingSystem.solid_fuel, "solid"),
}
_USER_HOT_WATER = {"from_boiler": "from_main", "cylinder_from_boiler": "from_main",
                   "electric_immersion": "electric_immersion", "heat_pump_cylinder": "heat_pump_cylinder",
                   "instantaneous_electric": "instantaneous"}


def apply_user_heating(prop: Property, heating_system: Optional[UserHeating],
                       hot_water_source: Optional[UserHotWater]) -> tuple[Property, list[str], dict[str, str]]:
    """The user's own heating and hot-water answers replace the EPC's (EPCs can be years old). Returns the updated
    property, conflicts to show the user, and sources. A gas-boiler answer keeps the EPC's condensing/non-condensing
    type when the EPC also says gas boiler."""
    if not heating_system and not hot_water_source:
        return prop, [], {}
    h = prop.heating.model_copy()
    conflicts: list[str] = []
    src: dict[str, str] = {}
    if heating_system:
        system, fuel = _USER_HEATING[heating_system]
        gas_boilers = (HeatingSystem.gas_boiler_condensing, HeatingSystem.gas_boiler_non_condensing)
        if heating_system.startswith("gas_boiler") and h.main_system in gas_boilers:
            system = h.main_system
        if system != h.main_system or fuel != h.main_fuel:
            conflicts.append(f"EPC says {h.main_system.value.replace('_', ' ')} ({h.main_fuel}); you said "
                             f"{heating_system.replace('_', ' ')}: using your answer")
            h.seasonal_efficiency = None  # the EPC's efficiency belongs to the old system
        h.main_system, h.main_fuel = system, fuel
        if heating_system == "gas_boiler_combi":
            h.hot_water, h.cylinder_insulated = "from_main", None
        elif heating_system == "gas_boiler_regular" and h.cylinder_insulated is None:
            h.cylinder_insulated = True
        src["heating"] = "user answer (overrides EPC)"
    if hot_water_source:
        h.hot_water = _USER_HOT_WATER[hot_water_source]
        if hot_water_source == "cylinder_from_boiler" and h.cylinder_insulated is None:
            h.cylinder_insulated = True
        if hot_water_source == "from_boiler":
            h.cylinder_insulated = None
        src["hot_water"] = "user answer (overrides EPC)"
    return prop.model_copy(update={"heating": h}), conflicts, src


def meter_fuel_conflicts(prop: Property, gas_kwh: Sequence[Optional[float]], has_gas_cooking: bool) -> list[str]:
    """Bills that don't match the heating fuel, e.g. gas bills in an all-electric home."""
    gas = [g for g in gas_kwh if g is not None]
    if not gas:
        return []
    total = sum(gas)
    if prop.heating.main_fuel != "mains_gas" and prop.heating.hot_water != "from_main" and total > 1500 and not has_gas_cooking:
        return [f"Gas bills show {total:,.0f} kWh but the heating and hot water are not gas: check the heating answer"]
    if prop.heating.main_fuel == "mains_gas" and total < 1:
        return ["Heating is gas but the gas readings are zero: check the heating answer or the bills"]
    return []
