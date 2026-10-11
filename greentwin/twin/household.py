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
# Share of the day someone is home, for metabolic gains. Working assumption. SAP's 60 W/person is already a 24-hour
# average for a typical household, taken here as "part_day"; other answers scale it by their share relative to that.
PRESENCE_FRACTION: dict[str, float] = {"out_all_day": 0.55, "part_day": 0.7, "home_all_day": 0.85}
SAP_TYPICAL_PRESENCE = "part_day"


class Occupancy(BaseModel):
    occupants: float
    presence: Presence
    presence_fraction: float
    metabolic_gains_w: float = Field(description="60 W per person (SAP 2012 Table 5, a 24 h average), scaled by "
                                                  "daytime presence relative to a typical household; before the "
                                                  "40 W/person evaporation loss (net 20 W/person, as P2's formula)")
    source: str


class HeatingBehaviour(BaseModel):
    setpoint_c: float
    setpoint_sd: float = Field(description="Prior standard deviation, K; 0 = the customer answered, so fixed")
    heating_hours_per_day: float
    hours_sd: float = Field(description="Prior standard deviation, h; 0 = the customer gave a schedule, so fixed")
    periods: list[tuple[int, int]] = Field(default_factory=list, description="Heating periods (start, end hour)")
    setpoint_source: str
    hours_source: str
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
    src = "user answer: occupants (fixed)" if occupants else "sap_2012 table 1b from floor area (uncertain)"
    return Occupancy(occupants=round(n, 2), presence=pres, presence_fraction=frac,
                     metabolic_gains_w=round(60 * n * frac / PRESENCE_FRACTION[SAP_TYPICAL_PRESENCE], 1), source=src)


def heating_behaviour(thermostat_c: Optional[float], periods: Optional[Sequence[tuple[int, int]]],
                      presence: Presence) -> HeatingBehaviour:
    # A customer's explicit answer is fixed (sd 0); a missing one becomes an uncertain input with a wider prior.
    if periods:
        hours = float(sum((end - start) % 24 or 24 for start, end in periods))
        hours_sd, hsrc = 0.0, "user answer: heating schedule (fixed)"
    else:
        hours = {"out_all_day": 8.0, "part_day": 9.0, "home_all_day": 12.0}[presence]
        hours_sd, hsrc = 2.5, "default by daytime presence (uncertain)"
    if thermostat_c is not None:
        sp, sp_sd, ssrc = thermostat_c - 1.0, 0.0, "user answer: thermostat less 1 K, rooms away from the stat run cooler (fixed)"
    else:
        sp, sp_sd, ssrc = 19.5, 1.5, "default 19.5 C (uncertain)"
    return HeatingBehaviour(setpoint_c=sp, setpoint_sd=sp_sd, heating_hours_per_day=min(hours, 24.0), hours_sd=hours_sd,
                            periods=list(periods or []), setpoint_source=ssrc, hours_source=hsrc,
                            source=f"setpoint: {ssrc}; hours: {hsrc}")


# --------------------------------------------------------------------------- optional follow-up questions
# Asked only when relevant; None = "not sure" (or not asked), which keeps a default with a wider prior.
PlugInHeaterUse = Literal["never", "rarely", "some_evenings", "most_days"]
LedLighting = Literal["all", "most", "about_half", "few"]
# Hours a day per plug-in heater over the heating season. Working assumption.
HEATER_HOURS: dict[str, float] = {"never": 0.0, "rarely": 0.5, "some_evenings": 2.0, "most_days": 5.0}
DEFAULT_HEATER_HOURS = 2.0  # appliance_energy.json electric_heater hours_per_day
# Share of low-energy (LED/CFL) lighting, %, for SAP 2012 Appendix L.
LED_PCT: dict[str, float] = {"all": 100.0, "most": 80.0, "about_half": 50.0, "few": 15.0}
DEFAULT_LED_PCT = 50.0

# Homes where plug-in heaters commonly top up the main heating (storage heaters run out in the evening; solid-fuel
# room heaters warm one room). Intake heating answer values and EPC systems.
_TOP_UP_USER = {"electric_storage", "electric_storage_hhr", "solid_fuel"}
_TOP_UP_EPC = {HeatingSystem.electric_storage, HeatingSystem.electric_storage_hhr, HeatingSystem.solid_fuel}


# --------------------------------------------------------------------------- solar, battery, appliance list
YesNo = Literal["yes", "no"]
Completeness = Literal["all", "most", "few"]
TYPICAL_PV_KWP = 3.5        # assumption: typical UK home system when the owner doesn't know the size
TYPICAL_BATTERY_KWH = 5.0   # assumption: typical home battery when the owner doesn't know the capacity
# Share of the household's appliance use the list covers; the rest of the gap to the SAP typical is added.
LISTED_SHARE: dict[str, float] = {"all": 1.0, "most": 0.75, "few": 0.3}
LISTED_SHARE_NOT_SURE = 0.5
BASELOAD_SIGMA: dict[str, float] = {"all": 0.15, "most": 0.2, "few": 0.3}


class Ownership(BaseModel):
    """What the customer said about solar panels and a battery, applied over the EPC."""

    pv_kwp: float
    battery_kwh: float
    sources: dict[str, str]
    sigma: dict[str, float] = Field(description="Prior sigma per field: 0 = fixed answer; >0 = lognormal, uncertain")
    flags: list[str] = Field(default_factory=list)


def _owned(name: str, unit: str, has: Optional[str], size: Optional[float], epc_value: float, typical: float
           ) -> tuple[float, str, Optional[float], list[str]]:
    if has == "no":
        flag = [f"EPC records {name} ({epc_value:g} {unit}); you said none: using your answer"] if epc_value else []
        return 0.0, f"user answer: no {name} (fixed)", 0.0, flag
    if has == "yes" and size:
        return float(size), f"user answer: {name} {size:g} {unit} (fixed)", 0.0, []
    if has == "yes":
        value = epc_value or typical
        how = "EPC size" if epc_value else f"typical {typical:g} {unit} (assumption)"
        return value, f"user answer: has {name}, size not given: {how} (uncertain)", 0.3, []
    if epc_value:  # not sure: the EPC's record stands, with some doubt
        return epc_value, f"epc {name} {epc_value:g} {unit} (customer not sure, uncertain)", 0.2, []
    return 0.0, f"no {name} recorded (customer not sure)", None, [f"Not sure whether the home has {name}: assumed none"]


def apply_user_ownership(prop: Property, solar: Optional[YesNo], pv_kwp: Optional[float], battery: Optional[YesNo],
                         battery_kwh: Optional[float]) -> tuple[Property, Ownership]:
    r = prop.renewables
    pv, pv_src, pv_sd, pv_flags = _owned("solar panels", "kWp", solar, pv_kwp, r.pv_kwp, TYPICAL_PV_KWP)
    bat, bat_src, bat_sd, bat_flags = _owned("a home battery", "kWh", battery, battery_kwh, r.battery_kwh,
                                             TYPICAL_BATTERY_KWH)
    flags = pv_flags + bat_flags
    if bat > 0 and pv == 0:
        flags.append("A battery without solar panels only shifts grid electricity (no self-consumption gain)")
    sigma = {k: v for k, v in (("pv_kwp", pv_sd), ("battery_kwh", bat_sd)) if v is not None}
    own = Ownership(pv_kwp=pv, battery_kwh=bat, sources={"pv_kwp": pv_src, "battery_kwh": bat_src}, sigma=sigma,
                    flags=flags)
    return prop.model_copy(update={"renewables": r.model_copy(update={"pv_kwp": pv, "battery_kwh": bat})}), own


class FollowUp(BaseModel):
    key: Literal["plug_in_heater_use", "led_lighting"]
    question: str
    options: list[tuple[str, str]]
    reason: str = Field(description="Why it is asked for this home")


def follow_up_questions(prop: Optional[Property], heating_system: Optional[str],
                        appliance_types: Sequence[str]) -> list[FollowUp]:
    """The optional follow-ups worth asking this home. Plug-in heaters: one is listed, or the heating is storage or
    solid-fuel room heaters. Lighting: no EPC, the EPC doesn't record it, or it records less than all low-energy."""
    out: list[FollowUp] = []
    listed = "electric_heater" in appliance_types
    top_up = heating_system in _TOP_UP_USER if heating_system else bool(prop and prop.heating.main_system in _TOP_UP_EPC)
    if listed or top_up:
        out.append(FollowUp(
            key="plug_in_heater_use", question="How often are plug-in heaters used in winter?",
            options=[("never", "Never"), ("rarely", "Rarely"), ("some_evenings", "Some evenings"),
                     ("most_days", "Most days")],
            reason="a plug-in heater is listed" if listed else "storage or room heaters are often topped up"))
    pct = prop.low_energy_lighting_pct if prop else None
    if pct is None or pct < 100:
        out.append(FollowUp(
            key="led_lighting", question="How much of the lighting is LED?",
            options=[("all", "All"), ("most", "Most"), ("about_half", "About half"), ("few", "Few or none")],
            reason="no EPC lighting record" if pct is None else f"EPC records {pct:.0f}% low-energy lighting"))
    return out


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
