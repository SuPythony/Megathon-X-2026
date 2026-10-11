"""The physics inputs package: everything the physics model and the Bayesian calibration need for one home, in one
JSON document (``PhysicsInputs``, version ``physics_inputs/0.1``).

Contents: fabric (U-values x areas), ventilation, heating system, hot-water system (separate equipment, fuel,
efficiency), occupancy and heating behaviour, appliance consumption schedules and the matching internal gains,
weather aligned to the meter months, monthly meter readings (missing months marked, never zero), priors for the
calibrated parameters, a suggested observation model and a calibration-readiness check. Every value carries its source.

Customers are asked only what they can know; technical quantities (U-values, ventilation, efficiency/COP, weather,
solar irradiation, load profiles, battery losses) are generated or sourced and listed in ``technical_inputs``. A
customer's explicit answer is fixed (prior kind ``fixed``); a "not sure" becomes an uncertain input with a wider prior.

    inputs = build_physics_inputs(prop, layout=layout, appliances=placements, intake=intake)
    inputs.model_dump_json(indent=2)           # hand this to the physics / calibration team
    run_physics(inputs.physics_params(), inputs.physics_timeseries(), mode, inputs.fuel_config())
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from contracts.models import EnergyUsageMonth, Property, WeatherMonth
from greentwin.twin.appliances import DAYS, ApplianceSummary, summarise_appliances
from greentwin.twin.archetypes import TwinLayout
from greentwin.twin.geometry import GeometryConfirmation, apply_geometry, from_layout
from greentwin.twin.household import (
    BASELOAD_SIGMA,
    DEFAULT_HEATER_HOURS,
    HEATER_HOURS,
    FollowUp,
    HotWaterSystem,
    apply_user_heating,
    apply_user_ownership,
    follow_up_questions,
    heating_behaviour,
    hot_water_system,
    meter_fuel_conflicts,
    occupancy,
)
from greentwin.twin.intake import TwinIntake
from greentwin.twin.params import ExtendedTwinParams, fuel_config, params_with_sources, sap_occupancy
from greentwin.twin.physics_api import EXTRA_PARAM_KEYS, PHYSICS_PARAM_KEYS, FuelConfig
from greentwin.twin.provenance import InputProvenance, ProvenanceSummary, build_provenance, provenance_summary
from greentwin.twin.scene import AppliancePlacement
from greentwin.twin.thermal import ThermalMass, thermal_mass
from greentwin.twin.weather import filled_fields, synthetic_months, typical_year, weather_arrays

INPUTS_VERSION = "physics_inputs/0.1"
WINTER = {11, 12, 1, 2, 3}
SUMMER = {6, 7, 8}


class Element(BaseModel):
    element: str
    u_value: float
    area_m2: float
    ua_w_per_k: float
    source: str


class Ventilation(BaseModel):
    air_changes_per_hour: float
    volume_m3: float
    heat_loss_w_per_k: float = Field(description="0.33 x ach x volume")
    source: str


class HeatingSystemInfo(BaseModel):
    system: str
    fuel: str
    efficiency: float
    source: str


class Prior(BaseModel):
    """Prior for one parameter. kind: lognormal -> value = base x exp(N(mu, sigma)); normal_add -> base + N(mu, sigma);
    fixed -> the customer's explicit answer, held at base (not sampled, not calibrated)."""

    param: str
    kind: Literal["lognormal", "normal_add", "fixed"]
    mu: float
    sigma: float
    base: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    rationale: str


class ObservationModel(BaseModel):
    """Suggested likelihood for monthly meter readings. The calibration team owns the final choice."""

    likelihood: str = "lognormal on monthly kWh per fuel"
    relative_noise: dict[str, float] = Field(default_factory=lambda: {"actual": 0.05, "estimated": 0.15})
    missing_months: str = "latent (no likelihood term); never treated as 0"
    notes: list[str] = Field(default_factory=list)


class Meter(BaseModel):
    months: list[str]
    electricity_kwh: list[Optional[float]]
    gas_kwh: list[Optional[float]]
    estimated: list[bool]
    source: str


class Readiness(BaseModel):
    method: Literal["prior_only", "weak", "ready"]
    months: int
    gas_months: int
    electricity_months: int
    winter_months: int
    summer_months: int
    estimated_months: int
    identifiable: list[str]
    weakly_identified: list[str]
    notes: list[str]


class Weather(BaseModel):
    months: list[str]
    arrays: dict[str, list[float]]
    source: str
    filled: list[str]


class TechnicalInput(BaseModel):
    """A technical quantity the twin generates or sources itself; never asked of the customer."""

    quantity: str
    value: Optional[float] = None
    unit: str
    source: str
    status: str = Field(description="Provenance status (measured, recorded, estimated, to_verify, assumption, ...)")
    replace_with: str = Field(description="What would make it better, and who provides it")


class FollowUpAnswer(BaseModel):
    question: FollowUp
    answer: Optional[str] = Field(None, description="None = not sure")


class PhysicsInputs(BaseModel):
    inputs_version: str = INPUTS_VERSION
    property_id: str
    created: str
    home: dict[str, str | float | int | None]
    params: ExtendedTwinParams
    sources: dict[str, str]
    fuel: dict[str, Optional[str | float]] = Field(description="heating_fuel, hot_water_fuel, hot_water_efficiency")
    fabric: list[Element]
    ventilation: Ventilation
    heating_system: HeatingSystemInfo
    hot_water: HotWaterSystem
    thermal_mass: ThermalMass
    geometry: GeometryConfirmation = Field(description="The geometry used, and whether the user confirmed it")
    input_provenance: list[InputProvenance]
    provenance_summary: ProvenanceSummary
    technical_inputs: list[TechnicalInput] = Field(description="Generated or sourced, never asked of the customer")
    follow_ups: list[FollowUpAnswer] = Field(default_factory=list, description="Optional questions asked for this home")
    occupancy: dict[str, str | float]
    heating_behaviour: dict[str, object]
    appliances: ApplianceSummary | None
    weather: Weather
    schedules: dict[str, list[float]] = Field(description="Aligned to weather.months; physics_api.SCHEDULE_KEYS")
    meter: Meter | None
    priors: list[Prior]
    observation_model: ObservationModel
    calibration_readiness: Readiness
    data_quality: dict[str, list[str]]

    # ---- helpers for running the physics from the package
    def fuel_config(self) -> FuelConfig:
        return FuelConfig(heating_fuel=self.fuel["heating_fuel"], hot_water_fuel=self.fuel["hot_water_fuel"],
                          hot_water_efficiency=self.fuel.get("hot_water_efficiency"))

    def physics_params(self) -> dict[str, float]:
        keys = [*PHYSICS_PARAM_KEYS, *EXTRA_PARAM_KEYS]
        return {k: float(getattr(self.params, k)) for k in keys}

    def physics_timeseries(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v, dtype=np.float32) for k, v in self.weather.arrays.items()}
        out.update({k: np.asarray(v, dtype=np.float32) for k, v in self.schedules.items()})
        return out

    def prior_table(self) -> dict[str, tuple[str, float, float]]:
        """Priors in the form simulate.prior_samples takes (with bounds); fixed answers are left out, so they stay at
        their value."""
        return {p.param: (p.kind, p.mu, p.sigma, p.lower, p.upper) for p in self.priors if p.kind != "fixed"}


def _schedules(months: Sequence[str], summary: Optional[ApplianceSummary]) -> dict[str, list[float]]:
    idx = [int(m[5:7]) - 1 for m in months]
    if summary is None:
        return {"appliance_month_factor": [1.0] * len(idx), "gains_month_factor": [1.0] * len(idx),
                "secondary_heating_kwh_monthly": [0.0] * len(idx), "extra_electricity_kwh_monthly": [0.0] * len(idx),
                "extra_gas_kwh_monthly": [0.0] * len(idx)}
    gas_daily = summary.gas_cooking_kwh_per_day
    days = [DAYS[i] for i in idx]  # calendar-month days (Feb as non-leap)
    return {
        "appliance_month_factor": [summary.appliance_month_factor[i] for i in idx],
        "gains_month_factor": [summary.gains_month_factor[i] for i in idx],
        "secondary_heating_kwh_monthly": [summary.secondary_heating_kwh_monthly[i] for i in idx],
        "extra_electricity_kwh_monthly": [summary.outdoor_kwh_monthly[i] for i in idx],
        "extra_gas_kwh_monthly": [round(gas_daily * d, 2) for d in days],
    }


def heat_loss_prior_sigma(params: ExtendedTwinParams, prov: list[InputProvenance], geometry_confirmed: bool) -> float:
    """Wider when the fabric rests on unverified U-values and unconfirmed geometry: 0.10 (all measured, confirmed)
    to 0.30 (all defaults, house-type geometry)."""
    status = {p.field: p.status for p in prov}
    pairs = [("wall_area_m2", "wall_u"), ("roof_area_m2", "roof_u"), ("floor_area_m2", "floor_u"),
             ("window_area_m2", "window_u"), ("door_area_m2", "door_u")]
    ua = {u: getattr(params, a) * getattr(params, u) for a, u in pairs}
    total = sum(ua.values()) or 1.0
    unknown_u = sum(v for u, v in ua.items() if status.get(u) not in ("measured", "recorded", "confirmed")) / total
    return round(0.10 + 0.15 * unknown_u + (0.0 if geometry_confirmed else 0.05), 3)


def _behaviour_prior(param: str, base: float, sd: float, source: str, lower: float, upper: float) -> Prior:
    if sd == 0:
        return Prior(param=param, kind="fixed", mu=0.0, sigma=0.0, base=base, rationale=f"Customer answer, fixed: {source}")
    return Prior(param=param, kind="normal_add", mu=0.0, sigma=sd, base=base, lower=lower, upper=upper,
                 rationale=f"Not given, so uncertain: {source}")


def _priors(params: ExtendedTwinParams, sources: dict[str, str], household: dict, has_appliances: bool,
            hl_sigma: float = 0.25) -> list[Prior]:
    """Explicit customer answers are fixed; missing ones widen the priors of what they feed."""
    occupants_given = "user answer" in sources.get("occupants", "")
    lighting_known = not sources.get("low_energy_lighting_pct", "").startswith("assumption")
    heater = sources.get("secondary_heating_scale", "")
    priors = [
        Prior(param="heat_loss_multiplier", kind="lognormal", mu=0.0, sigma=hl_sigma, base=params.heat_loss_multiplier,
              lower=0.4, upper=2.0, rationale="Scales the fabric + ventilation loss; width set by input provenance "
              "(unverified U-values and unconfirmed geometry widen it)"),
        _behaviour_prior("setpoint_c", params.setpoint_c, household["setpoint_sd"], sources["setpoint_c"], 14.0, 24.0),
        Prior(param="baseload_kwh_per_day", kind="lognormal", mu=0.0,
              sigma=(household.get("baseload_sigma") or (0.2 if has_appliances else 0.3))
              + (0.0 if lighting_known else 0.05),
              base=params.baseload_kwh_per_day,
              rationale=("From the appliance list, width by how complete the customer said it is" if has_appliances
                         else "SAP default") + ("" if lighting_known else "; lighting share not known, wider")),
        Prior(param="heating_efficiency", kind="lognormal", mu=0.0, sigma=0.07, base=params.heating_efficiency,
              lower=0.3, upper=0.99 if params.heating_fuel != "electricity" else None,
              rationale="Tight: confounded with heat_loss_multiplier on gas data alone; a combustion system stays "
                        "below 100%"),
        Prior(param="hot_water_kwh_per_day", kind="lognormal", mu=0.0, sigma=0.2 if occupants_given else 0.35,
              base=params.hot_water_kwh_per_day,
              rationale="SAP demand for " + ("the stated number of occupants (fixed); use per person still varies"
                                             if occupants_given else "occupants estimated from floor area (not given)")),
        _behaviour_prior("heating_hours_per_day", params.heating_hours_per_day, household["hours_sd"],
                         sources["heating_hours_per_day"], 2.0, 24.0),
        Prior(param="heat_capacity_kj_per_k", kind="lognormal", mu=0.0, sigma=0.3, base=params.heat_capacity_kj_per_k,
              rationale="Thermal mass from construction (to_verify kappa); weakly identified from monthly bills: "
                        "propagate as uncertainty, don't calibrate"),
    ]
    for key in ("pv_kwp", "battery_kwh"):
        sd = household.get("ownership_sigma", {}).get(key)
        if sd == 0:
            priors.append(Prior(param=key, kind="fixed", mu=0.0, sigma=0.0, base=getattr(params, key),
                                rationale=f"Customer answer, fixed: {sources[key]}"))
        elif sd:
            priors.append(Prior(param=key, kind="lognormal", mu=0.0, sigma=sd, base=getattr(params, key),
                                rationale=f"Size not known, so uncertain: {sources[key]}"))
    if "user answer" in heater:
        priors.append(Prior(param="secondary_heating_scale", kind="fixed", mu=0.0, sigma=0.0, base=1.0,
                            rationale=f"Customer answer, fixed: {heater}"))
    elif "uncertain" in heater:
        priors.append(Prior(param="secondary_heating_scale", kind="lognormal", mu=0.0, sigma=0.5, base=1.0,
                            rationale=f"Not given, so uncertain: {heater}"))
    return priors


def _readiness(meter: Optional[Meter], prop: Property) -> Readiness:
    if meter is None or not meter.months:
        return Readiness(method="prior_only", months=0, gas_months=0, electricity_months=0, winter_months=0,
                         summer_months=0, estimated_months=0, identifiable=[], weakly_identified=[],
                         notes=["No meter readings: outputs are prior predictions only"])
    moy = [int(m[5:7]) for m in meter.months]
    gas_n = sum(v is not None for v in meter.gas_kwh)
    elec_n = sum(v is not None for v in meter.electricity_kwh)
    winter = sum(1 for m, g, e in zip(moy, meter.gas_kwh, meter.electricity_kwh) if m in WINTER and (g is not None or e is not None))
    summer = sum(1 for m, g, e in zip(moy, meter.gas_kwh, meter.electricity_kwh) if m in SUMMER and (g is not None or e is not None))
    heat_meter = gas_n if prop.heating.main_fuel == "mains_gas" else elec_n
    ident, weak, notes = [], [], []
    if winter >= 2 and summer >= 1 and heat_meter >= 6:
        ident += ["heat_loss_multiplier (with efficiency fixed by its prior)",
                  "setpoint_c (partly, via the winter slope; not if the customer gave the thermostat)"]
    else:
        notes.append("Need winter and summer months on the heating fuel to separate heating from hot water")
    if summer >= 1:
        ident.append("hot_water_kwh_per_day (summer heating-fuel use)")
    if elec_n >= 3:
        ident.append("baseload_kwh_per_day (electricity)")
    weak += ["heating_efficiency vs heat_loss_multiplier (confounded: keep efficiency prior tight)",
             "air_changes_per_hour (inside heat_loss_multiplier; don't calibrate separately)",
             "heating_hours_per_day vs setpoint_c (fix one)"]
    method = "ready" if len(meter.months) >= 12 and winter >= 3 and summer >= 2 else "weak"
    if method == "weak":
        notes.append("Fewer than 12 months or poor season coverage: expect wide posteriors")
    return Readiness(method=method, months=len(meter.months), gas_months=gas_n, electricity_months=elec_n,
                     winter_months=winter, summer_months=summer, estimated_months=sum(meter.estimated),
                     identifiable=ident, weakly_identified=weak, notes=notes)


def _layout_for(prop: Property) -> TwinLayout:
    from greentwin.twin.archetypes import build_layout
    from greentwin.twin.params import choose_archetype

    return build_layout(choose_archetype(prop), total_floor_area_m2=prop.floor_area_m2, built_form=prop.built_form)


def _band(prop: Property) -> int:
    from greentwin.twin.params import age_band_index

    return age_band_index(prop.construction_age_band, prop.year_built)


def technical_inputs(params: ExtendedTwinParams, sources: dict[str, str], prov: list[InputProvenance],
                     weather: Weather, summary: Optional[ApplianceSummary]) -> list[TechnicalInput]:
    """The technical quantities the twin generates or sources itself, each with its source and how to improve it."""
    status = {p.field: p.status for p in prov}
    epc_or_tables = "EPC measured U-value, else verified RdSAP 10 tables (Part 0 / data team)"
    heat_pump = params.heating_efficiency > 1.5
    weather_status = "synthetic" if weather.source == "synthetic" else "recorded"
    real_weather = "DataLayer.load_weather(district, months): Met Office HadUK-Grid (Part 0)"
    ghi = weather.arrays.get("ghi_kwh_m2", [])
    rows = [
        *(TechnicalInput(quantity=f"{name} U-value", value=getattr(params, key), unit="W/m2K", source=sources[key],
                         status=status.get(key, "default"), replace_with=epc_or_tables)
          for name, key in (("Wall", "wall_u"), ("Roof", "roof_u"), ("Floor", "floor_u"), ("Window", "window_u"),
                            ("Door", "door_u"))),
        TechnicalInput(quantity="Ventilation rate", value=params.air_changes_per_hour, unit="air changes/h",
                       source=sources["air_changes_per_hour"], status=status.get("air_changes_per_hour", "assumption"),
                       replace_with="SAP 2012 section 2 ventilation from chimneys, fans and draught proofing; "
                                    "an air-tightness test if available"),
        TechnicalInput(quantity="Heating seasonal COP" if heat_pump else "Heating seasonal efficiency",
                       value=params.heating_efficiency, unit="COP" if heat_pump else "fraction",
                       source=sources["heating_efficiency"], status=status.get("heating_efficiency", "default"),
                       replace_with="EPC seasonal efficiency or PCDB product record; MCS design SCOP for heat pumps"),
        TechnicalInput(quantity="Hot-water efficiency", value=params.hot_water_efficiency, unit="fraction",
                       source=sources["hot_water_efficiency"], status=status.get("hot_water_efficiency", "assumption"),
                       replace_with="SAP 2012 Table 4a/4b by system"),
        TechnicalInput(quantity="Outdoor temperature and degree days", value=None, unit="deg C, K day",
                       source=f"weather source: {weather.source}", status=weather_status, replace_with=real_weather),
        TechnicalInput(quantity="Solar irradiation (horizontal)", value=round(sum(ghi), 1) if ghi else None,
                       unit="kWh/m2 over the months", source=f"weather source: {weather.source}", status=weather_status,
                       replace_with=real_weather + "; PVGIS for tilted panels"),
        TechnicalInput(quantity="PV performance ratio", value=0.85, unit="fraction",
                       source="assumption: typical UK system losses (inverter, temperature, soiling)",
                       status="assumption", replace_with="PVGIS system-loss estimate or the installer's MCS figure"),
        TechnicalInput(quantity="Appliance load profile", value=round(sum(summary.hourly_profile_kw), 2) if summary else None,
                       unit="kWh/day over 24 hourly steps",
                       source="appliance_energy.json daily and monthly profiles" if summary
                       else "sap_2012 appendix L monthly only (no appliance list)",
                       status="estimated" if summary else "default",
                       replace_with="half-hourly shapes from London smart-meter data (planned); the customer's smart meter"),
        TechnicalInput(quantity="Battery round-trip efficiency", value=params.battery_round_trip_efficiency,
                       unit="fraction", source=sources["battery_round_trip_efficiency"],
                       status=status.get("battery_round_trip_efficiency", "assumption"),
                       replace_with="the installed battery's datasheet"),
    ]
    return rows


def build_physics_inputs(
    prop: Property,
    layout: Optional[TwinLayout] = None,
    appliances: Optional[Sequence[AppliancePlacement]] = None,
    intake: Optional[TwinIntake] = None,
    usage: Optional[Sequence[EnergyUsageMonth]] = None,
    weather_months: Optional[Sequence[WeatherMonth]] = None,
) -> PhysicsInputs:
    """Assemble the package. ``usage`` (contract meter records) wins over ``intake.meter_readings``; real
    ``weather_months`` (from Part 0) win over the synthetic stand-in."""
    # The user's heating answers replace the EPC's (EPCs can be years old)
    prop, heating_conflicts, heating_sources = apply_user_heating(
        prop, intake.heating_system if intake else None, intake.hot_water_source if intake else None)
    # Solar and battery as the customer knows them replace the EPC's
    prop, ownership = apply_user_ownership(
        prop, intake.solar_pv if intake else None, intake.pv_kwp if intake else None,
        intake.home_battery if intake else None, intake.battery_kwh if intake else None)
    geometry_flags: list[str] = []
    if intake and intake.geometry:  # the user's confirmed (or corrected) geometry wins over any fitted layout
        layout, geometry_flags = apply_geometry(intake.geometry, prop)
    occupants = intake.occupants if intake else None
    presence = intake.daytime_occupancy if intake else None
    n_guess = float(occupants) if occupants else sap_occupancy(prop.floor_area_m2)
    placements = list(appliances) if appliances is not None else (list(intake.appliances) if intake and intake.appliances else [])

    # Optional follow-ups: only the relevant ones count as asked; None = not sure
    asked = follow_up_questions(prop, intake.heating_system if intake else None, [p.type for p in placements]) if intake else []
    answers = {"plug_in_heater_use": intake.plug_in_heater_use, "led_lighting": intake.led_lighting} if intake else {}
    asked_keys = {q.key for q in asked}
    heater_use = answers.get("plug_in_heater_use") if "plug_in_heater_use" in asked_keys else None
    led = answers.get("led_lighting") if "led_lighting" in asked_keys else None
    heater_hours = HEATER_HOURS[heater_use] if heater_use else None
    has_heater = any(p.type == "electric_heater" for p in placements)
    if heater_hours and not has_heater:  # heaters in use but none placed: one typical 2 kW heater, not in a room
        placements.append(AppliancePlacement(appliance_id="plug_in_heater_stated", type="electric_heater", room_id=None,
                                             label="Plug-in heater (stated)", answers={"rating_kw": "2"}))
        has_heater = True
    summary = summarise_appliances(placements, n_guess, heater_hours) if placements else None

    params, sources = params_with_sources(
        prop, layout=layout, appliances=summary, occupants=occupants, presence=presence,
        thermostat_c=intake.thermostat_c if intake else None,
        heating_periods=intake.heating_periods if intake and intake.heating_periods else None,
        internal_walls=intake.internal_walls if intake else None,
        led_lighting=led,
        appliance_completeness=intake.appliance_completeness if intake and placements else None,
    )
    if intake:
        sources.update(ownership.sources)
    sources.update({f"user_{k}": v for k, v in heating_sources.items()})
    if heater_use:
        sources["secondary_heating_scale"] = f"user answer: plug-in heaters {heater_use.replace('_', ' ')} ({heater_hours:g} h/day) (fixed)"
    elif has_heater:
        sources["secondary_heating_scale"] = f"default {DEFAULT_HEATER_HOURS:g} h/day per plug-in heater (uncertain)"
    occ = occupancy(prop, occupants, presence, sap_occupancy(prop.floor_area_m2))
    behaviour = heating_behaviour(intake.thermostat_c if intake else None,
                                  intake.heating_periods if intake and intake.heating_periods else None, occ.presence)
    hws = hot_water_system(prop, params.heating_efficiency, (25 * occ.occupants + 36) * 4.19 * 37 / 3600, summary)

    # Meter readings
    meter: Optional[Meter] = None
    if usage:
        rows = sorted(usage, key=lambda r: r.month)
        meter = Meter(months=[r.month for r in rows], electricity_kwh=[r.electricity_import_kwh for r in rows],
                      gas_kwh=[r.gas_kwh for r in rows], estimated=[r.estimated_read for r in rows], source="usage records")
    elif intake and intake.meter_readings:
        rows = sorted(intake.meter_readings, key=lambda r: r.month)
        meter = Meter(months=[r.month for r in rows], electricity_kwh=[r.electricity_kwh for r in rows],
                      gas_kwh=[r.gas_kwh for r in rows], estimated=[r.estimated for r in rows], source="intake (user entered)")

    # Weather aligned to the meter months (or a typical year without meters)
    district = prop.location.postcode_district
    if weather_months:
        wm = list(weather_months)
    elif meter:
        wm = synthetic_months(district, meter.months)
    else:
        wm = typical_year(district)
    arrays = {k: [round(float(x), 3) for x in v] for k, v in weather_arrays(wm).items()}
    weather = Weather(months=[w.month for w in wm], arrays=arrays, source=wm[0].source, filled=filled_fields(wm))

    env_elements = [
        ("walls", params.wall_u, params.wall_area_m2, sources["wall_u"]),
        ("roof", params.roof_u, params.roof_area_m2, sources["roof_u"]),
        ("floor", params.floor_u, params.floor_area_m2, sources["floor_u"]),
        ("windows", params.window_u, params.window_area_m2, sources["window_u"]),
        ("doors", params.door_u, params.door_area_m2, sources["door_u"]),
        ("party_walls", 0.0, params.party_wall_area_m2, "U = 0 provisional (open decision)"),
    ]
    fabric = [Element(element=e, u_value=u, area_m2=a, ua_w_per_k=round(u * a, 2), source=s) for e, u, a, s in env_elements]
    vent = Ventilation(air_changes_per_hour=params.air_changes_per_hour, volume_m3=params.volume_m3,
                       heat_loss_w_per_k=round(0.33 * params.air_changes_per_hour * params.volume_m3, 2),
                       source=sources["air_changes_per_hour"])

    to_verify = sorted(k for k, s in sources.items() if "to_verify" in s)
    assumptions = sorted(k for k, s in sources.items() if "assumption" in s or "default" in s)
    flags = list(heating_conflicts) + geometry_flags + (ownership.flags if intake else [])
    if meter:
        flags += meter_fuel_conflicts(prop, meter.gas_kwh, bool(summary and summary.gas_cooking_kwh_per_day > 0))
    if weather.source == "synthetic":
        flags.append("weather is a synthetic typical year: replace with load_weather(district, months)")
    if summary and any(a.defaulted for a in summary.appliances):
        flags.append("some appliance size questions were answered by default")
    if prop.heating.main_fuel in ("oil", "lpg", "solid", "district"):
        flags.append(f"heating fuel {prop.heating.main_fuel}: TwinRun has no output field for it yet (open decision)")

    completeness = intake.appliance_completeness if intake and placements else None
    household = {"setpoint_sd": behaviour.setpoint_sd, "hours_sd": behaviour.hours_sd,
                 "baseload_sigma": BASELOAD_SIGMA.get(completeness, 0.3) if summary else None,
                 "ownership_sigma": ownership.sigma if intake else {}}
    used_layout = layout or _layout_for(prop)
    geometry_source = used_layout.geometry_source if layout is not None else "epc_fitted"
    refs = {"house_type": f"house_type:{used_layout.archetype_id}"}
    if intake:
        refs["intake"] = f"intake:{intake.intake_id}"
        if intake.epc_rrn:
            refs["epc"] = f"epc:{intake.epc_rrn}"
    answered = {"occupants": bool(intake and intake.occupants), "daytime_occupancy": bool(intake and intake.daytime_occupancy),
                "setpoint_c": bool(intake and intake.thermostat_c is not None),
                "heating_hours_per_day": bool(intake and intake.heating_periods),
                "low_energy_lighting_pct": led is not None, "secondary_heating_scale": heater_use is not None,
                "pv_kwp": bool(intake and intake.solar_pv), "battery_kwh": bool(intake and intake.home_battery),
                "appliance_completeness": completeness is not None}
    questions = {"occupants", "daytime_occupancy", "setpoint_c", "heating_hours_per_day", "pv_kwp",
                 "battery_kwh"} if intake else set()
    questions |= {"appliance_completeness"} if intake and placements else set()
    questions |= {"low_energy_lighting_pct"} if "led_lighting" in asked_keys else set()
    questions |= {"secondary_heating_scale"} if "plug_in_heater_use" in asked_keys else set()
    customer = {f: ("given" if answered[f] else "not_sure") for f in questions}
    prov = build_provenance(params.model_dump(), sources, geometry_source, refs, weather.source,
                            meter.source if meter else None, customer)
    geometry = intake.geometry if intake and intake.geometry else from_layout(used_layout)
    hl_sigma = heat_loss_prior_sigma(params, prov, geometry.confirmed_by_user)
    return PhysicsInputs(
        property_id=prop.property_id,
        created=date.today().isoformat(),
        home={"postcode_district": district, "property_type": prop.property_type.value,
              "built_form": prop.built_form.value, "age_band": prop.construction_age_band,
              "floor_area_m2": prop.floor_area_m2, "storeys": prop.storeys,
              "epc_rrn": intake.epc_rrn if intake else None},
        params=params, sources=sources,
        fuel=fuel_config(params, hot_water_efficiency=params.hot_water_efficiency).__dict__,
        fabric=fabric, ventilation=vent,
        heating_system=HeatingSystemInfo(system=prop.heating.main_system.value, fuel=prop.heating.main_fuel,
                                         efficiency=params.heating_efficiency, source=sources["heating_efficiency"]),
        hot_water=hws,
        thermal_mass=thermal_mass(prop, used_layout, intake.internal_walls if intake else None, _band(prop)),
        geometry=geometry,
        input_provenance=prov,
        provenance_summary=provenance_summary(prov, params.model_dump(), geometry_source),
        technical_inputs=technical_inputs(params, sources, prov, weather, summary),
        follow_ups=[FollowUpAnswer(question=q, answer=answers.get(q.key)) for q in asked],
        occupancy=occ.model_dump(),
        heating_behaviour=behaviour.model_dump(),
        appliances=summary,
        weather=weather,
        schedules=_schedules(weather.months, summary),
        meter=meter,
        priors=_priors(params, sources, household, summary is not None, hl_sigma),
        observation_model=ObservationModel(notes=[
            "Gas: summer months identify hot water (+ gas cooking); winter slope vs temperature identifies heat loss.",
            "Electricity: identifies baseload; for electric heating it also carries the heating signal.",
            "Estimated reads get the wider noise; months with no reading are latent.",
        ]),
        calibration_readiness=_readiness(meter, prop),
        data_quality={"to_verify": to_verify, "assumptions_or_defaults": assumptions, "flags": flags},
    )
