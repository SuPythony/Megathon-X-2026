"""The physics inputs package: everything the physics model and the Bayesian calibration need for one home, in one
JSON document (``PhysicsInputs``, version ``physics_inputs/0.1``).

Contents: fabric (U-values x areas), ventilation, heating system, hot-water system (separate equipment, fuel,
efficiency), occupancy and heating behaviour, appliance consumption schedules and the matching internal gains,
weather aligned to the meter months, monthly meter readings (missing months marked, never zero), priors for the
calibrated parameters, a suggested observation model and a calibration-readiness check. Every value carries its source.

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
    HotWaterSystem,
    apply_user_heating,
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
    """Prior for one calibrated parameter. kind: lognormal -> value = base x exp(N(mu, sigma)); normal -> base + N."""

    param: str
    kind: Literal["lognormal", "normal_add"]
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


class PhysicsInputs(BaseModel):
    inputs_version: str = INPUTS_VERSION
    property_id: str
    created: str
    home: dict[str, str | float | int | None]
    params: ExtendedTwinParams
    sources: dict[str, str]
    fuel: dict[str, str]
    fabric: list[Element]
    ventilation: Ventilation
    heating_system: HeatingSystemInfo
    hot_water: HotWaterSystem
    thermal_mass: ThermalMass
    geometry: GeometryConfirmation = Field(description="The geometry used, and whether the user confirmed it")
    input_provenance: list[InputProvenance]
    provenance_summary: ProvenanceSummary
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
        return FuelConfig(heating_fuel=self.fuel["heating_fuel"], hot_water_fuel=self.fuel["hot_water_fuel"])

    def physics_params(self) -> dict[str, float]:
        keys = [*PHYSICS_PARAM_KEYS, *EXTRA_PARAM_KEYS]
        return {k: float(getattr(self.params, k)) for k in keys}

    def physics_timeseries(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v, dtype=np.float32) for k, v in self.weather.arrays.items()}
        out.update({k: np.asarray(v, dtype=np.float32) for k, v in self.schedules.items()})
        return out

    def prior_table(self) -> dict[str, tuple[str, float, float]]:
        """Priors in the form simulate.prior_samples takes."""
        return {p.param: (p.kind, p.mu, p.sigma) for p in self.priors}


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


def _priors(params: ExtendedTwinParams, sources: dict[str, str], household: dict, has_appliances: bool,
            hl_sigma: float = 0.25) -> list[Prior]:
    user_stat = "user" in sources.get("setpoint_c", "")
    return [
        Prior(param="heat_loss_multiplier", kind="lognormal", mu=0.0, sigma=hl_sigma, base=params.heat_loss_multiplier,
              lower=0.4, upper=2.0, rationale="Scales the fabric + ventilation loss; width set by input provenance "
              "(unverified U-values and unconfirmed geometry widen it)"),
        Prior(param="setpoint_c", kind="normal_add", mu=0.0, sigma=household["setpoint_sd"], base=params.setpoint_c,
              lower=14.0, upper=24.0, rationale="Mean internal temperature: " + ("user thermostat" if user_stat else "default")),
        Prior(param="baseload_kwh_per_day", kind="lognormal", mu=0.0, sigma=0.2 if has_appliances else 0.3,
              base=params.baseload_kwh_per_day, rationale="From the appliance list" if has_appliances else "SAP default"),
        Prior(param="heating_efficiency", kind="lognormal", mu=0.0, sigma=0.07, base=params.heating_efficiency,
              rationale="Tight: confounded with heat_loss_multiplier on gas data alone"),
        Prior(param="hot_water_kwh_per_day", kind="lognormal", mu=0.0, sigma=0.3, base=params.hot_water_kwh_per_day,
              rationale="SAP occupancy-based demand; identified by summer gas/electricity"),
        Prior(param="heating_hours_per_day", kind="normal_add", mu=0.0, sigma=household["hours_sd"],
              base=params.heating_hours_per_day, lower=2.0, upper=24.0,
              rationale="Optional: weakly identified with monthly data; fix unless the user gave a schedule"),
        Prior(param="heat_capacity_kj_per_k", kind="lognormal", mu=0.0, sigma=0.3, base=params.heat_capacity_kj_per_k,
              rationale="Thermal mass from construction (to_verify kappa); weakly identified from monthly bills: "
                        "propagate as uncertainty, don't calibrate"),
    ]


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
        ident += ["heat_loss_multiplier (with efficiency fixed by its prior)", "setpoint_c (partly, via the winter slope)"]
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
    geometry_flags: list[str] = []
    if intake and intake.geometry:  # the user's confirmed (or corrected) geometry wins over any fitted layout
        layout, geometry_flags = apply_geometry(intake.geometry, prop)
    occupants = intake.occupants if intake else None
    presence = intake.daytime_occupancy if intake else None
    n_guess = float(occupants) if occupants else sap_occupancy(prop.floor_area_m2)
    placements = list(appliances) if appliances is not None else (list(intake.appliances) if intake and intake.appliances else [])
    summary = summarise_appliances(placements, n_guess) if placements else None

    params, sources = params_with_sources(
        prop, layout=layout, appliances=summary, occupants=occupants, presence=presence,
        thermostat_c=intake.thermostat_c if intake else None,
        heating_periods=intake.heating_periods if intake and intake.heating_periods else None,
        internal_walls=intake.internal_walls if intake else None,
    )
    sources.update({f"user_{k}": v for k, v in heating_sources.items()})
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
    flags = list(heating_conflicts) + geometry_flags
    if meter:
        flags += meter_fuel_conflicts(prop, meter.gas_kwh, bool(summary and summary.gas_cooking_kwh_per_day > 0))
    if weather.source == "synthetic":
        flags.append("weather is a synthetic typical year: replace with load_weather(district, months)")
    if summary and any(a.defaulted for a in summary.appliances):
        flags.append("some appliance size questions were answered by default")
    if prop.heating.main_fuel in ("oil", "lpg", "solid", "district"):
        flags.append(f"heating fuel {prop.heating.main_fuel}: TwinRun has no output field for it yet (open decision)")

    household = {"setpoint_sd": behaviour.setpoint_sd, "hours_sd": behaviour.hours_sd}
    used_layout = layout or _layout_for(prop)
    geometry_source = used_layout.geometry_source if layout is not None else "epc_fitted"
    refs = {"house_type": f"house_type:{used_layout.archetype_id}"}
    if intake:
        refs["intake"] = f"intake:{intake.intake_id}"
        if intake.epc_rrn:
            refs["epc"] = f"epc:{intake.epc_rrn}"
    prov = build_provenance(params.model_dump(), sources, geometry_source, refs, weather.source,
                            meter.source if meter else None)
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
        fuel=fuel_config(params).__dict__,
        fabric=fabric, ventilation=vent,
        heating_system=HeatingSystemInfo(system=prop.heating.main_system.value, fuel=prop.heating.main_fuel,
                                         efficiency=params.heating_efficiency, source=sources["heating_efficiency"]),
        hot_water=hws,
        thermal_mass=thermal_mass(prop, used_layout, intake.internal_walls if intake else None, _band(prop)),
        geometry=geometry,
        input_provenance=prov,
        provenance_summary=provenance_summary(prov, params.model_dump(), geometry_source),
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
