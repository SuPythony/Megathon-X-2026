"""Evidence/context -> initial distributions. No JAX and no meter calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from contracts.models import Property
from contracts.uncertain_twin import HouseholdRequest, PriorBundle, PriorMarginal
from greentwin.twin.electricity_inputs import timeline
from greentwin.twin.params import params_with_sources
from greentwin.twin.physics_api import PHYSICS_PARAM_KEYS
from greentwin.twin.satyarth_inputs import adapt_satyarth_package
from greentwin.twin.weather import typical_year

ROOT = Path(__file__).resolve().parents[2]
EXTRAS = {"hot_water_efficiency", "hot_water_standing_loss_kwh_per_day", "shower_electric_kwh_per_day"}
HOUSEHOLD_FIELDS = {
    "postcode_district",
    "region",
    "property_type",
    "built_form",
    "floor_area_m2",
    "storeys",
    "construction_age_band",
    "bedrooms",
    "occupants",
    "daytime_occupancy",
    "thermostat_c",
    "heating_periods",
    "heating_system",
    "hot_water_source",
    "cooking_fuel",
    "gas_cooking_kwh_per_day",
    "extra_electricity_kwh_per_day",
    "auxiliary_electricity_kwh_per_day",
    "secondary_heating_kwh_per_day",
    "cylinder_present",
    "cylinder_insulated",
    "appliances_complete",
    "appliances",
    "roof_access",
    "latitude",
    "longitude",
    "battery_charge_kw",
    "battery_discharge_kw",
    "battery_usable_fraction",
    "battery_charge_efficiency",
    "battery_discharge_efficiency",
    "battery_initial_fraction",
    "pv_inverter_kw",
    "export_limit_kw",
    "baseload_multiplier",
    "hot_water_multiplier",
    "gains_multiplier",
    "secondary_heating_kwh_monthly",
    "extra_electricity_kwh_monthly",
    "extra_gas_kwh_monthly",
    "monthly_temperature_c",
    "monthly_ghi_kwh_m2",
    "exposed_floor_area_m2",
    "flat_position",
    "has_solar_pv",
    "has_battery",
    "retained_electric_heat_fraction",
    "heating_auxiliary_power_kw",
    "water_temperature_rise_c",
    "water_specific_heat_kj_per_kgk",
    "cylinder_loss_multiplier",
    "pv_yield_multiplier",
    "battery_power_c_rate",
    "inverter_capacity_ratio",
}
SYSTEMS = {
    "gas_boiler_combi": ("gas_boiler_condensing", "mains_gas"),
    "gas_boiler_regular": ("gas_boiler_condensing", "mains_gas"),
    "electric_radiators": ("electric_direct", "electricity"),
    "district_or_communal": ("district_heating", "district"),
    "gas_boiler_condensing": ("gas_boiler_condensing", "mains_gas"),
    "gas_boiler_non_condensing": ("gas_boiler_non_condensing", "mains_gas"),
    "oil_boiler": ("oil_boiler", "oil"),
    "lpg_boiler": ("lpg_boiler", "lpg"),
    "solid_fuel": ("solid_fuel", "solid"),
    "communal_boiler": ("communal_boiler", "district"),
    "district_heating": ("district_heating", "district"),
    "air_source_heat_pump": ("air_source_heat_pump", "electricity"),
    "ground_source_heat_pump": ("ground_source_heat_pump", "electricity"),
    "electric_direct": ("electric_direct", "electricity"),
    "electric_storage": ("electric_storage", "electricity"),
    "electric_storage_hhr": ("electric_storage_hhr", "electricity"),
}


def request_hash(request):
    return hashlib.sha256(request.model_dump_json().encode()).hexdigest()


def _fallback_property(request, facts):
    required = {"postcode_district", "floor_area_m2", "property_type", "heating_system"}
    if missing := required - facts.keys():
        raise ValueError(f"Without a property/EPC package, supply these basic facts: {sorted(missing)}")
    if facts["heating_system"] not in SYSTEMS:
        raise ValueError("Unknown heating system; supply a supported equipment type")
    pt = facts["property_type"]
    template = "SH3" if pt in ("flat", "maisonette") else "SH2"
    values = json.loads((ROOT / f"contracts/examples/{template}_property.json").read_text())
    # Showcase is a schema/template, not evidence that the new household owns its equipment.
    values.update(
        property_id=request.property_id,
        customer_id=None,
        source="self_serve",
        split=None,
        property_type=pt,
        floor_area_m2=facts["floor_area_m2"],
        construction_age_band=facts.get("construction_age_band", "unknown"),
        year_built=None,
        bedrooms=facts.get("bedrooms"),
        storeys=facts.get("storeys", 1 if pt in ("flat", "maisonette", "bungalow") else 2),
        built_form=facts.get(
            "built_form",
            "not_applicable"
            if pt in ("flat", "maisonette")
            else "detached"
            if pt == "bungalow"
            else "semi_detached",
        ),
        epc={},
        current_value_estimate_gbp=None,
        value_estimate_date=None,
        controls=[],
        data_quality={},
        draught_proofed_pct=None,
        low_energy_lighting_pct=None,
    )
    values["location"] = {
        "postcode_district": facts["postcode_district"],
        "region": facts.get("region", "unknown"),
    }
    values["heating"] = {
        "main_system": SYSTEMS[facts["heating_system"]][0],
        "main_fuel": SYSTEMS[facts["heating_system"]][1],
        "hot_water": "from_main",
    }
    values["walls"] = {"type": "cavity", "insulation": "none"}
    values["roof"] = {"type": "another_dwelling_above" if pt in ("flat", "maisonette") else "pitched_loft"}
    values["floor"] = {"type": "another_dwelling_below" if pt in ("flat", "maisonette") else "solid"}
    values["glazing"] = []
    values["renewables"] = {}
    values["constraints"] = {"roof_access": facts.get("roof_access")}
    prop = Property.model_validate(values)
    if prop.floor_area_m2 <= 0 or prop.storeys < 1:
        raise ValueError("Floor area and storeys must be positive")
    return prop


def _m(centre, scale, lower=0, upper=None, *, family="lognormal", role="constant", source=None, unit="1"):
    return PriorMarginal(
        family="exact" if scale == 0 else family,
        centre=centre,
        scale=scale,
        lower=lower,
        upper=upper,
        role=role,
        unit=unit,
        source=source or "development assumption; rdsap_defaults.json to_verify",
        rationale="Conditional initial distribution; not fitted to this home's meters",
    )


def build_priors(request: HouseholdRequest, *, history=None, history_as_of=None) -> PriorBundle:
    """Build one authoritative joint specification; user numeric facts are exact."""
    facts = {k: v for k, v in request.user_inputs.items() if v is not None}
    if unknown := set(facts) - (HOUSEHOLD_FIELDS | set(PHYSICS_PARAM_KEYS) | EXTRAS):
        raise ValueError(f"Unsupported input fields (never silently ignored): {sorted(unknown)}")
    for key in (
        "has_solar_pv",
        "has_battery",
        "cylinder_present",
        "cylinder_insulated",
        "roof_access",
        "appliances_complete",
    ):
        if key in facts and not isinstance(facts[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in (
        "gas_cooking_kwh_per_day",
        "extra_electricity_kwh_per_day",
        "auxiliary_electricity_kwh_per_day",
        "secondary_heating_kwh_per_day",
    ):
        if key in facts and (not np.isfinite(float(facts[key])) or float(facts[key]) < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    immersion = next(
        (
            item
            for item in facts.get("appliances", [])
            if isinstance(item, dict) and item.get("type") == "immersion_heater"
        ),
        None,
    )
    if immersion is not None:
        facts.setdefault("hot_water_source", "electric_immersion")
        facts.setdefault("cylinder_present", True)
        if not facts["cylinder_present"]:
            raise ValueError("Immersion heater requires a hot-water cylinder")
    timeline(tuple(request.months))
    warnings = [
        "Prior predictive only: Bayesian calibration is not performed",
        "Fabric defaults and distribution widths are provisional, not NEED-fitted or empirically calibrated",
    ]
    profiles, exact_profiles = {}, []
    package = request.physics_package
    adapted = adapt_satyarth_package(package) if package else None
    prop = request.property
    if prop is None:
        base_facts = (package or {}).get("home", {})
        facts = {k: v for k, v in base_facts.items() if k in HOUSEHOLD_FIELDS} | facts
        if base_facts.get("age_band"):
            facts.setdefault("construction_age_band", base_facts["age_band"])
        if package:
            facts.setdefault("heating_system", package["heating_system"]["system"])
            if "user" in package.get("occupancy", {}).get("source", ""):
                facts.setdefault("occupants", package.get("occupancy", {}).get("occupants"))
                facts.setdefault("daytime_occupancy", package.get("occupancy", {}).get("presence"))
            if package.get("heating_behaviour", {}).get("periods"):
                facts.setdefault("heating_periods", package["heating_behaviour"]["periods"])
            # Recover only the user-stated physical value, never lock all defaults in the handoff.
            if "user" in package.get("sources", {}).get("setpoint_c", ""):
                facts.setdefault("setpoint_c", package["params"]["setpoint_c"])
        facts = {k: v for k, v in facts.items() if v is not None}
        prop = _fallback_property(request, facts)
        warnings.append("Incomplete property: archetype/construction defaults are explicitly uncertain")
    elif any(k in facts for k in ("floor_area_m2", "storeys", "built_form")):
        prop = Property.model_validate(
            prop.model_dump()
            | {k: facts[k] for k in ("floor_area_m2", "storeys", "built_form") if k in facts}
        )
    if prop.renewables.solar_thermal:
        raise ValueError("Solar thermal is unsupported; supply a supported physical scenario")
    if prop.floor_area_m2 <= 0 or prop.storeys < 1:
        raise ValueError("Floor area and storeys must be positive")
    p, sources = params_with_sources(prop)
    if adapted:
        geometry = p.model_dump()
        p = adapted.params
        if any(k in request.user_inputs for k in ("floor_area_m2", "storeys", "built_form")):
            p = p.model_copy(
                update={
                    k: geometry[k]
                    for k in (
                        "wall_area_m2",
                        "roof_area_m2",
                        "floor_area_m2",
                        "window_area_m2",
                        "door_area_m2",
                        "party_wall_area_m2",
                        "volume_m3",
                    )
                }
            )
            warnings.append(
                "Changed geometry regenerated from household facts; original handoff layout is not rescaled as a surveyed plan"
            )
        sources = package.get("sources", sources)
        profiles = {k: np.asarray(v).tolist() for k, v in adapted.profiles.items()}
        if adapted.weather and [w.month for w in adapted.weather] != request.months:
            raise ValueError("Handoff months must match request; prepare/reindex explicitly")
    base = p.model_dump()
    facts.setdefault("postcode_district", prop.location.postcode_district)
    facts.setdefault("region", prop.location.region)
    facts.setdefault("floor_area_m2", prop.floor_area_m2)
    facts.setdefault("property_type", prop.property_type.value)
    facts.setdefault("storeys", prop.storeys)
    facts.setdefault("built_form", prop.built_form.value)
    facts.setdefault("latitude", prop.location.lat)
    facts.setdefault("longitude", prop.location.lon)
    facts.setdefault("heating_system", prop.heating.main_system.value)
    if facts["heating_system"] not in SYSTEMS:
        raise ValueError("Unsupported heating system")
    system, heating_fuel = SYSTEMS[facts["heating_system"]]
    if system != prop.heating.main_system.value or heating_fuel != p.heating_fuel:
        from greentwin.twin.params import rdsap

        base["heating_efficiency"] = rdsap()["heating_efficiency"][system]
    base["heating_fuel"] = heating_fuel
    hot_source = facts.get("hot_water_source", prop.heating.hot_water)
    hw_fuel = heating_fuel
    if hot_source in ("electric_immersion", "instantaneous_electric", "heat_pump_cylinder"):
        hw_fuel = "electricity"
    if hot_source == "heat_pump_cylinder":
        hw_eff = 2.0
    elif hw_fuel == "electricity" and hot_source in ("electric_immersion", "instantaneous_electric"):
        hw_eff = 1.0
    else:
        hw_eff = 2.0 if "heat_pump" in system else base["heating_efficiency"]
    if adapted and "hot_water_source" not in facts and "heating_system" not in request.user_inputs:
        hw_fuel = adapted.fuel.hot_water_fuel
        hw_eff = adapted.fuel.hot_water_efficiency
    equipment = {
        "heating_system": system,
        "heating_fuel": heating_fuel,
        "hot_water_fuel": hw_fuel,
        "hot_water_source": hot_source,
        "fuel_energy_basis": "gross CV for mains gas; purchased kWh for all fuels",
    }
    # Explicit values in supplied Property are facts; inferred defaults are not.
    for field, value in (
        ("wall_u", prop.walls.u_value),
        ("roof_u", prop.roof.u_value),
        ("floor_u", prop.floor.u_value),
        ("heating_efficiency", prop.heating.seasonal_efficiency),
    ):
        if value is not None and not (
            field == "heating_efficiency" and system != prop.heating.main_system.value
        ):
            facts.setdefault(field, value)
    if request.property:
        for key in ("pv_kwp", "battery_kwh", "pv_azimuth_deg", "pv_tilt_deg"):
            value = getattr(prop.renewables, key)
            if value is not None and key in prop.renewables.model_fields_set:
                facts.setdefault(key, value)
    if package:
        for key in ("pv_kwp", "battery_kwh"):
            facts.setdefault(key, package["params"][key])
    marginals = {}
    for field in ("wall_u", "roof_u", "floor_u", "window_u", "door_u"):
        marginals[field] = _m(base[field], 0 if base[field] == 0 else 0.22, unit="W/m2K")
    for field, scale, lo, hi in (
        ("air_changes_per_hour", 0.25, 0.1, 3),
        ("thermal_bridging_y", 0.25, 0.001, 0.5),
        ("solar_gain_factor", 0.15, 0.1, 1.8),
    ):
        marginals[field] = _m(base[field], scale, lo, hi)
    # No redundant heat-loss multiplier uncertainty on top of independently uncertain fabric.
    marginals["heat_loss_multiplier"] = _m(1, 0, 0.01, 5)
    hp = "heat_pump" in system
    direct = heating_fuel == "electricity" and not hp
    marginals["heating_efficiency"] = _m(
        base["heating_efficiency"],
        0 if direct else 0.12 if hp else 45,
        1 if hp else 0.1,
        7 if hp else 1,
        family="lognormal" if hp or direct else "beta",
    )
    hw_hp = hw_eff > 1
    marginals["hot_water_efficiency"] = _m(
        hw_eff,
        0.15 if hw_hp else 0 if hw_eff == 1 else 40,
        1 if hw_hp else 0.1,
        7 if hw_hp else 1,
        family="lognormal" if hw_hp or hw_eff == 1 else "beta",
    )
    marginals["geometry_aspect_multiplier"] = _m(1, 0.12, 0.5, 2, role="missing_input")
    marginals["volume_multiplier"] = _m(1, 0.06, 0.7, 1.3, role="missing_input")
    marginals["opening_multiplier"] = _m(1, 0.12, 0.5, 1.5, role="missing_input")
    marginals["occupants"] = PriorMarginal(
        family="categorical",
        categories={str(i): v for i, v in [(1, 0.2), (2, 0.35), (3, 0.2), (4, 0.15), (5, 0.1)]},
        source="development household prior",
        rationale="Occupants unknown; categorical count, no fabricated census fit",
    )
    marginals["presence"] = PriorMarginal(
        family="categorical",
        categories={"out_all_day": 0.25, "part_day": 0.45, "home_all_day": 0.30},
        source="development household prior",
        rationale="Presence unknown; categorical mixture",
    )
    marginals["setpoint_c"] = _m(
        base["setpoint_c"], 1.5, 10, 28, family="normal", role="missing_input", unit="degC"
    )
    marginals["heating_hours_per_day"] = _m(
        base["heating_hours_per_day"], 2, 0, 24, family="normal", role="missing_input", unit="h/day"
    )
    marginals["appliance_demand_multiplier"] = _m(1, 0.25, 0.2, 3, role="missing_input")
    marginals["lighting_demand_multiplier"] = _m(1, 0.2, 0.2, 3, role="missing_input")
    marginals["water_use_multiplier"] = _m(1, 0.22, 0.2, 3, role="missing_input")
    marginals["retained_electric_heat_fraction"] = _m(0.85, 70, 0.05, 0.999, family="beta")
    marginals["heating_auxiliary_power_kw"] = _m(0.04, 0.3, 0.005, 0.2, unit="kW")
    marginals["cylinder_loss_multiplier"] = _m(1, 0.25, 0.2, 3)
    cylinder = facts.get("cylinder_present")
    if cylinder is None and package:
        cylinder = package.get("hot_water", {}).get("standing_loss_kwh_per_day", 0) > 0
    if cylinder is None and facts["heating_system"] == "gas_boiler_combi":
        cylinder = False
    marginals["cylinder_presence"] = PriorMarginal(
        family="categorical",
        categories={"yes" if cylinder else "no": 1.0} if cylinder is not None else {"no": 0.5, "yes": 0.5},
        source="recorded cylinder fact"
        if cylinder is not None
        else "development cylinder-presence assumption",
        rationale="Unknown stored hot water is a mixture; combi is a physical no-cylinder condition",
    )
    marginals["pv_azimuth_deg"] = _m(
        base["pv_azimuth_deg"], 45, 0, 360, family="normal", role="missing_input", unit="degree"
    )
    marginals["pv_tilt_deg"] = _m(
        base["pv_tilt_deg"], 8, 0, 90, family="normal", role="missing_input", unit="degree"
    )
    marginals["cooking_fuel"] = PriorMarginal(
        family="categorical",
        categories={"electricity": 0.5, "mains_gas": 0.5}
        if prop.heating.main_fuel == "mains_gas"
        else {"electricity": 1},
        source="development cooking-fuel assumption",
        rationale="Unknown cooking fuel remains a mixture",
    )
    if request.property is None and facts["property_type"] in ("flat", "maisonette"):
        marginals["flat_position"] = PriorMarginal(
            family="categorical",
            categories={"ground": 0.2, "middle": 0.6, "top": 0.2},
            source="development flat-position assumption",
            rationale="Unknown exposure remains uncertain; not a measured stock distribution",
        )
        marginals["roof_u"] = _m(1, 0.4, 0.05, 3, unit="W/m2K")
        marginals["floor_u"] = _m(0.6, 0.3, 0.05, 2, unit="W/m2K")
    for flag, capacity, mean in (("has_solar_pv", "pv_kwp", 3.5), ("has_battery", "battery_kwh", 5)):
        if capacity not in facts:
            categories = {"no": 0.9, "yes": 0.1}
            if flag in facts:
                categories = {"yes" if facts[flag] else "no": 1.0}
            marginals[flag] = PriorMarginal(
                family="categorical",
                categories=categories,
                source="development equipment-presence assumption",
                rationale="Unknown ownership is not a known zero",
            )
            marginals[f"installed_{capacity}"] = _m(
                mean, 0.3, 0.5, 12 if capacity == "pv_kwp" else 30, role="missing_input"
            )
            warnings.append(
                f"{capacity} unknown: development ownership/capacity mixture; confirm this simple fact"
            )
        elif flag in facts and (float(facts[capacity]) > 0) != bool(facts[flag]):
            raise ValueError(f"Conflicting {flag} and {capacity}")
    marginals["water_temperature_rise_c"] = _m(37, 3, 20, 55, family="normal", unit="K")
    marginals["water_specific_heat_kj_per_kgk"] = _m(
        4.19, 0, unit="kJ/kg/K", source="Existing SAP-style preparation; water heat capacity approximation"
    )
    for field, centre, sd in (("pv_yield_multiplier", 1, 0.12),):
        marginals[field] = _m(centre, sd, 0.2, 3, role="missing_input")
    if request.electricity_series:
        marginals["pv_yield_multiplier"] = _m(
            1, 0, source="Supplied interval PV series held fixed", role="missing_input"
        )
    for name, centre, lo, hi in (
        ("battery_usable_fraction", 0.90, 0.5, 0.999),
        ("battery_charge_efficiency", 0.95, 0.6, 0.999),
        ("battery_discharge_efficiency", 0.95, 0.6, 0.999),
    ):
        marginals[name] = _m(centre, 150, lo, hi, family="beta")
    marginals["battery_power_c_rate"] = _m(0.5, 0.2, 0.1, 2)
    marginals["inverter_capacity_ratio"] = _m(1, 0.12, 0.5, 1.5)
    # Lock supplied numbers, including exact physical overrides, after assigning defaults.
    aliases = {"thermostat_c": "setpoint_c", "daytime_occupancy": "presence"}
    for key, value in facts.items():
        target = aliases.get(key, key)
        if key in ("has_solar_pv", "has_battery"):
            continue  # boolean answer already converted to exact yes/no categorical presence
        if target in marginals and marginals[target].family == "categorical":
            if target == "occupants" and (float(value) != int(value) or not 1 <= int(value) <= 12):
                raise ValueError("Occupants must be an integer in 1..12")
            if str(value) not in marginals[target].categories and target != "occupants":
                raise ValueError(f"Invalid {key}")
            marginals[target] = marginals[target].model_copy(
                update={"categories": {str(value): 1.0}, "source": "user/recorded exact fact"}
            )
        elif (
            target in marginals or target in PHYSICS_PARAM_KEYS or target in EXTRAS
        ) and target != "floor_area_m2":
            old = marginals.get(target)
            marginals[target] = _m(
                float(value),
                0,
                old.lower if old else 0,
                old.upper if old else None,
                source="user/recorded exact fact",
                role=old.role if old else "missing_input",
                unit=old.unit if old else "1",
            )
    if facts.get("heating_periods") and "heating_hours_per_day" not in facts:
        from contracts.electricity import ElectricityIntake

        periods = ElectricityIntake(heating_periods=facts["heating_periods"]).heating_periods
        duration = sum((end - start) % 24 or 24 for start, end in periods)
        marginals["heating_hours_per_day"] = _m(duration, 0, 0, 24, source="Exact supplied heating periods")
    if facts.get("roof_access") is False and float(facts.get("pv_kwp", base["pv_kwp"])) > 0:
        raise ValueError("PV installation conflicts with explicitly unavailable roof")
    # Requested dates, not fixed 2025 appliance day counts.
    if request.weather:
        weather = request.weather
        if [w.month for w in weather] != request.months:
            raise ValueError("Weather must cover the exact requested months")
    elif adapted:
        weather = adapted.weather
    else:
        weather = [
            typical_year(facts["postcode_district"], int(m[:4]))[int(m[5:]) - 1] for m in request.months
        ]
    if any(w.mean_temp_c is None or w.ghi_kwh_m2 is None for w in weather):
        raise ValueError(
            "Monthly temperature and GHI must be explicit; provide context or labelled development weather"
        )
    if any(w.source == "synthetic" for w in weather) and not request.allow_development_weather:
        raise ValueError("Synthetic weather requires allow_development_weather=True")
    if any(w.area_code != facts["postcode_district"] for w in weather):
        raise ValueError("Weather area must match household district")
    for w in weather:
        mm = w.month[5:]
        if f"mean_temp_c:{mm}" not in marginals:
            estimated = w.source == "synthetic"
            marginals[f"mean_temp_c:{mm}"] = _m(
                w.mean_temp_c,
                1.5 if estimated else 0,
                -40,
                50,
                family="normal",
                role="context",
                source=f"Monthly weather: {w.source}",
                unit="degC",
            )
            marginals[f"ghi_kwh_m2:{mm}"] = _m(
                w.ghi_kwh_m2,
                0.15 if estimated and w.ghi_kwh_m2 else 0,
                0,
                800,
                role="context",
                source=f"Monthly weather: {w.source}",
                unit="kWh/m2",
            )
    application_audit = {}
    if history is not None:
        from greentwin.twin.application_history import inform_priors

        for field, width in (("baseload_kwh_per_day", 0.25), ("hot_water_kwh_per_day", 0.22)):
            if field not in marginals:
                marginals[field] = _m(base[field], width, 0, role="missing_input", unit="kWh/day")
        application_audit = inform_priors(request, facts, marginals, history, as_of=history_as_of)
        warnings.append(
            "Application answers inform initial priors; unmeasured constants retain provisional defaults"
        )
    context_audit = []
    pooled_variables = set()
    # A district summary and its regional parent may describe the same homes.
    # Select the most specific recent eligible cohort rather than counting both.
    records = sorted(
        request.context,
        key=lambda r: (
            bool(r.area_code),
            bool(r.region),
            bool(r.property_type),
            bool(r.heating_system),
            r.year,
            len(r.values),
            r.source,
        ),
        reverse=True,
    )
    for record in records:
        variable = (
            f"{record.variable}:{record.month_of_year:02d}"
            if record.month_of_year is not None
            else record.variable
        )
        reason = None
        if record.split != "train":
            reason = "held-out evidence excluded"
        elif record.year > request.as_of_year:
            reason = "future evidence excluded"
        elif request.property_id in record.property_ids:
            reason = "same-home evidence excluded from population prior"
        elif record.status == "public" and "synthetic" in record.source.lower():
            reason = "synthetic evidence cannot be labelled public"
        elif record.area_code and record.area_code != facts["postcode_district"]:
            reason = "different area"
        elif record.region and record.region != facts["region"]:
            reason = "different region"
        elif record.property_type and record.property_type != facts["property_type"]:
            reason = "different home type"
        elif record.heating_system and record.heating_system != system:
            reason = "different equipment"
        elif variable not in marginals:
            reason = "unknown prior variable"
        elif marginals[variable].source == "user/recorded exact fact":
            reason = "supplied fact remains fixed"
        elif variable in pooled_variables:
            reason = "overlapping aggregate excluded; more specific eligible cohort selected"
        values = np.asarray(record.values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Context values must be finite")
        audit = {
            "variable": variable,
            "source": record.source,
            "status": record.status,
            "year": record.year,
            "n": len(values),
            "reason": reason,
        }
        if reason is None:
            old = marginals[variable]
            if old.family in ("exact", "categorical"):
                audit["reason"] = "structural or categorical prior cannot be overwritten by numeric rows"
            else:
                if (old.lower is not None and (values < old.lower).any()) or (
                    old.upper is not None and (values > old.upper).any()
                ):
                    raise ValueError(f"Context outside physical domain: {record.variable}")
                weight = len(values) / (len(values) + 20) * np.exp(-(request.as_of_year - record.year) / 10)
                mean = (1 - weight) * old.centre + weight * float(values.mean())
                # Preserve a heterogeneity floor; sparse or identical aggregate rows are not exact facts.
                if old.family == "normal":
                    prior_sd = old.scale
                elif old.family == "lognormal":
                    prior_sd = old.centre * np.sqrt(np.expm1(old.scale**2))
                elif old.family == "beta":
                    fraction = (old.centre - old.lower) / (old.upper - old.lower)
                    prior_sd = (old.upper - old.lower) * np.sqrt(fraction * (1 - fraction) / (old.scale + 1))
                else:
                    prior_sd = (old.upper - old.lower) / np.sqrt(12)
                sd = max(float(values.std(ddof=1)), prior_sd * 0.5)
                var = (
                    (1 - weight) * prior_sd**2
                    + weight * sd**2
                    + weight * (1 - weight) * (old.centre - values.mean()) ** 2
                )
                marginals[variable] = _m(
                    float(mean),
                    float(np.sqrt(var)),
                    old.lower,
                    old.upper,
                    family="normal",
                    role=old.role,
                    source=f"partial pooling: {record.source} ({record.status})",
                    unit=old.unit,
                )
                audit["pooling_weight"] = float(weight)
                pooled_variables.add(variable)
        context_audit.append(audit)
    for key in (
        "baseload_multiplier",
        "hot_water_multiplier",
        "gains_multiplier",
        "secondary_heating_kwh_monthly",
        "extra_electricity_kwh_monthly",
        "extra_gas_kwh_monthly",
    ):
        if key in facts:
            values = np.asarray(facts[key], dtype=float)
            if values.shape != (len(request.months),) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid supplied monthly {key}")
            profiles[key] = values.tolist()
            exact_profiles.append(key)
    if "exposed_floor_area_m2" in facts:
        marginals["floor_area_m2"] = _m(
            float(facts["exposed_floor_area_m2"]), 0, source="user/recorded exact fact"
        )
    base.update({k: v for k, v in facts.items() if k in PHYSICS_PARAM_KEYS and k != "floor_area_m2"})
    return PriorBundle(
        property_id=request.property_id,
        facts=facts,
        marginals=marginals,
        equipment=equipment,
        base_parameters=base,
        weather=weather,
        profile_centres=profiles,
        profile_exact=exact_profiles,
        context_audit=context_audit,
        request_hash=request_hash(request),
        application_history=application_audit,
        warnings=warnings,
        dependencies={
            "geometry": "floor area fixed; footprint aspect changes exposed perimeter jointly",
            "internal_gains": "occupants + sampled electricity x retained heat fraction",
            "dhw": "water use x mass x heat capacity x temperature rise; subtract electric-shower share",
            "monthly_profiles": "AR(1) rho=0.65 log variation; monthly electrical allocation conserves energy",
            "scenarios": "stable named random streams preserve common random numbers",
        },
    )
