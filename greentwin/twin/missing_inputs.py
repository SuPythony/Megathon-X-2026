"""Joint uncertain household inputs, prepared without importing the simulator/JAX."""

from __future__ import annotations

import calendar
import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import beta

from contracts.electricity import ElectricityIntake
from contracts.models import TwinParams
from contracts.summarise import summarise
from contracts.uncertain_twin import GeneratedInputs, HouseholdRequest, MomentDistribution, PriorBundle
from greentwin.twin.electricity_inputs import prepare_electricity_inputs
from greentwin.twin.params import sap_occupancy
from greentwin.twin.priors import build_priors


def input_unit(name):
    if name.endswith("_area_m2"):
        return "m2"
    if name.endswith("_u") or name == "thermal_bridging_y":
        return "W/m2K"
    if name.endswith("_kwh_per_day"):
        return "kWh/day"
    if name.endswith("_kwh_monthly"):
        return "kWh"
    if name.endswith("_kw"):
        return "kW"
    return {
        "setpoint_c": "degC",
        "mean_temp_c": "degC",
        "ghi_kwh_m2": "kWh/m2",
        "volume_m3": "m3",
        "internal_gains_w": "W",
        "pv_kwp": "kWp",
        "battery_kwh": "kWh",
        "heating_hours_per_day": "h/day",
        "days_in_month": "day",
        "hdd_15_5": "K day",
        "air_changes_per_hour": "1/h",
        "pv_azimuth_deg": "degree",
        "pv_tilt_deg": "degree",
    }.get(name, "1")


def moments(values, unit="1"):
    d = summarise(values, unit)
    return MomentDistribution(
        **d.model_dump(), variance=d.std**2, monte_carlo_standard_error=d.std / np.sqrt(d.n_samples)
    )


def stream(seed, property_id, name):
    digest = hashlib.sha256(f"{property_id}:{name}".encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([seed, *np.frombuffer(digest[:16], dtype="<u4")]))


def sample_marginal(spec, uniforms):
    """Inverse-CDF sampling, retaining common uniforms across modified scenarios."""
    u = np.clip(uniforms, 1e-12, 1 - 1e-12)
    if spec.family == "exact":
        return np.full(len(u), spec.centre)
    if spec.family == "categorical":
        labels, probs = zip(*spec.categories.items())
        return np.asarray(labels)[np.minimum(np.searchsorted(np.cumsum(probs), u), len(labels) - 1)]
    if spec.family == "uniform":
        return spec.lower + u * (spec.upper - spec.lower)
    if spec.family == "beta":
        mean = (spec.centre - spec.lower) / (spec.upper - spec.lower)
        return spec.lower + (spec.upper - spec.lower) * beta.ppf(
            u, mean * spec.scale, (1 - mean) * spec.scale
        )
    if spec.family == "lognormal":
        # centre denotes the UNTRUNCATED arithmetic mean, not exp(log-median).
        mu, sd = np.log(spec.centre) - 0.5 * spec.scale**2, spec.scale
        lo = 0 if spec.lower is None or spec.lower <= 0 else ndtr((np.log(spec.lower) - mu) / sd)
        hi = 1 if spec.upper is None else ndtr((np.log(spec.upper) - mu) / sd)
        return np.exp(mu + sd * ndtri(lo + u * (hi - lo)))
    lo = 0 if spec.lower is None else ndtr((spec.lower - spec.centre) / spec.scale)
    hi = 1 if spec.upper is None else ndtr((spec.upper - spec.centre) / spec.scale)
    return spec.centre + spec.scale * ndtri(lo + u * (hi - lo))


@dataclass
class InputSamples:
    """Internal joint samples; serialise summaries separately from large arrays."""

    request: HouseholdRequest
    prior: PriorBundle
    parameters: dict[str, np.ndarray]
    constants: dict[str, np.ndarray]
    weather: dict[str, np.ndarray]
    profiles: dict[str, np.ndarray]
    electricity: list
    summary: GeneratedInputs
    latent: dict[str, np.ndarray]


def _temporal(seed, identity, name, n, months, sigma):
    innovations = stream(seed, identity, name).normal(size=(n, months))
    values = np.empty_like(innovations)
    values[:, 0] = innovations[:, 0]
    for i in range(1, months):
        values[:, i] = 0.65 * values[:, i - 1] + np.sqrt(1 - 0.65**2) * innovations[:, i]
    return np.exp(sigma * values - 0.5 * sigma**2)


def generate_missing_inputs(
    request: HouseholdRequest, n_samples=128, *, prior=None, prepare_electricity=False
) -> InputSamples:
    """Every missing numeric input has a distribution; exact answers remain fixed."""
    if not isinstance(n_samples, int) or not 2 <= n_samples <= 10000:
        raise ValueError("n_samples must be an integer in 2..10000")
    prior = prior or build_priors(request)
    from greentwin.twin.priors import request_hash

    if prior.request_hash != request_hash(request):
        raise ValueError("Prior belongs to another request revision")
    draws = {
        key: sample_marginal(spec, stream(request.seed, request.property_id, key).uniform(size=n_samples))
        for key, spec in prior.marginals.items()
    }
    facts, base = prior.facts, prior.base_parameters
    m = len(request.months)
    p = {
        key: np.full(n_samples, float(value))
        for key, value in base.items()
        if isinstance(value, (int, float))
    }
    for flag, capacity in (("has_solar_pv", "pv_kwp"), ("has_battery", "battery_kwh")):
        if flag in draws:
            p[capacity] = np.where(draws[flag] == "yes", draws[f"installed_{capacity}"], 0)
    for key in p:
        if key in draws:
            p[key] = draws[key].astype(float)
    for key in ("hot_water_efficiency", "hot_water_standing_loss_kwh_per_day", "shower_electric_kwh_per_day"):
        if key in draws:
            p[key] = draws[key].astype(float)
    p["hot_water_efficiency"] = draws["hot_water_efficiency"].astype(float)
    # Geometry: keep total floor area fixed and scale perimeter from a common footprint aspect.
    aspect = draws["geometry_aspect_multiplier"]
    perimeter_factor = (np.sqrt(aspect) + 1 / np.sqrt(aspect)) / 2
    opening = draws["opening_multiplier"]
    p["window_area_m2"] = base["window_area_m2"] * opening
    p["door_area_m2"] = base["door_area_m2"] * opening
    gross = (base["wall_area_m2"] + base["window_area_m2"] + base["door_area_m2"]) * perimeter_factor
    p["wall_area_m2"] = np.maximum(gross - p["window_area_m2"] - p["door_area_m2"], 0)
    p["party_wall_area_m2"] = base["party_wall_area_m2"] * perimeter_factor
    p["volume_m3"] = base["volume_m3"] * draws["volume_multiplier"]
    if "flat_position" in draws:
        footprint = float(facts["floor_area_m2"]) / float(facts["storeys"])
        p["roof_area_m2"] = np.where(draws["flat_position"] == "top", footprint, 0)
        p["floor_area_m2"] = np.where(draws["flat_position"] == "ground", footprint, 0)
    occupants = draws["occupants"].astype(float)
    occ_scale = (occupants / sap_occupancy(float(facts["floor_area_m2"]))) ** 0.6
    lighting = (
        base["baseload_kwh_per_day"]
        * base["lighting_share_of_baseload"]
        * draws["lighting_demand_multiplier"]
    )
    appliances = (
        base["baseload_kwh_per_day"]
        * (1 - base["lighting_share_of_baseload"])
        * occ_scale
        * draws["appliance_demand_multiplier"]
    )
    package = request.physics_package or {}
    if package.get("appliances"):
        # Catalogue total is a prior centre, not a measured fact.
        appliances = package["appliances"]["baseload_kwh_per_day"] * draws["appliance_demand_multiplier"]
    if facts.get("appliances"):
        from greentwin.twin.appliance_inputs import appliance_centres

        appliance_base, appliance_gas, appliance_shower, appliance_outdoor, appliance_secondary = (
            appliance_centres(facts["appliances"], occupants)
        )
        appliances = appliance_base * draws["appliance_demand_multiplier"]
    else:
        appliance_gas, appliance_shower = np.zeros(n_samples), np.zeros(n_samples)
        appliance_outdoor, appliance_secondary = np.zeros(n_samples), np.zeros(n_samples)
        if not package.get("appliances"):
            cook = (
                (35 + 7 * sap_occupancy(float(facts["floor_area_m2"])))
                * 8.76
                / 365
                * occ_scale
                * draws["appliance_demand_multiplier"]
            )
            gas_mask = draws["cooking_fuel"] == "mains_gas"
            appliances = np.maximum(appliances - np.where(gas_mask, cook, 0), 0)
            appliance_gas = np.where(gas_mask, cook * 1.5, 0)
    p["baseload_kwh_per_day"] = appliances + lighting
    p["lighting_share_of_baseload"] = lighting / np.maximum(appliances + lighting, 1e-12)
    retained = draws["retained_electric_heat_fraction"]
    p["internal_gains_w"] = 20 * occupants + retained * p["baseload_kwh_per_day"] * 1000 / 24
    water = (25 * occupants + 36) * draws["water_use_multiplier"]
    dhw = water * draws["water_specific_heat_kj_per_kgk"] * draws["water_temperature_rise_c"] / 3600
    shower_centre = float(package.get("hot_water", {}).get("shower_electric_kwh_per_day", 0))
    shower = np.maximum(appliance_shower, shower_centre) * draws["water_use_multiplier"]
    shower = np.minimum(shower, dhw)
    p["shower_electric_kwh_per_day"] = shower
    p["hot_water_kwh_per_day"] = dhw - shower
    cylinder = draws["cylinder_presence"] == "yes"
    loss = float(
        package.get("hot_water", {}).get(
            "standing_loss_kwh_per_day", 1 if facts.get("cylinder_insulated", True) else 3
        )
    )
    p["hot_water_standing_loss_kwh_per_day"] = np.where(cylinder, loss * draws["cylinder_loss_multiplier"], 0)
    for appliance in facts.get("appliances", []):
        if appliance.get("type") == "immersion_heater":
            jacket = appliance.get("answers", {}).get("jacket", "factory")
            loss = {"factory": 1, "jacket": 1.6, "none": 3}[jacket]
            p["hot_water_standing_loss_kwh_per_day"] = loss * draws["cylinder_loss_multiplier"]
    # Explicit numerical physics answers override every inferred dependency.
    for key in p:
        if (
            key in prior.marginals
            and prior.marginals[key].source.startswith("application history:")
            and key in ("baseload_kwh_per_day", "hot_water_kwh_per_day")
        ):
            p[key] = draws[key].astype(float)
        if key in facts and key != "floor_area_m2":
            p[key] = np.full(n_samples, float(facts[key]))
        if key in prior.marginals and prior.marginals[key].family == "exact":
            p[key] = np.full(n_samples, prior.marginals[key].centre)
    if "exposed_floor_area_m2" in facts:
        p["floor_area_m2"] = np.full(n_samples, float(facts["exposed_floor_area_m2"]))
    if "internal_gains_w" not in facts:
        p["internal_gains_w"] = 20 * occupants + retained * p["baseload_kwh_per_day"] * 1000 / 24
    if "thermostat_c" in facts:
        p["setpoint_c"] = np.full(n_samples, float(facts["thermostat_c"]))
    days = np.array([calendar.monthrange(int(t[:4]), int(t[5:]))[1] for t in request.months])
    temp_centre = np.array([w.mean_temp_c for w in prior.weather])
    ghi_centre = np.array([w.ghi_kwh_m2 for w in prior.weather])
    synthetic_weather = any(w.source == "synthetic" for w in prior.weather)
    temperature = np.tile(temp_centre, (n_samples, 1))
    ghi = np.tile(ghi_centre, (n_samples, 1))
    if synthetic_weather:
        temp_u = ndtr(
            np.log(_temporal(request.seed, request.property_id, "weather_temperature", n_samples, m, 1)) + 0.5
        )
        ghi_u = ndtr(
            np.log(_temporal(request.seed, request.property_id, "weather_ghi", n_samples, m, 1)) + 0.5
        )
        for j, month in enumerate(request.months):
            temperature[:, j] = sample_marginal(prior.marginals[f"mean_temp_c:{month[5:]}"], temp_u[:, j])
            ghi[:, j] = sample_marginal(prior.marginals[f"ghi_kwh_m2:{month[5:]}"], ghi_u[:, j])
    for key, target in (("monthly_temperature_c", "temperature"), ("monthly_ghi_kwh_m2", "ghi")):
        if key in facts:
            a = np.asarray(facts[key], dtype=float)
            if a.shape != (m,) or not np.isfinite(a).all() or (target == "ghi" and (a < 0).any()):
                raise ValueError(f"Invalid {key}")
            if target == "temperature":
                temperature = np.tile(a, (n_samples, 1))
            else:
                ghi = np.tile(a, (n_samples, 1))
    w = {
        "days_in_month": np.tile(days, (n_samples, 1)),
        "mean_temp_c": temperature,
        "ghi_kwh_m2": ghi,
        "hdd_15_5": np.maximum(15.5 - temperature, 0) * days,
        "month_of_year": np.tile([int(t[5:]) for t in request.months], (n_samples, 1)),
    }
    profiles = {}
    for key in ("baseload_multiplier", "hot_water_multiplier"):
        centre = np.asarray(prior.profile_centres.get(key, [1] * m), dtype=float)
        profiles[key] = np.tile(centre, (n_samples, 1))
        if key not in prior.profile_exact:
            profiles[key] *= _temporal(request.seed, request.property_id, key, n_samples, m, 0.10)
    if "gains_multiplier" in prior.profile_centres:
        profiles["gains_multiplier"] = np.tile(prior.profile_centres["gains_multiplier"], (n_samples, 1))
    for key in ("secondary_heating_kwh_monthly", "extra_electricity_kwh_monthly", "extra_gas_kwh_monthly"):
        centre = np.array(prior.profile_centres.get(key, [0] * m), dtype=float)
        profiles[key] = np.tile(centre, (n_samples, 1))
        if key not in prior.profile_exact:
            profiles[key] *= draws["appliance_demand_multiplier"][:, None]
    # Cooking and outdoor loads remain after a heating-fuel change.
    profiles["extra_electricity_kwh_monthly"] += (
        appliance_outdoor[:, None] * draws["appliance_demand_multiplier"][:, None] * days
    )
    profiles["secondary_heating_kwh_monthly"] += (
        appliance_secondary[:, None] * draws["appliance_demand_multiplier"][:, None] * days
    )
    if "extra_gas_kwh_monthly" not in prior.profile_centres:
        cooking = appliance_gas.copy()
        if "gas_cooking_kwh_per_day" in facts:
            cooking = np.full(n_samples, facts["gas_cooking_kwh_per_day"])
        profiles["extra_gas_kwh_monthly"] += cooking[:, None] * days
    if "extra_electricity_kwh_per_day" in facts:
        profiles["extra_electricity_kwh_monthly"] += float(facts["extra_electricity_kwh_per_day"]) * days
    if "secondary_heating_kwh_per_day" in facts:
        profiles["secondary_heating_kwh_monthly"] += float(facts["secondary_heating_kwh_per_day"]) * days
    # Separate pump/fan electricity (not added as fabricated gas consumption).
    aux = np.zeros((n_samples, m))
    if prior.equipment["heating_fuel"] != "electricity":
        aux = (
            draws["heating_auxiliary_power_kw"][:, None]
            * p["heating_hours_per_day"][:, None]
            * days
            * (temperature < p["setpoint_c"][:, None])
        )
    if "auxiliary_electricity_kwh_per_day" in facts:
        aux = np.tile(float(facts["auxiliary_electricity_kwh_per_day"]) * days, (n_samples, 1))
    w["heating_auxiliary_electricity_kwh_monthly"] = aux
    w.update(profiles)
    electricity = []
    constants = {
        k: np.asarray(v, dtype=float)
        for k, v in draws.items()
        if prior.marginals[k].role == "constant" and prior.marginals[k].family != "categorical"
    }
    p["retained_electric_heat_fraction"] = retained
    categories = {
        k: {str(label): float(np.mean(v == label)) for label in np.unique(v)}
        for k, v in draws.items()
        if prior.marginals[k].family == "categorical"
    }
    summary = GeneratedInputs(
        property_id=request.property_id,
        n_samples=n_samples,
        seed=request.seed,
        request_hash=prior.request_hash,
        facts=facts,
        prior=prior,
        parameters={k: moments(a, input_unit(k)) for k, a in p.items()},
        constants={k: moments(a, prior.marginals[k].unit) for k, a in constants.items()},
        categories=categories,
        monthly_inputs={k: [moments(a[:, j], input_unit(k)) for j in range(m)] for k, a in w.items()},
    )
    samples = InputSamples(request, prior, p, constants, w, profiles, electricity, summary, draws)
    if prepare_electricity:
        samples.electricity = [prepare_electricity_sample(samples, i) for i in range(n_samples)]
    return samples


def prepare_electricity_sample(samples: InputSamples, i: int):
    """Bounded-memory preparation, outside the solver; same sample/seed on replay."""
    request, prior, p, draws = samples.request, samples.prior, samples.parameters, samples.latent
    facts = prior.facts
    parameters = TwinParams(
        **{k: a[i] for k, a in p.items()},
        property_id=request.property_id,
        heating_fuel=prior.equipment["heating_fuel"],
        tariff="standard",
    )
    records = [
        record.model_copy(
            update={
                "mean_temp_c": float(samples.weather["mean_temp_c"][i, j]),
                "ghi_kwh_m2": float(samples.weather["ghi_kwh_m2"][i, j]),
            }
        )
        for j, record in enumerate(prior.weather)
    ]
    kw = {}
    if parameters.battery_kwh > 0:
        for name in ("battery_usable_fraction", "battery_charge_efficiency", "battery_discharge_efficiency"):
            kw[name] = float(facts.get(name, draws[name][i]))
        kw["battery_charge_kw"] = float(
            facts.get("battery_charge_kw", parameters.battery_kwh * draws["battery_power_c_rate"][i])
        )
        kw["battery_discharge_kw"] = float(
            facts.get("battery_discharge_kw", parameters.battery_kwh * draws["battery_power_c_rate"][i])
        )
    kw["initial_battery_fraction"] = float(facts.get("battery_initial_fraction", 0))
    if parameters.pv_kwp > 0:
        kw["pv_inverter_kw"] = float(
            facts.get("pv_inverter_kw", parameters.pv_kwp * draws["inverter_capacity_ratio"][i])
        )
    intake = ElectricityIntake(
        presence=str(draws["presence"][i]),
        heating_periods=facts.get("heating_periods", []),
        export_limit_kw=facts.get("export_limit_kw"),
        **kw,
    )
    package = request.physics_package or {}
    inputs = prepare_electricity_inputs(
        parameters,
        records,
        intake=intake,
        development=request.electricity_series is None,
        series=request.electricity_series,
        latitude=facts.get("latitude"),
        longitude=facts.get("longitude"),
        appliance_hourly_profile_kw=package.get("appliances", {}).get("hourly_profile_kw")
        if package.get("appliances")
        else None,
        seed=int(stream(
            request.seed, request.property_id,
            f"intervals:{int(draws['interval_seed_index'][i]) if 'interval_seed_index' in draws else i}",
        ).integers(0, 2**31)),
    )
    inputs.pv_kwh_per_kwp = (np.asarray(inputs.pv_kwh_per_kwp) * draws["pv_yield_multiplier"][i]).tolist()
    return type(inputs).model_validate(inputs.model_dump())
