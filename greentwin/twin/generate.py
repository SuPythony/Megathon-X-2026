"""Synthetic homes and observations; energy always comes from an injected simulator.

This module has no JAX dependency and contains no heat/electricity solver. It uses
Satyarth's parameter preparation and accepts his run_physics(params, weather, mode,
fuel) seam. See docs/SYNTHETIC_DATA.md for the additional conservation diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from contracts.models import (
    AreaEPCStats,
    DataQuality,
    EnergyUsageMonth,
    Property,
    Provenance,
    ReferenceLabel,
    TwinParams,
    WeatherMonth,
)
from greentwin.twin.params import fuel_config, params_with_sources, sap_occupancy
from greentwin.twin.physics_api import PHYSICS_PARAM_KEYS, FuelConfig
from greentwin.twin.weather import resolve_weather, typical_year

VERSION = "synthetic/0.3.0"
ROOT = Path(__file__).resolve().parents[2]


class GeneratorConfig(BaseModel):
    n_homes: int = Field(100, ge=1)
    seed: int = Field(42, ge=0)
    start_month: str = "2025-01"
    n_months: int = Field(24, ge=1, le=120)
    data_version: str = "development-v1"
    development: bool = False
    missing_rate: float = Field(0.08, ge=0, le=1)
    missing_block_rate: float = Field(0.15, ge=0, le=1)
    estimated_read_rate: float = Field(0.10, ge=0, le=1)
    meter_noise_relative: float = Field(0.025, ge=0, le=0.5)
    estimated_noise_multiplier: float = Field(3.0, ge=1, le=10)
    missing_input_rate: float = Field(0.10, ge=0, le=1)
    conflict_rate: float = Field(0.05, ge=0, le=1)

    @model_validator(mode="after")
    def valid_start(self):
        month_sequence(self.start_month, self.n_months)
        return self


class LatentHousehold(BaseModel):
    occupants: int = Field(ge=1, le=10)
    behaviour: Literal["mostly_away", "evenings", "mostly_home"]
    rebound_fraction: float = Field(ge=0, le=1)
    # Correlated seasonal deviations, positive and mean-one; passed as runtime inputs.
    baseload_multiplier: list[float]
    hot_water_multiplier: list[float]


class PhysicsTruth(BaseModel):
    """Deterministic physical truth, not a posterior prediction or a meter observation."""

    totals: dict[str, float]
    monthly: dict[str, list[float]]
    element_heat_loss_w_per_k: dict[str, float]
    conservation_max_error_kwh: dict[str, float]
    provenance: Provenance


class GeneratedHome(BaseModel):
    source_property_id: str
    split_group: str
    true_property: Property
    observed_property: Property
    base_params: TwinParams
    true_params: TwinParams
    equipment: dict[str, str | float | None]
    parameter_sources: dict[str, str]
    latent: LatentHousehold
    weather: list[WeatherMonth]
    weather_filled_fields: list[str] = Field(default_factory=list)
    electricity_input_summary: dict | None = None
    baseline: PhysicsTruth | None = None
    scenarios: dict[str, PhysicsTruth] = Field(default_factory=dict)
    observations: list[EnergyUsageMonth] = Field(default_factory=list)
    reference_labels: list[ReferenceLabel] = Field(default_factory=list)


class SyntheticDataset(BaseModel):
    generator_version: str = VERSION
    config: GeneratorConfig
    homes: list[GeneratedHome]
    metadata: dict


@dataclass(frozen=True)
class Simulator:
    run_physics: Callable
    model_version: str
    engine: str = "physics_l1"

    def __post_init__(self):
        if self.engine != "physics_l1" or not self.model_version:
            raise ValueError("Synthetic energy requires a versioned physics_l1 simulator, never a stub/mock")
        identity = f"{self.run_physics.__module__}.{self.run_physics.__name__}".lower()
        if "stub" in identity or "mock" in identity:
            raise ValueError("A stub/mock cannot generate physics truth")


@dataclass(frozen=True)
class RetrofitScenario:
    name: str
    retrofit_ids: tuple[str, ...]
    params: TwinParams
    fuel: FuelConfig
    # Set only when the feasibility owner has supplied the full compatible package.
    all_feasible: bool = False


ScenarioFactory = Callable[[Property, TwinParams, FuelConfig], Sequence[RetrofitScenario]]


def month_sequence(start: str, count: int) -> list[str]:
    try:
        year, month = (int(x) for x in start.split("-"))
        if start != f"{year:04d}-{month:02d}" or not 1 <= month <= 12 or year < 1900:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("start_month must be YYYY-MM, year >= 1900") from None
    return [f"{(year * 12 + month - 1 + i) // 12:04d}-{(month - 1 + i) % 12 + 1:02d}" for i in range(count)]


def _split(key: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < 0.70 else "val" if value < 0.85 else "test"


def _bounded_normal(rng, centre, scale, low, high):
    # Rejection sampling avoids artificial piles at the bounds from clipping.
    for _ in range(1000):
        value = rng.normal(centre, scale)
        if low <= value <= high:
            return float(value)
    raise ValueError("Could not sample bounded normal; check distribution parameters")


def _seasonal(rng, n, scale):
    values = np.empty(n)
    state = rng.normal(0, scale)
    for i in range(n):
        state = 0.65 * state + rng.normal(0, scale * np.sqrt(1 - 0.65**2))
        values[i] = np.exp(state)
    return (values / values.mean()).tolist()


def sample_latent(prop: Property, base: TwinParams, rng, n_months: int):
    """Conditional household variation; physical parameters remain fixed over time.

    These are documented development priors, not claimed NEED-fitted distributions.
    Geometry/categories stay with the source EPC record, preserving their joint mix.
    """
    centre = sap_occupancy(prop.floor_area_m2)
    max_people = min(10, max(2, 2 * (prop.bedrooms or int(prop.floor_area_m2 / 30))))
    occupants = min(max_people, 1 + int(rng.poisson(max(0.1, centre - 1))))
    behaviour = str(rng.choice(["mostly_away", "evenings", "mostly_home"], p=[0.25, 0.45, 0.30]))
    hours_mean = {"mostly_away": 6, "evenings": 9, "mostly_home": 15}[behaviour]
    latent = LatentHousehold(
        occupants=occupants,
        behaviour=behaviour,
        rebound_fraction=float(rng.beta(2, 12)),
        baseload_multiplier=_seasonal(rng, n_months, 0.10),
        hot_water_multiplier=_seasonal(rng, n_months, 0.08),
    )
    p = base.model_dump()
    p["heat_loss_multiplier"] = float(rng.lognormal(0, 0.22))
    p["setpoint_c"] = _bounded_normal(rng, 19.5, 1.5, 15, 24)
    p["heating_hours_per_day"] = _bounded_normal(rng, hours_mean, 2, 2, 24)
    people_factor = (occupants / centre) ** 0.6
    p["baseload_kwh_per_day"] = base.baseload_kwh_per_day * people_factor * float(rng.lognormal(0, 0.25))
    # Demand follows water mass, heat capacity and temperature rise, not random fuel kWh.
    litres = (25 * occupants + 36) * float(rng.lognormal(0, 0.20))
    p["hot_water_kwh_per_day"] = litres * 4.19 * 37 / 3600
    # Couple electrical consumption and its heat gains; metabolic gains track occupants.
    p["internal_gains_w"] = 20 * occupants + 0.85 * p["baseload_kwh_per_day"] * 1000 / 24
    is_hp = "heat_pump" in prop.heating.main_system.value
    if is_hp:
        p["heating_efficiency"] = _bounded_normal(rng, base.heating_efficiency, 0.35, 1.1, 6)
    elif prop.heating.main_fuel == "electricity":
        p["heating_efficiency"] = 1.0
    else:
        eff = min(0.98, max(0.1, base.heating_efficiency))
        p["heating_efficiency"] = float(rng.beta(eff * 60, (1 - eff) * 60))
    result = TwinParams.model_validate(p)
    validate_inputs(prop, result, fuel_config(prop))
    return latent, result


def validate_inputs(prop: Property, params: TwinParams, fuel: FuelConfig):
    if params.property_id != prop.property_id or params.heating_fuel != fuel.heating_fuel:
        raise ValueError("Parameter property/fuel identity must match the simulated home")
    if prop.floor_area_m2 <= 0 or prop.storeys < 1:
        raise ValueError("Positive total floor area and storeys are required")
    for key in PHYSICS_PARAM_KEYS:
        value = float(getattr(params, key))
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite physical input: {key}")
        if key != "setpoint_c" and value < 0:
            raise ValueError(f"Negative physical input: {key}")
    if params.volume_m3 <= 0 or not 0 < params.heating_hours_per_day <= 24:
        raise ValueError("Positive heated volume and heating hours <= 24 required")
    if not 0 <= params.lighting_share_of_baseload <= 1:
        raise ValueError("Lighting share must be a fraction")
    if not 0 <= params.pv_azimuth_deg < 360 or not 0 <= params.pv_tilt_deg <= 90:
        raise ValueError("Invalid PV orientation/tilt")
    if not 0 < params.heating_efficiency or (
        fuel.heating_fuel != "electricity" and params.heating_efficiency > 1
    ):
        raise ValueError("Invalid fuel conversion efficiency")
    if fuel.hot_water_efficiency is not None and (
        not np.isfinite(fuel.hot_water_efficiency)
        or fuel.hot_water_efficiency <= 0
        or (fuel.hot_water_fuel != "electricity" and fuel.hot_water_efficiency > 1)
    ):
        raise ValueError("Invalid hot-water efficiency")
    if params.pv_kwp > 0 and prop.constraints.roof_access is False:
        raise ValueError("PV requires an accessible roof in the v1 catalogue")
    if params.roof_area_m2 > 0 and prop.roof.type.value == "another_dwelling_above":
        raise ValueError("An intermediate flat cannot have exposed roof area")
    if params.floor_area_m2 > 0 and prop.floor.type.value == "another_dwelling_below":
        raise ValueError("An intermediate flat cannot have exposed ground-floor area")


def population_weights(properties: Sequence[Property], area_stats: Sequence[AreaEPCStats] = ()):
    """Rake EPC categories to optional LA lodgement marginals (not dwelling-stock counts)."""
    weights = np.ones(len(properties), dtype=float)
    seen = set()
    for stats in area_stats:
        if stats.area_type != "local_authority":
            continue
        if stats.area_code in seen:
            raise ValueError("Supply one comparable period per LA for population weighting")
        seen.add(stats.area_code)
        ix = np.array(
            [i for i, p in enumerate(properties) if p.location.local_authority_code == stats.area_code]
        )
        if not len(ix):
            raise ValueError(f"No source properties for area target {stats.area_code}")
        margins = [
            (stats.band_counts, lambda p: p.epc.current_band),
            (stats.property_type_counts, lambda p: p.property_type.value),
        ]
        for counts, category in margins:
            if any(count < 0 for count in counts.values()):
                raise ValueError("Population target counts cannot be negative")
            for key, count in counts.items():
                if count > 0 and key not in {category(properties[i]) for i in ix}:
                    raise ValueError(
                        f"Population target category absent in source homes: {stats.area_code}/{key}"
                    )
            if counts and sum(counts.values()) > 0:
                for i in ix:
                    if counts.get(category(properties[i]), 0) == 0:
                        weights[i] = 0
        for _ in range(100):
            previous = weights[ix].copy()
            for counts, category in margins:
                if not counts or sum(counts.values()) <= 0:
                    continue
                total = weights[ix].sum()
                for key, count in counts.items():
                    members = [i for i in ix if category(properties[i]) == key]
                    mass = weights[members].sum()
                    if mass > 0:
                        weights[members] *= total * count / sum(counts.values()) / mass
            if np.max(np.abs(weights[ix] - previous)) < 1e-8:
                break
        for counts, category in margins:
            if not counts or sum(counts.values()) <= 0:
                continue
            mass = weights[ix].sum()
            if mass <= 0:
                raise ValueError("Incompatible population marginal targets")
            for key, count in counts.items():
                actual = sum(weights[i] for i in ix if category(properties[i]) == key) / mass
                if abs(actual - count / sum(counts.values())) > 1e-4:
                    raise ValueError("Population weighting failed to converge; incompatible margins")
        if stats.lodgements > 0:
            weights[ix] *= stats.lodgements / weights[ix].sum()
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Invalid population weights")
    return weights / weights.sum()


_MONTHLY = {
    "electricity_import_kwh": "monthly_electricity_kwh",
    "gas_kwh": "monthly_gas_kwh",
    "space_heating_kwh": "monthly_space_heating_kwh",
    "hot_water_kwh": "monthly_hot_water_kwh",
    "pv_generation_kwh": "monthly_pv_generation_kwh",
    "export_kwh": "monthly_export_kwh",
}
_DIAGNOSTICS = (
    "monthly_electricity_demand_kwh",
    "monthly_battery_loss_kwh",
    "monthly_battery_delta_kwh",
    "monthly_heat_loss_kwh",
    "monthly_heat_gains_kwh",
    "monthly_unused_heat_gains_kwh",
)


def validate_physics_output(
    raw: Mapping,
    n_months: int,
    params: TwinParams,
    fuel: FuelConfig,
    weather: Mapping,
    provenance: Provenance,
    extra_params: Mapping | None = None,
) -> PhysicsTruth:
    """Reject inconsistent outputs. An injected simulator must supply balance evidence.

    Signed storage changes are allowed. Heat is useful delivered heat before fuel
    conversion. Additional fuel fields stay in truth instead of being dropped.
    """
    monthly, totals = {}, {}
    required = list(_MONTHLY.values()) + list(_DIAGNOSTICS)
    for key in required:
        if key not in raw:
            raise ValueError(f"Simulator must provide {key} for conservation validation")
    for key, value in raw.items():
        if not key.startswith("monthly_"):
            continue
        a = np.asarray(value, dtype=float)
        if a.shape != (n_months,) or not np.isfinite(a).all():
            raise ValueError(f"{key} must be finite with shape ({n_months},)")
        signed = key in (
            "monthly_battery_delta_kwh",
            "monthly_thermal_storage_delta_kwh",
            "monthly_heat_loss_kwh",
        )
        if not signed and (a < 0).any():
            raise ValueError(f"Negative physical output: {key}")
        monthly[key] = a.tolist()
    for total_key, month_key in _MONTHLY.items():
        total = float(raw[total_key])
        if (
            not np.isfinite(total)
            or total < 0
            or not np.isclose(total, sum(monthly[month_key]), rtol=2e-5, atol=0.02)
        ):
            raise ValueError(f"{total_key} must equal its monthly sum")
        totals[total_key] = total
    a = {k: np.array(v) for k, v in monthly.items()}
    errors = {}

    def balance(name, lhs, rhs):
        error = np.abs(lhs - rhs)
        errors[name] = float(error.max())
        if not np.allclose(lhs, rhs, rtol=2e-5, atol=0.02):
            raise ValueError(f"{name} energy balance violated (max error {error.max():.6g} kWh)")

    balance(
        "electricity",
        a["monthly_electricity_kwh"] + a["monthly_pv_generation_kwh"],
        a["monthly_electricity_demand_kwh"]
        + a["monthly_export_kwh"]
        + a["monthly_battery_loss_kwh"]
        + a["monthly_battery_delta_kwh"]
        + a.get("monthly_curtailment_kwh", np.zeros(n_months)),
    )
    balance(
        "heat",
        a["monthly_space_heating_kwh"] + a["monthly_heat_gains_kwh"],
        a["monthly_heat_loss_kwh"]
        + a["monthly_unused_heat_gains_kwh"]
        + a.get("monthly_thermal_storage_delta_kwh", np.zeros(n_months)),
    )
    # Conversion law: useful heat / efficiency goes to the correct purchased fuel.
    extra_params = extra_params or {}
    dhw_eff = extra_params.get("hot_water_efficiency", fuel.hot_water_efficiency or params.heating_efficiency)
    days = np.asarray(weather["days_in_month"])
    secondary = a.get("monthly_secondary_electricity_kwh", np.zeros(n_months))
    expected_secondary = np.minimum(
        np.asarray(weather.get("secondary_heating_kwh_monthly", np.zeros(n_months))),
        a["monthly_space_heating_kwh"],
    )
    balance("secondary_heating", secondary, expected_secondary)
    shower = (
        float(extra_params.get("shower_electric_kwh_per_day", 0))
        * days
        * np.asarray(weather.get("hot_water_multiplier", np.ones(n_months)))
    )
    balance("shower_demand", a.get("monthly_shower_electricity_kwh", np.zeros(n_months)), shower)
    standing = float(extra_params.get("hot_water_standing_loss_kwh_per_day", 0)) * days
    balance("cylinder_loss", a.get("monthly_hot_water_system_loss_kwh", np.zeros(n_months)), standing)
    conversion = {}
    for f, energy in (
        (fuel.heating_fuel, (a["monthly_space_heating_kwh"] - secondary) / params.heating_efficiency),
        (fuel.hot_water_fuel, (a["monthly_hot_water_kwh"] - shower + standing) / dhw_eff),
    ):
        conversion[f] = conversion.get(f, np.zeros(n_months)) + energy
    conversion["electricity"] = conversion.get("electricity", np.zeros(n_months)) + secondary + shower
    extra_gas = np.asarray(weather.get("extra_gas_kwh_monthly", np.zeros(n_months)))
    conversion["mains_gas"] = conversion.get("mains_gas", np.zeros(n_months)) + extra_gas
    extra_electricity = np.asarray(weather.get("extra_electricity_kwh_monthly", np.zeros(n_months)))
    heating_auxiliary = np.asarray(weather.get("heating_auxiliary_electricity_kwh_monthly", np.zeros(n_months)))
    balance(
        "extra_electricity", a.get("monthly_auxiliary_electricity_kwh", np.zeros(n_months)),
        extra_electricity + heating_auxiliary
    )
    base = (
        params.baseload_kwh_per_day
        * np.asarray(weather["days_in_month"])
        * np.asarray(weather.get("baseload_multiplier", np.ones(n_months)))
    )
    balance(
        "electrical_end_uses",
        a["monthly_electricity_demand_kwh"],
        base
        + conversion.get("electricity", np.zeros(n_months))
        + a.get("monthly_auxiliary_electricity_kwh", np.zeros(n_months)),
    )
    balance("gas_conversion", a["monthly_gas_kwh"], conversion.get("mains_gas", np.zeros(n_months)))
    expected_dhw = (
        params.hot_water_kwh_per_day
        * np.asarray(weather["days_in_month"])
        * np.asarray(weather.get("hot_water_multiplier", np.ones(n_months)))
    )
    balance("hot_water_demand", a["monthly_hot_water_kwh"], expected_dhw + shower)
    for f, energy in conversion.items():
        if f not in ("mains_gas", "electricity"):
            energy = energy + np.asarray(weather.get(f"extra_{f}_kwh_monthly", np.zeros(n_months)))
            key = f"monthly_{f}_kwh"
            if key not in a:
                raise ValueError(f"Simulator must preserve {f} consumption in {key}")
            balance(f"{f}_conversion", a[key], energy)
            totals[f"{f}_kwh"] = float(a[key].sum())
    for f in ("oil", "lpg", "solid", "district"):
        if f not in conversion and f"monthly_{f}_kwh" in a:
            balance(f"{f}_other", a[f"monthly_{f}_kwh"],
                    np.asarray(weather.get(f"extra_{f}_kwh_monthly", np.zeros(n_months))))
        if f"monthly_{f}_kwh" in a:
            totals[f"{f}_kwh"] = float(a[f"monthly_{f}_kwh"].sum())
    if params.pv_kwp == 0 and totals["pv_generation_kwh"] != 0:
        raise ValueError("PV generation without PV capacity")
    initial = float(raw.get("battery_initial_kwh", 0))
    if not np.isfinite(initial) or initial < 0 or initial > params.battery_kwh:
        raise ValueError("Invalid initial battery energy")
    soc = initial + np.cumsum(a["monthly_battery_delta_kwh"])
    if (soc < -0.02).any() or (soc > params.battery_kwh + 0.02).any():
        raise ValueError("Battery energy exceeds capacity or is borrowed from an empty initial state")
    if params.battery_kwh == 0 and (
        np.any(a["monthly_battery_delta_kwh"] != 0) or np.any(a["monthly_battery_loss_kwh"] != 0)
    ):
        raise ValueError("Battery energy flows without a battery")
    elements = {k: float(v) for k, v in raw["element_heat_loss_w_per_k"].items()}
    expected_elements = {"walls", "roof", "floor", "windows", "doors", "ventilation", "party_walls"}
    if set(elements) != expected_elements or any(not np.isfinite(v) or v < 0 for v in elements.values()):
        raise ValueError("Invalid heat-loss element coefficients")
    return PhysicsTruth(
        totals=totals,
        monthly=monthly,
        element_heat_loss_w_per_k=elements,
        conservation_max_error_kwh=errors,
        provenance=provenance,
    )


def _observations(home: Property, truth: PhysicsTruth, months, config, rng):
    out = []
    block = set()
    if rng.random() < config.missing_block_rate:
        start = int(rng.integers(len(months)))
        block = set(range(start, min(len(months), start + int(rng.integers(2, 5)))))
    for i, month in enumerate(months):
        estimated = bool(rng.random() < config.estimated_read_rate)
        sigma = config.meter_noise_relative * (config.estimated_noise_multiplier if estimated else 1)

        def meter(key, structural_zero=False, i=i, sigma=sigma):
            value = truth.monthly[key][i]
            if structural_zero:
                return 0.0
            if i in block or rng.random() < config.missing_rate:
                return None
            # Mean-one multiplicative noise preserves positivity and true physical zeros.
            return float(value * rng.lognormal(-0.5 * sigma**2, sigma))

        out.append(
            EnergyUsageMonth(
                property_id=home.property_id,
                month=month,
                electricity_import_kwh=meter("monthly_electricity_kwh"),
                gas_kwh=meter("monthly_gas_kwh", truth.totals["gas_kwh"] == 0),
                # These are clean synthetic sensor values, allowing independent balance auditing.
                pv_generation_kwh=truth.monthly["monthly_pv_generation_kwh"][i],
                electricity_export_kwh=truth.monthly["monthly_export_kwh"][i],
                estimated_read=estimated,
            )
        )
    return out


def _labels(prop, baseline, scenario, after, config):
    result = []
    label_type = "per_fuel_saving" if len(scenario.retrofit_ids) == 1 else "package_saving"
    for fuel, keys in (
        ("electricity", ["electricity_import_kwh"]),
        ("gas", ["gas_kwh"]),
        ("purchased", ["electricity_import_kwh", "gas_kwh"]),
    ):
        saving = sum(baseline.totals[k] - after.totals[k] for k in keys)
        result.append(
            ReferenceLabel(
                property_id=prop.property_id,
                label_type=label_type,
                retrofit_ids=list(scenario.retrofit_ids),
                fuel=fuel,
                value=saving,
                unit="kWh/simulated-period",
                origin="synthetic_generator",
                rule_id="baseline_minus_package_rerun",
                generator_version=VERSION,
            )
        )
    if scenario.all_feasible:
        before = baseline.totals["electricity_import_kwh"] + baseline.totals["gas_kwh"]
        after_total = after.totals["electricity_import_kwh"] + after.totals["gas_kwh"]
        supported = prop.heating.main_fuel in ("mains_gas", "electricity")
        result.append(
            ReferenceLabel(
                property_id=prop.property_id,
                label_type="improvement_potential",
                retrofit_ids=list(scenario.retrofit_ids),
                fuel="purchased",
                value=100 * (before - after_total) / before if before > 0 and supported else None,
                unit="%",
                origin="synthetic_generator",
                rule_id="all_feasible_one_run_null_if_zero_or_unsupported",
                generator_version=VERSION,
            )
        )
    return result


def generate_dataset(
    properties: Sequence[Property],
    config: GeneratorConfig,
    *,
    simulator: Simulator | None = None,
    weather_records: Sequence[WeatherMonth] = (),
    area_stats: Sequence[AreaEPCStats] = (),
    scenario_factory: ScenarioFactory | None = None,
    electricity_factory: Callable | None = None,
) -> SyntheticDataset:
    """Inputs-only when simulator=None; never fabricate energy with a fallback model."""
    if not properties:
        raise ValueError("Provide source properties")
    if not config.development and any(p.source != "epc_real" for p in properties):
        raise ValueError(
            "Production generation requires real EPC properties; use development=True for examples"
        )
    ids = [p.property_id for p in properties]
    if len(ids) != len(set(ids)):
        raise ValueError("Source property IDs must be unique")
    groups, group_splits = {}, {}
    for prop in properties:
        key = prop.customer_id or prop.property_id
        groups[prop.property_id] = key
        if prop.split:
            if key in group_splits and group_splits[key] != prop.split:
                raise ValueError("Source customer/property occurs in multiple splits")
            group_splits[key] = prop.split
    for key in groups.values():
        group_splits.setdefault(key, _split(key, config.seed))
    months = month_sequence(config.start_month, config.n_months)
    lookup = {}
    for w in weather_records:
        key = (w.area_code, w.month)
        if key in lookup:
            raise ValueError(f"Duplicate weather row: {key}")
        lookup[key] = w
    weights = population_weights(properties, area_stats)
    selection_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    selection = selection_rng.choice(len(properties), config.n_homes, p=weights)
    homes = []
    for i, source_index in enumerate(selection):
        # Independent streams ensure meter/noise options cannot change physical truth.
        physical_rng = np.random.default_rng(np.random.SeedSequence([config.seed, i, 1]))
        observation_rng = np.random.default_rng(np.random.SeedSequence([config.seed, i, 2]))
        source = properties[int(source_index)]
        group = groups[source.property_id]
        prop = source.model_copy(deep=True)
        prop.property_id = f"SYN-{config.seed}-{i:07d}"
        prop.customer_id = "SYN-GROUP-" + hashlib.sha256(group.encode()).hexdigest()[:16]
        prop.split = group_splits[group]
        prop.source = "synthetic"
        # Potential EPC fields are reference labels, never inputs to this generator.
        prop.epc.potential_band = None
        prop.epc.potential_sap_score = None
        prop.current_value_estimate_gbp = None
        prop.value_estimate_date = None
        prop.data_quality = DataQuality()
        base, sources = params_with_sources(prop)
        latent, true = sample_latent(prop, base, physical_rng, config.n_months)
        fuel = fuel_config(prop)
        if simulator:
            for feature in getattr(simulator.run_physics, "unsupported_property_features", ()):
                if getattr(prop.renewables, feature):
                    raise ValueError(f"Simulator {simulator.model_version} does not support {feature}")
        records = []
        for month in months:
            record = lookup.get((prop.location.postcode_district, month))
            if record is None:
                if not config.development:
                    raise ValueError(f"Missing real weather for {prop.location.postcode_district}/{month}")
                record = typical_year(prop.location.postcode_district, int(month[:4]))[int(month[5:]) - 1]
            if not config.development and (
                record.source == "synthetic" or record.mean_temp_c is None or record.ghi_kwh_m2 is None
            ):
                raise ValueError("Production generation requires sourced temperature and irradiation")
            values = [record.hdd_15_5, record.mean_temp_c, record.ghi_kwh_m2]
            if any(v is not None and not np.isfinite(v) for v in values):
                raise ValueError("Weather inputs must be finite")
            if record.hdd_15_5 < 0 or (record.ghi_kwh_m2 is not None and record.ghi_kwh_m2 < 0):
                raise ValueError("Negative degree days/irradiation")
            records.append(record.model_copy(deep=True))
        weather, _ = resolve_weather(prop.location.postcode_district, records)
        # Explicit runtime profiles: future solver must honour them or its checks fail.
        weather["baseload_multiplier"] = np.array(latent.baseload_multiplier)
        weather["hot_water_multiplier"] = np.array(latent.hot_water_multiplier)
        observed = prop.model_copy(deep=True)
        for field in ("draught_proofed_pct", "low_energy_lighting_pct", "bedrooms"):
            if observation_rng.random() < config.missing_input_rate:
                setattr(observed, field, None)
                observed.data_quality.missing_fields.append(field)
        if observation_rng.random() < config.conflict_rate:
            observed.epc.current_band = "A" if prop.epc.current_band != "A" else "G"
            observed.data_quality.conflicts.append("EPC band intentionally corrupted in observed record")
        home = GeneratedHome(
            source_property_id=source.property_id,
            split_group=prop.customer_id,
            true_property=prop,
            observed_property=observed,
            base_params=base,
            true_params=true,
            equipment=asdict(fuel),
            parameter_sources=sources,
            latent=latent,
            weather=records,
            weather_filled_fields=[
                f"{w.month}:{k}"
                for w in records
                for k in ("mean_temp_c", "ghi_kwh_m2")
                if getattr(w, k) is None
            ],
        )
        if simulator:
            provenance = Provenance(
                engine=simulator.engine,
                model_version=simulator.model_version,
                data_version=config.data_version,
                seed=config.seed,
                n_samples=1,
            )

            def run(
                p,
                f,
                prop=prop,
                weather=weather,
                provenance=provenance,
                records=records,
                latent=latent,
                electricity_seed=config.seed + i,
                home=home,
            ):
                validate_inputs(prop, p, f)
                numeric = {k: float(getattr(p, k)) for k in PHYSICS_PARAM_KEYS}
                # Copies prevent an impure callback from corrupting paired scenarios.
                runtime = {k: v.copy() for k, v in weather.items()}
                prepared = None
                if electricity_factory is not None:
                    prepared = electricity_factory(prop, p, records, latent, electricity_seed)
                    if prepared.development and not config.development:
                        raise ValueError("Generated interval weather/equipment require explicit development mode")
                    if prepared.property_id != p.property_id or prepared.months != months:
                        raise ValueError(
                            "Prepared electricity inputs must match this synthetic home and period"
                        )
                    runtime["electricity_inputs"] = prepared.numeric_arrays()
                    if home.electricity_input_summary is None:
                        home.electricity_input_summary = {
                            "version": prepared.version,
                            "seed": prepared.seed,
                            "development": prepared.development,
                            "sources": prepared.sources,
                            "warnings": prepared.warnings,
                            "interval_count": len(prepared.timestamps_utc),
                            "battery": prepared.battery.model_dump(),
                            "pv_inverter_kw": prepared.pv_inverter_kw,
                            "export_limit_kw": prepared.export_limit_kw,
                        }
                raw = simulator.run_physics(numeric, runtime, "uncalibrated", f)
                truth = validate_physics_output(raw, config.n_months, p, f, runtime, provenance)
                if prepared is not None:
                    from greentwin.twin.electricity_simulation import validate_interval_output

                    truth.conservation_max_error_kwh.update(validate_interval_output(raw, p, prepared))
                return truth

            home.baseline = run(true, fuel)
            scenarios = (
                scenario_factory(prop.model_copy(deep=True), true.model_copy(deep=True), fuel)
                if scenario_factory
                else ()
            )
            for scenario in scenarios:
                if scenario.name in home.scenarios:
                    raise ValueError("Scenario names must be unique")
                after = run(scenario.params, scenario.fuel)
                home.scenarios[scenario.name] = after
                home.reference_labels.extend(_labels(prop, home.baseline, scenario, after, config))
            home.observations = _observations(prop, home.baseline, months, config, observation_rng)
            observed.data_quality.months_of_metered_data = sum(
                r.electricity_import_kwh is not None and r.gas_kwh is not None for r in home.observations
            )
            observed.data_quality.missing_fields.extend(
                f"usage.{r.month}.{f}"
                for r in home.observations
                for f in ("electricity_import_kwh", "gas_kwh")
                if getattr(r, f) is None
            )
        homes.append(home)
    return SyntheticDataset(
        config=config,
        homes=homes,
        metadata={
            "mode": "energy_and_observations" if simulator else "inputs_only",
            "source_property_count": len(properties),
            "source_mix": sorted({p.source for p in properties}),
            "weather_sources": sorted({w.source for h in homes for w in h.weather}),
            "empirical_validation": "not_run",
            "priors": "development_assumptions_not_NEED_fitted",
            "defaults_status": "to_verify",
            "population_weighting": "LA_lodgement_marginals" if area_stats else "source_mix",
            "split_basis": "source_customer_else_source_property; variants remain together",
            "simulator_version": simulator.model_version if simulator else None,
            "calibration_status": "not_performed",
            "limitations": [
                "Synthetic agreement is not real-world accuracy",
                "No value-uplift evidence generated",
                "Monthly balances do not certify timestep power limits; simulator must check those",
            ],
        },
    )


def write_dataset(dataset: SyntheticDataset, output: Path):
    """Write separated observation/truth JSONL and a manifest; refuse existing outputs."""
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside the Physics & Bayesian worktree")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new dataset path")
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "properties": [h.observed_property for h in dataset.homes],
        "usage": [r for h in dataset.homes for r in h.observations],
        "reference_labels": [r for h in dataset.homes for r in h.reference_labels],
        "truth": dataset.homes,
    }
    counts, hashes = {}, {}
    for name, rows in tables.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(row.model_dump_json() + "\n")
        counts[name] = len(rows)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "generator_version": VERSION,
        "config": dataset.config.model_dump(),
        "metadata": dataset.metadata,
        "row_counts": counts,
        "sha256": hashes,
    }
    (output / "meta.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    metrics = {
        "seed": dataset.config.seed,
        "data_version": dataset.config.data_version,
        "homes": len(dataset.homes),
        "physics_checks": "passed" if dataset.homes[0].baseline else "not_run",
        "empirical_validation": "not_run",
        "split_counts": {
            s: sum(h.true_property.split == s for h in dataset.homes) for s in ("train", "val", "test")
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
