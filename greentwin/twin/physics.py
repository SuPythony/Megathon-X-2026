"""Differentiable monthly, single-zone steady-state RC heat balance.

No calibration or thermal capacitance. Prepared electricity inputs enable the
half-hour PV and battery layer; without them only no-PV/no-battery runs are valid.
``run_physics`` is the numeric JAX seam; ``simulate_thermal`` is the validated,
contract-based boundary. See docs/THERMAL_PHYSICS.md for equations and limitations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

from contracts.models import ElementHeatLoss, Provenance, TwinParams, WeatherMonth
from greentwin.twin.physics_api import PHYSICS_PARAM_KEYS, FuelConfig

VERSION = "thermal-electricity-fuels/0.3.0"
FUELS = ("mains_gas", "electricity", "oil", "lpg", "solid", "district")


@dataclass(frozen=True)
class ThermalAssumptions:
    """Explicit development assumptions, not fitted or SAP-certified values.

    GHI-to-window transmission combines orientation, glazing, frame and shading.
    Party walls default to an equal-temperature neighbour (adiabatic boundary).
    The optional party-wall U is an effective outdoor-equivalent bypass loss.
    """

    setback_delta_c: float = 3.0
    solar_aperture_factor: float = 0.45
    party_wall_u: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.solar_aperture_factor > 1:
            raise ValueError("solar_aperture_factor must be <= 1")


DEFAULT_ASSUMPTIONS = ThermalAssumptions()


def _input_arrays(params, weather):
    # Structural checks use shapes only and work under jit/vmap/grad.
    p = {key: jnp.asarray(params[key]) for key in PHYSICS_PARAM_KEYS}
    for key in ("hot_water_standing_loss_kwh_per_day", "shower_electric_kwh_per_day"):
        p[key] = jnp.asarray(params.get(key, 0.0))
    p["retained_electric_heat_fraction"] = jnp.asarray(params.get("retained_electric_heat_fraction", .85))
    if "hot_water_efficiency" in params:
        p["hot_water_efficiency"] = jnp.asarray(params["hot_water_efficiency"])
    if any(value.ndim != 0 for value in p.values()):
        raise ValueError("Parameters must be scalars; use vmap for multiple homes/draws")
    w = {key: jnp.asarray(weather[key]) for key in ("days_in_month", "mean_temp_c", "ghi_kwh_m2")}
    shape = w["days_in_month"].shape
    if len(shape) != 1 or shape[0] == 0 or any(value.shape != shape for value in w.values()):
        raise ValueError("Weather arrays must have the same nonempty (n_months,) shape")
    for key in ("baseload_multiplier", "hot_water_multiplier"):
        default = (
            weather.get("appliance_month_factor", jnp.ones(shape))
            if key == "baseload_multiplier"
            else jnp.ones(shape)
        )
        w[key] = jnp.asarray(weather.get(key, default))
        if w[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    for key in ("secondary_heating_kwh_monthly", "extra_electricity_kwh_monthly", "extra_gas_kwh_monthly"):
        w[key] = jnp.asarray(weather.get(key, jnp.zeros(shape)))
        if w[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    for key in ("heating_auxiliary_electricity_kwh_monthly", "extra_oil_kwh_monthly", "extra_lpg_kwh_monthly",
                "extra_solid_kwh_monthly", "extra_district_kwh_monthly"):
        w[key] = jnp.asarray(weather.get(key, jnp.zeros(shape)))
        if w[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    if "gains_multiplier" in weather or "gains_month_factor" in weather:
        w["gains_multiplier"] = jnp.asarray(
            weather.get("gains_multiplier", weather.get("gains_month_factor"))
        )
        if w["gains_multiplier"].shape != shape:
            raise ValueError("gains_multiplier must align to weather months")
    return p, w


def _valid_domain(p, w, fuel, has_electricity=False):
    valid = jnp.all(jnp.stack([jnp.isfinite(v) for v in p.values()]))
    for key, value in p.items():
        if key != "setpoint_c":
            valid = valid & (value >= 0)
    valid = valid & (p["volume_m3"] > 0) & (p["heat_loss_multiplier"] > 0)
    valid = valid & (p["heating_efficiency"] > 0) & (p["heating_hours_per_day"] <= 24)
    valid = valid & (p["lighting_share_of_baseload"] <= 1)
    valid = valid & (p["retained_electric_heat_fraction"] <= 1)
    valid = valid & (p["pv_azimuth_deg"] < 360) & (p["pv_tilt_deg"] <= 90)
    if fuel.heating_fuel != "electricity":
        valid = valid & (p["heating_efficiency"] <= 1)
    if "hot_water_efficiency" in p:
        valid = valid & (p["hot_water_efficiency"] > 0)
        if fuel.hot_water_fuel != "electricity":
            valid = valid & (p["hot_water_efficiency"] <= 1)
    for key, value in w.items():
        valid = valid & jnp.all(jnp.isfinite(value))
        if key != "mean_temp_c":
            valid = valid & jnp.all(value >= 0)
    valid = valid & jnp.all((w["days_in_month"] >= 28) & (w["days_in_month"] <= 31))
    valid = valid & jnp.all(w["days_in_month"] == jnp.floor(w["days_in_month"]))
    # Fail closed for installed PV/batteries unless interval inputs are supplied.
    return valid if has_electricity else valid & (p["pv_kwp"] == 0) & (p["battery_kwh"] == 0)


def run_physics(
    params: Mapping,
    weather: Mapping,
    mode: str,
    fuel: FuelConfig,
    assumptions: ThermalAssumptions = DEFAULT_ASSUMPTIONS,
):
    """Return deterministic period totals and monthly JAX arrays (kWh).

    ``mode``/``fuel``/``assumptions`` must be static when jitted. ``calibrated``
    is a legacy seam alias for supplied parameters; it performs NO calibration.
    Standard-occupancy preparation is not implemented in this phase.
    Concrete invalid inputs raise; invalid traced inputs produce NaN outputs so
    they cannot masquerade as physical truth. Constrain future Bayesian priors.
    """
    if mode not in ("uncalibrated", "calibrated"):
        raise ValueError("Use uncalibrated mode; standard-occupancy preparation is deferred")
    if fuel.heating_fuel not in FUELS or fuel.hot_water_fuel not in FUELS:
        raise ValueError("Unsupported fuel")
    static_dhw_efficiency = getattr(fuel, "hot_water_efficiency", None)
    if static_dhw_efficiency is not None:
        efficiency = static_dhw_efficiency
        if (
            not math.isfinite(efficiency)
            or efficiency <= 0
            or (fuel.hot_water_fuel != "electricity" and efficiency > 1)
        ):
            raise ValueError("Invalid independent hot-water efficiency")
    elif fuel.hot_water_fuel != fuel.heating_fuel and "hot_water_efficiency" not in params:
        raise ValueError("Different hot-water fuel requires an explicit hot_water_efficiency")
    p, w = _input_arrays(params, weather)
    valid = _valid_domain(p, w, fuel, "electricity_inputs" in weather) & (
        p["solar_gain_factor"] * assumptions.solar_aperture_factor <= 1
    )
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        if float(p["pv_kwp"]) > 0 or float(p["battery_kwh"]) > 0:
            raise ValueError(
                "PV/battery grid accounting is deferred; supply an explicit no-PV/no-battery scenario"
            )
        raise ValueError("Invalid thermal parameters/weather: require finite values within physical domains")

    # U*A and y*A share the same exposed surface. Allocate bridges to their
    # elements for the seven-element contract, retaining the separate diagnostic.
    multiplier = p["heat_loss_multiplier"]
    elements, bridges = {}, {}
    for name, prefix in (
        ("walls", "wall"),
        ("roof", "roof"),
        ("floor", "floor"),
        ("windows", "window"),
        ("doors", "door"),
    ):
        area = p[f"{prefix}_area_m2"]
        bridges[name] = multiplier * p["thermal_bridging_y"] * area
        elements[name] = multiplier * p[f"{prefix}_u"] * area + bridges[name]
    elements["ventilation"] = multiplier * 0.33 * p["air_changes_per_hour"] * p["volume_m3"]
    elements["party_walls"] = multiplier * assumptions.party_wall_u * p["party_wall_area_m2"]
    coefficient = sum(elements.values())

    days = w["days_in_month"]
    hours = 24 * days
    normal_fraction = p["heating_hours_per_day"] / 24
    # Changing appliances changes retained appliance heat without scaling the
    # non-appliance (e.g. occupant) part of the supplied internal-gain estimate.
    appliance_w = p["retained_electric_heat_fraction"] * p["baseload_kwh_per_day"] * 1000 / 24
    retained_appliance_w = jnp.minimum(appliance_w, p["internal_gains_w"])
    internal_w = p["internal_gains_w"] + retained_appliance_w * (w["baseload_multiplier"] - 1)
    if "gains_multiplier" in w:
        internal_w = p["internal_gains_w"] * w["gains_multiplier"]
    internal = internal_w * hours / 1000
    solar = w["ghi_kwh_m2"] * p["window_area_m2"] * p["solar_gain_factor"] * assumptions.solar_aperture_factor
    gains = internal + solar

    # Monthly-mean temperature degree days at EACH controlled temperature.
    # HDD15.5 is not subtracted again: its reference is not the thermostat.
    loss_normal = coefficient * jnp.maximum(p["setpoint_c"] - w["mean_temp_c"], 0) * hours / 1000
    loss_setback = (
        coefficient
        * jnp.maximum(p["setpoint_c"] - assumptions.setback_delta_c - w["mean_temp_c"], 0)
        * hours
        / 1000
    )
    space_normal = jnp.maximum(loss_normal - gains, 0)
    space_setback = jnp.maximum(loss_setback - gains, 0)
    space = normal_fraction * space_normal + (1 - normal_fraction) * space_setback
    loss = normal_fraction * loss_normal + (1 - normal_fraction) * loss_setback
    unused = normal_fraction * jnp.maximum(gains - loss_normal, 0) + (1 - normal_fraction) * jnp.maximum(
        gains - loss_setback, 0
    )

    hot = p["hot_water_kwh_per_day"] * days * w["hot_water_multiplier"]
    standing_loss = p["hot_water_standing_loss_kwh_per_day"] * days
    shower = p["shower_electric_kwh_per_day"] * days * w["hot_water_multiplier"]
    secondary = jnp.minimum(w["secondary_heating_kwh_monthly"], space)
    main_space = space - secondary
    dhw_efficiency = p.get(
        "hot_water_efficiency",
        p["heating_efficiency"] if static_dhw_efficiency is None else static_dhw_efficiency,
    )
    zero = jnp.zeros_like(space)
    purchased = {f: zero for f in FUELS}
    purchased[fuel.heating_fuel] = purchased[fuel.heating_fuel] + main_space / p["heating_efficiency"]
    purchased[fuel.hot_water_fuel] = purchased[fuel.hot_water_fuel] + (hot + standing_loss) / dhw_efficiency
    purchased["electricity"] = purchased["electricity"] + secondary + shower
    purchased["mains_gas"] = purchased["mains_gas"] + w["extra_gas_kwh_monthly"]
    for f in ("oil", "lpg", "solid", "district"):
        purchased[f] = purchased[f] + w[f"extra_{f}_kwh_monthly"]
    baseload = p["baseload_kwh_per_day"] * days * w["baseload_multiplier"]
    auxiliary = w["extra_electricity_kwh_monthly"] + w["heating_auxiliary_electricity_kwh_monthly"]
    electricity = baseload + purchased["electricity"] + auxiliary
    output = {
        "monthly_electricity_kwh": electricity,
        "monthly_electricity_demand_kwh": electricity,
        "monthly_gas_kwh": purchased["mains_gas"],
        "monthly_space_heating_kwh": space,
        "monthly_hot_water_kwh": hot + shower,
        "monthly_main_hot_water_kwh": hot,
        "monthly_hot_water_system_loss_kwh": standing_loss,
        "monthly_shower_electricity_kwh": shower,
        "monthly_main_space_heating_kwh": main_space,
        "monthly_secondary_electricity_kwh": secondary,
        "monthly_secondary_unused_request_kwh": w["secondary_heating_kwh_monthly"] - secondary,
        "monthly_auxiliary_electricity_kwh": auxiliary,
        "monthly_heating_auxiliary_electricity_kwh": w["heating_auxiliary_electricity_kwh_monthly"],
        "monthly_outdoor_other_electricity_kwh": w["extra_electricity_kwh_monthly"],
        "monthly_extra_gas_kwh": w["extra_gas_kwh_monthly"],
        "monthly_heat_loss_kwh": loss,
        "monthly_heat_gains_kwh": gains,
        "monthly_internal_gains_kwh": internal,
        "monthly_solar_gains_kwh": solar,
        "monthly_unused_heat_gains_kwh": unused,
        "monthly_baseload_kwh": baseload,
        "monthly_heating_fuel_kwh": main_space / p["heating_efficiency"],
        "monthly_hot_water_fuel_kwh": (hot + standing_loss) / dhw_efficiency,
        "monthly_heating_conversion_loss_kwh": jnp.maximum(
            main_space / p["heating_efficiency"] - main_space, 0
        ),
        "monthly_heating_ambient_heat_kwh": jnp.maximum(main_space - main_space / p["heating_efficiency"], 0),
        "monthly_hot_water_conversion_loss_kwh": jnp.maximum(
            (hot + standing_loss) / dhw_efficiency - hot - standing_loss, 0
        ),
        "monthly_hot_water_ambient_heat_kwh": jnp.maximum(
            hot + standing_loss - (hot + standing_loss) / dhw_efficiency, 0
        ),
        "monthly_pv_generation_kwh": zero,
        "monthly_export_kwh": zero,
        "monthly_battery_loss_kwh": zero,
        "monthly_battery_delta_kwh": zero,
        "monthly_thermal_storage_delta_kwh": zero,
    }
    for f in ("oil", "lpg", "solid", "district"):
        output[f"monthly_{f}_kwh"] = purchased[f]
        output[f"{f}_kwh"] = jnp.sum(purchased[f])
    for f in FUELS:
        output[f"monthly_{f}_space_heating_fuel_kwh"] = (
            main_space / p["heating_efficiency"] if fuel.heating_fuel == f else zero)
        output[f"monthly_{f}_hot_water_fuel_kwh"] = (
            (hot+standing_loss) / dhw_efficiency if fuel.hot_water_fuel == f else zero)
        output[f"monthly_{f}_other_fuel_kwh"] = (
            w["extra_gas_kwh_monthly"] if f == "mains_gas" else
            secondary + shower if f == "electricity" else w[f"extra_{f}_kwh_monthly"])
    for total, monthly in (
        ("electricity_import_kwh", "monthly_electricity_kwh"),
        ("gas_kwh", "monthly_gas_kwh"),
        ("space_heating_kwh", "monthly_space_heating_kwh"),
        ("hot_water_kwh", "monthly_hot_water_kwh"),
        ("pv_generation_kwh", "monthly_pv_generation_kwh"),
        ("export_kwh", "monthly_export_kwh"),
    ):
        output[total] = jnp.sum(output[monthly])
    output["heat_loss_coefficient_w_per_k"] = coefficient
    output["thermal_bridging_w_per_k"] = sum(bridges.values())
    output["element_heat_loss_w_per_k"] = elements
    output["element_monthly_heat_loss_kwh"] = {
        name: value
        * (
            normal_fraction * jnp.maximum(p["setpoint_c"] - w["mean_temp_c"], 0)
            + (1 - normal_fraction)
            * jnp.maximum(p["setpoint_c"] - assumptions.setback_delta_c - w["mean_temp_c"], 0)
        )
        * hours
        / 1000
        for name, value in elements.items()
    }
    # Python exceptions cannot depend on traced values. Fail closed under JAX
    # transformations without stopping differentiation for valid parameters.
    if "electricity_inputs" in weather:
        from greentwin.twin.electricity import run_electricity

        output.update(
            run_electricity(p, output, weather["electricity_inputs"], fuel, assumptions.setback_delta_c)
        )
    output["electricity_demand_kwh"] = jnp.sum(output["monthly_electricity_demand_kwh"])
    output["monthly_purchased_electricity_gas_kwh"] = output["monthly_electricity_kwh"] + output["monthly_gas_kwh"]
    output["monthly_all_fuel_purchased_kwh"] = output["monthly_purchased_electricity_gas_kwh"] + sum(
        output[f"monthly_{f}_kwh"] for f in ("oil", "lpg", "solid", "district"))
    output["purchased_electricity_gas_kwh"] = jnp.sum(output["monthly_purchased_electricity_gas_kwh"])
    output["all_fuel_purchased_kwh"] = jnp.sum(output["monthly_all_fuel_purchased_kwh"])
    return jax.tree.map(lambda value: jnp.where(valid, value, jnp.nan), output)


# Intake-only flags are absent from TwinParams. The generator checks this
# declaration before invoking the numeric seam, which cannot inspect Property.
run_physics.unsupported_property_features = ("solar_thermal",)


class ThermalResult(BaseModel):
    """Exact conditional calculation; no fitted posterior or accuracy claim."""

    property_id: str
    parameters: TwinParams
    equipment: dict[str, str | float | None]
    weather_inputs: list[WeatherMonth]
    months: list[str]
    totals: dict[str, float]
    monthly: dict[str, list[float]]
    elements: list[ElementHeatLoss]
    heat_loss_coefficient_w_per_k: float
    thermal_bridging_w_per_k: float
    monthly_element_heat_loss_kwh: dict[str, list[float]]
    conservation_max_error_kwh: dict[str, float]
    assumptions: dict[str, float]
    calibration_status: Literal["uncalibrated"] = "uncalibrated"
    weather_filled_fields: list[str]
    weather_sources: list[str]
    provenance: Provenance


def simulate_thermal(
    params: TwinParams,
    weather: Sequence[WeatherMonth],
    fuel: FuelConfig,
    *,
    assumptions: ThermalAssumptions = DEFAULT_ASSUMPTIONS,
    data_version: str = "unspecified",
) -> ThermalResult:
    """Validate contract inputs and conservation; never silently fill weather."""
    from greentwin.twin.generate import validate_physics_output
    from greentwin.twin.weather import weather_arrays

    if params.heating_fuel != fuel.heating_fuel:
        raise ValueError("TwinParams and FuelConfig heating fuels must agree")
    if not weather:
        raise ValueError("At least one sourced or explicitly synthetic weather month is required")
    months = [record.month for record in weather]
    if months != sorted(set(months)):
        raise ValueError("Weather months must be unique and chronological")
    if len({record.area_code for record in weather}) != 1:
        raise ValueError("All weather months must describe the same area")
    if any(record.mean_temp_c is None or record.ghi_kwh_m2 is None for record in weather):
        raise ValueError("Explicit temperature and irradiation are required; fill and label upstream")
    numeric = {key: getattr(params, key) for key in PHYSICS_PARAM_KEYS}
    arrays = weather_arrays(weather)
    raw = run_physics(numeric, arrays, "uncalibrated", fuel, assumptions)
    provenance = Provenance(
        engine="physics_l1", model_version=VERSION, data_version=data_version, n_samples=1
    )
    # Nested per-element arrays are separate from monthly scalar series.
    truth = validate_physics_output(raw, len(weather), params, fuel, arrays, provenance)
    total_h = float(raw["heat_loss_coefficient_w_per_k"])
    elements = [
        ElementHeatLoss(
            element=name,
            heat_loss_w_per_k=value,
            share_pct=100 * value / total_h if total_h else 0,
            estimated=True,
            basis="Estimated from supplied geometry and fabric assumptions; uncalibrated steady-state model",
        )
        for name, value in truth.element_heat_loss_w_per_k.items()
    ]
    totals = truth.totals | {f"{f}_kwh": float(raw[f"{f}_kwh"]) for f in ("oil", "lpg", "solid", "district")}
    return ThermalResult(
        property_id=params.property_id,
        parameters=params,
        equipment=asdict(fuel),
        weather_inputs=list(weather),
        months=months,
        totals=totals,
        monthly=truth.monthly,
        elements=elements,
        heat_loss_coefficient_w_per_k=total_h,
        thermal_bridging_w_per_k=float(raw["thermal_bridging_w_per_k"]),
        monthly_element_heat_loss_kwh={
            key: np.asarray(value).tolist() for key, value in raw["element_monthly_heat_loss_kwh"].items()
        },
        conservation_max_error_kwh=truth.conservation_max_error_kwh,
        assumptions=asdict(assumptions),
        weather_filled_fields=[],
        weather_sources=sorted({record.source for record in weather}),
        provenance=provenance,
    )
