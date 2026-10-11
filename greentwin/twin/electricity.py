"""JAX half-hour end uses, AC PV and power-limited PV-only battery dispatch."""

from __future__ import annotations

import jax
import jax.numpy as jnp

VERSION = "electricity/0.1.0"


def _monthly_sum(values, index, months):
    return jax.ops.segment_sum(values, index, num_segments=months)


def _allocate(energy, weights, index):
    sums = _monthly_sum(weights, index, energy.shape[0])
    return energy[index] * weights / jnp.maximum(sums[index], 1e-12)


@jax.jit
def _dispatch(demand, pv, dt, capacity, charge_kw, discharge_kw, eta_c, eta_d, initial, export_limit):
    """Every flow is electrical-bus kWh; storage accounts for conversion losses."""

    def step(stored, inputs):
        load, generation, duration = inputs
        surplus = jnp.maximum(generation - load, 0)
        deficit = jnp.maximum(load - generation, 0)
        charge = jnp.minimum(
            surplus, jnp.minimum(charge_kw * duration, jnp.maximum(capacity - stored, 0) / eta_c)
        )
        discharge = jnp.minimum(deficit, jnp.minimum(discharge_kw * duration, jnp.maximum(stored, 0) * eta_d))
        next_stored = stored + eta_c * charge - discharge / eta_d
        export = jnp.minimum(surplus - charge, export_limit * duration)
        curtailed = surplus - charge - export
        imported = deficit - discharge
        loss = charge * (1 - eta_c) + discharge * (1 / eta_d - 1)
        return next_stored, (
            imported,
            export,
            charge,
            discharge,
            loss,
            next_stored - stored,
            next_stored,
            curtailed,
            jnp.minimum(generation, load),
        )

    _, flows = jax.lax.scan(step, jnp.asarray(initial, dtype=demand.dtype), (demand, pv, dt))
    names = (
        "import",
        "export",
        "charge",
        "discharge",
        "battery_loss",
        "battery_delta",
        "stored",
        "curtailment",
        "direct_pv",
    )
    return dict(zip(names, flows))


def run_electricity(params, thermal, inputs, fuel, setback_delta_c=3.0):
    """Deterministic numeric seam; validated PreparedElectricityInputs is the boundary."""
    x = {key: jnp.asarray(value) for key, value in inputs.items()}
    index = x["month_index"]
    if index.ndim != 1 or index.shape[0] == 0 or not jnp.issubdtype(index.dtype, jnp.integer):
        raise ValueError("month_index must be a nonempty integer interval array")
    interval_keys = (
        "local_hour",
        "duration_hours",
        "temperature_c",
        "pv_kwh_per_kwp",
        "baseload_weights",
        "lighting_weights",
        "heating_on_fraction",
        "hot_water_weights",
        "extra_weights",
    )
    if any(x[key].shape != index.shape for key in interval_keys):
        raise ValueError("Electricity interval arrays must align")
    n_months = thermal["monthly_baseload_kwh"].shape[0]
    valid = jnp.all((index >= 0) & (index < n_months)) & jnp.all(jnp.diff(index) >= 0)
    for key, value in x.items():
        valid = valid & jnp.all(jnp.isfinite(value))
        if key != "temperature_c":
            valid = valid & jnp.all(value >= 0)
    valid = valid & jnp.all(x["duration_hours"] == 0.5) & jnp.all(x["heating_on_fraction"] <= 1)
    valid = valid & jnp.all(x["local_hour"] < 24)
    for key in ("baseload_weights", "lighting_weights", "hot_water_weights", "extra_weights"):
        valid = valid & jnp.all(_monthly_sum(x[key], index, n_months) > 0)
    valid = valid & (x["battery_usable_fraction"] > 0) & (x["battery_usable_fraction"] <= 1)
    valid = valid & (x["battery_initial_fraction"] <= 1)
    for key in ("battery_charge_efficiency", "battery_discharge_efficiency"):
        valid = valid & (x[key] > 0) & (x[key] <= 1)
    valid = valid & (
        (params["battery_kwh"] == 0) | ((x["battery_charge_kw"] > 0) & (x["battery_discharge_kw"] > 0))
    )
    valid = valid & ((params["battery_kwh"] > 0) | (x["battery_initial_fraction"] == 0))
    valid = valid & ((params["pv_kwp"] == 0) | (x["pv_inverter_kw"] > 0))
    valid = (
        valid
        & (params["pv_azimuth_deg"] == x["pv_azimuth_deg"])
        & (params["pv_tilt_deg"] == x["pv_tilt_deg"])
    )
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError("Invalid electricity domains or PV geometry: prepare inputs for this scenario")

    base = thermal["monthly_baseload_kwh"]
    lighting = _allocate(base * params["lighting_share_of_baseload"], x["lighting_weights"], index)
    appliance = _allocate(base * (1 - params["lighting_share_of_baseload"]), x["baseload_weights"], index)
    controlled = params["setpoint_c"] - setback_delta_c * (1 - x["heating_on_fraction"])
    degree_weights = jnp.maximum(controlled - x["temperature_c"], 0) * x["duration_hours"]
    # In a contradictory monthly/interval-weather case, use a labelled allocation
    # fallback rather than silently lose the monthly thermal demand.
    degree_sums = _monthly_sum(degree_weights, index, n_months)
    degree_weights = jnp.where(degree_sums[index] > 0, degree_weights, x["duration_hours"])
    zero = jnp.zeros_like(base)
    heating_electric = thermal["monthly_heating_fuel_kwh"] if fuel.heating_fuel == "electricity" else zero
    heating_electric = heating_electric + thermal.get("monthly_secondary_electricity_kwh", zero)
    heat = _allocate(heating_electric, degree_weights, index)
    dhw_electric = thermal["monthly_hot_water_fuel_kwh"] if fuel.hot_water_fuel == "electricity" else zero
    dhw_electric = dhw_electric + thermal.get("monthly_shower_electricity_kwh", zero)
    dhw = _allocate(dhw_electric, x["hot_water_weights"], index)
    extra = _allocate(thermal.get("monthly_auxiliary_electricity_kwh", zero), x["extra_weights"], index)
    demand = appliance + lighting + heat + dhw + extra
    available_pv = params["pv_kwp"] * x["pv_kwh_per_kwp"]
    pv = jnp.minimum(available_pv, x["pv_inverter_kw"] * x["duration_hours"])
    capacity = params["battery_kwh"] * x["battery_usable_fraction"]
    initial = capacity * x["battery_initial_fraction"]
    flows = _dispatch(
        demand,
        pv,
        x["duration_hours"],
        capacity,
        x["battery_charge_kw"],
        x["battery_discharge_kw"],
        x["battery_charge_efficiency"],
        x["battery_discharge_efficiency"],
        initial,
        x["export_limit_kw"],
    )
    output = {f"interval_{key}_kwh": value for key, value in flows.items()}
    output.update(
        interval_appliances_kwh=appliance,
        interval_lighting_kwh=lighting,
        interval_heating_kwh=heat,
        interval_hot_water_kwh=dhw,
        interval_extra_kwh=extra,
        interval_demand_kwh=demand,
        interval_pv_generation_kwh=pv,
        interval_pv_clipping_kwh=available_pv - pv,
        battery_initial_kwh=initial,
        battery_final_kwh=flows["stored"][-1],
        battery_usable_capacity_kwh=capacity,
        peak_import_kw=jnp.max(flows["import"] / x["duration_hours"]),
        heating_shape_fallback_months=jnp.sum((degree_sums == 0) & (heating_electric > 0)),
    )
    monthly_map = {
        "monthly_electricity_kwh": "import",
        "monthly_export_kwh": "export",
        "monthly_battery_loss_kwh": "battery_loss",
        "monthly_battery_delta_kwh": "battery_delta",
        "monthly_curtailment_kwh": "curtailment",
        "monthly_direct_pv_kwh": "direct_pv",
        "monthly_battery_charge_kwh": "charge",
        "monthly_battery_discharge_kwh": "discharge",
    }
    for key, flow in monthly_map.items():
        output[key] = _monthly_sum(flows[flow], index, n_months)
    output["monthly_pv_generation_kwh"] = _monthly_sum(pv, index, n_months)
    output["monthly_electricity_demand_kwh"] = _monthly_sum(demand, index, n_months)
    for total, monthly in (
        ("electricity_import_kwh", "monthly_electricity_kwh"),
        ("pv_generation_kwh", "monthly_pv_generation_kwh"),
        ("export_kwh", "monthly_export_kwh"),
    ):
        output[total] = jnp.sum(output[monthly])
    return jax.tree.map(lambda value: jnp.where(valid, value, jnp.nan), output)
