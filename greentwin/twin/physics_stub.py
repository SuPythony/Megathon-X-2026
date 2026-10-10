"""STAND-IN physics: a simple monthly steady-state model with the exact ``run_physics`` signature.

It exists so the twin pipeline (params -> weather -> posterior samples -> simulate -> TwinRun) runs end to end
before the real physics lands. It is NOT the physics model: the teammate's ``greentwin/twin/physics.py`` replaces it
automatically (see ``physics_api.get_run_physics``). Every simplification is listed in STUB_ASSUMPTIONS; outputs
built on it carry provenance engine ``physics_stub``.

Written in jax.numpy, pure, smooth (softplus instead of max) and vmap-able, so calibration code built on it works
unchanged with the real model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from greentwin.twin.physics_api import FuelConfig

STUB_ASSUMPTIONS = (
    "Monthly steady state: heat loss = HLC x (internal - external temperature) x hours; no thermal mass.",
    "Mean internal temperature = setpoint - 1.5 K x (16 - heating hours) / 16.",
    "Party walls lose no heat (U = 0) until a party-wall U-value is agreed.",
    "Solar gains = window area x GHI x 0.5 (vertical/horizontal) x 0.63 (g-value) x 0.7 (frame, shading).",
    "Gain utilisation fixed at 0.9.",
    "PV yield = kWp x GHI x 0.85; self-consumption 30%, rising to 65% with a large battery.",
    "Hot water: its own efficiency (hot_water_efficiency) and fuel; cylinder standing loss added to the demand.",
    "Electric showers add electricity at efficiency 1; plug-in heaters add electricity and displace main heating.",
    "Standard mode: setpoint 19.0 C and 11 heating hours/day (SAP-style), other behaviour from the caller.",
    "Non-gas fossil fuels (oil, LPG, solid, district) are returned as other_fuel_kwh (provisional key).",
)

_SMOOTH_KWH = 5.0


def _smooth_pos(x):
    return jax.nn.softplus(x / _SMOOTH_KWH) * _SMOOTH_KWH


def run_physics(params: dict, weather: dict, mode: str, fuel: FuelConfig) -> dict:
    p = params
    days = weather["days_in_month"]
    t_out = weather["mean_temp_c"]
    ghi = weather["ghi_kwh_m2"]
    ones, zeros = jnp.ones_like(days), jnp.zeros_like(days)
    appliance_factor = weather.get("appliance_month_factor", ones)
    gains_factor = weather.get("gains_month_factor", ones)
    secondary = weather.get("secondary_heating_kwh_monthly", zeros)
    extra_elec = weather.get("extra_electricity_kwh_monthly", zeros)
    extra_gas = weather.get("extra_gas_kwh_monthly", zeros)

    setpoint = jnp.asarray(19.0) if mode == "standard" else p["setpoint_c"]
    hours = jnp.asarray(11.0) if mode == "standard" else p["heating_hours_per_day"]

    m = p["heat_loss_multiplier"]
    exposed = p["wall_area_m2"] + p["roof_area_m2"] + p["floor_area_m2"] + p["window_area_m2"] + p["door_area_m2"]
    elements = {
        "walls": m * (p["wall_u"] * p["wall_area_m2"] + p["thermal_bridging_y"] * exposed),
        "roof": m * p["roof_u"] * p["roof_area_m2"],
        "floor": m * p["floor_u"] * p["floor_area_m2"],
        "windows": m * p["window_u"] * p["window_area_m2"],
        "doors": m * p["door_u"] * p["door_area_m2"],
        "ventilation": m * 0.33 * p["air_changes_per_hour"] * p["volume_m3"],
        "party_walls": m * 0.0 * p["party_wall_area_m2"],
    }
    hlc = sum(elements.values())  # W/K

    t_in = setpoint - 1.5 * (16.0 - hours) / 16.0
    loss_kwh = hlc * jax.nn.softplus(t_in - t_out) * 24.0 * days / 1000.0
    solar_kwh = p["solar_gain_factor"] * p["window_area_m2"] * ghi * 0.5 * 0.63 * 0.7
    internal_kwh = p["internal_gains_w"] * gains_factor * 24.0 * days / 1000.0
    space_heat = _smooth_pos(loss_kwh - 0.9 * (solar_kwh + internal_kwh))  # useful heat, kWh/month
    main_heat = _smooth_pos(space_heat - secondary)  # plug-in heaters cover part of it

    hot_water = (p["hot_water_kwh_per_day"] + p.get("hot_water_standing_loss_kwh_per_day", 0.0)) * days
    heat_fuel = main_heat / p["heating_efficiency"]
    dhw_fuel = hot_water / p.get("hot_water_efficiency", p["heating_efficiency"])
    shower = p.get("shower_electric_kwh_per_day", 0.0) * days

    zero = jnp.zeros_like(days)
    by_fuel = {"electricity": zero, "mains_gas": zero, "other": zero}

    def add(f, kwh):
        key = f if f in ("electricity", "mains_gas") else "other"
        by_fuel[key] = by_fuel[key] + kwh

    add(fuel.heating_fuel, heat_fuel)
    add(fuel.hot_water_fuel, dhw_fuel)

    pv = p["pv_kwp"] * ghi * 0.85
    self_share = 0.30 + 0.35 * (1.0 - jnp.exp(-p["battery_kwh"] / 5.0))
    demand_elec = (p["baseload_kwh_per_day"] * days * appliance_factor + by_fuel["electricity"] + shower + secondary
                   + extra_elec)
    self_used = jnp.minimum(self_share * pv, demand_elec)
    export = pv - self_used
    monthly_elec = demand_elec - self_used
    monthly_gas = by_fuel["mains_gas"] + extra_gas

    return {
        "electricity_import_kwh": monthly_elec.sum(),
        "gas_kwh": monthly_gas.sum(),
        "space_heating_kwh": space_heat.sum(),
        "hot_water_kwh": (hot_water + shower).sum(),
        "pv_generation_kwh": pv.sum(),
        "export_kwh": export.sum(),
        "monthly_electricity_kwh": monthly_elec,
        "monthly_gas_kwh": monthly_gas,
        "other_fuel_kwh": by_fuel["other"].sum(),
        "element_heat_loss_w_per_k": elements,
    }
