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
    "Monthly quasi-steady state with thermal mass: time constant tau = C / (3.6 x HLC) hours.",
    "Heating off for 24 - hours each day: the home cools exponentially with tau; mean internal temperature is the "
    "day average (heavier homes cool less, so intermittent heating saves less).",
    "Gain utilisation from SAP 2012 Table 9a: eta = (1 - g^a) / (1 - g^(a+1)), a = 1 + tau / 15, g = gains / losses.",
    "Party walls lose no heat (U = 0) until a party-wall U-value is agreed.",
    "Solar gains = window area x GHI x 0.5 (vertical/horizontal) x 0.63 (g-value) x 0.7 (frame, shading).",
    "PV yield = kWp x GHI x 0.85; self-consumption 30%, plus up to 35% more through a battery at its round-trip efficiency.",
    "Hot water: its own efficiency (hot_water_efficiency) and fuel; cylinder standing loss added to the demand.",
    "Electric showers add electricity at efficiency 1; plug-in heaters add electricity and displace main heating.",
    "Standard mode: setpoint 19.0 C and 11 heating hours/day (SAP-style), other behaviour from the caller.",
    "Non-gas fossil fuels (oil, LPG, solid, district) are returned as other_fuel_kwh (provisional key).",
)

_SMOOTH_KWH = 5.0


def _smooth_pos(x):
    return jax.nn.softplus(x / _SMOOTH_KWH) * _SMOOTH_KWH


def _utilisation(gains, losses, tau):
    """SAP 2012 Table 9a gain utilisation factor; the gamma = 1 case uses its limit a / (a + 1).
    Double jnp.where keeps gradients finite at the singular point."""
    a = 1.0 + tau / 15.0
    gamma = jnp.clip(gains / jnp.maximum(losses, 1e-6), 0.0, 1e3)
    near_one = jnp.abs(gamma - 1.0) < 1e-3
    above = gamma > 1.0
    g_lo = jnp.where(near_one | above, 0.5, gamma)          # gamma < 1: direct form
    g_hi = jnp.where(near_one | ~above, 2.0, gamma)         # gamma > 1: divided by g^(a+1), no overflow
    eta_lo = (1.0 - g_lo ** a) / (1.0 - g_lo ** (a + 1.0))
    inv = 1.0 / g_hi
    eta_hi = (inv - inv ** (a + 1.0)) / (1.0 - inv ** (a + 1.0))
    return jnp.where(near_one, a / (a + 1.0), jnp.where(above, eta_hi, eta_lo))


def run_physics(params: dict, weather: dict, mode: str, fuel: FuelConfig) -> dict:
    p = params
    days = weather["days_in_month"]
    t_out = weather["mean_temp_c"]
    ghi = weather["ghi_kwh_m2"]
    ones, zeros = jnp.ones_like(days), jnp.zeros_like(days)
    appliance_factor = weather.get("appliance_month_factor", ones)
    gains_factor = weather.get("gains_month_factor", ones)
    secondary = weather.get("secondary_heating_kwh_monthly", zeros) * p.get("secondary_heating_scale", 1.0)
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

    # Thermal mass: time constant (hours). Default heat capacity: SAP medium (250 kJ/m2K) on floor area ~ volume / 2.5.
    capacity = p.get("heat_capacity_kj_per_k", 250.0 * p["volume_m3"] / 2.5)
    tau = capacity / (3.6 * hlc)

    # Intermittent heating: off for t_off hours, cooling towards outside with time constant tau.
    t_off = jnp.clip(24.0 - hours, 0.01, 24.0)
    off_drop = 1.0 - (tau / t_off) * (1.0 - jnp.exp(-t_off / tau))  # mean relative drop while off
    t_in = setpoint - (setpoint - t_out) * off_drop * t_off / 24.0
    loss_kwh = hlc * jax.nn.softplus(t_in - t_out) * 24.0 * days / 1000.0

    solar_kwh = p["solar_gain_factor"] * p["window_area_m2"] * ghi * 0.5 * 0.63 * 0.7
    internal_kwh = p["internal_gains_w"] * gains_factor * 24.0 * days / 1000.0
    gains = solar_kwh + internal_kwh
    space_heat = _smooth_pos(loss_kwh - _utilisation(gains, loss_kwh, tau) * gains)  # useful heat, kWh/month
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
    demand_elec = (p["baseload_kwh_per_day"] * days * appliance_factor + by_fuel["electricity"] + shower + secondary
                   + extra_elec)
    # 30% of PV is used as it is made; a battery stores up to 35% more for later, losing (1 - round trip) of it.
    rte = p.get("battery_round_trip_efficiency", 0.9)
    direct = jnp.minimum(0.30 * pv, demand_elec)
    stored = jnp.minimum(0.35 * (1.0 - jnp.exp(-p["battery_kwh"] / 5.0)) * pv, (demand_elec - direct) / rte)
    self_used = direct + stored * rte
    export = pv - direct - stored
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
