"""Satyarth-compatible, static equipment configuration for differentiable physics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FuelConfig:
    heating_fuel: str
    hot_water_fuel: str
    # None means same equipment as space heating. Explicit for immersion/independent DHW.
    hot_water_efficiency: float | None = None


PHYSICS_PARAM_KEYS = (
    "wall_area_m2",
    "roof_area_m2",
    "floor_area_m2",
    "window_area_m2",
    "door_area_m2",
    "party_wall_area_m2",
    "wall_u",
    "roof_u",
    "floor_u",
    "window_u",
    "door_u",
    "thermal_bridging_y",
    "air_changes_per_hour",
    "volume_m3",
    "heat_loss_multiplier",
    "setpoint_c",
    "heating_hours_per_day",
    "heating_efficiency",
    "hot_water_kwh_per_day",
    "baseload_kwh_per_day",
    "lighting_share_of_baseload",
    "internal_gains_w",
    "solar_gain_factor",
    "pv_kwp",
    "pv_azimuth_deg",
    "pv_tilt_deg",
    "battery_kwh",
)


def get_run_physics():
    from greentwin.twin.physics import run_physics

    return run_physics, "physics_l1"


# --------------------------------------------------------------------------- twin wrapper (Part 1) additions
# Added on merge so the twin wrapper, scenarios and the physics inputs package keep importing from this seam.
# P2's definitions above are unchanged. Spec: docs/digitaltwinvariables.md; plug-in: docs/DIGITAL_TWIN_HANDOFF.md.
from typing import Callable, Literal  # noqa: E402

FuelName = Literal["mains_gas", "electricity", "oil", "lpg", "solid", "district"]

# Extra numeric inputs from ExtendedTwinParams; optional for run_physics, defaults in brackets.
EXTRA_PARAM_KEYS: tuple[str, ...] = (
    "hot_water_efficiency",                 # [heating_efficiency] delivered hot water / fuel, main DHW system
    "hot_water_standing_loss_kwh_per_day",  # [0] cylinder loss, extra useful heat the DHW system must supply
    "shower_electric_kwh_per_day",          # [0] electric showers: electricity at efficiency 1
    "heat_capacity_kj_per_k",               # [250 x volume / 2.5] thermal mass; time constant = C / (3.6 x HLC) h
    "secondary_heating_scale",              # [1] multiplies secondary_heating_kwh_monthly (plug-in heater use)
    "battery_round_trip_efficiency",        # [0.9] PV energy shifted through the battery loses (1 - this)
)

# Behaviour fields: in "standard" mode the wrapper passes the home's uncalibrated SAP-based values for these.
BEHAVIOUR_KEYS: tuple[str, ...] = (
    "setpoint_c", "heating_hours_per_day", "hot_water_kwh_per_day", "baseload_kwh_per_day", "internal_gains_w",
)

WEATHER_KEYS: tuple[str, ...] = ("hdd_15_5", "mean_temp_c", "ghi_kwh_m2", "days_in_month", "month_of_year")

# Monthly schedules from the appliance list, passed with the weather (shape (n_months,)); defaults in brackets.
SCHEDULE_KEYS: tuple[str, ...] = (
    "appliance_month_factor",          # [1] baseload shape
    "gains_month_factor",              # [1] internal gains shape
    "secondary_heating_kwh_monthly",   # [0] plug-in electric heaters
    "extra_electricity_kwh_monthly",   # [0] outside the envelope (EV charger, hot tub)
    "extra_gas_kwh_monthly",           # [0] gas cooking
)

TOTAL_KEYS: tuple[str, ...] = (
    "electricity_import_kwh", "gas_kwh", "space_heating_kwh", "hot_water_kwh", "pv_generation_kwh", "export_kwh",
)
MONTHLY_KEYS: tuple[str, ...] = ("monthly_electricity_kwh", "monthly_gas_kwh")
ELEMENTS: tuple[str, ...] = ("roof", "walls", "floor", "windows", "doors", "ventilation", "party_walls")

RunPhysics = Callable[..., dict]
