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
