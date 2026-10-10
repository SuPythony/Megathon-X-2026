"""The seam between the twin wrapper (Part 1, Satyarth) and the physics model (teammate).

The physics owner writes ``run_physics`` in ``greentwin/twin/physics.py`` with the signature below. Until that file
exists, ``get_run_physics()`` returns the stand-in from ``physics_stub.py``, so the whole pipeline already runs.
Spec for the physics owner: ``docs/digitaltwinvariables.md``. How to plug in: ``docs/DIGITAL_TWIN_HANDOFF.md``.

    run_physics(params: dict[str, jax.Array],    # one home, one sample: PHYSICS_PARAM_KEYS -> scalars
                weather: dict[str, jax.Array],   # WEATHER_KEYS -> arrays of shape (n_months,)
                mode: "calibrated" | "standard", # static
                fuel: FuelConfig)                # static
        -> dict[str, jax.Array]                  # OUTPUT_KEYS (+ element_heat_loss_w_per_k dict)

Must be pure ``jax.numpy``, differentiable and ``jax.vmap``-able over ``params``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Literal

FuelName = Literal["mains_gas", "electricity", "oil", "lpg", "solid", "district"]


@dataclass(frozen=True)
class FuelConfig:
    """Static (not traced): the same for every posterior sample of a home."""

    heating_fuel: FuelName
    hot_water_fuel: FuelName


# Numeric TwinParams fields passed to physics (property_id and tariff are not physics inputs).
PHYSICS_PARAM_KEYS: tuple[str, ...] = (
    "wall_area_m2", "roof_area_m2", "floor_area_m2", "window_area_m2", "door_area_m2", "party_wall_area_m2",
    "wall_u", "roof_u", "floor_u", "window_u", "door_u", "thermal_bridging_y", "air_changes_per_hour", "volume_m3",
    "heat_loss_multiplier", "setpoint_c", "heating_hours_per_day", "heating_efficiency", "hot_water_kwh_per_day",
    "baseload_kwh_per_day", "lighting_share_of_baseload", "internal_gains_w", "solar_gain_factor", "pv_kwp",
    "pv_azimuth_deg", "pv_tilt_deg", "battery_kwh",
)

# Behaviour fields: in "standard" mode the wrapper passes the home's uncalibrated SAP-based values for these,
# never the calibrated samples (the rating must not depend on how this household behaves).
BEHAVIOUR_KEYS: tuple[str, ...] = (
    "setpoint_c", "heating_hours_per_day", "hot_water_kwh_per_day", "baseload_kwh_per_day", "internal_gains_w",
)

WEATHER_KEYS: tuple[str, ...] = ("hdd_15_5", "mean_temp_c", "ghi_kwh_m2", "days_in_month", "month_of_year")

TOTAL_KEYS: tuple[str, ...] = (
    "electricity_import_kwh", "gas_kwh", "space_heating_kwh", "hot_water_kwh", "pv_generation_kwh", "export_kwh",
)
MONTHLY_KEYS: tuple[str, ...] = ("monthly_electricity_kwh", "monthly_gas_kwh")
ELEMENTS: tuple[str, ...] = ("roof", "walls", "floor", "windows", "doors", "ventilation", "party_walls")

RunPhysics = Callable[..., dict]


def get_run_physics() -> tuple[RunPhysics, str]:
    """The teammate's ``greentwin.twin.physics.run_physics`` if it exists, else the stub. Returns (fn, engine name)."""
    try:
        module = importlib.import_module("greentwin.twin.physics")
        return module.run_physics, "physics"
    except ModuleNotFoundError as e:
        if e.name != "greentwin.twin.physics":
            raise  # physics.py exists but one of its imports is missing: surface that, don't hide it
        from greentwin.twin.physics_stub import run_physics

        return run_physics, "physics_stub"
