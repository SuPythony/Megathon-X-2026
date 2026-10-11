"""Read-only adapter for Satyarth's physics_inputs/0.1 JSON handoff."""

from dataclasses import dataclass

import numpy as np

from contracts.electricity import ElectricityIntake
from contracts.models import TwinParams, WeatherMonth
from greentwin.twin.physics_api import FuelConfig


@dataclass
class AdaptedInputs:
    params: TwinParams
    extra_params: dict[str, float]
    fuel: FuelConfig
    weather: list[WeatherMonth]
    profiles: dict[str, np.ndarray]
    intake: ElectricityIntake
    appliance_hourly_profile_kw: list[float] | None
    sources: dict


def adapt_satyarth_package(package: dict) -> AdaptedInputs:
    """Preserve household totals and separate end uses; never count whole-home hourly totals as baseload."""
    if package.get("inputs_version") != "physics_inputs/0.1":
        raise ValueError("Expected Satyarth physics_inputs/0.1 package")
    p = package["params"]
    params = TwinParams.model_validate(p)
    hot_water = package["hot_water"]
    if params.property_id != package["property_id"]:
        raise ValueError("Package and parameter property identities differ")
    fuel = FuelConfig(params.heating_fuel, hot_water["fuel"], hot_water["efficiency"])
    extra = {
        "hot_water_standing_loss_kwh_per_day": hot_water["standing_loss_kwh_per_day"],
        "shower_electric_kwh_per_day": hot_water["shower_electric_kwh_per_day"],
    }
    w = package["weather"]
    months, arrays = w["months"], w["arrays"]
    if any(len(arrays[key]) != len(months) for key in ("mean_temp_c", "ghi_kwh_m2", "hdd_15_5")):
        raise ValueError("Package weather arrays do not match months")
    # Contract source distinguishes synthetic fallback from factual weather.
    source = "synthetic" if "synthetic" in w["source"].lower() else w["source"]
    weather = [
        WeatherMonth(
            area_code=package["home"]["postcode_district"],
            month=month,
            hdd_15_5=arrays["hdd_15_5"][i],
            mean_temp_c=arrays["mean_temp_c"][i],
            ghi_kwh_m2=arrays["ghi_kwh_m2"][i],
            source=source,
        )
        for i, month in enumerate(months)
    ]
    schedules = package.get("schedules", {})
    rename = {
        "appliance_month_factor": "baseload_multiplier",
        "gains_month_factor": "gains_multiplier",
        "secondary_heating_kwh_monthly": "secondary_heating_kwh_monthly",
        "extra_electricity_kwh_monthly": "extra_electricity_kwh_monthly",
        "extra_gas_kwh_monthly": "extra_gas_kwh_monthly",
    }
    profiles = {}
    for key, target in rename.items():
        if key in schedules:
            values = np.asarray(schedules[key], dtype=float)
            if values.shape != (len(months),) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid {key} schedule")
            profiles[target] = values
    occupancy = package.get("occupancy", {})
    behaviour = package.get("heating_behaviour", {})
    intake = ElectricityIntake(
        presence=occupancy.get("presence"), heating_periods=behaviour.get("periods", [])
    )
    appliances = package.get("appliances")
    # The aggregate supplied by Satyarth excludes DHW/secondary/outdoor loads.
    hourly = appliances.get("hourly_profile_kw") if appliances else None
    if hourly is not None and sum(hourly) == 0:
        hourly = None  # explicit no appliance timing information, not zero demand
    return AdaptedInputs(params, extra, fuel, weather, profiles, intake, hourly, package.get("sources", {}))
