"""Prepare electricity inputs independently of JAX; synthetic profiles are labelled."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

from contracts.electricity import (
    BatteryInputs,
    ElectricityIntake,
    ElectricitySeries,
    PreparedElectricityInputs,
)
from contracts.models import TwinParams, WeatherMonth

LONDON = ZoneInfo("Europe/London")


@lru_cache(maxsize=8)
def timeline(months: tuple[str, ...]):
    """Full local calendar months, continuous UTC intervals including DST changes."""
    dates, index = [], []
    previous_end = None
    for i, month in enumerate(months):
        year, number = map(int, month.split("-"))
        start = datetime(year, number, 1, tzinfo=LONDON)
        end = datetime(year + (number == 12), 1 if number == 12 else number + 1, 1, tzinfo=LONDON)
        if i and start.astimezone(UTC) != previous_end:
            raise ValueError("Electricity simulation requires consecutive whole calendar months")
        t = start.astimezone(UTC)
        previous_end = end.astimezone(UTC)
        while t < previous_end:
            dates.append(t)
            index.append(i)
            t += timedelta(minutes=30)
    return tuple(dates), np.asarray(index, dtype=np.int32)


def _solar_geometry(dates, latitude, longitude, azimuth, tilt):
    # Approximate astronomical position. Generated geometry, not a PVGIS substitute.
    day = np.array([d.timetuple().tm_yday for d in dates])
    hour = np.array([d.hour + d.minute / 60 + 0.25 for d in dates]) + longitude / 15
    declination = np.deg2rad(23.44) * np.sin(2 * np.pi * (day - 81) / 365.25)
    angle = np.pi * (hour - 12) / 12
    lat, az, slope = np.deg2rad([latitude, azimuth, tilt])
    up = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(angle)
    north = np.cos(lat) * np.sin(declination) - np.sin(lat) * np.cos(declination) * np.cos(angle)
    east = -np.cos(declination) * np.sin(angle)
    incident = east * np.sin(az) * np.sin(slope) + north * np.cos(az) * np.sin(slope) + up * np.cos(slope)
    return np.maximum(up, 0), np.maximum(incident, 0)


def prepare_electricity_inputs(
    params: TwinParams,
    weather: list[WeatherMonth],
    *,
    intake: ElectricityIntake | None = None,
    seed: int = 42,
    development: bool = False,
    latitude: float | None = None,
    longitude: float | None = None,
    series: ElectricitySeries | None = None,
    appliance_hourly_profile_kw: list[float] | None = None,
) -> PreparedElectricityInputs:
    """Explicit development generation or externally sourced weather/PV series.

    Non-development mode never invents equipment specifications or weather.
    Missing appliance timing may use labelled archetype assumptions in either mode.
    """
    intake = intake or ElectricityIntake()
    months = tuple(w.month for w in weather)
    if not weather or months != tuple(sorted(set(months))):
        raise ValueError("Unique chronological weather months are required")
    if len({w.area_code for w in weather}) != 1:
        raise ValueError("Weather must describe one area")
    if params.pv_kwp < 0 or params.battery_kwh < 0:
        raise ValueError("PV and battery capacities cannot be negative")
    dates, index = timeline(months)
    local = [d.astimezone(LONDON) for d in dates]
    hour = np.array([d.hour + d.minute / 60 + 0.25 for d in local])
    weekday = np.array([d.weekday() for d in local])
    timestamps = [d.isoformat().replace("+00:00", "Z") for d in dates]
    rng = np.random.default_rng(seed)
    sources, warnings = {}, []
    if series is not None:
        if series.timestamps_utc != timestamps:
            raise ValueError("Sourced interval series must cover the exact UTC timeline")
        if not development and (
            "synthetic" in series.temperature_source.lower()
            or (params.pv_kwp > 0 and "synthetic" in series.pv_source.lower())
        ):
            raise ValueError("Synthetic series require explicit development mode")
        temperature = np.asarray(series.temperature_c)
        pv = np.asarray(series.pv_kwh_per_kwp)
        sources.update(temperature=series.temperature_source, pv=series.pv_source)
    elif development:
        if any(w.mean_temp_c is None or w.ghi_kwh_m2 is None for w in weather):
            raise ValueError("Generated profiles require explicit monthly temperature and irradiation")
        if latitude is None or longitude is None:
            latitude, longitude = 52.0, 0.0
            warnings.append("Location missing: synthetic 52 N, 0 E solar geometry assumption")
        if not np.isfinite([latitude, longitude]).all() or not (
            -90 < latitude < 90 and -180 <= longitude <= 180
        ):
            raise ValueError("Invalid solar geometry coordinates")
        up, incident = _solar_geometry(dates, latitude, longitude, params.pv_azimuth_deg, params.pv_tilt_deg)
        perturbation = 2 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1, len(dates))
        temperature, ghi = np.zeros(len(dates)), np.zeros(len(dates))
        # Clouds persist for a day, avoiding independent noisy sunshine at every step.
        day_keys = np.array([d.date().toordinal() for d in local])
        _, day_index = np.unique(day_keys, return_inverse=True)
        cloud = rng.uniform(0.35, 1.0, day_index.max() + 1)[day_index]
        daylight = up * cloud
        for i, record in enumerate(weather):
            if not np.isfinite([record.mean_temp_c, record.ghi_kwh_m2]).all() or record.ghi_kwh_m2 < 0:
                raise ValueError("Monthly weather must be finite with nonnegative irradiation")
            mask = index == i
            temperature[mask] = record.mean_temp_c + perturbation[mask] - perturbation[mask].mean()
            weights = daylight[mask]
            if weights.sum() == 0 and record.ghi_kwh_m2 > 0:
                raise ValueError("No daylight available for supplied irradiation")
            ghi[mask] = record.ghi_kwh_m2 * weights / max(weights.sum(), 1e-12)
        slope = np.deg2rad(params.pv_tilt_deg)
        plane = ghi * (
            0.8 * incident / np.maximum(up, 0.05)
            + 0.2 * (1 + np.cos(slope)) / 2
            + 0.2 * (1 - np.cos(slope)) / 2
        )
        pv = np.where(up > 0, 0.80 * plane, 0)
        sources.update(
            temperature="synthetic: centred on existing monthly mean",
            pv="synthetic: monthly GHI + approximate solar geometry",
        )
        warnings.append(
            "Generated half-hour weather/PV is a conditional development scenario, not measured or PVGIS data"
        )
    else:
        raise ValueError("Supply sourced ElectricitySeries, or explicitly enable development generation")

    presence = intake.presence or "part_day"
    morning = np.exp(-0.5 * ((hour - 7.5) / 1.3) ** 2)
    evening = np.exp(-0.5 * ((hour - 19) / 2) ** 2)
    daytime = np.exp(-0.5 * ((hour - 13) / 3) ** 2)
    home_day = (weekday >= 5) | (presence == "home_all_day")
    daytime_weight = np.where(home_day, 0.8, 0.1 if presence == "out_all_day" else 0.4)
    if appliance_hourly_profile_kw is not None:
        shape = np.asarray(appliance_hourly_profile_kw, dtype=float)
        if shape.shape != (24,) or not np.isfinite(shape).all() or (shape < 0).any() or shape.sum() <= 0:
            raise ValueError("Appliance profile must contain 24 nonnegative hourly values with positive sum")
        base = shape[np.floor(hour).astype(int)]
        sources["baseload_shape"] = (
            "supplied appliance hourly profile (renormalised); includes no heating/DHW"
        )
    else:
        base = 0.35 + morning * 0.5 + evening + daytime * daytime_weight
        sources["baseload_shape"] = f"assumed {presence} archetype; not fitted to London smart meters"
    # Separate day-to-day fluctuations preserve monthly energy through solver normalisation.
    day_keys = np.array([d.date().toordinal() for d in local])
    _, day_index = np.unique(day_keys, return_inverse=True)
    base = base * rng.lognormal(-0.5 * 0.15**2, 0.15, day_index.max() + 1)[day_index]
    lighting = 0.1 + morning * 0.4 + evening
    dhw = 0.05 + morning + evening * 0.6
    periods = intake.heating_periods
    if periods:
        heating = np.zeros(len(dates))
        for start, end in periods:
            heating += ((hour >= start) & (hour < end)) if end > start else ((hour >= start) | (hour < end))
        sources["heating_schedule"] = "supplied heating periods (local clock)"
    else:
        # Schedule length follows the existing synthetic household's heating hours.
        length = params.heating_hours_per_day
        distance = np.mod(hour - 15, 24)
        heating = np.clip((length - distance + 0.25) / 0.5, 0, 1)
        sources["heating_schedule"] = (
            "assumed block starting 15:00; duration from existing thermal parameters"
        )

    battery = BatteryInputs()
    if params.battery_kwh > 0:
        keys = ("charge_kw", "discharge_kw", "usable_fraction", "charge_efficiency", "discharge_efficiency")
        defaults = (0.5 * params.battery_kwh, 0.5 * params.battery_kwh, 0.90, 0.95, 0.95)
        values = {}
        for key, default in zip(keys, defaults):
            value = getattr(intake, f"battery_{key}")
            if value is None and not development:
                raise ValueError(f"Battery needs {key} from model/installer specifications")
            values[key] = default if value is None else value
            sources[f"battery_{key}"] = (
                "development assumption" if value is None else "supplied equipment specification"
            )
        battery = BatteryInputs(**values, initial_fraction=intake.initial_battery_fraction)
    elif intake.initial_battery_fraction > 0:
        raise ValueError("Initial battery charge requires an installed battery")
    inverter = 0.0
    if params.pv_kwp > 0:
        if intake.pv_inverter_kw is None and not development:
            raise ValueError("PV needs inverter power from equipment specifications")
        inverter = intake.pv_inverter_kw if intake.pv_inverter_kw is not None else params.pv_kwp
        sources["pv_inverter_kw"] = (
            "supplied specification"
            if intake.pv_inverter_kw is not None
            else "development assumption: inverter kW = PV kWp"
        )
    sources["export_limit"] = (
        "supplied connection limit"
        if intake.export_limit_kw is not None
        else "assumed unrestricted; confirm connection limit"
    )
    sources["tariffs"] = "not required for energy; ToU and export prices remain unavailable until sourced"
    return PreparedElectricityInputs(
        property_id=params.property_id,
        months=list(months),
        seed=seed,
        development=development,
        pv_azimuth_deg=params.pv_azimuth_deg,
        pv_tilt_deg=params.pv_tilt_deg,
        battery=battery,
        pv_inverter_kw=inverter,
        export_limit_kw=intake.export_limit_kw,
        sources=sources,
        warnings=warnings,
        timestamps_utc=timestamps,
        month_index=index.tolist(),
        local_hour=hour.tolist(),
        duration_hours=[0.5] * len(dates),
        temperature_c=temperature.tolist(),
        pv_kwh_per_kwp=pv.tolist(),
        baseload_weights=base.tolist(),
        lighting_weights=lighting.tolist(),
        heating_on_fraction=heating.tolist(),
        hot_water_weights=dhw.tolist(),
        extra_weights=(0.1 + evening).tolist(),
    )


def development_electricity_factory(prop, params, weather, latent, seed):
    """Generator hook: same household/seed for baseline and paired scenarios."""
    presence = {"mostly_away": "out_all_day", "evenings": "part_day", "mostly_home": "home_all_day"}[
        latent.behaviour
    ]
    return prepare_electricity_inputs(
        params,
        list(weather),
        development=True,
        seed=seed,
        latitude=prop.location.lat,
        longitude=prop.location.lon,
        intake=ElectricityIntake(presence=presence),
    )
