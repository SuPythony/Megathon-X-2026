"""Electricity input extensions; existing shared schemas remain unchanged."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ElectricityIntake(BaseModel):
    """Easy optional household answers, plus installer/device specification values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    presence: Literal["out_all_day", "part_day", "home_all_day"] | None = None
    heating_periods: list[tuple[float, float]] = Field(default_factory=list)
    battery_charge_kw: float | None = Field(None, gt=0)
    battery_discharge_kw: float | None = Field(None, gt=0)
    battery_usable_fraction: float | None = Field(None, gt=0, le=1)
    battery_charge_efficiency: float | None = Field(None, gt=0, le=1)
    battery_discharge_efficiency: float | None = Field(None, gt=0, le=1)
    initial_battery_fraction: float = Field(0, ge=0, le=1)
    pv_inverter_kw: float | None = Field(None, gt=0)
    export_limit_kw: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def periods_valid(self):
        occupied = np.zeros(1440, dtype=bool)
        for start, end in self.heating_periods:
            if not (0 <= start < 24 and 0 <= end <= 24) or start == end:
                raise ValueError("Use distinct start/end times in 0..24; full day is (0,24)")
            indices = (
                np.arange(round(start * 60), round(end * 60))
                if end > start
                else np.concatenate((np.arange(round(start * 60), 1440), np.arange(0, round(end * 60))))
            )
            if occupied[indices].any():
                raise ValueError("Heating periods overlap")
            occupied[indices] = True
        return self


class BatteryInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    usable_fraction: float = Field(1, gt=0, le=1)
    charge_kw: float = Field(0, ge=0)
    discharge_kw: float = Field(0, ge=0)
    charge_efficiency: float = Field(1, gt=0, le=1)
    discharge_efficiency: float = Field(1, gt=0, le=1)
    initial_fraction: float = Field(0, ge=0, le=1)


class ElectricitySeries(BaseModel):
    """Externally prepared interval data. PV yield is AC kWh per installed kWp."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamps_utc: list[str]
    temperature_c: list[float]
    pv_kwh_per_kwp: list[float]
    temperature_source: str
    pv_source: str


class PreparedElectricityInputs(BaseModel):
    """Validated, reproducible input package before invoking any JAX solver."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: str = "electricity-inputs/0.1.0"
    property_id: str
    months: list[str]
    seed: int = Field(ge=0)
    development: bool
    pv_azimuth_deg: float
    pv_tilt_deg: float
    battery: BatteryInputs
    pv_inverter_kw: float = Field(ge=0)
    export_limit_kw: float | None = Field(None, ge=0)
    sources: dict[str, str]
    warnings: list[str]
    timestamps_utc: list[str]
    month_index: list[int]
    local_hour: list[float]
    duration_hours: list[float]
    temperature_c: list[float]
    pv_kwh_per_kwp: list[float]
    baseload_weights: list[float]
    lighting_weights: list[float]
    heating_on_fraction: list[float]
    hot_water_weights: list[float]
    extra_weights: list[float]

    @model_validator(mode="after")
    def series_valid(self):
        n = len(self.timestamps_utc)
        if n == 0 or not self.months or self.months != sorted(set(self.months)):
            raise ValueError("A nonempty chronological timeline is required")
        fields = (
            "month_index",
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
        for name in fields:
            a = np.asarray(getattr(self, name))
            if a.shape != (n,) or not np.isfinite(a).all():
                raise ValueError(f"{name} must be finite and aligned to the timeline")
            if name != "temperature_c" and (a < 0).any():
                raise ValueError(f"{name} must be nonnegative")
        index = np.asarray(self.month_index)
        if set(index) != set(range(len(self.months))) or (np.diff(index) < 0).any():
            raise ValueError("Every month needs ordered intervals")
        if not np.allclose(self.duration_hours, 0.5) or max(self.local_hour) >= 24:
            raise ValueError("Use half-hour intervals and local hours in [0,24)")
        if max(self.heating_on_fraction) > 1:
            raise ValueError("Heating schedule must be a fraction")
        for field in ("baseload_weights", "lighting_weights", "hot_water_weights", "extra_weights"):
            if any(
                np.sum(np.asarray(getattr(self, field))[index == i]) <= 0 for i in range(len(self.months))
            ):
                raise ValueError(f"{field} must have positive mass in every month")
        return self

    def numeric_arrays(self):
        """JAX-compatible pytree; no strings or network access inside simulation."""
        names = (
            "month_index",
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
        arrays = {
            name: np.asarray(getattr(self, name), dtype=np.int32 if name == "month_index" else np.float32)
            for name in names
        }
        arrays.update(
            {
                f"battery_{key}": np.asarray(value, dtype=np.float32)
                for key, value in self.battery.model_dump().items()
            }
        )
        arrays["pv_inverter_kw"] = np.asarray(self.pv_inverter_kw, dtype=np.float32)
        arrays["pv_azimuth_deg"] = np.asarray(self.pv_azimuth_deg, dtype=np.float32)
        arrays["pv_tilt_deg"] = np.asarray(self.pv_tilt_deg, dtype=np.float32)
        arrays["export_limit_kw"] = np.asarray(
            self.export_limit_kw if self.export_limit_kw is not None else 1e9, dtype=np.float32
        )
        return arrays
