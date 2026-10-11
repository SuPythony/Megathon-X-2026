"""Contract boundary and independent conservation checks for the coupled twin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from contracts.electricity import PreparedElectricityInputs
from contracts.models import Provenance, TwinParams, WeatherMonth
from greentwin.twin.electricity_inputs import timeline
from greentwin.twin.generate import validate_physics_output
from greentwin.twin.physics import DEFAULT_ASSUMPTIONS, VERSION, ThermalAssumptions, run_physics
from greentwin.twin.physics_api import PHYSICS_PARAM_KEYS, FuelConfig
from greentwin.twin.weather import weather_arrays


class ElectricityResult(BaseModel):
    property_id: str
    parameters: TwinParams
    extra_parameters: dict[str, float]
    equipment: dict[str, str | float | None]
    months: list[str]
    totals: dict[str, float]
    monthly: dict[str, list[float]]
    interval_count: int
    timestamps_utc: list[str] | None = None
    intervals: dict[str, list[float]] | None = None
    element_heat_loss_w_per_k: dict[str, float]
    element_heat_loss_estimated: Literal[True] = True
    element_heat_loss_basis: str = "Estimated from supplied geometry and fabric assumptions; uncalibrated"
    thermal_assumptions: dict[str, float]
    electricity_input_summary: dict
    peak_import_kw: float
    peak_import_share_pct: float | None
    self_consumption_pct: float | None
    self_sufficiency_pct: float | None
    battery_initial_kwh: float
    battery_final_kwh: float
    import_bill_gbp: float | None
    export_revenue_gbp: float | None
    net_bill_gbp: float | None
    time_of_use_bill_gbp: None = None
    billing_notes: list[str]
    input_sources: dict[str, str]
    warnings: list[str]
    conservation_max_error_kwh: dict[str, float]
    calibration_status: Literal["uncalibrated"] = "uncalibrated"
    provenance: Provenance


def validate_interval_output(raw, params, inputs):
    n = len(inputs.timestamps_utc)
    a = {key: np.asarray(value, dtype=float) for key, value in raw.items() if key.startswith("interval_")}
    for key, value in a.items():
        if value.shape != (n,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} interval output")
        if key != "interval_battery_delta_kwh" and (value < -2e-5).any():
            raise ValueError(f"Negative {key}")
    errors = {}

    def balance(name, left, right):
        error = np.abs(left - right)
        errors[name] = float(np.max(error))
        if not np.allclose(left, right, rtol=2e-5, atol=2e-4):
            raise ValueError(f"Interval {name} balance violated")

    balance(
        "interval_electricity",
        a["interval_import_kwh"] + a["interval_pv_generation_kwh"],
        a["interval_demand_kwh"]
        + a["interval_export_kwh"]
        + a["interval_curtailment_kwh"]
        + a["interval_battery_loss_kwh"]
        + a["interval_battery_delta_kwh"],
    )
    balance(
        "interval_end_uses",
        a["interval_demand_kwh"],
        sum(a[f"interval_{key}_kwh"] for key in ("appliances", "lighting", "heating", "hot_water", "extra")),
    )
    b = inputs.battery
    balance(
        "interval_battery_losses",
        a["interval_battery_loss_kwh"],
        a["interval_charge_kwh"] * (1 - b.charge_efficiency)
        + a["interval_discharge_kwh"] * (1 / b.discharge_efficiency - 1),
    )
    balance(
        "interval_storage_change",
        a["interval_battery_delta_kwh"],
        a["interval_charge_kwh"] * b.charge_efficiency - a["interval_discharge_kwh"] / b.discharge_efficiency,
    )
    capacity = params.battery_kwh * b.usable_fraction
    initial = capacity * b.initial_fraction
    balance(
        "interval_storage_state",
        a["interval_stored_kwh"],
        initial + np.cumsum(a["interval_battery_delta_kwh"]),
    )
    if (a["interval_stored_kwh"] < -2e-4).any() or (a["interval_stored_kwh"] > capacity + 2e-4).any():
        raise ValueError("Battery state outside usable capacity")
    dt = np.asarray(inputs.duration_hours)
    for key, limit in (
        ("charge", b.charge_kw),
        ("discharge", b.discharge_kw),
        ("pv_generation", inputs.pv_inverter_kw),
    ):
        if (a[f"interval_{key}_kwh"] > limit * dt + 2e-4).any():
            raise ValueError(f"{key} power limit violated")
    if (
        inputs.export_limit_kw is not None
        and (a["interval_export_kwh"] > inputs.export_limit_kw * dt + 2e-4).any()
    ):
        raise ValueError("Export power limit violated")
    if ((a["interval_charge_kwh"] > 2e-5) & (a["interval_discharge_kwh"] > 2e-5)).any():
        raise ValueError("Simultaneous battery charge/discharge")
    balance(
        "interval_pv_availability",
        a["interval_pv_generation_kwh"] + a["interval_pv_clipping_kwh"],
        params.pv_kwp * np.asarray(inputs.pv_kwh_per_kwp),
    )
    index = np.asarray(inputs.month_index)
    for interval, monthly in (
        ("import", "electricity"),
        ("export", "export"),
        ("pv_generation", "pv_generation"),
        ("demand", "electricity_demand"),
        ("battery_delta", "battery_delta"),
        ("battery_loss", "battery_loss"),
        ("curtailment", "curtailment"),
    ):
        sums = np.bincount(index, weights=a[f"interval_{interval}_kwh"], minlength=len(inputs.months))
        balance(f"monthly_{monthly}_aggregation", raw[f"monthly_{monthly}_kwh"], sums)
    return errors


def _bills(imported, exported, weather):
    config = json.loads((Path(__file__).resolve().parents[2] / "contracts/tariffs.json").read_text())
    first, last = weather[0].month, weather[-1].month
    notes = ["Time-of-use and export tariffs are unsourced; no invented prices"]
    import_bill = None
    if first >= config["period"]["start"][:7] and last <= config["period"]["end"][:7]:
        arrays = weather_arrays(weather)
        rates = config["electricity"]
        import_bill = (
            imported * rates["unit_rate_p_per_kwh"]
            + sum(arrays["days_in_month"]) * rates["standing_charge_p_per_day"]
        ) / 100
        notes.append(f"Standard import charges: {config['source']['url']} ({config['version']})")
    else:
        notes.append("Simulation dates fall outside the sourced standard tariff's validity period")
    export_revenue = 0.0 if exported == 0 else None
    net = import_bill if import_bill is not None and export_revenue is not None else None
    return import_bill, export_revenue, net, notes


def simulate_electricity(
    params: TwinParams,
    weather: list[WeatherMonth],
    fuel: FuelConfig,
    inputs: PreparedElectricityInputs,
    *,
    profiles: dict | None = None,
    extra_params: dict[str, float] | None = None,
    assumptions: ThermalAssumptions = DEFAULT_ASSUMPTIONS,
    data_version: str = "electricity-development-v1",
    include_intervals: bool = False,
) -> ElectricityResult:
    """Inputs must have been prepared and validated before calling the solver."""
    from dataclasses import asdict

    if params.property_id != inputs.property_id or params.heating_fuel != fuel.heating_fuel:
        raise ValueError("Home and fuel identities must match prepared inputs")
    months = [w.month for w in weather]
    if months != inputs.months or len({w.area_code for w in weather}) != 1:
        raise ValueError("Weather months/area must match prepared inputs")
    if any(w.mean_temp_c is None or w.ghi_kwh_m2 is None for w in weather):
        raise ValueError("Monthly temperature and irradiation must be explicit")
    dates, index = timeline(tuple(months))
    if inputs.timestamps_utc != [d.isoformat().replace("+00:00", "Z") for d in dates] or not np.array_equal(
        index, inputs.month_index
    ):
        raise ValueError("Prepared timeline does not cover these complete calendar months")
    extra_params = extra_params or {}
    allowed = {"hot_water_efficiency", "hot_water_standing_loss_kwh_per_day", "shower_electric_kwh_per_day"}
    if set(extra_params) - allowed:
        raise ValueError("Unsupported extra parameters; do not silently ignore equipment")
    numeric = {key: getattr(params, key) for key in PHYSICS_PARAM_KEYS} | extra_params
    arrays = weather_arrays(weather)
    for key, value in (profiles or {}).items():
        if key not in {
            "baseload_multiplier",
            "hot_water_multiplier",
            "gains_multiplier",
            "secondary_heating_kwh_monthly",
            "extra_electricity_kwh_monthly",
            "extra_gas_kwh_monthly",
        }:
            raise ValueError(f"Unsupported monthly profile {key}")
        arrays[key] = np.asarray(value, dtype=np.float32)
    arrays["electricity_inputs"] = inputs.numeric_arrays()
    raw = run_physics(numeric, arrays, "uncalibrated", fuel, assumptions)
    provenance = Provenance(
        engine="physics_l1", model_version=VERSION, data_version=data_version, seed=inputs.seed, n_samples=1
    )
    truth = validate_physics_output(raw, len(months), params, fuel, arrays, provenance, extra_params)
    errors = truth.conservation_max_error_kwh | validate_interval_output(raw, params, inputs)
    imported, exported = truth.totals["electricity_import_kwh"], truth.totals["export_kwh"]
    # Ratios use the same interval arrays in float64; mixed float32 period and
    # Python monthly reductions can otherwise give negative self-sufficiency.
    pv = float(np.asarray(raw["interval_pv_generation_kwh"], dtype=float).sum())
    demand = float(np.asarray(raw["interval_demand_kwh"], dtype=float).sum())
    interval_import = np.asarray(raw["interval_import_kwh"], dtype=float)
    imported_exact = float(interval_import.sum())
    direct = float(np.asarray(raw["interval_direct_pv_kwh"], dtype=float).sum())
    charge = float(np.asarray(raw["interval_charge_kwh"], dtype=float).sum())
    intervals = (
        {key: np.asarray(value).tolist() for key, value in raw.items() if key.startswith("interval_")}
        if include_intervals
        else None
    )
    local_hour = np.asarray(inputs.local_hour)
    peak = float(np.sum(interval_import[(local_hour >= 16) & (local_hour < 19)]))
    bill, export_revenue, net, notes = _bills(imported, exported, weather)
    warnings = list(inputs.warnings)
    if int(raw["heating_shape_fallback_months"]) > 0:
        warnings.append(
            "Monthly heating demand was allocated uniformly where interval degree-hours were zero"
        )
    warnings.append(
        "Heating timing is allocated from monthly steady-state demand, not a dynamic indoor-temperature simulation"
    )
    return ElectricityResult(
        property_id=params.property_id,
        parameters=params,
        extra_parameters=extra_params,
        equipment=asdict(fuel),
        months=months,
        totals=truth.totals,
        monthly=truth.monthly,
        interval_count=len(inputs.timestamps_utc),
        timestamps_utc=inputs.timestamps_utc if include_intervals else None,
        intervals=intervals,
        element_heat_loss_w_per_k=truth.element_heat_loss_w_per_k,
        thermal_assumptions=asdict(assumptions),
        electricity_input_summary={
            "version": inputs.version,
            "seed": inputs.seed,
            "development": inputs.development,
            "battery": inputs.battery.model_dump(),
            "pv_inverter_kw": inputs.pv_inverter_kw,
            "export_limit_kw": inputs.export_limit_kw,
            "dispatch_policy": "pv_self_consumption",
        },
        peak_import_kw=float(raw["peak_import_kw"]),
        peak_import_share_pct=float(np.clip(100 * peak / imported_exact, 0, 100))
        if imported_exact > 0
        else None,
        self_consumption_pct=float(np.clip(100 * (direct + charge) / pv, 0, 100)) if pv > 0 else None,
        self_sufficiency_pct=float(np.clip(100 * (demand - imported_exact) / demand, 0, 100))
        if demand > 0
        else None,
        battery_initial_kwh=float(raw["battery_initial_kwh"]),
        battery_final_kwh=float(raw["battery_final_kwh"]),
        import_bill_gbp=bill,
        export_revenue_gbp=export_revenue,
        net_bill_gbp=net,
        billing_notes=notes,
        input_sources=inputs.sources,
        warnings=warnings,
        conservation_max_error_kwh=errors,
        provenance=provenance,
    )
