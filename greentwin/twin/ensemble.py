"""Prior-predictive Monte Carlo around the deterministic JAX twin; no calibration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import jax
import numpy as np

from contracts.models import Provenance, TwinParams
from contracts.uncertain_twin import EnsembleResult
from greentwin.twin.electricity_simulation import validate_interval_output
from greentwin.twin.generate import validate_physics_output
from greentwin.twin.missing_inputs import InputSamples, moments, prepare_electricity_sample
from greentwin.twin.physics import VERSION, run_physics
from greentwin.twin.physics_api import FuelConfig


@dataclass
class EnsembleRun:
    result: EnsembleResult
    samples: dict[str, np.ndarray]
    elapsed_seconds: float


@lru_cache(maxsize=12)
def _kernel(fuel):
    return jax.jit(jax.vmap(lambda params, weather: run_physics(params, weather, "uncalibrated", fuel)))


def simulate_ensemble(inputs: InputSamples, *, batch_size=8, include_intervals=False) -> EnsembleRun:
    """Run only previously generated inputs, validating each physical realisation."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    started = time.perf_counter()
    request, prior, p = inputs.request, inputs.prior, inputs.parameters
    n = inputs.summary.n_samples
    if any(a.shape != (n,) or not np.isfinite(a).all() for a in p.values()):
        raise ValueError("Prepared parameter samples must be finite and aligned")
    fuel = FuelConfig(prior.equipment["heating_fuel"], prior.equipment["hot_water_fuel"])
    kernel = _kernel(fuel)
    collected, errors = {}, {}
    metric_samples = {
        k: []
        for k in (
            "peak_import_kw",
            "peak_import_share_pct",
            "self_consumption_pct",
            "self_sufficiency_pct",
            "battery_initial_kwh",
            "battery_final_kwh",
        )
    }
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        prepared = [
            inputs.electricity[i] if inputs.electricity else prepare_electricity_sample(inputs, i)
            for i in range(start, end)
        ]
        w = {k: a[start:end].astype(np.float32) for k, a in inputs.weather.items()}
        arrays = [item.numeric_arrays() for item in prepared]
        w["electricity_inputs"] = {key: np.stack([a[key] for a in arrays]) for key in arrays[0]}
        raw = jax.device_get(kernel({k: a[start:end].astype(np.float32) for k, a in p.items()}, w))
        for offset, item in enumerate(prepared):
            i = start + offset
            single = jax.tree.map(lambda a, offset=offset: a[offset], raw)
            params = TwinParams(
                **{k: a[i] for k, a in p.items()},
                property_id=request.property_id,
                heating_fuel=fuel.heating_fuel,
            )
            extra = {
                k: float(p[k][i])
                for k in (
                    "hot_water_efficiency",
                    "hot_water_standing_loss_kwh_per_day",
                    "shower_electric_kwh_per_day",
                )
            }
            weather = {k: a[i] for k, a in inputs.weather.items()}
            provenance = Provenance(
                engine="physics_l1",
                model_version=VERSION,
                data_version=request.data_version,
                seed=request.seed,
                n_samples=n,
            )
            checked = validate_physics_output(
                single, len(request.months), params, fuel, weather, provenance, extra
            )
            checked_errors = checked.conservation_max_error_kwh | validate_interval_output(
                single, params, item
            )
            for key, value in checked_errors.items():
                errors[key] = max(errors.get(key, 0), value)
            import_a = np.asarray(single["interval_import_kwh"], dtype=float)
            demand = float(np.asarray(single["interval_demand_kwh"], dtype=float).sum())
            pv = float(np.asarray(single["interval_pv_generation_kwh"], dtype=float).sum())
            captured = float(
                np.asarray(single["interval_direct_pv_kwh"], dtype=float).sum()
                + np.asarray(single["interval_charge_kwh"], dtype=float).sum()
            )
            hours = np.asarray(item.local_hour)
            peak_energy = float(import_a[(hours >= 16) & (hours < 19)].sum())
            values = {
                "peak_import_kw": float(single["peak_import_kw"]),
                "peak_import_share_pct": 100 * peak_energy / import_a.sum() if import_a.sum() > 0 else np.nan,
                "self_consumption_pct": 100 * captured / pv if pv > 0 else np.nan,
                "self_sufficiency_pct": np.clip(100 * (1 - import_a.sum() / demand), 0, 100)
                if demand > 0
                else np.nan,
                "battery_initial_kwh": float(single["battery_initial_kwh"]),
                "battery_final_kwh": float(single["battery_final_kwh"]),
            }
            for key, value in values.items():
                metric_samples[key].append(value)
        for key, value in raw.items():
            if isinstance(value, dict):
                for sub, a in value.items():
                    collected.setdefault(f"{key}:{sub}", []).append(np.asarray(a))
            elif include_intervals or not key.startswith("interval_"):
                collected.setdefault(key, []).append(np.asarray(value))
    samples = {k: np.concatenate(parts, axis=0) for k, parts in collected.items()}
    totals = {
        k: moments(a, "W/K" if "w_per_k" in k else "kWh")
        for k, a in samples.items()
        if a.ndim == 1 and k.endswith("kwh")
    }
    monthly = {
        k: [moments(a[:, j], "kWh") for j in range(a.shape[1])]
        for k, a in samples.items()
        if k.startswith("monthly_")
    }
    elements = {
        k.split(":")[1]: moments(a, "W/K")
        for k, a in samples.items()
        if k.startswith("element_heat_loss_w_per_k:")
    }
    h = samples["heat_loss_coefficient_w_per_k"]
    shares = {
        k: moments(
            np.divide(samples[f"element_heat_loss_w_per_k:{k}"] * 100, h, out=np.zeros(n), where=h > 0), "%"
        )
        for k in elements
    }
    metrics, unavailable = {}, {}
    for key, a in metric_samples.items():
        a = np.asarray(a)
        samples[f"metric:{key}"] = a
        if np.isfinite(a).all():
            metrics[key] = moments(a, "kW" if key.endswith("kw") else "kWh" if key.endswith("kwh") else "%")
        else:
            metrics[key] = None
            unavailable[key] = (
                "Ratio undefined for one or more draws (zero PV/import/demand); do not drop these draws"
            )
    metrics["heat_loss_coefficient_w_per_k"] = moments(h, "W/K")
    metrics["thermal_bridging_w_per_k"] = moments(samples["thermal_bridging_w_per_k"], "W/K")
    # Complete calendar years only; no implicit x12 annualisation.
    annual = {}
    for year in sorted({month[:4] for month in request.months}):
        indices = [j for j, t in enumerate(request.months) if t[:4] == year]
        if [request.months[j][5:] for j in indices] == [f"{m:02d}" for m in range(1, 13)]:
            annual[year] = {
                k: moments(a[:, indices].sum(axis=1), "kWh/yr")
                for k, a in samples.items()
                if k.startswith("monthly_")
            }
    result = EnsembleResult(
        property_id=request.property_id,
        n_samples=n,
        seed=request.seed,
        request_hash=prior.request_hash,
        months=request.months,
        totals=totals,
        monthly=monthly,
        elements=elements,
        element_shares_pct=shares,
        metrics=metrics,
        annual_totals=annual,
        unavailable=unavailable
        | {
            "carbon": "Part 2 applies sourced emissions factors",
            "bills": "Part 2 applies date-valid sourced tariffs; no fabricated prices",
            "score": "Part 3 consumes energy and property state",
            "value": "Part 4 owns indicative value estimates",
        },
        input_summary=inputs.summary,
        max_conservation_error_kwh=errors,
        calibration_status=inputs.summary.calibration_status,
        uncertainty_basis="posterior_predictive"
        if inputs.summary.calibration_status == "performed" else "prior_predictive_not_calibrated",
        calibration_id=inputs.summary.calibration_id,
        calibration_diagnostics=inputs.summary.calibration_diagnostics,
        warnings=prior.warnings
        + [
            "Generated half-hour profiles are conditional scenarios, not observed London/PVGIS data",
            "Element losses are estimated; meter calibration does not independently measure each element"
            if inputs.summary.calibration_status == "performed"
            else "Element losses are estimated; no total or element has been calibrated",
            "Monthly steady-state heat; half-hour heating profiles are allocated, not dynamic temperatures",
        ],
        provenance=Provenance(
            engine="physics_l1",
            model_version=VERSION,
            data_version=request.data_version,
            seed=request.seed,
            n_samples=n,
        ),
    )
    return EnsembleRun(result, samples, time.perf_counter() - started)


def paired_differences(baseline: EnsembleRun, after: EnsembleRun):
    """Small downstream utility; correlated samples, signed baseline minus after."""
    a, b = baseline.result, after.result
    if (a.property_id, a.seed, a.n_samples, a.months) != (b.property_id, b.seed, b.n_samples, b.months):
        raise ValueError("Paired runs must share household, seed, sample count and dates")
    if a.calibration_id != b.calibration_id:
        raise ValueError("Paired calibrated runs must share the same posterior sample population")
    keys = (
        "electricity_import_kwh",
        "gas_kwh",
        "oil_kwh",
        "lpg_kwh",
        "solid_kwh",
        "district_kwh",
        "purchased_electricity_gas_kwh",
        "all_fuel_purchased_kwh",
    )
    return {k: moments(baseline.samples[k] - after.samples[k], "kWh") for k in keys}
