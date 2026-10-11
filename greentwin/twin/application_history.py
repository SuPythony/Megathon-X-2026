"""Application answers -> population priors. No simulation, generated labels or calibration.

Only submitted facts are retained. The application adapter must identify baseline
submissions; previews/retrofits must never become observed household history.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import AwareDatetime, Field

from contracts.uncertain_twin import HouseholdRequest, PriorMarginal, StrictModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = ROOT / "data/processed/application_history.sqlite"
DESCRIPTORS = {
    "postcode_district",
    "region",
    "property_type",
    "built_form",
    "construction_age_band",
    "floor_area_m2",
    "heating_system",
    "hot_water_source",
}
NUMERIC = {
    "wall_u",
    "roof_u",
    "floor_u",
    "window_u",
    "door_u",
    "air_changes_per_hour",
    "thermal_bridging_y",
    "heating_efficiency",
    "hot_water_efficiency",
    "solar_gain_factor",
    "retained_electric_heat_fraction",
    "heating_auxiliary_power_kw",
    "pv_azimuth_deg",
    "pv_tilt_deg",
    "battery_usable_fraction",
    "battery_charge_efficiency",
    "battery_discharge_efficiency",
    "thermostat_c",
    "setpoint_c",
    "heating_hours_per_day",
    "baseload_kwh_per_day",
    "hot_water_kwh_per_day",
    "water_temperature_rise_c",
    "pv_kwp",
    "battery_kwh",
}
CATEGORICAL = {
    "occupants",
    "daytime_occupancy",
    "cooking_fuel",
    "flat_position",
    "has_solar_pv",
    "has_battery",
    "cylinder_present",
}
ALIASES = {
    "thermostat_c": "setpoint_c",
    "daytime_occupancy": "presence",
    "cylinder_present": "cylinder_presence",
}


def identity(raw):
    return hashlib.sha256(f"application-history/1:{raw}".encode()).hexdigest()


class ApplicationObservation(StrictModel):
    version: str = "application-observation/0.1"
    household_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    customer_key: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    submitted_at: AwareDatetime
    split: Literal["train", "val", "test"] = "train"
    source: str = "application_answer"
    facts: dict


def observation_from_request(request: HouseholdRequest, *, submitted_at=None):
    """Extract recorded facts, never generated prior means or simulation outputs."""
    facts = {
        k: v
        for k, v in request.user_inputs.items()
        if k in DESCRIPTORS | NUMERIC | CATEGORICAL and v is not None
    }
    prop = request.property
    if prop and prop.source in ("showcase", "synthetic"):
        raise ValueError("Showcase/synthetic properties are not application observations")
    customer, split = None, "train"
    if prop:
        customer = identity(prop.customer_id) if prop.customer_id else None
        split = prop.split if prop.split else "train"
        fields = {
            "postcode_district": prop.location.postcode_district,
            "region": prop.location.region,
            "property_type": prop.property_type.value,
            "built_form": prop.built_form.value,
            "construction_age_band": prop.construction_age_band,
            "floor_area_m2": prop.floor_area_m2,
            "heating_system": prop.heating.main_system.value,
            "hot_water_source": prop.heating.hot_water,
            "wall_u": prop.walls.u_value,
            "roof_u": prop.roof.u_value,
            "floor_u": prop.floor.u_value,
            "heating_efficiency": prop.heating.seasonal_efficiency,
        }
        for k, v in fields.items():
            if v is not None and k not in prop.data_quality.imputed_fields:
                facts.setdefault(k, v)
        for k in ("pv_kwp", "battery_kwh", "pv_azimuth_deg", "pv_tilt_deg"):
            if k in prop.renewables.model_fields_set and getattr(prop.renewables, k) is not None:
                facts.setdefault(k, getattr(prop.renewables, k))
    package = request.physics_package or {}
    for k, v in package.get("home", {}).items():
        if k in DESCRIPTORS and v is not None:
            facts.setdefault(k, v)
    # Only explicit observed numeric sources from a handoff are eligible.
    for k, v in package.get("params", {}).items():
        source = package.get("sources", {}).get(k, "").lower()
        if k in NUMERIC and re.match(r"^(user|measured|surveyed|epc_recorded)\b", source):
            facts.setdefault(k, v)
    occupancy = package.get("occupancy", {})
    if re.match(r"^user\b", occupancy.get("source", "").lower()):
        for k, pkey in (("occupants", "occupants"), ("daytime_occupancy", "presence")):
            if occupancy.get(pkey) is not None:
                facts.setdefault(k, occupancy[pkey])
    if "heating_hours_per_day" not in facts and request.user_inputs.get("heating_periods"):
        from contracts.electricity import ElectricityIntake

        periods = ElectricityIntake(heating_periods=request.user_inputs["heating_periods"]).heating_periods
        facts["heating_hours_per_day"] = sum((end - start) % 24 or 24 for start, end in periods)
    return ApplicationObservation(
        household_key=identity(request.property_id),
        customer_key=customer,
        submitted_at=submitted_at or datetime.now(UTC),
        split=split,
        facts=facts,
    )


class ApplicationHistory:
    """Versioned baseline revisions; reads never create a database, writes stay here."""

    def __init__(self, path=DEFAULT_HISTORY):
        self.path = Path(path).resolve()

    def record(self, request, *, submitted_at=None, kind="baseline"):
        if kind != "baseline":
            return False
        if not self.path.is_relative_to(ROOT):
            raise ValueError("Application history writes must stay inside the physics worktree")
        row = observation_from_request(request, submitted_at=submitted_at)
        # Validate primitive, nonidentifying fields before persistence.
        if "postcode_district" in row.facts:
            row.facts["postcode_district"] = row.facts["postcode_district"].strip().upper()
            if not re.fullmatch(r"[A-Z]{1,2}\d[A-Z\d]?", row.facts["postcode_district"]):
                raise ValueError("History accepts postcode districts, never full postcodes")
        for k, v in row.facts.items():
            if (k in NUMERIC or k == "floor_area_m2") and (
                isinstance(v, bool) or not np.isfinite(float(v)) or float(v) < 0
            ):
                raise ValueError(f"Invalid recorded numeric {k}")
            if k == "occupants" and (isinstance(v, bool) or float(v) != int(v) or not 1 <= int(v) <= 12):
                raise ValueError("Invalid recorded occupant count")
            if k in {"has_solar_pv", "has_battery", "cylinder_present"} and not isinstance(v, bool):
                raise ValueError(f"Invalid recorded boolean {k}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS observations (household TEXT, submitted TEXT, payload TEXT, PRIMARY KEY(household,submitted))"
            )
            db.execute(
                "INSERT OR REPLACE INTO observations VALUES (?,?,?)",
                (row.household_key, row.submitted_at.astimezone(UTC).isoformat(), row.model_dump_json()),
            )
        return True

    def snapshot(self, as_of):
        if not self.path.exists():
            return []
        # URI read-only mode prevents SQLite from creating/changing another app's DB.
        with sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute("SELECT payload FROM observations ORDER BY household,submitted").fetchall()
        latest = {}
        for (payload,) in rows:
            row = ApplicationObservation.model_validate_json(payload)
            if row.submitted_at <= as_of:
                latest[row.household_key] = row
        return list(latest.values())


def _cohort(records, facts, variable):
    # Equipment variables cannot borrow efficiencies from another technology.
    equipment = (
        "efficiency" in variable
        or variable.startswith("battery_")
        or variable == "heating_auxiliary_power_kw"
    )
    from greentwin.twin.priors import SYSTEMS

    def system(value):
        return SYSTEMS.get(value, (value, None))[0]

    if equipment and variable.startswith("heating"):
        records = [
            r for r in records if system(r.facts.get("heating_system")) == system(facts.get("heating_system"))
        ]
    if variable == "hot_water_efficiency":
        records = [
            r
            for r in records
            if system(r.facts.get("heating_system")) == system(facts.get("heating_system"))
            and r.facts.get("hot_water_source", "from_main") == facts.get("hot_water_source", "from_main")
        ]
    typed = [r for r in records if r.facts.get("property_type") == facts.get("property_type")]
    if typed:
        records = typed
    area = facts.get("floor_area_m2")
    if area and variable in ("occupants", "setpoint_c", "baseload_kwh_per_day", "hot_water_kwh_per_day"):
        sized = [
            r
            for r in records
            if r.facts.get("floor_area_m2")
            and 0.7 * float(area) <= float(r.facts["floor_area_m2"]) <= 1.3 * float(area)
        ]
        if sized:
            records = sized
    if variable in ("baseload_kwh_per_day", "hot_water_kwh_per_day") and facts.get("occupants"):
        occupied = [r for r in records if r.facts.get("occupants") == facts["occupants"]]
        if occupied:
            records = occupied
    local = [r for r in records if r.facts.get("postcode_district") == facts.get("postcode_district")]
    regional = [
        r
        for r in records
        if r.facts.get("region") == facts.get("region") and facts.get("region") != "unknown"
    ]
    records, level = (
        (local, "district") if local else (regional, "region") if regional else (records, "application-wide")
    )
    age = facts.get("construction_age_band")
    if variable.endswith("_u") and age and age != "unknown":
        matched = [r for r in records if r.facts.get("construction_age_band") == age]
        if matched:
            records, level = matched, level + " + age"
    return records, level


def inform_priors(request, facts, marginals, history, *, as_of=None):
    """Shrink empirical answers toward existing priors, with a finite-data floor."""
    as_of = as_of or datetime(request.as_of_year, 12, 31, 23, 59, 59, tzinfo=UTC)
    all_rows = history.snapshot(as_of)
    current = identity(request.property_id)
    customer = (
        identity(request.property.customer_id) if request.property and request.property.customer_id else None
    )
    rows = [
        r
        for r in all_rows
        if r.split == "train" and r.household_key != current and not (customer and r.customer_key == customer)
    ]
    snapshot_hash = hashlib.sha256("\n".join(r.model_dump_json() for r in rows).encode()).hexdigest()
    audit = {
        "path": str(history.path),
        "snapshot_sha256": snapshot_hash,
        "eligible_households": len(rows),
        "excluded_households": len(all_rows) - len(rows),
        "as_of": as_of.isoformat(),
        "variables": [],
        "basis": "recorded application answers; initial population distributions, not meter calibration",
    }
    for variable, old in list(marginals.items()):
        if old.family == "exact" or old.source == "user/recorded exact fact":
            continue
        if old.family == "categorical" and len(old.categories) == 1:
            continue
        candidates = []
        for row in rows:
            mapped = {ALIASES.get(k, k): v for k, v in row.facts.items()}
            if variable.startswith("installed_"):
                capacity = variable.removeprefix("installed_")
                if capacity in mapped and float(mapped[capacity]) > 0:
                    mapped[variable] = mapped[capacity]
            if variable in ("has_solar_pv", "has_battery") and variable not in mapped:
                capacity = "pv_kwp" if variable == "has_solar_pv" else "battery_kwh"
                if capacity in mapped:
                    mapped[variable] = float(mapped[capacity]) > 0
            if variable in mapped:
                candidates.append(row.model_copy(update={"facts": row.facts | {variable: mapped[variable]}}))
        cohort, level = _cohort(candidates, facts, variable)
        values, weights = [], []
        for row in cohort:
            v = row.facts[variable]
            if old.family == "categorical":
                v = "yes" if v is True else "no" if v is False else str(v)
                if variable == "occupants" and str(v).replace(".0", "").isdigit():
                    v = str(int(float(v)))
                if variable != "occupants" and v not in old.categories:
                    continue
            else:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if (
                    not np.isfinite(v)
                    or (old.lower is not None and v < old.lower)
                    or (old.upper is not None and v > old.upper)
                ):
                    continue
            values.append(v)
            weights.append(float(np.exp(-max((as_of - row.submitted_at).total_seconds() / 31557600, 0) / 10)))
        if not values:
            continue
        w = np.asarray(weights)
        effective = float(w.sum())
        pooling = effective / (effective + 20)
        source = f"application history: {level}; {len(values)} households; {snapshot_hash[:12]}"
        if old.family == "categorical":
            labels = sorted(set(old.categories) | set(values))
            probs = {
                label: float(
                    (
                        20 * old.categories.get(label, 0)
                        + sum(weight for value, weight in zip(values, w) if value == label)
                    )
                    / (20 + effective)
                )
                for label in labels
            }
            marginals[variable] = old.model_copy(update={"categories": probs, "source": source})
        else:
            mean = float(np.average(values, weights=w))
            if old.family == "normal":
                prior_var = old.scale**2
            elif old.family == "lognormal":
                prior_var = old.centre**2 * np.expm1(old.scale**2)
            elif old.family == "beta":
                f = (old.centre - old.lower) / (old.upper - old.lower)
                prior_var = (old.upper - old.lower) ** 2 * f * (1 - f) / (old.scale + 1)
            else:
                prior_var = (old.upper - old.lower) ** 2 / 12
            empirical_var = float(np.average((np.asarray(values) - mean) ** 2, weights=w))
            variance = (
                (1 - pooling) * prior_var
                + pooling * max(empirical_var, prior_var * 0.25)
                + pooling * (1 - pooling) * (old.centre - mean) ** 2
            )
            marginals[variable] = PriorMarginal(
                family="normal",
                centre=(1 - pooling) * old.centre + pooling * mean,
                scale=float(np.sqrt(variance)),
                lower=old.lower,
                upper=old.upper,
                role=old.role,
                source=source,
                unit=old.unit,
                rationale="Recency-weighted recorded answers with fallback shrinkage and heterogeneity floor",
            )
        audit["variables"].append(
            {
                "variable": variable,
                "n_households": len(values),
                "cohort": level,
                "effective_n": effective,
                "pooling_weight": pooling,
            }
        )
    return audit


def prepare_application_inputs(request, n_samples=128, *, history=None, kind="baseline", submitted_at=None):
    """Application-facing seam: learn from old answers, generate, then record new facts."""
    from greentwin.twin.missing_inputs import generate_missing_inputs
    from greentwin.twin.priors import build_priors

    history = history or ApplicationHistory()
    prior = build_priors(request, history=history, history_as_of=submitted_at)
    samples = generate_missing_inputs(request, n_samples, prior=prior)
    # Current home was excluded from population learning; its answers were exact.
    if kind == "baseline":
        history.record(request, submitted_at=submitted_at)
    return samples
