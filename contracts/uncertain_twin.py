"""Part-1 prior-predictive extensions; shared schema 0.4 remains unchanged."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.electricity import ElectricitySeries
from contracts.models import Distribution, Property, Provenance, WeatherMonth


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PriorMarginal(StrictModel):
    family: Literal["exact", "normal", "lognormal", "beta", "uniform", "categorical"]
    centre: float = 0.0
    scale: float = Field(0, ge=0)
    lower: float | None = None
    upper: float | None = None
    categories: dict[str, float] = Field(default_factory=dict)
    unit: str = "1"
    source: str
    rationale: str
    role: Literal["missing_input", "constant", "context"] = "missing_input"

    @model_validator(mode="after")
    def domains(self):
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError("Prior bounds must be ordered")
        if self.family == "categorical":
            if not self.categories or any(p < 0 for p in self.categories.values()):
                raise ValueError("Category probabilities must be nonnegative")
            if abs(sum(self.categories.values()) - 1) > 1e-8:
                raise ValueError("Category probabilities must sum to one")
        elif self.family != "exact" and self.scale <= 0 and self.family != "uniform":
            raise ValueError("Nondegenerate priors require positive scale")
        if self.family in ("beta", "uniform") and (self.lower is None or self.upper is None):
            raise ValueError("Beta/uniform require finite bounds")
        if self.family == "beta" and not self.lower < self.centre < self.upper:
            raise ValueError("Beta centre must lie inside bounds")
        if self.family == "lognormal" and self.centre <= 0:
            raise ValueError("Lognormal mean must be positive")
        if self.family == "exact" and (
            (self.lower is not None and self.centre < self.lower)
            or (self.upper is not None and self.centre > self.upper)
        ):
            raise ValueError("Exact value lies outside physical domain")
        return self


class ContextRecord(StrictModel):
    """Public/population parameter evidence, never this household's meter likelihood."""

    variable: str
    values: list[float] = Field(min_length=2)
    source: str
    status: Literal["public", "development"] = "public"
    area_code: str | None = None
    region: str | None = None
    property_type: str | None = None
    heating_system: str | None = None
    year: int = Field(ge=1900, le=2200)
    month_of_year: int | None = Field(None, ge=1, le=12)
    split: Literal["train", "validation", "test"] = "train"
    property_ids: list[str] = Field(default_factory=list)


class HouseholdRequest(StrictModel):
    property_id: str = Field(min_length=1)
    property: Property | None = None
    physics_package: dict[str, Any] | None = None
    user_inputs: dict[str, Any] = Field(default_factory=dict)
    months: list[str] = Field(default_factory=lambda: [f"2025-{m:02d}" for m in range(1, 13)])
    weather: list[WeatherMonth] = Field(default_factory=list)
    electricity_series: ElectricitySeries | None = None
    context: list[ContextRecord] = Field(default_factory=list)
    as_of_year: int = Field(2026, ge=1900, le=2200)
    seed: int = Field(42, ge=0)
    data_version: str = "prior-predictive/0.1"
    allow_development_weather: bool = False

    @model_validator(mode="after")
    def identity(self):
        if self.property and self.property.property_id != self.property_id:
            raise ValueError("Property identities differ")
        if self.physics_package and self.physics_package.get("property_id") != self.property_id:
            raise ValueError("Handoff identities differ")
        if not self.months or self.months != sorted(set(self.months)):
            raise ValueError("Months must be nonempty, unique and chronological")
        return self


class PriorBundle(StrictModel):
    version: str = "household-priors/0.1.0"
    property_id: str
    facts: dict[str, Any]
    marginals: dict[str, PriorMarginal]
    equipment: dict[str, Any]
    base_parameters: dict[str, Any]
    weather: list[WeatherMonth]
    profile_centres: dict[str, list[float]] = Field(default_factory=dict)
    profile_exact: list[str] = Field(default_factory=list)
    context_audit: list[dict[str, Any]] = Field(default_factory=list)
    dependencies: dict[str, str]
    warnings: list[str]
    request_hash: str
    application_history: dict[str, Any] = Field(default_factory=dict)


class MomentDistribution(Distribution):
    """Shared distribution summary with explicit sample variance requested by the caller."""

    variance: float = Field(ge=0)
    monte_carlo_standard_error: float = Field(ge=0)


class GeneratedInputs(StrictModel):
    version: str = "missing-inputs/0.1.0"
    property_id: str
    n_samples: int = Field(ge=2)
    seed: int
    request_hash: str
    sampling: Literal["independent_monte_carlo", "posterior_joint_samples"] = "independent_monte_carlo"
    facts: dict[str, Any]
    parameters: dict[str, MomentDistribution]
    constants: dict[str, MomentDistribution]
    monthly_inputs: dict[str, list[MomentDistribution]]
    categories: dict[str, dict[str, float]]
    prior: PriorBundle
    sample_ref: str | None = None
    calibration_status: Literal["not_performed", "performed"] = "not_performed"
    calibration_id: str | None = None
    calibration_diagnostics: dict[str, Any] = Field(default_factory=dict)


class EnsembleResult(StrictModel):
    version: str = "prior-predictive-twin/0.1.0"
    property_id: str
    n_samples: int
    seed: int
    request_hash: str
    months: list[str]
    time_basis: Literal["supplied_period"] = "supplied_period"
    annual_totals: dict[str, dict[str, MomentDistribution]] = Field(default_factory=dict)
    totals: dict[str, MomentDistribution]
    monthly: dict[str, list[MomentDistribution]]
    elements: dict[str, MomentDistribution]
    element_shares_pct: dict[str, MomentDistribution]
    metrics: dict[str, MomentDistribution | None]
    unavailable: dict[str, str]
    input_summary: GeneratedInputs
    max_conservation_error_kwh: dict[str, float]
    sample_ref: str | None = None
    uncertainty_basis: Literal["prior_predictive_not_calibrated", "posterior_predictive"] = "prior_predictive_not_calibrated"
    calibration_status: Literal["not_performed", "performed"] = "not_performed"
    calibration_id: str | None = None
    calibration_diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str]
    provenance: Provenance
