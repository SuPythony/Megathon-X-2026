"""Home-level calibration extensions, preserving shared schema 0.4."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, model_validator

from contracts.models import Provenance
from contracts.uncertain_twin import MomentDistribution, PriorMarginal, StrictModel


class MeterObservation(StrictModel):
    month: str
    electricity_import_kwh: float | None = Field(None, ge=0)
    gas_kwh: float | None = Field(None, ge=0)
    oil_kwh: float | None = Field(None, ge=0)
    lpg_kwh: float | None = Field(None, ge=0)
    solid_kwh: float | None = Field(None, ge=0)
    district_kwh: float | None = Field(None, ge=0)
    export_kwh: float | None = Field(None, ge=0)
    pv_generation_kwh: float | None = Field(None, ge=0)
    estimated: bool = False
    use_for_calibration: bool = True
    source: str = "customer monthly meter energy"

    @model_validator(mode="after")
    def calendar(self):
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", self.month):
            raise ValueError("Meter month must be YYYY-MM")
        return self


class CalibrationConfig(StrictModel):
    method: Literal["nuts", "svi"] = "nuts"
    parameters: list[str] | None = None
    nuisance_samples: int = Field(16, ge=2, le=256)
    warmup: int = Field(300, ge=20)
    samples: int = Field(300, ge=20)
    chains: int = Field(2, ge=1, le=4)
    target_accept: float = Field(0.9, gt=0.5, lt=1)
    max_tree_depth: int = Field(8, ge=2, le=12)
    svi_steps: int = Field(1500, ge=100)
    learning_rate: float = Field(0.01, gt=0, le=0.1)
    relative_noise_actual: float = Field(0.05, gt=0, le=1)
    relative_noise_estimated: float = Field(0.15, gt=0, le=1)
    absolute_noise_kwh: float = Field(5, gt=0)
    student_df: float = Field(4, gt=2)
    seed: int = Field(42, ge=0)
    max_r_hat: float = Field(1.01, gt=1)
    min_effective_samples: float = Field(50, gt=0)

    @model_validator(mode="after")
    def unique_parameters(self):
        if self.parameters is not None and (not self.parameters or len(set(self.parameters)) != len(self.parameters)):
            raise ValueError("Calibration parameters must be a nonempty unique list")
        if self.relative_noise_estimated < self.relative_noise_actual:
            raise ValueError("Estimated readings must have at least the actual-reading noise")
        return self


class HomeCalibrationResult(StrictModel):
    version: str = "home-calibration/0.1.0"
    property_id: str
    calibration_id: str
    request_hash: str
    method: Literal["nuts", "svi"]
    status: Literal["converged", "approximate", "needs_review"]
    config: CalibrationConfig
    fitted_parameters: list[str]
    fixed_parameters: list[str]
    prior_specs: dict[str, PriorMarginal]
    prior_parameters: dict[str, MomentDistribution]
    posterior_parameters: dict[str, MomentDistribution]
    observations: list[MeterObservation]
    observation_count: int
    training_months: list[str]
    held_out_months: list[str]
    diagnostics: dict[str, Any]
    nuisance_basis: str = "finite joint prior bank, marginalized over the complete observed series"
    elapsed_seconds: float = Field(ge=0)
    warnings: list[str]
    provenance: Provenance
