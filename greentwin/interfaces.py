"""In-process interfaces between GreenTwin's parts (Python side of the contracts).

Each part implements one Protocol below and talks to the others ONLY through these
methods and the Pydantic contracts in ``contracts/models.py``. That is what lets the
parts be built in parallel: code against the Protocol, start with a stub, swap in the
real implementation later without touching the caller.

Parts
-----
0. DataLayer            greentwin/data, greentwin/geo
1. DigitalTwin          greentwin/twin (JAX physics), greentwin/bayes (NumPyro), 3D scene, photo capture
2. EnergyRetrofitEngine greentwin/models (EBM prior), greentwin/retrofit (ladder, rating, Pareto)
3. GreenHomeScorer      greentwin/score
4. ValueModel           greentwin/value
5. FeasibilityEngine, Router, Explainer   greentwin/retrofit/rules.py, greentwin/jev, greentwin/llm, greentwin/kb
-. Orchestrator         greentwin/api (run_scenario + FastAPI)

Rules every implementation follows: uncertain outputs are ``Distribution``s built with
``contracts.summarise``; every result carries ``Provenance``; missing data is ``None``;
savings are per fuel and packages are re-simulated, never summed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, runtime_checkable

from contracts.models import (
    AreaEPCStats,
    CalibrationResult,
    CaptureResult,
    ChatRequest,
    ChatResponse,
    CustomerMemory,
    DashboardArea,
    ElectricityProfileSummary,
    ElementHeatLoss,
    EnergyBreakdown,
    EnergyPrior,
    EnergyUsageMonth,
    EPCRecommendation,
    FeasibilityReport,
    GreenTwin,
    ImprovementPotential,
    LoadProfileArchetype,
    LSOAConsumption,
    ParetoPoint,
    Property,
    RatingEstimate,
    RetrofitAction,
    RetrofitEffect,
    RetrofitItem,
    SaleRecord,
    ScenarioResult,
    ScenarioSpec,
    ScoreResult,
    TwinParams,
    TwinScene,
    ValueImpact,
    WeatherMonth,
    WhatIfRoute,
)

Array = Any  # jax.Array / numpy.ndarray; kept loose so this module imports without JAX installed
Weather = Literal["typical", "cold_year", "warm_year", "forecast_week", "ukcp18_2050"]
Goal = Literal["bills", "carbon"]


@dataclass
class PosteriorSamples:
    """Posterior draws for one home, shared in-process between Part 1 and Parts 2-4.

    ``params`` maps a TwinParams field name to an array of shape (n_samples,). Fields that were
    not calibrated hold the RdSAP/EBM prior value repeated. Persisted to Parquet at
    ``CalibrationResult.posterior_ref``; never sent over the API (send Distributions instead).
    """

    property_id: str
    n_samples: int
    params: Mapping[str, Array]
    base: TwinParams
    seed: int = 42


@dataclass
class TwinRun:
    """Vectorised twin output over samples: arrays of shape (n_samples,) or (n_samples, 12) monthly."""

    electricity_import_kwh: Array
    gas_kwh: Array
    space_heating_kwh: Array
    hot_water_kwh: Array
    pv_generation_kwh: Array
    export_kwh: Array
    monthly_electricity_kwh: Optional[Array] = None
    monthly_gas_kwh: Optional[Array] = None
    element_heat_loss_w_per_k: Mapping[str, Array] = field(default_factory=dict)


# --------------------------------------------------------------------------- Part 0
@runtime_checkable
class DataLayer(Protocol):
    def load_property(self, property_id: str) -> Property: ...
    def load_usage(self, property_id: str) -> list[EnergyUsageMonth]: ...
    def load_weather(self, area_code: str, months: Sequence[str]) -> list[WeatherMonth]: ...
    def load_sales(self, region: Optional[str] = None, split: Optional[str] = None) -> Sequence[SaleRecord]: ...
    def load_epc_recommendations(self, property_id: str) -> list[EPCRecommendation]: ...
    def load_area_epc_stats(self, local_authority_code: str, period: Optional[str] = None) -> list[AreaEPCStats]: ...
    def load_lsoa_consumption(self, year: Optional[int] = None) -> Sequence[LSOAConsumption]: ...
    def load_load_profiles(self) -> list[LoadProfileArchetype]: ...
    def catalogue(self) -> list[RetrofitItem]: ...
    def emission_factors(self) -> dict[str, float]: ...
    def tariffs(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- Part 1
@runtime_checkable
class DigitalTwin(Protocol):
    """Physics (JAX), Bayesian calibration (NumPyro), 3D scene and photo capture."""

    def params_from_property(self, prop: Property) -> TwinParams:
        """RdSAP defaults (+ captured room geometry) -> one parameter set."""

    def apply_retrofits(self, params: TwinParams, actions: Sequence[RetrofitAction]) -> TwinParams:
        """Apply RetrofitItem.twin_edits in order. Pure function; never sums savings."""

    def simulate(self, samples: PosteriorSamples, actions: Sequence[RetrofitAction], weather: Weather,
                 mode: Literal["calibrated", "standard"] = "calibrated") -> TwinRun:
        """Run the twin on every posterior sample (jax.vmap). 'standard' = SAP occupancy for ratings."""

    def calibrate(self, prop: Property, usage: Sequence[EnergyUsageMonth], weather: Sequence[WeatherMonth],
                  prior: Optional[EnergyPrior] = None) -> tuple[CalibrationResult, PosteriorSamples]:
        """NumPyro: SVI per home (NUTS offline). Missing months are latent; no meters -> prior_only."""

    def element_heat_loss(self, samples: PosteriorSamples, actions: Sequence[RetrofitAction] = ()) -> list[ElementHeatLoss]:
        """Prior split scaled by the calibrated total; always estimated=True."""

    def electricity_profile(self, samples: PosteriorSamples, actions: Sequence[RetrofitAction],
                            profiles: Sequence[LoadProfileArchetype], tariffs: Mapping[str, Any]) -> ElectricityProfileSummary:
        """Half-hourly layer: load shape + heat pump + PV - battery -> self-consumption, ToU bill."""

    def scene(self, prop: Property, samples: PosteriorSamples, actions: Sequence[RetrofitAction] = (),
              feasibility: Optional[FeasibilityReport] = None) -> TwinScene:
        """Geometry + surface heat-loss shares + retrofit slots for the 3D viewer. Never calculates savings."""

    def capture(self, property_id: str, floor_plan_image: Optional[bytes],
                room_photos: Sequence[bytes]) -> CaptureResult:
        """Photo-to-twin: Claude Haiku 5.5 vision drafts rooms + tags with confidence; agent confirms later."""


# --------------------------------------------------------------------------- Part 2
@runtime_checkable
class EnergyRetrofitEngine(Protocol):
    """Energy prediction, retrofit ladder, savings/carbon/bills, improvement potential, rating after, Pareto."""

    def energy_prior(self, prop: Property) -> EnergyPrior:
        """EBM trained on NEED: kWh/m2 per fuel + feature contributions (prior for homes without meters)."""

    def baseline(self, prop: Property, samples: PosteriorSamples, weather: Weather = "typical") -> EnergyBreakdown:
        """Per-fuel baseline as Distributions; intensity = annualised metered energy / m2."""

    def ladder(self, prop: Property, samples: PosteriorSamples, feasibility: FeasibilityReport,
               goal: Goal = "bills") -> list[RetrofitEffect]:
        """Feasible retrofits in fabric-first order; marginal effect of each step on every sample; verdicts."""

    def package(self, prop: Property, samples: PosteriorSamples,
                actions: Sequence[RetrofitAction]) -> EnergyBreakdown:
        """'After' energy for a package: ONE twin run on the whole package."""

    def improvement_potential(self, prop: Property, samples: PosteriorSamples,
                              feasibility: FeasibilityReport) -> ImprovementPotential:
        """All feasible retrofits together; null + reason if the baseline is 0 or missing."""

    def estimate_rating(self, prop: Property, samples: PosteriorSamples,
                        actions: Sequence[RetrofitAction]) -> Optional[RatingEstimate]:
        """contracts/sap_estimator.json: SAP 2012 on standard mode, anchored to the recorded SAP."""

    def pareto(self, prop: Property, samples: PosteriorSamples, feasibility: FeasibilityReport,
               budget_gbp: Optional[float] = None) -> list[ParetoPoint]:
        """Enumerate feasible packages; P(on front) across samples; knee."""


# --------------------------------------------------------------------------- Part 3
@runtime_checkable
class GreenHomeScorer(Protocol):
    def score(self, prop: Property, energy: EnergyBreakdown, rating: Optional[RatingEstimate] = None,
              peer: Optional[AreaEPCStats] = None) -> ScoreResult:
        """Fixed weights x monotonic sub-scores (contracts/score_mapping.json). rating given -> projected score.
        Distribution over posterior intensity samples; key issues with evidence; peer context from EPC live tables."""


# --------------------------------------------------------------------------- Part 4
@runtime_checkable
class ValueModel(Protocol):
    def value_impact(self, prop: Property, rating_after: Optional[RatingEstimate]) -> ValueImpact:
        """Bayesian hierarchical hedonic: same home at recorded vs estimated-after rating. Always 'indicative'."""

    def green_twins(self, prop: Property, k: int = 3) -> list[GreenTwin]:
        """Matched greener comparable homes, as ranges (no addresses)."""


# --------------------------------------------------------------------------- Part 5
@runtime_checkable
class FeasibilityEngine(Protocol):
    def assess(self, prop: Property, catalogue: Sequence[RetrofitItem],
               memory: Optional[CustomerMemory] = None) -> FeasibilityReport:
        """Layer 1 rule vetoes (contracts/feasibility_rules.json) always win; layer 2 Jev on survivors
        (Claude structured output if no key); low confidence -> needs_survey."""


@runtime_checkable
class Router(Protocol):
    def route(self, property_id: str, message: str) -> WhatIfRoute:
        """Jev System-1: intent + ScenarioSpec slots with calibrated confidence; below tau -> clarify."""


@runtime_checkable
class Explainer(Protocol):
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Tool-only LLM (get_home_facts, get_results, run_scenario, search_docs, query_graph); numeric
        validator; regenerate once then template fallback. Never computes a number."""


# --------------------------------------------------------------------------- Orchestration
@runtime_checkable
class Orchestrator(Protocol):
    def run_scenario(self, spec: ScenarioSpec) -> ScenarioResult:
        """calibrate (cached) -> feasibility -> baseline/package/ladder -> rating -> score before/after
        -> value -> pareto; one ScenarioResult with provenance; cached by hash(spec + model versions)."""

    def dashboard_areas(self, layer: str, region: Optional[str] = None) -> list[DashboardArea]: ...
