"""GreenTwin shared data contracts (source of truth).

Every module exchanges data through these Pydantic v2 models. The TypeScript
mirror lives in ``contracts/types.ts`` and must be updated in the same commit
as any change here. Regenerate JSON Schema with::

    uv run python -m contracts.export_schemas

Conventions
-----------
* Energy in kWh/year unless the field name says otherwise; money in GBP;
  carbon in kgCO2e; areas in m2; U-values in W/m2K; heat-loss in W/K.
* Savings are ``baseline - after``: positive = saving, negative = increase
  (e.g. electricity rises after a heat pump).
* Every model output that is uncertain is a ``Distribution``: the full summary
  of its posterior samples (mean, median, mode, std, percentiles P1-P99, 50%/90%
  credible intervals, tail probabilities, worst-tail mean, skewness, histogram).
  Build it with ``contracts.summarise.summarise(samples, unit)`` so every module
  summarises the same way. Only documented constants (weights, emission factors,
  tariffs, grants, rule vetoes) are bare numbers.
* Fields marked ``leakage`` in their description must never be model inputs.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "0.3.0"


# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #
class Percentiles(BaseModel):
    """Posterior percentiles; must be non-decreasing from p1 to p99."""

    p1: float
    p5: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float


class CredibleInterval(BaseModel):
    """Central (equal-tailed) credible interval, e.g. level 0.9 = P5..P95."""

    level: float = Field(gt=0, lt=1)
    lo: float
    hi: float


class TailProbability(BaseModel):
    """Probability that the quantity falls beyond a meaningful threshold, e.g. P(saving < 0)."""

    label: str = Field(description="Human-readable event, e.g. 'bill rises by more than £100/yr'")
    threshold: float
    direction: Literal["below", "above"]
    probability: float = Field(ge=0, le=1)


class Histogram(BaseModel):
    """Sample histogram for the UI density plot; len(bin_edges) == len(counts) + 1."""

    bin_edges: list[float]
    counts: list[int]


class Distribution(BaseModel):
    """Full posterior summary of an uncertain quantity (replaces the old mean/lo/hi Interval).

    Built from posterior samples by ``contracts.summarise.summarise``. A known-exact value
    (e.g. gas after a heat pump removes the boiler) is a degenerate distribution: std = 0.
    """

    unit: str = Field(description="e.g. 'kWh/yr', 'GBP/yr', 'kgCO2e/yr', 'points', '%'")
    n_samples: int = Field(ge=1, description="Posterior samples the summary was computed from")
    mean: float
    median: float = Field(description="Equals percentiles.p50")
    mode: float = Field(description="Most likely value: peak of the sample density (KDE)")
    std: float = Field(ge=0)
    skewness: float = Field(0.0, description="Sample skewness; |skew| > 0.5 -> headline uses the median")
    percentiles: Percentiles
    credible_intervals: list[CredibleInterval] = Field(description="At least the 50% and 90% intervals")
    tail_probabilities: list[TailProbability] = Field(default_factory=list)
    worst_tail: Literal["low", "high"] = Field(
        "low", description="Which side is the bad outcome (low for savings, high for costs)"
    )
    worst_5pct_mean: float = Field(description="Mean of the worst 5% of samples (expected shortfall)")
    histogram: Optional[Histogram] = None
    headline: Literal["mean", "median"] = Field("mean", description="Which centre the UI/LLM shows first")

    @model_validator(mode="after")
    def _consistent(self) -> "Distribution":
        p = self.percentiles
        seq = [p.p1, p.p5, p.p10, p.p25, p.p50, p.p75, p.p90, p.p95, p.p99]
        if any(b < a - 1e-9 for a, b in zip(seq, seq[1:])):
            raise ValueError("percentiles must be non-decreasing")
        if abs(self.median - p.p50) > 1e-6 * max(1.0, abs(p.p50)):
            raise ValueError("median must equal percentiles.p50")
        levels = {round(ci.level, 6) for ci in self.credible_intervals}
        if not {0.5, 0.9} <= levels:
            raise ValueError("credible_intervals must include the 50% and 90% intervals")
        for ci in self.credible_intervals:
            if ci.lo > ci.hi:
                raise ValueError("credible interval lo > hi")
        if self.histogram and len(self.histogram.bin_edges) != len(self.histogram.counts) + 1:
            raise ValueError("histogram needs len(bin_edges) == len(counts) + 1")
        return self

    def interval(self, level: float = 0.9) -> CredibleInterval:
        """The credible interval at ``level`` (0.5 or 0.9 are always present)."""
        for ci in self.credible_intervals:
            if abs(ci.level - level) < 1e-9:
                return ci
        raise KeyError(f"no {level:.0%} interval")


class Provenance(BaseModel):
    """Where a number came from. Every result carries one."""

    engine: Literal["physics_l1", "surrogate_deeponet", "sap_estimator", "ebm", "xgboost", "hedonic", "dr_learner", "rules", "mock"]
    model_version: str
    data_version: str
    seed: Optional[int] = None
    n_samples: Optional[int] = Field(None, description="Posterior samples used")


# --------------------------------------------------------------------------- #
# Property (input)
# --------------------------------------------------------------------------- #
class PropertyType(str, Enum):
    house = "house"
    bungalow = "bungalow"
    flat = "flat"
    maisonette = "maisonette"


class BuiltForm(str, Enum):
    detached = "detached"
    semi_detached = "semi_detached"
    mid_terrace = "mid_terrace"
    end_terrace = "end_terrace"
    enclosed_mid_terrace = "enclosed_mid_terrace"
    not_applicable = "not_applicable"


class WallType(str, Enum):
    solid_brick = "solid_brick"
    solid_stone = "solid_stone"
    cavity = "cavity"
    timber_frame = "timber_frame"
    system_built = "system_built"


class WallInsulation(str, Enum):
    none = "none"
    cavity_filled = "cavity_filled"
    internal = "internal"
    external = "external"
    as_built = "as_built"  # insulated to building regs at construction


class RoofType(str, Enum):
    pitched_loft = "pitched_loft"
    pitched_room_in_roof = "pitched_room_in_roof"
    flat = "flat"
    another_dwelling_above = "another_dwelling_above"


class FloorType(str, Enum):
    suspended_timber = "suspended_timber"
    solid = "solid"
    another_dwelling_below = "another_dwelling_below"


class GlazingType(str, Enum):
    single = "single"
    secondary = "secondary"
    double_pre_2002 = "double_pre_2002"
    double_post_2002 = "double_post_2002"
    triple = "triple"


class HeatingSystem(str, Enum):
    gas_boiler_condensing = "gas_boiler_condensing"
    gas_boiler_non_condensing = "gas_boiler_non_condensing"
    oil_boiler = "oil_boiler"
    lpg_boiler = "lpg_boiler"
    electric_storage = "electric_storage"
    electric_storage_hhr = "electric_storage_hhr"
    electric_direct = "electric_direct"
    air_source_heat_pump = "air_source_heat_pump"
    ground_source_heat_pump = "ground_source_heat_pump"
    district_heating = "district_heating"
    communal_boiler = "communal_boiler"
    solid_fuel = "solid_fuel"


class Control(str, Enum):
    none = "none"
    programmer = "programmer"
    room_thermostat = "room_thermostat"
    trvs = "trvs"
    smart_thermostat = "smart_thermostat"
    zoned_control = "zoned_control"
    smart_meter = "smart_meter"


class Tenure(str, Enum):
    owner_occupied = "owner_occupied"
    private_rented = "private_rented"
    social_rented = "social_rented"
    unknown = "unknown"


class Location(BaseModel):
    postcode_district: str = Field(description="Outward code only, e.g. 'M14'. No full postcodes/addresses.")
    lsoa_code: Optional[str] = None
    local_authority: Optional[str] = None
    region: str = Field(description="ONS region, e.g. 'North West'")
    lat: Optional[float] = Field(None, description="Rounded to 3 dp (~100 m) for privacy")
    lon: Optional[float] = None
    altitude_m: Optional[float] = None


class EPCInfo(BaseModel):
    current_band: Optional[Literal["A", "B", "C", "D", "E", "F", "G"]] = None
    sap_score: Optional[int] = Field(None, ge=1, le=100)
    lodgement_date: Optional[str] = Field(None, description="ISO date")
    potential_band: Optional[str] = Field(None, description="leakage: reference label only, never a model input")
    potential_sap_score: Optional[int] = Field(None, description="leakage: reference label only")
    estimated_annual_energy_cost_gbp: Optional[float] = Field(None, description="EPC modelled current cost")


class Walls(BaseModel):
    type: WallType
    insulation: WallInsulation
    u_value: Optional[float] = Field(None, description="W/m2K if measured; else from RdSAP defaults")


class Roof(BaseModel):
    type: RoofType
    loft_insulation_mm: Optional[int] = Field(None, description="Null if no loft")
    u_value: Optional[float] = None


class Floor(BaseModel):
    type: FloorType
    insulated: bool = False
    u_value: Optional[float] = None


class Glazing(BaseModel):
    type: GlazingType
    share_pct: float = Field(100, ge=0, le=100, description="Share of windows with this glazing")
    window_to_floor_ratio: Optional[float] = None


class Heating(BaseModel):
    main_system: HeatingSystem
    main_fuel: Literal["mains_gas", "electricity", "oil", "lpg", "solid", "district"]
    system_age_years: Optional[int] = None
    seasonal_efficiency: Optional[float] = Field(None, description="Boiler efficiency or heat-pump SCOP")
    hot_water: Literal["from_main", "electric_immersion", "heat_pump_cylinder", "instantaneous"] = "from_main"
    cylinder_insulated: Optional[bool] = None


class Renewables(BaseModel):
    pv_kwp: float = 0.0
    pv_azimuth_deg: Optional[float] = Field(None, description="180 = south")
    pv_tilt_deg: Optional[float] = None
    battery_kwh: float = 0.0
    solar_thermal: bool = False


class Constraints(BaseModel):
    """Feasibility inputs. Rules read these; Jev reads them plus notes."""

    conservation_area: bool = False
    listed_building: bool = False
    leasehold: bool = False
    has_outdoor_space: Optional[bool] = Field(None, description="For heat-pump outdoor unit")
    roof_access: Optional[bool] = Field(None, description="Own roof available for PV")
    best_roof_azimuth_deg: Optional[float] = None
    off_gas_grid: bool = False
    floor_level: Optional[int] = Field(None, description="0 = ground; flats only")
    free_text_notes: list[str] = Field(default_factory=list, description="Agent/customer notes; read by Jev")


class DataQuality(BaseModel):
    missing_fields: list[str] = Field(default_factory=list)
    imputed_fields: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list, description="e.g. 'EPC A but 400 kWh/m2 metered'")
    months_of_metered_data: int = 0


class Property(BaseModel):
    schema_version: str = SCHEMA_VERSION
    property_id: str
    customer_id: Optional[str] = Field(None, description="Split key when present (customer-level splits)")
    source: Literal["epc_real", "synthetic", "showcase", "agent_capture", "self_serve"]
    split: Optional[Literal["train", "val", "test"]] = Field(None, description="Assigned once, by property/customer")
    location: Location
    property_type: PropertyType
    built_form: BuiltForm
    construction_age_band: str = Field(description="RdSAP band, e.g. '1900-1929'")
    year_built: Optional[int] = None
    storeys: int = 1
    floor_area_m2: float
    habitable_rooms: Optional[int] = None
    bedrooms: Optional[int] = None
    tenure: Tenure = Tenure.unknown
    epc: EPCInfo = Field(default_factory=EPCInfo)
    walls: Walls
    roof: Roof
    floor: Floor
    glazing: list[Glazing]
    draught_proofed_pct: Optional[float] = None
    heating: Heating
    renewables: Renewables = Field(default_factory=Renewables)
    controls: list[Control] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    current_value_estimate_gbp: Optional[float] = Field(
        None, description="Baseline valuation used for value impact. NOT a historical sale price."
    )
    value_estimate_date: Optional[str] = None
    data_quality: DataQuality = Field(default_factory=DataQuality)


class EnergyUsageMonth(BaseModel):
    property_id: str
    month: str = Field(description="YYYY-MM")
    electricity_import_kwh: Optional[float] = Field(None, description="Null = missing, never 0 by default")
    gas_kwh: Optional[float] = Field(None, description="Gross CV kWh. Null = missing; 0 only if no gas supply")
    pv_generation_kwh: Optional[float] = None
    electricity_export_kwh: Optional[float] = None
    estimated_read: bool = False


class SaleRecord(BaseModel):
    """Historical transaction. Never used as the baseline value."""

    property_id: str
    date: str
    price_gbp: float
    energy_rating_at_sale: Optional[str] = None
    new_build: Optional[bool] = None


class WeatherMonth(BaseModel):
    area_code: str = Field(description="LSOA or region code")
    month: str
    hdd_15_5: float = Field(description="Heating degree days, base 15.5 C")
    cdd_22: Optional[float] = None
    mean_temp_c: Optional[float] = None
    ghi_kwh_m2: Optional[float] = Field(None, description="Global horizontal irradiation")
    source: Literal["haduk_grid", "open_meteo_era5", "ukcp18", "met_office_datahub", "synthetic"]


# --------------------------------------------------------------------------- #
# Retrofit catalogue
# --------------------------------------------------------------------------- #
class RetrofitItem(BaseModel):
    id: str
    name: str
    category: Literal["fabric", "glazing", "heating", "renewables", "controls", "airtightness"]
    twin_edits: dict[str, float | str] = Field(description="Parameter changes applied to the physics twin")
    blocked_by_rules: list[str] = Field(default_factory=list, description="Rule ids that veto this retrofit")
    depends_on: list[str] = Field(default_factory=list, description="Should come after these (ordering)")
    excludes: list[str] = Field(default_factory=list, description="Mutually exclusive retrofits")
    cost_gbp_low: Optional[float] = None
    cost_gbp_central: Optional[float] = None
    cost_gbp_high: Optional[float] = None
    cost_basis: str = Field("per_home", description="per_home | per_m2 | per_kwp | per_window")
    cost_price_year: Optional[str] = None
    cost_status: Literal["sourced", "secondary_source", "unsourced"] = Field(
        "unsourced", description="unsourced -> no cost figure is ever shown (cost = null + reason)"
    )
    lifetime_years: Optional[int] = None
    grants: list[str] = Field(default_factory=list)
    sizing: Optional[dict[str, float]] = Field(None, description="Continuous sizing bounds, e.g. {'kwp_min':1,'kwp_max':6}")
    source: str = Field(description="Citation for cost/impact assumptions")


# --------------------------------------------------------------------------- #
# Scenario (what-if) request and result
# --------------------------------------------------------------------------- #
class RetrofitAction(BaseModel):
    retrofit_id: str
    size: Optional[float] = Field(None, description="e.g. kWp, mm, kW")
    stage_year: Optional[int] = Field(None, description="Roadmap year (0 = now)")


class ScenarioSpec(BaseModel):
    schema_version: str = SCHEMA_VERSION
    scenario_id: str
    property_id: str
    retrofits: list[RetrofitAction] = Field(default_factory=list, description="Empty = baseline only")
    weather: Literal["typical", "cold_year", "warm_year", "forecast_week", "ukcp18_2050"] = "typical"
    n_samples: int = Field(500, description="Posterior samples to propagate")
    include_value: bool = True
    include_pareto: bool = False
    include_element_heat_loss: bool = True
    seed: int = 42
    goal: Literal["bills", "carbon"] = Field(
        "bills", description="Defines 'net benefit' for suitability/verdicts; never changes computed numbers"
    )
    requested_by: Literal["ui", "llm", "batch", "jev_router", "eval"] = "ui"


class EnergyBreakdown(BaseModel):
    electricity_import_kwh: Distribution
    gas_kwh: Distribution
    purchased_total_kwh: Distribution
    space_heating_kwh: Optional[Distribution] = None
    hot_water_kwh: Optional[Distribution] = None
    pv_generation_kwh: Optional[Distribution] = None
    intensity_kwh_m2: Distribution = Field(
        description="Annualised METERED energy (grid import + gas) / floor area, as defined in the brief; used by the score"
    )
    intensity_weather_normalised_kwh_m2: Optional[Distribution] = Field(
        None, description="Sensitivity only; never used in the score"
    )


class ScoreComponent(BaseModel):
    name: Literal[
        "energy_efficiency", "energy_intensity", "insulation", "heating_system",
        "windows_glazing", "renewable_energy", "smart_energy_management",
    ]
    weight: float
    sub_score: float = Field(ge=0, le=100)
    contribution: float = Field(description="weight * sub_score (posterior mean if the input is uncertain)")
    contribution_distribution: Optional[Distribution] = Field(
        None, description="Set when the component's input is uncertain (intensity, imputed fields)"
    )
    inputs_used: dict[str, float | str | None] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    score: int = Field(ge=0, le=100, description="Rounded weighted sum (of the posterior mean inputs)")
    score_distribution: Optional[Distribution] = Field(None, description="Score over posterior samples")
    band: Literal["Highly efficient", "Good", "Moderate", "Poor", "Very poor"] = Field(
        description="Mode of band_probabilities (most probable band)"
    )
    band_probabilities: dict[str, float] = Field(default_factory=dict, description="P(each band); sums to 1")
    p_poor_or_very_poor: Optional[float] = Field(None, ge=0, le=1)
    components: list[ScoreComponent]
    key_issues: list[str] = Field(default_factory=list)


class Feasibility(BaseModel):
    retrofit_id: str
    status: Literal["feasible", "feasible_with_conditions", "needs_survey", "infeasible"]
    decided_by: Literal["rules", "jev", "claude_fallback", "agent"]
    reason_codes: list[str] = Field(default_factory=list)
    p_feasible: float = Field(ge=0, le=1, description="1.0/0.0 for rules; recalibrated confidence for Jev")


class RetrofitEffect(BaseModel):
    retrofit_id: str
    order: int = Field(description="Position in the recommended sequence")
    feasibility: Feasibility
    electricity_saving_kwh: Distribution = Field(description="Marginal, in sequence; negative = increase")
    gas_saving_kwh: Distribution
    carbon_saving_kgco2e: Distribution
    bill_saving_gbp: Distribution
    cost_gbp: Optional[Distribution] = None
    cost_unavailable_reason: Optional[str] = Field(None, description="Set when the catalogue cost is unsourced")
    purchased_energy_saving_kwh: Optional[Distribution] = Field(None, description="Marginal grid import + gas saving")
    score_after: Optional[Distribution] = Field(None, description="Projected score after this step (cumulative)")
    improvement_potential_cumulative_pct: Optional[Distribution] = Field(
        None, description="Improvement potential reached after steps 1..order"
    )
    payback_years: Optional[Distribution] = None
    p_net_benefit: float = Field(ge=0, le=1, description="Under ScenarioSpec.goal, cost uncertainty included")
    p_suitable: float = Field(ge=0, le=1, description="p_feasible * p_net_benefit")
    suitability: Literal["High", "Medium", "Low"]
    p_gain_above_threshold: Optional[float] = Field(
        None, ge=0, le=1, description="P(step adds >= the documented gain threshold)"
    )
    verdict: Optional[Literal["recommended", "optional", "not_worth_adding", "needs_survey", "not_feasible"]] = None
    reason: str = Field(description="Short human-readable reason built from rule codes/KB, not LLM")


class RatingEstimate(BaseModel):
    """Post-retrofit EPC-style rating from contracts/sap_estimator.json. Never presented as an EPC."""

    sap: Distribution
    band: Literal["A", "B", "C", "D", "E", "F", "G"]
    band_probabilities: dict[str, float] = Field(default_factory=dict)
    anchored_to_recorded_sap: bool = Field(description="False when the home has no recorded EPC")
    method: Literal["sap2012_ecf_delta"] = "sap2012_ecf_delta"
    label: Literal["estimated rating, not an EPC"] = "estimated rating, not an EPC"


class ValueImpact(BaseModel):
    current_value_gbp: Optional[float] = Field(description="Baseline valuation (never a historical sale price)")
    current_value_distribution: Optional[Distribution] = Field(
        None, description="Set when the current value is model-estimated rather than provided"
    )
    band_before: Optional[str] = Field(None, description="Recorded EPC band")
    band_after: Optional[str] = Field(None, description="From RatingEstimate (sap_estimator), not a lodged EPC")
    uplift_pct: Optional[Distribution] = None
    value_after_gbp: Optional[Distribution] = None
    method: Literal["bayesian_hedonic", "hedonic_quantile", "dr_learner", "green_twin_matching", "unavailable"]
    label: Literal["indicative"] = "indicative"
    wording: str = Field(
        "Indicative range based on historical associations in sales data; not a promised or proven increase.",
        description="Must accompany every display of the value range",
    )
    unavailable_reason: Optional[str] = None


class ElementHeatLoss(BaseModel):
    """Level 1 heat loss per envelope element; drives 3D colouring."""

    element: Literal["roof", "walls", "floor", "windows", "doors", "ventilation", "party_walls"]
    heat_loss_w_per_k: float
    share_pct: float
    u_value: Optional[float] = None
    area_m2: Optional[float] = None
    estimated: bool = Field(True, description="Per-element split comes from priors, not measurement")
    basis: str = Field(
        "Estimated from home type and EPC descriptions; only the total heat loss is calibrated to bills",
        description="Label shown wherever element colours/values are displayed",
    )


class ParetoPoint(BaseModel):
    package: list[str]
    net_cost_gbp: Distribution
    carbon_saving_kgco2e: Distribution
    bill_saving_gbp: Distribution
    score_gain: Distribution
    p_on_front: float
    is_knee: bool = False


class ScenarioResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    scenario_id: str
    property_id: str
    provenance: Provenance
    baseline: EnergyBreakdown
    after: EnergyBreakdown
    electricity_saving_kwh: Distribution = Field(description="baseline - after; negative = increase")
    gas_saving_kwh: Distribution
    purchased_energy_saving_kwh: Distribution
    improvement_potential_pct: Optional[Distribution] = Field(
        None, description="100*(E_base - E_after)/E_base on purchased energy; null if baseline 0/unavailable"
    )
    improvement_potential_unavailable_reason: Optional[str] = None
    improvement_potential_p_top_decile: Optional[float] = Field(
        None, ge=0, le=1, description="P(this home is in the top 10% by improvement potential) for ranking"
    )
    carbon_saving_kgco2e: Optional[Distribution] = Field(None, description="Null if a fuel has no sourced factor")
    carbon_unavailable_reason: Optional[str] = None
    emission_factors: dict[str, float] = Field(
        description="kgCO2e/kWh from contracts/emission_factors.json, e.g. {'electricity':0.19553,'gas':0.18296}"
    )
    emission_factor_source: str
    bill_saving_gbp: Distribution
    score_before: ScoreResult
    score_after: ScoreResult = Field(description="Uses rating_after (estimated) for the energy-efficiency component")
    rating_after: Optional[RatingEstimate] = None
    retrofits: list[RetrofitEffect] = Field(default_factory=list)
    value: ValueImpact
    element_heat_loss_before: list[ElementHeatLoss] = Field(default_factory=list)
    element_heat_loss_after: list[ElementHeatLoss] = Field(default_factory=list)
    pareto: list[ParetoPoint] = Field(default_factory=list)
    data_confidence: Literal["high", "medium", "low"] = "medium"
    warnings: list[str] = Field(default_factory=list, description="Missing/conflicting data, OOD, assumptions")


# --------------------------------------------------------------------------- #
# LLM / Jev boundary
# --------------------------------------------------------------------------- #
class WhatIfRoute(BaseModel):
    """Output of the System-1 router (Jev or Claude fallback)."""

    intent: Literal["numeric", "comparative", "feasibility", "explain", "out_of_scope"]
    confidence: float = Field(ge=0, le=1)
    scenario: Optional[ScenarioSpec] = None
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    router: Literal["jev", "claude_fallback"]


class ExplanationCheck(BaseModel):
    """Validator output for one LLM answer."""

    numbers_found: list[str]
    unsupported_numbers: list[str]
    jev_supported_probability: Optional[float] = None
    passed: bool


class CustomerMemory(BaseModel):
    customer_id: str
    facts: list[dict[str, str]] = Field(
        default_factory=list, description="e.g. {'type':'constraint','value':'renting'}"
    )
    completed_retrofits: list[RetrofitAction] = Field(default_factory=list)
    rejected: list[dict[str, str]] = Field(default_factory=list, description="{'retrofit_id':..,'reason':..}")
    saved_scenarios: list[str] = Field(default_factory=list)
