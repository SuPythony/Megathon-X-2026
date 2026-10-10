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
* Every model output that is uncertain is an ``Interval`` (posterior mean and
  90% credible interval by default).
* Fields marked ``leakage`` in their description must never be model inputs.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #
class Interval(BaseModel):
    """Posterior summary of an uncertain quantity."""

    mean: float
    lo: float = Field(description="Lower bound of the credible interval")
    hi: float = Field(description="Upper bound of the credible interval")
    level: float = Field(0.9, description="Credible level, e.g. 0.9 for 90%")
    unit: str = Field(description="e.g. 'kWh/yr', 'GBP', 'kgCO2e/yr', 'points'")


class Provenance(BaseModel):
    """Where a number came from. Every result carries one."""

    engine: Literal["physics_l1", "surrogate_deeponet", "ebm", "xgboost", "hedonic", "dr_learner", "rules", "mock"]
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
    cost_gbp_high: Optional[float] = None
    cost_basis: str = Field("per_home", description="per_home | per_m2 | per_kwp | per_window")
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
    requested_by: Literal["ui", "llm", "batch", "jev_router", "eval"] = "ui"


class EnergyBreakdown(BaseModel):
    electricity_import_kwh: Interval
    gas_kwh: Interval
    purchased_total_kwh: Interval
    space_heating_kwh: Optional[Interval] = None
    hot_water_kwh: Optional[Interval] = None
    pv_generation_kwh: Optional[Interval] = None
    intensity_kwh_m2: Interval


class ScoreComponent(BaseModel):
    name: Literal[
        "energy_efficiency", "energy_intensity", "insulation", "heating_system",
        "windows_glazing", "renewable_energy", "smart_energy_management",
    ]
    weight: float
    sub_score: float = Field(ge=0, le=100)
    contribution: float = Field(description="weight * sub_score")
    inputs_used: dict[str, float | str | None] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    score: int = Field(ge=0, le=100, description="Rounded weighted sum")
    score_interval: Optional[Interval] = None
    band: Literal["Highly efficient", "Good", "Moderate", "Poor", "Very poor"]
    band_probabilities: dict[str, float] = Field(default_factory=dict)
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
    electricity_saving_kwh: Interval = Field(description="Marginal, in sequence; negative = increase")
    gas_saving_kwh: Interval
    carbon_saving_kgco2e: Interval
    bill_saving_gbp: Interval
    cost_gbp: Optional[Interval] = None
    p_net_benefit: float = Field(ge=0, le=1)
    p_suitable: float = Field(ge=0, le=1, description="p_feasible * p_net_benefit")
    suitability: Literal["High", "Medium", "Low"]
    reason: str = Field(description="Short human-readable reason built from rule codes/KB, not LLM")


class ValueImpact(BaseModel):
    current_value_gbp: Optional[float]
    uplift_pct: Optional[Interval] = None
    value_after_gbp: Optional[Interval] = None
    method: Literal["hedonic_quantile", "dr_learner", "green_twin_matching", "unavailable"]
    label: Literal["indicative"] = "indicative"
    unavailable_reason: Optional[str] = None


class ElementHeatLoss(BaseModel):
    """Level 1 heat loss per envelope element; drives 3D colouring."""

    element: Literal["roof", "walls", "floor", "windows", "doors", "ventilation", "party_walls"]
    heat_loss_w_per_k: float
    share_pct: float
    u_value: Optional[float] = None
    area_m2: Optional[float] = None


class ParetoPoint(BaseModel):
    package: list[str]
    net_cost_gbp: Interval
    carbon_saving_kgco2e: Interval
    bill_saving_gbp: Interval
    score_gain: Interval
    p_on_front: float
    is_knee: bool = False


class ScenarioResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    scenario_id: str
    property_id: str
    provenance: Provenance
    baseline: EnergyBreakdown
    after: EnergyBreakdown
    electricity_saving_kwh: Interval = Field(description="baseline - after; negative = increase")
    gas_saving_kwh: Interval
    purchased_energy_saving_kwh: Interval
    improvement_potential_pct: Optional[Interval] = Field(
        None, description="100*(E_base - E_after)/E_base on purchased energy; null if baseline 0/unavailable"
    )
    improvement_potential_unavailable_reason: Optional[str] = None
    carbon_saving_kgco2e: Interval
    emission_factors: dict[str, float] = Field(description="kgCO2e/kWh used, e.g. {'electricity':0.177,'gas':0.183}")
    emission_factor_source: str
    bill_saving_gbp: Interval
    score_before: ScoreResult
    score_after: ScoreResult
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
