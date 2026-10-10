"""GreenTwin shared data contracts (source of truth).

Every part of the system exchanges data through these Pydantic v2 models. The
TypeScript mirror ``contracts/types.ts`` is GENERATED from this file
(``python -m contracts.export_ts``); JSON Schema with
``python -m contracts.export_schemas``. In-process Python interfaces between the
parts live in ``greentwin/interfaces.py``.

The contracts are grouped by the system's parts:

* Part 0  Data layer            - inputs, public datasets, labels
* Part 1  Digital twin          - physics parameters, Bayesian calibration, 3D scene, photo capture
* Part 2  Energy & retrofit     - energy prediction, retrofit ladder, savings, carbon, rating after
* Part 3  Green Home Score      - score, components, key issues
* Part 4  Property value        - value impact, comparable greener homes
* Part 5  Feasibility & LLM     - rules/Jev feasibility, routing, chat, validation, memory
* Orchestration                 - ScenarioSpec -> ScenarioResult, dashboard areas

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
* Missing values are ``None``, never 0 (0 only for a true zero, e.g. no gas supply).
* Fields marked ``leakage`` in their description must never be model inputs.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "0.4.0"

EPCBand = Literal["A", "B", "C", "D", "E", "F", "G"]
ScoreBand = Literal["Highly efficient", "Good", "Moderate", "Poor", "Very poor"]


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

    engine: Literal[
        "physics_l1", "bayes_calibration", "surrogate_deeponet", "sap_estimator", "ebm", "xgboost", "gnn",
        "bayesian_hedonic", "hedonic", "dr_learner", "rules", "jev", "claude", "haiku_vision", "template", "mock",
    ]
    model_version: str
    data_version: str
    seed: Optional[int] = None
    n_samples: Optional[int] = Field(None, description="Posterior samples used")


class FeatureContribution(BaseModel):
    """One additive term of a glass-box model (EBM shape function or hedonic coefficient x value)."""

    feature: str
    value: Optional[float | str] = Field(None, description="The input value for this home")
    contribution: float = Field(description="Additive contribution in the model's output unit")
    unit: str


# =========================================================================== #
# Part 0 - Data layer (inputs, public datasets, labels)
# =========================================================================== #
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
    lsoa_code: Optional[str] = Field(None, description="LSOA 2021 code (ONS postcode directory)")
    local_authority: Optional[str] = Field(None, description="Local authority name")
    local_authority_code: Optional[str] = Field(None, description="ONS LA code, e.g. 'E08000003'; joins EPC live tables")
    region: str = Field(description="ONS region, e.g. 'North West'")
    lat: Optional[float] = Field(None, description="Rounded to 3 dp (~100 m) for privacy")
    lon: Optional[float] = None
    altitude_m: Optional[float] = None


class EPCInfo(BaseModel):
    current_band: Optional[EPCBand] = None
    sap_score: Optional[int] = Field(None, ge=1, le=100)
    lodgement_date: Optional[str] = Field(None, description="ISO date")
    potential_band: Optional[str] = Field(None, description="leakage: reference label only, never a model input")
    potential_sap_score: Optional[int] = Field(None, description="leakage: reference label only")
    estimated_annual_energy_cost_gbp: Optional[float] = Field(None, description="EPC modelled current cost")
    energy_consumption_kwh_m2: Optional[float] = Field(
        None, description="EPC modelled primary energy (kWh/m2/yr); validation target for the twin's standard mode"
    )


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
    low_energy_lighting_pct: Optional[float] = Field(None, ge=0, le=100, description="EPC LOW_ENERGY_LIGHTING")
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
    """Historical transaction (UCL Price Paid-EPC linked data). Never used as the baseline value."""

    property_id: str
    date: str
    price_gbp: float
    energy_rating_at_sale: Optional[str] = None
    sap_at_sale: Optional[int] = None
    new_build: Optional[bool] = None
    tenure: Optional[Literal["freehold", "leasehold"]] = None
    split: Optional[Literal["train", "val", "test"]] = Field(None, description="By property, assigned once")


class WeatherMonth(BaseModel):
    area_code: str = Field(description="Postcode district, LSOA or region code")
    month: str
    hdd_15_5: float = Field(description="Heating degree days, base 15.5 C")
    cdd_22: Optional[float] = None
    mean_temp_c: Optional[float] = None
    ghi_kwh_m2: Optional[float] = Field(None, description="Global horizontal irradiation")
    source: Literal["haduk_grid", "open_meteo_era5", "ukcp18", "met_office_datahub", "synthetic"]


class DatasetMeta(BaseModel):
    """Sidecar ``<file>.meta.json`` written next to every processed Parquet file."""

    dataset_id: str = Field(description="e.g. 'epc_certificates', 'need_50k', 'epc_live_d1'")
    title: str
    publisher: str
    url: str
    licence: str
    accessed: str = Field(description="ISO date downloaded")
    release: Optional[str] = Field(None, description="Publisher's release/version, e.g. 'data to June 2026'")
    rows: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    filters: list[str] = Field(default_factory=list, description="e.g. ['regions: NW, SW, London']")
    notes: list[str] = Field(default_factory=list, description="Caveats: suppression, opt-outs, revisions")


class EPCRecommendation(BaseModel):
    """One assessor recommendation from the EPC recommendations table. Label for recommendation checks."""

    property_id: str
    measure_code: str = Field(description="EPC improvement id, mapped to a catalogue id where possible")
    catalogue_id: Optional[str] = None
    order: Optional[int] = None
    indicative_cost_text: Optional[str] = None
    typical_saving_text: Optional[str] = None


class AreaEPCStats(BaseModel):
    """One row of the GOV.UK 'Live tables on Energy Performance of Buildings Certificates' (MHCLG).

    Counts of EPCs LODGED in a period (not the dwelling stock): tables D1 (bands), D3 (floor area, modelled
    energy, CO2, fuel cost), EB4 (property type). Used for population calibration targets, download/mapper
    validation, twin standard-mode validation, peer context and dashboard layers.
    """

    area_code: str = Field(description="LA code, region name or 'E+W'")
    area_type: Literal["local_authority", "region", "england_wales", "england", "wales"]
    period: str = Field(description="'YYYY/Q' quarter or 'YYYY' year")
    dwellings: Literal["all", "existing", "new"] = "all"
    lodgements: int
    total_floor_area_m2: Optional[float] = None
    band_counts: dict[str, int] = Field(default_factory=dict, description="A..G + 'not_recorded' (table D1/EB1)")
    mean_energy_kwh_m2: Optional[float] = Field(None, description="EPC modelled energy (table D3/EB3)")
    mean_energy_kwh_per_dwelling: Optional[float] = None
    mean_co2_t_per_dwelling: Optional[float] = None
    mean_fuel_cost_gbp: Optional[float] = None
    property_type_counts: dict[str, int] = Field(default_factory=dict, description="table EB4/NB4")
    source_tables: list[str] = Field(default_factory=list, description="e.g. ['D1', 'D3']")
    release: str = Field(description="e.g. 'data to June 2026 (30 July 2026)'")


class LSOAConsumption(BaseModel):
    """DESNZ sub-national domestic consumption by LSOA (real, annual). Label for the GNN area model."""

    lsoa_code: str
    year: int
    fuel: Literal["electricity", "gas"]
    meters: Optional[int] = None
    mean_kwh: Optional[float] = Field(None, description="Null when suppressed")
    median_kwh: Optional[float] = None


class LoadProfileArchetype(BaseModel):
    """Half-hourly electricity load shape clustered from London smart-meter data."""

    archetype_id: str
    season: Literal["winter", "spring", "summer", "autumn"]
    day_type: Literal["weekday", "weekend"]
    half_hourly_share: list[float] = Field(description="48 values summing to 1 (share of daily kWh)")
    n_households: int
    source: str = "London Datastore: Smart Meter Energy Consumption Data in London Households (LCL)"

    @model_validator(mode="after")
    def _shape(self) -> "LoadProfileArchetype":
        if len(self.half_hourly_share) != 48 or abs(sum(self.half_hourly_share) - 1) > 1e-6:
            raise ValueError("half_hourly_share needs 48 values summing to 1")
        return self


class ReferenceLabel(BaseModel):
    """Evaluation label (PS section 9). Synthetic labels name the generator rule that made them."""

    property_id: str
    label_type: Literal[
        "per_fuel_saving", "package_saving", "improvement_potential", "feasibility", "relevance",
        "value_impact_method_check", "rating_change",
    ]
    retrofit_ids: list[str] = Field(default_factory=list)
    fuel: Optional[Literal["electricity", "gas", "purchased"]] = None
    value: Optional[float | str] = None
    unit: Optional[str] = None
    origin: Literal["synthetic_generator", "epc_recommendation", "epc_history", "rules", "observed"]
    rule_id: Optional[str] = Field(None, description="Generator/rule that produced it (documented)")
    generator_version: Optional[str] = None


# =========================================================================== #
# Part 1 - Digital twin (physics, Bayesian calibration, 3D scene, photo capture)
# =========================================================================== #
class TwinParams(BaseModel):
    """Physics parameters of one home for the Level-1 twin (one value per parameter, i.e. one sample).

    Built by ``params_from_property`` from RdSAP defaults (+ room geometry when captured); retrofits are edits
    to these fields (``RetrofitItem.twin_edits``). Calibration multiplies/overrides the uncertain ones.
    """

    property_id: str
    wall_area_m2: float
    roof_area_m2: float
    floor_area_m2: float
    window_area_m2: float
    door_area_m2: float = 3.7
    party_wall_area_m2: float = 0.0
    wall_u: float
    roof_u: float
    floor_u: float
    window_u: float
    door_u: float = 3.0
    thermal_bridging_y: float = Field(0.15, description="W/m2K added over exposed area")
    air_changes_per_hour: float
    volume_m3: float
    heat_loss_multiplier: float = Field(1.0, description="Calibrated: scales total fabric + ventilation loss")
    setpoint_c: float = 19.5
    heating_hours_per_day: float = 9.0
    heating_efficiency: float = Field(description="Boiler seasonal efficiency or heat-pump SCOP")
    heating_fuel: Literal["mains_gas", "electricity", "oil", "lpg", "solid", "district"]
    hot_water_kwh_per_day: float
    baseload_kwh_per_day: float = Field(description="Appliances + lighting + cooking electricity")
    lighting_share_of_baseload: float = 0.15
    internal_gains_w: float = 400.0
    solar_gain_factor: float = 1.0
    pv_kwp: float = 0.0
    pv_azimuth_deg: float = 180.0
    pv_tilt_deg: float = 35.0
    battery_kwh: float = 0.0
    tariff: Literal["standard", "time_of_use"] = "standard"


class CalibrationDiagnostics(BaseModel):
    max_r_hat: Optional[float] = Field(None, description="NUTS only; < 1.01 required")
    min_ess: Optional[float] = None
    elbo: Optional[float] = Field(None, description="SVI only")
    divergences: Optional[int] = None
    posterior_predictive_coverage_90: Optional[float] = Field(None, description="Share of observed months in 90% PI")


class CalibrationResult(BaseModel):
    """Bayesian calibration of one home's twin on its meter readings (summary; samples stored separately)."""

    property_id: str
    method: Literal["nuts", "svi", "prior_only"] = Field(description="prior_only when no meter data exists")
    n_samples: int
    months_used: int
    months_imputed: list[str] = Field(default_factory=list, description="Missing months treated as latent")
    parameters: dict[str, Distribution] = Field(
        description="Posterior of the calibrated parameters: heat_loss_multiplier, setpoint_c, baseload, efficiency"
    )
    diagnostics: CalibrationDiagnostics = Field(default_factory=CalibrationDiagnostics)
    data_confidence: Literal["high", "medium", "low"]
    posterior_ref: str = Field(description="Where the samples live, e.g. 'duckdb:posteriors/SH2/v3'")
    prior_source: Literal["ebm_prior", "rdsap_defaults", "hierarchy"] = "ebm_prior"
    provenance: Provenance


class ElementHeatLoss(BaseModel):
    """Level 1 heat loss per envelope element; drives 3D colouring."""

    element: Literal["roof", "walls", "floor", "windows", "doors", "ventilation", "party_walls"]
    heat_loss_w_per_k: float = Field(description="Posterior mean")
    heat_loss_distribution: Optional[Distribution] = None
    share_pct: float
    u_value: Optional[float] = None
    area_m2: Optional[float] = None
    estimated: bool = Field(True, description="Per-element split comes from priors, not measurement")
    basis: str = Field(
        "Estimated from home type and EPC descriptions; only the total heat loss is calibrated to bills",
        description="Label shown wherever element colours/values are displayed",
    )


class Room(BaseModel):
    room_id: str
    label: str = Field(description="e.g. 'living room', 'kitchen', 'bedroom 1', 'loft'")
    storey: int = 0
    polygon_m: list[tuple[float, float]] = Field(description="Floor polygon in metres, house-local coordinates")
    area_m2: float
    ceiling_height_m: Optional[float] = None
    confidence: float = Field(1.0, ge=0, le=1, description="1.0 once confirmed by the agent")


class FloorPlan(BaseModel):
    property_id: str
    rooms: list[Room]
    total_area_m2: float
    area_vs_epc_pct: Optional[float] = Field(None, description="(plan - EPC floor area) / EPC floor area x 100")
    exterior_wall_length_m: Optional[float] = None
    window_area_m2: Optional[float] = None
    source: Literal["haiku_vision", "agent_drawn", "parametric"]
    confirmed_by_agent: bool = False


class ElementTag(BaseModel):
    """One element observed in a photo (vision draft or manual tag)."""

    photo_id: str
    room_id: Optional[str] = None
    field: str = Field(description="Property field it fills, e.g. 'glazing.type', 'heating.main_system', "
                                   "'roof.loft_insulation_mm'")
    value: float | str
    confidence: float = Field(ge=0, le=1)
    source: Literal["haiku_vision", "manual"]
    confirmed_by_agent: bool = False


class CaptureResult(BaseModel):
    """Photo-to-twin output: drafts only. Nothing reaches the twin until the agent confirms it."""

    property_id: str
    floor_plan: Optional[FloorPlan] = None
    tags: list[ElementTag] = Field(default_factory=list)
    draft_property_fields: dict[str, float | str] = Field(default_factory=dict)
    needs_confirmation: list[str] = Field(default_factory=list, description="Fields below the confidence threshold")
    vision_model: Optional[str] = Field("claude-haiku-5-5", description="None for manual capture")
    consent_recorded: bool = False
    faces_blurred: bool = True


class SceneSurface(BaseModel):
    """One renderable building surface for the 3D twin."""

    surface_id: str
    element: Literal["roof", "walls", "floor", "windows", "doors", "party_walls"]
    orientation_deg: Optional[float] = Field(None, description="Outward normal azimuth, 180 = south")
    area_m2: float
    u_value: float
    heat_loss_share_pct: float = Field(description="Posterior mean share of total heat loss; drives colour")
    heat_loss_share: Optional[Distribution] = None
    estimated: bool = True


class RetrofitSlot(BaseModel):
    """Where a retrofit object appears in the 3D scene."""

    retrofit_id: str
    anchor: Literal["roof_face", "outside_wall", "loft", "wall_inner", "windows", "plant_cupboard", "floor", "room"]
    surface_id: Optional[str] = None
    position_m: Optional[tuple[float, float, float]] = None
    status: Literal["feasible", "feasible_with_conditions", "needs_survey", "infeasible"]
    reason: Optional[str] = None


class TwinScene(BaseModel):
    """Everything the 3D viewer needs. The scene never calculates: numbers come from ScenarioResult."""

    property_id: str
    scenario_id: Optional[str] = Field(None, description="Set when the scene shows a retrofit package")
    geometry_source: Literal["parametric", "floor_plan"]
    property_type: PropertyType
    built_form: BuiltForm
    storeys: int
    footprint_m: tuple[float, float] = Field(description="(width, depth) of the parametric box")
    roof_type: RoofType
    roof_pitch_deg: Optional[float] = None
    surfaces: list[SceneSurface]
    rooms: list[Room] = Field(default_factory=list)
    retrofit_slots: list[RetrofitSlot] = Field(default_factory=list)
    heat_loss_label: str = "Estimated from home type and EPC descriptions; only the total heat loss is calibrated to bills"


# =========================================================================== #
# Part 2 - Energy & retrofit engine
# =========================================================================== #
class EnergyPrior(BaseModel):
    """EBM energy-use model output (trained on NEED): the prior for homes without meter data."""

    property_id: str
    electricity_kwh_m2: Distribution
    gas_kwh_m2: Distribution
    contributions_electricity: list[FeatureContribution] = Field(default_factory=list)
    contributions_gas: list[FeatureContribution] = Field(default_factory=list)
    out_of_distribution: bool = False
    provenance: Provenance


class MonthlyEnergy(BaseModel):
    month: str = Field(description="YYYY-MM")
    electricity_import_kwh: Distribution
    gas_kwh: Distribution
    imputed: bool = Field(False, description="True when the meter reading was missing")


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
    monthly: list[MonthlyEnergy] = Field(default_factory=list, description="Optional monthly series")


class ElectricityProfileSummary(BaseModel):
    """Half-hourly electricity layer summary (smart-meter load shapes + PV + battery + tariff)."""

    self_consumption_pct: Optional[Distribution] = None
    export_kwh: Optional[Distribution] = None
    peak_share_pct: Optional[Distribution] = Field(None, description="Share of import in 16:00-19:00")
    bill_standard_gbp: Distribution
    bill_time_of_use_gbp: Optional[Distribution] = Field(None, description="Null until a ToU tariff is sourced")
    load_archetype_id: Optional[str] = None


class RetrofitEffect(BaseModel):
    """One step of the improvement ladder: the marginal effect of adding this retrofit after the earlier steps."""

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


class ImprovementPotential(BaseModel):
    """PS section 3: 100 x (E_base - E_after) / E_base on purchased energy, with ALL feasible retrofits run together."""

    pct: Optional[Distribution] = Field(None, description="Null when the baseline is 0 or unavailable")
    unavailable_reason: Optional[str] = None
    electricity_change_kwh: Distribution = Field(description="baseline - after; negative = increase")
    gas_change_kwh: Distribution
    included_retrofits: list[str] = Field(default_factory=list, description="feasible + feasible_with_conditions")
    excluded_needs_survey: list[str] = Field(default_factory=list)
    pct_if_surveys_pass: Optional[Distribution] = None
    p_top_decile: Optional[float] = Field(None, ge=0, le=1, description="P(home in top 10% by potential), for ranking")


class RatingEstimate(BaseModel):
    """Post-retrofit EPC-style rating from contracts/sap_estimator.json. Never presented as an EPC."""

    sap: Distribution
    band: EPCBand = Field(description="Mode of band_probabilities")
    band_probabilities: dict[str, float] = Field(default_factory=dict)
    anchored_to_recorded_sap: bool = Field(description="False when the home has no recorded EPC")
    method: Literal["sap2012_ecf_delta"] = "sap2012_ecf_delta"
    label: Literal["estimated rating, not an EPC"] = "estimated rating, not an EPC"


class ParetoPoint(BaseModel):
    package: list[str]
    net_cost_gbp: Distribution
    carbon_saving_kgco2e: Distribution
    bill_saving_gbp: Distribution
    score_gain: Distribution
    p_on_front: float
    is_knee: bool = False


# =========================================================================== #
# Part 3 - Green Home Score
# =========================================================================== #
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


class KeyIssue(BaseModel):
    """PS section 3: why a home performs poorly, with the evidence behind it."""

    code: Literal[
        "high_intensity", "poor_insulation", "old_heating", "single_glazing", "no_renewables",
        "no_controls", "data_conflict", "missing_data",
    ]
    label: str = Field(description="e.g. 'High energy use per m2 for a home like this'")
    severity: Literal["high", "medium", "low"]
    evidence: dict[str, float | str] = Field(default_factory=dict, description="e.g. {'kwh_m2': 231, 'peer_p50': 165}")
    probability: Optional[float] = Field(None, ge=0, le=1, description="e.g. P(intensity above peers)")


class AlternativeScore(BaseModel):
    """The same seven sub-scores combined with different weights (e.g. weights learned from data).

    Reported beside the headline score, which always uses the brief's fixed weights.
    """

    name: Literal["learned_weights"] = "learned_weights"
    weights: dict[str, float] = Field(description="Component -> weight; sums to 1")
    score: int = Field(ge=0, le=100)
    band: ScoreBand
    score_distribution: Optional[Distribution] = None
    weights_version: str = Field(description="e.g. 'artifacts/score_weights/v1'")
    method: str = Field(description="How the weights were learned (documented in docs/score)")


class ScoreResult(BaseModel):
    score: int = Field(ge=0, le=100, description="Rounded weighted sum (of the posterior mean inputs)")
    score_distribution: Optional[Distribution] = Field(None, description="Score over posterior samples")
    band: ScoreBand = Field(description="Classification of the rounded score (brief section 2); "
                                        "band_probabilities gives the spread over posterior samples")
    band_probabilities: dict[str, float] = Field(default_factory=dict, description="P(each band); sums to 1")
    p_poor_or_very_poor: Optional[float] = Field(None, ge=0, le=1)
    components: list[ScoreComponent]
    key_issues: list[KeyIssue] = Field(default_factory=list)
    projected: bool = Field(False, description="True for a post-retrofit score (uses RatingEstimate)")
    rating_source: Literal["recorded_epc", "estimated", "missing"] = "recorded_epc"
    peer_context: dict[str, float | str] = Field(
        default_factory=dict, description="e.g. local-authority band shares from EPC live table D1"
    )
    mapping_version: str = Field("score_mapping.json", description="Version of contracts/score_mapping.json used")
    alternatives: list[AlternativeScore] = Field(
        default_factory=list, description="Companion scores with other weights (never replaces the headline score)"
    )


# =========================================================================== #
# Part 4 - Property value
# =========================================================================== #
class ValueImpact(BaseModel):
    current_value_gbp: Optional[float] = Field(description="Baseline valuation (never a historical sale price)")
    current_value_distribution: Optional[Distribution] = Field(
        None, description="Set when the current value is model-estimated rather than provided"
    )
    band_before: Optional[str] = Field(None, description="Recorded EPC band")
    band_after: Optional[str] = Field(None, description="From RatingEstimate (sap_estimator), not a lodged EPC")
    uplift_pct: Optional[Distribution] = None
    value_after_gbp: Optional[Distribution] = None
    rating_effect_pct_per_sap_point: Optional[Distribution] = Field(None, description="Posterior of beta for this region x type")
    price_drivers: list[FeatureContribution] = Field(default_factory=list, description="Why the home is worth what it is")
    method: Literal["bayesian_hedonic", "hedonic_quantile", "dr_learner", "green_twin_matching", "unavailable"]
    label: Literal["indicative"] = "indicative"
    wording: str = Field(
        "Indicative range based on historical associations in sales data; not a promised or proven increase.",
        description="Must accompany every display of the value range",
    )
    unavailable_reason: Optional[str] = None


class GreenTwin(BaseModel):
    """A matched greener comparable home, shown only as ranges (never an address)."""

    match_rank: int
    score_gap: float = Field(description="Green Home Score points greener than this home")
    price_per_m2_gbp: Distribution
    energy_cost_gbp: Optional[Distribution] = None
    epc_band: Optional[str] = None
    same: list[str] = Field(default_factory=list, description="Matched on, e.g. ['type', 'age band', 'LSOA']")


# =========================================================================== #
# Part 5 - Feasibility, routing, LLM explanation and chat, memory
# =========================================================================== #
class Feasibility(BaseModel):
    retrofit_id: str
    status: Literal["feasible", "feasible_with_conditions", "needs_survey", "infeasible"]
    decided_by: Literal["rules", "jev", "claude_fallback", "agent"]
    reason_codes: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list, description="Human-readable, from rules/KB (never the LLM)")
    p_feasible: float = Field(ge=0, le=1, description="1.0/0.0 for rules; recalibrated confidence for Jev")


class FeasibilityReport(BaseModel):
    property_id: str
    items: list[Feasibility]
    rules_version: str
    jev_used: bool = False
    router: Literal["jev", "claude_fallback", "rules_only"] = "rules_only"


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


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Citation(BaseModel):
    source: Literal["tool_result", "doc", "rule", "dataset"]
    ref: str = Field(description="scenario_id, doc id, rule id or dataset id")
    snippet: Optional[str] = None


class ToolCallRecord(BaseModel):
    name: Literal["get_home_facts", "get_results", "run_scenario", "search_docs", "query_graph"]
    input: dict
    output_ref: Optional[str] = Field(None, description="e.g. the scenario_id a run_scenario call produced")


class ChatRequest(BaseModel):
    property_id: str
    customer_id: Optional[str] = None
    message: str
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    route: Optional[WhatIfRoute] = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    check: ExplanationCheck
    scenario_result_id: Optional[str] = None
    fallback: Literal["none", "regenerated", "template"] = "none"
    model: Optional[str] = Field(None, description="LLM model id, or None for the template explainer")


class CustomerMemory(BaseModel):
    customer_id: str
    facts: list[dict[str, str]] = Field(
        default_factory=list, description="e.g. {'type':'constraint','value':'renting'}"
    )
    completed_retrofits: list[RetrofitAction] = Field(default_factory=list)
    rejected: list[dict[str, str]] = Field(default_factory=list, description="{'retrofit_id':..,'reason':..}")
    saved_scenarios: list[str] = Field(default_factory=list)


# =========================================================================== #
# Retrofit catalogue (shared config, validated by this model)
# =========================================================================== #
class RetrofitItem(BaseModel):
    id: str
    name: str
    category: Literal["fabric", "glazing", "heating", "renewables", "controls", "airtightness", "lighting", "tariff"]
    twin_edits: dict[str, float | str] = Field(
        description="TwinParams edits: a number sets the field; a string 'x<k>' multiplies it (e.g. 'x0.25'); "
                    "other strings set a categorical field (e.g. tariff)"
    )
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
    changes_kwh: bool = Field(True, description="False for tariff switches: bills change, energy does not")
    source: str = Field(description="Citation for cost/impact assumptions")


# =========================================================================== #
# Orchestration - one result per what-if, and dashboard areas
# =========================================================================== #
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
    include_electricity_profile: bool = False
    seed: int = 42
    goal: Literal["bills", "carbon"] = Field(
        "bills", description="Defines 'net benefit' for suitability/verdicts; never changes computed numbers"
    )
    requested_by: Literal["ui", "llm", "batch", "jev_router", "eval"] = "ui"


class ScenarioResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    scenario_id: str
    property_id: str
    provenance: Provenance
    goal: Literal["bills", "carbon"] = "bills"
    calibration: Optional[CalibrationResult] = None
    baseline: EnergyBreakdown
    after: EnergyBreakdown
    electricity_saving_kwh: Distribution = Field(description="baseline - after; negative = increase")
    gas_saving_kwh: Distribution
    purchased_energy_saving_kwh: Distribution
    improvement_potential: ImprovementPotential
    carbon_saving_kgco2e: Optional[Distribution] = Field(None, description="Null if a fuel has no sourced factor")
    carbon_unavailable_reason: Optional[str] = None
    emission_factors: dict[str, float] = Field(
        description="kgCO2e/kWh from contracts/emission_factors.json, e.g. {'electricity':0.19553,'gas':0.18296}"
    )
    emission_factor_source: str
    bill_saving_gbp: Distribution
    electricity_profile_after: Optional[ElectricityProfileSummary] = None
    score_before: ScoreResult
    score_after: ScoreResult = Field(description="Uses rating_after (estimated) for the energy-efficiency component")
    rating_after: Optional[RatingEstimate] = None
    feasibility: Optional[FeasibilityReport] = None
    retrofits: list[RetrofitEffect] = Field(default_factory=list, description="The improvement ladder, in order")
    value: ValueImpact
    element_heat_loss_before: list[ElementHeatLoss] = Field(default_factory=list)
    element_heat_loss_after: list[ElementHeatLoss] = Field(default_factory=list)
    pareto: list[ParetoPoint] = Field(default_factory=list)
    data_confidence: Literal["high", "medium", "low"] = "medium"
    warnings: list[str] = Field(default_factory=list, description="Missing/conflicting data, OOD, assumptions")


class DashboardArea(BaseModel):
    """One area on the Lloyds heat map (LSOA or local authority). Synthetic customer layers are labelled."""

    area_code: str
    area_type: Literal["lsoa", "local_authority", "postcode_district"]
    n_homes_modelled: int
    share_green: Optional[Distribution] = Field(None, description="Share with score >= 65")
    share_poor: Optional[Distribution] = None
    finance_demand_gbp: Optional[Distribution] = None
    carbon_potential_t: Optional[Distribution] = None
    gnn_expected_electricity_kwh: Optional[Distribution] = None
    gnn_expected_gas_kwh: Optional[Distribution] = None
    observed: Optional[LSOAConsumption] = None
    epc_stats: Optional[AreaEPCStats] = Field(None, description="Official EPC live-table figures (LA level)")
    off_gas_grid_share: Optional[float] = None
    synthetic_customers: Optional[int] = Field(None, description="Synthetic customer base; always labelled")


# Resolve forward references (models are grouped by part, not by dependency order).
for _m in list(globals().values()):
    if isinstance(_m, type) and issubclass(_m, BaseModel) and _m is not BaseModel:
        _m.model_rebuild()
