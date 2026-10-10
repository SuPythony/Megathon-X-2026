// GreenTwin shared data contracts — TypeScript mirror of contracts/models.py.
// Keep in sync with models.py in the SAME commit. Units: kWh/yr, GBP, kgCO2e, m2, W/m2K, W/K.
// Savings = baseline - after (positive = saving, negative = increase).

export const SCHEMA_VERSION = "0.2.0";

export interface Interval { mean: number; lo: number; hi: number; level: number; unit: string; }

export interface Provenance {
  engine: "physics_l1" | "surrogate_deeponet" | "sap_estimator" | "ebm" | "xgboost" | "hedonic" | "dr_learner" | "rules" | "mock";
  model_version: string; data_version: string; seed?: number | null; n_samples?: number | null;
}

export type PropertyType = "house" | "bungalow" | "flat" | "maisonette";
export type BuiltForm = "detached" | "semi_detached" | "mid_terrace" | "end_terrace" | "enclosed_mid_terrace" | "not_applicable";
export type WallType = "solid_brick" | "solid_stone" | "cavity" | "timber_frame" | "system_built";
export type WallInsulation = "none" | "cavity_filled" | "internal" | "external" | "as_built";
export type RoofType = "pitched_loft" | "pitched_room_in_roof" | "flat" | "another_dwelling_above";
export type FloorType = "suspended_timber" | "solid" | "another_dwelling_below";
export type GlazingType = "single" | "secondary" | "double_pre_2002" | "double_post_2002" | "triple";
export type HeatingSystem =
  | "gas_boiler_condensing" | "gas_boiler_non_condensing" | "oil_boiler" | "lpg_boiler"
  | "electric_storage" | "electric_storage_hhr" | "electric_direct"
  | "air_source_heat_pump" | "ground_source_heat_pump" | "district_heating" | "communal_boiler" | "solid_fuel";
export type Control = "none" | "programmer" | "room_thermostat" | "trvs" | "smart_thermostat" | "zoned_control" | "smart_meter";
export type Tenure = "owner_occupied" | "private_rented" | "social_rented" | "unknown";

export interface Location {
  postcode_district: string; lsoa_code?: string | null; local_authority?: string | null; region: string;
  lat?: number | null; lon?: number | null; altitude_m?: number | null;
}
export interface EPCInfo {
  current_band?: "A" | "B" | "C" | "D" | "E" | "F" | "G" | null; sap_score?: number | null; lodgement_date?: string | null;
  potential_band?: string | null; potential_sap_score?: number | null; // leakage: labels only
  estimated_annual_energy_cost_gbp?: number | null;
}
export interface Walls { type: WallType; insulation: WallInsulation; u_value?: number | null; }
export interface Roof { type: RoofType; loft_insulation_mm?: number | null; u_value?: number | null; }
export interface Floor { type: FloorType; insulated: boolean; u_value?: number | null; }
export interface Glazing { type: GlazingType; share_pct: number; window_to_floor_ratio?: number | null; }
export interface Heating {
  main_system: HeatingSystem; main_fuel: "mains_gas" | "electricity" | "oil" | "lpg" | "solid" | "district";
  system_age_years?: number | null; seasonal_efficiency?: number | null;
  hot_water: "from_main" | "electric_immersion" | "heat_pump_cylinder" | "instantaneous"; cylinder_insulated?: boolean | null;
}
export interface Renewables { pv_kwp: number; pv_azimuth_deg?: number | null; pv_tilt_deg?: number | null; battery_kwh: number; solar_thermal: boolean; }
export interface Constraints {
  conservation_area: boolean; listed_building: boolean; leasehold: boolean; has_outdoor_space?: boolean | null;
  roof_access?: boolean | null; best_roof_azimuth_deg?: number | null; off_gas_grid: boolean; floor_level?: number | null;
  free_text_notes: string[];
}
export interface DataQuality { missing_fields: string[]; imputed_fields: string[]; conflicts: string[]; months_of_metered_data: number; }

export interface Property {
  schema_version: string; property_id: string; customer_id?: string | null;
  source: "epc_real" | "synthetic" | "showcase" | "agent_capture" | "self_serve";
  split?: "train" | "val" | "test" | null; location: Location; property_type: PropertyType; built_form: BuiltForm;
  construction_age_band: string; year_built?: number | null; storeys: number; floor_area_m2: number;
  habitable_rooms?: number | null; bedrooms?: number | null; tenure: Tenure; epc: EPCInfo; walls: Walls; roof: Roof;
  floor: Floor; glazing: Glazing[]; draught_proofed_pct?: number | null; heating: Heating; renewables: Renewables;
  controls: Control[]; constraints: Constraints; current_value_estimate_gbp?: number | null;
  value_estimate_date?: string | null; data_quality: DataQuality;
}

export interface EnergyUsageMonth {
  property_id: string; month: string; electricity_import_kwh: number | null; gas_kwh: number | null;
  pv_generation_kwh?: number | null; electricity_export_kwh?: number | null; estimated_read: boolean;
}

export interface RetrofitAction { retrofit_id: string; size?: number | null; stage_year?: number | null; }
export interface ScenarioSpec {
  schema_version: string; scenario_id: string; property_id: string; retrofits: RetrofitAction[];
  weather: "typical" | "cold_year" | "warm_year" | "forecast_week" | "ukcp18_2050";
  n_samples: number; include_value: boolean; include_pareto: boolean; include_element_heat_loss: boolean; seed: number;
  requested_by: "ui" | "llm" | "batch" | "jev_router" | "eval";
}

export interface EnergyBreakdown {
  electricity_import_kwh: Interval; gas_kwh: Interval; purchased_total_kwh: Interval;
  space_heating_kwh?: Interval | null; hot_water_kwh?: Interval | null; pv_generation_kwh?: Interval | null;
  intensity_kwh_m2: Interval; // annualised METERED (import + gas) / m2 — used by the score
  intensity_weather_normalised_kwh_m2?: Interval | null; // sensitivity only
}
export type ScoreComponentName =
  | "energy_efficiency" | "energy_intensity" | "insulation" | "heating_system"
  | "windows_glazing" | "renewable_energy" | "smart_energy_management";
export interface ScoreComponent { name: ScoreComponentName; weight: number; sub_score: number; contribution: number; inputs_used: Record<string, number | string | null>; }
export interface ScoreResult {
  score: number; score_interval?: Interval | null; band: "Highly efficient" | "Good" | "Moderate" | "Poor" | "Very poor";
  band_probabilities: Record<string, number>; components: ScoreComponent[]; key_issues: string[];
}
export interface Feasibility {
  retrofit_id: string; status: "feasible" | "feasible_with_conditions" | "needs_survey" | "infeasible";
  decided_by: "rules" | "jev" | "claude_fallback" | "agent"; reason_codes: string[]; p_feasible: number;
}
export interface RetrofitEffect {
  retrofit_id: string; order: number; feasibility: Feasibility; electricity_saving_kwh: Interval; gas_saving_kwh: Interval;
  carbon_saving_kgco2e: Interval; bill_saving_gbp: Interval; cost_gbp?: Interval | null; cost_unavailable_reason?: string | null; p_net_benefit: number; p_suitable: number;
  suitability: "High" | "Medium" | "Low"; reason: string;
}
export interface RatingEstimate {
  sap: Interval; band: "A" | "B" | "C" | "D" | "E" | "F" | "G"; band_probabilities: Record<string, number>;
  anchored_to_recorded_sap: boolean; method: "sap2012_ecf_delta"; label: "estimated rating, not an EPC";
}
export interface ValueImpact {
  current_value_gbp: number | null; band_before?: string | null; band_after?: string | null; uplift_pct?: Interval | null; value_after_gbp?: Interval | null;
  method: "hedonic_quantile" | "dr_learner" | "green_twin_matching" | "unavailable"; label: "indicative";
  wording: string; // always displayed with the range
  unavailable_reason?: string | null;
}
export interface ElementHeatLoss {
  element: "roof" | "walls" | "floor" | "windows" | "doors" | "ventilation" | "party_walls";
  heat_loss_w_per_k: number; share_pct: number; u_value?: number | null; area_m2?: number | null;
  estimated: boolean; basis: string; // show basis label wherever element colours are displayed
}
export interface ParetoPoint {
  package: string[]; net_cost_gbp: Interval; carbon_saving_kgco2e: Interval; bill_saving_gbp: Interval; score_gain: Interval;
  p_on_front: number; is_knee: boolean;
}
export interface ScenarioResult {
  schema_version: string; scenario_id: string; property_id: string; provenance: Provenance;
  baseline: EnergyBreakdown; after: EnergyBreakdown;
  electricity_saving_kwh: Interval; gas_saving_kwh: Interval; purchased_energy_saving_kwh: Interval;
  improvement_potential_pct?: Interval | null; improvement_potential_unavailable_reason?: string | null;
  carbon_saving_kgco2e?: Interval | null; carbon_unavailable_reason?: string | null;
  emission_factors: Record<string, number>; emission_factor_source: string;
  bill_saving_gbp: Interval; score_before: ScoreResult; score_after: ScoreResult; rating_after?: RatingEstimate | null;
  retrofits: RetrofitEffect[];
  value: ValueImpact; element_heat_loss_before: ElementHeatLoss[]; element_heat_loss_after: ElementHeatLoss[];
  pareto: ParetoPoint[]; data_confidence: "high" | "medium" | "low"; warnings: string[];
}

export interface WhatIfRoute {
  intent: "numeric" | "comparative" | "feasibility" | "explain" | "out_of_scope"; confidence: number;
  scenario?: ScenarioSpec | null; needs_clarification: boolean; clarifying_question?: string | null; router: "jev" | "claude_fallback";
}
