# GreenTwin — Low-Level Design

**Megathon X: Singularity · Lloyds Problem Statement 2 · Team MegaMind · schema {{SCHEMA_VERSION}} · built {{BUILD_DATE}}**

**Audience:** engineers building GreenTwin, and AI coding agents working in the repo. This document is the build reference. It names the contracts, interfaces, algorithms, data and configuration each part implements. Sections 5 and 6 (contracts and interfaces) are **generated from the code** (`contracts/models.py`, `greentwin/interfaces.py`), so they always match it. The high-level view is in `SYSTEM_DESIGN.pdf`. Binding rules are in `AGENTS.md` and `contracts/`, which win if anything here disagrees.

---

## 1. The parts and how they connect

| Part | Package | Responsibility | Produces (contracts) | Consumes |
| --- | --- | --- | --- | --- |
| **0 · Data layer** | `greentwin/data`, `greentwin/geo` | Download, clean, map, split and store all data; sourced factors | `Property`, `EnergyUsageMonth`, `SaleRecord`, `WeatherMonth`, `AreaEPCStats`, `LSOAConsumption`, `LoadProfileArchetype`, `EPCRecommendation`, `ReferenceLabel`, `DatasetMeta` | Public datasets, customer input |
| **1 · Digital twin** | `greentwin/twin` (JAX), `greentwin/bayes` (NumPyro), `web/twin3d` | Physics of each home, Bayesian calibration on its meters, 3D scene, photo-to-twin capture | `TwinParams`, `CalibrationResult`, `PosteriorSamples`*, `ElementHeatLoss`, `TwinScene`, `CaptureResult`, `ElectricityProfileSummary` | `Property`, usage, weather, `EnergyPrior`, load profiles |
| **2 · Energy & retrofit engine** | `greentwin/models` (EBM), `greentwin/retrofit`, `greentwin/optimise` | Energy prediction, retrofit ladder, savings, carbon, bills, improvement potential, rating after, Pareto | `EnergyPrior`, `EnergyBreakdown`, `RetrofitEffect`, `ImprovementPotential`, `RatingEstimate`, `ParetoPoint` | `PosteriorSamples`, `FeasibilityReport`, catalogue, factors, tariffs |
| **3 · Green Home Score** | `greentwin/score` | Score, band, contributions, key issues, projected score | `ScoreResult`, `KeyIssue` | `Property`, `EnergyBreakdown`, `RatingEstimate`, `AreaEPCStats` |
| **4 · Property value** | `greentwin/value` | Indicative value range before vs after; comparable greener homes | `ValueImpact`, `GreenTwin` | `Property`, `SaleRecord` (training), `RatingEstimate` |
| **5 · Feasibility, Jev, LLM** | `greentwin/retrofit/rules.py`, `greentwin/jev`, `greentwin/llm`, `greentwin/kb` | Rule vetoes + Jev feasibility; what-if routing; tool-only explanation and chat; memory | `FeasibilityReport`, `WhatIfRoute`, `ChatResponse`, `ExplanationCheck`, `CustomerMemory` | `Property`, catalogue, `ScenarioResult` (through tools) |
| **Orchestrator** | `greentwin/api` | `run_scenario`, caching, REST API, dashboard aggregation | `ScenarioResult`, `DashboardArea` | All parts through their interfaces |

\* `PosteriorSamples` is an in-process dataclass (arrays), not a wire contract. It is persisted to Parquet and summarised into `Distribution`s before leaving Python.

**Coupling.** Parts 1 and 2 form the **coupled core**: everything numeric runs through the twin. Parts 0, 3, 4 and 5 are **independent**. Each codes against its interface and a stub, and plugs into the core at four points:
1. EBM prior → calibration prior
2. feasibility list → ladder
3. estimated rating → projected score and value "after"
4. `ScenarioResult` → LLM and UI

```
            Part 0 Data ──────────────┬──────────────────────────────┐
               │ Property, usage,     │ sales, HPI                   │ EPC live tables, DESNZ LSOA
               ▼ weather              ▼                              ▼
   ┌──── Part 1 Digital twin ────┐  Part 4 Value ◄─ RatingEstimate ─┐   Part 3 Score ◄─┐
   │ physics (JAX) · calibration │                                   │                  │
   │ (NumPyro) · 3D · capture    │──PosteriorSamples──► Part 2 Energy & retrofit ───────┘
   └─────────────────────────────┘                    ▲      │ EnergyBreakdown, RetrofitEffect,
                                 Part 5 Feasibility ──┘      │ ImprovementPotential, Pareto
                                                             ▼
                                   Orchestrator run_scenario → ScenarioResult → Part 5 LLM/chat, UI
```

## 2. Conventions (all parts)

- **Units:** kWh/yr, GBP, kgCO2e, m², W/m²K, W/K. Field names carry other units explicitly (`_kwh_m2`, `_pct`, `_t`).
- **Savings = baseline − after**, per fuel. Negative = increase (electricity after a heat pump). Packages are one twin run, never a sum.
- **Uncertainty:** every uncertain output is a `Distribution`, built only with `contracts.summarise.summarise(samples, unit, tails=…, worst_tail=…)`. Use `exact()` for known values and `category_probabilities()` for bands. Fixed constants (weights, emission factors, tariffs, grants, rule vetoes) are bare numbers.
- **Missing ≠ 0:** `None` everywhere; `summarise` rejects NaN. Missing months are latent in calibration and flagged.
- **Provenance** on every result: engine, model version, data version, seed, sample count.
- **Leakage:** `epc.potential_*` are labels only. Splits are assigned once by customer, else property (`Property.split`); spatial-block splits for area models.
- **Seeds:** 42 by default; recorded in every `metrics.json`. **Schema version:** `contracts.models.SCHEMA_VERSION`, bumped on any breaking change.
- **Generated files:** `contracts/types.ts` (`python -m contracts.export_ts`) and `contracts/schema/*.json` (`python -m contracts.export_schemas`). A test fails if `types.ts` is stale.

## 3. Repository layout

```
contracts/              models.py (source of truth) · summarise.py · export_ts.py · export_schemas.py
                        types.ts (generated) · schema/ (generated) · examples/ (SH1–SH3 + mocks)
                        score_mapping.json · feasibility_rules.json · retrofit_catalogue.json
                        sap_estimator.json · emission_factors.json · tariffs.json · green_profits_assumptions.json
greentwin/interfaces.py Protocols for every part (section 6)
greentwin/data/         download_*.py, epc_mapper.py, splits.py, store.py (DuckDB), live_tables.py
greentwin/geo/          postcode→LSOA, boundaries, adjacency graph, hotspots
greentwin/twin/         physics.py (JAX), params.py (RdSAP defaults), electricity.py, scene.py, capture.py, generate.py
greentwin/bayes/        model.py (NumPyro), calibrate.py, hierarchy.py
greentwin/models/       energy_prior.py (EBM), gnn_lsoa.py (GraphSAGE)
greentwin/retrofit/     rules.py, ladder.py, potential.py, rating.py, catalogue.py
greentwin/optimise/     pareto.py
greentwin/score/        score.py, issues.py
greentwin/value/        hedonic.py (NumPyro), green_twins.py
greentwin/jev/          client.py, router.py, feasibility.py, calibration.py
greentwin/llm/          chat.py, tools.py, validator.py, template.py, prompts/
greentwin/kb/           sql.py, vector.py, graph.py
greentwin/api/          app.py (FastAPI), orchestrator.py, cache.py, dashboard.py
web/                    React app: twin3d/ (react-three-fiber), capture/ (react-konva), app/, dashboard/
scripts/                train_<model>.py --seed 42, make_showcase_examples.py, eval_all.py
data/                   raw/ interim/ processed/ (git-ignored) · greentwin.duckdb
artifacts/<model>/<ver>/ model.*, metrics.json, config.json      eval/ scenarios/, reports/
```

## 4. Part designs

### 4.0 Part 0 — Data layer

**Pipeline:** `data/raw` (as published, never edited) → `scripts/download_*.py` → `data/interim` (filtered to the North West, South West and London) → `greentwin/data/*` mappers → `data/processed/*.parquet` + `*.meta.json` (`DatasetMeta`) → `splits.py` → DuckDB `data/greentwin.duckdb`. Every other part reads only through `DataLayer` (section 6).

**EPC record → `Property` mapper** (`epc_mapper.py`). Free-text EPC descriptions become contract enums through ordered regex rules. Every unmatched value is logged, with a target of ≥ 95% mapped per field.

| EPC field | Contract field | Rule (examples) |
| --- | --- | --- |
| `PROPERTY_TYPE`, `BUILT_FORM` | `property_type`, `built_form` | House/Bungalow/Flat/Maisonette; "Mid-Terrace" → `mid_terrace` |
| `CONSTRUCTION_AGE_BAND` | `construction_age_band` | "England and Wales: 1900-1929" → `1900-1929` |
| `TOTAL_FLOOR_AREA`, `NUMBER_HABITABLE_ROOMS` | `floor_area_m2`, `habitable_rooms` | numeric; drop < 15 m² or > 1,000 m² |
| `WALLS_DESCRIPTION` | `walls.type`, `walls.insulation`, `walls.u_value` | "Cavity wall, as built, no insulation (assumed)" → cavity/none; "Solid brick, with internal insulation" → solid_brick/internal; "Average thermal transmittance 0.28 W/m²K" → u_value |
| `ROOF_DESCRIPTION` | `roof.type`, `roof.loft_insulation_mm` | "Pitched, 270 mm loft insulation" → pitched_loft/270; "(another dwelling above)" → another_dwelling_above |
| `FLOOR_DESCRIPTION` | `floor.type`, `floor.insulated` | "Suspended, no insulation (assumed)"; "(another dwelling below)" |
| `WINDOWS_DESCRIPTION`, `MULTI_GLAZE_PROPORTION`, `GLAZED_TYPE` | `glazing[]` | "Fully double glazed" + "double glazing installed during or after 2002" → double_post_2002 100%; partial → two entries by proportion |
| `MAINHEAT_DESCRIPTION`, `MAIN_FUEL` | `heating.main_system`, `heating.main_fuel` | "Boiler and radiators, mains gas" + condensing flag; "Electric storage heaters"; "Air source heat pump" |
| `MAINHEATCONT_DESCRIPTION` | `controls[]` | "Programmer, room thermostat and TRVs" → programmer, room_thermostat, trvs |
| `LOW_ENERGY_LIGHTING` | `low_energy_lighting_pct` | numeric % |
| `PHOTO_SUPPLY`, `SOLAR_WATER_HEATING_FLAG` | `renewables` | PV share of roof → kWp estimate; solar thermal flag |
| `CURRENT_ENERGY_RATING`, `CURRENT_ENERGY_EFFICIENCY`, `ENERGY_CONSUMPTION_CURRENT` | `epc.current_band`, `epc.sap_score`, `epc.energy_consumption_kwh_m2` | numeric; the modelled energy is a validation target only |
| `POTENTIAL_ENERGY_RATING`, `POTENTIAL_ENERGY_EFFICIENCY` | `epc.potential_*` | **leakage: label only** |
| `MAINS_GAS_FLAG` | `constraints.off_gas_grid` | `N` → off-gas (primary off-gas indicator; postcode lists only as fallback) |
| `FLOOR_HEIGHT`, `GLAZED_AREA`, `NUMBER_HEATED_ROOMS` | `TwinParams.volume_m3`, `window_area_m2` | storey height × floor area; glazed-area band → window-to-floor ratio |
| `HEATING_COST_CURRENT`, `HOT_WATER_COST_CURRENT`, `LIGHTING_COST_CURRENT` | `epc.estimated_annual_energy_cost_gbp` | sum; validation target for standard mode |
| `TENURE`, `FLOOR_LEVEL`, `LODGEMENT_DATE`, `TRANSACTION_TYPE`, `POSTCODE`, `UPRN`, `LOCAL_AUTHORITY` | `tenure`, `constraints.floor_level`, `epc.lodgement_date`, `location.*` | postcode → district + LSOA + LA code (ONSPD), then the full postcode is dropped; UPRN is a private join key; latest certificate per UPRN = current state |
| `CERTIFICATE_NUMBER` | private key | links certificates to recommendations (`EPCRecommendation`); replaces the old `LMK_KEY` |

Field names are from the EPC service's domestic data dictionary (`/download/data-dictionary?property_type=domestic`, checked 11 Oct 2026). That dictionary covers the CSV download only; API JSON names differ.

**Splits** (`splits.py`): `hash(customer_id or property_id, seed=42)` → 70/15/15, written once to `Property.split` and copied to every row of that home (usage, sales, labels). Spatial-block splits by postcode district are used for area models (GNN). Test: no id appears in two splits.

**DuckDB tables** (one per contract; nested contracts stored as JSON, plus flattened key columns):

| Table | Key | Key columns | Contract |
| --- | --- | --- | --- |
| `properties` | property_id | split, lsoa_code, local_authority_code, region, type, form, age_band, floor_area_m2, sap, band, json | `Property` |
| `usage_monthly` | property_id, month | electricity_import_kwh, gas_kwh, pv, export, estimated_read | `EnergyUsageMonth` |
| `sales` | sale_id | property_id, date, price_gbp, band, sap, new_build, tenure, lsoa, la, type, floor_area, split | `SaleRecord` |
| `weather_monthly` | area_code, month | hdd_15_5, cdd_22, mean_temp_c, ghi, source | `WeatherMonth` |
| `epc_recommendations` | property_id, order | measure_code, catalogue_id | `EPCRecommendation` |
| `area_epc_stats` | area_code, period, dwellings | lodgements, floor area, band A–G, mean kWh/m², CO2, cost, type counts | `AreaEPCStats` |
| `lsoa_consumption` | lsoa_code, year, fuel | meters, mean_kwh, median_kwh | `LSOAConsumption` |
| `load_profiles` | archetype_id, season, day_type | shares[48], n_households | `LoadProfileArchetype` |
| `reference_labels` | property_id, label_type, retrofit_ids | value, unit, origin, rule_id | `ReferenceLabel` |
| `calibrations` | property_id, version | json | `CalibrationResult` |
| `posteriors` (Parquet) | property_id, version, sample | one column per `TwinParams` field | `PosteriorSamples` |
| `scenario_results` | scenario_id | property_id, spec_hash, model_versions, key metrics, json | `ScenarioResult` |
| `memory` | customer_id | json | `CustomerMemory` |
| `dashboard_areas` | area_code, area_type | layer columns, json | `DashboardArea` |

### 4.1 Part 1 — Digital twin

**4.1.1 Physics (JAX, `twin/physics.py`).** A monthly steady-state RC model. It is written in `jax.numpy` and `jax.jit`-compiled, then `jax.vmap`ped over posterior samples. It runs inside the NumPyro model, so NUTS and SVI get exact gradients.

```
H_fabric = Σ_e U_e·A_e + y·A_exposed                     [W/K]   e ∈ walls, roof, floor, windows, doors
H_vent   = 0.33 · ach · V                                [W/K]
H        = m · (H_fabric + H_vent)                       m = heat_loss_multiplier (calibrated)
HDD*_t   = HDD15.5_t + (T_set − 19.5) · D_t · h/24       setpoint shift over D_t heating days
Q_sh,t   = max(0, H·24·HDD*_t/1000 − η_g·(G_int,t + f_sol·G_sol,t))      [kWh]
Q_hw,t   = q_hw · days_t ;  Q_base,t = b · days_t
Fuel     = Q_sh/η (boiler) or Q_sh/SCOP (heat pump) to its fuel; hot water by its source
Elec_t   = Q_base,t + pumps_t + electric heat/hot water − PV_self,t          (import)
Gas_t    = gas heat + gas hot water ;  Export_t = PV_t − PV_self,t
PV_t     = kWp · PVGIS_yield(lat, lon, azimuth, tilt)_t ;  PV_self from the electricity layer (default 0.4)
```

**Modes:** `calibrated` (posterior behaviour) and `standard` (SAP heating pattern and occupancy by floor area, for the rating estimator). **Retrofits:** `apply_retrofits` applies `RetrofitItem.twin_edits` (number = set, `x<k>` = multiply, string = categorical). Monotone by construction: lower U never raises heat loss, enforced by a sweep test.

**Element heat loss** = prior split (U·A shares) × calibrated total. Always `estimated=True` with the basis label; bills only identify the total.

**4.1.2 Bayesian calibration (NumPyro, `bayes/model.py`).**

| Parameter | Prior | Hierarchy |
| --- | --- | --- |
| `heat_loss_multiplier` m | LogNormal(μ_type,region, 0.25) | μ_type,region ~ Normal(μ_global, τ), τ ~ HalfNormal(0.2); home → LSOA → region |
| `setpoint_c` | Normal(19.5, 1.5), truncated [15, 24] | — |
| `baseload_kwh_per_day` | LogNormal(log b₀, 0.35), b₀ from the EBM electricity prior | by property type |
| `heating_efficiency` | Boiler: Beta around the RdSAP value; heat pump: LogNormal around the Electrification of Heat SCOP | by system |
| meter noise σ_f | HalfNormal(0.1) relative; ×2 for estimated reads | — |

**Likelihood:** `y_t,f ~ StudentT(ν=5, twin_t,f(θ), σ_f·twin_t,f)` for every observed month t and fuel f. Missing months are left out of the likelihood and **predicted** from the posterior (latent, flagged).

**Inference:**
- **Offline:** NUTS (4 chains × 1,000) on a training subset fits the hierarchy.
- **Per home:** SVI (`AutoMultivariateNormal`, ~2,000 steps), initialised from the population posterior; under 1 s on CPU. Draw S = 500 samples.
- **Diagnostics:** R̂ < 1.01 and ESS (NUTS); ELBO convergence (SVI); posterior-predictive coverage of observed months.
- **No meters:** `method = prior_only` (prior centred on the EBM prediction, wider distributions, `data_confidence = low`).

**4.1.3 3D scene (`twin/scene.py` → `TwinScene`; viewer `web/twin3d`).**
- **Parametric geometry:** footprint = floor area / storeys, with aspect 4:3 adjusted by built form (party walls for terraces and semis), a roof from `roof.type`, and windows from glazing share × window-to-floor ratio.
- **Surfaces** carry orientation, area, U and the posterior-mean heat-loss share (colour), with the distribution available on hover.
- **Retrofit slots:** PV on the best roof face, heat pump on an outside wall, loft, internal walls, plant cupboard; each slot's status comes from `FeasibilityReport`.
- **Captured homes:** with a `FloorPlan`, the rooms replace the box. The viewer never computes; numbers come from `ScenarioResult`.

**4.1.4 Photo-to-twin (`twin/capture.py` → `CaptureResult`; editor `web/capture`).**
1. Upload a floor-plan image and room photos, with consent recorded.
2. Blur faces and personal items client-side.
3. Call **Claude Haiku 5.5** (`claude-haiku-5-5`) with image content blocks and **structured outputs** (`output_config.format` / `client.messages.parse`, JSON schema = rooms + element tags with confidence), low effort.
4. Mark fields with confidence < 0.7 as `needs_confirmation`.
5. The agent edits polygons and tags in a **react-konva** canvas.
6. Confirmed `FloorPlan` + tags fill `Property` and the twin's areas: wall area = exterior wall length × storey height − window area.

Plan area is checked against the EPC floor area (`area_vs_epc_pct`, warning above 15%). The physics stays home-level. Vision never outputs energy numbers. Always check `stop_reason` (handle `refusal`); fall back to manual tags on error.

**4.1.5 Electricity layer (`twin/electricity.py` → `ElectricityProfileSummary`).**
- **Archetypes:** k-means (k ≈ 8) on normalised daily half-hourly profiles from the London smart-meter data, by season × day type, gives `LoadProfileArchetype`s.
- **Per home:** the archetype is scaled to the calibrated baseload; heat-pump demand is distributed by degree-hours, PV by the PVGIS hourly series, and a battery is dispatched greedily (charge from surplus PV / off-peak, discharge at peak).
- **Outputs:** self-consumption, export, peak share, and bills (standard, plus time-of-use once a tariff is cited). Monthly totals are reconciled to the monthly twin.

### 4.2 Part 2 — Energy & retrofit engine

**Energy prior (EBM, `models/energy_prior.py` → `EnergyPrior`).**
- **Target:** annual electricity and gas kWh/m² (NEED, record-level, split by property).
- **Features:** type, form, age band, floor area, region, IMD decile, main fuel, heating system, wall/roof/glazing state, degree days.
- **Model:** monotone shape functions where physics requires (better insulation never raises use); conformal 90% intervals.
- **Baselines:** NEED median by type × age × region, and XGBoost (`monotone_constraints`) + SHAP.
- **Output:** `Distribution` per fuel plus `FeatureContribution`s (glass-box explanations).

**Baseline (`baseline`).** `TwinRun` over S samples → `EnergyBreakdown` via `summarise`. `intensity_kwh_m2` = **annualised metered** (grid import + gas) / floor area. Missing months come from the calibrated fit for that month's actual degree days, never ×12 and never 0. The weather-normalised intensity is a sensitivity only.

**Ladder (`retrofit/ladder.py`).**
```
feasible = [r for r in feasibility.items if r.status in (feasible, feasible_with_conditions)]
order    = sort(feasible, by=(ordering group from feasibility_rules.json, expected net benefit))
prev     = twin.simulate(samples, [], weather)                        # baseline, S samples
for k, r in enumerate(order, 1):
    cur  = twin.simulate(samples, order[:k], weather)                 # cumulative package
    Δ    = prev − cur   per fuel, per sample                          # marginal effect of step k
    carbon = Σ_f Δ_f · EF_f ;  bill = Σ_f Δ_f · rate_f (tariffs.json; standing charge only if a supply is removed)
    cost ~ catalogue range (null + reason if unsourced) − grants
    p_net_benefit = P(goal metric > 0 per sample) ;  p_suitable = p_feasible · p_net_benefit
    p_gain = P(Δ purchased ≥ 3% of baseline or Δscore ≥ 2)            # thresholds in config, to agree
    verdict = rules(p_suitable, p_gain, cost per kWh)  → RetrofitEffect(order=k, …)
    prev = cur
```
The marginals add up exactly to the package. A different combination is a **new run**, never "package minus a row".

**Improvement potential (`retrofit/potential.py`).** One run with all feasible + with-conditions retrofits. `pct = 100·(E_base − E_after)/E_base` per sample, on purchased energy. It is null + reason if E_base is 0 or missing. `pct_if_surveys_pass` adds the needs-survey items; `p_top_decile` is computed across homes in batch.

**Rating estimator (`retrofit/rating.py` → `RatingEstimate`).** It follows `contracts/sap_estimator.json`:
1. Run the twin in **standard** mode before and after the package, on every sample.
2. Price the energy with `tariffs.json`.
3. `ECF = 0.42·cost/(TFA + 45)`; `SAP = 117 − 121·log10(ECF)` if ECF ≥ 3.5, else `100 − 13.95·ECF`; clamp to 1–100.
4. **Anchor:** `SAP_after = SAP_recorded + (SAP_model_after − SAP_model_before)`.
5. Band from the thresholds; `band_probabilities` via `category_probabilities`.

It is labelled "estimated rating, not an EPC" and returns `None` + warning if inputs are missing or out of distribution.

**Pareto (`optimise/pareto.py`).**
- Enumerate the feasible subsets (≤ 256, respecting `excludes`/`depends_on`).
- Objectives per sample: net cost after grants (sourced costs only), CO2 cut, bill saving, score gain.
- Run a non-dominated sort per sample → `p_on_front`; the knee comes from the posterior-median front.
- Baselines: greedy kWh/£ and CO2-only. Metrics: hypervolume.

### 4.3 Part 3 — Green Home Score (`score/score.py`)

- `S = Σ_k w_k·s_k` (weights from `score_mapping.json`), rounded half-up, then band.
- **Inputs:** energy efficiency = recorded SAP (or `RatingEstimate.sap` for a projected score); intensity = `EnergyBreakdown.intensity_kwh_m2` (annualised metered); the other five come from `Property` fields.
- **Uncertainty:** evaluated per posterior sample of intensity (and SAP when estimated) → `score_distribution`, `band_probabilities`, `p_poor_or_very_poor`.
- **Key issues** (`score/issues.py`):

  | Issue | Rule |
  | --- | --- |
  | `high_intensity` | P(intensity > peer P50 for type × age × region) > 0.7 |
  | `poor_insulation` | insulation sub-score < 40 |
  | `old_heating` | heating sub-score < 40 or boiler age > 15 |
  | `single_glazing` | any single glazing |
  | `no_renewables` | renewables sub-score 0 |
  | `no_controls` | controls sub-score < 40 |
  | `data_conflict` / `missing_data` | `data_quality` |

  Each issue carries its evidence; per-m² comparisons keep large homes from being penalised for size.
- **Peer context:** local-authority band shares and mean modelled kWh/m² from the **EPC live tables** (D1, D3; `AreaEPCStats`).
- **Tests:** monotonicity sweep per sub-score; every `HeatingSystem` mapped; projected ≥ current for improvement-only packages.

### 4.4 Part 4 — Property value (`value/hedonic.py`)

```
log(price_i / HPI_LA,type,month_i) ~ StudentT(ν, μ_i, σ_type)
μ_i = α_region + α_LA[region] + α_LSOA[LA]           pooled location effects
    + f(log floor_area) + γ_type×form + γ_age + γ_tenure + γ_rooms
    + β_region×type · SAP_i                          β ~ Normal(β₀, 0.01); β₀ from DECC 2013 / Nationwide
```
- **Data:** UCL Price Paid–EPC linked sales, split by property. Fitted with NumPyro SVI (minibatch) per region on a stratified sample; NUTS for the repeat-sales check.
- **Prediction:** V_s = current value (or posterior predictive if missing); ΔSAP_s from `RatingEstimate`; `uplift_s = exp(β_s·ΔSAP_s) − 1` → `uplift_pct`, `value_after_gbp`, `rating_effect_pct_per_sap_point`, `price_drivers` (coefficient × value).
- **Wording:** always `ValueImpact.wording`; if the 90% interval includes 0, say no clear effect.
- **Green twins:** coarsened exact matching (type, form, age band, tenure, LSOA, sale quarter; floor area ±10%) with a ≥ 15-point score gap, shown as ranges only.
- **Evaluation:** held-out coverage and pinball loss vs OLS without the rating; β vs published premiums; planted-premium recovery reported as a method check only.

### 4.5 Part 5 — Feasibility, Jev, LLM explanation and chat

**Feasibility (`retrofit/rules.py` + `jev/feasibility.py` → `FeasibilityReport`).**
1. **Layer 1:** evaluate every rule in `feasibility_rules.json` on `Property` (+ memory constraints). A `blocks` match → `infeasible`, `p_feasible = 0`. A `downgrade` → that status. **Rule vetoes always win.**
2. **Layer 2** (survivors only): Jev Choice {feasible, with conditions, needs survey, infeasible} on a state built from home facts + EPC text + agent notes + memory. Its confidence is recalibrated with isotonic regression; below τ → `needs_survey`.
3. **Without a Jev key:** OpenAI structured output (`gpt-6-luna`) on the same schema (`router = llm_fallback`); otherwise rules only.

Reasons shown to users come from rule codes and the knowledge base, never from the LLM.

**Router (`jev/router.py` → `WhatIfRoute`).**
- Intent ∈ {numeric, comparative, feasibility, explain, out_of_scope}, plus `ScenarioSpec` slots, each with confidence.
- τ is chosen on 30–50 labelled prompts so decisions above it are ≥ 95% correct; below τ → clarifying question.
- Jev never produces kWh, £, CO2 or score values.

**Chat (`llm/chat.py` → `ChatResponse`).**
- **Model:** `LLM_MODEL` from `.env`: OpenAI `gpt-6-luna` for every call (team-lead decision, 11 Oct 2026). Responses API tool loop; see `docs/LLM.md`.
- **Tools:** `get_home_facts`, `get_results`, `run_scenario`, `search_docs`, `query_graph`, defined with `strict: true` and `tool_choice: auto`. Forced tool choice is rejected on current models, so the system prompt instructs the model to call `run_scenario` for every what-if.
- **Loop:** manual tool loop or the SDK tool runner. Check `stop_reason` (`refusal`, `max_tokens`) before reading content.
- **Caching:** the frozen system prompt and tool list are prompt-cached.
- **System prompt rules:** never compute; quote tool numbers with unit and interval; say "estimate", "projected", "indicative"; state missing, conflicting or infeasible data; no financial advice.

**Validator (`llm/validator.py` → `ExplanationCheck`).**
- Regex-extract every number (with units, %, £, ranges) from the answer.
- Each must match a number in that turn's tool outputs (relative tolerance 1% or rounding-equivalent); dates and years are allowed by an allow-list.
- On failure: regenerate once with the unsupported numbers named, then fall back to the **template explainer** (`llm/template.py`), which fills fixed sentences from `ScenarioResult` and is also used when there is no API key.
- Optional layer 2: Jev "is every claim supported by this JSON?". Every call is cached by hash(prompt + tool outputs + model) in `cache/llm/`.

**Knowledge base (`kb/`).**
- **SQL:** DuckDB views, the only source of numbers.
- **Vector store:** methods docs, grant rules, EPC recommendation text; cited chunks.
- **Graph:** embedded Kùzu/networkx with Home → elements, Home → LSOA → region, Retrofit → requires/blocked-by, Grant → applies-to. Queried by templated `query_graph(template_id, params)` only.

**Memory:** `CustomerMemory` constraints feed `FeasibilityEngine.assess`; rejected items are demoted; new bills trigger recalibration (posteriors narrow).

### 4.6 Orchestrator (`api/orchestrator.py`)

```
run_scenario(spec):
  prop      = data.load_property(spec.property_id)
  calib, S  = cache or twin.calibrate(prop, usage, weather, engine.energy_prior(prop))
  feas      = feasibility.assess(prop, catalogue, memory)
  base      = engine.baseline(prop, S, spec.weather)
  after     = engine.package(prop, S, spec.retrofits)              # one run
  ladder    = engine.ladder(prop, S, feas, spec.goal)
  potential = engine.improvement_potential(prop, S, feas)
  rating    = engine.estimate_rating(prop, S, spec.retrofits)
  score_b   = scorer.score(prop, base, None, peer) ; score_a = scorer.score(prop, after, rating, peer)
  value     = value_model.value_impact(prop, rating)            if spec.include_value
  pareto    = engine.pareto(prop, S, feas)                       if spec.include_pareto
  → ScenarioResult(provenance, distributions, warnings) ; cache[hash(spec, model_versions)]
```

**Latency budget** (showcase homes are precomputed):

| Step | Budget |
| --- | --- |
| Calibration | cached, else < 1 s |
| Ladder (S = 500, ≤ 10 steps, `vmap`) | < 1 s |
| Pareto (≤ 256 packages) | < 3 s, or precomputed |
| LLM answer | typically under 10 s |

{{CONTRACTS}}

{{INTERFACES}}

## 7. REST API (`greentwin/api/app.py`)

| Method & path | Request | Response | Notes |
| --- | --- | --- | --- |
| `GET /homes/{id}` | — | `Property` + latest `ScoreResult` | `get_home_facts` tool |
| `POST /homes` | `Property` + `EnergyUsageMonth[]` | `{property_id}` | agent capture / self-serve |
| `POST /capture/{id}` | multipart images | `CaptureResult` | Haiku vision drafts; agent confirms via `PUT /homes/{id}` |
| `POST /scenario` | `ScenarioSpec` | `ScenarioResult` | `run_scenario`; cached |
| `GET /homes/{id}/scene?scenario_id=` | — | `TwinScene` | 3D viewer |
| `GET /homes/{id}/feasibility` | — | `FeasibilityReport` | rules → Jev |
| `GET /homes/{id}/pareto?budget=` | — | `ParetoPoint[]` | precomputed for showcase homes |
| `GET /homes/{id}/green-twins` | — | `GreenTwin[]` | ranges only |
| `POST /route` | `{property_id, message}` | `WhatIfRoute` | Jev / OpenAI fallback |
| `POST /chat` | `ChatRequest` | `ChatResponse` | tool-only LLM + validator |
| `GET /dashboard/areas?layer=&region=` | — | `DashboardArea[]` (GeoJSON-wrapped) | Lloyds heat map |
| `GET /dashboard/segments` · `POST /dashboard/green-profits` | sliders | segment table · Monte Carlo `Distribution`s | scenario, labelled assumptions |
| `GET /memory/{customer_id}` · `POST /memory/{customer_id}` | `CustomerMemory` | `CustomerMemory` | |

Errors: 404 unknown id; 422 contract validation; 200 with `null` + reason for unavailable figures (never a fake number). Mock mode serves `contracts/examples/*_MOCK.json`, which never reaches the demo.

{{DATA}}

## 9. Configuration files (`contracts/`)

| File | Owner part | Contents |
| --- | --- | --- |
| `score_mapping.json` | 3 | Fixed weights, monotone sub-score mappings, intensity definition, overlap justification |
| `feasibility_rules.json` | 5 | Rule vetoes / downgrades with reason codes; ordering groups |
| `retrofit_catalogue.json` | 2 | Retrofits, `twin_edits`, dependencies, sourced costs + `cost_status`, grants |
| `sap_estimator.json` | 2 | Post-retrofit rating method (SAP 2012 ECF, anchoring, band thresholds) |
| `emission_factors.json` | 2 | DESNZ 2025 factors (electricity 0.19553, gas 0.18296 gross CV) with row ids |
| `tariffs.json` | 2 | Ofgem Q4 2026 rates; time-of-use entry unsourced until cited |
| `green_profits_assumptions.json` | Orchestrator | Scenario rule for take-up; every coefficient a labelled assumption |

## 10. Testing, evaluation, fallbacks

| Part | Must-pass tests | Held-out metrics (vs baseline) |
| --- | --- | --- |
| 0 | contracts validate; no id spans splits; no leakage fields in features; mapper coverage ≥ 95%; our EPC counts per LA × quarter reconcile with live table D1 | — |
| 1 | twin monotone sweep; SH1–SH3 within ±15% of metered; `estimated=True` on element heat loss | calibration coverage ≈ 90%, CRPS, R̂ (vs degree-day OLS, LightGBM quantile); standard-mode kWh/m² vs EPC D3 by LA; masked-month MAE; capture room-area error + tag accuracy |
| 2 | per-fuel signs; package = one run; potential null at baseline 0; rating monotone | EBM MAE/MAPE + coverage (vs NEED median, XGBoost); rating Δ on EPC before/after pairs; ranking NDCG; precision@k vs EPC recommendations; Pareto hypervolume |
| 3 | weights sum to 1; monotone sub-scores; every heating type mapped | — (deterministic); band-probability calibration |
| 4 | wording always present; uses current value, never a sale price | coverage + pinball (vs OLS without rating); β vs published |
| 5 | rule veto beats Jev; validator rejects unsupported numbers; works with no keys | routing accuracy ≥ 95% above τ; feasibility precision/recall; groundedness ≥ 90%, accuracy ≥ 90%, unsupported < 5% on ≥ 30 scenarios |

**Fallbacks:**

| Missing | Replacement |
| --- | --- |
| No LLM key | Template explainer |
| No Jev key | OpenAI structured output → rules |
| No EPC API | Cached sample files |
| Calibration failure | `prior_only` + warning |
| Unsourced cost / factor / tariff | `null` + reason |
| No WebGL | 2D view |
