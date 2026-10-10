# GreenTwin — Implementation Plan v2 (build blocks)

Megathon X · Lloyds PS2 · Team MegaMind

v1 (`IMPLEMENTATION_PLAN.md`) listed every idea by tier. This version groups the work into **8 build blocks**. Each block is a complete piece of the system that can be built, tested and documented from start to finish. Blocks connect only through the contracts in `contracts/`, so all of them can start on day one against stubs or mocks. Owners are assigned separately.

**Precedence.** `AGENTS.md` (including its binding "Compliance fixes" section) and `contracts/` win over this plan. Design detail for each component is in `docs/STRATEGY_REPORT.md` (system design) and `docs/STRATEGY_REPORT_OVERVIEW.md` (rationale); this plan says **what to build, in which block, and when it is done**.

**Core philosophies:**
1. Physics is the source of truth.
2. Bayesian in thought.
3. Rerun, never add up.
4. Grounded explanation.
5. Explainability first.
6. See it to believe it.

---

## Physics library: JAX (decided) and the open twin-ambition decision

**Library: JAX (`jax.numpy`), not numpy or PyTorch.** The Bayesian calibration (B3) puts the twin *inside* the NumPyro model, and NUTS/SVI need gradients of the likelihood through it. A numpy twin can't be sampled by NUTS. Written in `jax.numpy`, the same twin code:
- gives exact gradients for calibration,
- runs over all posterior samples at once with `jax.vmap` (fast ladder, Pareto and what-ifs),
- is `jax.jit`-compiled, and runs on CPU (JAX GPU is Linux-only, which is fine).

The steady-state twin is simple algebra, so writing it in `jax.numpy` costs almost nothing extra.

**Still open: how ambitious is the twin?**

| | Option A — monthly steady-state twin first (default) | Option B — differentiable RC graph first (`STRATEGY_REPORT.md`, "Must" tier) |
| --- | --- | --- |
| Level-1 twin | Monthly steady-state RC element model in JAX | 4–6 node RC graph in JAX; an MLP maps EPC features → conductances; ODE solve with `diffrax` |
| GP discrepancy (metered = twin + GP + noise) | Stretch | Core, inside the calibration |
| Risk | Low; fast to an end-to-end skeleton | Higher; checkpoint 1 slips if the ODE or training stalls |
| Gain | — | Learned conductances, hourly runs, end-to-end training on meter data |

**Recommendation:** Option A for checkpoint 1, then upgrade to the differentiable graph behind the same `simulate()` interface if checkpoint 2 is on time. Both options use JAX and keep the same contracts, so nothing downstream changes.

---

## The pipeline at a glance

```
 B1 Real data + sourced factors ──► B2 Physics twin + rating estimator + synthetic data ──► B3 Bayesian calibration + prediction
      │                                    │                                                     │
      │                                    ▼                                                     ▼
      │                            B4 Score + retrofit recommendations ◄─────────────────────────┘
      │                                    │
      └──► B5 Property value ──────────────┤
                                           ▼
                B6 Scenario engine + API + knowledge base + grounded LLM + memory
                                           │
                           ┌───────────────┴───────────────┐
                           ▼                               ▼
                 B7 Customer app + 3D twin        B8 Lloyds dashboard + geography
```

Problem-statement coverage:

| PS section | Requirement | Block |
| --- | --- | --- |
| §2 | Green Home Score: fixed weights, monotonic and fully documented sub-scores, contributions, justified overlaps | B4 |
| §3 | Flag Poor/Very poor homes, list key issues, rank by improvement potential | B3 (energy) + B4 (flag, rank) |
| §4 | Prioritised, feasible retrofits with suitability and reasons; catalogue source and assumptions documented | B1 (sources) + B4 |
| §5 | Per-retrofit and package savings, per fuel, interactions | B2 (twin) + B3 (intervals) |
| §6 | Carbon from documented per-fuel factors (units + source) | B1 (factors) + B3 |
| §7 | Value range: before vs after, location, size, type, market; indicative wording | B2c (rating after) + B5 |
| §8 | Pipeline and what-if flow (rerun the models, then explain) | B6 |
| §9 | Data, schema, splits, leakage, documented and plausible label generation | B1 + B2 |
| §10 | Customer memory (optional) | B6 + B7 |
| §11 | Evaluation vs baselines, held-out metrics, LLM targets | each block + evaluation section |
| §13 | Submission items A–I | final section |

---

## Block 1 — Real data foundation + sourced factors
**Starts:** immediately · **Feeds:** B2, B3, B4, B5, B8

**Goal:** clean, licensed, split, joinable real data, with every external factor traceable to a cited source.

**Build:**
1. Set up the environment: `uv` + Python 3.12, `uv sync`, and CI running `pytest` (fresh clone → green).
2. Download scripts (`scripts/download_*.py`). Each writes Parquet to `data/processed/` with a `*.meta.json` (source, licence, date, rows):
   - **EPC certificates + recommendations** for North West, South West and London. Store property features, SAP score/band, `potential_*` fields (labels only), the EPC recommendation list, and **certificate history per property**, which B2c needs for validation.
   - **NEED 50k anonymised**: real annual gas and electricity with property attributes.
   - **UCL Land Registry–EPC linked sales** (854942 / 857911) + **UK HPI** by local authority and type.
   - **Weather**: monthly HDD (base 15.5 °C), mean temperature and irradiance by area from HadUK-Grid (primary) or Open-Meteo. Also 30–50-year history for trend layers. Cache every response.
   - **ONS postcode → LSOA / local authority**, IMD deprivation, off-gas-grid postcodes, conservation areas / listed buildings (planning.data.gov.uk, Historic England), and OSM footprints (roof azimuth, party walls).
   - **Priors:** Electrification of Heat trial (heat-pump SCOP by house type) and English Housing Survey stock distributions.
   - **London smart meters** (London Datastore, half-hourly, 5,567 homes, 2011–14, electricity only): cluster daily profiles by season and day type into **load-shape archetypes** for the electricity layer (B2).
   - **DESNZ sub-national LSOA consumption** (OGL): real annual domestic electricity and gas per meter by LSOA, the labels for the GNN area model (B8) and a check on synthetic area totals.
3. **Sourced factors.** These already exist in `contracts/` and are verified. This block keeps them correct and closes the gaps:
   - `emission_factors.json`: DESNZ 2025. Electricity 0.19553 kgCO2e/kWh (generation 0.177 + T&D 0.01853); gas 0.18296 (gross CV). **Gap:** oil, LPG, solid fuel and district heating return `carbon = null` until their factors are added from the same file.
   - `tariffs.json`: Ofgem Q4 2026.
   - `retrofit_catalogue.json` costs: BEIS 2017 Table 3 × ONS CPI 1.3888. **Gaps:** 4 items are `cost_status: unsourced` and 2 are `secondary_source`. Source them or leave them returning `cost = null` + reason, excluded from cost ranking.
   - **Still TO_VERIFY:** BUS £9,000 off-gas dates, ECO4/GBIS status, the SAP 2012 formula text (B2c), UCL variable names.
   - **Test:** every factor and cost the engine reads has a source or explicitly returns `null`.
4. Write mappers from EPC rows to `Property` (contracts enums) and from the sales table to `SaleRecord`.
5. Implement `greentwin/data/splits.py`: assign 70/15/15 by `customer_id`, else `property_id` (seed 42), once, into `Property.split`; use spatial-block splits by postcode district for geo work.
6. Leakage audit: `epc.potential_*` and anything recorded after the prediction date are labels only and never model inputs.
7. Load everything into DuckDB `data/greentwin.duckdb`.

**Outputs:**
- Tables: `properties`, `epc_history`, `epc_recommendations`, `need`, `sales`, `hpi`, `weather_monthly`, `geo_lsoa`.
- A `load_property(id)` helper.

**Done when:**
- Tests show no ID spans two splits and no leakage fields are present in feature lists.
- Factor and cost tests pass.
- `docs/DATA.md` covers sources, licences, schema, missing-value handling, the factor citations and limitations (submission item I, part 1).

---

## Block 2 — Physics twin + post-retrofit rating estimator + synthetic data
**Starts:** immediately (RdSAP defaults; real data plugs in from B1) · **Feeds:** B3, B4, B5, B7

**Goal:** a documented physics model that turns any home and any retrofit package into per-fuel energy and an estimated rating. A generator then uses that model to create realistic homes, meter readings and reference labels.

**2a. Level-1 twin** (`greentwin/twin`, written in **JAX** so B3 can sample it with NUTS/SVI and `vmap` it over posterior samples; Option A, see the decision above):
- `params_from_property(Property) -> TwinParams`:
  - Element U·A from RdSAP default U-values by age band, wall type and insulation.
  - Areas estimated from floor area, built form, storeys and party walls (OSM).
  - Ventilation heat loss, thermal-bridging allowance.
  - Heating-system efficiency or SCOP (Electrification of Heat prior), hot-water demand, baseload, PV kWp.
- `simulate(params, weather, mode) -> TwinOutput` (monthly):
  - Space heat = (ΣUA + vent) × HDD, less solar and internal gains.
  - Plus hot water; converted to fuel through efficiency or COP.
  - PV yield (PVGIS, roof azimuth) reduces grid import, not demand.
  - Returns per-fuel kWh (electricity import, gas, PV generation, export) and heat loss per element (`ElementHeatLoss`).
  - **`mode = "calibrated"`** uses the home's posterior behaviour (B3). **`mode = "standard"`** uses a SAP-style fixed heating pattern and occupancy by floor area (for B2c).
- **Per-element heat loss is prior-based.** It comes from RdSAP defaults for the home type and its EPC element descriptions; only the total is calibrated to bills. Always emit `ElementHeatLoss.estimated = true` with its `basis` label.
- `apply_retrofits(params, actions) -> TwinParams`:
  - Each catalogue item is a parameter edit, e.g. loft → roof U, heat pump → COP + fuel switch, PV → generation at the electricity node.
  - Packages are re-simulated as a whole, never summed, so interactions come for free.
- **Monotonicity guaranteed by construction:** lower U never increases heat loss. A parameter-sweep test checks this.
- **Out-of-distribution flag** when a home is outside the training ranges. Only the physics twin extrapolates (new combinations, future climate).

**2c. Post-retrofit rating estimator** (`greentwin/twin/rating.py`). Implements `contracts/sap_estimator.json` exactly; provenance engine `sap_estimator`, output `RatingEstimate`:
1. Run the twin in **standard mode** before and after the package.
2. Price the energy with `tariffs.json` (a documented deviation from SAP's own price table).
3. Compute SAP_model_before and SAP_model_after with the SAP 2012 energy-cost-factor formula: ECF = 0.42 × cost / (floor area + 45).
4. **Anchor:** SAP_after = SAP_recorded + (SAP_model_after − SAP_model_before). Because only the model's *change* is used, its systematic bias cancels. With no recorded SAP, use SAP_model_after flagged "no recorded EPC".
5. Push posterior samples through → SAP_after interval + band probabilities → `band_after` for B4 and B5.
6. Always labelled **"Estimated rating, not an EPC."**
7. **Validation:**
   - Primary: homes in the EPC register with certificates before and after a recorded measure. Report MAE and interval coverage of predicted vs lodged SAP change.
   - Secondary: the EPC `potential_energy_efficiency` field, the assessor's after-recommendations rating, used as a reference label only.
   - Baseline: the median band shift per measure.
8. Missing inputs or out of distribution → projected score and value are **unavailable + reason**, never guessed.

**2b. Synthetic data generator** (`greentwin/twin/generate.py`):
1. **Homes:** take **real EPC records from B1** as the population, so features keep their real joint distributions. Fill gaps from EHS stock distributions.
2. **Latent behaviour:** sample setpoint, heating hours, occupants, baseload and rebound from NEED-calibrated priors.
3. **Usage:** run the twin with real monthly weather and add meter noise → 12–24 months of `EnergyUsageMonth`. Inject realistic gaps (missing months, estimated reads) and conflicting records (e.g. rating A but 400 kWh/m2).
4. **Reference labels** (the PS §9 labels), with the true latent parameters kept for B3 coverage checks:
   - Per-retrofit and package per-fuel savings.
   - Improvement potential.
   - Feasibility, from the rules.
5. **Synthetic values (method check only):** price = hedonic model fitted on real sales (B5) + noise drawn from its real residuals + an explicit green-premium term. Recovering that premium shows only that the estimators work. It is never evidence of real uplift and never appears in the UI or LLM.

**Validation:** synthetic annual gas and electricity vs real NEED distributions by type × age × region (KS test, quantiles). Twin on real NEED homes vs their metered totals.

**Done when:**
- SH1–SH3 baselines are within ±15% of their metered annual totals.
- Monotonicity sweeps pass for both twin and rating.
- Rating validation metrics are in `artifacts/sap_estimator/<ver>/metrics.json`.
- `eval/reports/synthetic_validation.md` is written.
- `docs/SYNTHETIC_DATA.md` lists every generation and label rule and states that agreement with synthetic labels is not real-world accuracy.

**2d. Electricity layer** (`greentwin/twin/electricity.py`): a half-hourly electricity profile per home. It's the smart-meter load-shape archetype scaled to the calibrated baseload, plus heat-pump demand from degree-hours, plus PVGIS hourly PV, minus battery dispatch. Gives PV self-consumption, export and time-of-use bills, with monthly totals kept consistent with 2a. Also provides the lighting share used by LED retrofits.

**Upgrade (Option B, if chosen or after checkpoint 2):** a differentiable 4–6 node RC graph in JAX (`diffrax` ODE solver) behind the same `simulate()` interface. An MLP maps EPC features → conductances (softplus > 0) and the model is trained end to end on metered kWh.

---

## Block 3 — Bayesian calibration + prediction engine
**Starts:** on synthetic data from 2b · **Feeds:** B4, B6

**Goal:** fit each home's unknown physics to its own meter data, then predict baseline and post-retrofit per-fuel energy, carbon and bills with **90% credible intervals**.

**3a. Energy-intensity prior** (`greentwin/models/intensity.py`):
- **EBM** (InterpretML) predicts annual electricity and gas kWh/m2 from property + weather + geography. It uses monotonic shape functions where physics requires them.
- Trained on real NEED (+ EPC features).
- Used as the prior mean for homes **without** meter data, and as the PS-suggested regression model. Per-home EBM contributions are stored as numbers so the LLM and dashboard can quote them.
- **Baselines:** NEED median by type/age/region, and XGBoost with `monotone_constraints` + SHAP.
- **Metrics:** MAE/MAPE per fuel, plus 90% conformal coverage, on the held-out split.

**3b. Hierarchical Bayesian calibration** (`greentwin/bayes`, NumPyro):
- **Unknowns per home:** a multiplier on total heat loss (UA), heating setpoint, electrical baseload, heat-pump/boiler efficiency.
- **Priors:** hierarchical home → LSOA → region, by property type, so sparse areas borrow strength. Centred on RdSAP defaults, the 3a prediction and the Electrification of Heat SCOP.
- **Likelihood:** monthly meter readings through the twin. Missing months are **latent**, never zero; estimated reads get wider noise.
- **GP discrepancy** (Kennedy–O'Hagan: metered = twin + GP + noise): Core under Option B, stretch under Option A. Keep its amplitude small so monotonicity holds.
- **Fitting:** NUTS offline for the population hyperpriors; **SVI per home** (< 1 s) for new or showcase homes. Store posterior samples in the `posteriors` table. No live MCMC in the request path.
- Element-level heat loss is the prior split (2a) scaled by the posterior total, and stays labelled "estimated".

**3c. Prediction** (`predict(posterior, ScenarioSpec, weather) -> samples`):
- Push every posterior sample through `apply_retrofits` + `simulate` for the baseline and the package, and for each retrofit **in sequence** (marginal savings).
- From the samples, compute these as `Distribution`s (via `contracts.summarise.summarise`):
  - Per-fuel baseline/after/savings (savings = baseline − after; increases are negative).
  - Purchased energy, and improvement potential = 100 × (E_base − E_after)/E_base, or `null` + reason when the baseline is 0 or missing.
  - Carbon ΔC = Σ ΔE_f × EF_f from `emission_factors.json` (`null` + reason for unsupported fuels).
  - Bills from `tariffs.json`.
- **Weather variability:** sample typical, cold and warm years so intervals include weather, not only parameter uncertainty.
- **Annualised metered intensity** (the §2 quantity, consumed by B4):
  - (grid import + gas) / floor area from the available months.
  - Missing months are filled from that month's actual degree days via the calibrated fit (not ×12), and flagged.
  - No weather normalisation.
- The **weather-normalised** typical-year figure is reported only as a sensitivity.
- Data-confidence score (months of data, imputed fields, EPC age) that also widens intervals.

**Done when:**
- On held-out synthetic homes (true parameters known): 90% interval coverage ≈ 90%, CRPS, flat PIT histogram, R-hat < 1.01. Baselines are degree-day OLS, LightGBM quantile and conformal.
- Missing-month imputation MAE on artificially masked months, vs seasonal-naive ×12.
- SH1's missing month is imputed and flagged visibly.

---

## Block 4 — Green Home Score + retrofit recommendations
**Starts:** immediately (the score needs only `Property` + usage; recommendations plug into B2/B3) · **Feeds:** B6, B7, B8

**Goal:** answer "how green is my home" and "what should I improve", with feasible, ordered, explained recommendations.

**4a. Score** (`greentwin/score`, deterministic, from `contracts/score_mapping.json`):
- S = Σ w_k s_k with the brief's fixed weights; rounded score, band and the 7 component contributions for the waterfall.
- **Energy efficiency:** recorded SAP for the current score; the B2c `RatingEstimate` for a projected score, labelled "projected".
- **Energy intensity:** annualised **metered** energy per m2 from B3c, exactly as §2 defines it.
- **Heating:** the mapping already covers every `HeatingSystem` value, including communal heating. Keep the test that enforces it.
- **Overlaps:** justified in `score_mapping.json`, including the PV overlap. Report its size: the score with and without self-consumed PV added back.
- **Tests:** monotonicity sweep on every sub-score (mandatory); projected score ≥ current score for any package that only improves inputs.
- Score interval and band probabilities come from evaluating the formula on B3 posterior samples, e.g. "38 (90% CI 33–44), P(Poor) = 0.81".

**4b. Poor-home detector** (`detect(property, result) -> issues`):
- Flag Poor/Very poor homes.
- Key issues from rules on sub-scores, e.g. high kWh/m2 vs peers, poor insulation, old boiler, single glazing, no PV. Kept per m2 with peer context so large homes are not penalised for size.
- Rank homes by improvement potential, which comes from the twin package in B3.

**4c. Recommendation engine** (`greentwin/retrofit`):
1. **Feasibility layer 1 (rules):** `feasibility_rules.json` vetoes with reason codes (conservation area/listed → no EWI, no roof access or north-facing roof → no PV, no loft, flat → heat pump needs a survey, no outdoor space, already installed). A rule veto cannot be overridden.
2. **Feasibility layer 2:** Jev or the Claude fallback, from B6. Only for retrofits that pass the rules. It reads EPC descriptions, agent notes and memory, and returns feasible / with conditions / needs survey / infeasible with recalibrated confidence. Low confidence → `needs_survey`. Shown reasons come from rule codes and the knowledge base.
3. **Relevance model:** an **EBM classifier** per measure trained on the **real EPC recommendations table**: P(an assessor would recommend X | features).
   - **Baseline:** rules only. **Metrics:** AUC and precision@k on held-out properties; zero infeasible items recommended.
4. **Ordering + staging:** fabric → glazing → controls → heating system (sized on reduced demand) → generation/storage. Marginal savings come from B3 in that sequence. A small dynamic programme stages the package by dependencies and cash flow.
5. **Suitability:** P(suitable) = P(feasible) × P(net benefit) → High (≥ 0.8) / Medium (0.5–0.8) / Low, with a reason. P(net benefit) comes from the posterior; independence is assumed and stated.
6. **Electricity retrofits** (via the 2d layer):
   - **Battery:** stores PV and cheap-rate electricity.
   - **LED lighting:** cuts the lighting share of baseload.
   - **Time-of-use tariff:** changes bills, not kWh, and is reported that way. Needs a cited supplier tariff before any bill figure is shown.
   - The catalogue already has `battery`; add `led_lighting` and `time_of_use_tariff`.
7. **Pareto optimiser:**
   - Enumerate feasible packages (≤ 256) on net cost after grants (sourced costs only), CO2e cut, bill saving and score gain.
   - Compute the front per posterior sample → **P(on front)**; the knee is the default; add a budget slider and a reference-point mode ("≥ 2 t CO2, ≤ £10k").
   - Memory-learned preference weights re-rank points without relaxing hard constraints.
   - Value is shown next to packages but is not an objective.
   - Precompute every package for SH1–SH3.
   - **Baselines:** greedy kWh/£ and a CO2-only optimiser. **Metrics:** hypervolume, share of greedy picks the front dominates.

**Done when:**
- SH1 blocks EWI and suggests IWI; SH2 shows electricity rising after a heat pump; SH3 blocks loft/PV and marks the heat pump `needs_survey`.
- Ranking metric vs the generator's labels: Spearman/NDCG against "rank by total kWh".
- `docs/SCORE.md` (submission item D) includes the full mapping tables and the overlap justification.

---

## Block 5 — Property value impact
**Starts:** once B1 sales data lands; the "after" band comes from B2c · **Feeds:** B6, B7

**Goal:** an **indicative value range**, before vs after, that accounts for location, size, type and market, and is honest about causation.

**Build** (`greentwin/value`), in order of strength:
1. **Hedonic quantile model (baseline product):**
   - LightGBM quantile (or EBM) predicts log price from location (LSOA/district), floor area, rooms, type, built form, age, sale period (HPI-adjusted) and EPC band.
   - Conformal calibration for the interval. Optional: a spatial Matérn GP location effect.
   - **Before vs after:** the same home at its recorded band vs the B2c `band_after` (weighted by band probabilities), all else fixed → Δ% range.
   - Applied to `current_value_estimate_gbp`, never to a historical sale price. No `band_after` → "insufficient information" + reason.
2. **Green-twin matching (transparent check + product feature):**
   - Match real sales exactly on type/form/age/tenure/LSOA/quarter, and closely on area (±10%), rooms and glazing.
   - Keep pairs with a ≥ 15-point score gap, compared on sub-scores, not only band.
   - Check balance (SMD < 0.1); bootstrap the premium curves by region and type. Also compute running cost per 10 points.
   - Benchmark against Nationwide (A/B ~1.6–1.7% vs D), Oxford Economics (F/G −7.4%) and DECC 2013 (Wales band premiums).
   - Powers "green twins of your home": the 3 closest greener matched homes, shown as ranges, never addresses.
3. **DR-learner on repeat sales (upgrade, Should tier):**
   - EconML `DRLearner`: treatment = band improved between sales, outcome = Δ log price, covariates = location, type, size, period, HPI.
   - Gives uplift by region × type. Replaces the hedonic Δ in the range once its checks pass.
   - **Checks:** propensity overlap + trimming; sensitivity (E-values / EconML); DoWhy refutations (placebo, random common cause, subsets); BLP/GATES on real data.
   - Learner choice is benchmarked on B2's synthetic sales with a known effect (method check only).

**Wording, everywhere:** `ValueImpact.wording`: "Indicative range based on historical associations in sales data; not a promised or proven increase." Never a reason to borrow.

**Done when:**
- Held-out interval coverage and pinball loss vs hedonic OLS without band; comparison with DECC 2013 premiums.
- SH1–SH3 have value ranges.
- `docs/VALUE.md` states association vs causation and lists confounders (other renovations, selection, market timing).

---

## Block 6 — Scenario engine + API + knowledge base + grounded LLM + memory
**Starts:** immediately (mock server first) · **Feeds:** B7, B8

**Goal:** one `run_scenario` that reruns every model for any what-if, an API over it, a knowledge base the LLM can reach only through tools, and an LLM that can only explain numbers it gets back.

**6a. Scenario engine + storage:**
- `run_scenario(ScenarioSpec) -> ScenarioResult` calls B3 predict, B2c rating, B4 score/feasibility/Pareto and B5 value, then assembles the contract with `provenance` on every number.
- Unsourced inputs or unavailable ratings return `null` + reason, never placeholders.
- Cache by hash(spec + model versions); store in DuckDB `scenario_results`.
- Mock mode returns `contracts/examples/SH2_scenario_result_MOCK.json` until the blocks land. A mock result must never reach the demo.

**6b. FastAPI** (`greentwin/api`), the endpoints from v1 §3.4:
- `/homes`, `/scenario`, `/homes/{id}/feasibility|pareto|green-twins`
- `/chat`, `/route`
- `/dashboard/*` (served from B8 functions)
- `/memory/{customer_id}`

**6c. Knowledge base:**

| Store | Holds | Tool |
| --- | --- | --- |
| SQL (DuckDB) | Facts, results, posteriors, EBM contributions; **the only source of numbers** | `get_home_facts`, `get_results` |
| Vector store | Method docs, retrofit guidance, grant rules, EPC recommendation text | `search_docs` (cited) |
| Graph (embedded Kùzu or networkx) | Home → elements, Home → LSOA → region, Retrofit → requires / blocked-by, Grant → applies-to | `query_graph` (templated queries only) |
| Scenario engine | Reruns the models | `run_scenario` |

GraphRAG pattern: retrieve the home's subgraph (constraints, eligible grants, blocked retrofits) plus cited text, then explain tool numbers.

**6d. Grounded LLM** (`greentwin/llm`, Claude tool use):
- **System prompt:**
  - Never compute; quote only tool numbers with unit and interval.
  - Say "estimate", "projected" and "indicative"; use `ValueImpact.wording` for value and the `ElementHeatLoss.basis` label for heat loss.
  - Rerun for every what-if.
  - State missing, conflicting or infeasible data plainly.
  - No financial advice; loans are illustrative.
- **Validator:**
  - Layer 1: regex-extract every number and match it within tolerance to that turn's tool outputs.
  - Layer 2 (optional): a Jev "is every claim supported by this JSON?" check.
  - On failure: regenerate once, then fall back to a **template explainer**, which also covers having no API key.
- Cache every call in `cache/llm/` so evaluations replay offline.

**6e. Jev System-1 layer** (`greentwin/jev`, if `JEV_API_KEY` exists; otherwise Claude structured output on the same interface, otherwise rules):
- **What-if routing:** intent Choice {numeric, comparative, feasibility, out-of-scope} + scenario slots with confidence. Below τ → Claude parses or asks the user to clarify.
- **Comparative questions:** Jev Choice over options already computed by physics/Pareto.
- **Answerability gate:** "Is there enough data to estimate value?"
- **Feasibility layer 2**, for B4.
- **Never** produces kWh, £, CO2 or score values.
- **Calibration (isotonic):** on 30–50 prompts + ~200 home × retrofit pairs. Choose τ so decisions above it are ≥ 95% correct; report a reliability diagram and ECE.
- Cache everything in `cache/jev/`.

**6f. Customer memory:** `CustomerMemory` (completed retrofits, constraints, rejected options with reasons, saved plans).
- Constraints feed B4, and Beta posteriors over acceptance re-rank only feasible options.
- New bills and completed work tighten the home's posterior.
- Demo: recommendations change and intervals narrow from session 1 to 3 (PS §10).

**6g. LLM evaluation set:**
- ≥ 30 scenarios growing to ~100 in `eval/scenarios/`, covering missing months, conflicting data, infeasible requests, electricity rising after a heat pump, not enough data for value, and out-of-scope questions.
- Scored for groundedness ≥ 90%, factual accuracy ≥ 90% and unsupported claims < 5%, vs a no-validator baseline.

**Done when:**
- Real `ScenarioResult`s for SH1–SH3 validate against the contracts.
- 0 unvalidated numbers across the scenario set.
- The service works with no keys at all.
- `docs/LLM.md` (submission item E) is written.

---

## Block 7 — Customer app + 3D twin
**Starts:** immediately against `contracts/examples` and the B6 mock server · **Feeds:** the demo

**Goal:** the customer journey from "how green is my home" to a green-loan conversation, centred on an explorable 3D twin. The 3D scene never calculates anything; every number and colour comes from `run_scenario`.

**Stack:** React + Vite + TypeScript, `contracts/types.ts`, Zustand, Recharts, react-three-fiber + drei, SunCalc.

**Screens:**
1. **Home + score:**
   - Score with interval and band probabilities.
   - **Waterfall of the 7 component contributions.**
   - Key issues, data-confidence badge, warnings.
   - Projected values labelled "projected"; the rating labelled "estimated rating, not an EPC".
2. **Retrofit plan:**
   - Ordered list with suitability and reason.
   - **Per-fuel** marginal savings, with negatives in red.
   - Carbon (factor source on hover), bills, and cost − grants (cost source and `cost_status`).
   - Value with `ValueImpact.wording`.
3. **What-if chat:** message → route chip + confidence → computed-numbers card → explanation with source tags and validator badge.
4. **3D twin:** the parametric house is built procedurally from EPC archetype fields (type, storeys, floor area, roof, glazing ratio, wall type), with optional low-poly glTF archetypes with named meshes.
   - **Views:**
     - Outside orbit with sun path and a weather slider (typical / cold snap / heatwave).
     - Inside cutaway "dollhouse" (clipping planes) with fly-to loft, living room and plant cupboard.
     - **Thermal X-ray** coloured by element heat-loss share, with a colour-blind-safe scale and W/K on hover. Always labelled "Estimated from home type and EPC descriptions; only the total heat loss is calibrated to your bills".
     - **Before/after split slider.**
   - **Each retrofit appears on the house:**

     | Retrofit | Outside | Inside |
     | --- | --- | --- |
     | Loft / roof insulation | Roof colour drops | Insulation between joists |
     | Cavity / external wall insulation | Thicker render or cladding; wall colour drops | Filled cavity in section |
     | Internal wall insulation | Wall colour drops | Board on inner wall; room slightly smaller |
     | Double / triple glazing | Frames and glass swap | Same windows inside |
     | Floor insulation | Floor edge colour drops | Insulation under floorboards |
     | Draught proofing | Fewer leakage particles | Seals on doors and windows |
     | Heat pump | Outdoor unit beside the house | Boiler removed; cylinder added; radiators may upsize |
     | Solar PV (+ battery) | Panels on the best roof face | Inverter and battery in plant cupboard |
     | Smart controls | — | Thermostat on wall; schedule panel |

   - **Interaction:**
     - Retrofit palette: infeasible items greyed out with a reason; "needs survey" items marked.
     - Live numbers panel (score, per-fuel kWh, CO2, bills, value, P(suitable), all with intervals) via a debounced `run_scenario`.
     - Timeline scrubber through the staged roadmap.
     - Click any element → U-value, heat-loss share and score contribution, plus an "Explain" button.
     - Picking a Pareto point loads that package onto the house.
   - **Showcase homes:** all feasible packages (≤ 256) are precomputed, so toggles are instant and the demo can't fail on a slow call.
   - **Build in versions:**
     - v1: parametric house, thermal colouring, toggles, numbers panel.
     - v2: cutaway interior and retrofit objects.
     - v3: timeline animation, sun path, weather slider.
5. **Pareto + green loan:**
   - Cost vs CO2 scatter (colour = value, size = P(on front)) with the knee, budget slider and parallel coordinates for 4+ objectives.
   - Loan = package − grants; repayment vs monthly bill saving ("cash-flow positive?").
   - Marked "illustrative, not an offer"; a "book an adviser" button.
   - Value shown as separate information, never as part of the borrowing case.
6. **Green twins of your home** (from B5) and **memory/history**.
7. **Photo-to-twin (agent capture)**, with B6 providing the vision endpoint:
   - Upload a floor-plan photo + room photos.
   - **Claude Haiku 5.5** (`claude-haiku-5-5`) vision with structured outputs returns rooms (label, polygon, estimated dimensions) and element tags (glazing, boiler/heat pump, radiators, visible loft depth, …), each with a confidence.
   - The agent corrects polygons and tags in a **react-konva** floor-plan editor; every field needs agent confirmation.
   - Confirmed fields fill `Property`. **Room-level geometry** gives floor area (cross-checked with the EPC), wall lengths and window areas for the twin's U·A areas, and real rooms in the 3D house.
   - The physics stays home-level (monthly bills can't identify per-room parameters). Vision outputs are tags, never energy numbers.
   - Consent and blur notice. **v1 fallback:** photos + manual tags.
   - **Evaluation:** room-area error and tag accuracy on a labelled set (CubiCasa5k + our photos), confidence calibration, share of fields the agent corrected.

**UX rules:**
- Every number shows its unit and interval; every colour is paired with a number.
- Colour-blind-safe scales, keyboard navigation, reduced-motion mode.
- 2D fallback without WebGL; under ~100k triangles at 60 fps.
- Offline showcase demo.

**Demo script:**
1. Outside thermal view → cut away inside.
2. Add loft insulation (roof cools, gas falls).
3. Add a heat pump (gas to zero, electricity rises, CO2 falls).
4. Add solar.
5. Scrub the timeline.
6. Pick a Pareto point.
7. Ask the LLM to explain.

**Done when:** the full journey works live on SH1–SH3 (submission item A), and the demo script runs without errors.

---

## Block 8 — Lloyds dashboard + geography
**Starts:** after B1; uses B3/B4 batch outputs · **Feeds:** the business story

**Goal:** turn home-level results into green-lending demand for Lloyds (business relevance, 10%). Every number is either **computed** by our models or an **explicit, labelled assumption**. Nothing is trained on synthetic labels.

**Build:**
1. **Batch scoring:** run score, improvement potential, package cost (sourced) and carbon over the EPC regions. Aggregate by LSOA and postcode district into `dashboard_areas`.
2. **Synthetic customer base, with a documented generating rule** (in `docs/SYNTHETIC_DATA.md`):
   - Customers per LSOA = real EPC home count × an assumed market share.
   - Each customer is a real EPC record from that LSOA, with no personal attributes.
   - Labelled "synthetic customer base over real geography".
3. **Geography analysis:**
   - Hotspots of poor homes (Getis-Ord Gi\*, local Moran's I via PySAL).
   - Regional heating and cooling degree-day trends with intervals (Mann–Kendall + Sen's slope).
   - Overheating-risk flag layer.
4. **Heat map** (MapLibre + deck.gl), switchable layers:
   - Share of green homes (score ≥ 65) and share of Poor/Very poor homes.
   - Finance demand (£ = package cost − grants), carbon potential, off-gas grant eligibility.
   - Synthetic customer density and in-app interest.
   - Hotspots and degree-day trends.
5. **Segments table:** e.g. "1930s semi, gas boiler, owner-occupier", with computed size, average loan need, carbon potential and share of Poor homes.
   - **Take-up** comes from the documented rule in `contracts/green_profits_assumptions.json`: a logistic function of model outputs with **assumption coefficients on sliders**, labelled "assumption".
6. **Green Profits:**
   - Lending (balance × margin + fees) + retention (churn reduction × CLV) + acquisition (new customers × CLV − acquisition and visit costs), with financed emissions alongside.
   - All parameters are sliders with stated ranges; Monte Carlo output is labelled "Illustrative scenario on a synthetic customer base. Not a forecast."
7. **Funnel:** users → profiles → plans → visits → applications → loans; cost per acquired customer and £ lent per area.
8. **GNN area model:**
   - GraphSAGE (PyTorch Geometric) on the LSOA adjacency graph.
   - **Features:** EPC stock mix, IMD, weather, off-gas share. **Target:** DESNZ mean domestic electricity and gas per meter.
   - **Uses:** area-level priors for the Bayesian hierarchy (B3); a "uses more than expected" map layer; filling suppressed LSOAs.
   - **Baseline:** EBM/LightGBM without the graph. **Validation:** spatial-block CV.
9. **Optional:** a portfolio Pareto front for allocating a green-finance budget across homes (multi-objective knapsack: ε-constraint or NSGA-II).

The idea that green homes carry lower credit risk is presented as a hypothesis, not a finding.

**Done when:** the dashboard runs on the three EPC regions through `/dashboard/*`, and every assumption is visibly labelled.

---

## Evaluation and submission

Each block writes its own `metrics.json` (seed, data version, baseline, held-out split). `scripts/eval_all.py` gathers them into `eval/reports/eval_v1.md` with one command (submission item G):

| Output | Metric (held-out) | Baseline | Block |
| --- | --- | --- | --- |
| Energy intensity | MAE/MAPE per fuel, 90% coverage | NEED median, XGBoost | B3a |
| Calibrated per-fuel energy | coverage ≈ 90%, CRPS, PIT, R-hat | degree-day OLS, LightGBM quantile, conformal | B3b |
| Missing-month imputation | MAE on masked months | seasonal-naive ×12 | B3c |
| Post-retrofit rating | MAE + coverage of SAP change on EPC before/after pairs | median band shift per measure | B2c |
| Score | 0 monotonicity violations, every heating type mapped, reproducible | — | B4 |
| Improvement-potential ranking | Spearman / NDCG vs labels | rank by total kWh | B4 |
| Retrofit relevance | AUC, precision@k vs real EPC recommendations; 0 infeasible recommended | rules only / recommend all | B4 |
| Pareto | hypervolume, % greedy picks dominated | greedy kWh/£, CO2-only | B4 |
| Value | coverage, pinball loss on real held-out sales; vs DECC 2013 | hedonic OLS without band | B5 |
| Value method check | recovery of the planted synthetic premium (method validation only) | — | B5 |
| Jev / router | routing accuracy, feasibility precision/recall, ECE, latency | rules only, Claude structured output | B6 |
| LLM | groundedness, accuracy, unsupported-claim rate | no validator | B6 |
| Synthetic data | KS / quantiles vs NEED | — | B2 |

Submission checklist:

| Item | Content | Block |
| --- | --- | --- |
| A | Working prototype | B7 + B6 |
| B | Code + scripts + README | all |
| C | Trained models in `artifacts/` | B2c, B3, B4, B5 |
| D | `docs/SCORE.md` | B4 |
| E | `docs/LLM.md` + prompts | B6 |
| F | Architecture diagram | B6 + B7 |
| G | Evaluation report | all |
| H | `docs/RESPONSIBLE_AI.md`: privacy (incl. interior-photo consent, blur, retention), size bias, explainability, hallucination, human adviser, data limits | B6 + B1 |
| I | Held-out test predictions in `deliverables/test_predictions/` + `docs/DATA.md` | B1 + B3/B4/B5 |

---

## Order of work

```
B1 ──► B2a twin ──► B2c rating ──► B2b generator ──► B3 calibration + prediction
B4a score (on recorded inputs) ──► B4c recommendations + Pareto (needs B3)
B5 hedonic (needs B1 sales; "after" needs B2c) ──► matching ──► DR-learner
B6a engine on mocks ──► B6b API ──► B6d LLM + validator ──► B6g eval set ──► B6c KB extras ──► B6e Jev ──► B6f memory
B7 screens 1–3 on mocks ──► live ──► 3D v1 ──► Pareto/loan ──► 3D v2 ──► 3D v3 ──► memory/capture
B8 after B1, fed by B3/B4 batch outputs
```

**Checkpoints:**
1. **End-to-end skeleton:** SH1–SH3 flow through real twin → score → recommendations → API → UI. Point predictions only; intervals may be placeholders.
2. **Complete submission:** Bayesian intervals, rating estimator, value ranges, grounded LLM with validator, evaluation report, docs D–I.
3. **Demo build:** 3D twin v1–v3, Pareto/loan, Lloyds dashboard, memory. Feature freeze.

**Stop rules:**
- If checkpoint 1 slips, drop Pareto P(on front), matching, the DR-learner and agent capture.
- If checkpoint 2 slips, drop Jev (keep the Claude fallback), the graph store, 3D v2/v3 and the portfolio front.

**Stretch only, after checkpoint 3:**
- Differentiable RC graph in JAX/diffrax (if Option A was chosen).
- GP emulator / BoTorch continuous sizing.
- UKCP18 lifetime savings.
- Physics-informed DeepONet surrogate.
- Home-level neighbourhood GraphSAGE (the LSOA-level GNN is in B8).
- Spatial PINO visual (gated ±5% vs Level 1).
- Automatic floor-plan segmentation without agent correction (photo-to-twin with agent review is in B7).
- NEED savings CATE vs physics.
- JEPA-Anything factorised load model (merged only if an ablation shows a gain).

---

## Compliance with the problem statement

The 10 Oct constraint review found 8 issues. All are fixed in `contracts/` and binding in `AGENTS.md`; each block must implement them:

| # | Issue | Fixed in | Implemented by |
| --- | --- | --- | --- |
| 1 | Projected score and value had no documented post-retrofit rating | `sap_estimator.json`, `RatingEstimate` | B2c → B4a, B5 |
| 2 | Emission factors and costs were unverified | `emission_factors.json`, `tariffs.json`, catalogue `cost_status` | B1 (gaps), B3c, B6a |
| 3 | Intensity was weather-normalised; PV overlap unjustified | `score_mapping.json` | B3c, B4a |
| 4 | Heating mapping incomplete | `score_mapping.json` + test | B4a |
| 5 | Take-up model on arbitrary synthetic labels | `green_profits_assumptions.json` (assumption-only rule) | B8 |
| 6 | Value framed as a reason to borrow | `ValueImpact.wording` | B5, B6d, B7 |
| 7 | Synthetic premium recovery could read as evidence | method check only | B2b, B5, evaluation |
| 8 | 3D colours looked measured | `ElementHeatLoss.estimated` + `basis` | B2a, B3b, B6d, B7 |

**Wording leftovers to fix in the docs:**
- `docs/STRATEGY_REPORT.md` line 43, customer journey: "value uplift range" → "indicative value range".
- Its open-questions list still asks about team size, GPU and region, which are all decided.
