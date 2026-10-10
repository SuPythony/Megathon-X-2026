# GreenTwin — Implementation Plan

Megathon X: Singularity · Problem Statement 2 (Lloyds) · Team MegaMind · v1.0 (10 Oct 2026)

GreenTwin gives every UK home a calibrated physics **digital twin**. Every score, saving, carbon figure and value range comes from documented models with Bayesian credible intervals. A tool-only LLM explains those numbers. A Three.js 3D twin lets customers add retrofits inside and outside the house. A Lloyds dashboard turns all of this into green-lending demand.

Related documents:
- **Strategy report** (living doc): https://claude.ai/code/artifact/5db02f30-587d-4e5a-9bec-16e560798f48. It has two tabs: "Strategy Report" and "System design and build plan".
- **Submission PDF:** `deliverables/M1.pdf`.
- **Contracts (shared data formats):** `contracts/`. The source of truth is `contracts/models.py`; the TypeScript mirror is `contracts/types.ts`.
- **Team knowledge:** `AGENTS.md`. Read it before starting work.

---

## 1. Team and ownership (5 members)

Keep this table current.

| ID | Name | Role | Owns (code) | Owns (deliverables) |
| --- | --- | --- | --- | --- |
| **P1** | A | Data & Geography engineer | `greentwin/data`, `greentwin/geo`, data scripts | Data sources/licences doc, schema doc, splits, maps, Lloyds heat-map layers |
| **P2** | B | Physics & Bayesian lead | `greentwin/twin`, `greentwin/bayes`, `greentwin/score`, `contracts/score_mapping.json` | Sub-score mapping doc (item D), synthetic-data generation doc, calibration report |
| **P3** | C | ML, Value & Causal engineer | `greentwin/models`, `greentwin/value`, `greentwin/optimise`, `eval/` metrics | Evaluation report (item G), held-out predictions (item I), value-method note |
| **P4** | D | LLM, Knowledge-base, Jev & API engineer | `greentwin/api`, `greentwin/kb`, `greentwin/llm`, `greentwin/jev`, `greentwin/retrofit` | LLM implementation doc (item E), adversarial scenario set, Responsible AI note (item H, with P1) |
| **P5** | E | Front-end & 3D engineer | `web/` (React + Vite + react-three-fiber) | Prototype (item A), demo script and video, architecture diagram (item F, with P4) |

Contract changes (`contracts/`) need two reviewers: the producer and the main consumer.

---

## 2. Repository layout

```
contracts/            Shared data formats (Pydantic + TS mirror + JSON configs + examples)
  models.py           SOURCE OF TRUTH for all data exchanged between modules
  types.ts            TypeScript mirror (front end)
  score_mapping.json  Fixed weights + monotonic sub-score mappings
  feasibility_rules.json  Layer-1 hard rules + retrofit ordering
  retrofit_catalogue.json Retrofits, twin parameter edits, grants (costs TO_SOURCE)
  examples/           3 showcase homes, 12-month usage, scenario spec, MOCK result
  schema/             Generated JSON Schema (python -m contracts.export_schemas)
greentwin/            Backend package (data, score, twin, bayes, retrofit, models, value, geo, optimise, kb, llm, jev, api)
scripts/              One-off/reproducible scripts (downloads, example generation)
data/                 raw/ interim/ processed/ cache/   (git-ignored except .gitkeep)
artifacts/            Trained models + metrics JSON (large binaries git-ignored)
cache/                llm/ jev/ response caches for replay (git-ignored)
eval/                 scenarios/ (adversarial LLM set), reports/ (metrics, plots)
web/                  React app: customer app, agent capture, 3D twin, Lloyds dashboard
deliverables/         M1.pdf and final submission artefacts
docs/                 This plan, method docs, data docs
tests/                Contract, score monotonicity, rules, twin, API tests
```

---

## 3. Shared data formats

All modules exchange the Pydantic models in `contracts/models.py`. JSON on the wire uses the same field names.

### 3.1 Conventions (non-negotiable)
- **Units:**
  - Energy in **kWh/yr**, unless the field name says otherwise.
  - Money in **GBP**; carbon in **kgCO2e**.
  - Area in **m2**; U-values in **W/m2K**; heat loss in **W/K**.
- **Savings = baseline − after.** Positive means a saving. Negative means an increase, for example electricity after a heat pump. Report electricity and gas separately, always.
- **Purchased energy** = grid electricity import + gas, in kWh/yr. Solar reduces imports, not demand.
- **Uncertain numbers are `Interval`:** `{mean, lo, hi, level=0.9, unit}`. A bare float is only allowed for deterministic quantities, such as the score formula on fixed inputs.
- **Missing values are `null`, never 0.** The exception is a true zero, such as a home with no gas supply.
- **`provenance.engine`** on every result says which model produced it. `mock` results must never reach the demo.
- **Leakage fields** (`epc.potential_*`) are reference labels only and are never model inputs. The schema descriptions mark them.
- **Values:**
  - `current_value_estimate_gbp` is the baseline valuation.
  - `SaleRecord.price_gbp` is historical only.
  - They are never interchanged.
- **Privacy:** location is the postcode district + LSOA, with lat/lon rounded to 3 dp. No full addresses, no names.

### 3.2 Entities

| Contract | Producer | Consumers | Key fields |
| --- | --- | --- | --- |
| `Property` | P1 (EPC/synthetic), P4 (agent/self-serve capture) | all | location, type/form/age, floor area, walls/roof/floor/glazing, heating, renewables, controls, constraints, value baseline, data_quality |
| `EnergyUsageMonth` | P1 | P2, P3 | month `YYYY-MM`, electricity_import_kwh, gas_kwh (gross CV), pv/export, estimated_read |
| `SaleRecord` | P1 | P3 | date, price, energy_rating_at_sale |
| `WeatherMonth` | P1 | P2, P3 | area_code, month, hdd_15_5, cdd_22, mean_temp, ghi, source |
| `RetrofitItem` (catalogue) | P2 + P4 | twin, rules, UI | twin_edits, blocked_by_rules, depends_on, excludes, cost range, grants, sizing |
| `ScenarioSpec` | UI, LLM/Jev router, batch | `run_scenario` | retrofits[{id,size,stage_year}], weather, n_samples, include_value/pareto, seed |
| `ScenarioResult` | `run_scenario` (P2 twin + P3 value/Pareto + P4 assembly) | UI, LLM, dashboard, eval | baseline/after per fuel, savings per fuel, improvement potential, carbon + factors, bills, score before/after with components, per-retrofit effects + feasibility + suitability, value range, element heat loss (3D), pareto, warnings |
| `WhatIfRoute` | Jev / Claude fallback | API | intent, confidence, ScenarioSpec, clarification |
| `ExplanationCheck` | validator | API, eval | numbers found, unsupported numbers, Jev probability, passed |
| `CustomerMemory` | API | optimiser, LLM, UI | facts, completed, rejected, saved scenarios |

### 3.3 Storage
- **DuckDB** `data/greentwin.duckdb`, with tables `properties`, `usage_monthly`, `sales`, `weather_monthly`, `scenario_results` (JSON column + flattened key metrics), `posteriors` (per-home parameter samples, Parquet), `memory`, `dashboard_areas`.
- **Parquet** in `data/processed/` for training tables. Each file has a sibling `*.meta.json` with source, licence, date, row count and split counts.
- **Splits:** assign once in `greentwin/data/splits.py`.
  - By `customer_id` when present, else `property_id`; ratio 70/15/15, seed 42.
  - For geography/GNN work, use spatial-block splits by postcode district.
  - Stored as `Property.split`. Never re-split.

### 3.4 API (FastAPI, `greentwin/api`)

| Method & path | Request | Response | Notes |
| --- | --- | --- | --- |
| `GET /homes/{id}` | — | `Property` + latest `ScoreResult` | `get_home_facts` tool |
| `POST /homes` | `Property` (+ usage[]) | `{property_id}` | agent capture / self-serve |
| `POST /scenario` | `ScenarioSpec` | `ScenarioResult` | `run_scenario`, cached by hash(spec + model versions) |
| `GET /homes/{id}/feasibility` | — | `Feasibility[]` | rules, then Jev layer 2 |
| `GET /homes/{id}/pareto` | `?budget=` | `ParetoPoint[]` | precomputed for showcase homes |
| `GET /homes/{id}/green-twins` | — | ranges of 3 matched greener homes | no addresses |
| `POST /chat` | `{property_id, message, history}` | `{answer, citations, check: ExplanationCheck, scenario_result_id?}` | tool-only LLM + validator |
| `POST /route` | `{property_id, message}` | `WhatIfRoute` | Jev or fallback |
| `GET /dashboard/areas` | `?layer=&region=` | GeoJSON features with layer values | Lloyds heat map |
| `GET /dashboard/segments` | — | segment table | |
| `POST /dashboard/green-profits` | assumption sliders | Monte Carlo ranges | |
| `GET /memory/{customer_id}`, `POST /memory/{customer_id}` | `CustomerMemory` | | |

The front end develops against `contracts/examples/*` and a mock server (`/scenario` returns the MOCK result) until the real engine lands.

---

## 4. Inputs and factors to consider

### 4.1 Property and fabric
- Type, built form, attachment, and exposed vs party walls.
- Construction age band, storeys, floor area, room count, ceiling height (RdSAP default).
- Walls: type, insulation, U-value.
- Roof: type, loft depth, room-in-roof.
- Floor: type, insulation.
- Glazing: type, share, window-to-floor ratio, frame.
- Doors; draught-proofing share; ventilation and air-change rate; thermal bridging allowance; thermal mass.

### 4.2 Systems
- Main heating system, fuel, age, seasonal efficiency or SCOP, emitters (radiator sizing for heat pumps).
- Hot water source, cylinder and insulation.
- Secondary heating; controls (programmer, thermostat, TRVs, smart, zoning).
- PV kWp, azimuth, tilt and shading; battery; solar thermal; smart meter.

### 4.3 Occupancy and behaviour (latent, calibrated)
- Heating setpoint and hours.
- Occupants and hot-water demand; appliance and lighting baseload.
- Rebound after retrofit (comfort take-back), modelled as a prior on setpoint shift.

### 4.4 Weather and climate
- Monthly HDD (base 15.5 °C) and CDD by area; hourly temperature and irradiance for the hourly twin.
- Typical vs cold vs warm years; forecast week; UKCP18/CMIP6 futures.
- Regional degree-day trends; hot days (overheating).

### 4.5 Geography and planning
- Region, LSOA, local authority, deprivation index, altitude and exposure.
- Conservation area, listed building, leasehold or freeholder consent.
- Outdoor space for a heat-pump unit, roof access and orientation (OSM footprints).
- Off-gas-grid status; urban vs rural.

### 4.6 Economics and policy
- Tariffs: Ofgem cap unit rates and standing charges (verify VAT basis); export tariff.
- Retrofit costs (BEIS 2017 + CPI), lifetimes and maintenance.
- Grants: BUS £7,500, or £9,000 off-gas to 31 Mar 2027; ECO4/GBIS, to verify.
- Discount rate for NPV; loan APR and term for the green-finance view; MEES / EPC C context.

### 4.7 Carbon
- DESNZ 2025 factors: electricity about 0.177 kgCO2e/kWh (verify).
- Gas: gross vs net CV, about 0.183 vs 0.2049. Meter kWh are gross CV, so use the gross factor (verify).
- Projected grid factors for lifetime carbon.

### 4.8 Market and value
- Baseline valuation, local HPI index by type, sale date and period effects, location premium.
- EPC band at sale; green-twin matched premiums; confounders such as other renovations.

### 4.9 Data quality and uncertainty
- Months of metered data, estimated reads, EPC age, imputed fields, conflicts (e.g. rating A but 400 kWh/m2), out-of-distribution flags.
- Sources of uncertainty:
  - Parameter (posterior).
  - Model discrepancy (GP).
  - Weather variability.
  - Behaviour.
  - Cost ranges.
  - Value model.

### 4.10 Customer and lending
- Constraints: renting, budget, disruption tolerance, planned move.
- Preferences learned from accept/reject (memory).
- Customer segment, existing Lloyds products (synthetic), take-up propensity.

---

## 5. Models to build and train

**Physics before ML.** Learned models never produce customer savings directly. They calibrate, explain, rank, estimate value, or act as a surrogate of the twin.

| ID | Model | Kind | Inputs → Target | Data | Method / lib | Metric (held-out) | Baseline | Owner | Tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01 | Green Home Score | Deterministic | Property + annualised intensity → score, band, contributions | — | `score_mapping.json` | Monotonicity sweep: 0 violations; reproducibility | — | P2 | A |
| M02 | Twin v1 (steady-state RC element model) | Physics | U·A per element, ventilation, degree days, system efficiency/COP, PV yield → per-fuel kWh, element heat loss | RdSAP defaults, PVGIS | numpy | Error vs NEED intensity distributions | NEED medians | P2 | A |
| M03 | Synthetic generator | Simulator | Sampled homes (EPC/EHS joint dists) + weather + behaviour noise → monthly usage, per-retrofit reference labels, synthetic values | EPC, EHS, NEED priors, M10 | numpy | Distribution match vs NEED (KS test, quantiles) | — | P2 | A |
| M04 | Energy-intensity model | ML | Property + weather/geo features → annual elec & gas kWh/m2 (homes without meters; prior mean) | NEED 50k (real) + EPC | **EBM** (InterpretML), XGBoost baseline | MAE/MAPE per fuel, 90% coverage (conformal) | NEED median by type/age/region | P3 | A |
| M05 | Improvement-potential ranking | Derived + check | Twin before/after feasible package → % purchased-energy reduction | Synthetic labels | Twin + LightGBM ranker (ablation) | Spearman/NDCG vs reference | Rank by total kWh | P3 | A |
| M06 | Value model | ML | Size, type, age, location, period, HPI, EPC band → price quantiles | UCL EPC–PPD linked | EBM / LightGBM quantile + conformal | Interval coverage, pinball loss | Hedonic OLS without band | P3 | A |
| M07 | Retrofit relevance prior | ML | Property features → P(EPC would recommend measure X) | EPC recommendations table | EBM classifier per measure | AUC, precision@k vs EPC recs | Rules only | P3 | B |
| M08 | Missing-month imputation | Stats | Available months + HDD → missing months | NEED/London subsample (masked) | Degree-day regression (GP stretch) | MAE on masked months | ×12 seasonal-naive | P2 | C |
| M09 | Bayesian calibration | Bayesian | Priors on UA, setpoint, baseload, COP + monthly meter data → posterior samples | Showcase + NEED-like synthetic + real where meters exist | NumPyro NUTS offline, SVI per home; hierarchy type×region | 90% coverage ≈ 90%, CRPS, R-hat < 1.01 | LightGBM quantile; conformal | P2 | C |
| M10 | Green-twin matching | Causal (design) | Matched pairs on type/form/age/tenure/LSOA/quarter/area → Δlog price, Δcost per 10 score pts | UCL linked | CEM + NN matching, bootstrap | SMD < 0.1, CI width, vs Nationwide/DECC | Hedonic coefficient | P3 | C |
| M11 | DR-learner value uplift | Causal ML | Repeat sales, treatment = band improved → CATE on Δlog price | UCL linked | EconML DRLearner (+ BCF optional) | Recovery of known synthetic premium; BLP/GATES; refutations | Matching (M10) | P3 | C |
| M12 | Pareto optimiser | Optimisation | Feasible subsets × posterior samples → front, P(on front), knee, staging | M02/M09 | Enumeration + non-dominated sort; DP staging | Hypervolume vs greedy | Greedy kWh/£ | P3 | C |
| M13 | Jev threshold calibration | Calibration | Jev confidences on labelled routes/feasibility → calibrated P | ~50 prompts, ~200 home×retrofit pairs | Isotonic regression | ECE, reliability, routing accuracy ≥ 95% above τ | Claude structured output | P4 | C |
| M14 | Lloyds take-up / uplift | ML (synthetic) | Synthetic customer features → P(take-up), uplift | Synthetic customer base | EconML/CausalML uplift | Qini/AUUC | Random targeting | P3 | Could |
| M15 | NEED savings CATE | Causal ML | Insulation installed (treatment) → Δgas | NEED 50k | DR/X-learner | vs twin predictions by archetype | Twin only | P3 | Could |
| M16 | Differentiable RC graph (twin v2) | Physics-ML | EPC element features → conductances/capacitances (MLP/GNN) → hourly ODE | Meter data + synthetic | PyTorch + torchdiffeq (GPU) | Per-fuel MAE, monotonic sweep | Twin v1 | P2 | D |
| M17 | DeepONet surrogate | Neural operator | Weather series + twin params → hourly series | Synthetic from M16 | DeepXDE/PyTorch (GPU) | Rel. L2, speed-up, 0 monotonic violations | GP emulator | P2 | D |
| M18 | GraphSAGE neighbourhood | GNN | Homes graph (party walls, proximity) → imputed fields, peer benchmark | EPC region subset | PyG (GPU) | MAE, spatial-block CV | kNN | P3 | D |
| M19 | Spatial PINO (visual only) | Neural operator | Archetype sections → heat-flow field | FEM solves (scikit-fem) | neuraloperator (GPU) | ±5% vs L1 element heat loss | L1 colouring | P2 | D |
| M20 | Photo-to-twin vision | CV | Floor-plan photo → rooms/areas; room photos → element tags + confidence | CubiCasa5k (+ labelled photos) | Segmentation model / VLM (GPU) | IoU/area error; tag accuracy; ECE | Manual tagging (v1) | P5+P4 | D |

**GPU:** a GPU is available. Use it for M16–M20 (PyTorch CUDA).
- JAX GPU wheels are Linux-only, so run GPU NumPyro inside WSL2 or on Linux. On Windows, NumPyro on CPU is fine for the per-home SVI.
- Torch models use `GREENTWIN_DEVICE` from `.env`.

**Artefacts:** `artifacts/<model_id>/<version>/`, containing `model.*`, `metrics.json`, `config.json` and `data_version.txt`. Every metric reported in the evaluation report must be reproducible with `scripts/train_<model_id>.py --seed 42`.

---

## 6. Work plan by checkpoint

Order follows the build order in the strategy report. Each checkpoint leaves a submittable system.

### Tier A — required outputs end to end → Checkpoint A (3 showcase homes)

| # | Task | Owner | Done when |
| --- | --- | --- | --- |
| A1 | Env: `uv` + Python 3.12; `uv sync`; CI runs `pytest` | P4 | Fresh clone → tests green |
| A2 | Data downloads: EPC (regions: North West, South West, London), NEED 50k, UCL linked (854942 / 857911), UK HPI, Open-Meteo/HadUK-Grid HDD, ONS postcode→LSOA. Record licences in `docs/DATA.md` | P1 | Parquet + meta.json present; licences documented |
| A3 | Splits by customer/property; leakage field audit | P1 | `splits.py` + test that no id spans splits |
| A4 | Score module (M01) from `score_mapping.json` + monotonic sweep test | P2 | Showcase scores computed; tests pass |
| A5 | Twin v1 (M02): element heat loss, per-fuel conversion, PV via PVGIS, emission factors config | P2 | SH1–SH3 baselines within ±15% of their metered annual totals |
| A6 | Rules engine + catalogue loader + ordering (`feasibility_rules.json`) | P4 | SH1 blocks EWI; SH3 blocks loft/PV, ASHP = needs_survey |
| A7 | `run_scenario` assembling `ScenarioResult`; DuckDB store; mock server for UI | P4 | Real result for SH1–SH3 validates against contracts |
| A8 | Synthetic generator (M03) + reference labels; validation vs NEED | P2 | Distribution report in `eval/reports/` |
| A9 | M04 EBM + XGBoost + NEED-median baseline; M05 ranking | P3 | Metrics JSON + comparison table |
| A10 | M06 value model with quantile ranges, labelled indicative | P3 | SH1–SH3 value ranges + coverage on held-out sales |
| A11 | UI shell: home page, score + contribution waterfall, recommendations list, per-fuel savings, value range (against mocks, then live) | P5 | Works on all 3 homes with live API |

### Tier B — grounded LLM + evaluation → Checkpoint B (complete submission)

| # | Task | Owner | Done when |
| --- | --- | --- | --- |
| B1 | Tool-only LLM: tools `get_home_facts`, `get_results`, `run_scenario`, `search_docs`; system prompt rules; numeric validator; template fallback without key | P4 | 0 unvalidated numbers on 20 scenarios |
| B2 | What-if chat in UI; shows scenario JSON → numbers → explanation with source tags | P5 | Demo what-if works on SH2 |
| B3 | Eval harness: held-out metrics vs baselines for M04/M05/M06; 30 adversarial LLM scenarios with groundedness, accuracy and unsupported-claim rates | P3 + P4 | `eval/reports/eval_v1.md` generated by one command |

### Tier C — differentiators, in order → Checkpoint C (demo build, feature freeze)

| # | Task | Owner |
| --- | --- | --- |
| C1 | Weather normalisation + missing-month handling (M08); data-confidence score | P1 + P2 |
| C2 | Bayesian calibration (M09) → intervals everywhere, P(suitable) | P2 |
| C3 | Pareto (M12) + P(on front) + staging; precompute showcase packages | P3 |
| C4 | Geography: LSOA maps, hotspots, HDD trends; Lloyds dashboard API (layers, segments, Green Profits Monte Carlo) | P1 (+P3 for segments) |
| C5 | 3D twin v1 → v2 → v3 (outside/inside, retrofit objects, timeline) | P5 |
| C6 | Lloyds dashboard UI: heat map with switchable layers, segments table, Green Profits view, funnel | P5 (+P1) |
| C7 | Knowledge-base extras: vector docs (methods, grants, guidance), embedded graph of rules/grants; Jev router + feasibility layer 2 + explanation check **if `JEV_API_KEY` exists**, else Claude fallback (M13) | P4 |
| C8 | Green-twin matching (M10) + DR-learner (M11) + "green twins of your home" | P3 |
| C9 | Customer memory + agent capture v1 (photo upload, manual tags, agent confirmation) | P4 + P5 |

### Tier D — stretch (only after Checkpoint C)
M16 → M17 → M18/M19/M20, UKCP18 lifetime savings and the overheating flag, GP discrepancy, M14/M15.

### Parallel start
- **P1:** A2 → A3 → C1 → C4.
- **P2:** A4 → A5 → A8 → C2.
- **P3:** A9 → A10 → B3 → C3 → C8.
- **P4:** A1 → A6 → A7 → B1 → C7 → C9.
- **P5:** A11 on mocks → B2 → C5 → C6.

### Stop rules
- If Checkpoint A slips past about a third of the time, Tier C shrinks to C1–C3.
- If Checkpoint B is late, drop the 3D v2/v3 work and Jev.
- Documentation items D–I are written alongside each task.

---

## 7. LLM design (P4)

- **Model:** `LLM_MODEL` from `.env`; default `claude-sonnet-5-5`. The key comes from `ANTHROPIC_API_KEY`, which is a placeholder for now: ask the team lead when needed.
- **Tools only:**
  - `get_home_facts(property_id)`
  - `get_results(scenario_id)`
  - `run_scenario(ScenarioSpec)`
  - `search_docs(query)` (cited text)
  - `query_graph(template_id, params)` (templated only)
- **System prompt rules:**
  - Never compute or estimate numbers.
  - Quote numbers only from tool JSON, with the unit and the interval.
  - Say "estimate" and "indicative" where they apply.
  - Rerun `run_scenario` for every what-if.
  - State missing or conflicting data and infeasible retrofits plainly.
  - No financial advice; loans are illustrative.
- **Validator:**
  - Regex-extract every number from the answer and match it, with a tolerance, against the tool outputs of that turn.
  - On failure, regenerate once, then fall back to the template explainer.
  - Optional Jev check: "is every claim supported by this JSON?"
- **Caching:** hash(prompt + tool outputs + model) → `cache/llm/`, so eval runs are replayable.
- **Eval set:** `eval/scenarios/*.json`, at least 30 and growing to 100. It covers:
  - Missing months.
  - Conflicting data.
  - Infeasible requests ("heat pump in my 3rd-floor flat").
  - Electricity rising after a heat pump.
  - Insufficient data for value.
  - Out-of-scope questions.
- **Targets:** groundedness ≥ 90%, factual accuracy ≥ 90%, unsupported claims < 5%.

## 8. Jev integration (P4)

- **Client:**
  - `greentwin/jev/client.py` → `POST {JEV_BASE_URL}` with `Authorization: Bearer {JEV_API_KEY}`, model `JEV_MODEL`.
  - Questions are typed: Choice, Score, yes/no.
  - Every call is cached in `cache/jev/`.
- **Uses:**
  1. What-if routing → `WhatIfRoute`.
  2. Feasibility layer 2 → `Feasibility` (status + confidence), only for retrofits that pass the rules.
  3. Explanation check.
- **Jev never produces kWh, £, CO2 or score values.**
- **`JEV_ENABLED=false` or no key:** use the same interface with Claude structured output (`router="claude_fallback"`), or rules only.
- **Calibration (M13):** isotonic recalibration; τ chosen so decisions above it are ≥ 95% correct; reliability diagram in the evaluation report.
- **Suitability:** P(suitable) = P(feasible) × P(net benefit). Rule vetoes always win. Low confidence → `needs_survey`.

---

## 9. UI (P5): screens, components, data

**Stack:**
- React + Vite + TypeScript.
- react-three-fiber + drei for 3D.
- MapLibre + deck.gl for maps.
- Recharts or visx for charts.
- Zustand for state.
- Types from `contracts/types.ts`.

### 9.1 Customer app

| Screen | What it shows | Data |
| --- | --- | --- |
| Onboarding / home profile | EPC lookup by postcode (EPC API) or manual form; optional photos; constraints and preferences | `Property` draft → `POST /homes` |
| Score | Score + band with interval, band probabilities, **waterfall of 7 component contributions**, key issues, data-confidence badge, warnings | `ScoreResult` |
| 3D twin | Outside/inside cutaway, thermal X-ray (element heat-loss share), before/after split, retrofit palette (feasible / needs survey / blocked with reason), live numbers panel, timeline scrubber, sun path, weather slider | `ScenarioResult.element_heat_loss_*`, `retrofits[]`, cached packages |
| Retrofit plan | Prioritised list with suitability, per-fuel marginal savings (negative shown red), carbon, bills, cost, grants; staged roadmap | `RetrofitEffect[]` |
| Pareto | Cost vs CO2 scatter (colour = value uplift, size = P(on front)), knee highlighted, budget slider; click → loads the package into 3D | `ParetoPoint[]` |
| Green loan | Package cost − grants = indicative loan; repayment vs monthly bill saving; value range; "book adviser". Disclaimer: illustrative, not an offer | `ScenarioResult` + loan assumptions |
| Green twins | 3 matched greener homes as ranges (price/m2, energy cost, score), never addresses | `/green-twins` |
| What-if chat | Message → routed intent + confidence chip → computed numbers card → explanation with source tags; validator badge | `/chat`, `/route` |
| History / memory | Saved plans, completed retrofits, rejected options with reasons; how recommendations changed | `CustomerMemory` |

### 9.2 Agent capture (tablet)
1. Pre-fill from EPC.
2. Floor-plan photo + room photos.
3. Element tag checklist (v1 manual; v2 auto-tags with confidence).
4. Agent confirms each field.
5. Consent and blur notice.
6. Twin created; open the 3D view with the customer.

### 9.3 Lloyds dashboard
- **Locality heat map** by LSOA / postcode district, with switchable layers:
  - Share of green homes (score ≥ 65).
  - Share of Poor/Very poor homes.
  - Customers interested (what-ifs, saved plans, adviser requests).
  - Retrofit finance demand (£).
  - Carbon potential.
  - Grant eligibility (off-gas).
  - Lloyds customer density (synthetic).
- **Segments table**: size, average loan need, take-up probability, carbon potential.
- **Green Profits view**: lending margin + retention + acquisition − costs. Sliders for assumptions; Monte Carlo ranges; financed-emissions reduction alongside.
- **Funnel**: users → profiles → plans → visits → applications → loans; cost per acquired customer.
- Everything is labelled "synthetic customer base".

### 9.4 UX rules
- Every number shows its interval and unit; every colour is paired with a number.
- Colour-blind-safe scales; keyboard navigation; reduced-motion mode; 2D fallback without WebGL.
- Showcase homes use precomputed results, so the demo works offline.

---

## 10. Showcase homes (synthetic archetypes, `contracts/examples/`)

| ID | Home | Why it is in the demo |
| --- | --- | --- |
| SH1 | 1905 solid-brick mid-terrace, Manchester (M14), 85 m2, EPC E/48, loft 50 mm, mostly single glazing, 22-yr non-condensing gas boiler, conservation area; 1 missing month | Conservation area blocks EWI → internal wall insulation; fabric-first then heat pump; missing-data handling |
| SH2 | 1972 cavity semi, Bristol area (BS16), 95 m2, EPC D/60, unfilled cavity, loft 100 mm, double glazing pre-2002, condensing combi | Everything feasible: Pareto, green loan, electricity rises after heat pump (negative saving) |
| SH3 | 2004 3rd-floor flat, Croydon (CR0), 62 m2, EPC C/70, all-electric storage heaters, leasehold, no roof access | Rules block loft/PV; heat pump `needs_survey` (Jev + agent); gas baseline 0 → per-fuel handling |

Regenerate the files with `python scripts/make_showcase_examples.py`. The mock result is shape-only.

---

## 11. Submission mapping (problem statement §13)

| Item | Content | Owner | Location |
| --- | --- | --- | --- |
| A | Working prototype | P5 | `web/` + `greentwin/api` |
| B | Source, data scripts, run instructions | all | repo, `README.md` |
| C | Trained models + inference code | P2/P3 | `artifacts/`, `greentwin/` |
| D | Sub-score mappings | P2 | `contracts/score_mapping.json`, `docs/SCORE.md` |
| E | LLM prompts, model/API, eval approach | P4 | `greentwin/llm/prompts/`, `docs/LLM.md` |
| F | Architecture diagram | P4 + P5 | `docs/architecture.png`, M1.pdf |
| G | Evaluation report on validation set | P3 | `eval/reports/` |
| H | Responsible AI note | P4 + P1 | `docs/RESPONSIBLE_AI.md` |
| I | Held-out test predictions + schema/label docs | P3 | `deliverables/test_predictions/`, `docs/DATA.md` |

---

## 12. Keys and secrets
- `.env.example` lists every key. Values marked `__ASK_USER__` are not available yet.
- **Ask the team lead when a task first needs one.** Never hard-code, commit or log keys.
- Code must degrade gracefully without keys:
  - The LLM falls back to the template explainer.
  - Jev falls back to Claude or rules.
  - EPC falls back to cached sample files.

## 13. Definition of done (every task)
- Inputs and outputs use `contracts/` models, and validation passes.
- Tests added: unit tests, plus a monotonicity test wherever a model sets savings or score.
- Metrics JSON written to `artifacts/` or `eval/reports/` with seed and data version.
- The doc section for the matching submission item is updated.
- No new number reaches the UI or LLM without `provenance`.

## 14. Risks
| Risk | Mitigation |
| --- | --- |
| Too many techniques | Tiers + checkpoints; Could/D tiers only after Checkpoint C |
| Python 3.14 on the dev machine | Pinned 3.12 via `uv` |
| EPC login delay | Start A2 first; cached samples; NEED + UCL don't need One Login |
| Synthetic labels mirror the generator | Validate vs NEED; say so in the report |
| Value read as causal | Ranges, "indicative", sensitivity + refutation tests |
| LLM/Jev keys unavailable | Fallbacks behind the same interface |
| Demo failure | Precomputed showcase packages, offline mode, recorded video |
