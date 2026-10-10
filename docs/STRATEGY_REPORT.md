<!-- Exported from the GreenTwin strategy doc (Claude Docs artifact https://claude.ai/artifact/Ca1BntuvXtnKYtD4Uuz1GT), tab "System design and build plan", rev 34, on 2026-10-10. Embedded charts/diagrams are replaced by placeholders; the architecture, two-level and confidence diagrams are reproduced in deliverables/M1.pdf. -->

# System design and build plan

Oct 10, 2026 · @Sumanyu Aggarwal

## Recommendation

Build a weather- and location-aware **differentiable physics twin** of each home, calibrated with **Bayesian inference**, explained by an **EBM**, queried through a **hybrid knowledge base**, and shown as a **3D twin** whose numbers all come from the model. Every output carries a posterior (mean + 90% credible interval), and every retrofit's suitability is a probability.

| Tier | Techniques | Why |
| --- | --- | --- |
| Must | Differentiable RC physics twin (a small physics graph); physics-based synthetic data generator; Bayesian calibration (NumPyro); EBM primary + XGBoost baseline; weather + geography features; SQL + vector knowledge base with tool-only LLM and validator | Covers 65% of the rubric; each piece is achievable and testable |
| Should | Geography analysis (maps, hotspots, spatial hierarchy); GP discrepancy + missing-month imputation; 3D twin UI with retrofit toggles; graph DB for retrofit rules, constraints and grants; multi-objective Pareto front with P(on front); DR-learner value uplift | Strong differentiation at moderate risk |
| Could | Neighbourhood GraphSAGE; GP emulator + Bayesian optimisation for continuous sizing; future-climate extrapolation (UKCP18); customer memory; NSGA/BoTorch continuous sizing; portfolio-level Pareto front; neural-operator surrogate (physics-informed DeepONet); spatial PINO heat-flow visual; Jev System-1 routing and feasibility layer; NEED savings CATE vs physics; Lloyds uplift targeting | Add only after the gate at the end of Phase 2 |
| Won't (this hackathon) | Room-level multi-zone graphs; LV-grid graph; JEPA-Anything in the core path | Not identifiable from public data, or too immature |

**EBM vs XGBoost:** use **EBM** (InterpretML Explainable Boosting Machine) as the primary learned model. It is a glass-box additive model with exact per-feature and pairwise contributions, close to XGBoost accuracy on tabular data, and it supports monotonic shape functions. Its contributions can be stored as plain numbers in the knowledge base for the LLM to quote. Keep XGBoost with `monotone_constraints` + SHAP as the benchmark, and report both.

## Business plan: Lloyds green home lending

GreenTwin is a free customer-acquisition and green-lending tool for Lloyds: customers see what their home could gain from retrofits and how a green home or retrofit loan would pay for it, and Lloyds sees where to target. The plan is built on sections 2, 3, 4, 7 and 10 of the problem statement.

| Problem-statement section | Role in the business plan |
| --- | --- |
| 2 — Green Home Score | Shareable headline score that starts every conversation |
| 3 — Poorly performing homes | Finds the homes and areas with the most improvement potential, i.e. loan demand |
| 4 — Retrofit recommendations | The prioritised, feasible package becomes the loan's purpose and amount |
| 7 — Property value | Indicative value range, based on historical associations in sales data, shown for information only; never a promised increase or a reason to borrow |
| 10 — Customer memory | Keeps each customer's plan, constraints and responses, so follow-ups and offers improve over time |

**Two channels.**

1. **Agent-assisted.** A Lloyds agent visits, photographs the floor plan and every room, and the app builds the digital twin on the spot. The customer sees their score, the retrofit path on the 3D house, and what a green loan would mean for them.
2. **Self-serve.** The customer looks up their EPC by postcode, uploads a home profile (and optional photos), and explores improvements themselves; a "talk to an adviser" button turns interest into a lead.

**Photo to digital twin.**

- **v1 (build):** the agent uploads the floor-plan photo and room photos and tags elements manually (window type, boiler, radiators, visible loft insulation depth); the app pre-fills from EPC, and the agent confirms each field.
- **v2 (stretch):** floor-plan parsing (a model trained on the open CubiCasa5k floor-plan dataset) gives room polygons and areas; a vision model tags elements in room photos with confidence scores; low-confidence fields go back to the agent. Floor area is cross-checked against the EPC.
- Confirmed fields fill the property schema, which drives both the physics twin and the 3D geometry.

**Customer journey.** Score and key issues → recommended retrofit path on the 3D house → package cost minus grants = indicative loan amount → monthly repayment vs monthly bill saving ("cash-flow positive?") → value uplift range → save the plan or book an adviser. Loan figures are illustrative, not an offer or advice; any credit decision follows Lloyds' regulated process with a human adviser.

**Lloyds dashboard.**

- **Locality heat map (centrepiece):** LSOA or postcode-district map with switchable layers: share of green homes (score ≥ 65); share of Poor/Very poor homes; customers interested in going green (app what-ifs run, plans saved, adviser requests); estimated retrofit finance demand (£); carbon reduction potential; off-gas-grid / heat-pump grant eligibility; density of existing Lloyds customers.
- **Segments:** e.g. "1930s semi, gas boiler, owner-occupier, low loan-to-value", each with size, average loan need, take-up from a documented scenario rule (coefficients are labelled assumptions on sliders; no model trained on synthetic labels) and carbon potential, ranked for outreach.
- **Green Profits view:** how incentivising green homes earns money for Lloyds through three channels: green lending (loan balances × net interest margin, plus fees); retention (lower churn among mortgage customers who retrofit, valued as customer lifetime value); acquisition (app users who become customers, net of acquisition cost and agent-visit cost). Financed-emissions reduction is shown alongside for net-zero reporting. Every assumption (take-up, margin, churn, visit cost) is a slider, and results are shown as ranges from Monte Carlo over those assumptions.

**Funnel metrics.** App users → completed home profiles → saved plans → adviser or agent visits → loan applications → loans drawn, plus the cost per acquired customer and £ lent per area.

**Data and responsible AI.** The problem statement forbids real customer data, so the dashboard runs on a synthetic Lloyds customer base laid over real geography and real EPC distributions per area, labelled as synthetic. Interior photos are personal data: explicit consent, faces and personal items blurred, a stated retention period, and use only for building the twin. Greener-home profit claims are illustrative scenarios, and the higher-collateral-value or lower-default benefit of green homes is presented as a hypothesis, not a finding.

## Confidence map

Each technique gets a Bayesian confidence score: a Beta prior on the chance of finishing it well, based on our judgement of effort, data availability and maturity. Tiers then weigh that against rubric value.

> _Figure: Team judgement, 10 Oct 2026 · 15 techniques · Beta(10) priors (rendered in the live doc and in deliverables/M1.pdf)_

These are priors, not measurements. Update them at each gate: a finished milestone is evidence for success, a blocked one against, which turns the plan itself into a Bayesian update.

## System architecture

Data and environment feed one physics twin; calibration, models and analysis all build on it; their outputs land in a knowledge base that the LLM can reach only through tools.

> _Figure: System architecture · 6 layers, from data to UI (rendered in the live doc and in deliverables/M1.pdf)_

A what-if from the 3D twin or the chat calls `run_scenario`, which edits the twin's graph, reruns calibration samples and the downstream models, and writes fresh results to SQL before the LLM explains them.

## Component designs

Each component below states what it is, how to build it, and the library to use.

### 1. Differentiable physics twin (physics graph)

- Per home: a 4–6 node RC network (indoor air, walls, roof, floor, glazing, outside); edges are conductances U × A, nodes have heat capacities.
- A small GNN/MLP maps EPC element features to conductances and capacitances (softplus keeps them positive).
- Solved as an ODE (`torchdiffeq`) for hourly runs, or in steady-state degree-day form for monthly runs; outputs space heating, hot water, cooling, appliances, then per-fuel via the heating system (boiler efficiency, heat-pump COP).
- Retrofits are edits to the graph: insulation lowers a roof or wall conductance, glazing lowers the window edge, a heat pump changes the fuel conversion, PV adds generation at the electricity node.
- Guarantees: energy conservation and monotonicity by construction.

### 2. Environment and geography inputs

- Weather: hourly temperature and irradiance from ERA5 or NASA POWER; regional degree days; PVGIS for PV yield by roof orientation.
- Geography: postcode → LSOA/local authority via the ONS Postcode Directory; deprivation index; conservation areas and listed buildings (planning.data.gov.uk, Historic England) for feasibility; off-gas-grid postcodes for the £9,000 heat-pump grant; altitude and exposure.
- Building footprints (OpenStreetMap) give roof azimuth for PV and attachment (party walls) for heat loss.

### 3. Physics-based synthetic data and extrapolation

- Sample homes from EPC/EHS joint distributions (type × age × wall × heating × region), attach real weather, run the twin, add behaviour noise (setpoint, occupancy, appliance load) drawn from NEED-calibrated priors.
- Outputs monthly electricity and gas, plus per-retrofit reference labels for savings and improvement potential; all rules are disclosed.
- Extrapolation: the physics twin, not the ML, runs scenarios outside the data (new retrofit combinations, future climate from UKCP18 warmer winters and hotter summers, cooling demand).
- Out-of-distribution flag: GP predictive variance and EBM data density mark when a home is far from the training data.
- Validate the generator against real NEED distributions (energy intensity by type and age) before using its labels.

### 4. Bayesian calibration

- NumPyro model: priors on conductances, setpoint, baseload and COP; likelihood from the home's metered monthly fuel use given weather.
- Kennedy–O'Hagan form: metered = twin + GP discrepancy + noise.
- Hierarchical pooling by property type × region; offline fit, per-home updates by variational inference or importance reweighting (< 1 s).

### 5. Learned models: EBM primary, XGBoost baseline

- EBM predicts the residual between the twin and metered energy (behaviour and unmodelled effects), and the property value.
- Monotonic shape functions where the brief requires them (better insulation never raises demand).
- Per-home contributions are stored as numbers in the knowledge base; the dashboard shows them as a contribution chart.
- Baseline: XGBoost with `monotone_constraints` + SHAP; report accuracy and explanation quality side by side.

### 6. Gaussian processes and surrogates

- GP discrepancy term in calibration; per-home GP to impute missing months (yearly periodic + degree-day kernels).
- Spatial Matérn GP as the location effect in the value model (sparse GP via GPyTorch).
- If the twin runs hourly over many scenarios: a GP emulator trained on a Latin hypercube design gives millisecond what-ifs for the 3D UI sliders; BoTorch for continuous sizing (insulation thickness, PV kWp).

### 7. Physics graph NN for retrofit analysis

- The twin in section 1 is the physics graph NN: message passing follows heat flow along conductance edges.
- Retrofit analysis = graph edits + rerun; interactions (insulation before heat pump) emerge from the shared graph.
- Optional neighbourhood graph (homes linked by party walls, proximity, similarity) with GraphSAGE for imputation and peer benchmarks; spatial-block splits to avoid leakage.

### 8. Geography-based analysis

- Maps (MapLibre + deck.gl) of score, improvement potential and carbon by LSOA.
- Hotspot detection (Getis-Ord Gi\*, local Moran's I via PySAL) to find clusters of poor homes.
- Spatial hierarchy in the Bayesian model (home → LSOA → region) so sparse areas borrow strength.
- Portfolio view for Lloyds: where retrofit finance would cut the most carbon per £.

### 9. Knowledge base for the LLM

| Store | Holds | Tool the LLM calls |
| --- | --- | --- |
| SQL (DuckDB/Postgres) | Property facts, model outputs, posteriors, EBM contributions; the only source of numbers | `get_home_facts`, `get_results` |
| Vector DB (pgvector or Qdrant) | Methodology docs, retrofit guidance, grant rules, EPC recommendation text | `search_docs` |
| Graph DB (Neo4j or embedded Kùzu) | Home → elements, Home → LSOA → region, Retrofit → requires/blocked-by, Grant → applies-to | `query_graph` (templated Cypher only) |
| Scenario engine | Reruns twin + models for a what-if | `run_scenario` |

GraphRAG pattern: the LLM retrieves the home's subgraph (constraints, eligible grants, blocked retrofits) plus cited text, then explains numbers returned by tools. A validator checks every number against tool outputs.

### 10. 3D digital twin UI

- React + Three.js (react-three-fiber): a parametric house built from EPC type, floors, floor area and roof type.
- Surfaces coloured by heat-loss share from the twin; toggling a retrofit animates the change and updates numbers with their intervals.
- Sun path (SunCalc) over the roof for PV; a weather slider (cold snap, heatwave) reruns the scenario.
- Side panels: score contribution chart, retrofit roadmap, what-if chat; a map tab for geography and the portfolio.

### 10a. Interactive 3D retrofit experience (final presentation)

The final demo is a Three.js model of the customer's house that can be explored outside and inside, where the user adds any retrofit, sees it appear on the house, and watches the computed numbers change. The 3D scene never calculates anything: every number and every colour comes from `run_scenario` outputs.

**Stack.** React + react-three-fiber + drei; geometry generated procedurally from EPC archetype fields (detached, semi-detached, mid/end terrace, flat, bungalow; storeys, floor area, roof type, glazing ratio, wall type). Optional low-poly glTF archetypes from Blender with named meshes per building element (roof, each wall, windows, floor, loft).

**Views.**

- **Outside:** orbit camera, sun path from SunCalc over the roof, weather slider (typical, cold snap, heatwave).
- **Inside:** cutaway "dollhouse" view using clipping planes (roof and one façade removed), plus fly-to rooms (loft, living room, boiler/plant cupboard).
- **X-ray / thermal view:** surfaces coloured by their share of Level 1 heat loss, with a colour-blind-safe scale, legend and W/K values on hover, always labelled "estimated from home type and EPC descriptions; only the total heat loss is calibrated to your bills".
- **Before/after split:** a slider divides the screen between the current home and the chosen package.

| Retrofit | Shown outside | Shown inside |
| --- | --- | --- |
| Loft / roof insulation | Roof heat colour drops | Insulation layer appears between joists |
| Cavity or external wall insulation | Thicker render or cladding on walls; wall colour drops | Wall section shows filled cavity |
| Internal wall insulation | Wall colour drops | Insulation board on inner wall; room slightly smaller |
| Double / triple glazing | Window frames and glass swap | Same windows from inside |
| Floor insulation | Floor edge colour drops | Insulation under floorboards |
| Draught proofing | Fewer leakage particles at doors and windows | Seals on doors and windows |
| Heat pump | Outdoor unit appears beside the house | Gas boiler removed; hot-water cylinder added; radiators may be upsized |
| Solar PV (+ battery) | Panels on the best roof face, sized in kWp | Inverter and optional battery in the plant cupboard |
| Smart thermostat / controls | No exterior change | Thermostat on the wall; heating schedule panel |

**Interaction.**

- **Retrofit palette:** every retrofit can be added or removed; infeasible ones are greyed out with the rule or Jev reason, and "needs survey" items are marked.
- **Live numbers panel:** score, per-fuel kWh (electricity rises shown as a negative saving), CO2, bills, value range and P(suitable), each with its credible interval, refreshed through a debounced `run_scenario` call.
- **Retrofit path timeline:** a scrubber steps through the staged roadmap year by year (insulation → glazing → heat pump → solar); each step animates onto the house and the numbers update.
- **Click any element** → its U-value, heat-loss share and score contribution; "Explain" sends the computed values to the LLM.
- **Pareto link:** picking a point on the Pareto chart loads that package onto the house.

**Data flow.** Scene state = the scenario spec JSON → `run_scenario` → results JSON → colours, objects and numbers. For the showcase homes, all feasible packages (≤ 256) are precomputed and cached, so toggles respond instantly and the demo cannot fail on a slow call.

**Performance and accessibility.** Low-poly meshes, instancing, under \~100k triangles, 60 fps on a laptop; reduced-motion option; keyboard controls; every colour paired with a number; a 2D fallback if WebGL is unavailable.

**Build in three versions.** v1: parametric house, thermal colouring, retrofit toggles and the numbers panel. v2: cutaway interior and the retrofit objects in the table above. v3: path timeline animation, sun path and weather slider. Estimated success about 60% for v3, about 85% for v1.

**Demo script.** Start outside on the thermal view → cut away to the inside → add loft insulation (roof cools, gas falls) → add a heat pump (gas to zero, electricity rises, CO2 falls) → add solar → scrub the timeline through the staged path → pick a Pareto point → ask the LLM to explain.

### 11. Multi-objective Pareto optimisation

Retrofit choice trades cost against carbon, bills, score and value, so there is no single best package: the system shows the Pareto front (packages no other option beats on every objective) and lets the user pick.

- **Decision variables:** retrofit subset; continuous sizing (insulation thickness, PV kWp, heat-pump capacity); staging order, which affects cash flow but not final physics.
- **Objectives:** minimise net cost after grants and disruption; maximise CO2e cut, bill saving, score gain and value uplift; optionally minimise uncertainty (90% CI width).
- **Constraints:** feasibility rules, budget, memory constraints (e.g. renting rules out a heat pump), planning rules such as conservation areas.

| Case | Method | Tool |
| --- | --- | --- |
| Discrete packages (5–8 retrofits, ≤ 256 subsets) | Enumerate all, evaluate each with the twin, non-dominated sort → exact front | numpy, `pymoo` utilities |
| Continuous sizing | NSGA-II (2–3 objectives) or NSGA-III (4+) | `pymoo` |
| Slow hourly twin | Multi-objective Bayesian optimisation (qNEHVI) on GP surrogates | BoTorch |
| Staging the chosen package | Dependency rules + cash flow (insulation before heat pump, fastest payback first) | Small dynamic programme |

**Bayesian robust front.** Objectives are posteriors, so a front built from means overstates certainty. Compute the front for each posterior sample; each package's membership frequency is its **P(on front)**, shown as e.g. "loft insulation + heat pump: 87% chance of being Pareto-optimal". A conservative mode optimises P10 savings and P90 cost.

**Choosing a point.** Default suggestion = the knee point; a budget slider shrinks the front live; a reference-point method finds the package closest to goals such as "≥ 2 t CO2, ≤ £10k"; customer memory learns preference weights from accept/reject choices and re-ranks points without relaxing hard constraints.

**Lloyds portfolio front.** Allocating a green-finance budget across homes is a multi-objective knapsack (maximise CO2 cut and customer value, minimise risk), solved with NSGA-II or an ε-constraint integer programme, giving a lending-policy front.

**UI.** Scatter of cost vs CO2 cut (colour = value uplift, size = P(on front)); parallel coordinates for 4+ objectives; clicking a point updates the 3D twin and the LLM explains the trade-off from front data only.

**Evaluation.** Hypervolume vs a greedy kWh-per-£ ranking and a CO2-only optimiser; share of greedy picks dominated by the front; stability of P(on front) across posterior samples. Enumeration is exact and reproducible.

### 12. Neural-operator surrogate (optional)

Use a **physics-informed DeepONet** as a fast copy of the twin's hourly simulation, only when the workload reaches portfolio scale; the twin stays the source of truth.

- **Why:** one home's RC ODE is already fast, but portfolio scale (10⁴–10⁶ homes) × \~1,000 posterior samples × up to 256 Pareto packages × 8,760 hours × future-climate years is too many ODE solves. A neural operator turns that into batched GPU inference in milliseconds, and also drives instant 3D-UI sliders and UKCP18 sweeps.
- **Mapping learned:** hourly weather series + twin parameters (conductances, capacitance, COP, PV kWp) + retrofit set → hourly heat demand, indoor temperature and per-fuel series.
- **Architecture:** branch 1 = weather series; branch 2 = twin parameters; trunk = time. Loss = error vs twin output + RC ODE residual (physics-informed).
- **Training data:** synthetic only, generated by the twin over sampled homes, retrofits and weather years including UKCP18. It is a documented surrogate of a documented model, not a new source of truth.

| Type | Fit |
| --- | --- |
| Physics-informed DeepONet | Best: handles weather functions plus parameter vectors cleanly |
| FNO-1D | Strong second; works at any time resolution; used as the ablation |
| PINO | FNO with ODE-residual loss; the physics-informed variant of FNO |
| GNO / GINO, FNO-2D/3D, transformer operators | Skip: the RC graph is small and there is no spatial-field data to validate against |

**Safeguards.** 1% of operator calls are rerun on the twin and the error logged; a monotonicity sweep must show zero violations or calls fall back to the twin; inputs outside training ranges are routed to the twin; posterior samples are pushed through the operator to keep intervals, and a deep-ensemble spread covers the surrogate's own error.

**GP emulator vs neural operator.** The GP emulator suits scalar outputs (annual kWh), up to \~10k runs, with native uncertainty, for calibration and small sweeps. The operator suits hourly series, 10⁴–10⁶ runs, needs an ensemble for uncertainty, and serves portfolio and future-climate scale. They are complementary.

**Evaluation.** Relative L2 error on held-out series and annual-kWh MAPE vs the twin; speed-up in solves per second; the monotonicity sweep; ablation of DeepONet vs FNO-1D vs GP emulator.

**Tools.** `neuraloperator` (FNO, PINO), DeepXDE (DeepONet, physics-informed), NVIDIA PhysicsNeMo (heavier).

**Priority.** Could tier, about 55% estimated success; needs a GPU and a finished twin and synthetic generator. Skip it if the portfolio feature is dropped; the GP emulator is then enough.

### 13. Two-level simulation

Adopt a two-level system, with information flowing one way: Level 1 produces every number and drives all decisions; Level 2 only visualises, and only when it agrees with Level 1.

> _Figure: Two-level simulation · one-way from Level 1 (rendered in the live doc and in deliverables/M1.pdf)_

- **Level 1 — RC + equipment simulator.** "Validated" means Bayesian-calibrated against metered energy, with interval coverage reported on held-out homes. It is the source of every score, saving, carbon and value figure, and the fallback for every home.
- **Decisions.** Bayesian simulation and the Pareto optimiser use posterior samples of Level 1, or the temporal DeepONet surrogate of Level 1 at portfolio scale (component 12). They never use Level 2.
- **Level 2 — visual only.** The default 3D view colours each envelope surface by its Level 1 heat flux (U × A × ΔT): consistent with the numbers by construction and no ML risk. An optional spatial PINO adds within-surface detail (corners, thermal bridges) for archetype geometries only (mid-terrace, semi-detached, detached, flat).

**If the spatial PINO is built:** train it on FEM solves (FEniCS or scikit-fem) of archetype wall and corner sections plus the heat-equation residual; render it only if its integrated heat flux is within ±5% of Level 1's heat loss for that element; label it "illustrative heat-flow detail, not used in estimates". It is a Could-tier item at about 38% estimated success, because EPC data has no geometry or surface temperatures to validate it against.

### 14. Jev System-1 layer: what-if routing and feasibility checks

Use Jev for fast, calibrated, bounded decisions (routing what-ifs, screening feasibility, checking explanations), while every number still comes from the physics tools.

**What Jev is.** TypeSafe's "System 1" model: a text state plus typed questions in, Choice / Score / yes-no probability out, each with a calibrated confidence, in 70–500 ms. It is hosted and proprietary, in early access via a waitlist, and its vendor-run benchmark shows 67.8% agreement with reference answers ([DataCamp overview](https://www.datacamp.com/blog/system-one-models-jev), [launch post](https://www.mindstudio.ai/blog/jev-system-one-model-launch)). It gives no rationale, and its confidence describes its own decision accuracy, not physics error, so it must not produce kWh, £, CO2 or score values.

| What-if question type | Answered by | Jev's role |
| --- | --- | --- |
| Numeric (savings, carbon, value, score, payback) | Physics tool `run_scenario` only (Level 1, or its temporal surrogate) | Parse the request and route it |
| Comparative ("which retrofit first?", "is it worth it for me?") | Jev Choice over options already computed by physics and Pareto | Pick, with confidence |
| Answerable? ("is there enough data to estimate value?") | Jev yes/no probability; low → say data is insufficient | Gate |
| Explanation check | Jev yes/no: "is every claim supported by this JSON?" | Guardrail layer 2; the regex number check stays layer 1 |

```
User what-if
  → Jev (System 1): intent Choice {numeric, comparative, feasibility, out-of-scope}
                    + scenario slots, each with confidence
     ├─ confidence ≥ τ  → build scenario spec
     └─ confidence < τ  → Claude parses or asks the user to clarify
  → feasibility check (rules, then Jev)
  → run_scenario (System 2 physics): posterior samples, credible intervals
  → Claude explains the computed JSON
  → validator: regex number check + Jev "supported?" → regenerate on failure
```

**Jev-based feasibility check.**

- **Layer 1, hard rules** (deterministic, from the knowledge base): solid walls, flats, listed buildings, conservation areas, roof orientation. A rule veto cannot be overridden.
- **Layer 2, Jev** for each retrofit that passes the rules: Choice {feasible, feasible with conditions, needs survey, infeasible} with confidence. The state includes home facts plus free text that rules cannot parse: EPC element descriptions, user notes, customer memory and planning details.
- Confidence below τ → "needs survey" or a clarifying question to the user, never a silent pass.
- Reasons shown to the user come from rule codes and the knowledge base, because Jev gives no rationale.
- **Combined suitability:** P(suitable) = P(feasible) × P(net benefit), with P(net benefit) from the Bayesian posterior and independence assumed and stated. Jev's confidence is used as P(feasible) only after recalibration (isotonic regression) on our labelled set.

**Using the confidence.** Calibrate τ on 30–50 held-out what-if prompts and about 200 labelled home × retrofit feasibility pairs; pick τ so decisions above it are at least 95% correct; report a reliability diagram and expected calibration error. The UI shows the interpretation ("loft insulation + heat pump, confidence 0.93"). Jev's confidence never stands in for an estimate's credible interval.

**Evaluation.** Routing accuracy; feasibility precision and recall on the infeasible class; calibration error; latency. Baselines: rules only, and Claude with structured output on the same schema.

**Fallback and reproducibility.** Without an API key, Claude with structured output fills the same interface. Every Jev request and response is cached and logged so organisers can replay runs. Vendor claims are re-measured on our own data. Could tier, about 55% estimated success, mainly because access is uncertain.

### 15. Causal meta-learners for retrofit effects

Use causal meta-learners to estimate how a retrofit or an EPC upgrade affects energy use and price differently for different homes (the conditional average treatment effect, CATE). The DR-learner is the default; the twin still produces each customer's numbers.

| Learner | Idea | When |
| --- | --- | --- |
| S-learner | One model with treatment as a feature | Baseline only; tends to shrink effects towards zero |
| T-learner | Separate models for treated and control homes | Large, balanced groups |
| X-learner | Imputed effects weighted by propensity | Small treated groups, e.g. cavity insulation in one region |
| R-learner / DML | Residualise outcome and treatment | Many confounders, smooth effects |
| DR-learner | Doubly robust with cross-fitting: right if either the outcome or the propensity model is right | Default |
| Bayesian Causal Forest | Posterior over CATE | Matches the Bayesian design; intervals built in |

**Uses.**

1. **Value uplift (Should tier, \~65% estimated success).** On UCL EPC–Price Paid repeat sales: treatment = EPC band improved between sales; outcome = change in log price; covariates = location, type, size, period and the UK House Price Index. The DR-learner gives uplift by region × property type, which becomes the required value range.
2. **Observed energy savings (Could, \~55%).** On NEED 50k: treatment = loft or cavity insulation installed in a given year; outcome = change in gas use before vs after, so stable differences between homes cancel out. CATE by archetype checks the physics twin and supplies data-driven priors for the hierarchical retrofit model, shown as "physics-predicted vs observed savings by home type".
3. **Lloyds targeting (Could, \~50%).** Dropped while only synthetic customers exist (problem statement section 9 forbids arbitrary synthetic labels): take-up uses the documented Green Profits scenario rule instead; an uplift model is trained only if real take-up data becomes available.

Heat pumps have no control group in public data, so their effects come from physics only, and the report says so.

**Checks.**

- The key assumption is no unmeasured confounding, which other renovations (a new kitchen alongside an EPC upgrade) can break. State it, run a sensitivity analysis (E-values or EconML's sensitivity tools), and label results indicative.
- Check propensity overlap and trim extreme cases.
- Refutation tests with DoWhy: placebo treatment, random common cause and data subsets.
- Benchmark every learner on synthetic data from the physics generator, where the true effect is known, to choose the learner without real ground truth.
- On real data, BLP and GATES tests check whether the predicted differences between homes hold up.

**Tools.** EconML (DRLearner, XLearner, CausalForestDML, sensitivity), CausalML (S/T/X/R learners, uplift metrics), DoWhy (refutation), `stochtree` (Bayesian Causal Forest).

### 16. Property value: green-twin matching and synthetic values

Find pairs of real homes that match on everything except greenness, and measure how much price and running cost differ between them. Real data already exists for this, so no synthetic prices are needed for the analysis; synthetic values are generated only for synthetic homes.

**Data (already available).** The UCL attribute-linked Price Paid–EPC dataset ([UK Data Service 854942](https://reshare.ukdataservice.ac.uk/854942/)): 5.73M sales in England and Wales, 2011–19, with floor area, habitable rooms and EPC energy information; 79% of market sales linked; linkage scripts included. The newer linked release to 2024 (UK Data Service 857911) extends it. EPC records also carry estimated current heating, hot-water and lighting costs, which give the running-cost side.

**Finding green twins.**

1. **Exact match** on property type and built form, construction-age band, tenure, LSOA (or the same street where possible) and sale quarter.
2. **Close match** on floor area (±10%), habitable rooms and glazing type, by coarsened exact matching then nearest neighbour on the remaining features.
3. **Keep pairs** whose Green Home Score differs by at least 15 points. Compare on component sub-scores, not only the EPC band, because the energy-efficiency component is itself rating-based.
4. **Balance check:** standardised mean differences of all non-green features below 0.1 across pairs.

**What it measures.**

| Relationship | Pair measure | Output |
| --- | --- | --- |
| Greenness vs price | Δ log price per 10 score points, within pairs | Premium curve by region and type, with bootstrap intervals |
| Greenness vs running cost | Δ EPC estimated energy cost and, where available, metered cost | £/yr per 10 score points |
| Price vs running cost | Price premium vs capitalised bill saving | Whether buyers price in the savings |

Results are benchmarked against published estimates: Nationwide finds A/B homes sell for about 1.6–1.7% more than comparable D homes and F/G about 1.4–3.5% less ([Mortgage Professional summary](https://www.mpamag.com/uk/mortgage-types/residential/nationwide-epc-ratings-create-house-price-premium/390239)); The Mortgage Works reports +10.9% for buy-to-let vs +2.2% for owner-occupiers ([Mortgage Solutions](https://www.mortgagesolutions.co.uk/mortgage-news/2024/10/17/greener-btls-worth-up-to-20000-more-than-lower-rated-homes/)); Oxford Economics reports F/G homes at −7.4% ([Oxford Economics](https://oxfordeconomics.com/resource/the-growing-importance-of-energy-efficiency-in-home-buying-decisions)); DECC 2013 gives band premiums for Wales.

**Limits.** Pairs that match on observed features can still differ in unobserved condition (a new kitchen often comes with an EPC upgrade), and homes with large score gaps inside a matched group may be unusual. Results are associations, labelled indicative; repeat sales of the same home and the DR-learner (component 15) are the stronger designs, and matching is the transparent baseline beside them.

**In the product.** "Green twins of your home": the three closest matched homes that are greener, shown as ranges (price per m², energy cost, score) and never as addresses, even though Price Paid data is public.

**Synthetic property values.** Synthetic homes get a value from the hedonic model fitted on real data, plus residual noise drawn from its real residual distribution, plus a green-premium term we set explicitly (for example from the matched-pair estimate). Because that premium is known, it becomes ground truth for benchmarking the matching study, the DR-learner and the hedonic model. The report states that this premium is our assumption, so recovering it shows the methods work, not what the real premium is.

Should tier, about 70% estimated success: the data exists and matching is simple; the effort is in the linkage clean-up and balance checks.

## Bayesian confidence in every output

Every number the customer sees comes from posterior samples, so each one carries a credible interval and each decision a probability.

| Output | Shown as | How it is computed |
| --- | --- | --- |
| Green Home Score | 38 (90% CI 33–44); P(Poor) = 0.81 | Deterministic weighted formula applied to posterior samples of uncertain inputs |
| Classification | Most probable band + its probability | Share of samples in each band |
| Per-fuel savings | e.g. gas −6,100 kWh/yr (CI −7,000 to −5,200); electricity +1,900 kWh/yr | Twin rerun on each posterior sample, before vs after |
| Carbon reduction | tCO2e/yr with CI | Per-fuel savings samples × documented emission factors |
| Value impact | +2–5% range, labelled indicative | Posterior predictive of the hedonic model, before vs after |
| Retrofit suitability | High / Medium / Low | P(suitable) = P(feasible) × P(net benefit): ≥ 0.8 = High, 0.5–0.8 = Medium, < 0.5 = Low; rule-infeasible items excluded |
| Data confidence | Evidence score per home | Months of metered data, EPC age, fields imputed; drives interval width |

The LLM receives posterior summaries only (mean, 90% CI, probabilities) and maps them to calibrated wording. Calibration is checked on held-out homes: 90% intervals should contain the truth about 90% of the time.

## Compliance with the brief

A constraint review on 10 Oct 2026 found five violations and three wording risks; all are now fixed in the repo (`AGENTS.md` → Compliance fixes, `contracts/`, `deliverables/M1.pdf`).

| # | Issue | Fix |
| --- | --- | --- |
| 1 | Projected score and value had no documented post-retrofit rating | `sap_estimator.json`: SAP 2012 energy-cost-factor formula on the twin under standard conditions, change anchored to the recorded SAP; output labelled "estimated rating, not an EPC"; feeds the score's energy-efficiency component and the value model's band-after |
| 2 | Emission factors and costs were TO\_VERIFY / TO\_SOURCE | Factors verified in the DESNZ 2025 file (electricity 0.19553 = generation 0.177 + T&D 0.01853; gas 0.18296 gross CV); Ofgem Q4 2026 tariffs; costs from BEIS 2017 Table 3 × ONS CPI 1.3888; heat pump and PV from secondary reports (flagged); unsourced costs return no figure |
| 3 | Intensity was weather-normalised; PV overlap unjustified | Intensity = annualised metered energy per m² as the brief defines; weather-normalised only as a sensitivity; PV overlap justified and its size reported |
| 4 | Heating mapping seemed incomplete | Every heating system mapped (incl. communal boiler), enforced by a test; full table in M1 |
| 5 | Take-up model on arbitrary synthetic labels | Green Profits is a scenario: documented logistic rule with labelled assumption coefficients; no model trained on synthetic labels |
| 6 | Value framed as a reason to borrow | Fixed wording: indicative, historical association, not a promised increase |
| 7 | Synthetic premium recovery could read as evidence | Reported only as a method check |
| 8 | 3D element colours looked measured | Labelled as estimated from home type; only the total is calibrated to bills |

## Build plan and team

Four phases, each ending in a gate; nothing from a later tier starts until the gate before it passes. Phase lengths are shares of the total time.

1. **Phase 1 — Foundations (\~20%).** Data download and region choice; property-level splits; sub-score mapping table; single-zone physics twin in PyTorch; synthetic generator v1; EBM and XGBoost baselines; `run_scenario` API stub.
   - Gate: the score and per-fuel savings run end to end for one home, with tests for monotonicity.
2. **Phase 2 — Core intelligence (\~35%).** Multi-node RC graph; Bayesian calibration with hierarchy; weather and geography features; knowledge base (SQL + vectors); tool-only LLM with validator; basic 2D dashboard.
   - Gate: posterior intervals on held-out homes with coverage reported; the LLM answers 20 test scenarios with zero unvalidated numbers.
3. **Phase 3 — Differentiators (\~30%).** 3D twin UI; geography maps and hotspots; graph DB for rules and grants; Pareto front of retrofit packages; Jev routing and feasibility layer; GP discrepancy and imputation; then, only if time allows, GraphSAGE, the GP emulator or UKCP18 scenarios.
   - Gate: demo script runs without errors on 3 showcase homes.
4. **Phase 4 — Evaluation and submission (\~15%).** Ablations, adversarial LLM evaluation, evaluation report, architecture diagram, Responsible AI note, held-out test predictions, run instructions.

| Role | Owns |
| --- | --- |
| Data and geography engineer | Downloads, joins, splits, weather and geography features, maps, hotspots |
| Physics and Bayesian lead | RC twin, synthetic generator, NumPyro calibration, GP discrepancy |
| ML engineer | EBM/XGBoost, value model, GraphSAGE, evaluation and ablations |
| LLM and knowledge-base engineer | SQL/vector/graph stores, tools, Jev layer, validator, adversarial scenarios |
| Front-end engineer | 3D twin, dashboard, what-if chat, portfolio view |

With fewer than five people, merge the ML and LLM roles first and drop the graph DB and GraphSAGE.

## Priority build order

Build a thin end-to-end path that meets every required output first, then deepen it in order of rubric weight × confidence. Each checkpoint leaves a submittable system, so running out of time costs features, not the submission.

**Tier A — required outputs, end to end**

1. **Data schema, property-level splits, sub-score mapping table.** Everything depends on these; fixing splits first prevents leakage later.
2. **Steady-state RC twin v1** (monthly degree-day form) with per-fuel conversion and documented emission factors. Unlocks savings, carbon and every what-if.
3. **Green Home Score, classification and component contributions.** Pure formula on EPC fields and metered intensity; can run in parallel with step 2.
4. **Retrofit catalogue and feasibility rules** (JSON). Needed before any recommendation is shown.
5. **`run_scenario` API and SQL results store.** The single interface the LLM, dashboard and optimiser all call.
6. **Synthetic generator using the twin.** Produces the per-fuel and improvement-potential reference labels.
7. **EBM and XGBoost baselines** for energy intensity, plus improvement-potential ranking against a simple baseline.
8. **Value model** on the UCL EPC–Price Paid set, with quantile ranges labelled indicative; upgrade it to a DR-learner on repeat sales (component 15) once the baseline works; run the green-twin matching study (component 16) alongside as the transparent check.

*Checkpoint A: score, class, recommendations, per-fuel savings, carbon and value range for three demo homes.*

**Tier B — grounded LLM and minimum product**

9. **Tool-only LLM with numeric validator, and a 2D dashboard** (score contributions, recommendations, what-if chat).
10. **Evaluation harness:** held-out metrics vs baselines and the first 30 adversarial LLM scenarios. Started now so every later feature is measured as it lands.

*Checkpoint B: a complete, evaluated submission that meets the whole brief.*

**Tier C — differentiators, in order**

11. **Weather normalisation and missing-month handling** (Open-Meteo / HadUK-Grid). Cheap, and it improves every number.
12. **Bayesian calibration (NumPyro)** → credible intervals on all outputs and P(suitable).
13. **Pareto front with P(on front)** and staged roadmap.
14. **Geography layer:** LSOA maps, hotspots, regional degree-day trends, and the Lloyds dashboard (heat-map layers, segments, Green Profits view).
15. **3D twin** coloured by Level 1 element heat loss, with retrofit toggles; then the inside-and-outside retrofit path experience (component 10a), built v1 → v3.
16. **Knowledge base extras:** vector docs, graph rules and grants; Jev routing and feasibility if an API key is available.
17. **Customer memory** showing recommendations changing over sessions; agent capture v1 (photo upload with manual element tags, confirmed by the agent).

*Checkpoint C: the demo build; freeze features here and spend remaining time on evaluation and documents.*

**Tier D — stretch, only after Checkpoint C**

18. GP discrepancy term and GP imputation.
19. Multi-node differentiable RC graph.
20. Lifetime savings under UKCP18 scenarios; overheating flag.
21. Neighbourhood GraphSAGE; neural-operator surrogate; spatial PINO visual; physics-vs-observed interaction tensor; photo-to-twin vision auto-tagging (agent capture v2).

**Parallel start.** All five roles start at once: data engineer on step 1 then 14; physics lead on steps 2 and 6 then 12; ML engineer on steps 7–8 then 10 and 13; LLM engineer on steps 4–5 then 9 and 16; front-end engineer on the dashboard shell against mocked `run_scenario` output, then 15.

**Stop rules.** If Checkpoint A is not reached by about a third of the time, cut Tier C to steps 11–13. If Checkpoint B is late, skip the 3D twin and Jev. Documentation (submission items D–I) is written alongside each step, never left to the end.

## Risks and open questions

The main risk is breadth: ten techniques in a hackathon. The tiers and gates exist so the core is finished before anything flashy starts.

| Risk | Mitigation |
| --- | --- |
| Too many techniques, nothing finished | Must tier first; gates enforced; Could tier only after Phase 2 |
| 3D UI eats front-end time | Parametric box-and-roof house only; no detailed geometry |
| Graph DB adds ops overhead | Embedded Kùzu or networkx instead of a Neo4j server |
| Synthetic labels just mirror the generator | Validate against NEED; say plainly that agreement with synthetic targets does not prove real-world accuracy |
| Extrapolation beyond data | Only the physics twin extrapolates; out-of-distribution flag shown to the user |
| Bayesian fitting too slow for live demo | Offline fit; fast per-home updates; cache showcase homes |

Open questions:

- [ ] How long is the hackathon, and how many people are on the team?
- [ ] Is a GPU available (affects the GNN and any JEPA work)?
- [ ] Region: UK only (decided 10 Oct 2026)
- [ ] Which LLM and API budget is allowed?
