<!-- Rebuilt on 2026-10-10 from the authored content of the deleted "Strategy Report" (main) tab of the GreenTwin strategy doc (https://claude.ai/artifact/Ca1BntuvXtnKYtD4Uuz1GT), including all later edits (UK-only decision, weather wording, verified emission factors). The doc's other tab is exported as docs/STRATEGY_REPORT.md. Embedded figures are replaced by placeholders; see deliverables/M1.pdf. Where this file and AGENTS.md / contracts/ disagree, AGENTS.md and contracts/ win (they hold the later compliance fixes). -->

# Green Home Score & Retrofit Engine — Megathon Strategy Report

Megathon X: Singularity · Problem Statement 2 (Lloyds) · Oct 10, 2026

## Executive summary

We should build a physics-based "digital twin" of each home, calibrated with Bayesian inference against metered energy, with a grounded LLM that only explains computed outputs. This targets the 50% of marks that sit in ML quality, savings/carbon/value estimation and LLM grounding, rather than the 5% for innovation.

- **Core engine:** RC thermal model + Bayesian calibration produces every score input, per-fuel saving, carbon figure and value range, each with a credible interval.
- **Data:** real public data first (EPC, Land Registry, NEED, smart meters), synthetic only to fill gaps, with simulator-generated labels that follow physics.
- **Recommendations:** feasibility rules + a sequence optimiser that reruns the model per candidate, so retrofit interactions are computed, never summed.
- **LLM:** tool-only access to `run_scenario`, plus a numeric validator that rejects any number not in the computed output.
- **Lloyds angle:** green-finance cash-flow view and a mortgage-portfolio dashboard.
- **Stretch:** JEPA-Anything factorised load model, merged only if an ablation shows a gain.
- **Region packs:** UK only (team decision, 10 Oct 2026); the India pack is dropped and kept below for reference.

## Problem and rubric

Lloyds asks for an AI system that answers: how green is my home, what should I improve, how much energy and carbon could I save, and could it raise my property's value. The **core rule**: score, savings, carbon and value come from documented models; the LLM only explains them and must rerun the models for every what-if.

Required outputs:

- Green Home Score 0–100, a fixed weighted sum of seven monotonic sub-scores (energy rating 25%, energy intensity 20%, insulation 15%, heating 15%, glazing 10%, renewables 10%, smart controls 5%), with each component's contribution shown.
- Poor / Very poor flagging and ranking by improvement potential, with electricity and gas reported separately.
- A prioritised, feasible retrofit list (High/Medium/Low suitability) whose order reflects interactions.
- Per-fuel kWh savings, carbon from documented emission factors, and a property-value impact as a range.

> _Figure: rubric weights — four criteria (ML and recommendation quality 20%, Green Home Score 15%, savings/carbon/value 15%, GenAI/LLM 15%) carry 65% of the marks; business relevance 10%, architecture 10%, explainability/RAI 5%, UX 5%, innovation 5%._

Judges also run hidden scenarios: missing or conflicting data, infeasible retrofits, electricity rising after a heating change, and too little data. LLM targets are groundedness ≥90%, factual accuracy ≥90% and unsupported claims <5%. Unsupported precision and arbitrary synthetic labels are penalised.

## Idea catalogue

Fourteen ideas came out of the brainstorm; each is tied to the rubric criterion it strengthens. Tier says whether it is core, cheap-but-required, or stretch.

| # | Idea | What it does | Main criterion | Tier |
| --- | --- | --- | --- | --- |
| 1 | Physics simulator as label generator | RdSAP-style heat-loss model (U-values × areas × degree days) generates per-fuel labels; interactions emerge naturally | ML & data quality | Core |
| 2 | EPC ↔ Land Registry join | Real sale prices with energy rating at sale; repeat-sales difference-in-differences for value uplift | Value estimation | Core |
| 3 | Uncertainty everywhere | Credible intervals on savings, carbon, value and the score itself | Savings & value | Must-have |
| 4 | Sequence optimiser / roadmap | Beam search over retrofit orderings; Pareto of cost, kWh, CO2e, value; "cheapest path to EPC C"; budget-staged plan | Business relevance | Core |
| 5 | Feasibility rules engine | Hard constraints (solid wall, flat, listed building, roof orientation) with a stated reason per rejection | Recommendations | Must-have |
| 6 | Solar from smart-meter load shape | PV generation curve vs household load → self-consumption → realistic import cut | Savings | Stretch |
| 7 | Tool-only LLM + numeric validator | LLM can only call `run_scenario`; every number tagged and checked against computed JSON | LLM grounding | Core |
| 8 | Adversarial eval harness | ~100 held-out scenarios scored for groundedness, accuracy, unsupported claims | LLM grounding | Core |
| 9 | Green-finance view | Loan repayment vs bill saving; grants such as the Boiler Upgrade Scheme | Business relevance | Stretch |
| 10 | Bank portfolio dashboard | Financed emissions, rating mix, top homes by improvement potential | Business relevance | Core |
| 11 | Grid-decarbonisation timeline | Projected carbon factors → lifetime tCO2e of a heat pump | Carbon | Stretch |
| 12 | Structured customer memory | Facts store (constraints, rejections, completed work) re-ranks feasible options; posteriors tighten over sessions | Innovation | Must-have |
| 13 | EPC certificate upload | LLM extracts fields, user confirms, models compute | UX | Stretch |
| 14 | Explainable score | Monotonic piecewise-linear sub-scores, peer-benchmarked intensity, waterfall of contributions | Green Home Score | Must-have |

## Recommended stack and architecture

Build ideas 1, 2, 4, 7, 8 and 10 as the core, add 3, 5, 12 and 14 because they are cheap and rubric-critical, and treat 6, 9, 11 and 13 as stretch goals.

> _Figure: system architecture — data layer → home digital twin (RC thermal model + Bayesian calibration, with memory adjusting priors and constraints) → Green Home Score / retrofit engine / savings-carbon-value → grounded LLM (tool-only `run_scenario`, numeric validator) → customer dashboard and Lloyds portfolio view; a what-if loop reruns the twin._

A what-if from the user goes back through the twin, so the LLM never sees a number the models did not compute. Memory adjusts priors and constraints, not outputs.

| Layer | Choice |
| --- | --- |
| Data & features | Python, pandas/Polars, DuckDB for the large EPC and Price Paid files |
| Physics + Bayesian | NumPyro (or PyMC), ArviZ diagnostics |
| Baseline ML | LightGBM + conformal intervals |
| API | FastAPI; `run_scenario` as the single tool the LLM can call |
| LLM | Claude via tool use, with a post-generation numeric validator |
| Front end | React dashboard: score waterfall, retrofit roadmap, what-if chat, portfolio view |

## ML approach: physics-informed model with Bayesian calibration

A physics backbone gives us counterfactuals (what happens if the U-value changes) that pure ML cannot learn without before/after retrofit data; Bayesian calibration ties it to each home's real meter readings and turns every output into a range.

**1. Physics backbone.** A lumped RC thermal model: heat loss = UA × ΔT plus thermal mass, driven by regional degree days. Heat-pump electricity = heat demand / COP. Insulation lowers demand before the COP applies, so retrofit interactions are computed rather than summed.

**2. Bayesian calibration (core).**

- Priors: U-values by wall type and build age (RdSAP default tables), heat-pump COP from the Electrification of Heat trial, occupant setpoint.
- Likelihood: the home's metered monthly gas and electricity, given degree days.
- Posterior: heat-loss coefficient (UA), setpoint and baseload → per-fuel savings credible intervals. A what-if changes a parameter and pushes the posterior samples through again.
- Missing months are latent variables, never zeros.

**3. Hierarchical retrofit effects.** Priors = the problem statement's illustrative ranges (loft insulation 12–18%, etc.), updated with NEED before/after data. Partial pooling across property type × region stops small groups overfitting.

**4. Bayesian hedonic value model.** Price ~ size + type + location + time + energy band, with priors from the DECC 2013 band premiums. The posterior predictive before vs after retrofit gives the required range. It remains an observational association, not a proven causal effect.

**5. Score uncertainty.** The score formula stays deterministic and monotonic; missing inputs are sampled from their posteriors, giving outputs such as "Score 38 (90% CI 33–44), P(Poor) = 0.81".

**6. Memory as Bayesian updating.** Each new bill or completed retrofit tightens the home's posterior; Beta posteriors over acceptance re-rank only feasible options. The demo shows intervals narrowing from session 1 to session 3.

**Speed.** Fit hierarchical models offline; per-home updates use variational inference or importance reweighting of prior samples, under one second. No live MCMC in the request path.

**Baseline.** LightGBM point estimates, LightGBM quantile regression and conformal intervals, so judges see what the Bayesian layer adds.

### Gaussian processes and surrogate models

The strongest uses are a GP discrepancy term inside the calibration and GP imputation of missing months; an emulator and Bayesian optimisation are only worth adding if the simulator becomes slow or retrofits are sized continuously.

| Use | How | Priority |
| --- | --- | --- |
| Kennedy–O'Hagan calibration | Metered energy = simulator output + GP discrepancy + noise; the GP absorbs model error (occupancy, thermal bridges, appliances) while UA, COP and setpoint stay interpretable | Core |
| Missing-month imputation | Per-home GP with yearly periodic + degree-day linear + noise kernels; gaps filled with uncertainty bands (≤ 60 months per home, so exact GPs are fast) | Core |
| Spatial value effect | Matérn GP over latitude/longitude + time as the location effect in the hedonic model; sparse variational GP or per-region subsamples because 21M sales is too many for an exact GP | Optional |
| Simulator emulator | Latin hypercube design → 500–2,000 simulator runs → GP surrogate for millisecond what-ifs; unnecessary while the RC model itself runs in milliseconds | Only with a heavy simulator |
| Bayesian optimisation | Continuous sizing (insulation thickness, PV kWp, heat-pump capacity) for best NPV or CO2 per £; discrete packages (32 sets) are brute-forced instead | Only with continuous sizing |

**Monotonicity risk.** A plain GP can produce small wiggles such as "more insulation, slightly more heat demand", which breaks the brief's rule that a better state never scores lower. Keep the physics backbone monotonic with a small-amplitude GP discrepancy, or use a monotonic GP with virtual derivative observations, and report a parameter-sweep test with zero violations. LightGBM with `monotone_constraints` is a cheap surrogate baseline that is monotonic by construction.

**Tools.** scikit-learn or GPflow for small exact GPs; GPyTorch for sparse GPs; BoTorch for optimisation; emukit, SMT and SALib for emulation, experimental design and sensitivity analysis; the NumPyro/PyMC HSGP approximation for GPs inside the Bayesian model.

**Evaluation.** Interval coverage, negative log predictive density and CRPS on held-out homes; emulator RMSE against the true simulator on held-out design points; the monotonicity sweep; and an ablation of physics only vs physics + GP discrepancy vs LightGBM.

### Graph neural networks for physics-based ML

A differentiable RC thermal network is the graph worth building: it upgrades the core physics model while keeping energy conservation and monotonicity guaranteed. A neighbourhood graph of homes is an optional extra. Room-level graphs are out, because public data is home-level and cannot identify per-room parameters.

**1. Differentiable RC thermal network.** The RC model is already a graph: nodes are thermal masses (indoor air, wall, roof, floor, outside) and edges are conductances (U × A).

- A small GNN/MLP predicts edge conductances and node capacitances from EPC element features (wall type, age, glazing).
- The RC equations (an ODE via `torchdiffeq`, or the steady-state degree-day form) turn them into energy use, trained end to end on metered kWh.
- Guarantees by construction: energy conservation at each node, conductance > 0 via softplus, and monotonicity, because a retrofit lowers U and therefore heat loss.
- It stays a documented physics model with named parameters, and the conductances can take Bayesian priors.
- Keep it to 4–6 nodes per home.

**2. Neighbourhood graph of homes.** Nodes are homes; edges are party walls (real physical coupling: mid-terrace homes lose less heat than end-terrace), spatial proximity, and similarity of era, type and size.

- GraphSAGE imputes missing fields and generalises to unseen homes.
- Graph neighbours give peer benchmarks for the intensity sub-score.
- Neighbour sale prices add a spatial signal to the value model.
- Leakage controls: only neighbour prices dated before the sale; spatial-block splits (whole postcodes per split); test labels masked during training.

**Skipped.** A low-voltage grid graph (no public data) and a GNN over retrofit dependencies (a rule table is enough).

**Evaluation.** Per-fuel kWh MAE and interval coverage against the single-zone RC + Bayesian model and LightGBM with spatial features; a monotonicity sweep with zero violations; spatial-block vs random cross-validation to show the leakage gap; and a check that the mid- vs end-terrace heat-loss difference matches physics. GNNs often fail to beat LightGBM with good spatial features, so the ablation is reported either way.

**Tools.** PyTorch Geometric or DGL, `torchdiffeq`, NumPyro for priors on conductances.

**Priority across the advanced methods:** physics RC + Bayesian calibration, then GP discrepancy and imputation, then the GNN extensions, then JEPA-Anything.

## Stretch track: JEPA-Anything factorised load model

Use [JEPA-Anything](https://github.com/Gen-Verse/JEPA-Anything) to split each home's load into weather-driven, behaviour and baseload factors, and check the weather factor against physics. It must not produce the savings numbers.

**What it is.** A domain-agnostic Joint-Embedding Predictive Architecture with Orthogonal Predictive Factorization (OPF), which splits the latent state into complementary factors ([paper](https://arxiv.org/abs/2609.20800)). It was tested on vision, biology, clinical trajectories, control, molecular dynamics, physical fields and weather. It reports a 34.8% cut in intervention-prediction error on Interventional Pong, and latent modes that recovered Kepler's scaling law (slope −1.4991). Apache 2.0; about 296 stars and 6 commits; no energy-domain weights.

**How we would use it.**

- Pretrain on real smart-meter data only (London smart meters), split by household before pretraining.
- Inputs: 48×7 half-hourly load window, daily degree days, static property features. Target: embedding of the next 7-day window.
- Physics check, the same idea as the Kepler result: regress the weather factor on heating degree days; the slope should approximate the heat-loss coefficient (UA) from the Bayesian twin.
- Uses: missing-month imputation, separating a large house from heavy usage, peer benchmarking, explanations such as "your weather-sensitive load is 2.1× peers".
- The weather-factor slope can feed the Bayesian calibration as an extra likelihood term.

**Guardrails.** Pin the commit hash. One person, timeboxed to about 40% of the hackathon, with a go/no-go checkpoint. Merge only if an ablation (LightGBM vs physics vs physics + OPF) shows a gain; report the result either way.

Related work: [Phys-JEPA](https://arxiv.org/pdf/2606.16076) (physics constraints in JEPA latent space) and [Distributed JEPA for energy](https://arxiv.org/pdf/2609.17029) (energy time series plus building and PV context).

## Datasets: UK region pack

The biggest shortcut is UCL's published EPC–Price Paid linked dataset, which saves us doing the hardest join ourselves. Register for EPC access today, because the sign-in can be slow.

| Dataset | Use in pipeline | Access / licence | Size / notes |
| --- | --- | --- | --- |
| [EPC domestic data](https://get-energy-performance-data.communities.gov.uk/) | Property features, EPC recommendations, feasibility rules | Free; GOV.UK One Login; bulk CSV + API | ~30M certificates, England & Wales; UPRN since Nov 2021 |
| [Land Registry Price Paid](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | Sale history for the value model | OGL v3, attribution required | Yearly files 115–230 MB; full file ~5 GB |
| [UCL EPC–PPD linked dataset](https://reshare.ukdataservice.ac.uk/857911/) | Sales with EPC rating + floor area → hedonic and repeat-sales models | UK Data Service ReShare, open | 21M sales, 107 variables, 1995–2024 |
| [NEED anonymised 50k](https://www.data.gov.uk/dataset/nationalenergyefficiencydataframeworkanonymiseddata) | Real metered gas + electricity with installed measures → intensity model, before/after signal | OGL | 50k rows (a 4M version exists) |
| [London smart meters](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d) | Half-hourly load shapes, solar self-consumption, JEPA pretraining | Open | 5,567 homes, 2011–14, ~10 GB; electricity only |
| [Electrification of Heat trial](https://doc.ukdataservice.ac.uk/doc/9050/mrdoc/pdf/9050_eoh_Interim_insights_from_heat_pump_performance_data.pdf) | Real heat-pump seasonal efficiency by house type → COP prior | UKDS registration | 2020–23 |
| [English Housing Survey energy report](https://www.gov.uk/government/statistics/english-housing-survey-2022-to-2023-energy/english-housing-survey-2022-to-2023-energy-report) | Stock distributions for the synthetic generator | Official statistics | Wall type, age, glazing |
| [PVGIS API](https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en) | Hourly PV generation by location and orientation | Free API | v5.3 |
| [UK House Price Index](https://landregistry.data.gov.uk/app/ukhpi/) | Market conditions by local authority × property type | Open | Monthly since 1995 |

**Reference constants** (see `contracts/emission_factors.json`, `contracts/tariffs.json`, `contracts/retrofit_catalogue.json` for the verified, current values):

| Item | Value | Source |
| --- | --- | --- |
| Electricity emission factor | 0.19553 kgCO2e/kWh = generation 0.177 + T&D 0.01853 (verified, DESNZ 2025 flat file) | [DESNZ 2025 GHG factors](https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2025) |
| Gas emission factor | 0.18296 kgCO2e/kWh gross CV (verified; net CV 0.2027) | Same; meter kWh are gross CV |
| Electricity / gas price | 26.32p / 7.97p per kWh (Oct–Dec 2026) | [Ofgem price cap](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-october-and-31-december-2026) |
| Retrofit costs | BEIS 2017 Table 3 low/mean/high per home × ONS CPI 1.3888 | [BEIS cost assumptions](https://www.gov.uk/government/publications/domestic-cost-assumptions-what-does-it-cost-to-retrofit-homes) |
| Heat-pump grant | £7,500; £9,000 off-gas-grid to 31 Mar 2027 (to confirm) | [Energy Saving Trust](https://www.energysavingtrust.org.uk/grants-and-schemes/boiler-upgrade-scheme/) |
| Value uplift prior | Wales vs band D: A/B +12.8%, C +3.5%, E −3.6%, F −6.5% | [DECC 2013, Fuerst et al.](https://www.gov.uk/government/publications/an-investigation-of-the-effect-of-epc-ratings-on-house-prices) |

Leakage: drop EPC "potential rating" and "potential energy" from model inputs; they are EPC's own post-retrofit estimates and may only serve as reference labels.

## Weather and climate data

Weather drives every energy number, so it feeds the core model, not just a trends panel. Use the official Met Office and UKCP18 data for documented figures, Open-Meteo for convenient hourly, forecast and demo data, and cache every response for reproducibility.

| Source | What | Licence / limits | Use |
| --- | --- | --- | --- |
| [HadUK-Grid](https://www.metoffice.gov.uk/research/climate/maps-and-data/data/haduk-grid/overview) via [CEDA](https://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb) | 1 km daily max/min temperature from 1931; latest v1.3.1 to end-2024 | OGL; check CEDA registration per version | Primary history: degree days per postcode, long-term trends |
| [Open-Meteo](https://open-meteo.com/en/terms) | Historical archive (ERA5), forecast API, [climate API](https://openmeteo.substack.com/p/climate-change-api) (CMIP6 models, 1950–2050) | Data CC BY 4.0; free API non-commercial, < 10k calls/day | Hourly temperature and irradiance, recent months, forecast, projections |
| [Met Office Weather DataHub](https://datahub.metoffice.gov.uk/docs) | Site forecasts, land observations | Free observations tier, 360 calls/day | Optional live "this week" panel |
| UKCP18 (CEDA) | Official UK projections under several emission scenarios | Open | Future-climate scenarios (Open-Meteo CMIP6 has no scenario choice and stops at 2050) |
| PVGIS | Irradiance → PV yield | Free | Solar |

**Uses.**

1. **Weather-normalised baseline:** per-home degree-day regression (kWh = baseload + effective UA × HDD) separates a cold year from a poor house for the Bayesian likelihood and a sensitivity check; the score itself uses annualised metered intensity exactly as the brief defines it.
2. **Missing-month annualisation:** fill gaps using that month's actual degree days instead of multiplying by 12.
3. **Regional trends:** 30–50-year heating and cooling degree-day trends per region or LSOA (Mann–Kendall test with Sen's slope, or a Bayesian linear trend), shown as a map layer.
4. **Lifetime value:** UKCP18 or CMIP6 scenarios turn first-year savings into 15–25-year lifetime savings, carbon and payback ranges, which feed the Pareto objectives.
5. **Overheating risk:** trends in hot days (e.g. Tmax > 28°C) and the 2022 heatwave flag homes where insulation without shading or ventilation could raise summer overheating.
6. **Live demo:** a forecast cold snap shows next week's heat loss and cost before vs after retrofit on the 3D twin.
7. **Weather uncertainty:** sampling typical, cold and warm years widens savings intervals to include weather variability, not just model uncertainty.

**Rules.** Weather data is a model input; the LLM quotes trend figures only from computed outputs. Document the source, grid, base temperature (15.5°C UK convention for heating degree days) and units. Trends are reported with confidence intervals and are not presented as causal attribution.

## Datasets: India region pack (dropped)

Dropped: the team is going UK only, so this section is reference material. India would work as a second region pack, but its homes are cooling-dominated and "gas" mostly means LPG/PNG for cooking and water heating, so the retrofit set changes. The brief is UK-framed and Lloyds judges may expect the UK, so India should be the optional second pack, not the main one.

| UK retrofit (brief) | India equivalent |
| --- | --- |
| Loft / roof insulation | Roof insulation or cool roof, checked against the ENS roof U-value ≤ 1.2 |
| Heat pump | 5-star inverter AC; heat-pump or solar water heater replacing geyser/LPG |
| Double glazing | Low-e glazing and shading within the ENS RETV limit (≤ 15 W/m²) |
| Solar panels | Rooftop PV under PM Surya Ghar |
| Smart thermostat | Smart AC controls, BLDC fans, smart plugs, time-of-day tariffs |
| Energy rating band | BEE residential label / ENS compliance (rarely recorded → derived rating-based measure) |

| Dataset | Use in pipeline | Notes |
| --- | --- | --- |
| [CEEW smart meters, Mathura & Bareilly](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GOCHJH) | Load shapes, annualisation, solar self-consumption | ~100 homes, 3-min, 2019–21, CC0 |
| [Kaggle Indian household electricity](https://www.kaggle.com/datasets/suraj520/indian-household-electricity-bill) | Appliance hours → consumption | Check licence and plausibility |
| [CLASP/BEE 2024 survey](https://www.clasp.ngo/research/all/residential-appliance-energy-consumption-survey-india/) | Appliance ownership and use priors | 4,321 households, 20 states |
| [Prayas 2-state report](https://energy.prayaspune.org/images/pdf/energy-consumption-patterns-2-states-report.pdf) | End-use patterns | 3,000 households, UP & Maharashtra |
| [HCES 2023-24 (MoSPI)](https://microdata.gov.in/NADA/index.php/catalog/237/data-dictionary/F1) | Cooking fuel and energy spend → LPG baseline | ~262k households, unit-level |
| [Census 2011 HH-3 / HH-14](https://censusindia.gov.in/nada/index.php/catalog/9573) | Roof and wall material by state → generator priors, feasibility rules | Excel tables |
| [ENS 2024 code (BEE)](https://beeindia.gov.in/sites/default/files/ENS%202024_Buildign%20Code%20for%20Residential%20Buildings.pdf) | RETV formula, U-values, climate zones → simulator core | Documented rating-based measure |
| [Kaggle metro housing prices](https://baselight.app/u/kaggle/dataset/ruchi798_housing_prices_in_metropolitan_areas_of_india) | Hedonic value model | Asking prices, not sales |
| [NHB RESIDEX](https://www.nhb.org.in/about-residex/) | Market trend, 26+ cities, quarterly | Built from mortgage-registry data |

**Constants.** Grid factor 0.727 tCO2/MWh for FY2023-24 ([CEA CO2 Baseline Database v20](https://cea.nic.in/wp-content/uploads/2021/03/User_Guide_Version_20.0.pdf)). LPG ≈ 0.227 kgCO2/kWh from the IPCC 2006 default of 63.1 tCO2/TJ. Rooftop solar subsidy ₹30k/kW for the first 2 kW plus ₹18k for the third, capped at ₹78k ([PM Surya Ghar summary](https://www.solarsquare.in/blog/types-of-government-solar-panel-subsidy-schemes-in-india/)). Tariffs from 2–3 state regulator tariff orders.

**Gaps.** India has no public property-level energy-rating registry and no public join of sale prices with energy ratings, so the value model is the weakest part: use listings plus a wide green-premium prior (JLL cites 10–15% for certified homes) and present a wide indicative range.

## Evaluation plan

Every component gets a held-out metric and a simple baseline, using property-level (or customer-level) splits made before any training or pretraining.

| Component | Metric | Baseline |
| --- | --- | --- |
| Energy model | MAE / MAPE on annual kWh per fuel; 90% interval coverage; CRPS | NEED median by type, size and region; LightGBM |
| Savings | Error vs held-out simulator reference labels and NEED before/after cases | Sum of illustrative percentages |
| Calibration | 90% interval coverage ≈ 90%; flat PIT histogram; R-hat < 1.01 | LightGBM quantile regression; conformal intervals |
| Improvement ranking | Spearman / NDCG vs reference ranking | Rank by raw total consumption |
| Retrofit recommendations | Precision@k vs feasibility/relevance labels; zero infeasible items recommended | Recommend every retrofit |
| Value model | Interval coverage vs held-out sales; comparison with DECC 2013 premiums | Hedonic OLS without energy band |
| Green Home Score | Monotonicity unit tests (0 violations across a parameter sweep); reproducibility | None needed (deterministic formula) |
| LLM | Groundedness ≥ 90%, factual accuracy ≥ 90%, unsupported claims < 5% on ~100 adversarial scenarios | LLM without validator |
| Imputation (stretch) | MAE on artificially masked months | Seasonal-naive ×12; degree-day regression |

Adversarial scenarios cover missing months, conflicting data (rating A but 400 kWh/m²), infeasible requests (heat pump in a 3rd-floor flat), electricity rising after a heat-pump switch, and too little data to estimate value.

## Risks, guardrails and responsible AI

The biggest risk is overreach: an ambitious model that cannot be reproduced or that claims more precision than the data supports. Each risk below has a fallback.

| Risk | Mitigation |
| --- | --- |
| EPC sign-in or big downloads slow us down | Register today; work on 2–3 regions; DuckDB over raw CSVs |
| Synthetic labels look arbitrary | Generate them from the documented physics simulator; disclose every rule; validate against NEED |
| Value model read as causal | Always a range, labelled indicative; repeat-sales design; state confounders |
| LLM invents numbers | Tool-only access; numeric validator; regenerate or strip on mismatch |
| MCMC too slow for the demo | Offline fitting; variational or importance-reweighted per-home updates |
| JEPA-Anything does not converge | Timebox with a go/no-go; fall back to degree-day regression; report the attempt |
| India pack dilutes the UK story | Resolved: UK only |

**Guardrails we commit to:**

- Property-level (or customer-level) splits; no fields unavailable at prediction time.
- Monthly energy annualised from available months; missing data never treated as zero.
- Per-fuel savings reported separately; increases shown as negative savings.
- Emission factors documented with units and source.

**Responsible AI note** (submission item H): no personal data (public and synthetic only); bias check that large or older homes are not penalised for size alone; every score shows its component contributions; the LLM states uncertainty and missing data plainly; a human advisor reviews before any financial decision.

## Next steps

- [ ] Register for EPC bulk data (GOV.UK One Login) and UK Data Service
- [ ] Download the UCL linked dataset, NEED 50k and a London smart-meter subsample
- [ ] Pick 2–3 regions; fix property-level splits
- [ ] Write the sub-score mapping table and monotonicity tests
- [ ] Build the RC physics model and the LightGBM baseline
- [ ] Add Bayesian calibration and the hierarchical retrofit model
- [ ] Build `run_scenario`, the LLM validator and the adversarial scenario set
- [ ] Dashboard: score waterfall, roadmap, what-if chat, portfolio view
- [ ] Stretch: clone JEPA-Anything, run the quickstart, draft `design.json`
- [ ] Decide team owners

## Sources

Problem statement: Megathon X: Singularity, Problem Statement 2 (Lloyds), `LLOYDS-ps2.pdf`. Opened directly: [JEPA-Anything repository](https://github.com/Gen-Verse/JEPA-Anything), [JEPA-Anything paper](https://arxiv.org/abs/2609.20800), the DESNZ 2025 GHG conversion factors flat file, the Ofgem Q4 2026 price-cap page, BEIS (2017) domestic cost assumptions and the ONS CPI series D7BT. Other dataset and constant links came from web search results; confirm each figure on its official page before the final submission.
