# Green Home Score: UK standards, parameters and how to collect them

Part 3 (Green Home Score) research note. It answers three questions:

1. Which accepted UK standards already rate how "green" a home is, and what does each one measure?
2. What is the full list of parameters those standards use, and which of our seven score components does each feed?
3. How do we get each parameter from what a homeowner can give us (an EPC lookup, a few questions, photos, bills)?

The last section proposes how to **learn the weights with ML**. Decisions and numerical evidence are logged in `docs/score/DECISIONS.md`.

Status labels: **[verified]** = read in the primary source this session (link given); **[secondary]** = from commentary or a secondary source; **[to verify]** = from memory or unconfirmed, must be checked before it is quoted in a deliverable.

---

## 1. The UK standards

### 1.1 SAP 10.2 / RdSAP 10: the EPC we have today
The Energy Performance Certificate for an existing home is produced by **RdSAP** (reduced-data SAP). An assessor collects a reduced data set, software expands it with default tables (U-values by age band, etc.), then runs the full **SAP 10.2** calculation. [verified: [RdSAP10 specification, BRE, Feb 2024](https://files.bregroup.com/SAP/RdSAP10-dt13.02.2024.pdf), §1 and Table 31 "Data to be collected"]

SAP produces several ratings. All of them are **normalised per floor area** so large homes are not penalised for size alone. [verified: [SAP 10.2 specification, BRE](https://bregroup.com/documents/d/bre-group/sap-10-2-11-04-2023-pdf), §13–14]

| Rating | Formula (SAP 10.2) | Measures |
| --- | --- | --- |
| **Energy Efficiency Rating (EER, "SAP rating")**, bands A–G | `ECF = 0.36 × total cost / (TFA + 45)`; `SAP = 108.8 − 120.5·log10(ECF)` if ECF ≥ 3.5, else `100 − 16.21·ECF` | Modelled **energy cost** per m² under standard occupancy and standard prices |
| **Environmental Impact Rating (EIR)**, bands A–G | `CF = CO2 / (TFA + 45)`; `EI = 200 − 95·log10(CF)` if CF ≥ 28.3, else `100 − 1.34·CF` | Modelled **CO2 emissions** per m² |
| Dwelling Primary Energy Rate (DPER), kWh/m²/yr | — | Modelled primary energy per m² (EPC field `ENERGY_CONSUMPTION_CURRENT`) |
| **Element ratings** (1–5 stars: Very poor → Very good) | Assigned by the software per element; the thresholds are in a separate BRE document ("basis of star ratings") that is referenced in RdSAP 10 Table 31, item 12-1 [verified reference; thresholds **to verify**] | Walls, roof, floor, windows, main heating, main heating controls, hot water, lighting, secondary heating, each for energy efficiency and for CO2 (`*_ENERGY_EFF`, `*_ENV_EFF`) |

Band thresholds for SAP: A ≥ 92, B 81–91, C 69–80, D 55–68, E 39–54, F 21–38, G 1–20 [verified, matches `contracts/sap_estimator.json`].

> **Flag for Part 2 (rating estimator).** `contracts/sap_estimator.json` uses the **SAP 2012** formula (deflator 0.42, `117 − 121·log10`, `100 − 13.95·ECF`). SAP 10.2, which current EPCs use, has deflator **0.36**, `108.8 − 120.5·log10(ECF)` and `100 − 16.21·ECF`. Logged in DECISIONS.md D-07.

**Why this matters for us:** the EPC already *is* a multi-component green score. The element ratings are per-component labels for almost exactly our sub-scores (insulation ≈ walls + roof + floor, glazing ≈ windows, heating ≈ main heating + hot water, smart ≈ controls). The EPC register gives millions of labelled homes.

### 1.2 Reformed EPC (HEM:EPC), from October 2026
Government confirmed that domestic EPCs move from one headline number to **four headline metrics**, calculated with the new **Home Energy Model (HEM)** instead of SAP. [secondary: [Mayer Brown, May 2026](https://www.mayerbrown.com/insights/publications/2026/05/reforming-the-energy-performance-of-buildings-regime---new-uk-government-proposals); [Element Sustainability](https://elementsustainability.co.uk/reform-to-energy-performance-of-buildings-partial-government-response-overview/); [Building Energy Experts](https://buildingenergyexperts.co.uk/resources/2026-epc-reform-technical-update/)] Metric definitions: [verified: [DESNZ, The Home Energy Model: Energy Performance Certificates, 2026](https://assets.publishing.service.gov.uk/media/69b93b7b635612b767a46600/home-energy-model-energy-performance-certificates-methodology.pdf), ch. 3]

| HEM:EPC metric | Definition (DESNZ) | Maps to our component |
| --- | --- | --- |
| **Fabric performance** | Fabric Energy Efficiency (FEE): modelled space heating + cooling demand, with the real heating system replaced by one standard system so the system type does not leak in. Depends on insulation, windows, airtightness, solar gains. C/D boundary aligned with today's EER C/D. Scotland's equivalent: "Heat Retention Rating". | Insulation, Windows & glazing |
| **Heating system** | Modelled efficiency **and** emissions intensity of space heating, hot water and cooking (time-varying grid carbon factor). No home relying on a fossil-fuel primary system can reach C; hybrid with fossil ≤ D; heat pumps and low-carbon heat networks always ≥ C; direct electric without storage ≤ D; efficient storage heating typically ≥ C. More efficient fossil systems still rank above less efficient ones (condensing gas > old gas > oil). | Heating system |
| **Smart readiness** | 1–100 score, A–G. Self-generation (PV capacity and self-consumption), electric batteries, smart thermal storage (heat batteries, storage heaters, cylinders), smart heating controls, smart meter, smart/bidirectional EV charger, grid connection. | Renewable energy, Smart energy management |
| **Energy cost** | Predicted £ bill, shown in £ rather than a band. | Energy efficiency (today's EER is cost-based) |
| Secondary | Delivered energy demand (kWh/m²) and carbon emissions. Legacy EER kept for transition. | Energy intensity (but metered, not modelled) |

**Why this matters for us:** the government's own decomposition of "green" lines up with the brief's seven components. We can cite it as the external justification for our sub-score definitions, and our heating sub-score ordering should follow the DESNZ banding principles above.

### 1.3 Other standards and benchmarks

| Standard | Type | What it measures / sets | Use for us |
| --- | --- | --- | --- |
| **NEED** (DESNZ) | Real metered statistics | Annual gas and electricity per home with type, age band, floor-area band, region, EPC band, loft/cavity/PV measures [verified: NEED 2025 metadata, downloaded] | Metered kWh/m² peer benchmarks and an ML target (real meters) |
| **EPC live tables** (MHCLG) | Area statistics | Band mix and mean modelled kWh/m² by LA × quarter | Peer context; sampling the band when the rating is missing (decision D-02) |
| **Approved Document L 2021** (Part L) | Building regulations | Limiting U-values for new and renovated elements in existing homes (Tables 4.2/4.3); e.g. replacement windows 1.4 W/m²K or WER band B [secondary; exact table values **to verify**] | "Good practice" anchors for U-value → sub-score curves |
| **Future Homes Standard / FEE** | New-build | Fabric Energy Efficiency target (kWh/m²/yr) | Upper anchor for fabric |
| **PAS 2035 / PAS 2030** | Retrofit **process** | No energy targets: whole-house assessment, medium-term improvement plan, fabric first, ventilation, risk paths [secondary: gov.scot retrofit review] | Ordering of recommendations (Part 2), not scoring |
| **AECB Retrofit Standard** | Voluntary performance | Space heating ≤ 50 kWh/m²/yr (≤ 100 by exemption); airtightness 2 ACH@50 Pa (Level 2), looser Level 1 [secondary: [Passive House Plus](https://passivehouseplus.co.uk/news/general/aecb-launches-two-levels-of-retrofit-standards)] | Top anchor for intensity / fabric |
| **LETI Climate Emergency Retrofit Guide** | Voluntary guidance | Retrofit space heating ~130 → 50 kWh/m²/yr; whole-home energy-use-intensity target (≈ 60 kWh/m²/yr **to verify**); 60–80% cut for the average home [secondary] | Top anchor for intensity curve |
| **EnerPHit** (Passivhaus retrofit) | Certified performance | Heating demand limit by climate (≈ 25 kWh/m²/yr cool-temperate **to verify**); airtightness; MVHR required [secondary] | "Best possible" anchor |
| **MEES** (private rented) | Regulation | Today EPC E minimum; reform: fabric standard first, then heating-system or smart-readiness standard; higher standards from 2028/2030 [secondary] | Business relevance (Lloyds buy-to-let book) |
| **ECO4** | Grant scheme | Scores measures by modelled SAP band movement [secondary: [BRE report for Ofgem](https://www.ofgem.gov.uk/sites/default/files/2022-01/Calculations%20of%20data%20to%20support%20ECO4%20scoring%20methodology1641913651196.pdf)] | Retrofit value of each measure (Part 2) |
| **Green mortgages** | Lender products | Rate or cashback for EPC A/B (often C) | Business relevance; not a scoring method |
| **Home Quality Mark** (BRE) | New-build quality | Broad: energy, wellbeing, running costs. Not applicable to existing homes. Not researched further. | — |

---

## 2. Parameter catalogue

Every parameter the standards above use, the component(s) it feeds, and where it comes from. Columns:
- **EPC field**: the column in the EPC register (CSV data dictionary, verified [here](https://get-energy-performance-data.communities.gov.uk/download/data-dictionary?property_type=domestic)).
- **NEED**: the column in NEED anonymised 2025, if any.
- **Homeowner route**: how we get it without an EPC (question, photo, document, bill), and the fallback when it is unknown.
- **Component**: EE = energy efficiency, INT = intensity, INS = insulation, HEAT = heating, GLZ = windows & glazing, REN = renewables, SMART = smart management. "ctx" = context or normaliser, not scored directly.

### 2.1 Dwelling and context

| Parameter | Standard(s) | EPC field | NEED | Homeowner route | Fallback when unknown | Component |
| --- | --- | --- | --- | --- | --- | --- |
| Postcode → region, LA, LSOA, climate (HDD), off-gas, conservation area | SAP climate region; RdSAP 1-1/1-2 | `POSTCODE`, `LOCAL_AUTHORITY` | `REGION` | Ask postcode; ONSPD lookup, then keep only district + LSOA | — (required) | ctx |
| Property type, built form | RdSAP 1-5/1-6 | `PROPERTY_TYPE`, `BUILT_FORM` | `PROP_TYPE` | Question (pick list with pictures) | — (required) | ctx (peer group) |
| Construction age band | RdSAP 2-1 (age band drives default U-values) | `CONSTRUCTION_AGE_BAND` | `PROP_AGE_BAND` (4 bands) | Question "roughly when built?"; Land Registry new-build date if sold new | EPC live-table mix for LA × type | ctx; drives default U-values for INS |
| Total floor area (TFA) | SAP normaliser (`TFA + 45`) | `TOTAL_FLOOR_AREA` | `FLOOR_AREA_BAND` | EPC lookup; room count × median room area; floor-plan photo (Part 1 capture) | Habitable rooms → TFA regression on EPC | ctx (denominator for EE, INT) |
| Storeys, storey height | RdSAP 2-4 | `FLOOR_HEIGHT`, `FLAT_STOREY_COUNT` | — | Question | 2.5 m default | ctx |
| Habitable / heated rooms | RdSAP 1-7 | `NUMBER_HABITABLE_ROOMS`, `NUMBER_HEATED_ROOMS` | — | Question | Bedrooms + 2 | ctx; partly-heated home flag |
| Flat position (floor level, top storey, corridor) | RdSAP 1-11..1-14 | `FLOOR_LEVEL`, `FLAT_TOP_STOREY`, `HEAT_LOSS_CORRIDOR` | — | Question | — | INS (which elements exist) |
| Tenure | RdSAP 1-4 | `TENURE` | — | Question | unknown | ctx (MEES) |

### 2.2 Fabric (heat retention)

| Parameter | Standard(s) | EPC field | NEED | Homeowner route | Fallback | Component |
| --- | --- | --- | --- | --- | --- | --- |
| Wall construction (solid brick/stone, cavity, timber, system) | RdSAP 4-1..4-3; HEM fabric | `WALLS_DESCRIPTION` | — | Question + photo: wall thickness at a window reveal (> ~260 mm ⇒ likely cavity for brick); brick bond (headers visible ⇒ solid) | Age band × type prior (EHS / EPC) | INS |
| Wall insulation (none, cavity fill, internal, external, as built) and thickness | RdSAP 4-4..4-7; ECO | `WALLS_DESCRIPTION`, `WALLS_ENERGY_EFF` | `CWI_FLAG` (+ date) | Question; photo of drill-hole pattern (cavity fill) or render (EWI); installer guarantee (CIGA) | As-built default for age band | INS |
| Wall U-value | SAP | (in description when measured) | — | Rarely known; from type + insulation + age via RdSAP tables | RdSAP default table | INS |
| Roof type (pitched loft, room in roof, flat, dwelling above) | RdSAP 2-3, 5-x | `ROOF_DESCRIPTION` | — | Question | — | INS |
| Loft insulation depth (mm) | RdSAP 5-2 (12…400 mm) | `ROOF_DESCRIPTION`, `ROOF_ENERGY_EFF` | `LI_FLAG` (+ date) | Photo of a tape measure between joists; question with depth pictures | **Never 0** when unknown: sample from age-band prior, flag `missing_data` | INS |
| Floor type and insulation | RdSAP 3-1..3-4 | `FLOOR_DESCRIPTION`, `FLOOR_ENERGY_EFF` | — | Question (suspended timber / solid; insulated?) | As built for age band | INS |
| Glazing type (single, secondary, double pre/post-2002, triple) | RdSAP 6-4..6-11 | `WINDOWS_DESCRIPTION`, `GLAZED_TYPE`, `WINDOWS_ENERGY_EFF` | — | Photo of the spacer bar (date stamp ⇒ pre/post 2002); question | — | GLZ |
| Share of multiple glazing | RdSAP | `MULTI_GLAZE_PROPORTION` | — | Question (all / most / some) | 100% if "double" | GLZ |
| Glazed area relative to typical | RdSAP | `GLAZED_AREA` | — | Question | Typical | GLZ (solar gain, ctx) |
| Doors (number, insulated) | RdSAP 6-1..6-3 | — | — | Question | Defaults | INS (minor) |
| Draught proofing % | RdSAP 9.1 | (in windows description) | — | Question | Age default | INS / GLZ (airtightness) |
| Open fireplaces, ventilation type, air-test result | RdSAP 7-1, 10-1; HEM fabric; AECB | `NUMBER_OPEN_FIREPLACES`, `MECHANICAL_VENTILATION` | — | Question; air-test certificate if any | Natural ventilation | INS (airtightness) |
| Conservatory | RdSAP 1-9 | — | `CONSERVATORY_FLAG` | Question | none | ctx |

### 2.3 Heating, hot water, lighting

| Parameter | Standard(s) | EPC field | NEED | Homeowner route | Fallback | Component |
| --- | --- | --- | --- | --- | --- | --- |
| Main heating system and fuel | RdSAP 7-2..7-5; HEM heating system metric | `MAINHEAT_DESCRIPTION`, `MAIN_FUEL`, `MAINS_GAS_FLAG`, `MAINHEAT_ENERGY_EFF`, `MAINHEAT_ENV_EFF` | `MAIN_HEAT_FUEL` | Question + photo of the boiler/heat-pump label | — (required) | HEAT |
| Boiler make/model → seasonal efficiency (PCDB) | SAP Appendix D / PCDB | (expanded in software) | — | Photo of data plate; PCDB lookup | Table default by type and age | HEAT |
| System age, condensing or not | SAP | (description) | — | Question; install date on plate | Age-band default | HEAT |
| Heat-pump SCOP | SAP/HEM | (expanded) | — | MCS certificate | Electrification of Heat prior | HEAT |
| Secondary heating | RdSAP 7-8 | `SECONDHEAT_DESCRIPTION`, `SHEATING_ENERGY_EFF` | — | Question | none | HEAT |
| Hot-water system, cylinder insulation and thermostat | RdSAP 7-9..7-15 | `HOTWATER_DESCRIPTION`, `HOT_WATER_ENERGY_EFF` | — | Question / photo of cylinder | From main | HEAT |
| Heating controls (programmer, room stat, TRVs, smart, zoning, weather compensation) | SAP Table 4e; HEM smart readiness | `MAINHEATCONT_DESCRIPTION`, `MAINHEATC_ENERGY_EFF` | — | Question with pictures; smart thermostat app | Programmer only | SMART |
| Lighting (LED / CFL / incandescent count) | RdSAP 10 item 12-1 | `LOW_ENERGY_LIGHTING`, `LOW_ENERGY_FIXED_LIGHT_COUNT`, `FIXED_LIGHTING_OUTLETS_COUNT`, `LIGHTING_ENERGY_EFF` | — | Question ("about how many bulbs are LED?") | Age default | SMART (not scored today, see D-08) |

### 2.4 Generation, storage, smart

| Parameter | Standard(s) | EPC field | NEED | Homeowner route | Fallback | Component |
| --- | --- | --- | --- | --- | --- | --- |
| Solar PV (kWp, orientation, tilt) | RdSAP 11-1; HEM smart readiness | `PHOTO_SUPPLY` (% of roof) | `PV_FLAG` (+ date) | MCS certificate / inverter label; or panel count × ~400 W; roof photo | % roof × roof area × 0.2 kWp/m² | REN |
| PV battery (kWh), diverter | RdSAP 11-3/11-4; HEM | — | — | Question / label | 0 | REN, SMART |
| Solar thermal | RdSAP 8-1 | `SOLAR_WATER_HEATING_FLAG` | — | Question | no | REN |
| Wind turbine, hydro | RdSAP 11-5..11-7 | `WIND_TURBINE_COUNT` | — | Question | 0 | REN |
| Smart meter present | RdSAP 1-17/1-18 (does not affect SAP); HEM smart readiness | — | — | Question | unknown | SMART |
| Electricity tariff (single, dual, time-of-use) | RdSAP 13; HEM | `ENERGY_TARIFF` | — | Bill | single | SMART |
| EV charger (smart / bidirectional) | HEM smart readiness | — | — | Question | none | SMART |
| Grid connection (phase, looped supply, fuse) | HEM smart readiness | — | — | Question (often unknown) | unknown | SMART (future) |

### 2.5 Outcomes (targets, or the outcome-type components)

| Parameter | Source | Field | Homeowner route | Component |
| --- | --- | --- | --- | --- |
| SAP rating / band (EER) | EPC | `CURRENT_ENERGY_EFFICIENCY`, `CURRENT_ENERGY_RATING` | EPC lookup by address; else **sampled from LA peer band mix** (decision D-02) | EE |
| Environmental Impact Rating | EPC | `ENVIRONMENT_IMPACT_CURRENT` | EPC lookup | ML target |
| Modelled CO2 / m² | EPC | `CO2_EMISS_CURR_PER_FLOOR_AREA` | EPC lookup | ML target |
| Modelled primary energy / m² | EPC | `ENERGY_CONSUMPTION_CURRENT` | EPC lookup | ML target; twin validation |
| Modelled heating / hot water / lighting cost | EPC | `HEATING_COST_CURRENT`, … | EPC lookup | ML target (cost) |
| **Metered** gas and electricity (kWh/yr) | NEED (real); homeowner | `Gcons*`, `Econs*` | 12 monthly bills; annual statement; smart-meter data with consent | INT |
| Occupants, working from home | — (not in SAP) | — | Question (optional) | Context only: explains metered vs modelled gap; never lowers the score (fairness) |

---

## 3. From homeowner input to the seven components

| Component (brief weight) | Inputs (Section 2) | Minimum homeowner effort | Best available evidence |
| --- | --- | --- | --- |
| Energy efficiency (25%) | SAP rating | Address → EPC lookup | Lodged EPC; projected via `sap_estimator` after retrofits |
| Energy intensity (20%) | Metered grid import + gas, TFA | Upload 12 bills or one annual statement | Smart-meter monthly data |
| Insulation (15%) | Walls, roof/loft depth, floor, airtightness | 4 questions with pictures | Loft tape photo, CIGA guarantee, EPC descriptions |
| Heating system (15%) | System, fuel, age, efficiency | 2 questions + boiler photo | Data plate → PCDB efficiency; MCS certificate |
| Windows & glazing (10%) | Glazing type, share | 1 question + spacer-bar photo | EPC windows description |
| Renewable energy (10%) | PV kWp, battery, solar thermal | 1 question | MCS certificate, inverter label |
| Smart energy management (5%) | Controls, smart meter, tariff, EV charger | 1 checklist question | Thermostat app / bill |

**Missing inputs never become 0.** Each unknown is sampled from its prior (age-band, LA or peer distribution) inside the score's Monte Carlo, so it widens the score interval and raises a `missing_data` key issue. See D-02 and D-04 in DECISIONS.md.

---

## 4. Learning the weights: options and recommendation

The headline score keeps the brief's fixed weights (decision D-05). A **second, learned-weights score** is fitted from data and reported beside it, with the comparison.

### 4.1 What do we predict? (multi-target, decision D-06)

| Target | Data | Real or modelled | Rows | Risk |
| --- | --- | --- | --- | --- |
| SAP rating (EER, cost) | EPC register | Modelled | millions | **Circular** with the EE component (EE *is* SAP). Must drop EE from the inputs when SAP is the target. |
| EIR / CO2 per m² (carbon) | EPC register | Modelled | millions | Shares inputs with SAP, but is a different outcome (carbon, not cost) |
| Primary energy kWh/m² | EPC register | Modelled | millions | Similar |
| **Metered** kWh/m² (gas + electricity) | NEED 50k / 4M | **Real** | 50k / 4M | Coarse features (bands, flags); behaviour noise; floor area only banded |

Proposal (needs sign-off, see the open question in DECISIONS.md D-06): learn weights for the **five attribute components** (INS, HEAT, GLZ, REN, SMART) on each target; the two outcome components (EE, INT) are themselves targets, so their weights are set from out-of-sample predictive value (Section 4.3), not by regressing a target on itself.

### 4.2 Models

| # | Model | What it gives | Pros | Cons | Role |
| --- | --- | --- | --- | --- | --- |
| M1 | **Constrained linear regression** (weights ≥ 0, Σw = 1, on sub-scores 0–100) | Weights directly in the brief's form | Same functional form as the score; transparent; scipy/cvxpy | Assumes additivity; collinearity can zero a weight | **Primary weights** |
| M2 | Ridge / OLS on standardised sub-scores | Standardised coefficients → normalised weights | Simple baseline | Negative weights possible | Baseline |
| M3 | **Shapley / dominance (LMG) R² decomposition** | Share of explained variance per component | Robust to correlated components (insulation ↔ EE) | Not a prediction model | Robustness check |
| M4 | **XGBoost / LightGBM with monotone constraints** on raw parameters + **grouped SHAP** | Importance share per component; non-linear fit ceiling | Captures interactions; monotone; fast | Importances ≠ linear weights | Upper bound + importance check |
| M5 | **EBM (InterpretML)** with monotone shape functions | Learned sub-score *curves* (e.g. loft mm → points) and term importances | Glass box; learns the mappings the brief leaves to us | Slower to train | **Learned sub-score mappings** |
| M6 | **Bayesian linear model, Dirichlet prior on w** centred on the brief's weights (NumPyro) | Posterior over weights with credible intervals | Fits our Bayesian philosophy; shows how far data moves from the brief | Needs care with priors | **Final reported weights with uncertainty** |

Recommended pipeline: M5 learns monotone sub-score curves → sub-scores 0–100 → M1 and M6 learn the weights (M6 for intervals) → M3 and M4 check robustness and the non-linear ceiling → compare against the brief's weights on held-out R², rank correlation (Spearman) and band agreement.

### 4.3 Cross-database fitting
- **EPC register:** per-home components (from descriptions) + SAP, EIR, CO2/m², primary energy. Fits weights for INS, HEAT, GLZ, REN, SMART on the modelled targets.
- **NEED:** real metered kWh/m² with coarse components (EPC band, loft and cavity insulation flags, PV flag, fuel). Fits weights on real consumption for the components NEED can see.
- **Join by archetype**, not by home: type × age band × floor-area band × region cells. Aggregate EPC component means per cell and join NEED metered medians, then fit a cell-level regression (an ecological check, reported as such).
- **Splits:** by property (EPC: UPRN; NEED: row), 70/15/15, seed 42; held-out test used once. Spatial-block check by local authority.

> **As built (11 Oct 2026):** the per-home NEED fit was tried and rejected (held-out R² 0.04: behaviour dominates single-home meters). The archetype join is what identifies the energy-efficiency and intensity weights. It uses type × age × floor-area band (no region: the EPC sample has no region code yet), compares within floor-area bands, and uses Shapley/LMG R² shares because SAP summarises the attributes. See DECISIONS.md D-18 and the results in LEARNED_WEIGHTS.md.

### 4.4 Evaluation (what we will report)
- Held-out R² / MAE of each weighting (learned vs brief vs equal weights) for each target.
- Rank agreement (Spearman) and band agreement between the learned and the brief score.
- Weight posteriors (M6) with 90% intervals; stability across bootstrap resamples and regions.
- Monotonicity: 0 violations in a sweep (learned curves are monotone by construction).
- Fairness: residuals by floor-area band and property type (large homes not penalised).

---

## Sources
- [RdSAP 10 specification (BRE, 12 Feb 2024)](https://files.bregroup.com/SAP/RdSAP10-dt13.02.2024.pdf)
- [RdSAP 10 specification (BRE, Jun 2025)](https://bregroup.com/documents/d/bre-group/rdsap-10-specification-10-06-2025)
- [SAP 10.2 specification (BRE)](https://bregroup.com/documents/d/bre-group/sap-10-2-11-04-2023-pdf)
- [SAP 10.3 specification (BRE, Jun 2026), not yet reviewed](https://bregroup.com/documents/d/bre-group/sap-10-3-full-specification-2026-06-12)
- [DESNZ: The Home Energy Model: Energy Performance Certificates (2026)](https://assets.publishing.service.gov.uk/media/69b93b7b635612b767a46600/home-energy-model-energy-performance-certificates-methodology.pdf)
- [EPC domestic data dictionary](https://get-energy-performance-data.communities.gov.uk/download/data-dictionary?property_type=domestic)
- [NEED anonymised data 2025](https://www.gov.uk/government/statistics/national-energy-efficiency-data-framework-need-anonymised-data-2025)
- [Mayer Brown: Reforming the EPB regime (May 2026)](https://www.mayerbrown.com/insights/publications/2026/05/reforming-the-energy-performance-of-buildings-regime---new-uk-government-proposals)
- [Element Sustainability: partial Government response overview](https://elementsustainability.co.uk/reform-to-energy-performance-of-buildings-partial-government-response-overview/)
- [Building Energy Experts: EPC reform 2026](https://buildingenergyexperts.co.uk/resources/2026-epc-reform-technical-update/)
- [AECB retrofit standard levels (Passive House Plus)](https://passivehouseplus.co.uk/news/general/aecb-launches-two-levels-of-retrofit-standards)
- [LETI Climate Emergency Retrofit Guide (Good Homes summary)](https://kb.goodhomes.org.uk/guidance/climate-emergency-retrofit-guide/)
- [gov.scot: retrofit assessment review (PAS 2035)](https://www.gov.scot/publications/review-retrofit-assessment-scotland-improving-home-energy-efficiency-policy-report/pages/11/)
- [BRE for Ofgem: ECO4 scoring data](https://www.ofgem.gov.uk/sites/default/files/2022-01/Calculations%20of%20data%20to%20support%20ECO4%20scoring%20methodology1641913651196.pdf)
- [Pilkington: Part L existing dwellings](https://www.pilkington.com/en-gb/uk/architects/standards-and-regs/building-regulations/england/conservation-of-fuel-and-power/existing-dwellings)
