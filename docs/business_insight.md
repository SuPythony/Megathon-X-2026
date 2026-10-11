# Business insights: the GreenTwin demo city and the Lloyds dashboard

Owner: Satyarth (Part 1, digital twin), with the dashboard owner (Block 8). Status: **plan**, 11 Oct 2026.
Read with: `docs/DIGITAL_TWIN_HANDOFF.md` (what the twin produces), `docs/IMPLEMENTATION_PLAN_v2.md` Block 8
(dashboard), `contracts/green_profits_assumptions.json` (Green Profits scenario rule).

This doc does three things:
1. Defines a **demo city of precomputed digital twins**: built once in batch, never simulated live, feeding the dashboard.
2. Lists the **business insights** for Lloyds, including **rental increase potential from greening**, each with its
   formula, outputs, visual and caveats.
3. States, for every insight, **which part produces each input** (Parts 0–5, orchestrator, assumptions).

---

## 0. Ground rules (apply to every number on the dashboard)

- Every number is either **computed** by a documented model or an **explicit, labelled assumption** on a slider.
  Nothing is trained on synthetic labels (AGENTS.md, problem statement §9).
- Uncertain numbers are `Distribution`s (`contracts.summarise`): headline + 90% range everywhere.
- **Provenance rule:** results from the stub physics carry contract engine `mock` and must not appear as real results.
  The demo city is rebuilt automatically when the real physics (`greentwin/twin/physics.py`) and calibration land.
- Value and rent effects are **indicative** associations, never promises. "Greener homes are lower credit risk" is
  a **hypothesis**, not a finding.
- **Privacy:** no addresses or names. Location is shown only as LSOA or postcode district. Customers are a
  **synthetic customer base over real geography**, always labelled.

---

## 1. The demo city

### 1.1 What and why
A city's worth of homes, each with a digital twin and a set of pre-run retrofit scenarios. It's computed **once, in
batch**, stored, and read by the dashboard. Live simulation is only for the individual-home demo (3D viewer + twin
server); the city view must load instantly and replay offline.

### 1.2 Which city and how many homes
- **City:** Greater Manchester (postcode area `M`) as the default, since it's the SH1 showcase district (`M14`). The
  pipeline takes any list of postcode districts, so Bristol (`BS`, SH2) and Croydon (`CR`, SH3) can be added for
  the three EPC regions the plan names.
- **Homes:** real EPC records (domestic EPC open data) from those districts, **deterministically sampled** (e.g. 2,000–
  5,000 homes; seed 42). p3 already holds a 25% EPC extract (`data/interim/epc25/`, built by
  `scripts/epc_bulk_extract.py`) that can be the source.
- **Drill-down homes:** ~50 homes across the main house types also get the full room-level scene (3D) so the dashboard
  can open "a home like this" in the viewer.

### 1.3 Build pipeline (`scripts/build_demo_city.py`, to build)

```
EPC records (Part 0) ──► Property per home (Part 0 mapper)
        │
        ├─► twin: archetype layout, params_from_property, thermal mass, provenance      (Part 1)
        ├─► prior samples (n = 200 per home, seeded) → simulate baseline                  (Part 1)
        ├─► feasibility per retrofit (rules)                                              (Part 5)
        ├─► standard packages, each re-simulated as a whole                               (Part 1 run_scenario /
        │     "fabric first", "heat pump", "PV + battery", "all feasible"                  Part 2 package/ladder)
        ├─► score before / after (fixed weights + learned companion)                      (Part 3)
        ├─► rating after (SAP estimator)                                                  (Part 2)
        ├─► value impact (indicative)                                                     (Part 4)
        ├─► rent impact (indicative, new; §2.8)                                           (Part 4 + Part 0)
        └─► stress test: bills under price and weather scenarios                          (Part 1 + Part 0)
                 │
                 ▼
data/processed/demo_city/homes.parquet   (one row per home: CityHome, §5)
data/processed/demo_city/areas.json      (DashboardArea per LSOA / district, + extensions §5)
greentwin/twin/examples/city/<id>.scene.json   (drill-down homes)
```

- Batch, cached by `hash(home + model versions)`, so re-runs only recompute what changed.
- Runtime: stub physics with 200 samples is about 0.1–0.3 s per home per package (jit-compiled, vmapped), so a few
  thousand homes × 5 runs fit in minutes. Re-run when the real physics or calibration lands.
- **Synthetic customer base** (Block 8): customers per LSOA = real home count × assumed market share (slider).
  Each customer is a real EPC record from that LSOA with no personal attributes; tenure and mortgage flags are
  synthetic and labelled.

---

## 2. Insights catalogue

Legend for **Inputs**: P0 data layer · P1 digital twin · P2 energy & retrofit engine · P3 Green Home Score · P4 value ·
P5 feasibility / LLM · ORCH orchestrator · ASSUME labelled assumption (slider) · SRC sourced config in `contracts/`.

### 2.1 Green Home Score map and poor-home hotspots
- **Question:** where are the green homes and the poor ones?
- **Metric:** share with score ≥ 65 (`share_green`); share Poor / Very poor (`share_poor`); hotspots (Getis-Ord Gi*,
  local Moran's I) of poor homes.
- **Visual:** heat map layer + hotspot overlay; distribution of scores by area.

| Input | From | Field / source | Status |
|---|---|---|---|
| Score per home, band, P(Poor) | P3 | `ScoreResult.score`, `band`, `p_poor_or_very_poor` | Built (p3, merged) |
| Energy intensity for the score | P2 (P1 until then) | `EnergyBreakdown.intensity_kwh_m2` | Stub: `greentwin/score/stub_energy.py` |
| LSOA geometry, adjacency | P0 | ONS boundaries | To build (P0) |

### 2.2 Improvement potential and finance demand
- **Question:** how much could homes improve, and how much finance does that need?
- **Metric:** improvement potential = 100 × (E_base − E_after) / E_base (all feasible retrofits together); finance demand
  = package cost − grants, summed per area.
- **Visual:** finance demand (£) layer; segments table ("1930s semi, gas boiler, owner-occupier": size, average loan
  need, carbon potential, share Poor).

| Input | From | Field / source | Status |
|---|---|---|---|
| Baseline and package energy per fuel | P1 (re-simulated) / P2 | `ScenarioView` / `ImprovementPotential` | P1 built (stub physics); P2 to build |
| Feasibility per retrofit | P5 | `FeasibilityReport` (rules) | To build (P5) |
| Package cost | SRC | `contracts/retrofit_catalogue.json` (BEIS 2017 × CPI; `cost_status`) | Sourced; some unsourced → excluded |
| Grants | SRC | catalogue `grants` | BUS amount **to verify** (AGENTS.md) |
| Segment keys | P0 + P1 | property type, age band, heating, tenure; archetype id | Built (P1 archetypes) |

### 2.3 Self-funding green loan share ⭐
- **Question:** for how many homes does a green loan pay for itself from bill savings?
- **Metric per home:** monthly repayment R = P · r / (1 − (1 + r)^−n), with P = package cost − grants, r = monthly
  rate, n = months. **P(self-funding) = P(monthly bill saving ≥ R)** over the twin's posterior samples. City metric =
  share of homes with P(self-funding) ≥ 0.8, and the £ lending they represent.
- **Visual:** map layer "self-funding share"; per-segment bar; slider for rate and term.

| Input | From | Field / source | Status |
|---|---|---|---|
| Bill saving per sample (gas + electricity × unit rates) | P1 | `run_scenario` savings samples; `contracts/tariffs.json` (Ofgem Q4 2026) | Built (stub physics) |
| Package cost, grants | SRC | retrofit catalogue | Sourced / partly to verify |
| Interest rate, term | ASSUME | slider (no rate quoted as fact) | To add to `green_profits_assumptions.json` |

### 2.4 Energy-cost stress test ⭐
- **Question:** how exposed are customers to energy price and weather shocks, before and after retrofit?
- **Metric per home:** annual bill under scenarios {price cap +0%, +30%, −20%} × {typical, cold year}; P(bill > X% of
  the area's median household income). City metric: share of customers above the threshold, before vs after the
  package.
- **Visual:** stress-test panel with scenario toggles; "customers protected by retrofit" count.

| Input | From | Field / source | Status |
|---|---|---|---|
| Bills per sample per scenario | P1 | `simulate` with `weather="cold_year"`; tariffs × shock | Price shock: built logic; cold year needs P0 weather |
| Cold-year weather by district | P0 | `load_weather(district, months)` / scenario | To build (P0) |
| Area income proxy | P0 | ONS small-area income estimates (MSOA), public | To source (P0) |
| Threshold X%, shock sizes | ASSUME | sliders | Labelled assumption |

### 2.5 Financed emissions and carbon per £ lent
- **Question:** where does green lending cut the most carbon per pound?
- **Metric:** tCO₂e/yr avoided per home = Σ_fuel ΔkWh × DESNZ factor; **kgCO₂e avoided per £1k lent** by segment;
  portfolio financed-emissions baseline vs after (PCAF-style attribution = loan share of property value, labelled).
- **Visual:** carbon potential layer; "carbon per £" ranking of segments.

| Input | From | Field / source | Status |
|---|---|---|---|
| ΔkWh per fuel (package) | P1 / P2 | savings samples | Built (stub physics) |
| Emission factors | SRC | `contracts/emission_factors.json` (DESNZ 2025) | Sourced |
| Loan amount | P2 + SRC | package cost − grants | As 2.2 |
| Property value (attribution) | P4 | `ValueImpact.current_value_gbp` | P4 |

### 2.6 Confidence-tiered offers
- **Question:** which customers can get an instant offer, and which need a survey first?
- **Metric:** tier A (instant): confirmed geometry or ≥ 50% of heat loss on recorded/measured inputs, ≥ 12 months of
  bills; tier B (desk check); tier C (survey needed: any retrofit `needs_survey`, or mostly default inputs). City
  metric: share per tier; cost per acquired customer (survey cost slider).
- **Visual:** funnel with tiers; map of tier A share.

| Input | From | Field / source | Status |
|---|---|---|---|
| Heat-loss share by input status | P1 | `PhysicsInputs.provenance_summary` | Built |
| Geometry confirmed | P1 | `PhysicsInputs.geometry.confirmed_by_user` | Built |
| Calibration readiness | P1 | `calibration_readiness.method` | Built |
| Needs-survey retrofits | P5 | `FeasibilityReport` | To build |
| Survey / visit cost | ASSUME | slider | Labelled assumption |

### 2.7 Heat-pump readiness
- **Question:** which homes are ready for a heat pump now?
- **Metric:** gas-heated + fabric heat-loss coefficient per m² below a threshold (or after "fabric first") + outdoor
  space + feasible; with the heat-pump package's P(cost saving > 0) and carbon saving.
- **Visual:** "heat-pump ready" layer; ready-now vs ready-after-insulation.

| Input | From | Field / source | Status |
|---|---|---|---|
| Heat-loss coefficient, fuel | P1 | element heat loss, `params.heating_fuel` | Built |
| Heat-pump package result | P1 | `run_scenario(["air_source_heat_pump", ...])` | Built (stub physics) |
| Outdoor space, constraints | P0 + P5 | `Property.constraints`, feasibility rules | P5 to build |
| BUS grant | SRC | catalogue grants | **To verify** |

### 2.8 Rental: compliance gap and rent increase potential from greening ⭐ (new)
- **Questions:** (a) which privately rented homes are below the minimum standard, and what does it cost to fix them?
  (b) How much could greening raise the rent? (c) Who benefits: landlord or tenant?
- **Compliance:** the Minimum Energy Efficiency Standard for private rentals in England & Wales requires at least
  band **E** (with exemptions). A move to **C** has been proposed. **Verify the current status and dates before
  citing.** Metric: share of rentals below E / below C now vs after the package; upgrade cost to reach each.
- **Rent increase potential:** rent_after = area rent × (1 + premium(band_after) − premium(band_before)), where area rent
  = ONS private rent level for the local authority × bedrooms, and premium(band) = the rent premium for that EPC band
  vs the area average. **Premium is a labelled assumption** (slider with a range) until a rent model is estimated
  (see below). Outputs: £/month and % rent uplift (Distribution, via the rating's band probabilities), annual rent
  gain, payback for the landlord = (cost − grants) / annual rent gain, change in gross yield = annual rent / value.
- **Split incentive:** tenant gains the bill saving (P1), landlord gains rent uplift + compliance + value (P4). Showing
  both explains why landlord loans need a different product.
- **Upgrade path to a model:** a rental hedonic in Part 4, mirroring the sales model: rent ~ EPC band + type +
  bedrooms + area. Data: ONS rents by LA (public) for levels; listing-level rents would be needed for the band effect
  (not public; flag as a data gap). Until then: assumption slider, labelled "indicative".
- **Visual:** rentals layer (share below E / C), landlord payback by segment, rent uplift range, split-incentive card.

| Input | From | Field / source | Status |
|---|---|---|---|
| Tenure (private rented) | P0 | `Property.tenure` | EPC field / synthetic customer flag |
| Band before | P0 | `Property.epc.current_band` | EPC |
| Band after (+ probabilities) | P2 | `RatingEstimate.band`, `band_probabilities` (SAP estimator) | To build (P2); note p3's SAP 2012 vs 10.2 flag |
| Area rent by LA × bedrooms | P0 | ONS private rents (public, OGL) | **To source** (new P0 dataset) |
| Rent premium per band | ASSUME → P4 | slider now; rental hedonic later | New `rent_assumptions` block |
| Package cost, grants | SRC | retrofit catalogue | As 2.2 |
| Property value (yield) | P4 | `ValueImpact` | P4 |
| Bill saving (tenant) | P1 | savings samples | Built (stub physics) |

### 2.9 Property value uplift (indicative)
- **Metric:** `ValueImpact.uplift_pct` and `value_after_gbp` from band before → band after; always with
  `ValueImpact.wording`.
- **Inputs:** P4 (Bayesian hedonic on real sales), P2 (band after), P0 (sales, HPI deflation; p3 note D-20).

### 2.10 Street deals (area-based installation)
- **Question:** where can bulk installs cut cost per home?
- **Metric:** clusters of ≥ N same-archetype homes with the same top package within an LSOA / street block (spatial
  clustering on LSOA centroids + archetype id). City metric: homes in clusters, indicative cost saving per home
  (discount slider).
- **Inputs:** P1 (archetype id, package), P0 (LSOA geography), ASSUME (bulk discount).

### 2.11 Timing triggers
- **Question:** when is the right moment to make an offer?
- **Metric:** heating system age ≥ typical life (boiler replacement window); EPC lodgement age (stale EPC = re-assess);
  synthetic remortgage date (labelled). Count of homes per trigger per month.
- **Inputs:** P0 (`Heating.system_age_years`, `EPCInfo.lodgement_date`), synthetic customer data (labelled), ASSUME
  (typical boiler life).

### 2.12 Customer engagement (individual homes, from the intake + 3D twin)
- Room story ("your bedrooms lose 40% of the heat"), appliance quick wins ("your old fridge-freezer costs about £X a
  year"), before/after 3D. These are **not** city-level: they come from homes that used the intake.
- **Inputs:** P1 (`room_energy`, appliance schedules, scenes), intake (user answers).
- **Dashboard use:** engagement funnel (users → profiles → plans → surveys → applications → loans; Block 8 step 7).

### 2.13 Green Profits (scenario)
- **Metric:** lending (balance × margin × term, discounted, + fees) + retention (customers retained × CLV × churn
  reduction) + acquisition (new customers × CLV − acquisition and survey costs). Take-up = the logistic rule in
  `green_profits_assumptions.json` on improvement potential, bill saving, owner-occupation and deprivation.
- **Inputs:** P1/P2 (improvement potential, bill saving), P0 (tenure, IMD decile), ASSUME (all coefficients and
  financial parameters), §2.3 (loan sizes), §2.8 (landlord loans).
- **Label:** "Illustrative scenario on a synthetic customer base. Not a forecast."

---

## 3. Input matrix: which part feeds which insight

| Insight | P0 data | P1 twin | P2 engine | P3 score | P4 value | P5 feasibility | SRC config | ASSUME |
|---|---|---|---|---|---|---|---|---|
| 2.1 Score map, hotspots | ● geography | ○ (energy until P2) | ● intensity | ● | | | | |
| 2.2 Potential, finance demand | ● | ● re-simulated packages | ● package / potential | | | ● | ● costs, grants | |
| 2.3 Self-funding loans | | ● savings samples | ○ | | | ○ | ● tariffs, costs | ● rate, term |
| 2.4 Stress test | ● weather, income | ● bills per scenario | | | | | ● tariffs | ● shocks, threshold |
| 2.5 Financed emissions | | ● ΔkWh | ○ | | ● value | | ● emission factors | ○ attribution |
| 2.6 Confidence tiers | | ● provenance, readiness | | | | ● needs survey | | ● survey cost |
| 2.7 Heat-pump readiness | ● constraints | ● HLC, package | | | | ● | ● grants | |
| 2.8 Rentals, rent uplift | ● tenure, band, ONS rents | ● tenant bill saving | ● band after | | ● value, rent model | ● | ● costs | ● rent premium |
| 2.9 Value uplift | ● sales, HPI | | ● band after | | ● | | | |
| 2.10 Street deals | ● LSOA | ● archetype, package | | | | | | ● bulk discount |
| 2.11 Timing triggers | ● system age, lodgement | | | | | | | ● boiler life |
| 2.12 Engagement | | ● rooms, appliances | | | | | | |
| 2.13 Green Profits | ● tenure, IMD | ● | ● | | | | | ● all coefficients |

● main input · ○ secondary.

---

## 4. Dashboard layout (proposed)

1. **City map** (MapLibre + deck.gl), layer switch:
   - share green
   - share Poor + hotspots
   - finance demand
   - self-funding share
   - carbon potential
   - heat-pump ready
   - rentals below E / C
   - rent uplift
   - confidence tier A share
   - street-deal clusters
2. **Segments table:** archetype × heating × tenure, with:
   - homes
   - average loan need
   - self-funding share
   - carbon per £1k
   - rent uplift (rentals)
   - share Poor
3. **Stress test panel:** price and weather toggles, with customers above the threshold before and after retrofit.
4. **Rentals panel:**
   - compliance gap (E / C)
   - landlord payback
   - rent uplift range
   - split-incentive card
5. **Green Profits:** sliders, Monte Carlo range and funnel, labelled.
6. **Drill-down:** click an area → a representative home → open its room-level twin in the 3D viewer (before/after).

Every panel shows its labels:
- "estimated" / "indicative"
- "assumption" next to slider values
- "synthetic customer base"
- "stand-in physics" until the real model lands

---

## 5. Data formats to add (proposed, additive)

- **`CityHome`** (one row per home in `homes.parquet`):
  - `property_id`, `lsoa`, `district`, `archetype_id`, `property_type`, `built_form`, `age_band`, `tenure`,
    `heating_fuel`
  - `score`, `band`, `p_poor`
  - per package: `{energy before/after per fuel, bill saving, carbon saving, cost, grants, p_self_funding, band_after,
    value_uplift, rent_uplift}` (headline + 90% range)
  - `stress: {scenario → bill, p_over_threshold}`
  - `confidence_tier`, `heat_pump_ready`
  - `triggers[]`
  - `engine`, `model_versions`
- **`DashboardArea` extensions** (the contract already has `share_green`, `share_poor`, `finance_demand_gbp`,
  `carbon_potential_t`, `synthetic_customers`). Add, all optional and Distribution-typed:
  - `self_funding_share`
  - `stress_share_before`, `stress_share_after`
  - `rentals_below_e_share`, `rentals_below_c_share`
  - `rent_uplift_pct`
  - `heat_pump_ready_share`
  - `tier_a_share`
  - `carbon_per_1k_lent_kg`
  - `street_deal_homes`
- **`rent_assumptions`** block in `contracts/green_profits_assumptions.json`: rent premium per band (range, status
  assumption), MEES thresholds (with "verify" note).
- **Loan assumptions** in the same file: rate and term sliders.

These are contract changes, so the consumers (dashboard, API) review them before merging.

---

## 6. Gaps and who closes them

| Gap | Owner | Needed for |
|---|---|---|
| Real physics (`physics.py`) and calibration | Physics teammate, calibration team | Every energy-based number |
| EPC mapper, LSOA geography, weather by district, cold-year scenario | P0 | 2.1, 2.4, demo city |
| ONS private rents by LA × bedrooms | P0 | 2.8 |
| ONS small-area income | P0 | 2.4 |
| Feasibility rules (`needs_survey`, constraints) | P5 | 2.2, 2.6, 2.7, 2.8 |
| Package / ladder / improvement potential, SAP estimator (band after; SAP 10.2 flag) | P2 | 2.2, 2.8, 2.9 |
| Rental hedonic (optional upgrade) | P4 | 2.8 |
| Verify: BUS grant value, MEES status, RdSAP defaults | Us + team lead | Labels |
| `build_demo_city.py`, `CityHome`, area extensions | Us (P1) + dashboard owner | All |

## 7. Does the data show that greening is better?

**Position: the data shows what the physics shows. Nothing is tuned to produce a result.**
- AGENTS.md: physics is the only source of numbers.
- Compliance fix 7: a synthetic green premium we insert may only be reported as a method check.
- A rigged result would also fail with judges and with Lloyds' own credit and climate teams.

What the physics shows is still a strong case. `config.json → summary.greening_evidence`. The figures below are from
the first run (500 synthetic homes, **stand-in physics**); the current build (60 homes, P2's `physics_l1`) is under
"First run" below:

| Package | Carbon falls | Bills fall | Median bill saving | Median carbon saving | Median cost | Saving covers the loan repayment: 6%/10y → 6%/25y → 4%/25y → 2%/25y |
|---|---|---|---|---|---|---|
| Fabric first | 100% | 100% | £148/yr | 292 kg/yr | £870 | 42% → 62% → 71% → 78% |
| Heat pump (+ fabric) | 100% | 100% | £225/yr | 1,984 kg/yr | £13,370 | 3% → 4% → 6% → 9% |
| Solar PV (3.5 kWp) | 100% | 100% | £230/yr | 171 kg/yr | £6,566 | 0% at every term |

**How to read it (and how to pitch it honestly)**
1. **Greening is better on carbon and bills in every modelled home.** It also makes homes more resilient: under a
   +30% price shock, the share of homes over the bill threshold falls from 61% to 46%.
2. **Fabric-first pays for itself** for a majority of homes on a long term. This is the "self-funding green loan"
   product.
3. **Heat pumps and PV are the big carbon wins but don't pay back on bills alone.** That isn't greening losing: it's
   the gap Lloyds' product design closes. The levers are all sliders, and each must be sourced before quoting:
   - longer terms and lower green rates
   - grants (BUS, not yet verified)
   - export payments for PV (no sourced export tariff yet)
   - value uplift (Part 4, indicative)
   - rent uplift (2.8, assumption)
4. **Not counted yet, and plausibly positive:** comfort, health, property value, rent, lower arrears risk. Each is
   either indicative (value, rent) or a hypothesis (credit risk), and labelled so.

**What would make the evidence real rather than illustrative:**
- the real physics and calibration on real meter data
- real EPC records (needs `EPC_API_TOKEN`)
- verified grants and export tariffs
- the Part 4 value model on real sales

The same code re-runs on them unchanged.

## 8. Status: step 1 built (11 Oct 2026)

**Built:** `greentwin/twin/city.py`, `greentwin/twin/city_assumptions.json`, `scripts/build_demo_city.py`, tests
`tests/test_demo_city.py`. Output: `artifacts/demo_city/v1/{homes.json, areas.json, config.json}` (60 homes by
default: P2's half-hourly PV/battery layer is too memory-heavy for hundreds of homes on a laptop).

- **Homes are synthetic** (no EPC extract or `EPC_API_TOKEN` on this machine). They're placed over real Manchester
  postcode districts (M8, M14, M19, M20, M21) with synthetic small areas (`M14-03`, never real LSOA codes). The housing
  mix is a labelled assumption. Loft insulation and low-energy lighting follow p3's EPC-fitted priors. Swap
  `synthetic_homes()` for an EPC mapper when real records are available; nothing downstream changes.
- Per home: twin params + provenance, 60 prior samples, baseline + 3 packages (`fabric_first`, `heat_pump`,
  `solar_pv`), each re-simulated as a whole. Homes run in batched, vmapped physics calls, grouped by fuel and by
  what P2's PV/battery inputs depend on (PV angles, heating hours): 60 homes in about 4 s on `physics_l1`.
- **Insights built:**
  - 2.3 self-funding loans: P(saving ≥ repayment) per package; best package per home.
  - 2.4 price stress test: base / +30% / −20% before and after the best package.
  - 2.5 carbon per £1k lent.
  - 2.6 confidence tiers for synthetic customers.
  - 2.10 street deals: ≥ 5 same-type homes with the same best package in a small area, plus the bulk saving.
- `areas.json` rows are `DashboardArea` (contract) plus the step-1 layers (`CityArea`, the proposed extension in §5).

**First run** (500 synthetic homes, **stand-in physics, uncalibrated, synthetic weather: illustrative only**):

| Metric | Value |
|---|---|
| Finance demand (best costed package per home) | £2.66M |
| Homes whose best package self-funds (P ≥ 0.8, 6% over 10 years) | 27.6% |
| Lending in those self-funding packages | £134k |
| Carbon saving (best packages) | 368 tCO₂e/yr |
| Homes over the £2,500 bill threshold: today / +30% price shock / +30% after retrofit | 32% / 61% / 46% |
| Customer tiers A / B / C | 2 / 53 / 39 |
| Homes in street-deal clusters | 146 |

**Current build** (60 synthetic homes, P2's `physics_l1`, uncalibrated, synthetic weather), share of homes where the
saving covers the repayment at 6% over 25 years: fabric first 58%, solar PV 8%, heat pump 0%. Carbon and bills fall
in 100% of modelled homes for all three packages. Same direction as the first run.

**What the first run already says:**
- Self-funding loans are mostly cheap fabric measures. The big-ticket packages (heat pump, PV) don't pay for
  themselves at the assumed rate and term, so the lending story for them is grants, longer terms or a lower rate.
  These are sliders in `city_assumptions.json`.
- Retrofit is a price-shock hedge: it cuts the share of homes over the bill threshold under a +30% shock by about a
  quarter.
- Few customers reach tier A, because every U-value is still an unverified RdSAP default. Verifying the defaults
  and collecting 12 months of bills is what unlocks instant offers.

**Not yet:**
- rentals (2.8), rating after, value, score map: these need P2, P3, P4 and P5
- the cold-year weather and income-based stress: these need P0
- drill-down scenes; a larger city needs the PV layer run in chunks

## 9. Build order (thinnest path first)

1. `build_demo_city.py` on 500 homes from one district with stub physics: `CityHome` + `areas.json`.
2. Insights that only need P1 + configs: **2.3 self-funding**, **2.4 stress test (price shocks)**, **2.5 carbon per £**,
   **2.6 confidence tiers**, **2.10 street deals**.
3. Add P3 score (merged) for 2.1; P5 rules and P2 rating as they land for 2.2, 2.7, **2.8 rentals**, 2.9.
4. Scale to the full city sample; drill-down scenes for ~50 homes.
5. Re-run everything when the real physics and calibration land: the banner switches off automatically when
   `engine == "physics"`.
