# Green Home Score: decisions, proofs and worked examples

Running log for Part 3 (Green Home Score). Every decision has an id, the reason, and (where it applies) a quantitative proof or a worked example. Numbers in "Proof" blocks are produced by code in the repo and can be rerun; the command is given next to each.

Branch: `worktree-p3-green-score`. Research background: `docs/score/GREEN_SCORE_STANDARDS.md`.

## Decision index

| Id | Decision | Status |
| --- | --- | --- |
| D-01 | Score band = band of the **rounded** score (brief), not the mode of the band probabilities | Decided (user, 11 Oct 2026) |
| D-02 | Missing SAP/band → **sample the band from local-authority peers** (EPC live table D1), then SAP uniformly within the band | Decided (user, 11 Oct 2026) |
| D-03 | Rounding is half-up: `floor(x + 0.5)`, not Python's banker's `round` | Decided |
| D-04 | Unknown loft depth / roof state is never 0: sampled from the age-band prior and flagged | Decided |
| D-05 | Report **both** scores: headline uses the brief's fixed weights; a learned-weights score is reported beside it with a comparison | Decided (user, 11 Oct 2026) |
| D-06 | Weights are learned on **multiple targets** (SAP, carbon, metered kWh/m²) | Decided (user); circularity handling **open** |
| D-07 | `sap_estimator.json` uses the SAP 2012 formula; current EPCs use SAP 10.2 | Flagged to Part 2 owner |
| D-08 | Low-energy lighting is not scored in v0.1 mappings | Open |
| D-09 | Environment: Python 3.12 via uv; `odfpy` added for ODS files (NEED metadata, EPC live tables) | Done |
| D-10 | Wall `as_built` counts as insulated only for age bands from 1976 on | Decided |
| D-11 | Flat / room-in-roof insulation inferred from roof U-value, else loft depth, else age band (≥ 1983) | Decided |
| D-12 | Score uses posterior-mean inputs for the point score; samples give the distribution | Decided |
| D-13 | Stub `EnergyBreakdown` from metered bills until Part 2's baseline lands | Decided (temporary) |
| D-14 | EPC data: bulk zip (8.3 GB) → deterministic property sample by UPRN hash (10% remote; 25% from the local copy) | Decided |
| D-15 | U-value curves for walls, roof, floor (`score_mapping.json` v0.2.0); roof U goes through RdSAP Table 16 | Decided |
| D-16 | EPC text → sub-score parsing rules (assumed depths, unknown condensing status, glazing date) | Decided |
| D-17 | NEED assumptions: floor-area band midpoints (band 5 = 250 m²), PV flag = 3 kWp, "not gas" = electric | Decided |
| D-18 | Cross-target weights via EPC↔NEED archetype cells, within-floor-area-band, Shapley (LMG) ratios | Decided |
| D-19 | Learned score reported as `ScoreResult.alternatives` (contract addition) | Decided; needs consumer review |
| D-20 | EPC targets are era-specific (fuel prices, carbon factors): fit final weights on the RdSAP 10 era only; other eras = stability check | Decided (raised by user, 11 Oct 2026) |
| D-21 | Priors fitted from real data (band mix, loft depth by age, NEED peer kWh/m²) | Done |
| D-22 | Missing `PHOTO_SUPPLY` = unknown PV, not "no PV" (bug found in v1 results) | Fixed |
| D-23 | EBM curve check: loft curve too linear (data + physics: diminishing returns); heating map between cost and carbon orderings | Findings |
| D-24 | Loft curve = share of max heat-loss cut (RdSAP Table 16), `score_mapping.json` v0.3.0; insulation key issue judged per element | Decided (user, 11 Oct 2026) |
| D-25 | Heating map: physics (SAP 10.2 carbon + primary energy per kWh heat) tried and **rejected**: fits the data worse than the current map | Experiment; map unchanged, **open** |
| D-26 | Lighting scored inside smart management: 0.53 × controls + 0.47 × low-energy lighting % (share learned), mapping v0.4.0 | Decided (user asked to score lighting) |

---

## D-01 Band from the rounded score
**Brief §2:** "Round the final score to the nearest integer for reporting and classification." The contract comment on `ScoreResult.band` said "mode of band_probabilities". These can disagree near a boundary: e.g. a score of 49.4 rounds to 49 (Poor) while 55% of posterior samples land in Moderate.

**Decision:** `band` = classification of the rounded score. `band_probabilities` stays as extra information. The contract field description is updated to match (contract change; needs review by consumers: Parts 5 and UI).

## D-02 Missing rating: sample from peers
When a home has neither `epc.sap_score` nor `epc.current_band`, the energy-efficiency sub-score cannot be dropped (weights are fixed). We draw the band from the home's local authority band mix (EPC live table D1, latest four quarters), then draw SAP uniformly within that band's range, per Monte Carlo sample. `rating_source = "missing"`, a `missing_data` key issue is raised, and the score interval widens accordingly. If the LA is unknown, the regional, then England & Wales, mix is used.

## D-03 Half-up rounding
Python `round(48.5) == 48` (banker's rounding). The brief expects 48.5 → 49. We use `math.floor(x + 0.5)`. A unit test pins this.

## D-04 Unknown roof state
`pitched_loft` with `loft_insulation_mm = None` is unknown, not 0 mm. It is sampled from the age-band prior (to be fitted on EPC roof descriptions) and flagged.

## D-05 Two scores
Judging criterion "Correct use of the fixed weights" (brief §11) and AGENTS.md both require the brief's weights. The **headline** Green Home Score uses them. The **learned-weights score** is an evidence-based companion: same sub-scores, weights fitted from data (GREEN_SCORE_STANDARDS.md §4). Both are shown with their difference.

## D-06 Multi-target weights (open sub-question)
Targets: SAP (EPC), EIR or CO2/m² (EPC), metered kWh/m² (NEED). **Circularity:** the EE component *is* the SAP rating and the INT component *is* metered kWh/m², so regressing SAP on all seven components gives EE a weight near 1 by construction. Proposed handling: learn weights for the five attribute components per target; set the EE and INT weights from their out-of-sample predictive value for the *other* target (e.g. EE predicting metered use, INT predicting carbon). Awaiting confirmation.

## D-07 SAP formula version
`contracts/sap_estimator.json`: SAP 2012 (deflator 0.42; `117 − 121·log10(ECF)`; `100 − 13.95·ECF`). SAP 10.2 §13 [verified in the BRE PDF]: deflator 0.36; `108.8 − 120.5·log10(ECF)` for ECF ≥ 3.5; `100 − 16.21·ECF` otherwise. Because the estimator only uses the *change* anchored to the recorded SAP, the effect is second-order, but the cited method should match the EPCs it anchors to. Owner: Part 2.

## D-08 Lighting
SAP and RdSAP 10 rate lighting (`LIGHTING_ENERGY_EFF`), and `Property.low_energy_lighting_pct` exists, but `score_mapping.json` does not score it. Candidate: fold into smart energy management or insulation-free "efficiency extras". Left open; the ML weights (D-06) will tell us whether it carries signal.

## D-09 Environment
`uv python install 3.12 && uv sync` → baseline `uv run pytest -q`: **30 passed** (11 Oct 2026). Added `odfpy` (needed by pandas to read `.ods`).

## D-10 Wall "as built"
`score_mapping.json` gives `as_built` 100 points ("insulated to building regs at construction"). EPC mapper output marks e.g. "Solid brick, as built, no insulation (assumed)" as as-built, and a 1910 solid wall is not insulated. Building Regulations first set a wall U-value limit in the mid-1970s [to verify exact year], so `as_built` is treated as `none` for age bands starting before 1976. Monotonicity is unaffected: `as_built ≥ none` holds for every type and age (test `test_wall_insulation_never_lowers_score`).

## D-11 Flat and room-in-roof insulation
`Roof` has no `insulated` flag, but `score_mapping.json` scores `flat_insulated` vs `flat_uninsulated`. Rule, in order: roof U-value ≤ 0.35 W/m²K (RdSAP 10 note 13 threshold) ⇒ insulated; else loft depth ≥ 100 mm ⇒ insulated; else age band from 1983 ⇒ insulated as built [to verify against RdSAP Table 16 roof U-values by age]; otherwise uninsulated.

## D-12 Point score vs distribution
The contract says `score` is the weighted sum at the posterior-mean inputs. For non-linear (clamped, piecewise) sub-scores the mean of sample scores can differ slightly from the score at mean inputs. Both are reported: `score` (mean inputs) and `score_distribution` (per-sample). SH1–SH3 below differ by < 0.3 points.

## D-13 Stub energy input
`greentwin/score/stub_energy.py` builds `EnergyBreakdown.intensity_kwh_m2` from monthly bills: observed months summed; missing gas months filled from a per-home degree-day fit `gas = a + b·HDD` (HDD from SAP 10.2 Table U1 UK-average temperatures, base 15.5 °C); missing electricity months from the observed monthly mean; bootstrap over months + 3% meter noise → distribution. Never ×12 and never 0. Replaced by Part 2's `baseline()` when it lands.

## D-14 EPC data acquisition
The API's bulk file is an 8.3 GB zip (19 yearly certificate CSVs + 19 recommendation CSVs, updated 9 Oct 2026). Free disk: E: 2.6 GB, D: 14 GB, so the zip cannot be stored and unzipped (≈ 30 GB of certificate CSVs). `scripts/epc_bulk_extract.py` opens the zip remotely (HTTP Range on the pre-signed S3 URL), streams each yearly CSV, keeps 70 columns and a deterministic 10% sample of properties (`md5(UPRN) mod 100 < 10`, so all certificates of a sampled home stay together for history and splits). Test on 2008: 69,322 / 691,496 rows kept (10.02%).

## D-15 U-value curves
New-build EPCs (SAP, report type 3) describe elements only as "Average thermal transmittance X W/m²K", and some homeowners know U-values. When a U-value is known it is scored by a curve; lower U never lowers the sub-score (test `test_u_value_curves_monotonic_and_consistent_with_loft_table`).
- Roof: U → equivalent loft depth through **RdSAP 10 Table 16** [verified in the BRE PDF: 0 mm 2.3, 50 mm 0.68, 100 mm 0.40, 150 mm 0.30, 200 mm 0.21, 270 mm 0.16, 400 mm 0.11 W/m²K] → loft curve. Proof of consistency: `roof_u_sub(0.16) == loft_sub(270)` and `roof_u_sub(0.40) == loft_sub(100)` exactly.
- Walls: `[[0.18,100],[0.30,92],[0.45,82],[0.60,70],[1.0,40],[1.5,25],[1.7,0]]`, anchored on RdSAP 10 Table 6 (England): cavity as built pre-1976 1.5 ⇒ 25 (= `cavity_none`), solid brick as built 1.7 ⇒ 0 (= `solid_none`), modern as built 0.18–0.30 ⇒ 92–100.
- Floor: `[[0.15,100],[0.25,90],[0.45,60],[0.70,30]]`; uninsulated floors (~0.7) ⇒ 30 (= `uninsulated`).

## D-16 EPC text parsing (`greentwin/score/epc_features.py`)
Parsed once per distinct string. Rules that need a judgement:
- "Pitched, insulated (assumed)" with no depth ⇒ 100 mm (cautious); "limited insulation" ⇒ 25 mm.
- Flat / room-in-roof "limited insulation" ⇒ midpoint of insulated and uninsulated.
- Walls "partial insulation" ⇒ midpoint of none and as built; explicit "insulated" on the EPC is not age-gated (D-10 applies only to the ambiguous contract value).
- **Condensing status is not in EPC text.** Main-heating stars: Very good ⇒ condensing, Average/Poor ⇒ non-condensing, Good ⇒ unknown, scored as the 50/50 mix (42.5). Evidence: in the 2019 sample 99.5% of gas boilers are "Good" (2008: Good 61.6%, Very good 22.2%, Average 16.2%), so stars cannot separate them in recent EPCs. Limitation: within-gas heating variation is lost on the EPC.
- Glazing: double with unknown install date ⇒ half pre-2002, half post-2002; "High performance glazing" ⇒ post-2002 double; partial/mostly/some ⇒ `MULTI_GLAZE_PROPORTION`, else 50/80/20%, remainder single.
- PV: `PHOTO_SUPPLY` (% of roof) × footprint (TFA ÷ storeys: 2 for houses, 1 otherwise) × 1.15 pitch × 0.18 kWp/m².
- Controls: words → contract controls (e.g. "Time and temperature zone control" ⇒ programmer + room stat + TRVs + zoning; HHR storage controls ⇒ programmer + stat + TRVs).

**Validation against EPC element stars** (Spearman, 159k 2019 certificates):

| Our sub-score | EPC element rating | Spearman ρ | n |
| --- | --- | --- | --- |
| walls | WALLS_ENERGY_EFF | 0.892 | 158,630 |
| roof | ROOF_ENERGY_EFF | 0.903 | 122,588 |
| floor | FLOOR_ENERGY_EFF | 0.635 | 19,096 |
| windows & glazing | WINDOWS_ENERGY_EFF | 0.903 | 159,074 |
| heating system | MAINHEAT_ENERGY_EFF | 0.893 | 159,073 |
| smart management | MAINHEATC_ENERGY_EFF | 0.770 | 158,590 |

Our documented mappings rank homes almost exactly as the official SAP element ratings do, while staying transparent and monotonic.

## D-17 NEED assumptions
Floor-area band midpoints 40 / 75.5 / 125.5 / 175.5 / 250 m² (band 5 "over 200" is open-ended; 250 assumed). Only meter reads flagged V (valid) are used; gas = 0 only when the home has no gas meter and is not gas-heated. PV flag ⇒ 3 kWp. "Gas is not the main heating fuel" ⇒ scored as direct electric (most non-gas homes; oil/LPG cannot be told apart in NEED).

## D-18 How the cross-target weights are actually identified
First attempt (per-home NEED): metered kWh/m² is barely explained by NEED's coarse fields: **test R² 0.044** (EE + proxies → metered) and **0.055** (intensity + proxies → SAP) on 600k held-out homes. Behaviour dominates single-home meter readings, so per-home NEED cannot weigh EE against attributes (the fit gave EE 96% purely because the proxies are weak).

Fix, the archetype join planned in GREEN_SCORE_STANDARDS.md §4.3:
1. Cells = NEED property type × NEED age band × NEED floor-area band. EPC homes are mapped onto the same bands (`greentwin/score/archetypes.py`). Each cell gets mean EPC sub-scores and mean SAP from the EPC sample, and median **metered** kWh/m² from NEED (4M sample). Cells need ≥ 30 homes in both sources.
2. Within each floor-area band (fixed effects): kWh/m² falls with size (median 164 → 80 kWh/m² from band 1 to band 5) and NEED's area is banded, which otherwise swamps everything (pooled R² ≈ 0). Within bands, cell SAP correlates −0.63 to −0.71 with metered kWh/m².
3. SAP summarises the attributes, so plain NNLS gives it 100% of the weight (collinearity). The ratio therefore uses **Shapley/LMG R² shares**, which split credit fairly between correlated predictors: `r_EE = LMG(EE) / Σ LMG(attributes)` when predicting −metered kWh/m², and `r_INT = LMG(INT) / Σ LMG(attributes)` when predicting SAP.
4. Final: `T = 1 / (1 + r_EE + r_INT)`, `w_EE = r_EE·T`, `w_INT = r_INT·T`; the five attributes share `T` in proportion to their per-home EPC weights (mean over the four modelled targets). Bootstrap over cells gives intervals.

This is an ecological (cell-level) analysis: it shows how components relate on average across home types, not within a single home.

## D-19 Where the learned score lives
`ScoreResult.alternatives: list[AlternativeScore]` (name, weights, score, band, distribution, weights_version, method). The headline `score` and `band` always use the brief's weights. `GreenHomeScorer` loads the latest `artifacts/score_weights/v*/weights.json` automatically (`learned_weights_dir=None` turns it off). Contract addition: needs review by the UI and LLM owners.

## D-20 Prices and carbon factors change with the EPC era
The SAP rating is a **cost** rating and the EIR is a **carbon** rating. Each SAP version prices energy with its own fixed fuel-price table and uses its own carbon factors, and the EPC register mixes certificates from 2008 to 2026. Pooling them means a home's target depends on *when* it was assessed, not only on its fabric and systems. Electric heating moves most: SAP 10.2's grid electricity factor (≈ 0.136 kgCO₂/kWh vs ≈ 0.519 in SAP 2012 [to verify exact values]) raises the EIR of electrically heated homes, and the new price table moves their SAP rating. That would bias the **heating** weight.

| Era (`ERAS` in `train_score_weights.py`) | Lodgement dates [boundaries to verify] | Method |
| --- | --- | --- |
| `sap2009` | 2008 – Mar 2012 | RdSAP on SAP 2009 |
| `sap2012` | Apr 2012 – 14 Jun 2025 | RdSAP 9.9x on SAP 2012 |
| `rdsap10` | from 15 Jun 2025 | RdSAP 10 on SAP 10.2 (deflator 0.36) |

Decision: the final weights are fitted on **RdSAP 10 certificates only** (the methodology, prices and grid carbon that current EPCs use, and what a homeowner sees on a new EPC). Each property's latest certificate *within the era* is used. The same fit on the other eras is reported as a stability check (`metrics.era_stability`). Stage 2 cell means use the same era. NEED meter data is one year (2023), so it has no price drift. Using the £ cost fields was avoided from the start because they are in each era's price year.

Dev evidence of the effect (10% sample): SAP-target heating weight 0.590 in the SAP 2009 era vs 0.416 pooled.

## D-21 Priors fitted from data
`scripts/fit_score_priors.py` → `greentwin/score/priors.json`:
- National band mix (latest certificate per property, lodged in the last 10 years, n = 3,529,238 in the 25% sample): A 1.1%, B 15.7%, C 39.8%, D 33.3%, E 8.4%, F 1.3%, G 0.4%. Used when neither the local-authority mix nor a rating is available.
- Loft depth by age band (pitched lofts with a recorded depth, n = 3,103,402): e.g. 1900–1929 homes have 0 mm in 29.1% of cases, 2007–2011 homes in 0.4%. Used for unknown loft depths (D-04).
- NEED peer median metered kWh/m² by property type × floor-area band (cells with ≥ 100 homes). Used for the `high_intensity` key issue, so large homes are compared with large homes.

## D-22 Missing PV field is unknown, not zero
The first full v1 run gave renewables a weight of 0.020, and exactly 0 on both carbon targets. Investigation (2026 certificates, 25% sample, n = 345,722): `PHOTO_SUPPLY` is null on 14.8% of rows, and on **100% of new-build SAP certificates** (report type 3) vs 4.1% of RdSAP ones. Those homes average SAP 84.5 vs 68.3. The parser had turned null into 0 kWp, so the greenest homes looked like "no PV", which pulled the renewables weight down. This broke our own "missing ≠ 0" rule. Fix: null ⇒ NaN; such rows are excluded from the weight fits instead of being scored as having no PV (test `test_missing_pv_is_unknown_not_zero`). Separately, a genuinely lower carbon weight for PV under SAP 10.2 is expected: the low grid-electricity factor means PV displaces less carbon.

## D-23 EBM curve check (does each sub-score curve rank homes like the data?)
`uv run python scripts/fit_score_curves_ebm.py --target sap` and `--target eir` → `artifacts/score_curves/v1/{sap,eir}/metrics.json`. EBM (InterpretML, monotone, no interactions) on raw parameters of the latest certificate per property, 4 most recent years, 300k sample, split by property (210k train / 45k test). Held-out R²: SAP 0.632, EIR 0.613.

**Tautology disclosed:** walls, floor, glazing and smart terms were fed our own sub-score, so their Spearman of 1.0 is by construction and is not evidence. The independent checks are loft (raw mm), heating (category) and PV (kWp).

| Independent term | Spearman vs our sub-score (SAP / EIR) |
| --- | --- |
| PV kWp | 0.932 / 0.932 |
| Heating system (per home) | 0.705 / 0.912 |
| Heating system (12 category means) | 0.837 / 0.739 |
| Loft depth | 0.578 / 0.560 |

**Loft: diminishing returns.** Mean EBM points by recorded depth (SAP; EIR similar): 0 mm −6.6, 50 mm −1.7, 100 mm −0.5, 200 mm +0.1, 270 mm +0.4, 400 mm +1.3. About 78% (SAP) / 74% (EIR) of the 0→400 mm gain is reached by 100 mm. RdSAP 10 Table 16 says the same: U falls 2.3 → 0.40 W/m²K by 100 mm, 87% of the possible reduction. Our curve gives 45 of 100 points at 100 mm. Ranking is preserved (both monotone), but the spacing over-rewards top-ups and under-rewards the first insulation. **Proposal (not applied):** loft sub-score = 100 × (U₀ − U(mm)) / (U₀ − U_min) with U from Table 16 (0 mm 0; 50 mm 74; 100 mm 87; 200 mm 95; 270 mm 99; 400 mm 100). Still monotone, physics-based, consistent with the roof U-value path (D-15).

**Heating: our order sits between cost and carbon.** Cost (SAP) ranks condensing gas top, heat pumps near neutral, LPG worst. Carbon (EIR, SAP 10.2 factors) ranks ASHP best (+7.9), district heating and GSHP next, oil worst (−15.5). Our map (heat pumps top, solid fuel / oil bottom) follows the carbon end and the reformed-EPC heating-metric principles (§1.2 of the standards doc). Mismatches on carbon: direct electric −1.3 (ours 30, low), storage −7.3 / HHR −6.3 (ours 35 / 50, high), and "solid fuel" mixes coal and biomass (no biomass enum value). Left as is for now; noted for the heating-map backlog item.

## D-24 Physics-based loft curve (applied)
Approved after D-23. `roof_loft_mm_points` = `100 × (2.3 − U(mm)) / (2.3 − 0.11)` at every RdSAP 10 Table 16 depth:

| mm | 0 | 12 | 25 | 50 | 75 | 100 | 150 | 200 | 270 | 300 | 400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Old | 0 | 6 | 12.5 | 25 | 35 | 45 | 65 | 80 | 100 | 100 | 100 |
| **New** | 0 | 36.5 | 59.4 | 74.0 | 82.2 | 86.8 | 91.3 | 95.4 | 97.7 | 98.6 | 100 |

Why: the sub-score now measures what loft insulation does, the share of the possible heat-loss cut, and its spacing matches both physics and the EBM-learned curve (D-23). It is monotone (test `test_loft_curve_is_share_of_heat_loss_cut` pins the formula at 6 table depths), and the roof U-value path (D-15) goes through the same table, so a known U and a known depth give the same score.

Effect on the showcase homes (headline, brief weights): SH1 30 → **33** Poor; SH2 49 Poor → **51 Moderate**; SH3 unchanged at 65 (no loft). SH2's band flip is a real change in meaning: a 1970s semi with 100 mm of loft insulation has already captured most of the loft benefit.

Side effect fixed: with a better roof score, SH2's insulation average (50.5) no longer triggered `poor_insulation`, hiding its unfilled cavity walls (sub-score 25). The key issue is now raised if the average **or any element** is below 40, and names the weak elements (test `test_good_loft_does_not_hide_unfilled_cavity`).

Learned weights retrained as **v2** on the new sub-scores (v1 kept for the record).

## D-25 Heating map: physics version tried and rejected
Goal (user, after D-23): fix the heating map. Tried a sourced physics map: per kWh of delivered heat, carbon `C = ef / eff` and primary energy `P = pef / eff`, both from SAP 10.2 (Table 12 factors: mains gas 0.210 kgCO₂e/kWh and PEF 1.130, electricity 0.136 / 1.501, LPG 0.241 / 1.141, heating oil 0.298 / 1.180, house coal 0.395 / 1.064; Table 4a/4b default efficiencies: GSHP 230%, ASHP 170%, electric 100%, gas condensing 84%, non-condensing 74%, oil 80%, LPG 79% (mean), coal appliance 65%), each min-max scaled and averaged.

Result, Spearman across 11 heating categories against the EBM points learned from EPC data (D-23):

| Map | vs cost ordering (SAP) | vs carbon ordering (EIR) |
| --- | --- | --- |
| **Current hand-ordered map (kept)** | **0.879** | **0.697** |
| Physics, carbon only | 0.593 | 0.506 |
| Physics, carbon + primary energy | 0.556 | 0.533 |

Why physics loses: (1) SAP models storage heaters with poor responsiveness and off-peak tariffs, so per-kWh carbon ties them with direct electric while the data separates them; (2) the EPC "solid fuel" class mixes coal with wood (low carbon), so a coal-only assumption is wrong for the class; (3) the cost and carbon data disagree with each other (storage beats direct on cost and loses on carbon; LPG is worst on cost but mid-table on carbon), so no single physics quantity matches both. Decision: keep the current map (`score_mapping.json` unchanged for heating). The open question is which target defines "green" for heating (see the question to the user).

## D-26 Lighting scored (inside smart energy management)
The brief fixes seven components, so lighting cannot be an eighth: it joins `smart_energy_management`, documented as "controls and efficient electricity use". `sub = (1 − a) × controls_level + a × low_energy_lighting_pct`, with `a` learned by `scripts/fit_lighting_split.py` (RdSAP 10 era, simplex NNLS with lighting as its own attribute, 100 bootstraps):

| Target | Lighting share of the component | 90% bootstrap | Held-out R² without → with lighting |
| --- | --- | --- | --- |
| SAP (cost) | 0.463 | 0.456–0.468 | 0.303 → 0.329 |
| Primary energy / m² | 0.472 | 0.459–0.479 | 0.274 → 0.294 |
| EIR (carbon) | 1.0 (controls weight already 0: not informative) | — | 0.224 → 0.242 |
| CO₂ / m² | 1.0 (same) | — | 0.209 → 0.226 |

`a = 0.47`, the mean of the two targets where the split is identified. Lighting improves fit on every target. Missing lighting % is sampled from the EPC prior (`priors.json`: 60.5% of homes have 100% low-energy lighting, 3.6% have 0%; n = 3,528,294) and widens the interval. The `no_controls` key issue is judged on the controls level alone, so good lighting never hides missing controls. Overlap with energy efficiency (SAP counts lighting) is accepted on the same basis as insulation. Tests: `test_more_low_energy_lighting_never_lowers_score`, `test_missing_lighting_is_sampled_and_controls_issue_ignores_lighting`; the controls monotonicity sweep now runs at lighting 0, 50 and 100%. Learned weights retrained as **v3**.

---

## Proofs

### P-01 Monotonicity sweeps (brief §2: a better state never gets a lower sub-score)
`uv run pytest -q tests/test_score.py`: 16 passed, **0 violations**.

| Sub-score | Sweep | Points checked |
| --- | --- | --- |
| Energy efficiency | SAP −20…129 | 150 |
| Energy intensity | 0…800 kWh/m² | 2,001 |
| Loft depth | 0…499 mm | 500 |
| PV | 0…10 kWp, ± battery, ± solar thermal | 101 + 6 |
| Walls | 5 wall types × 3 age bands × (none → cavity fill → internal → external) and as_built ≥ none | 75 |
| Glazing | single → secondary → double pre/post-2002 → triple; and single→triple share swaps | 5 + 10 |
| Controls | every subset of 6 controls (64) × adding each control | 384 |
| Heating | 12 systems × boiler age 0…59 | 720 |

Also: every `HeatingSystem` value is mapped; weights sum to 1; contributions sum to the score within rounding; band = classification of the rounded score; half-up rounding (`48.5 → 49`, while Python `round(48.5) = 48`).

### P-02 Missing data widens, never zeroes
- Missing rating on SH2, peer mix B 10% / C 40% / D 40% / E 10%: `rating_source = missing`, `missing_data` issue raised, score std larger than with the recorded SAP (test `test_missing_rating_sampled_from_peers_and_widens_interval`).
- Unknown loft depth scores **higher** than a recorded 0 mm and carries a contribution distribution (test `test_unknown_loft_is_sampled_not_zero`).

### P-03 Projected score not lower for an improvement
SH1 with an estimated rating SAP ~ N(60, 3) vs recorded 48: projected score ≥ current (test `test_projected_score_not_lower_for_improvement`).

---

## Worked examples (headline score, brief's fixed weights; mapping v0.3.0)
`uv run python scripts/score_showcase.py` (4,000 samples, seed 42; intensity from the stub, D-13). Scores before the D-24 loft change are kept in brackets.

### SH1: Victorian mid-terrace, 85 m², EPC E (SAP 48), 11 months metered
Intensity 219.2 kWh/m²/yr (90% 210.0–228.4); NEED peer median for a mid-terrace of 51–100 m² is 144.4. **Score 33/100, Poor** (90% 32.8–33.9; P(Poor or Very poor) = 1.00) [was 30].

| Component | Weight | Sub-score | Contribution | Why |
| --- | --- | --- | --- | --- |
| Energy efficiency | 0.25 | 48.0 | 12.00 | recorded SAP 48 |
| Energy intensity | 0.20 | 49.2 | 9.85 | 219 kWh/m², between the 200 (55) and 250 (40) points |
| Insulation | 0.15 | 34.1 | 5.11 | solid wall none (0) ×0.45 + 50 mm loft (74.0) ×0.40 + uninsulated floor (30) ×0.15 = 34.1 |
| Heating system | 0.15 | 20.0 | 3.00 | non-condensing gas 30 − (22 − 12) years = 20 |
| Windows & glazing | 0.10 | 24.0 | 2.40 | 60% single (0) + 40% double pre-2002 (60) = 24 |
| Renewables | 0.10 | 0.0 | 0.00 | no PV |
| Smart | 0.05 | 20.0 | 1.00 | programmer only |
| **Total** | 1.00 | | **33.36 → 33** | |

Key issues: high intensity vs peers (high), poor insulation: walls (high), old heating (medium), single glazing (high), no renewables, basic controls. Compare the brief's own example (rating E, poor insulation, single glazing, gas boiler → 38, Poor).

### SH2: 1970s cavity semi, 95 m², EPC D (SAP 60), 12 months
Intensity 170.5 kWh/m²/yr (90% 163.8–177.4); peer median 162.9. **Score 51/100, Moderate** (90% 51.0–51.8; P(Poor or Very poor) = 0.00) [was 49, Poor]. Contributions: EE 15.00, intensity 12.78, insulation 7.57 (unfilled cavity 25 ×0.45 + 100 mm loft 86.8 ×0.40 + floor 30 ×0.15 = 50.5), heating 8.10 (condensing gas 55 − 1 for 13 years), glazing 6.00, renewables 0, smart 2.00 → 51.45 → 51. Key issues still name the unfilled cavity (walls sub-score 25, D-24). Filling the cavity (wall 25 → 75) adds 0.15 × 0.45 × 50 = 3.4 points → 55.

### SH3: 2004 all-electric flat, 62 m², EPC C (SAP 70), 12 months
Intensity 74.2 kWh/m²/yr; peer median 87.4. **Score 65/100, Good** [unchanged]. Insulation 100 (roof and floor dropped: dwellings above and below; cavity as built 2004 → 100). Heating 35 (storage heaters) is the main drag, plus programmer-only controls.

---

## Results: learned weights v2 (current; RdSAP 10 era, PV fix, mapping v0.3.0)
Full tables: [LEARNED_WEIGHTS.md](LEARNED_WEIGHTS.md). Reproduce: `uv run python scripts/train_score_weights.py --epc-dir data/interim/epc25 --boot 100 --version v2`. v1 (mapping v0.2.0) kept in `artifacts/score_weights/v1` for comparison.

| Component | Brief | Learned v1 | **Learned v2** | v2 90% bootstrap |
| --- | --- | --- | --- | --- |
| Energy efficiency | 0.25 | 0.213 | **0.223** | 0.170–0.324 |
| Energy intensity | 0.20 | 0.203 | **0.208** | 0.149–0.253 |
| Insulation | 0.15 | 0.247 | **0.230** | 0.187–0.268 |
| Heating system | 0.15 | 0.192 | **0.196** | 0.159–0.229 |
| Windows & glazing | 0.10 | 0.049 | **0.051** | 0.042–0.060 |
| Renewables | 0.10 | 0.045 | **0.046** | 0.037–0.053 |
| Smart management | 0.05 | 0.050 | **0.046** | 0.038–0.054 |

- **The brief's weights are close to the data on energy efficiency, intensity and smart management.** The data puts more on insulation and heating and less on glazing and renewables. The weights barely moved between v1 and v2 (largest change 0.017), so they are robust to the loft-curve change.
- Held-out R² (447,139 RdSAP 10 properties, attribute block only): learned beats the brief's weights and equal weights on all four modelled targets, e.g. SAP 0.303 vs 0.275 vs 0.262. v2's R² is slightly below v1's (SAP 0.320): with the new curve, "insulated (assumed)" lofts (scored at 100 mm) sit at 86.8 points, compressing loft differences on the EPC. Non-linear ceiling (monotone LightGBM incl. extras) 0.589.
- Era stability (D-20), SAP target heating weight (v2): SAP 2009 era 0.572 (n = 1,342,875), SAP 2012 era 0.466 (n = 3,623,688), RdSAP 10 era 0.440 (n = 447,139, used). It moves with the methodology, which is why only one era is used. (v1, before the PV fix and loft change: 0.52 / 0.38 / 0.43.)
- Candidate extras (LightGBM |SHAP| share, SAP target, v1): low-energy lighting 0.101 and open fireplaces 0.074 carry real signal. Evidence for D-08 (score lighting).
- Brief vs learned on 66,761 EPC test homes: Spearman 0.983, same band 83.2%, mean |difference| 2.7 points; Poor or Very poor 11.0% (brief) vs 10.7% (learned).

### Worked examples with both scores
`uv run python scripts/score_showcase.py` (priors and NEED peers fitted from data, D-21):

| Home | Headline (brief weights) | Learned v2 | Peer median kWh/m² (NEED, same type × size) | Key issues |
| --- | --- | --- | --- | --- |
| SH1 Victorian terrace, EPC E | **33, Poor** | 35, Poor | 144 (home: 219) | high intensity (high), poor insulation: walls, old heating, single glazing, no renewables, basic controls |
| SH2 1970s semi, EPC D | **51, Moderate** | 54, Moderate | 163 (home: 171) | high intensity (medium), poor insulation: walls, no renewables |
| SH3 2004 flat, EPC C | **65, Good** | 70, Good | 87 (home: 74) | storage heating, no renewables, basic controls |

SH3 gains 5 points under the learned weights because full insulation (sub-score 100) is weighted 0.230 instead of 0.15.

---

## Data on hand
| Dataset | Path (git-ignored) | Rows | Notes |
| --- | --- | --- | --- |
| NEED anonymised 2025, 50k sample | `data/raw/need/anon2025_50k.csv` | 50,000 | Real annual gas/electricity 2005–2023; type, age band (4), floor-area band (5), region, EPC band (A/B…F/G), loft/cavity/PV flags, main fuel, IMD, council-tax band |
| NEED metadata | `data/raw/need/NEED-2025-metadata.ods` | — | Field definitions |
| EPC register, 10% property sample, 2008–2026 | `data/interim/epc/certificates-<year>_s10.parquet` | ≈ 2.8M (extracting) | 70 columns incl. SAP, EIR, CO2/m², element ratings and descriptions (D-14) |
