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
| D-14 | EPC data: stream the 8.3 GB bulk zip over HTTP Range, keep a 10% property sample by UPRN hash | Decided |

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

## Worked examples (headline score, brief's fixed weights)
`uv run python scripts/score_showcase.py` (4,000 samples, seed 42; intensity from the stub, D-13).

### SH1: Victorian mid-terrace, 85 m², EPC E (SAP 48), 11 months metered
Intensity 219.2 kWh/m²/yr (90% 210.0–228.4). **Score 30/100, Poor** (90% 29.9–31.0; P(Poor or Very poor) = 1.00)

| Component | Weight | Sub-score | Contribution | Why |
| --- | --- | --- | --- | --- |
| Energy efficiency | 0.25 | 48.0 | 12.00 | recorded SAP 48 |
| Energy intensity | 0.20 | 49.2 | 9.85 | 219 kWh/m², between the 200 (55) and 250 (40) points |
| Insulation | 0.15 | 14.5 | 2.17 | solid wall none (0) ×0.45 + 50 mm loft (25) ×0.40 + uninsulated floor (30) ×0.15 = 14.5 |
| Heating system | 0.15 | 20.0 | 3.00 | non-condensing gas 30 − (22 − 12) years = 20 |
| Windows & glazing | 0.10 | 24.0 | 2.40 | 60% single (0) + 40% double pre-2002 (60) = 24 |
| Renewables | 0.10 | 0.0 | 0.00 | no PV |
| Smart | 0.05 | 20.0 | 1.00 | programmer only |
| **Total** | 1.00 | | **30.43 → 30** | |

Key issues: poor insulation (high), old heating (medium), single glazing (high), no renewables, basic controls. Compare the brief's own example (rating E, poor insulation, single glazing, gas boiler → 38, Poor).

### SH2: 1970s cavity semi, 95 m², EPC D (SAP 60), 12 months
Intensity 170.5 kWh/m²/yr (90% 163.8–177.4). **Score 49/100, Poor** (90% 48.5–49.3; P(Poor or Very poor) = 0.99). Contributions: EE 15.00, intensity 12.78, insulation 5.06 (unfilled cavity 25, 100 mm loft 45, floor 30), heating 8.10 (condensing gas 55 − 1 for 13 years), glazing 6.00, renewables 0, smart 2.00 → 48.94 → 49. One point below Moderate: filling the cavity alone (wall 25 → 75) adds 0.15 × 0.45 × 50 = 3.4 points → 52, Moderate.

### SH3: 2004 all-electric flat, 62 m², EPC C (SAP 70), 12 months
Intensity 74.2 kWh/m²/yr. **Score 65/100, Good**. Insulation 100 (roof and floor dropped: dwellings above and below; cavity as built 2004 → 100). Heating 35 (storage heaters) is the main drag, plus programmer-only controls.

---

## Data on hand
| Dataset | Path (git-ignored) | Rows | Notes |
| --- | --- | --- | --- |
| NEED anonymised 2025, 50k sample | `data/raw/need/anon2025_50k.csv` | 50,000 | Real annual gas/electricity 2005–2023; type, age band (4), floor-area band (5), region, EPC band (A/B…F/G), loft/cavity/PV flags, main fuel, IMD, council-tax band |
| NEED metadata | `data/raw/need/NEED-2025-metadata.ods` | — | Field definitions |
| EPC register, 10% property sample, 2008–2026 | `data/interim/epc/certificates-<year>_s10.parquet` | ≈ 2.8M (extracting) | 70 columns incl. SAP, EIR, CO2/m², element ratings and descriptions (D-14) |
