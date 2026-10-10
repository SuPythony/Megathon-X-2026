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

---

## Data on hand
| Dataset | Path (git-ignored) | Rows | Notes |
| --- | --- | --- | --- |
| NEED anonymised 2025, 50k sample | `data/raw/need/anon2025_50k.csv` | 50,000 | Real annual gas/electricity 2005–2023; type, age band (4), floor-area band (5), region, EPC band (A/B…F/G), loft/cavity/PV flags, main fuel, IMD, council-tax band |
| NEED metadata | `data/raw/need/NEED-2025-metadata.ods` | — | Field definitions |
| EPC register | **pending: path from user** | — | Needed for SAP/EIR/CO2 targets and per-element components |
