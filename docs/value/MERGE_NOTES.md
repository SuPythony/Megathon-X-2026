# Merge notes: Part 4 Property value → `main`

Branch `p4/property-value` (built on `p3/green-home-score`; merge that first, see `docs/score/MERGE_NOTES.md`). Decisions and evidence: [DECISIONS.md](DECISIONS.md) (V-xx). Research and benchmarks: [VALUE_RESEARCH.md](VALUE_RESEARCH.md).

**Contracts:** none changed. Implements `greentwin.interfaces.ValueModel` (`HedonicValueModel` in `greentwin/value/model.py`). Output is `ValueImpact` and `GreenTwin`, as already defined. All tests pass (`uv run pytest -q`).

## What each part should know or update
### Orchestrator / API (P4)
- `HedonicValueModel().value_impact(prop, rating_after)` needs `RatingEstimate` from Part 2's sap_estimator and `Property.epc.sap_score`. Without either it returns `method="unavailable"` with a reason (brief §11 scenario "insufficient information").
- The uplift applies to `Property.current_value_estimate_gbp`, never to a sale price. Without a current value, the uplift is given in % only, with a reason.
- `green_twins(prop)` returns `[]` when there aren't enough comparable greener sales; the UI shows a message. It needs `data/processed/sales_scored_v1.parquet` (git-ignored; build with the scripts below) or falls back to the precomputed SH1–SH3 file.

### Part 2 (rating after)
- The value step consumes `RatingEstimate.sap` (Distribution). ΔSAP = estimate − recorded SAP; the anchoring rule must match `sap_estimator.json` (and see the SAP 10.2 flag in score D-07).
- Worked examples use **illustrative** ratings until the estimator lands (`scripts/value_showcase.py`).

### LLM / explainer (P4)
- Always quote `ValueImpact.wording` with any value figure. If the 90% interval of `uplift_pct` includes 0 (`tail_probabilities` "no clear increase"), say there is no clear effect (SH2 example).
- `rating_effect_pct_per_sap_point` is a **mixture of three designs** (V-13). Describe it as "historical association", never "each SAP point adds X% to value".
- Green twins: always name the geography from `GreenTwin.same`, and never compare £/m² across geographies (V-10 caveat).

### UI (P5)
- Value range card: median + 90% range in £ and %, the wording, and "estimated rating, not an EPC" for band_after.
- Green twins card: tier, score gap, £/m² range, geography, number of sales; empty state "Not enough comparable greener sales nearby".

### Data (P1)
- New sources: UCL linked PPD+EPC (ReShare 857911, CC BY 4.0, attribution in `scripts/ucl_sales_extract.py`), NSPL Nov 2024 (in the UCL bundle), UK HPI July 2026 (OGL). Add to `docs/DATA.md` with licences.
- Pipeline (all streamed, nothing large stored):
  ```bash
  uv run python scripts/ucl_sales_extract.py --bundle bundle_2 --member "857911_bundle_2/tranall_link_26122024.zip::tranall_link_26122024.csv" --sales-cols --key address --pct 10 --out data/interim/sales
  uv run python scripts/ucl_sales_extract.py --bundle bundle_2 --member "857911_bundle_2/NSPL21_NOV_2024_UK.zip::NSPL21_NOV_2024_UK.csv" --usecols pcds laua rgn ctry lsoa21 msoa21 lat long imd ru11ind doterm --key pcds --out data/interim/geo
  curl -L -o data/raw/hpi/UK-HPI-full-file-2026-07.csv https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-2026-07.csv
  uv run python scripts/train_value_model.py --version v1        # fit (~10 min CPU)
  uv run python scripts/eval_value_model.py --version v1         # evaluation, confounding checks, rating_effects.json
  uv run python scripts/build_green_twins.py --version v1
  uv run python scripts/value_showcase.py
  ```

## Artifacts committed
`artifacts/value_model/v1/`: `posterior.npz` (small parameters as samples; location effects as mean/std), `encoding.json`, `rating_effects.json` (V-13 mixture), `metrics.json` (evaluation), `green_twins_showcase.json`, `config.json`.

## Proposed AGENTS.md edits (team lead)
- Decisions → Value: primary model is the Bayesian hierarchical hedonic; the customer range uses a **triangulated rating effect** (OLS + within-neighbourhood + Bayesian, V-13), because the designs disagree 0.08–0.25%/SAP point while published premiums are lower.
- Verified: UCL 857911 structure, licence and Range support; UK HPI full-file URL and columns.

## Known limitations to state in the evaluation report
- Rating effect is an association; all designs have known biases (V-11). Repeat sales overstate it (renovations).
- Mean-field SVI under-covers slightly (90% → 0.88 / 0.87).
- 10% property sample; the full dataset would tighten region × type slopes and find more green twins.
