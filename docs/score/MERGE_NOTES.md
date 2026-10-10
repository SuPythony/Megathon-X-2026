# Merge notes: Part 3 Green Home Score → `main`

Branch: `p3/green-home-score` (12 commits, 11 Oct 2026). This page lists everything a teammate must know or update when this branch is merged. Detail and evidence for every item: [DECISIONS.md](DECISIONS.md) (decision ids D-xx). Index: [README.md](README.md).

**Merge risk:** low. Checked against `origin/main` (unchanged since branch point `a1ba6e4`) and `origin/satyarth` (twin work): **no file overlaps**. All 57 tests pass (`uv run pytest -q`).

---

## 1. Contract changes (review needed by consumers)

| Change | File | Who must act | What to do |
| --- | --- | --- | --- |
| **New model `AlternativeScore`** and field **`ScoreResult.alternatives: list[AlternativeScore]`** (default `[]`) | `contracts/models.py` | **UI (P5), LLM / API (P4)** | Review. The headline `score` / `band` are unchanged and always use the brief's fixed weights. `alternatives[0]` (name `learned_weights`) is a companion score with data-learned weights (D-05, D-19). UI: show it as secondary ("score with data-learned weights"), never instead of the headline. LLM: may quote it from the tool output, labelled as the learned-weights score. |
| `ScoreResult.band` **description** changed: band = classification of the **rounded** score (brief §2), not the mode of `band_probabilities` | `contracts/models.py` | UI, LLM | Behaviour change only if you computed band yourself: use `ScoreResult.band`; `band_probabilities` is the spread over samples (D-01). |
| `contracts/types.ts` and `contracts/schema/*.json` **regenerated** | generated | UI | Pull and rebuild; do not hand-edit (`python -m contracts.export_ts`). |
| `SCHEMA_VERSION` **not bumped** (still 0.4.0) | `contracts/models.py` | Team lead | Additive change (new optional field). Bump to 0.4.1 if the team wants every addition versioned. |

## 2. Config changes: `contracts/score_mapping.json` v0.1.0 → **v0.4.0**

Weights are **unchanged** (brief's 25/20/15/15/10/10/5). Sub-score mappings changed; every change is monotone and tested.

| Version | Change | Decision |
| --- | --- | --- |
| 0.2.0 | U-value curves for walls, roof, floor (used when a U-value is known, e.g. new-build EPCs "Average thermal transmittance") | D-15 |
| 0.3.0 | **Loft curve is physics-based**: share of max heat-loss cut from RdSAP 10 Table 16 (50 mm → 74, 100 mm → 87, 270 mm → 98) | D-23, D-24 |
| 0.4.0 | **Lighting scored inside `smart_energy_management`**: 0.53 × controls level + 0.47 × low-energy lighting %, share learned from EPC data | D-26 |
| — | Heating map **unchanged** (physics alternative tested and rejected; team choice A) | D-25 |

**Effect:** scores rise for homes with some loft insulation and for homes with efficient lighting. Showcase homes: SH1 30 → **35** (Poor), SH2 49 Poor → **52 Moderate**, SH3 65 → **67** (Good). Anything that hard-codes old SH scores (e.g. mocks, demo script text, UI snapshots) must be updated. `contracts/examples/SH2_scenario_result_MOCK.json` still shows 46 (it is a mock; replace when the orchestrator produces real results).

## 3. What each part should update

### Part 0: data layer (P1)
- `Property.low_energy_lighting_pct` is now **scored**. Populate it from EPC `LOW_ENERGY_LIGHTING` in the mapper; when missing the score samples it from the EPC prior (never 0) and raises `missing_data`.
- The EPC text → sub-score rules in `greentwin/score/epc_features.py` (D-16) overlap with the planned `epc_mapper.py`. Reuse or align them (they are validated: Spearman 0.64–0.90 against EPC element star ratings). Known traps found here: EPC text can contain `|`; `PHOTO_SUPPLY` is null on **all** new-build SAP certificates (unknown PV, not zero, D-22); "no room thermostat" contains "room thermostat".
- EPC data acquisition (D-14): bulk zip via the API (`scripts/download_epc.py bulk`, 8.3 GB) → `scripts/epc_bulk_extract.py --zip … --pct 25` keeps a deterministic property sample by UPRN hash. A 25% extract of all years sits in `data/interim/epc25/` (git-ignored); the zip is on the owner's `D:/greentwin_data/epc/`.
- NEED 2025 50k and 4M samples are in `data/raw/need/` (git-ignored); loader `greentwin/score/need_features.load` (chunked).

### Parts 1 + 2: twin and energy engine (P2)
- **Replace the stub.** The score currently builds its intensity input with `greentwin/score/stub_energy.py` (metered bills, degree-day fill, D-13). When `EnergyRetrofitEngine.baseline()` returns `EnergyBreakdown.intensity_kwh_m2`, pass that to `GreenHomeScorer.score()` and stop using the stub (only `scripts/score_showcase.py` and tests call it).
- **Projected score** expects `RatingEstimate.sap` (a `Distribution`) for the after-retrofit energy-efficiency component.
- **SAP formula flag (D-07):** `contracts/sap_estimator.json` uses **SAP 2012** (deflator 0.42; `117 − 121·log10`; `100 − 13.95·ECF`). Current EPCs use **SAP 10.2** (deflator 0.36; `108.8 − 120.5·log10(ECF)`; `100 − 16.21·ECF`; verified in the BRE SAP 10.2 PDF §13). Please align.
- **Verified RdSAP 10 values you can reuse:** Table 16 loft U-values (0 mm 2.3, 50 mm 0.68, 100 mm 0.40, 150 mm 0.30, 200 mm 0.21, 270 mm 0.16, 400 mm 0.11 W/m²K) are now in `score_mapping.json` → `roof_u_to_loft_mm`. `greentwin/twin/rdsap_defaults.json` (branch `satyarth`) marks its roof values `to_verify` and uses RdSAP 2012 tables.
- SAP 10.2 Table 12 fuel factors and Table 4a/4b default efficiencies are summarised in DECISIONS.md D-25 if the twin needs them.

### Part 4: value (P3 / next)
- Unaffected by the merge. Same price-drift lesson applies (D-20): deflate sale prices with UK HPI by LA × type × month.

### Part 5: LLM / API / KB (P4)
- `ScoreResult.alternatives` is a new number source: add it to the validator's allowed tool outputs and to the template explainer (optional line: "With weights learned from EPC data the score would be X").
- New `KeyIssue` evidence keys: `peer_p50_kwh_m2` (NEED peer median by type × floor-area band), `walls_sub_score` / `roof_sub_score` / `floor_sub_score` (weak elements), `basis`. Labels now name weak elements, e.g. "Poorly insulated walls".
- `ScoreComponent.inputs_used` for smart management now includes `controls_level` and `low_energy_lighting_pct` (a number, or "sampled, mean N").
- `ScoreResult.peer_context` now carries `peer_intensity_p50_kwh_m2`.
- Explanation wording for the learned score: it is "evidence-based companion, not the official score".

### UI (P5)
- `ScoreWaterfall`: smart management is now "Controls & efficient lighting".
- Show `alternatives[0]` beside the headline with its weights (tooltip) and the label from §1.
- Wider score intervals are expected for homes with unknown lighting or loft depth (sampled inputs).

### Evaluation / docs owners
- `docs/score/` is submission item **D** (scoring method): README, GREEN_SCORE_STANDARDS.md (standards + parameter catalogue), DECISIONS.md, LEARNED_WEIGHTS.md (generated).
- Evaluation-report inputs: `artifacts/score_weights/v3/metrics.json`, `artifacts/score_curves/v1/{sap,eir}/metrics.json`, `artifacts/score_lighting/v1/metrics.json`.

## 4. Dependencies and environment
- **New dependency:** `odfpy` (pandas needs it to read `.ods`: NEED metadata, EPC live tables). `pyproject.toml` and `uv.lock` updated → run `uv sync`.
- `uv.lock` is committed for the first time on this branch (was untracked). Reproducible installs: `uv sync`.
- `.env` needs `EPC_API_TOKEN` only to re-download EPC data; nothing in the score needs a key at runtime.

## 5. New files (all under Part 3 ownership)
`greentwin/score/`: `score.py`, `mapping.py`, `sampling.py`, `priors.json`, `issues.py`, `stub_energy.py`, `epc_features.py`, `need_features.py`, `archetypes.py`, `learn.py`.
`scripts/`: `download_epc.py`, `epc_bulk_extract.py`, `train_score_weights.py`, `report_score_weights.py`, `fit_score_priors.py`, `fit_score_curves_ebm.py`, `fit_lighting_split.py`, `score_showcase.py`.
`tests/`: `test_score.py`, `test_score_epc_features.py`.
`artifacts/`: `score_weights/v1..v3` (weights, metrics, config, cells), `score_curves/v1`, `score_lighting/v1` (small JSON/CSV only).

## 6. Proposed AGENTS.md edits (team lead to approve; AGENTS.md asks for decision changes in the same PR)
- **Decisions table**, new row: "Learned-weights companion score | Headline score keeps the brief's fixed weights. A companion score with weights learned from EPC + NEED data is reported in `ScoreResult.alternatives`; never replaces the headline (docs/score/DECISIONS.md D-05, D-19)."
- **Shared data formats → Configs:** `score_mapping.json` is v0.4.0 (U-value curves, physics loft curve, lighting in smart management).
- **Still to verify:** add "EPC methodology era boundaries (SAP 2009 / 2012 / RdSAP 10 by lodgement date, D-20)" and "Part L Table 4.3 limiting U-values".
- **Verified:** add "SAP 10.2 rating formulas (§13–14) and RdSAP 10 Table 16 loft U-values (checked 11 Oct 2026)".

## 7. Merge checklist
1. Consumers (P4 LLM/API, P5 UI) review §1 `AlternativeScore` / `alternatives`.
2. `uv sync` (odfpy) → `uv run pytest -q` → 57 passed.
3. `uv run python -m contracts.export_ts` → no diff (types.ts is current).
4. `uv run python scripts/score_showcase.py` → SH1 35, SH2 52, SH3 67 (headline).
5. Team lead applies or rejects §6 AGENTS.md edits.
6. P2 picks up the SAP 2012 → 10.2 flag (D-07) and the stub replacement.
