# AGENTS.md — GreenTwin team knowledge

Read this before working in this repo, whether you are a human or an AI agent. It holds the decisions, rules and conventions everyone must follow. When a decision changes, update this file in the same PR.

## What we are building
**GreenTwin** is the Megathon X (Singularity) entry for **Lloyds Problem Statement 2: Green Home Score and Retrofit Recommendation Engine** (`LLOYDS-ps2.pdf`, repo root).

Every UK home gets a calibrated **physics digital twin**. From it we compute:
1. The **Green Home Score** (0–100, fixed weights, monotonic sub-scores).
2. Poor-home flagging and an **improvement-potential** ranking.
3. A **prioritised, feasible retrofit path**.
4. **Per-fuel energy savings** and **carbon**.
5. An indicative **property-value range**.

Every uncertain number carries a **full Bayesian distribution** (mean, median, mode, std, percentiles, 90% interval, tail probabilities). A **tool-only LLM** explains results and answers what-ifs by rerunning the models. **Jev**, a System-1 model, routes questions and screens feasibility. A **Three.js 3D twin** shows retrofits inside and outside the house. A **Lloyds dashboard** (locality heat map, segments, Green Profits) turns this into green-lending demand.

- Plan: `docs/IMPLEMENTATION_PLAN.md` (tiers) and `docs/IMPLEMENTATION_PLAN_v2.md` (build blocks)
- **System design: `SYSTEM_DESIGN.pdf`** (repo root; source `deliverables/src/build_system_design.py`): the five parts, architecture, data, build plan.
- **Low-level design: `LLD.pdf`** (repo root; source `deliverables/src/build_lld.py` + `lld_content.md` + `lld_data.md`): every contract field, interface, algorithm, table schema and dataset download step. Contract and interface sections are generated from the code.
- Dataset list (verified 11 Oct 2026): `deliverables/src/datasets.py`, shared by both PDFs.
- Submission PDF: `deliverables/M1.pdf` (source `deliverables/src/build_m1.py`)
- Living strategy doc: https://claude.ai/code/artifact/5db02f30-587d-4e5a-9bec-16e560798f48
- Git remote: https://github.com/IIIT-ECell/megathon26-megamind.git

## Core philosophies
1. **Physics is the source of truth.** Every number (energy, savings, carbon, rating) comes from the Level-1 digital twin of each home. ML sets priors, ranks, estimates value and checks results; it never produces a customer's savings directly.
2. **Bayesian in thought.** Each home is calibrated against its own meter readings. Every output is a full distribution (mean, median, mode, std, percentiles, 90% interval, tail probabilities) plus P(Poor), P(suitable), P(on front). Missing data never becomes zero; it widens the range.
3. **Rerun, never add up.** A retrofit is a change to the twin's parameters; every what-if reruns the model. Interactions come from the physics. Savings are always per fuel.
4. **Grounded explanation.** The LLM only quotes tool outputs; a validator rejects any other number; Jev only routes questions and screens feasibility. No backing → "unavailable", never a guess.
5. **Explainability first.** Score waterfalls, heat loss per building element, EBM contributions, value coefficients, and a reason on every recommendation or veto.
6. **See it to believe it.** A living 3D twin: heat loss glows, retrofits appear as you add them, numbers update live, all driven by the same calibrated model.

*In one line:* a Bayesian-calibrated physics twin of every home, which you can see in 3D, explain in plain language and act on, where every number is backed by physics and the range it comes with.

## Coding guidelines (hackathon: speed first)
This is a hackathon. **Speed is the goal, not bug-free code.** A working, good-looking demo beats a perfect module nobody sees.

**Move fast**
- Build the thinnest end-to-end path first (SH1–SH3 through twin → score → retrofits → API → UI), then deepen. Never polish a block in isolation.
- Work against `contracts/examples/` and mocks from day one; never wait for another block.
- Reuse libraries and existing code; copy-paste is fine; no premature abstractions or frameworks.
- Hardcode and precompute where it helps the demo (showcase homes, cached packages). Cache every slow or paid call.
- Timebox: stuck for ~45 minutes → cut scope, use the fallback, or ask. `# TODO` / `# HACK` comments are fine.
- Commit small and often; merge to `main` quickly; don't break the demo path.
- Use AI coding agents freely, but read the numbers they produce.

**Tests: only where judges look.** Don't chase coverage. Keep these green, and add little else:
- Contracts validate; monotonic sub-scores; per-fuel signs (electricity can rise); improvement potential `null` when the baseline is 0.
- The LLM numeric validator; no `mock` result reaches the demo.

**Non-negotiables, even when rushing** (each one costs marks if broken):
- Inputs and outputs use `contracts/`; every result has `provenance`.
- Uncertain numbers are `Distribution`s built with `contracts.summarise`.
- Missing ≠ 0; savings per fuel; packages are rerun, never summed.
- The LLM never produces a number; value is always "indicative"; no keys in git.

**UI: one look everywhere**
- One design system in `web/src/theme` (colour, type, spacing tokens) and shared components: `NumberWithRange`, `DistributionCard`, `ScoreWaterfall`, `ProvenanceTag`, `StatusBadge`. Never restyle per page.
- Brand palette, matching the PDFs: green `#0b6e4f` primary, blue `#2a6fdb` accent, neutral greys. Colour-blind-safe scales (viridis/cividis) for heat loss and maps. Red only for increases and negative savings.
- One font family (Inter), an 8 px spacing grid, card layout, the same chart style (Recharts) on every screen. The 3D twin uses the same palette.
- Every number shows unit + headline + 90% range; tapping it opens the `DistributionCard` (mean, median, mode, std, percentiles, tails). Labels: "estimate", "projected", "indicative", "estimated from home type".
- Polish the demo journey first (score → ladder → 3D → loan → chat) and make it look great; simple loading, empty and error states; works offline for showcase homes.

## Decisions (do not relitigate without the team lead)
| Decision | Detail |
| --- | --- |
| Five parts | **1 Digital twin** (JAX physics, Bayesian calibration, 3D, photo capture) · **2 Energy & retrofit engine** (EBM prior, baseline, ladder, savings/carbon/bills, improvement potential, rating after, Pareto) · **3 Green Home Score** · **4 Property value** · **5 Feasibility (rules → Jev), LLM explanation and chat**; plus Part 0 data layer and the orchestrator. Parts 1+2 are the coupled core; 0, 3, 4, 5 and the UI are independent and code against `greentwin/interfaces.py`. |
| EPC live tables | MHCLG live tables (D1/D3/EB4, LA × quarter) are aggregates: used for population calibration targets, download/mapper reconciliation, twin standard-mode validation, score peer context and dashboard layers. Never as per-home training data. |
| Region | **UK only** (England & Wales data). The India pack was dropped. |
| Core engine | **Level 1 RC + equipment simulator**, Bayesian-calibrated, is the **only source of numbers**. Level 2 (spatial PINO, 3D heat-flow detail) is **visual only** and is gated to agree within ±5% with Level 1. |
| Learned models | **EBM primary**, XGBoost baseline. Learned models calibrate, rank, value, or act as surrogates; they never directly produce customer savings. |
| LLM | Tool-only. Never computes or estimates numbers. Every what-if → `run_scenario`. A numeric validator checks every answer. **Provider (P4, 11 Oct 2026, approved by the team lead): OpenAI Responses API, `gpt-6-luna` for every call** (`LLM_MODEL`, key `OPENAI_API_KEY`); knowledge base in DuckDB (BM25 full-text); see `docs/LLM.md`. |
| Jev | TypeSafe "System 1" model (not JEPA). Used **only** for what-if routing, feasibility layer 2 and the explanation check. **Never** for kWh/£/CO2/score. Rule vetoes always win. Falls back to OpenAI structured output (`gpt-6-luna`) or rules. |
| Value | Always a **range**, labelled **indicative** (association, not proven causation). Primary: **Bayesian hierarchical hedonic** model on real UK sales; baseline OLS without the rating; repeat-sales and matching only as stretch checks. |
| Plan | Phases and checkpoints, not clock time. Build order is Tier A → B → C → D. |
| Physics library | **JAX** (`jax.numpy`). The twin runs inside the NumPyro model (NUTS/SVI need its gradients) and is `vmap`ped over posterior samples. Option B's ODE twin uses `diffrax`. Not numpy, not PyTorch. |
| Electricity depth | Half-hourly electricity layer from London smart-meter load shapes; battery, LED and time-of-use retrofits; GraphSAGE GNN on DESNZ LSOA consumption for area priors and the dashboard. |
| Photo-to-twin | The team's LLM provider (`gpt-6-luna`, `greentwin.llm.client`, per the LLM decision) with structured outputs. Two readers on that one client: floor plan + fuse box → layout and circuits (`greentwin/twin/vision.py`, P1); room photos → element tags with confidence (`greentwin/twin/capture.py`, P4). The person confirms every draft; v1 fallback is manual tags. Vision never produces energy numbers. |
| Hardware | GPU available, used for Tier-D models (twin v2, DeepONet, home-level GraphSAGE, PINO). |
| Showcase homes | SH1 (Victorian terrace, conservation area), SH2 (1970s cavity semi), SH3 (all-electric 3rd-floor flat). See `contracts/examples/`. |

## Core rules (from the problem statement; judges test these)
1. Score, savings, carbon and value come **only** from documented models and calculations. The LLM explains them.
2. Savings = baseline − after, **per fuel** (electricity and gas separately). Increases are negative savings.
3. Purchased energy = grid import + gas. Solar cuts imports, not demand. Improvement potential = 100 × (E_base − E_after)/E_base; if the baseline is 0 or missing → `null` + reason.
4. Package savings must account for **interactions**: rerun the twin on the package, never sum individual savings.
5. Carbon = Σ_fuel ΔE × EF, with documented factors (units + source).
6. A large home is not inefficient just because it uses more: use **kWh/m2** and peer context.
7. Missing months/fields are **never 0 by default**. Impute explicitly, flag them, and widen intervals.
8. **No real personal or customer data.** Use public data (with licences) plus synthetic data. Show location only as the postcode district or LSOA.
9. Splits are by **property / customer**, assigned once (`Property.split`). No fields that aren't available at prediction time (`epc.potential_*` = leakage).
10. Better state → never a lower sub-score (monotonicity tests are mandatory).

## Compliance fixes (from the constraint review, 10 Oct 2026) — binding
1. **Post-retrofit rating.** The projected score's energy-efficiency component and the value model's "band after" come **only** from `contracts/sap_estimator.json`: SAP 2012 ECF formula, anchored to the recorded SAP, run under standard conditions. The output is `RatingEstimate`, always labelled "estimated rating, not an EPC".
2. **Sourced inputs only.**
   - Carbon uses `contracts/emission_factors.json`, verified against the DESNZ 2025 file:
     - Electricity 0.19553 kgCO2e/kWh (generation 0.177 + T&D 0.01853).
     - Gas 0.18296 kgCO2e/kWh (gross CV).
   - Bills use `contracts/tariffs.json` (Ofgem Q4 2026).
   - Costs use `retrofit_catalogue.json` (BEIS 2017 Table 3 × ONS CPI 1.3888).
   - Any item with `cost_status: unsourced` returns `cost = null` plus a reason, and is excluded from cost ranking.
   - A fuel with no sourced factor → carbon `null` plus a reason.
3. **Energy intensity = the brief's definition:** annualised **metered** grid import + gas, divided by floor area. It is not weather-normalised; the weather-normalised figure is reported as a sensitivity only. The PV overlap with renewables is justified in `score_mapping.json`, and its size is reported (score with and without PV added back).
4. **Heating mapping** covers every `HeatingSystem` value (test-enforced), including electric storage, direct electric, district and communal heating.
5. **Green Profits is a scenario, not a trained model.** Take-up follows the documented rule in `contracts/green_profits_assumptions.json`; every coefficient is a labelled assumption on a slider. No model is trained on synthetic labels.
6. **Value wording.** Every display of the value range uses `ValueImpact.wording`: "Indicative range based on historical associations in sales data; not a promised or proven increase." Never present it as a reason to borrow.
7. **Synthetic green premium.** Recovering the premium we inserted into synthetic values is reported **only** as a method check, never as evidence of real uplift.
8. **3D element colours.** `ElementHeatLoss.estimated = true`. Show the `basis` label: "Estimated from home type and EPC descriptions; only the total heat loss is calibrated to bills".

## Shared data formats
- **Source of truth:** `contracts/models.py` (Pydantic v2, schema 0.4.0, grouped by part). `contracts/types.ts` is **generated** (`python -m contracts.export_ts`); never edit it by hand (a test fails if it is stale).
- **Interfaces:** `greentwin/interfaces.py` has one Protocol per part (DataLayer, DigitalTwin, EnergyRetrofitEngine, GreenHomeScorer, ValueModel, FeasibilityEngine, Router, Explainer, Orchestrator). Call other parts only through these.
- Configs:
  - `contracts/score_mapping.json`: weights + sub-score mappings.
  - `contracts/feasibility_rules.json`: rules + ordering.
  - `contracts/retrofit_catalogue.json`: twin edits, grants, sourced costs with `cost_status`.
  - `contracts/emission_factors.json` (DESNZ 2025) and `contracts/tariffs.json` (Ofgem Q4 2026).
  - `contracts/sap_estimator.json`: the post-retrofit rating method.
  - `contracts/green_profits_assumptions.json`: the scenario rule, assumptions only.
- Examples: `contracts/examples/` (showcase homes, usage, scenario spec, and **MOCK** ScenarioResult, TwinScene and ChatResponse, shape only).
- JSON Schema: `python -m contracts.export_schemas` → `contracts/schema/`.
- Units:
  - kWh/yr, GBP, kgCO2e, m2, W/m2K, W/K.
  - Uncertain values are a `Distribution` (full posterior summary: mean, median, mode, std, P1–P99, 50%/90% credible intervals, tail probabilities, worst-5% mean, skewness, histogram). Build it only with `contracts.summarise.summarise(samples, unit)`; known-exact values use `exact()`, categories use `category_probabilities()`.
  - Every result has `provenance.engine`.

## Repo map
`contracts/` · `greentwin/{data,score,twin,bayes,retrofit,models,value,geo,optimise,kb,llm,jev,api}` · `scripts/` · `data/{raw,interim,processed,cache}` (git-ignored) · `artifacts/` · `cache/{llm,jev}` · `eval/{scenarios,reports}` · `web/` · `docs/` · `deliverables/` · `tests/`

## Setup
```bash
# Python: use 3.12 via uv. The machine default is 3.14, which lacks some wheels (torch/jax/interpret).
uv python install 3.12
uv sync                         # add --extra stretch for GPU/stretch libs
uv run pytest -q                # contract + monotonicity tests must pass
uv run python scripts/make_showcase_examples.py
cp .env.example .env            # fill only what you have; see Keys
```
- GPU: torch uses `GREENTWIN_DEVICE=cuda`.
- JAX GPU is Linux-only. Run GPU NumPyro in WSL2; CPU is fine for per-home SVI.

## Keys
- `.env.example` lists every key: `OPENAI_API_KEY` (LLM and photo-to-twin), `JEV_API_KEY`, `EPC_API_TOKEN`, CEDA, Met Office. `__ASK_USER__` means not provided yet.
- **Agents/teammates: ask the team lead when a task first needs a key.** Never invent, commit or log keys.
- Code must work without keys:
  - The template explainer replaces the LLM.
  - OpenAI structured output or rules replace Jev.
  - Cached samples replace the EPC API.

## Conventions
- Branches: `p<N>/<short-task>`; PRs into `main`; a contract change needs review from its consumer.
- Seeds: 42 by default; record `seed` + `data_version` in every `metrics.json`.
- Artefacts: `artifacts/<model_id>/<version>/{model.*, metrics.json, config.json}`.
- Cache every LLM and Jev call (`cache/`) so evaluations replay without keys.
- Code comments/docs in English; numbers in the UI always show the unit + headline + 90% range (full `Distribution` on tap).
- Never let a `provenance.engine == "mock"` result reach the demo or evaluation.

## Verified (10 Oct 2026)
- **DESNZ 2025 factors** (flat file), with row IDs in `emission_factors.json`:
  - Electricity generation 0.177, T&D 0.01853.
  - Natural gas 0.18296 (gross CV) / 0.2027 (net CV).
- **Ofgem 1 Oct – 31 Dec 2026:**
  - Electricity 26.32 p/kWh and 54.83 p/day. Ofgem states it excludes VAT for Oct 2026 – Mar 2027.
  - Gas 7.97 p/kWh and 29.68 p/day, including 5% VAT.
- **BEIS 2017 Table 3 costs + ONS CPI** (D7BT: 2017 = 103.4, Aug 2026 = 143.6) for draught proofing, loft, cavity, internal/external wall, underfloor, double glazing and heating controls.

## Still to verify (do not present as confirmed)
- Air-source heat pump cost: £12,500 for 8 kW, Apr–Jun 2025, as reported by Which? citing DESNZ BUS statistics. Status `secondary_source`.
- PV cost: £1,876/kW, ≤ 4 kW, 2024/25, as reported by pv magazine citing DESNZ. Status `secondary_source`.
- Still unsourced: triple glazing, secondary glazing, HHR storage heaters, battery costs.
- BUS £9,000 off-gas dates; ECO4/GBIS status (excluded from net cost until confirmed).
- The SAP 2012 ECF formula and deflator against the published BRE specification.
- UCL linked dataset variable names (854942 / 857911).

## Owners
- P1 (A): Data & Geography
- P2 (B): Physics & Bayesian
- P3 (C): ML, Value & Causal
- P4 (D): LLM / KB / Jev / API
- P5 (E): Front-end & 3D

See `docs/IMPLEMENTATION_PLAN.md` §1 and §6.
