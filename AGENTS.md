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

Every uncertain number carries a **Bayesian 90% credible interval**. A **tool-only LLM** explains results and answers what-ifs by rerunning the models. **Jev**, a System-1 model, routes questions and screens feasibility. A **Three.js 3D twin** shows retrofits inside and outside the house. A **Lloyds dashboard** (locality heat map, segments, Green Profits) turns this into green-lending demand.

- Plan: `docs/IMPLEMENTATION_PLAN.md`
- Submission PDF: `deliverables/M1.pdf` (source `deliverables/src/build_m1.py`)
- Living strategy doc: https://claude.ai/code/artifact/5db02f30-587d-4e5a-9bec-16e560798f48
- Git remote: https://github.com/IIIT-ECell/megathon26-megamind.git

## Decisions (do not relitigate without the team lead)
| Decision | Detail |
| --- | --- |
| Region | **UK only** (England & Wales data). The India pack was dropped. |
| Core engine | **Level 1 RC + equipment simulator**, Bayesian-calibrated, is the **only source of numbers**. Level 2 (spatial PINO, 3D heat-flow detail) is **visual only** and is gated to agree within ±5% with Level 1. |
| Learned models | **EBM primary**, XGBoost baseline. Learned models calibrate, rank, value, or act as surrogates; they never directly produce customer savings. |
| LLM | Tool-only. Never computes or estimates numbers. Every what-if → `run_scenario`. A numeric validator checks every answer. |
| Jev | TypeSafe "System 1" model (not JEPA). Used **only** for what-if routing, feasibility layer 2 and the explanation check. **Never** for kWh/£/CO2/score. Rule vetoes always win. Falls back to Claude structured output or rules. |
| Value | Always a **range**, labelled **indicative** (association, not proven causation). Methods: hedonic quantile → green-twin matching → DR-learner. |
| Plan | Phases and checkpoints, not clock time. Build order is Tier A → B → C → D. |
| Hardware | GPU available, used for Tier-D models (twin v2, DeepONet, GraphSAGE, PINO, vision). |
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

## Shared data formats
- **Source of truth:** `contracts/models.py` (Pydantic v2). TS mirror: `contracts/types.ts`. Change both in the same commit.
- Configs:
  - `contracts/score_mapping.json`: weights + sub-score mappings.
  - `contracts/feasibility_rules.json`: rules + ordering.
  - `contracts/retrofit_catalogue.json`: twin edits, grants; costs are TO_SOURCE.
- Examples: `contracts/examples/` (showcase homes, usage, scenario spec, a **MOCK** result that is shape only).
- JSON Schema: `python -m contracts.export_schemas` → `contracts/schema/`.
- Units:
  - kWh/yr, GBP, kgCO2e, m2, W/m2K, W/K.
  - Uncertain values are `Interval {mean, lo, hi, level, unit}`.
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
- `.env.example` lists every key: `ANTHROPIC_API_KEY`, `JEV_API_KEY`, `EPC_API_TOKEN`, CEDA, Met Office. `__ASK_USER__` means not provided yet.
- **Agents/teammates: ask the team lead when a task first needs a key.** Never invent, commit or log keys.
- Code must work without keys:
  - The template explainer replaces the LLM.
  - Claude or rules replace Jev.
  - Cached samples replace the EPC API.

## Conventions
- Branches: `p<N>/<short-task>`; PRs into `main`; a contract change needs review from its consumer.
- Seeds: 42 by default; record `seed` + `data_version` in every `metrics.json`.
- Artefacts: `artifacts/<model_id>/<version>/{model.*, metrics.json, config.json}`.
- Cache every LLM and Jev call (`cache/`) so evaluations replay without keys.
- Code comments/docs in English; numbers in the UI always show the unit + interval.
- Never let a `provenance.engine == "mock"` result reach the demo or evaluation.

## Facts still to verify (mark as TO_VERIFY until checked against official files)
- DESNZ 2025 electricity factor (~0.177 kgCO2e/kWh) and the gas gross vs net CV factor (~0.183 vs 0.2049). Meter kWh are gross CV.
- Ofgem Oct–Dec 2026 unit rates (26.32p elec / 7.97p gas) and their VAT basis.
- BUS £9,000 off-gas uplift dates; ECO4/GBIS status.
- Retrofit costs: BEIS 2017 + CPI (catalogue `source: TO_SOURCE`).
- UCL linked dataset variable names (854942 / 857911).

## Owners
- P1 (A): Data & Geography
- P2 (B): Physics & Bayesian
- P3 (C): ML, Value & Causal
- P4 (D): LLM / KB / Jev / API
- P5 (E): Front-end & 3D

See `docs/IMPLEMENTATION_PLAN.md` §1 and §6.
