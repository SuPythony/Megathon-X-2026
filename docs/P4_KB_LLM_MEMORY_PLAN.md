# Knowledge base, grounded LLM explanations and customer memory — build plan

Part 5 (P4) · Branch `p4/kb-llm` · Extends `IMPLEMENTATION_PLAN_v2.md` Block 6 and `LLD.pdf` §4.5. Where this plan
adds something those documents do not cover, it says so; it never relaxes a rule in `AGENTS.md`.

## 1. What we are building

One explanation layer that answers a customer or agent about **their** home, using only numbers our models
produced, with sources, and that remembers the customer across sessions.

```
 user message ─► Router (Jev | Claude structured | rules) ─► intent + ScenarioSpec slots
                                                │
                     ┌──────────────────────────┴───────────────────────────┐
                     ▼                                                      ▼
            Claude tool loop (strict tools)                         Template explainer
      get_home_facts · get_energy_prior · run_scenario               (no key / validator
      get_results · rank_recommendations · search_docs               fails twice)
      query_graph · get_customer_memory · propose_memory
                     │
                     ▼
   Explanation plan (claims → evidence refs) ─► Validator (numbers + evidence + wording) ─► ChatResponse
                     │                                                                     │
                     └──────────── chat log, tool log, memory events ──► DuckDB knowledge base ◄┘
```

Five properties, each checkable:
1. **Grounded.** Every number in an answer traces to a tool result from the same turn. The LLM never computes.
2. **Cited.** Every non-numeric claim about methods, grants or rules cites a document chunk or rule id.
3. **Personal, but honest.** Memory changes the *order and framing* of options, never a kWh, £, CO2 or score
   value, and never hides a better option. Every reorder states its reason.
4. **Works offline.** No API key → the router falls back to rules and the explainer to templates. Every
   LLM/Jev call is cached so evaluations replay without keys.
5. **Private by design.** No real personal data (AGENTS.md rule 8). Customers are synthetic or pseudonymous;
   location is postcode district or LSOA only; free text is screened before it is stored.

## 2. What exists today (inputs this plan builds on)

| Exists | Where | Used for |
|---|---|---|
| Contracts: `ChatRequest/Response`, `ToolCallRecord`, `Citation`, `ExplanationCheck`, `WhatIfRoute`, `Feasibility(Report)`, `CustomerMemory` | `contracts/models.py` Part 5 | Chat API shape (memory contract is extended in §6) |
| Protocols `Router`, `Explainer`, `FeasibilityEngine`, `Orchestrator` | `greentwin/interfaces.py` | The seams this plan implements |
| Configs: retrofit catalogue (16 items, sourced costs), feasibility rules (17), emission factors, tariffs, score mapping, SAP estimator, Green Profits assumptions | `contracts/*.json` | Facts, KB documents and graph edges |
| Showcase homes SH1–SH3: property, 2025 usage, mock scenario/chat/twin | `contracts/examples/` | Seed data and evaluation fixtures |
| **M04 energy prior (real, served):** `energy_prior()`, `update_with_meters()`, HTTP router, model card, report | `greentwin/models/`, `greentwin/api/energy_prior.py`, `docs/M04_ENERGY_PRIOR.md` | First real tool (`get_energy_prior`); first method document |
| Plans, LLD, system design, AGENTS.md rules | `docs/`, repo root PDFs | Method documents and the system prompt's rules |
| `run_scenario` | **mock only** (`SH2_scenario_result_MOCK.json`) | Tool stays behind the `Orchestrator` protocol; mock results are flagged and blocked from the demo |

## 3. Storage: one structured knowledge base

Single embedded **DuckDB** file `data/greentwin.duckdb` (git-ignored), built reproducibly by
`python -m greentwin.kb.build`. DuckDB gives SQL, full-text search (`fts` extension, BM25) and vector search
(`vss` extension, HNSW) in one file, so there is one store to back up, version and test. Large arrays
(posterior samples, per-home explanations) stay in Parquet and are registered as DuckDB views.

### 3.1 Conventions on every table
- `*_id` primary keys; `created_at` (UTC); `source` (`dataset|model|config|user|agent|llm_extracted|system`);
  `provenance` JSON (`engine`, `model_version`, `data_version`, `seed`) for anything a model produced.
- **Append-only** for events (memory events, chat turns, tool calls). Current state comes from views.
- `content_hash` on documents, configs and scenario specs, so rebuilds are idempotent and caches key on it.
- No column ever holds a full postcode, name, email, phone or free-form address.

### 3.2 Tables

| Group | Table | Key columns | Filled by |
|---|---|---|---|
| **Homes** | `properties` | property_id, source, split, location (district/LSOA/region), fabric JSON, heating JSON, epc JSON | data layer / capture |
| | `usage_monthly` | property_id, month, electricity_import_kwh, gas_kwh (NULL = missing), estimated_read | data layer |
| **Model outputs** | `energy_priors` | property_id, model_version, electricity/gas `Distribution` JSON, contributions JSON | M04 (`energy_prior`) |
| | `posteriors` | property_id, model, samples_ref (Parquet), calibration summary | M04 update now, M09 later |
| | `scenario_results` | scenario_id, spec_hash, property_id, result JSON, key metrics (flattened), engine, is_mock | orchestrator |
| | `feasibility` | property_id, retrofit_id, status, decided_by, reason_codes, p_feasible, rules_version | feasibility engine |
| **Registries** | `configs` | config_id (`retrofit_catalogue`…), version, content_hash, JSON | `contracts/*.json` |
| | `datasets` | dataset_id, publisher, licence, url, rows, sha256, use | `deliverables/src/datasets.py`, M04 proof |
| | `model_cards` | model_id, version, metrics JSON, card_path | `artifacts/*/metrics.json` |
| **Documents** | `documents` | doc_id, title, kind (`method|rule|retrofit|grant|dataset|policy|faq`), applies_to JSON, licence, source_url, version, content_hash | KB ingest |
| | `chunks` | chunk_id, doc_id, ord, heading_path, text, n_tokens, embedding FLOAT[384] | ingest + embedder |
| **Graph** | `nodes` | node_id, type (`Home|Element|Retrofit|Rule|Grant|LSOA|Region|Customer|Goal|Doc`), props JSON | build |
| | `edges` | src, rel (`HAS_ELEMENT|IN_LSOA|IN_REGION|REQUIRES|EXCLUDES|BLOCKED_BY|GRANT_APPLIES|OWNS|PREFERS|REJECTED|COMPLETED|CITES`), dst, props JSON | build + memory |
| **Customers & memory** (§6) | `customers` | customer_id (pseudonymous), created_at, consent JSON | onboarding |
| | `customer_homes` | customer_id, property_id, role (`owner|tenant|agent_view`) | onboarding |
| | `memory_items` | item_id, customer_id, kind, key, value JSON, status (`proposed|confirmed|retracted|expired`), confidence, source, evidence_ref, valid_from, expires_at | memory service |
| | `memory_events` | event_id, customer_id, property_id, retrofit_id, type (`viewed|saved|accepted|rejected|dismissed|completed|asked_about`), reason_code, weight, session_id | UI + chat |
| | `memory_scores` | customer_id, property_id, retrofit_id, alpha, beta, components JSON, computed_at, version | scorer (derived, recomputable) |
| **Conversation** | `chat_sessions` | session_id, customer_id, property_id, channel, started_at | chat API |
| | `chat_turns` | turn_id, session_id, role, text, route JSON, check JSON, fallback, model, latency_ms | chat API |
| | `tool_calls` | call_id, turn_id, name, input JSON, output_ref, output_hash, ms, error | tool loop |
| **Evaluation** | `eval_runs`, `eval_results` | run_id, scenario_id, metrics JSON, model, prompt_version | eval harness |

Views: `v_home_facts(property_id)` (property + latest prior/score/feasibility), `v_memory_current(customer_id)`
(latest confirmed item per key), `v_recent_events(customer_id, days)`.

### 3.3 Build and refresh
`greentwin/kb/build.py` is idempotent: load configs and examples → render documents (§4.1) → chunk → embed →
build FTS and HNSW indexes → build graph → load model outputs (M04 priors for every known home). A `kb_version`
row records the content hashes of every input; the health check fails if any input changed without a rebuild.

## 4. Retrieval

Three kinds of retrieval, each behind a tool, each returning the evidence the validator needs.

### 4.1 Document corpus (what `search_docs` can find)
| Kind | Content | Source |
|---|---|---|
| `method` | How each model works and its limits (M04 card, LLD sections, score mapping, SAP estimator notes) | our docs, rendered to Markdown |
| `retrofit` | One document per catalogue item: what it is, twin edits, cost basis and status, lifetime, grants, depends_on/excludes | `retrofit_catalogue.json` → template |
| `rule` | One per feasibility rule: condition, blocked retrofits, plain-English reason | `feasibility_rules.json` → template |
| `grant` / `policy` | BUS, ECO4/GBIS, VAT relief; each with status and "verify" flag where `AGENTS.md` lists it as unconfirmed | gov.uk pages (OGL), stored with URL and retrieval date |
| `dataset` | What each dataset is, licence, limits | `datasets.py` |
| `faq` | Wording rules: "estimate", "indicative", value disclaimer, heat-loss basis label | `AGENTS.md` compliance fixes |

Chunking: split on Markdown headings, ~350–500 tokens, 15% overlap, keep `heading_path` for context.

### 4.2 Hybrid search (`search_docs`)
1. **BM25** over `chunks.text` (DuckDB `fts`): exact terms such as "cavity", "BUS", rule ids.
2. **Dense** over `chunks.embedding` (DuckDB `vss`, cosine). Embeddings from a local CPU model
   (`fastembed`, `BAAI/bge-small-en-v1.5`, 384-d, ONNX, no torch). Anthropic has no embeddings endpoint, and
   local embedding keeps the build offline and free.
3. **Fusion:** reciprocal rank fusion (k = 60) of the two lists, then **metadata filters** from the request
   (`kind`, `applies_to.retrofit_id`, `applies_to.property_type`).
4. Return top-k (default 5) chunks with `doc_id`, `chunk_id`, `heading_path`, snippet and source URL. These
   become `Citation(source="doc", ref="doc_id#chunk_id")`.
Stretch: rerank the top 20 with a small cross-encoder if retrieval evaluation (§9) shows it helps.

### 4.3 Structured retrieval (the only source of home-specific numbers)
SQL views behind typed tools (`get_home_facts`, `get_energy_prior`, `get_results`). Tools return compact JSON
with units and intervals already attached (headline, median, 90% interval), so the model quotes, never derives.

### 4.4 Graph retrieval (`query_graph`)
Templated queries only (no free Cypher/SQL from the model), e.g. `blocked_retrofits(property_id)`,
`eligible_grants(property_id, retrofit_id)`, `dependencies(retrofit_id)`, `customer_constraints(customer_id)`,
`similar_homes_in_area(property_id)`. Implemented as recursive SQL over `nodes/edges` (no extra service). This
is the GraphRAG step: fetch the home's subgraph (constraints, eligible grants, blocked items), then explain.

### 4.5 Memory retrieval (`get_customer_memory`)
Always returns confirmed constraints and goals; adds the top relevant preferences and the last N events, ranked
by recency and (for free-text notes) BM25 match to the question. Proposed (unconfirmed) items are returned
separately, marked as unconfirmed.

## 5. Tools and the LLM loop

### 5.1 Tool set (all `strict: true`, `additionalProperties: false`)

| Tool | Input | Returns | Notes |
|---|---|---|---|
| `get_home_facts` | property_id | facts, data quality, missing fields | SQL view |
| `get_energy_prior` | property_id, include_update? | M04 `EnergyPrior` (+ posterior if meters) | **real today** |
| `run_scenario` | `ScenarioSpec` fields | scenario_id + key metrics with intervals | orchestrator; mock flagged |
| `get_results` | scenario_id, fields[] | requested parts of a `ScenarioResult` | avoids resending large JSON |
| `rank_recommendations` | property_id, customer_id?, goal | ranked feasible options, physics utility, memory adjustment, reasons | §6.4; computed in code |
| `search_docs` | query, kind?, retrofit_id?, k | cited chunks | §4.2 |
| `query_graph` | template_id, params | rows | templated only |
| `get_customer_memory` | customer_id, topic? | constraints, goals, preferences, recent events | §4.5 |
| `propose_memory` | customer_id, items[] (kind, key, value, evidence quote) | stored as `proposed` | never confirms itself; §6.2 |

The `ToolCallRecord.name` literal in `contracts/models.py` is extended with the four new tool names (§10).

### 5.2 Loop (manual, so every step is logged and validated)
- Model from `LLM_MODEL` (team default `claude-sonnet-5-5`; `claude-opus-5-5` for evaluation runs if budget
  allows). Adaptive thinking at `effort: "low"` for chat routes, raised only if evaluation shows a gain.
- `tool_choice: auto`. Forced tool choice is rejected on current models, so the system prompt requires
  `run_scenario` for every what-if and the validator rejects answers to what-if routes without one.
- Up to 6 tool rounds; parallel tool calls executed concurrently and returned in **one** user message; failed
  tools return `is_error: true`, never dropped.
- Check `stop_reason` before reading content (`refusal` → template answer; `max_tokens` → one retry with a
  higher cap). Server-side refusal fallback enabled (`fallbacks: "default"`).
- **Prompt caching:** tools + frozen system prompt first, cached. Per-session memory and home context are
  injected as a mid-conversation `system` message after the first user turn, so they never invalidate the
  cached prefix. No timestamps or IDs in the system prompt.
- Every request/response cached in `cache/llm/` by hash(model + prompt version + messages + tool outputs).

### 5.3 Explanations as claims, then text
The model returns an **explanation plan** via structured output (`output_config.format`):
`claims[] = {text, kind: number|fact|advice|caveat, value?, unit?, evidence: [{tool_call_id, json_path} |
{doc_id, chunk_id} | {rule_id}]}` plus `caveats[]` and `follow_up?`. Code renders the plan into the answer and
the UI can show "why?" per claim. This makes validation exact (each number has a declared source path) instead
of guessing from regex alone.

### 5.4 Validator (`llm/validator.py` → `ExplanationCheck`)
1. **Numbers:** every number in the rendered text must match its claim's evidence value (relative 1%, or
   rounding-equivalent); any number with no claim fails. Regex sweep as a second net, as in the LLD.
2. **Evidence exists:** every referenced tool call, JSON path, chunk or rule exists in this turn.
3. **Policy numbers:** a number may come from a document (e.g. a grant amount) only if cited and the doc is
   not marked "unverified"; home-specific kWh/£/CO2/score numbers must come from structured tools.
4. **Wording:** value claims use `ValueImpact.wording`; heat-loss claims carry the basis label; projected
   numbers say "estimate" or "projected"; no financial advice; loans are illustrative.
5. **Mock guard:** any evidence with `engine == "mock"` fails the answer outside eval mode.
6. Optional layer: Jev "is every claim supported by this JSON?".
On failure: regenerate once naming the failed claims; then the template explainer.

### 5.5 Template explainer (no key, or after two failures)
Fixed sentences filled from `EnergyPrior` (headline, 90% range, top three contributions), `ScenarioResult`
(per-fuel savings with intervals, bill, carbon), feasibility reasons and the memory-ranked list with reasons.

## 6. Customer memory and the memory score

### 6.1 What is remembered
| Kind | Examples | Effect |
|---|---|---|
| `constraint` | renting, listed building, no outdoor space, budget cap £X (user-stated) | **Hard:** passed to `FeasibilityEngine.assess`; rule vetoes win |
| `goal` | lower bills / lower carbon / comfort | Sets `ScenarioSpec.goal` and the utility weights |
| `preference` | low disruption, prefers staged work, dislikes heat pumps (with reason) | **Soft:** memory score |
| `event` | viewed, saved, accepted, rejected (+ reason code), completed, asked about | Updates the acceptance posterior |
| `fact` | completed loft insulation in 2025, new boiler | Changes the home record → twin rerun (new bills tighten the posterior) |
| `note` | free text from an agent visit | Searchable; never used for numbers |

Every item stores source, confidence, status, evidence quote/ref and an expiry (preferences decay; constraints
persist until retracted).

### 6.2 How memory is written
1. **Explicit:** UI actions (save, reject with reason, mark completed) write `memory_events` / `memory_items`
   directly (`source=user`).
2. **From conversation:** the model calls `propose_memory` with the exact quote as evidence; items are stored
   as `proposed`. Constraints and facts need confirmation (one tap in the UI, or the agent) before they affect
   feasibility; low-risk preferences may auto-confirm after being stated twice.
3. **Screening before storage:** drop anything that looks like personal data (names, emails, phone numbers,
   full postcodes, addresses) and anything that is a number the model produced rather than the user stated.
4. **Corrections:** "that's wrong" retracts an item (status change, never a delete), so the history explains
   past recommendations.

### 6.3 The memory score (Bayesian acceptance, per customer × retrofit)
For each feasible retrofit *r* and customer *c*, keep a Beta posterior over P(c accepts r):

- **Prior** Beta(α₀, β₀) from the population: start from an assessor-relevance prior where available (M07,
  P(EPC would recommend r | home)), else uniform Beta(1, 1); scaled to a prior strength of 2 pseudo-events.
- **Updates** from `memory_events` with weights: accepted/completed +3 to α; saved +1 to α; viewed +0.2 to α;
  rejected +3 to β; dismissed +1 to β. Events on a *similar* retrofit (same category) count at 0.3×, so
  "no heat pump" also lowers air-to-water variants.
- **Preference fit** f ∈ [−1, 1] from confirmed preferences matched to retrofit attributes in the catalogue
  (disruption, upfront cost band, staging, category).
- **Decay:** event weights halve every 180 days.
- **Memory score** m = posterior mean of P(accept) adjusted by preference fit:
  `m = clip(E[p] + 0.15·f, 0, 1)`, reported with its 90% Beta interval and the components that produced it.

### 6.4 How the score pushes recommendations, safely
`rank_recommendations` (code, not the LLM):
1. **Feasibility first:** only items the feasibility engine marks feasible (with memory constraints applied).
2. **Physics utility** U from the scenario results for the goal (e.g. P50 bill saving per £ for "bills",
   kgCO2e per £ for "carbon"), using posterior medians and intervals from `run_scenario`.
3. **Bounded nudge:** `rank_score = U × (1 + λ·(m − 0.5))` with λ = 0.3, so memory moves an option by at most
   ±15% of its utility. It can reorder near-ties, but cannot lift a clearly worse option above a clearly better
   one.
4. **Guardrails:** the best option by physics is always shown, labelled as such, even if memory ranks it
   lower; dominated options are never promoted; no option is hidden because of memory. Memory never changes
   any number in a `ScenarioResult`.
5. **Explained:** every moved item carries a reason string from its components ("moved up: you said you prefer
   low-disruption work"; "moved down: you rejected heat pumps on 3 Oct, reason: upfront cost").
6. **Stored:** `memory_scores` (α, β, components, version) and a graph edge `Customer -PREFERS/REJECTED->
   Retrofit` so `query_graph` and `get_customer_memory` can explain the ranking later.

This makes the existing plan line "Beta posteriors over acceptance re-rank only feasible options" concrete.

### 6.5 Demo behaviour (PS §10)
Session 1: no memory → physics order. Session 2: customer rejects heat pump (cost) and says they rent → heat
pump vetoed by the renting constraint where rules require ownership; low-disruption items move up with stated
reasons. Session 3: completed loft insulation and new bills uploaded → home record updated, M04/M09 posterior
narrows, ladder recomputed without loft insulation.

## 7. API surface (FastAPI, mounted into the main app)
`POST /chat` (`ChatRequest` → `ChatResponse`), `POST /route`, `GET/POST /memory/{customer_id}`,
`POST /memory/{customer_id}/confirm`, `POST /memory/{customer_id}/events`,
`GET /homes/{id}/recommendations?customer_id=&goal=`, `GET /kb/search?q=` (debug, agent view only).

## 8. Code layout
```
greentwin/kb/        build.py · schema.sql · ingest.py (render configs/docs) · chunk.py · embed.py
                     search.py (BM25 + vss + RRF) · graph.py (templated queries) · store.py (DuckDB access)
greentwin/memory/    models.py · service.py (write/confirm/retract, screening) · score.py (Beta + nudge)
                     rank.py (rank_recommendations)
greentwin/llm/       tools.py (schemas + handlers) · chat.py (loop, caching, stop reasons) · plan.py
                     (claims schema) · validator.py · template.py · prompts/system_v1.md · cache.py
greentwin/jev/       client.py · router.py (Jev → Claude structured → rules)
greentwin/api/       chat.py · memory.py (routers)
eval/scenarios/      *.json (≥30 → 100) · eval/run_llm_eval.py
tests/               test_kb_*.py · test_memory_*.py · test_validator.py · test_chat_offline.py
```

## 9. Evaluation (what proves it works)
| Area | Measure | Target |
|---|---|---|
| Groundedness | share of numbers traced to evidence | ≥ 99% (0 unvalidated numbers shipped) |
| Factual accuracy | graded on the scenario set | ≥ 90% |
| Unsupported claims | claims without evidence | < 5% before validator, 0 after |
| Tool use | what-if turns that called `run_scenario` | 100% |
| Retrieval | recall@5 and MRR on ~40 labelled question → chunk pairs | recall@5 ≥ 0.85 |
| Memory | session-1→3 script: expected reorders happen, guardrails hold (best-physics option always shown) | all pass |
| Safety | PII never stored; value wording present; no financial advice; out-of-scope declined | 100% |
| Offline | full scenario set with no keys (router rules + templates) | runs, 0 unvalidated numbers |
Scenarios include: missing months, conflicting data, infeasible requests ("heat pump in my third-floor flat"),
electricity rising after a heat pump, not enough data for value, out-of-scope, memory contradiction, and a
prompt-injection note in agent free text. Baseline: the same set without the validator.

## 10. Contract changes (need review by consumers: P5 UI, P2 orchestrator)
- `ToolCallRecord.name`: add `get_energy_prior`, `rank_recommendations`, `get_customer_memory`, `propose_memory`.
- New `MemoryItem`, `MemoryEvent`, `MemoryScore` (α, β, m as `Distribution`, components, reasons),
  `RankedRecommendation` (retrofit_id, physics utility, memory score, rank, moved_by, reasons, is_best_physics).
- `CustomerMemory` gains `items: list[MemoryItem]`; existing fields kept for compatibility.
- `ChatResponse` gains optional `claims` (the explanation plan) and `memory_proposals`.
Bump schema to 0.5.0, regenerate `contracts/types.ts` and JSON Schemas, keep contract tests green.

## 11. Build order (phases with checkpoints)
| Phase | Delivers | Checkpoint | Effort |
|---|---|---|---|
| **1 · KB store** | `schema.sql`, build script, configs/examples/M04 priors loaded, registries | `kb.build` reproducible; `v_home_facts` returns SH1–SH3 | 0.5 day |
| **2 · Documents + search** | rendered corpus, chunking, local embeddings, BM25 + vss + RRF, `search_docs` | retrieval eval recall@5 ≥ 0.85 | 1 day |
| **3 · Graph** | nodes/edges from catalogue, rules, homes; 5 templated queries | `blocked_retrofits(SH3)` and `eligible_grants` correct | 0.5 day |
| **4 · Tools + loop + validator + template** | tool schemas/handlers, manual loop, claims plan, validator, template, LLM cache | offline chat for SH1–SH3 passes; 0 unvalidated numbers | 1.5 days |
| **5 · Memory** | contracts 0.5.0, memory service + screening, Beta score, `rank_recommendations`, memory API | session 1→3 script passes all guardrails | 1 day |
| **6 · Router + Jev** | Jev client, Claude structured fallback, rules fallback, τ calibration hook | routing accuracy ≥ 95% above τ on labelled prompts | 0.5–1 day |
| **7 · Evaluation + docs** | ≥30 scenarios, eval runner, `docs/LLM.md` (submission item E), health checks | targets in §9 met or gaps documented | 1 day |
`run_scenario` stays on the mock until the orchestrator lands; everything else uses real data now (M04, configs).

## 12. Risks and decisions needed
| Item | Decision / mitigation |
|---|---|
| `ANTHROPIC_API_KEY` / `JEV_API_KEY` not provided yet | Ask the team lead when phase 4/6 needs a live run; everything is built and tested offline first |
| Contract change (0.5.0) | Needs review from P5 (UI) and P2 (orchestrator) before merge |
| Grant facts out of date (BUS, ECO4 status unverified per AGENTS.md) | Documents carry `verified` flags; validator blocks unverified policy numbers |
| Memory nudging could feel manipulative | Bounded λ, best-physics option always shown, every move explained, no loan push (value never a reason to borrow) |
| Personal data in free text | Screening before storage; no PII columns; synthetic customers only |
| Prompt injection via agent notes or documents | Tool outputs and notes are data, never instructions; system prompt says so; eval scenario covers it |
| Embedding model download on a slow link | One-off ~130 MB; BM25-only mode works without it |
