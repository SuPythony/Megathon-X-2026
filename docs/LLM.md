# GreenTwin assistant: knowledge base, grounded LLM chat and customer memory

Part 5 (P4). Plan: `docs/P4_KB_LLM_MEMORY_PLAN.md`.

## Run it
```bash
uv run python -m greentwin.kb.build                      # builds data/greentwin.duckdb (stop the app first)
uv run uvicorn greentwin.api.app:app --port 8050          # open http://127.0.0.1:8050
```
`.env` needs `OPENAI_API_KEY=sk-...` (a bare `sk-...` line is also accepted). Without a key the app still works:
answers come from a fixed template filled only from tool outputs. Model: `gpt-6-luna` for every call (`LLM_MODEL`).

## What it does
- **Chat about one home** (SH1–SH3) as one of two synthetic demo customers, or anonymously.
- **Every number is checked.** The answer's numbers must appear in that turn's tool outputs (or the customer's
  message). If not, the model is asked once to correct itself; if it still fails, a template answer built only from
  tool outputs is shown. Each answer shows the check, the tools it called (click to see input and output) and its
  sources (click to read the document).
- **Visual panels:** energy estimate per fuel (from home type vs after the home's own meter readings, 90% ranges,
  metered value), monthly meter readings (missing months shown as gaps, never zero), what drives the estimate
  (M04 contributions), recommended next steps with conditions and memory moves, and what is not possible and why.
- **Memory:** constraints, goals and preferences, either added directly or proposed by the assistant from what the
  customer said (proposals need one-tap confirmation). Save / Not for me on a recommendation records an action.

## How it works
| Piece | Where | Notes |
|---|---|---|
| Knowledge base | `greentwin/kb/` (DuckDB `data/greentwin.duckdb`) | Homes, meter readings, configs, 73 documents (retrofits, rules, grants, factors, tariffs, M04 method, wording rules, datasets), config graph edges, memory, chat and tool logs |
| Document search | DuckDB full-text index (BM25) | `search_docs`; grant documents carry a verified flag |
| Feasibility | `greentwin/retrofit/rules.py` | The 17 rules in `contracts/feasibility_rules.json` plus confirmed memory constraints; rule vetoes always win |
| Tools | `greentwin/llm/tools.py` | `get_home_facts`, `get_energy_prior` (real M04 model + Bayesian meter update), `run_scenario`, `rank_recommendations`, `search_docs`, `query_graph`, `get_customer_memory`, `propose_memory`; strict JSON schemas |
| Chat loop | `greentwin/llm/chat.py` | Rule router; OpenAI Responses API tool loop (≤ 6 rounds); `run_scenario` forced for what-if questions; validator; template fallback; replay cache in `cache/llm/` |
| Memory + ranking | `greentwin/memory/service.py` | Beta acceptance score per customer × retrofit from actions (accepted/completed +3, saved +1, viewed +0.2; rejected −3, dismissed −1; similar options 30%; 180-day half-life) plus preference fit |
| App | `greentwin/api/app.py`, `greentwin/api/static/index.html` | FastAPI + one page; also mounts the M04 `/m04/*` endpoints |

**Ranking.** Feasible options are ordered fabric first by group (fabric, glazing, controls, heating, generation,
storage, tariff), as in `contracts/feasibility_rules.json`. Within a group the config orders by expected net benefit,
which needs physics savings; until the scenario engine is connected those options tie, and the memory score breaks
the tie. Memory can move a score by at most ±15%, never changes a number, never lifts a rule veto, and the best first
step is always shown. Every move states its reason.

## Honest limits
- **Savings, bills and carbon for retrofit packages are unavailable** until the physics scenario engine (P2) is
  connected. `run_scenario` says so; the assistant does not estimate instead.
- Jev routing is not wired (no key); a small rule router is used.
- Document search is keyword-based (BM25). It is enough for this 73-document corpus; dense embeddings can be added
  behind the same `search_docs` tool.
- Customers are synthetic; personal details (emails, phone numbers, full postcodes, street addresses) are refused
  before storage.
