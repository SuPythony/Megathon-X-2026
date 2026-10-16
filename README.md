# GreenTwin (Team MegaMind)

🏆 **Runner-up, Lloyds Problem Statement 2 track, Megathon X 2026.**

Megathon X: Singularity · Lloyds Problem Statement 2 — Green Home Score and Retrofit Recommendation Engine.

GreenTwin builds a calibrated physics digital twin of each UK home and uses it to compute:
- The Green Home Score.
- A feasible retrofit path.
- Per-fuel energy and carbon savings.
- An indicative value range.

Every uncertain number carries a Bayesian credible interval. A tool-only LLM explains the results. A Three.js 3D twin lets customers explore retrofits. A Lloyds dashboard turns this into green-lending demand.

## System overview
Five parts, plus a data layer (Part 0) and an orchestrator. Each part codes against `greentwin/interfaces.py`.
1. **Digital twin**: a Level-1 RC heat-balance plus equipment simulator for each home, Bayesian-calibrated to its meter readings. It is the only source of numbers. It also has photo-to-twin capture and a 3D view.
2. **Energy and retrofit engine**: an EBM prior, the baseline, the retrofit ladder and per-fuel savings, carbon and bills. Every package is rerun on the twin, never summed. It also computes improvement potential, the estimated rating after retrofit and a Pareto front.
3. **Green Home Score**: 0–100 with fixed weights and monotonic sub-scores.
4. **Property value**: a Bayesian hierarchical hedonic model. It gives an indicative range only.
5. **Feasibility, explanation and chat**: rule vetoes, then Jev (a System-1 model) for routing and feasibility. A tool-only LLM explains results and reruns scenarios; a numeric validator rejects any number not from a tool.

The API serves all of this to a 3D twin viewer and a Lloyds dashboard (locality heat map, segments, Green Profits scenario).

## Main tech
- **Physics and Bayesian:** JAX, NumPyro (NUTS/SVI), ArviZ
- **ML:** InterpretML EBM, XGBoost, LightGBM, SHAP, MAPIE (conformal), PyTorch Geometric (GraphSAGE)
- **Optimisation and causal checks:** pymoo, EconML, DoWhy
- **Data and geo:** Polars, pandas, DuckDB, GeoPandas, PySAL
- **LLM and knowledge base:** OpenAI Responses API with DuckDB BM25 retrieval
- **Contracts and API:** Pydantic v2, FastAPI, Uvicorn
- **Front end and 3D:** React 19, TypeScript, Vite, Three.js (react-three-fiber, drei), Zustand
- **Data sources:** EPC register, HM Land Registry, DESNZ emission factors, Ofgem tariffs, London smart-meter data. See the strategy report for the full list.

## Architecture and system design
**The main architecture and system design live in two documents. Start here:**
- **`docs/STRATEGY_REPORT_OVERVIEW.md`**: the overview (ideas, data, ML approach, evaluation).
- **`docs/STRATEGY_REPORT.md`**: the full report (system design, architecture and build plan).

## Other references
- **Implementation plan:** `docs/IMPLEMENTATION_PLAN.md`
- **Team knowledge / rules:** `AGENTS.md`
- **Shared data formats:** `contracts/`

## Quick start
```bash
uv python install 3.12
uv sync
uv run pytest -q
uv run python scripts/make_showcase_examples.py
cp .env.example .env   # keys are placeholders; ask the team lead when needed
```

Data licences and attributions are recorded in `docs/DATA.md`, which is written during data ingestion. It includes: "Contains HM Land Registry data © Crown copyright and database right. Licensed under the Open Government Licence v3.0."
