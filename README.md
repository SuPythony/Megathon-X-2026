# GreenTwin (Team MegaMind)

Megathon X: Singularity · Lloyds Problem Statement 2 — Green Home Score and Retrofit Recommendation Engine.

GreenTwin builds a calibrated physics digital twin of each UK home and uses it to compute:
- The Green Home Score.
- A feasible retrofit path.
- Per-fuel energy and carbon savings.
- An indicative value range.

Every uncertain number carries a Bayesian credible interval. A tool-only LLM explains the results. A Three.js 3D twin lets customers explore retrofits. A Lloyds dashboard turns this into green-lending demand.

- **Approach (submission):** `deliverables/M1.pdf`
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

Rebuild the submission PDF:
```bash
uv run python deliverables/src/build_m1.py
```

Data licences and attributions are recorded in `docs/DATA.md`, which is written during data ingestion. It includes: "Contains HM Land Registry data © Crown copyright and database right. Licensed under the Open Government Licence v3.0."
