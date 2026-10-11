# Retrofit engine (Part 2): what it does and where

Implements `EnergyRetrofitEngine` (`greentwin/interfaces.py`) as `greentwin.retrofit.engine.RetrofitEngine`. Design: `LLD.pdf` §4.2. Decisions, proofs and the worked example: `docs/retrofit/DECISIONS.md`.

| Module | What it does |
| --- | --- |
| `retrofit/catalogue.py` | Validated catalogue, fabric-first ordering, alternatives, sampled costs (null + reason when unsourced), confirmed grants only |
| `retrofit/rules.py` | Feasibility layer 1 (`feasibility_rules.json`), from the `satyarth` branch |
| `retrofit/runner.py` | Batched twin runs per package (never summed), meter anchoring, bills (Ofgem Q4 2026), carbon (DESNZ 2025), `EnergyBreakdown` |
| `retrofit/property_edits.py` | `property_after`: the recorded features after a package, for the projected score |
| `retrofit/rating.py` | `sap_estimator.json`: standard-mode runs → anchored SAP → `RatingEstimate` ("estimated rating, not an EPC") |
| `retrofit/ladder.py` | Selection, order, per-fuel marginal effects per sample, cost, payback, p_net_benefit, p_suitable, p_gain, verdict, reason, projected score |
| `retrofit/potential.py` | Improvement potential: all feasible retrofits in ONE run; null + reason at baseline 0; with surveys |
| `optimise/pareto.py` | ≤ 256 packages, P(on front) per sample, knee per goal, budget filter, greedy baseline |
| `retrofit/engine.py` | `RetrofitEngine`: the Protocol, per-home warnings, provenance (`mock` while the stand-in physics runs) |

```bash
uv run pytest -q tests/test_retrofit_catalogue.py tests/test_retrofit_engine.py
uv run python scripts/retrofit_showcase.py      # SH1-SH3 -> artifacts/retrofit/v1/*.json + Markdown tables
```

The real physics and calibration (branch `satyarth`) plug in through `greentwin/twin/physics_api.get_run_physics`. Once a calibrated posterior replaces `prior_samples`, the meter anchor is close to a no-op (R-07).
