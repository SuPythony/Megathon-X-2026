# Part 3: Green Home Score: documentation index

| Doc | What it holds |
| --- | --- |
| [GREEN_SCORE_STANDARDS.md](GREEN_SCORE_STANDARDS.md) | UK standards research (SAP 10.2 / RdSAP 10, reformed HEM EPC, NEED, Part L, AECB, LETI, EnerPHit, MEES, ECO4); full parameter catalogue with EPC/NEED fields and how a homeowner supplies each one; ML options for learning weights |
| [DECISIONS.md](DECISIONS.md) | Every decision (D-xx) with its reason, quantitative proofs (P-xx), worked examples for SH1–SH3, data on hand |
| [LEARNED_WEIGHTS.md](LEARNED_WEIGHTS.md) | Generated results of the weight learning: final weights with intervals, per-target fits, baselines, Bayesian check, brief-vs-learned agreement, limitations |

## Code map
| Path | Role |
| --- | --- |
| `contracts/score_mapping.json` | Fixed weights + monotone sub-score mappings (v0.2.0 adds U-value curves) |
| `greentwin/score/score.py` | `GreenHomeScorer` (headline score, distribution, band, contributions, learned companion) |
| `greentwin/score/mapping.py` | Sub-score functions (vectorised, monotone) |
| `greentwin/score/sampling.py`, `priors.json` | Missing-input sampling (peer band mix, loft depth by age) |
| `greentwin/score/issues.py` | Key issues with evidence (peer kWh/m² from NEED) |
| `greentwin/score/stub_energy.py` | Temporary metered-intensity input until Part 2's baseline lands |
| `greentwin/score/epc_features.py`, `need_features.py`, `archetypes.py`, `learn.py` | Training data and weight learning |
| `scripts/train_score_weights.py`, `report_score_weights.py`, `fit_score_priors.py`, `fit_score_curves_ebm.py` | Reproducible runs (seed 42) |
| `scripts/download_epc.py`, `epc_bulk_extract.py` | EPC API bulk download and property sampling |
| `tests/test_score.py`, `tests/test_score_epc_features.py` | Monotonicity sweeps, rules, parser checks |

## Reproduce
```bash
uv run python scripts/download_epc.py bulk --env-file .env --out D:/greentwin_data/epc
uv run python scripts/epc_bulk_extract.py --zip D:/greentwin_data/epc/domestic-csv.zip --pct 25 --out data/interim/epc25
uv run python scripts/fit_score_priors.py
uv run python scripts/train_score_weights.py --epc-dir data/interim/epc25 --boot 100 --version v1
uv run python scripts/report_score_weights.py v1
uv run python scripts/score_showcase.py
uv run pytest -q
```

## Ideas backlog (not built yet)
- Region in the archetype cells (needs an LA → region lookup for EPC rows).
- Score PV overlap on real PV homes once Part 2 gives self-consumption (currently a 40% default).
- Lighting as a scored parameter (D-08): decide from the learned-weights evidence.
- Align the heating sub-score order with the reformed EPC's heating-system banding (no fossil system ≥ C).
- Validate the band-probability calibration once Part 1 posteriors exist.
