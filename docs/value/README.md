# Part 4: Property value: documentation index

| Doc | What it holds |
| --- | --- |
| [DECISIONS.md](DECISIONS.md) | Every decision (V-01 … V-16) with reasons, evidence and worked examples |
| [EVALUATION.md](EVALUATION.md) | Test-set evaluation: MSE, RMSE, MAE, R², MAPE, Spearman, CRPS, coverage, pinball, PIT, log score for all models; confounding checks; held-out predictions file; full-distribution outputs |
| [VALUE_RESEARCH.md](VALUE_RESEARCH.md) | Published premiums used as benchmarks; confounders; why the estimate is indicative |
| [MERGE_NOTES.md](MERGE_NOTES.md) | What teammates must know or update when this merges |

## Code map
| Path | Role |
| --- | --- |
| `greentwin/value/prepare.py` | Sales cleaning, EPC-before-sale rule, NSPL join, HPI adjustment, splits |
| `greentwin/value/hedonic.py` | Bayesian hierarchical hedonic model (NumPyro SVI) and posterior predictive |
| `greentwin/value/model.py` | `HedonicValueModel` (ValueModel interface): value distribution, uplift, green twins |
| `greentwin/value/metrics.py` | Distribution metrics shared by all models |
| `greentwin/value/green_twins.py` | Greener comparable sales as ranges |
| `scripts/ucl_sales_extract.py` | Streams UCL 857911 (sales) and NSPL over HTTP Range |
| `scripts/train_value_model.py`, `eval_value_model.py`, `report_value_eval.py` | Fit, evaluate, render tables |
| `scripts/build_green_twins.py`, `value_showcase.py`, `prepare_sales.py` | Twins, worked examples, prep check |
| `tests/test_value.py`, `tests/test_value_metrics.py` | Guarantees and metric correctness |
