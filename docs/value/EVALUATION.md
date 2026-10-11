# Property value model: evaluation report (v1)

Submission item G (evaluation) for Part 4, plus item I (held-out predictions). Decisions referenced as V-xx are in [DECISIONS.md](DECISIONS.md). Reproduce:

```bash
uv run python scripts/train_value_model.py --version v1      # fit (SVI, 8,000 steps, seed 42)
uv run python scripts/eval_value_model.py --version v1       # all metrics below + predictions file
uv run python scripts/report_value_eval.py v1                # renders the tables
```

## 1. What is evaluated
The model predicts the **log sale price** of a home, adjusted to July 2026 money with UK HPI (V-07), and returns a **full predictive distribution** for every home, not a single number (V-16). The value uplift is built on the same posterior. So the main test is: on homes the model has never seen, are the predicted distributions accurate (a good centre) **and** well calibrated (the right width)?

## 2. Data and splits (V-04, V-05, V-08)
- UCL linked Land Registry + EPC sales, 10% property sample: 2,241,184 sales → **1,047,182** usable after keeping only sales whose EPC was lodged before the sale (V-05).
- Split by property (address hash), assigned once; no property is in two splits.
- **Training:** 712,053 sales, 2008–2023.
- **Held-out test:** 50,000 sales (random sample of 152,630) of unseen properties, 2008–2023.
- **Out-of-time test:** all 8,602 sales in 2024 of unseen properties: unseen homes in an unseen year, the brief's "market conditions" check.

## 3. Models compared (brief §11: compare with a simple baseline)
| Model | Predictive distribution from |
| --- | --- |
| **Bayesian hierarchical hedonic** (product) | posterior predictive draws: 300 posterior samples of every parameter + StudentT noise |
| LightGBM quantile | 5 predicted quantiles (5/25/50/75/95%) turned into draws by piecewise-linear interpolation (approximate) |
| OLS with SAP / without SAP | prediction + bootstrap draws of training residuals; LA, type/form, age band, year fixed effects |
| Naive baseline | local authority × property type median + bootstrap residuals |

All models are scored by **one function** (`greentwin/value/metrics.py`) on the same rows from 300 draws each, so the numbers are directly comparable. The metric code is unit-tested against closed-form answers (`tests/test_value_metrics.py`: CRPS of a normal forecast, calibrated vs over-confident forecasts).

## 4. Metrics
| Metric | Scale | Meaning | Better |
| --- | --- | --- | --- |
| MSE, RMSE, MAE | log price | error of the predictive mean | lower |
| R² (log) | log price | share of variance explained by the predictive mean | higher |
| R² (£), RMSE £, MAE £ | price | error of exp(predictive median) in pounds | R² higher, errors lower |
| MAPE, median APE | price | mean / median absolute % error | lower |
| Spearman | price | rank agreement of predicted vs actual price | higher |
| **CRPS** | log price | proper scoring rule for the whole distribution: rewards sharp **and** calibrated forecasts | lower |
| Coverage 50/80/90/95% | log price | share of actual prices inside the central interval; should equal the nominal level | ≈ nominal |
| Width 90% | log price | average width of the 90% interval (sharpness) | narrower, if coverage holds |
| Pinball (5/50/95%) | log price | quantile loss, averaged over three quantiles | lower |
| PIT KS | — | distance of the probability integral transform from uniform (0 = perfectly calibrated) | lower |
| Log score | log price | mean log density of the actual value under a Gaussian fitted to each forecast's draws | higher |

## 5. Results

### Held-out test (unseen properties, 2008–2023): n = 50,000

| Model | MSE (log) | RMSE (log) | MAE (log) | R² (log) | R² (£) | RMSE £ | MAE £ | MAPE | Median APE | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Bayesian hierarchical hedonic** | **0.0494** | **0.2222** | **0.1568** | **0.870** | 0.820 | 140,205 | **61,137** | **16.5%** | **11.3%** | **0.935** |
| LightGBM quantile (approx. draws) | 0.0560 | 0.2367 | 0.1696 | 0.852 | **0.823** | **139,027** | 64,654 | 17.6% | 11.9% | 0.925 |
| OLS with SAP | 0.0790 | 0.2811 | 0.2077 | 0.792 | 0.754 | 163,849 | 80,423 | 21.9% | 15.8% | 0.892 |
| OLS without SAP | 0.0791 | 0.2812 | 0.2078 | 0.791 | 0.754 | 163,923 | 80,471 | 21.9% | 15.8% | 0.891 |
| Naive: LA × type median | 0.1406 | 0.3750 | 0.2778 | 0.629 | 0.483 | 237,813 | 111,862 | 28.0% | 20.8% | 0.800 |

| Model | CRPS ↓ | Cov 50% | Cov 80% | Cov 90% | Cov 95% | Width 90% | Pinball ↓ | PIT KS ↓ | Log score ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Bayesian hierarchical hedonic** | **0.1158** | 0.502 | 0.789 | 0.882 | 0.930 | **0.629** | **0.0433** | 0.014 | 0.093 |
| LightGBM quantile (approx. draws) | 0.1217 | 0.475 | 0.795 | 0.852 | 0.874 | 0.648 | 0.0448 | 0.055 | **0.109** |
| OLS with SAP | 0.1517 | 0.490 | 0.792 | 0.891 | 0.941 | 0.864 | 0.0559 | **0.009** | −0.155 |
| OLS without SAP | 0.1517 | 0.488 | 0.794 | 0.890 | 0.941 | 0.864 | 0.0560 | 0.010 | −0.154 |
| Naive: LA × type median | 0.2032 | 0.486 | 0.787 | 0.888 | 0.939 | 1.169 | 0.0747 | 0.011 | −0.442 |

### Out-of-time (unseen properties, 2024): n = 8,602

| Model | MSE (log) | RMSE (log) | MAE (log) | R² (log) | R² (£) | RMSE £ | MAE £ | MAPE | Median APE | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Bayesian hierarchical hedonic** | **0.0576** | **0.2401** | **0.1664** | **0.854** | **0.851** | **119,520** | **58,728** | **18.0%** | **11.8%** | **0.928** |
| LightGBM quantile (approx. draws) | 0.0626 | 0.2502 | 0.1758 | 0.841 | 0.814 | 133,575 | 62,296 | 18.8% | 12.1% | 0.923 |
| OLS with SAP | 0.0849 | 0.2914 | 0.2101 | 0.784 | 0.759 | 152,281 | 75,777 | 23.4% | 15.6% | 0.893 |
| OLS without SAP | 0.0846 | 0.2909 | 0.2095 | 0.785 | 0.757 | 152,701 | 75,658 | 23.3% | 15.4% | 0.893 |
| Naive: LA × type median | 0.1385 | 0.3722 | 0.2762 | 0.648 | 0.557 | 206,264 | 102,794 | 28.9% | 20.3% | 0.817 |

| Model | CRPS ↓ | Cov 50% | Cov 80% | Cov 90% | Cov 95% | Width 90% | Pinball ↓ | PIT KS ↓ | Log score ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Bayesian hierarchical hedonic** | **0.1238** | 0.485 | 0.768 | 0.871 | 0.920 | **0.629** | **0.0464** | 0.040 | −0.006 |
| LightGBM quantile (approx. draws) | 0.1278 | 0.484 | 0.799 | 0.853 | 0.874 | 0.668 | 0.0474 | 0.053 | **0.024** |
| OLS with SAP | 0.1549 | 0.504 | 0.790 | 0.887 | 0.936 | 0.863 | 0.0576 | 0.051 | −0.195 |
| OLS without SAP | 0.1545 | 0.501 | 0.795 | 0.886 | 0.937 | 0.865 | 0.0576 | 0.043 | −0.194 |
| Naive: LA × type median | 0.2015 | 0.498 | 0.790 | 0.884 | 0.938 | 1.170 | 0.0743 | **0.027** | −0.438 |

## 6. Reading the results
1. **The Bayesian model is the best overall.** On held-out homes it has the lowest MSE (0.049) and CRPS (0.116), the highest R² on log price (0.870) and the best ranks (Spearman 0.935). It keeps that lead on 2024 sales of unseen homes (R² 0.854, CRPS 0.124, median error 11.8%), so it generalises to a later market. On the pound scale LightGBM is level on the test set (R² 0.823 vs 0.820) but behind out of time (0.814 vs 0.851).
2. **Calibration.** Every model with residual-based intervals (OLS, naive) is well calibrated (PIT KS ≈ 0.01) but much wider (90% width 0.86–1.17 log points vs 0.63). The Bayesian intervals are sharp and close to calibrated: 50% → 0.50, 80% → 0.79, 90% → 0.88, 95% → 0.93 on test. The slight under-coverage in the outer tails is the known effect of mean-field SVI (posterior variance is underestimated). Mitigation options: NUTS on a subsample, or a conformal widening factor fitted on the validation split. LightGBM under-covers most (95% → 0.87).
3. **Log score** slightly favours LightGBM because the Gaussian fitted to its interpolated draws has lighter tails. CRPS, which uses the draws directly, favours the Bayesian model on both test sets.
4. **The rating barely predicts price.** OLS with and without SAP are the same to the 4th decimal (MSE 0.0790 vs 0.0791). Once location, size, type, age and year are known, the rating adds almost no predictive power. This is why the value uplift is small, wide and labelled indicative (V-13).
5. **Naive baseline** (median by LA × type): R² 0.63. Every model beats it by a wide margin, so the features carry real information.
6. In pounds, RMSE (£140k) is dominated by expensive homes. Median APE (11.3%) is the more representative error for a typical home.

## 7. Rating effect: confounding checks (V-11)
| Design | % value per SAP point | Implied A/B vs D |
| --- | --- | --- |
| OLS, LA fixed effects | 0.076 (SE 0.006) | ≈ +2.3% |
| Within LSOA × type/form × age band × year cells (93,720 cells, 224,603 sales) | 0.205 (90% 0.196–0.216) | +6.2% |
| Bayesian hierarchical (pooled) | 0.255 (90% 0.20–0.31) | +7.8% |
| Repeat sales, same floor area (17,850 pairs) | 0.483 | ≈ +15% (renovation bias) |
| Published owner-occupied studies | ≈ 0.05–0.11 | +1.6% to +3.4% |

The product mixes the first three designs equally (V-13), so the disagreement between designs becomes part of the customer's range.

## 8. Held-out predictions (submission item I)
`deliverables/test_predictions/value_v1_test_predictions.csv.gz` (58,602 rows): one row per held-out sale (test and out-of-time) with the full predictive summary.

| Column | Meaning |
| --- | --- |
| `set` | `test` or `out_of_time` |
| `transactionid` | Land Registry transaction id (public dataset key; no address) |
| `postcode_district`, `lsoa21` | location at district / LSOA level only (brief §9) |
| `ptype`, `year`, `sap` | property type, sale year, EPC SAP at sale |
| `price_adj_gbp` | actual price, HPI-adjusted to July 2026 |
| `pred_mean_log`, `pred_sd_log` | predictive mean and standard deviation of log price |
| `pred_p5_gbp` … `pred_p95_gbp` | predictive 5 / 25 / 50 / 75 / 95% quantiles in £ |
| `pit` | probability integral transform of the actual price (uniform if calibrated) |

## 9. Full-distribution outputs in the product (V-16)
Every number the value model returns is a `Distribution` (mean, median, mode, std, P1–P99, 50%/90% intervals, tail probabilities, skewness, histogram) built with `contracts.summarise`:

| Output | Distribution of |
| --- | --- |
| `ValueImpact.uplift_pct` | % change from the triangulated rating effect × ΔSAP, per draw; tail P(uplift ≤ 0) |
| `ValueImpact.rating_effect_pct_per_sap_point` | the mixed rating effect |
| `ValueImpact.current_value_distribution` | the home's posterior predictive value today, **when no valuation is supplied** |
| `ValueImpact.value_after_gbp` | value × (1 + uplift), computed draw by draw (joint, not interval arithmetic) |
| `GreenTwin.price_per_m2_gbp` | price per m² across matched greener sales |

Showcase predictive values (`HedonicValueModel.value_samples`, 4,000 draws; prices are right-skewed, so the headline is the median):

| Home | Stated valuation | Model median | Mean | Mode | 50% interval | 90% interval |
| --- | --- | --- | --- | --- | --- | --- |
| SH1 Manchester terrace | £215,000 | £238,134 | £246,013 | £225,329 | £197,170 – £282,414 | £148,262 – £374,745 |
| SH2 South Gloucestershire semi | £325,000 | £375,278 | £391,966 | £344,179 | £316,526 – £449,064 | £242,850 – £592,018 |
| SH3 Croydon flat | £330,000 | £272,103 | £286,306 | £255,918 | £222,369 – £329,389 | £164,711 – £452,080 |

All three stated valuations fall inside the model's 90% intervals (SH1 and SH2 inside the 50% interval). The showcase homes have no LSOA code, so their LSOA effect is drawn from the fitted LSOA spread, which correctly widens their intervals.

## 10. Limitations
- Mean-field SVI slightly under-covers at 90–95%; region × type slope intervals are too narrow (V-13 mixture compensates for the rating effect).
- 10% property sample; the full data would tighten local effects.
- Sales prices are an imperfect proxy for value (sale-specific noise is included in the predictive distribution by design).
- Evaluation is on observed sales; agreement with them is not evidence that a retrofit causes a price change (VALUE_RESEARCH §2).
