# M04 energy-intensity prior — build, tuning and evaluation plan

Owner: P3 · Branch: `p3/energy-pred-model` · Model card (results): `docs/M04_ENERGY_PRIOR.md`

## Goal
Predict annual **electricity** and **gas** intensity (kWh/m²/yr) for a home with no meter data, from fields a
`Property` carries. Output is the `EnergyPrior` contract: a full predictive `Distribution` per fuel plus additive
EBM contributions. It is the prior for Bayesian calibration (M09). It never produces customer savings.

## Data
DESNZ **NEED anonymised data 2025**, 4M-row sample (OGL v3, real metered data, England & Wales).
Target = mean of valid (flag `V`) readings 2021–2023 ÷ representative floor area of the NEED band.
Gas: gas-heated homes only (DESNZ weather-corrected). Electricity: all homes (not weather-corrected).

## 1. Split (stratified, by property, assigned once)
- Each NEED row is one property → property-level split; no leakage across splits.
- Strata = `prop_type × age_band × floor_band × region × main_heat_fuel` (≈2,400 cells).
- Inside each stratum, rows are ordered by a seeded hash (seed 42) and cut **60 / 10 / 10 / 20**:
  - `train`: model fitting
  - `tune`: hyperparameter selection (never used for final metrics)
  - `cal`: conformal calibration of the 90% intervals (never used for fitting or tuning)
  - `test`: final metrics, explanations and plots only
- Report a balance table: the share of each stratum level per split (should match to ±0.1 pp).

## 2. Model ladder (same features, same split)
| # | Model | Why it is in the ladder |
|---|---|---|
| 0 | Global mean | R² = 0 reference |
| 1 | NEED median by type × age × region | The domain baseline from the plan |
| 2 | OLS, one-hot main effects | "Simple regression" |
| 3 | OLS on log target (smearing back-transform) | Simple regression done properly for a skewed target |
| 4 | LightGBM (monotone) | Strong gradient-boosting reference + TreeSHAP |
| 5 | XGBoost (monotone) | The plan's boosting baseline |
| 6 | **EBM (monotone)** | **Primary: glass-box, additive kWh/m² contributions** |

Monotone constraints (physics): better EPC band, loft and cavity insulation never raise gas; PV never raises grid import.
Monotone features are excluded from EBM interaction terms so the guarantee holds overall.

## 3. Hyperparameter search
- Seeded random search (`sklearn.ParameterSampler`), fit on a `train` subsample, scored by **MAE on `tune`**.
- Selection rule: lowest tune MAE, subject to 0 monotonicity violations. Ties → the simpler model.
- Spaces:
  - **EBM:** learning_rate, max_leaves, min_samples_leaf, smoothing_rounds, max_bins, number of interaction pairs, target (raw vs log).
  - **LightGBM:** num_leaves, min_child_samples, learning_rate, feature_fraction, lambda_l2, target (raw vs log).
  - **XGBoost:** max_depth, min_child_weight, learning_rate, subsample, reg_lambda.
- All trials go in `artifacts/m04_energy_prior/v2/trials.csv`; the chosen values go in `config.json`.
- The winners are refit on the full `train`, with early stopping on `tune`.

## 4. Evaluation on `test` (only after selection)
- **Point metrics:** R², MAE, RMSE, MAPE, MdAPE, bias, Spearman ρ.
- **Proof of improvement:** paired bootstrap (1,000 resamples of test homes) gives a 95% CI for each model's R² and MAE, and for **ΔMAE and ΔR² of EBM vs every other model**.
- **Uncertainty:** split-conformal 90% intervals from `cal`, per stratum, for every model. Report coverage (target 0.90) and mean width. A sharper interval at the same coverage is better.
- **Ceiling:** the share of variance between feature cells (saturated cell means), estimated in-sample on `train` and also by cross-fit. It is an upper bound on any model's R² from these features. Report EBM R² as a % of the ceiling.
- **Archetype level:** R² of mean predicted vs mean actual over type × age × floor × region cells with n ≥ 50. This is the accuracy that matters for area priors and the dashboard, and it is labelled as aggregate-level.
- **Segments:** MAE, R² and coverage by property type, age band, region and EPC band.
- **Monotonicity sweep:** 0 violations required.

## 5. Explainability (logged for the test set)
- **Global:** EBM term importances and every shape function (bin → kWh/m²), plus LightGBM mean |SHAP| for comparison. Rank agreement shows the explanations are robust.
- **Local:** per-home EBM contributions for every test home → `test_explanations.parquet` (git-ignored). Check that the contributions sum to the prediction for every row.
- **Exemplars:** waterfalls for SH1–SH3 and for typical, high and low test homes.

## 6. Visual report
`eval/reports/m04/` holds the PNG figures and `report.html`, which is also published as an artifact for the team:
the model ladder with CIs, predicted-vs-actual hexbins (EBM vs OLS), decile calibration, coverage by stratum,
shape functions, importance (EBM vs SHAP), waterfalls, the hyperparameter search, the ceiling, and archetype R².

## 7. Tests (only what judges look at)
Contract validation; provenance `ebm`; monotonic sweeps; structural zero gas for no-gas homes;
contributions sum to prediction; split is stratified and disjoint.

## Honest framing
Per-home intensity is driven heavily by occupant behaviour, which none of these features capture. Per-home R²
is therefore bounded well below 1 for **any** model. The case we make: EBM beats simple regression with
confidence intervals, captures most of the explainable variance (the ceiling), is very accurate at archetype
level, and its 90% intervals are honest. The calibration step (M09) then narrows each home's prior using its own meters.
