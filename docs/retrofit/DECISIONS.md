# Retrofit engine: decisions, proofs and worked examples

Running log for the Part 2 retrofit engine (`greentwin/retrofit`, `greentwin/optimise`). Every decision has an id, the reason and, where it applies, a proof or worked example that can be rerun.

Branch: `p2/retrofit-engine` (from `main`). Design source: `LLD.pdf` §4.2 (`deliverables/src/lld_content.md`).

## Decision index

| Id | Decision | Status |
| --- | --- | --- |
| R-01 | Branch from `main`, code against `GreenDigitalTwin`; take only `greentwin/retrofit/rules.py` from `satyarth` | Decided (user, 11 Oct 2026) |
| R-02 | Ranged costs → two-piece lognormal: median = central, P5 = low, P95 = high | Decided |
| R-03 | Central-only costs → lognormal, log-sd 0.25, flagged as an assumption | Decided (assumption) |
| R-04 | Only confirmed grants reduce net cost: BUS £7,500. BUS £9,000 off-gas, ECO4, GBIS listed but not applied | Decided |
| R-05 | Package cost = sum of item costs (costs add, savings never do); `None` if any item is unsourced | Decided |
| R-06 | Build order: catalogue → rating → ladder → potential → Pareto | Decided (user, 11 Oct 2026); done |
| R-07 | Meter anchoring: each fuel's "after" = metered × twin_after / twin_base, per sample; extra electricity from a fuel switch takes the heating fuel's ratio | Decided (temporary until calibration lands) |
| R-08 | Batched twin runs: packages with the same fuel config share one cached `jit(vmap)` call | Decided |
| R-09 | Verdicts: recommended if p_suitable ≥ 0.8 (sourced cost); optional if p_suitable ≥ 0.5 or p_gain ≥ 0.5; else not worth adding. p_gain = P(step saves ≥ 3% of the baseline goal metric) | Decided (revised once) |
| R-10 | `property_after`: each retrofit also edits the EPC-style `Property` fields, so the projected score sees the package | Decided |
| R-11 | Rating: clamp only the final anchored SAP, not the model SAP before the delta | Decided (bug found) |
| R-12 | Ladder keeps one of each set of alternatives (catalogue `excludes`, any two heating systems), chosen across feasible and needs-survey items together | Decided (revised once) |
| R-13 | Order: ordering group, then the item's median goal saving when run alone; needs-survey items after the feasible ones | Decided |
| R-14 | Pareto: sourced-cost items only, ≤ 256 packages; weakest net benefit dropped first if over | Decided |
| R-15 | Pareto score gain = projected point score − current score (fixed across samples in the dominance check) | Decided (limitation) |
| R-16 | Knee = furthest point above the cost–benefit chord on the 2-D median front (benefit = goal metric) | Decided (revised once) |
| R-17 | Fix in `twin.retrofit_samples`: the controls retrofit shifts each sample's setpoint instead of overwriting it with one value | Fixed (Part 1 file; tell the owner) |
| R-18 | Ladder leaves out the time-of-use tariff (changes bills not kWh, no sourced tariff) and keeps unsourced-cost items, which are capped at "optional" | Decided |
| R-19 | After main gained P2's physics + calibration: `run_packages` delegates to the twin's own `simulate` (standard mode, PV/battery layer, per-sample weather) instead of re-implementing it; showcase uses `GreenDigitalTwin.calibrate` (SVI) | Decided (11 Oct 2026) |
| R-20 | Twin fixes found on the calibrated posterior: retrofits never make a draw worse (U = min(draw, new); draught proofing cuts each draw's air changes); fixed draw values copied into `base` | Fixed (Part 1 files; tell the owner) |
| R-21 | Twin fix: `simulate` splits draws by PV orientation, one electricity layer per roof face (SH1's posterior has two faces, and every PV run was NaN) | Fixed (Part 1 file; tell the owner) |
| R-22 | Staging: DP over done-sets, year by year; yearly budget + (optionally) reinvested bill savings; `depends_on` within the package respected; minimise savings lost while waiting | Decided (11 Oct 2026) |
| R-23 | Pareto reference-point mode: hard targets met with P ≥ 0.5 over paired samples; best goal saving, then cheapest; nothing relaxed; closest package reported when none qualifies | Decided |
| R-24 | Memory re-rank: orders the packages with P(on front) > 0 by weighted normalised objectives (weights from memory goals) + 0.3 × (acceptance − 0.5); never adds/removes a package | Decided |
| R-25 | Baselines: greedy bill-per-£ and CO2-only (max median carbon in budget); hypervolume of the in-budget median front vs each baseline (Monte Carlo, normalised 4-D box) | Decided |
| R-26 | Relevance: one EBM per measure on real EPC certificates + assessor recommendations (2025, hashed 2% sample); rules-only baseline; ranks, never vetoes | Decided |
| R-27 | Orchestrator `greentwin/api/orchestrator.py`: `run_scenario` assembles `ScenarioResult` from the parts; `plan()` adds staging, Pareto modes, baselines, relevance; cached in memory + `cache/scenarios` | Decided |
| R-28 | Ranking check: synthetic city homes, prior-draw truth, noisy meters → calibrate → potential; Spearman/NDCG vs generator-rule labels, baseline "rank by total kWh"; `p_top_decile` from the posterior | Decided |

---

## R-01 Base branch
`satyarth` holds the real physics, calibration and rules, but isn't merged. Working from `main` keeps this branch small; `GreenDigitalTwin` loads `greentwin/twin/physics.py` automatically once it exists, so the real physics drops in without code changes here. `rules.py` (feasibility layer 1) is copied unchanged from `origin/satyarth`; it uses only contract fields already on `main`.

**Proof (showcase rules).** `assess(SH*)` on the example properties:
- SH1: `external_wall_insulation` is vetoed (conservation area) and `internal_wall_insulation` is feasible.
- SH2: `cavity_wall_insulation` and `air_source_heat_pump` are feasible.
- SH3: loft and PV are vetoed, and `air_source_heat_pump` is `needs_survey`.

These match the Block 4 "done when" checks.

## R-02 Cost distribution from low / central / high
BEIS 2017 Table 3 gives the lowest, mean and highest cost per home across literature sources. These are skewed (cavity £670 / £1,040 / £1,940), so a triangular with mean = central has no valid mode. We read low/high as P5/P95 and central as the median of a lognormal with a different log-sd on each side. It's always positive, matches all three published points and keeps the skew.

**Caveat:** BEIS's central is a mean, and we treat it as a median. For right-skewed items, the true median is a little lower.

**Proof:** `tests/test_retrofit_catalogue.py::test_ranged_cost_matches_source_percentiles`. On loft insulation, P5/P50/P95 are within 3% of £260 / £620 / £930.

## R-03 Central-only costs
Smart controls (£620), the heat pump (£12,500) and PV (£1,876/kWp) have one figure each. Rather than present a point cost as certain, we sample a lognormal with log-sd 0.25 (P5–P95 ≈ ×0.66–×1.51) and add a warning naming the assumption.

**Worked example:**
- Heat pump gross median £12,500 minus BUS £7,500 gives a net median of about £4,850 (90% range about £640–£11,400).
- Most of that width comes from the assumed spread, not from data.
- Replace the assumption once the DESNZ BUS statistics give a range.

## R-04 Grants
`GRANTS` in `catalogue.py` records an amount and a `confirmed` flag. Only confirmed schemes are subtracted:
- BUS rates are alternatives, never stacked.
- The £9,000 off-gas rate is considered only for off-gas-grid homes, and is listed as not applied until it's confirmed on gov.uk.
- ECO4/GBIS depend on eligibility and are excluded (AGENTS.md "Still to verify").

Net cost is floored at 0.

## R-05 Package cost
Installation costs are additive, unlike savings, which always come from one twin run on the whole package. If any item in a package has no sourced cost, the package cost is `None`, because a partial sum would make the package look cheaper than it is. Such packages are excluded from cost ranking (compliance fix 2).

## R-06 Build order
1. `catalogue.py` (done).
2. `rating.py`: unblocks the projected score and the value "after".
3. `ladder.py`.
4. `potential.py`.
5. `optimise/pareto.py`.

## R-07 Meter anchoring
**Problem.** On `main` the twin runs on prior samples, not a calibrated posterior. It sits far from the bills: SH1 gas is about 25,000 kWh from the twin against about 15,000 metered.

**Fix.** Per sample, each fuel is scaled by metered ÷ twin baseline, for the baseline and for every package. The physics decides the relative change, interactions included; the meter decides the level.

A first version scaled heat-pump electricity by the *electricity* ratio. That ratio describes the baseload, not the heat demand, so SH1's heat-pump electricity spread from −4,608 to −1,150 kWh. Now any electricity above the twin's baseline is treated as new heating demand and takes the *heating fuel's* ratio (gas), which measured this home's heat demand. SH1's range became −2,089 to −1,542 kWh.

With a calibrated posterior the ratio is about 1 and the anchor is close to a no-op. Without meter readings, the twin prior is used and a warning says so. Code: `runner.anchored`.

## R-08 Batched runs
`DigitalTwin.simulate` builds a new `jit` on every call (about 0.5 s of compile time each). The ladder needs 10–20 runs and Pareto up to 256. `runner.run_packages` applies each package with the twin's own `retrofit_samples` and stacks all packages that share a fuel config into one call with a cached `jit(vmap)`.

**Proof:** `tests/test_retrofit_engine.py::test_batched_runs_match_single_runs` (rtol 1e-5 against `GreenDigitalTwin.simulate`).

## R-09 Verdicts
The first rule required p_gain ≥ 0.5 for "recommended". That marked SH1's £250 draught-proofing "not worth adding", even though it pays back with probability 1.0, only because its £52/yr saving is under 3% of the £1,950 bill.

Revised rule:
- **recommended** when p_suitable ≥ 0.8 and the cost is sourced (p_suitable = p_feasible × p_net_benefit).
- **optional** when p_suitable ≥ 0.5, or when p_gain ≥ 0.5: a big step that doesn't pay back on the goal alone (e.g. double glazing). The reason says so.
- **not worth adding** otherwise.
- **needs survey** whenever the rules say so.

Net benefit under the bills goal is P(bill saving × lifetime − net cost > 0), undiscounted. With no sourced cost it falls back to P(bill saving > 0), and the reason says the cost is ignored. Under the carbon goal it is P(carbon saving > 0).

## R-10 Property after a package
The score's insulation, heating, glazing, renewables and controls sub-scores read `Property`, not twin parameters. `property_edits.property_after` maps each catalogue retrofit to those fields, e.g.:
- loft → 270 mm, with the catalogue roof U.
- cavity → `cavity_filled`.
- heat pump → `air_source_heat_pump`, electricity, SCOP 2.9, heat-pump cylinder.
- PV → +3.5 kWp.
- controls → smart thermostat + TRVs.
- LED → 100%.

The projected score uses `property_after` + the anchored package energy + `RatingEstimate`.

**Proof:** `test_property_after_and_projected_score_not_lower`.

## R-11 Rating clamp
`sap_estimator.json` clamps SAP to 1–100. Clamping the *model* SAP before taking the delta erased the change for samples whose modelled SAP fell below 1. At current prices SH1's dearest samples have an ECF above 9, so SH1's P5 showed no gain (48 → 48). Now the model SAPs are left unclamped and the final anchored SAP is clamped.

## R-12 / R-13 Selection and order
The ladder is one coherent package. Applying double and triple glazing, or two heating systems, makes no physical sense. Of each set of alternatives the ladder keeps one: sourced cost first, then the highest median lifetime net benefit.

The first version resolved feasible items before needs-survey ones. SH3 then kept HHR storage heaters (saving 5 kWh) over a heat pump (saving 1,028 kWh) only because the heat pump needs a survey. Now alternatives are resolved together, and the dropped feasible rival is named as the fallback if the survey fails.

Order follows the groups in `feasibility_rules.json`; within a group, the item that saves most on its own goes first.

## R-14 – R-16 Pareto
- **Candidates:** feasible items with a sourced cost that change kWh. SH1/SH2 give exactly 256 packages (8 items, one glazing type).
- **Objectives per sample:** net cost (min), carbon saving, bill saving, score gain (max).
- **Score gain** is the point projected score minus the current score. The scorer returns a distribution, not samples, so it is fixed across samples (R-15).
- **Front size:** with 4 objectives most packages are non-dominated somewhere; SH1 has about 160 points with P(on front) > 0.
- **Knee (R-16):** the first version took the point closest to the 4-D ideal. That picked almost the whole catalogue. Now the knee is the point furthest above the chord on the 2-D median front of cost vs the goal's benefit.
  - SH2 bills: fabric + controls + PV, £9.3k.
  - SH2 carbon: draught proofing + cavity + heat pump, £6.3k.
- **Baseline:** greedy bill-saving-per-£ under £10k. On SH1–SH3 the greedy package is itself on the median front (not dominated).

## R-17 Setpoint fix in the twin
`retrofit_samples` replaced every sample's `setpoint_c` with `base − 0.5`, which removed the calibrated spread of the setpoint. It is now an additive shift, like the existing multiplicative rule for baseload and heating hours. This is a Part 1 file, so the owner needs to know.

## R-19 – R-21 Moving onto P2's physics and calibration
**R-19 Delegate to `simulate`.** On main, `greentwin/twin/simulate.py` now does three things my batched runner skipped:
- Sets SAP standard occupancy for `physics_l1`, which has no standard mode of its own.
- Builds the PV/battery half-hourly layer.
- Replays each draw's own calibrated monthly weather and schedules (`PosteriorSamples.inputs`).

`run_packages` now calls `simulate` once per group of packages that it would treat alike: fuel config, keys, PV layer and standard-mode behaviour.
- After calibration, the twin is within about 3% of the bills. SH1 gas: twin 16,010 vs metered 15,493 kWh. SH2 gas: 13,221 vs 12,792.
- So the meter anchor (R-07) now only trims the small residual.

**R-20 Retrofits never make a draw worse.** `retrofit_samples` set every draw of an edited field to one value. On the calibrated posterior this backfired:
- Draught proofing *raised* SH1 gas from 16,010 to 16,805 kWh. The calibrated air-change draws sat below `base − 0.15`, so overwriting them with `base − 0.15` increased ventilation.
- Cavity fill (U 0.55) raised use on newer walls.

The fixes:
- Air changes: each draw minus the same cut.
- Fabric U: min(draw, new U).
- `calibrate` copies values that P2 held fixed across draws into `base`. P2's fixed SH2 roof angles are 168°/28.7°, not the 180°/35° default.

**R-21 PV orientation.** SH1's posterior samples its PV roof face per draw (186°/29° or 241°/47°). `simulate` built one shared PV layer from `base`, and `physics_l1` fails closed on a geometry mismatch, so every PV run on SH1 was NaN. `simulate` now splits draws by orientation, builds a layer per face and merges the runs back in draw order.

## R-22 Staging by dependencies and cash flow
`greentwin/optimise/staging.py`. The ladder gives the steps (recommended and optional, with a sourced cost); the household sets aside a budget each year, and bill savings from finished work go back into the pot when `reinvest` is on.
- **State:** the set of steps already done. Each year any subset of the rest can be installed if the pot covers its median net cost and every `depends_on` step in the package is done or installed that year.
- **Objective:** savings lost while waiting = Σ over years of the annual saving of the steps not yet installed. Tie-break: more cash left.
- **Exactness:** labels (lost, cash) are kept only when not dominated, so the search is exact for this objective. A stage's saving is the sum of ladder marginals: exact in ladder order, an approximation when the plan reorders steps (stated in every result). The package itself is still one twin run.
- **Left out, with a reason:** `needs_survey`, `not_worth_adding`, unsourced cost (can't be cash-planned).
- **Proof:** `tests/test_retrofit_plan.py`: every stage's cost ≤ its pot; heat pump never before the fabric it depends on; reinvesting is never slower.

## R-23 – R-25 Pareto modes and baselines
- **Reference point (R-23):** e.g. "≥ 2 t CO2, ≤ £10k". Each package's P(meets every target) uses the same paired cost/saving samples as P(on front). Qualifying packages (P ≥ 0.5) are ranked by median goal saving, then cost. If none qualifies, the closest package and its probability are returned; targets are never relaxed.
- **Memory re-rank (R-24):** customer-memory goals (`bills`, `carbon`, …) raise the weight of the matching objective; acceptance scores (the memory service's Beta posterior) nudge by at most ±0.15. Only packages that can be on the front are ordered, so the hard constraints and the front itself don't change.
- **Baselines (R-25):** greedy bill-saving-per-£ (existing) and a CO2-only optimiser (largest median carbon cut in budget). Hypervolume: share of the normalised box [0, budget] × [0, max saving]³ dominated by the median points, 200k Monte Carlo points, seed 42. The front's hypervolume is ≥ any single baseline package by construction; the report shows how much more.

## R-26 Relevance model (assessor recommendations)
`greentwin/models/relevance.py`, `scripts/epc_recs_extract.py`, `scripts/train_relevance.py`.
- **Data:** the EPC bulk zip read over HTTP Range; certificates-2025 and recommendations-2025 filtered on the same certificate-number hash (2%), so each kept certificate has all its recommendations.
- **Labels:** `improvement_summary_text` → catalogue ids by documented regexes ("internal or external wall insulation" labels both IWI and EWI). Unmapped texts are counted in `metrics.json`.
- **Features:** only what the certificate records (element sub-scores parsed by `greentwin/score/epc_features`, heating, glazing, controls, lighting, PV, SAP, form, age, floor area). No `potential_*` fields.
- **Split:** by property (UPRN, else certificate) hash, 70/10/20.
- **Baseline:** rules only (element in a poor state → recommend), same metrics: AUC, Brier, precision@3 per home.
- **Use:** `plan()` shows P(an assessor would recommend it) with the top EBM contributions, for feasible items only. It never vetoes and never produces savings.

**Results** (`artifacts/relevance/v1/metrics.json`; 35,065 certificates from 2025, 89,195 recommendation rows; test = 6,944 held-out properties):

| Measure | Train positives | EBM AUC | Rules AUC |
| --- | --- | --- | --- |
| Loft / flat roof / room-in-roof insulation | 3,638 | 0.904 | 0.771 |
| Cavity wall insulation | 3,294 | 0.977 | 0.841 |
| Internal or external wall insulation | 5,864 | 0.964 | 0.911 |
| Floor insulation | 12,910 | 0.973 | 0.906 |
| Draught proofing | 851 | 0.925 | 0.718 |
| Double / replacement glazing | 1,437 | 0.971 | 0.829 |
| Heating controls | 1,739 | 0.986 | 0.871 |
| Air-source heat pump | 44 | 0.995 | 0.836 |
| HHR storage heaters | 1,414 | 0.996 | 0.669 |
| Solar PV | 14,575 | 1.000 | 0.841 |
| Low-energy lighting | 3,246 | 0.999 | 0.920 |
| **Mean** | | **0.971** | **0.835** |

Precision@3 per home: **EBM 0.725 vs rules 0.529**. Secondary glazing has 2 positives and isn't trained.

Read with care:
- EPC recommendations are generated by the RdSAP software from the recorded fields, so a near-perfect AUC (PV, lighting) means the EBM has learnt that software's rules. That is what this label measures: "an assessor's report would list it". It is not evidence that the measure pays.
- The heat pump has only 44 training positives; its AUC is unstable.
- IWI and EWI share one EPC label ("internal or external wall insulation").
- Showcase check: SH2 cavity fill 0.99, SH1 IWI/EWI 0.97 (EWI is vetoed by rules anyway), SH3 HHR storage heaters 0.66.

## R-27 Orchestrator
`GreenTwinOrchestrator.run_scenario(spec)`: calibrate (cached per home) → rules feasibility → baseline and package (one run, meter-anchored) → per-fuel savings, bills, carbon → ladder → improvement potential → rating after → score before/after → value (Part 4) → element heat loss → Pareto. Ruled-out items in a requested package still run but carry a warning. Results are cached by hash(spec + versions); `mock` results are never written to disk. Exposed as `POST /api/scenario`, `GET /api/plan/{pid}` and the LLM tool `run_scenario` (which used to return "unavailable").

## R-28 Ranking check
`scripts/eval_retrofit_ranking.py` → `eval/reports/retrofit_ranking/`. Each synthetic city home's truth is one prior draw; its meters are that draw's months plus 2.5% noise (the generator's `meter_noise_relative`). Label = the generator's rule `all_feasible_one_run_null_if_zero_or_unsupported`, applied to the truth draw with the engine's all-feasible package. Prediction = the engine's potential after SVI calibration on the noisy meters. Baselines: rank by total metered kWh, and by kWh/m². Caveat: truth and prediction share the physics, so this measures identifiability from meters, not physics error on real homes.

## Worked example, first version (stand-in physics, prior samples, meter-anchored)
**To regenerate:** `artifacts/retrofit/v1/` and this table still come from the stand-in physics (provenance `mock`, so they never reach the demo). Rerun `uv run python scripts/retrofit_showcase.py` on P2's physics with calibrated posteriors. It takes more than 10 minutes, because Pareto recompiles per package group.
Reproduce with `uv run python scripts/retrofit_showcase.py`; the full tables are printed and written to `artifacts/retrofit/v1/`. These numbers come from engine `mock` (stand-in physics). They show the method, not results.

| | SH1 Victorian terrace | SH2 1970s semi | SH3 electric flat |
| --- | --- | --- | --- |
| Ladder verdicts | IWI, loft, draught: recommended · glazing, LED, ASHP, PV: optional · floor, controls: not worth adding | cavity, draught, loft: recommended · floor, glazing, LED, PV: optional · controls, ASHP: not worth adding (bills goal) | draught: recommended · triple glazing, LED: optional (no sourced cost) · ASHP: needs survey |
| Heat-pump step | gas saving 6,969, electricity −1,815 (rises) kWh/yr | gas saving 6,203, electricity −1,934 (rises); bill −£18 (−74 to +52) | electricity saving 1,028 kWh/yr if the survey passes |
| Improvement potential | 79% (76–81) | 73% (70–77) | 18% (11–24); 40% if surveys pass |
| Score: now → after the full ladder | 35 → 90 | 52 → 90 | 67 → 87 |
| Pareto knee, bills | draught + loft + IWI + controls + PV, £19.1k | draught + loft + cavity + controls + PV, £9.3k | draught, £252 |
| Pareto knee, carbon | draught + loft + controls + ASHP, £6.3k | draught + cavity + ASHP, £6.3k | draught |

These satisfy the Block 4 checks:
- SH1 vetoes external wall insulation and suggests internal.
- SH2's electricity rises after the heat pump: P(electricity rises) > 0.9 (test).
- SH3 vetoes loft and PV and marks the heat pump `needs_survey`.

## Open
- R-03's cost spread is an assumption, to replace once ranges are sourced.
- p_gain uses the goal metric (bills or carbon) rather than the LLD's "3% of purchased energy or Δscore ≥ 2". Agree with the team.
- Triple glazing stays feasible on SH3 (`R_GLAZING_DONE` blocks double and secondary only); it ranks "optional" with no cost.
- HHR storage heaters change almost nothing in the stand-in physics: same efficiency, and controllability is modelled as heating hours, which standard mode ignores.
- `p_top_decile` is computed in the ranking batch (R-28, `eval/reports/retrofit_ranking/homes.json`), not per request: it needs the population.
- Feasibility layer 2 (Jev / OpenAI fallback) isn't wired into the engine yet; only layer 1 (rules) runs.
- Staging sums ladder marginals when it reorders steps (R-22): exact only in ladder order.
- PV export isn't credited in bills or the rating (no sourced export tariff), so PV is understated.
- `ScenarioResult` assembly: done in `greentwin/api/orchestrator.py` (R-27). Weather scenarios other than `typical` need Part 0's `load_weather`.
