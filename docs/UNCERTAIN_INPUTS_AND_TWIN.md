# Part 1: uncertain inputs, initial priors and deterministic physics

This implementation covers the requested three layers and gas/fuel accounting. It does **not** perform Bayesian calibration. Outputs describe a conditional **prior predictive** distribution, not a meter-calibrated posterior or a guarantee.

**Bayesian extension:** the separate meter-calibration and posterior-prediction workflow is now implemented. See [Bayesian calibration: prior to posterior worked example](BAYESIAN_CALIBRATION_PRIOR_TO_POSTERIOR_WORKED_EXAMPLE.md) for the new API, CLI, numerical constant recovery and held-out prediction results. The preparation workflow described below remains available for initial priors and prior-only predictions.

**Application-history extension:** request-based CLI preparation now learns initial distributions from recorded application answers and saves new baseline facts for later households. See [APPLICATION_HISTORY_PRIORS.md](APPLICATION_HISTORY_PRIORS.md) for import format, snapshot provenance and exclusions. Actual older application submissions require a database/export connection; showcase/generated values are not historical observations.

## 1. The boundary between the layers

1. `build_priors(HouseholdRequest) -> PriorBundle` constructs initial distributions for unknown inputs and constants. It uses property descriptors, local provisional defaults, the appliance catalogue and eligible supplied population/context records. It does not run JAX or fit to this household's meters.
2. `generate_missing_inputs(request, n_samples, prior=...) -> InputSamples` draws a joint set of physically bounded household realizations. It computes dependent geometry, gains, water demand and monthly profiles. It imports no JAX. `prepare_electricity_sample` supplies a reproducible interval realization outside the physics solver; preparation is lazy to bound memory.
3. `simulate_ensemble(InputSamples) -> EnsembleRun` passes numeric realizations to the deterministic JAX solver using `jit` and `vmap`. Every realization passes thermal/fuel and interval conservation checks. It summarizes joint output samples using the shared `contracts.summarise`, with additional explicit variance and Monte Carlo standard error.

The solver contains no random-number generation. Constants remain fixed throughout one realization. Prepared artifacts can be saved and replayed without rebuilding priors. The separate legacy synthetic **label** generator in `generate.py` is a different purpose: it creates development training/evaluation tables using physics. Agreement against those labels is not empirical household accuracy.

## 2. Input audit and the customer experience

The read-only Satyarth handoff supplies geometry, fabric defaults and their sources, heating equipment, occupancy/presence, heating periods, appliance answers and summaries, hot-water configuration, monthly weather and profiles. It includes meter and prior suggestions, which this phase deliberately does not use for calibration. A cached SH2 handoff and a versioned copy of its appliance catalogue are available locally. Catalogue consumption is an estimated centre, not a measured customer fact.

There is no separate local Satyarth worktree: Git currently lists the main checkout and this dedicated worktree. The Satyarth code/handoff was inspected read-only; no branch was checked out or modified.

| Input | Treatment now | Easy question if unavailable |
|---|---|---|
| Location, total floor area, property type, heating equipment | Property/handoff or explicit facts; essential for a new home | Postcode district, approximate total area, house/flat, main heating type |
| Construction age and built form | Property descriptors; otherwise uncertain development archetype | Approximate building era; detached/semi/terrace |
| Occupants and daytime presence | Exact when answered; categorical distributions otherwise | How many people? Mostly home/part day/out all day? |
| Thermostat and heating periods | Exact when answered; bounded distributions otherwise | Typical temperature and approximate heating hours |
| Existing PV and battery | Explicit capacity/absence preserved; unknown ownership is a labelled mixture | Do you already have solar or a battery? Capacity if known |
| Cooking fuel | Exact when answered; conditional mixture otherwise | Gas or electric hob/oven? |
| Cylinder and flat exposure | Exact when answered; unknown state is a mixture | Hot-water tank? Ground/middle/top flat? |
| Appliances | Questionnaire catalogue supplies centres with uncertain demand multipliers | Optional simple appliance selections; avoid model numbers |
| U-values, air changes, efficiencies, geometry detail | Bounded priors; supplied physical measurements stay exact | No technical survey questions required initially |
| Weather and half-hour load/PV series | Supplied series, or explicitly enabled development approximation | Obtain from data service, not the customer |
| Local population evidence | Explicit `ContextRecord` rows, with provenance and cohort filters | Obtain licensed data through Part 0 |

The minimum for a home without `Property`/handoff is four basic facts. Zero facts cannot identify a building, location or fuel system. Unknown fields fail explicitly instead of being ignored. `None` means missing. A numerical zero or a boolean false is an explicit answer. `floor_area_m2` means **total dwelling floor area**; use `exposed_floor_area_m2` for a surveyed external-floor area. A supplied `Property`'s explicit renewable zero is treated as a known absence.

## 3. Available evidence and probability assumptions

The local RdSAP-style defaults are marked `to_verify`; the appliance catalogue contains typical estimates. Verified NEED microdata, local cohort fits, London learned archetypes and PVGIS simulations are **not** present in this checkout. The program does not pretend to have fitted them. Development weather is opt-in, and its uncertainty is propagated. Public/local data can replace a prior through `context`; population evidence is separate from future meter calibration.

Each `PriorMarginal` records family, centre, scale, physical bounds, unit, source, rationale and role. Numeric summaries also retain their joint sample arrays. Categorical missing inputs have probabilities, not invented fractional people or averaged equipment states.

| Quantity | Initial family | Domain / meaning |
|---|---|---|
| Fabric U-values | Lognormal, log-SD 0.22 | Positive; exact observed U-values freeze |
| Ventilation ACH | Lognormal, log-SD 0.25 | 0.1–3 per hour |
| Thermal bridges | Lognormal, log-SD 0.25 | 0.001–0.5 W/m²K |
| Combustion/district efficiency | Scaled beta, concentration 45 | 0.1–1; cannot create heat from fuel |
| Heat-pump seasonal COP | Lognormal, log-SD 0.12 | 1–7; ambient heat is reported separately |
| DHW efficiency / COP | Beta 40 or lognormal 0.15 | Selected by equipment/fuel; direct electric is 1 |
| Setpoint / heating hours | Bounded normal | 10–28°C / 0–24 h/day |
| Geometry aspect, volume, opening factors | Lognormal | Coupled geometry, positive bounded factors |
| Appliance, lighting, water demand factors | Lognormal | Log-SD 0.25 / 0.20 / 0.22 |
| Retained appliance heat | Scaled beta | 0.05–0.999; does not create electricity |
| Pump/fan electrical power | Lognormal | 0.005–0.2 kW |
| Water temperature rise | Bounded normal, 37 ± 3 K | 20–55 K; water heat capacity is 4.19 kJ/kg/K |
| Development monthly temperature / GHI | Bounded normal / lognormal | SD 1.5°C / log-SD 0.15, temporal copula |
| Monthly baseload / DHW factors | Correlated lognormal | AR(1) rho 0.65; exact supplied profiles freeze |
| PV azimuth / tilt / yield | Bounded normal / lognormal | 0–360° / 0–90° / positive yield multiplier |
| Battery usable fraction / efficiencies | Scaled beta | Each less than 1; capacity and power limits apply |

These widths are explicit development choices, not experimentally established confidence. Structural assumptions (adiabatic equal-temperature party wall, 3 K setback, fixed solar-aperture factor, zero initial battery unless supplied) remain model assumptions. Not every source of model error is represented by the ensemble.

For a matching numeric population cohort of n rows and age a years, the initial pooling weight is `n/(n+20) * exp(-a/10)`. The prior mean blends the fallback and cohort mean; variance includes a heterogeneity floor and between-mean disagreement. The most geographically specific eligible recent cohort is selected for each variable, avoiding overlap between district and parent-region aggregates. Future, held-out, same-home and mismatched cohorts are excluded and audited. This is initial prior construction, not a meter-likelihood update. Row counts must mean actual population observations; do not submit repeated aggregate means as independent households.

## 4. Physics and dependencies

Heat-loss coefficient is `H = sum(U*A) + y*exposed_area + 0.33*ACH*volume`, in W/K. The monthly steady-state calculation has occupied and 3 K setback balances and subtracts internal/solar gains, clipping useful heating at zero. Unused gains are explicit. This is a single-zone resistance model; it has no transient capacitance or room-temperature trajectory.

Geometry draws share a footprint/perimeter factor and opening factor. Floor area and storeys determine footprint; ground/top-flat exposure is sampled only if unknown. Appliance energy and retained internal gains share the same realization. Water demand uses `mass * heat_capacity * temperature_rise / 3600`; electric shower energy is removed from the main hot-water load to avoid counting it twice. Pump/fan electricity, outdoor loads and secondary electric heat are separate.

Space heating and DHW are converted independently by the selected efficiency/COP. For combustion: purchased fuel = useful heat + conversion loss. For heat pumps: purchased electricity + ambient heat = delivered heat. Supported purchased energy carriers are grid electricity, mains gas, oil, LPG, solid fuel and district heat. Gas cooking remains when gas heating is replaced. This models **fuel energy**, not gas pressure/pipe flow, concentrations, litres, fuel mass or combustion chemistry.

Half-hour electricity dispatch accounts for demand, PV, direct PV consumption, imports, exports, charging, discharging, storage losses, curtailment and continuous state of charge. Power/capacity/export limits are enforced. Solar reduces grid imports rather than the underlying household demand. Monthly heating is allocated to half-hour profiles; an hourly temperature response or weather-dependent defrost COP is not inferred from that allocation.

## 5. Outputs and downstream handoff

Every available uncertain numeric summary includes mean, variance, SD, median, mode, percentiles, 50% and 90% intervals, histogram, skewness, sample count and `std/sqrt(n)` Monte Carlo standard error. The shared field name `credible_intervals` is retained for compatibility; here it means a **prior predictive** interval, not a calibrated posterior interval. Numeric inputs that are exact have zero variance; deterministic outputs can still be uncertain because other inputs are missing.

* Period totals and monthly trajectories: useful space heating, hot water, fuel input, conversion losses and ambient heat; electricity demand/import/export, PV and battery accounting; six-carrier totals and fuel-by-end-use breakdowns.
* Two purchased-energy conventions: `purchased_electricity_gas_kwh` implements the Lloyds grid-import-plus-gas definition. `all_fuel_purchased_kwh` additionally includes oil/LPG/solid/district energy. Downstream code must select the intended convention explicitly.
* Envelope diagnostics: H, thermal bridges and seven element contributions/shares. These are estimates from the parameter realization, not independently calibrated element losses.
* Electricity diagnostics: peak import, import fraction between 16:00 and 19:00 local time, self-consumption, self-sufficiency, initial/final stored energy. A ratio undefined in any realization returns `null` with reason rather than dropping those realizations.
* Full joint output draws in NPZ, enabling correlated per-fuel savings/carbon/bill calculations. Optional interval arrays are large; request them only when needed.
* Full-calendar-year totals only when all twelve months of that year are present. Partial periods remain partial; no automatic multiplication by twelve.
* Input distributions, prior/context audit, sources/warnings, request hash, seed, model/data versions, conservation residuals, `physics_l1` provenance, and explicit `calibration_status=not_performed`.

`paired_differences` reruns baseline and after scenarios with named common random streams and returns signed baseline-minus-after distributions per carrier. Negative electricity saving is preserved. Packages rerun the combined edits; individual savings are never added. This utility does not rank feasible recommendations or calculate the official improvement potential.

Part 2 can apply sourced date-valid prices and emissions factors **to the same paired draws**, calculate purchased-energy improvement potential with zero-baseline handling, estimate ratings in the appropriate standard mode, and rank feasible packages. Parts 3 and 4 combine those results with recorded/estimated rating and property state for the Green Home Score and indicative value. The scene owner can map estimated element shares to surfaces. Scores, carbon, bills and value are explicitly unavailable in this new Part-1 result; there is no fabricated solar score. Shared `DigitalTwin` orchestration/scene adapters remain a separate integration task; this ensemble contract does not change schema 0.4 or the generated TypeScript mirror.

## 6. Running and replaying

From `PHYSICS MODEL AND BAYESIAN`, a minimal request JSON is:

```json
{
  "property_id": "HOME-001",
  "user_inputs": {
    "postcode_district": "BS16",
    "floor_area_m2": 95,
    "property_type": "house",
    "heating_system": "gas_boiler_combi",
    "occupants": 3,
    "thermostat_c": 20,
    "has_solar_pv": false,
    "has_battery": false,
    "cooking_fuel": "mains_gas"
  },
  "months": ["2025-01", "2025-02"],
  "seed": 42,
  "allow_development_weather": true
}
```

Save a request **inside this worktree**, then:

```sh
.venv/bin/python scripts/run_uncertain_twin.py prepare --request request.json --samples 128 --output artifacts/home001-inputs
.venv/bin/python scripts/run_uncertain_twin.py simulate --prepared artifacts/home001-inputs --output artifacts/home001-result
```

Prepare writes request, initial priors, input summaries, joint input NPZ and a SHA-256 manifest. Simulate verifies hashes and reuses those inputs. Every output directory must be new and inside this worktree. To try a scenario, modify the facts and prepare a new revision with the same identity, seed and sample count. `--changes` accepts a JSON object of simple facts. Neither existing artifacts nor other worktrees are overwritten.

```sh
.venv/bin/python scripts/run_uncertain_twin.py run --demo SH2 --samples 128 --output artifacts/sh2-example
.venv/bin/python scripts/evaluate_uncertain_twin.py --samples 128 --output eval/reports/new-evidence
.venv/bin/python -m pytest -q --basetemp .pytest_tmp
```

The convenience `run` command stores outputs and their input/prior summaries; use **prepare then simulate** to retain the complete input artifact for direct replay. Sample count is configurable from 2 to 10,000, with default 128 and batch size 8. More draws reduce Monte Carlo error; they do not fix inaccurate priors or missing physics. The largest runs can take substantial time. Change batches for memory/performance without changing the intended sample population.

## 7. Remaining work beyond this request

Home-level NumPyro calibration now replaces initial uncertain parameter samples with posterior samples through the deterministic seam; see the separate Bayesian guide above. Verified local datasets, learned load archetypes, PVGIS weather/PV integration and empirical validation remain subsequent data work. Standard-occupancy/SAP preparation, full `DigitalTwin` Protocol integration, capture/scene wiring and downstream rating/score/bills/carbon/value completion remain separate. The prior-only evaluation in this document remains distinct from the controlled calibration experiment.
