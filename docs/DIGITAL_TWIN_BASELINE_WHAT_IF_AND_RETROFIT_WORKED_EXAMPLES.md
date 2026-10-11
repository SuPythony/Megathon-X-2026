# GreenTwin digital twin: baseline, what-if and retrofit worked examples

**Part 1 · Household inputs → initial distributions → JAX physics → output distributions**  
Team MegaMind · 11 October 2026  
Model: `thermal-electricity-fuels/0.3.0` · Worktree: `PHYSICS MODEL AND BAYESIAN`

This document explains exactly how the implemented physics block handles a household, a hypothetical change and a retrofit package. It follows the Part 1 separation in the system design: prepare uncertain inputs separately, run deterministic physics for each realization, and return distributions and joint samples for the next parts. The examples use saved, reproducible simulation runs. All ranges are **prior predictive estimates**, with Bayesian meter calibration reserved for the next phase.

## 1. The three cases use one physics block

| Case | What the caller supplies | What happens inside the block | What the caller receives |
|---|---|---|---|
| A. Baseline household | Recorded property and customer answers | Preserve supplied facts; construct distributions for unknown quantities; sample and simulate | Baseline energy, fuel, heat-loss and electricity distributions |
| B. User what-if | The baseline household plus proposed changed facts | Prepare the changed household and rerun the same solver | After-state distributions and signed per-fuel changes against baseline |
| C. Retrofit package | The baseline household plus the package's physical edits | Apply all edits together and rerun the coupled thermal, fuel and electricity calculation | Package distributions, interactions and signed per-fuel changes |

The simulator responds to the physical state presented to it. It does not need to know whether the proposed state came from a customer, chat or the retrofit engine. This makes it a separate, reusable computation block: the same building and equipment definitions drive every case.

Its three stages have distinct responsibilities:

1. **Initial prior construction** chooses probability distributions for unknown constants and household quantities from recorded property information, eligible previous application answers and physical defaults.
2. **Missing-input generation** samples those distributions and constructs consistent geometry, demand, weather and electricity profiles. It produces a prepared input artifact separately from simulation.
3. **Deterministic JAX simulation** takes one complete realization at a time. Repeating this over the prepared samples produces the final distribution of outcomes.

For a fixed complete realization, the physics calculation is deterministic. Randomness enters during input preparation, before the solver.

## 2. The example household and evaluation settings

All three examples use the recorded SH2 showcase household, a 1970s cavity-wall semi-detached home. It is an example household, rather than a real customer record.

| Input | Value used |
|---|---|
| Household identity | `SH2` |
| Location | BS16, South West England; latitude 51.487°, longitude −2.512° |
| Dwelling | House; semi-detached; built in 1972, age band 1967–1975 |
| Total floor area | 95 m² across 2 storeys |
| Wall description | Cavity wall, uninsulated |
| Roof description | Pitched loft, 100 mm loft insulation |
| Floor description | Solid floor, uninsulated |
| Windows | Older double glazing, pre-2002 |
| Baseline heating | Condensing mains-gas boiler |
| Baseline hot water | From the main heating system |
| Customer-confirmed occupants | 4 |
| Customer-confirmed thermostat | 20°C |
| Example-specified cooking | Mains gas, 1 kWh/day |
| Cylinder | Explicitly absent |
| Existing solar PV and battery | Explicitly absent, 0 kWp and 0 kWh |
| Simulation period | January–December 2025, all 12 months |
| Random seed | 42 |
| Ensemble size | 128 realizations per scenario |
| JAX batch size | 8 |
| Weather basis for these saved runs | Explicitly enabled development weather, sampled with uncertainty |

The 1 kWh/day cooking value is an exact assumption for this demonstration. Ordinary customer intake can use a simple cooking-fuel/appliance answer, from which the generator constructs uncertain consumption.

Each realization has **17,520 half-hour electricity intervals** over this 365-day year. The four main evaluations—baseline, fabric-only, heat-pump-only and combined package—therefore contain **512 household realizations** and **8,970,240 electricity dispatch intervals**.

## 3. Exactly what happens when a baseline household is submitted

### Step 1 — Validate the request and preserve recorded facts

The application passes a `HouseholdRequest`: household identity, property/handoff, customer inputs, requested months, seed and available weather or electricity data. A household without a property/handoff needs the basic location, total floor area, dwelling type and heating-system facts to describe the simulation.

Customer-provided facts remain fixed across realizations. Here, every realization has four occupants, a 20°C setpoint, 95 m² total dwelling area and the same chosen heating system. `false` means confirmed absence, `0` is a supplied numerical value, and a missing value is treated as unknown. Supplied physical measurements also remain exact within the requested calculation.

### Step 2 — Read eligible previous application answers

For application requests, `prepare_application_inputs` reads the household-history snapshot before recording the current baseline. It selects relevant previous answers by available district, region, home type, floor area, construction age and equipment. Demand cohorts prefer a floor area within 30% where suitable records exist, and daily energy evidence can additionally match occupant count.

Each distinct household contributes its latest eligible baseline revision. The current household/customer group, future records and held-out evaluation records are excluded from the population evidence. Proposed what-ifs and retrofits do not become observations. Generated samples and simulator results are never treated as customer answers.

Recent answers receive more weight: `recency weight = exp(−age in years / 10)`. With effective evidence weight `N`, the numeric pooling weight is `N / (N + 20)`. For illustration, 20 equally weighted observations give a 50% pooling weight; 80 give 80%. Categorical distributions use twenty fallback pseudo-observations. These examples explain the rule; the SH2 evidence runs below use their saved property/default distributions and do not learn from real customer history.

The new household's own answers are used immediately as exact facts. After successful baseline input generation, eligible recorded facts can inform later households. Prior construction records the snapshot identity and cohort details for replay.

### Step 3 — Construct initial distributions for constants and unknown inputs

The prior builder creates source-labelled, physically bounded marginals. A known value is exact; an unknown value receives a distribution appropriate to its role.

| Unknown quantity | Initial distribution used by the implementation |
|---|---|
| Fabric U-values | Positive lognormal, initial log-SD 0.22 |
| Ventilation air changes | Lognormal, initial log-SD 0.25; 0.1–3 per hour |
| Thermal bridging coefficient | Lognormal, initial log-SD 0.25; 0.001–0.5 W/m²K |
| Combustion/district efficiency | Scaled beta, initial concentration 45; bounded 0.1–1 |
| Heat-pump seasonal COP | Lognormal, initial log-SD 0.12; bounded 1–7 |
| Hot-water efficiency/COP | Equipment-dependent beta or lognormal; direct electric efficiency is 1 |
| Appliance, lighting and water demand factors | Lognormal, initial log-SDs 0.25, 0.20 and 0.22 |
| Geometry detail | Positive bounded factors coupled to the dwelling dimensions |
| Missing occupant/presence/equipment choices | Categorical probabilities |
| Pump/fan electrical power | Lognormal; 0.005–0.2 kW |
| Water temperature rise | Bounded normal, initial 37 ± 3 K; 20–55 K |
| Development monthly outdoor temperature | Bounded normal, initial SD 1.5°C |
| Development monthly solar irradiation | Lognormal, initial log-SD 0.15 |
| Battery usable fraction and efficiencies | Scaled beta, each bounded below 1 |

Application evidence can update the initial distributions while retaining their physical domains and heterogeneity. This is population-informed initialization. Meter calibration is a separate later operation.

### Step 4 — Generate 128 consistent complete inputs

The missing-input generator draws one instance of each unknown quantity for every realization. Named random streams make regeneration repeatable and preserve latent draws for unaffected variables across comparable scenarios.

The inputs are generated jointly where physics requires a dependency. The footprint follows total area and storeys; shared geometry factors determine envelope areas and volume. Appliance electricity and its retained internal heat use the same realization. Hot-water demand follows occupants, water use and water heating energy. Monthly demand/weather variation preserves temporal dependence; the monthly profile copula uses AR(1) correlation 0.65.

The prepared artifact contains both distribution summaries and the actual aligned joint draws. The simulator uses those draws, rather than independently sampling the summaries again.

### Step 5 — Calculate heat loss and useful heating with JAX

For each realization, the heat-loss coefficient is:

`H = Σ(U × exposed area) + thermal bridging coefficient × exposed area + 0.33 × ACH × volume`

`H` is in W/K. The monthly steady-state resistance calculation combines outdoor temperature, occupied heating hours, the setpoint, a 3 K setback balance, internal gains and solar gains. Useful heating is clipped at zero, with unused gains explicitly accounted for. This calculates monthly heat demand for the chosen household state.

### Step 6 — Convert useful heat into purchased energy

Space heating and hot water are calculated separately using their selected equipment efficiency or COP:

`purchased heating energy = useful heating / efficiency or COP`

For combustion, the balance includes useful heat and conversion loss. For a heat pump, the balance includes purchased electricity and heat collected from the environment. Cooking, boiler auxiliaries, secondary heating, electric showers and cylinder losses follow their respective end uses. Electric-shower demand is separated from the main hot-water load to avoid double counting.

Supported purchased carriers are **grid electricity, mains gas, oil, LPG, solid fuel and district heat**. The gas/fuel calculation returns energy in kWh, allocated by carrier and end use.

### Step 7 — Run half-hour electricity and storage dispatch

The realization's baseload, electric heating/hot water, auxiliaries and other electrical end uses are placed into half-hour profiles. The electricity engine balances demand, PV generation, direct solar consumption, grid import, export, battery charging/discharging, storage losses, curtailment and state of charge. Device power, usable-capacity and export limits apply.

PV reduces imported electricity. It does not erase the electrical demand being served. Battery state carries continuously through the simulated intervals.

### Step 8 — Check physical accounting for every realization

The wrapper validates thermal balances, per-fuel conversion, electrical end uses, interval dispatch, battery storage and monthly aggregation. It records the maximum residual for each checked identity in `max_conservation_error_kwh`.

### Step 9 — Summarize the 128 outcomes as distributions

For output draws `y₁…yₙ`, the reported mean is `Σy / n`; sample variance is `Σ(y − mean)² / (n − 1)`. SD is the square root of variance. The Monte Carlo standard error of the mean is `SD / √n`.

Each numeric distribution also carries median, mode, percentiles, 50% and 90% intervals, skewness, a histogram and sample count. Here the 90% range is P5–P95 of the prior predictive realizations. Monte Carlo standard error measures sampling precision of the mean; SD describes spread across the modeled household realizations.

### Step 10 — Return and save the result

The caller receives `EnsembleResult` plus aligned joint output samples. Preparation and simulation can be run separately: prepared inputs are hash-verified before replay, and results retain identity, seed, versions, input provenance and calibration status.

## 4. Case A — A baseline household: one realization and the final result

The first baseline realization shows the calculation concretely. Its setpoint is exactly 20°C. Sampled heating duration is 6.621636 hours/day, boiler efficiency is 0.869889 and hot-water efficiency is 0.947163.

| January 2025 calculation, realization 0 | Result |
|---|---:|
| Heat-loss coefficient | 313.09338 W/K |
| Heat loss before gains | 3,032.84619 kWh |
| Useful heat gains | 501.03958 kWh |
| Useful space heating | 2,531.80640 kWh |
| Purchased gas for space heating | 2,910.49341 kWh |
| Purchased gas for hot water | 183.13313 kWh |
| Gas cooking, 31 days × 1 kWh/day | 31.00000 kWh |
| Total gas | 3,124.62646 kWh |
| Grid electricity | 371.63287 kWh |

The path is visible in the arithmetic: `3,032.84619 − 501.03958 ≈ 2,531.80640` kWh of useful heating; `2,531.80640 / 0.869889 ≈ 2,910.49341` kWh of heating gas; and `2,910.49341 + 183.13313 + 31 ≈ 3,124.62646` kWh total gas. Small rounding differences come from displayed precision and numerical arithmetic.

This is one complete realization. The final annual estimate summarizes all 128 realizations:

| Baseline annual output | Mean, kWh | P5–P95, kWh |
|---|---:|---:|
| Useful space heating | 13,178.77 | 8,797.39–17,192.33 |
| Useful hot water | 2,143.63 | 1,431.35–2,914.89 |
| Purchased mains gas | 18,705.97 | 13,226.75–24,269.85 |
| Grid electricity import | 4,307.82 | 3,081.04–5,701.40 |
| All-fuel purchased energy | 23,013.79 | 17,843.19–28,973.26 |

Baseline gas variance is **11,689,846.14 kWh²**, its SD is **3,419.04 kWh**, and the Monte Carlo standard error of its mean is **302.20 kWh**. All-fuel purchased-energy variance is **11,588,959.41 kWh²**, with SD **3,404.26 kWh**.

For this household, all-fuel purchased energy equals grid import plus gas because oil, LPG, solid fuel and district heat are absent. PV generation and export are exactly zero because existing PV absence is explicit.

## 5. Case B — A user asks “What if I replace the gas boiler with a heat pump?”

The proposed change for this example is `heating_system = air_source_heat_pump`. The scenario starts from the baseline household and retains the same envelope, occupants, thermostat, cooking choice, seed, months and sample count. Heating equipment and its efficiency/COP distribution are prepared for the proposed state; the solver reruns that entire state.

The example's hot water follows the main heating system, so both main heating and hot-water energy move to electricity. The cooking appliance remains gas-fired. For an application comparison, use the same population/weather snapshot for both states and mark the new request as a what-if. Hypothetical answers stay outside the recorded baseline population.

| Annual output | Baseline mean, kWh | Heat-pump what-if mean, kWh |
|---|---:|---:|
| Useful space heating | 13,178.77 | 13,178.77 |
| Useful hot water | 2,143.63 | 2,143.63 |
| Purchased mains gas | 18,705.97 | 365.00 |
| Grid electricity import | 4,307.82 | 10,620.71 |
| All-fuel purchased energy | 23,013.79 | 10,985.71 |

The useful heat required by the home stays the same. The selected device changes the purchased energy needed to provide it. The remaining **365 kWh/year of gas** is exactly the supplied cooking assumption: `1 × 365`.

The final heat-pump grid-import distribution has mean **10,620.71 kWh**, variance **2,328,223.49 kWh²**, SD **1,525.85 kWh** and P5–P95 **8,412.88–13,320.72 kWh**. Total purchased energy has P5–P95 **8,777.88–13,685.72 kWh**.

Savings use matched draws: `saving[i] = baseline[i] − after[i]`. The mean gas saving is **18,340.97 kWh/year**, electricity saving is **−6,312.89 kWh/year**, and total purchased-energy saving is **12,028.08 kWh/year**. The negative electricity saving correctly expresses increased electricity use after electrification. Savings ranges come from the paired difference distribution, rather than subtracting unrelated percentile endpoints.

## 6. Case C — The retrofit engine submits a combined package

The retrofit caller describes the physical after-state. The example package specifies:

```json
{
  "wall_u": 0.30,
  "roof_u": 0.16,
  "window_u": 1.40,
  "heating_system": "air_source_heat_pump",
  "pv_kwp": 3.5,
  "battery_kwh": 5.0
}
```

The U-values are W/m²K; PV capacity is kWp; nominal battery capacity is kWh. These supplied values define the package being evaluated. Other unspecified quantities retain their appropriate distributions. The caller provides a coherent proposed state, including replacing any superseded exact equipment facts and updating ownership fields when present.

The twin computes the package as one state:

1. Improved fabric changes heat loss and useful heating.
2. The heat pump supplies the resulting useful heat and hot water.
3. Its electricity becomes part of household electrical demand.
4. PV is allocated against that demand at each half hour.
5. The battery shifts available solar energy subject to power, capacity and loss constraints.
6. The result is compared with the matched baseline by fuel.

These interactions are calculated by rerunning the model. Separate retrofit savings are never added together.

| Annual output | Baseline mean, kWh | Combined-package mean, kWh | Package P5–P95, kWh |
|---|---:|---:|---:|
| Useful space heating | 13,178.77 | 5,369.32 | 3,296.89–7,384.13 |
| Useful hot water | 2,143.63 | 2,143.63 | 1,431.35–2,914.89 |
| Purchased mains gas | 18,705.97 | 365.00 | 365.00–365.00 |
| Electricity demand | 4,307.82 | 7,447.02 | 5,929.95–9,224.55 |
| Grid electricity import | 4,307.82 | 4,602.06 | 3,141.88–6,492.91 |
| PV generation | 0.00 | 3,300.45 | 2,372.44–4,548.18 |
| Electricity export | 0.00 | 377.30 | 29.36–1,097.43 |
| All-fuel purchased energy | 23,013.79 | 4,967.06 | 3,506.88–6,857.91 |

Mean signed savings are **18,340.97 kWh/year gas**, **−294.23 kWh/year electricity**, and **18,046.73 kWh/year total purchased energy**. The total reduction is **78.42% of baseline mean purchased energy**, calculated as `100 × (23,013.791595 − 4,967.057247) / 23,013.791595`. This is a ratio of the displayed scenario means; a downstream distribution of percentage improvement uses matched sample-wise ratios.

The package reduces mean useful space-heating demand by **59.26%** and mean H from **304.81 to 172.65 W/K**, a **43.36%** reduction. Average PV self-consumption is **89.58%**, average electrical self-sufficiency is **38.79%**, and average peak import is **2.338 kW**. The package's purchased-energy variance is **1,050,328.35 kWh²**, with SD **1,024.86 kWh**.

The heat-pump-only scenario imports 10,620.71 kWh/year, while the full package imports 4,602.06 kWh/year. Fabric improvements, solar generation and storage therefore materially alter the electricity result after electrification. The separate fabric-only run below makes the interaction visible.

## 7. What the returned output contains

The saved baseline result contains **16 period-total distributions**, **60 monthly output series**, **7 element heat-loss distributions**, **7 corresponding element-share distributions** and **8 metric slots**. Complete calendar years additionally receive annual summaries; partial requested periods remain period totals.

| Output group | Content and use |
|---|---|
| Useful heat | Space heating and hot water, enabling comparison of thermal demand and equipment input |
| Purchased energy | Grid import, gas, oil, LPG, solid fuel and district heat; both grid-plus-gas and all-fuel totals |
| Fuel-by-end-use series | Space heating, hot water and other consumption by carrier |
| Thermal accounting | Heat loss, gains, unused gains, main/secondary heat, conversion losses and ambient heat |
| Electrical accounting | Demand, baseload, hot water, auxiliaries, secondary heat, outdoor loads and other electricity |
| Solar/storage accounting | Generation, direct consumption, charge/discharge, storage change/loss, export and curtailment |
| Envelope diagnostics | H, thermal bridges, walls, roof, floor, windows, doors, party walls and ventilation contributions |
| Electricity metrics | Peak import, 16:00–19:00 import share, PV self-consumption and electrical self-sufficiency |
| Traceability | Input/prior summaries, exact facts, sources, history/context audit, identity, seed, versions and conservation checks |
| Joint output samples | Aligned realizations for correlated downstream savings, carbon, bill and ranking calculations |

An undefined ratio, such as PV self-consumption when the household has no PV, has an explicit null status. Exact absence yields zero energy for that absent system.

Part 2 can use paired per-fuel draws with its sourced emissions factors and tariffs, and compute improvement potential and package rankings. Part 3 can consume energy intensity, property state and rating information for the Green Home Score. The property-value component can use the resulting scenario state and ratings. The scene component can use the estimated element contributions. Solar generation, import/export and self-consumption are physical inputs to downstream renewable scoring.

The returned physics distributions provide a traceable numerical basis for these parts. The same sample index should be preserved when downstream calculations combine dependent outputs.

## 8. Reproducible execution and artifact handoff

The functions used by the block are:

| Stage | Entry point |
|---|---|
| Standalone initial distributions | `build_priors(request)` in `greentwin/twin/priors.py` |
| Application history and input preparation | `prepare_application_inputs(request, n_samples, history=..., kind=...)` |
| Separate missing-input generation | `generate_missing_inputs(request, n_samples, prior=...)` |
| Deterministic ensemble simulation | `simulate_ensemble(prepared_inputs, batch_size=8)` |
| Signed baseline-minus-after summaries | `paired_differences(baseline_run, after_run)` |
| Prepared replay and saved results | `save_inputs`, `load_inputs` and `save_result` |

From this worktree, a customer request can be prepared and simulated separately:

```sh
.venv/bin/python scripts/run_uncertain_twin.py prepare --request household.json --samples 128 --output artifacts/household-inputs
.venv/bin/python scripts/run_uncertain_twin.py simulate --prepared artifacts/household-inputs --output artifacts/household-result
```

Prepare writes `request.json`, `priors.json`, `inputs.json`, `input_samples.npz` and `manifest.json`. The manifest records SHA-256 hashes. Simulation verifies them and uses the frozen draws. Its output contains `result.json`, `output_samples.npz` and `metrics.json`.

For a proposed physical change:

```sh
.venv/bin/python scripts/run_uncertain_twin.py prepare --request household.json --changes proposed-changes.json --samples 128 --output artifacts/proposed-inputs
.venv/bin/python scripts/run_uncertain_twin.py simulate --prepared artifacts/proposed-inputs --output artifacts/proposed-result
```

`--changes` is treated as a scenario, so it does not record proposed facts in baseline history. `--what-if` provides the same history behavior when passing an already modified request. For controlled comparisons, retain the same household, dates, seed, sample count and population/weather evidence; use named latent streams and compare aligned output draws. Each output directory is new and stays inside this worktree.

## 9. Evidence tables and complete output inventory

The tables below are extracted from the saved evaluation evidence and result artifacts. Means, variances, SDs and percentiles are reported directly from the simulations. Units are kWh for period energy, kWh² for energy variance, W/K for heat-loss coefficients, kW for peak power and percent for ratios. Display rounding does not change the stored full-precision values.

### 9.1 Scenario overview

![Purchased-energy means and 90% prior predictive ranges](../eval/reports/prior_predictive_v1/scenario_energy.png)

| Scenario | Gas mean | Grid-import mean | Purchased-energy mean | Purchased-energy P5–P95 | Measured runtime |
|---|---:|---:|---:|---:|---:|
| Baseline household | 18,705.97 | 4,307.82 | 23,013.79 | 17,843.19–28,973.26 | 14.41 s |
| Fabric-only comparison | 9,351.96 | 4,307.82 | 13,659.78 | 11,264.69–16,434.75 | 13.35 s |
| Heat-pump what-if | 365.00 | 10,620.71 | 10,985.71 | 8,777.88–13,685.72 | 16.74 s |
| Combined retrofit package | 365.00 | 4,602.06 | 4,967.06 | 3,506.88–6,857.91 | 12.66 s |

Runtimes describe the saved local runs at 128 realizations and batch size 8, including their preparation/dispatch work within the simulation wrapper and any JAX compilation incurred there. They are measured run durations, rather than a promised service latency.

### 9.2 Full period-total distribution summaries

#### Baseline household

| Output | Mean | Variance | SD | P5 | P95 | MCSE |
|---|---:|---:|---:|---:|---:|---:|
| `all_fuel_purchased_kwh` | 23,013.79 | 11,588,959.41 | 3,404.26 | 17,843.19 | 28,973.26 | 300.90 |
| `battery_final_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `battery_initial_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `battery_usable_capacity_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `district_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `electricity_demand_kwh` | 4,307.82 | 853,749.35 | 923.99 | 3,081.04 | 5,701.40 | 81.67 |
| `electricity_import_kwh` | 4,307.82 | 853,749.35 | 923.99 | 3,081.04 | 5,701.40 | 81.67 |
| `export_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `gas_kwh` | 18,705.97 | 11,689,846.14 | 3,419.04 | 13,226.75 | 24,269.85 | 302.20 |
| `hot_water_kwh` | 2,143.63 | 280,194.95 | 529.33 | 1,431.35 | 2,914.89 | 46.79 |
| `lpg_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `oil_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `purchased_electricity_gas_kwh` | 23,013.79 | 11,588,959.41 | 3,404.26 | 17,843.19 | 28,973.26 | 300.90 |
| `pv_generation_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `space_heating_kwh` | 13,178.77 | 7,486,066.83 | 2,736.07 | 8,797.39 | 17,192.33 | 241.84 |

#### Fabric-only comparison

| Output | Mean | Variance | SD | P5 | P95 | MCSE |
|---|---:|---:|---:|---:|---:|---:|
| `all_fuel_purchased_kwh` | 13,659.78 | 2,955,395.39 | 1,719.13 | 11,264.69 | 16,434.75 | 151.95 |
| `battery_final_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `battery_initial_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `battery_usable_capacity_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `district_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `electricity_demand_kwh` | 4,307.82 | 853,749.35 | 923.99 | 3,081.04 | 5,701.40 | 81.67 |
| `electricity_import_kwh` | 4,307.82 | 853,749.35 | 923.99 | 3,081.04 | 5,701.40 | 81.67 |
| `export_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `gas_kwh` | 9,351.96 | 2,851,433.04 | 1,688.62 | 6,899.85 | 12,037.05 | 149.25 |
| `hot_water_kwh` | 2,143.63 | 280,194.95 | 529.33 | 1,431.35 | 2,914.89 | 46.79 |
| `lpg_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `oil_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `purchased_electricity_gas_kwh` | 13,659.78 | 2,955,395.39 | 1,719.13 | 11,264.69 | 16,434.75 | 151.95 |
| `pv_generation_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `space_heating_kwh` | 5,369.32 | 1,623,045.40 | 1,273.99 | 3,296.89 | 7,384.13 | 112.61 |

#### Heat-pump what-if

| Output | Mean | Variance | SD | P5 | P95 | MCSE |
|---|---:|---:|---:|---:|---:|---:|
| `all_fuel_purchased_kwh` | 10,985.71 | 2,328,223.63 | 1,525.85 | 8,777.88 | 13,685.72 | 134.87 |
| `battery_final_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `battery_initial_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `battery_usable_capacity_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `district_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `electricity_demand_kwh` | 10,620.71 | 2,328,223.49 | 1,525.85 | 8,412.88 | 13,320.72 | 134.87 |
| `electricity_import_kwh` | 10,620.71 | 2,328,223.49 | 1,525.85 | 8,412.88 | 13,320.72 | 134.87 |
| `export_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `gas_kwh` | 365.00 | 0.00 | 0.00 | 365.00 | 365.00 | 0.00 |
| `hot_water_kwh` | 2,143.63 | 280,194.95 | 529.33 | 1,431.35 | 2,914.89 | 46.79 |
| `lpg_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `oil_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `purchased_electricity_gas_kwh` | 10,985.71 | 2,328,223.63 | 1,525.85 | 8,777.88 | 13,685.72 | 134.87 |
| `pv_generation_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `space_heating_kwh` | 13,178.77 | 7,486,066.83 | 2,736.07 | 8,797.39 | 17,192.33 | 241.84 |

#### Combined retrofit package

| Output | Mean | Variance | SD | P5 | P95 | MCSE |
|---|---:|---:|---:|---:|---:|---:|
| `all_fuel_purchased_kwh` | 4,967.06 | 1,050,328.35 | 1,024.86 | 3,506.88 | 6,857.91 | 90.59 |
| `battery_final_kwh` | -0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `battery_initial_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `battery_usable_capacity_kwh` | 4.50 | 0.01 | 0.09 | 4.36 | 4.65 | 0.01 |
| `district_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `electricity_demand_kwh` | 7,447.02 | 1,067,675.91 | 1,033.28 | 5,929.95 | 9,224.55 | 91.33 |
| `electricity_import_kwh` | 4,602.06 | 1,050,328.34 | 1,024.86 | 3,141.88 | 6,492.91 | 90.59 |
| `export_kwh` | 377.30 | 92,096.07 | 303.47 | 29.36 | 1,097.43 | 26.82 |
| `gas_kwh` | 365.00 | 0.00 | 0.00 | 365.00 | 365.00 | 0.00 |
| `hot_water_kwh` | 2,143.63 | 280,194.95 | 529.33 | 1,431.35 | 2,914.89 | 46.79 |
| `lpg_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `oil_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `purchased_electricity_gas_kwh` | 4,967.06 | 1,050,328.35 | 1,024.86 | 3,506.88 | 6,857.91 | 90.59 |
| `pv_generation_kwh` | 3,300.45 | 368,513.13 | 607.05 | 2,372.44 | 4,548.18 | 53.66 |
| `solid_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `space_heating_kwh` | 5,369.32 | 1,623,045.40 | 1,273.99 | 3,296.89 | 7,384.13 | 112.61 |

### 9.3 Paired savings distributions

Positive means a reduction; negative means an increase. All rows use baseline minus the scenario for matched realizations.

#### Fabric-only comparison

| Carrier / purchased-energy convention | Mean saving | Variance | SD | P5 saving | P95 saving |
|---|---:|---:|---:|---:|---:|
| `electricity_import_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `gas_kwh` | 9,354.01 | 6,416,256.49 | 2,533.03 | 6,010.84 | 13,918.47 |
| `oil_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `lpg_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `district_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `purchased_electricity_gas_kwh` | 9,354.01 | 6,416,256.63 | 2,533.03 | 6,010.84 | 13,918.47 |
| `all_fuel_purchased_kwh` | 9,354.01 | 6,416,256.63 | 2,533.03 | 6,010.84 | 13,918.47 |

#### Heat-pump what-if

| Carrier / purchased-energy convention | Mean saving | Variance | SD | P5 saving | P95 saving |
|---|---:|---:|---:|---:|---:|
| `electricity_import_kwh` | -6,312.89 | 1,733,493.59 | 1,316.62 | -8,504.59 | -4,323.48 |
| `gas_kwh` | 18,340.97 | 11,689,846.14 | 3,419.04 | 12,861.75 | 23,904.85 |
| `oil_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `lpg_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `district_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `purchased_electricity_gas_kwh` | 12,028.08 | 4,763,980.17 | 2,182.65 | 8,561.16 | 15,395.73 |
| `all_fuel_purchased_kwh` | 12,028.08 | 4,763,980.17 | 2,182.65 | 8,561.16 | 15,395.73 |

#### Combined retrofit package

| Carrier / purchased-energy convention | Mean saving | Variance | SD | P5 saving | P95 saving |
|---|---:|---:|---:|---:|---:|
| `electricity_import_kwh` | -294.23 | 618,351.87 | 786.35 | -1,537.50 | 799.11 |
| `gas_kwh` | 18,340.97 | 11,689,846.14 | 3,419.04 | 12,861.75 | 23,904.85 |
| `oil_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `lpg_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `solid_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `district_kwh` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `purchased_electricity_gas_kwh` | 18,046.73 | 9,139,442.22 | 3,023.15 | 13,379.00 | 23,400.64 |
| `all_fuel_purchased_kwh` | 18,046.73 | 9,139,442.22 | 3,023.15 | 13,379.00 | 23,400.64 |

### 9.4 All metric summaries

| Scenario | Metric | Mean | SD | P5 | P95 |
|---|---|---:|---:|---:|---:|
| Baseline household | `peak_import_kw` | 1.4572 | 0.3138 | 1.0377 | 1.9624 |
| Baseline household | `peak_import_share_pct` | 21.4795 | 0.5197 | 20.6459 | 22.3564 |
| Baseline household | `self_consumption_pct` | null: no PV denominator | — | — | — |
| Baseline household | `self_sufficiency_pct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Baseline household | `battery_initial_kwh` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Baseline household | `battery_final_kwh` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Baseline household | `heat_loss_coefficient_w_per_k` | 304.8083 | 33.5412 | 252.6221 | 364.2975 |
| Baseline household | `thermal_bridging_w_per_k` | 29.0920 | 7.6119 | 19.4327 | 41.8808 |
| Fabric-only comparison | `peak_import_kw` | 1.4572 | 0.3138 | 1.0377 | 1.9624 |
| Fabric-only comparison | `peak_import_share_pct` | 21.4795 | 0.5197 | 20.6459 | 22.3564 |
| Fabric-only comparison | `self_consumption_pct` | null: no PV denominator | — | — | — |
| Fabric-only comparison | `self_sufficiency_pct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fabric-only comparison | `battery_initial_kwh` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fabric-only comparison | `battery_final_kwh` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fabric-only comparison | `heat_loss_coefficient_w_per_k` | 172.6477 | 15.2500 | 148.7663 | 197.7367 |
| Fabric-only comparison | `thermal_bridging_w_per_k` | 29.0920 | 7.6119 | 19.4327 | 41.8808 |
| Heat-pump what-if | `peak_import_kw` | 3.2357 | 0.4468 | 2.6232 | 3.9486 |
| Heat-pump what-if | `peak_import_share_pct` | 16.8048 | 0.6393 | 15.7798 | 17.7948 |
| Heat-pump what-if | `self_consumption_pct` | null: no PV denominator | — | — | — |
| Heat-pump what-if | `self_sufficiency_pct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Heat-pump what-if | `battery_initial_kwh` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Heat-pump what-if | `battery_final_kwh` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Heat-pump what-if | `heat_loss_coefficient_w_per_k` | 304.8083 | 33.5412 | 252.6221 | 364.2975 |
| Heat-pump what-if | `thermal_bridging_w_per_k` | 29.0920 | 7.6119 | 19.4327 | 41.8808 |
| Combined retrofit package | `peak_import_kw` | 2.3378 | 0.3145 | 1.8556 | 2.8697 |
| Combined retrofit package | `peak_import_share_pct` | 15.4041 | 1.6466 | 13.2731 | 18.2586 |
| Combined retrofit package | `self_consumption_pct` | 89.5817 | 6.7266 | 76.1772 | 98.9297 |
| Combined retrofit package | `self_sufficiency_pct` | 38.7856 | 6.4801 | 28.8989 | 49.6439 |
| Combined retrofit package | `battery_initial_kwh` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Combined retrofit package | `battery_final_kwh` | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Combined retrofit package | `heat_loss_coefficient_w_per_k` | 172.6477 | 15.2500 | 148.7663 | 197.7367 |
| Combined retrofit package | `thermal_bridging_w_per_k` | 29.0920 | 7.6119 | 19.4327 | 41.8808 |

Self-consumption and self-sufficiency are means of realization-wise percentages. They need not equal a ratio formed from the mean energies. Battery usable capacity, initial/final storage and storage losses are available alongside these metrics.

### 9.5 Physics, repeatability and uncertainty evidence

| Check | Measured result |
|---|---:|
| Fabric realizations with useful heating no higher than baseline | 128/128 |
| Identical regenerated parameter arrays for the same request and seed | True |
| Largest checked residual across four annual scenarios | 0.00327297 kWh |
| H SD before seven explicit envelope measurements | 33.54120 W/K |
| H SD after those measurements | 3.93676 W/K |
| Reduction in H sample SD in that experiment | 88.2629% |

The seven exact envelope inputs in the information experiment are wall, roof, floor, window and door U-values, ventilation ACH and thermal bridging. It demonstrates how additional measured facts narrow the modeled uncertainty in H. Conservation residuals measure numerical accounting consistency. These numerical checks and scenario estimates are simulation evidence.

#### Sampling precision

![Gas distributions and Monte Carlo sampling precision](../eval/reports/prior_predictive_v1/uncertainty_checks.png)

| Baseline draws included | Gas sample mean, kWh | MCSE of mean, kWh |
|---|---:|---:|
| 16 | 18,743.01 | 865.07 |
| 32 | 19,083.27 | 699.97 |
| 64 | 19,111.28 | 446.36 |
| 128 | 18,705.97 | 302.20 |

The means above use prefixes of the same saved ensemble. Increasing the draw count improves sampling precision; the final customer range continues to represent uncertainty in the modeled inputs.

#### Six-carrier conversion checks

Each independent check uses eight January realizations with the same home and 1 kWh/day gas cooking. A non-gas heating system therefore retains 31 kWh of January gas cooking.

| Heating carrier | Mean all-fuel purchased energy, January kWh | Maximum heating-conversion residual, kWh | Mean January gas, kWh |
|---|---:|---:|---:|
| mains_gas | 3,695.4989 | 0.00074111 | 3,347.0975 |
| oil | 3,867.3282 | 0.00066172 | 31.0000 |
| lpg | 3,695.4989 | 0.00074111 | 31.0000 |
| solid | 5,081.9278 | 0.00086164 | 31.0000 |
| district | 3,467.0948 | 0.00064432 | 31.0000 |
| electricity | 1,509.9398 | 0.00006691 | 31.0000 |

### 9.6 Complete monthly-series inventory

Each of these 60 fields has one output distribution per requested month. For a full year, that is 720 monthly distribution summaries per scenario. The complete numerical summaries are retained in the linked result JSON files.

| Group | Exact output keys |
|---|---|
| Fuel and purchased energy | `monthly_all_fuel_purchased_kwh` |
| Fuel and purchased energy | `monthly_district_hot_water_fuel_kwh` |
| Fuel and purchased energy | `monthly_district_kwh` |
| Fuel and purchased energy | `monthly_district_other_fuel_kwh` |
| Fuel and purchased energy | `monthly_district_space_heating_fuel_kwh` |
| Fuel and purchased energy | `monthly_electricity_hot_water_fuel_kwh` |
| Fuel and purchased energy | `monthly_electricity_other_fuel_kwh` |
| Fuel and purchased energy | `monthly_electricity_space_heating_fuel_kwh` |
| Fuel and purchased energy | `monthly_extra_gas_kwh` |
| Fuel and purchased energy | `monthly_gas_kwh` |
| Fuel and purchased energy | `monthly_heating_fuel_kwh` |
| Fuel and purchased energy | `monthly_hot_water_fuel_kwh` |
| Fuel and purchased energy | `monthly_lpg_hot_water_fuel_kwh` |
| Fuel and purchased energy | `monthly_lpg_kwh` |
| Fuel and purchased energy | `monthly_lpg_other_fuel_kwh` |
| Fuel and purchased energy | `monthly_lpg_space_heating_fuel_kwh` |
| Fuel and purchased energy | `monthly_mains_gas_hot_water_fuel_kwh` |
| Fuel and purchased energy | `monthly_mains_gas_other_fuel_kwh` |
| Fuel and purchased energy | `monthly_mains_gas_space_heating_fuel_kwh` |
| Fuel and purchased energy | `monthly_oil_hot_water_fuel_kwh` |
| Fuel and purchased energy | `monthly_oil_kwh` |
| Fuel and purchased energy | `monthly_oil_other_fuel_kwh` |
| Fuel and purchased energy | `monthly_oil_space_heating_fuel_kwh` |
| Fuel and purchased energy | `monthly_purchased_electricity_gas_kwh` |
| Fuel and purchased energy | `monthly_solid_hot_water_fuel_kwh` |
| Fuel and purchased energy | `monthly_solid_kwh` |
| Fuel and purchased energy | `monthly_solid_other_fuel_kwh` |
| Fuel and purchased energy | `monthly_solid_space_heating_fuel_kwh` |
| Other monthly diagnostics | `monthly_auxiliary_electricity_kwh` |
| Other monthly diagnostics | `monthly_baseload_kwh` |
| Other monthly diagnostics | `monthly_battery_charge_kwh` |
| Other monthly diagnostics | `monthly_battery_delta_kwh` |
| Other monthly diagnostics | `monthly_battery_discharge_kwh` |
| Other monthly diagnostics | `monthly_battery_loss_kwh` |
| Other monthly diagnostics | `monthly_curtailment_kwh` |
| Other monthly diagnostics | `monthly_direct_pv_kwh` |
| Other monthly diagnostics | `monthly_electricity_demand_kwh` |
| Other monthly diagnostics | `monthly_electricity_kwh` |
| Other monthly diagnostics | `monthly_export_kwh` |
| Other monthly diagnostics | `monthly_heat_gains_kwh` |
| Other monthly diagnostics | `monthly_heat_loss_kwh` |
| Other monthly diagnostics | `monthly_heating_ambient_heat_kwh` |
| Other monthly diagnostics | `monthly_heating_auxiliary_electricity_kwh` |
| Other monthly diagnostics | `monthly_heating_conversion_loss_kwh` |
| Other monthly diagnostics | `monthly_hot_water_ambient_heat_kwh` |
| Other monthly diagnostics | `monthly_hot_water_conversion_loss_kwh` |
| Other monthly diagnostics | `monthly_hot_water_kwh` |
| Other monthly diagnostics | `monthly_hot_water_system_loss_kwh` |
| Other monthly diagnostics | `monthly_internal_gains_kwh` |
| Other monthly diagnostics | `monthly_main_hot_water_kwh` |
| Other monthly diagnostics | `monthly_main_space_heating_kwh` |
| Other monthly diagnostics | `monthly_outdoor_other_electricity_kwh` |
| Other monthly diagnostics | `monthly_pv_generation_kwh` |
| Other monthly diagnostics | `monthly_secondary_electricity_kwh` |
| Other monthly diagnostics | `monthly_secondary_unused_request_kwh` |
| Other monthly diagnostics | `monthly_shower_electricity_kwh` |
| Other monthly diagnostics | `monthly_solar_gains_kwh` |
| Other monthly diagnostics | `monthly_space_heating_kwh` |
| Other monthly diagnostics | `monthly_thermal_storage_delta_kwh` |
| Other monthly diagnostics | `monthly_unused_heat_gains_kwh` |

### 9.7 Building-element diagnostics

| Element | Baseline mean H contribution, W/K | Baseline mean share, % |
|---|---:|---:|
| doors | 11.1206 | 3.6773 |
| floor | 36.1994 | 11.9633 |
| party_walls | 0.0000 | 0.0000 |
| roof | 25.6221 | 8.4724 |
| ventilation | 50.0837 | 16.4751 |
| walls | 138.4099 | 45.1303 |
| windows | 43.3727 | 14.2816 |

These contributions are estimates from the simulated building state. Their joint draws allow the scene to explain which elements contribute to heat loss.

### 9.8 Full-precision evidence and replay references

Evidence SHA-256: `7658a58c0c7dbf14d468559421c879a93fe214e882633ed31aee9c149d755973`.

- [Complete measured evaluation evidence](../eval/reports/prior_predictive_v1/evidence.json)
- [Original prepared request](../eval/reports/prior_predictive_v1/prepared/request.json)
- [Prepared priors](../eval/reports/prior_predictive_v1/prepared/priors.json)
- [Prepared input distributions](../eval/reports/prior_predictive_v1/prepared/inputs.json)
- [Prepared joint inputs](../eval/reports/prior_predictive_v1/prepared/input_samples.npz)
- [Prepared hash manifest](../eval/reports/prior_predictive_v1/prepared/manifest.json)
- [Baseline household: full output distributions](../eval/reports/prior_predictive_v1/baseline/result.json)
- [Baseline household: aligned output samples](../eval/reports/prior_predictive_v1/baseline/output_samples.npz)
- [Fabric-only comparison: full output distributions](../eval/reports/prior_predictive_v1/fabric/result.json)
- [Fabric-only comparison: aligned output samples](../eval/reports/prior_predictive_v1/fabric/output_samples.npz)
- [Heat-pump what-if: full output distributions](../eval/reports/prior_predictive_v1/heat_pump/result.json)
- [Heat-pump what-if: aligned output samples](../eval/reports/prior_predictive_v1/heat_pump/output_samples.npz)
- [Combined retrofit package: full output distributions](../eval/reports/prior_predictive_v1/fabric_heat_pump_pv_battery/result.json)
- [Combined retrofit package: aligned output samples](../eval/reports/prior_predictive_v1/fabric_heat_pump_pv_battery/output_samples.npz)
- [Known-envelope experiment](../eval/reports/prior_predictive_v1/known_envelope/result.json)
- [Application-history distribution method](APPLICATION_HISTORY_PRIORS.md)

The evidence files retain the complete distribution summaries, histograms, percentiles, monthly values and conservation checks beyond the displayed rounded tables. This document and those artifacts together make the baseline, what-if and retrofit examples inspectable and reproducible.
