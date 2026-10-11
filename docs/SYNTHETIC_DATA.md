# Physics-backed synthetic data generator

Implements the generator described in `SYSTEM_DESIGN.pdf` sections 6–7 and
`docs/IMPLEMENTATION_PLAN_v2.md` block 2b. All changes live in the Physics & Bayesian
worktree. Geometry/parameter preparation is reused from Satyarth's commit
`3647772e197d55ca0d71c8ab9b426f2b84342312`; no other branch was fetched, switched, or modified.

## Build order and ownership

The generator samples homes and household behaviour; it does not implement a JAX
engine or an electricity solver. Physics and Bayesian calibration are owned by the
Physics & Bayesian track. Satyarth supplies geometry, user-input preparation, and
integration. There is no automatic physics stub. `--inputs-only` works before the
simulator exists. Meter readings and retrofit labels require an injected simulator.

Only the minimal generator dependencies are needed: Python 3.12, numpy, pydantic;
scipy/pytest are used by the repository tests. JAX is needed by your future simulator,
not by the sampler.

## Run now

From this worktree:

```bash
.venv/bin/python scripts/generate_synthetic.py --demo --inputs-only --homes 100 --months 24
```

`--demo` explicitly uses SH1–SH3 and synthetic typical weather. This is development
data, not a representative UK population. Variants of three showcase homes cannot
support a meaningful held-out population accuracy claim. Existing outputs are never
overwritten; choose a different `--output` inside this worktree for another run.

Use mapped public EPC and weather records for production-shaped inputs:

```bash
.venv/bin/python scripts/generate_synthetic.py \
  --properties data/processed/epc_properties.jsonl \
  --weather data/processed/weather_months.jsonl \
  --inputs-only --data-version epc-weather-v1 \
  --output data/processed/synthetic/epc-weather-v1
```

Each JSONL line validates against `Property` or `WeatherMonth`. JSON lists are also
accepted. Production mode requires `source="epc_real"`, sourced temperature and
irradiation, and complete location/month weather coverage. No network calls are
performed by the generator. Supply source licences and releases through the data
layer's `DatasetMeta`; `data_version` identifies that prepared input release.

## Random generation rules

1. Sample complete source EPC records, keeping their joint type/age/fabric/equipment
   combinations. Optional `--area-stats` rakes the source mix to band and property-type
   marginals per LA. Supply one comparable lodgement period per LA; these are not
   dwelling-stock counts. Missing target categories cause an error.
2. Use separate seeded random streams for source selection, physical truth, and
   observations. Changing meter noise or missingness cannot change clean physics.
3. Sample occupancy conditionally on floor area/bedrooms; select mostly-away,
   evening, or mostly-home behaviour. Setpoints and heating hours use bounded
   rejection-sampled normals. Household properties persist across all months.
4. Sample heat-loss multiplier, appliance demand and water use with positive
   distributions. Boiler efficiency is beta-bounded below 1, heat-pump performance
   is bounded separately, and resistance heating uses efficiency 1.
5. Hot-water useful heat follows litres × water heat capacity × temperature rise.
   Appliance internal gains follow sampled baseload; they are not sampled independently.
6. Monthly demand multipliers are positive correlated seasonal deviations normalised
   to mean 1. The simulator receives them as explicit runtime arrays.
7. Use real location/month weather intact. The inputs-only demo fallback is synthetic,
   labelled, and never described as observed historical weather.
8. Missing fields and corrupted EPC-band observations are applied only to the observed
   home. Clean truth is preserved. Gas absent from a home is a structural zero.
9. Meter readings use mean-one lognormal multiplicative noise; estimated readings
   have wider noise. Missing readings become null, including blocks of missing months.
   Clean PV/export sensor values are stored separately from noisy import readings.

All distribution coefficients are **development assumptions, not NEED-fitted priors**.
Defaults imported from Satyarth include `to_verify` engineering approximations. Real
EPC/weather input does not by itself make those coefficients empirically calibrated.
Before submission, fit/check these against NEED and EHS and verify defaults against
their published specifications. Potential EPC fields are removed from inputs.

## Connect the thermal simulator

The optional half-hour electricity integration is now available; see
[ELECTRICITY_PHYSICS.md](ELECTRICITY_PHYSICS.md). Use model version
`thermal-electricity/0.2.0` and `--electricity-development` for labelled generated
profiles in demo mode. Bayesian calibration remains deferred.

Phase 1's JAX engine is now available at `greentwin.twin.physics:run_physics`.
Use model version `thermal-rc/0.1.0`; see [THERMAL_PHYSICS.md](THERMAL_PHYSICS.md)
for assumptions and a complete demo command. It generates uncalibrated thermal
truth and basic fuel accounting. PV/battery homes require the next electricity
phase and are rejected by this engine. The generator remains independent of JAX
when `--inputs-only` is used.

```bash
.venv/bin/python scripts/generate_synthetic.py --properties data/processed/epc_properties.jsonl \
  --weather data/processed/weather_months.jsonl \
  --simulator greentwin.twin.physics:run_physics --simulator-version physics-l1-v1 \
  --data-version epc-weather-v1 --output data/processed/synthetic/energy-v1
```

The interface is Satyarth's `run_physics(params, weather, mode, fuel)`. Numeric
`TwinParams` fields are passed under the same names; categorical equipment uses
static `FuelConfig`. It adds independent `hot_water_efficiency` to avoid applying a
space-heating COP to immersion hot water. Weather contains monthly arrays plus
`baseload_multiplier` and `hot_water_multiplier`, which the solver must honour.
No random sampling happens inside the solver.

Return Satyarth's six period totals and monthly electricity/gas, plus monthly space
heat, hot water, PV and export using the corresponding `monthly_*_kwh` keys. The
generator also requires these energy-balance diagnostics:

| Key | Meaning |
| --- | --- |
| `monthly_electricity_demand_kwh` | End-use electricity before PV/storage |
| `monthly_battery_loss_kwh` | Nonnegative storage losses |
| `monthly_battery_delta_kwh` | Signed end-minus-start stored energy; initial state is empty |
| `monthly_heat_loss_kwh` | Net heat leaving the modelled thermal system |
| `monthly_heat_gains_kwh` | Internal and solar heat entering that system |
| `monthly_unused_heat_gains_kwh` | Gains rejected/removed rather than stored |
| `monthly_thermal_storage_delta_kwh` | Optional signed thermal storage change; zero for steady state |
| `monthly_auxiliary_electricity_kwh` | Optional pumps/controls electricity |
| `monthly_oil_kwh`, `monthly_lpg_kwh`, etc. | Required when those fuels supply heat; preserved in truth |

The thermal system boundary and losses/gains must be consistent with the simulator.
Each monthly array has length `n_months`; totals describe the simulated period, not
implicitly kWh/year. Return all seven element heat-loss coefficients in W/K.

The generator rejects nonfinite/negative outputs, wrong shapes, totals inconsistent
with monthly sums, invalid fuel conversion, mismatched hot-water/end-use demand,
generation without PV, and these balance failures:

```
grid import + PV = electricity demand + export + battery losses + storage change
delivered space heat + heat gains = heat loss + unused gains + thermal storage change
```

Battery monthly endpoints must stay within capacity. Your electricity solver must
also validate **timestep** SOC and power limits: monthly balances alone cannot certify
them. Likewise these checks enforce accounting identities, not the accuracy of heat
transfer equations. The generator does not substitute for physics-model validation.

## Retrofit labels

Supply `--scenario-factory package.module:function` or the Python `scenario_factory`
argument. It receives `(true_property, true_params, fuel)` and returns
`RetrofitScenario(name, retrofit_ids, params, fuel, all_feasible=False)` objects from
the feasibility/catalogue owner. It must apply complete packages, rule vetoes,
dependencies and mutual exclusions before returning them. The generator does not
invent feasible packages or implement the retrofit engine.

Baseline and every package rerun use identical weather and household multipliers.
Savings = baseline − after per fuel; electricity increases remain negative savings.
The label unit is `kWh/simulated-period`, explicitly avoiding annualising a partial
year. Set `all_feasible=True` only for the full compatible feasible package; that adds
the purchased-energy improvement-potential label. Zero baseline or unsupported fuels
yield a null percentage. Latent rebound is saved for explicitly defined future
behaviour scenarios; it is not silently applied to retrofit labels.

No synthetic value premium or value evidence is produced. That belongs to the value
model's separate method-check dataset.

## Output and Bayesian evaluation

- `properties.jsonl`: observed `Property` records (training/calibration inputs).
- `usage.jsonl`: observed `EnergyUsageMonth` records; empty in inputs-only mode.
- `reference_labels.jsonl`: exact generator-rule labels; empty without scenarios.
- `truth.jsonl`: hidden parameters, clean physical runs, household profiles, weather,
  source-field bases and provenance. **Never feed this file as calibration features.**
- `meta.json`: config, versions, limitations, counts and output hashes.
- `metrics.json`: seed, data version, split counts, conservation check status.

Source customers (else source properties) get one seeded 70/15/15 split, respecting
existing assignments. Every synthetic variant stays in its source group. Conflicting
source split assignments are rejected. Split fractions are expectations over source
groups, not forced row counts.

Synthetic truth is a known deterministic generator draw, so it is stored as exact
values. Bayesian predictions are separate posterior `Distribution`s built with
`contracts.summarise`, not copied from truth or assigned arbitrary intervals.

Evaluate identifiable parameter recovery, held-out-month predictions, artificially
masked-month imputation, posterior-predictive coverage and calibration diagnostics.
Compare annual fuel distributions by type/age/region with real NEED and stock mixes
with comparable EPC live tables. Use independent/model-mismatch cases too. The
manifest deliberately records `empirical_validation="not_run"`; passing synthetic
conservation tests is not evidence of real-world prediction accuracy.
