# Phase 1: JAX steady-state thermal twin

Phase 1 is now extended by the optional [electricity layer](ELECTRICITY_PHYSICS.md).
The standalone thermal entry point still rejects PV/batteries without prepared
electricity inputs. The combined model is `thermal-electricity/0.2.0`; the version
below records the original thermal-only build.

All code lives in the `PHYSICS MODEL AND BAYESIAN` worktree. This implements the
monthly Level-1 thermal core in `SYSTEM_DESIGN.pdf` and implementation plan block
2a. Bayesian calibration, half-hour electricity profiles, PV, battery dispatch,
tariffs, retrofit catalogue translation and standard-occupancy/SAP preparation
remain later work. No posterior, fitted accuracy or EPC rating is produced.

## Run

```bash
.venv/bin/python scripts/run_thermal_twin.py --demo SH1
.venv/bin/python scripts/run_thermal_twin.py --demo SH3
.venv/bin/python scripts/generate_synthetic.py --demo --homes 100 --months 24 \
  --simulator greentwin.twin.physics:run_physics --simulator-version thermal-rc/0.1.0 \
  --data-version thermal-development-v1 \
  --output data/processed/synthetic/thermal-development-v1
```

Demo homes use provisional fabric defaults and explicitly synthetic weather.
For a real prepared home, use `--property home.json --weather weather.jsonl`.
The JSON record follows `contracts.models.Property`; weather follows
`WeatherMonth` and must match its postcode district. Missing temperature or
irradiation is rejected: fill and label in the data layer before calling this
boundary. An optional `--output` writes a new file inside this worktree, never
overwriting a previous result. Default output is JSON on stdout.

JAX is already declared in `pyproject.toml` and is installed in this worktree's
Python 3.12 virtual environment. The input-only sampler can still run without JAX.

## Equations and assumptions

The model is one well-mixed thermal zone with a parallel resistance network.
The steady-state limit of `C dT/dt = Q_heat + Q_gain - H(T_inside-T_outside)`
sets the storage term to zero; it does not simulate thermal mass or warm-up.
There is no artificial capacitance or ODE engine in this phase.

For each exposed element, `H_element = multiplier * area * (U + y)` in W/K.
The bridging allowance `y*area` is allocated to that element for the existing
seven-element output contract, and separately reported as a diagnostic.
The roof and floor areas are **exposed envelope areas**, not total habitable
floor area. Intermediate flats can have zero exposed floor/roof. Opaque wall
area excludes windows and doors. Party walls are excluded from the exposed
bridging area.

Ventilation adds `multiplier * 0.33 * ACH * volume` W/K. This is a fixed air-change
approximation, with no wind, humidity, heat recovery or detailed infiltration
model. Party walls default to an equal-temperature neighbour, giving zero net
heat loss. `--party-wall-u` can supply an explicit effective outdoor-equivalent
bypass coefficient; this is not a model of neighbour temperature.

These U*A, bridging and ventilation relationships follow the heat-transfer
coefficient structure in [BRE SAP 2012](https://bregroup.com/documents/d/bre-group/sap-2012_9-92).
The monthly schedule and gain treatment below are simplified development
assumptions, **not a complete implementation of SAP**.

Each month has two steady-state controlled periods: the supplied setpoint for
`heating_hours_per_day`, and a setback of `setpoint - setback_delta_c` for the
remaining hours. The default setback is an explicit, unverified **3 K assumption**,
configurable with `ThermalAssumptions` or `--setback-delta-c`. Zero heating hours
means maintaining the setback all day; it does **not** mean heating is switched
off. This approximation avoids multiplying continuous heating energy by a duty
fraction. It still cannot reproduce the cooling and reheating of intermittent
heating without the future dynamic RC model.

For each period, gross loss is `H * max(T_control - T_outdoor_month_mean, 0) * hours / 1000`.
Thus the solver uses monthly-mean temperature degree days at the actual controlled
temperature. The input's HDD15.5 is not used: combining that already referenced
index with a second gain subtraction can miscount heat. Monthly means omit
within-month cold spells, so warm shoulder-season heating can be underestimated.
Actual-weather calibration later must account for this model discrepancy.

Internal gains are the supplied W multiplied by hours. Monthly baseload variation
changes the retained appliance component (assumed 85% retained as heat), leaving
the remaining supplied occupant/background component fixed. Retained appliance
heat is capped at the supplied total internal gain, avoiding negative inferred
background gains. This 85% assumption matches the existing generator preparation;
it needs empirical verification. Cooking fuel is included in the supplied
electric baseload approximation, not modelled independently.

Solar gains are `GHI * window_area * solar_gain_factor * solar_aperture_factor` kWh.
The **0.45 default aperture factor is an assumption**, combining glazing, frame,
shading and a coarse GHI-to-window exposure proxy. Supply a better externally
derived factor when available; it is not orientation-resolved solar modelling.
The product of the factors is bounded to [0,1] in this simplified model.
Gains are allocated uniformly across the two periods. Useful heating in each
period is `max(gross_loss - gains, 0)`. Excess gains are reported as unused,
not negative heating or a predicted cooling bill. Summer overheating and the
actual resulting free-floating indoor temperature are outside this model.

Useful hot water is supplied kWh/day × days × monthly multiplier. Fuel input is
useful heat divided by equipment efficiency or SCOP. Independent immersion DHW
uses efficiency 1 and electricity. A separate heat-pump cylinder needs an explicit
DHW COP; a gas space-heating efficiency must never be used as its COP. With
`from_main`, DHW inherits the main seasonal conversion factor. `instantaneous`
currently inherits the main fuel/efficiency because the intake schema has no
separate equipment identity; supply an explicit FuelConfig if this differs.
Cylinder/distribution losses and boiler auxiliary loads require better intake
data and are not independently modelled here.

For heat pumps, ambient-source heat = useful heat − purchased electricity;
for efficiencies below 1, conversion losses = purchased fuel − useful heat.
Both are reported so equipment energy conservation can be checked. SCOP is
constant, with no temperature-dependent performance, defrost or capacity limit.

## Outputs and integration

`simulate_thermal(TwinParams, weather_months, FuelConfig)` returns `ThermalResult`
with shared contract inputs, `Provenance(engine="physics_l1")`, months, monthly
and period-total useful space heat, useful DHW, fuel kWh, appliance baseload,
gains, unused gains, conversion losses, ambient-source heat, and element losses.
The result includes the input parameters, equipment conversion and weather
records so the calculation can be reproduced.
Totals refer to the supplied period; they are annual only for a full year.
Per-element W/K and percentage shares use `ElementHeatLoss(estimated=True)`.
Every result is labelled **uncalibrated**. Values are exact calculations
conditional on the supplied assumptions, not uncertainty distributions.

Basic electricity here is only appliance baseload plus electric heating/DHW
input. With no PV or battery, this equals grid import. Installed PV or batteries
are rejected rather than silently ignored; an explicit no-PV scenario may be
run separately but must not be presented as that home's actual grid imports.
The CLI and generator also reject solar thermal. The low-level TwinParams contract has no
solar-thermal flag, so upstream callers must enforce that restriction.

`run_physics(numeric_params, weather_arrays, mode, fuel, assumptions)` retains
Satyarth's integration seam. It is compatible with `jax.jit` (mark mode, fuel and
assumptions static), `jax.vmap` and `jax.grad`; see the tests for usage. The
heating-on/off `max` is piecewise differentiable, with a kink at zero demand.
Invalid traced domains produce NaN, never valid-looking zero demand. Validate
inputs at the application boundary and constrain future Bayesian priors.
The legacy `calibrated` mode string accepts already supplied parameters only;
it performs no calibration. `standard` is rejected until explicit occupancy
and behaviour preparation is implemented. The generator now calls uncalibrated mode.

Heat conservation: useful space heat + all gains = gross loss + unused gains.
Fuel conversion and electric end-use accounting are also checked by the same
validator used for synthetic truth. PV/export/battery/storage outputs are exact
zeros only for supported no-PV/no-battery scenarios. Re-running the thermal
engine with edited fabric or equipment enables paired scenarios; catalogue
translation and feasibility are still the integration layer's responsibility.

## Validation and next phase

Tests compare a hand-calculated U*A/ventilation/bridging example, seasonal heat
balances, all six fuels, separate DHW conversion, monotonic insulation response,
equipment balances, analytic gradients, JIT/vmap execution, invalid inputs and
the actual synthetic-generator pipeline. This verifies implementation and
conservation, **not accuracy against real buildings**.

Before real-customer claims: verify the imported `to_verify` fabric/equipment
defaults, validate geometry, obtain actual weather and meter records, test
predictions against independent observed consumption, and estimate calibration
uncertainty. Those are explicit future tasks. Calibration can reuse this
differentiable seam; the next electricity phase must replace zero PV/storage
flows with time-resolved generation and dispatch, preserving per-fuel savings.

JAX's [transformation documentation](https://docs.jax.dev/en/latest/101/transformations.html)
and [static-argument guidance](https://docs.jax.dev/en/latest/jit-compilation.html)
describe the compilation/differentiation interface used here.
