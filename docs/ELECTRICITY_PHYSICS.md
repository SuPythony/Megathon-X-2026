# Coupled JAX thermal and electricity twin

Model `thermal-electricity/0.2.0`. Inputs were audited and generated before the
electricity solver was implemented; see [ELECTRICITY_INPUT_AUDIT.md](ELECTRICITY_INPUT_AUDIT.md).
One hundred existing synthetic homes now have validated half-hour input packages
for 24 months. Inputs and all outputs stay in this worktree. No Bayesian calibration
has been added and no real-building accuracy claim is made.

## Run prepared inputs

```bash
.venv/bin/python scripts/run_electricity_twin.py \
  --prepared data/processed/synthetic/electricity-inputs-v1/home-0000.json
```

This reads the already prepared JSON + compressed interval array bundle. It does
not create weather or equipment inputs implicitly. Results default to stdout;
`--output` writes a new file inside this worktree, refusing overwrites.
`--include-intervals` includes the complete half-hour traces; otherwise results
include monthly energy, totals, peaks, ratios and validation errors.

Run the latest Satyarth example through the adapter:

```bash
.venv/bin/python scripts/run_electricity_twin.py \
  --prepared data/processed/synthetic/electricity-scenarios-inputs-v1/satyarth-SH2.json \
  --satyarth-package data/cache/satyarth/7eeb4d6-physics_inputs_SH2.json
```

The cached package was read from Satyarth commit
`7eeb4d651695fa7311509aa746daacbae06b6d79`, without fetching or modifying his branch.
`adapt_satyarth_package` preserves appliance schedules, distinct DHW conversion,
cylinder losses, electric showers, secondary heaters, gas cooking and outdoor
electrical demand. The 24-hour appliance profile excludes heating, DHW and outdoor
loads; it is normalised rather than added to the existing baseload.

Generate a new electricity-backed synthetic dataset:

```bash
.venv/bin/python scripts/generate_synthetic.py --demo --homes 100 --months 24 \
  --electricity-development \
  --simulator greentwin.twin.physics:run_physics \
  --simulator-version thermal-electricity/0.2.0 \
  --data-version electricity-development-v1 \
  --output data/processed/synthetic/electricity-development-v1
```

Each baseline/package first prepares and validates its own electricity inputs,
then runs the physics. Paired scenarios reuse the seed/household; changing PV or
battery does not resample appliance behaviour. Every truth output is checked at
interval and monthly resolution before meter noise/missingness is applied.
No scenario labels are invented when no scenario factory is supplied.

## Numerical model

`run_physics` retains the four-argument Satyarth seam, with optional
`weather['electricity_inputs']` containing only numeric arrays/scalars.
Without this package, installed PV/batteries remain rejected; no unphysical
fixed self-consumption percentage or hidden battery fallback is used.

Monthly thermal results remain authoritative for useful space heat and DHW.
The electricity solver normalises appliance, lighting and DHW profile weights
within each month. Heating demand is distributed using temperature degree-hours
and the heating schedule. It remains an allocation of steady-state monthly
energy, not a dynamic simulation of temperature, cycling or heat-pump capacity.
The heating weights are renormalised to preserve monthly equipment consumption.
When no degree-hours exist but monthly heating is positive, uniform allocation
is used and explicitly flagged. Future equipment power limits require an upgrade.

Monthly electrical end uses sum to appliances + lighting + electric space heat +
DHW + outdoor/other explicit loads. Lighting is a share of baseload, not another
complete baseload. Appliance retrofits must update both baseload and internal
gains before rerunning the thermal model. User/Satyarth supplied gain factors
override the approximate baseload-derived gain variation.

Plug-in secondary heat is capped by useful space demand and displaces main-system
heat. The unused requested secondary amount is reported separately. Cylinder
standing losses require additional main-DHW fuel but are not useful tap-water heat;
they are treated as unrecovered losses in this version. Electric showers supply
additional useful DHW at efficiency 1 and are never counted as appliance baseload.
Gas cooking uses its explicit gas schedule; its gains are already supplied by the
input preparation. Outdoor loads add electricity without indoor heat gains.

PV input is AC energy per kWp, supplied externally or explicitly synthetic.
It is scaled by installed capacity and clipped to the inverter power times
interval duration. Reported generation is available AC after inverter clipping;
unrealised clipping energy is a separate diagnostic. Roof geometry must match
the prepared profile: orientation-changing scenarios must reprepare their input.

The battery policy is PV self-consumption: PV serves demand first, surplus charges,
the battery discharges into unmet demand, then the grid imports/exports.
Charging/discharging occur at the AC household bus. With stored energy S:

`S_next = S + eta_charge*charge - discharge/eta_discharge`

Capacity, initial charge and both power limits are enforced. There is no simultaneous
charge/discharge, grid charging, battery export, degradation or standby load in this
version. `jax.lax.scan` carries state continuously across the entire timeline; there
is no daily/monthly reset. Final stored energy is reported, not counted as delivered
savings. Annual comparisons should use matched initial conditions and account for
different terminal stored energy; the default initial state is empty.

Export limits cause curtailment after demand and charging are served. For every
interval, `import + generation = demand + export + curtailment + battery_loss + delta_S`.
An independent validator also checks storage state, losses, power constraints,
nonnegative flows, PV availability and monthly aggregation. The synthetic truth
validator checks thermal balance, equipment conversion and purchased fuel totals.

All numeric runtime equations are JAX operations. Input generation/validation uses
NumPy and contract models outside the differentiable solver. `jit`, `vmap` and
gradients are tested; min/max dispatch is piecewise differentiable, with kinks at
operating limits. Observed dispatch policies should be fixed when calibrating.

## Interpretation and remaining work

Self-consumption is (direct PV + PV charging)/available PV, including charging losses
and energy remaining stored at the end. It is null with no PV. Self-sufficiency is
1 - import/demand; initial battery energy is separately reported and must be considered
when interpreting this ratio. Peak import is the maximum half-hour average kW,
not a sub-second fuse/connection sizing estimate. Peak share uses local 16:00–19:00.

Standard import bills are calculated only when all simulated months lie inside the
validity period of `contracts/tariffs.json`. Export revenue is unavailable for positive
exports until a sourced tariff exists, so net bills then remain unavailable. No ToU
prices are fabricated and tariff arbitrage is not enabled. Simulation of energy does
not require tariffs. The input audit identifies easy user questions and installer
specifications needed for real systems.

The existing showcase homes have zero PV/battery; hypothetical systems are explicit
scenarios, not claims that those homes actually have panels or batteries. Synthetic
solar profiles use monthly GHI and approximate sun geometry, not actual PVGIS.
London smart-meter archetype fitting and real interval-weather/PVGIS ingestion remain
data-layer tasks. Solar-thermal contribution remains unsupported.

Bayesian calibration remains the next phase. Its likelihood will compare monthly
meters to this coupled simulator's import/export/gas totals; posterior samples can
be propagated through the same JAX function. Conditional deterministic results here
are labelled uncalibrated and do not pretend to be posterior distributions.
