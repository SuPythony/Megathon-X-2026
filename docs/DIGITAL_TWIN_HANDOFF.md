# Digital Twin (Part 1): handoff for the other parts

Owner: Satyarth. Branch: `satyarth` (not merged into `main` yet). Status as of 11 Oct 2026 (updated: physics inputs
package, appliances, household, hot water, retrofits, rooms, `GreenDigitalTwin`).
Read this to find out what the twin gives you, how to call it, what it needs from you, and what isn't built yet.

Companion docs:
- `docs/digitaltwinvariables.md`: the `run_physics` contract for the physics owner (inputs, outputs, rules).
- `docs/ROOM_TWIN_DESIGN.md`: the room-level twin design (photos, archetypes, appliances, 3D).

---

## 1. What runs today

```
Intake (postcode district, EPC number, photos, appliances)          greentwin/twin/intake.py
  │
Property (EPC record) ──► archetype layout (geometry)                 greentwin/twin/archetypes.py, vision.py
  │                              │
  └──► params_from_property ◄────┘   (TwinParams + source of every value)   greentwin/twin/params.py
                 │
                 ├── weather arrays (postcode district)               greentwin/twin/weather.py
                 ├── prior samples (uncalibrated)                     greentwin/twin/simulate.py
                 ▼
          simulate ── jax.vmap(run_physics) ──► TwinRun               greentwin/twin/simulate.py
                 ▲
     run_physics: teammate's greentwin/twin/physics.py if present, else physics_stub.py   (physics_api.py)

Scene JSON for the 3D viewer                                          greentwin/twin/scene.py → web/twin-viewer
```

Try it:

```bash
uv run python scripts/run_twin.py            # SH1-SH3 end to end; prints ranges vs metered totals
uv run pytest tests/test_physics_contract.py tests/test_twin_params.py
cd web/twin-viewer && npm install && npm run dev   # 3D room viewer
```

If `uv sync` is too heavy for your machine, this runs the twin tests with only what they need:
`uv run --python 3.12 --no-project --with pydantic --with pytest --with numpy --with scipy --with jax python -m pytest tests/test_physics_contract.py`

Output of `scripts/run_twin.py` today (stub physics, synthetic weather, **uncalibrated**):

| Home | Gas median (90% range) | Metered gas | Electricity median (90% range) | Metered electricity |
|---|---|---|---|---|
| SH1 Victorian terrace | 25,053 (15,009–41,153) | 15,016 | 3,642 (2,279–5,932) | 2,871 |
| SH2 1970s semi | 21,961 (12,764–36,896) | 12,800 | 3,907 (2,445–6,364) | 3,400 |
| SH3 electric flat | 0 | 0 | 5,870 (4,417–8,290) | 4,600 |

Metered values sit inside every 90% range (SH1 gas only just), but the medians are high: an uncalibrated default-based model typically
over-predicts older homes. **These are not results.** They prove the pipeline runs. Real numbers need the real
physics, real weather and calibration (section 6).

---

## 2. The physics inputs package (what we hand to physics and calibration)

One JSON document per home, `PhysicsInputs` (`greentwin/twin/inputs.py`, version `physics_inputs/0.1`). Schema:
`greentwin/twin/examples/physics_inputs.schema.json`. Worked examples:
`greentwin/twin/examples/physics_inputs_SH1.json`, `_SH2.json` and `_SH3.json` (regenerate with
`uv run python scripts/make_twin_examples.py`).

```python
from greentwin.twin.twin import GreenDigitalTwin
pkg = GreenDigitalTwin().physics_inputs(prop, layout=layout, appliances=placements, intake=intake, usage=usage)
run_physics(pkg.physics_params(), pkg.physics_timeseries(), "calibrated", pkg.fuel_config())   # ready to run
```

| What you asked for | Where it is in the package | Built from |
|---|---|---|
| U-values per element | `fabric[]`: element, U, area, UA, source. Also `params.*_u` | EPC (measured) → RdSAP default tables (**to_verify**) → ISO 13370 for floors |
| Ventilation | `ventilation`: ach, volume, 0.33·ach·V, source | Age-band assumption less draught proofing (**assumption**) |
| Heating system + fuel | `heating_system`: system, fuel, efficiency | EPC; efficiency from EPC or typical by system (**to_verify**) |
| **Separate** hot-water equipment, efficiency, fuel | `hot_water`: system (combi / cylinder from main / immersion / heat-pump cylinder / instantaneous electric), fuel, efficiency, cylinder standing loss, electric shower kWh/day, main-system useful demand | EPC `heating.hot_water` + cylinder, appliance list (immersion answers, electric shower), SAP Appendix J |
| Appliances → consumption schedules | `appliances.appliances[]`: per appliance kWh/yr, fuel, 12 monthly kWh, 24-h average-day kW profile. `appliances.*` totals; `schedules.*` aligned to the weather months | User's appliance list + size answers → `appliance_energy.json` (**typical_estimate**) |
| Matching internal heat gains | per appliance `gains_w_monthly` (energy × share released indoors); `params.internal_gains_w` (metabolic × presence + appliances + lighting − evaporation); `schedules.gains_month_factor` | SAP 2012 Table 5 structure + appliance schedules |
| Weather | `weather`: months, arrays (`hdd_15_5`, `mean_temp_c`, `ghi_kwh_m2`, `days_in_month`, `month_of_year`), source, filled fields | Part 0 `load_weather` when available; until then **synthetic** typical values for the meter months |
| Monthly meter readings | `meter`: months, electricity/gas kWh (None = missing, never 0), estimated flags | Usage records or the intake's `meter_readings` |
| Occupancy | `occupancy`: occupants, presence, metabolic gains, source | User answer, else SAP Table 1b from floor area |
| Thermal mass | `thermal_mass`: heat capacity (kJ/K), SAP thermal mass parameter (kJ/m²K), low / medium / high, breakdown by element; `params.heat_capacity_kj_per_k` | Construction (walls, floors, partitions) × layout areas, κ values **to_verify**; user's "internal walls solid or hollow?" answer |
| Heating + hot water as the user knows it | intake `heating_system`, `hot_water_source`: these override the EPC; conflicts go in `data_quality.flags` (EPC vs answer, bills vs fuel) | User answers > EPC |
| Heating behaviour | `heating_behaviour`: setpoint (+ prior sd), heating hours (+ sd), periods, source | User thermostat and schedule, else defaults (wider priors) |
| Calibration priors | `priors[]`: param, lognormal / normal_add, mu, sigma, bounds, rationale | User-given values get tighter priors |
| Likelihood suggestion | `observation_model`: lognormal per fuel, 5% actual / 15% estimated reads, missing months latent | Suggestion; the calibration team decides |
| Can we calibrate? | `calibration_readiness`: prior_only / weak / ready, month counts by season, identifiable and confounded parameters | Meter coverage |
| Data quality | `data_quality`: to_verify fields, defaults/assumptions, flags | |

### Is it enough to run the physics?

**Yes.** The package covers every input in `docs/digitaltwinvariables.md` plus the extras below. The stub runs on it
today (`tests/test_twin_package.py::test_package_complete_and_runs_on_stub`). New keys the physics should read (all
optional, with defaults, in `physics_api.py`):
- `EXTRA_PARAM_KEYS`: `hot_water_efficiency`, `hot_water_standing_loss_kwh_per_day`, `shower_electric_kwh_per_day`.
- `SCHEDULE_KEYS` (monthly, in the time-series dict): `appliance_month_factor`, `gains_month_factor`,
  `secondary_heating_kwh_monthly`, `extra_electricity_kwh_monthly`, `extra_gas_kwh_monthly`.

### Is it enough for Bayesian calibration?

**Structurally yes. The numbers become trustworthy once three things are real:**
1. **Real weather for the meter months.** Synthetic weather would bias the heat-loss and setpoint posteriors. This
   needs Part 0's `load_weather(district, months)`.
2. **Real meter readings.** The intake now collects them (`meter_readings`; 12+ months covering winter and summer).
   Showcase homes have their usage files.
3. **Verified defaults.** The priors are centred on RdSAP values marked `to_verify`. Calibration absorbs fabric
   error through `heat_loss_multiplier`, but a biased centre shows up as a biased multiplier.

**What monthly data can and can't identify** (also written into each package's `calibration_readiness`):
- **Identifiable:**
  - `heat_loss_multiplier`, from the winter slope of heating fuel vs temperature
  - `hot_water_kwh_per_day`, from summer heating-fuel use
  - `baseload_kwh_per_day`, from electricity
  - `setpoint_c`, partly
- **Confounded:**
  - heating efficiency × heat-loss multiplier: keep the efficiency prior tight
  - air changes inside the multiplier: don't calibrate separately
  - heating hours × setpoint: fix one, unless the user gave a schedule
- **Not identifiable from monthly totals:**
  - per-element split (walls vs roof): always prior-based, labelled estimated
  - per-room numbers: allocation, labelled estimated
  - hourly behaviour: needs smart-meter half-hourly data, which isn't in scope

## 3. For the physics owner

**You write one function:** `run_physics(params, weather, mode, fuel)` in **`greentwin/twin/physics.py`**.
`physics_api.get_run_physics()` picks it up automatically: no other file changes. Until then the stub runs.

- Inputs and outputs: `docs/digitaltwinvariables.md` sections 2–5. The key lists are in code in
  `greentwin/twin/physics_api.py`: `PHYSICS_PARAM_KEYS`, `WEATHER_KEYS`, `TOTAL_KEYS`, `MONTHLY_KEYS`, `ELEMENTS`.
- `fuel` is a frozen `FuelConfig(heating_fuel, hot_water_fuel)`: static, never traced.
- **Check your model:** `pytest tests/test_physics_contract.py`. It runs the same contract against the stub and
  against your `physics.py`:
  - output keys and shapes
  - totals equal monthly sums; nothing negative; export ≤ PV
  - zero cases (no PV, electric home)
  - monotonicity (U-values, air changes, setpoint, efficiency)
  - `jit` + `vmap` over 1,000 samples
  - finite gradients for the 4 calibrated parameters
  - SH1–SH3 within ±15% of metered. This test is skipped for the stub and switches on for your model.
- **Standard mode:** the wrapper already passes the home's uncalibrated SAP-based values for behaviour fields
  (`BEHAVIOUR_KEYS`) in `"standard"` mode. You only set the SAP heating pattern (setpoint/hours) inside `run_physics`.
- **What the stub decided provisionally** (please confirm or replace; listed in `physics_stub.STUB_ASSUMPTIONS`):
  - `heating_fuel` as a static `FuelConfig` (open decision 1)
  - party wall U = 0 (open decision 5)
  - oil/LPG/solid/district returned as `other_fuel_kwh`, which `TwinRun` doesn't carry (open decision 2)
  - temperature-difference heating from `mean_temp_c`, so `setpoint_c` matters (open decision 3)
  - Option A, monthly steady state (open decision 4)

## 4. For Part 0 (data layer)

What the twin needs from you, in priority order:

1. **EPC lookup by certificate number.** Users give their EPC certificate number (RRN, `0000-0000-0000-0000-0000`) and
   postcode district. The twin calls `EPCLookup.by_rrn(rrn, postcode_district) -> Property | None`
   (`greentwin/twin/intake.py`). Today it's `NoEPCLookup` (always None), so homes fall back to house-type defaults.
   `DataLayer` has no lookup by RRN yet; this needs adding there or as a separate adapter.
2. **Weather by postcode district:** `DataLayer.load_weather(area_code, months) -> list[WeatherMonth]`. The twin uses
   `hdd_15_5`, `mean_temp_c`, `ghi_kwh_m2`. Missing temperature or irradiation is filled and flagged, never zero.
   Until this exists, `weather.typical_year()` is a **synthetic** typical year (`source="synthetic"`).
3. **Privacy:** the twin only ever holds the postcode **district**. The viewer reduces a full postcode in the browser.

## 5. For Part 2 (energy & retrofit engine) and Parts 3–4

Call the twin through `GreenDigitalTwin` (`greentwin/twin/twin.py`), which implements the `DigitalTwin` Protocol:

```python
from greentwin.twin.twin import GreenDigitalTwin
from greentwin.twin.simulate import prior_samples

twin = GreenDigitalTwin(n_samples=1000)
params = twin.params_from_property(prop)                       # ExtendedTwinParams (is a TwinParams)
samples = prior_samples(params, 1000)                          # until the calibration team's posterior exists
baseline = twin.simulate(samples, [])                          # TwinRun
package = twin.simulate(samples, [RetrofitAction(retrofit_id="loft_insulation"),
                                  RetrofitAction(retrofit_id="air_source_heat_pump")])   # re-simulated, never summed
elements = twin.element_heat_loss(samples)                     # list[ElementHeatLoss], estimated=True
rating_run = twin.simulate(samples, [], mode="standard")       # SAP-style run for the rating estimator
twin.last_engine                                               # 'physics' or 'physics_stub'
```

| Method | Status |
|---|---|
| `params_from_property` | Built |
| `apply_retrofits` | Built. All 16 catalogue items map. Mismatched keys are translated provisionally in `retrofits.KEY_MAP`, with each interpretation documented |
| `simulate` | Built. Only the `"typical"` weather scenario until Part 0 |
| `element_heat_loss` | Built |
| `scene` | Built (contract `TwinScene`) |
| `capture` | Built (Sonnet 5.5 floor-plan draft) |
| `calibrate` | `prior_only` with no meters. With meters it raises and points to the calibration team |
| `electricity_profile` | Not built (ownership open) |

- `TwinRun` fields are `(n_samples,)` arrays: `electricity_import_kwh`, `gas_kwh`, `space_heating_kwh`,
  `hot_water_kwh`, `pv_generation_kwh`, `export_kwh`.
  - Monthly: `monthly_electricity_kwh`, `monthly_gas_kwh`, shape `(n_samples, n_months)`.
  - `element_heat_loss_w_per_k` per element.
- Build `Distribution`s with `contracts.summarise.summarise(run.gas_kwh, "kWh/yr")`.
- `mode="standard"` gives the SAP-style run for the rating estimator (`contracts/sap_estimator.json`).
- **Provenance:** `twin.provenance(...)` maps the stub to contract engine `"mock"` (model_version
  `physics_stub/...`), so the existing "no mock in the demo" rule keeps stub numbers out of the demo. The real
  physics maps to `"physics_l1"` automatically.
- **Rooms:** `twin.room_energy(prop, layout, samples, appliances)` → per-room heat loss, heating, appliances,
  lighting and running cost as `Distribution`s. Rooms sum to the whole house, labelled estimated.

## 6. For the UI / Part 5

- **Retrofits in 3D + before/after (built):** `web/twin-viewer` shows every catalogue retrofit on the house and the
  re-simulated package's numbers. Python side: `greentwin/twin/scenarios.py` (`run_scenario` → `ScenarioView`:
  before / after / savings per sample for gas, electricity, £ at unit rates and carbon at the DESNZ factors, plus
  per-room numbers after), `scene.retrofit_options` (geometry applicability, costs with `cost_status`), dev server
  `scripts/twin_server.py` (`POST /api/twin/run`). The production API (Part 5) can reuse `run_scenario` directly.
- **Appliance size questions (built):** the viewer asks each appliance's questions; answers are part of
  `AppliancePlacement` and `SceneAppliance`.

- **3D room viewer:** `web/twin-viewer` (own folder; doesn't touch `web/`). It renders the scene JSON from
  `greentwin/twin/scene.py`. Schemas: `greentwin/twin/examples/*.schema.json`.
- **Intake:** the viewer's "New home" form produces `TwinIntake` JSON (`greentwin/twin/examples/twin_intake.schema.json`).
  Nothing processes it yet: that needs a twin API endpoint (orchestrator / `greentwin/api`).

---

## 7. Where every number comes from, and what's still provisional

`params_with_sources(prop)` returns, for every `TwinParams` field, where its value came from. Order of precedence:
measured on the EPC → fitted floor plan → RdSAP default table → SAP 2012 formula → labelled assumption.

| Input | Source | Status |
|---|---|---|
| Areas, volume, party walls | Fitted archetype layout (floor plan or EPC floor area + built form) | Geometry approximation |
| Wall / roof / window / door U | `greentwin/twin/rdsap_defaults.json` (RdSAP 2012 Tables S6, S9, S10, S14) | **to_verify**: approximations, not yet checked line by line against the published tables |
| Floor U | BS EN ISO 13370 slab method (RdSAP Appendix S5) on the layout's exposed perimeter | Method sourced; suspended floors approximated |
| Air changes per hour | Age band less draught proofing | **assumption** |
| Heating efficiency | EPC if given, else typical by system | **to_verify** |
| Occupancy, hot water, baseload, lighting, internal gains | SAP 2012 Appendix J / L, Table 1b, Table 5 | Formulas; verify coefficients before submission |
| Weather | Synthetic typical year until Part 0's `load_weather` | **synthetic** |
| Posterior | Prior only (working priors in `simulate.PRIOR`) until calibration | **prior_only** |
| Physics | `physics_stub.py` until `physics.py` lands | **stub** |
| Appliance kWh, schedules, gains | `appliance_energy.json` | **typical_estimate**: typical product values, not yet individually cited |
| Hot-water efficiency, cylinder loss | Main system with boiler summer penalty; cylinder insulation | **assumption** |
| Occupants, behaviour | User answers, else SAP Table 1b and defaults | Wider priors when defaulted |
| Retrofit key translations | `retrofits.KEY_MAP` | **provisional** until the catalogue keys are renamed |

### Still to do

**Ours (Part 1):**
1. Verify every `to_verify` value in `rdsap_defaults.json` against RdSAP 2012, and cite the appliance figures.
2. Run vision live on real photos (needs `ANTHROPIC_API_KEY` from the team lead); cache demo drafts.
3. Wire `TwinIntake` → twin once an API endpoint exists.

**Other teams:**
- **Calibration team:** Bayesian calibration from the physics inputs package.
- **Physics owner:** `physics.py`.
- **Part 0:**
  - EPC lookup by certificate number (RRN)
  - `load_weather(district, months)`
- **Contract owner:** rename the catalogue keys.
- **Open:** `electricity_profile` ownership.
