# Digital Twin (Part 1): handoff for the other parts

Owner: Satyarth. Branch: `satyarth` (not merged into `main` yet). Status as of 11 Oct 2026.
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
| SH1 Victorian terrace | 23,700 (13,600–39,800) | 15,016 | 3,643 (2,279–5,933) | 2,871 |
| SH2 1970s semi | 21,371 (12,018–36,375) | 12,800 | 3,906 (2,444–6,362) | 3,400 |
| SH3 electric flat | 0 | 0 | 5,334 (3,870–7,776) | 4,600 |

Metered values sit inside every 90% range, but the medians are high: an uncalibrated default-based model typically
over-predicts older homes. **These are not results.** They prove the pipeline runs. Real numbers need the real
physics, real weather and calibration (section 6).

---

## 2. For the physics owner

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

## 3. For Part 0 (data layer)

What the twin needs from you, in priority order:

1. **EPC lookup by certificate number.** Users give their EPC certificate number (RRN, `0000-0000-0000-0000-0000`) and
   postcode district. The twin calls `EPCLookup.by_rrn(rrn, postcode_district) -> Property | None`
   (`greentwin/twin/intake.py`). Today it's `NoEPCLookup` (always None), so homes fall back to house-type defaults.
   `DataLayer` has no lookup by RRN yet; this needs adding there or as a separate adapter.
2. **Weather by postcode district:** `DataLayer.load_weather(area_code, months) -> list[WeatherMonth]`. The twin uses
   `hdd_15_5`, `mean_temp_c`, `ghi_kwh_m2`. Missing temperature or irradiation is filled and flagged, never zero.
   Until this exists, `weather.typical_year()` is a **synthetic** typical year (`source="synthetic"`).
3. **Privacy:** the twin only ever holds the postcode **district**. The viewer reduces a full postcode in the browser.

## 4. For Part 2 (energy & retrofit engine) and Parts 3–4

```python
from greentwin.twin.params import params_from_property, fuel_config
from greentwin.twin.simulate import prior_samples, simulate
from greentwin.twin.weather import resolve_weather

params = params_from_property(prop)                 # contracts.TwinParams
samples = prior_samples(params, n_samples=1000)     # PosteriorSamples (prior only until calibration exists)
weather, weather_source = resolve_weather(prop.location.postcode_district)
run, engine = simulate(samples, weather, fuel_config(prop), mode="calibrated")   # TwinRun, arrays (n_samples,)
```

- `TwinRun` fields are `(n_samples,)` arrays: `electricity_import_kwh`, `gas_kwh`, `space_heating_kwh`,
  `hot_water_kwh`, `pv_generation_kwh`, `export_kwh`.
  - Monthly: `monthly_electricity_kwh`, `monthly_gas_kwh`, shape `(n_samples, n_months)`.
  - `element_heat_loss_w_per_k` per element.
- Build `Distribution`s with `contracts.summarise.summarise(run.gas_kwh, "kWh/yr")`.
- `mode="standard"` gives the SAP-style run for the rating estimator (`contracts/sap_estimator.json`).
- **Provenance:** pass `engine` through. `"physics_stub"` must never reach the demo or evaluation, the same rule as
  `"mock"`. When the real physics lands, `engine` becomes `"physics"` by itself.
- Retrofits: `apply_retrofits` is **not built yet**. It's blocked by the catalogue key mismatch (section 6).

## 5. For the UI / Part 5

- **3D room viewer:** `web/twin-viewer` (own folder; doesn't touch `web/`). It renders the scene JSON from
  `greentwin/twin/scene.py`. Schemas: `greentwin/twin/examples/*.schema.json`.
- **Intake:** the viewer's "New home" form produces `TwinIntake` JSON (`greentwin/twin/examples/twin_intake.schema.json`).
  Nothing processes it yet: that needs a twin API endpoint (orchestrator / `greentwin/api`).

---

## 6. Where every number comes from, and what's still provisional

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

### Not built yet (Part 1 backlog)

1. `apply_retrofits`. Blocked: catalogue `twin_edits` keys don't match `TwinParams` names (e.g. `roof_u_value` vs
   `roof_u`). Raised with the contract owner; see `docs/digitaltwinvariables.md` section 7.
2. `calibrate`: NumPyro SVI per home against monthly meter readings. Missing months are latent.
3. `element_heat_loss` → `ElementHeatLoss` list, and room allocation → per-room heat loss and electricity.
4. Appliance energy catalogue (sourced watts and typical use) → replaces the SAP baseload with the user's appliances.
5. `electricity_profile` (half-hourly layer).
6. A `DigitalTwin` class implementing the `greentwin/interfaces.py` Protocol around these functions.
7. Verify every `to_verify` value in `rdsap_defaults.json` against RdSAP 2012 and record citations.
