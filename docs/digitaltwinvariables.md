# Digital Twin: Physics Variables (handoff spec)

Physics model owner: _teammate_. Digital twin wrapper owner: Satyarth.
Based on repo commit `a1ba6e4`, contracts schema `0.4.0` (`contracts/models.py`, `greentwin/interfaces.py`).
Status: **draft to agree**. Items marked **AGREE** need a decision from both of us before coding.

> **Update (11 Oct 2026): the wrapper side is built.** `params_from_property`, weather arrays, prior samples and
> `simulate` (vmap) all exist and run end to end on a **stand-in** `run_physics` (`greentwin/twin/physics_stub.py`).
> Put your function in **`greentwin/twin/physics.py`** and it's picked up automatically. Then check it with
> `pytest tests/test_physics_contract.py`. Key lists in code: `greentwin/twin/physics_api.py`.
> The stub took a provisional position on each open decision (section 10); please confirm or replace.
> Full guide: `docs/DIGITAL_TWIN_HANDOFF.md`.

---

## 1. Who owns what

| Physics owner (you) | Digital twin owner (Satyarth) |
|---|---|
| `run_physics()` in `greentwin/twin/physics.py`: one home, one parameter set, weather in, energy out | `params_from_property` (RdSAP defaults → `TwinParams`) |
| Standard-mode (SAP) occupancy and heating pattern inside `run_physics` | `apply_retrofits` (catalogue `twin_edits` → `TwinParams`) |
| Physics unit tests (monotonicity, energy balance) | `simulate` (`jax.vmap` of `run_physics` over posterior samples) |
| | `calibrate` (NumPyro model, `run_physics` inside the likelihood) |
| | `element_heat_loss`, `scene` (3D), `capture` (photo-to-twin) |

You never need to touch NumPyro, posterior samples or the 3D scene. I never touch the physics equations.

---

## 2. The function

```python
# greentwin/twin/physics.py
def run_physics(
    params: dict[str, jax.Array],    # section 3: one scalar per key
    weather: dict[str, jax.Array],   # section 4: arrays of shape (n_months,)
    mode: Literal["calibrated", "standard"],   # static argument
    fuel: FuelConfig,                # section 3.3: static, not traced
) -> dict[str, jax.Array]:           # section 5
    ...
```

I call it as:

```python
jax.vmap(run_physics, in_axes=(0, None, None, None))(params_batched, weather, mode, fuel)
```

So `params` arrive batched over posterior samples, and weather is shared by all samples.

### Hard rules (calibration breaks without these)
1. **`jax.numpy` only.** No `numpy`, `scipy`, `math` or PyTorch on traced values.
2. **Pure.** No I/O, no globals, no randomness, no printing traced values.
3. **Differentiable.** NUTS/SVI take gradients through it. Use `jnp.where`, `jnp.maximum`, `jax.nn.softplus` instead of Python `if`/`max` on values. Avoid hard clips where the gradient matters.
4. **`vmap`-able.** No Python loops over values. Loops over months are fine with `jax.lax.scan` or plain vectorised maths.
5. **Fixed shapes.** Output shapes depend only on `n_months`, never on values.
6. **float32-safe.** No NaN at the edges, e.g. zero PV or zero window area.

---

## 3. Input: `params`

Keys are the `TwinParams` field names in `contracts/models.py`. Don't rename them, because retrofits and calibration edit these exact names.

### 3.1 Numeric parameters (traced, one scalar per key)

| Key | Unit | Meaning | Default | Calibrated? | Changed by retrofit |
|---|---|---|---|---|---|
| `wall_area_m2` | m² | Exposed external wall area, net of windows and doors | required | no | no |
| `roof_area_m2` | m² | Heat-loss roof area | required | no | no |
| `floor_area_m2` | m² | Ground-floor heat-loss area | required | no | internal wall insulation (×0.97) |
| `window_area_m2` | m² | Total glazing area | required | no | no |
| `door_area_m2` | m² | External door area | 3.7 | no | no |
| `party_wall_area_m2` | m² | Party wall area (semis, terraces, flats) | 0.0 | no | no |
| `wall_u` | W/m²K | Wall U-value | required | no | cavity, internal and external wall insulation |
| `roof_u` | W/m²K | Roof U-value | required | no | loft insulation |
| `floor_u` | W/m²K | Floor U-value | required | no | floor insulation |
| `window_u` | W/m²K | Window U-value | required | no | double, triple and secondary glazing |
| `door_u` | W/m²K | Door U-value | 3.0 | no | no |
| `thermal_bridging_y` | W/m²K | Bridging allowance over total exposed area | 0.15 | no | no |
| `air_changes_per_hour` | 1/h | Infiltration + ventilation rate | required | no | draught proofing |
| `volume_m3` | m³ | Heated volume | required | no | no |
| `heat_loss_multiplier` | – | Scales total fabric + ventilation loss | 1.0 | **yes** | no |
| `setpoint_c` | °C | Mean heating setpoint | 19.5 | **yes** | smart thermostat (−0.5) |
| `heating_hours_per_day` | h/day | Heating hours | 9.0 | no | smart thermostat (×0.93) |
| `heating_efficiency` | – | Boiler seasonal efficiency (e.g. 0.84) or heat-pump SCOP (e.g. 2.9) | required | **yes** | heat pump |
| `hot_water_kwh_per_day` | kWh/day | Useful hot-water heat demand | required | no | no |
| `baseload_kwh_per_day` | kWh/day | Appliances + lighting + cooking electricity | required | **yes** | LED lighting (via lighting share) |
| `lighting_share_of_baseload` | – | Lighting fraction of baseload | 0.15 | no | LED lighting (×0.25) |
| `internal_gains_w` | W | Mean internal heat gains | 400.0 | no | no |
| `solar_gain_factor` | – | Scales solar gains through glazing | 1.0 | no | no |
| `pv_kwp` | kWp | PV array size | 0.0 | no | solar PV |
| `pv_azimuth_deg` | ° | PV azimuth (180 = south) | 180.0 | no | no |
| `pv_tilt_deg` | ° | PV tilt | 35.0 | no | no |
| `battery_kwh` | kWh | Battery capacity | 0.0 | no | battery |

**Missing field: party wall U-value (AGREE).** `party_wall_area_m2` exists but there's no `party_wall_u`. As a result, the `party_walls` heat-loss output (§5.3) can't be computed. Options: (a) add `party_wall_u` to `TwinParams` (contract change, raised with the contract owner), or (b) hard-code an RdSAP default inside physics as a named, cited constant, until the field exists. Either way, emit `party_walls` (0.0 if area is 0).

"Calibrated" means it varies across posterior samples, so outputs must be smooth in it. All other fields can still vary per sample, because calibration may widen the list. Treat every numeric key as traced.

### 3.2 Not passed to physics
- `property_id`: bookkeeping only.
- `tariff` (`standard` / `time_of_use`): changes bills, not kWh. Bills are priced outside the physics.

### 3.3 Categorical: `heating_fuel` (**AGREE**)
`TwinParams.heating_fuel` is a string: `mains_gas | electricity | oil | lpg | solid | district`. JAX can't trace strings. It's also the same for every posterior sample. Proposal: pass it as a static `FuelConfig`, not inside `params`.

```python
@dataclass(frozen=True)
class FuelConfig:
    heating_fuel: str            # from TwinParams.heating_fuel
    hot_water_fuel: str          # same as heating_fuel unless electric immersion
```

Alternative: physics returns fuel-agnostic delivered energy (`heating_fuel_kwh`, `hot_water_fuel_kwh`) and my wrapper assigns it to gas or electricity. Pick one.

---

## 4. Input: `weather`

From `WeatherMonth` (`contracts/models.py`). Every array has shape `(n_months,)`. `n_months` is 12 for typical, cold or warm years. Calibration uses the home's actual meter months, usually 12–24.

| Key | Unit | Meaning | Always present? |
|---|---|---|---|
| `hdd_15_5` | K·day | Heating degree days, base 15.5 °C | yes |
| `mean_temp_c` | °C | Monthly mean outdoor temperature | wrapper fills it if the source lacks it |
| `ghi_kwh_m2` | kWh/m² | Global horizontal irradiation (solar gains, PV) | wrapper fills it if the source lacks it |
| `days_in_month` | days | Days in each month | yes (added by wrapper) |
| `month_of_year` | 1–12 | Calendar month, for seasonal profiles | yes (added by wrapper) |

`cdd_22` exists in the contract but there's no cooling in v1, so ignore it.

Weather scenarios (`typical`, `cold_year`, `warm_year`, `forecast_week`, `ukcp18_2050`) are resolved by me into these arrays. Physics doesn't need to know which scenario it is.

**AGREE:** HDD base 15.5 °C is fixed by the data. If your model uses `setpoint_c` directly, compute heating demand from `mean_temp_c` and treat HDD as a cross-check. Otherwise the setpoint has no effect, and calibrating it won't work.

---

## 5. Output

Return a dict with these keys. They map 1:1 onto `TwinRun` in `greentwin/interfaces.py`. After `vmap`, each array gains a leading `n_samples` axis.

### 5.1 Totals over the simulated period (scalars, kWh)
| Key | Meaning |
|---|---|
| `electricity_import_kwh` | Grid electricity bought: baseload + electric heating/hot water − PV self-consumed (− battery) |
| `gas_kwh` | Mains gas bought (0 if not gas-heated) |
| `space_heating_kwh` | Useful space-heat demand (before efficiency) |
| `hot_water_kwh` | Useful hot-water demand (before efficiency) |
| `pv_generation_kwh` | Total PV generated |
| `export_kwh` | PV exported to grid |

### 5.2 Monthly series (shape `(n_months,)`, kWh)
| Key | Meaning |
|---|---|
| `monthly_electricity_kwh` | Electricity import per month. **Calibration fits this.** |
| `monthly_gas_kwh` | Gas per month. **Calibration fits this.** |

Totals must equal the sum of the monthly values.

### 5.3 Heat loss per element (scalars, W/K)
Key `element_heat_loss_w_per_k`, a dict with exactly these keys (from `ElementHeatLoss.element`):

`roof`, `walls`, `floor`, `windows`, `doors`, `ventilation`, `party_walls`

Each value is U×A (or the ventilation equivalent) **after** `heat_loss_multiplier`. These drive the 3D heat-loss colours. Their sum is the home's total heat-loss coefficient.

### 5.4 Other fuels (**AGREE**)
`TwinRun` only has electricity and gas. Oil, LPG, solid and district heating have no output field. Options: (a) add `other_fuel_kwh` (contract change, other parts affected), or (b) skip non-gas fossil homes in v1. Don't add the key until we've agreed.

---

## 6. Mode: `calibrated` vs `standard`

| | `calibrated` | `standard` |
|---|---|---|
| Used for | Real bills, savings, what-ifs | Estimated rating (SAP 2012 ECF method, `contracts/sap_estimator.json`) |
| Behaviour params | Use `params` as given (`setpoint_c`, `heating_hours_per_day`, `hot_water_kwh_per_day`, `baseload_kwh_per_day`, `internal_gains_w`) | **Override** them with SAP-style fixed heating pattern and occupancy by floor area |
| Fabric params | as given | as given |

You own the standard-mode values. Source them from the SAP 2012 spec, cite them in a docstring and keep them in one constants block. The rating estimator only uses the before/after *difference*, so consistency matters more than exactness.

---

## 7. How retrofits reach you

Retrofits never call physics directly. `apply_retrofits` edits `params`, then `run_physics` runs again on the whole package. Physics must therefore respond correctly to these fields changing: U-values, `air_changes_per_hour`, `setpoint_c`, `heating_hours_per_day`, `heating_efficiency` + fuel, `pv_kwp`, `battery_kwh`, `lighting_share_of_baseload`, `floor_area_m2`.

FYI, a contract issue on my side: the catalogue keys in `contracts/retrofit_catalogue.json` don't match `TwinParams` names yet. I'm raising it with the contract owner. For reference:

| Catalogue key | Intended `TwinParams` effect |
|---|---|
| `roof_u_value`, `wall_u_value`, `floor_u_value`, `window_u_value` | set `roof_u`, `wall_u`, `floor_u`, `window_u` |
| `ach_reduction: 0.15` | reduce `air_changes_per_hour` (absolute or fraction: unclear) |
| `floor_area_factor: 0.97` | multiply `floor_area_m2` |
| `setpoint_reduction_c: 0.5` | `setpoint_c` −0.5 |
| `heating_hours_factor: 0.93` | multiply `heating_hours_per_day` |
| `main_system: air_source_heat_pump`, `scop: 2.9` | `heating_fuel = electricity`, `heating_efficiency = 2.9` |
| `main_system: electric_storage_hhr`, `controllability_gain: 0.15` | no matching field yet |
| `pv_kwp`, `battery_kwh`, `lighting_share_of_baseload`, `tariff` | match already |

---

## 8. Reference formula (Option A sketch, from `docs/IMPLEMENTATION_PLAN_v2.md` §2a)

Not prescriptive. It's here so we agree on the meaning of each variable.

```
HLC  = heat_loss_multiplier × ( Σ U_i·A_i  +  thermal_bridging_y·A_exposed  +  ventilation )
space_heat_month = max(0, HLC × degree_hours − useful(solar gains + internal gains))
fuel_heat  = space_heat / heating_efficiency
fuel_dhw   = hot_water_kwh_per_day × days / efficiency
elec       = baseload_kwh_per_day × days (+ electric heat/dhw) − PV self-consumed
```

Option A (monthly steady state) vs Option B (4–6 node RC graph, `diffrax` ODE) is **still open** (**AGREE**). The function signature in section 2 works for both.

---

## 9. Tests your function must pass

1. **Monotonicity:** lowering any U-value, `air_changes_per_hour` or `setpoint_c` never increases heat or fuel. Same for raising `heating_efficiency` or `pv_kwp` on electricity import.
2. **Energy balance:** totals = sum of monthly values. `export_kwh ≤ pv_generation_kwh`. Every output ≥ 0.
3. **Zero cases:** `pv_kwp = 0` → PV and export are 0. Electric-heated home → `gas_kwh = 0`.
4. **JAX:** `jax.jit`, `jax.vmap` over 1000 samples and `jax.grad` w.r.t. `heat_loss_multiplier`, `setpoint_c`, `heating_efficiency` and `baseload_kwh_per_day` all run and give finite values.
5. **Plausibility:** showcase homes SH1–SH3 (`contracts/examples/`) within ±15% of their metered annual totals, run with default params before calibration.

---

## 10. Open decisions (both of us)

Provisional positions taken by the stub (change them by changing your `physics.py`; the wrapper adapts):
1. `heating_fuel`: static `FuelConfig(heating_fuel, hot_water_fuel)` (`physics_api.FuelConfig`).
2. Other fuels: returned as `other_fuel_kwh` (not in `TwinRun` yet).
3. Setpoint: heating demand from `mean_temp_c`, so `setpoint_c` matters; HDD isn't used by the stub.
4. Option A (monthly steady state).
5. Party walls: U = 0.
6. Ownership of `rating.py`, `electricity.py`, `generate.py`: still open.


1. `heating_fuel` handling: static `FuelConfig`, or fuel-agnostic output (§3.3).
2. Oil / LPG / solid / district output (§5.4).
3. Setpoint vs fixed-base HDD (§4).
4. Option A vs B (§8).
5. Party wall U-value: add `party_wall_u` to the contract, or use a hard-coded default (§3.1).
6. Who owns `twin/rating.py` (SAP estimator), `twin/electricity.py` (half-hourly layer) and `twin/generate.py` (synthetic data)?
7. Until your function lands, I'll use a stub `run_physics` with this exact signature: a simple HLC × degree-day model. Your version replaces it without changes on my side.
