# What the GreenTwin digital twin can do

Part 1 (digital twin), owner Satyarth. Status as of 11 Oct 2026, branch `satyarth` (`main` up to the last merge).
One page per question: *what can it do, where is the code, and is it real yet?* For how other parts call it, see
`docs/DIGITAL_TWIN_HANDOFF.md`; for the physics contract, `docs/digitaltwinvariables.md`; for business use,
`docs/business_insight.md`.

**Status legend:** ✅ built and tested · 🟡 built, runs on stand-in physics or unverified defaults · ⏳ needs another part ·
❌ not built.

> **Honesty note.** Until the teammate's physics (`greentwin/twin/physics.py`) and the team's calibration land,
> every energy number comes from a **stand-in physics model** with **synthetic weather** and **uncalibrated priors**.
> Those results carry engine `physics_stub` (contract engine `mock`), are labelled in the UI, and must not be
> presented as real. When the real model drops in, everything re-runs with no code changes.

---

## 1. Take in a home

| Capability | Status | Where |
|---|---|---|
| **Intake form:** postcode (only the district is kept, reduced in the browser), EPC certificate number (validated, normalised), floor plan and fuse box photos with consent | ✅ | `greentwin/twin/intake.py`, viewer "New home" |
| **Household questions:** occupants, daytime presence, thermostat setting, heating schedule (presets or custom periods, validated); each allows "not sure", which means typical UK values with wider priors | ✅ | `intake.py`, `household.py`, viewer "New home" form |
| **Heating as the user knows it:** heating system, fuel, hot-water source. Overrides a stale EPC; conflicts flagged (EPC vs answer, bills vs fuel) | ✅ | `household.apply_user_heating`, `meter_fuel_conflicts` |
| **Monthly meter readings:** 12-month table or paste from a spreadsheet; missing months stay missing, never 0; estimated reads flagged; "no gas supply"; live check of whether there is enough for calibration (12+ months, 3 winter, 2 summer) | ✅ | `intake.MeterReading`, viewer "New home" form |
| **Internal walls "solid or hollow?"** (thermal mass) | ✅ | `intake.internal_walls` |
| **EPC lookup by certificate number** | ⏳ Part 0 | seam `intake.EPCLookup` (`NoEPCLookup` today) |

## 2. Build the home's geometry

| Capability | Status | Where |
|---|---|---|
| **6 prefit UK house types:** Victorian terrace, 1930s semi, 1960s–80s detached, post-2000 terrace, bungalow, flat. Rooms, walls (exterior / party / internal), doors and windows | ✅ | `archetypes.py` |
| **Fit to a home:** floor area, built form (which walls are shared), orientation, ceiling height, room names | ✅ | `build_layout` |
| **Read a floor-plan photo** (Claude Sonnet 5.5, structured outputs): house type, rooms, printed dimensions; drafts only, cached | 🟡 not yet run on real photos (needs `ANTHROPIC_API_KEY`) | `vision.py` |
| **Read a fuse-box photo:** breakers, ratings, labels, circuit types | 🟡 as above | `vision.extract_fuse_board` |
| **User-confirmed geometry:** width and depth (each scaled on its own), ceiling height, front direction, built form; floor-area mismatch vs the EPC flagged | ✅ | `geometry.py`, viewer Geometry tab |
| Edit individual room sizes | ❌ | rooms keep the house type's proportions |

## 3. Turn the home into physics inputs

| Capability | Status | Where |
|---|---|---|
| **All 27 `TwinParams`** with the source of every value (EPC → layout → RdSAP default → SAP formula → assumption) | 🟡 RdSAP defaults marked `to_verify` | `params.py`, `rdsap_defaults.json` |
| **U-values per element:** walls, roof (loft depth), floor (ISO 13370), windows (by glazing), doors | 🟡 | `params.py` |
| **Ventilation** (air changes from age and draught proofing) | 🟡 assumption | `params.py` |
| **Separate hot-water system:** combi / cylinder / immersion / heat-pump cylinder / instant electric, with its own fuel, efficiency, cylinder loss and electric showers | ✅ | `household.hot_water_system` |
| **Thermal mass:** heat capacity from construction × layout areas, SAP thermal-mass parameter and class | 🟡 κ values `to_verify` | `thermal.py` |
| **Occupancy and heating behaviour** (user answers, else SAP defaults with wider uncertainty) | ✅ | `household.py` |
| **Appliances → consumption:** 19 types, size questions, kWh/yr, fuel, 12-month and 24-hour schedules, matching internal heat gains, by category (baseload, gas cooking, showers, plug-in heaters, outdoor) | 🟡 typical estimates, not yet cited | `appliances.py`, `appliance_energy.json` |
| **Weather** aligned to the meter months; missing values filled and flagged | 🟡 synthetic until Part 0's `load_weather` | `weather.py` |
| **Input provenance:** per input, its source, status (confirmed > measured > recorded > stated > estimated > to_verify > assumption > default > synthetic), confirmed flag and reference; share of the home's heat loss by status | ✅ | `provenance.py` |
| **Calibration priors** whose width follows provenance, plus a suggested likelihood and a readiness check (prior-only / weak / ready; what monthly data can and can't identify) | ✅ | `inputs.py` |
| **Physics inputs package:** everything above in one JSON per home, ready for the physics and calibration teams | ✅ | `inputs.build_physics_inputs`, `examples/physics_inputs_SH*.json` |

## 4. Simulate

| Capability | Status | Where |
|---|---|---|
| **Physics seam:** `run_physics(params, weather, mode, fuel)`. The teammate's `physics.py` is picked up automatically, else the stand-in | ✅ seam · 🟡 stand-in physics | `physics_api.py`, `physics_stub.py` |
| **Stand-in physics:** monthly heat balance with thermal mass (time constant, SAP gain utilisation, cooling while the heating is off), per-fuel energy, PV, hot water, schedules | 🟡 | `physics_stub.py` |
| **Run every posterior sample at once** (`jax.vmap`); standard mode (SAP-style) for the rating | ✅ | `simulate.py` |
| **Physics contract tests** any physics model must pass (outputs, balances, monotonicity, jit/vmap/grad) | ✅ | `tests/test_physics_contract.py` |
| **Bayesian calibration** on meter readings | ⏳ calibration team | inputs ready in the package |

## 5. Retrofits and what-ifs

| Capability | Status | Where |
|---|---|---|
| **Apply any of the 16 catalogue retrofits** as parameter changes (provisional key translation documented) | ✅ | `retrofits.py` |
| **Re-simulate a whole package** (never summed): before / after / per-sample savings for gas, electricity, £ (unit rates) and CO₂ (DESNZ factors); increases shown as negative savings | 🟡 stand-in physics | `scenarios.py` |
| **Which retrofits can apply** to this home's geometry (no loft in a flat, no cavity fill on solid walls), with catalogue costs and cost status | ✅ | `scene.retrofit_options` |
| Ranking, packages, feasibility rules, post-retrofit rating | ⏳ Parts 2 and 5 | |

## 6. Outputs per home

| Capability | Status | Where |
|---|---|---|
| **Element heat loss** (walls, roof, floor, windows, doors, ventilation) with ranges, labelled estimated | 🟡 | `twin.element_heat_loss_from_run` |
| **Per-room numbers:** heat loss, heating, appliances, lighting, running cost; rooms always add up to the whole house | 🟡 | `rooms.py` |
| **`GreenDigitalTwin`:** implements the `DigitalTwin` interface the other parts call; calibrate is prior-only without meters | ✅ | `twin.py` |
| Half-hourly electricity profile | ❌ ownership open | |

## 7. The 3D twin (viewer)

| Capability | Status | Where |
|---|---|---|
| **Distinct look per house type**, real CC0 textures in metres (brick, render, roughcast, slate, clay tile, wood and tile floors, lawn), soft shadows, sky, reflections | ✅ | `web/twin-viewer` |
| **Click a room to isolate it** (cutaway walls), exploded floors, heat-loss glow with a colour-blind-safe legend | ✅ | `House.tsx` |
| **Appliances:** add with size questions, drag (never overlapping), rotate, remove, export | ✅ | `Appliances.tsx`, `Panel.tsx` |
| **Retrofits in 3D:** loft layer, PV on the south slope, heat pump, glazing, wall insulation, draught seals, LEDs, storage heaters, battery; before/after toggle; live re-simulated numbers and the saving | ✅ visuals · 🟡 numbers | `Retrofits.tsx`, `RetrofitPanel.tsx` |
| **Confirm geometry** in the viewer, with input-quality bar | ✅ | `GeometryCard.tsx` |
| **Live re-runs** via the dev twin server (~1 s); offline fallback to precomputed single retrofits | ✅ | `scripts/twin_server.py` |

## 8. Many homes: the demo city and business insights

| Capability | Status | Where |
|---|---|---|
| **Demo city:** 500 homes over real Manchester postcode districts, in a few batched physics calls (~2 s) | 🟡 synthetic homes until EPC data; stand-in physics | `city.py`, `scripts/build_demo_city.py`, `artifacts/demo_city/v1/` |
| **Insights:** self-funding loans, price stress test, carbon per £1k lent, confidence-tiered offers, street deals, greening evidence with loan-term sensitivity | 🟡 | `city.py`, `docs/business_insight.md` |
| **Area roll-ups** in the `DashboardArea` contract (+ proposed extensions) | ✅ | `areas.json` |
| Rentals and rent uplift, value uplift, score map, cold-year and income stress test | ⏳ Parts 0, 2, 3, 4, 5 | plan in `docs/business_insight.md` |
| Lloyds dashboard UI | ⏳ dashboard owner | reads `areas.json` |

---

## 9. Run it

```bash
uv run python scripts/run_twin.py                 # SH1-SH3 end to end in the terminal
uv run python scripts/make_twin_examples.py       # physics inputs packages + showcase scenes
uv run python scripts/build_demo_city.py          # demo city -> artifacts/demo_city/v1
uv run python scripts/twin_server.py              # live re-runs for the viewer (port 8765)
cd web/twin-viewer && npm install && npm run dev  # 3D viewer at http://localhost:5173
uv run pytest -q                                  # all tests
```

## 10. What would make the numbers real

1. **Physics:** the teammate's `greentwin/twin/physics.py`. It's auto-detected; run `tests/test_physics_contract.py` against it.
2. **Calibration:** the team's Bayesian calibration on real meter readings, from the physics inputs package.
3. **Data (Part 0):** EPC lookup by certificate number, real weather by district and month, EPC records for the city.
4. **Sources:**
   - verify the RdSAP defaults (p3 already verified the RdSAP 10 loft table)
   - cite the appliance figures
   - confirm the BUS grant, a PV export tariff and the rental-standard status
5. **Keys:** `ANTHROPIC_API_KEY` (photo reading) and `EPC_API_TOKEN` (EPC data), from the team lead.
