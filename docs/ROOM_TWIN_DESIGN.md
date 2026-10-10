# Room-level Digital Twin: Design

Owner: Satyarth (Part 1, digital twin). Status: **draft, proposed**. Nothing in `contracts/` has been changed yet. The contract additions in §6 need team sign-off first.
Based on commit `7220322` and contracts `0.4.0`. Companion spec for the physics teammate: `docs/digitaltwinvariables.md`. What the twin offers the other parts and how to call it: `docs/DIGITAL_TWIN_HANDOFF.md`.

---

## 1. What changes

The twin is now built from these user inputs (`greentwin/twin/intake.py`, `TwinIntake`):

1. **Location**: the user types their postcode, and only the **postcode district** (outward code, e.g. `M14`) is kept. The viewer reduces it in the browser, so the full postcode never leaves the device (AGENTS.md rule 8). The district gives the weather area and local context. Postcodes in Scotland or Northern Ireland get a warning, since our data covers England & Wales.
2. **EPC certificate number** (RRN, 20 digits as `0000-0000-0000-0000-0000`), accepted with or without dashes. The data layer uses it to fetch the home's EPC (floor area, built form, age, fabric, heating), which then drives `fit_layout` and `params_from_property`. "I don't have one" is allowed: the twin then starts from the archetype and floor plan only. **Needs Part 0:** a lookup by RRN; `EPCLookup` is the seam, `NoEPCLookup` the stand-in until then.
3. **Floor plan photo**
4. **Fuse box (consumer unit) photo**, with explicit consent for interior photos.
5. **High-energy appliances, entered room by room**: the user picks a type and answers basic size questions.

Schema: `greentwin/twin/examples/twin_intake.schema.json`. The viewer's "New home" form produces this JSON; processing it needs the twin API (not built yet).

In the demo, the user can **click any room in the 3D house and view it on its own**, with that room's heat loss and appliance energy.

What does **not** change: whole-house physics and Bayesian calibration stay the source of truth (`run_physics`, `calibrate`). Room-level numbers are an **allocation** of the calibrated whole-house totals (§5). No multi-zone physics.

---

## 2. Pipeline

**Prefit archetypes.** We don't reconstruct arbitrary geometry from the photo. We keep a library of prefit twins of standard UK house types (`greentwin/twin/archetypes.py`). The vision model picks the closest one and reads room labels and printed dimensions. Deterministic code then fits the archetype to the home: floor area, ceiling height, built form (which walls are party walls), orientation and room labels. Homes of the same type look alike in 3D, but their parameters (areas, U-values, appliances, behaviour, calibration) are their own.

```
 floor plan photo ──► [A] Vision LLM ──► FloorPlanExtraction
                                          (archetype_id, rooms, printed dims, confidence)
                                            │
                                            ▼
                     [B] fit_layout (deterministic) ◄── Property (EPC floor area, built form)
                                            │
                                            ▼
                     TwinLayout: rooms, walls (exterior / party / internal), openings, envelope areas
                                            │
 fuse box photo ──► [A] Vision LLM ──► FuseBoardExtraction (circuits, ratings, raw labels)
                                            │
 per-room appliance form ──► ApplianceEntry[] ──► [C] Appliance catalogue (watts, typical use)
                                            │
                                            ▼
                     [D] User confirms everything (archetype + labels, circuits, appliances)
                                            │
                                            ▼
                     [E] Twin builder
                         ├─ params_from_property: areas from TwinLayout.envelope(), baseload prior
                         │                         from appliances, hot-water fuel from shower/immersion
                         ├─ calibrate / simulate (whole house, unchanged)
                         ├─ room allocation (§5)
                         └─ scene(): TwinScene with rooms, walls, openings, appliances
                                            │
                                            ▼
                     [F] 3D viewer (react-three-fiber): click room → isolate → room panel
```

### Rules
1. **Vision produces drafts only.** Nothing reaches the twin until the user confirms it (same rule as the existing `CaptureResult`).
2. **Vision never produces energy numbers.** It reads type, labels, dimensions and ratings. Watts and usage come from the sourced appliance catalogue (§4).
3. **The LLM doesn't do geometry.** It chooses an archetype and reads printed text. Scaling, wall classification, openings and envelope areas are deterministic code (`build_layout`, `fit_layout`), which is tested.
4. **Every field carries a confidence.** Low-confidence fields (including the archetype choice) go to `needs_confirmation`. Rooms with no slot in the archetype (e.g. a conservatory) go to `unmatched_rooms` for the user to review.

---

## 3. Vision stage [A]

### 3.1 Model and API

| Item | Choice | Why |
|---|---|---|
| Model | `claude-sonnet-5-5` | Team decision. Good vision quality at Sonnet price ($2 / $10 per MTok). |
| Output | Structured outputs: `client.messages.parse(..., output_format=<PydanticModel>)` | Schema-validated Pydantic object; no hand JSON parsing. Verified against `anthropic` 1.13. |
| Thinking | Adaptive (default on). Effort `high` for floor plans, `medium` for fuse boxes. | Set explicitly per task. |
| Refusal handling | Check `stop_reason` before reading `parsed_output`; `refusal` / `max_tokens` raise `VisionError` and are never cached. | Server-side fallback on Sonnet 5.5 only retries `cyber`/`frontier_llm` declines, which don't apply to floor plans, so it's not enabled. |
| Caching | `cache/llm/vision/<Schema>_<hash>.json`, keyed by model + prompt version + image bytes | Same photo → same draft; demo replays offline; no key needed. |

**Correction for `AGENTS.md`:** it names `claude-haiku-5-5`, which isn't a real model ID. The photo-to-twin row should name `claude-sonnet-5-5`.

Schema notes for structured outputs: no recursive schemas, and no `minimum`/`maximum`/`minLength`. The Python SDK strips those and validates them client-side.

### 3.2 Archetype library (v1)

| `archetype_id` | Type | Storeys | Beds | Reference area | Showcase home |
|---|---|---|---|---|---|
| `victorian_terrace_2bed` | Pre-1919 solid-brick terrace | 2 | 2 | 77 m² | SH1 |
| `interwar_semi_3bed` | 1930s semi, bay window, side hall | 2 | 3 | 91 m² | SH2 (layout match) |
| `postwar_detached_4bed` | 1960s-80s detached | 2 | 4 | 135 m² | |
| `modern_terrace_3bed` | Post-2000 terrace | 2 | 3 | 85 m² | |
| `bungalow_2bed` | Detached bungalow | 1 | 2 | 70 m² | |
| `flat_2bed` | Mid-floor purpose-built flat | 1 | 2 | 60 m² | SH3 |

Each archetype is a rectangular footprint per storey, tiled by rectangular rooms with doors and windows on room walls. Fitting:
- **Area:** printed total > printed room dimensions > EPC floor area > archetype default. Both axes scale equally.
- **Built form → party walls:** detached none; semi and end-terrace one side; mid-terrace both sides; enclosed mid-terrace adds the back. Flats keep their own default. Openings on party walls are dropped.
- **Orientation:** from the plan's north arrow if shown, else south-facing front (180°). Each exterior wall gets an outward azimuth.
- **Room labels:** extracted rooms are matched by storey and type, by closest area when dimensions are printed, otherwise largest first.
- **Output:** `TwinLayout` with rooms (`contracts.Room`), walls, openings and `envelope()`, whose areas use `TwinParams` field names (`wall_area_m2`, `window_area_m2`, `party_wall_area_m2`, `volume_m3`, ...). Also `to_floor_plan()` gives a `contracts.FloorPlan` with `source="parametric"`.

More archetypes are cheap to add: one entry in `ARCHETYPES` plus the tiling test.

### 3.3 `FloorPlanExtraction` (vision output; internal to Part 1)

```python
class ExtractedRoom(BaseModel):
    label: str                              # as printed, e.g. "Kitchen/Diner"
    room_type: RoomType                     # living | kitchen | dining | bedroom | bathroom | wc | hall | ...
    storey: int                             # 0 = ground
    width_m: Optional[float]                # only if printed
    depth_m: Optional[float]                # only if printed
    window_count: Optional[int]
    confidence: float

class FloorPlanExtraction(BaseModel):
    archetype_id: ArchetypeId               # closest prefit archetype
    archetype_confidence: float
    archetype_reason: str
    built_form_guess: Optional[BuiltForm]
    storeys: int
    bedrooms: int
    rooms: list[ExtractedRoom]
    total_floor_area_m2: Optional[float]    # only if printed
    front_azimuth_deg: Optional[float]      # only if a north arrow is shown
    notes: list[str]
```

### 3.4 `FuseBoardExtraction` (vision output)

```python
class ExtractedCircuit(BaseModel):
    position: int                           # left-to-right slot number, from 1
    device: Literal["mcb", "rcbo", "rcd", "main_switch", "fuse", "spd", "unknown"]
    rating_a: Optional[int]                 # e.g. 6, 16, 32, 40, 45
    curve: Optional[Literal["B", "C", "D"]]
    label_raw: Optional[str]                # exactly as written; null if none, never invented
    circuit_kind: Literal["lighting", "sockets", "cooker", "shower", "immersion", "heating",
                          "ev_charger", "heat_pump", "storage_heaters", "smoke_alarm",
                          "spare", "other", "unknown"]
    rooms_guess: list[str]
    confidence: float

class FuseBoardExtraction(BaseModel):
    board_type: Literal["split_load_rcd", "all_rcbo", "single_rcd", "rewireable_fuses", "unknown"]
    main_switch_rating_a: Optional[int]
    circuits: list[ExtractedCircuit]
    notes: list[str]
```

Fuse labels are often missing, faded or handwritten. Expect low confidence, and let the user map circuits to rooms on the review screen.

**What the fuse box tells the twin:**
- Big fixed loads exist: shower (typically 32–45 A), cooker, immersion heater, EV charger, heat pump, storage heaters. These prompt the appliance form ("We saw a 40 A shower circuit. Which bathroom?").
- Fuel hints: an immersion heater or storage heaters suggests electric hot water or heating, cross-checked against the EPC heating fields.
- A circuit-to-room map for the 3D view. Optional: circuits can be drawn as coloured runs per room.

---

## 4. Appliance input [C]

### 4.1 Flow
For each confirmed room, the user adds appliances. For each one:
1. Pick a **type** from a short list. The list is filtered by room type, e.g. kitchen → fridge-freezer, oven, hob, dishwasher, kettle.
2. Answer **1–2 size questions** (below).
3. Usage is **pre-filled with UK typical values**, shown and editable. No usage question by default.

### 4.2 Question set (v1)

| Type | Size questions | Notes |
|---|---|---|
| Fridge-freezer / fridge / freezer | Size: small / medium / large (or litres); age: <5 / 5–10 / >10 years | Age affects consumption a lot |
| Washing machine | Drum size (kg) | |
| Tumble dryer | Type: vented / condenser / heat-pump; drum (kg) | |
| Dishwasher | Slimline / full-size | |
| Electric oven | Single / double | |
| Hob | Gas / electric ceramic / induction | Gas hob → gas, not electricity |
| Electric shower | Rating (kW) | Cross-check with fuse box circuit |
| Immersion heater | Tank size: small / medium / large | Switches hot-water fuel to electricity |
| TV | Screen size (inches) | |
| Desktop PC / gaming console | Type | |
| Plug-in electric heater | Rating (kW) | Space heating, not baseload |
| Dehumidifier | Size: small / large | |
| EV charger | Rating (kW); miles per week | Large; separate from baseload |
| Hot tub | Yes/no | |

### 4.3 Catalogue (proposed: `contracts/appliance_catalogue.json`)

One entry per type: questions, watts (or kWh per cycle) per answer, typical hours/cycles per week, and **source** with a `source_status` like `cost_status` (`sourced | secondary_source | unsourced`). Values come from published UK sources (e.g. Energy Saving Trust and DESNZ household electricity survey figures). **No unsourced numbers shown as fact.** Unsourced entries are labelled "typical estimate".

### 4.4 How appliances feed the physics (no new physics inputs)
- Σ (appliance kWh/day) excluding heaters and EV → prior mean for `baseload_kwh_per_day` (calibrated against meters, so the prior can be off without breaking anything).
- Lighting is not asked per bulb. `lighting_share_of_baseload` keeps its default.
- Electric shower / immersion heater → hot-water fuel = electricity (see the `FuelConfig` question in `digitaltwinvariables.md` §3.3).
- Plug-in heaters → flagged as secondary electric heating.
- EV charger → separate electricity term, kept out of baseload (**open question** for the physics teammate).

---

## 5. Room allocation [E]

Calibrated whole-house totals are split across rooms. **Rooms always sum to the whole house.** Every room number is labelled "estimated".

### 5.1 Heat loss per room
For room *r*, from the confirmed plan:
- External wall area = Σ (exterior wall length in room × ceiling height) − openings
- Window and door area = openings on its exterior walls
- Roof area = room footprint if on the top storey; floor area = footprint if on the ground storey
- Ventilation share = room volume / house volume

`UA_r = Σ U_element × A_element(r) + ventilation_r`, using the same U-values as `TwinParams`.
Then scale: `heat_loss_r = UA_r / Σ UA × calibrated total heat-loss coefficient` (posterior samples → `Distribution` per room).

### 5.2 Electricity per room
`room_appliance_kwh_r = Σ appliances in room r` (catalogue figures).
Scale to the calibrated baseload: `elec_r = room_appliance_kwh_r / Σ room_appliance_kwh × calibrated baseload`, **capped** so a room can't be scaled up by more than ×1.5. The remainder goes to a house-level **"unaccounted (standby, small appliances, lighting)"** bucket, shown in the UI. The cap stops a missing appliance from being blamed on whichever rooms were entered.

### 5.3 Space-heating energy per room
`heating_kwh_r = heat_loss_share_r × calibrated space-heating kWh`. Also gives £/year per room from the tariff.

---

## 6. Contract additions (proposed, additive only)

Nothing existing is renamed or removed. These need team sign-off before they go into `contracts/models.py` / `types.ts` / JSON schema.

| Change | Fields |
|---|---|
| **New** `Wall` | `wall_id`, `room_ids` (1 or 2), `start_m`, `end_m`, `storey`, `exterior: bool`, `orientation_deg` |
| **New** `Opening` | `opening_id`, `wall_id`, `kind` (door / window / patio_door), `width_m`, `height_m`, `sill_m` |
| **Extend** `FloorPlan` | `walls: list[Wall] = []`, `openings: list[Opening] = []`, `north_deg: Optional[float]` |
| **New** `Appliance` | `appliance_id`, `room_id`, `type`, `answers: dict`, `kwh_per_year: Distribution`, `fuel`, `source`, `user_edited: bool` |
| **New** `Circuit` | `position`, `device`, `rating_a`, `label`, `kind`, `room_ids`, `confirmed_by_user` |
| **Extend** `CaptureResult` | `circuits: list[Circuit] = []`, `appliances: list[Appliance] = []` |
| **New** `RoomEnergy` | `room_id`, `heat_loss_w_per_k: Distribution`, `heat_loss_share_pct`, `space_heating_kwh: Distribution`, `appliance_kwh: Distribution`, `bill_gbp: Distribution`, `estimated: bool = True`, `basis: str` |
| **Extend** `SceneSurface` | `room_id: Optional[str] = None`, `wall_id: Optional[str] = None` |
| **Extend** `TwinScene` | `walls`, `openings`, `appliances`, `room_energy: list[RoomEnergy] = []`, `geometry_source` gains `"floor_plan"` (already exists) |

All new list fields default to empty, so existing code and mocks keep validating.

---

## 7. 3D viewer [F]

Stack (from the plan): React + Vite + TypeScript, react-three-fiber + drei, Zustand.

### 7.1 Scene JSON (built, `greentwin/twin/scene.py`)
`build_scene(layout, property_id, appliances)` returns one JSON document (`RoomScene`, version `room_scene/0.1`) that the viewer renders directly. It's internal to Part 1 for now; no shared contract change. Schema: `greentwin/twin/examples/room_scene.schema.json`. One example per archetype: `greentwin/twin/examples/<archetype>.scene.json` (regenerate with `python -m greentwin.twin.scene`).

Contents: rooms (polygon, storey, floor height), walls (exterior / party / internal, outward azimuth), openings (with sill heights), roof (type, pitch, eave and ridge heights), exterior features, faded neighbour blocks, appliances.

Conventions: metres; x across the front, y from front (0) to back; z up. Box positions are bottom-centre. Rotation about z: 0 = front faces -y (street), 90 = +x, 180 = +y, 270 = -x.

### 7.2 Distinct look per house type (`greentwin/twin/house_styles.json`)
| Archetype | Walls | Roof | Features |
|---|---|---|---|
| Victorian terrace | red brick | slate gable | ground-floor bay, two party-wall chimneys, front garden wall, white sash frames |
| 1930s semi | brick below, pebbledash above | clay-tile hip | two-storey bay, canopy porch, chimney on the party side |
| 1960s-80s detached | buff brick | low-pitch concrete-tile gable | attached garage (visual only, unheated), picture windows, dark frames |
| Post-2000 terrace | orange brick | grey gable | canopy porch, white uPVC, no chimney |
| Bungalow | white render | brown hip | enclosed porch, central chimney, garden wall |
| Flat | concrete panel | block above | glass balcony, aluminium frames |

Party walls get faded neighbour blocks (left / right / back, and above / below for flats), so terraces, semis and flats read correctly. A garage on a party side moves to the free side, or is dropped for mid-terraces. Visual only: nothing in the style feeds the physics.

### 7.3 Appliances: custom placement (`greentwin/twin/appliance_visuals.json`)
- Each appliance type has its own size, shape, colours, icon and mounting (floor / counter / wall / outdoor). 19 types.
- **Custom:** the user drops an appliance in a room: `{"appliance_id", "type", "room_id", "u", "v", "rotation_deg"}`. `u`/`v` are 0..1 across the room's width and depth, so placements survive rescaling. Rotation defaults to facing away from the nearest wall. Out-of-room positions are clamped back in, with a warning.
- **No overlaps, ever.** No two appliances share floor space, at any height (nothing is stacked). This holds in the backend and in the viewer.
- **Auto:** no `u`/`v` → placed along free wall space (back, left, right, front), stepping around custom-placed items. When the walls are full, it goes in the free spot nearest the room centre.
- **Custom positions that would overlap** snap to the nearest free spot, with a warning. In the viewer, a drag slides along whatever it bumps into, rotation moves the item if it no longer fits, and a full room refuses new items.
- Worktop items (hob, microwave, kettle) are drawn on their own base cabinet.
- **Outdoor:** `room_id: "outside"`; EV charger auto-mounts on the front wall, hot tub in the garden; custom via `x_m`/`y_m`.
- Unusual rooms (hob in a bedroom) warn but are allowed. Unknown types or rooms raise.
- Schema for the input: `greentwin/twin/examples/appliance_placement.schema.json`.

### 7.4 Viewer behaviour
- **House view:** rooms extruded per storey; exterior walls tinted by heat-loss share once allocation exists; roof toggleable; storeys can be exploded.
- **Click a room → isolate:** other rooms fade to ~10% opacity, the camera frames the room, roof and upper storeys hide when the room is below them.
- **Room panel:** heat loss (W/K + share + interval), worst walls and windows, each appliance's kWh/yr and £/yr, heating £/yr, applicable retrofits.
- **Drag to place:** dragging an appliance writes back `u`/`v`/`rotation_deg`; the backend re-runs `build_scene`.
- **Labels:** the scene's `basis` text is shown wherever room numbers appear.
- **The scene never calculates.** Energy numbers come from the twin's results.

---

## 8. Evaluation and demo safety

- **Floor plan:** archetype-choice accuracy, room-count accuracy, area error (%) after fitting. Test set: estate-agent style plans + CubiCasa5k samples.
- **Fuse box:** rating read accuracy, circuit-kind accuracy, on 10–20 labelled photos.
- **Share of fields the user corrected**: tracks how much the review step is doing.
- **Demo:** one prepared floor plan + fuse box photo with **cached, hand-checked** drafts in `cache/llm/`. Live vision runs, but the demo never depends on it.

---

## 9. Build order (thinnest path first)

1. ✅ **Archetype library + fitting** (`greentwin/twin/archetypes.py`) and **vision extraction** on `claude-sonnet-5-5` with structured outputs + cache (`greentwin/twin/vision.py`). Tests: `tests/test_archetypes.py`.
2. ✅ **Scene JSON** (`greentwin/twin/scene.py`): distinct look per house type, custom + auto appliance placement, examples + JSON Schema. Tests: `tests/test_scene.py`.
2b. ✅ **3D viewer** (`web/twin-viewer`, React + react-three-fiber): renders the scene JSON, click-to-isolate rooms with cutaway walls, exploded floors, drag/rotate/add/remove appliances, export placements for `build_scene`.
3. **Appliance catalogue** + per-room form → baseload prior.
4. **Room allocation** on the stub `run_physics` → `RoomEnergy` → room panel.
5. **Confirm screen:** pick or override the archetype, fix labels, map circuits to rooms.
6. Prepare demo photos; hand-check and commit their cached drafts as an eval snapshot.
7. Swap the stub physics for the teammate's `run_physics`.

---

## 10. Open questions

1. Homes that match no archetype well (extensions, conversions): add archetypes, or let the user pick the closest and accept the visual mismatch?
2. Multi-storey plans on one photo vs one photo per storey?
3. Does the EV charger go through physics (new output) or a separate electricity term?
4. Who owns the confirm screen? It's in the UI block (B7) in the current plan.
5. Privacy: floor plans and fuse box photos are interior data, so the consent + retention note goes in `docs/RESPONSIBLE_AI.md`.
