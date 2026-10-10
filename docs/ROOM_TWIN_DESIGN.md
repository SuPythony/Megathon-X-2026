# Room-level Digital Twin: Design

Owner: Satyarth (Part 1, digital twin). Status: **draft, proposed**. Nothing in `contracts/` has been changed yet. The contract additions in §6 need team sign-off first.
Based on commit `38942ee` and contracts `0.4.0`. Companion spec for the physics teammate: `docs/digitaltwinvariables.md`.

---

## 1. What changes

The twin is now built from three user inputs:

1. **Floor plan photo**
2. **Fuse box (consumer unit) photo**
3. **High-energy appliances, entered room by room**: the user picks a type and answers basic size questions.

In the demo, the user can **click any room in the 3D house and view it on its own**, with that room's heat loss and appliance energy.

What does **not** change: whole-house physics and Bayesian calibration stay the source of truth (`run_physics`, `calibrate`). Room-level numbers are an **allocation** of the calibrated whole-house totals (§5). No multi-zone physics.

---

## 2. Pipeline

```
 floor plan photo ──► [A] Vision LLM ──► FloorPlanDraft (pixels, labels, confidence)
                                            │
                                            ▼
                                [B] Normaliser (deterministic)  ◄── one known dimension from user
                                            │                        (wall length or total area)
                                            ▼
                                FloorPlan (metres, rooms, walls, openings)
                                            │
 fuse box photo ──► [A] Vision LLM ──► FuseBoardDraft (circuits, ratings, raw labels)
                                            │
 per-room appliance form ──► ApplianceEntry[] ──► [C] Appliance catalogue (watts, typical use)
                                            │
                                            ▼
                     [D] User confirms everything (2D editor + review screen)
                                            │
                                            ▼
                     [E] Twin builder
                         ├─ params_from_property: areas from plan, baseload prior from appliances,
                         │                         hot-water fuel from shower/immersion
                         ├─ calibrate / simulate (whole house, unchanged)
                         ├─ room allocation (§5)
                         └─ scene(): TwinScene with rooms, walls, openings, appliances
                                            │
                                            ▼
                     [F] 3D viewer (react-three-fiber): click room → isolate → room panel
```

### Rules
1. **Vision produces drafts only.** Nothing reaches the twin until the user confirms it (same rule as the existing `CaptureResult`).
2. **Vision never produces energy numbers.** It reads geometry, labels and ratings. Watts and usage come from the sourced appliance catalogue (§4).
3. **The LLM doesn't do geometry maths.** It returns approximate pixel coordinates and labels. Scaling to metres, closing polygons, snapping walls and classifying external walls is deterministic code in the normaliser [B], which is testable and repeatable.
4. **Every field carries a confidence.** Low-confidence fields go to `needs_confirmation` and are highlighted in the editor.

---

## 3. Vision stage [A]

### 3.1 Model and API

| Item | Choice | Why |
|---|---|---|
| Model | `claude-opus-5-5` | Best vision accuracy for dense plans and small handwritten fuse labels. |
| Output | Structured outputs: `client.messages.parse(..., output_format=<PydanticModel>)` | Returns a validated Pydantic object; no hand JSON parsing. |
| Thinking | Adaptive (default on Opus 5.5). Effort `high` for floor plans, `medium` for fuse boxes. | Opus 5.5's default effort is `medium`, so set it explicitly. |
| Refusal handling | Check `stop_reason` before reading the output; enable server-side fallbacks (`fallbacks: "default"`) | `refusal` / `max_tokens` responses may not match the schema. |
| Caching | Store each response in `cache/llm/` keyed by image hash + prompt version | Same image → same draft; demo runs offline from cache. |

**Correction for `AGENTS.md`:** it names `claude-haiku-5-5`, which isn't a real model ID. The current Haiku is `claude-haiku-4-5`. If cost matters later, measure Haiku 4.5 against Opus 5.5 on our labelled set (§8) before switching.

Schema notes for structured outputs: no recursive schemas, and no `minimum`/`maximum`/`minLength`. The Python SDK strips those and validates them client-side. Every object needs `additionalProperties: false`, which Pydantic models get automatically through `.parse()`.

### 3.2 `FloorPlanDraft` (vision output; internal to Part 1, not a shared contract)

```python
class PixelPoint(BaseModel):
    x: float; y: float                      # image pixels, origin top-left

class DraftOpening(BaseModel):
    kind: Literal["door", "window", "patio_door"]
    wall_ref: str                           # id of the DraftWall it sits on
    position_px: PixelPoint                 # centre
    width_px: float
    confidence: float

class DraftWall(BaseModel):
    wall_id: str
    start: PixelPoint; end: PixelPoint
    exterior_guess: bool                    # vision's guess; normaliser recomputes
    confidence: float

class DraftRoom(BaseModel):
    room_id: str
    label: str                              # as printed, e.g. "Kitchen/Diner"
    room_type: Literal["living", "kitchen", "bedroom", "bathroom", "wc", "hall",
                       "landing", "utility", "dining", "study", "garage", "conservatory", "other"]
    storey: int
    polygon_px: list[PixelPoint]
    printed_dimensions: Optional[str]       # e.g. "4.20m x 3.10m" if written on the plan
    confidence: float

class FloorPlanDraft(BaseModel):
    storeys_detected: int
    rooms: list[DraftRoom]
    walls: list[DraftWall]
    openings: list[DraftOpening]
    scale_hint: Optional[str]               # scale bar text or printed dimension it found
    compass_north_deg: Optional[float]      # if a north arrow exists
    notes: list[str]                        # what it was unsure about
```

`printed_dimensions` and `scale_hint` matter: estate-agent plans usually print room sizes, which gives the normaliser a scale for free.

### 3.3 `FuseBoardDraft` (vision output)

```python
class DraftCircuit(BaseModel):
    position: int                           # left-to-right slot number
    device: Literal["mcb", "rcbo", "rcd", "main_switch", "fuse", "spd", "unknown"]
    rating_a: Optional[int]                 # e.g. 6, 16, 32, 40, 45
    curve: Optional[Literal["B", "C", "D"]]
    label_raw: Optional[str]                # exactly as written, e.g. "Upstairs skts"
    label_guess: Optional[str]              # normalised, e.g. "upstairs sockets"
    circuit_kind_guess: Literal["lighting", "sockets", "ring", "cooker", "shower", "immersion",
                                "heating", "ev_charger", "heat_pump", "storage_heaters",
                                "smoke_alarm", "spare", "other", "unknown"]
    rooms_guess: list[str]                  # room labels it may serve
    confidence: float

class FuseBoardDraft(BaseModel):
    board_type: Literal["split_load_rcd", "all_rcbo", "single_rcd", "rewireable_fuses", "unknown"]
    circuits: list[DraftCircuit]
    main_switch_rating_a: Optional[int]
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

Stack (from the plan): React + Vite + TypeScript, react-three-fiber + drei, Zustand, `contracts/types.ts`.

- **House view:** each room's polygon extruded to its ceiling height, per storey. Exterior walls tinted by heat-loss share, windows as glass panes, roof as a separate toggleable mesh. Storeys can be exploded or stacked.
- **Click a room → isolate:** other rooms fade to ~10% opacity, the camera eases to frame the room, and the roof and upper storeys hide when the room is below them.
- **Room panel:**
  - heat loss (W/K + share + interval)
  - which walls and windows lose the most
  - each appliance's kWh/yr and £/yr
  - heating £/yr
  - retrofits that apply to this room (from `RetrofitSlot`)
- **Appliances:** simple icon meshes placed inside the room polygon (auto-layout along walls; no exact positions needed).
- **Labels:** "Estimated from floor plan and home type; only whole-house totals are calibrated to bills."
- **The scene never calculates.** All numbers come from `TwinScene.room_energy` / `ScenarioResult`.

---

## 8. Evaluation and demo safety

- **Floor plan:** room-count accuracy, room-area error (%) after scaling, exterior-wall classification accuracy. Test set: CubiCasa5k samples + our own plans.
- **Fuse box:** rating read accuracy, circuit-kind accuracy, on 10–20 labelled photos.
- **Share of fields the user corrected**: tracks how much the review step is doing.
- **Demo:** one prepared floor plan + fuse box photo with **cached, hand-checked** drafts in `cache/llm/`. Live vision runs, but the demo never depends on it.

---

## 9. Build order (thinnest path first)

1. **Contracts:** add §6 models (after sign-off) + regenerate `types.ts` / JSON schema.
2. **Hand-written `FloorPlan` JSON** for the demo house → `scene()` → 3D viewer with room isolation. **No vision yet.**
3. **Appliance catalogue** + per-room form → baseload prior.
4. **Room allocation** on the stub `run_physics` → `RoomEnergy` → room panel.
5. **Normaliser [B]:** pixels → metres, wall snapping, exterior classification. Unit tests on synthetic drafts.
6. **Vision [A]** for floor plan, then fuse box. Structured outputs + cache.
7. **2D confirm editor** (react-konva) + fuse-box review screen.
8. Swap the stub physics for the teammate's `run_physics`.

---

## 10. Open questions

1. Is a plan without printed dimensions acceptable? Then the user must type one wall length.
2. Multi-storey plans on one photo vs one photo per storey?
3. Does the EV charger go through physics (new output) or a separate electricity term?
4. Who owns the 2D confirm editor? It's in the UI block (B7) in the current plan.
5. Privacy: floor plans and fuse box photos are interior data, so the consent + retention note goes in `docs/RESPONSIBLE_AI.md`.
