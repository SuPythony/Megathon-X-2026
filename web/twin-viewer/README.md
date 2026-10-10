# Room twin viewer

3D viewer for GreenTwin's room-level digital twin (Part 1). It renders the scene JSON produced by
`greentwin/twin/scene.py` (`room_scene/0.1`): each standard UK house type with its own look, every room
selectable on its own, and appliances the user can place.

```bash
cd web/twin-viewer
npm install
npm run dev        # copies greentwin/twin/examples/*.scene.json + appliance_visuals.json into src/data first
```

## What it does
- **House view:** pick a house type from the top bar. Toggle the roof, or explode floors to see inside.
- **Room view:** click a room (in 3D or in the side panel). Other rooms fade, upper floors hide, and the walls
  nearest the camera become see-through. `Esc` or "Whole house" goes back.
- **Appliances:** in room view, drag an appliance to move it, rotate it ±90°, remove it, or add one from the list.
  "Export appliance placements" downloads the input format of `build_scene` (`u`/`v` = position as a fraction of
  the room; untouched auto-placed items are exported without a position so the backend places them again).
- **Load scene JSON:** open any scene produced by `build_scene`.

## Retrofits and live numbers
- The **Retrofits** card lists the catalogue retrofits. The geometry check greys out ones that can't apply (no loft
  in a flat, cavity fill on solid walls). Ticking one shows it on the house, and **Before / After** switches views:
  loft layer, PV on the south-facing slope, outdoor heat-pump unit, glazing colour, wall insulation layers, draught
  seals, LED lights, storage heaters, battery.
- Numbers: the package is **re-simulated as a whole** (never summed). The table shows before, after and the change
  per year for gas, electricity, running cost (unit rates only) and carbon, with 90% ranges. Increases are red.
- **Live:** start the dev twin server, `uv run python scripts/twin_server.py` (port 8765; Vite proxies `/api`). It
  re-runs any package and any appliance changes in about 1 s.
- **Offline:** single retrofits come from scenarios precomputed into the scene JSON. Packages and changed
  appliances need the server, and the panel says so.

## Geometry and input quality
The **Geometry** card (house view, showcase homes) shows whether the layout is a house-type estimate, a photo draft or
confirmed by you. You can correct the width, depth, ceiling height, which way the front faces and the built form,
then press **Confirm geometry**. The twin server rebuilds the house and its energy on your dimensions, and later
retrofit runs use them. The bar shows how much of the home's heat loss rests on confirmed or recorded inputs vs
defaults.

## Appliance questions
Adding an appliance asks its size questions (`appliance_energy.json`: fridge size and age, drum kg, TV inches, shower
kW...). The answers can be edited for the selected appliance. Answers are exported with placements and drive the
appliance's energy, schedule and heat gains.

## Look and feel
- Full-screen 3D with floating glass panels. A left toolbar handles roof, exploded floors, heat-loss colours, back to
  the whole house and reset view; the right panel has tabs for overview / room, retrofits, geometry and data.
- Real materials: CC0 photo textures from Poly Haven (`public/textures/`, see `CREDITS.md`) mapped in metres, so
  bricks and tiles keep real size. Soft sun shadows, sky, a procedural environment for reflections on glass and PV
  (no runtime downloads).
- No jitter: outer walls are trimmed at corners (no coplanar faces), the shadow camera is fixed and fitted,
  rendering is on demand, and components subscribe only to the state they use.
- Retrofits drop gently into place. Heat-loss mode makes outer walls glow on a colour-blind-safe cividis scale,
  with a legend.

## Notes
- Showcase scenes (SH1-SH3) carry per-room energy from the twin; archetype examples show geometry only.
- Plan coordinates map to three.js as `(x, z, -y)`, so the plan stays right-handed on screen.
- `src/data/` is generated and git-ignored; edit the source files in `greentwin/twin/` instead.
- Lives in its own folder so it doesn't clash with the main web app in `web/`.
