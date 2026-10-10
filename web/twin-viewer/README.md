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

## Notes
- No energy numbers yet: the room panel shows geometry only, until the twin's room allocation is wired in.
- Plan coordinates map to three.js as `(x, z, -y)`, so the plan stays right-handed on screen.
- `src/data/` is generated and git-ignored; edit the source files in `greentwin/twin/` instead.
- Lives in its own folder so it doesn't clash with the main web app in `web/`.
