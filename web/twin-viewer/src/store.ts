import { create } from 'zustand'
import visualsJson from './data/appliance_visuals.json'
import { bbox, clampToRoom, findFree, findWallSpot, overlaps } from './geometry'
import type { AppliancePlacement, ApplianceVisual, RoomScene, SceneAppliance, Vec3 } from './types'

const sceneModules = import.meta.glob<RoomScene>('./data/scenes/*.scene.json', { eager: true, import: 'default' })
export const EXAMPLE_SCENES: Record<string, RoomScene> = Object.fromEntries(
  Object.values(sceneModules).map((s) => [s.archetype_id, s]),
)
export const VISUALS = (visualsJson as { types: Record<string, ApplianceVisual> }).types

interface ViewerState {
  scenes: Record<string, RoomScene>
  sceneKey: string
  appliances: SceneAppliance[]
  selectedRoom: string | null
  selectedAppliance: string | null
  dragging: boolean
  showRoof: boolean
  explode: boolean
  notice: string | null
  setScene: (key: string) => void
  addScene: (scene: RoomScene) => void
  selectRoom: (roomId: string | null) => void
  selectAppliance: (id: string | null) => void
  setDragging: (on: boolean) => void
  toggleRoof: () => void
  toggleExplode: () => void
  moveAppliance: (id: string, x: number, y: number) => void
  rotateAppliance: (id: string, by: number) => void
  removeAppliance: (id: string) => void
  addAppliance: (type: string, roomId: string) => void
}

const scene = (s: ViewerState) => s.scenes[s.sceneKey]
const firstKey = Object.keys(EXAMPLE_SCENES).sort()[0]

export const useViewer = create<ViewerState>((set, get) => ({
  scenes: EXAMPLE_SCENES,
  sceneKey: firstKey,
  appliances: EXAMPLE_SCENES[firstKey]?.appliances ?? [],
  selectedRoom: null,
  selectedAppliance: null,
  dragging: false,
  showRoof: true,
  explode: false,
  notice: null,

  setScene: (key) =>
    set((s) => ({ sceneKey: key, appliances: s.scenes[key].appliances, selectedRoom: null, selectedAppliance: null, notice: null })),
  addScene: (sc) => {
    const key = `${sc.archetype_id}:${sc.property_id}`
    set((s) => ({ scenes: { ...s.scenes, [key]: sc } }))
    get().setScene(key)
  },
  selectRoom: (roomId) => set({ selectedRoom: roomId, selectedAppliance: null, notice: null }),
  selectAppliance: (id) => set({ selectedAppliance: id }),
  setDragging: (on) => set({ dragging: on }),
  toggleRoof: () => set((s) => ({ showRoof: !s.showRoof })),
  toggleExplode: () => set((s) => ({ explode: !s.explode })),

  // Dragging slides along whatever it bumps into: try the full move, then each axis alone, else stay put.
  moveAppliance: (id, x, y) =>
    set((s) => {
      const sc = scene(s)
      const a = s.appliances.find((p) => p.appliance_id === id)
      if (!a || !a.room_id) return {}
      const room = sc.rooms.find((r) => r.room_id === a.room_id)!
      const others = s.appliances.filter((o) => o.room_id === a.room_id && o.appliance_id !== id)
      const [cx, cy] = clampToRoom(room, a, x, y)
      const [ox, oy] = a.position_m
      const at = (px: number, py: number) => ({ ...a, placement: 'custom' as const, position_m: [px, py, a.position_m[2]] as Vec3 })
      const next = [at(cx, cy), at(cx, oy), at(ox, cy)].find((c) => !others.some((o) => overlaps(c, o)))
      if (!next) return {}
      return { appliances: s.appliances.map((p) => (p.appliance_id === id ? next : p)) }
    }),
  // Rotating keeps the appliance where it is if it fits, otherwise moves it to the nearest free spot.
  rotateAppliance: (id, by) =>
    set((s) => {
      const sc = scene(s)
      const a = s.appliances.find((p) => p.appliance_id === id)
      if (!a) return {}
      const turned = { ...a, rotation_deg: (((a.rotation_deg + by) % 360) + 360) % 360, placement: 'custom' as const }
      let next: SceneAppliance = turned
      if (a.room_id) {
        const room = sc.rooms.find((r) => r.room_id === a.room_id)!
        const others = s.appliances.filter((o) => o.room_id === a.room_id && o.appliance_id !== id)
        const spot = findFree(room, turned, others, a.position_m[0], a.position_m[1])
        if (!spot) return {} // no room to turn it
        next = { ...turned, position_m: [spot[0], spot[1], a.position_m[2]] }
      }
      return { appliances: s.appliances.map((p) => (p.appliance_id === id ? next : p)) }
    }),
  removeAppliance: (id) =>
    set((s) => ({ appliances: s.appliances.filter((a) => a.appliance_id !== id), selectedAppliance: null })),
  addAppliance: (type, roomId) =>
    set((s) => {
      const sc = scene(s)
      const vis = VISUALS[type]
      const room = sc.rooms.find((r) => r.room_id === roomId)!
      const [x0, y0, x1, y1] = bbox(room)
      const used = new Set(s.appliances.map((a) => a.appliance_id))
      let n = s.appliances.length + 1
      while (used.has(`a${n}`)) n++
      const draft: SceneAppliance = {
        appliance_id: `a${n}`, type, label: vis.label, room_id: roomId, placement: 'custom',
        position_m: [(x0 + x1) / 2, (y0 + y1) / 2, room.floor_z_m + vis.mount_height_m], rotation_deg: 0,
        size_m: [vis.w, vis.d, vis.h], shape: vis.shape, mount: vis.mount, colour: vis.colour, accent: vis.accent,
        icon: vis.icon,
        warnings: vis.rooms.includes(room.room_type) ? [] : [`${vis.label} is unusual in a ${room.room_type}`],
      }
      // New appliances go along free wall space first, then anywhere free nearest the centre; a full room refuses.
      const others = s.appliances.filter((o) => o.room_id === roomId)
      const wall = findWallSpot(room, draft, others)
      const centre = wall ? null : findFree(room, draft, others, (x0 + x1) / 2, (y0 + y1) / 2)
      if (!wall && !centre) return { notice: `No free floor space in ${room.label} for a ${vis.label.toLowerCase()}.` }
      const [px, py, rot] = wall ?? [centre![0], centre![1], 0]
      const app = { ...draft, rotation_deg: rot, position_m: [px, py, draft.position_m[2]] as Vec3 }
      return { appliances: [...s.appliances, app], selectedAppliance: app.appliance_id, notice: null }
    }),
}))

/** The user's appliances in the input format of scene.py's build_scene (u/v within the room). */
export function exportPlacements(sc: RoomScene, apps: SceneAppliance[]): AppliancePlacement[] {
  return apps.map((a) => {
    const base = { appliance_id: a.appliance_id, type: a.type, room_id: a.room_id, label: a.label, rotation_deg: a.rotation_deg }
    if (a.placement === 'auto') return { ...base, room_id: a.room_id ?? 'outside' } // backend re-places it
    if (!a.room_id) return { ...base, room_id: 'outside', x_m: a.position_m[0], y_m: a.position_m[1] }
    const room = sc.rooms.find((r) => r.room_id === a.room_id)!
    const [x0, y0, x1, y1] = bbox(room)
    const round = (v: number) => Math.round(v * 1000) / 1000
    return { ...base, u: round((a.position_m[0] - x0) / (x1 - x0)), v: round((a.position_m[1] - y0) / (y1 - y0)) }
  })
}
