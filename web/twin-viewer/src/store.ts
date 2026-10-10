import { create } from 'zustand'
import energyJson from './data/appliance_energy.json'
import visualsJson from './data/appliance_visuals.json'
import { bbox, clampToRoom, findFree, findWallSpot, overlaps } from './geometry'
import type {
  ApplianceQuestion, AppliancePlacement, ApplianceVisual, EnergyOverlay, RoomScene, SceneAppliance, ScenarioView, Vec3,
} from './types'

const sceneModules = import.meta.glob<RoomScene>('./data/scenes/*.scene.json', { eager: true, import: 'default' })
// Keyed by file name: archetype examples ('flat_2bed') and showcase homes with energy ('SH1').
export const EXAMPLE_SCENES: Record<string, RoomScene> = Object.fromEntries(
  Object.entries(sceneModules).map(([path, s]) => [path.split('/').pop()!.replace('.scene.json', ''), s]),
)
export const VISUALS = (visualsJson as { types: Record<string, ApplianceVisual> }).types
export const QUESTIONS: Record<string, ApplianceQuestion[]> = Object.fromEntries(
  Object.entries((energyJson as unknown as { types: Record<string, { questions: ApplianceQuestion[] }> }).types)
    .map(([k, v]) => [k, v.questions]),
)

export function defaultAnswers(type: string): Record<string, string> {
  return Object.fromEntries((QUESTIONS[type] ?? []).map((q) => [q.key, q.default]))
}

export type TwinSource = 'live' | 'precomputed' | 'unavailable' | 'idle'

interface ViewerState {
  scenes: Record<string, RoomScene>
  sceneKey: string
  appliances: SceneAppliance[]
  selectedRoom: string | null
  selectedAppliance: string | null
  dragging: boolean
  showRoof: boolean
  explode: boolean
  heatTint: boolean
  notice: string | null
  retrofitIds: string[]
  view: 'before' | 'after'
  scenario: ScenarioView | null
  baselineRooms: EnergyOverlay | null
  twinSource: TwinSource
  twinBusy: boolean
  viewNonce: number
  panelTab: 'home' | 'retrofits' | 'geometry' | 'data'
  setPanelTab: (t: 'home' | 'retrofits' | 'geometry' | 'data') => void
  resetView: () => void
  setScene: (key: string) => void
  addScene: (scene: RoomScene) => void
  selectRoom: (roomId: string | null) => void
  selectAppliance: (id: string | null) => void
  setDragging: (on: boolean) => void
  toggleRoof: () => void
  toggleExplode: () => void
  toggleHeatTint: () => void
  toggleRetrofit: (id: string) => void
  setView: (v: 'before' | 'after') => void
  runTwin: () => Promise<void>
  updateAnswers: (id: string, answers: Record<string, string>) => void
  replaceScene: (scene: RoomScene) => void
  moveAppliance: (id: string, x: number, y: number) => void
  rotateAppliance: (id: string, by: number) => void
  removeAppliance: (id: string) => void
  addAppliance: (type: string, roomId: string, answers?: Record<string, string>) => void
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
  heatTint: false,
  notice: null,
  retrofitIds: [],
  view: 'before',
  scenario: null,
  baselineRooms: EXAMPLE_SCENES[firstKey]?.energy ?? null,
  twinSource: 'idle',
  twinBusy: false,
  viewNonce: 0,
  panelTab: 'home',
  setPanelTab: (t) => set({ panelTab: t }),
  resetView: () => set((s) => ({ viewNonce: s.viewNonce + 1 })),

  setScene: (key) =>
    set((s) => ({
      sceneKey: key, appliances: s.scenes[key].appliances, selectedRoom: null, selectedAppliance: null, notice: null,
      retrofitIds: [], view: 'before', scenario: null, baselineRooms: s.scenes[key].energy ?? null, twinSource: 'idle',
    })),
  addScene: (sc) => {
    const key = `${sc.archetype_id}:${sc.property_id}`
    set((s) => ({ scenes: { ...s.scenes, [key]: sc } }))
    get().setScene(key)
  },
  selectRoom: (roomId) => set((s) => ({ selectedRoom: roomId, selectedAppliance: null, notice: null, panelTab: roomId ? 'home' : s.panelTab })),
  selectAppliance: (id) => set({ selectedAppliance: id }),
  setDragging: (on) => set({ dragging: on }),
  toggleRoof: () => set((s) => ({ showRoof: !s.showRoof })),
  toggleExplode: () => set((s) => ({ explode: !s.explode })),
  toggleHeatTint: () => set((s) => ({ heatTint: !s.heatTint })),

  // Selecting a retrofit drops the ones it excludes (e.g. double vs triple glazing). Packages are re-simulated.
  toggleRetrofit: (id) =>
    set((s) => {
      const options = scene(s).retrofits ?? []
      if (s.retrofitIds.includes(id)) {
        const ids = s.retrofitIds.filter((r) => r !== id)
        return { retrofitIds: ids, view: ids.length ? s.view : 'before' }
      }
      const opt = options.find((o) => o.retrofit_id === id)
      const excluded = new Set([...(opt?.excludes ?? []), ...options.filter((o) => o.excludes.includes(id)).map((o) => o.retrofit_id)])
      return { retrofitIds: [...s.retrofitIds.filter((r) => !excluded.has(r)), id], view: 'after' }
    }),
  setView: (v) => set({ view: v }),

  // Live re-simulation from the dev twin server; offline, the precomputed scenarios in the scene JSON.
  runTwin: async () => {
    const s = get()
    const sc = scene(s)
    if (!sc.energy) return
    const key = [...s.retrofitIds].sort().join('+')
    const appsChanged = applianceKey(s.appliances) !== applianceKey(sc.appliances)
    set({ twinBusy: true })
    try {
      const res = await fetch('/api/twin/run', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          property_id: sc.property_id, retrofit_ids: s.retrofitIds, appliances: exportPlacements(sc, s.appliances),
          geometry: sc.geometry_source === 'user_confirmed' ? sc.geometry : null,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json() as { baseline_rooms: EnergyOverlay; scenario: ScenarioView }
      if (get().sceneKey !== s.sceneKey) return
      set({ scenario: data.scenario, baselineRooms: data.baseline_rooms, twinSource: 'live' })
    } catch {
      if (get().sceneKey !== s.sceneKey) return
      const pre = !appsChanged ? sc.scenarios?.[key] : undefined
      set(pre
        ? { scenario: pre, baselineRooms: sc.energy ?? null, twinSource: 'precomputed' }
        : { scenario: null, baselineRooms: appsChanged ? null : sc.energy ?? null, twinSource: 'unavailable' })
    } finally {
      set({ twinBusy: false })
    }
  },

  // A rebuilt scene (e.g. confirmed geometry) replaces the current one; the package re-runs on it.
  replaceScene: (sc) =>
    set((s) => ({
      scenes: { ...s.scenes, [s.sceneKey]: sc }, appliances: sc.appliances, baselineRooms: sc.energy ?? null,
      scenario: null, selectedAppliance: null,
    })),

  updateAnswers: (id, answers) =>
    set((s) => ({ appliances: s.appliances.map((a) => (a.appliance_id === id ? { ...a, answers } : a)) })),

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
  addAppliance: (type, roomId, answers) =>
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
        icon: vis.icon, answers: answers ?? defaultAnswers(type),
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

/** What changes the energy (not the position): used to tell if appliances differ from the scene's own. */
export function applianceKey(apps: SceneAppliance[]): string {
  return JSON.stringify(apps.map((a) => [a.appliance_id, a.type, a.room_id, a.answers ?? {}]))
}

/** The overlay for the current view: after the package when showing 'after', else the baseline. */
export function currentRooms(s: Pick<ViewerState, 'view' | 'scenario' | 'baselineRooms'>): EnergyOverlay | null {
  return s.view === 'after' && s.scenario ? s.scenario.rooms_after : s.baselineRooms
}

/** The user's appliances in the input format of scene.py's build_scene (u/v within the room). */
export function exportPlacements(sc: RoomScene, apps: SceneAppliance[]): AppliancePlacement[] {
  return apps.map((a) => {
    const base = { appliance_id: a.appliance_id, type: a.type, room_id: a.room_id, label: a.label,
      rotation_deg: a.rotation_deg, answers: a.answers ?? {} }
    if (a.placement === 'auto') return { ...base, room_id: a.room_id ?? 'outside' } // backend re-places it
    if (!a.room_id) return { ...base, room_id: 'outside', x_m: a.position_m[0], y_m: a.position_m[1] }
    const room = sc.rooms.find((r) => r.room_id === a.room_id)!
    const [x0, y0, x1, y1] = bbox(room)
    const round = (v: number) => Math.round(v * 1000) / 1000
    return { ...base, u: round((a.position_m[0] - x0) / (x1 - x0)), v: round((a.position_m[1] - y0) / (y1 - y0)) }
  })
}
