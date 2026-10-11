import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { toThree } from '../geometry'
import { currentRooms, useViewer } from '../store'
import type { RoomScene } from '../types'
import { HEAT_RAMP, storeyOffset } from './House'

// Animated heat leaving the house: particles drift out of every exterior wall and the roof. Each room's heat loss
// (W/K, from the twin's room overlay for the current Before/After view) is split over its exterior walls by area,
// plus the roof for top-floor rooms with no home above. More particles = more heat lost; colour = loss per m² of
// that surface (same ramp as the heat-loss colours). Ticking a retrofit and switching to After thins the flow.
// Visual only: element shares come from the room model and are estimates (see the overlay's basis label).

const PER_WK = 2.2          // particles per W/K of heat loss
const MAX_PER_SURFACE = 160
const DRIFT_M = 1.4         // how far a particle travels before it restarts
const SPEED = 0.45          // fraction of the drift per second

interface Emitter {
  origin: THREE.Vector3     // one corner of the surface
  u: THREE.Vector3          // first edge
  v: THREE.Vector3          // second edge
  normal: THREE.Vector3     // outward
  count: number
  colour: THREE.Color
}

function ramp(t: number): THREE.Color {
  const x = Math.max(0, Math.min(1, t)) * (HEAT_RAMP.length - 1)
  const i = Math.min(HEAT_RAMP.length - 2, Math.floor(x))
  return new THREE.Color(HEAT_RAMP[i]).lerp(new THREE.Color(HEAT_RAMP[i + 1]), x - i)
}

function emitters(scene: RoomScene, wk: Map<string, number>, explode: boolean): Emitter[] {
  const out: Omit<Emitter, 'colour'>[] = []
  const flux: number[] = []
  const hasAbove = scene.context.some((c) => c.relation === 'above')
  const top = Math.max(...scene.rooms.map((r) => r.storey))
  for (const room of scene.rooms) {
    const loss = wk.get(room.room_id) ?? 0
    if (loss <= 0) continue
    const walls = scene.walls.filter((w) => w.room_id === room.room_id && w.kind === 'exterior')
    const roofArea = room.storey === top && !hasAbove ? room.area_m2 : 0
    const wallArea = walls.reduce((a, w) => a + w.length_m * w.height_m, 0)
    const total = wallArea + roofArea
    if (total <= 0) continue
    const base = storeyOffset(room.storey, explode) + room.floor_z_m
    const cx = room.polygon_m.reduce((a, p) => a + p[0], 0) / room.polygon_m.length
    const cy = room.polygon_m.reduce((a, p) => a + p[1], 0) / room.polygon_m.length
    for (const w of walls) {
      const area = w.length_m * w.height_m
      const share = (loss * area) / total
      const [sx, sy] = w.start_m
      const [ex, ey] = w.end_m
      let [nx, ny] = [ey - sy, -(ex - sx)]
      const len = Math.hypot(nx, ny) || 1
      ;[nx, ny] = [nx / len, ny / len]
      if (((sx + ex) / 2 - cx) * nx + ((sy + ey) / 2 - cy) * ny < 0) {  // point away from the room's centre
        nx = -nx
        ny = -ny
      }
      const o = new THREE.Vector3(...toThree(sx, sy, base))
      out.push({
        origin: o,
        u: new THREE.Vector3(...toThree(ex, ey, base)).sub(o),
        v: new THREE.Vector3(0, w.height_m, 0),
        normal: new THREE.Vector3(...toThree(nx, ny, 0)).normalize(),
        count: Math.min(MAX_PER_SURFACE, Math.round(share * PER_WK)),
      })
      flux.push(share / area)
    }
    if (roofArea > 0) {
      const xs = room.polygon_m.map((p) => p[0])
      const ys = room.polygon_m.map((p) => p[1])
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
      const z = base + room.height_m
      const o = new THREE.Vector3(...toThree(x0, y0, z))
      out.push({
        origin: o,
        u: new THREE.Vector3(...toThree(x1, y0, z)).sub(o),
        v: new THREE.Vector3(...toThree(x0, y1, z)).sub(o),
        normal: new THREE.Vector3(0, 1, 0),
        count: Math.min(MAX_PER_SURFACE, Math.round(((loss * roofArea) / total) * PER_WK)),
      })
      flux.push((loss * roofArea) / total / roofArea)
    }
  }
  const maxFlux = Math.max(1e-9, ...flux)
  return out.map((e, i) => ({ ...e, colour: ramp(0.35 + 0.65 * (flux[i] / maxFlux)) }))
}

export function HeatFlow({ scene }: { scene: RoomScene }) {
  const on = useViewer((s) => s.heatFlow)
  const explode = useViewer((s) => s.explode)
  const selected = useViewer((s) => s.selectedRoom)
  const energy = useViewer(currentRooms)
  const wk = useMemo(() => new Map((energy?.rooms ?? []).map((r) => [r.room_id, r.heat_loss_w_per_k.headline])), [energy])
  const ems = useMemo(() => emitters(scene, wk, explode), [scene, wk, explode])

  // One particle = (emitter, u, v, phase); positions and colours are rewritten every frame.
  const { geometry, seeds } = useMemo(() => {
    const total = ems.reduce((a, e) => a + e.count, 0)
    const seeds = new Float32Array(total * 4)
    let k = 0
    ems.forEach((e, ei) => {
      for (let j = 0; j < e.count; j++, k++) {
        seeds.set([ei, Math.random(), Math.random(), Math.random()], k * 4)
      }
    })
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(total * 3), 3))
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(total * 3), 3))
    return { geometry: g, seeds }
  }, [ems])

  const t = useRef(0)
  const p = useMemo(() => new THREE.Vector3(), [])
  useFrame((_, dt) => {
    if (!on) return
    t.current += Math.min(dt, 0.1)
    const pos = geometry.attributes.position as THREE.BufferAttribute
    const col = geometry.attributes.color as THREE.BufferAttribute
    const n = seeds.length / 4
    for (let k = 0; k < n; k++) {
      const e = ems[seeds[k * 4]]
      const phase = (seeds[k * 4 + 3] + t.current * SPEED) % 1
      p.copy(e.origin).addScaledVector(e.u, seeds[k * 4 + 1]).addScaledVector(e.v, seeds[k * 4 + 2])
        .addScaledVector(e.normal, 0.15 + phase * DRIFT_M)
      p.y += phase * phase * 0.35                     // warm air rises a little as it leaves
      pos.setXYZ(k, p.x, p.y, p.z)
      const fade = Math.sin(Math.PI * phase)          // fade in at the wall, out at the end (additive blend)
      col.setXYZ(k, e.colour.r * fade, e.colour.g * fade, e.colour.b * fade)
    }
    pos.needsUpdate = true
    col.needsUpdate = true
  })

  if (!on || !energy || selected || ems.length === 0) return null
  return (
    <points geometry={geometry} frustumCulled={false} renderOrder={5}>
      <pointsMaterial size={0.11} vertexColors transparent depthWrite={false} blending={THREE.AdditiveBlending} sizeAttenuation />
    </points>
  )
}
