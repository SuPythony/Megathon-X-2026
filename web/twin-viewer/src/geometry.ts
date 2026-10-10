import * as THREE from 'three'
import { planarUV } from './materials'
import type { SceneAppliance, SceneOpening, SceneRoom, SceneWall, Side, Vec2, Vec3 } from './types'

export const EXPLODE_GAP_M = 2.0
export const WALL_THICKNESS = { exterior: 0.25, party: 0.25, internal: 0.08 } as const

// Plan (x, y, z-up) -> three (x, y-up, z). Plan y maps to three -z so the plan stays right-handed on screen:
// seen from the street (three +z looking towards -z), plan +x is on the viewer's right.
export function toThree(x: number, y: number, z: number): [number, number, number] {
  return [x, z, -y]
}

/** Three point -> plan (x, y). */
export function toPlan(v: { x: number; z: number }): [number, number] {
  return [v.x, -v.z]
}

// Plan rotation (0 = faces -y, the street) -> three rotation about Y, for objects whose front is local +z.
export function rotationY(rotationDeg: number): number {
  return THREE.MathUtils.degToRad(rotationDeg)
}

export function bbox(room: SceneRoom): [number, number, number, number] {
  const xs = room.polygon_m.map((p) => p[0])
  const ys = room.polygon_m.map((p) => p[1])
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
}

export function halfExtents(w: number, d: number, rotDeg: number): [number, number] {
  const r = THREE.MathUtils.degToRad(rotDeg)
  return [
    Math.abs((w / 2) * Math.cos(r)) + Math.abs((d / 2) * Math.sin(r)),
    Math.abs((w / 2) * Math.sin(r)) + Math.abs((d / 2) * Math.cos(r)),
  ]
}

export function clampToRoom(room: SceneRoom, app: SceneAppliance, x: number, y: number): [number, number] {
  const [x0, y0, x1, y1] = bbox(room)
  const [hx, hy] = halfExtents(app.size_m[0], app.size_m[1], app.rotation_deg)
  const cx = x1 - x0 >= 2 * hx ? Math.min(Math.max(x, x0 + hx), x1 - hx) : (x0 + x1) / 2
  const cy = y1 - y0 >= 2 * hy ? Math.min(Math.max(y, y0 + hy), y1 - hy) : (y0 + y1) / 2
  return [cx, cy]
}

const GAP_M = 0.05

/** True when two appliances' floor footprints overlap (with a small gap). Height is ignored: nothing stacks. */
export function overlaps(a: SceneAppliance, b: SceneAppliance): boolean {
  const [ahx, ahy] = halfExtents(a.size_m[0], a.size_m[1], a.rotation_deg)
  const [bhx, bhy] = halfExtents(b.size_m[0], b.size_m[1], b.rotation_deg)
  return (
    Math.abs(a.position_m[0] - b.position_m[0]) < ahx + bhx + GAP_M - 1e-6 &&
    Math.abs(a.position_m[1] - b.position_m[1]) < ahy + bhy + GAP_M - 1e-6
  )
}

/** Nearest free spot in the room to (x, y) for this appliance, or null if the room is full. Same rule as
 * scene.py's _WallCursor.nearest. */
export function findFree(room: SceneRoom, app: SceneAppliance, others: SceneAppliance[], x: number, y: number): Vec2 | null {
  const [x0, y0, x1, y1] = bbox(room)
  const [hx, hy] = halfExtents(app.size_m[0], app.size_m[1], app.rotation_deg)
  if (x1 - x0 < 2 * hx || y1 - y0 < 2 * hy) return null
  const step = 0.05
  const spots: Vec2[] = []
  for (let cx = x0 + hx; cx <= x1 - hx + 1e-9; cx += step)
    for (let cy = y0 + hy; cy <= y1 - hy + 1e-9; cy += step) spots.push([cx, cy])
  spots.sort((p, q) => (p[0] - x) ** 2 + (p[1] - y) ** 2 - ((q[0] - x) ** 2 + (q[1] - y) ** 2))
  for (const [cx, cy] of spots) {
    const moved = { ...app, position_m: [cx, cy, app.position_m[2]] as Vec3 }
    if (!others.some((o) => overlaps(moved, o))) return [cx, cy]
  }
  return null
}

const FACING: Record<Side, number> = { back: 0, right: 270, front: 180, left: 90 }

/** First free spot along the room's walls (back, left, right, front), facing into the room: the same walk as
 * scene.py's auto-placement. Returns [x, y, rotation] or null when the walls are full. */
export function findWallSpot(room: SceneRoom, app: SceneAppliance, others: SceneAppliance[]): [number, number, number] | null {
  const [x0, y0, x1, y1] = bbox(room)
  const [w, d] = app.size_m
  for (const side of ['back', 'left', 'right', 'front'] as Side[]) {
    const along = side === 'back' || side === 'front' ? x1 - x0 : y1 - y0
    const across = side === 'back' || side === 'front' ? y1 - y0 : x1 - x0
    if (w + 2 * GAP_M > along || d > across) continue
    for (let s = GAP_M; s + w + GAP_M <= along + 1e-9; s += 0.05) {
      const [x, y] =
        side === 'back' ? [x0 + s + w / 2, y1 - d / 2]
          : side === 'front' ? [x1 - s - w / 2, y0 + d / 2]
            : side === 'left' ? [x0 + d / 2, y1 - s - w / 2]
              : [x1 - d / 2, y0 + s + w / 2]
      const cand = { ...app, rotation_deg: FACING[side], position_m: [x, y, app.position_m[2]] as Vec3 }
      if (!others.some((o) => overlaps(cand, o))) return [x, y, FACING[side]]
    }
  }
  return null
}

const INWARD: Record<Side, Vec2> = { front: [0, 1], back: [0, -1], left: [1, 0], right: [-1, 0] }

export function wallSide(wall: SceneWall): Side {
  return wall.wall_id.split(':').pop() as Side
}

export interface WallPiece {
  key: string
  position: [number, number, number] // three coords, centre of the box
  size: [number, number, number] // length, height, thickness
  rotationY: number
  fill?: 'window' | 'door' | 'patio_door'
}

/** Splits one room wall into solid boxes around its openings, plus fills for the openings. Each room draws its
 * own walls inset into the room, so a room can be isolated on its own. */
export function wallPieces(wall: SceneWall, openings: SceneOpening[], baseY: number, footprint: Vec2): WallPiece[] {
  const [sx, sy] = wall.start_m
  const [ex, ey] = wall.end_m
  const full = Math.hypot(ex - sx, ey - sy)
  if (full < 1e-6) return []
  const dx = (ex - sx) / full
  const dy = (ey - sy) / full
  // Internal walls stop short of the outer walls, so they don't show through the facade.
  const onEdge = (x: number, y: number) =>
    Math.abs(x) < 1e-6 || Math.abs(y) < 1e-6 || Math.abs(x - footprint[0]) < 1e-6 || Math.abs(y - footprint[1]) < 1e-6
  // Front/back outer walls stop where the side walls run, so two boxes never share a face at a corner (z-fighting).
  const onSide = (x: number) => Math.abs(x) < 1e-6 || Math.abs(x - footprint[0]) < 1e-6
  const side = wallSide(wall)
  const outerFrontBack = wall.kind !== 'internal' && (side === 'front' || side === 'back')
  const trim = wall.kind === 'internal' ? WALL_THICKNESS.exterior : 0
  const t0 = outerFrontBack ? (onSide(sx) ? WALL_THICKNESS.exterior : 0) : onEdge(sx, sy) ? trim : 0
  const t1 = outerFrontBack ? (onSide(ex) ? WALL_THICKNESS.exterior : 0) : onEdge(ex, ey) ? trim : 0
  const [ax, ay] = [sx + dx * t0, sy + dy * t0]
  const len = full - t0 - t1
  if (len < 1e-3) return []
  const t: number = WALL_THICKNESS[wall.kind]
  const [nx, ny] = INWARD[wallSide(wall)]
  const H = wall.height_m
  const rotY = Math.atan2(dy, dx)

  const box = (key: string, s0: number, s1: number, z0: number, z1: number, inset = t / 2, thick = t,
    fill?: WallPiece['fill']): WallPiece => {
    const mid = (s0 + s1) / 2
    const px = ax + dx * mid + nx * inset
    const py = ay + dy * mid + ny * inset
    return { key, position: toThree(px, py, baseY + (z0 + z1) / 2), size: [s1 - s0, z1 - z0, thick], rotationY: rotY, fill }
  }

  const ops = openings
    .map((o) => ({ o, t: (o.centre_m[0] - ax) * dx + (o.centre_m[1] - ay) * dy }))
    .sort((p, q) => p.t - q.t)
  const pieces: WallPiece[] = []
  let cursor = 0
  ops.forEach(({ o, t: c }, i) => {
    const s0 = Math.max(cursor, c - o.width_m / 2)
    const s1 = Math.min(len, c + o.width_m / 2)
    if (s0 > cursor + 1e-3) pieces.push(box(`${wall.wall_id}:s${i}`, cursor, s0, 0, H))
    if (o.sill_m > 1e-3) pieces.push(box(`${wall.wall_id}:b${i}`, s0, s1, 0, o.sill_m))
    const top = o.sill_m + o.height_m
    if (top < H - 1e-3) pieces.push(box(`${wall.wall_id}:t${i}`, s0, s1, top, H))
    const fillThick = o.kind === 'door' ? 0.05 : 0.02
    pieces.push(box(`${wall.wall_id}:f${i}`, s0, s1, o.sill_m, top, t / 2, fillThick, o.kind))
    cursor = s1
  })
  if (cursor < len - 1e-3) pieces.push(box(`${wall.wall_id}:end`, cursor, len, 0, H))
  return pieces
}

/** Roof slopes and gable ends as triangle soups (three coords), built for ridge along x, swapped for y. */
export function roofGeometry(
  type: 'gable' | 'hip', width: number, depth: number, eaveY: number, rise: number, ridgeAxis: 'x' | 'y',
  overhang = 0.3, tile = 1.6,
): { slopes: THREE.BufferGeometry; gables: THREE.BufferGeometry | null } {
  // Work in (u along ridge, v across); map back to plan (x, y).
  const U = ridgeAxis === 'x' ? width : depth
  const V = ridgeAxis === 'x' ? depth : width
  const o = overhang
  const map = (u: number, v: number, h: number) =>
    ridgeAxis === 'x' ? toThree(u, v, eaveY + h) : toThree(v, u, eaveY + h)
  const A = map(-o, -o, 0), B = map(U + o, -o, 0), C = map(U + o, V + o, 0), D = map(-o, V + o, 0)
  let r1u = -o, r2u = U + o
  if (type === 'hip') {
    r1u = Math.min(V / 2, U / 2)
    r2u = Math.max(U - V / 2, U / 2)
  }
  const R1 = map(r1u, V / 2, rise), R2 = map(r2u, V / 2, rise)
  const tri = (...pts: number[][]) => pts.flat()
  const slopes = [...tri(A, B, R2), ...tri(A, R2, R1), ...tri(C, D, R1), ...tri(C, R1, R2)]
  const ends = [...tri(D, A, R1), ...tri(B, C, R2)]
  const geo = (arr: number[]) => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3))
    g.computeVertexNormals()
    return planarUV(g, tile)
  }
  if (type === 'hip') return { slopes: geo([...slopes, ...ends]), gables: null }
  return { slopes: geo(slopes), gables: geo(ends) }
}

/** Plan direction of north, given the azimuth the front facade faces (180 = south). */
export function northDirection(frontAzimuthDeg: number): Vec2 {
  const a = THREE.MathUtils.degToRad(frontAzimuthDeg)
  return [Math.sin(a), -Math.cos(a)]
}

export function roomStats(room: SceneRoom, walls: SceneWall[], openings: SceneOpening[]) {
  const mine = walls.filter((w) => w.room_id === room.room_id)
  const myOpenings = openings.filter((o) => o.room_id === room.room_id)
  const gross = (k: SceneWall['kind']) => mine.filter((w) => w.kind === k).reduce((s, w) => s + w.length_m * w.height_m, 0)
  const windows = myOpenings.filter((o) => o.kind !== 'door')
  const windowArea = windows.reduce((s, o) => s + o.width_m * o.height_m, 0)
  const doorArea = myOpenings.filter((o) => o.kind === 'door').reduce((s, o) => s + o.width_m * o.height_m, 0)
  return {
    exteriorWallNet: gross('exterior') - windowArea - doorArea,
    partyWall: gross('party'),
    windowCount: windows.length,
    windowArea,
    exteriorSides: mine.filter((w) => w.kind === 'exterior').map((w) => w.side),
  }
}
