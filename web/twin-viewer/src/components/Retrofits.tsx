import { useFrame } from '@react-three/fiber'
import { type JSX, type ReactNode, useRef } from 'react'
import * as THREE from 'three'
import { useShallow } from 'zustand/react/shallow'
import { WALL_THICKNESS, bbox, toThree, wallPieces, wallSide } from '../geometry'
import { grilleTexture, insulationTexture, pvTexture } from '../materials'
import { useViewer } from '../store'
import type { RoomScene, SceneRoom, Side } from '../types'
import { storeyOffset } from './House'

// What each retrofit looks like in the 'after' view. Visual only: the numbers come from the re-simulated package.
const CAVITY_FILL = '#f2a0b4'
const RENDER = '#ece6da'
const SEAL = '#2ecc71'
const PANEL = '#1d2b4a'
const INWARD: Record<Side, [number, number]> = { front: [0, 1], back: [0, -1], left: [1, 0], right: [-1, 0] }
const KWP_PER_PANEL = 0.4

function Box({ pos, size, colour, opacity = 1, rotY = 0, emissive, map, metal = 0, rough = 0.8 }: {
  pos: [number, number, number]; size: [number, number, number]; colour: string; opacity?: number; rotY?: number
  emissive?: string; map?: THREE.Texture; metal?: number; rough?: number
}) {
  return (
    <mesh position={pos} rotation={[0, rotY, 0]} castShadow receiveShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial color={colour} map={map ?? null} transparent={opacity < 1} opacity={opacity}
        emissive={emissive ?? '#000000'} emissiveIntensity={emissive ? 0.8 : 0} roughness={rough} metalness={metal} />
    </mesh>
  )
}

/** Drops a retrofit gently into place (~0.35 s) when it's added, then stops requesting frames. */
function PopIn({ children }: { children: ReactNode }) {
  const ref = useRef<THREE.Group>(null)
  const t = useRef(0)
  useFrame((state, delta) => {
    if (!ref.current || t.current >= 1) return
    t.current = Math.min(1, t.current + delta * 3)
    ref.current.position.y = (1 - (1 - (1 - t.current) ** 3)) * 0.8
    state.invalidate()
  })
  return <group ref={ref} position={[0, 0.8, 0]}>{children}</group>
}

/** Thin layers along every exterior wall: inside (internal insulation), in the cavity, or outside (external). */
function WallLayers({ scene, where, explode }: { scene: RoomScene; where: 'inside' | 'cavity' | 'outside'; explode: boolean }) {
  const t = WALL_THICKNESS.exterior
  const shift = where === 'inside' ? t / 2 + 0.04 : where === 'cavity' ? 0 : -(t / 2 + 0.06)
  const thick = where === 'outside' ? 0.1 : 0.05
  const colour = where === 'cavity' ? CAVITY_FILL : RENDER
  const map = where === 'inside' ? insulationTexture() : undefined
  const out: JSX.Element[] = []
  for (const w of scene.walls.filter((x) => x.kind === 'exterior')) {
    const room = scene.rooms.find((r) => r.room_id === w.room_id)!
    const [nx, ny] = INWARD[wallSide(w)]
    const pieces = wallPieces(w, scene.openings.filter((o) => o.wall_id === w.wall_id),
      storeyOffset(room.storey, explode) + room.floor_z_m, scene.footprint_m)
    for (const p of pieces) {
      if (p.fill) continue
      const pos: [number, number, number] = [p.position[0] + nx * shift, p.position[1] + (where === 'cavity' ? 0.02 : 0), p.position[2] - ny * shift]
      out.push(<Box key={`${p.key}:${where}`} pos={pos} size={[p.size[0], p.size[1] + (where === 'cavity' ? 0.04 : 0), thick]}
        colour={map ? '#ffffff' : colour} map={map} rough={1} rotY={p.rotationY} />)
    }
  }
  return <group>{out}</group>
}

function DraughtSeals({ scene, explode }: { scene: RoomScene; explode: boolean }) {
  const out: JSX.Element[] = []
  for (const w of scene.walls.filter((x) => x.kind === 'exterior')) {
    const room = scene.rooms.find((r) => r.room_id === w.room_id)!
    const pieces = wallPieces(w, scene.openings.filter((o) => o.wall_id === w.wall_id),
      storeyOffset(room.storey, explode) + room.floor_z_m, scene.footprint_m)
    for (const p of pieces.filter((x) => x.fill)) {
      out.push(
        <mesh key={`${p.key}:seal`} position={p.position} rotation={[0, p.rotationY, 0]}>
          <boxGeometry args={[p.size[0] + 0.08, p.size[1] + 0.08, WALL_THICKNESS.exterior + 0.04]} />
          <meshBasicMaterial color={SEAL} wireframe />
        </mesh>,
      )
    }
  }
  return <group>{out}</group>
}

/** One PV module: cell texture under glass, in an aluminium frame. */
function Panel({ pw, ph }: { pw: number; ph: number }) {
  return (
    <group>
      <mesh castShadow>
        <boxGeometry args={[pw, 0.035, ph]} />
        <meshStandardMaterial color="#c9ced3" metalness={0.8} roughness={0.35} />
      </mesh>
      <mesh position={[0, 0.019, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[pw - 0.05, ph - 0.05]} />
        <meshPhysicalMaterial map={pvTexture()} color={PANEL === '#1d2b4a' ? '#ffffff' : PANEL} roughness={0.15} metalness={0.3}
          clearcoat={1} clearcoatRoughness={0.05} envMapIntensity={1.4} />
      </mesh>
    </group>
  )
}

/** PV panels on the roof slope that faces closest to south (flat roofs: tilted rows). */
function SolarPanels({ scene, explode, kwp }: { scene: RoomScene; explode: boolean; kwp: number }) {
  const [W, D] = scene.footprint_m
  const eave = scene.roof.eave_z_m + storeyOffset(scene.storeys - 1, explode)
  const rise = scene.roof.ridge_z_m - scene.roof.eave_z_m
  const n = Math.max(1, Math.round(kwp / KWP_PER_PANEL))
  const [pw, ph] = [1.0, 1.7]
  if (scene.roof.type === 'flat' || rise <= 0) {
    const cols = Math.max(1, Math.floor((W - 1) / (pw + 0.1)))
    return (
      <group>
        {Array.from({ length: n }, (_, i) => {
          const c = i % cols, r = Math.floor(i / cols)
          const x = 0.5 + pw / 2 + c * (pw + 0.1), y = 1 + r * 2.2
          return (
            <group key={i} position={toThree(x, y, eave + 0.45)} rotation={[-0.6, 0, 0]}><Panel pw={pw} ph={ph} /></group>
          )
        })}
      </group>
    )
  }
  const frontAz = scene.front_azimuth_deg
  const dist = (a: number) => Math.abs(((a - 180 + 540) % 360) - 180)
  const useFront = dist(frontAz) <= dist(frontAz + 180)
  const theta = Math.atan2(rise, D / 2)
  const slopeLen = Math.hypot(D / 2, rise)
  const usable = scene.roof.type === 'hip' ? Math.max(W - D / 2, pw) : W - 0.6
  const cols = Math.max(1, Math.floor(usable / (pw + 0.05)))
  const rows = Math.max(1, Math.min(Math.ceil(n / cols), Math.floor((slopeLen - 0.6) / (ph + 0.05))))
  const panels = Math.min(n, cols * rows)
  const eaveZ = useFront ? 0 : -D
  return (
    <group position={[W / 2, eave + 0.06, eaveZ]} rotation={[useFront ? theta : -theta, 0, 0]}>
      {Array.from({ length: panels }, (_, i) => {
        const c = i % cols, r = Math.floor(i / cols)
        const rowCols = Math.min(cols, panels - r * cols)
        const x = (c - (rowCols - 1) / 2) * (pw + 0.05)
        const along = 0.4 + ph / 2 + r * (ph + 0.05)
        return (
          <group key={i} position={[x, 0.03, useFront ? -along : along]}><Panel pw={pw} ph={ph} /></group>
        )
      })}
    </group>
  )
}

function roomsOfType(scene: RoomScene, types: string[]): SceneRoom[] {
  return scene.rooms.filter((r) => types.includes(r.room_type))
}

function centre(room: SceneRoom): [number, number] {
  const [x0, y0, x1, y1] = bbox(room)
  return [(x0 + x1) / 2, (y0 + y1) / 2]
}

export function Retrofits({ scene }: { scene: RoomScene }) {
  const { retrofitIds, view, explode, selectedRoom } = useViewer(useShallow((s) => ({
    retrofitIds: s.retrofitIds, view: s.view, explode: s.explode, selectedRoom: s.selectedRoom,
  })))
  if (view !== 'after' || retrofitIds.length === 0) return null
  const has = (id: string) => retrofitIds.includes(id)
  const [W, D] = scene.footprint_m
  const isolated = !!selectedRoom
  const off = (r: SceneRoom) => storeyOffset(r.storey, explode) + r.floor_z_m
  const top = Math.max(...scene.rooms.map((r) => r.storey))
  const flat = scene.context.some((c) => c.relation === 'below' || c.relation === 'above')
  const living = roomsOfType(scene, ['living'])[0] ?? scene.rooms[0]
  const store = roomsOfType(scene, ['utility', 'garage', 'hall', 'landing'])[0] ?? living

  const items: [string, ReactNode][] = []
  const add = (id: string, cond: boolean, node: () => ReactNode) => { if (has(id) && cond) items.push([id, node()]) }

  add('loft_insulation', !isolated && scene.roof.type !== 'flat' && !flat, () => (
    <Box pos={toThree(W / 2, D / 2, scene.roof.eave_z_m + storeyOffset(top, explode) + 0.14)} size={[W - 0.3, 0.27, D - 0.3]}
      colour="#ffffff" map={insulationTexture()} rough={1} />
  ))
  add('floor_insulation', !scene.context.some((c) => c.relation === 'below'), () => scene.rooms.filter((r) => r.storey === 0).map((r) => {
    const [x0, y0, x1, y1] = bbox(r)
    return <Box key={`fi:${r.room_id}`} pos={toThree((x0 + x1) / 2, (y0 + y1) / 2, off(r) - 0.06)}
      size={[x1 - x0 - 0.1, 0.08, y1 - y0 - 0.1]} colour="#ffffff" map={insulationTexture()} rough={1} />
  }))
  add('internal_wall_insulation', true, () => <WallLayers scene={scene} where="inside" explode={explode} />)
  add('cavity_wall_insulation', true, () => <WallLayers scene={scene} where="cavity" explode={explode} />)
  add('external_wall_insulation', !isolated, () => <WallLayers scene={scene} where="outside" explode={explode} />)
  add('draught_proofing', true, () => <DraughtSeals scene={scene} explode={explode} />)
  add('solar_pv', !isolated && !explode && !flat, () => <SolarPanels scene={scene} explode={explode} kwp={3.5} />)
  add('air_source_heat_pump', !isolated, () => (
    <group position={flat ? toThree(W * 0.3, -0.7, off(living)) : toThree(W * 0.72, D + 0.4, 0)}>
      <Box pos={[0, 0.45, 0]} size={[1.0, 0.85, 0.4]} colour="#f4f5f6" rough={0.35} metal={0.2} />
      <mesh position={[0.15, 0.45, flat ? 0.205 : -0.205]} rotation={[0, flat ? 0 : Math.PI, 0]}>
        <circleGeometry args={[0.3, 40]} /><meshStandardMaterial map={grilleTexture()} roughness={0.6} metalness={0.4} />
      </mesh>
      <Box pos={[0, 0.03, 0]} size={[1.1, 0.06, 0.5]} colour="#6b737a" />
    </group>
  ))
  add('hhr_storage_heaters', true, () => roomsOfType(scene, ['living', 'bedroom', 'dining', 'study']).map((r) => {
    const [x0, , x1, y1] = bbox(r)
    return <Box key={`hhr:${r.room_id}`} pos={toThree((x0 + x1) / 2, y1 - 0.25, off(r) + 0.35)} size={[0.9, 0.7, 0.22]}
      colour="#e4e4df" rough={0.4} metal={0.3} />
  }))
  add('smart_thermostat', true, () => {
    const [cx] = centre(living)
    const [, y0] = bbox(living)
    return <Box pos={toThree(cx + 0.6, y0 + 0.2, off(living) + 1.5)} size={[0.12, 0.12, 0.03]} colour="#20262b" emissive="#2ecc71" />
  })
  add('battery', true, () => {
    const [x0, y0] = bbox(store)
    return <Box pos={toThree(x0 + 0.35, y0 + 0.25, off(store) + 0.45)} size={[0.6, 0.9, 0.2]} colour="#e6e8ea" metal={0.5} rough={0.3} emissive="#2a6fdb" />
  })
  add('led_lighting', true, () => scene.rooms.map((r) => {
    const [cx, cy] = centre(r)
    return (
      <mesh key={`led:${r.room_id}`} position={toThree(cx, cy, off(r) + r.height_m - 0.04)}>
        <cylinderGeometry args={[0.18, 0.18, 0.04, 20]} />
        <meshStandardMaterial color="#fff6d8" emissive="#ffe9a3" emissiveIntensity={1.4} />
      </mesh>
    )
  }))

  return <group>{items.map(([id, node]) => <PopIn key={id}>{node}</PopIn>)}</group>
}

export const glazingColour = (ids: string[]): string | null =>
  ids.includes('triple_glazing') ? '#4fb3d9' : ids.includes('double_glazing') ? '#7fd6c2'
    : ids.includes('secondary_glazing') ? '#a9c7e8' : null

export const RETROFIT_WALL_RENDER = RENDER
