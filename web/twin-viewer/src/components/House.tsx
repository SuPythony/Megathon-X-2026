import { Html } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import { useMemo, useState } from 'react'
import * as THREE from 'three'
import { EXPLODE_GAP_M, bbox, northDirection, roofGeometry, toThree, wallPieces } from '../geometry'
import { useViewer } from '../store'
import type { RoomScene, SceneFeature, SceneRoom } from '../types'

export type Visibility = 'full' | 'faded' | 'hidden'

const ROOM_COLOURS: Record<string, string> = {
  living: '#e9dcc4', kitchen: '#d9e4d0', dining: '#efe2cc', bedroom: '#dbe2ee', bathroom: '#d4e8ec', wc: '#d4e8ec',
  hall: '#e6e1da', landing: '#e6e1da', utility: '#dfe3dc', study: '#e8e0d0', garage: '#d8d8d8',
  conservatory: '#e3ecd8', other: '#e4e4e4',
}
const GLASS = '#9ec9e6'
const FADED_OPACITY = 0.1

export function storeyOffset(storey: number, explode: boolean): number {
  return explode ? storey * EXPLODE_GAP_M : 0
}

export function roomVisibility(room: SceneRoom, selected: SceneRoom | null): Visibility {
  if (!selected) return 'full'
  if (room.room_id === selected.room_id) return 'full'
  if (room.storey > selected.storey) return 'hidden'
  return 'faded'
}

function Mat({ colour, vis, opacity = 1, side }: { colour: string; vis: Visibility; opacity?: number; side?: THREE.Side }) {
  const faded = vis === 'faded'
  const o = faded ? Math.min(opacity, FADED_OPACITY) : opacity
  // Keyed so three rebuilds the material when transparency changes (it ignores later toggles).
  return (
    <meshStandardMaterial key={o < 1 ? 't' : 'o'} color={colour} transparent={o < 1} opacity={o} depthWrite={o >= 1}
      side={side} roughness={0.85} />
  )
}

/** Ignore clicks that were really orbit drags. */
function isClick(e: ThreeEvent<MouseEvent>) {
  return e.delta < 5
}

// When one room is isolated, the walls nearest the camera (front and right) become see-through: a cutaway.
const CUTAWAY_SIDES = new Set(['front', 'right'])

function Room({ scene, room, vis, isolated, explode, onHover }: {
  scene: RoomScene; room: SceneRoom; vis: Visibility; isolated: boolean; explode: boolean
  onHover: (id: string | null) => void
}) {
  const selectRoom = useViewer((s) => s.selectRoom)
  const base = storeyOffset(room.storey, explode)
  const [x0, y0, x1, y1] = bbox(room)
  const walls = scene.walls.filter((w) => w.room_id === room.room_id)
  const upper = room.storey > 0 && scene.style.wall_upper_colour ? scene.style.wall_upper_colour : scene.style.wall_colour

  const pieces = useMemo(
    () => walls.flatMap((w) => wallPieces(w, scene.openings.filter((o) => o.wall_id === w.wall_id), base + room.floor_z_m,
      scene.footprint_m).map((p) => ({ ...p, kind: w.kind, side: w.wall_id.split(':').pop() ?? '' }))),
    [walls, scene.openings, base, room.floor_z_m, scene.footprint_m],
  )
  if (vis === 'hidden') return null

  const handlers = {
    onClick: (e: ThreeEvent<MouseEvent>) => {
      if (!isClick(e)) return
      e.stopPropagation()
      selectRoom(room.room_id)
    },
    onPointerOver: (e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); onHover(room.room_id) },
    onPointerOut: () => onHover(null),
  }

  return (
    <group>
      <mesh position={toThree((x0 + x1) / 2, (y0 + y1) / 2, base + room.floor_z_m + 0.025)} {...handlers}>
        <boxGeometry args={[x1 - x0, 0.05, y1 - y0]} />
        <Mat colour={ROOM_COLOURS[room.room_type] ?? ROOM_COLOURS.other} vis={vis} />
      </mesh>
      {pieces.map((p) => {
        const colour =
          p.fill === 'window' || p.fill === 'patio_door' ? GLASS
            : p.fill === 'door' ? scene.style.door_colour
              : p.kind === 'internal' ? '#f3f1ec'
                : p.kind === 'party' ? '#b9a99a'
                  : upper
        const glass = p.fill === 'window' || p.fill === 'patio_door'
        const cut = isolated && CUTAWAY_SIDES.has(p.side)
        return (
          <mesh key={p.key} position={p.position} rotation={[0, p.rotationY, 0]} {...handlers}>
            <boxGeometry args={p.size} />
            <Mat colour={colour} vis={vis} opacity={cut ? 0.18 : glass ? 0.45 : 1} />
          </mesh>
        )
      })}
    </group>
  )
}

function Roof({ scene, explode }: { scene: RoomScene; explode: boolean }) {
  const [w, d] = scene.footprint_m
  const eave = scene.roof.eave_z_m + storeyOffset(scene.storeys - 1, explode)
  const rise = scene.roof.ridge_z_m - scene.roof.eave_z_m
  const geo = useMemo(
    () => (scene.roof.type === 'flat' ? null : roofGeometry(scene.roof.type, w, d, eave, rise, scene.roof.ridge_axis)),
    [scene.roof.type, scene.roof.ridge_axis, w, d, eave, rise],
  )
  if (!geo) {
    return (
      <mesh position={toThree(w / 2, d / 2, eave + 0.1)}>
        <boxGeometry args={[w + 0.3, 0.2, d + 0.3]} />
        <meshStandardMaterial color={scene.roof.colour} />
      </mesh>
    )
  }
  return (
    <group>
      <mesh geometry={geo.slopes}>
        <meshStandardMaterial color={scene.roof.colour} side={THREE.DoubleSide} roughness={0.9} />
      </mesh>
      {geo.gables && (
        <mesh geometry={geo.gables}>
          <meshStandardMaterial color={scene.style.wall_upper_colour ?? scene.style.wall_colour} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  )
}

function Feature({ f, scene, explode }: { f: SceneFeature; scene: RoomScene; explode: boolean }) {
  const [x, y, z] = f.position_m
  const [sw, sd, sh] = f.size_m
  const room = f.room_id ? scene.rooms.find((r) => r.room_id === f.room_id) : undefined
  const off = f.kind === 'chimney' ? storeyOffset(scene.storeys - 1, explode) : storeyOffset(room?.storey ?? 0, explode)
  const pos = toThree(x, y, z + off + sh / 2)
  const colour = f.colour ?? scene.style.wall_colour

  if (f.kind === 'bay_window') {
    return (
      <group position={pos}>
        <mesh>
          <boxGeometry args={[sw, sh * 0.96, sd]} />
          <meshStandardMaterial color={GLASS} transparent opacity={0.4} />
        </mesh>
        <mesh position={[0, sh / 2, 0]}>
          <boxGeometry args={[sw + 0.1, 0.12, sd + 0.1]} />
          <meshStandardMaterial color={colour} />
        </mesh>
      </group>
    )
  }
  if (f.kind === 'balcony') {
    return (
      <group position={toThree(x, y, z + off)}>
        <mesh position={[0, 0.05, 0]}>
          <boxGeometry args={[sw, 0.1, sd]} />
          <meshStandardMaterial color="#9a9fa3" />
        </mesh>
        <mesh position={[0, sh / 2, sd / 2]}>
          <boxGeometry args={[sw, sh, 0.03]} />
          <meshStandardMaterial color={GLASS} transparent opacity={0.35} />
        </mesh>
      </group>
    )
  }
  if (f.kind === 'garage') {
    return (
      <group position={pos}>
        <mesh>
          <boxGeometry args={[sw, sh, sd]} />
          <meshStandardMaterial color={colour} />
        </mesh>
        <mesh position={[0, -sh * 0.1, sd / 2 + 0.02]}>
          <boxGeometry args={[sw * 0.8, sh * 0.75, 0.04]} />
          <meshStandardMaterial color={String(f.extra.door_colour ?? '#ddd')} />
        </mesh>
      </group>
    )
  }
  return (
    <mesh position={pos}>
      <boxGeometry args={[sw, sh, sd]} />
      <meshStandardMaterial color={colour} />
    </mesh>
  )
}

function Compass({ scene }: { scene: RoomScene }) {
  const [w, d] = scene.footprint_m
  const [nx, ny] = northDirection(scene.front_azimuth_deg)
  const r = Math.max(w, d) / 2 + 2.5
  const [px, py] = [w / 2 + nx * r, d / 2 + ny * r]
  const angle = Math.atan2(nx, -ny) // rotate the cone (pointing +z) to face north
  return (
    <group position={toThree(px, py, 0.05)} rotation={[0, angle, 0]}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.35, 1, 3]} />
        <meshStandardMaterial color="#0b6e4f" />
      </mesh>
      <Html center position={[0, 0.6, 0.7]} style={{ pointerEvents: 'none' }}>
        <span className="compass-label">N</span>
      </Html>
    </group>
  )
}

function roomTop(room: SceneRoom, explode: boolean): [number, number, number] {
  const [x0, y0, x1, y1] = bbox(room)
  return toThree((x0 + x1) / 2, (y0 + y1) / 2, storeyOffset(room.storey, explode) + room.floor_z_m + room.height_m + 0.3)
}

export function House({ scene }: { scene: RoomScene }) {
  const selectedRoomId = useViewer((s) => s.selectedRoom)
  const showRoof = useViewer((s) => s.showRoof)
  const explode = useViewer((s) => s.explode)
  const [hover, setHover] = useState<string | null>(null)
  const selected = scene.rooms.find((r) => r.room_id === selectedRoomId) ?? null
  const [w, d] = scene.footprint_m
  const elevated = scene.context.some((c) => c.relation === 'below')
  const hovered = scene.rooms.find((r) => r.room_id === hover)

  return (
    <group>
      {!elevated && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[w / 2, -0.01, -d / 2]}>
          <planeGeometry args={[60, 60]} />
          <meshStandardMaterial color="#c9d6c1" />
        </mesh>
      )}
      {scene.rooms.map((room) => (
        <Room key={room.room_id} scene={scene} room={room} vis={roomVisibility(room, selected)}
          isolated={selected?.room_id === room.room_id} explode={explode} onHover={setHover} />
      ))}
      {!selected && !explode && showRoof && <Roof scene={scene} explode={explode} />}
      {!selected && !explode && scene.features.map((f, i) => <Feature key={`${f.kind}${i}`} f={f} scene={scene} explode={explode} />)}
      {!selected && scene.context.map((c) => {
        const [cx, cy, cz] = c.position_m
        const [cw, cd, ch] = c.size_m
        return (
          <mesh key={c.relation} position={toThree(cx, cy, cz + ch / 2)}>
            <boxGeometry args={[cw, ch, cd]} />
            <meshStandardMaterial color="#aab2b8" transparent opacity={0.18} depthWrite={false} />
          </mesh>
        )
      })}
      <Compass scene={scene} />
      {hovered && !selected && (
        <Html position={roomTop(hovered, explode)} center style={{ pointerEvents: 'none' }}>
          <span className="room-tag">{hovered.label}</span>
        </Html>
      )}
    </group>
  )
}
