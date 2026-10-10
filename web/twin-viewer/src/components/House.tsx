import { Html } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'
import { EXPLODE_GAP_M, bbox, northDirection, roofGeometry, toThree, wallPieces } from '../geometry'
import { TEXTURES, type TextureKey, type TexturePair, floorTexture, greyDetail, metricBox, roofTexture, useTextures, wallTexture } from '../materials'
import { currentRooms, useViewer } from '../store'
import type { RoomScene, SceneFeature, SceneRoom } from '../types'
import { glazingColour } from './Retrofits'

export type Visibility = 'full' | 'faded' | 'hidden'

const GLASS = '#a9cfe6'
const PLASTER = '#f1ede6'
const PARTY = '#b9b2aa' // shared walls: same finish, muted
// Cividis ramp (colour-blind safe, AGENTS.md): low heat-loss share -> high.
export const HEAT_RAMP = ['#00224e', '#35456c', '#666970', '#958f78', '#cbba69', '#fee838']
const FADED_OPACITY = 0.1
const NORMAL_SCALE = new THREE.Vector2(0.7, 0.7)

export function heatColour(share: number, maxShare: number): string {
  const t = maxShare > 0 ? Math.min(share / maxShare, 1) : 0
  return HEAT_RAMP[Math.round(t * (HEAT_RAMP.length - 1))]
}

export function storeyOffset(storey: number, explode: boolean): number {
  return explode ? storey * EXPLODE_GAP_M : 0
}

export function roomVisibility(room: SceneRoom, selected: SceneRoom | null): Visibility {
  if (!selected) return 'full'
  if (room.room_id === selected.room_id) return 'full'
  if (room.storey > selected.storey) return 'hidden'
  return 'faded'
}

/** Ignore clicks that were really orbit drags. */
function isClick(e: ThreeEvent<MouseEvent>) {
  return e.delta < 5
}

interface SurfaceLook {
  tex?: TexturePair
  tile?: number
  colour?: string
  emissive?: string
  opacity?: number
  glass?: boolean
  roughness?: number
}

/** One box with real-world-scaled texture coordinates. Geometry is rebuilt only when its size changes. */
function Surface({ size, position, rotationY = 0, look, vis = 'full', handlers, shadows = true }: {
  size: [number, number, number]; position: [number, number, number]; rotationY?: number; look: SurfaceLook
  vis?: Visibility; handlers?: object; shadows?: boolean
}) {
  const [w, h, d] = size
  const tile = look.tile ?? 1
  const geometry = useMemo(() => metricBox(w, h, d, tile), [w, h, d, tile])
  useEffect(() => () => geometry.dispose(), [geometry])
  const o = Math.min(look.opacity ?? 1, vis === 'faded' ? FADED_OPACITY : 1)
  const transparent = o < 1
  return (
    <mesh geometry={geometry} position={position} rotation={[0, rotationY, 0]} castShadow={shadows && !transparent}
      receiveShadow={shadows} renderOrder={transparent ? 2 : 0} {...handlers}>
      {look.glass ? (
        <meshPhysicalMaterial key={`g${transparent}`} color={look.colour ?? GLASS} transparent opacity={Math.min(o, 0.42)}
          roughness={0.04} metalness={0.1} clearcoat={1} envMapIntensity={1.6} depthWrite={false} />
      ) : (
        <meshStandardMaterial key={`${transparent}${look.tex ? 't' : 'c'}`} map={look.tex?.map ?? null}
          normalMap={look.tex?.normalMap ?? null} normalScale={NORMAL_SCALE}
          color={look.colour ?? '#ffffff'} emissive={look.emissive ?? '#000000'} emissiveIntensity={look.emissive ? 0.26 : 0}
          roughness={look.roughness ?? 0.92} transparent={transparent} opacity={o} depthWrite={!transparent} />
      )}
    </mesh>
  )
}

// When one room is isolated, the walls nearest the camera (front and right) become see-through: a cutaway.
const CUTAWAY_SIDES = new Set(['front', 'right'])

function Room({ scene, room, vis, isolated, explode, onHover, heat, hovered, glass, renderFinish, tex }: {
  scene: RoomScene; room: SceneRoom; vis: Visibility; isolated: boolean; explode: boolean
  onHover: (id: string | null) => void; heat: string | null; hovered: boolean; glass: string
  renderFinish: boolean; tex: Record<TextureKey, TexturePair>
}) {
  const selectRoom = useViewer((s) => s.selectRoom)
  const base = storeyOffset(room.storey, explode)
  const [x0, y0, x1, y1] = bbox(room)
  const pieces = useMemo(
    () => scene.walls.filter((w) => w.room_id === room.room_id).flatMap((w) =>
      wallPieces(w, scene.openings.filter((o) => o.wall_id === w.wall_id), base + room.floor_z_m, scene.footprint_m)
        .map((p) => ({ ...p, kind: w.kind, side: w.wall_id.split(':').pop() ?? '' }))),
    [scene.walls, scene.openings, room.room_id, base, room.floor_z_m, scene.footprint_m],
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
  const outer = renderFinish ? 'render' : wallTexture(scene.style.wall_material, room.storey > 0)
  const fk = floorTexture(room.room_type)

  return (
    <group>
      <Surface size={[x1 - x0, 0.05, y1 - y0]} position={toThree((x0 + x1) / 2, (y0 + y1) / 2, base + room.floor_z_m + 0.025)}
        look={heat ? { colour: heat, roughness: 0.6 } : { tex: tex[fk], tile: TEXTURES[fk], emissive: hovered ? '#2a6fdb' : undefined }}
        vis={vis} handlers={handlers} />
      {pieces.map((p) => {
        const isGlass = p.fill === 'window' || p.fill === 'patio_door'
        const cut = isolated && CUTAWAY_SIDES.has(p.side)
        let look: SurfaceLook
        if (isGlass) look = { glass: true, colour: glass }
        else if (p.fill === 'door') look = { colour: scene.style.door_colour, roughness: 0.45 }
        else if (p.kind === 'internal') look = { colour: PLASTER, roughness: 0.95 }
        else if (p.kind === 'party') look = { tex: tex[outer], tile: TEXTURES[outer], colour: PARTY }
        else look = { tex: tex[outer], tile: TEXTURES[outer], emissive: heat ?? undefined }
        if (cut) look = { ...look, opacity: 0.16 }
        return (
          <Surface key={p.key} size={p.size} position={p.position} rotationY={p.rotationY} look={look} vis={vis}
            handlers={handlers} shadows={!isGlass} />
        )
      })}
    </group>
  )
}

function Roof({ scene, explode, opacity = 1, tex }: {
  scene: RoomScene; explode: boolean; opacity?: number; tex: Record<TextureKey, TexturePair>
}) {
  const [w, d] = scene.footprint_m
  const eave = scene.roof.eave_z_m + storeyOffset(scene.storeys - 1, explode)
  const rise = scene.roof.ridge_z_m - scene.roof.eave_z_m
  const key = roofTexture(scene.roof.material)
  const gableKey = wallTexture(scene.style.wall_material, true)
  const geo = useMemo(
    () => (scene.roof.type === 'flat' ? null
      : roofGeometry(scene.roof.type, w, d, eave, rise, scene.roof.ridge_axis, 0.3, TEXTURES[key])),
    [scene.roof.type, scene.roof.ridge_axis, w, d, eave, rise, key],
  )
  useEffect(() => () => { geo?.slopes.dispose(); geo?.gables?.dispose() }, [geo])
  const transparent = opacity < 1
  if (!geo) {
    return <Surface size={[w + 0.3, 0.2, d + 0.3]} position={toThree(w / 2, d / 2, eave + 0.1)}
      look={{ tex: tex.concrete, tile: TEXTURES.concrete, colour: '#9a9d9f' }} />
  }
  return (
    <group>
      <mesh geometry={geo.slopes} castShadow={!transparent} receiveShadow renderOrder={transparent ? 3 : 0}>
        <meshStandardMaterial key={transparent ? 't' : 'o'} map={tex[key].map} normalMap={tex[key].normalMap}
          side={THREE.DoubleSide} roughness={0.85} transparent={transparent} opacity={opacity} depthWrite={!transparent} />
      </mesh>
      {geo.gables && (
        <mesh geometry={geo.gables} castShadow receiveShadow>
          <meshStandardMaterial map={tex[gableKey].map} normalMap={tex[gableKey].normalMap} side={THREE.DoubleSide} roughness={0.92} />
        </mesh>
      )}
    </group>
  )
}

function Feature({ f, scene, explode, tex }: { f: SceneFeature; scene: RoomScene; explode: boolean; tex: Record<TextureKey, TexturePair> }) {
  const [x, y, z] = f.position_m
  const [sw, sd, sh] = f.size_m
  const room = f.room_id ? scene.rooms.find((r) => r.room_id === f.room_id) : undefined
  const off = f.kind === 'chimney' ? storeyOffset(scene.storeys - 1, explode) : storeyOffset(room?.storey ?? 0, explode)
  const pos = toThree(x, y, z + off + sh / 2)
  const wallKey = wallTexture(scene.style.wall_material, false)
  const wallLook = { tex: tex[wallKey], tile: TEXTURES[wallKey] }

  if (f.kind === 'bay_window') {
    return (
      <group>
        <Surface size={[sw, sh * 0.96, sd]} position={pos} look={{ glass: true }} shadows={false} />
        <Surface size={[sw + 0.12, 0.14, sd + 0.12]} position={[pos[0], pos[1] + sh / 2, pos[2]]} look={{ colour: f.colour ?? '#ffffff', roughness: 0.5 }} />
      </group>
    )
  }
  if (f.kind === 'balcony') {
    const base = toThree(x, y, z + off)
    return (
      <group>
        <Surface size={[sw, 0.12, sd]} position={[base[0], base[1] + 0.06, base[2]]} look={{ tex: tex.concrete, tile: TEXTURES.concrete }} />
        <Surface size={[sw, sh, 0.03]} position={[base[0], base[1] + sh / 2, base[2] + sd / 2]} look={{ glass: true }} shadows={false} />
      </group>
    )
  }
  if (f.kind === 'garage') {
    return (
      <group>
        <Surface size={[sw, sh, sd]} position={pos} look={wallLook} />
        <Surface size={[sw * 0.8, sh * 0.75, 0.04]} position={[pos[0], pos[1] - sh * 0.1, pos[2] + sd / 2 + 0.02]}
          look={{ colour: String(f.extra.door_colour ?? '#e8e4dc'), roughness: 0.4 }} />
      </group>
    )
  }
  if (f.kind === 'porch' && f.style !== 'enclosed') {
    const rk = roofTexture(scene.roof.material)
    return <Surface size={[sw, sh, sd]} position={pos} look={{ tex: tex[rk], tile: TEXTURES[rk] }} />
  }
  return <Surface size={[sw, sh, sd]} position={pos} look={wallLook} />
}

function Ground({ scene, tex }: { scene: RoomScene; tex: Record<TextureKey, TexturePair> }) {
  const [w, d] = scene.footprint_m
  const size = 80
  const lawn = useMemo(() => greyDetail(tex.grass.map), [tex.grass.map])
  const geometry = useMemo(() => {
    const g = new THREE.CircleGeometry(size / 2, 64)
    const uv = g.attributes.uv as THREE.BufferAttribute
    for (let i = 0; i < uv.count; i++) uv.setXY(i, (uv.getX(i) * size) / TEXTURES.grass, (uv.getY(i) * size) / TEXTURES.grass)
    return g
  }, [])
  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} position={[w / 2, -0.02, -d / 2]} receiveShadow>
      <meshStandardMaterial map={lawn} normalMap={tex.grass.normalMap} roughness={1} color="#7fae5c" />
    </mesh>
  )
}

function Compass({ scene }: { scene: RoomScene }) {
  const [w, d] = scene.footprint_m
  const [nx, ny] = northDirection(scene.front_azimuth_deg)
  const r = Math.max(w, d) / 2 + 3
  const [cx, cy] = [w / 2 + nx * r, d / 2 + ny * r]
  const angle = Math.atan2(nx, -ny)
  return (
    <group position={toThree(cx, cy, 0.03)}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.55, 0.62, 48]} />
        <meshBasicMaterial color="#0b6e4f" transparent opacity={0.85} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, -angle]} position={[0, 0.01, 0]}>
        <shapeGeometry args={[new THREE.Shape([new THREE.Vector2(0, 0.5), new THREE.Vector2(0.18, -0.3), new THREE.Vector2(0, -0.15), new THREE.Vector2(-0.18, -0.3)])]} />
        <meshBasicMaterial color="#0b6e4f" />
      </mesh>
      <Html center position={[nx * 0.95, 0.05, -ny * 0.95]} style={{ pointerEvents: 'none' }} zIndexRange={[3, 0]}>
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
  const tex = useTextures()
  const selectedRoomId = useViewer((s) => s.selectedRoom)
  const showRoof = useViewer((s) => s.showRoof)
  const explode = useViewer((s) => s.explode)
  const heatTint = useViewer((s) => s.heatTint)
  const retrofitIds = useViewer((s) => s.retrofitIds)
  const afterView = useViewer((s) => s.view === 'after' && s.retrofitIds.length > 0)
  const energy = useViewer(currentRooms)
  const glass: string = (afterView ? glazingColour(retrofitIds) : null) ?? GLASS
  const renderFinish = afterView && retrofitIds.includes('external_wall_insulation')
  const maxShare = energy ? Math.max(...energy.rooms.map((r) => r.heat_loss_share_pct)) : 0
  const heatFor = (id: string) => {
    const r = heatTint && energy ? energy.rooms.find((x) => x.room_id === id) : undefined
    return r ? heatColour(r.heat_loss_share_pct, maxShare) : null
  }
  const [hover, setHover] = useState<string | null>(null)
  const selected = scene.rooms.find((r) => r.room_id === selectedRoomId) ?? null
  const elevated = scene.context.some((c) => c.relation === 'below')
  const hovered = scene.rooms.find((r) => r.room_id === hover)

  return (
    <group>
      {!elevated && <Ground scene={scene} tex={tex} />}
      {scene.rooms.map((room) => (
        <Room key={room.room_id} scene={scene} room={room} vis={roomVisibility(room, selected)}
          isolated={selected?.room_id === room.room_id} explode={explode} onHover={setHover} heat={heatFor(room.room_id)}
          hovered={hover === room.room_id && !selected} glass={glass} renderFinish={renderFinish} tex={tex} />
      ))}
      {!selected && !explode && showRoof && <Roof scene={scene} explode={explode} opacity={afterView ? 0.45 : 1} tex={tex} />}
      {!selected && !explode && scene.features.map((f, i) => <Feature key={`${f.kind}${i}`} f={f} scene={scene} explode={explode} tex={tex} />)}
      {!selected && scene.context.map((c) => {
        const [cx, cy, cz] = c.position_m
        const [cw, cd, ch] = c.size_m
        return (
          <mesh key={c.relation} position={toThree(cx, cy, cz + ch / 2)} renderOrder={1}>
            <boxGeometry args={[cw, ch, cd]} />
            <meshStandardMaterial color="#c4cbd0" transparent opacity={0.22} depthWrite={false} roughness={1} />
          </mesh>
        )
      })}
      <Compass scene={scene} />
      {hovered && !selected && (
        <Html position={roomTop(hovered, explode)} center style={{ pointerEvents: 'none' }} zIndexRange={[3, 0]}>
          <span className="room-tag">{hovered.label}</span>
        </Html>
      )}
    </group>
  )
}
