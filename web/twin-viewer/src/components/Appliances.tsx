import { Edges } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'
import { useShallow } from 'zustand/react/shallow'
import { rotationY, toPlan, toThree } from '../geometry'
import { useViewer } from '../store'
import type { RoomScene, SceneAppliance } from '../types'
import { roomVisibility, storeyOffset } from './House'

function Body({ app, opacity }: { app: SceneAppliance; opacity: number }) {
  const [w, d, h] = app.size_m
  const mat = (colour: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) => (
    <meshStandardMaterial key={opacity < 1 ? 't' : 'o'} color={colour} transparent={opacity < 1} opacity={opacity}
      depthWrite={opacity >= 1} {...extra} />
  )
  const front = d / 2 + 0.006 // local +z is the appliance's front

  if (app.shape === 'cylinder') {
    return (
      <group>
        <mesh castShadow receiveShadow>
          <cylinderGeometry args={[w / 2, w / 2, h, 24]} />
          {mat(app.colour, { metalness: 0.3 })}
        </mesh>
        <mesh position={[0, h / 2 + 0.01, 0]}>
          <cylinderGeometry args={[w / 2.4, w / 2.4, 0.02, 24]} />
          {mat(app.accent)}
        </mesh>
      </group>
    )
  }
  return (
    <group>
      <mesh>
        <boxGeometry args={[w, h, d]} />
        {mat(app.colour, { roughness: 0.5 })}
      </mesh>
      {app.shape === 'panel' && (
        <mesh position={[0, 0, front]}>
          <boxGeometry args={[w * 0.95, h * 0.9, 0.01]} />
          {mat('#0d1a26', { emissive: app.accent, emissiveIntensity: 0.25 })}
        </mesh>
      )}
      {app.shape === 'drum_box' && (
        <mesh position={[0, -h * 0.08, front]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[Math.min(w, h) * 0.3, Math.min(w, h) * 0.3, 0.01, 24]} />
          {mat(app.accent)}
        </mesh>
      )}
      {app.shape === 'tub' && (
        <mesh position={[0, h / 2 + 0.005, 0]}>
          <boxGeometry args={[w * 0.85, 0.01, d * 0.85]} />
          {mat(app.accent)}
        </mesh>
      )}
      {!['panel', 'drum_box', 'tub'].includes(app.shape) && (
        <mesh position={[0, h * 0.3, front]}>
          <boxGeometry args={[w * 0.8, Math.max(0.02, h * 0.08), 0.01]} />
          {mat(app.accent)}
        </mesh>
      )}
    </group>
  )
}

function Appliance({ app, scene }: { app: SceneAppliance; scene: RoomScene }) {
  const { selectedRoom, selectedAppliance, explode, selectRoom, selectAppliance, moveAppliance, setDragging } = useViewer(
    useShallow((s) => ({
      selectedRoom: s.selectedRoom, selectedAppliance: s.selectedAppliance, explode: s.explode, selectRoom: s.selectRoom,
      selectAppliance: s.selectAppliance, moveAppliance: s.moveAppliance, setDragging: s.setDragging,
    })),
  )
  const dragging = useRef(false)
  const room = app.room_id ? scene.rooms.find((r) => r.room_id === app.room_id) : undefined
  const selected = scene.rooms.find((r) => r.room_id === selectedRoom) ?? null
  const vis = room ? roomVisibility(room, selected) : selected ? 'faded' : 'full'
  if (vis === 'hidden') return null

  const [x, y, z] = app.position_m
  const [w, d, h] = app.size_m
  const off = storeyOffset(room?.storey ?? 0, explode)
  const editable = !!room && selectedRoom === room.room_id
  const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(z + off))
  const isSelected = selectedAppliance === app.appliance_id

  const onPointerDown = (e: ThreeEvent<PointerEvent>) => {
    if (!editable) return
    e.stopPropagation()
    ;(e.target as Element).setPointerCapture(e.pointerId)
    dragging.current = true
    setDragging(true)
    selectAppliance(app.appliance_id)
  }
  const onPointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!dragging.current) return
    e.stopPropagation()
    const hit = new THREE.Vector3()
    if (e.ray.intersectPlane(floorPlane, hit)) moveAppliance(app.appliance_id, ...toPlan(hit))
  }
  const onPointerUp = (e: ThreeEvent<PointerEvent>) => {
    if (!dragging.current) return
    e.stopPropagation()
    ;(e.target as Element).releasePointerCapture(e.pointerId)
    dragging.current = false
    setDragging(false)
  }
  const onClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 5) return
    e.stopPropagation()
    if (room && selectedRoom !== room.room_id) selectRoom(room.room_id)
    selectAppliance(app.appliance_id)
  }

  return (
    <group
      position={toThree(x, y, z + off + h / 2)}
      rotation={[0, rotationY(app.rotation_deg), 0]}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onClick={onClick}
      onPointerOver={() => { if (editable) document.body.style.cursor = 'grab' }}
      onPointerOut={() => { document.body.style.cursor = 'auto' }}
    >
      <Body app={app} opacity={vis === 'faded' ? 0.15 : 1} />
      {app.mount === 'counter' && room && z - room.floor_z_m > 0.05 && (
        // Worktop items stand on their own base cabinet, since nothing is stacked under them.
        <mesh position={[0, -h / 2 - (z - room.floor_z_m) / 2, 0]}>
          <boxGeometry args={[w, z - room.floor_z_m, d]} />
          <meshStandardMaterial key={vis === 'faded' ? 't' : 'o'} color="#d9d2c4" transparent={vis === 'faded'}
            opacity={vis === 'faded' ? 0.15 : 1} roughness={0.7} />
        </mesh>
      )}
      {isSelected && (
        <mesh>
          <boxGeometry args={[w + 0.05, h + 0.05, d + 0.05]} />
          <meshBasicMaterial transparent opacity={0.08} color="#2a6fdb" depthWrite={false} />
          <Edges color="#2a6fdb" lineWidth={2} />
        </mesh>
      )}
    </group>
  )
}

export function Appliances({ scene }: { scene: RoomScene }) {
  const appliances = useViewer((s) => s.appliances)
  return (
    <group>
      {appliances.map((a) => <Appliance key={a.appliance_id} app={a} scene={scene} />)}
    </group>
  )
}
