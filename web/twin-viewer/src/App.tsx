import { CameraControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { Appliances } from './components/Appliances'
import { House, storeyOffset } from './components/House'
import { Panel } from './components/Panel'
import { bbox } from './geometry'
import { useViewer } from './store'
import type { RoomScene } from './types'

const SCENE_NAMES: Record<string, string> = {
  victorian_terrace_2bed: 'Victorian terrace',
  interwar_semi_3bed: '1930s semi',
  postwar_detached_4bed: '1960s-80s detached',
  modern_terrace_3bed: 'Post-2000 terrace',
  bungalow_2bed: 'Bungalow',
  flat_2bed: 'Flat',
}

function frameBox(scene: RoomScene, roomId: string | null, explode: boolean): THREE.Box3 {
  const room = scene.rooms.find((r) => r.room_id === roomId)
  if (room) {
    const [x0, y0, x1, y1] = bbox(room)
    const base = storeyOffset(room.storey, explode) + room.floor_z_m
    return new THREE.Box3(new THREE.Vector3(x0, base, -y1), new THREE.Vector3(x1, base + room.height_m, -y0))
  }
  const [w, d] = scene.footprint_m
  const top = scene.roof.ridge_z_m + storeyOffset(scene.storeys - 1, explode)
  return new THREE.Box3(new THREE.Vector3(-1, 0, -d - 1), new THREE.Vector3(w + 1, top, 1.5))
}

function Viewport({ scene }: { scene: RoomScene }) {
  const controls = useRef<CameraControls>(null)
  const { selectedRoom, explode, dragging, sceneKey } = useViewer()

  useEffect(() => {
    const c = controls.current
    if (!c) return
    const box = frameBox(scene, selectedRoom, explode)
    const centre = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3()).length()
    // Look from the front (street side, +z) and above; a closer, steeper view for a single room.
    const dist = selectedRoom ? size * 1.1 : size * 1.2
    const elev = selectedRoom ? 1.0 : 0.55
    c.setLookAt(centre.x + dist * 0.45, centre.y + dist * elev, centre.z + dist * 0.85, centre.x, centre.y, centre.z, true)
  }, [scene, selectedRoom, explode, sceneKey])

  return (
    <Canvas camera={{ position: [12, 10, 14], fov: 40 }} dpr={[1, 2]}>
      <color attach="background" args={['#eef2f0']} />
      <hemisphereLight args={['#ffffff', '#b8c4b0', 1.1]} />
      <directionalLight position={[8, 14, 6]} intensity={1.4} />
      <directionalLight position={[-6, 8, -10]} intensity={0.4} />
      <House scene={scene} />
      <Appliances scene={scene} />
      <CameraControls ref={controls} enabled={!dragging} makeDefault minDistance={2} maxDistance={60} />
    </Canvas>
  )
}

export default function App() {
  const { scenes, sceneKey, setScene, showRoof, toggleRoof, explode, toggleExplode, selectedRoom, selectRoom } =
    useViewer()
  const scene = scenes[sceneKey]

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') selectRoom(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectRoom])

  if (!scene) return <p className="empty">No scenes found. Run <code>npm run sync-data</code>.</p>

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><span className="dot" />GreenTwin · Room twin</div>
        <select value={sceneKey} onChange={(e) => setScene(e.target.value)} aria-label="House type">
          {Object.keys(scenes).sort().map((k) => (
            <option key={k} value={k}>{SCENE_NAMES[k] ?? k}</option>
          ))}
        </select>
        <button className={showRoof ? 'toggle on' : 'toggle'} onClick={toggleRoof} disabled={!!selectedRoom || explode}>Roof</button>
        <button className={explode ? 'toggle on' : 'toggle'} onClick={toggleExplode}>Explode floors</button>
        <span className="hint-inline">{selectedRoom ? 'Esc to return to the whole house' : 'Click a room to isolate it'}</span>
      </header>
      <main className="stage">
        <div className="viewport"><Viewport scene={scene} /></div>
        <Panel scene={scene} />
      </main>
    </div>
  )
}
