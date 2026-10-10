import { CameraControls, Environment, Lightformer, Sky, useProgress } from '@react-three/drei'
import { Canvas, useThree } from '@react-three/fiber'
import { Flame, Home, Layers, Plus, RotateCcw, Triangle } from 'lucide-react'
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { useShallow } from 'zustand/react/shallow'
import { Appliances } from './components/Appliances'
import { HEAT_RAMP, House, storeyOffset } from './components/House'
import { IntakeForm } from './components/IntakeForm'
import { Panel } from './components/Panel'
import { Retrofits } from './components/Retrofits'
import { bbox } from './geometry'
import { applianceKey, useViewer } from './store'
import type { RoomScene } from './types'

const SCENE_NAMES: Record<string, string> = {
  SH1: 'SH1 · Victorian terrace',
  SH2: 'SH2 · 1970s semi',
  SH3: 'SH3 · Electric flat',
  victorian_terrace_2bed: 'Victorian terrace (layout only)',
  interwar_semi_3bed: '1930s semi (layout only)',
  postwar_detached_4bed: '1960s-80s detached (layout only)',
  modern_terrace_3bed: 'Post-2000 terrace (layout only)',
  bungalow_2bed: 'Bungalow (layout only)',
  flat_2bed: 'Flat (layout only)',
}
const SCENE_ORDER = ['SH1', 'SH2', 'SH3']

function frameBox(scene: RoomScene, roomId: string | null, explode: boolean): THREE.Box3 {
  const room = scene.rooms.find((r) => r.room_id === roomId)
  if (room) {
    const [x0, y0, x1, y1] = bbox(room)
    const base = storeyOffset(room.storey, explode) + room.floor_z_m
    return new THREE.Box3(new THREE.Vector3(x0, base, -y1), new THREE.Vector3(x1, base + room.height_m, -y0))
  }
  const [w, d] = scene.footprint_m
  const top = scene.roof.ridge_z_m + storeyOffset(scene.storeys - 1, explode)
  return new THREE.Box3(new THREE.Vector3(-1, 0, -d - 1.5), new THREE.Vector3(w + 1, top, 1.5))
}

/** Lighting that stays still: one shadow-casting sun with a fixed, fitted shadow camera (no shimmering), a soft sky,
 * and a procedural environment for reflections on glass and panels (no network fetch). */
function Lighting({ scene }: { scene: RoomScene }) {
  const [w, d] = scene.footprint_m
  const target = useMemo(() => {
    const o = new THREE.Object3D()
    o.position.set(w / 2, 0, -d / 2)
    o.updateMatrixWorld()
    return o
  }, [w, d])
  return (
    <>
      <Sky distance={4500} sunPosition={[60, 40, 40]} turbidity={6} rayleigh={1.2} mieCoefficient={0.004} mieDirectionalG={0.85} />
      <fog attach="fog" args={['#dfe7ee', 35, 110]} />
      <hemisphereLight args={['#eaf2ff', '#8a9878', 0.85]} />
      <primitive object={target} />
      <directionalLight position={[w / 2 + 14, 22, -d / 2 + 12]} intensity={2.4} castShadow target={target}
        shadow-mapSize={[2048, 2048]} shadow-bias={-0.0004} shadow-normalBias={0.03}
        shadow-camera-left={-16} shadow-camera-right={16} shadow-camera-top={16} shadow-camera-bottom={-16}
        shadow-camera-near={1} shadow-camera-far={70} />
      <Environment resolution={256} frames={1}>
        <Lightformer intensity={2} position={[0, 6, 6]} scale={[12, 4, 1]} color="#ffffff" />
        <Lightformer intensity={1.2} position={[-8, 3, -2]} rotation-y={Math.PI / 2} scale={[10, 3, 1]} color="#dbe8ff" />
        <Lightformer intensity={0.8} position={[8, 2, -4]} rotation-y={-Math.PI / 2} scale={[10, 2, 1]} color="#fff1dc" />
        <Lightformer intensity={0.6} form="ring" position={[0, 10, -6]} scale={4} color="#ffffff" />
      </Environment>
    </>
  )
}

/** Frames the house (or one room) from the street side, then shifts the view so it sits in the middle of the part of
 * the screen the panel and top bar don't cover. Runs only when what's framed changes, never per frame. */
function Framer({ controls, scene }: { controls: React.RefObject<CameraControls | null>; scene: RoomScene }) {
  const { camera, size } = useThree()
  const { selectedRoom, explode, sceneKey, viewNonce } = useViewer(useShallow((s) => ({
    selectedRoom: s.selectedRoom, explode: s.explode, sceneKey: s.sceneKey, viewNonce: s.viewNonce,
  })))
  useEffect(() => {
    const c = controls.current
    if (!c) return
    const box = frameBox(scene, selectedRoom, explode)
    const centre = box.getCenter(new THREE.Vector3())
    const diag = box.getSize(new THREE.Vector3()).length()
    const narrow = size.width / size.height < 1 ? 1.7 : 1  // portrait screens need more distance to fit the width
    const dist = (selectedRoom ? diag * 1.35 : diag * 1.85) * narrow
    const elev = selectedRoom ? 1.0 : 0.5
    const dir = new THREE.Vector3(0.45, elev, 0.85)
    const eye = centre.clone().addScaledVector(dir, dist)
    void c.setLookAt(eye.x, eye.y, eye.z, centre.x, centre.y, centre.z, true)
    const wide = size.width > 900
    const panel = wide ? 396 : 0, top = wide ? 72 : 64, bottom = wide ? 0 : size.height * 0.45 + 16
    const fov = THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov)
    const visH = 2 * dist * dir.length() * Math.tan(fov / 2)
    const visW = visH * (size.width / size.height)
    // camera-controls: +x shifts the scene left on screen, +y shifts it up.
    void c.setFocalOffset((visW * panel) / 2 / size.width, (visH * (bottom - top)) / 2 / size.height, 0, true)
  }, [scene, selectedRoom, explode, sceneKey, viewNonce, size.width, size.height, camera, controls])
  return null
}

function Viewport({ scene }: { scene: RoomScene }) {
  const controls = useRef<CameraControls>(null)
  const dragging = useViewer((s) => s.dragging)

  return (
    <Canvas shadows="soft" frameloop="demand" dpr={[1, 2]} camera={{ position: [14, 10, 16], fov: 38, near: 0.1, far: 400 }}
      gl={{ antialias: true, powerPreference: 'high-performance' }}>
      <Lighting scene={scene} />
      <Suspense fallback={null}>
        <House scene={scene} />
        <Appliances scene={scene} />
        <Retrofits scene={scene} />
      </Suspense>
      <CameraControls ref={controls} enabled={!dragging} makeDefault minDistance={2.5} maxDistance={70}
        maxPolarAngle={Math.PI * 0.47} smoothTime={0.35} draggingSmoothTime={0.08} dollySpeed={0.6} />
      <Framer controls={controls} scene={scene} />
    </Canvas>
  )
}

function LoadingVeil() {
  const { active, progress } = useProgress()
  if (!active) return null
  return (
    <div className="veil">
      <div className="spinner" />
      <span>Loading textures… {Math.round(progress)}%</span>
    </div>
  )
}

function ToolButton({ on, disabled, onClick, label, children }: {
  on?: boolean; disabled?: boolean; onClick: () => void; label: string; children: React.ReactNode
}) {
  return (
    <button className={`tool ${on ? 'on' : ''}`} disabled={disabled} onClick={onClick} aria-label={label} aria-pressed={on} title={label}>
      {children}
      <span className="tip">{label}</span>
    </button>
  )
}

function HeatLegend() {
  return (
    <div className="legend glass">
      <span className="legend-title">Share of heat loss</span>
      <div className="ramp" style={{ background: `linear-gradient(90deg, ${HEAT_RAMP.join(', ')})` }} />
      <div className="ramp-labels"><span>Low</span><span>High</span></div>
    </div>
  )
}

const money = (v: number) => `£${Math.round(Math.abs(v)).toLocaleString('en-GB')}`

export default function App() {
  const s = useViewer(useShallow((st) => ({
    scenes: st.scenes, sceneKey: st.sceneKey, setScene: st.setScene, showRoof: st.showRoof, toggleRoof: st.toggleRoof,
    explode: st.explode, toggleExplode: st.toggleExplode, selectedRoom: st.selectedRoom, selectRoom: st.selectRoom,
    heatTint: st.heatTint, toggleHeatTint: st.toggleHeatTint, retrofitIds: st.retrofitIds, view: st.view,
    setView: st.setView, runTwin: st.runTwin, appliances: st.appliances, scenario: st.scenario, resetView: st.resetView,
    twinBusy: st.twinBusy,
  })))
  const scene = s.scenes[s.sceneKey]
  const [intakeOpen, setIntakeOpen] = useState(false)
  const energyKey = applianceKey(s.appliances) + [...s.retrofitIds].sort().join('+') + (scene?.geometry?.confirmed_at ?? '')

  // Re-run the twin when the package, appliances or confirmed geometry change (debounced).
  useEffect(() => {
    if (!scene?.energy) return
    const t = setTimeout(() => { void s.runTwin() }, 350)
    return () => clearTimeout(t)
  }, [energyKey, s.sceneKey]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (intakeOpen) setIntakeOpen(false)
      else s.selectRoom(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [s.selectRoom, intakeOpen]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!scene) return <p className="empty">No scenes found. Run <code>npm run sync-data</code>.</p>
  const keys = Object.keys(s.scenes).sort((a, b) => {
    const ia = SCENE_ORDER.indexOf(a), ib = SCENE_ORDER.indexOf(b)
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b)
  })
  const saving = s.scenario && s.scenario.key === [...s.retrofitIds].sort().join('+') ? s.scenario.savings.cost_gbp : null

  return (
    <div className="app">
      <div className="viewport"><Viewport scene={scene} /></div>
      <LoadingVeil />

      <header className="topbar glass">
        <div className="brand"><span className="logo">G</span><div><b>GreenTwin</b><small>Room twin</small></div></div>
        <select className="scene-select" value={s.sceneKey} onChange={(e) => s.setScene(e.target.value)} aria-label="Home">
          {keys.map((k) => <option key={k} value={k}>{SCENE_NAMES[k] ?? k}</option>)}
        </select>
        {scene.energy?.engine === 'physics_stub' && (
          <span className="pill warn-pill" title={`Stand-in physics, uncalibrated, ${scene.energy.weather_source} weather`}>
            Stand-in physics · not real results
          </span>
        )}
        <button className="btn-primary" onClick={() => setIntakeOpen(true)}><Plus size={16} />New home</button>
      </header>

      <nav className="toolbar glass" aria-label="View controls">
        <ToolButton label="Roof" on={s.showRoof} disabled={!!s.selectedRoom || s.explode} onClick={s.toggleRoof}><Triangle size={18} /></ToolButton>
        <ToolButton label="Explode floors" on={s.explode} onClick={s.toggleExplode}><Layers size={18} /></ToolButton>
        <ToolButton label="Heat loss colours" on={s.heatTint} disabled={!scene.energy} onClick={s.toggleHeatTint}><Flame size={18} /></ToolButton>
        <span className="tool-sep" />
        <ToolButton label="Whole house" disabled={!s.selectedRoom} onClick={() => s.selectRoom(null)}><Home size={18} /></ToolButton>
        <ToolButton label="Reset view" onClick={s.resetView}><RotateCcw size={18} /></ToolButton>
      </nav>

      {s.heatTint && scene.energy && <HeatLegend />}

      {s.retrofitIds.length > 0 && (
        <div className="ba glass" role="group" aria-label="Before or after retrofits">
          <button className={s.view === 'before' ? 'on' : ''} onClick={() => s.setView('before')}>Before</button>
          <button className={s.view === 'after' ? 'on' : ''} onClick={() => s.setView('after')}>After · {s.retrofitIds.length}</button>
          <span className={`ba-saving ${saving && saving.headline < 0 ? 'neg' : ''}`}>
            {s.twinBusy ? 'Re-simulating…' : saving ? `${saving.headline >= 0 ? 'Saves' : 'Costs'} ${money(saving.headline)}/yr` : ''}
          </span>
        </div>
      )}

      {!s.selectedRoom && <div className="hint-chip glass">Click a room to look inside · drag to orbit · scroll to zoom</div>}

      <Panel scene={scene} />
      {intakeOpen && <IntakeForm onClose={() => setIntakeOpen(false)} />}
    </div>
  )
}
