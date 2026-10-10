import { useState } from 'react'
import { bbox, roomStats } from '../geometry'
import { VISUALS, exportPlacements, useViewer } from '../store'
import type { RoomScene } from '../types'

const fmt = (v: number, dp = 1) => v.toFixed(dp)
const STOREY_NAMES = ['Ground floor', 'First floor', 'Second floor', 'Third floor']

function download(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  a.click()
  URL.revokeObjectURL(url)
}

function HouseSummary({ scene }: { scene: RoomScene }) {
  const selectRoom = useViewer((s) => s.selectRoom)
  const total = scene.rooms.reduce((s, r) => s + r.area_m2, 0)
  const storeys = [...new Set(scene.rooms.map((r) => r.storey))].sort()
  return (
    <>
      <section className="card">
        <h2>{scene.archetype_name}</h2>
        <dl className="facts">
          <dt>Built form</dt><dd className="cap">{scene.built_form.replace(/_/g, ' ')}</dd>
          <dt>Floor area</dt><dd>{fmt(total)} m²</dd>
          <dt>Footprint</dt><dd>{fmt(scene.footprint_m[0])} × {fmt(scene.footprint_m[1])} m</dd>
          <dt>Storeys</dt><dd>{scene.storeys}</dd>
          <dt>Front faces</dt><dd>{Math.round(scene.front_azimuth_deg)}°</dd>
        </dl>
      </section>
      <section className="card">
        <h3>Rooms</h3>
        <p className="hint">Click a room in 3D or here to view it on its own.</p>
        {storeys.map((st) => (
          <div key={st} className="storey">
            <h4>{STOREY_NAMES[st] ?? `Storey ${st}`}</h4>
            <div className="room-list">
              {scene.rooms.filter((r) => r.storey === st).map((r) => (
                <button key={r.room_id} className="room-btn" onClick={() => selectRoom(r.room_id)}>
                  <span>{r.label}</span><span className="muted">{fmt(r.area_m2)} m²</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </section>
    </>
  )
}

function RoomDetail({ scene, roomId }: { scene: RoomScene; roomId: string }) {
  const { appliances, selectedAppliance, selectRoom, selectAppliance, rotateAppliance, removeAppliance, addAppliance } =
    useViewer()
  const [newType, setNewType] = useState('')
  const room = scene.rooms.find((r) => r.room_id === roomId)!
  const stats = roomStats(room, scene.walls, scene.openings)
  const mine = appliances.filter((a) => a.room_id === roomId)
  const sel = mine.find((a) => a.appliance_id === selectedAppliance)
  const types = Object.entries(VISUALS)
    .filter(([, v]) => v.mount !== 'outdoor')
    .sort(([, a], [, b]) => Number(b.rooms.includes(room.room_type)) - Number(a.rooms.includes(room.room_type)) || a.label.localeCompare(b.label))
  const [x0, y0, x1, y1] = bbox(room)

  return (
    <>
      <section className="card">
        <button className="link" onClick={() => selectRoom(null)}>← Whole house</button>
        <h2>{room.label}</h2>
        <dl className="facts">
          <dt>Type</dt><dd className="cap">{room.room_type}</dd>
          <dt>Floor</dt><dd>{STOREY_NAMES[room.storey] ?? `Storey ${room.storey}`}</dd>
          <dt>Size</dt><dd>{fmt(x1 - x0)} × {fmt(y1 - y0)} m · {fmt(room.area_m2)} m²</dd>
          <dt>Outside walls</dt><dd>{stats.exteriorSides.length ? `${stats.exteriorSides.join(', ')} · ${fmt(stats.exteriorWallNet)} m²` : 'none'}</dd>
          <dt>Windows</dt><dd>{stats.windowCount} · {fmt(stats.windowArea, 2)} m²</dd>
          {stats.partyWall > 0 && <><dt>Party wall</dt><dd>{fmt(stats.partyWall)} m²</dd></>}
        </dl>
        <p className="pending">Heat loss and energy for this room appear once the twin has run on this home.</p>
      </section>

      <section className="card">
        <h3>Appliances ({mine.length})</h3>
        <p className="hint">Drag an appliance to move it. Positions are saved as a fraction of the room, so they survive rescaling.</p>
        <ul className="app-list">
          {mine.map((a) => (
            <li key={a.appliance_id} className={a.appliance_id === selectedAppliance ? 'active' : ''}>
              <button className="app-btn" onClick={() => selectAppliance(a.appliance_id)}>
                <span className="swatch" style={{ background: a.accent }} />
                {a.label}
                <span className="muted">{a.placement}</span>
              </button>
            </li>
          ))}
        </ul>
        {sel && (
          <div className="app-tools">
            <button onClick={() => rotateAppliance(sel.appliance_id, -90)}>⟲ 90°</button>
            <button onClick={() => rotateAppliance(sel.appliance_id, 90)}>⟳ 90°</button>
            <button className="danger" onClick={() => removeAppliance(sel.appliance_id)}>Remove</button>
            {sel.warnings.map((w) => <p key={w} className="warn">{w}</p>)}
          </div>
        )}
        <div className="add-row">
          <select value={newType} onChange={(e) => setNewType(e.target.value)} aria-label="Appliance type">
            <option value="">Add appliance…</option>
            {types.map(([k, v]) => (
              <option key={k} value={k}>{v.label}{v.rooms.includes(room.room_type) ? '' : ' (unusual here)'}</option>
            ))}
          </select>
          <button disabled={!newType} onClick={() => { addAppliance(newType, roomId); setNewType('') }}>Add</button>
        </div>
      </section>
    </>
  )
}

export function Panel({ scene }: { scene: RoomScene }) {
  const { selectedRoom, appliances, addScene } = useViewer()
  const onUpload = async (file: File | undefined) => {
    if (!file) return
    try {
      addScene(JSON.parse(await file.text()))
    } catch {
      alert('That file is not a valid room scene JSON.')
    }
  }
  return (
    <aside className="panel">
      {selectedRoom ? <RoomDetail scene={scene} roomId={selectedRoom} /> : <HouseSummary scene={scene} />}
      <section className="card actions">
        <button onClick={() => download(`${scene.property_id}_appliances.json`, exportPlacements(scene, appliances))}>
          Export appliance placements
        </button>
        <label className="upload">
          Load scene JSON
          <input type="file" accept=".json,application/json" onChange={(e) => onUpload(e.target.files?.[0])} />
        </label>
      </section>
      <p className="basis">{scene.basis}</p>
    </aside>
  )
}
