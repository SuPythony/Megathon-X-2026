import { useState } from 'react'
import { exportPlacements, useViewer } from '../store'
import type { GeometryConfirmation, RoomScene } from '../types'

const DIRECTIONS: [number, string][] = [
  [0, 'North'], [45, 'North-east'], [90, 'East'], [135, 'South-east'],
  [180, 'South'], [225, 'South-west'], [270, 'West'], [315, 'North-west'],
]
const FORMS: [string, string][] = [
  ['detached', 'Detached'], ['semi_detached', 'Semi-detached'], ['end_terrace', 'End terrace'],
  ['mid_terrace', 'Mid terrace'], ['enclosed_mid_terrace', 'Enclosed mid terrace'], ['not_applicable', 'Flat (block)'],
]
const STATUS: Record<string, string> = {
  archetype_default: 'Estimated: standard house type',
  epc_fitted: 'Estimated: house type fitted to your EPC floor area',
  vision_fitted: 'Draft from your floor plan photo',
  user_confirmed: 'Confirmed by you',
}

function nearestDirection(az: number) {
  return DIRECTIONS.reduce((best, d) => (Math.abs(((az - d[0] + 540) % 360) - 180) < Math.abs(((az - best[0] + 540) % 360) - 180) ? d : best))[0]
}

/** Confirm or correct the fitted geometry. Rebuilding the house needs the twin server. */
export function GeometryCard({ scene }: { scene: RoomScene }) {
  const { appliances, replaceScene } = useViewer()
  const g = scene.geometry
  const [w, setW] = useState(String(scene.footprint_m[0]))
  const [d, setD] = useState(String(scene.footprint_m[1]))
  const [h, setH] = useState(String(scene.ceiling_height_m))
  const [az, setAz] = useState(nearestDirection(scene.front_azimuth_deg))
  const [form, setForm] = useState(scene.built_form)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const area = Number(w) * Number(d) * scene.storeys
  const q = scene.input_quality

  const confirm = async () => {
    setBusy(true); setMsg(null)
    const geometry: GeometryConfirmation = {
      archetype_id: scene.archetype_id, footprint_width_m: Number(w), footprint_depth_m: Number(d),
      ceiling_height_m: Number(h), front_azimuth_deg: az, built_form: form,
      room_labels: Object.fromEntries(scene.rooms.map((r) => [r.room_id, r.label])),
      confirmed_by_user: true, confirmed_at: new Date().toISOString().slice(0, 19) + '+00:00',
    }
    try {
      const res = await fetch('/api/twin/scene', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ property_id: scene.property_id, geometry, appliances: exportPlacements(scene, appliances) }),
      })
      if (!res.ok) throw new Error(await res.text())
      replaceScene({ ...(await res.json()), scenarios: {} })
      setMsg('Geometry confirmed. The twin now uses your dimensions.')
    } catch {
      setMsg('Needs the twin server to rebuild the house: uv run python scripts/twin_server.py')
    } finally {
      setBusy(false)
    }
  }

  if (!scene.energy) return null
  return (
    <section className="card">
      <h3>Geometry</h3>
      <p className={`badge ${scene.geometry_source === 'user_confirmed' ? 'ok' : 'est'}`}>
        {STATUS[scene.geometry_source] ?? scene.geometry_source}
        {g?.confirmed_at && scene.geometry_source === 'user_confirmed' ? ` on ${g.confirmed_at.slice(0, 10)}` : ''}
      </p>
      <div className="geo-grid">
        <label>Width across front (m)<input type="number" step="0.1" min="2" max="40" value={w} onChange={(e) => setW(e.target.value)} /></label>
        <label>Depth front to back (m)<input type="number" step="0.1" min="2" max="40" value={d} onChange={(e) => setD(e.target.value)} /></label>
        <label>Ceiling height (m)<input type="number" step="0.05" min="2" max="4" value={h} onChange={(e) => setH(e.target.value)} /></label>
        <label>Front faces
          <select value={az} onChange={(e) => setAz(Number(e.target.value))}>
            {DIRECTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
        <label className="span2">Built form
          <select value={form} onChange={(e) => setForm(e.target.value)}>
            {FORMS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
      </div>
      <p className="muted">Floor area with these dimensions: {area.toFixed(0)} m². Rooms keep the house type's proportions.</p>
      <button className="confirm-btn" disabled={busy} onClick={confirm}>{busy ? 'Rebuilding…' : 'Confirm geometry'}</button>
      {msg && <p className={msg.startsWith('Needs') ? 'warn' : 'ok-text'}>{msg}</p>}
      {q && (
        <div className="quality">
          <h4>Input quality (heat loss)</h4>
          <div className="qbar">
            <span style={{ width: `${q.confirmed_or_recorded_heat_loss_pct}%` }} className="good" />
            <span style={{ width: `${100 - q.confirmed_or_recorded_heat_loss_pct - q.default_or_unverified_heat_loss_pct}%` }} className="mid" />
            <span style={{ width: `${q.default_or_unverified_heat_loss_pct}%` }} className="bad" />
          </div>
          <p className="muted">
            {q.confirmed_or_recorded_heat_loss_pct}% confirmed or recorded · {q.default_or_unverified_heat_loss_pct}% defaults or
            unverified. U-values are still RdSAP defaults until verified or measured.
          </p>
        </div>
      )}
    </section>
  )
}
