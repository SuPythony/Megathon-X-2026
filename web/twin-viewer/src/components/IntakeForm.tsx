import { useState } from 'react'
import {
  HEATING_OPTIONS, HOT_WATER_OPTIONS, MAX_OCCUPANTS, PRESENCE_OPTIONS, type Presence, SCHEDULE_PRESETS, type SchedulePreset,
  THERMOSTAT_RANGE, type TwinIntake, intakeWarnings, normaliseRrn, periodErrors, postcodeDistrict,
} from '../intake'
import { exportPlacements, useViewer } from '../store'
import { MeterReadings, type MeterRow, emptyMeterRows, meterErrors, toReadings } from './MeterReadings'

const FIND_EPC_URL = 'https://find-energy-certificate.service.gov.uk/'

function download(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  a.click()
  URL.revokeObjectURL(url)
}

/** Collects what's needed to build a home's twin. Processing (EPC fetch, vision) needs the twin API; until it
 * exists, the form produces the intake JSON in the format of greentwin/twin/intake.py. */
export function IntakeForm({ onClose }: { onClose: () => void }) {
  const { scenes, sceneKey, appliances } = useViewer()
  const [postcode, setPostcode] = useState('')
  const [epc, setEpc] = useState('')
  const [noEpc, setNoEpc] = useState(false)
  const [plan, setPlan] = useState<File | null>(null)
  const [fuse, setFuse] = useState<File | null>(null)
  const [consent, setConsent] = useState(false)
  const [heating, setHeating] = useState('')
  const [hotWater, setHotWater] = useState('')
  const [walls, setWalls] = useState<'' | 'solid' | 'hollow'>('')
  const [occupants, setOccupants] = useState('')
  const [presence, setPresence] = useState<'' | Presence>('')
  const [thermostat, setThermostat] = useState('')
  const [schedule, setSchedule] = useState<'' | SchedulePreset>('')
  const [custom, setCustom] = useState<[number, number][]>([[6, 9], [17, 22]])
  const [meterRows, setMeterRows] = useState<MeterRow[]>(emptyMeterRows)
  const [noGas, setNoGas] = useState(false)
  const [done, setDone] = useState<TwinIntake | null>(null)

  const district = postcode.trim() ? postcodeDistrict(postcode) : null
  const rrn = epc.trim() ? normaliseRrn(epc) : null
  const postcodeError = postcode.trim() && !district ? 'Enter a UK postcode, e.g. M14 5AB' : ''
  const epcError = !noEpc && epc.trim() && !rrn ? 'An EPC certificate number has 20 digits, e.g. 0390-2954-3640-2196-6705' : ''
  const photosNeedConsent = (plan || fuse) && !consent
  const stat = thermostat.trim() === '' ? null : Number(thermostat)
  const statError = stat !== null && !(stat >= THERMOSTAT_RANGE[0] && stat <= THERMOSTAT_RANGE[1])
    ? `Enter a temperature between ${THERMOSTAT_RANGE[0]} and ${THERMOSTAT_RANGE[1]} °C` : ''
  const periods: [number, number][] = schedule === 'custom' ? custom : schedule ? SCHEDULE_PRESETS[schedule].periods : []
  const scheduleErrors = schedule === 'custom' ? periodErrors(custom) : []
  const heatingHours = periods.reduce((h, [a, b]) => h + Math.max(0, b - a), 0)
  const canSubmit = !!district && (noEpc || !!rrn) && !photosNeedConsent && !statError && !scheduleErrors.length
    && !meterErrors(meterRows).length

  const submit = () => {
    if (!district) return
    const base = {
      intake_id: `intake-${Date.now()}`,
      postcode_district: district,
      epc_rrn: noEpc ? null : rrn,
      floor_plan_image: plan?.name ?? null,
      fuse_box_image: fuse?.name ?? null,
      appliances: exportPlacements(scenes[sceneKey], appliances),
      photo_consent: consent,
      heating_system: heating || null,
      hot_water_source: hotWater || null,
      internal_walls: walls || null,
      occupants: occupants ? Number(occupants) : null,
      daytime_occupancy: presence || null,
      thermostat_c: stat,
      heating_periods: periods,
      meter_readings: toReadings(meterRows, noGas),
    }
    const intake: TwinIntake = { ...base, warnings: intakeWarnings(base) }
    download(`${intake.intake_id}.json`, intake)
    setDone(intake)
  }

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="intake-title" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2 id="intake-title">New home twin</h2>
        {done ? (
          <>
            <p>Intake saved as <code>{done.intake_id}.json</code>.</p>
            <dl className="facts">
              <dt>Postcode district</dt><dd>{done.postcode_district}</dd>
              <dt>EPC number</dt><dd>{done.epc_rrn ?? 'none'}</dd>
              <dt>Floor plan</dt><dd>{done.floor_plan_image ?? 'none'}</dd>
              <dt>Fuse box</dt><dd>{done.fuse_box_image ?? 'none'}</dd>
              <dt>Heating</dt><dd>{HEATING_OPTIONS.find(([v]) => v === done.heating_system)?.[1] ?? 'from EPC'}</dd>
              <dt>Hot water</dt><dd>{HOT_WATER_OPTIONS.find(([v]) => v === done.hot_water_source)?.[1] ?? 'from EPC'}</dd>
              <dt>Internal walls</dt><dd>{done.internal_walls ?? 'not sure'}</dd>
              <dt>Occupants</dt><dd>{done.occupants ?? 'not sure'}</dd>
              <dt>Daytime</dt><dd>{PRESENCE_OPTIONS.find(([v]) => v === done.daytime_occupancy)?.[1] ?? 'not sure'}</dd>
              <dt>Thermostat</dt><dd>{done.thermostat_c != null ? `${done.thermostat_c} °C` : 'not sure'}</dd>
              <dt>Heating on</dt><dd>{done.heating_periods.length
                ? done.heating_periods.map(([a, b]) => `${a}:00–${b}:00`).join(', ') : 'not sure'}</dd>
              <dt>Bills</dt><dd>{done.meter_readings.length ? `${done.meter_readings.length} months` : 'none'}</dd>
              <dt>Appliances</dt><dd>{done.appliances.length}</dd>
            </dl>
            {done.warnings.map((w) => <p key={w} className="warn">{w}</p>)}
            <p className="pending">Building the twin from this intake (EPC fetch and photo reading) needs the twin API, which isn't connected yet.</p>
            <div className="modal-actions"><button className="primary" onClick={onClose}>Close</button></div>
          </>
        ) : (
          <>
            <label className="field">
              <span>Postcode</span>
              <input value={postcode} onChange={(e) => setPostcode(e.target.value)} placeholder="M14 5AB" autoComplete="postal-code" />
              {district && <small className="ok">Only the district <b>{district}</b> is kept. Your full postcode isn't stored or sent.</small>}
              {postcodeError && <small className="err">{postcodeError}</small>}
            </label>

            <label className="field">
              <span>EPC certificate number</span>
              <input value={epc} onChange={(e) => setEpc(e.target.value)} placeholder="0390-2954-3640-2196-6705"
                disabled={noEpc} inputMode="numeric" />
              {rrn && !noEpc && <small className="ok">Certificate {rrn}</small>}
              {epcError && <small className="err">{epcError}</small>}
              <small className="muted">
                Find it on your certificate, or search by address at{' '}
                <a href={FIND_EPC_URL} target="_blank" rel="noreferrer">find-energy-certificate.service.gov.uk</a>.
              </small>
            </label>
            <label className="check">
              <input type="checkbox" checked={noEpc} onChange={(e) => setNoEpc(e.target.checked)} />
              I don't have an EPC number
            </label>

            <fieldset className="group">
              <legend>Your household</legend>
              <p className="muted">Any you're not sure about: the twin uses typical UK values and widens its ranges.</p>
              <div className="row2">
                <label className="field">
                  <span>How many people usually live here?</span>
                  <select value={occupants} onChange={(e) => setOccupants(e.target.value)}>
                    <option value="">Not sure</option>
                    {Array.from({ length: MAX_OCCUPANTS }, (_, i) => i + 1).map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span>Is someone usually home during the day?</span>
                  <select value={presence} onChange={(e) => setPresence(e.target.value as '' | Presence)}>
                    <option value="">Not sure</option>
                    {PRESENCE_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                  </select>
                </label>
              </div>
              <label className="field">
                <span>What temperature do you usually set the thermostat to?</span>
                <div className="inline">
                  <input type="number" inputMode="decimal" step="0.5" min={THERMOSTAT_RANGE[0]} max={THERMOSTAT_RANGE[1]}
                    value={thermostat} onChange={(e) => setThermostat(e.target.value)} placeholder="e.g. 20" aria-label="Thermostat °C" />
                  <span className="unit">°C</span>
                  <button type="button" className="chip" onClick={() => setThermostat('')}>Not sure</button>
                </div>
                {statError && <small className="err">{statError}</small>}
              </label>
              <label className="field">
                <span>When is the heating usually on?</span>
                <select value={schedule} onChange={(e) => setSchedule(e.target.value as '' | SchedulePreset)}>
                  <option value="">Not sure</option>
                  {Object.entries(SCHEDULE_PRESETS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
                  <option value="custom">Custom times</option>
                </select>
              </label>
              {schedule === 'custom' && (
                <div className="periods">
                  {custom.map(([a, b], i) => (
                    <div key={i} className="period">
                      <span className="muted">On</span>
                      <select aria-label={`Period ${i + 1} on`} value={a}
                        onChange={(e) => setCustom(custom.map((p, j) => (j === i ? [Number(e.target.value), p[1]] : p)))}>
                        {Array.from({ length: 24 }, (_, h) => <option key={h} value={h}>{`${h}:00`}</option>)}
                      </select>
                      <span className="muted">off</span>
                      <select aria-label={`Period ${i + 1} off`} value={b}
                        onChange={(e) => setCustom(custom.map((p, j) => (j === i ? [p[0], Number(e.target.value)] : p)))}>
                        {Array.from({ length: 24 }, (_, h) => h + 1).map((h) => <option key={h} value={h}>{`${h}:00`}</option>)}
                      </select>
                      {custom.length > 1 && (
                        <button type="button" className="chip" onClick={() => setCustom(custom.filter((_, j) => j !== i))}>Remove</button>
                      )}
                    </div>
                  ))}
                  {custom.length < 3 && (
                    <button type="button" className="chip" onClick={() => setCustom([...custom, [12, 14]])}>Add a period</button>
                  )}
                  {scheduleErrors.map((e) => <small key={e} className="err">{e}</small>)}
                </div>
              )}
              {periods.length > 0 && !scheduleErrors.length && <small className="ok">About {heatingHours} hours of heating a day.</small>}
            </fieldset>

            <MeterReadings rows={meterRows} setRows={setMeterRows} noGas={noGas} setNoGas={setNoGas} />

            <label className="field">
              <span>Heating system</span>
              <select value={heating} onChange={(e) => setHeating(e.target.value)}>
                <option value="">Use my EPC</option>
                {HEATING_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
              <small className="muted">What heats the home today. Overrides the EPC, which may be years old.</small>
            </label>
            <label className="field">
              <span>Hot water</span>
              <select value={hotWater} onChange={(e) => setHotWater(e.target.value)}>
                <option value="">Use my EPC</option>
                {HOT_WATER_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Internal walls</span>
              <select value={walls} onChange={(e) => setWalls(e.target.value as '' | 'solid' | 'hollow')}>
                <option value="">Not sure</option>
                <option value="solid">Solid when knocked (brick or block)</option>
                <option value="hollow">Hollow when knocked (plasterboard)</option>
              </select>
              <small className="muted">Tells the twin how much heat the house stores (thermal mass).</small>
            </label>

            <label className="field">
              <span>Floor plan photo</span>
              <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setPlan(e.target.files?.[0] ?? null)} />
            </label>
            <label className="field">
              <span>Fuse box photo</span>
              <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setFuse(e.target.files?.[0] ?? null)} />
            </label>
            <label className="check">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              I agree to these interior photos being processed to build my home's twin
            </label>
            {photosNeedConsent && <small className="err">Tick the consent box to include photos, or remove them.</small>}

            <p className="muted">Appliances: the {appliances.length} currently placed in the viewer are included. You can add more room by room.</p>
            <div className="modal-actions">
              <button onClick={onClose}>Cancel</button>
              <button className="primary" disabled={!canSubmit} onClick={submit}>Save intake</button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
