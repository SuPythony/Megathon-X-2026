import { useState } from 'react'
import { type TwinIntake, intakeWarnings, normaliseRrn, postcodeDistrict } from '../intake'
import { exportPlacements, useViewer } from '../store'

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
  const [done, setDone] = useState<TwinIntake | null>(null)

  const district = postcode.trim() ? postcodeDistrict(postcode) : null
  const rrn = epc.trim() ? normaliseRrn(epc) : null
  const postcodeError = postcode.trim() && !district ? 'Enter a UK postcode, e.g. M14 5AB' : ''
  const epcError = !noEpc && epc.trim() && !rrn ? 'An EPC certificate number has 20 digits, e.g. 0390-2954-3640-2196-6705' : ''
  const photosNeedConsent = (plan || fuse) && !consent
  const canSubmit = !!district && (noEpc || !!rrn) && !photosNeedConsent

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
