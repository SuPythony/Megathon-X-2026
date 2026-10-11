import { type ReactNode, useEffect, useMemo, useState } from 'react'
import {
  Briefcase, Building2, Check, ChevronLeft, ChevronRight, Droplets, Flame, Fuel, Layers, Lightbulb, MapPin,
  BatteryCharging, Refrigerator, Sofa, Sun, Sunrise, TreePine, Wind, X, Zap,
} from 'lucide-react'
import {
  COMPLETENESS_OPTIONS, type Completeness, HEATER_USE_OPTIONS, HEATING_OPTIONS, HOT_WATER_OPTIONS, type HeaterUse, LED_OPTIONS, type LedShare, MAX_OCCUPANTS,
  type Presence, SCHEDULE_PRESETS, type SchedulePreset, THERMOSTAT_SLIDER, type TwinIntake, askHeaterUse, askLighting,
  type YesNo, hoursToPeriods, intakeWarnings, normaliseRrn, periodsToHours, postcodeDistrict,
} from '../intake'
import { exportPlacements, useViewer } from '../store'
import { Choice, HourStrip, PhotoDrop, Question, Slider } from './intake/controls'
import { MeterReadings, type MeterRow, emptyMeterRows, meterErrors, toReadings } from './intake/MeterReadings'
import './intake/intake.css'

const FIND_EPC_URL = 'https://find-energy-certificate.service.gov.uk/'
const STEPS = ['Home', 'Energy', 'Household', 'Bills', 'Rooms', 'Review'] as const
type Step = (typeof STEPS)[number]

const HEATING_ICONS: Record<string, ReactNode> = {
  gas_boiler_combi: <Flame size={18} />, gas_boiler_regular: <Flame size={18} />, air_source_heat_pump: <Wind size={18} />,
  ground_source_heat_pump: <Layers size={18} />, electric_storage: <Zap size={18} />, electric_storage_hhr: <Zap size={18} />,
  electric_radiators: <Zap size={18} />, oil_boiler: <Droplets size={18} />, lpg_boiler: <Fuel size={18} />,
  district_or_communal: <Building2 size={18} />, solid_fuel: <TreePine size={18} />,
}
const PRESENCE: [Presence, string, ReactNode][] = [
  ['out_all_day', 'Usually out', <Briefcase key="out" size={18} />],
  ['part_day', 'Sometimes home', <Sunrise key="part" size={18} />],
  ['home_all_day', 'Usually home', <Sofa key="home" size={18} />],
]

function download(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  a.click()
  URL.revokeObjectURL(url)
}

/** Groups typed digits as an EPC number (0000-0000-0000-0000-0000) while the user types. */
function formatRrn(text: string) {
  const d = text.replace(/\D/g, '').slice(0, 20)
  return d.match(/.{1,4}/g)?.join('-') ?? ''
}

const people = (n: number) => (n >= MAX_OCCUPANTS ? `${MAX_OCCUPANTS}+ people` : `${n} ${n === 1 ? 'person' : 'people'}`)

/** New home intake, one short step at a time. Only the postcode is required; every other question has "Not sure",
 * which the twin turns into an uncertain input, while explicit answers are held fixed. Produces the intake JSON in
 * the format of greentwin/twin/intake.py. */
export function IntakeForm({ onClose }: { onClose: () => void }) {
  const { scenes, sceneKey, appliances } = useViewer()
  const [step, setStep] = useState<Step>('Home')
  const [postcode, setPostcode] = useState('')
  const [epc, setEpc] = useState('')
  const [noEpc, setNoEpc] = useState(false)
  const [walls, setWalls] = useState<'solid' | 'hollow' | null>(null)
  const [heating, setHeating] = useState<string | null>(null)
  const [hotWater, setHotWater] = useState<string | null>(null)
  const [occupants, setOccupants] = useState<number | null>(null)
  const [presence, setPresence] = useState<Presence | null>(null)
  const [thermostat, setThermostat] = useState<number | null>(null)
  const [schedule, setSchedule] = useState<SchedulePreset | null>(null)
  const [hours, setHours] = useState<boolean[]>(() => periodsToHours(SCHEDULE_PRESETS.morning_evening.periods))
  const [solar, setSolar] = useState<YesNo | null>(null)
  const [pvKwp, setPvKwp] = useState<number | null>(null)
  const [battery, setBattery] = useState<YesNo | null>(null)
  const [batteryKwh, setBatteryKwh] = useState<number | null>(null)
  const [complete, setComplete] = useState<Completeness | null>(null)
  const [heaterUse, setHeaterUse] = useState<HeaterUse | null>(null)
  const [led, setLed] = useState<LedShare | null>(null)
  const [meterRows, setMeterRows] = useState<MeterRow[]>(emptyMeterRows)
  const [noGas, setNoGas] = useState(false)
  const [plan, setPlan] = useState<File | null>(null)
  const [fuse, setFuse] = useState<File | null>(null)
  const [consent, setConsent] = useState(false)
  const [done, setDone] = useState<TwinIntake | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const district = postcode.trim() ? postcodeDistrict(postcode) : null
  const rrn = !noEpc && epc ? normaliseRrn(epc) : null
  const epcError = !noEpc && epc.replace(/\D/g, '').length > 0 && !rrn ? 'An EPC number has 20 digits' : ''
  const combi = heating === 'gas_boiler_combi'
  const periods = schedule ? hoursToPeriods(hours) : []
  const heatingHours = hours.filter(Boolean).length
  const placements = useMemo(() => exportPlacements(scenes[sceneKey], appliances), [scenes, sceneKey, appliances])
  const heaterWhy = askHeaterUse(heating, placements.map((p) => p.type))
  const lightWhy = askLighting(!!rrn)
  const meterBad = meterErrors(meterRows).length > 0
  const photosNeedConsent = !!(plan || fuse) && !consent

  const blocked: Partial<Record<Step, string>> = {
    Home: !district ? 'Enter your postcode to continue' : epcError || '',
    Bills: meterBad ? 'Fix the highlighted months' : '',
    Rooms: photosNeedConsent ? 'Tick the consent switch, or remove the photos' : '',
  }
  const idx = STEPS.indexOf(step)
  const canLeave = !blocked[step]
  const reachable = (s: Step) => !!district && !epcError && STEPS.indexOf(s) <= STEPS.length - 1

  const build = (intakeId: string): TwinIntake | null => {
    if (!district) return null
    const base = {
      intake_id: intakeId,
      postcode_district: district,
      epc_rrn: rrn,
      floor_plan_image: plan?.name ?? null,
      fuse_box_image: fuse?.name ?? null,
      appliances: placements,
      photo_consent: consent,
      heating_system: heating,
      hot_water_source: combi ? null : hotWater,
      internal_walls: walls,
      occupants,
      daytime_occupancy: presence,
      thermostat_c: thermostat,
      heating_periods: periods,
      meter_readings: toReadings(meterRows, noGas),
      solar_pv: solar,
      pv_kwp: solar === 'yes' ? pvKwp : null,
      home_battery: battery,
      battery_kwh: battery === 'yes' ? batteryKwh : null,
      appliance_completeness: placements.length ? complete : null,
      plug_in_heater_use: heaterWhy ? heaterUse : null,
      led_lighting: lightWhy ? led : null,
    }
    return { ...base, warnings: intakeWarnings(base) }
  }

  const save = () => {
    const intake = build(`intake-${Date.now()}`)
    if (!intake) return
    download(`${intake.intake_id}.json`, intake)
    setDone(intake)
  }

  const nBills = toReadings(meterRows, noGas).length
  // none = an empty optional extra (shown as None), not a "not sure" answer
  const answers: { step: Step; label: string; value: string | null; none?: boolean }[] = [
    { step: 'Home', label: 'Postcode district', value: district },
    { step: 'Home', label: 'EPC number', value: rrn ?? (noEpc ? 'None' : null) },
    { step: 'Home', label: 'Internal walls', value: walls && (walls === 'solid' ? 'Solid' : 'Hollow') },
    { step: 'Energy', label: 'Heating', value: HEATING_OPTIONS.find(([v]) => v === heating)?.[1] ?? null },
    ...(combi ? [] : [{ step: 'Energy' as Step, label: 'Hot water', value: HOT_WATER_OPTIONS.find(([v]) => v === hotWater)?.[1] ?? null }]),
    { step: 'Energy', label: 'Solar panels', value: solar === 'no' ? 'None' : solar === 'yes' ? (pvKwp ? `${pvKwp} kWp` : 'Yes, size not known') : null },
    { step: 'Energy', label: 'Home battery', value: battery === 'no' ? 'None' : battery === 'yes' ? (batteryKwh ? `${batteryKwh} kWh` : 'Yes, size not known') : null },
    { step: 'Household', label: 'People', value: occupants !== null ? people(occupants) : null },
    { step: 'Household', label: 'Daytime', value: PRESENCE.find(([v]) => v === presence)?.[1] ?? null },
    { step: 'Household', label: 'Thermostat', value: thermostat !== null ? `${thermostat} °C` : null },
    { step: 'Household', label: 'Heating on', value: periods.length ? periods.map(([a, b]) => `${a}–${b}h`).join(', ') : null },
    ...(heaterWhy ? [{ step: 'Household' as Step, label: 'Plug-in heaters', value: HEATER_USE_OPTIONS.find(([v]) => v === heaterUse)?.[1] ?? null }] : []),
    ...(lightWhy ? [{ step: 'Household' as Step, label: 'LED lighting', value: LED_OPTIONS.find(([v]) => v === led)?.[1] ?? null }] : []),
    ...(placements.length ? [{ step: 'Rooms' as Step, label: 'Appliances', value: `${placements.length} placed`, none: true }] : []),
    ...(placements.length ? [{ step: 'Rooms' as Step, label: 'All main ones?', value: COMPLETENESS_OPTIONS.find(([v]) => v === complete)?.[1] ?? null }] : []),
    { step: 'Bills', label: 'Bills', value: nBills ? `${nBills} month${nBills === 1 ? '' : 's'}` : null, none: true },
    { step: 'Rooms', label: 'Photos', value: [plan && 'Floor plan', fuse && 'Fuse box'].filter(Boolean).join(', ') || null, none: true },
  ]
  const unsure = answers.filter((a) => a.value === null && !a.none).length
  const given = answers.filter((a) => a.value !== null).length

  return (
    <div className="iw-overlay" role="dialog" aria-modal="true" aria-labelledby="iw-title" onClick={onClose}>
      <div className="iw-card" onClick={(e) => e.stopPropagation()}>
        <header className="iw-header">
          <div className="iw-title-row">
            <h2 id="iw-title">New home twin</h2>
            <button type="button" className="iw-icon-btn" aria-label="Close" onClick={onClose}><X size={18} /></button>
          </div>
          {!done && (
            <ol className="iw-steps">
              {STEPS.map((s, i) => (
                <li key={s} className={i < idx ? 'past' : i === idx ? 'now' : ''}>
                  <button type="button" disabled={!reachable(s) || (i > idx && !canLeave)} onClick={() => setStep(s)}
                    aria-current={i === idx ? 'step' : undefined}>
                    <span className="iw-dot">{i < idx ? <Check size={12} /> : i + 1}</span>
                    <span className="iw-step-label">{s}</span>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </header>

        <div className="iw-body" key={done ? 'done' : step}>
          {done ? (
            <div className="iw-done">
              <div className="iw-done-mark"><Check size={28} /></div>
              <h3>Intake saved</h3>
              <p>Saved as <code>{done.intake_id}.json</code>. Answers marked not sure stay uncertain in the twin; the ones you gave are held fixed.</p>
              {done.warnings.map((w) => <p key={w} className="iw-warn">{w}</p>)}
              <p className="iw-hint">Building the twin from this intake (EPC fetch and photo reading) needs the twin API, which isn't connected yet.</p>
            </div>
          ) : step === 'Home' ? (
            <>
              <p className="iw-lead">Only your postcode is needed. Everything else is optional: answer what you know.</p>
              <Question label="Postcode" why="Required">
                <div className="iw-input-icon">
                  <MapPin size={16} />
                  <input autoFocus value={postcode} onChange={(e) => setPostcode(e.target.value)} placeholder="M14 5AB"
                    autoComplete="postal-code" aria-label="Postcode"
                    onKeyDown={(e) => { if (e.key === 'Enter' && district) setStep('Energy') }} />
                </div>
                {district ? <p className="iw-ok">Only the district <b>{district}</b> is kept. Your full postcode never leaves this device.</p>
                  : postcode.trim() ? <p className="iw-err">Enter a UK postcode, e.g. M14 5AB</p> : null}
              </Question>
              <Question label="EPC certificate number" hint="Fills in your walls, roof, windows and heating from the certificate.">
                <input className="iw-input mono" value={noEpc ? '' : epc} disabled={noEpc} inputMode="numeric"
                  placeholder="0390-2954-3640-2196-6705" aria-label="EPC certificate number"
                  onChange={(e) => setEpc(formatRrn(e.target.value))} />
                <div className="iw-row">
                  <button type="button" className={`iw-chip ghost small ${noEpc ? 'on' : ''}`} aria-pressed={noEpc}
                    onClick={() => setNoEpc(!noEpc)}>I don't have one</button>
                  <a className="iw-link" href={FIND_EPC_URL} target="_blank" rel="noreferrer">Find yours by address</a>
                </div>
                {rrn && <p className="iw-ok">Certificate {rrn}</p>}
                {epcError && <p className="iw-err">{epcError}</p>}
              </Question>
              <Question label="Knock on an inside wall. Does it sound…" hint="Tells the twin how much heat the house stores.">
                <Choice label="Internal walls" value={walls} onChange={setWalls} columns={3}
                  options={[['solid', 'Solid'], ['hollow', 'Hollow']]} />
              </Question>
            </>
          ) : step === 'Energy' ? (
            <>
              <Question label="How is your home heated?" hint="Your answer replaces the EPC, which may be out of date.">
                <Choice label="Heating system" value={heating} onChange={setHeating} columns={2}
                  options={HEATING_OPTIONS.map(([v, l]) => [v, l, HEATING_ICONS[v]] as [string, string, ReactNode])} />
              </Question>
              {combi ? (
                <p className="iw-note"><Flame size={14} /> Combi boilers heat water on demand, so there's no hot-water question.</p>
              ) : (
                <Question label="Where does your hot water come from?">
                  <Choice label="Hot water" value={hotWater} onChange={setHotWater} columns={2} options={HOT_WATER_OPTIONS} />
                </Question>
              )}
              <Question label="Do you have solar panels?" hint="Your answer replaces the EPC.">
                <Choice label="Solar panels" value={solar} onChange={setSolar} columns={3}
                  options={[['yes', 'Yes', <Sun key="y" size={18} />], ['no', 'No']]} />
                {solar === 'yes' && (
                  <div className="iw-sub">
                    <p className="iw-hint">Size, from the installer's MCS certificate. Not sure? We'll use a typical size and keep it uncertain.</p>
                    <Slider label="Solar size" value={pvKwp} onChange={setPvKwp} min={1} max={10} step={0.5} start={4}
                      format={(v) => `${v} kWp · about ${Math.round(v / 0.4)} panels`} />
                  </div>
                )}
              </Question>
              <Question label="Do you have a home battery?">
                <Choice label="Home battery" value={battery} onChange={setBattery} columns={3}
                  options={[['yes', 'Yes', <BatteryCharging key="y" size={18} />], ['no', 'No']]} />
                {battery === 'yes' && (
                  <div className="iw-sub">
                    <p className="iw-hint">Usable capacity, on the battery or its app.</p>
                    <Slider label="Battery capacity" value={batteryKwh} onChange={setBatteryKwh} min={2} max={20} step={0.5} start={5} unit=" kWh" />
                  </div>
                )}
              </Question>
            </>
          ) : step === 'Household' ? (
            <>
              <Question label="How many people usually live here?" hint="Sets hot water, appliance use and body heat.">
                <Slider label="People" value={occupants} onChange={setOccupants} min={1} max={MAX_OCCUPANTS} start={2} format={people} />
              </Question>
              <Question label="Is someone usually home during the day?">
                <Choice label="Daytime" value={presence} onChange={setPresence} columns={4} options={PRESENCE} />
              </Question>
              <Question label="What do you usually set the thermostat to?">
                <Slider label="Thermostat" value={thermostat} onChange={setThermostat} min={THERMOSTAT_SLIDER[0]}
                  max={THERMOSTAT_SLIDER[1]} step={0.5} start={20} unit=" °C" />
              </Question>
              <Question label="When is the heating usually on?">
                <Choice label="Heating schedule" value={schedule} columns={4}
                  onChange={(v) => {
                    setSchedule(v)
                    if (v && v !== 'custom') setHours(periodsToHours(SCHEDULE_PRESETS[v].periods))
                  }}
                  options={[
                    ['morning_evening', SCHEDULE_PRESETS.morning_evening.label],
                    ['all_day', SCHEDULE_PRESETS.all_day.label],
                    ['custom', 'Custom'],
                  ]} />
                {schedule && (
                  <>
                    <HourStrip hours={hours} onChange={(h) => { setHours(h); setSchedule('custom') }} />
                    <p className="iw-hint">{heatingHours ? `About ${heatingHours} hours a day. Tap or drag across hours to change.` : 'Tap the hours the heating is on.'}</p>
                  </>
                )}
              </Question>
              {(heaterWhy || lightWhy) && (
                <div className="iw-extras">
                  <h4><Lightbulb size={15} /> A couple of extras for your home</h4>
                  {heaterWhy && (
                    <Question label="How often are plug-in heaters used in winter?" why={heaterWhy}>
                      <Choice label="Plug-in heaters" value={heaterUse} onChange={setHeaterUse} columns={5} options={HEATER_USE_OPTIONS} />
                    </Question>
                  )}
                  {lightWhy && (
                    <Question label="How much of your lighting is LED?" why={lightWhy}>
                      <Choice label="LED lighting" value={led} onChange={setLed} columns={5} options={LED_OPTIONS} />
                    </Question>
                  )}
                </div>
              )}
            </>
          ) : step === 'Bills' ? (
            <>
              <p className="iw-lead">Monthly kWh from your bills or smart-meter app lets the twin learn how your home really
                uses energy. Leave a month blank if you don't have it: blank means missing, not zero.</p>
              <MeterReadings rows={meterRows} setRows={setMeterRows} noGas={noGas} setNoGas={setNoGas} />
            </>
          ) : step === 'Rooms' ? (
            <>
              {placements.length ? (
                <Question label={`You've placed ${placements.length} appliance${placements.length === 1 ? '' : 's'}. Is that all your main ones?`}
                  hint="Things like the fridge, washing machine, cooker, TV, dishwasher, tumble dryer, electric shower. Anything missing is filled in with typical use.">
                  <Choice label="Appliance list complete" value={complete} onChange={setComplete} columns={4}
                    options={COMPLETENESS_OPTIONS.map(([v, l]) => [v, l] as [Completeness, string])} />
                </Question>
              ) : (
                <p className="iw-note"><Refrigerator size={14} /> No appliances placed yet: the twin uses typical use for a home your size. Add them room by room in the viewer.</p>
              )}
              <p className="iw-lead">Photos help draw your rooms and circuits. They're optional and only used to build your twin.</p>
              <div className="iw-drops">
                <PhotoDrop label="Floor plan" hint="From an estate agent listing or a sketch" file={plan} onFile={setPlan} />
                <PhotoDrop label="Fuse box" hint="With the labels readable" file={fuse} onFile={setFuse} />
              </div>
              {(plan || fuse) && (
                <label className="iw-switch">
                  <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
                  <span className="iw-switch-track" aria-hidden />
                  I agree to these photos being processed to build my home's twin
                </label>
              )}
            </>
          ) : (
            <>
              <div className="iw-summary">
                <span><b>{given}</b> answered</span>
                <span><b>{unsure}</b> not sure</span>
              </div>
              <p className="iw-hint">Answers you gave are held fixed. Not sure ones use typical UK values with a wider range, so nothing is guessed silently.</p>
              <dl className="iw-review">
                {answers.map((a) => (
                  <div key={a.label} className="iw-review-row">
                    <dt>{a.label}</dt>
                    <dd>{a.value ?? <span className="iw-pill">{a.none ? 'None' : 'Not sure'}</span>}</dd>
                    <button type="button" className="iw-link" onClick={() => setStep(a.step)}>Edit</button>
                  </div>
                ))}
              </dl>
              {build('review')?.warnings.map((w) => <p key={w} className="iw-warn">{w}</p>)}
            </>
          )}
        </div>

        <footer className="iw-footer">
          {done ? (
            <button type="button" className="iw-btn primary" onClick={onClose}>Done</button>
          ) : (
            <>
              <button type="button" className="iw-btn" onClick={() => (idx ? setStep(STEPS[idx - 1]) : onClose())}>
                {idx ? <><ChevronLeft size={16} />Back</> : 'Cancel'}
              </button>
              <span className="iw-footer-msg">{blocked[step]}</span>
              {step === 'Review' ? (
                <button type="button" className="iw-btn primary" disabled={!district || !!epcError || meterBad || photosNeedConsent} onClick={save}>
                  <Check size={16} />Save intake
                </button>
              ) : (
                <button type="button" className="iw-btn primary" disabled={!canLeave} onClick={() => setStep(STEPS[idx + 1])}>
                  Continue<ChevronRight size={16} />
                </button>
              )}
            </>
          )}
        </footer>
      </div>
    </div>
  )
}
