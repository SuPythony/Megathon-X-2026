// Mirrors greentwin/twin/intake.py. Location is reduced to the postcode district here, in the browser, so the
// full postcode never leaves the device (AGENTS.md rule 8).
import type { AppliancePlacement } from './types'

const FULL_POSTCODE = /^([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})$/
const DISTRICT = /^[A-Z]{1,2}\d[A-Z\d]?$/
const OUTSIDE_EW: Record<string, string> = {
  BT: 'Northern Ireland',
  ...Object.fromEntries(['AB', 'DD', 'DG', 'EH', 'FK', 'G', 'HS', 'IV', 'KA', 'KW', 'KY', 'ML', 'PA', 'PH', 'ZE'].map((a) => [a, 'Scotland'])),
}

export function postcodeDistrict(text: string): string | null {
  const s = text.trim().toUpperCase().replace(/\s+/g, ' ')
  const m = s.match(FULL_POSTCODE)
  if (m) return m[1]
  const compact = s.replace(/ /g, '')
  return DISTRICT.test(compact) ? compact : null
}

export function normaliseRrn(text: string): string | null {
  const digits = text.replace(/[\s-]/g, '')
  if (!/^\d{20}$/.test(digits)) return null
  return digits.match(/.{4}/g)!.join('-')
}

export interface TwinIntake {
  intake_id: string
  postcode_district: string
  epc_rrn: string | null
  floor_plan_image: string | null
  fuse_box_image: string | null
  appliances: AppliancePlacement[]
  photo_consent: boolean
  heating_system: string | null
  hot_water_source: string | null
  internal_walls: 'solid' | 'hollow' | null
  // Household (greentwin/twin/household.py): null / [] = not sure, so typical UK values with wider uncertainty are used
  occupants: number | null
  daytime_occupancy: Presence | null
  thermostat_c: number | null
  heating_periods: [number, number][]
  meter_readings: MeterReading[]
  warnings: string[]
}

// One month of bills, as greentwin/twin/intake.py MeterReading. null = missing (never 0); gas 0 only with no gas supply.
export interface MeterReading {
  month: string // YYYY-MM
  electricity_kwh: number | null
  gas_kwh: number | null
  estimated: boolean
}

export const WINTER = new Set([11, 12, 1, 2, 3])
export const SUMMER = new Set([6, 7, 8])

/** The last `n` complete months before `today`, oldest first, as YYYY-MM. */
export function lastMonths(n: number, today = new Date()): string[] {
  const out: string[] = []
  for (let i = n; i >= 1; i--) {
    const d = new Date(today.getFullYear(), today.getMonth() - i, 1)
    out.push(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`)
  }
  return out
}

/** Same rule as greentwin/twin/inputs.py _readiness: what calibration can do with these bills. */
export function meterReadiness(rows: MeterReading[]): { method: 'prior_only' | 'weak' | 'ready'; months: number; winter: number; summer: number } {
  const used = rows.filter((r) => r.electricity_kwh != null || r.gas_kwh != null)
  const moy = used.map((r) => Number(r.month.slice(5, 7)))
  const winter = moy.filter((m) => WINTER.has(m)).length
  const summer = moy.filter((m) => SUMMER.has(m)).length
  const method = !used.length ? 'prior_only' : used.length >= 12 && winter >= 3 && summer >= 2 ? 'ready' : 'weak'
  return { method, months: used.length, winter, summer }
}

/** Parse pasted rows: "2025-01, 310, 1840" or "2025-01\t310\t1840\tE" (E = estimated); blank cells = missing. */
export function parseMeterPaste(text: string): { rows: MeterReading[]; errors: string[] } {
  const rows: MeterReading[] = []
  const errors: string[] = []
  text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).forEach((line, i) => {
    const cells = line.split(/[,\t;]/).map((c) => c.trim())
    const [month, elec, gas, flag] = cells
    if (!/^\d{4}-(0[1-9]|1[0-2])$/.test(month ?? '')) { errors.push(`Line ${i + 1}: month must be YYYY-MM`); return }
    const num = (v?: string) => (v === undefined || v === '' ? null : Number(v))
    const e = num(elec), g = num(gas)
    if ((e != null && !(e >= 0)) || (g != null && !(g >= 0))) { errors.push(`Line ${i + 1}: kWh must be numbers ≥ 0`); return }
    rows.push({ month, electricity_kwh: e, gas_kwh: g, estimated: /^e(st(imated)?)?$/i.test(flag ?? '') })
  })
  const months = rows.map((r) => r.month)
  if (new Set(months).size !== months.length) errors.push('A month appears twice')
  return { rows, errors }
}

export type Presence = 'out_all_day' | 'part_day' | 'home_all_day'
export const PRESENCE_OPTIONS: [Presence, string][] = [
  ['out_all_day', 'Usually out during the day'],
  ['part_day', 'Sometimes home during the day'],
  ['home_all_day', 'Usually home during the day'],
]

export type SchedulePreset = 'morning_evening' | 'all_day' | 'custom'
export const SCHEDULE_PRESETS: Record<Exclude<SchedulePreset, 'custom'>, { label: string; periods: [number, number][] }> = {
  morning_evening: { label: 'Mornings and evenings (6–9am, 5–10pm)', periods: [[6, 9], [17, 22]] },
  all_day: { label: 'All day (6am–10pm)', periods: [[6, 22]] },
}

export const THERMOSTAT_RANGE: [number, number] = [10, 28] // matches TwinIntake.thermostat_c in intake.py
export const MAX_OCCUPANTS = 12

/** Problems with custom heating periods (hours 0-24, start before end, no overlaps); empty when fine. */
export function periodErrors(periods: [number, number][]): string[] {
  const out: string[] = []
  periods.forEach(([s, e], i) => {
    if (!(s >= 0 && s <= 23 && e >= 1 && e <= 24)) out.push(`Period ${i + 1}: hours must be between 0 and 24`)
    else if (s >= e) out.push(`Period ${i + 1}: the heating must go on before it goes off`)
  })
  const sorted = [...periods].sort((a, b) => a[0] - b[0])
  for (let i = 1; i < sorted.length; i++) if (sorted[i][0] < sorted[i - 1][1]) { out.push('Periods overlap'); break }
  return out
}

// Heating as the user knows it today; overrides the EPC (greentwin/twin/household.py apply_user_heating).
export const HEATING_OPTIONS: [string, string][] = [
  ['gas_boiler_combi', 'Gas boiler (combi, no hot water tank)'],
  ['gas_boiler_regular', 'Gas boiler with a hot water tank'],
  ['air_source_heat_pump', 'Air-source heat pump'],
  ['ground_source_heat_pump', 'Ground-source heat pump'],
  ['electric_storage', 'Electric storage heaters'],
  ['electric_storage_hhr', 'High-heat-retention storage heaters'],
  ['electric_radiators', 'Electric radiators / panel heaters'],
  ['oil_boiler', 'Oil boiler'],
  ['lpg_boiler', 'LPG boiler'],
  ['district_or_communal', 'District / communal heating'],
  ['solid_fuel', 'Solid fuel (wood, coal)'],
]
export const HOT_WATER_OPTIONS: [string, string][] = [
  ['from_boiler', 'From the boiler on demand (combi)'],
  ['cylinder_from_boiler', 'Hot water tank heated by the boiler'],
  ['electric_immersion', 'Electric immersion heater in a tank'],
  ['heat_pump_cylinder', 'Heat pump with a hot water tank'],
  ['instantaneous_electric', 'Instant electric water heater'],
]

export function intakeWarnings(i: Omit<TwinIntake, 'warnings'>): string[] {
  const out: string[] = []
  const nation = OUTSIDE_EW[i.postcode_district.match(/^[A-Z]+/)![0]]
  if (nation) out.push(`${i.postcode_district} is in ${nation}; our EPC and energy data cover England & Wales only`)
  if (!i.epc_rrn) out.push('No EPC number: the twin starts from the closest house type and the floor plan only')
  if ((i.floor_plan_image || i.fuse_box_image) && !i.photo_consent)
    out.push('Photos supplied without consent; they will not be processed')
  const used = i.meter_readings.filter((r) => r.electricity_kwh != null || r.gas_kwh != null).length
  if (!used) out.push("No meter readings: the twin can't be calibrated to this home yet (prior only)")
  else if (used < 12) out.push(`Only ${used} months of meter readings; 12+ covering winter and summer are needed for a reliable calibration`)
  if (i.occupants == null || i.thermostat_c == null || !i.heating_periods.length)
    out.push('Some household answers are "not sure": the twin uses typical UK values with wider uncertainty')
  return out
}
