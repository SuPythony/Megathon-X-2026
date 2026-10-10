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
  warnings: string[]
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
  return out
}
