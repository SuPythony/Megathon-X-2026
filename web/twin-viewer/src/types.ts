// Mirrors greentwin/twin/scene.py (RoomScene, room_scene/0.1). Schema: greentwin/twin/examples/room_scene.schema.json
// Plan coordinates in metres: x across the front, y from the front (0) to the back, z up.
// Box positions are bottom-centre. Rotation: 0 = front faces -y (street), 90 = +x, 180 = +y, 270 = -x.

export type Vec2 = [number, number]
export type Vec3 = [number, number, number]
export type Side = 'front' | 'back' | 'left' | 'right'

export interface SceneRoom {
  room_id: string
  label: string
  room_type: string
  storey: number
  polygon_m: Vec2[]
  area_m2: number
  floor_z_m: number
  height_m: number
}

export interface SceneWall {
  wall_id: string
  room_id: string
  storey: number
  side: Side | null
  kind: 'exterior' | 'party' | 'internal'
  start_m: Vec2
  end_m: Vec2
  length_m: number
  height_m: number
  orientation_deg: number | null
  z0_m: number
}

export interface SceneOpening {
  opening_id: string
  wall_id: string
  room_id: string
  kind: 'door' | 'window' | 'patio_door'
  width_m: number
  height_m: number
  centre_m: Vec2
  sill_m: number
  z0_m: number
}

export interface SceneRoof {
  type: 'gable' | 'hip' | 'flat'
  pitch_deg: number
  ridge_axis: 'x' | 'y'
  eave_z_m: number
  ridge_z_m: number
  colour: string
  material: string
}

export interface SceneFeature {
  kind: string
  position_m: Vec3
  size_m: Vec3
  side: Side | null
  room_id: string | null
  style: string | null
  colour: string | null
  extra: Record<string, string | number>
}

export interface ContextBlock {
  relation: 'left' | 'right' | 'back' | 'above' | 'below'
  position_m: Vec3
  size_m: Vec3
}

export interface SceneAppliance {
  appliance_id: string
  type: string
  label: string
  room_id: string | null
  placement: 'custom' | 'auto'
  position_m: Vec3
  rotation_deg: number
  size_m: Vec3
  shape: string
  mount: string
  colour: string
  accent: string
  icon: string
  answers?: Record<string, string>
  warnings: string[]
}

export interface SceneStyle {
  wall_material: string
  wall_colour: string
  wall_upper_colour: string | null
  frame_colour: string
  window_style: string
  door_colour: string
}

export interface Range {
  headline: number
  lo: number
  hi: number
  unit: string
}

export interface RoomEnergyView {
  room_id: string
  heat_loss_w_per_k: Range
  heat_loss_share_pct: number
  space_heating_kwh: Range
  appliance_kwh: Range | null
  lighting_kwh: Range
  running_cost_gbp: Range | null
  other_kwh: Record<string, number>
}

export interface EnergyOverlay {
  engine: string // 'physics' or 'physics_stub' (stand-in, never real results)
  calibrated: boolean
  weather_source: string
  rooms: RoomEnergyView[]
  unaccounted_electricity_kwh: Range | null
  basis: string
}

export interface RetrofitOption {
  retrofit_id: string
  name: string
  category: string
  applicable: boolean
  reason: string | null
  cost_gbp_low: number | null
  cost_gbp_central: number | null
  cost_gbp_high: number | null
  cost_basis: string
  cost_status: string
  excludes: string[]
  depends_on: string[]
}

export interface HouseNumbers {
  gas_kwh: Range
  electricity_kwh: Range
  cost_gbp: Range | null
  carbon_kg: Range | null
}

// One retrofit package, re-simulated as a whole. Savings = before - after per sample (negative = use goes up).
export interface ScenarioView {
  key: string
  retrofit_ids: string[]
  engine: string
  calibrated: boolean
  weather_source: string
  before: HouseNumbers
  after: HouseNumbers
  savings: HouseNumbers
  rooms_after: EnergyOverlay
  warnings: string[]
}

export interface RoomScene {
  scene_version: string
  property_id: string
  archetype_id: string
  archetype_name: string
  built_form: string
  storeys: number
  footprint_m: Vec2
  ceiling_height_m: number
  front_azimuth_deg: number
  style: SceneStyle
  rooms: SceneRoom[]
  walls: SceneWall[]
  openings: SceneOpening[]
  roof: SceneRoof
  features: SceneFeature[]
  context: ContextBlock[]
  appliances: SceneAppliance[]
  energy?: EnergyOverlay | null
  retrofits?: RetrofitOption[]
  scenarios?: Record<string, ScenarioView>
  geometry_source: string
  geometry?: GeometryConfirmation | null
  input_quality?: ProvenanceSummary | null
  basis: string
}

// greentwin/twin/geometry.py
export interface GeometryConfirmation {
  archetype_id: string
  footprint_width_m: number
  footprint_depth_m: number
  ceiling_height_m: number
  front_azimuth_deg: number
  built_form: string | null
  room_labels: Record<string, string>
  confirmed_by_user: boolean
  confirmed_at: string | null
}

// greentwin/twin/provenance.py
export interface ProvenanceSummary {
  geometry_source: string
  counts_by_status: Record<string, number>
  heat_loss_share_by_status: Record<string, number>
  confirmed_or_recorded_heat_loss_pct: number
  default_or_unverified_heat_loss_pct: number
}

// appliance_energy.json: the size questions per appliance type
export interface ApplianceQuestion {
  key: string
  label: string
  options: [string, string][]
  default: string
}

export interface ApplianceVisual {
  label: string
  shape: string
  w: number
  d: number
  h: number
  mount: string
  mount_height_m: number
  colour: string
  accent: string
  icon: string
  rooms: string[]
}

// Matches AppliancePlacement in scene.py: what the viewer sends back to build_scene.
export interface AppliancePlacement {
  appliance_id: string
  type: string
  room_id: string | null
  label: string
  u?: number
  v?: number
  x_m?: number
  y_m?: number
  rotation_deg: number
  answers?: Record<string, string>
}
