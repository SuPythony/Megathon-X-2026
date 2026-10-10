import { useTexture } from '@react-three/drei'
import { useMemo } from 'react'
import * as THREE from 'three'

// CC0 textures from Poly Haven (public/textures/CREDITS.md). Each is mapped in metres: TILE_M is how many metres one
// repeat of the texture covers, so bricks and tiles keep a real-world size on any wall or roof.
export const TEXTURES = {
  brick_red: 1.2, brick_buff: 1.2, brick_orange: 1.2, render: 2.0, pebbledash: 2.0, concrete: 2.5,
  roof_slate: 1.6, roof_clay: 1.6, roof_concrete: 1.8, floor_wood: 1.6, floor_tiles: 1.2, grass: 5.0,
} as const
export type TextureKey = keyof typeof TEXTURES

const WALL_TEXTURE: Record<string, TextureKey> = {
  red_brick: 'brick_red', buff_brick: 'brick_buff', orange_brick: 'brick_orange', white_render: 'render',
  concrete_panel: 'concrete', brick_and_pebbledash: 'brick_red',
}
const ROOF_TEXTURE: Record<string, TextureKey> = {
  slate: 'roof_slate', clay_tile: 'roof_clay', concrete_tile: 'roof_concrete', block_above: 'concrete',
}
const WET_ROOMS = new Set(['kitchen', 'bathroom', 'wc', 'utility'])

export function wallTexture(material: string, upper: boolean): TextureKey {
  if (material === 'brick_and_pebbledash' && upper) return 'pebbledash'
  return WALL_TEXTURE[material] ?? 'render'
}
export const roofTexture = (material: string): TextureKey => ROOF_TEXTURE[material] ?? 'roof_concrete'
export const floorTexture = (roomType: string): TextureKey => (WET_ROOMS.has(roomType) ? 'floor_tiles' : 'floor_wood')

export interface TexturePair { map: THREE.Texture; normalMap: THREE.Texture }

function repeat(textures: THREE.Texture | THREE.Texture[], srgb: boolean) {
  for (const t of Array.isArray(textures) ? textures : [textures]) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping
    t.anisotropy = 8
    if (srgb) t.colorSpace = THREE.SRGBColorSpace
  }
}
const asColour = (t: THREE.Texture | THREE.Texture[]) => repeat(t, true)
const asData = (t: THREE.Texture | THREE.Texture[]) => repeat(t, false)
const KEYS = Object.keys(TEXTURES) as TextureKey[]
const DIFFUSE = KEYS.map((k) => `/textures/${k}_diff.jpg`)
const NORMAL = KEYS.map((k) => `/textures/${k}_nor.jpg`)

/** All textures, loaded once (suspends until ready, so nothing pops in mid-frame). Colour maps are sRGB, normal maps
 * linear. */
export function useTextures(): Record<TextureKey, TexturePair> {
  const maps = useTexture(DIFFUSE, asColour) as THREE.Texture[]
  const normals = useTexture(NORMAL, asData) as THREE.Texture[]
  return useMemo(() => {
    const out = {} as Record<TextureKey, TexturePair>
    KEYS.forEach((k, i) => { out[k] = { map: maps[i], normalMap: normals[i] } })
    return out
  }, [maps, normals])
}

const greyCache = new WeakMap<THREE.Texture, THREE.CanvasTexture>()
/** A greyscale, brightened copy of a colour texture: keeps its detail so a material colour can tint it (e.g. lawn). */
export function greyDetail(src: THREE.Texture): THREE.Texture {
  const hit = greyCache.get(src)
  if (hit) return hit
  const img = src.image as HTMLImageElement
  const canvas = document.createElement('canvas')
  canvas.width = img.width; canvas.height = img.height
  const c = canvas.getContext('2d')!
  c.filter = 'grayscale(1) contrast(1.15) brightness(1.55)'
  c.drawImage(img, 0, 0)
  const t = new THREE.CanvasTexture(canvas)
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 8
  greyCache.set(src, t)
  return t
}

/** Box geometry whose UVs are in metres / tile, so a shared texture repeats at real-world scale on every face. */
export function metricBox(w: number, h: number, d: number, tile: number): THREE.BoxGeometry {
  const g = new THREE.BoxGeometry(w, h, d)
  const uv = g.attributes.uv as THREE.BufferAttribute
  // BoxGeometry face order: +x, -x, +y, -y, +z, -z; 4 vertices each. Face (u, v) extents in metres:
  const dims: [number, number][] = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]]
  for (let f = 0; f < 6; f++) {
    const [su, sv] = dims[f]
    for (let i = 0; i < 4; i++) {
      const k = f * 4 + i
      uv.setXY(k, (uv.getX(k) * su) / tile, (uv.getY(k) * sv) / tile)
    }
  }
  uv.needsUpdate = true
  return g
}

/** Planar UVs in metres for custom geometry (roofs): u along x, v along the slope (projected y + z). */
export function planarUV(g: THREE.BufferGeometry, tile: number): THREE.BufferGeometry {
  const pos = g.attributes.position as THREE.BufferAttribute
  const uv = new Float32Array(pos.count * 2)
  for (let i = 0; i < pos.count; i++) {
    uv[2 * i] = pos.getX(i) / tile
    uv[2 * i + 1] = (pos.getZ(i) + pos.getY(i) * 1.2) / tile
  }
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  return g
}

/** Procedural canvas textures for things no photo texture fits: PV cells and a fan grille. */
function canvasTexture(size: number, draw: (c: CanvasRenderingContext2D) => void): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  draw(canvas.getContext('2d')!)
  const t = new THREE.CanvasTexture(canvas)
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 8
  return t
}

let pvCells: THREE.CanvasTexture | null = null
export function pvTexture(): THREE.CanvasTexture {
  pvCells ??= canvasTexture(256, (c) => {
    c.fillStyle = '#0f1d3a'
    c.fillRect(0, 0, 256, 256)
    const n = 6, gap = 3, cell = (256 - gap * (n + 1)) / n
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = gap + i * (cell + gap), y = gap + j * (cell + gap)
      const g = c.createLinearGradient(x, y, x + cell, y + cell)
      g.addColorStop(0, '#1d3a73'); g.addColorStop(1, '#0d1f45')
      c.fillStyle = g
      c.fillRect(x, y, cell, cell)
      c.strokeStyle = 'rgba(190,205,230,0.35)'
      c.lineWidth = 1
      for (let k = 1; k < 4; k++) { c.beginPath(); c.moveTo(x + (cell * k) / 4, y); c.lineTo(x + (cell * k) / 4, y + cell); c.stroke() }
    }
  })
  return pvCells
}

let grille: THREE.CanvasTexture | null = null
export function grilleTexture(): THREE.CanvasTexture {
  grille ??= canvasTexture(256, (c) => {
    c.fillStyle = '#2f3840'
    c.fillRect(0, 0, 256, 256)
    c.strokeStyle = '#8c969e'
    c.lineWidth = 6
    for (let r = 20; r < 128; r += 18) { c.beginPath(); c.arc(128, 128, r, 0, Math.PI * 2); c.stroke() }
    c.lineWidth = 8
    for (let a = 0; a < 4; a++) {
      c.beginPath(); c.moveTo(128, 128)
      c.lineTo(128 + 120 * Math.cos((a * Math.PI) / 2 + 0.4), 128 + 120 * Math.sin((a * Math.PI) / 2 + 0.4)); c.stroke()
    }
  })
  return grille
}

let fluff: THREE.CanvasTexture | null = null
export function insulationTexture(): THREE.CanvasTexture {
  fluff ??= canvasTexture(256, (c) => {
    c.fillStyle = '#f4c542'
    c.fillRect(0, 0, 256, 256)
    for (let i = 0; i < 2600; i++) {
      const x = Math.random() * 256, y = Math.random() * 256, r = Math.random() * 3 + 1
      c.fillStyle = Math.random() > 0.5 ? 'rgba(255,236,170,0.55)' : 'rgba(205,150,30,0.35)'
      c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill()
    }
  })
  fluff.wrapS = fluff.wrapT = THREE.RepeatWrapping
  return fluff
}
