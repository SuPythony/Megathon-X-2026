import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import { Camera, Check, X } from 'lucide-react'

/** One question: label, optional hint and "why we ask", then its control. */
export function Question({ label, hint, why, children }: { label: string; hint?: string; why?: string; children: ReactNode }) {
  return (
    <section className="iw-q">
      <div className="iw-q-head">
        <h3>{label}</h3>
        {why && <span className="iw-why">{why}</span>}
      </div>
      {hint && <p className="iw-hint">{hint}</p>}
      {children}
    </section>
  )
}

/** Single choice as chips, always with a "Not sure" chip (value null). */
export function Choice<T extends string>({ value, onChange, options, label, columns, notSure = true }: {
  value: T | null; onChange: (v: T | null) => void; options: [T, string, ReactNode?][]; label: string
  columns?: number; notSure?: boolean
}) {
  return (
    <div className="iw-choice" role="radiogroup" aria-label={label}
      style={columns ? { gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` } : undefined}>
      {options.map(([v, text, icon]) => (
        <button key={v} type="button" role="radio" aria-checked={value === v}
          className={`iw-chip ${value === v ? 'on' : ''} ${icon ? 'tile' : ''}`} onClick={() => onChange(value === v ? null : v)}>
          {icon && <span className="iw-chip-icon">{icon}</span>}
          <span>{text}</span>
        </button>
      ))}
      {notSure && (
        <button type="button" role="radio" aria-checked={value === null}
          className={`iw-chip ghost ${value === null ? 'on' : ''}`} onClick={() => onChange(null)}>
          Not sure
        </button>
      )}
    </div>
  )
}

/** Slider with a value bubble; null = not sure (slider greyed, first touch sets a value). */
export function Slider({ value, onChange, min, max, step = 1, unit = '', label, format, start }: {
  value: number | null; onChange: (v: number | null) => void; min: number; max: number; step?: number
  unit?: string; label: string; format?: (v: number) => string; start: number
}) {
  const v = value ?? start
  const pct = ((v - min) / (max - min)) * 100
  const show = format ?? ((x: number) => `${x}${unit}`)
  return (
    <div className={`iw-slider ${value === null ? 'unset' : ''}`}>
      <div className="iw-slider-value" aria-live="polite">{value === null ? 'Not sure' : show(v)}</div>
      <input type="range" min={min} max={max} step={step} value={v} aria-label={label}
        aria-valuetext={value === null ? 'not sure' : show(v)} style={{ ['--pct' as string]: `${pct}%` }}
        onChange={(e) => onChange(Number(e.target.value))}
        onPointerDown={() => { if (value === null) onChange(v) }} />
      <div className="iw-slider-scale"><span>{show(min)}</span><span>{show(max)}</span></div>
      <button type="button" className={`iw-chip ghost small ${value === null ? 'on' : ''}`} onClick={() => onChange(null)}>
        Not sure
      </button>
    </div>
  )
}

/** 24 hour cells; tap, or press and drag across, to switch hours on and off. */
export function HourStrip({ hours, onChange }: { hours: boolean[]; onChange: (h: boolean[]) => void }) {
  const painting = useRef<boolean | null>(null)
  useEffect(() => {
    const up = () => { painting.current = null }
    window.addEventListener('pointerup', up)
    return () => window.removeEventListener('pointerup', up)
  }, [])
  const paint = (h: number, to: boolean) => { if (hours[h] !== to) onChange(hours.map((x, i) => (i === h ? to : x))) }
  return (
    <div className="iw-hours" role="group" aria-label="Heating hours">
      <div className="iw-hours-cells">
        {hours.map((on, h) => (
          <button key={h} type="button" aria-pressed={on} aria-label={`${h}:00 to ${h + 1}:00`}
            className={`iw-hour ${on ? 'on' : ''}`}
            onPointerDown={(e) => {
              e.preventDefault()
              ;(e.target as HTMLElement).releasePointerCapture?.(e.pointerId)  // let pointerenter fire on other cells
              painting.current = !on
              paint(h, !on)
            }}
            onPointerEnter={() => { if (painting.current !== null) paint(h, painting.current) }}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); paint(h, !on) } }} />
        ))}
      </div>
      <div className="iw-hours-scale">{[0, 6, 12, 18, 24].map((h) => <span key={h}>{h === 24 ? '24:00' : `${h}:00`}</span>)}</div>
    </div>
  )
}

/** Photo drop zone with a preview. */
export function PhotoDrop({ label, hint, file, onFile }: { label: string; hint: string; file: File | null; onFile: (f: File | null) => void }) {
  const [over, setOver] = useState(false)
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => () => { if (url) URL.revokeObjectURL(url) }, [url])
  return (
    <div className={`iw-drop ${over ? 'over' : ''} ${file ? 'has' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f?.type.startsWith('image/')) onFile(f) }}>
      {file && url ? (
        <>
          <img src={url} alt="" />
          <div className="iw-drop-meta">
            <span><Check size={14} /> {file.name}</span>
            <button type="button" className="iw-icon-btn" aria-label={`Remove ${label}`} onClick={() => onFile(null)}><X size={16} /></button>
          </div>
        </>
      ) : (
        <label className="iw-drop-empty">
          <Camera size={22} />
          <b>{label}</b>
          <span>{hint}</span>
          <span className="iw-link">Choose a photo or drop it here</span>
          <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
        </label>
      )}
    </div>
  )
}
