import { useState } from 'react'
import { type MeterReading, lastMonths, meterReadiness, parseMeterPaste } from '../intake'

export interface MeterRow { month: string; elec: string; gas: string; estimated: boolean }

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
// Monthly kWh above these is very unusual for a home: shown as a "check units" hint, never blocked.
const CHECK_ELEC = 2500, CHECK_GAS = 8000

export function emptyMeterRows(): MeterRow[] {
  return lastMonths(12).map((month) => ({ month, elec: '', gas: '', estimated: false }))
}

const num = (v: string) => (v.trim() === '' ? null : Number(v))

/** Rows -> MeterReading[] for the intake: blank = missing (left out), never 0; gas 0 only with no gas supply. */
export function toReadings(rows: MeterRow[], noGas: boolean): MeterReading[] {
  return rows
    .map((r) => ({ month: r.month, electricity_kwh: num(r.elec), gas_kwh: noGas ? 0 : num(r.gas), estimated: r.estimated }))
    .filter((r) => r.electricity_kwh != null || (!noGas && r.gas_kwh != null))
}

export function meterErrors(rows: MeterRow[]): string[] {
  const bad = rows.filter((r) => [r.elec, r.gas].some((v) => v.trim() !== '' && !(Number(v) >= 0)))
  return bad.length ? [`${bad.length} month(s) have values that aren't numbers ≥ 0`] : []
}

function label(month: string) {
  return `${MONTH_NAMES[Number(month.slice(5, 7)) - 1]} ${month.slice(0, 4)}`
}

/** Optional monthly bills: the data the Bayesian calibration fits the twin to. */
export function MeterReadings({ rows, setRows, noGas, setNoGas }: {
  rows: MeterRow[]; setRows: (r: MeterRow[]) => void; noGas: boolean; setNoGas: (v: boolean) => void
}) {
  const [paste, setPaste] = useState('')
  const [pasteMsg, setPasteMsg] = useState<string[]>([])
  const readings = toReadings(rows, noGas)
  const ready = meterReadiness(readings)
  const errors = meterErrors(rows)
  const set = (i: number, patch: Partial<MeterRow>) => setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  const applyPaste = () => {
    const { rows: parsed, errors: errs } = parseMeterPaste(paste)
    if (errs.length) { setPasteMsg(errs); return }
    const fmt = (v: number | null) => (v == null ? '' : String(v))
    setRows(parsed.sort((a, b) => a.month.localeCompare(b.month))
      .map((r) => ({ month: r.month, elec: fmt(r.electricity_kwh), gas: fmt(r.gas_kwh), estimated: r.estimated })))
    setPasteMsg([`Loaded ${parsed.length} months`])
  }

  return (
    <details className="group bills">
      <summary>
        <span>Your energy bills <span className="muted">(optional, recommended)</span></span>
        <span className={`badge ${ready.method === 'ready' ? 'ok' : 'est'}`}>
          {ready.method === 'ready' ? 'Enough to calibrate' : ready.method === 'weak' ? 'Partial' : 'None yet'}
        </span>
      </summary>
      <p className="muted">
        Monthly kWh from your bills or smart-meter app lets the twin learn how your home really uses energy. Leave a month
        blank if you don't have it: blank means missing, not zero.
      </p>
      <label className="check">
        <input type="checkbox" checked={noGas} onChange={(e) => setNoGas(e.target.checked)} />
        This home has no gas supply
      </label>
      <div className="meter-table" role="table" aria-label="Monthly meter readings">
        <div className="meter-head" role="row">
          <span>Month</span><span>Electricity kWh</span><span>Gas kWh</span><span title="Estimated read on the bill">Est.</span>
        </div>
        {rows.map((r, i) => {
          const hi = (Number(r.elec) > CHECK_ELEC) || (!noGas && Number(r.gas) > CHECK_GAS)
          return (
            <div key={r.month} className={`meter-row ${hi ? 'hi' : ''}`} role="row">
              <span>{label(r.month)}</span>
              <input inputMode="decimal" aria-label={`Electricity ${r.month}`} value={r.elec} placeholder="—"
                onChange={(e) => set(i, { elec: e.target.value })} />
              <input inputMode="decimal" aria-label={`Gas ${r.month}`} value={noGas ? '0' : r.gas} placeholder="—"
                disabled={noGas} onChange={(e) => set(i, { gas: e.target.value })} />
              <input type="checkbox" aria-label={`Estimated read ${r.month}`} checked={r.estimated}
                onChange={(e) => set(i, { estimated: e.target.checked })} />
            </div>
          )
        })}
      </div>
      {errors.map((e) => <small key={e} className="err">{e}</small>)}
      {rows.some((r) => Number(r.elec) > CHECK_ELEC || (!noGas && Number(r.gas) > CHECK_GAS)) &&
        <small className="warn">Some months look very high: check they're kWh, not £ or meter readings.</small>}
      <p className={ready.method === 'ready' ? 'ok-text' : 'muted'}>
        {ready.months} months · {ready.winter} winter · {ready.summer} summer.{' '}
        {ready.method === 'ready' ? 'Enough for the twin to calibrate to your home.'
          : 'For a reliable calibration: 12+ months including at least 3 winter and 2 summer months.'}
      </p>
      <details className="paste">
        <summary>Paste from a spreadsheet</summary>
        <textarea rows={4} value={paste} onChange={(e) => setPaste(e.target.value)}
          placeholder={'2025-01, 310, 1840\n2025-02, 290, 1620, E\n(month, electricity kWh, gas kWh, E = estimated)'} />
        <div className="inline">
          <button type="button" className="chip" onClick={applyPaste} disabled={!paste.trim()}>Load</button>
          <button type="button" className="chip" onClick={() => { setRows(emptyMeterRows()); setPasteMsg([]) }}>Reset to last 12 months</button>
        </div>
        {pasteMsg.map((m) => <small key={m} className={m.startsWith('Loaded') ? 'ok' : 'err'}>{m}</small>)}
      </details>
    </details>
  )
}
