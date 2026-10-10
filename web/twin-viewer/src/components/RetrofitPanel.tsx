import { useViewer } from '../store'
import type { HouseNumbers, Range, RoomScene } from '../types'

const n0 = (v: number) => Math.round(v).toLocaleString('en-GB')

function money(r: Range) {
  return `£${n0(r.headline)}`
}

/** Change per year = after - before, from the per-sample savings: negative is a reduction (green), positive means
 * use goes up (red, e.g. electricity after a heat pump). Range is the 90% interval of the change. */
function Saving({ r, unit, prefix = '' }: { r: Range | null; unit: string; prefix?: string }) {
  if (!r) return <span className="muted">n/a</span>
  const change = -r.headline
  const [lo, hi] = [-r.hi, -r.lo]
  const signed = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${prefix}${n0(Math.abs(v))}`
  return (
    <span className={change > 0 ? 'neg' : 'pos'}>
      {signed(change)} {unit} <span className="muted">(90%: {signed(lo)} to {signed(hi)})</span>
    </span>
  )
}

function Row({ label, before, after, saving, unit, prefix }: {
  label: string; before: Range | null; after: Range | null; saving: Range | null; unit: string; prefix?: string
}) {
  return (
    <tr>
      <td>{label}</td>
      <td>{before ? (prefix ? money(before) : n0(before.headline)) : 'n/a'}</td>
      <td>{after ? (prefix ? money(after) : n0(after.headline)) : 'n/a'}</td>
      <td><Saving r={saving} unit={unit} prefix={prefix} /></td>
    </tr>
  )
}

function Results({ before, after, savings }: { before: HouseNumbers; after: HouseNumbers; savings: HouseNumbers }) {
  return (
    <table className="results">
      <thead><tr><th /><th>Before</th><th>After</th><th>Change per year</th></tr></thead>
      <tbody>
        <Row label="Gas (kWh)" before={before.gas_kwh} after={after.gas_kwh} saving={savings.gas_kwh} unit="kWh" />
        <Row label="Electricity (kWh)" before={before.electricity_kwh} after={after.electricity_kwh} saving={savings.electricity_kwh} unit="kWh" />
        <Row label="Running cost" before={before.cost_gbp} after={after.cost_gbp} saving={savings.cost_gbp} unit="/yr" prefix="£" />
        <Row label="Carbon (kgCO2e)" before={before.carbon_kg} after={after.carbon_kg} saving={savings.carbon_kg} unit="kg" />
      </tbody>
    </table>
  )
}

const SOURCE_TEXT = {
  live: 'Re-simulated live by the twin server',
  precomputed: 'Precomputed scenario (start the twin server to combine retrofits)',
  unavailable: 'Not computed: start the twin server (uv run python scripts/twin_server.py) for packages or changed appliances',
  idle: '',
}

export function RetrofitPanel({ scene }: { scene: RoomScene }) {
  const { retrofitIds, toggleRetrofit, scenario, twinSource, twinBusy } = useViewer()
  const options = scene.retrofits ?? []
  if (!options.length) return null
  const byCat = options.reduce<Record<string, typeof options>>((acc, o) => ((acc[o.category] ??= []).push(o), acc), {})
  const costTotal = retrofitIds.reduce((sum, id) => {
    const o = options.find((x) => x.retrofit_id === id)
    return sum === null || !o || o.cost_gbp_central == null || o.cost_basis !== 'per_home' ? null : sum + o.cost_gbp_central
  }, 0 as number | null)

  return (
    <section className="card">
      <h3>Retrofits</h3>
      <p className="hint">Tick retrofits to see them on the house. The package is re-simulated as a whole, never added up.</p>
      {Object.entries(byCat).map(([cat, opts]) => (
        <div key={cat} className="retro-cat">
          <h4>{cat}</h4>
          {opts.map((o) => (
            <label key={o.retrofit_id} className={`retro ${o.applicable ? '' : 'disabled'}`} title={o.reason ?? ''}>
              <input type="checkbox" disabled={!o.applicable} checked={retrofitIds.includes(o.retrofit_id)}
                onChange={() => toggleRetrofit(o.retrofit_id)} />
              <span>{o.name}</span>
              <span className="muted">
                {o.cost_gbp_central != null ? `£${n0(o.cost_gbp_central)}${o.cost_basis === 'per_kwp' ? '/kWp' : ''}` : 'cost unsourced'}
              </span>
              {o.reason && <small className="muted reason">{o.reason}</small>}
            </label>
          ))}
        </div>
      ))}
      {retrofitIds.length > 0 && (
        <div className="retro-results">
          {twinBusy && <p className="muted">Re-simulating…</p>}
          {scenario && scenario.key === [...retrofitIds].sort().join('+') ? (
            <>
              <Results before={scenario.before} after={scenario.after} savings={scenario.savings} />
              {scenario.warnings.map((w) => <p key={w} className="warn">{w}</p>)}
            </>
          ) : !twinBusy && <p className="warn">{SOURCE_TEXT[twinSource] || 'Waiting for the twin…'}</p>}
          <p className="muted">
            {SOURCE_TEXT[twinSource]}. Installed cost: {costTotal != null ? `about £${n0(costTotal)}` : 'not available (some costs unsourced or sized)'}.
            Unit rates only; standing charges excluded.
          </p>
        </div>
      )}
    </section>
  )
}
