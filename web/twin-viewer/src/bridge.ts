import { useViewer } from './store'

// Shared selection with the dashboard (greentwin/api/static/index.html), when the viewer runs inside its iframe.
// Same-origin postMessage only:
//   viewer -> dashboard  {type: 'greentwin:ready'} once, then {type: 'greentwin:state', property_id, retrofit_ids}
//   dashboard -> viewer  {type: 'greentwin:select', property_id, retrofit_ids}
// Both sides ignore messages that match what they already show, so the echo stops after one round.
export function startBridge() {
  if (window.parent === window) return
  const send = (msg: object) => window.parent.postMessage(msg, location.origin)
  let last = ''
  useViewer.subscribe((s) => {
    const sc = s.scenes[s.sceneKey]
    const msg = { type: 'greentwin:state', property_id: sc?.property_id, retrofit_ids: [...s.retrofitIds].sort() }
    const k = JSON.stringify(msg)
    if (k !== last) {
      last = k
      send(msg)
    }
  })
  window.addEventListener('message', (e: MessageEvent) => {
    if (e.origin !== location.origin || e.data?.type !== 'greentwin:select') return
    const { property_id, retrofit_ids } = e.data as { property_id?: string; retrofit_ids?: string[] }
    const st = useViewer.getState()
    const key = Object.keys(st.scenes).find((k) => st.scenes[k].property_id === property_id)
    if (!key) return
    if (key !== st.sceneKey) st.setScene(key)
    if (!Array.isArray(retrofit_ids)) return
    const sc = useViewer.getState().scenes[key]
    const ok = new Set((sc.retrofits ?? []).filter((o) => o.applicable).map((o) => o.retrofit_id))
    const want = retrofit_ids.filter((r) => ok.has(r)).sort()
    if (want.join('+') === [...useViewer.getState().retrofitIds].sort().join('+')) return
    // Apply through the viewer's own toggle so alternatives (e.g. double vs triple glazing) resolve the same way.
    useViewer.setState({ retrofitIds: [], view: 'before' })
    for (const r of want) useViewer.getState().toggleRetrofit(r)
  })
  send({ type: 'greentwin:ready' })
}
