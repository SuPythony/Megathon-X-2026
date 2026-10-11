import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { useViewer } from './store'
import { startBridge } from './bridge'

// Dev-only handle for browser smoke tests.
if (import.meta.env.DEV) Object.assign(window, { __viewer: useViewer })

startBridge()  // shared home + retrofit selection with the dashboard when embedded

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
