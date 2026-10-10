import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// /api goes to the dev twin server (scripts/twin_server.py) for live re-simulation; the viewer works without it.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
})
