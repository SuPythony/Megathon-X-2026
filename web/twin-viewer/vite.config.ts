import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// /api goes to the dev twin server (scripts/twin_server.py) for live re-simulation; the viewer works without it.
// `npm run build` writes into the main app (greentwin/api/static/twin), served at /twin with /api/twin on the same origin.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: '../../greentwin/api/static/twin', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
})
