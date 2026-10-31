import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // maplibre-gl ships its own web worker; Vite's dep optimizer drops the worker
  // chunk, which leaves the map blank in dev.
  optimizeDeps: { exclude: ['maplibre-gl'] },
})
