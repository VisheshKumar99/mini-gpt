import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
 
// The FastAPI backend (backend/app/main.py) is assumed to run on :8000.
// Everything under /api and /ws is proxied there so the browser sees one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})