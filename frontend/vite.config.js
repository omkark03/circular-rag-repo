import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api and /files are proxied to the FastAPI backend on :8000
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000',
                timeout: 600000, proxyTimeout: 600000 },
      '/files': 'http://localhost:8000',
    },
  },
})
