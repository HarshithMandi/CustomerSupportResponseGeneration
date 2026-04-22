import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

// Vite config for the React frontend.
// Loads VITE_* variables from the repo-root `.env` (one level above /frontend).
const __dirname = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  // Load VITE_* variables from the repo root .env
  envDir: resolve(__dirname, '..'),
  plugins: [react()],
  server: {
    port: 5173
  }
})
