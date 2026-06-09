import { defineConfig, configDefaults } from 'vitest/config';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [react(), wasm(), topLevelAwait()],
  build: { target: 'esnext' },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/setupTests.ts'],
    // Extend Vitest's default excludes so dist/config globs are preserved.
    // E2E specs run via `pnpm e2e`, not vitest.
    exclude: [...configDefaults.exclude, 'e2e/**'],
  },
});
