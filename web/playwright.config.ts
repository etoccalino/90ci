import { defineConfig, devices } from '@playwright/test';

/**
 * Release-smoke Playwright config.
 *
 * Serves the already-built web/dist/ via `vite preview` on port 4173.
 * Run with: `pnpm -C web e2e`
 * Ensure the bundle is up-to-date first: `pnpm -C web build`
 *
 * Browsers: Chromium only (cached at ~/.cache/ms-playwright/chromium-1223).
 * Do NOT run `playwright install` — browsers are already present offline.
 */
export default defineConfig({
  testDir: './e2e',

  // No retries for the release smoke — a flaky test here is a real signal.
  retries: 0,

  // One worker keeps the webServer lifecycle simple.
  workers: 1,

  reporter: 'list',

  use: {
    baseURL: 'http://localhost:4173',
    // Capture a trace on first retry (there are none, but keep for future).
    trace: 'on-first-retry',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: {
    // Serve the pre-built static bundle. The caller must run `pnpm build` first.
    command: 'pnpm preview',
    port: 4173,
    // Reuse a running server so incremental local runs are fast.
    reuseExistingServer: true,
    // Fail fast if the server doesn't come up within 10 s.
    timeout: 10_000,
  },
});
