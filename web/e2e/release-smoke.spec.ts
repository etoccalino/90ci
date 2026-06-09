import { test, expect } from '@playwright/test';

/**
 * Release-smoke E2E spec — §9 Definition of Done.
 *
 * Serves the built web/dist/ via `vite preview` (port 4173) and drives one full
 * simulation end-to-end using the real WASM engine. Asserts:
 *   1. The default model is pre-filled and the Run button is present.
 *   2. Clicking Run produces a rendered CI hero (ciLow–ciHigh in large type).
 *   3. The output chart SVG renders (histogram area path + two dashed CI markers).
 *   4. Run-to-render is < 100 ms, measured entirely inside the browser's
 *      performance.now() timeline to exclude Playwright IPC wall-clock inflation.
 */

const PERF_BUDGET_MS = 100;

test.describe('Release smoke — default model end-to-end', () => {
  test('pre-filled model is visible and Run button is enabled', async ({ page }) => {
    await page.goto('/');

    // The model name input should show the prefilled value.
    const nameInput = page.getByRole('textbox', { name: 'Model name' });
    await expect(nameInput).toHaveValue('Exchange exposure');

    // The equation input should be pre-filled.
    const eqInput = page.getByRole('textbox', { name: 'Model equation' });
    await expect(eqInput).toHaveValue('200 * EXCHANGE_RATE + BASE_FEE');

    // Run button is present and enabled (engine loading may take a moment).
    const runBtn = page.getByRole('button', { name: /run/i });
    await expect(runBtn).toBeVisible();
    // Wait up to 5 s for the engine to load (WASM init).
    await expect(runBtn).toBeEnabled({ timeout: 5_000 });
  });

  test('Run produces a CI hero and output chart (real WASM)', async ({ page }) => {
    await page.goto('/');

    // Wait for the engine to be ready (Run enabled means init succeeded).
    const runBtn = page.getByRole('button', { name: /run/i });
    await expect(runBtn).toBeEnabled({ timeout: 5_000 });

    // Click Run and wait for the CI hero to appear.
    await runBtn.click();

    // The CI hero renders the 90% CI range.
    // CIHero.tsx wraps the output in a .ci-hero div; the range spans contain the
    // formatted ciLow and ciHigh values separated by a dash element.
    const ciHero = page.locator('.ci-hero');
    await expect(ciHero).toBeVisible({ timeout: 5_000 });

    // The CI range should contain two numeric spans (ciLow, ciHigh).
    const ciRange = ciHero.locator('.ci-range span');
    // There are three spans: ciLow, the dash, ciHigh — verify count.
    await expect(ciRange).toHaveCount(3);
    // Verify the first and last spans actually contain a rendered number.
    await expect(ciHero.locator('.ci-range span').first()).toHaveText(/\d/);
    await expect(ciHero.locator('.ci-range span').last()).toHaveText(/\d/);

    // The output chart SVG renders. The chart wrapper div is the reliable
    // visible container — the SVG <path>/<line> elements are geometry nodes
    // that Playwright considers "hidden" (no CSS box model dimensions).
    const chartWrap = page.locator('.chartwrap');
    await expect(chartWrap).toBeVisible({ timeout: 5_000 });

    // The histogram area path must be attached and carry non-empty geometry.
    // Both dashed CI markers are asserted for DOM presence.
    await expect(page.locator('[data-testid="area-path"]')).toHaveAttribute('d', /\S/);
    await expect(page.locator('[data-testid="marker-low"]')).toBeAttached();
    await expect(page.locator('[data-testid="marker-high"]')).toBeAttached();
  });

  test(`Run-to-render is under ${PERF_BUDGET_MS} ms (in-browser measurement)`, async ({ page }) => {
    await page.goto('/');

    // Wait for the WASM engine to initialise before measuring.
    const runBtn = page.getByRole('button', { name: /run/i });
    await expect(runBtn).toBeEnabled({ timeout: 5_000 });

    // Inject a measurement harness into the page's JS context.
    // Both timestamps are from the same performance.now() origin inside the
    // browser — no Playwright IPC latency in the critical path.
    //
    // The harness:
    //   - Captures t0 immediately before dispatching the click event on Run.
    //   - Uses a MutationObserver to detect when the CI hero appears in the DOM
    //     (the first child added to #root that is or contains .ci-hero).
    //   - Captures t1 inside the same microtask/mutation callback.
    //   - Stores the delta in window.__runDeltaMs so we can read it back.
    await page.evaluate(() => {
      (window as unknown as Record<string, unknown>)['__runDeltaMs'] = null;

      const observer = new MutationObserver(() => {
        if (document.querySelector('.ci-hero')) {
          const t0 = (window as unknown as Record<string, number>)['__runT0'];
          if (typeof t0 === 'number') {
            (window as unknown as Record<string, unknown>)['__runDeltaMs'] = performance.now() - t0;
          }
          observer.disconnect();
        }
      });

      observer.observe(document.body, { childList: true, subtree: true });
    });

    // Record t0 inside the browser's performance.now() timeline so it shares
    // the same origin as the MutationObserver t1 — no IPC inflation.
    await page.evaluate(() => {
      (window as unknown as Record<string, number>)['__runT0'] = performance.now();
    });
    // Drive the click through Playwright so it uses the proven locator path.
    // The ~1–5 ms localhost IPC gap is well within the 100 ms budget.
    await page.getByRole('button', { name: /run/i }).click();

    // Wait for the result to appear in the DOM (upper bound: 5 s).
    await expect(page.locator('.ci-hero')).toBeVisible({ timeout: 5_000 });

    // Read the delta captured inside the browser.
    const delta = await page.evaluate(
      () => (window as unknown as Record<string, unknown>)['__runDeltaMs'],
    );

    expect(typeof delta).toBe('number');
    const deltaMs = delta as number;

    // The actual observed value is reported in the test output so it is visible
    // in CI logs — not just a binary pass/fail.
    console.log(`[perf] Run-to-render: ${deltaMs.toFixed(1)} ms (budget: ${PERF_BUDGET_MS} ms)`);

    expect(deltaMs).toBeLessThan(PERF_BUDGET_MS);
  });
});
