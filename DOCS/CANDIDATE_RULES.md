# Candidate rules

Learnings gathered during implementation, candidates for promotion to project rules.

## Testing

- The Vitest config must import `defineConfig` from `vitest/config`, not `vite` — only that overload type-checks the `test` block (otherwise `tsc` errors even though tests run).
- `setupTests.ts` must import `@testing-library/jest-dom/vitest` (the vitest entry), not the bare `@testing-library/jest-dom` — only the vitest entry augments Vitest's `Assertion` type so matchers like `toBeInTheDocument` type-check.
- Prefer explicit `import { describe, it, expect, vi } from 'vitest'` in test files over Vitest `globals: true`; the explicit form gives TypeScript resolution without a `tsconfig` `types` entry and avoids a hidden ambient-globals dependency.
- `getAllByText`/`getByText` match DOM **text content** and never match an `<input value=...>`; use `getByDisplayValue` to assert on input values. A test that mixes these passes for the wrong reason.
- When mocking the WASM glue in component tests, the `vi.mock(...)` path must match the import path used by the hook (`../wasm/ninety_ci_wasm`) — a stale path makes the mock silently not apply and the test exercise the wrong path.

## WASM boundary

- The boundary harness runs under `wasm-pack test --headless --chrome` with `wasm_bindgen_test_configure!(run_in_browser)` — a true browser is required because `getrandom` uses the JS backend.
- `crates/wasm/webdriver.json` carries the Chrome flags (`--no-sandbox`, `--headless`, `--disable-dev-shm-usage`) needed to launch headless Chrome in sandboxed/CI environments.
- wasm-pack 0.15 ignores `$CHROMEDRIVER` and uses its own cached driver; a Chrome/chromedriver major-version mismatch SIGKILLs the run. Keep the cached driver in sync with the installed Chrome.

## Cross-stage hygiene

- Assertions that encode a *currently-true* invariant can silently break a future stage. The boundary test's `sum(counts) == samples` was weakened to `<=` ahead of Stage 1's non-finite filtering, with a comment naming the stage — pin the intended invariant, not the incidental one.
