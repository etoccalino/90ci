# Candidate rules

Learnings staged for promotion into `.claude/CLAUDE.md`. Added per stage; phrased in the `ALWAYS/NEVER … BECAUSE …` house style.

## From stage 7 (non-functional gates & release)

### Front-end
- ALWAYS extend Vitest's `configDefaults.exclude` (`exclude: [...configDefaults.exclude, 'e2e/**']`) rather than replacing it with a bare array BECAUSE a replacement silently drops the built-in `**/dist/**` and config-file globs, so after a `vite build` Vitest can start scanning `dist/` (and config files) for tests.
- ALWAYS make an E2E/release-smoke script build the artifact it validates inside its own script (`"e2e": "pnpm build && playwright test"`) rather than relying on a documented "build first" prerequisite BECAUSE a human/CI step that is only documented (not enforced) lets the smoke silently pass against a stale or absent `dist/`, defeating the gate.
- ALWAYS exclude each runner's specs from the other when co-locating Vitest and Playwright (Vitest `test.exclude` for `e2e/**`; Playwright `testDir`) BECAUSE both default-match `**/*.spec.*`, so a Playwright spec will otherwise be collected and run under jsdom where it breaks.
- ALWAYS measure a cross-browser perf budget inside the page's own `performance.now()` timeline (bracketed in `page.evaluate`/`MutationObserver`), never as Playwright wall-clock around `click()`+`waitFor` BECAUSE the driver IPC round-trip adds tens of ms of latency that is not part of the Run-to-render cost being asserted.
- ALWAYS drive a test interaction through the proven locator (`page.getByRole('button', { name: /run/i }).click()`) rather than a raw-DOM `querySelector(...)?.click()` BECAUSE optional-chaining on a missed selector is a silent no-op, turning a real failure into a misleading downstream timeout.

### Rust
- ALWAYS seed the RNG (or derive the tolerance from N) in a Monte-Carlo *integration* test rather than asserting a fixed absolute tolerance against an unseeded `thread_rng` BECAUSE sampling variance at N=5000 intermittently exceeds a hardcoded tolerance, producing a ~10% flaky failure (observed: `crates/core/tests/simple.rs::integration_single_variable_normal`).
