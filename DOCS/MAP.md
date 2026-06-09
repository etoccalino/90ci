# Repository Map

A high-level guide to where things live in the `90ci` codebase, for developers and code reviewers. Config files and documentation are intentionally omitted; this map covers the code: UI, designs, Rust, the WASM wrapper, and tooling.

## Birds-eye layout

The repo is a hybrid Rust + TypeScript monorepo. A Rust workspace (`crates/`) holds the Monte-Carlo simulation engine and a WASM wrapper that compiles it for the browser. A Vite + React app (`web/`) is the front-end, which loads the WASM module to run simulations client-side. Design mockups live in `design/`, and the single build script lives in `scripts/`.

```
crates/        Rust workspace: core engine + WASM wrapper
  core/        ninety_ci_core — Monte-Carlo CI library (+ optional CLI binary)
  wasm/        cdylib that exposes core over the JS↔WASM boundary
web/           Vite + React + TypeScript front-end
  src/         UI source (components, hooks, model types, styles)
  src/wasm/    generated wasm-pack output (git-ignored, built on demand)
design/        UI mockups, wireframes, and design-direction prototypes
scripts/       build automation
```

The data flow is one direction: React UI → `web/src/wasm` glue (generated) → `crates/wasm` FFI → `crates/core` engine, and the `Simulation` result flows back out the same path.

## Front-end UI (`web/`)

React + TypeScript, built with Vite. The app is a two-panel calculator: a model editor on the left, a result panel on the right.

Entry & app shell:
- `web/index.html` — HTML entry point; mounts the root div and loads `main.tsx`.
- `web/src/main.tsx` — React bootstrap; renders `<App>` into the root.
- `web/src/App.tsx` — root component; owns model and result state; lays out `TopBar`, `ModelEditor`, and `ResultPanel`. On a run error it clears `result` so a stale chart never sits beside the error banner (E-10), and it disables Run while the engine has failed to load (`runDisabled`, E-09).
- `web/src/model.ts` — shared TypeScript types and constants: `Shape`, `Variable`, `Model`, `SimResult`, `SHAPES`. The sample count is fixed at 5,000 (PRD §1), so `Model` carries no `samples` field; `SimResult.samples` is echoed back by the engine. `Variable.p5`/`p95` are `number | null` — `null` represents a blank bound, kept distinct from a real `0` so an empty cell is never silently coerced (E-04).
- `web/src/validation.ts` — client-side model validation (Stage 4, PRD §3). `firstValidationError(model)` reports one violation at a time (mirroring the engine's one-per-run precedence); today it covers blank bounds (E-04), naming the offending variable and which bound: "`X`: enter a number for the 5th/95th bound." `hasBlankP5`/`hasBlankP95` are per-field predicates the editor uses to mark cells; `parseBound` maps a raw input string to `number | null` (blank → `null`, non-finite → `null`). Shared by the hook (to block the engine call) and `ModelEditor` (to parse/mark cells) so the rule lives in one place.
- `web/src/vite-env.d.ts` — Vite ambient type declarations.

Components (`web/src/components/`):
- `TopBar.tsx` — breadcrumb header ("Models / [name]").
- `ModelEditor.tsx` — left panel: equation input, variable management, Run button. (Sample count is fixed at 5,000; there is no sample-count control.) Parses a blank input to `null` instead of `0` via `parseBound` from `validation.ts` (E-04); blank bound cells get a `num-input--invalid` marker driven by the `validationError` prop (the client validation channel only, never engine/init errors). Accepts optional `runDisabled` (E-09) and `validationError` props.
- `ResultPanel.tsx` — right panel: renders results, or the empty/error state. The error banner carries `role="alert"` (PRD §3 / §7 a11y).
- `CIHero.tsx` — headline display of the 90% confidence interval bounds and width.
- `OutputChart.tsx` — SVG histogram of the output distribution with the 90% CI band shaded and two dashed `ciLow`/`ciHigh` markers carrying numeric labels (PRD §6). A zero-width / degenerate model (`buckets.length < 2`, E-03b) renders a visible single-spike marker with its value instead of a blank `<svg/>`. Very-narrow spreads enforce a min marker separation and collapse the two edge labels into one centred label so they stay legible. Bucket widths come from the engine (no front-end `step`).
- `Sparkline.tsx` — miniature distribution preview per variable (bell for normal, plateau for uniform/range).

Hooks & WASM glue (`web/src/`):
- `hooks/useNinetyCi.ts` — hook that initializes the WASM module, invokes `simulate`, and tracks loading/error state; maps UI percentiles to engine bounds and shapes the output into `SimResult`. Passes a hard-wired `SAMPLES = 5_000` to the engine (PRD §1). Runs `firstValidationError` before calling the engine and, on a violation, sets a dedicated `validationError` (a channel separate from engine/init `error`, so an inline blank-bound cell marker never fires under an unrelated failure) and does **not** call `wasmSimulate` (E-04). Tracks engine init via `engineReady` (`null` pending / `true` loaded / `false` failed); an `init()` rejection sets a persistent "Couldn't load the calculator engine — reload the page." error and `engineReady === false`, which `App` turns into `runDisabled` (E-09). Calls the engine's 3-arg `simulate(equation, vars, SAMPLES)` — the bucket `step` is no longer a front-end constant; `core` computes it from the output magnitude (PRD §6).
- `web/src/wasm/` — wasm-pack output (ES module + `.d.ts` bindings). Generated by the build script and git-ignored; do not hand-edit.

Styles:
- `web/src/styles/app.css` — global CSS: design tokens (color, type, spacing) plus component styles.

Tests (`web/`):
- `web/vite.config.ts` — Vitest config (`defineConfig` from `vitest/config`): `test.environment = 'jsdom'`, `globals: true`, `setupFiles: ['./src/setupTests.ts']`. `globals: true` is required for `@testing-library/react`'s auto-cleanup (it registers via a global `afterEach`), which keeps prior renders from accumulating in the jsdom DOM across tests in a file.
- `web/src/setupTests.ts` — imports `@testing-library/jest-dom/vitest` to register DOM matchers (and their Vitest `Assertion` types).
- `web/src/App.test.tsx` — smoke test: mocks the WASM glue module, renders `<App>`, asserts the prefilled model name and an enabled Run button; plus a test asserting `simulate` is invoked with `5000` as its samples argument (PRD §1). Run with `pnpm -C web test`.
- `web/src/components/ModelEditor.test.tsx` — asserts the sample-count control is gone (no "Samples" label, no sample-count combobox) and the percentile-bounds hint note still renders.
- `web/src/validation.test.ts` — unit tests for `validation.ts`: `firstValidationError` returns null for a valid model and a named message for blank/NaN bounds (reporting only the first violation); `hasBlankP5`/`hasBlankP95` treat `null`/`NaN` as blank but `0` as valid (E-04).
- `web/src/components/OutputChart.test.tsx` — Stage 5 §6 graph tests: a normal-spread `SimResult` renders the area, blue outline, shaded band, exactly two dashed markers with numeric edge labels, and 3 lo/mid/hi x-ticks; a degenerate (`buckets.length < 2`) model renders a visible spike with its value label (not a blank svg); marker x-positions match the `ciLow`/`ciHigh` mapping; negative-spanning results render negative axis ticks.
- `web/src/components/ErrorUX.test.tsx` — Stage 4 error-UX behavior: error banner has `role="alert"` (§3); a blank bound blocks Run, names the variable, and `simulate` is not called (E-04); the offending cell is marked `.num-input--invalid`; a rejected `init` shows the persistent reload banner and disables Run (E-09); an erroring Run after a successful one clears the chart (E-10).
- Dev deps: `jsdom`, `@testing-library/react`, `@testing-library/jest-dom`, `@testing-library/user-event`.

## Designs (`design/`)

Static mockups and throwaway prototypes that informed the UI. Not wired into the build.

`design/version-1/project/`:
- `90ci Calculator - Wireframes.html` — low-fidelity wireframes.
- `90ci Hi-Fi Directions.html` — high-fidelity design direction.
- `90ci Calculator - Spreadsheet.html` — spreadsheet-view mockup.
- `design-canvas.jsx` — React prototype of the canvas/layout pattern.
- `tweaks-panel.jsx` — React prototype of the variable-tweaking panel.

## Rust core engine (`crates/core`)

`ninety_ci_core` — the Monte-Carlo confidence-interval library. Pure Rust, no browser dependency. An optional `cli` feature builds a TUI binary.

- `crates/core/src/lib.rs` — the engine. Key items: `Distro` (Normal / Uniform / DiscreteUniform sampling); `Equation<Status>` — a phantom-typed state machine (`UnderDefined` → `FullyDefined` → `Evaluated`) that extracts variables, samples distributions, evaluates the equation per sample, builds a histogram, and computes the 90% CI; `VariableDescription` (input) and `Simulation` (output) types; `simulate()` entrypoint and the `ci90()` convenience wrapper (both 3-arg: `eq, vars, iterations` — the bucket `step` is no longer a parameter). `compute_step(lowest, highest)` derives the histogram bucket width from the observed output range, targeting ~50–100 buckets (divisor 75; degenerate `highest <= lowest` → a positive default of `1.0` yielding a single-bucket spike); `evaluate()` calls it on the finite series before `compute_histogram` (which still takes an explicit `bucket_size`). `evaluate()` filters out non-finite per-sample outputs (±inf / NaN from degenerate models such as division by zero) so they never reach `compute_histogram`; when no finite samples survive, `simulate` returns a human-readable `Err` instead of trapping (E-07). `compute_histogram` sorts with `total_cmp` and clamps bucket indices to absorb float rounding at the range extremes; `ninety_ci` normalizes by the surviving finite-sample count (not the configured resolution) and propagates an `Err` rather than unwrapping on a degenerate histogram. **Authored-model validation (Stage 3, §4 error matrix)** reports one violation per run, in this precedence: empty/blank equation → `"Enter an equation."` (E-05a); empty vars → `"Add at least one variable."` (E-05b); duplicate variable names, scanned in `add_variables` before the `HashMap` can silently merge them → `"Two variables are named \`X\` — names must be unique."` (E-12); then per variable in `add_variable`: a name not referenced in the equation → `"\`X\` is defined but not used — use it or remove the row."` (E-02), strictly-inverted bounds `lower > upper` (equal bounds stay valid, e.g. `range(0,0)`) → `"\`X\`: 5th (lo) must be below 95th (hi)."` (E-03), unsupported shape → the `Distro::new` guard `"Unsupported distribution…"` (E-11); finally, equation tokens with no variable row, named individually → `"\`X\` is used in the equation but not defined — add a variable row for it (or remove it)."` (E-01, with a comma-joined multi-token variant). Includes the unit-test module.
- `crates/core/src/bin/main.rs` — CLI binary, gated behind the `cli` feature. Clap parser for `--equation` and repeatable `--var name,distribution,lower,upper`; prints the computed 90% CI.

## WASM wrapper (`crates/wasm`)

A thin `cdylib` that exposes `ninety_ci_core` to JavaScript via `wasm-bindgen`. This is the only FFI boundary.

- `crates/wasm/src/lib.rs` — the FFI layer. `VarInput` (owned strings deserialized from JS) and `SimOutput` (serialized back to JS); `init()` installs `console_error_panic_hook` so Rust panics surface in the browser console; `simulate(equation, vars, iterations)` borrows `VariableDescription`s from the inputs, calls the core engine, and returns `{ ci_low, ci_high, buckets, counts, samples }`. (The bucket `step` is computed engine-side and is no longer a boundary parameter — PRD §6.)
- `crates/wasm/tests/boundary.rs` — WASM boundary harness (`wasm-bindgen-test`, `run_in_browser`): a round-trip test that calls `simulate` over the built glue and asserts a plausible `SimOutput`, plus a divide-by-zero test asserting that a degenerate model returns a readable error string (never a wasm `"unreachable"` trap; E-07). Stage 3 adds one test per §4 validation row (E-05a, E-05b, E-12, E-02, E-03, E-11, and both the single- and multi-token E-01 branches) asserting the thrown JS string names the offending token/row and carries the corrective phrasing; a header comment documents the one-violation-per-run precedence. Run with `wasm-pack test --headless --chrome crates/wasm`.
- `crates/wasm/webdriver.json` — ChromeDriver capabilities (`--no-sandbox`, `--headless`, `--disable-gpu`, `--disable-dev-shm-usage`) so the boundary test launches Chrome in headless/sandboxed environments.

## Tools & scripts

- `scripts/build-wasm.sh` — runs `wasm-pack build crates/wasm --target web` and emits the ES module + bindings into `web/src/wasm/` (git-ignored).
- `web/package.json` scripts — `build:wasm` (invokes the script above), `dev` (`build:wasm` then `vite`), `build` (`build:wasm` then `vite build`), `preview` (Vite preview), `test` (`vitest run`).
