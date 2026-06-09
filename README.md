# 90ci

`90ci` estimates the **90% confidence interval** of a model under uncertainty. A model is an equation over one or more random variables (each described by a distribution and its 5th/95th-percentile bounds); 90ci samples those variables, evaluates the equation many times via Monte-Carlo simulation, and reports the range in which the middle 90% of outcomes land — along with the output distribution.

The same Rust engine drives two presentations:

- **Web UI** — a standalone, back-end-free React app that embeds the engine as a **WASM module**. A spreadsheet-style builder: edit the equation, fill a table of random variables, hit *Run*, and see the output histogram plus the 90% confidence-interval result.
- **TUI** — the original `clap`-based command-line binary, e.g. `90ci -e "200 * EXCHANGE_RATE + BASE_FEE" --var "EXCHANGE_RATE,uniform,1000,1200" --var "BASE_FEE,normal,0,50"`.

## Setting up

### Rust (CLI tool and WASM module)

Install rustup, then add the WASM compilation target and `wasm-pack`:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

### Web UI

Install Node ≥ 22.13 and pnpm, then install the front-end dependencies:

```sh
# with Volta (recommended):
volta install node@22
volta install pnpm

# then, from the repo root:
pnpm -C web install
```

### Headless browser (for WASM boundary tests)

`wasm-pack test --headless --chrome` requires `chromedriver` in `PATH`. On Arch Linux one package provides both:

```sh
sudo pacman -S chromium
```

On Debian/Ubuntu: `sudo apt install chromium-driver`. On macOS: `brew install chromedriver`.

## Dependencies, per component

### Web app (front-end)
- **Toolchain**: Node ≥ 22.13 (required by pnpm 11), pnpm
- `react`, `react-dom`
- `vite`, `@vitejs/plugin-react`
- `typescript`
- `vite-plugin-wasm`, `vite-plugin-top-level-await`
- Consumes the WASM package produced by `wasm-pack` (see below)

### Rust TUI app (the `core` crate, built natively)
- `rand` 0.8
- `statrs` 0.15
- `regex` 1
- `lazy_static` 1
- `meval` 0.2
- `anyhow` 1.0
- `clap` 2.33 — TUI-only

### WASM module (the `wasm` wrapper crate)
- **Toolchain**: rustup, the `wasm32-unknown-unknown` target, `wasm-pack`
- `core` — the engine, which brings `rand` / `statrs` / `regex` / `lazy_static` / `meval` / `anyhow` (but **not** `clap`)
- `wasm-bindgen`
- `serde`, `serde-wasm-bindgen`
- `console_error_panic_hook`
- `getrandom = { version = "0.2", features = ["js"] }` — required for `rand` on `wasm32-unknown-unknown`; the `js` feature is scoped to this crate only, never to `core`

## Running the tests

Every Rust test in the workspace at once (from the repo root):

```sh
cargo test
```

Per component:

### CLI / engine (`crates/core`)
```sh
cargo test -p ninety_ci_core
```
This is where the test coverage currently lives: unit tests in `src/lib.rs` plus integration tests in `tests/simple.rs`.

### WASM crate (`crates/wasm`)
```sh
cargo test -p ninety-ci-wasm                       # native unit tests
wasm-pack test --headless --chrome crates/wasm     # tests across the wasm boundary
```
The boundary tests live in `crates/wasm/tests/boundary.rs` (a `simulate` round-trip plus one assertion per `§4` engine-error row); the native unit slot is empty since the crate is a thin marshalling layer over `core`. See [Setting up — Headless browser](#headless-browser-for-wasm-boundary-tests) for the `chromedriver` prerequisite.

### Web app (`web`)

Unit and component tests run under **Vitest** (jsdom + Testing Library):
```sh
pnpm -C web test
```

#### End-to-end release smoke (Playwright)

The release smoke builds the static bundle, serves it with `vite preview`, and drives one real simulation through the WASM engine in a headless browser. It needs Playwright's own Chromium, installed once:
```sh
pnpm -C web install                              # installs @playwright/test (pinned to match the cached browser)
pnpm -C web exec playwright install chromium     # downloads Playwright's Chromium
```
Then run it:
```sh
pnpm -C web e2e
```
The `e2e` script is `pnpm build && playwright test`, so it always rebuilds `web/dist/` first and validates the freshly built bundle (never a stale one). The spec lives in `web/e2e/release-smoke.spec.ts` and the config in `web/playwright.config.ts` (Chromium only, `vite preview` on port 4173). It is intentionally excluded from the Vitest run, so `pnpm -C web test` and `pnpm -C web e2e` stay separate.
