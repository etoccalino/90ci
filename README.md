# 90ci

`90ci` estimates the **90% confidence interval** of a model under uncertainty. A model is an equation over one or more random variables (each described by a distribution and its 5th/95th-percentile bounds); 90ci samples those variables, evaluates the equation many times via Monte-Carlo simulation, and reports the range in which the middle 90% of outcomes land — along with the output distribution.

The same Rust engine drives two presentations:

- **Web UI** — a standalone, back-end-free React app that embeds the engine as a **WASM module**. A spreadsheet-style builder: edit the equation, fill a table of random variables, hit *Run*, and see the output histogram plus the 90% confidence-interval result.
- **TUI** — the original `clap`-based command-line binary, e.g. `90ci -e "200 * EXCHANGE_RATE + BASE_FEE" --var "EXCHANGE_RATE,uniform,1000,1200" --var "BASE_FEE,normal,0,50"`.

## Dependencies, per component

### Web app (front-end)
- **Toolchain**: Node 20, pnpm
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
