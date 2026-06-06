# Architecture — Rust → WASM module

## Split rationale

`core` is a plain Rust library (plus the TUI binary) with no knowledge of the web. The `wasm` crate is the only place that depends on `wasm-bindgen` and enables `getrandom`'s `js` feature — required for `rand` to get entropy on `wasm32-unknown-unknown` (via Web Crypto). Enabling that feature anywhere in the native build path would break non-web targets, so it is confined here. See `overview.md` for the layout.

## Required change in `core`: expose the histogram

Today `core::ci90(eq, vars, iterations, step)` returns only `(f64, f64)` and **discards the histogram** — `compute_histogram` and `ninety_ci` are private, and the `Equation<Evaluated>` that holds the `(buckets, counts)` is dropped. The web chart needs that distribution, so `core` must expose it.

Add a public, result-bearing entry point:

```rust
pub struct Simulation {
    pub ci_low: f64,
    pub ci_high: f64,
    pub buckets: Vec<f64>,   // bucket lower bounds
    pub counts: Vec<usize>,  // samples per bucket (same length as buckets)
    pub samples: usize,      // total iterations run
}

pub fn simulate(
    eq: &str,
    vars: &[VariableDescription],
    iterations: &usize,
    step: &f64,
) -> anyhow::Result<Simulation>;
```

`simulate` runs the same `Equation<UnderDefined> → add_variables → evaluate` flow already in `lib.rs`, then reads both `ninety_ci()` and the stored `hist` into a `Simulation`. Keep `ci90` as a thin wrapper so the TUI is untouched:

```rust
pub fn ci90(eq, vars, iterations, step) -> anyhow::Result<(f64, f64)> {
    simulate(eq, vars, iterations, step).map(|s| (s.ci_low, s.ci_high))
}
```

This is the **only** functional change `core` needs for the web app.

## The `wasm` wrapper crate

`crates/wasm/Cargo.toml`:
- `[lib] crate-type = ["cdylib"]`
- deps: `core` (path), `wasm-bindgen`, `serde` (derive), `serde-wasm-bindgen`, `console_error_panic_hook`, `getrandom = { version = "0.2", features = ["js"] }`

`crates/wasm/src/lib.rs` exposes one entry point and owns the JS ↔ Rust marshalling:

```rust
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct VarInput { name: String, shape: String, lower: f64, upper: f64 }

#[derive(Serialize)]
struct SimOutput {
    ci_low: f64, ci_high: f64,
    buckets: Vec<f64>, counts: Vec<usize>, samples: usize,
}

#[wasm_bindgen(start)]
pub fn init() { console_error_panic_hook::set_once(); }

#[wasm_bindgen]
pub fn simulate(equation: &str, vars: JsValue, iterations: usize, step: f64)
    -> Result<JsValue, JsValue>
{
    let inputs: Vec<VarInput> = serde_wasm_bindgen::from_value(vars)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    // Hold the owned Strings alive, then borrow into core's VariableDescription<'a>.
    let descs: Vec<core::VariableDescription> = inputs.iter().map(|v|
        core::VariableDescription { name: &v.name, shape: &v.shape, lower: v.lower, upper: v.upper }
    ).collect();
    let s = core::simulate(equation, &descs, &iterations, &step)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = SimOutput { ci_low: s.ci_low, ci_high: s.ci_high,
                          buckets: s.buckets, counts: s.counts, samples: s.samples };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}
```

## Lifetime note

`core::VariableDescription<'a>` borrows `&str` for `name`/`shape`. The wrapper deserializes into owned `VarInput` (with `String`s), keeps that `Vec` on the stack for the duration of the call, and builds the borrowed `VariableDescription`s referencing it. No change to `core`'s borrowing API is required.

## Threading & performance

5k–100k samples evaluate in milliseconds; v1 runs `simulate` synchronously on the main thread, which is acceptable. Moving it to a **Web Worker** (to keep the UI responsive at very high sample counts) is a documented future option, not a v1 requirement. `thread_rng` works in wasm through the `js` getrandom backend.

## Known engine limitations (recorded, out of scope)

- `extract_variable_names` (`src/lib.rs`) matches `[[:alpha:]]\w*`, so it would treat math functions like `sin`/`exp`/`cos` as variables if an equation uses them. The current model assumes plain variable identifiers only.
- The TUI parser (`parse_variables_descriptions` in `src/bin/main.rs`) uses `unwrap()` on field parsing; not relevant to the WASM path, which validates via serde, but noted.

Neither is fixed as part of this work.
