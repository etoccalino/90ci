use ninety_ci_core::{simulate as core_simulate, VariableDescription};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// A variable as it arrives from JavaScript: owned strings, so the data outlives
/// the borrowed `VariableDescription<'a>` we build from it for the call.
#[derive(Deserialize)]
struct VarInput {
    name: String,
    shape: String,
    lower: f64,
    upper: f64,
}

/// The simulation result, serialized back to a plain JS object.
#[derive(Serialize)]
struct SimOutput {
    ci_low: f64,
    ci_high: f64,
    buckets: Vec<f64>,
    counts: Vec<usize>,
    samples: usize,
}

/// Installed once on module load: routes Rust panics to `console.error`.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Run the 90% confidence-interval simulation.
///
/// `vars` is a JS array of `{ name, shape, lower, upper }`. Returns a JS object
/// `{ ci_low, ci_high, buckets, counts, samples }`, or throws a string on error.
#[wasm_bindgen]
pub fn simulate(equation: &str, vars: JsValue, iterations: usize) -> Result<JsValue, JsValue> {
    let inputs: Vec<VarInput> =
        serde_wasm_bindgen::from_value(vars).map_err(|e| JsValue::from_str(&e.to_string()))?;

    // `inputs` owns the strings; the borrowed descriptions reference it for the call.
    let descs: Vec<VariableDescription> = inputs
        .iter()
        .map(|v| VariableDescription {
            name: &v.name,
            shape: &v.shape,
            lower: v.lower,
            upper: v.upper,
        })
        .collect();

    let s = core_simulate(equation, &descs, &iterations)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let out = SimOutput {
        ci_low: s.ci_low,
        ci_high: s.ci_high,
        buckets: s.buckets,
        counts: s.counts,
        samples: s.samples,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}
