use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Build a `JsValue` array of variable descriptors from plain Rust data.
/// We serialise via `serde-wasm-bindgen` to stay consistent with what the
/// production path expects on the other side of `simulate`.
fn make_vars(vars: &[(&str, &str, f64, f64)]) -> JsValue {
    use serde::Serialize;

    #[derive(Serialize)]
    struct VarObj<'a> {
        name: &'a str,
        shape: &'a str,
        lower: f64,
        upper: f64,
    }

    let objs: Vec<VarObj> = vars
        .iter()
        .map(|(name, shape, lower, upper)| VarObj {
            name,
            shape,
            lower: *lower,
            upper: *upper,
        })
        .collect();

    serde_wasm_bindgen::to_value(&objs).expect("serialising test vars must not fail")
}

/// Happy-path round-trip: `simulate` with a simple two-variable normal model
/// must return `Ok` and a structurally plausible `SimOutput`.
#[wasm_bindgen_test]
fn simulate_round_trip_returns_ok_with_plausible_output() {
    use ninety_ci_wasm::simulate;

    let vars = make_vars(&[
        ("A", "normal", 0.0, 10.0),
        ("B", "normal", 0.0, 10.0),
    ]);

    let result = simulate("A + B", vars, 1_000, 1.0);

    // The call must succeed.
    assert!(result.is_ok(), "simulate returned Err: {:?}", result.err());

    let js_out = result.unwrap();

    // Deserialise back to a typed struct to assert on the values.
    #[derive(serde::Deserialize, Debug)]
    struct SimOutput {
        ci_low: f64,
        ci_high: f64,
        buckets: Vec<f64>,
        counts: Vec<usize>,
        samples: usize,
    }

    let out: SimOutput =
        serde_wasm_bindgen::from_value(js_out).expect("deserialising SimOutput must not fail");

    assert!(
        out.samples > 0,
        "samples must be positive, got {}",
        out.samples
    );
    assert!(
        out.ci_low < out.ci_high,
        "ci_low ({}) must be less than ci_high ({})",
        out.ci_low,
        out.ci_high
    );
    assert!(
        !out.buckets.is_empty(),
        "buckets must not be empty"
    );
    assert_eq!(
        out.buckets.len(),
        out.counts.len(),
        "buckets and counts must have equal length"
    );
    // The sum of counts must equal the number of samples run.
    let total: usize = out.counts.iter().sum();
    assert_eq!(
        total, out.samples,
        "sum of counts ({}) must equal samples ({})",
        total, out.samples
    );
}
