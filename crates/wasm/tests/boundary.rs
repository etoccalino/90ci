use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// Serialise via serde-wasm-bindgen so simulate()'s real from_value deserializer is exercised with well-typed input.
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

/// Divide-by-zero model: `X / Y` where Y is always 0 must return `Err`,
/// never trap the wasm module. The error must be a non-empty string that
/// does not contain "unreachable" (the wasm-trap signature).
#[wasm_bindgen_test]
fn simulate_div_zero_returns_err_not_trap() {
    use ninety_ci_wasm::simulate;

    // Y is a "range" variable with lower=upper=0, so every sample is exactly 0.
    // X / Y produces inf for every sample; the engine must return Err.
    let vars = make_vars(&[
        ("X", "normal", 1.0, 10.0),
        ("Y", "range", 0.0, 0.0),
    ]);

    let result = simulate("X / Y", vars, 1_000, 0.1);

    let err_val = result.expect_err("expected Err for divide-by-zero model");

    // The error value must be a string (not an opaque JS exception).
    let err_str = err_val
        .as_string()
        .expect("error JsValue must be a string");

    assert!(!err_str.is_empty(), "error string must not be empty");
    assert!(
        !err_str.contains("unreachable"),
        "error must not be a wasm trap, got: {err_str}"
    );
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

    let js_out = result.expect("simulate returned Err");

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
    // total may be less than samples once non-finite outputs are filtered (Stage 1).
    let total: usize = out.counts.iter().sum();
    assert!(total <= out.samples, "bucket counts ({total}) should not exceed samples ({})", out.samples);
}
