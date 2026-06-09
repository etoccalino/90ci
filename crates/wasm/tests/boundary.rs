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
    assert_eq!(
        err_str,
        "The equation produced an undefined result (e.g. division by zero) — check the formula/bounds.",
        "error string must match the exact engine diagnostic"
    );
}

// Stage 3 boundary tests — engine-side validation & honest error messages.
//
// Validation precedence (one violation per run, lowest wins):
//   E-05a: empty/blank equation  → "Enter an equation."
//   E-05b: zero variable rows    → "Add at least one variable."
//   E-12:  duplicate var names   → "Two variables are named `X` — names must be unique."
//   per-variable (in slice order):
//     E-02: var defined, not used → "`X` is defined but not used — use it or remove the row."
//     E-03: inverted bounds       → "`X`: 5th (…) must be below 95th (…)."
//     E-11: bad shape             → "Unsupported distribution. Use either 'normal', 'range' or 'uniform'."
//   E-01: token with no var row  → "`X` is used in the equation but not defined — add a variable row for it (or remove it)."

/// E-05a: empty equation string must produce a readable error naming the fix.
#[wasm_bindgen_test]
fn simulate_empty_equation_returns_err() {
    use ninety_ci_wasm::simulate;

    let vars = make_vars(&[("X", "uniform", 1.0, 2.0)]);
    let result = simulate("", vars, 100, 0.1);

    let err_str = result
        .expect_err("expected Err for empty equation")
        .as_string()
        .expect("error JsValue must be a string");

    assert_eq!(err_str, "Enter an equation.", "E-05a: wrong error for empty equation");
}

/// E-05a: whitespace-only equation is treated as empty.
#[wasm_bindgen_test]
fn simulate_whitespace_equation_returns_err() {
    use ninety_ci_wasm::simulate;

    let vars = make_vars(&[("X", "uniform", 1.0, 2.0)]);
    let result = simulate("   ", vars, 100, 0.1);

    let err_str = result
        .expect_err("expected Err for whitespace equation")
        .as_string()
        .expect("error JsValue must be a string");

    assert_eq!(err_str, "Enter an equation.", "E-05a: wrong error for whitespace equation");
}

/// E-05b: empty variable slice must produce a readable error naming the fix.
#[wasm_bindgen_test]
fn simulate_empty_vars_returns_err() {
    use ninety_ci_wasm::simulate;

    let vars = make_vars(&[]);
    let result = simulate("X + Y", vars, 100, 0.1);

    let err_str = result
        .expect_err("expected Err for empty vars")
        .as_string()
        .expect("error JsValue must be a string");

    assert_eq!(err_str, "Add at least one variable.", "E-05b: wrong error for empty vars");
}

/// E-12: duplicate variable names must be blocked with the offending name in the message.
#[wasm_bindgen_test]
fn simulate_duplicate_variable_names_returns_err() {
    use ninety_ci_wasm::simulate;

    let vars = make_vars(&[
        ("X", "uniform", 1.0, 2.0),
        ("X", "normal", 3.0, 8.0),
    ]);
    let result = simulate("X", vars, 100, 0.1);

    let err_str = result
        .expect_err("expected Err for duplicate variable name")
        .as_string()
        .expect("error JsValue must be a string");

    assert!(
        err_str.contains("`X`"),
        "E-12: error must name the duplicate, got: {}",
        err_str
    );
    assert!(
        err_str.contains("names must be unique"),
        "E-12: error must mention unique names, got: {}",
        err_str
    );
    assert_eq!(
        err_str,
        "Two variables are named `X` — names must be unique.",
        "E-12: wrong error message"
    );
}

/// E-02: variable defined but not used in the equation must produce a message naming the row
/// and the corrective action.
#[wasm_bindgen_test]
fn simulate_variable_not_used_returns_err() {
    use ninety_ci_wasm::simulate;

    let vars = make_vars(&[
        ("X", "uniform", 1.0, 2.0),
        ("Unused", "uniform", 1.0, 2.0),
    ]);
    let result = simulate("X", vars, 100, 0.1);

    let err_str = result
        .expect_err("expected Err for unused variable")
        .as_string()
        .expect("error JsValue must be a string");

    assert!(
        err_str.contains("`Unused`"),
        "E-02: error must name the unused variable, got: {}",
        err_str
    );
    assert!(
        err_str.contains("remove the row"),
        "E-02: error must contain corrective phrasing, got: {}",
        err_str
    );
    assert_eq!(
        err_str,
        "`Unused` is defined but not used — use it or remove the row.",
        "E-02: wrong error message"
    );
}

/// E-03: inverted bounds (lower > upper) must produce a message naming the variable and the values.
#[wasm_bindgen_test]
fn simulate_inverted_bounds_returns_named_err() {
    use ninety_ci_wasm::simulate;

    let vars = make_vars(&[("Revenue", "normal", 10.0, 5.0)]);
    let result = simulate("Revenue", vars, 100, 0.1);

    let err_str = result
        .expect_err("expected Err for inverted bounds")
        .as_string()
        .expect("error JsValue must be a string");

    assert!(
        err_str.contains("`Revenue`"),
        "E-03: error must name the variable, got: {}",
        err_str
    );
    assert!(
        err_str.contains("must be below"),
        "E-03: error must describe the constraint, got: {}",
        err_str
    );
    assert_eq!(
        err_str,
        "`Revenue`: 5th (10) must be below 95th (5).",
        "E-03: wrong error message"
    );
}

/// E-01: equation token with no variable row must produce a message naming the undeclared token
/// and the corrective action.
#[wasm_bindgen_test]
fn simulate_missing_token_returns_named_err() {
    use ninety_ci_wasm::simulate;

    let vars = make_vars(&[("A", "uniform", 1.0, 2.0)]);
    // Equation references B which has no variable row.
    let result = simulate("A + B", vars, 100, 0.1);

    let err_str = result
        .expect_err("expected Err for missing token")
        .as_string()
        .expect("error JsValue must be a string");

    assert!(
        err_str.contains("`B`"),
        "E-01: error must name the missing token, got: {}",
        err_str
    );
    assert!(
        err_str.contains("add a variable row"),
        "E-01: error must contain corrective phrasing, got: {}",
        err_str
    );
    assert_eq!(
        err_str,
        "`B` is used in the equation but not defined — add a variable row for it (or remove it).",
        "E-01: wrong error message"
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
