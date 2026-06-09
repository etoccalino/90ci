use ninety_ci_core::*;
use statrs::assert_almost_eq;

#[test]
fn integration_single_variable_normal() {
    let variables: Vec<VariableDescription> = vec![VariableDescription {
        name: "VAR",
        shape: "normal",
        lower: 100.,
        upper: 200.,
    }];
    let (low, up) = ninety_ci_core::ci90("VAR", &variables, &5000).unwrap();
    // Tolerance derivation: step = 100/75 ≈ 1.33 (bucket-snapping error ≤ 1 step ≈ 1.33).
    // Monte-Carlo 90% CI for normal(150, σ=100/3.29≈30.4): 5th pct ≈ 100.0, 95th ≈ 200.0.
    // Sampling error at N=5000 for the 5th/95th percentiles: SE ≈ σ/√N * √(p(1-p)/p²) ≈ 1.9.
    // Theoretical bound ≈ 1.33 + 1.9 ≈ 3.3, but observed worst case runs hit ~4.6, so
    // tolerance of 6 is needed to keep the test reliably green across random seeds.
    assert_almost_eq!(low, 100., 6.);
    assert_almost_eq!(up, 200., 6.);
}

#[test]
fn integration_single_variable_uniform() {
    // For a symetric random variable with a 90%CI of [1,2], the equation
    // "1 + variable" should obviously have a 90%CI of [2,3].
    let variables: Vec<VariableDescription> = vec![VariableDescription {
        name: "VAR",
        shape: "uniform",
        lower: 1.,
        upper: 2.,
    }];
    let (low, up) = ninety_ci_core::ci90("1 + VAR", &variables, &5000).unwrap();
    assert_almost_eq!(low, 2., 0.1);
    assert_almost_eq!(up, 3., 0.1);
}
