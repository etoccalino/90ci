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
    // With the computed step (~100/75 ≈ 1.33 for this range), bucket-snapping error is
    // ~1 step plus Monte Carlo variance; tolerance of 6 covers both comfortably.
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
