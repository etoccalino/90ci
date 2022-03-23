use cli_90::*;
use statrs::assert_almost_eq;

#[test]
fn integration_single_variable_normal() {
    let variables: Vec<VariableDescription> = vec![VariableDescription {
        name: "VAR",
        shape: "normal",
        lower: 100.,
        upper: 200.,
    }];
    let (low, up) = cli_90::ci90("VAR", &variables, &5000, &0.1).unwrap();
    assert_almost_eq!(low, 100., 3.);
    assert_almost_eq!(up, 200., 3.);
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
    let (low, up) = cli_90::ci90("1 + VAR", &variables, &5000, &0.1).unwrap();
    assert_almost_eq!(low, 2., 0.1);
    assert_almost_eq!(up, 3., 0.1);
}
