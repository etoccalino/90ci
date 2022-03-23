use anyhow::{bail, Result};
use clap::{App, Arg};

fn main() {
    let matches = App::new("90ci")
        .about("Returns the 90% confidence interval for a model")
        .arg(
            Arg::with_name("equation")
                .short("e")
                .long("equation")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("vars")
                .long("var")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1),
        )
        .get_matches();

    // Get and validate input.
    let equation: &str = matches.value_of("equation").unwrap();
    let vars: Vec<&str> = matches.values_of("vars").unwrap().collect();

    const ITERATIONS: usize = 5000;
    const BUCKET_SIZE: f64 = 0.1;
    let (lower, upper) = cli_90::ci90(
        equation,
        &parse_variables_descriptions(&vars).unwrap(),
        &ITERATIONS,
        &BUCKET_SIZE,
    )
    .unwrap();

    println!("-----------------------------------------");
    println!("90% C.I.: [{:.1?} ; {:.1?}]", lower, upper);
    println!("-----------------------------------------");
}

///////////////////////////////////////////////////////////////////////////////

/// Ensure the variable descriptions are valid, i.e.:
///     var_name,distro,lower,upper
/// where:
/// * all of the variables in the equation are present as `var_name`
/// * `distro` is either "uniform" or "normal"
/// * each of `lower` and `upper` parse to a f64
/// * `lower < upper`
///
/// TODO: use a more robust parsing implementation (regex).
fn parse_variables_descriptions<'a>(
    descriptions: &[&'a str],
) -> Result<Vec<cli_90::VariableDescription<'a>>> {
    let mut res: Vec<cli_90::VariableDescription> = Vec::with_capacity(descriptions.len());
    let mut _lower: f64 = 0.;
    let mut _upper: f64 = 0.;
    let mut fields: Vec<&str>;
    for description in descriptions.iter() {
        fields = description.split(',').collect();
        if fields.len() != 4 {
            bail!("Incorrect number of fields in description: {}", description);
        }
        res.push(cli_90::VariableDescription {
            name: fields[0],
            shape: fields[1],
            lower: fields[2].parse().unwrap(),
            upper: fields[3].parse().unwrap(),
        });
    }
    Ok(res)
}
