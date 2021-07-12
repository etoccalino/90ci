use clap::{App, Arg};
use cli_90;

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
    // println!("Vars: {:#?}", vars);
    // println!("-----------------");
    let parsed_variables = parse_variables_descriptions(&vars).unwrap();
    // println!("Parsed: {:#?}", parsed_variables);
    let mut names: Vec<&str> = parsed_variables
        .iter()
        .map(|&(name, _, _, _)| name)
        .collect();
    assert!(validate_variables(equation, &mut names));

    // Build the frequency data and compute the 90% C.I.
    const iterations: usize = 5000;
    const bucket_size: f64 = 0.1;
    let (buckets, freqs) =
        cli_90::generate_freq_data(equation, &parsed_variables, &iterations, &bucket_size).unwrap();
    let (lower, upper) = cli_90::ninety_ci(&buckets, &freqs, &iterations);

    println!("-----------------------------------------");
    println!("90% C.I.: [{:.2?} ; {:.2?}]", lower, upper);
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
fn parse_variables_descriptions<'a>(
    descriptions: &Vec<&'a str>,
) -> Result<Vec<(&'a str, &'a str, f64, f64)>, String> {
    let mut res: Vec<(&str, &str, f64, f64)> = Vec::with_capacity(descriptions.len());
    let mut _lower: f64 = 0.;
    let mut _upper: f64 = 0.;
    let mut fields: Vec<&str>;
    for description in descriptions.iter() {
        fields = description.split(',').collect();
        if fields.len() != 4 {
            return Err(format!(
                "Incorrect number of fields in description: {}",
                description
            ));
        }
        _lower = fields[2].parse().unwrap();
        _upper = fields[3].parse().unwrap();
        res.push((fields[0], fields[1], _lower, _upper));
    }
    Ok(res)
}

fn validate_variables(equation: &str, variables: &mut Vec<&str>) -> bool {
    let extracted_names: Vec<&str> = cli_90::extract_variable_names(equation);
    if extracted_names.len() != variables.len() {
        return false;
    }
    let mut found: bool;
    for name in extracted_names.iter() {
        found = false;
        for var in variables.iter() {
            if name == var {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}
