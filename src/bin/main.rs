use clap::{App, Arg};
use cli_90::*;
use regex::Regex;

/// Ensure the variable descriptions are valid, i.e.:
///     var_name,distro,lower,upper
/// where:
/// * all of the variables in the equation are present as `var_name`
/// * `distro` is either "uniform" or "normal"
/// * each of `lower` and `upper` parse to a f64
/// * `lower < upper`
fn parse_variables_descriptions(
    descriptions: Vec<&str>,
) -> Result<Vec<(&str, &str, f64, f64)>, String> {
    //     // The regex to match floats in different variations:
    //     // "[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)"
    //     // See: https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
    //     let RE: Regex = Regex::new(
    //         r"
    // (?P<name>[[:alnum:]]+),
    // (?P<distribution>[[:alnum:]]+),
    // (?P<lower>[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)),
    // (?P<upper>[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))",
    //     )
    //     .unwrap();

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

fn main() {
    let matches = App::new("90ci")
        .about("Returns the 90% confidence interval for a model")
        .arg(
            Arg::with_name("equation")
                .short("e")
                .long("equation")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("vars")
                .long("var")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1),
        )
        .get_matches();

    let vars: Vec<&str> = matches.values_of("vars").unwrap().collect();
    println!("Vars: {:#?}", vars);
    println!("-----------------");
    let res = parse_variables_descriptions(vars);
    println!("Parsed: {:#?}", res);
}
