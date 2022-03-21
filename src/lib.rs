extern crate meval;

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Error, Result};
use lazy_static::lazy_static;
use rand::distributions::Distribution;
use rand::{thread_rng, Rng};
use regex::Regex;
use statrs::distribution::{DiscreteUniform, Normal, Uniform};

///////////////////////////////////////////////////////////////////////////////
/// A Distro can be sampled, and therefor used by the `rand` package.
enum Distro {
    N(Normal),
    U(Uniform),
    DU(DiscreteUniform),
}

impl Distro {
    pub fn new(name: &str, lower_bound: f64, upper_bound: f64) -> Result<Self> {
        match name {
            "range" => {
                let l: i64 = (lower_bound).floor() as i64;
                let u = (upper_bound).floor() as i64;
                Ok(Distro::DU(DiscreteUniform::new(l, u)?))
            }
            "uniform" => Ok(Distro::U(Uniform::new(lower_bound, upper_bound)?)),
            "normal" => Ok(Distro::N(Normal::new(
                (upper_bound + lower_bound) / 2.,
                (upper_bound - lower_bound) / 3.29,
            )?)),
            _ => bail!("Unsupported distribution. Use either 'normal', 'range' or 'uniform'."),
        }
    }
}

impl Distribution<f64> for Distro {
    // https://docs.rs/rand/0.8.5/rand/distributions/trait.Distribution.html
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            Distro::N(distro) => distro.sample(rng),
            Distro::U(distro) => distro.sample(rng),
            Distro::DU(distro) => distro.sample(rng),
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Use VariableDescription to define a variable
pub struct VariableDescription<'a> {
    pub name: &'a str,
    pub shape: &'a str,
    pub lower: f64,
    pub upper: f64,
}

impl<'a> VariableDescription<'a> {
    pub fn new(name: &'a str, distribution: &'a str, lower_bound: f64, upper_bound: f64) -> Self {
        VariableDescription {
            name,
            shape: distribution,
            lower: lower_bound,
            upper: upper_bound,
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

enum Equation {
    UnderDefined,
    FullyDefined,
    OverDefined,
    Invalid,
}

struct UnderDefined<'a> {
    eq: &'a str,                      // String representation of the equation
    step: usize,                      // size of buckets
    vars: HashMap<&'a str, Vec<f64>>, // Variables ready to be used
    var_names: HashSet<&'a str>,      // List variables needed to define the equation
}

fn extract_variable_names(equation: &str) -> Vec<&str> {
    // Compile the regex once
    lazy_static! {
        // static ref RE: Regex = Regex::new(r"\W").unwrap();
        static ref RE: Regex = Regex::new(r"[[:alpha:]]\w*").unwrap();
    }
    // RE.split(equation).filter(|v| !v.is_empty()).collect()
    RE.find_iter(equation).map(|v| v.as_str()).collect()
}

impl<'a> UnderDefined<'a> {
    pub fn new(eq: &'a str, step: usize) -> Self {
        let names = extract_variable_names(eq);
        UnderDefined {
            eq,
            step,
            vars: HashMap::with_capacity(names.len()),
            var_names: names.into_iter().collect(),
        }
    }

    pub fn add_var(self, var: &'a VariableDescription) -> Result<Equation> {
        if !self.var_names.contains(var.name) {
            bail!("Variable {} not mentioned in the equation", var.name);
        }

        let sample = sample_variable(var.shape, &var.lower, &var.upper, self.step)?;
        self.vars.insert(var.name, sample);

        if self.vars.len() < self.var_names.len() {
            Ok(self)
        } else {
            Ok(FullyDefined {
                vars: self.vars,
                ..self
            })
        }
    }
}

struct FullyDefined<'a> {
    eq: &'a str,                    // String representation of the equation
    step: usize,                    // size of buckets
    vars: HashMap<&'a str, Distro>, // Variables ready to be used
}

struct OverDefined<'a> {
    eq: &'a str,                    // String representation of the equation
    vars: HashMap<&'a str, Distro>, // Variables ready to be used
    var_names: HashSet<&'a str>,    // List variables needed to define the equation
}

struct Invalid<'a> {
    eq: &'a str, // String representation of the equation
    error: Error,
}

///////////////////////////////////////////////////////////////////////////////

/// Return a series of samples of a random variable described by either "uniform" or "normal".
/// Fail if a variable has type other than "uniform", "range" or "normal", or a lower bound is
/// greater than an upper bound.
fn sample_variable(distribution: &str, lower: &f64, upper: &f64, n: usize) -> Result<Vec<f64>> {
    if *lower >= *upper {
        bail!("Lower bound >= upper bound");
    }

    let dist = match distribution {
        "range" => {
            let l: i64 = (*lower).floor() as i64;
            let u = (*upper).floor() as i64;
            Distro::DU(DiscreteUniform::new(l, u).unwrap())
        }
        "uniform" => Distro::U(Uniform::new(*lower, *upper).unwrap()),
        "normal" => Distro::N(Normal::new((upper + lower) / 2., (upper - lower) / 3.29).unwrap()),
        _ => bail!("Unsupported distribution. Use either 'normal', 'range' or 'uniform'."),
    };

    let rng = thread_rng();
    Ok(rng.sample_iter(&dist).take(n).collect())
}

/// Given a data series and a bucket size, return a pair of vectors:
/// - first vector carries the buckets in the series, and
/// - second vector carries the number of data points in the corresponding bucket.
/// Fails if the vector is empty or has a single element.
fn bucketize_series(mut series: Vec<f64>, bucket_size: &f64) -> Option<(Vec<f64>, Vec<usize>)> {
    if series.len() < 2 {
        return None;
    }

    let mut buckets: Vec<f64> = Vec::new();

    // Part one:
    //   Get the lowest and highest values in the vector.
    //   Generate the array of buckets using lowest, highest and bucket_size.
    //   Initialize the results vector with zeroes.
    series.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lowest = series.first()?;
    let highest = series.last()?;
    let mut bucket = lowest - lowest.rem_euclid(*bucket_size);
    while bucket <= *highest {
        buckets.push(bucket);
        bucket += bucket_size;
    }
    let mut freqs: Vec<usize> = vec![0; buckets.len()];

    // Part two:
    //   For each value: compute which bucket it corresponds to using `bucket_size * (val % bucket_size)`
    //                   increment the bucket's entry in the results vector.
    let buckets_offset = lowest.div_euclid(*bucket_size); // can be negative
    let mut bucket_index: usize;
    for val in series {
        bucket_index = (val.div_euclid(*bucket_size) - buckets_offset) as usize;
        freqs[bucket_index] += 1;
    }

    Some((buckets, freqs))
}

/// Generates a frequency data series: first vector is the list of "buckets"
/// and second vector is the number of values that fall on each bucket,
/// i.e. values that are `bucket <= x < next_bucket`.
/// Fails if:
///     there are no variables,
///     a variable has type other than "uniform" or "normal",
///     a lower bound is greater than an upper bound.
pub fn generate_freq_data(
    equation: &str,
    variables_description: &[VariableDescription],
    n: &usize,
    bucket_size: &f64,
) -> Result<(Vec<f64>, Vec<usize>)> {
    if variables_description.is_empty() {
        bail!("No variables to evaluate");
    }

    let mut values: Vec<(&str, Vec<f64>)> = Vec::with_capacity(variables_description.len()); // Hold the random variable samples.
    let mut ctx = meval::Context::new(); // The context to pass samples to evaluate the equation.
    let mut series: Vec<f64> = Vec::with_capacity(*n); // Hold results of evaluating the equation.

    // Sample all the variables in the equation.
    for description in variables_description {
        values.push((
            description.name,
            sample_variable(
                description.shape,
                &description.lower,
                &description.upper,
                *n,
            )?,
        ));
    }
    // Evaluate the equation using the samples.
    for i in 0..*n {
        // Update the evaluation context.
        for (var_name, var_values) in values.iter() {
            ctx.var(String::from(*var_name), var_values[i]);
        }
        match meval::eval_str_with_context(equation, &ctx) {
            Ok(result) => series.push(result),
            Err(e) => {
                bail!("Error evaluating the equation: {:?}", e);
            }
        }
    }

    bucketize_series(series, bucket_size).ok_or_else(|| anyhow!("Error bucket'ing the data series"))
}

/// Given the frequency data (buckets, frequencies), return a pair of buckets
/// which represent the range for the 90% confidence interval of the sample.
pub fn ninety_ci(
    buckets: &[f64],
    frequencies: &[usize],
    n: &usize, // sum(frequencies)
) -> (f64, f64) {
    // Accumulating the frequencies from first to last, the 90CI range is found as:
    // - Lower bound is LAST bucket BEFORE accumulating >5%
    // - Upper bound is FIRST bucket AFTER accumulating >95%
    let mut acc = 0.;
    let mut lower: &f64 = buckets.first().unwrap();
    let mut upper: &f64 = buckets.last().unwrap();
    for (bucket, freq) in buckets.iter().zip(frequencies.iter()) {
        acc += *freq as f32 / *n as f32;

        if acc <= 0.05 {
            // Drag the lower bound up
            lower = bucket;
        }
        if acc >= 0.95 {
            // Drop the upper bound and stop accumulating
            upper = bucket;
            break;
        }
    }
    (*lower, *upper)
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::assert_almost_eq;

    #[test]
    fn extract_variable_names_simple_names() {
        let variables = extract_variable_names("A1 + A2 - B");
        assert_eq!(variables.len(), 3);
        assert_eq!(variables[0], "A1");
        assert_eq!(variables[1], "A2");
        assert_eq!(variables[2], "B");
    }

    #[test]
    fn extract_variable_names_complex_names() {
        let variables = extract_variable_names("A_1 + some_A*B1");
        assert_eq!(variables.len(), 3);
        assert_eq!(variables[0], "A_1");
        assert_eq!(variables[1], "some_A");
        assert_eq!(variables[2], "B1");
    }

    #[test]
    fn extract_variable_names_with_numbers() {
        let variables = extract_variable_names("4.5 * A_1 + 2 * some");
        assert_eq!(variables.len(), 2);
        assert_eq!(variables[0], "A_1");
        assert_eq!(variables[1], "some");
    }

    //////////////////////////////////////////////////////////////////////

    #[test]
    fn sample_variable_incorrect_type() {
        assert!(sample_variable("incorrect", &1., &2., 1).is_err());
    }

    #[test]
    fn sample_variable_incorrect_bounds() {
        assert!(sample_variable("incorrect", &2., &1., 1).is_err());
    }

    #[test]
    fn sample_variable_size_correct() {
        let sample = sample_variable("uniform", &1., &2., 100).unwrap();
        assert_eq!(sample.len(), 100);
        let sample = sample_variable("normal", &1., &2., 100).unwrap();
        assert_eq!(sample.len(), 100);
        let sample = sample_variable("range", &1., &2., 100).unwrap();
        assert_eq!(sample.len(), 100);
    }

    #[test]
    fn generate_freq_data_empty_variable_type() {
        let vars: Vec<VariableDescription> = vec![];
        assert!(generate_freq_data("1", &vars, &100, &1.).is_err());
    }

    #[test]
    fn generate_freq_data_incorrect_variable_type() {
        let vars = vec![VariableDescription::new("V1", "incorrect", 1., 1.)];
        assert!(generate_freq_data("1", &vars, &100, &1.).is_err());
    }

    #[test]
    fn generate_freq_data_incorrect_bounds() {
        let vars = vec![VariableDescription::new("V1", "uniform", 2., 1.)];
        assert!(generate_freq_data("1", &vars, &100, &1.).is_err());
    }

    #[test]
    fn generate_freq_data_check_size() {
        let vars = vec![
            VariableDescription::new("V1", "uniform", 1., 2.),
            VariableDescription::new("V2", "normal", 1., 2.),
        ];
        let (buckets, freqs) = generate_freq_data("1", &vars, &100, &1.).unwrap();
        assert_eq!(buckets.len(), freqs.len());
    }

    #[test]
    fn bucketize_series_single() {
        let data = vec![1.];
        assert!(bucketize_series(data, &0.1).is_none());
    }

    #[test]
    fn bucketize_series_bucket_size_smaller_than_1() {
        let data = vec![1., 3.];
        let (buckets, freqs) = bucketize_series(data, &0.5).unwrap();
        assert_eq!(buckets, vec![1., 1.5, 2., 2.5, 3.]);
        assert_eq!(freqs, vec![1, 0, 0, 0, 1]);
    }

    #[test]
    fn bucketize_series_negative_values_and_small_bucket() {
        let data = vec![-1., 2.];
        let (buckets, freqs) = bucketize_series(data, &0.5).unwrap();
        assert_eq!(buckets, vec![-1., -0.5, 0., 0.5, 1., 1.5, 2.]);
        assert_eq!(freqs, vec![1, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn bucketize_series_larger_test() {
        let data = vec![0.33, 1.1, 1.6, 6.0, 5.5, 6.0, 4.3, 7.1, -1.1];
        let (buckets, freqs) = bucketize_series(data, &2.0).unwrap();
        assert_eq!(buckets, vec![-2., 0., 2., 4., 6.]);
        assert_eq!(freqs, vec![1, 3, 0, 2, 3]);
    }

    #[test]
    fn bucketize_sub_unit() {
        let data = vec![0.83, 0.96, 1.15];
        let (buckets, freqs) = bucketize_series(data, &0.1).unwrap();
        assert_eq!(buckets, vec![0.8, 0.9, 1.0, 1.1]);
        assert_eq!(freqs, vec![1, 1, 0, 1]);
    }

    //////////////////////////////////////////////////////////////////////

    #[test]
    fn ninety_ci_bucket_size_smaller_than_1() {
        let buckets = vec![1., 1.5, 2., 2.5, 3.];
        let freqs = vec![1, 0, 0, 0, 1];
        let (low, up) = ninety_ci(&buckets, &freqs, &2);
        assert_eq!(low, 1.);
        assert_eq!(up, 3.);
    }

    #[test]
    fn ninety_ci_negative_values_and_small_bucket() {
        let buckets = vec![-1., -0.5, 0., 0.5, 1., 1.5, 2.];
        let freqs = vec![1, 0, 0, 0, 0, 0, 1];
        let (low, up) = ninety_ci(&buckets, &freqs, &2);
        assert_eq!(low, -1.);
        assert_eq!(up, 2.);
    }

    #[test]
    fn ninety_ci_larger_test() {
        let buckets = vec![-4., -2., 0., 2., 4., 6.];
        let freqs = vec![1, 1, 4, 40, 3, 1];
        let (low, up) = ninety_ci(&buckets, &freqs, &50);
        assert_eq!(low, -2.);
        assert_eq!(up, 4.);
    }

    ///////////////////////////////////////////////////////////////////////////////

    #[test]
    fn integration_single_variable_normal() {
        let equation: &str = "VAR";
        let variables: Vec<VariableDescription> =
            vec![VariableDescription::new("VAR", "normal", 100., 200.)];
        const ITERATIONS: usize = 5000;
        const BUCKET_SIZE: f64 = 0.1;

        let (buckets, freqs) =
            generate_freq_data(equation, &variables, &ITERATIONS, &BUCKET_SIZE).unwrap();
        let (low, up) = ninety_ci(&buckets, &freqs, &ITERATIONS);

        println!("DEBUG - test 90% CI: [{}, {}]", low, up);
        assert_almost_eq!(low, 100., 1.);
        assert_almost_eq!(up, 200., 1.);
    }

    #[test]
    fn integration_single_variable_uniform() {
        // For a symetric random variable with a 90%CI of [1,2], the equation
        // "1 + variable" should obviously have a 90%CI of [2,3].
        let equation: &str = "1 + VAR";
        let variables: Vec<VariableDescription> =
            vec![VariableDescription::new("VAR", "uniform", 1., 2.)];
        const ITERATIONS: usize = 5000;
        const BUCKET_SIZE: f64 = 0.1;

        let (buckets, freqs) =
            generate_freq_data(equation, &variables, &ITERATIONS, &BUCKET_SIZE).unwrap();
        let (low, up) = ninety_ci(&buckets, &freqs, &ITERATIONS);

        println!("DEBUG - test 90% CI: [{}, {}]", low, up);
        assert_almost_eq!(low, 2., 0.1);
        assert_almost_eq!(up, 3., 0.1);
    }
}
