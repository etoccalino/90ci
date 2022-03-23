extern crate meval;

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
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
    fn new(name: &str, lower_bound: f64, upper_bound: f64) -> Result<Self> {
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

/// A histogram represented as a list of "bucket (upper) bounds" and the number of samples that
/// fell in it.
type Histogram = (Vec<f64>, Vec<usize>);

struct Equation<'a, Status> {
    _status: Status,                  // Current status of the equation
    eq: &'a str,                      // String representation of the equation
    step: f64,                        // size of buckets
    resolution: usize,                // number of samples to take for each distribution
    vars: HashMap<&'a str, Vec<f64>>, // statrs distributions for the Variables, ready to be used
    var_names: HashSet<&'a str>,      // List variables needed to define the equation
    hist: Option<Histogram>,
}

struct UnderDefined;
struct FullyDefined;
struct Evaluated;

enum ValidEquation<'a> {
    Partial(Equation<'a, UnderDefined>),
    Full(Equation<'a, FullyDefined>),
}
// struct Invalid { errors: &[Error] };

/// To easily transition an equation
impl<'a, Status> Equation<'a, Status> {
    fn new(
        eq: &'a str,
        samples_num: Option<usize>, // Optional, defaults to 5.000 samples
        step_size: Option<f64>,     // Optional, defaults to 0.1 bucket size
    ) -> Equation<'a, UnderDefined> {
        let var_names = Equation::<UnderDefined>::extract_variable_names(eq);
        let resolution = samples_num.unwrap_or(5_000);
        let step = step_size.unwrap_or(0.1);
        Equation {
            _status: UnderDefined,
            eq,
            step,
            resolution,
            vars: HashMap::with_capacity(var_names.len()),
            var_names: var_names.into_iter().collect(),
            hist: None,
        }
    }

    fn with_status<NewStatus>(self, new: NewStatus) -> Equation<'a, NewStatus> {
        Equation {
            _status: new,
            eq: self.eq,
            step: self.step,
            resolution: self.resolution,
            vars: self.vars,
            var_names: self.var_names,
            hist: self.hist,
        }
        // I wish I could replace that with: Equation { status: new, ..self }
    }
}

/// A "under defined" equation requires its variable descriptions.
impl<'a> Equation<'a, UnderDefined> {
    /// Find the variable names involved in an equation.
    fn extract_variable_names(equation: &str) -> Vec<&str> {
        // Compile the regex once
        lazy_static! {
            // static ref RE: Regex = Regex::new(r"\W").unwrap();
            static ref RE: Regex = Regex::new(r"[[:alpha:]]\w*").unwrap();
        }
        // RE.split(equation).filter(|v| !v.is_empty()).collect()
        RE.find_iter(equation).map(|v| v.as_str()).collect()
    }

    /// Return a series of samples of a random variable described by either "uniform" or "normal".
    /// Fail if a variable has type other than "uniform", "range" or "normal", or a lower bound is
    /// greater than an upper bound.
    fn sample_variable(distribution: &str, lower: &f64, upper: &f64, n: usize) -> Result<Vec<f64>> {
        let dist: Distro = Distro::new(distribution, *lower, *upper)?;
        let rng = thread_rng();
        Ok(rng.sample_iter(&dist).take(n).collect())
    }

    /// Add a variable description to the equation.
    /// Fails if the variable passed is not present in the equation.
    fn add_variable(&mut self, var: &'a VariableDescription) -> Result<()> {
        if !self.var_names.contains(var.name) {
            bail!("Variable {} not mentioned in the equation", var.name);
        }
        self.vars.insert(
            var.name,
            Equation::sample_variable(var.shape, &var.lower, &var.upper, self.resolution)?,
        );
        Ok(())
    }

    /// Add variables to the equation. This operation may fully define the equation,
    /// and so the return type will be either a `Equation<UnderDefined>`, which can continue to
    /// receive variable descriptions, or a `Equation<Fullydefined>`, which does not support this
    /// operation.
    fn add_variables(mut self, vars: &'a [VariableDescription]) -> Result<ValidEquation<'a>> {
        for var in vars.iter() {
            self.add_variable(var)?;
        }

        if self.vars.len() < self.var_names.len() {
            Ok(ValidEquation::Partial(self))
        } else {
            Ok(ValidEquation::Full(Equation::with_status(
                self,
                FullyDefined,
            )))
        }
    }
}

/// A fully defined equation is ready to be evaluated.
impl<'a> Equation<'a, FullyDefined> {
    /// Given a data series and a bucket size, return a pair of vectors:
    /// - first vector carries the buckets in the series, and
    /// - second vector carries the number of data points in the corresponding bucket.
    /// Fails if the vector is empty or has a single element.
    fn compute_histogram(series: &mut Vec<f64>, bucket_size: &f64) -> Option<Histogram> {
        if series.len() < 2 {
            return None;
        }
        let mut buckets: Vec<f64> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();

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
            counts.push(0);
            bucket += bucket_size;
        }
        counts.resize(buckets.len(), 0);

        // Part two:
        //   For each value: compute which bucket it corresponds to using `bucket_size * (val % bucket_size)`
        //                   increment the bucket's entry in the results vector.
        let buckets_offset = lowest.div_euclid(*bucket_size); // can be negative
        let mut bucket_index: usize;
        for val in series {
            bucket_index = (val.div_euclid(*bucket_size) - buckets_offset) as usize;
            counts[bucket_index] += 1;
        }
        Some((buckets, counts))
    }

    fn evaluate(self) -> Result<Equation<'a, Evaluated>> {
        let mut ctx = meval::Context::new(); // The context to pass samples to evaluate the equation.
        let mut series: Vec<f64> = Vec::with_capacity(self.resolution); // Hold results of evaluating the equation.

        // Evaluate the equation using the samples.
        for i in 0..self.resolution {
            // Update the evaluation context.
            for (var_name, var_values) in self.vars.iter() {
                ctx.var(String::from(*var_name), var_values[i]);
            }
            match meval::eval_str_with_context(self.eq, &ctx) {
                Ok(result) => series.push(result),
                Err(e) => {
                    bail!("Error evaluating the equation: {:?}", e);
                }
            }
        }
        let histogram = Equation::compute_histogram(&mut series, &self.step)
            .ok_or_else(|| anyhow!("Error bucket'ing the data series"))?;
        let mut evaluated_self = Equation::with_status(self, Evaluated);
        evaluated_self.hist = Some(histogram);
        Ok(evaluated_self)
    }
}

/// The Equation<Evaluated> exposes methods to interpret and visualize the results.
impl<'a> Equation<'a, Evaluated> {
    /// Given the frequency data (buckets, frequencies), return a pair of buckets
    /// which represent the range for the 90% confidence interval of the sample.
    fn ninety_ci(&self) -> Result<(f64, f64)> {
        // Accumulating the frequencies from first to last, the 90CI range is found as:
        // - Lower bound is LAST bucket BEFORE accumulating >5%
        // - Upper bound is FIRST bucket AFTER accumulating >95%

        // Get a ref to each of the vectors (to avoid copies)
        let hist = self
            .hist
            .as_ref()
            .ok_or(anyhow!("Equation was not evaluated!?"))?;
        let buckets = &hist.0;
        let counts = &hist.1;

        let mut lower: &f64 = buckets.first().unwrap();
        let mut upper: &f64 = buckets.last().unwrap();
        let mut acc = 0.;
        for ix in 0..buckets.len() {
            acc += counts[ix] as f32 / self.resolution as f32;

            if acc <= 0.05 {
                // Drag the lower bound up
                lower = &buckets[ix];
            }
            if acc >= 0.95 {
                // Drop the upper bound and stop accumulating
                upper = &buckets[ix];
                break;
            }
        }
        Ok((*lower, *upper))
    }
}

///////////////////////////////////////////////////////////////////////////////

/// Use VariableDescription to define a variable
#[derive(Debug)]
pub struct VariableDescription<'a> {
    pub name: &'a str,
    pub shape: &'a str,
    pub lower: f64,
    pub upper: f64,
}

/// Entrypoint to the library.
/// Use the `VariableDescription` and the `ci90()` function.
pub fn ci90(
    eq: &str,
    vars: &[VariableDescription],
    iterations: &usize,
    step: &f64,
) -> Result<(f64, f64)> {
    if vars.is_empty() {
        bail!("No variables for the equation");
    }
    let initial_model: Equation<UnderDefined> =
        Equation::<UnderDefined>::new(eq, Some(*iterations), Some(*step)); // I have to explicitly annotate Equation::<UnderDefined> !?

    match initial_model.add_variables(vars)? {
        ValidEquation::Full(equation) => equation.evaluate()?.ninety_ci(),
        _ => bail!("Variables missing"),
    }
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    //     use statrs::assert_almost_eq;

    // Equation<UnderDefined> /////////////////////////////////////////////////

    #[test]
    fn extract_variable_names_simple_names() {
        let variables = Equation::<UnderDefined>::extract_variable_names("A1 + A2 - B");
        assert_eq!(variables.len(), 3);
        assert_eq!(variables[0], "A1");
        assert_eq!(variables[1], "A2");
        assert_eq!(variables[2], "B");
    }

    #[test]
    fn extract_variable_names_complex_names() {
        let variables = Equation::<UnderDefined>::extract_variable_names("A_1 + some_A*B1");
        assert_eq!(variables.len(), 3);
        assert_eq!(variables[0], "A_1");
        assert_eq!(variables[1], "some_A");
        assert_eq!(variables[2], "B1");
    }

    #[test]
    fn extract_variable_names_with_numbers() {
        let variables = Equation::<UnderDefined>::extract_variable_names("4.5 * A_1 + 2 * some");
        assert_eq!(variables.len(), 2);
        assert_eq!(variables[0], "A_1");
        assert_eq!(variables[1], "some");
    }

    #[test]
    fn sample_variable_incorrect_type() {
        assert!(Equation::<UnderDefined>::sample_variable("incorrect", &1., &2., 1).is_err());
    }

    #[test]
    fn sample_variable_incorrect_bounds() {
        assert!(Equation::<UnderDefined>::sample_variable("incorrect", &2., &1., 1).is_err());
    }

    #[test]
    fn sample_variable_size_correct() {
        let sample = Equation::<UnderDefined>::sample_variable("uniform", &1., &2., 100).unwrap();
        assert_eq!(sample.len(), 100);
        let sample = Equation::<UnderDefined>::sample_variable("normal", &1., &2., 100).unwrap();
        assert_eq!(sample.len(), 100);
        let sample = Equation::<UnderDefined>::sample_variable("range", &1., &2., 100).unwrap();
        assert_eq!(sample.len(), 100);
    }

    #[test]
    fn add_variable_not_in_equation() {
        let mut eq = Equation::<UnderDefined>::new("V1", None, None);
        let var = VariableDescription {
            name: "V1",
            shape: "incorrect",
            lower: 1.,
            upper: 2.,
        };
        assert!(eq.add_variable(&var).is_err());
    }

    #[test]
    fn add_variable_adds_var() {
        let mut eq = Equation::<UnderDefined>::new("A + B", None, None);
        eq.add_variable(
            &(VariableDescription {
                name: "A",
                shape: "uniform",
                lower: 1.,
                upper: 2.,
            }),
        )
        .unwrap();
        assert_eq!(eq.vars.len(), 1);
        assert!(eq.vars.contains_key("A"));
    }

    #[test]
    fn add_variable_incorrect_type() {
        let mut eq = Equation::<UnderDefined>::new("A + B", None, None);
        assert!(eq
            .add_variable(
                &(VariableDescription {
                    name: "A",
                    shape: "incorrect",
                    lower: 1.,
                    upper: 2.,
                }),
            )
            .is_err());
    }

    #[test]
    fn add_variable_incorrect_bounds() {
        let mut eq = Equation::<UnderDefined>::new("A + B", None, None);
        assert!(eq
            .add_variable(
                &(VariableDescription {
                    name: "A",
                    shape: "incorrect",
                    lower: 100.,
                    upper: 2.,
                }),
            )
            .is_err());
    }

    #[test]
    fn add_variables_returns_partial() {
        let eq = Equation::<UnderDefined>::new("A + B", None, None);
        let vars = vec![VariableDescription {
            name: "A",
            shape: "uniform",
            lower: 1.,
            upper: 2.,
        }];
        if let ValidEquation::Full(..) = eq.add_variables(&vars).unwrap() {
            panic!("wrong type")
        }
    }

    #[test]
    fn add_variables_returns_full() {
        let eq = Equation::<UnderDefined>::new("A + B", None, None);
        let vars = vec![
            VariableDescription {
                name: "A",
                shape: "uniform",
                lower: 1.,
                upper: 2.,
            },
            VariableDescription {
                name: "B",
                shape: "uniform",
                lower: 1.,
                upper: 2.,
            },
        ];
        if let ValidEquation::Partial(..) = eq.add_variables(&vars).unwrap() {
            panic!("wrong type")
        }
    }

    // Equation<FullyDefined> /////////////////////////////////////////////////

    #[test]
    fn compute_histogram_series_single() {
        let mut data = vec![1.];
        assert!(Equation::<FullyDefined>::compute_histogram(&mut data, &0.1).is_none());
    }

    #[test]
    fn compute_histogram_check_size() {
        let mut data = vec![1., 2., 3., 4.];
        let (buckets, counts) =
            Equation::<FullyDefined>::compute_histogram(&mut data, &0.1).unwrap();
        assert_eq!(buckets.len(), counts.len());
    }

    #[test]
    fn compute_histogram_bucket_size_smaller_than_1() {
        let mut data: Vec<f64> = vec![1., 3.];
        let (buckets, freqs) =
            Equation::<FullyDefined>::compute_histogram(&mut data, &0.5).unwrap();
        assert_eq!(buckets, vec![1., 1.5, 2., 2.5, 3.]);
        assert_eq!(freqs, vec![1, 0, 0, 0, 1]);
    }

    #[test]
    fn compute_histogram_negative_values_and_small_bucket() {
        let mut data: Vec<f64> = vec![-1., 2.];
        let (buckets, freqs) =
            Equation::<FullyDefined>::compute_histogram(&mut data, &0.5).unwrap();
        assert_eq!(buckets, vec![-1., -0.5, 0., 0.5, 1., 1.5, 2.]);
        assert_eq!(freqs, vec![1, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn compute_histogram_larger_test() {
        let mut data: Vec<f64> = vec![0.33, 1.1, 1.6, 6.0, 5.5, 6.0, 4.3, 7.1, -1.1];
        let (buckets, freqs) =
            Equation::<FullyDefined>::compute_histogram(&mut data, &2.0).unwrap();
        assert_eq!(buckets, vec![-2., 0., 2., 4., 6.]);
        assert_eq!(freqs, vec![1, 3, 0, 2, 3]);
    }

    #[test]
    fn compute_histogram_sub_unit() {
        let mut data: Vec<f64> = vec![0.83, 0.96, 1.15];
        let (buckets, freqs) =
            Equation::<FullyDefined>::compute_histogram(&mut data, &0.1).unwrap();
        assert_eq!(buckets, vec![0.8, 0.9, 1.0, 1.1]);
        assert_eq!(freqs, vec![1, 1, 0, 1]);
    }

    //     //////////////////////////////////////////////////////////////////////

    //     #[test]
    //     fn ninety_ci_bucket_size_smaller_than_1() {
    //         let buckets = vec![1., 1.5, 2., 2.5, 3.];
    //         let freqs = vec![1, 0, 0, 0, 1];
    //         let (low, up) = ninety_ci(&buckets, &freqs, &2);
    //         assert_eq!(low, 1.);
    //         assert_eq!(up, 3.);
    //     }

    //     #[test]
    //     fn ninety_ci_negative_values_and_small_bucket() {
    //         let buckets = vec![-1., -0.5, 0., 0.5, 1., 1.5, 2.];
    //         let freqs = vec![1, 0, 0, 0, 0, 0, 1];
    //         let (low, up) = ninety_ci(&buckets, &freqs, &2);
    //         assert_eq!(low, -1.);
    //         assert_eq!(up, 2.);
    //     }

    //     #[test]
    //     fn ninety_ci_larger_test() {
    //         let buckets = vec![-4., -2., 0., 2., 4., 6.];
    //         let freqs = vec![1, 1, 4, 40, 3, 1];
    //         let (low, up) = ninety_ci(&buckets, &freqs, &50);
    //         assert_eq!(low, -2.);
    //         assert_eq!(up, 4.);
    //     }

    //     ///////////////////////////////////////////////////////////////////////////////

    //     #[test]
    //     fn integration_single_variable_normal() {
    //         let equation: &str = "VAR";
    //         let variables: Vec<VariableDescription> =
    //             vec![VariableDescription::new("VAR", "normal", 100., 200.)];
    //         const ITERATIONS: usize = 5000;
    //         const BUCKET_SIZE: f64 = 0.1;

    //         let (buckets, freqs) =
    //             generate_freq_data(equation, &variables, &ITERATIONS, &BUCKET_SIZE).unwrap();
    //         let (low, up) = ninety_ci(&buckets, &freqs, &ITERATIONS);

    //         println!("DEBUG - test 90% CI: [{}, {}]", low, up);
    //         assert_almost_eq!(low, 100., 1.);
    //         assert_almost_eq!(up, 200., 1.);
    //     }

    //     #[test]
    //     fn integration_single_variable_uniform() {
    //         // For a symetric random variable with a 90%CI of [1,2], the equation
    //         // "1 + variable" should obviously have a 90%CI of [2,3].
    //         let equation: &str = "1 + VAR";
    //         let variables: Vec<VariableDescription> =
    //             vec![VariableDescription::new("VAR", "uniform", 1., 2.)];
    //         const ITERATIONS: usize = 5000;
    //         const BUCKET_SIZE: f64 = 0.1;

    //         let (buckets, freqs) =
    //             generate_freq_data(equation, &variables, &ITERATIONS, &BUCKET_SIZE).unwrap();
    //         let (low, up) = ninety_ci(&buckets, &freqs, &ITERATIONS);

    //         println!("DEBUG - test 90% CI: [{}, {}]", low, up);
    //         assert_almost_eq!(low, 2., 0.1);
    //         assert_almost_eq!(up, 3., 0.1);
    //     }
}
