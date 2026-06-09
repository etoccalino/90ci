extern crate meval;

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use lazy_static::lazy_static;
use rand::distributions::Distribution;
use rand::{thread_rng, Rng};
use regex::Regex;
use statrs::distribution::{DiscreteUniform, Normal, Uniform};

///////////////////////////////////////////////////////////////////////////////
/// A Distro can be sampled, and therefore used by the `rand` package.
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
    Partial(()),
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

    /// Return a series of samples of a random variable. Fail if a variable has type other than
    /// "uniform", "range" or "normal", or a lower bound is greater than an upper bound.
    fn sample_variable(distribution: &str, lower: &f64, upper: &f64, n: usize) -> Result<Vec<f64>> {
        let dist: Distro = Distro::new(distribution, *lower, *upper)?;
        let rng = thread_rng();
        Ok(rng.sample_iter(&dist).take(n).collect())
    }

    /// Add a variable description to the equation.
    /// Fails if the variable is not referenced in the equation (E-02),
    /// if the bounds are inverted — strictly lower > upper (E-03),
    /// or if the shape is unsupported (E-11).
    fn add_variable(&mut self, var: &'a VariableDescription) -> Result<()> {
        // E-02: variable defined but not present in the equation
        if !self.var_names.contains(var.name) {
            bail!(
                "`{}` is defined but not used — use it or remove the row.",
                var.name
            );
        }
        // E-03: inverted bounds (strict: lower == upper is valid for `range`)
        if var.lower > var.upper {
            bail!(
                "`{}`: 5th ({}) must be below 95th ({}).",
                var.name,
                var.lower,
                var.upper
            );
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
        // E-12: detect duplicate variable names before the HashMap can silently overwrite one.
        // Scan in a single pass; report the first duplicate found.
        let mut seen: HashSet<&str> = HashSet::with_capacity(vars.len());
        for var in vars.iter() {
            if !seen.insert(var.name) {
                bail!(
                    "Two variables are named `{}` — names must be unique.",
                    var.name
                );
            }
        }

        for var in vars.iter() {
            self.add_variable(var)?;
        }

        if self.vars.len() < self.var_names.len() {
            Ok(ValidEquation::Partial(()))
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
    ///   Fails if the vector is empty or has a single element.
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
        series.sort_by(|a, b| a.total_cmp(b));
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
        let last_ix = buckets.len() - 1; // guaranteed >= 0 since len >= 1 (we pushed at least one bucket above)
        for val in series {
            let raw = val.div_euclid(*bucket_size) - buckets_offset;
            // Guard against a negative raw index (shouldn't happen for finite values >= lowest,
            // but floating-point arithmetic can produce a tiny negative due to rounding).
            // Also clamp to last_ix: floating-point rounding near the highest value can push
            // the computed index one past the end.
            let bucket_index = if raw < 0.0 {
                0_usize
            } else {
                (raw as usize).min(last_ix)
            };
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
                Ok(result) => {
                    // Only retain finite results; non-finite outputs (inf, NaN) arise
                    // from degenerate models (e.g. division by zero) and must never
                    // reach compute_histogram, which would panic on NaN comparisons
                    // or corrupt the bucket index arithmetic on inf.
                    if result.is_finite() {
                        series.push(result);
                    }
                }
                Err(e) => {
                    bail!("Error evaluating the equation: {:?}", e);
                }
            }
        }
        let histogram = Equation::compute_histogram(&mut series, &self.step)
            .ok_or_else(|| anyhow!("The equation produced an undefined result (e.g. division by zero) — check the formula/bounds."))?;
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
        let (buckets, counts): &(Vec<f64>, Vec<usize>) = self
            .hist
            .as_ref()
            .ok_or_else(|| anyhow!("Equation was not evaluated!?"))?;

        let mut lower: &f64 = buckets
            .first()
            .ok_or_else(|| anyhow!("Histogram has no buckets — equation may be degenerate."))?;
        let mut upper: &f64 = buckets
            .last()
            .ok_or_else(|| anyhow!("Histogram has no buckets — equation may be degenerate."))?;
        let total: usize = counts.iter().sum();
        // total > 0 is guaranteed: compute_histogram returns Some only when series.len() >= 2,
        // and evaluate only calls it after filtering non-finite values, so the Ok path here
        // requires at least two finite samples.
        let mut acc = 0_f64;
        for ix in 0..buckets.len() {
            acc += counts[ix] as f64 / total as f64;

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

/// The full result of a simulation: the 90% confidence interval plus the
/// histogram of the output distribution that produced it.
#[derive(Debug, Clone)]
pub struct Simulation {
    pub ci_low: f64,
    pub ci_high: f64,
    pub buckets: Vec<f64>, // bucket lower bounds
    pub counts: Vec<usize>, // samples that fell in each bucket (same length as `buckets`)
    pub samples: usize,    // total iterations run
}

/// Entrypoint to the library: run the Monte-Carlo simulation and return both the
/// 90% confidence interval and the output histogram.
/// Use the `VariableDescription` struct to describe each variable.
pub fn simulate(
    eq: &str,
    vars: &[VariableDescription],
    iterations: &usize,
    step: &f64,
) -> Result<Simulation> {
    // E-05a: empty or blank equation
    if eq.trim().is_empty() {
        bail!("Enter an equation.");
    }
    // E-05b: no variable rows provided
    if vars.is_empty() {
        bail!("Add at least one variable.");
    }

    let initial_model: Equation<UnderDefined> =
        Equation::<UnderDefined>::new(eq, Some(*iterations), Some(*step));

    match initial_model.add_variables(vars)? {
        ValidEquation::Full(equation) => {
            let evaluated = equation.evaluate()?;
            let samples = evaluated.resolution;
            let (ci_low, ci_high) = evaluated.ninety_ci()?;
            let (buckets, counts) = evaluated
                .hist
                .ok_or_else(|| anyhow!("Equation was not evaluated!?"))?;
            Ok(Simulation {
                ci_low,
                ci_high,
                buckets,
                counts,
                samples,
            })
        }
        ValidEquation::Partial(()) => {
            // E-01: find the tokens in the equation that have no corresponding variable row.
            // Call extract_variable_names directly on the equation string to get the token
            // list, then diff against the supplied names.
            let eq_tokens = Equation::<UnderDefined>::extract_variable_names(eq);
            let supplied: HashSet<&str> = vars.iter().map(|v| v.name).collect();
            let mut missing: Vec<&str> = eq_tokens
                .into_iter()
                .filter(|t| !supplied.contains(t))
                .collect();
            // Deduplicate while preserving first-seen order.
            {
                let mut seen = HashSet::with_capacity(missing.len());
                missing.retain(|t| seen.insert(*t));
            }

            if missing.len() == 1 {
                bail!(
                    "`{}` is used in the equation but not defined — add a variable row for it (or remove it).",
                    missing[0]
                );
            } else {
                // Multiple missing tokens: list them all, comma-separated, each backtick-quoted.
                let list: Vec<String> = missing.iter().map(|t| format!("`{}`", t)).collect();
                bail!(
                    "{} are used in the equation but not defined — add variable rows for them (or remove them).",
                    list.join(", ")
                );
            }
        }
    }
}

/// Convenience wrapper returning only the 90% confidence interval, as `(low, high)`.
pub fn ci90(
    eq: &str,
    vars: &[VariableDescription],
    iterations: &usize,
    step: &f64,
) -> Result<(f64, f64)> {
    let s = simulate(eq, vars, iterations, step)?;
    Ok((s.ci_low, s.ci_high))
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

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
        // lower=100 > upper=2 trips E-03 before E-11 (bounds are checked before shape).
        let mut eq = Equation::<UnderDefined>::new("A + B", None, None);
        let result = eq.add_variable(
            &(VariableDescription {
                name: "A",
                shape: "incorrect",
                lower: 100.,
                upper: 2.,
            }),
        );
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("must be below"),
            "inverted bounds must produce E-03 message (not E-11)"
        );
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

    // simulate — non-finite output must never panic ///////////////////////////

    const NON_FINITE_ERR: &str =
        "The equation produced an undefined result (e.g. division by zero) — check the formula/bounds.";

    /// `X / Y` where Y is always 0 (range lower=upper=0) produces inf for every
    /// sample; after filtering non-finite values the surviving series is empty,
    /// so simulate must return Err with the exact diagnostic message.
    #[test]
    fn simulate_div_zero_range_returns_err() {
        let vars = vec![
            VariableDescription {
                name: "X",
                shape: "normal",
                lower: 1.,
                upper: 10.,
            },
            VariableDescription {
                name: "Y",
                shape: "range",
                lower: 0.,
                upper: 0.,
            },
        ];
        let result = simulate("X / Y", &vars, &1_000, &0.1);
        assert!(result.is_err(), "expected Err, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            NON_FINITE_ERR,
            "wrong error message"
        );
    }

    /// `X / 0` (literal zero denominator) — meval evaluates this to inf for
    /// every sample; same expected Err path as above.
    #[test]
    fn simulate_literal_div_zero_returns_err() {
        let vars = vec![VariableDescription {
            name: "X",
            shape: "uniform",
            lower: 1.,
            upper: 2.,
        }];
        let result = simulate("X / 0", &vars, &1_000, &0.1);
        assert!(result.is_err(), "expected Err, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            NON_FINITE_ERR,
            "wrong error message"
        );
    }

    /// `0 / 0` produces NaN for every sample. is_finite() is false for NaN, so all
    /// samples are dropped and simulate must return the same Err as the inf case.
    /// This confirms that the non-finite filter covers NaN, not only inf.
    #[test]
    fn simulate_nan_model_returns_err() {
        // X is range(0,0) so every sample is exactly 0; 0/0 = NaN in IEEE 754.
        let vars = vec![VariableDescription {
            name: "X",
            shape: "range",
            lower: 0.,
            upper: 0.,
        }];
        let result = simulate("X / X", &vars, &1_000, &0.1);
        assert!(result.is_err(), "expected Err for NaN model, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            NON_FINITE_ERR,
            "wrong error message for NaN model"
        );
    }

    /// Partially-degenerate model: `X / (X - 5)` where X is range(0, 10).
    /// When X == 5, the denominator is 0 and the result is inf (non-finite, filtered out).
    /// The remaining ~90 % of samples are finite, so simulate must return Ok.
    /// The returned CI must satisfy the invariant ci_low <= ci_high.
    ///
    /// Because the samples are random we cannot pin exact bounds, so we assert
    /// the robust invariants only. The denominator fix (dividing by surviving-sample
    /// count, not configured resolution) is what makes the CI reach the 95 % threshold
    /// even when some samples are dropped.
    #[test]
    fn simulate_partial_degenerate_returns_ok_with_valid_ci() {
        let vars = vec![VariableDescription {
            name: "X",
            shape: "range",
            lower: 0.,
            upper: 10.,
        }];
        // X = 5 is one of the 11 integer values (0..=10), so ~1/11 of samples are non-finite.
        let result = simulate("X / (X - 5)", &vars, &5_000, &0.1);
        assert!(
            result.is_ok(),
            "expected Ok for partially-degenerate model, got: {:?}",
            result.err()
        );
        let sim = result.unwrap();
        assert!(
            sim.ci_low <= sim.ci_high,
            "ci_low ({}) must be <= ci_high ({})",
            sim.ci_low,
            sim.ci_high
        );
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

    // simulate — validation & honest error messages (Stage 3) //////////////////
    //
    // Validation precedence (one violation per run):
    //   empty-equation (E-05a)
    //   → empty-vars (E-05b)
    //   → duplicate-names (E-12)
    //   → per-variable: not-used (E-02), then inverted-bounds (E-03), then bad-shape (E-11)
    //   → missing-token (E-01)

    // E-05a: empty equation string
    #[test]
    fn simulate_empty_equation_returns_err() {
        let vars = vec![VariableDescription {
            name: "X",
            shape: "uniform",
            lower: 1.,
            upper: 2.,
        }];
        let result = simulate("", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for empty equation, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            "Enter an equation.",
            "wrong error message for E-05a"
        );
    }

    // E-05a: whitespace-only equation (same guard as empty)
    #[test]
    fn simulate_whitespace_equation_returns_err() {
        let vars = vec![VariableDescription {
            name: "X",
            shape: "uniform",
            lower: 1.,
            upper: 2.,
        }];
        let result = simulate("   ", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for whitespace equation, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            "Enter an equation.",
            "wrong error message for E-05a whitespace"
        );
    }

    // E-05b: zero variables
    #[test]
    fn simulate_empty_vars_returns_err() {
        let result = simulate("X + Y", &[], &100, &0.1);
        assert!(result.is_err(), "expected Err for empty vars, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            "Add at least one variable.",
            "wrong error message for E-05b"
        );
    }

    // E-05 ordering: empty-equation is checked before empty-vars
    #[test]
    fn simulate_empty_equation_takes_precedence_over_empty_vars() {
        let result = simulate("", &[], &100, &0.1);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Enter an equation.",
            "empty-equation must be reported before empty-vars"
        );
    }

    // E-12: duplicate variable names must be detected before HashMap merge
    #[test]
    fn simulate_duplicate_variable_names_returns_err() {
        let vars = vec![
            VariableDescription {
                name: "X",
                shape: "uniform",
                lower: 1.,
                upper: 2.,
            },
            VariableDescription {
                name: "X",
                shape: "normal",
                lower: 3.,
                upper: 8.,
            },
        ];
        let result = simulate("X", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for duplicate variable name, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            "Two variables are named `X` — names must be unique.",
            "wrong error message for E-12"
        );
    }

    // E-12 before E-02: duplicate-name must fire even when neither variable is in the equation.
    // Two vars named `Y` with equation `"X"`: must report duplicate-name, not not-used.
    #[test]
    fn simulate_duplicate_name_takes_precedence_over_not_used() {
        let vars = vec![
            VariableDescription {
                name: "Y",
                shape: "uniform",
                lower: 1.,
                upper: 2.,
            },
            VariableDescription {
                name: "Y",
                shape: "uniform",
                lower: 1.,
                upper: 2.,
            },
        ];
        let result = simulate("X", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for duplicate name, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("names must be unique"),
            "E-12 must fire before E-02, got: {}",
            msg
        );
        assert!(
            !msg.contains("not used"),
            "E-02 must not fire when E-12 applies, got: {}",
            msg
        );
    }

    // E-02: variable defined but not used in equation
    #[test]
    fn simulate_variable_not_used_returns_err() {
        let vars = vec![
            VariableDescription {
                name: "X",
                shape: "uniform",
                lower: 1.,
                upper: 2.,
            },
            VariableDescription {
                name: "Unused",
                shape: "uniform",
                lower: 1.,
                upper: 2.,
            },
        ];
        let result = simulate("X", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for unused variable, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            "`Unused` is defined but not used — use it or remove the row.",
            "wrong error message for E-02"
        );
    }

    // E-03: inverted bounds (lower strictly > upper) named in error message
    #[test]
    fn simulate_inverted_bounds_returns_named_err() {
        let vars = vec![VariableDescription {
            name: "Revenue",
            shape: "normal",
            lower: 10.,
            upper: 5.,
        }];
        let result = simulate("Revenue", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for inverted bounds, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            "`Revenue`: 5th (10) must be below 95th (5).",
            "wrong error message for E-03"
        );
    }

    // E-03: lower == upper for `range` must remain valid (existing E-07 tests depend on this)
    #[test]
    fn simulate_range_equal_bounds_is_valid() {
        let vars = vec![VariableDescription {
            name: "X",
            shape: "range",
            lower: 3.,
            upper: 3.,
        }];
        // X is always 3; a constant series has length >= 2 so compute_histogram runs fine,
        // producing a single bucket with all samples. simulate must return Ok.
        let result = simulate("X", &vars, &100, &0.1);
        assert!(
            result.is_ok(),
            "equal bounds for range must succeed (not E-03), got: {:?}",
            result.err()
        );
    }

    // E-11: unsupported distribution shape — exact message must be preserved
    #[test]
    fn simulate_bad_shape_returns_err_with_exact_message() {
        let vars = vec![VariableDescription {
            name: "X",
            shape: "poisson",
            lower: 1.,
            upper: 2.,
        }];
        let result = simulate("X", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for bad shape, got Ok");
        assert_eq!(
            result.unwrap_err().to_string(),
            "Unsupported distribution. Use either 'normal', 'range' or 'uniform'.",
            "wrong error message for E-11"
        );
    }

    // E-01: token in equation with no corresponding variable row
    #[test]
    fn simulate_missing_token_returns_named_err() {
        let vars = vec![VariableDescription {
            name: "A",
            shape: "uniform",
            lower: 1.,
            upper: 2.,
        }];
        // Equation references B which has no variable row
        let result = simulate("A + B", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for missing token, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("`B`"),
            "error message must name the missing token, got: {}",
            msg
        );
        assert!(
            msg.contains("is used in the equation but not defined"),
            "error message must contain corrective phrasing, got: {}",
            msg
        );
        assert_eq!(
            msg,
            "`B` is used in the equation but not defined — add a variable row for it (or remove it).",
            "wrong error message for E-01 (single missing token)"
        );
    }

    // E-01: multiple missing tokens — all must appear in the error
    #[test]
    fn simulate_multiple_missing_tokens_names_all() {
        let vars = vec![VariableDescription {
            name: "A",
            shape: "uniform",
            lower: 1.,
            upper: 2.,
        }];
        let result = simulate("A + B + C", &vars, &100, &0.1);
        assert!(result.is_err(), "expected Err for multiple missing tokens");
        let msg = result.unwrap_err().to_string();
        // Both must appear backtick-quoted in the message (pinning the actual message format).
        assert!(
            msg.contains("`B`"),
            "error message must name missing token B as `B`, got: {}",
            msg
        );
        assert!(
            msg.contains("`C`"),
            "error message must name missing token C as `C`, got: {}",
            msg
        );
    }

    // Equation<Evaluated> ////////////////////////////////////////////////////

    #[test]
    fn ninety_ci_bucket_size_smaller_than_1() {
        let buckets = vec![1., 1.5, 2., 2.5, 3.];
        let freqs = vec![1, 0, 0, 0, 1];
        let eq = Equation::<Evaluated> {
            _status: Evaluated,
            eq: "A",
            step: 0.5,
            resolution: 2,
            vars: HashMap::new(),
            var_names: HashSet::new(),
            hist: Some((buckets, freqs)),
        };
        let (low, up) = eq.ninety_ci().unwrap();
        assert_eq!(low, 1.);
        assert_eq!(up, 3.);
    }

    #[test]
    fn ninety_ci_negative_values_and_small_bucket() {
        let buckets = vec![-1., -0.5, 0., 0.5, 1., 1.5, 2.];
        let freqs = vec![1, 0, 0, 0, 0, 0, 1];
        let eq = Equation::<Evaluated> {
            _status: Evaluated,
            eq: "A",
            step: 0.1,
            resolution: 2,
            vars: HashMap::new(),
            var_names: HashSet::new(),
            hist: Some((buckets, freqs)),
        };
        let (low, up) = eq.ninety_ci().unwrap();
        assert_eq!(low, -1.);
        assert_eq!(up, 2.);
    }

    #[test]
    fn ninety_ci_larger_test() {
        let buckets = vec![-4., -2., 0., 2., 4., 6.];
        let freqs = vec![1, 1, 4, 40, 3, 1];
        let eq = Equation::<Evaluated> {
            _status: Evaluated,
            eq: "A",
            step: 0.1,
            resolution: 50,
            vars: HashMap::new(),
            var_names: HashSet::new(),
            hist: Some((buckets, freqs)),
        };
        let (low, up) = eq.ninety_ci().unwrap();
        assert_eq!(low, -2.);
        assert_eq!(up, 4.);
    }

    // Probe: degenerate equal-bound cases for uniform and normal.
    // These are NOT E-03 (inverted bounds); lower == upper is a degenerate (zero-width)
    // input. statrs behaviour differs:
    //   - uniform(5, 5): statrs ACCEPTS it (Uniform::new(5.0,5.0) is Ok); every sample
    //     is exactly 5.0. A 1000-sample constant series has len >= 2 but all values equal,
    //     so compute_histogram produces a single bucket — the CI computation still works,
    //     giving ci_low == ci_high == 5.0. simulate returns Ok. No panic.
    //   - normal(5, 5): lower==upper means std_dev=(5-5)/3.29=0.0; statrs Normal::new(5.0,0.0)
    //     returns Err("Bad distribution parameters") — a clean Err, never a panic.
    //
    // Both are clean (no panic); the normal case message is the terse statrs text, which
    // could be improved in a future stage but is out of E-03 scope (as noted in the spec).

    #[test]
    fn probe_uniform_equal_bounds_simulate_does_not_panic() {
        // uniform(5, 5): statrs accepts it; simulate returns Ok (constant output, valid CI).
        let vars = vec![VariableDescription { name: "X", shape: "uniform", lower: 5., upper: 5. }];
        let result = simulate("X", &vars, &100, &0.1);
        // Must not panic; we don't enforce Ok vs Err here — just document the behaviour.
        let _ = result; // Ok(ci_low=5, ci_high=5) expected but not required by this probe
    }

    #[test]
    fn probe_normal_equal_bounds_is_clean_err() {
        // normal(5, 5): std_dev=0 → statrs returns Err, not a panic.
        let result = Equation::<UnderDefined>::sample_variable("normal", &5.0, &5.0, 1);
        assert!(result.is_err(), "normal(5,5) must be Err (std_dev=0 rejected by statrs)");
        let msg = result.unwrap_err().to_string();
        assert!(!msg.is_empty(), "statrs error message must not be empty");
        // Document the actual terse message from statrs (not our E-03 message).
        // This message is poor but is out of E-03 scope per the spec.
        assert!(
            !msg.contains("must be below"),
            "normal equal-bounds must NOT produce E-03 message, got: {}",
            msg
        );
    }
}
