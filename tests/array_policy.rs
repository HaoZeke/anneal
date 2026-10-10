use std::convert::Infallible;

use anneal_core::accept::{AcceptRule, Metropolis, ProbabilityArithmetic, TsallisAccept};
use anneal_core::movekernel::{MoveKernel, TsallisVisit};
use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

struct VectorArithmetic(usize);

impl ProbabilityArithmetic<f64> for VectorArithmetic {
    type Value = Vec<f64>;
    type Error = Infallible;

    fn constant(&self, value: f64) -> Result<Self::Value, Infallible> {
        Ok(vec![value; self.0])
    }
    fn scale(&self, value: &Self::Value, factor: f64) -> Result<Self::Value, Infallible> {
        Ok(value.iter().map(|value| value * factor).collect())
    }
    fn divide(&self, value: &Self::Value, divisor: f64) -> Result<Self::Value, Infallible> {
        Ok(value.iter().map(|value| value / divisor).collect())
    }
    fn offset(&self, value: &Self::Value, addend: f64) -> Result<Self::Value, Infallible> {
        Ok(value.iter().map(|value| value + addend).collect())
    }
    fn exp(&self, value: &Self::Value) -> Result<Self::Value, Infallible> {
        Ok(value.iter().map(|value| value.exp()).collect())
    }
    fn powf(&self, value: &Self::Value, exponent: f64) -> Result<Self::Value, Infallible> {
        Ok(value.iter().map(|value| value.powf(exponent)).collect())
    }
    fn select_nonpositive(
        &self,
        condition: &Self::Value,
        nonpositive: &Self::Value,
        positive: &Self::Value,
    ) -> Result<Self::Value, Infallible> {
        assert_eq!(condition.len(), self.0);
        assert_eq!(nonpositive.len(), self.0);
        assert_eq!(positive.len(), self.0);
        Ok((0..self.0)
            .map(|index| {
                if condition[index] <= 0.0 {
                    nonpositive[index]
                } else {
                    positive[index]
                }
            })
            .collect())
    }
}

#[test]
fn metropolis_arrays_use_the_native_scalar_rule() {
    let deltas = vec![-100.0, -1.0, 0.0, 0.125, 1.0, 10.0, f64::INFINITY];
    let arithmetic = VectorArithmetic(deltas.len());
    for temperature in [0.125, 1.0, 10.0] {
        let actual = Metropolis
            .probabilities_with(&deltas, temperature, &arithmetic)
            .unwrap();
        let expected: Vec<f64> = deltas
            .iter()
            .map(|&delta| Metropolis.accept_prob(delta, temperature))
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn tsallis_arrays_use_the_native_scalar_rule() {
    let deltas = vec![-100.0, -1.0, 0.0, 0.125, 1.0, 10.0, f64::INFINITY];
    let arithmetic = VectorArithmetic(deltas.len());
    for q in [-2.0, 0.0, 0.5, 1.0, 1.0 + f64::EPSILON, 1.7, 2.7] {
        let rule = TsallisAccept::new(q);
        for temperature in [0.125, 1.0, 10.0] {
            let actual = rule
                .probabilities_with(&deltas, temperature, &arithmetic)
                .unwrap();
            let expected: Vec<f64> = deltas
                .iter()
                .map(|&delta| rule.accept_prob(delta, temperature))
                .collect();
            assert_eq!(actual, expected, "q={q}, temperature={temperature}");
        }
    }
}

#[test]
fn array_rules_retain_compact_support_and_downhill_acceptance() {
    let deltas = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
    let arithmetic = VectorArithmetic(deltas.len());
    let compact = TsallisAccept::new(0.0)
        .probabilities_with(&deltas, 1.0, &arithmetic)
        .unwrap();
    assert_eq!(compact, vec![1.0, 1.0, 0.5, 0.0, 0.0]);
    let heavy = TsallisAccept::new(2.0)
        .probabilities_with(&deltas, 1.0, &arithmetic)
        .unwrap();
    assert_eq!(heavy, vec![1.0, 1.0, 2.0 / 3.0, 0.5, 1.0 / 3.0]);
}

#[test]
fn nan_energy_differences_do_not_become_finite_acceptance_probabilities() {
    let deltas = vec![f64::NAN];
    let arithmetic = VectorArithmetic(1);
    assert!(
        Metropolis
            .probabilities_with(&deltas, 1.0, &arithmetic)
            .unwrap()[0]
            .is_nan()
    );
    for q in [0.0, 1.0, 1.7] {
        assert!(
            TsallisAccept::new(q)
                .probabilities_with(&deltas, 1.0, &arithmetic)
                .unwrap()[0]
                .is_nan()
        );
    }
}

#[test]
fn native_visit_draws_consume_the_shared_prepared_parameters() {
    for q in [1.5, 2.0, 2.62] {
        let kernel = TsallisVisit::new(q);
        let parameters = kernel.parameters(0.75);
        let start = array![0.25, -0.5, 1.0];
        let mut actual_rng = StdRng::seed_from_u64(71);
        let mut expected_rng = actual_rng.clone();
        let expected = start.mapv(|coordinate| {
            let x: f64 = StandardNormal.sample(&mut expected_rng);
            let y: f64 = StandardNormal.sample(&mut expected_rng);
            let displacement = parameters.scale * x / y.abs().powf(parameters.exponent);
            assert!(displacement.abs() < parameters.tail_limit);
            coordinate + displacement
        });
        let actual = kernel.propose(start.view(), 0.75, &mut actual_rng);
        assert_eq!(actual, expected);
    }
}
