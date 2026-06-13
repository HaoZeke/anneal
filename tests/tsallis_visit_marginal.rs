//! Pins the per-coordinate factorization of the Tsallis visiting kernel.
//!
//! `TsallisVisit` samples each coordinate independently as a Student-t with
//! `dof = (3 - q_v) / (q_v - 1)` degrees of freedom, scaled by
//! `T^(1/(3 - q_v))` (manuscript Appendix C). The displayed isotropic
//! D-dimensional density carries the limit theorems; the sampler uses this
//! coordinate factorization, exact at `D = 1` and isotropic Gaussian (`q_v ->
//! 1`) / Cauchy (`q_v = 2`) at the two named limits. These tests lock that
//! behavior so a future change to the sampler is caught.

use anneal_core::movekernel::{MoveKernel, TsallisVisit};
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Collects single-coordinate increments from `propose` at the origin, where
/// the increment equals `T^(1/(3-q_v))` times a `dof`-Student-t draw.
fn increments(q_v: f64, temp: f64, n: usize, seed: u64) -> Vec<f64> {
    let k = TsallisVisit::new(q_v);
    let x = array![0.0_f64];
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| k.propose(x.view(), temp, &mut rng)[0])
        .collect()
}

fn fraction_within(samples: &[f64], bound: f64) -> f64 {
    let c = samples.iter().filter(|&&v| v.abs() <= bound).count();
    c as f64 / samples.len() as f64
}

#[test]
fn qv_two_is_standard_cauchy() {
    // q_v = 2 -> dof = 1 -> standard Cauchy at T = 1 (scale = 1).
    // P(|X| <= 1) = 0.5 for Cauchy(0, 1).
    let s = increments(2.0, 1.0, 200_000, 11);
    let f = fraction_within(&s, 1.0);
    assert!((f - 0.5).abs() < 0.02, "q_v=2 not Cauchy: P(|X|<=1) = {f:.4}");
}

#[test]
fn temperature_scales_the_cauchy_width() {
    // At T = 4, scale = 4^(1/(3-2)) = 4, so increments are Cauchy(0, 4):
    // P(|X| <= 4) = 0.5.
    let s = increments(2.0, 4.0, 200_000, 12);
    let f = fraction_within(&s, 4.0);
    assert!(
        (f - 0.5).abs() < 0.02,
        "T-scaling broken: P(|X|<=4) at T=4 = {f:.4}"
    );
    // And the unscaled bound 1.0 should capture far less than half.
    let f1 = fraction_within(&s, 1.0);
    assert!(f1 < 0.30, "width did not scale with T: P(|X|<=1) = {f1:.4}");
}

#[test]
fn near_one_qv_approaches_gaussian() {
    // q_v = 1.05 -> dof = 39, close to normal: P(|X| <= 1.96) ~ 0.95 and the
    // sample stays light-tailed (finite, ~1.05 variance).
    let s = increments(1.05, 1.0, 200_000, 13);
    let f = fraction_within(&s, 1.96);
    assert!(
        (f - 0.95).abs() < 0.02,
        "q_v->1 not near-Gaussian: P(|X|<=1.96) = {f:.4}"
    );
}
