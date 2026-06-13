//! Validates the Tsallis (GenSA) visiting kernel against the SOTA reference
//! implementation, SciPy `dual_annealing`'s `VisitingDistribution.visit_fn`
//! (the Schuur/Xiang transform).
//!
//! The kernel draws each coordinate independently as
//! `dx = sigma(T, q_v) * x / |y|^{(q_v-1)/(3-q_v)}`, `x, y ~ N(0,1)`. The
//! reference statistics below were generated from SciPy's `visit_fn` and an
//! identical NumPy reimplementation (KS two-sample stat <= 0.005, p >= 0.2 for
//! q_v in {1.5, 2.0, 2.62, 2.9}); these tests pin the Rust port (including the
//! Lanczos `gamma_fn` used for the normalization constant) to those references.

use anneal_core::movekernel::{MoveKernel, TsallisVisit};
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Single-coordinate increments from `propose` at the origin (dim 1).
fn increments(q_v: f64, temp: f64, n: usize, seed: u64) -> Vec<f64> {
    let k = TsallisVisit::new(q_v);
    let x = array![0.0_f64];
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| k.propose(x.view(), temp, &mut rng)[0])
        .collect()
}

fn median_abs(samples: &[f64]) -> f64 {
    let mut a: Vec<f64> = samples.iter().map(|v| v.abs()).collect();
    a.sort_by(|x, y| x.total_cmp(y));
    a[a.len() / 2]
}

fn fraction_within(samples: &[f64], bound: f64) -> f64 {
    samples.iter().filter(|&&v| v.abs() <= bound).count() as f64 / samples.len() as f64
}

#[test]
fn qv_two_is_standard_cauchy() {
    // q_v = 2 -> sigma = T and the Schuur transform is x/|y| = standard Cauchy
    // at T = 1. P(|X| <= 1) = 0.5 for Cauchy(0, 1).
    let s = increments(2.0, 1.0, 200_000, 11);
    let f = fraction_within(&s, 1.0);
    assert!((f - 0.5).abs() < 0.02, "q_v=2 not Cauchy: P(|X|<=1) = {f:.4}");
}

#[test]
fn temperature_scales_qv2() {
    // q_v = 2, T = 4 -> sigma = 4, so increments are 4 * Cauchy(0,1):
    // P(|X| <= 4) = 0.5, P(|X| <= 1) well under half.
    let s = increments(2.0, 4.0, 200_000, 12);
    assert!(
        (fraction_within(&s, 4.0) - 0.5).abs() < 0.02,
        "T-scaling broken at q_v=2"
    );
    assert!(fraction_within(&s, 1.0) < 0.30, "width did not scale with T");
}

#[test]
fn matches_scipy_reference_qv262_default() {
    // GenSA / SciPy dual_annealing default q_v = 2.62, T = 1. Reference from
    // SciPy visit_fn: median|x| ~ 15.84, P(|x| <= 1) ~ 0.211. This pins the
    // Lanczos gamma_fn normalization and the Schuur exponent.
    let s = increments(2.62, 1.0, 1_000_000, 13);
    let med = median_abs(&s);
    let p1 = fraction_within(&s, 1.0);
    assert!(
        (med / 15.84 - 1.0).abs() < 0.08,
        "q_v=2.62 median|x| = {med:.3} (scipy ref 15.84)"
    );
    assert!(
        (p1 - 0.211).abs() < 0.01,
        "q_v=2.62 P(|x|<=1) = {p1:.4} (scipy ref 0.211)"
    );
}

#[test]
fn matches_scipy_reference_qv15() {
    // q_v = 1.5, T = 1. SciPy reference median|x| ~ 0.394.
    let s = increments(1.5, 1.0, 1_000_000, 14);
    let med = median_abs(&s);
    assert!(
        (med / 0.394 - 1.0).abs() < 0.05,
        "q_v=1.5 median|x| = {med:.4} (scipy ref 0.394)"
    );
}
