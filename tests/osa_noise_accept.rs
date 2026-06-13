//! Integration tests for the noise-aware OSA acceptance component
//! (`noise_accept::OsaAccept`), mirroring the checks in the reference
//! `experiments/osa.py::_self_test`.
//!
//! OSA is the sequential acceptance rule of Ball, Branke & Meisel (2018) for
//! simulated annealing under noisy cost differences. Two properties pin it:
//!
//! 1. Detailed balance: the steady-state acceptance ratio `PA(+Delta) /
//!    PA(-Delta)` equals the Metropolis factor `exp(-beta Delta)`.
//! 2. Noise-free Metropolis limit: as the noise scale shrinks, the first
//!    sample decides, so `PA(Delta) -> min(1, exp(-beta Delta))`.

use anneal_core::noise_accept::OsaAccept;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn osa_detailed_balance_matches_metropolis_factor() {
    let osa = OsaAccept::new();
    let temp = 1.0;
    let sigma = 0.5;
    let beta = 1.0 / temp;
    let trials = 40_000;

    let mut worst = 0.0_f64;
    for &delta in &[0.25_f64, 0.5, 1.0] {
        let mut rng_pos = StdRng::seed_from_u64(1);
        let mut rng_neg = StdRng::seed_from_u64(2);
        let (pa_pos, _) = osa.acceptance_rate(delta, temp, sigma, trials, &mut rng_pos);
        let (pa_neg, _) = osa.acceptance_rate(-delta, temp, sigma, trials, &mut rng_neg);
        let ratio = pa_pos / pa_neg;
        let target = (-beta * delta).exp();
        let rel = (ratio - target).abs() / target;
        worst = worst.max(rel);
    }
    assert!(
        worst < 0.10,
        "OSA detailed balance violated: worst relative error {worst:.4}"
    );
}

#[test]
fn osa_noise_free_limit_is_metropolis() {
    let osa = OsaAccept::new();
    // Tiny noise: the n = 1 rule is min(1, exp(-beta Delta)) because c* = 0
    // accepts or rejects on the first sample.
    let mut rng = StdRng::seed_from_u64(3);
    let (pa_uphill, _) = osa.acceptance_rate(1.0, 1.0, 1e-3, 20_000, &mut rng);
    let metropolis = (-1.0_f64).exp();
    assert!(
        (pa_uphill - metropolis).abs() < 0.03,
        "OSA Metropolis limit off: PA(1) = {pa_uphill:.4}, exp(-1) = {metropolis:.4}"
    );
}

#[test]
fn osa_downhill_accepts_readily() {
    // A clearly downhill true difference accepts with high probability and few
    // samples.
    let osa = OsaAccept::new();
    let mut rng = StdRng::seed_from_u64(4);
    let (pa, mean_n) = osa.acceptance_rate(-2.0, 1.0, 0.5, 20_000, &mut rng);
    assert!(pa > 0.9, "downhill acceptance too low: {pa:.4}");
    assert!(mean_n < 5.0, "downhill decision too slow: <n> = {mean_n:.2}");
}
