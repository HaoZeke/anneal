//! Witness for Method B Phase 1: HMC-driven SA inside the typed
//! algebra. Same `run_rs` driver, different sampler.

use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::grad::{FiniteDiffGradient, Gradient};
use anneal_core::hmc::{HmcSaSampler, LeapfrogIntegrator};
use anneal_core::run_rs;

use eindir_core::objectives::StybTang2D;

#[test]
fn finite_diff_gradient_matches_analytic_at_zero() {
    // Styblinski-Tang's analytic gradient at x = (0, 0) is
    // f'(x) = 4x^3 - 32x + 5 = 5 per coordinate; whole gradient = (5, 5).
    let g = FiniteDiffGradient::new(StybTang2D::new());
    let grad = g.grad(ndarray::ArrayView1::from(&[0.0, 0.0]));
    // Gradient of f / 2 (the StybTang2D normalisation) = 0.5*(5, 5) = (2.5, 2.5).
    // The crate's StybTang2D includes the /2 prefactor -- check the actual eval-difference.
    for &g_i in grad.iter() {
        let g: f64 = g_i;
        assert!(g.is_finite());
        assert!((g - 2.5).abs() < 1e-3, "FD gradient {} far from 2.5", g);
    }
}

#[test]
fn hmc_sa_finds_negative_minimum() {
    let obj = StybTang2D::new();
    let grad = FiniteDiffGradient::new(StybTang2D::new());
    let cool = LogCool::new(5.0_f64, 2.0);
    let integrator = LeapfrogIntegrator::new(0.05, 5, 5.0);
    let sampler = HmcSaSampler::new(obj, grad, cool.clone(), integrator);
    let history = run_rs(sampler, &cool, 50, 50, 42);
    assert!(
        history.best.val < 0.0,
        "HMC-SA should find a negative value on StybTang2D; got {}",
        history.best.val
    );
}

#[test]
fn hmc_sa_acceptance_in_unit_interval() {
    use anneal_core::history::State;
    use anneal_core::sampler::Sampler;
    use rand::SeedableRng;

    let obj = StybTang2D::new();
    let grad = FiniteDiffGradient::new(StybTang2D::new());
    let cool = LogCool::new(5.0_f64, 2.0);
    let integrator = LeapfrogIntegrator::new(0.05, 5, 5.0);
    let sampler = HmcSaSampler::new(obj, grad, cool, integrator);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut state: State = sampler.initial_state(&mut rng);

    let mut accepted = 0;
    let mut total = 0;
    for epoch in 0..10 {
        for _ in 0..50 {
            if sampler.step(&mut state, epoch, &mut rng) {
                accepted += 1;
            }
            total += 1;
        }
    }
    let rate = accepted as f64 / total as f64;
    assert!(
        rate > 0.0 && rate <= 1.0,
        "HMC-SA acceptance rate {} out of [0, 1]",
        rate
    );
    // Suppress unused-import warning when Metropolis is not directly named.
    let _ = std::any::type_name::<Metropolis>();
}
