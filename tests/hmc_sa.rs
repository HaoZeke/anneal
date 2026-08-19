//! Witness for HMC-driven SA inside the typed algebra. Same `run_rs`
//! driver, different sampler.

use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::hmc::{HmcIntegrator, HmcSaSampler, LeapfrogIntegrator, OmelyanIntegrator};
use anneal_core::run_rs;
use anneal_core::sampler::Sampler;
use eindir_core::{AnalyticGradient, FiniteDiffGradient, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::objectives::StybTang2D;

struct CountingZeroGradient {
    calls: AtomicUsize,
    dim: usize,
}

impl CountingZeroGradient {
    fn new(dim: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            dim,
        }
    }
}

impl eindir_core::Gradient<f64> for CountingZeroGradient {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Array1::zeros(x.len())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

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
    let integrator = OmelyanIntegrator::new(0.05, 5, 5.0);
    let sampler = HmcSaSampler::new(obj, grad, cool.clone(), integrator);
    let history = run_rs(sampler, &cool, 50, 50, 42);
    assert!(
        history.best.val < 0.0,
        "HMC-SA should find a negative value on StybTang2D; got {}",
        history.best.val
    );
}

#[test]
fn analytic_gradient_matches_finite_diff_on_styb_tang() {
    // Analytic d/dx (x^4 - 16x^2 + 5x)/2 = (4x^3 - 32x + 5)/2 per coord.
    let analytic = AnalyticGradient::new(2, |x: ndarray::ArrayView1<f64>| {
        Array1::from_iter(
            x.iter()
                .map(|&xi| (4.0 * xi.powi(3) - 32.0 * xi + 5.0) / 2.0),
        )
    });
    let fd = FiniteDiffGradient::new(eindir_core::objectives::StybTang2D::new());
    let test_pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [-2.9, -2.9], [1.5, -3.0]];
    for pt in test_pts {
        let av = analytic.grad(ndarray::ArrayView1::from(&pt));
        let fv = fd.grad(ndarray::ArrayView1::from(&pt));
        for i in 0..2 {
            assert!(
                (av[i] - fv[i]).abs() < 1e-3,
                "analytic {} vs FD {} at {:?}",
                av[i],
                fv[i],
                pt
            );
        }
    }
}

#[test]
fn hmc_sa_acceptance_in_unit_interval() {
    use anneal_core::history::State;

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

#[test]
fn hmc_sampler_can_start_from_supplied_position() {
    let obj = StybTang2D::new();
    let grad = FiniteDiffGradient::new(StybTang2D::new());
    let cool = LogCool::new(5.0_f64, 2.0);
    let integrator = OmelyanIntegrator::new(0.05, 5, 5.0);
    let x0 = Array1::from_vec(vec![-2.903534, -2.903534]);
    let sampler = HmcSaSampler::new(obj, grad, cool, integrator).with_initial_pos(x0.clone());
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);

    let state = sampler.initial_state(&mut rng);

    assert_eq!(state.cur.pos, x0);
    assert!(state.cur.val < -78.0);
}

#[test]
fn hmc_sampler_clips_an_out_of_bounds_initial_position() {
    let obj = StybTang2D::new();
    let grad = FiniteDiffGradient::new(StybTang2D::new());
    let cool = LogCool::new(5.0_f64, 2.0);
    let integrator = OmelyanIntegrator::new(0.05, 5, 5.0);
    let sampler = HmcSaSampler::new(obj, grad, cool, integrator)
        .with_initial_pos(Array1::from_vec(vec![9.0, -9.0]));
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);

    let state = sampler.initial_state(&mut rng);

    assert_eq!(state.cur.pos.to_vec(), vec![5.0, -5.0]);
}

#[test]
fn omelyan_integrator_uses_three_force_stages_per_step() {
    let grad = CountingZeroGradient::new(2);
    let integrator = OmelyanIntegrator::new(0.01, 1, 1.0);
    let x0 = Array1::zeros(2);
    let p0 = Array1::zeros(2);
    let objective = |_: &Array1<f64>| 0.0;
    let momentum = anneal_core::hmc::GaussianMomentum;

    let result = integrator.evolve(x0, p0, 0.0, 1.0, &grad, &momentum, &objective);

    assert!(!result.diverged);
    assert_eq!(grad.calls.load(Ordering::SeqCst), 3);
}
