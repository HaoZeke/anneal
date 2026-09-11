use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEscape, GleEscapeConfig,
    box_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::gle_langevin::GleNoise;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};

struct Rastrigin {
    bounds: Bounds<f64>,
    evaluations: AtomicUsize,
    gradients: AtomicUsize,
}

impl Rastrigin {
    fn value(x: ArrayView1<f64>) -> f64 {
        x.iter()
            .map(|x| x * x + 10.0 * (1.0 - (std::f64::consts::TAU * x).cos()))
            .sum()
    }
}

impl Objective<f64> for Rastrigin {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        assert_eq!(x.len(), 8);
        assert!(x.iter().all(|x| x.is_finite() && (-5.12..=5.12).contains(x)));
        Self::value(x)
    }

    fn dim(&self) -> usize { 8 }
    fn bounds(&self) -> &Bounds<f64> { &self.bounds }
}

impl Gradient<f64> for Rastrigin {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        x.mapv(|x| 2.0 * x + 10.0 * std::f64::consts::TAU * (std::f64::consts::TAU * x).sin())
    }

    fn dim(&self) -> usize { 8 }
}

fn excursion_contract(noise: GleNoise) {
    let mut sum = 0.0;
    for seed in 0..4 {
        let objective = Rastrigin {
            bounds: Bounds::new(Array1::from_elem(8, -5.12), Array1::from_elem(8, 5.12), 0.0),
            evaluations: AtomicUsize::new(0),
            gradients: AtomicUsize::new(0),
        };
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5354_4152_545f_424f);
        let start = Array1::from_shape_fn(8, |_| -5.12 + 10.24 * rng.random::<f64>());
        let config = BoxEnsembleConfig {
            replicas: 4,
            budget: 8000,
            history: HistoryMode::None,
            escape: BoxEscape::Langevin(GleEscapeConfig { noise, ..GleEscapeConfig::default() }),
            ..BoxEnsembleConfig::default()
        };
        let coverage = BoxCoverageConfig { shared: false, ..BoxCoverageConfig::default() };
        let result = box_ensemble_optimize_with_coverage(
            &objective, &objective, seed, Some(start.view()), &config, &coverage,
        );
        assert_eq!(result.n_evals, objective.evaluations.load(Ordering::Relaxed));
        assert_eq!(result.n_grads, objective.gradients.load(Ordering::Relaxed));
        assert!(result.n_evals + result.n_grads <= config.budget);
        assert_eq!(result.history_observations, 0);
        assert_eq!(result.best_val, Rastrigin::value(result.best_pos.view()));
        sum += result.best_val;
    }
    let mean = sum / 4.0;
    assert!(mean <= 20.0, "Langevin exploration freezes on relaxed endpoints: mean {mean}");
}

#[test]
fn colored_noise_explores_beyond_the_initial_relaxed_regions() {
    excursion_contract(GleNoise::Colored);
}

#[test]
fn white_noise_explores_beyond_the_initial_relaxed_regions() {
    excursion_contract(GleNoise::White { friction: 4.0 });
}
