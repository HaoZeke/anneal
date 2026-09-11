use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::box_hopping::{
    BoxEnsembleConfig, box_ensemble_optimize, box_values_ensemble_optimize,
};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

struct Quadratic {
    bounds: Bounds<f64>,
    evaluations: AtomicUsize,
    gradients: AtomicUsize,
}

impl Quadratic {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(8, -5.12), Array1::from_elem(8, 5.12), 0.0),
            evaluations: AtomicUsize::new(0),
            gradients: AtomicUsize::new(0),
        }
    }

    fn curvature(axis: usize) -> f64 {
        1000.0_f64.powf(axis as f64 / 7.0)
    }
}

impl Objective<f64> for Quadratic {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        assert_eq!(x.len(), 8);
        assert!(
            x.iter()
                .all(|v| v.is_finite() && (-5.12..=5.12).contains(v))
        );
        x.iter()
            .enumerate()
            .map(|(axis, value)| 0.5 * Self::curvature(axis) * value * value)
            .sum()
    }

    fn dim(&self) -> usize {
        8
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

impl Gradient<f64> for Quadratic {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        Array1::from_iter(
            x.iter()
                .enumerate()
                .map(|(axis, value)| Self::curvature(axis) * value),
        )
    }

    fn dim(&self) -> usize {
        8
    }
}

fn check_coverage(gradient: bool, history: HistoryMode) {
    let objective = Quadratic::new();
    let start = Array1::from_elem(8, 2.5);
    let config = BoxEnsembleConfig {
        replicas: 4,
        budget: if gradient { 256 } else { 512 },
        history,
        ..BoxEnsembleConfig::default()
    };
    let result = if gradient {
        box_ensemble_optimize(&objective, &objective, 0, Some(start.view()), &config)
    } else {
        box_values_ensemble_optimize(&objective, 0, Some(start.view()), &config)
    };
    let actual = (
        objective.evaluations.load(Ordering::Relaxed),
        objective.gradients.load(Ordering::Relaxed),
    );
    assert_eq!((result.n_evals, result.n_grads), actual);
    assert!(actual.0 + actual.1 <= config.budget);
    assert!(result.hops >= config.replicas);
    assert_eq!(
        result.history_observations, 0,
        "no quench reaches the certificate"
    );
    assert_eq!(result.history_minima, 0);
    if matches!(history, HistoryMode::Shared) {
        assert!(
            result.shared_deposits > 0,
            "paid feasible coverage must be shared without a minimum certificate"
        );
    } else {
        assert_eq!(result.shared_deposits, 0, "private coverage stays private");
    }
}

#[test]
fn gradient_chains_share_coverage_without_certified_minima() {
    check_coverage(true, HistoryMode::Shared);
}

#[test]
fn values_only_chains_share_coverage_without_certified_minima() {
    check_coverage(false, HistoryMode::Shared);
}

#[test]
fn private_chains_do_not_receive_coverage_from_other_replicas() {
    check_coverage(true, HistoryMode::Private);
    check_coverage(false, HistoryMode::Private);
}
