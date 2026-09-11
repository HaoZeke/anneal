use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::box_hopping::{BoxEnsembleConfig, box_ensemble_optimize};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

struct ConditionedQuadratic {
    bounds: Bounds<f64>,
    evaluations: AtomicUsize,
    gradients: AtomicUsize,
}

impl ConditionedQuadratic {
    fn curvature(axis: usize) -> f64 {
        1000.0_f64.powf(axis as f64 / 7.0)
    }
}

impl Objective<f64> for ConditionedQuadratic {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        assert_eq!(x.len(), 8);
        assert!(x.iter().all(|v| v.is_finite() && (-5.12..=5.12).contains(v)));
        x.iter().enumerate().map(|(j, v)| 0.5 * Self::curvature(j) * v * v).sum()
    }

    fn dim(&self) -> usize { 8 }

    fn bounds(&self) -> &Bounds<f64> { &self.bounds }
}

impl Gradient<f64> for ConditionedQuadratic {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        Array1::from_iter(x.iter().enumerate().map(|(j, v)| Self::curvature(j) * v))
    }

    fn dim(&self) -> usize { 8 }
}

#[test]
fn funded_box_quenches_supply_a_certified_shared_minimum() {
    let objective = ConditionedQuadratic {
        bounds: Bounds::new(Array1::from_elem(8, -5.12), Array1::from_elem(8, 5.12), 0.0),
        evaluations: AtomicUsize::new(0), gradients: AtomicUsize::new(0),
    };
    let config = BoxEnsembleConfig {
        replicas: 4, budget: 8_000, history: HistoryMode::Shared,
        ..BoxEnsembleConfig::default()
    };
    let result = box_ensemble_optimize(
        &objective, &objective, 0, Some(Array1::from_elem(8, 2.5).view()), &config,
    );
    let counts = (
        objective.evaluations.load(Ordering::Relaxed),
        objective.gradients.load(Ordering::Relaxed),
    );
    assert_eq!((result.n_evals, result.n_grads), counts);
    assert!(counts.0 + counts.1 <= config.budget);
    assert!(result.hops > 0);
    assert_eq!(result.history_minima, 1, "the convex basin must enter the shared book");
    assert!(result.history_observations >= config.replicas);
}
