use std::sync::Mutex;

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::portfolio::values_local_polish;
use eindir_core::{Bounds, Objective};
use ndarray::{ArrayView1, array};

struct CorrelatedLoss {
    bounds: Bounds<f64>,
    observed: Mutex<Vec<(Vec<f64>, f64)>>,
}

impl CorrelatedLoss {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-2.0, -2.0], array![2.0, 2.0], 0.0),
            observed: Mutex::new(Vec::new()),
        }
    }
}

impl Objective<f64> for CorrelatedLoss {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 2);
        assert!(self.bounds.contains(x));
        let a = x[0] - 0.37;
        let b = x[1] + 0.61;
        let value = (a + b).powi(2) + 100.0 * (a - b).powi(2);
        self.observed.lock().unwrap().push((x.to_vec(), value));
        value
    }

    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

#[test]
fn communicating_values_chains_use_the_common_scalar_refiner() {
    let local = CorrelatedLoss::new();
    let start = array![1.4, 1.3];
    let local_result = values_local_polish(&local, start.clone(), 24, 0.1, 1e-12);
    let local_trace = local.observed.lock().unwrap().clone();
    assert_eq!(local_result.n_evals, local_trace.len());
    assert!(local_trace.len() > 1);
    let ensemble = CorrelatedLoss::new();
    let config = BoxEnsembleConfig {
        replicas: 4,
        budget: 160,
        history: HistoryMode::None,
        ..BoxEnsembleConfig::default()
    };
    let coverage = BoxCoverageConfig {
        shared: true,
        ..BoxCoverageConfig::default()
    };
    let result = box_values_ensemble_optimize_with_coverage(
        &ensemble,
        7,
        Some(start.view()),
        &config,
        &coverage,
    );
    let observed = ensemble.observed.lock().unwrap();
    assert_eq!(&observed[..local_trace.len()], local_trace.as_slice());
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.n_evals, observed.len());
    assert_eq!(result.n_evals, config.budget);
    assert_eq!(result.history_observations, 0);
    assert!(result.coverage.applied_foreign_samples > 0);
    let best = observed.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    assert_eq!(result.best_val, best.1);
    assert_eq!(result.best_pos.to_vec(), best.0);
}
