use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

struct ScalarBox {
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}

impl Objective<f64> for ScalarBox {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), self.bounds.dims);
        assert!(x.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
        self.calls.fetch_add(1, Ordering::Relaxed);
        0.0
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

fn run(dim: usize, budget: usize, history: HistoryMode) {
    let objective = ScalarBox {
        bounds: Bounds::new(
            Array1::from_elem(dim, -1.0),
            Array1::from_elem(dim, 1.0),
            0.0,
        ),
        calls: AtomicUsize::new(0),
    };
    let start = Array1::zeros(dim);
    let config = BoxEnsembleConfig {
        replicas: 4,
        budget,
        history,
        ..BoxEnsembleConfig::default()
    };
    let coverage = BoxCoverageConfig {
        shared: true,
        ..BoxCoverageConfig::default()
    };
    let result = box_values_ensemble_optimize_with_coverage(
        &objective,
        7,
        Some(start.view()),
        &config,
        &coverage,
    );
    assert_eq!(result.n_evals, objective.calls.load(Ordering::Relaxed));
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.best_pos, start);
    assert_eq!(
        result.n_evals, budget,
        "dim={dim}, history={history:?}: unfunded full quenches must not suppress funded scalar proposals"
    );
    assert!(result.coverage.published_samples > 0);
}

#[test]
fn scalar_search_uses_terminal_work_without_requiring_a_full_quench() {
    for dim in [2, 128] {
        for budget in [240, 241] {
            for history in [HistoryMode::None, HistoryMode::Private, HistoryMode::Shared] {
                run(dim, budget, history);
            }
        }
    }
}

#[test]
fn a_partial_initial_round_never_exceeds_the_aggregate_budget() {
    run(2, 3, HistoryMode::None);
}
