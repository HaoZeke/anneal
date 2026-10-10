use std::sync::Mutex;

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

struct FlatBox {
    bounds: Bounds<f64>,
    evaluations: Mutex<Vec<Array1<f64>>>,
}

impl Objective<f64> for FlatBox {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 8);
        assert!(x.iter().all(|x| x.is_finite() && (-1.0..=1.0).contains(x)));
        self.evaluations.lock().unwrap().push(x.to_owned());
        0.0
    }

    fn dim(&self) -> usize {
        8
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

fn run(history: HistoryMode, shared: bool) {
    let objective = FlatBox {
        bounds: Bounds::new(Array1::from_elem(8, -1.0), Array1::from_elem(8, 1.0), 0.0),
        evaluations: Mutex::new(Vec::new()),
    };
    let config = BoxEnsembleConfig {
        replicas: 4,
        budget: 512,
        history,
        ..BoxEnsembleConfig::default()
    };
    let coverage = BoxCoverageConfig {
        shared,
        height: 0.0,
        ..BoxCoverageConfig::default()
    };
    let start = Array1::zeros(8);
    let result = box_values_ensemble_optimize_with_coverage(
        &objective,
        71,
        Some(start.view()),
        &config,
        &coverage,
    );
    let evaluations = objective.evaluations.lock().unwrap();
    assert_eq!(result.n_evals, evaluations.len());
    assert_eq!(result.n_grads, 0);
    assert!(result.n_evals <= config.budget);
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.best_pos, start);
    let certificate_probes = evaluations
        .iter()
        .filter(|point| {
            let maximum = point.iter().map(|value| value.abs()).fold(0.0, f64::max);
            maximum > 0.0 && maximum <= 1e-5
        })
        .count();
    if matches!(history, HistoryMode::None) {
        // This allowance funds complete pattern-search sweeps. Their smallest
        // displacement is 0.05, distinct from the certificate's 1e-6 probes.
        assert_eq!(
            certificate_probes, 0,
            "disabled history consumes certificate probes"
        );
        assert!(
            result.hops >= 2 * config.replicas,
            "certificate reservations waste funded hops"
        );
        assert_eq!(result.history_observations, 0);
        assert_eq!(result.history_minima, 0);
        assert_eq!(result.history_cost, (0, 0, 0.0));
    } else {
        assert!(certificate_probes >= 8);
        assert!(result.history_observations >= config.replicas);
        assert!(result.history_minima > 0);
    }
    assert_eq!(
        result.coverage.local_observations,
        config.replicas + result.hops
    );
    if shared {
        assert!(result.coverage.applied_foreign_visits > 0);
    } else {
        assert_eq!(result.coverage.applied_foreign_visits, 0);
    }
}

#[test]
fn coverage_only_search_does_not_fund_unused_minimum_certificates() {
    run(HistoryMode::None, true);
    run(HistoryMode::None, false);
}

#[test]
fn requested_minimum_history_keeps_its_certificate_probes() {
    run(HistoryMode::Private, false);
    run(HistoryMode::Shared, true);
}
