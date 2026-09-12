use std::sync::Mutex;

use anneal_core::methods::box_hopping::{BoxCoverageConfig, BoxEnsembleConfig, box_values_ensemble_optimize_with_coverage};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::portfolio::{PortfolioEnsembleConfig, portfolio_values_ensemble_optimize};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

struct ScalarLoss {
    bounds: Bounds<f64>,
    samples: Mutex<Vec<Vec<u64>>>,
}

impl ScalarLoss {
    fn new(dim: usize) -> Self {
        Self { bounds: Bounds::new(Array1::from_elem(dim, -2.0), Array1::from_elem(dim, 2.0), 0.0), samples: Mutex::new(Vec::new()) }
    }
    fn trace(&self) -> Vec<Vec<u64>> {
        let mut result = self.samples.lock().unwrap().clone();
        result.sort();
        result
    }
}

impl Objective<f64> for ScalarLoss {
    fn dim(&self) -> usize { self.bounds.dims }
    fn bounds(&self) -> &Bounds<f64> { &self.bounds }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), self.bounds.dims);
        assert!(x.iter().all(|v| v.is_finite() && (-2.0..=2.0).contains(v)));
        self.samples.lock().unwrap().push(x.iter().map(|v| v.to_bits()).collect());
        1.0
    }
}

#[test]
fn scalar_portfolio_delivers_only_funded_ring_neighbors() {
    for (replicas, neighbors, delivered) in [(2, 1, 2), (3, 1, 6), (5, 0, 20), (5, 1, 10), (5, 2, 20), (5, usize::MAX, 20)] {
        let loss = ScalarLoss::new(8);
        let config = PortfolioEnsembleConfig {
            replicas, budget: 2 * replicas,
            coverage: BoxCoverageConfig { radius: 0.8, neighbors, ..Default::default() },
            ..Default::default()
        };
        let result = portfolio_values_ensemble_optimize(&loss, 17, None, &config);
        assert_eq!(result.n_evals, loss.trace().len());
        assert_eq!(result.n_evals, 2 * replicas);
        assert_eq!(result.n_grads, 0);
        assert_eq!(result.best_val, 1.0);
        assert_eq!(result.coverage.published_samples, (2 * replicas) as u64);
        assert_eq!(result.coverage.applied_foreign_samples, delivered);
        assert_eq!(result.coverage.repelled_proposals, replicas);
    }
}

#[test]
fn scalar_hops_restrict_samples_and_deposits_with_no_minimum_history() {
    for (neighbors, delivered) in [(0, 30), (1, 15), (2, 30), (usize::MAX, 30)] {
        let loss = ScalarLoss::new(1);
        let config = BoxEnsembleConfig { replicas: 5, budget: 30, history: HistoryMode::None, shared_deposits: usize::MAX, ..Default::default() };
        let coverage = BoxCoverageConfig { neighbors, radius: 1e-12, ..Default::default() };
        let result = box_values_ensemble_optimize_with_coverage(&loss, 17, None, &config, &coverage);
        assert_eq!(result.n_evals, loss.trace().len());
        assert_eq!(result.n_evals, 30);
        assert_eq!(result.n_grads, 0);
        assert_eq!(result.best_val, 1.0);
        assert_eq!(result.hops, 5);
        assert_eq!(result.history_observations, 0);
        assert_eq!(result.history_minima, 0);
        assert_eq!(result.coverage.published_samples, 10);
        assert_eq!(result.coverage.published_visits, 10);
        assert_eq!(result.coverage.applied_foreign_samples, delivered);
        assert_eq!(result.coverage.applied_foreign_visits, delivered);
        assert_eq!(result.coverage.capped_foreign_visits, 0);
    }
}

#[test]
fn private_and_zero_weight_portfolios_preserve_paid_positions_for_any_graph() {
    let mut expected = None;
    for (shared, peer_weight) in [(false, 1.0), (true, 0.0)] {
        for neighbors in [0, 1, usize::MAX] {
            let loss = ScalarLoss::new(8);
            let config = PortfolioEnsembleConfig {
                replicas: 5, budget: 200,
                coverage: BoxCoverageConfig { shared, peer_weight, neighbors, ..Default::default() },
                ..Default::default()
            };
            let result = portfolio_values_ensemble_optimize(&loss, 17, None, &config);
            let trace = loss.trace();
            assert_eq!(result.n_evals, trace.len());
            assert_eq!(result.n_evals, 200);
            assert_eq!(result.n_grads, 0);
            assert_eq!(result.coverage.published_samples, 0);
            assert_eq!(result.coverage.applied_foreign_samples, 0);
            assert_eq!(result.coverage.repelled_proposals, 0);
            if let Some(expected) = &expected { assert_eq!(&trace, expected); } else { expected = Some(trace); }
        }
    }
}
