//! A scalar client implements Objective, with no gradient trait or placeholder.

use std::sync::Mutex;

use anneal_core::methods::{PortfolioEnsembleConfig, portfolio_values_ensemble_optimize};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

struct ScalarLoss {
    bounds: Bounds<f64>,
    measured: Mutex<Vec<f64>>,
}

impl Objective<f64> for ScalarLoss {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert!(self.bounds.contains(x));
        let value = x.iter().map(|v| (v - 0.371).powi(2)).sum();
        self.measured.lock().unwrap().push(value);
        value
    }
}

fn scalar_client(replicas: usize) {
    let loss = ScalarLoss {
        bounds: Bounds::new(Array1::from_elem(5, -2.0), Array1::from_elem(5, 2.0), 0.0),
        measured: Mutex::new(Vec::new()),
    };
    let mut config = PortfolioEnsembleConfig {
        replicas,
        budget: 2_003,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.radius = 0.8;
    let result = portfolio_values_ensemble_optimize(&loss, 17, None, &config);
    let values = loss.measured.into_inner().unwrap();
    assert_eq!(result.n_evals, values.len());
    assert_eq!(result.n_evals, config.budget);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.replicas.len(), replicas);
    assert!(result.replicas.iter().all(|replica| replica.n_grads == 0));
    assert_eq!(result.best_val, values.into_iter().fold(f64::INFINITY, f64::min));
    if replicas > 1 {
        assert!(result.coverage.applied_foreign_samples > 0);
        assert!(result.coverage.repelled_proposals > 0);
    } else {
        assert_eq!(result.coverage.published_samples, 0);
    }
}

#[test]
fn scalar_single_portfolio_needs_only_objective() {
    scalar_client(1);
}

#[test]
fn scalar_communicating_portfolios_need_only_objective() {
    scalar_client(4);
}
