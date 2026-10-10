//! Scalar warmup funds the active operators before learning their allocation.

use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::{PortfolioEnsembleConfig, portfolio_values_ensemble_optimize};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

struct ScalarPlateau {
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}

impl Objective<f64> for ScalarPlateau {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, position: ArrayView1<f64>) -> f64 {
        assert!(self.bounds.contains(position));
        self.calls.fetch_add(1, Ordering::Relaxed);
        1.0
    }
}

#[test]
fn scalar_replicas_fund_each_active_warmup_arm() {
    for seed in [7, 17] {
        for shared in [false, true] {
            let objective = ScalarPlateau {
                bounds: Bounds::new(
                    Array1::from_elem(128, -5.12),
                    Array1::from_elem(128, 5.12),
                    0.0,
                ),
                calls: AtomicUsize::new(0),
            };
            let mut config = PortfolioEnsembleConfig {
                replicas: 4,
                budget: 8_000,
                ..PortfolioEnsembleConfig::default()
            };
            config.coverage.shared = shared;
            let initial = Array1::from_elem(128, 0.1875);
            let result =
                portfolio_values_ensemble_optimize(&objective, seed, Some(initial.view()), &config);
            assert_eq!(result.n_evals, config.budget);
            assert_eq!(result.n_evals, objective.calls.load(Ordering::Relaxed));
            assert_eq!(result.n_grads, 0);
            assert_eq!(result.best_val, 1.0);
            for (replica, search) in result.replicas.iter().enumerate() {
                assert!(search.arm_stats.len() >= 4);
                for arm in &search.arm_stats {
                    assert!(
                        arm.pulls > 0,
                        "seed={seed}, shared={shared}, replica={replica}: active {} must receive its warmup allocation",
                        arm.name
                    );
                }
            }
        }
    }
}
