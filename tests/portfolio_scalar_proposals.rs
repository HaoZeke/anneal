//! Scalar global candidates participate in the position channel without forces.

use std::sync::Mutex;

use anneal_core::{PortfolioEnsembleConfig, portfolio_values_ensemble_optimize};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

struct ScalarSamples {
    bounds: Bounds<f64>,
    positions: Mutex<Vec<Vec<u64>>>,
}

impl ScalarSamples {
    fn new(dim: usize) -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_elem(dim, -2.0),
                Array1::from_elem(dim, 2.0),
                0.0,
            ),
            positions: Mutex::new(Vec::new()),
        }
    }

    fn positions(&self) -> Vec<Vec<u64>> {
        let mut positions = self.positions.lock().unwrap().clone();
        positions.sort();
        positions
    }
}

impl Objective<f64> for ScalarSamples {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert!(self.bounds.contains(x));
        self.positions
            .lock()
            .unwrap()
            .push(x.iter().map(|v| v.to_bits()).collect());
        1.0
    }
}

#[test]
fn sub_stencil_allowances_exchange_and_prepare_their_global_candidates() {
    for (replicas, budget) in [(2, 10), (4, 13)] {
        let objective = ScalarSamples::new(8);
        let mut config = PortfolioEnsembleConfig {
            replicas,
            budget,
            ..PortfolioEnsembleConfig::default()
        };
        config.coverage.radius = 0.4;
        let result = portfolio_values_ensemble_optimize(&objective, 17, None, &config);
        assert_eq!(objective.positions().len(), budget);
        assert_eq!(result.n_evals, budget);
        assert_eq!(result.n_grads, 0);
        assert_eq!(result.best_val, 1.0);
        assert!(
            result
                .replicas
                .iter()
                .all(|replica| replica.arm_stats.iter().all(|arm| arm.pulls == 0))
        );
        // One initial paid position per chain, followed entirely by global draws.
        assert_eq!(result.coverage.published_samples, budget as u64);
        assert_eq!(result.coverage.sample_peer_checks, budget - replicas);
        assert!(result.coverage.applied_foreign_samples >= replicas);
        assert!(result.coverage.repelled_proposals > 0);
    }
}

#[test]
fn scalar_allowances_below_a_complete_start_fund_global_positions() {
    for replicas in [1, 4] {
        for allowance in [8, 9, 16, 32, 33] {
            for shared in [false, true] {
                let objective = ScalarSamples::new(16);
                let mut config = PortfolioEnsembleConfig {
                    replicas,
                    budget: replicas * allowance,
                    ..PortfolioEnsembleConfig::default()
                };
                config.coverage.shared = shared;
                config.coverage.radius = 1e-12;
                let result = portfolio_values_ensemble_optimize(&objective, 17, None, &config);
                let positions = objective.positions();
                assert_eq!(positions.len(), config.budget);
                assert_eq!(result.n_evals, config.budget);
                assert_eq!(result.n_grads, 0);
                assert_eq!(result.best_val, 1.0);
                let global_positions = positions
                    .iter()
                    .filter(|position| {
                        position
                            .iter()
                            .any(|&bits| f64::from_bits(bits).abs() > 0.01)
                    })
                    .count();
                assert_eq!(
                    global_positions,
                    config.budget - replicas,
                    "replicas={replicas}, allowance={allowance}, shared={shared}: an unfunded local start must leave paid work for global draws"
                );
            }
        }
    }
}

#[test]
fn sub_stencil_scalar_work_above_the_legacy_floor_reaches_the_sample_channel() {
    for allowance in [8, 9, 16, 32, 33] {
        let objective = ScalarSamples::new(16);
        let mut config = PortfolioEnsembleConfig {
            replicas: 4,
            budget: 4 * allowance,
            ..PortfolioEnsembleConfig::default()
        };
        config.coverage.radius = 1e-12;
        let result = portfolio_values_ensemble_optimize(&objective, 17, None, &config);
        assert_eq!(result.n_evals, objective.positions().len());
        assert_eq!(result.n_evals, config.budget);
        assert_eq!(result.n_grads, 0);
        assert_eq!(result.coverage.published_samples, config.budget as u64);
        assert_eq!(
            result.coverage.sample_peer_checks,
            config.budget - config.replicas
        );
        assert_eq!(
            result.coverage.applied_foreign_samples,
            config.replicas * (config.replicas - 1)
        );
        assert_eq!(result.coverage.repelled_proposals, 0);
    }
}

#[test]
fn sub_stencil_private_and_zero_height_controls_keep_the_same_positions() {
    let mut config = PortfolioEnsembleConfig {
        replicas: 4,
        budget: 13,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.shared = false;
    let private = ScalarSamples::new(8);
    let reference = portfolio_values_ensemble_optimize(&private, 17, None, &config);
    config.coverage.shared = true;
    config.coverage.height = 0.0;
    let disabled = ScalarSamples::new(8);
    let result = portfolio_values_ensemble_optimize(&disabled, 17, None, &config);
    assert_eq!(disabled.positions(), private.positions());
    assert_eq!(result.n_evals, config.budget);
    assert_eq!(result.n_evals, reference.n_evals);
    assert_eq!(result.coverage.published_samples, 0);
    assert_eq!(result.coverage.sample_peer_checks, 0);
}

#[test]
fn shared_scalar_proposals_differ_from_the_private_trace() {
    // From ten dimensions the quenched chain starts at the incumbent, so
    // there is no separate low-discrepancy population to correct. Sharing
    // still moves later proposals.
    let mut config = PortfolioEnsembleConfig {
        replicas: 2,
        budget: 2_048,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.shared = false;
    let private = ScalarSamples::new(16);
    let reference = portfolio_values_ensemble_optimize(&private, 17, None, &config);
    config.coverage.shared = true;
    config.coverage.radius = 0.8;
    let shared = ScalarSamples::new(16);
    let result = portfolio_values_ensemble_optimize(&shared, 17, None, &config);
    assert_eq!(reference.n_evals + reference.n_answered, config.budget);
    assert_eq!(result.n_evals + result.n_answered, config.budget);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, 1.0);
    assert_eq!(private.positions().len(), reference.n_evals);
    assert_eq!(shared.positions().len(), result.n_evals);
    assert_ne!(private.positions(), shared.positions());
    assert!(result.coverage.applied_foreign_samples > 0);
    assert!(result.coverage.repelled_proposals > 0);
}
