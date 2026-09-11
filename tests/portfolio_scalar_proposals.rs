//! Scalar global candidates participate in the position channel without forces.

use std::sync::Mutex;

use anneal_core::{PortfolioEnsembleConfig, portfolio_values_ensemble_optimize};
use eindir_core::{Bounds, Objective, shifted_low_discrepancy_points};
use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};

struct ScalarSamples {
    bounds: Bounds<f64>,
    positions: Mutex<Vec<Vec<u64>>>,
}

impl ScalarSamples {
    fn new(dim: usize) -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(dim, -2.0), Array1::from_elem(dim, 2.0), 0.0),
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
        self.positions.lock().unwrap().push(x.iter().map(|v| v.to_bits()).collect());
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
        assert!(result.replicas.iter().all(|replica| replica.arm_stats.iter().all(|arm| arm.pulls == 0)));
        // One initial paid position per chain, followed entirely by global draws.
        assert_eq!(result.coverage.published_samples, budget as u64);
        assert_eq!(result.coverage.sample_peer_checks, budget - replicas);
        assert!(result.coverage.applied_foreign_samples >= replicas);
        assert!(result.coverage.repelled_proposals > 0);
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
fn gsa_initial_population_uses_peer_corrected_measured_positions() {
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
    assert_eq!(reference.n_evals, config.budget);
    assert_eq!(result.n_evals, config.budget);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, 1.0);
    let private_positions = private.positions();
    let shared_positions = shared.positions();
    assert_eq!(private_positions.len(), config.budget);
    assert_eq!(shared_positions.len(), config.budget);
    for replica in 0..config.replicas {
        // The front-loaded DE and GSA arms have independent seeded streams.
        // Identify GSA initialization by its generating stream, not a callback index.
        let replica_seed = 17 ^ (replica as u64).wrapping_mul(0x9E37_79B9);
        let front_seed = (replica_seed ^ 0xBEEF).wrapping_add(2);
        let gsa_seed = StdRng::seed_from_u64(front_seed).random::<u64>() ^ 2;
        let starts = shifted_low_discrepancy_points(
            &private.bounds,
            1,
            anneal_core::runner::qmc_skip_from_seed(gsa_seed),
            gsa_seed,
        );
        let start: Vec<u64> = starts.row(0).iter().map(|v| v.to_bits()).collect();
        assert!(private_positions.contains(&start), "GSA initialization must be exercised");
        assert!(!shared_positions.contains(&start), "GSA must evaluate its corrected start");
    }
}
