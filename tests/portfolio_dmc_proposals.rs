//! Population starts consume the same paid-position geometry as other global arms.

use std::sync::Mutex;

use anneal_core::methods::dmc_population::recommend_target_n;
use anneal_core::movekernel::reflect_into_box;
use anneal_core::{PortfolioEnsembleConfig, portfolio_values_ensemble_optimize};
use eindir_core::{Bounds, Objective, shifted_low_discrepancy_points};
use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIM: usize = 8;
const SEED: u64 = 17;

struct ScalarSamples {
    bounds: Bounds<f64>,
    positions: Mutex<Vec<Vec<u64>>>,
}

impl ScalarSamples {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_elem(DIM, -5.12),
                Array1::from_elem(DIM, 5.12),
                0.0,
            ),
            positions: Mutex::new(Vec::new()),
        }
    }
}

impl Objective<f64> for ScalarSamples {
    fn dim(&self) -> usize {
        DIM
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert!(self.bounds.contains(x));
        self.positions
            .lock()
            .unwrap()
            .push(x.iter().map(|value| value.to_bits()).collect());
        1.0
    }
}

fn run(shared: bool, height: f64) -> Vec<Vec<u64>> {
    let objective = ScalarSamples::new();
    let mut config = PortfolioEnsembleConfig {
        replicas: 2,
        budget: 4_000,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.shared = shared;
    config.coverage.height = height;
    config.coverage.radius = 0.8;
    let start = Array1::from_elem(DIM, 0.1875);
    let result = portfolio_values_ensemble_optimize(&objective, SEED, Some(start.view()), &config);
    assert_eq!(result.n_evals, config.budget);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, 1.0);
    assert!(result.replicas.iter().all(|replica| {
        replica
            .arm_stats
            .iter()
            .any(|arm| arm.name == "dmc_pop" && arm.pulls > 0)
    }));
    let mut positions = objective.positions.into_inner().unwrap();
    assert_eq!(positions.len(), config.budget);
    positions.sort();
    positions
}

fn raw_population_starts() -> Vec<Vec<u64>> {
    let objective = ScalarSamples::new();
    // The width-selected warmup is Explore, GSA, DE, DMC. Each draws one
    // arm seed; the first three arms keep their internal random streams.
    let allowance = 49;
    let walkers = recommend_target_n(allowance, DIM)
        .clamp(6, 32)
        .min(allowance / 3);
    let mut positions = Vec::new();
    for replica in 0..2_u64 {
        let replica_seed = SEED ^ replica.wrapping_mul(0x9E37_79B9);
        let mut rng = StdRng::seed_from_u64(replica_seed);
        let arm_seed = (1..=4)
            .map(|index| rng.random::<u64>() ^ index)
            .last()
            .unwrap();
        let points = shifted_low_discrepancy_points(&objective.bounds, walkers, 1, arm_seed);
        // Walker zero is the incumbent; walkers 1--3 are incumbent jitter.
        for index in 4..walkers {
            let position = reflect_into_box(points.row(index), &objective.bounds);
            positions.push(position.iter().map(|value| value.to_bits()).collect());
        }
    }
    positions
}

#[test]
fn shared_scalar_population_starts_are_peer_corrected_before_evaluation() {
    let raw = raw_population_starts();
    assert!(!raw.is_empty());
    let private = run(false, 0.1);
    let shared = run(true, 0.1);
    for point in raw {
        assert!(
            private.contains(&point),
            "the DMC generator must be exercised"
        );
        assert!(
            !shared.contains(&point),
            "a nearby DMC start must evaluate its peer-corrected position"
        );
    }
}

#[test]
fn zero_height_population_control_keeps_private_paid_positions() {
    assert_eq!(run(false, 0.1), run(true, 0.0));
}
