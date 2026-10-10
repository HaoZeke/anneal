//! A values-only scalar run has no population-diffusion arm. Peer
//! correction still moves paid positions when the coverage height is
//! positive, and a zero height leaves the private trace unchanged.

use std::sync::Mutex;

use anneal_core::{PortfolioEnsembleConfig, portfolio_values_ensemble_optimize};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

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
    assert!(
        result
            .replicas
            .iter()
            .all(|replica| { replica.arm_stats.iter().all(|arm| arm.name != "dmc_pop") })
    );
    let mut positions = objective.positions.into_inner().unwrap();
    assert_eq!(positions.len(), config.budget);
    positions.sort();
    positions
}

#[test]
fn shared_scalar_positions_differ_when_coverage_height_is_positive() {
    let start_point: Array1<f64> = Array1::from_elem(DIM, 0.1875);
    let start: Vec<u64> = start_point.iter().map(|value| value.to_bits()).collect();
    let private = run(false, 0.1);
    let shared = run(true, 0.1);
    assert!(private.contains(&start) && shared.contains(&start));
    assert_ne!(
        private, shared,
        "a positive coverage height must move a paid position"
    );
}

#[test]
fn zero_height_population_control_keeps_private_paid_positions() {
    assert_eq!(run(false, 0.1), run(true, 0.0));
}
