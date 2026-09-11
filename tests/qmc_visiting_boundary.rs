use std::sync::Mutex;

use anneal_core::movekernel::{MoveKernel, TsallisVisit, reflect_into_box};
use anneal_core::{qmc_gsa_global_search, qmc_skip_from_seed};
use eindir_core::{Bounds, Objective, shifted_low_discrepancy_points};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

struct ScalarTrace {
    bounds: Bounds<f64>,
    positions: Mutex<Vec<Array1<f64>>>,
}

impl Objective<f64> for ScalarTrace {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, position: ArrayView1<f64>) -> f64 {
        for axis in 0..self.dim() {
            assert!(position[axis].is_finite());
            assert!(position[axis] >= self.bounds.low[axis]);
            assert!(position[axis] <= self.bounds.high[axis]);
        }
        self.positions.lock().unwrap().push(position.to_owned());
        1.0
    }
}

fn check_reflected_visits(bounds: Bounds<f64>) {
    let chain_count = 2;
    let budget = 32;
    let unit_bounds = Bounds::new(Array1::zeros(bounds.dims), Array1::ones(bounds.dims), 0.0);
    let mut crossed_boundary = false;

    for seed in 0..8 {
        let starts =
            shifted_low_discrepancy_points(&bounds, chain_count, qmc_skip_from_seed(seed), seed);
        let initial_unit = Array1::from_iter((0..bounds.dims).map(|axis| {
            let width = bounds.high[axis] - bounds.low[axis];
            if width > 0.0 {
                ((starts[[0, axis]] - bounds.low[axis]) / width).clamp(0.0, 1.0)
            } else {
                0.5
            }
        }));
        let mut rng = StdRng::seed_from_u64(seed);
        let raw = TsallisVisit::new(2.62).propose(initial_unit.view(), 1.0, &mut rng);
        crossed_boundary |= raw.iter().enumerate().any(|(axis, value)| {
            bounds.high[axis] > bounds.low[axis] && !(0.0..=1.0).contains(value)
        });
        let reflected = reflect_into_box(raw.view(), &unit_bounds);
        let expected = Array1::from_iter((0..bounds.dims).map(|axis| {
            bounds.low[axis] + (bounds.high[axis] - bounds.low[axis]) * reflected[axis]
        }));
        let objective = ScalarTrace {
            bounds: bounds.clone(),
            positions: Mutex::new(Vec::new()),
        };

        let result = qmc_gsa_global_search(&objective, budget, seed, chain_count, 1.0, 2.62, -5.0);
        let positions = objective.positions.lock().unwrap();
        assert_eq!(result.n_evals, positions.len());
        assert!(result.n_evals <= budget);
        assert_eq!(result.n_grads, 0);
        assert_eq!(result.n_starts, chain_count);
        assert_eq!(result.best_val, 1.0);
        assert!(positions.contains(&result.best_pos));
        assert!(positions.len() > chain_count);
        for axis in 0..bounds.dims {
            let tolerance = 1e-12 * (bounds.high[axis] - bounds.low[axis]).max(1.0);
            assert!(
                (positions[chain_count][axis] - expected[axis]).abs() <= tolerance,
                "seed={seed}, axis={axis}: paid visit {} must equal reflected visit {}, raw unit visit {}",
                positions[chain_count][axis],
                expected[axis],
                raw[axis],
            );
        }
    }
    assert!(
        crossed_boundary,
        "the replay must exercise a heavy-tailed boundary crossing"
    );
}

#[test]
fn qmc_gsa_reflects_scalar_only_heavy_tailed_visits() {
    check_reflected_visits(Bounds::new(array![0.0, 0.0], array![1.0, 1.0], 0.0));
}

#[test]
fn qmc_gsa_reflection_preserves_anisotropic_and_fixed_coordinates() {
    check_reflected_visits(Bounds::new(
        array![2.0, 7.0, -4.0],
        array![10.0, 7.0, 0.0],
        0.0,
    ));
}
