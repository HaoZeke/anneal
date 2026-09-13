//! Continuous population search does not privilege an absolute coordinate origin.

use std::sync::Mutex;

use anneal_core::methods::dmc_population::run_dmc_population_seeded;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};

struct ConstantLandscape {
    bounds: Bounds<f64>,
    positions: Mutex<Vec<Array1<f64>>>,
}

impl ConstantLandscape {
    fn new(translation: &Array1<f64>) -> Self {
        Self {
            bounds: Bounds::new(translation - 4.0, translation + 4.0, 0.0),
            positions: Mutex::new(Vec::new()),
        }
    }
}

impl Objective<f64> for ConstantLandscape {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert!(self.bounds.contains(x));
        self.positions.lock().unwrap().push(x.to_owned());
        1.0
    }
}

struct NoGradient;

impl Gradient<f64> for NoGradient {
    fn dim(&self) -> usize {
        panic!("scalar search has no gradient provider")
    }

    fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> {
        panic!("scalar search has no gradient provider")
    }
}

fn run(translation: &Array1<f64>, budget: usize, seed: u64) -> Vec<Array1<f64>> {
    let objective = ConstantLandscape::new(translation);
    let initial = translation + 0.1875;
    let mut rng = StdRng::seed_from_u64(seed);
    let result = run_dmc_population_seeded::<_, NoGradient, _>(
        &objective,
        None,
        budget,
        seed,
        6,
        3,
        1.0,
        Some(initial.view()),
        &mut rng,
    );
    let positions = objective.positions.into_inner().unwrap();
    assert_eq!(result.n_evals, positions.len());
    assert!(result.n_evals <= budget);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, 1.0);
    assert!(positions.contains(&result.best_pos));
    positions
}

#[test]
fn scalar_population_trace_translates_with_its_box() {
    let origin = Array1::zeros(4);
    let translation = Array1::from_vec(vec![0.375, -0.625, 0.8125, -0.3125]);
    for budget in [49, 400, 800] {
        for seed in [7, 11, 33] {
            let reference = run(&origin, budget, seed);
            let translated = run(&translation, budget, seed);
            assert_eq!(
                translated.len(),
                reference.len(),
                "budget={budget}, seed={seed}: changing the origin must not change paid work"
            );
            for (call, (base, shifted)) in reference.iter().zip(&translated).enumerate() {
                for axis in 0..origin.len() {
                    assert!(
                        (shifted[axis] - translation[axis] - base[axis]).abs() < 1e-7,
                        "budget={budget}, seed={seed}, call={call}, axis={axis}: base={}, translated={}, offset={}",
                        base[axis],
                        shifted[axis],
                        translation[axis]
                    );
                }
            }
        }
    }
}
