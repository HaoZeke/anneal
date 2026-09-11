use std::sync::Mutex;

use anneal_core::methods::box_hopping::ensemble_hop_optimize;
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::minima_hopping::HistoryMembership;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

struct ScalarLoss {
    bounds: Bounds<f64>,
    nonfinite: bool,
    observed: Mutex<Vec<(Vec<f64>, f64)>>,
}

impl Objective<f64> for ScalarLoss {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), self.bounds.dims);
        assert!(
            x.iter()
                .all(|v| v.is_finite() && (-5.12..=5.12).contains(v))
        );
        let value = if self.nonfinite {
            f64::INFINITY
        } else {
            x.iter()
                .enumerate()
                .map(|(j, &v)| {
                    let optimum = 0.7 + 0.3 * ((j + 1) as f64 * std::f64::consts::SQRT_2).sin();
                    let shifted = v - optimum;
                    shifted * shifted + 10.0 * (1.0 - (2.0 * std::f64::consts::PI * shifted).cos())
                })
                .sum()
        };
        self.observed.lock().unwrap().push((x.to_vec(), value));
        value
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

struct NoUserGradient;

impl Gradient<f64> for NoUserGradient {
    fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> {
        panic!("scalar search must not call a user gradient")
    }

    fn dim(&self) -> usize {
        panic!("scalar search must not inspect a user gradient")
    }
}

fn check(dim: usize, budget: usize, seed: u64, nonfinite: bool) {
    let objective = ScalarLoss {
        bounds: Bounds::new(
            Array1::from_elem(dim, -5.12),
            Array1::from_elem(dim, 5.12),
            0.0,
        ),
        nonfinite,
        observed: Mutex::new(Vec::new()),
    };
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5354_4152_545f_424f);
    let start = Array1::from_shape_fn(dim, |_| -5.12 + 10.24 * rng.random::<f64>());
    let result = ensemble_hop_optimize::<_, NoUserGradient>(
        &objective,
        None,
        seed,
        Some(start.view()),
        budget,
        1,
        HistoryMode::None,
        HistoryMembership::Accepted,
    );
    let observed = objective.observed.lock().unwrap();
    assert_eq!(result.n_evals, observed.len());
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.history_observations, 0);
    assert_eq!(result.charged, budget, "dim={dim}, seed={seed}");
    if nonfinite {
        assert_eq!(result.best_val, f64::INFINITY);
    } else {
        let best = observed.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        assert_eq!(result.best_val, best.1);
        assert_eq!(result.best_pos.to_vec(), best.0);
    }
}

#[test]
fn scalar_endgame_reuses_work_left_by_local_convergence() {
    for seed in 0..4 {
        check(8, 8000, seed, false);
    }
}

#[test]
fn terminal_scalar_search_does_not_require_a_full_stencil() {
    for budget in [1, 2, 3, 7, 8, 17, 31, 257] {
        check(2, budget, 7, false);
    }
}

#[test]
fn nonfinite_local_probes_cannot_stall_the_global_allowance() {
    check(2, 257, 7, true);
}
