use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::box_hopping::{
    BoxEnsembleConfig, box_ensemble_optimize, box_values_ensemble_optimize,
};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};

const ENERGY_SCALE: f64 = 1e-20;

struct CountedDoubleWell {
    bounds: Bounds<f64>,
    evaluations: Mutex<Vec<(f64, f64)>>,
    gradients: AtomicUsize,
}

impl CountedDoubleWell {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-2.0], array![2.0], 0.0),
            evaluations: Mutex::new(Vec::new()),
            gradients: AtomicUsize::new(0),
        }
    }

    fn value(x: f64) -> f64 {
        // The positive second factor makes x=-1 the unique zero-energy
        // minimum; the tilted second well has strictly positive energy.
        ENERGY_SCALE * (x + 1.0).powi(2) * ((x - 1.0).powi(2) + 0.1)
    }
}

impl Objective<f64> for CountedDoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 1);
        assert!(x[0].is_finite() && (-2.0..=2.0).contains(&x[0]));
        let value = Self::value(x[0]);
        self.evaluations.lock().unwrap().push((x[0], value));
        value
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn dim(&self) -> usize {
        1
    }
}

impl Gradient<f64> for CountedDoubleWell {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        let left = x[0] + 1.0;
        let right = x[0] - 1.0;
        array![2.0 * ENERGY_SCALE * left * (right * right + 0.1 + left * right)]
    }

    fn dim(&self) -> usize {
        1
    }
}

fn check_incumbent_retention(with_gradient: bool) {
    let objective = CountedDoubleWell::new();
    let start = array![-1.0];
    let config = BoxEnsembleConfig {
        replicas: 2,
        budget: if with_gradient { 12 } else { 56 },
        history: HistoryMode::None,
        shared_deposits: 0,
        ..BoxEnsembleConfig::default()
    };
    // Both replicas have exactly one hop. The energy scale makes the first
    // unbiased uphill acceptance probability round to one throughout the box.
    assert_eq!((-1e-18_f64 / 5.0).exp(), 1.0);
    let result = if with_gradient {
        box_ensemble_optimize(&objective, &objective, 7, Some(start.view()), &config)
    } else {
        box_values_ensemble_optimize(&objective, 7, Some(start.view()), &config)
    };

    let evaluations = objective.evaluations.lock().unwrap();
    let gradients = objective.gradients.load(Ordering::Relaxed);
    let expected_work = if with_gradient { (4, 4) } else { (52, 0) };
    assert_eq!((evaluations.len(), gradients), expected_work);
    assert_eq!((result.n_evals, result.n_grads), expected_work);
    assert!(result.n_evals + result.n_grads <= config.budget);
    assert_eq!(result.hops, 2);
    assert_eq!(evaluations[0], (-1.0, 0.0));
    assert!(evaluations.iter().any(|(_, value)| *value > 0.0));

    let best_measured = evaluations
        .iter()
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    assert_eq!(*best_measured, (-1.0, 0.0));
    assert_eq!(
        result.best_val, best_measured.1,
        "an accepted uphill hop must not replace the ensemble's measured incumbent"
    );
    assert_eq!(result.best_pos, start);
    assert_eq!(
        result.best_val,
        CountedDoubleWell::value(result.best_pos[0])
    );
}

#[test]
fn gradient_box_ensemble_retains_the_best_measured_incumbent() {
    check_incumbent_retention(true);
}

#[test]
fn values_box_ensemble_retains_the_best_measured_incumbent() {
    check_incumbent_retention(false);
}
