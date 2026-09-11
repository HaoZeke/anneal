use std::sync::Mutex;

use anneal_core::methods::box_hopping::{BoxEnsembleConfig, box_values_ensemble_optimize};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Objective};
use ndarray::{ArrayView1, array};

struct CountedDescendingLine {
    bounds: Bounds<f64>,
    evaluations: Mutex<Vec<f64>>,
}

impl Objective<f64> for CountedDescendingLine {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 1);
        self.evaluations.lock().unwrap().push(x[0]);
        -x[0]
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn dim(&self) -> usize {
        1
    }
}

#[test]
fn values_box_ensemble_retains_a_better_certificate_probe() {
    let objective = CountedDescendingLine {
        bounds: Bounds::new(array![0.0], array![1.0], 0.0),
        evaluations: Mutex::new(Vec::new()),
    };
    let start = array![0.0];
    let config = BoxEnsembleConfig {
        replicas: 1,
        budget: 3,
        history: HistoryMode::None,
        shared_deposits: 0,
        ..BoxEnsembleConfig::default()
    };

    let result = box_values_ensemble_optimize(&objective, 7, Some(start.view()), &config);

    let evaluations = objective.evaluations.lock().unwrap();
    assert_eq!(evaluations.as_slice(), &[0.0, 0.0, 1e-6]);
    assert_eq!((result.n_evals, result.n_grads), (3, 0));
    assert_eq!(result.n_evals + result.n_grads, config.budget);
    assert_eq!(result.hops, 0);
    assert_eq!(
        result.best_val, -1e-6,
        "a paid feasible certificate probe belongs to the measured incumbent"
    );
    assert_eq!(result.best_pos, array![1e-6]);
    assert_eq!(result.best_val, -result.best_pos[0]);
}
