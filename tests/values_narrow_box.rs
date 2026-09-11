use std::sync::Mutex;

use anneal_core::methods::box_hopping::ensemble_hop_optimize;
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::minima_hopping::HistoryMembership;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

struct NarrowBox {
    bounds: Bounds<f64>,
    observed: Mutex<Vec<(Vec<f64>, f64)>>,
}

impl Objective<f64> for NarrowBox {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), self.bounds.dims);
        assert!(self.bounds.contains(x));
        let value = x.iter().enumerate().map(|(j, &v)| {
            let width = self.bounds.high[j] - self.bounds.low[j];
            ((v - self.bounds.low[j]) / width - 0.37).powi(2)
        }).sum();
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
        panic!("the values-only entry must not request a user gradient")
    }

    fn dim(&self) -> usize {
        panic!("the values-only entry must not inspect a user gradient")
    }
}

#[test]
fn scalar_portfolio_accepts_narrow_finite_parameter_boxes() {
    for width in [1e-7, 1e-9, 1e-16] {
        let objective = NarrowBox {
            bounds: Bounds::new(Array1::zeros(2), Array1::from_elem(2, width), 0.0),
            observed: Mutex::new(Vec::new()),
        };
        let start = Array1::from_elem(2, width * 0.9);
        let result = ensemble_hop_optimize::<_, NoUserGradient>(
            &objective, None, 7, Some(start.view()), 256, 1,
            HistoryMode::None, HistoryMembership::Accepted,
        );
        let observed = objective.observed.lock().unwrap();
        assert_eq!(result.n_grads, 0);
        assert_eq!(result.n_evals, observed.len());
        assert!(result.n_evals > 0 && result.n_evals <= 256);
        let best = observed.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        assert_eq!(result.best_val, best.1);
        assert_eq!(result.best_pos.to_vec(), best.0);
        assert!(result.best_val < observed[0].1);
    }
}
