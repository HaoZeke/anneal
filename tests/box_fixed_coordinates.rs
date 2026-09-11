use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::box_hopping::ensemble_hop_optimize;
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::minima_hopping::HistoryMembership;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};

struct LoggedBox {
    bounds: Bounds<f64>,
    trace: Mutex<Vec<Array1<f64>>>,
}

impl Objective<f64> for LoggedBox {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), self.bounds.dims);
        let mut value = 0.0;
        for (i, &coordinate) in x.iter().enumerate() {
            assert!(coordinate.is_finite());
            assert!(coordinate >= self.bounds.low[i] && coordinate <= self.bounds.high[i]);
            if self.bounds.low[i] < self.bounds.high[i] {
                value += coordinate * coordinate;
            }
        }
        self.trace.lock().unwrap().push(x.to_owned());
        value
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

impl Gradient<f64> for LoggedBox {
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        panic!("values-only search must not request a gradient")
    }
    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

#[test]
fn values_portfolio_keeps_fixed_coordinates_in_every_callback() {
    let obj = LoggedBox {
        bounds: Bounds::new(array![-2.0, 3.0, -2.0], array![2.0, 3.0, 2.0], 0.0),
        trace: Mutex::new(Vec::new()),
    };
    let result = ensemble_hop_optimize::<_, LoggedBox>(
        &obj,
        None,
        7,
        Some(array![1.0, 3.0, 1.0].view()),
        64,
        1,
        HistoryMode::None,
        HistoryMembership::Accepted,
    );
    let trace = obj.trace.lock().unwrap();
    assert_eq!(result.charged, trace.len());
    assert!(result.charged > 1 && result.charged <= 64);
    assert_eq!(trace[0], array![1.0, 3.0, 1.0]);
    assert_eq!(result.best_pos[1], 3.0);
    assert_eq!(
        result.best_val,
        result.best_pos[0].powi(2) + result.best_pos[2].powi(2)
    );
}

#[test]
fn all_fixed_values_box_evaluates_its_only_point_once() {
    let obj = LoggedBox {
        bounds: Bounds::new(array![3.0, -4.0], array![3.0, -4.0], 0.0),
        trace: Mutex::new(Vec::new()),
    };
    let result = ensemble_hop_optimize::<_, LoggedBox>(
        &obj,
        None,
        7,
        None,
        64,
        1,
        HistoryMode::None,
        HistoryMembership::Accepted,
    );
    assert_eq!(obj.trace.lock().unwrap().as_slice(), &[array![3.0, -4.0]]);
    assert_eq!(result.charged, 1);
    assert_eq!(result.best_pos, array![3.0, -4.0]);
    assert_eq!(result.best_val, 0.0);
}

#[test]
fn fixed_axes_preserve_the_free_coordinate_portfolio_trace() {
    let fixed = LoggedBox {
        bounds: Bounds::new(array![-2.0, 3.0, -2.0], array![2.0, 3.0, 2.0], 0.0),
        trace: Mutex::new(Vec::new()),
    };
    let free = LoggedBox {
        bounds: Bounds::new(array![-2.0, -2.0], array![2.0, 2.0], 0.0),
        trace: Mutex::new(Vec::new()),
    };
    let expanded = ensemble_hop_optimize::<_, LoggedBox>(
        &fixed,
        None,
        7,
        Some(array![1.0, 3.0, 1.0].view()),
        256,
        1,
        HistoryMode::None,
        HistoryMembership::Accepted,
    );
    let reduced = ensemble_hop_optimize::<_, LoggedBox>(
        &free,
        None,
        7,
        Some(array![1.0, 1.0].view()),
        256,
        1,
        HistoryMode::None,
        HistoryMembership::Accepted,
    );
    let fixed_trace = fixed.trace.lock().unwrap();
    let free_trace = free.trace.lock().unwrap();
    assert_eq!(fixed_trace.len(), free_trace.len());
    for (expanded, reduced) in fixed_trace.iter().zip(free_trace.iter()) {
        assert_eq!(array![expanded[0], expanded[2]], *reduced);
    }
    assert_eq!(expanded.charged, reduced.charged);
    assert_eq!(expanded.best_val, reduced.best_val);
}

#[test]
fn all_fixed_values_box_rejects_nonfinite_bounds_before_evaluation() {
    struct InvalidBox {
        bounds: Bounds<f64>,
        calls: AtomicUsize,
    }
    impl Objective<f64> for InvalidBox {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, _x: ArrayView1<f64>) -> f64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            0.0
        }
    }
    let objective = InvalidBox {
        bounds: Bounds::new(array![f64::INFINITY], array![f64::INFINITY], 0.0),
        calls: AtomicUsize::new(0),
    };
    let result = std::panic::catch_unwind(|| {
        ensemble_hop_optimize::<_, LoggedBox>(
            &objective,
            None,
            0,
            None,
            10,
            1,
            HistoryMode::None,
            HistoryMembership::Accepted,
        )
    });
    assert!(result.is_err(), "nonfinite box bounds must be rejected");
    assert_eq!(objective.calls.load(Ordering::Relaxed), 0);
}
