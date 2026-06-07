use anneal_core::{projected_gradient_polish, AnalyticGradient};
use eindir_core::{Bounds, Objective};
use ndarray::{array, Array1, ArrayView1};

struct ShiftedQuadratic {
    bounds: Bounds<f64>,
    weights: Array1<f64>,
}

impl ShiftedQuadratic {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 1e-9),
            weights: array![1.0, 1.0],
        }
    }

    fn weighted(weights: Array1<f64>) -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 1e-9),
            weights,
        }
    }
}

impl Objective<f64> for ShiftedQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.weights[0] * (x[0] - 0.25).powi(2) + self.weights[1] * (x[1] + 0.4).powi(2)
    }
}

#[test]
fn projected_gradient_polish_refines_bounded_quadratic() {
    let obj = ShiftedQuadratic::new();
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        Array1::from_vec(vec![2.0 * (x[0] - 0.25), 2.0 * (x[1] + 0.4)])
    });

    let result = projected_gradient_polish(&obj, &grad, array![0.9, 0.9], 64, 1.0, 1e-10);

    assert!(result.best_val < 1e-10);
    assert!(result.n_evals <= 64);
    assert!(result.n_grads <= 64);
    assert!((result.best_pos[0] - 0.25).abs() < 1e-5);
    assert!((result.best_pos[1] + 0.4).abs() < 1e-5);
}

#[test]
fn projected_gradient_polish_expands_steps_in_flat_directions() {
    let obj = ShiftedQuadratic::weighted(array![1_000.0, 1e-3]);
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        Array1::from_vec(vec![2_000.0 * (x[0] - 0.25), 2e-3 * (x[1] + 0.4)])
    });

    let result = projected_gradient_polish(&obj, &grad, array![0.9, 0.9], 128, 1.0, 1e-10);

    assert!(result.best_val < 1e-8);
    assert!((result.best_pos[0] - 0.25).abs() < 1e-5);
    assert!((result.best_pos[1] + 0.4).abs() < 1e-3);
}
