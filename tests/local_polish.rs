use anneal_core::{projected_gradient_polish, qmc_projected_gradient_polish, AnalyticGradient};
use eindir_core::{Bounds, Objective};
use ndarray::{array, Array1, ArrayView1};

struct ShiftedQuadratic {
    bounds: Bounds<f64>,
    weights: Array1<f64>,
}

struct HighCornerLinear {
    bounds: Bounds<f64>,
}

impl HighCornerLinear {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 0.0),
        }
    }
}

impl Objective<f64> for HighCornerLinear {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        -(x[0] + x[1])
    }
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
fn qmc_polish_screens_boundary_anchors() {
    let obj = HighCornerLinear::new();
    let grad = AnalyticGradient::new(2, |_x: ArrayView1<f64>| Array1::from_vec(vec![-1.0, -1.0]));

    let result = qmc_projected_gradient_polish(&obj, &grad, 2, 8, 0, 1.0, 1e-10, 2);

    assert!(result.best_val <= -2.0);
    assert_eq!(result.best_pos.to_vec(), vec![1.0, 1.0]);
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

struct RotatedQuadratic {
    bounds: Bounds<f64>,
}

impl RotatedQuadratic {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 1e-9),
        }
    }

    fn shifted(x: ArrayView1<f64>) -> (f64, f64) {
        (x[0] - 0.25, x[1] + 0.4)
    }

    fn hessian_times(dx0: f64, dx1: f64) -> (f64, f64) {
        let theta = 0.7_f64;
        let (sin_t, cos_t) = theta.sin_cos();
        let z0 = cos_t * dx0 + sin_t * dx1;
        let z1 = -sin_t * dx0 + cos_t * dx1;
        let hz0 = 1_000.0 * z0;
        let hz1 = 1e-2 * z1;
        (cos_t * hz0 - sin_t * hz1, sin_t * hz0 + cos_t * hz1)
    }
}

impl Objective<f64> for RotatedQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let (dx0, dx1) = Self::shifted(x);
        let (hx0, hx1) = Self::hessian_times(dx0, dx1);
        0.5 * (dx0 * hx0 + dx1 * hx1)
    }
}

#[test]
fn projected_gradient_polish_uses_rotated_curvature() {
    let obj = RotatedQuadratic::new();
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        let (dx0, dx1) = RotatedQuadratic::shifted(x);
        let (gx0, gx1) = RotatedQuadratic::hessian_times(dx0, dx1);
        Array1::from_vec(vec![gx0, gx1])
    });

    let result = projected_gradient_polish(&obj, &grad, array![0.9, 0.9], 64, 1.0, 1e-10);

    assert!(result.best_val < 1e-8);
    assert!(result.n_evals + result.n_grads <= 64);
    assert!((result.best_pos[0] - 0.25).abs() < 1e-3);
    assert!((result.best_pos[1] + 0.4).abs() < 1e-3);
}

#[test]
fn projected_gradient_polish_recovers_from_oversized_initial_scaling() {
    let obj = ShiftedQuadratic::new();
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        Array1::from_vec(vec![2.0 * (x[0] - 0.25), 2.0 * (x[1] + 0.4)])
    });

    let result = projected_gradient_polish(&obj, &grad, array![0.9, 0.9], 16, 1e9, 1e-10);

    assert!(result.best_val < 1e-3);
    assert!(result.n_evals <= 16);
    assert!(result.n_grads <= 16);
}
