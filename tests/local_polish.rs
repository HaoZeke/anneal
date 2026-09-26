use anneal_core::{
    AnalyticGradient, projected_gradient_polish, qmc_best1bin_scout, qmc_gsa_global_search,
    qmc_projected_gradient_polish, qmc_trust_region_poll, shifted_qmc_projected_gradient_polish,
};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1, array};

struct ShiftedQuadratic {
    bounds: Bounds<f64>,
    weights: Array1<f64>,
}

struct HighCornerLinear {
    bounds: Bounds<f64>,
}

struct CenterQuadratic {
    bounds: Bounds<f64>,
}

struct SmoothNeedle {
    bounds: Bounds<f64>,
}

struct LocalShiftedQuadratic {
    bounds: Bounds<f64>,
}

impl CenterQuadratic {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 0.0),
        }
    }
}

impl SmoothNeedle {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 0.0),
        }
    }
}

impl LocalShiftedQuadratic {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 0.0),
        }
    }
}

impl Objective<f64> for CenterQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0].powi(2) + x[1].powi(2)
    }
}

impl Objective<f64> for SmoothNeedle {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let dx0 = x[0] - 0.5;
        let dx1 = x[1] + 0.5;
        -(-40.0 * (dx0 * dx0 + dx1 * dx1)).exp()
    }
}

impl Objective<f64> for LocalShiftedQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (x[0] - 0.12).powi(2) + (x[1] + 0.08).powi(2)
    }
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
fn qmc_polish_screens_center_anchor() {
    let obj = CenterQuadratic::new();
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]])
    });

    let result = qmc_projected_gradient_polish(&obj, &grad, 5, 8, 0, 1.0, 1e-10, 1);

    assert!(result.best_val <= 1e-12);
    assert_eq!(result.best_pos.to_vec(), vec![0.0, 0.0]);
}

#[test]
fn qmc_polish_reports_polished_values() {
    let obj = CenterQuadratic::new();
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]])
    });

    let result = qmc_projected_gradient_polish(&obj, &grad, 5, 8, 0, 1.0, 1e-10, 3);

    assert_eq!(result.polished_values.len(), result.n_polished);
    assert_eq!(result.polished_values.len(), 3);
    assert!(result.polished_values.iter().all(|v| v.is_finite()));
    assert!(result.polished_values.contains(&result.best_val));
}

#[test]
fn qmc_polish_reports_projected_stationarity() {
    let obj = CenterQuadratic::new();
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]])
    });

    let result = qmc_projected_gradient_polish(&obj, &grad, 5, 8, 0, 1.0, 1e-10, 3);

    assert_eq!(
        result.polished_projected_grad_norms.len(),
        result.n_polished
    );
    assert_eq!(result.polished_projected_grad_norms.len(), 3);
    assert!(
        result
            .polished_projected_grad_norms
            .iter()
            .all(|norm| norm.is_finite() && *norm <= 1e-8)
    );
}

#[test]
fn qmc_best1bin_scout_refines_smooth_low_dimensional_basin() {
    let obj = SmoothNeedle::new();

    let result = qmc_best1bin_scout(&obj, 240, 0, 30, 0.5, 0.5, 0.7);

    assert!(result.best_val < -0.85, "best value: {}", result.best_val);
    assert!(result.n_evals <= 240);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.n_starts, 30);
    assert_eq!(result.n_polished, 0);
}

#[test]
fn qmc_gsa_global_search_uses_bounded_visiting_distribution() {
    let obj = SmoothNeedle::new();

    let result = qmc_gsa_global_search(&obj, 240, 0, 30, 1.0, 2.62, 1.7);

    assert!(result.best_val < -0.85, "best value: {}", result.best_val);
    assert!(result.n_evals <= 240);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.n_starts, 30);
    assert_eq!(result.n_polished, 0);
    assert!(
        result
            .best_pos
            .iter()
            .all(|value| (-1.0..=1.0).contains(value))
    );
}

#[test]
fn qmc_trust_region_poll_improves_local_shifted_basin() {
    let obj = LocalShiftedQuadratic::new();

    let result = qmc_trust_region_poll(&obj, array![0.0, 0.0], 96, 7, 0.25, 2, 24);

    assert!(result.best_val < 0.0025, "best value: {}", result.best_val);
    assert!(result.n_evals <= 96);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.n_polished, 0);
    assert!(result.best_pos[0] > 0.08);
    assert!(result.best_pos[1] < -0.04);
}

#[test]
fn shifted_qmc_polish_replicates_screened_designs() {
    let obj = CenterQuadratic::new();
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]])
    });

    let result = shifted_qmc_projected_gradient_polish(&obj, &grad, 4, 16, 7, 2, 1.0, 1e-10, 1);

    assert!(result.best_val < 1e-12);
    assert_eq!(result.n_starts, 8);
    assert_eq!(result.n_polished, 2);
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

/// Kappa ~ 1e8 diagonal quadratic: the LANCZOS-class valley in miniature.
///
/// The pre-fix line search (Armijo halving, sqrt(eps) curvature floor, break
/// on first stall) plateaus around 1e-6 here with budget left; interpolated
/// backtracking plus stall recovery must polish to the deep floor.
struct IllConditionedQuadratic {
    bounds: Bounds<f64>,
}

impl IllConditionedQuadratic {
    const KAPPA: f64 = 1e8;

    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-5.0, -5.0], array![5.0, 5.0], 0.0),
        }
    }
}

impl Objective<f64> for IllConditionedQuadratic {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0] * x[0] + Self::KAPPA * x[1] * x[1]
    }
}

#[test]
fn projected_gradient_polish_reaches_deep_floor_on_ill_conditioned_valley() {
    let obj = IllConditionedQuadratic::new();
    let grad = AnalyticGradient::new(2, |x: ArrayView1<f64>| {
        Array1::from_vec(vec![
            2.0 * x[0],
            2.0 * IllConditionedQuadratic::KAPPA * x[1],
        ])
    });

    let result = projected_gradient_polish(&obj, &grad, array![4.0, 3.0], 2000, 1.0, 1e-12);

    assert!(
        result.best_val < 1e-14,
        "polish stalled at {:.3e} after {} evals + {} grads",
        result.best_val,
        result.n_evals,
        result.n_grads
    );
}
