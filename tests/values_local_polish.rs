use std::sync::Mutex;

use anneal_core::methods::portfolio::values_local_polish;
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1, array};

struct ScalarObjective<F> {
    bounds: Bounds<f64>,
    value: F,
    observed: Mutex<Vec<(Vec<f64>, f64)>>,
}

impl<F: Fn(ArrayView1<f64>) -> f64 + Send + Sync> Objective<f64> for ScalarObjective<F> {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), self.bounds.dims);
        assert!(self.bounds.contains(x));
        let value = (self.value)(x);
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

fn objective<F: Fn(ArrayView1<f64>) -> f64>(dim: usize, value: F) -> ScalarObjective<F> {
    ScalarObjective {
        bounds: Bounds::new(
            Array1::from_elem(dim, -2.0),
            Array1::from_elem(dim, 2.0),
            0.0,
        ),
        value,
        observed: Mutex::new(Vec::new()),
    }
}

#[test]
fn an_unfunded_stencil_cannot_certify_stationarity() {
    let surface = objective(2, |x| x.dot(&x));
    let start = array![0.7, 0.9];
    let result = values_local_polish(&surface, start.clone(), 1, 0.1, 1e-12);
    assert_eq!(result.n_evals, 1);
    assert_eq!(surface.observed.lock().unwrap().len(), 1);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_pos, start);
    assert_eq!(result.best_val, 1.3);
    assert!(!result.projected_stationary);
    assert!(result.best_grad.is_none());
}

#[test]
fn nonfinite_probe_values_cannot_certify_stationarity() {
    for invalid in [f64::INFINITY, f64::NAN] {
        let surface = objective(2, |x| {
            if x.iter().all(|v| *v == 0.0) {
                0.0
            } else {
                invalid
            }
        });
        let result = values_local_polish(&surface, Array1::zeros(2), 32, 0.1, 1e-12);
        assert_eq!(result.n_grads, 0);
        assert_eq!(result.n_evals, surface.observed.lock().unwrap().len());
        assert!(result.n_evals <= 32);
        assert_eq!(result.best_val, 0.0);
        assert!(!result.projected_stationary);
        assert!(result.best_grad.is_none());
    }
}

#[test]
fn a_funded_stationary_stencil_retains_its_derivative() {
    let surface = objective(2, |x| x.dot(&x));
    let result = values_local_polish(&surface, Array1::zeros(2), 32, 0.1, 1e-12);
    assert_eq!(result.n_evals, surface.observed.lock().unwrap().len());
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, 0.0);
    assert!(result.projected_stationary);
    assert_eq!(result.best_grad.unwrap(), Array1::<f64>::zeros(2));
}

#[test]
fn the_raw_incumbent_includes_a_terminal_stencil_probe() {
    let surface = objective(2, |x| x.dot(&x));
    let result = values_local_polish(&surface, array![0.7, 0.9], 3, 0.1, 1e-12);
    let seen = surface.observed.lock().unwrap();
    assert_eq!(result.n_evals, seen.len());
    assert_eq!(seen.len(), 3);
    let best = seen.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    assert_eq!(result.best_val, best.1);
    assert_eq!(result.best_pos.to_vec(), best.0);
    assert!(result.best_val < seen[0].1);
    assert!(!result.projected_stationary);
    assert!(result.best_grad.is_none());
}

#[test]
fn scalar_refinement_resolves_a_correlated_quadratic() {
    let surface = objective(8, |x| {
        let shifted = x.mapv(|v| v - 0.37);
        let reflection = 2.0 * shifted.sum() / shifted.len() as f64;
        shifted
            .iter()
            .enumerate()
            .map(|(i, v)| 0.5 * 1000.0_f64.powf(i as f64 / 7.0) * (v - reflection).powi(2))
            .sum()
    });
    let result = values_local_polish(&surface, Array1::from_elem(8, 1.4), 2000, 0.1, 1e-8);
    let seen = surface.observed.lock().unwrap();
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.n_evals, seen.len());
    assert!(result.n_evals <= 2000);
    assert!(result.best_val < 1e-6, "{}", result.best_val);
    assert_eq!(
        result.best_val,
        seen.iter()
            .map(|(_, value)| *value)
            .fold(f64::INFINITY, f64::min)
    );
}

#[test]
fn fixed_coordinates_do_not_consume_stencil_work() {
    let surface = ScalarObjective {
        bounds: Bounds::new(array![5.0, -2.0], array![5.0, 2.0], 0.0),
        value: |x: ArrayView1<f64>| x[1] * x[1],
        observed: Mutex::new(Vec::new()),
    };
    let result = values_local_polish(&surface, array![5.0, 0.0], 16, 0.1, 1e-12);
    assert_eq!(result.n_evals, 3);
    assert_eq!(surface.observed.lock().unwrap().len(), 3);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_pos, array![5.0, 0.0]);
    assert!(result.projected_stationary);
    assert_eq!(result.best_grad.unwrap(), array![0.0, 0.0]);
}

#[test]
fn small_stencil_widths_do_not_rescale_a_valid_derivative() {
    let surface = ScalarObjective {
        bounds: Bounds::new(array![0.0], array![1e-18], 0.0),
        value: |x: ArrayView1<f64>| x[0],
        observed: Mutex::new(Vec::new()),
    };
    let result = values_local_polish(&surface, array![0.0], 16, 0.1, 1e-12);
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.n_evals, surface.observed.lock().unwrap().len());
    assert!(result.projected_stationary);
    assert!((result.best_grad.unwrap()[0] - 1.0).abs() < 1e-12);
}

#[test]
fn rounded_stencil_steps_use_distinct_representable_points() {
    let low = 1e10;
    let high = low + 1e-3;
    let surface = ScalarObjective {
        bounds: Bounds::new(array![low], array![high], 0.0),
        value: |x: ArrayView1<f64>| x[0] - low,
        observed: Mutex::new(Vec::new()),
    };
    let result = values_local_polish(&surface, array![(low + high) * 0.5], 128, 0.1, 1e-12);
    assert_eq!(result.best_pos, array![low]);
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.n_evals, surface.observed.lock().unwrap().len());
    assert!(result.n_evals <= 128);
    assert!(result.projected_stationary);
    assert!((result.best_grad.unwrap()[0] - 1.0).abs() < 1e-12);
}
