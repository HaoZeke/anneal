//! Classical presets stay inside the box they are given, and a supplied
//! start is the first point evaluated.

use std::sync::{Arc, Mutex};

use anneal_core::runner::{run_rs_variant, run_rs_variant_at};
use anneal_core::variant::{boltzmann, boltzmann_box, fast_box};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

struct Trace {
    bounds: Bounds<f64>,
    seen: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl Trace {
    fn new(low: f64, high: f64) -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(2, low), Array1::from_elem(2, high), 1e-12),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

fn points(seen: &Arc<Mutex<Vec<Vec<f64>>>>) -> Vec<Vec<f64>> {
    seen.lock().expect("trace").clone()
}

impl Objective<f64> for Trace {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.seen.lock().expect("trace").push(x.to_vec());
        x.iter().map(|v| v * v).sum()
    }
}

fn inside(points: &[Vec<f64>], low: f64, high: f64) -> bool {
    points
        .iter()
        .flatten()
        .all(|v| *v >= low - 1e-9 && *v <= high + 1e-9)
}

#[test]
fn reflected_boltzmann_never_leaves_a_narrow_box() {
    let obj = Trace::new(-0.2, 0.2);
    let seen = Arc::clone(&obj.seen);
    let variant = boltzmann_box(obj, 5.0, 8.0).expect("boxed boltzmann");
    let _ = run_rs_variant_at(variant, 4, 30, 7, None);
    let seen = points(&seen);
    assert!(!seen.is_empty());
    assert!(
        inside(&seen, -0.2, 0.2),
        "a reflected chain left the box: {seen:?}"
    );
}

#[test]
fn reflected_fast_never_leaves_a_narrow_box() {
    let obj = Trace::new(-0.2, 0.2);
    let seen = Arc::clone(&obj.seen);
    let variant = fast_box(obj, 3.0, 20.0).expect("boxed fast");
    let _ = run_rs_variant_at(variant, 3, 20, 9, None);
    assert!(inside(&points(&seen), -0.2, 0.2));
}

#[test]
fn unbounded_boltzmann_can_leave_the_same_box() {
    let obj = Trace::new(-0.2, 0.2);
    let seen = Arc::clone(&obj.seen);
    let variant = boltzmann(obj, 5.0, 8.0).expect("unbounded boltzmann");
    let _ = run_rs_variant(variant, 4, 30, 7);
    assert!(
        !inside(&points(&seen), -0.2, 0.2),
        "the unbounded preset was expected to walk outside the sampling box"
    );
}

#[test]
fn x0_is_the_first_evaluated_point_and_is_clipped() {
    let obj = Trace::new(-1.0, 1.0);
    let seen = Arc::clone(&obj.seen);
    let variant = boltzmann_box(obj, 1.0, 0.1).expect("boxed boltzmann");
    let start = Array1::from_vec(vec![4.0, -0.25]);
    let history = run_rs_variant_at(variant, 0, 1, 1, Some(start));
    let points = points(&seen);
    assert_eq!(points.len(), 1);
    assert!((points[0][0] - 1.0).abs() < 1e-12);
    assert!((points[0][1] + 0.25).abs() < 1e-12);
    assert!((history.best.pos[0] - 1.0).abs() < 1e-12);
    assert!((history.best.pos[1] + 0.25).abs() < 1e-12);
}
