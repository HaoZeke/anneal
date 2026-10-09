//! Integration tests for the box-constrained `run` surface.
//!
//! The Python `anneal.run` / `anneal.run_qmc` entry points drive the
//! `*_bounded` variants, so every evaluation point must lie inside the
//! declared box, and an explicit start (`x0`) must seed the chain. These
//! tests pin that contract at the Rust level with a recording objective.

use std::sync::{Arc, Mutex};

use anneal_core::methods::portfolio_optimize_seeded;
use anneal_core::variant::{boltzmann_bounded, fast_bounded, gsa_bounded};
use anneal_core::{run_rs_qmc_variant_start, run_rs_variant_start};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

/// Sphere on `[-3, 3]^dim` that records every evaluated point into a
/// shared log, so the test can inspect evaluations after the objective
/// is consumed by the variant.
#[derive(Clone)]
struct RecordingSphere {
    bounds: Bounds<f64>,
    seen: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl RecordingSphere {
    fn new(dim: usize) -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_elem(dim, -3.0),
                Array1::from_elem(dim, 3.0),
                1e-12,
            ),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn assert_all_in_bounds(&self) {
        let seen = self.seen.lock().expect("seen mutex");
        assert!(
            !seen.is_empty(),
            "objective must be evaluated at least once"
        );
        for point in seen.iter() {
            for (k, value) in point.iter().enumerate() {
                assert!(
                    *value >= -3.0 && *value <= 3.0,
                    "evaluation point escaped the box at dim {k}: {value}",
                );
            }
        }
    }
}

impl Objective<f64> for RecordingSphere {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.seen
            .lock()
            .expect("seen mutex")
            .push(x.iter().copied().collect());
        x.iter().map(|v| v * v).sum()
    }
}

/// Gradient handle for the gradient-free portfolio path (unused).
struct RecordingGrad {
    dim: usize,
}

impl Gradient<f64> for RecordingGrad {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        2.0 * &x.to_owned()
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[test]
fn bounded_boltzmann_never_leaves_the_box() {
    let obj = RecordingSphere::new(4);
    let probe = obj.clone();
    let variant = boltzmann_bounded(obj, 5.0, 0.5).expect("bounded construction");
    let history = run_rs_variant_start(variant, None, 20, 50, 7);
    assert!(history.best.val.is_finite());
    probe.assert_all_in_bounds();
}

#[test]
fn bounded_fast_never_leaves_the_box() {
    let obj = RecordingSphere::new(4);
    let probe = obj.clone();
    let variant = fast_bounded(obj, 3.0, 0.5).expect("bounded construction");
    let history = run_rs_variant_start(variant, None, 20, 50, 7);
    assert!(history.best.val.is_finite());
    probe.assert_all_in_bounds();
}

#[test]
fn bounded_gsa_never_leaves_the_box() {
    let obj = RecordingSphere::new(4);
    let probe = obj.clone();
    let variant = gsa_bounded(obj, 3.0, 2.62, 1.7).expect("bounded construction");
    let history = run_rs_variant_start(variant, None, 20, 50, 7);
    assert!(history.best.val.is_finite());
    probe.assert_all_in_bounds();
}

#[test]
fn bounded_qmc_starts_never_leave_the_box() {
    let obj = RecordingSphere::new(6);
    let probe = obj.clone();
    let variant = fast_bounded(obj, 3.0, 0.5).expect("bounded construction");
    let history = run_rs_qmc_variant_start(variant, None, 6, 10, 25, 11);
    assert!(history.best.val >= 0.0);
    probe.assert_all_in_bounds();
}

#[test]
fn seeded_start_installs_x0_as_initial_best() {
    let x0 = Array1::from_vec(vec![1.0, 2.0, -1.0]);
    let expected = 1.0 + 4.0 + 1.0;
    let obj = RecordingSphere::new(3);
    let variant = boltzmann_bounded(obj, 5.0, 0.5).expect("bounded construction");
    // Zero epochs: the history best is exactly the start evaluation.
    let history = run_rs_variant_start(variant, Some(x0.clone()), 0, 10, 0);
    assert_eq!(history.best.val, expected);
    assert_eq!(history.best.pos, x0);
}

#[test]
fn seeded_chain_never_regresses_past_x0() {
    let x0 = Array1::from_vec(vec![1.0, 2.0, -1.0]);
    let f0 = 6.0;
    let obj = RecordingSphere::new(3);
    let probe = obj.clone();
    let variant = gsa_bounded(obj, 3.0, 2.62, 1.7).expect("bounded construction");
    let history = run_rs_variant_start(variant, Some(x0), 10, 25, 4);
    assert!(
        history.best.val <= f0,
        "seeded chain regressed past its start: {} > {f0}",
        history.best.val
    );
    probe.assert_all_in_bounds();
}

#[test]
fn seeded_qmc_extra_chain_starts_from_x0() {
    let x0 = Array1::from_vec(vec![1.0, 2.0, -1.0]);
    let f0 = 6.0;
    let obj = RecordingSphere::new(3);
    let probe = obj.clone();
    let variant = boltzmann_bounded(obj, 5.0, 0.5).expect("bounded construction");
    let history = run_rs_qmc_variant_start(variant, Some(x0), 4, 5, 20, 9);
    assert!(
        history.best.val <= f0,
        "seeded multistart regressed past x0: {} > {f0}",
        history.best.val
    );
    probe.assert_all_in_bounds();
}

#[test]
fn portfolio_seeded_incumbent_is_x0_floor() {
    let x0 = Array1::from_vec(vec![1.0, 2.0, -1.0]);
    let f0 = 6.0;
    let obj = RecordingSphere::new(3);
    let probe = obj.clone();
    let result =
        portfolio_optimize_seeded(&obj, None::<&RecordingGrad>, 400, 5, None, Some(x0.view()));
    assert!(
        result.best_val <= f0,
        "seeded portfolio regressed past its start: {} > {f0}",
        result.best_val
    );
    for (k, value) in result.best_pos.iter().enumerate() {
        assert!(
            *value >= -3.0 && *value <= 3.0,
            "portfolio best escaped the box at dim {k}: {value}",
        );
    }
    probe.assert_all_in_bounds();
    let _ = RecordingGrad { dim: 3 }.dim();
}
