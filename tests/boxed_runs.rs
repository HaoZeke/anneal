//! Bounds contract for the boxed classical presets.
//!
//! Regression test for the ChemFit report: `run` with a Boltzmann preset let
//! chain positions escape `[low, high]` because the unconstrained presets
//! (`ContinuousR_n` + bare kernels) ignored the box. The boxed constructors
//! (`boltzmann_boxed` / `fast_boxed` / `gsa_boxed`) pair `BoxConstrained`
//! with `Reflected` kernels, and `run_rs_variant_from_position` clips the
//! caller-supplied start. Every evaluated position, including the first,
//! must lie inside the box.

use std::sync::{Arc, Mutex};

use anneal_core::runner::{
    run_rs_qmc_variant_from_position, run_rs_variant, run_rs_variant_from_position,
};
use anneal_core::variant::{boltzmann_boxed, fast_boxed, gsa_boxed};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};

#[derive(Clone)]
struct RecordingQuadratic {
    bounds: Bounds<f64>,
    seen: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl RecordingQuadratic {
    fn new(lo: f64, hi: f64, dim: usize) -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(dim, lo), Array1::from_elem(dim, hi), 1e-9),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn seen(&self) -> Vec<Vec<f64>> {
        self.seen.lock().expect("seen mutex poisoned").clone()
    }
}

impl Objective<f64> for RecordingQuadratic {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.seen
            .lock()
            .expect("seen mutex poisoned")
            .push(x.iter().copied().collect());
        x.iter().map(|v| v * v).sum()
    }
}

fn assert_all_in_box(seen: &[Vec<f64>], lo: f64, hi: f64) {
    assert!(!seen.is_empty(), "objective was never evaluated");
    for pos in seen {
        for &v in pos {
            assert!(
                v >= lo - 1e-9 && v <= hi + 1e-9,
                "out-of-box evaluation: {v} outside [{lo}, {hi}]"
            );
        }
    }
}

#[test]
fn boxed_boltzmann_never_leaves_the_box() {
    let obj = RecordingQuadratic::new(-3.0, 3.0, 6);
    let probe = obj.clone();
    let variant = boltzmann_boxed(obj, 5.0, 2.0).expect("boxed boltzmann");
    let history = run_rs_variant(variant, 20, 50, 0);
    assert_all_in_box(&probe.seen(), -3.0, 3.0);
    for &v in &history.best.pos {
        assert!(v >= -3.0 - 1e-9 && v <= 3.0 + 1e-9, "best escaped: {v}");
    }
}

#[test]
fn boxed_fast_and_gsa_keep_every_eval_in_box() {
    let obj = RecordingQuadratic::new(-3.0, 3.0, 6);
    let probe = obj.clone();
    let variant = fast_boxed(obj, 3.0, 2.0).expect("boxed fast");
    let _ = run_rs_variant(variant, 10, 30, 7);
    assert_all_in_box(&probe.seen(), -3.0, 3.0);

    let obj = RecordingQuadratic::new(-3.0, 3.0, 6);
    let probe = obj.clone();
    let variant = gsa_boxed(obj, 3.0, 2.62, 1.7).expect("boxed gsa");
    let _ = run_rs_variant(variant, 10, 30, 3);
    assert_all_in_box(&probe.seen(), -3.0, 3.0);
}

#[test]
fn boxed_start_from_position_anchors_and_clips() {
    // In-box anchor: best starts at the anchor, so it can only improve on it.
    let obj = RecordingQuadratic::new(-3.0, 3.0, 2);
    let probe = obj.clone();
    let x0 = Array1::from_vec(vec![0.1, -0.2]);
    let x0_val: f64 = x0.iter().map(|v| v * v).sum();
    let variant = boltzmann_boxed(obj, 5.0, 0.5).expect("boxed boltzmann");
    let history = run_rs_variant_from_position(variant, 1, 1, 123, Some(x0));
    assert!(
        history.best.val <= x0_val + 1e-12,
        "anchor not honored: {} > {x0_val}",
        history.best.val
    );
    assert_all_in_box(&probe.seen(), -3.0, 3.0);

    // Out-of-box anchor is clipped, never evaluated raw.
    let obj = RecordingQuadratic::new(-3.0, 3.0, 2);
    let probe = obj.clone();
    let far = Array1::from_vec(vec![100.0, -100.0]);
    let variant = boltzmann_boxed(obj, 5.0, 0.5).expect("boxed boltzmann");
    let history = run_rs_variant_from_position(variant, 1, 1, 123, Some(far));
    assert_all_in_box(&probe.seen(), -3.0, 3.0);
    for &v in &history.best.pos {
        assert!(v >= -3.0 - 1e-9 && v <= 3.0 + 1e-9, "best escaped: {v}");
    }
}

#[test]
fn boxed_qmc_anchor_runs_as_extra_chain_in_box() {
    let obj = RecordingQuadratic::new(-3.0, 3.0, 4);
    let probe = obj.clone();
    let x0 = Array1::from_vec(vec![0.5, -0.5, 0.25, 0.75]);
    let variant = gsa_boxed(obj, 3.0, 2.2, 1.5).expect("boxed gsa");
    let history = run_rs_qmc_variant_from_position(variant, 4, 3, 5, 1, Some(x0));
    assert_all_in_box(&probe.seen(), -3.0, 3.0);
    assert!(history.best.val.is_finite());
    for &v in &history.best.pos {
        assert!(v >= -3.0 - 1e-9 && v <= 3.0 + 1e-9, "best escaped: {v}");
    }
}
