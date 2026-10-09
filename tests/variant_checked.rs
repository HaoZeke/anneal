//! Tests for `SaVariant::checked`: happy path on the three IISE-manuscript
//! preset variants (Boltzmann, Fast, GSA), and a negative path that
//! constructs a deliberately broken `Neighborhood` to confirm the
//! `LawViolation::Symmetry` arm fires.

use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::laws::LawViolation;
use anneal_core::movekernel::{Gaussian, Reflected};
use anneal_core::neigh::{BoxConstrained, Neighborhood};
use anneal_core::runner::run_rs_variant_from;
use anneal_core::variant::{
    SaVariant, SweepBudget, ValidationEvidence, boltzmann, boltzmann_in_box, fast, fast_in_box,
    gsa, gsa_in_box,
};

use eindir_core::objectives::StybTang2D;
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1, array};
use std::sync::{Arc, Mutex};

#[test]
fn boltzmann_preset_constructs() {
    let v = boltzmann(StybTang2D::new(), 1.0, 0.5).expect("Boltzmann should pass L1-L4");
    assert_eq!(v.obj.dim(), 2);
}

#[test]
fn fast_preset_constructs() {
    let v = fast(StybTang2D::new(), 1.0, 0.3).expect("Fast should pass L1-L4");
    assert_eq!(v.obj.dim(), 2);
}

#[test]
fn gsa_preset_constructs() {
    let v = gsa(StybTang2D::new(), 1.0, 2.62, 1.7).expect("GSA should pass L1-L4");
    assert_eq!(v.obj.dim(), 2);
}

/// Every point an objective was asked to evaluate, in call order.
type Seen = Arc<Mutex<Vec<Array1<f64>>>>;

/// Wraps an objective and records every point it is asked to evaluate.
struct Recorded<O> {
    inner: O,
    seen: Seen,
}

impl<O: Objective<f64>> Objective<f64> for Recorded<O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.seen.lock().unwrap().push(x.to_owned());
        self.inner.eval(x)
    }
}

fn recorded() -> (Recorded<StybTang2D>, Seen) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let objective = Recorded {
        inner: StybTang2D::new(),
        seen: Arc::clone(&seen),
    };
    (objective, seen)
}

fn assert_all_inside(seen: &[Array1<f64>], bounds: &Bounds<f64>) {
    assert!(!seen.is_empty());
    for x in seen {
        for k in 0..x.len() {
            assert!(
                x[k] >= bounds.low[k] && x[k] <= bounds.high[k],
                "evaluated {x} outside [{}, {}]",
                bounds.low[k],
                bounds.high[k]
            );
        }
    }
}

#[test]
fn box_presets_are_certified() {
    for validation in [
        boltzmann_in_box(StybTang2D::new(), 1.0, 0.5).map(|v| v.validation),
        fast_in_box(StybTang2D::new(), 1.0, 0.3).map(|v| v.validation),
        gsa_in_box(StybTang2D::new(), 1.0, 2.62, 1.7).map(|v| v.validation),
    ] {
        assert_eq!(
            validation.expect("box preset passes L1-L4"),
            ValidationEvidence::Certified
        );
    }
}

#[test]
fn box_presets_evaluate_only_inside_the_box() {
    // Steps far wider than the box: an unreflected walk leaves it at once.
    let bounds = StybTang2D::new().bounds().clone();

    let (objective, seen) = recorded();
    let h = run_rs_variant_from(
        boltzmann_in_box(objective, 5.0, 25.0).unwrap(),
        20,
        50,
        7,
        None,
    );
    assert_all_inside(&seen.lock().unwrap(), &bounds);
    assert_eq!(seen.lock().unwrap().len(), 1 + 20 * 50);
    assert!(bounds.contains(h.best.pos.view()));

    let (objective, seen) = recorded();
    let h = run_rs_variant_from(fast_in_box(objective, 5.0, 25.0).unwrap(), 20, 50, 7, None);
    assert_all_inside(&seen.lock().unwrap(), &bounds);
    assert!(bounds.contains(h.best.pos.view()));

    let (objective, seen) = recorded();
    let h = run_rs_variant_from(
        gsa_in_box(objective, 5.0, 2.9, 1.7).unwrap(),
        20,
        50,
        7,
        None,
    );
    assert_all_inside(&seen.lock().unwrap(), &bounds);
    assert!(bounds.contains(h.best.pos.view()));
}

#[test]
fn box_preset_starts_from_x0() {
    let x0 = array![-2.903534, -2.903534];
    let (objective, seen) = recorded();
    let h = run_rs_variant_from(
        boltzmann_in_box(objective, 1.0, 0.5).unwrap(),
        1,
        3,
        11,
        Some(x0.clone()),
    );
    assert_eq!(seen.lock().unwrap()[0], x0);
    assert!(h.best.val <= StybTang2D::new().eval(x0.view()));
}

#[test]
fn matching_reflected_box_pair_is_certified() {
    let objective = StybTang2D::new();
    let bounds = objective.bounds().clone();
    let neighborhood = BoxConstrained::new(bounds.clone());
    let mover = Reflected::new(Gaussian::new(0.5), bounds);
    let variant = SaVariant::checked(
        objective,
        LogCool::new(1.0_f64, 2.0),
        neighborhood,
        mover,
        Metropolis,
    )
    .expect("matching reflected move and box are certified");
    assert_eq!(variant.validation, ValidationEvidence::Certified);
}

#[test]
fn mismatched_reflected_box_pair_is_rejected() {
    let objective = StybTang2D::new();
    let neighborhood = BoxConstrained::new(objective.bounds().clone());
    let other_bounds = Bounds::new(array![-4.0, -4.0], array![4.0, 4.0], 0.0);
    let mover = Reflected::new(Gaussian::new(0.5), other_bounds);
    let result = SaVariant::checked(
        objective,
        LogCool::new(1.0_f64, 2.0),
        neighborhood,
        mover,
        Metropolis,
    );
    assert!(matches!(result, Err(LawViolation::SupportEscape)));
}

/// Mock neighborhood that lies about symmetry to exercise the runtime
/// validation path for third-party components.
struct BadNeigh;

impl Neighborhood<f64> for BadNeigh {
    fn contains(&self, _i: ArrayView1<f64>, _j: ArrayView1<f64>) -> bool {
        true
    }
    fn is_symmetric(&self) -> bool {
        false
    }
}

#[test]
fn checked_rejects_non_symmetric_neighborhood() {
    let result = SaVariant::checked_with_sweep(
        StybTang2D::new(),
        LogCool::new(1.0_f64, 2.0),
        BadNeigh,
        Gaussian::new(0.5),
        Metropolis,
        SweepBudget::Default,
        2,
        5.0,
        0,
    );
    match result {
        Err(LawViolation::Symmetry) => {}
        Err(other) => panic!("expected Symmetry, got {other:?}"),
        Ok(_) => panic!("expected Err(Symmetry), got Ok"),
    }
}
