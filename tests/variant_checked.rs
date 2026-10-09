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
        None,
    );
    assert_all_inside(&seen.lock().unwrap(), &bounds);
    assert_eq!(seen.lock().unwrap().len(), 1 + 20 * 50);
    assert!(bounds.contains(h.best.pos.view()));

    let (objective, seen) = recorded();
    let h = run_rs_variant_from(
        fast_in_box(objective, 5.0, 25.0).unwrap(),
        20,
        50,
        7,
        None,
        None,
    );
    assert_all_inside(&seen.lock().unwrap(), &bounds);
    assert!(bounds.contains(h.best.pos.view()));

    let (objective, seen) = recorded();
    let h = run_rs_variant_from(
        gsa_in_box(objective, 5.0, 2.9, 1.7).unwrap(),
        20,
        50,
        7,
        None,
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
        None,
    );
    assert_eq!(seen.lock().unwrap()[0], x0);
    assert!(h.best.val <= StybTang2D::new().eval(x0.view()));
}

#[test]
fn max_evals_caps_the_calls_including_the_start() {
    for cap in [1, 2, 37, 1001, 5000] {
        let (objective, seen) = recorded();
        let h = run_rs_variant_from(
            boltzmann_in_box(objective, 1.0, 0.5).unwrap(),
            20,
            50,
            3,
            None,
            Some(cap),
        );
        let calls = seen.lock().unwrap().len();
        assert_eq!(calls, cap.min(1 + 20 * 50), "cap {cap}");
        let steps: usize = h.epochs.iter().map(|e| e.accepted + e.rejected).sum();
        assert_eq!(steps + 1, calls);
    }
}

#[test]
fn gsa_near_q_v_three_evaluates_every_scheduled_step() {
    let bounds = StybTang2D::new().bounds().clone();
    for q_v in [2.99, 2.999] {
        let (objective, seen) = recorded();
        run_rs_variant_from(
            gsa_in_box(objective, 1.0, q_v, 1.7).unwrap(),
            20,
            100,
            5,
            None,
            None,
        );
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1 + 20 * 100, "q_v {q_v}");
        assert_all_inside(&seen, &bounds);
    }
}

/// Styblinski-Tang on a box whose upper wall `0.7` is not a sum `lo + w`
/// that rounds back to itself.
struct OddBox {
    bounds: Bounds<f64>,
}

impl Objective<f64> for OddBox {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        StybTang2D::new().eval(x)
    }
}

#[test]
fn reflection_never_rounds_past_the_upper_wall() {
    let bounds = Bounds::new(array![-3.0, -1.0], array![0.7, 0.3], 0.0);
    let seen: Seen = Arc::default();
    let objective = Recorded {
        inner: OddBox {
            bounds: bounds.clone(),
        },
        seen: Arc::clone(&seen),
    };
    run_rs_variant_from(
        gsa_in_box(objective, 1.0, 2.99, 1.7).unwrap(),
        20,
        100,
        9,
        Some(array![0.7, 0.3]),
        None,
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0], array![0.7, 0.3]);
    assert_all_inside(&seen, &bounds);
}

/// An objective that is NaN at its start and finite elsewhere.
struct NanAtOrigin {
    inner: StybTang2D,
}

impl Objective<f64> for NanAtOrigin {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if x.iter().all(|&v| v == 0.0) {
            f64::NAN
        } else {
            self.inner.eval(x)
        }
    }
}

#[test]
fn a_nan_start_does_not_freeze_the_walk() {
    let objective = NanAtOrigin {
        inner: StybTang2D::new(),
    };
    let h = run_rs_variant_from(
        boltzmann_in_box(objective, 1.0, 0.5).unwrap(),
        5,
        100,
        2,
        Some(array![0.0, 0.0]),
        None,
    );
    assert!(h.best.val.is_finite());
    assert!(h.epochs.iter().map(|e| e.accepted).sum::<usize>() > 0);
}

/// Styblinski-Tang that is infeasible (NaN) on the half-plane `x0 > 1`.
struct InfeasibleHalfPlane {
    inner: StybTang2D,
}

impl Objective<f64> for InfeasibleHalfPlane {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if x[0] > 1.0 {
            f64::NAN
        } else {
            self.inner.eval(x)
        }
    }
}

#[test]
fn a_walk_started_on_an_infeasible_plateau_reaches_the_feasible_region() {
    for seed in 0..10 {
        let objective = InfeasibleHalfPlane {
            inner: StybTang2D::new(),
        };
        let h = run_rs_variant_from(
            boltzmann_in_box(objective, 1.0, 0.5).unwrap(),
            10,
            100,
            seed,
            Some(array![4.5, 0.0]),
            None,
        );
        assert!(h.best.val.is_finite(), "seed {seed} never left the plateau");
    }
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
