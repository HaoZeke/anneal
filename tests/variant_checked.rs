//! Tests for `SaVariant::checked`: happy path on the three IISE-manuscript
//! preset variants (Boltzmann, Fast, GSA), and a negative path that
//! constructs a deliberately broken `Neighborhood` to confirm the
//! `LawViolation::Symmetry` arm fires.

use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::laws::LawViolation;
use anneal_core::movekernel::{Gaussian, Reflected};
use anneal_core::neigh::{BoxConstrained, Neighborhood};
use anneal_core::variant::{SaVariant, SweepBudget, ValidationEvidence, boltzmann, fast, gsa};

use eindir_core::objectives::StybTang2D;
use eindir_core::{Bounds, Objective};
use ndarray::{ArrayView1, array};

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
