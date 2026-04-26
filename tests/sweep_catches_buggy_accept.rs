//! A6 witness: the runtime law sweep in `SaVariant::checked_with_sweep`
//! catches a deliberately-broken `AcceptRule` that the pre-A6 Boolean
//! witness (just a trait method) would have admitted as valid.

use anneal_core::accept::AcceptRule;
use anneal_core::cool::LogCool;
use anneal_core::laws::LawViolation;
use anneal_core::movekernel::Gaussian;
use anneal_core::neigh::ContinuousR_n;
use anneal_core::variant::{SaVariant, SweepBudget};

use eindir_core::objectives::StybTang2D;

/// A buggy acceptance rule that violates L3: returns 0 for all uphill
/// moves, but ALSO returns 0 for downhill moves (delta_e <= 0). The
/// pre-A6 SaVariant::checked has no method that catches this -- there
/// is no `accept_rule.is_l3_compliant()` Boolean -- so it constructed
/// cleanly. The A6 sweep catches it on the first downhill sample.
struct AlwaysReject;

impl AcceptRule<f64> for AlwaysReject {
    fn accept_prob(&self, _delta_e: f64, _temp: f64) -> f64 {
        0.0
    }
}

#[test]
fn checked_admits_buggy_accept_pre_a6() {
    let result = SaVariant::checked(
        StybTang2D::new(),
        LogCool::new(1.0_f64, 2.0),
        ContinuousR_n::new(2),
        Gaussian::new(0.5),
        AlwaysReject,
    );
    // The pre-A6 cheap path returns Ok: AlwaysReject has no Boolean
    // witness that catches the L3 violation, and SaVariant::checked
    // only inspects cool.is_monotone, neigh.is_symmetric, mover.supports_in.
    assert!(
        result.is_ok(),
        "checked() incorrectly rejected (pre-A6 baseline)"
    );
    drop(result);
}

#[test]
fn checked_with_sweep_rejects_buggy_accept() {
    let result = SaVariant::checked_with_sweep(
        StybTang2D::new(),
        LogCool::new(1.0_f64, 2.0),
        ContinuousR_n::new(2),
        Gaussian::new(0.5),
        AlwaysReject,
        SweepBudget::Default,
        2,
        5.0,
        0,
    );
    match result {
        Err(LawViolation::DownhillNotAccepted { p, .. }) => {
            assert!(
                (p - 0.0).abs() < 1e-12,
                "expected p=0 from AlwaysReject, got {p}"
            );
        }
        Err(other) => panic!("expected DownhillNotAccepted, got {other:?}"),
        Ok(_) => panic!("expected Err(DownhillNotAccepted), got Ok(SaVariant)"),
    }
}

#[test]
fn checked_with_sweep_passes_legitimate_variants() {
    // Boltzmann preset must pass the sweep since every component
    // satisfies its law. This pins SweepBudget::Default as a
    // non-trivial smoke test on every CI run.
    let result = SaVariant::checked_with_sweep(
        StybTang2D::new(),
        LogCool::new(1.0_f64, 2.0),
        ContinuousR_n::new(2),
        Gaussian::new(0.5),
        anneal_core::accept::Metropolis,
        SweepBudget::Default,
        2,
        5.0,
        42,
    );
    if let Err(e) = result {
        panic!("Boltzmann preset failed sweep: {e:?}");
    }
}
