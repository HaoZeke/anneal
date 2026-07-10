//! A6 witness: the runtime law sweep in `SaVariant::checked_with_sweep`
//! catches a deliberately broken third-party `AcceptRule`.

use anneal_core::accept::AcceptRule;
use anneal_core::cool::LogCool;
use anneal_core::laws::LawViolation;
use anneal_core::movekernel::Gaussian;
use anneal_core::neigh::ContinuousR_n;
use anneal_core::variant::{SaVariant, SweepBudget};

use eindir_core::objectives::StybTang2D;

/// A buggy acceptance rule that violates L3: returns 0 for all uphill
/// moves, but ALSO returns 0 for downhill moves (delta_e <= 0). The
/// The certified constructor rejects this type at compile time. The sampled
/// constructor catches the executable L3 violation.
struct AlwaysReject;

impl AcceptRule<f64> for AlwaysReject {
    fn accept_prob(&self, _delta_e: f64, _temp: f64) -> f64 {
        0.0
    }
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
