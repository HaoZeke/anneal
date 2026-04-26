//! Integration test: `run_rs` on the Boltzmann variant of Styblinski-Tang
//! 2D. Two acceptance criteria:
//!   1. Final `History.best.val` is below 0 (better than uniform-random
//!      sampling on `StybTang2D`'s `[-5, 5]^2` domain, which sees a mean
//!      objective of approximately +50).
//!   2. Determinism: two runs with the same seed produce bitwise-identical
//!      `History.best.pos` and identical per-epoch counters.

use anneal_core::run_rs;
use anneal_core::variant::boltzmann;

use eindir_core::objectives::StybTang2D;

const N_EPOCHS: usize = 100;
const STEPS_PER_EPOCH: usize = 200;
const SEED: u64 = 42;

#[test]
fn run_rs_styb_tang_finds_negative_minimum() {
    let v = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("Boltzmann construction");
    let h = run_rs(v, N_EPOCHS, STEPS_PER_EPOCH, SEED);
    assert!(
        h.best.val < 0.0,
        "Boltzmann SA should find a negative value on StybTang2D; got {}",
        h.best.val
    );
}

#[test]
fn run_rs_styb_tang_is_deterministic() {
    let h1 = run_rs(
        boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant"),
        N_EPOCHS,
        STEPS_PER_EPOCH,
        SEED,
    );
    let h2 = run_rs(
        boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant"),
        N_EPOCHS,
        STEPS_PER_EPOCH,
        SEED,
    );
    assert_eq!(h1.best.pos, h2.best.pos, "best.pos must be reproducible");
    assert_eq!(h1.best.val, h2.best.val, "best.val must be reproducible");
    for (a, b) in h1.epochs.iter().zip(h2.epochs.iter()) {
        assert_eq!(a.accepted, b.accepted);
        assert_eq!(a.rejected, b.rejected);
        assert_eq!(a.best_val, b.best_val);
    }
}
