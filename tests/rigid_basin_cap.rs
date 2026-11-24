// Copyright (c) anneal developers
//! A rigid chain caps its basin index and must keep running past the cap.
//!
//! The rigid path seats at most a fixed number of basins. With a merge
//! radius small enough that every quench is a new basin, a run reaches the
//! cap early. Later unmatched quenches reuse the latest seat, so the run
//! neither aborts nor opens another basin.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, Outcome, run_with_gradient,
};
use anneal_core::potentials::random_tip4p_cluster;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Every quench returns its input at zero energy, so with a vanishing
/// merge radius every hop lands in a basin no seat holds.
fn run_identity_relax(cfg: &Config, start: &Array1<f64>, budget: usize, seed: u64) -> Outcome {
    let mut ledger = Ledger::new(budget);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, _iterations: usize| {
        let _ = led.charge();
        (0.0, x.to_owned())
    };
    let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
        let _ = led.charge();
        Some(Array1::zeros(point.len()))
    };
    let mut rng = StdRng::seed_from_u64(seed);
    run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start")
}

#[test]
fn a_rigid_chain_runs_on_past_its_basin_cap() {
    const RIGID_BASIN_CAP: usize = 2_048;
    let n = 4;
    let mut cfg = Config::for_tip4p(n);
    assert!(matches!(cfg.move_library, MoveLibrary::RigidBody { .. }));
    cfg.merge_radius = 1.0e-12;
    let mut source = StdRng::seed_from_u64(47);
    let start = random_tip4p_cluster(n, &mut source);
    let outcome = run_identity_relax(&cfg, &start, 20_000, 53);
    assert!(
        outcome.hops > RIGID_BASIN_CAP,
        "{} hops cannot reach the cap",
        outcome.hops
    );
    assert_eq!(outcome.basins, RIGID_BASIN_CAP);
    // A miss past the cap reuses the latest seat, so it is same or known.
    // Counting it as new is the refusal the cap used to do.
    let (same, known, new) = outcome.visit_counts;
    assert!(
        new <= RIGID_BASIN_CAP,
        "new visits {new} exceed the cap; a later basin was not aliased"
    );
    assert!(
        same + known > 0,
        "no revisit was recorded, so the cap was not reused"
    );
    assert!(outcome.best.is_finite());
}

#[test]
fn the_recommended_tip4p_tetramer_runs_on_past_its_basin_cap() {
    const RIGID_BASIN_CAP: usize = 2_048;
    let n = 4;
    let mut cfg = Config::recommended_tip4p(n);
    assert!(matches!(cfg.move_library, MoveLibrary::RigidBody { .. }));
    cfg.merge_radius = 1.0e-12;
    let mut source = StdRng::seed_from_u64(59);
    let start = random_tip4p_cluster(n, &mut source);
    let outcome = run_identity_relax(&cfg, &start, 10_000, 61);
    assert!(outcome.hops > RIGID_BASIN_CAP, "{} hops", outcome.hops);
    assert_eq!(outcome.basins, RIGID_BASIN_CAP);
    let (same, known, new) = outcome.visit_counts;
    assert!(
        new <= RIGID_BASIN_CAP,
        "new visits {new} exceed the cap; a later basin was not aliased"
    );
    assert!(
        same + known > 0,
        "no revisit was recorded, so the cap was not reused"
    );
    assert!(outcome.best.is_finite());
}
