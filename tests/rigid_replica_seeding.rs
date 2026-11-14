//! A replica ladder on a rigid chart seeds its extra rungs as rigid bodies.
//!
//! A rigid-body chart holds six coordinates per body: three centre, three
//! rotation. The extra rungs of a replica ladder used to draw a plain centre
//! cluster, three per body, and the six-per-body potential rejected the length
//! the moment it quenched that rung's start. A rigid ladder therefore panicked
//! at startup for any `replicas > 1`. The run below seeds four rigid rungs and
//! must reach a finite best without panicking.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, run_with_energy_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Tip4pCluster, random_tip4p_cluster};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn a_rigid_replica_ladder_seeds_extra_rungs_as_rigid_bodies() {
    let n = 8;
    let mut cfg = Config::for_tip4p(n);
    assert!(
        matches!(cfg.move_library, MoveLibrary::RigidBody { .. }),
        "for_tip4p moves rigid bodies"
    );
    // More than one rung is what drives the extra-rung seeding; before the fix
    // this alone panicked.
    cfg.replicas = 4;

    let pot = Tip4pCluster::new(n);
    let mut rng = StdRng::seed_from_u64(20_261_114);
    let start = random_tip4p_cluster(n, &mut rng);
    assert_eq!(start.len(), 6 * n, "a rigid start is six per body");

    let mut ledger = Ledger::new(20_000);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, mut xr, _) = opt.minimize(x, iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(pot.value_and_gradient(v))
        });
        if let Some(slice) = xr.as_slice_mut() {
            pot.fold_rotations(slice);
        }
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(pot.value_and_gradient(x).1)
    };

    let out = run_with_energy_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        None,
        &mut rng,
    )
    .expect("finite start");

    assert!(out.best.is_finite(), "the ladder returns a finite best");
    assert!(
        out.best_state.as_ref().map(Array1::len) == Some(6 * n),
        "the reported state is six coordinates per body"
    );
}
