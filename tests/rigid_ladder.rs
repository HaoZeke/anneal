//! A replica ladder on a rigid-body chart.

use anneal_core::methods::cluster_hopping::{Config, LadderMode, Ledger, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Tip4pCluster, random_tip4p_cluster};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn every_rung_of_a_rigid_ladder_starts_on_the_rigid_chart() {
    let n = 4;
    let pot = Tip4pCluster::new(n);
    let mut cfg = Config::for_tip4p(n);
    cfg.replicas = 4;
    cfg.ladder_mode = LadderMode::Independent;
    let mut rng = StdRng::seed_from_u64(11);
    let start = random_tip4p_cluster(n, &mut rng);
    let mut opt = WarmLbfgs::default();
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        assert_eq!(x.len(), 6 * n, "a rung left the rigid chart");
        opt.forget();
        let (f, xr, _) = opt.minimize(x, iters, |v| {
            if !ledger.charge() {
                return None;
            }
            Some(pot.value_and_gradient(v))
        });
        (f, xr)
    };
    let mut grad = |ledger: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
        ledger.charge().then(|| pot.value_and_gradient(x).1)
    };
    let mut ledger = Ledger::new(4_000);
    let out = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    assert_eq!(out.rungs.len(), 4);
    assert!(out.hops > 0);
}
