//! A hop cap bounds every hop the chain takes.
//!
//! Delayed acceptance rejects a trial at its first stage and moves on without
//! paying the quench. That rejection is still a hop, so a run whose cap falls
//! on one has to stop there like any other.

use anneal_core::methods::cluster_hopping::{Config, Ledger, Outcome, optimize_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};

fn lj_run(cfg: &Config, seed: u64) -> Outcome {
    let pot = PairPotential::lennard_jones(cfg.n_points);
    let mut ledger = Ledger::new(1_000_000);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(pot.value_and_gradient(v))
            },
            |_, _| true,
        );
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(pot.value_and_gradient(x).1)
    };
    optimize_with_gradient(cfg, &mut ledger, &mut relax, Some(&mut grad), seed)
        .expect("finite start")
}

#[test]
fn a_stage_one_reject_counts_against_the_hop_cap() {
    let mut cfg = Config::for_cluster(13);
    cfg.delayed_acceptance = true;
    // A surrogate that never abstains decides every hop after its warmup, so
    // the first stage rejects often enough for a cap to fall on a rejection.
    cfg.surrogate_tolerance = f64::INFINITY;
    let mut stage_one_rejects = 0;
    // The surrogate abstains through its first 64 quenches, so the caps start
    // past that.
    for cap in 80..=140 {
        cfg.max_hops = Some(cap);
        let out = lj_run(&cfg, 7);
        assert!(
            out.hops <= cap,
            "{} hops ran under a cap of {cap}",
            out.hops
        );
        stage_one_rejects += out.delayed.map_or(0, |d| d.1);
    }
    assert!(
        stage_one_rejects > 0,
        "the first stage never rejected, so the cap was not tested against it"
    );
}
