// Copyright (c) anneal developers
//! The cap turn joins the call-plateau leave only when `Config::cap_twist`
//! asks for it, and a chain that asks for it takes it.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn run(cfg: &Config, budget: usize, seed: u64) -> Outcome {
    let potential = PairPotential::lennard_jones(cfg.n_points);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        cfg.n_points,
        cfg.start_radius(),
        cfg.min_separation,
        &mut rng,
    );
    let mut ledger = Ledger::new(budget);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(point).1)
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

fn plateau_config(cap_twist: bool) -> Config {
    let mut cfg = Config::recommended(38);
    // Short plateaus, so a small budget sees several leaves.
    cfg.restart_patience = 4_000;
    cfg.cap_twist = cap_twist;
    cfg
}

#[test]
fn a_chain_that_asks_for_the_cap_turn_leaves_plateaus_through_it() {
    anneal_core::install_rgpot_kernels!();
    let outcome = run(&plateau_config(true), 80_000, 2003);
    assert!(
        outcome.cap_escapes >= 1,
        "no cap turn in {} stall escapes",
        outcome.stall_escapes
    );
    assert!(outcome.stall_escapes >= outcome.cap_escapes);
    assert!(outcome.best.is_finite());
}

#[test]
fn the_cap_turn_is_off_unless_asked_for() {
    anneal_core::install_rgpot_kernels!();
    let off = run(&plateau_config(false), 80_000, 2003);
    assert_eq!(off.cap_escapes, 0);
    assert!(off.stall_escapes > 0, "the plateau leave still runs");
}
