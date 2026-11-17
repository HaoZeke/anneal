// Copyright (c) anneal developers
//! `Config::branch_on_stall` is off unless asked for. A branch that never
//! fires leaves the walk as it was, and one that fires stays inside the
//! budget and repeats.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;

const ATOMS: usize = 19;
const SEED: u64 = 2000;

fn search(cfg: &Config, budget: usize) -> Outcome {
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(SEED);
    let start = random_cluster_in_radius(ATOMS, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ndarray::ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(SEED);
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

fn same(a: &Outcome, b: &Outcome) -> bool {
    a.best.to_bits() == b.best.to_bits()
        && a.final_energy.to_bits() == b.final_energy.to_bits()
        && a.charged == b.charged
        && a.hops == b.hops
        && a.improvements == b.improvements
        && a.accepted_transitions.len() == b.accepted_transitions.len()
}

#[test]
fn the_branch_is_off_and_unserialised_by_default() {
    let mut cfg = Config::recommended(ATOMS);
    assert!(!cfg.branch_on_stall);
    assert!(!cfg.resolved_json().unwrap().contains("branch_on_stall"));
    cfg.branch_on_stall = true;
    assert!(
        cfg.resolved_json()
            .unwrap()
            .contains("\"branch_on_stall\":true")
    );
}

#[test]
fn a_branch_that_never_fires_leaves_the_walk_unchanged() {
    let mut off = Config::recommended(ATOMS);
    off.restart_patience = usize::MAX / 4;
    let mut on = off.clone();
    on.branch_on_stall = true;
    let budget = 6_000;
    assert!(same(&search(&off, budget), &search(&on, budget)));
}

#[test]
fn a_branch_stays_inside_the_budget_and_repeats() {
    let mut off = Config::recommended(ATOMS);
    off.restart_patience = 200;
    let mut on = off.clone();
    on.branch_on_stall = true;
    let budget = 12_000;
    let walk = search(&off, budget);
    let first = search(&on, budget);
    let again = search(&on, budget);
    assert!(same(&first, &again));
    assert!(first.charged <= budget);
    // The branches spend calls the walk would have hopped with.
    assert!(
        first.hops < walk.hops,
        "{} hops against {}",
        first.hops,
        walk.hops
    );
    assert!(first.best.is_finite());
}
