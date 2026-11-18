// Copyright (c) anneal developers
//! The cap turn joins the call-plateau leave only when `Config::cap_twist`
//! asks for it, and a chain that asks for it takes it.

use anneal_core::methods::cluster_hopping::{
    Config, ContinuousSymmetry, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
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
    // A turn that finds no axis is redrawn as a climb, and all ten cap
    // draws in this seed's first 80k calls find none.
    let outcome = run(&plateau_config(true), 160_000, 2003);
    assert!(
        outcome.cap_escapes >= 1,
        "no cap turn in {} stall escapes",
        outcome.stall_escapes
    );
    assert!(outcome.stall_escapes >= outcome.cap_escapes);
    let cap = outcome
        .proposal_counts
        .iter()
        .find(|row| row.move_type == "cap")
        .expect("cap arm accounting");
    assert!(cap.proposed >= cap.accepted);
    assert_eq!(cap.accepted, outcome.cap_escapes);
    assert!(outcome.best.is_finite());
}

#[test]
fn the_cap_turn_is_off_unless_asked_for() {
    anneal_core::install_rgpot_kernels!();
    let off = run(&plateau_config(false), 80_000, 2003);
    assert_eq!(off.cap_escapes, 0);
    assert!(off.stall_escapes > 0, "the plateau leave still runs");
}

fn cap_only_config(n: usize) -> Config {
    let mut cfg = Config::recommended(n);
    cfg.trail_on_stall = false;
    cfg.escape_on_stall = false;
    cfg.restart_on_stall = false;
    cfg.continuous_symmetry = ContinuousSymmetry::Off;
    cfg.tabu_on_stall = false;
    cfg.cap_twist = true;
    cfg.escape_stall_patience = 1;
    cfg.escape_stall_factor = 0.0;
    cfg
}

#[test]
fn an_identity_cap_mutation_is_not_an_escape() {
    let start = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let outcome = run_identity_relax(&cap_only_config(3), &start, 80, 7);
    assert_eq!(outcome.cap_escapes, 0);
    assert_eq!(outcome.stall_escapes, 0);
    let cap = outcome
        .proposal_counts
        .iter()
        .find(|row| row.move_type == "cap")
        .expect("cap arm accounting");
    assert!(cap.proposed > 0);
    assert_eq!(cap.accepted, 0);
}

#[test]
fn frozen_atoms_are_bit_identical_when_cap_twist_is_requested() {
    let mut source = StdRng::seed_from_u64(11);
    let start = random_cluster_in_radius(13, 2.0, 0.7, &mut source);
    let mut cfg = cap_only_config(13);
    cfg.frozen = Some(vec![true; 13]);
    let outcome = run_identity_relax(&cfg, &start, 80, 19);
    let final_state = outcome.final_state.expect("final state");
    assert_eq!(final_state.len(), start.len());
    for (actual, expected) in final_state.iter().zip(&start) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(outcome.cap_escapes, 0);
}

#[test]
fn an_atomic_cap_search_keeps_every_basin() {
    const BASIN_CAP: usize = 2_048;
    let mut source = StdRng::seed_from_u64(23);
    let start = random_cluster_in_radius(4, 1.4, 0.4, &mut source);
    let mut cfg = cap_only_config(4);
    cfg.merge_radius = 1.0e-12;
    let outcome = run_identity_relax(&cfg, &start, 10_000, 29);
    assert!(
        outcome.basins > BASIN_CAP,
        "{} basins show that the atomic index was capped",
        outcome.basins
    );
}

#[test]
fn a_default_atomic_path_never_caps() {
    const RIGID_BASIN_CAP: usize = 2_048;
    let mut source = StdRng::seed_from_u64(31);
    let start = random_cluster_in_radius(4, 1.4, 0.4, &mut source);
    let mut cfg = Config::recommended(4);
    assert!(!cfg.cap_twist);
    cfg.merge_radius = 1.0e-12;
    let outcome = run_identity_relax(&cfg, &start, 10_000, 37);
    assert!(
        outcome.basins > RIGID_BASIN_CAP,
        "{} basins show that the default atomic index was capped",
        outcome.basins
    );
}
