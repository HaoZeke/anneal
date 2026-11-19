// Copyright (c) anneal developers
//! `plateau_heat` is an acceptance temperature. Each test pins a walk
//! that moves when one heated site is reverted.

use anneal_core::methods::cluster_hopping::{
    Config, ContinuousSymmetry, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;

const ATOMS: usize = 26;

fn walk(cfg: &Config, budget: usize, seed: u64) -> Outcome {
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(seed);
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

/// FNV-1a over the accepted edges. Stable for this toolchain, unlike
/// `std` hashing.
fn fingerprint(outcome: &Outcome) -> u64 {
    let mut state = 0xcbf29ce484222325u64;
    let mut mix = |bytes: &[u8]| {
        for byte in bytes {
            state ^= *byte as u64;
            state = state.wrapping_mul(0x100000001b3);
        }
    };
    for edge in &outcome.accepted_transitions {
        mix(&edge.hop.to_le_bytes());
        mix(edge.action.as_bytes());
        mix(&edge.from_energy.to_bits().to_le_bytes());
        mix(&edge.to_energy.to_bits().to_le_bytes());
    }
    mix(&outcome.best.to_bits().to_le_bytes());
    mix(&(outcome.hops as u64).to_le_bytes());
    mix(&(outcome.charged as u64).to_le_bytes());
    state
}

fn pair(cfg: &Config, budget: usize, seed: u64) -> (Outcome, Outcome) {
    let mut hot = cfg.clone();
    hot.plateau_heat = true;
    (walk(cfg, budget, seed), walk(&hot, budget, seed))
}

#[test]
fn a_chain_without_a_plateau_walks_the_preset_trajectory() {
    let budget = 30_000;
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = budget * 10;
    let (cold, hot) = pair(&cfg, budget, 2000);
    assert_eq!(hot.best, cold.best);
    assert_eq!(hot.hops, cold.hops);
    assert_eq!(hot.charged, cold.charged);
    assert_eq!(hot.accepted_transitions, cold.accepted_transitions);
}

#[test]
fn metropolis_heat_follows_the_plateau() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = 2_000;
    cfg.energy_lattice = false;
    let (cold, hot) = pair(&cfg, 30_000, 2000);
    let cold_fp = fingerprint(&cold);
    let hot_fp = fingerprint(&hot);
    assert_eq!(cold_fp, 0x89cf5273a413bb1c, "preset walk");
    assert_ne!(hot_fp, cold_fp, "temperature *= heat");
    assert_eq!(hot_fp, 0x89a108df8ad62f0c, "PLATEAU_HEAT_CAP");
}

#[test]
fn lattice_exchange_follows_the_plateau() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = 2_000;
    let (cold, hot) = pair(&cfg, 30_000, 2000);
    assert_eq!(fingerprint(&cold), 0x0e954b89e974b441, "preset walk");
    assert_eq!(
        fingerprint(&hot),
        0x62b6612c4728920f,
        "heat * cfg.temperature"
    );
}

/// Seed 2 reaches a new lowest while the heat is above 1, and the walk
/// differs when that reset is left out. Seeds where the reset never fires
/// pass whether or not it exists.
#[test]
fn a_new_lowest_returns_the_walk_to_the_preset() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = 80;
    cfg.relax_steps = 20;
    cfg.energy_lattice = false;
    let (cold, hot) = pair(&cfg, 8_000, 2);
    assert_ne!(fingerprint(&hot), fingerprint(&cold), "heat moves the walk");
    assert_eq!(fingerprint(&cold), 0xc0cf0528311a79f2, "preset walk");
    assert_eq!(fingerprint(&hot), 0x68a82c0212136f37, "heat = 1.0");
}

/// Seed 76 lowers the best from a quench after the loop head, with the heat
/// above 1, and the walk differs when that reset is left out.
#[test]
fn a_same_iteration_record_returns_to_the_preset() {
    let mut cfg = Config::recommended(ATOMS);
    cfg.restart_patience = 60;
    cfg.relax_steps = 40;
    cfg.general_collective = true;
    cfg.continuous_symmetry = ContinuousSymmetry::Measure { interval: 1 };
    let (cold, hot) = pair(&cfg, 6_000, 76);
    assert_ne!(fingerprint(&hot), fingerprint(&cold), "heat moves the walk");
    assert_eq!(fingerprint(&cold), 0x4e21f286e8d17063, "preset walk");
    assert_eq!(
        fingerprint(&hot),
        0x3ae26ce650b0fc5c,
        "same-iteration heat = 1.0"
    );
}
