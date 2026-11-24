// Copyright (c) anneal developers
//! The generic packing catalogue is not part of a hit the walk records.
//!
//! Lennard-Jones 26, seed 2000, budget 30_000, stops at -108.315616.
//! develop `9b65fda3972fd8c014bce886a3ea5c05956d64af` charges 502 at that
//! hit. Settling the catalogue onto the parent before the walk charged
//! 1433 for the same geometry and the same hop.

use anneal_core::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;

const ATOMS: usize = 26;
const SEED: u64 = 2000;
const BUDGET: usize = 30_000;
const REFERENCE: f64 = -108.315616;

#[test]
fn a_walk_hit_keeps_develop_charged_count() {
    anneal_core::install_rgpot_kernels!();
    let cfg = Config::recommended(ATOMS);
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(SEED);
    let start = anneal_core::methods::cluster_hopping::random_cluster_in_radius(
        ATOMS,
        cfg.start_radius(),
        cfg.min_separation,
        &mut rng,
    );
    let mut ledger = Ledger::new(BUDGET);
    ledger.stop_below(REFERENCE + anneal_core::DEFAULT_HIT_TOL);
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
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let at_hit = outcome
        .improvements
        .iter()
        .find(|row| anneal_core::energy_hits(row.3, REFERENCE, anneal_core::DEFAULT_HIT_TOL))
        .map(|row| row.1);
    assert!(
        anneal_core::energy_hits(outcome.best, REFERENCE, anneal_core::DEFAULT_HIT_TOL),
        "best {:.6}",
        outcome.best
    );
    assert_eq!(at_hit, Some(502), "hit spend");
    assert_eq!(outcome.charged, 502, "charged calls");
}
