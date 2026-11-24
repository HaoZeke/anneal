// Copyright (c) anneal developers
//! The generic packing probe's potential calls are all on the ledger.
//!
//! Lennard-Jones 26, seed 2000, budget 30_000, stops at -108.315616.
//! The probe spends 931 potential calls before the walk. develop
//! `983f6f69ea1555e51a71874870410966ac726616` has no probe and charges 502
//! at the hit. With the probe the same geometry is hit at 502 + 931 = 1433,
//! and the ledger, the improvement row and `outcome.charged` all say so.

use anneal_core::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicUsize, Ordering};

const ATOMS: usize = 26;
const SEED: u64 = 2000;
const BUDGET: usize = 30_000;
const REFERENCE: f64 = -108.315616;

/// The energy-lattice arm charges its site scans straight to the ledger,
/// with no potential call. This walk is charged four calls that way. develop
/// shows the same four: 498 potential calls against 502 charged.
const SITE_SCAN_CHARGE: usize = 4;

#[test]
fn a_walk_hit_is_charged_for_every_potential_call_including_the_packing_probe() {
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
    let evaluations = AtomicUsize::new(0);
    let mut ledger = Ledger::new(BUDGET);
    ledger.stop_below(REFERENCE + anneal_core::DEFAULT_HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            evaluations.fetch_add(1, Ordering::Relaxed);
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ndarray::ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        evaluations.fetch_add(1, Ordering::Relaxed);
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
    assert_eq!(at_hit, Some(1433), "hit spend");
    assert_eq!(outcome.charged, 1433, "charged calls");
    assert_eq!(outcome.charged, ledger.spent(), "outcome and ledger agree");
    assert_eq!(
        evaluations.load(Ordering::Relaxed) + SITE_SCAN_CHARGE,
        ledger.spent(),
        "every potential call is charged"
    );
}
