//! GOAT's gradient boost on the cluster hop.

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, run};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

const FULL: usize = 2;

fn bare() -> Config {
    let mut cfg = Config::for_cluster(3);
    cfg.move_library = MoveLibrary::WalesDoye;
    cfg.angular_moves = false;
    cfg.return_screen = false;
    cfg.bayes_screen = false;
    cfg.calibrated_screen = false;
    cfg.screen_margin = f64::INFINITY;
    cfg.screen_steps = 1;
    cfg.relax_steps = FULL;
    cfg.polish_records = 0;
    cfg.orbit_complete_on_new = false;
    cfg.surfaces.clear();
    cfg.merge_radius = 1e-6;
    cfg.max_hops = Some(90);
    cfg
}

fn triangle(scale: f64) -> Array1<f64> {
    array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] * scale
}

/// Forty hops alternating a new high minimum with a return home, thirty
/// returns home, then one high minimum offered over and over.
fn scripted(cfg: &Config) -> anneal_core::methods::cluster_hopping::Outcome {
    let mut full = 0usize;
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        let _ = ledger.charge();
        if steps != FULL {
            return (1.0e3, x.to_owned());
        }
        full += 1;
        let k = full;
        if k <= 1 {
            (0.0, triangle(1.0))
        } else if k <= 41 && k.is_multiple_of(2) {
            (1.0e3, triangle(1.0 + 0.01 * k as f64))
        } else if k <= 71 {
            (0.0, triangle(1.0))
        } else {
            (1.0e3, triangle(2.0))
        }
    };
    let mut rng = StdRng::seed_from_u64(11);
    let mut ledger = Ledger::new(100_000);
    run(cfg, triangle(1.0).view(), &mut ledger, &mut relax, &mut rng).expect("finite start")
}

#[test]
fn a_stalled_walker_takes_the_next_new_minimum() {
    let mut cfg = bare();
    cfg.boost_on_stall = true;
    let out = scripted(&cfg);
    assert!(out.boosts >= 1);
    assert_eq!(out.final_energy, 1.0e3);
}

#[test]
fn without_the_boost_the_high_minimum_is_refused() {
    let out = scripted(&bare());
    assert_eq!(out.boosts, 0);
    assert_eq!(out.final_energy, 0.0);
}

#[test]
fn a_boosted_search_still_finds_lj13() {
    let pot = PairPotential::lennard_jones(13);
    let mut cfg = Config::for_cluster(13);
    cfg.boost_on_stall = true;
    let mut ledger = Ledger::new(40_000);
    let (out, _) = search(&pot, &cfg, &mut ledger, 5).expect("finite start");
    assert!(out.best < -44.32);
}
