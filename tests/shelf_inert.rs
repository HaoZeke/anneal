//! A recommended search that ends before any call plateau.
//!
//! The shelf quarantine records a geometry only after that plateau, and
//! must not move a walk that never reaches one. Best, hops and charged
//! calls are the trajectory of `develop` at
//! `bc16d35cf13e71c2edaf343b80fd8c41368631d4`, LJ13, seed 2000, 1500 calls.
//! `search_lennard_jones` is the monomorphized scalar walk.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search_lennard_jones;
use anneal_core::potentials::PairPotential;

#[test]
fn lj13_before_a_call_plateau_matches_develop() {
    anneal_core::install_rgpot_kernels!();
    let n = 13usize;
    let budget = 1500usize;
    let cfg = Config::recommended(n);
    assert!(
        cfg.restart_patience > budget,
        "budget {} reaches the call plateau at {}",
        budget,
        cfg.restart_patience
    );
    assert!(cfg.escape_stall_patience > 100);
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    let (out, _) = search_lennard_jones(&pot, &cfg, &mut ledger, 2000).expect("finite start");
    assert_eq!(
        out.best.to_bits(),
        0xc04629d4a10097a8,
        "best {:.17} hops {} charged {}",
        out.best,
        out.hops,
        out.charged
    );
    assert_eq!(out.hops, 25);
    assert_eq!(out.charged, budget);
    assert_eq!(ledger.spent(), budget);
    assert_eq!(out.tabu, (0, 0), "a plateau armed on a short walk");
}
