//! GOAT's conformer ensemble and entropy stop on the cluster hop.

use anneal_core::methods::cluster_hopping::{Config, HopStop, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;

#[test]
fn the_report_counts_minima_inside_the_window() {
    let pot = PairPotential::lennard_jones(13);
    let mut cfg = Config::for_cluster(13);
    cfg.ensemble_report = true;
    let mut ledger = Ledger::new(20_000);
    let (out, _) = search(&pot, &cfg, &mut ledger, 2).expect("finite start");
    let ens = out.conformers.expect("report on");
    assert!(ens.members >= 1);
    assert!(ens.entropy >= 0.0 && ens.entropy <= (ens.members as f64).ln() + 1e-12);
    assert!(ens.free_energy <= 1e-12);
    assert_eq!(ens.temperature, cfg.conformer_temperature());
    assert_eq!(ens.rounds, 0);
}

#[test]
fn the_entropy_stop_ends_an_lj13_run_early() {
    let pot = PairPotential::lennard_jones(13);
    let mut cfg = Config::for_cluster(13);
    cfg.entropy_stop = true;
    let budget = 2_000_000;
    let mut ledger = Ledger::new(budget);
    let (out, _) = search(&pot, &cfg, &mut ledger, 4).expect("finite start");
    assert_eq!(out.stop, Some(HopStop::EntropyConverged));
    assert!(out.charged < budget);
    assert!(out.conformers.expect("report on").rounds >= 3);
    assert!(out.best < -44.32);
}

#[test]
fn rounds_and_temperature_follow_the_chart() {
    // A water decamer has 30 atoms in 10 fragments: 260 optimisations on
    // each of four workers.
    let water = Config::for_tip4p(10);
    assert_eq!(water.entropy_round(), 1040);
    assert!((water.conformer_temperature() - 2.4790).abs() < 1e-4);
    // An atomic cluster is one fragment: 0.5 * 75^1.5 rounds up to 325.
    let lj = Config::for_cluster(75);
    assert_eq!(lj.entropy_round(), 1300);
    assert!((lj.conformer_temperature() - 0.5925).abs() < 1e-4);
}

#[test]
fn without_the_flags_there_is_no_report() {
    let pot = PairPotential::lennard_jones(13);
    let cfg = Config::for_cluster(13);
    let mut ledger = Ledger::new(5_000);
    let (out, _) = search(&pot, &cfg, &mut ledger, 2).expect("finite start");
    assert!(out.conformers.is_none());
    assert_ne!(out.stop, Some(HopStop::EntropyConverged));
}
