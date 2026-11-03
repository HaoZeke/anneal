//! GOAT's conformer ensemble and entropy stop on the cluster hop.

use std::sync::Mutex;

use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_hopping::{Config, HopStop, Ledger, random_cluster_in_radius};
use anneal_core::methods::cluster_search::search;
use anneal_core::methods::ensemble::{
    EnsembleConfig, EnsembleProblem, EnsembleReport, HistoryMode, ObjectiveFactory, StartFactory,
    run_ensemble,
};
use anneal_core::methods::minima_hopping::{HistoryMembership, SerializedWitness};
use anneal_core::pes_exploration::StructureContext;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;

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

fn sorted_pairs(x: ArrayView1<f64>) -> Vec<f64> {
    let n = x.len() / 3;
    let mut d = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let r2: f64 = (0..3).map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2)).sum();
            d.push(r2.sqrt());
        }
    }
    d.sort_by(f64::total_cmp);
    d
}

fn lj13_ensemble(
    cfg: &Config,
    history: HistoryMode,
    budget: usize,
) -> Result<EnsembleReport, String> {
    let n = 13;
    let pot = PairPotential::lennard_jones(n);
    let ens = EnsembleConfig {
        replicas: 2,
        budget,
        history,
        membership: HistoryMembership::Accepted,
        shared_bias: None,
        gossip: None,
        two_choice_stall: None,
        checkpoint_interval: 1_000,
        target: None,
    };
    let descriptor = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let context = StructureContext::new(Some(vec![18; n]), None, Some("goat-lj13".into()));
    let witness = SerializedWitness(Mutex::new(|a: ArrayView1<f64>, b: ArrayView1<f64>| {
        sorted_pairs(a)
            .iter()
            .zip(sorted_pairs(b))
            .all(|(p, q)| (p - q).abs() < 1e-3)
    }));
    let objective: ObjectiveFactory<'_> = &|_| Box::new(|x| pot.value_and_gradient(x));
    let start: StartFactory<'_> =
        &|_, rng| random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, rng);
    run_ensemble(
        cfg,
        &ens,
        5,
        &EnsembleProblem {
            objective,
            start,
            descriptor: &descriptor,
            context: &context,
            witness: &witness,
            same_family: &|_, _| false,
            certificate: 1e-5,
            polish_below: 1e-3,
            callbacks_per_objective: 1,
        },
    )
}

#[test]
fn an_ensemble_reads_its_conformers_from_the_shared_history() {
    let mut cfg = Config::for_cluster(13);
    cfg.ensemble_report = true;
    let report = lj13_ensemble(&cfg, HistoryMode::Shared, 40_000).expect("shared history");
    let conformers = report.conformers.expect("report on");
    let (minima, _, _) = report.histories[0];
    assert!(conformers.members >= 1 && conformers.members <= minima);
    assert_eq!(conformers.temperature, cfg.conformer_temperature());
    assert_eq!(conformers.rounds, 0);
    for replica in &report.replicas {
        assert!(replica.outcome.conformers.is_none());
    }
}

#[test]
fn the_shared_entropy_stop_retires_every_replica() {
    let mut cfg = Config::for_cluster(13);
    cfg.entropy_stop = true;
    let budget = 2_000_000;
    let report = lj13_ensemble(&cfg, HistoryMode::Shared, budget).expect("shared history");
    assert!(report.conformers.expect("report on").rounds >= 3);
    assert!(report.aggregate_charged < budget);
    for replica in &report.replicas {
        assert_eq!(
            replica.outcome.stop,
            Some(HopStop::Retired {
                reason: "entropy_converged".into()
            })
        );
    }
}

#[test]
fn a_private_history_refuses_the_conformer_ensemble() {
    let mut cfg = Config::for_cluster(13);
    cfg.ensemble_report = true;
    assert!(lj13_ensemble(&cfg, HistoryMode::Private, 4_000).is_err());
    assert!(lj13_ensemble(&cfg, HistoryMode::None, 4_000).is_err());
}
