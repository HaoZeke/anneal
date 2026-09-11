use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_hopping::Config;
use anneal_core::methods::ensemble::{
    EnsembleConfig, EnsembleProblem, EnsembleReport, GossipConfig, GossipTopology, HistoryMode,
    ObjectiveFactory, StartFactory, run_ensemble,
};
use anneal_core::methods::minima_hopping::{HistoryMembership, SerializedWitness};
use anneal_core::pes_exploration::StructureContext;
use ndarray::{Array1, ArrayView1, array};

fn tetrahedron() -> Array1<f64> {
    array![
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0
    ]
}

fn run(shared: bool, gossip: bool, restart: bool) -> EnsembleReport {
    let mut cfg = Config::for_cluster(4);
    cfg.screen_steps = 1;
    cfg.relax_steps = 120;
    cfg.screen_margin = f64::INFINITY;
    cfg.return_screen = false;
    cfg.max_hops = Some(64);
    let ens = EnsembleConfig {
        replicas: 3,
        budget: 90_000,
        history: HistoryMode::None,
        membership: HistoryMembership::Accepted,
        shared_bias: shared.then_some(0.5),
        gossip: gossip.then_some(GossipConfig {
            topology: GossipTopology::Ring,
            interval: 1,
            weight: 0.5,
            adaptive: false,
            top: None,
        }),
        two_choice_stall: restart.then_some(1),
        checkpoint_interval: 1,
        target: None,
    };
    let descriptor = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let context = StructureContext::new(Some(vec![18; 4]), None, Some("channel-contract".into()));
    let witness = SerializedWitness(Mutex::new(|a: ArrayView1<f64>, b: ArrayView1<f64>| a == b));
    let calls = AtomicUsize::new(0);
    let objective: ObjectiveFactory<'_> = &|_| {
        let calls = &calls;
        Box::new(move |x| {
            calls.fetch_add(1, Ordering::Relaxed);
            let delta = &x - &tetrahedron();
            (delta.dot(&delta), 2.0 * delta)
        })
    };
    let start: StartFactory<'_> = &|_, _| tetrahedron();
    let report = run_ensemble(
        &cfg,
        &ens,
        7,
        &EnsembleProblem {
            objective,
            start,
            descriptor: &descriptor,
            context: &context,
            witness: &witness,
            same_family: &|_, _| true,
            certificate: 1e-5,
            polish_below: 1e-3,
            callbacks_per_objective: 1,
        },
    )
    .unwrap();
    assert_eq!(report.aggregate_charged, calls.load(Ordering::Relaxed));
    assert!(report.aggregate_charged <= ens.budget);
    assert_eq!(
        report.aggregate_charged,
        report
            .replicas
            .iter()
            .map(|replica| replica.charged)
            .sum::<usize>()
    );
    assert!(report.histories.is_empty());
    assert_eq!(report.best, 0.0);
    report
}

#[test]
fn every_completed_checkpoint_publishes_local_visits_during_gossip() {
    let report = run(true, true, false);
    assert!(report.replicas.iter().any(|r| r.outcome.gossip_rounds > 0));
    for replica in &report.replicas {
        assert_eq!(replica.outcome.hops, 64);
        assert_eq!(
            replica.bias_published + 1,
            replica.outcome.hops as u64,
            "replica {} lost local visit publications while gossiping",
            replica.replica
        );
    }
    assert!(report.exchange.1 > 0);
}

#[test]
fn restarts_do_not_suppress_gossip_or_local_visit_publication() {
    let report = run(true, true, true);
    assert!(report.replicas.iter().any(|r| r.two_choice_restarts > 0));
    assert!(report.replicas.iter().any(|r| r.outcome.gossip_rounds > 0));
    assert!(report.exchange.1 > 0);
    for replica in &report.replicas {
        assert!(
            replica.bias_published + 2 >= replica.outcome.hops as u64,
            "replica {} lost local visit publications during restarts: {} for {} hops",
            replica.replica,
            replica.bias_published,
            replica.outcome.hops
        );
    }
}

#[test]
fn the_deposit_only_control_publishes_every_completed_checkpoint() {
    let report = run(true, false, false);
    for replica in &report.replicas {
        assert_eq!(replica.bias_published + 1, replica.outcome.hops as u64);
        assert_eq!(replica.outcome.gossip_rounds, 0);
        assert_eq!(replica.two_choice_restarts, 0);
    }
}

#[test]
fn an_independent_control_has_no_communication_and_preserves_its_streams() {
    let first = run(false, false, false);
    let second = run(false, false, false);
    assert_eq!(first.exchange, (0, 0));
    assert_eq!(second.exchange, (0, 0));
    for (a, b) in first.replicas.iter().zip(&second.replicas) {
        assert_eq!(a.bias_published, 0);
        assert_eq!(a.outcome.shared_deposits, 0);
        assert_eq!(a.outcome.gossip_rounds, 0);
        assert_eq!(a.two_choice_restarts, 0);
        assert_eq!(a.charged, b.charged);
        assert_eq!(a.outcome.hops, b.outcome.hops);
        assert_eq!(a.outcome.best.to_bits(), b.outcome.best.to_bits());
    }
}
