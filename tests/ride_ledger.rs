use std::collections::BTreeSet;

use anneal_core::pes_exploration::RideMethod;
use anneal_core::ride_ledger::{
    EnvironmentTarget, RideFailure, RideLedger, RideOutcome, RidePortfolio, RideSource,
};

fn source(basin: u64, energy: f64, environments: &[(u32, u32)]) -> RideSource {
    RideSource {
        basin,
        energy,
        environments: environments
            .iter()
            .map(|&(class, atom)| EnvironmentTarget { class, atom })
            .collect(),
    }
}

#[test]
fn live_replicas_claim_each_transition_experiment_at_most_once() {
    let portfolio = RidePortfolio::new(2, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6), (9, 11)]))
        .unwrap();

    let mut arms = BTreeSet::new();
    for replica in 0..16 {
        let work = ledger.claim(replica, 1000 + u64::from(replica)).unwrap();
        assert!(arms.insert(work.arm.clone()));
    }

    assert_eq!(arms.len(), 2 * 2 * 2 * 2);
    assert!(ledger.claim(99, 9999).is_none());
}

#[test]
fn shared_failure_evidence_redirects_the_next_replica_to_an_untried_arm() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();

    let failed = ledger.claim(2, 101).unwrap();
    ledger
        .report(
            2,
            failed.id,
            83,
            RideOutcome::Failed(RideFailure::SaddleNotConverged),
        )
        .unwrap();
    let reassigned = ledger.claim(7, 102).unwrap();

    assert_ne!(reassigned.arm, failed.arm);
    assert_eq!(ledger.completed_attempts(), 1);
    assert_eq!(ledger.charged_evaluations(), 83);
}

#[test]
fn duplicate_certified_connections_do_not_masquerade_as_new_pes_edges() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();

    let first = ledger.claim(2, 101).unwrap();
    let first_credit = ledger
        .report(
            2,
            first.id,
            140,
            RideOutcome::Certified {
                saddle: 70,
                endpoints: [17, 29],
            },
        )
        .unwrap();
    let second = ledger.claim(7, 102).unwrap();
    let duplicate_credit = ledger
        .report(
            7,
            second.id,
            131,
            RideOutcome::Certified {
                saddle: 71,
                endpoints: [29, 17],
            },
        )
        .unwrap();

    assert!(first_credit.novel_edge);
    assert!(!duplicate_credit.novel_edge);
    assert_eq!(ledger.unique_edges(), 1);
    assert_eq!(ledger.certified_connections(), 2);
}

#[test]
fn ape_energy_priority_selects_the_lowest_untried_source() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(10, -91.0, &[(2, 3)]))
        .unwrap();
    ledger
        .register_source(source(20, -107.5, &[(5, 8)]))
        .unwrap();

    let work = ledger.claim(3, 44).unwrap();

    assert_eq!(work.arm.source_basin, 20);
    assert_eq!(work.representative_atom, 8);
}
