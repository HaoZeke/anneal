use std::collections::BTreeSet;

use anneal_core::pes_exploration::RideMethod;
use anneal_core::ride_ledger::{
    EnvironmentBook, EnvironmentTarget, RideFailure, RideLedger, RideOutcome, RidePortfolio,
    RideSource,
};
use ndarray::array;

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
    let credit = ledger
        .report(
            2,
            failed.id,
            83,
            RideOutcome::Failed(RideFailure::SaddleNotConverged),
        )
        .unwrap();
    let reassigned = ledger.claim(7, 102).unwrap();

    assert_ne!(reassigned.arm, failed.arm);
    assert_eq!(credit.failure, Some(RideFailure::SaddleNotConverged));
    assert_eq!(ledger.completed_attempts(), 1);
    assert_eq!(ledger.charged_evaluations(), 83);
}

#[test]
fn environment_coverage_precedes_repeated_local_portfolio_arms() {
    let portfolio = RidePortfolio::new(2, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut serial = RideLedger::new(portfolio.clone());
    serial
        .register_source(source(17, -104.2, &[(4, 6), (9, 11)]))
        .unwrap();

    let first = serial.claim(2, 101).unwrap();
    serial
        .report(
            2,
            first.id,
            83,
            RideOutcome::Failed(RideFailure::SaddleNotConverged),
        )
        .unwrap();
    let second = serial.claim(2, 102).unwrap();

    assert_ne!(second.arm.environment_class, first.arm.environment_class);

    let mut parallel = RideLedger::new(portfolio);
    parallel
        .register_source(source(17, -104.2, &[(4, 6), (9, 11)]))
        .unwrap();
    let first = parallel.claim(2, 201).unwrap();
    let second = parallel.claim(7, 202).unwrap();

    assert_ne!(second.arm.environment_class, first.arm.environment_class);
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

    assert!(first_credit.certified_connection);
    assert!(duplicate_credit.certified_connection);
    assert_eq!(first_credit.failure, None);
    assert_eq!(duplicate_credit.failure, None);
    assert!(first_credit.novel_edge);
    assert!(!duplicate_credit.novel_edge);
    assert_eq!(first_credit.total_charged_evaluations, 140);
    assert_eq!(duplicate_credit.total_charged_evaluations, 131);
    assert_eq!(ledger.unique_edges(), 1);
    assert_eq!(ledger.certified_connections(), 2);
}

#[test]
fn a_certified_edge_is_useful_even_when_it_misses_the_nominal_source() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();
    let work = ledger.claim(2, 101).unwrap();

    let credit = ledger
        .report(
            2,
            work.id,
            140,
            RideOutcome::Certified {
                saddle: 70,
                endpoints: [29, 31],
            },
        )
        .unwrap();

    assert!(credit.certified_connection);
    assert!(credit.novel_edge);
    assert_eq!(ledger.unique_edges(), 1);
}

#[test]
fn an_unresolved_saddle_counts_as_method_reliability_not_edge_novelty() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();
    let first = ledger.claim(2, 101).unwrap();
    let second = ledger.claim(7, 102).unwrap();
    let (dimer, lanczos) = if first.arm.method == RideMethod::Dimer {
        (first, second)
    } else {
        (second, first)
    };
    ledger
        .report(
            dimer.replica,
            dimer.id,
            4_000,
            RideOutcome::Failed(RideFailure::MinimumModeLostCurvature),
        )
        .unwrap();
    let credit = ledger
        .report(
            lanczos.replica,
            lanczos.id,
            2_000,
            RideOutcome::Unresolved {
                saddle: 70,
                failure: RideFailure::CollapsedConnection,
            },
        )
        .unwrap();
    ledger
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();

    let guided = ledger.claim(9, 103).unwrap();

    assert!(!credit.certified_connection);
    assert_eq!(credit.failure, Some(RideFailure::CollapsedConnection));
    assert!(!credit.novel_edge);
    assert_eq!(guided.arm.method, RideMethod::Lanczos);
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

#[test]
fn local_environment_book_exposes_one_stable_representative_per_class() {
    let mut book = EnvironmentBook::new(0.12).unwrap();
    let first = book
        .observe(array![[1.0, 0.0], [0.96, 0.04], [0.0, 1.0]].view())
        .unwrap();
    let second = book
        .observe(array![[0.98, 0.01], [0.02, 0.97], [0.55, 0.55]].view())
        .unwrap();

    assert_eq!(
        first,
        vec![
            EnvironmentTarget { class: 0, atom: 0 },
            EnvironmentTarget { class: 1, atom: 2 },
        ]
    );
    assert_eq!(
        second,
        vec![
            EnvironmentTarget { class: 0, atom: 0 },
            EnvironmentTarget { class: 1, atom: 1 },
            EnvironmentTarget { class: 2, atom: 2 },
        ]
    );
    assert_eq!(book.class_count(), 3);
}

#[test]
fn novel_edge_yield_outweighs_a_failed_arm_after_initial_coverage() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();

    let productive = ledger.claim(2, 101).unwrap();
    ledger
        .report(
            2,
            productive.id,
            140,
            RideOutcome::Certified {
                saddle: 70,
                endpoints: [17, 29],
            },
        )
        .unwrap();
    let failed = ledger.claim(7, 102).unwrap();
    ledger
        .report(
            7,
            failed.id,
            90,
            RideOutcome::Failed(RideFailure::SaddleNotConverged),
        )
        .unwrap();

    let repeated = ledger.claim(9, 103).unwrap();

    assert_eq!(repeated.arm, productive.arm);
    assert_ne!(repeated.arm, failed.arm);
}

#[test]
fn same_system_method_evidence_guides_untried_arms() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();

    let first = ledger.claim(2, 101).unwrap();
    let second = ledger.claim(7, 102).unwrap();
    let (dimer, lanczos) = if first.arm.method == RideMethod::Dimer {
        (first, second)
    } else {
        (second, first)
    };
    assert_eq!(dimer.arm.method, RideMethod::Dimer);
    assert_eq!(lanczos.arm.method, RideMethod::Lanczos);

    ledger
        .report(
            dimer.replica,
            dimer.id,
            4_000,
            RideOutcome::Failed(RideFailure::MinimumModeLostCurvature),
        )
        .unwrap();
    ledger
        .report(
            lanczos.replica,
            lanczos.id,
            2_000,
            RideOutcome::Certified {
                saddle: 70,
                endpoints: [17, 29],
            },
        )
        .unwrap();
    ledger
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();

    let guided = ledger.claim(9, 103).unwrap();

    assert_eq!(guided.arm.method, RideMethod::Lanczos);
}

#[test]
fn certification_reliability_transfers_when_a_source_edge_is_saturated() {
    let portfolio = RidePortfolio::new(2, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();

    let orders = (0..8)
        .map(|replica| ledger.claim(replica, 1_000 + u64::from(replica)).unwrap())
        .collect::<Vec<_>>();
    let mut dimers = orders
        .iter()
        .filter(|order| order.arm.method == RideMethod::Dimer)
        .cloned()
        .collect::<Vec<_>>();
    let lanczos = orders
        .iter()
        .filter(|order| order.arm.method == RideMethod::Lanczos)
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(dimers.len(), 4);
    assert_eq!(lanczos.len(), 4);

    let first_dimer = dimers.remove(0);
    ledger
        .report(
            first_dimer.replica,
            first_dimer.id,
            3_000,
            RideOutcome::Certified {
                saddle: 70,
                endpoints: [17, 29],
            },
        )
        .unwrap();
    for order in dimers {
        ledger
            .report(
                order.replica,
                order.id,
                4_000,
                RideOutcome::Failed(RideFailure::MinimumModeLostCurvature),
            )
            .unwrap();
    }
    for (offset, order) in lanczos.into_iter().enumerate() {
        ledger
            .report(
                order.replica,
                order.id,
                2_000,
                RideOutcome::Certified {
                    saddle: 80 + offset as u64,
                    endpoints: [17, 29],
                },
            )
            .unwrap();
    }
    ledger
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();

    let guided = ledger.claim(9, 2_000).unwrap();

    assert_eq!(guided.arm.source_basin, 23);
    assert_eq!(guided.arm.method, RideMethod::Lanczos);
}
