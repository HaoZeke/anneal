use std::collections::{BTreeMap, BTreeSet};

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
fn minimum_information_scores_are_the_only_ranked_claim_reward() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();
    ledger
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();
    let arms = ledger.claimable_arms();
    assert_eq!(arms.len(), 4);

    let preferred = arms
        .iter()
        .find(|(arm, _)| arm.source_basin == 17)
        .unwrap()
        .0
        .clone();
    let scores = arms
        .iter()
        .map(|(arm, _)| (arm.clone(), f64::from(arm == &preferred)))
        .collect::<BTreeMap<_, _>>();
    let first = ledger.claim_ranked(2, 101, &scores).unwrap();
    assert_eq!(first.arm, preferred);
    ledger
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

    let alternative = ledger
        .claimable_arms()
        .into_iter()
        .find(|(arm, _)| arm.source_basin == 23)
        .unwrap()
        .0;
    let scores = ledger
        .claimable_arms()
        .into_iter()
        .map(|(arm, _)| {
            let score = f64::from(arm == alternative);
            (arm, score)
        })
        .collect::<BTreeMap<_, _>>();
    let second = ledger.claim_ranked(7, 102, &scores).unwrap();

    assert_eq!(second.arm, alternative);
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
fn reobserving_an_exact_basin_does_not_expand_its_environment_portfolio() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();
    ledger
        .register_source(source(17, -104.3, &[(4, 2), (99, 1)]))
        .unwrap();

    let first = ledger.claim(1, 101).unwrap();
    let second = ledger.claim(2, 102).unwrap();

    assert_eq!(first.arm.environment_class, 4);
    assert_eq!(second.arm.environment_class, 4);
    assert!(ledger.claim(3, 103).is_none());
}

#[test]
fn mode_rank_coverage_precedes_the_opposite_sign_of_a_tried_rank() {
    let portfolio = RidePortfolio::new(2, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();

    let first = ledger.claim(2, 101).unwrap();
    ledger
        .report(
            2,
            first.id,
            83,
            RideOutcome::Failed(RideFailure::SaddleNotConverged),
        )
        .unwrap();
    let second = ledger.claim(2, 102).unwrap();

    assert_ne!(second.arm.mode_rank, first.arm.mode_rank);
    assert_eq!(second.arm.direction, first.arm.direction);
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
    let third = ledger.claim(9, 103).unwrap();
    let repeated_saddle_credit = ledger
        .report(
            9,
            third.id,
            125,
            RideOutcome::Certified {
                saddle: 71,
                endpoints: [17, 29],
            },
        )
        .unwrap();

    assert!(first_credit.certified_connection);
    assert!(duplicate_credit.certified_connection);
    assert_eq!(first_credit.failure, None);
    assert_eq!(duplicate_credit.failure, None);
    assert!(first_credit.novel_saddle);
    assert!(duplicate_credit.novel_saddle);
    assert!(!repeated_saddle_credit.novel_saddle);
    assert!(first_credit.novel_edge);
    assert!(!duplicate_credit.novel_edge);
    assert!(!repeated_saddle_credit.novel_edge);
    assert_eq!(first_credit.total_charged_evaluations, 140);
    assert_eq!(duplicate_credit.total_charged_evaluations, 131);
    assert_eq!(ledger.unique_saddles(), 2);
    assert_eq!(ledger.unique_edges(), 1);
    assert_eq!(ledger.certified_connections(), 3);
}

#[test]
fn certified_same_basin_saddle_is_counted_as_a_degenerate_rearrangement() {
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
                endpoints: [17, 17],
            },
        )
        .unwrap();

    assert!(credit.certified_connection);
    assert!(credit.degenerate_rearrangement);
    assert!(credit.novel_saddle);
    assert!(!credit.novel_edge);
    assert_eq!(credit.failure, None);
    assert_eq!(ledger.unique_saddles(), 1);
    assert_eq!(ledger.unique_edges(), 0);
    assert_eq!(ledger.unique_degenerate_rearrangements(), 1);
    assert_eq!(ledger.certified_connections(), 1);
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
fn an_unresolved_saddle_is_diagnostic_not_an_allocation_reward() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut diagnostic = RideLedger::new(portfolio);
    diagnostic
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();
    let mut control = diagnostic.clone();
    let diagnostic_first = diagnostic.claim(2, 101).unwrap();
    let diagnostic_second = diagnostic.claim(7, 102).unwrap();
    let control_first = control.claim(2, 101).unwrap();
    let control_second = control.claim(7, 102).unwrap();
    assert_eq!(diagnostic_first.arm, control_first.arm);
    assert_eq!(diagnostic_second.arm, control_second.arm);

    diagnostic
        .report(
            diagnostic_first.replica,
            diagnostic_first.id,
            4_000,
            RideOutcome::Failed(RideFailure::MinimumModeLostCurvature),
        )
        .unwrap();
    let credit = diagnostic
        .report(
            diagnostic_second.replica,
            diagnostic_second.id,
            2_000,
            RideOutcome::Unresolved {
                saddle: 70,
                failure: RideFailure::CollapsedConnection,
            },
        )
        .unwrap();
    control
        .report(
            control_first.replica,
            control_first.id,
            4_000,
            RideOutcome::Failed(RideFailure::MinimumModeLostCurvature),
        )
        .unwrap();
    control
        .report(
            control_second.replica,
            control_second.id,
            2_000,
            RideOutcome::Failed(RideFailure::CollapsedConnection),
        )
        .unwrap();
    diagnostic
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();
    control
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();

    let diagnostic_next = diagnostic.claim(9, 103).unwrap();
    let control_next = control.claim(9, 103).unwrap();

    assert!(!credit.certified_connection);
    assert_eq!(credit.failure, Some(RideFailure::CollapsedConnection));
    assert!(credit.novel_saddle);
    assert!(!credit.novel_edge);
    assert_eq!(diagnostic.unique_saddles(), 1);
    assert_eq!(control.unique_saddles(), 0);
    assert_eq!(diagnostic_next.arm, control_next.arm);
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
fn edge_novelty_is_recorded_without_becoming_an_allocation_reward() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut diagnostic = RideLedger::new(portfolio);
    diagnostic
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();
    let mut control = diagnostic.clone();

    let diagnostic_first = diagnostic.claim(2, 101).unwrap();
    let diagnostic_second = diagnostic.claim(7, 102).unwrap();
    let control_first = control.claim(2, 101).unwrap();
    let control_second = control.claim(7, 102).unwrap();
    assert_eq!(diagnostic_first.arm, control_first.arm);
    assert_eq!(diagnostic_second.arm, control_second.arm);

    let credit = diagnostic
        .report(
            2,
            diagnostic_first.id,
            140,
            RideOutcome::Certified {
                saddle: 70,
                endpoints: [17, 29],
            },
        )
        .unwrap();
    diagnostic
        .report(
            7,
            diagnostic_second.id,
            90,
            RideOutcome::Failed(RideFailure::SaddleNotConverged),
        )
        .unwrap();
    control
        .report(
            2,
            control_first.id,
            140,
            RideOutcome::Failed(RideFailure::SaddleNotConverged),
        )
        .unwrap();
    control
        .report(
            7,
            control_second.id,
            90,
            RideOutcome::Failed(RideFailure::SaddleNotConverged),
        )
        .unwrap();

    let diagnostic_next = diagnostic.claim(9, 103).unwrap();
    let control_next = control.claim(9, 103).unwrap();

    assert!(credit.novel_saddle);
    assert!(credit.novel_edge);
    assert_eq!(diagnostic.unique_edges(), 1);
    assert_eq!(control.unique_edges(), 0);
    assert_eq!(diagnostic_next.arm, control_next.arm);
}

#[test]
fn method_outcomes_do_not_transfer_between_source_minima() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut lanczos_succeeds = RideLedger::new(portfolio);
    lanczos_succeeds
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();
    let mut dimer_succeeds = lanczos_succeeds.clone();

    let first = lanczos_succeeds.claim(2, 101).unwrap();
    let second = lanczos_succeeds.claim(7, 102).unwrap();
    let (dimer, lanczos) = if first.arm.method == RideMethod::Dimer {
        (first, second)
    } else {
        (second, first)
    };
    let dimer_control = dimer_succeeds.claim(dimer.replica, dimer.seed).unwrap();
    let lanczos_control = dimer_succeeds.claim(lanczos.replica, lanczos.seed).unwrap();
    assert_eq!(dimer.arm, dimer_control.arm);
    assert_eq!(lanczos.arm, lanczos_control.arm);

    lanczos_succeeds
        .report(
            dimer.replica,
            dimer.id,
            4_000,
            RideOutcome::Failed(RideFailure::MinimumModeLostCurvature),
        )
        .unwrap();
    lanczos_succeeds
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
    dimer_succeeds
        .report(
            dimer_control.replica,
            dimer_control.id,
            4_000,
            RideOutcome::Certified {
                saddle: 71,
                endpoints: [17, 31],
            },
        )
        .unwrap();
    dimer_succeeds
        .report(
            lanczos_control.replica,
            lanczos_control.id,
            2_000,
            RideOutcome::Failed(RideFailure::MinimumModeLostCurvature),
        )
        .unwrap();
    lanczos_succeeds
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();
    dimer_succeeds
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();

    let after_lanczos_success = lanczos_succeeds.claim(9, 103).unwrap();
    let after_dimer_success = dimer_succeeds.claim(9, 103).unwrap();

    assert_eq!(after_lanczos_success.arm, after_dimer_success.arm);
}

#[test]
fn charged_cost_cannot_rank_default_claims() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut dimer_expensive = RideLedger::new(portfolio);
    dimer_expensive
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();
    let mut lanczos_expensive = dimer_expensive.clone();

    let dimer_expensive_orders = (0..4)
        .map(|replica| {
            dimer_expensive
                .claim(replica, 1_000 + u64::from(replica))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let lanczos_expensive_orders = (0..4)
        .map(|replica| {
            lanczos_expensive
                .claim(replica, 1_000 + u64::from(replica))
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        dimer_expensive_orders
            .iter()
            .map(|order| &order.arm)
            .collect::<Vec<_>>(),
        lanczos_expensive_orders
            .iter()
            .map(|order| &order.arm)
            .collect::<Vec<_>>()
    );
    for (offset, (left, right)) in dimer_expensive_orders
        .into_iter()
        .zip(lanczos_expensive_orders)
        .enumerate()
    {
        let left_charged = match left.arm.method {
            RideMethod::Dimer => 4_000,
            RideMethod::Lanczos => 500,
        };
        let right_charged = match right.arm.method {
            RideMethod::Dimer => 500,
            RideMethod::Lanczos => 4_000,
        };
        dimer_expensive
            .report(
                left.replica,
                left.id,
                left_charged,
                RideOutcome::Certified {
                    saddle: 70 + offset as u64,
                    endpoints: [17, 29 + offset as u64],
                },
            )
            .unwrap();
        lanczos_expensive
            .report(
                right.replica,
                right.id,
                right_charged,
                RideOutcome::Certified {
                    saddle: 70 + offset as u64,
                    endpoints: [17, 29 + offset as u64],
                },
            )
            .unwrap();
    }
    dimer_expensive
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();
    lanczos_expensive
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();

    let after_expensive_dimers = dimer_expensive.claim(9, 2_000).unwrap();
    let after_expensive_lanczos = lanczos_expensive.claim(9, 2_000).unwrap();

    assert_eq!(after_expensive_dimers.arm, after_expensive_lanczos.arm);
}

#[test]
fn explicit_information_scores_override_network_history() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer, RideMethod::Lanczos]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();

    let orders = (0..4)
        .map(|replica| ledger.claim(replica, 1_000 + u64::from(replica)).unwrap())
        .collect::<Vec<_>>();
    for (offset, order) in orders.into_iter().enumerate() {
        ledger
            .report(
                order.replica,
                order.id,
                500 + offset as u64,
                RideOutcome::Certified {
                    saddle: 70,
                    endpoints: [17, 29],
                },
            )
            .unwrap();
    }
    ledger
        .register_source(source(23, -107.5, &[(4, 8)]))
        .unwrap();

    let preferred = ledger
        .claimable_arms()
        .into_iter()
        .find(|(arm, _)| arm.source_basin == 23 && arm.method == RideMethod::Lanczos)
        .unwrap()
        .0;
    let scores = ledger
        .claimable_arms()
        .into_iter()
        .map(|(arm, _)| {
            let score = f64::from(arm == preferred);
            (arm, score)
        })
        .collect::<BTreeMap<_, _>>();
    let ranked = ledger.claim_ranked(9, 2_000, &scores).unwrap();

    assert_eq!(ledger.unique_saddles(), 1);
    assert_eq!(ledger.unique_edges(), 1);
    assert_eq!(ranked.arm, preferred);
}

#[test]
fn exact_saddle_reobservations_produce_coverage_evidence() {
    let portfolio = RidePortfolio::new(1, vec![RideMethod::Dimer]).unwrap();
    let mut ledger = RideLedger::new(portfolio);
    ledger
        .register_source(source(17, -104.2, &[(4, 6)]))
        .unwrap();

    for attempt in 0..20_u64 {
        let work = ledger.claim(2, 3_000 + attempt).unwrap();
        ledger
            .report(
                2,
                work.id,
                100,
                RideOutcome::Certified {
                    saddle: 70,
                    endpoints: [17, 29],
                },
            )
            .unwrap();
    }

    let covered = ledger.saddle_coverage();
    assert_eq!(covered.observations, 20);
    assert_eq!(covered.singletons, 0);
    assert_eq!(covered.doubletons, 0);
    assert_eq!(covered.unseen_mass_upper, Some(0.0));
    assert!(covered.saturated);

    let work = ledger.claim(2, 4_000).unwrap();
    let credit = ledger
        .report(
            2,
            work.id,
            100,
            RideOutcome::Unresolved {
                saddle: 71,
                failure: RideFailure::CollapsedConnection,
            },
        )
        .unwrap();
    let reopened = ledger.saddle_coverage();

    assert!(credit.novel_saddle);
    assert_eq!(reopened.observations, 21);
    assert_eq!(reopened.singletons, 1);
    assert!(!reopened.saturated);
}
