use anneal_core::cooperative_search::ledger::{
    ChargeKind, CooperativeLedger, LedgerError, LedgerUpdate, ReplicaLedgerEvent,
};

fn event(
    replica: u32,
    sequence: u64,
    kind: ChargeKind,
    charged_calls: u64,
    cumulative_charged: u64,
) -> ReplicaLedgerEvent {
    ReplicaLedgerEvent {
        replica,
        sequence,
        kind,
        charged_calls,
        cumulative_charged,
    }
}

#[test]
fn four_replica_total_is_the_sum_of_unique_charged_work() {
    let mut ledger = CooperativeLedger::new([0, 1, 2, 3], 100).unwrap();
    for (replica, calls) in [(0, 7), (1, 11), (2, 13), (3, 17)] {
        ledger
            .record(event(replica, 1, ChargeKind::AcceptedQuench, calls, calls))
            .unwrap();
    }

    assert_eq!(ledger.replica_total(0), Some(7));
    assert_eq!(ledger.replica_total(3), Some(17));
    assert_eq!(ledger.ensemble_total(), 48);
}

#[test]
fn replay_and_out_of_order_delivery_are_idempotent() {
    let mut ledger = CooperativeLedger::new([7], 100).unwrap();
    let second = event(7, 2, ChargeKind::FreshValidation, 3, 8);
    let first = event(7, 1, ChargeKind::RejectedQuench, 5, 5);

    assert_eq!(ledger.record(second).unwrap(), LedgerUpdate::Recorded);
    assert_eq!(ledger.ensemble_total(), 8);
    assert_eq!(ledger.record(first).unwrap(), LedgerUpdate::Recorded);
    assert_eq!(ledger.record(second).unwrap(), LedgerUpdate::Duplicate);
    assert_eq!(ledger.ensemble_total(), 8);
}

#[test]
fn counter_regression_and_conflicting_replay_are_rejected() {
    let mut ledger = CooperativeLedger::new([2], 100).unwrap();
    let first = event(2, 1, ChargeKind::AcceptedQuench, 4, 4);
    ledger.record(first).unwrap();

    assert_eq!(
        ledger
            .record(event(2, 2, ChargeKind::Retry, 1, 3))
            .unwrap_err(),
        LedgerError::CounterRegression {
            replica: 2,
            sequence: 2,
        }
    );
    assert_eq!(
        ledger
            .record(event(2, 1, ChargeKind::AcceptedQuench, 5, 5))
            .unwrap_err(),
        LedgerError::ConflictingReplay {
            replica: 2,
            sequence: 1,
        }
    );
    assert_eq!(ledger.ensemble_total(), 4);
}

#[test]
fn every_work_boundary_has_explicit_charge_semantics() {
    let mut ledger = CooperativeLedger::new([0], 100).unwrap();
    let records = [
        (ChargeKind::LocalProposal, 0),
        (ChargeKind::RemoteProposal, 0),
        (ChargeKind::DescriptorEvaluation, 0),
        (ChargeKind::AcceptedQuench, 4),
        (ChargeKind::RejectedQuench, 3),
        (ChargeKind::FreshValidation, 1),
        (ChargeKind::Retry, 2),
        (ChargeKind::RpcFallback, 0),
    ];
    let mut cumulative = 0;
    for (index, (kind, calls)) in records.into_iter().enumerate() {
        cumulative += calls;
        ledger
            .record(event(0, index as u64 + 1, kind, calls, cumulative))
            .unwrap();
    }

    assert_eq!(ledger.ensemble_total(), 10);
    assert_eq!(ledger.event_count(), 8);
}

#[test]
fn invalid_charge_semantics_and_budget_overrun_are_rejected() {
    let mut ledger = CooperativeLedger::new([0], 5).unwrap();
    assert_eq!(
        ledger
            .record(event(0, 1, ChargeKind::RpcFallback, 1, 1))
            .unwrap_err(),
        LedgerError::UnchargedKindHasCalls {
            kind: ChargeKind::RpcFallback,
            charged_calls: 1,
        }
    );
    assert_eq!(
        ledger
            .record(event(0, 1, ChargeKind::AcceptedQuench, 6, 6))
            .unwrap_err(),
        LedgerError::BudgetExceeded {
            replica: 0,
            charged: 6,
            budget: 5,
        }
    );
    assert_eq!(ledger.ensemble_total(), 0);
}

#[test]
fn first_encounter_freezes_the_complete_counter_vector() {
    let mut ledger = CooperativeLedger::new([0, 1, 2, 3], 100).unwrap();
    ledger
        .record(event(0, 1, ChargeKind::AcceptedQuench, 5, 5))
        .unwrap();
    ledger
        .record(event(2, 1, ChargeKind::FreshValidation, 3, 3))
        .unwrap();
    let encounter = ledger.record_first_encounter().clone();
    ledger
        .record(event(1, 1, ChargeKind::RejectedQuench, 7, 7))
        .unwrap();

    assert_eq!(
        encounter.replica_totals(),
        &[(0, 5), (1, 0), (2, 3), (3, 0)]
    );
    assert_eq!(encounter.ensemble_total(), 8);
    assert_eq!(ledger.record_first_encounter(), &encounter);
    assert_eq!(ledger.ensemble_total(), 15);
}
