use anneal_core::methods::cluster_hopping::{Ledger, QuenchStatus};
use ndarray::{Array1, array};

#[test]
fn quench_boundaries_partition_charged_relaxation_work() {
    let mut ledger = Ledger::new(20);

    let first = ledger.spent();
    assert!(ledger.charge_many(7));
    assert!(ledger.record_quench_boundary(
        first,
        -4.0,
        array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        Some(Array1::zeros(6)),
    ));

    let second = ledger.spent();
    assert!(ledger.charge_many(5));
    assert!(ledger.record_quench_boundary(
        second,
        -3.0,
        array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
        None,
    ));

    let boundaries = ledger.quench_boundaries();
    assert_eq!(boundaries.len(), 2);
    assert_eq!(boundaries[0].status(), QuenchStatus::Validated);
    assert_eq!(boundaries[0].charged_calls(), 7);
    assert_eq!(boundaries[1].status(), QuenchStatus::Rejected);
    assert_eq!(boundaries[1].charged_calls(), 5);
    assert_eq!(
        boundaries
            .iter()
            .map(|event| event.charged_calls())
            .sum::<usize>(),
        ledger.spent()
    );
}

#[test]
fn an_uncharged_relaxation_is_not_a_quench_boundary() {
    let mut ledger = Ledger::new(3);
    assert!(!ledger.record_quench_boundary(ledger.spent(), 0.0, array![0.0, 0.0, 0.0], None,));
    assert!(ledger.quench_boundaries().is_empty());
}
