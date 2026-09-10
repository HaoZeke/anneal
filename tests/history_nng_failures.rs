#![cfg(feature = "history-nng")]

use anneal_core::history_nng::{HistoryNngClient, HistoryNngServer};
use anneal_core::methods::minima_hopping::{HistoryHook, HistoryMembership};
use ndarray::array;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn accepting_an_unknown_remote_identity_cannot_report_success() {
    let url = format!("inproc://anneal-history-unknown-{}", std::process::id());
    let _server = HistoryNngServer::bind(&url, 1e-3, 1e-3).unwrap();
    let mut client =
        HistoryNngClient::dial(&url, array![2.0], HistoryMembership::Accepted).unwrap();
    let outcome = catch_unwind(AssertUnwindSafe(|| client.mark_accepted(417)));
    assert!(
        outcome.is_err(),
        "an unknown identity is not a successful acceptance"
    );
    let observer = HistoryNngClient::dial(&url, array![2.0], HistoryMembership::Accepted).unwrap();
    assert_eq!(observer.minimum_count(), Some(0));
}

#[test]
fn remote_history_cost_counts_admissions_and_scientific_refusals() {
    let url = format!("inproc://anneal-history-cost-{}", std::process::id());
    let _server = HistoryNngServer::bind(&url, 1e-3, 1e-3).unwrap();
    let mut client =
        HistoryNngClient::dial(&url, array![2.0], HistoryMembership::Accepted).unwrap();
    assert_eq!(client.cost(), (0, 0, 0.0));
    let admitted = client
        .observe(0.0, array![0.0].view(), array![0.0].view())
        .unwrap();
    client.mark_accepted(admitted.minimum);
    assert!(
        client
            .observe(0.5, array![0.5].view(), array![1.0].view())
            .is_none()
    );
    let (observations, refusals, seconds) = client.cost();
    assert_eq!((observations, refusals), (1, 1));
    assert!(seconds.is_finite() && seconds > 0.0);
    assert_eq!(client.minimum_count(), Some(1));
}
