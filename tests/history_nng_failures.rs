#![cfg(feature = "history-nng")]

use anneal_core::history_nng::{HistoryNngClient, HistoryNngServer};
use anneal_core::methods::minima_hopping::{HistoryHook, HistoryMembership};
use ndarray::array;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn a_connected_peer_that_never_replies_returns_a_transport_error() {
    const CHILD: &str = "ANNEAL_HISTORY_NNG_SILENT_CHILD";
    const NAME: &str = "a_connected_peer_that_never_replies_returns_a_transport_error";
    if std::env::var_os(CHILD).is_some() {
        let url = format!("inproc://anneal-history-silent-{}", std::process::id());
        let peer = nng::Socket::new(nng::Protocol::Rep0).unwrap();
        peer.listen(&url).unwrap();
        let mut client =
            HistoryNngClient::dial(&url, array![2.0], HistoryMembership::Accepted).unwrap();
        let result = client.try_observe(0.0, array![0.0].view(), array![0.0].view());
        assert!(
            result.is_err(),
            "missing replies are transport failures, not scientific refusals"
        );
        let (observations, refusals, seconds) = client.cost();
        assert_eq!((observations, refusals), (0, 0));
        assert!(seconds.is_finite() && seconds > 0.0);
        return;
    }

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", NAME, "--nocapture"])
        .env(CHILD, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    let terminated = loop {
        if child.try_wait().unwrap().is_some() {
            break true;
        }
        if started.elapsed() >= Duration::from_secs(10) {
            child.kill().unwrap();
            break false;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let output = child.wait_with_output().unwrap();
    assert!(
        terminated && output.status.success(),
        "the history request must fail within its deadline; terminated={terminated}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

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
