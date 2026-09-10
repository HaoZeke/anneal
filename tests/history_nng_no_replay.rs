#![cfg(feature = "history-nng")]

use anneal_core::history_nng::HistoryNngClient;
use anneal_core::methods::minima_hopping::HistoryMembership;
use ndarray::array;
use nng::options::{Options, RecvTimeout};
use nng::{Protocol, Socket};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[test]
fn a_disconnect_cannot_replay_an_unacknowledged_history_mutation() {
    const CHILD: &str = "ANNEAL_HISTORY_NNG_REPLAY_CHILD";
    const NAME: &str = "a_disconnect_cannot_replay_an_unacknowledged_history_mutation";
    if std::env::var_os(CHILD).is_some() {
        let url = format!("inproc://anneal-history-replay-{}", std::process::id());
        let peer = Socket::new(Protocol::Rep0).unwrap();
        peer.set_opt::<RecvTimeout>(Some(Duration::from_secs(2)))
            .unwrap();
        peer.listen(&url).unwrap();
        let mut client =
            HistoryNngClient::dial(&url, array![2.0], HistoryMembership::Accepted).unwrap();
        let (sent, received) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = client.try_observe(0.0, array![0.0].view(), array![0.0].view());
            sent.send(result).unwrap();
        });
        let first = peer
            .recv()
            .expect("the server must receive the original mutation");
        assert!(!first.is_empty());
        peer.close();

        let replacement = Socket::new(Protocol::Rep0).unwrap();
        replacement
            .set_opt::<RecvTimeout>(Some(Duration::from_secs(2)))
            .unwrap();
        replacement.listen(&url).unwrap();
        let replay = replacement.recv();
        assert!(
            matches!(replay, Err(nng::Error::TimedOut)),
            "an unacknowledged mutation cannot be resent to the replacement peer: {replay:?}",
        );
        let result = received
            .recv_timeout(Duration::from_secs(2))
            .expect("the ambiguous mutation must return an error, not wait for a replay");
        assert!(result.is_err());
        worker.join().unwrap();
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
        "the request must fail without replay; terminated={terminated}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
