#![cfg(feature = "bank-rpc")]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{
    CatalogIdentity, CatalogOperation, CatalogReply, CatalogRequest, CoordinatorEvent,
    PROTOCOL_VERSION,
};

const CHILD: &str = "ANNEAL_CATALOG_EVENT_REPLAY_CHILD";
const TEST_NAME: &str = "committed_tick_replay_does_not_redeliver_subscriber_event";
const REQUEST_LIMIT: Duration = Duration::from_millis(500);
const CHILD_LIMIT: Duration = Duration::from_secs(15);

fn drain_events(client: &mut CatalogClient) -> Vec<CoordinatorEvent> {
    let deadline = Instant::now() + Duration::from_millis(100);
    let mut events = Vec::new();
    loop {
        events.extend(client.events());
        if Instant::now() >= deadline {
            return events;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn exercise_event_replay() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("catalog-protocol", "event-replay", [0x7e; 32], [0]).unwrap(),
    )
    .unwrap();
    let identity = CatalogIdentity {
        campaign: "catalog-protocol".into(),
        ensemble: "event-replay".into(),
        replica: 0,
        signature_digest: [0x7e; 32],
    };
    let mut client = CatalogClient::connect(
        server.addr(),
        identity.clone(),
        ClientConfig {
            connect_timeout: REQUEST_LIMIT,
            io_timeout: REQUEST_LIMIT,
        },
    )
    .unwrap();
    let request = CatalogRequest {
        protocol_version: PROTOCOL_VERSION,
        identity,
        event_sequence: 1,
        snapshot_version: 0,
        operation: CatalogOperation::Tick { millis: 17 },
    };

    let CatalogReply::Accepted(first) = client.session_call(request.clone()).unwrap() else {
        panic!("the first tick must be accepted");
    };
    assert!(!first.duplicate);
    assert_eq!(first.event_sequence, request.event_sequence);
    assert_eq!(client.observe().unwrap().ticks, 1);
    assert_eq!(drain_events(&mut client), vec![CoordinatorEvent::Tick(17)]);

    let CatalogReply::Accepted(replay) = client.session_call(request.clone()).unwrap() else {
        panic!("an exact committed request must replay successfully");
    };
    assert!(replay.duplicate);
    assert_eq!(replay.event_sequence, first.event_sequence);
    assert_eq!(replay.snapshot, first.snapshot);
    assert_eq!(replay.payload, first.payload);
    assert_eq!(client.observe().unwrap().ticks, 1);
    assert_eq!(
        drain_events(&mut client),
        Vec::<CoordinatorEvent>::new(),
        "an exact request replay must not deliver the committed tick twice"
    );

    let distinct = CatalogRequest {
        event_sequence: 2,
        snapshot_version: replay.snapshot.version,
        operation: CatalogOperation::Tick { millis: 23 },
        ..request
    };
    let CatalogReply::Accepted(second) = client.session_call(distinct).unwrap() else {
        panic!("a distinct tick must remain deliverable");
    };
    assert!(!second.duplicate);
    assert_eq!(client.observe().unwrap().ticks, 2);
    assert_eq!(drain_events(&mut client), vec![CoordinatorEvent::Tick(23)]);
}

#[test]
fn committed_tick_replay_does_not_redeliver_subscriber_event() {
    if std::env::var(CHILD).as_deref() == Ok(TEST_NAME) {
        exercise_event_replay();
        return;
    }

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", TEST_NAME, "--nocapture"])
        .env(CHILD, TEST_NAME)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    let terminated = loop {
        if child.try_wait().unwrap().is_some() {
            break true;
        }
        if started.elapsed() >= CHILD_LIMIT {
            child.kill().unwrap();
            break false;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let output = child.wait_with_output().unwrap();
    assert!(
        terminated && output.status.success(),
        "catalogue event replay check must finish within {CHILD_LIMIT:?}; terminated={terminated}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
