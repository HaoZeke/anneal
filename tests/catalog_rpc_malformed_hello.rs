#![cfg(feature = "bank-rpc")]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anneal_core::catalog_rpc::CatalogIdentity;
use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use nng::options::{Options, RecvTimeout, SendTimeout};
use nng::{Protocol, Socket};

const CHILD: &str = "ANNEAL_CATALOG_MALFORMED_HELLO_CHILD";
const TEST_NAME: &str = "malformed_carrier_hello_preserves_catalog_availability";
const REQUEST_LIMIT: Duration = Duration::from_millis(500);
const CHILD_LIMIT: Duration = Duration::from_secs(15);

fn exercise_malformed_hello() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("catalog-protocol", "malformed-hello", [0x7d; 32], [0]).unwrap(),
    )
    .unwrap();
    let identity = CatalogIdentity {
        campaign: "catalog-protocol".into(),
        ensemble: "malformed-hello".into(),
        replica: 0,
        signature_digest: [0x7d; 32],
    };
    let config = ClientConfig {
        connect_timeout: REQUEST_LIMIT,
        io_timeout: REQUEST_LIMIT,
    };
    {
        let mut baseline = CatalogClient::connect(server.addr(), identity.clone(), config).unwrap();
        let snapshot = baseline.snapshot(1).expect("the catalogue must be live");
        assert_eq!(snapshot.version, 0);
        assert_eq!(snapshot.census_visits, 0);
    }

    let malformed = Socket::new(Protocol::Req0).unwrap();
    malformed
        .set_opt::<SendTimeout>(Some(REQUEST_LIMIT))
        .unwrap();
    malformed
        .set_opt::<RecvTimeout>(Some(REQUEST_LIMIT))
        .unwrap();
    malformed.dial(&format!("tcp://{}", server.addr())).unwrap();
    malformed.send(&b"INVALID-CATALOG-HELLO"[..]).unwrap();
    let rejection = malformed
        .recv()
        .expect("an invalid carrier hello must receive its rejection reply");
    assert!(
        rejection.is_empty(),
        "a rejected hello must not receive a pair endpoint"
    );
    drop(malformed);

    let mut client = CatalogClient::connect(server.addr(), identity, config).unwrap();
    let snapshot = client
        .snapshot(2)
        .expect("rejecting a malformed hello must preserve valid catalogue connections");
    assert_eq!(snapshot.version, 0);
    assert_eq!(snapshot.census_visits, 0);
    assert_eq!(snapshot.active_entries, 0);
    assert_eq!(snapshot.aggregate_charged, 0);
    assert_eq!(snapshot.aggregate_budget, 0);
}

#[test]
fn malformed_carrier_hello_preserves_catalog_availability() {
    if std::env::var(CHILD).as_deref() == Ok(TEST_NAME) {
        exercise_malformed_hello();
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
        "catalogue availability check must finish within {CHILD_LIMIT:?}; terminated={terminated}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
