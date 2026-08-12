#![cfg(feature = "bank-rpc")]

use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use anneal_core::catalog_rpc::CatalogIdentity;
use anneal_core::catalog_rpc::client::{
    CatalogAccess, CatalogClient, CatalogClientEvent, ClientConfig, SyncSchedule,
};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};

fn identity() -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: "fault-ensemble".into(),
        replica: 0,
        signature_digest: [0x6b; 32],
    }
}

fn short_timeouts() -> ClientConfig {
    ClientConfig {
        connect_timeout: Duration::from_millis(50),
        io_timeout: Duration::from_millis(20),
    }
}

#[test]
fn disconnect_produces_a_recorded_local_fallback() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let peer = thread::spawn(move || drop(listener.accept().unwrap()));
    let mut client = CatalogClient::connect(addr, identity(), short_timeouts()).unwrap();
    peer.join().unwrap();
    let mut events = Vec::new();

    let access = client.snapshot_or_fallback(1, &mut events);

    assert_eq!(access, CatalogAccess::LocalFallback);
    assert_eq!(
        events,
        vec![CatalogClientEvent::LocalFallback { event_sequence: 1 }]
    );
}

#[test]
fn delayed_reply_times_out_into_the_same_local_fallback() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let peer = thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        thread::sleep(Duration::from_millis(80));
    });
    let mut client = CatalogClient::connect(addr, identity(), short_timeouts()).unwrap();
    let mut events = Vec::new();

    assert_eq!(
        client.snapshot_or_fallback(1, &mut events),
        CatalogAccess::LocalFallback
    );
    assert_eq!(events.len(), 1);
    peer.join().unwrap();
}

#[test]
fn synchronization_schedule_enforces_the_k_b_staleness_bound() {
    let mut schedule = SyncSchedule::new(3, 7).unwrap();
    assert_eq!(schedule.maximum_staleness_calls(), 21);
    assert!(!schedule.record_slice(7).unwrap());
    assert!(!schedule.record_slice(4).unwrap());
    assert!(schedule.record_slice(6).unwrap());
    schedule.synchronized();
    assert!(!schedule.record_slice(1).unwrap());
    assert!(schedule.record_slice(8).is_err());
}

#[test]
fn server_restart_constructs_a_new_empty_ensemble_state() {
    let first_addr = {
        let server = CatalogServer::start(
            "127.0.0.1:0",
            ServerConfig::new("jcc-2026", "fault-ensemble", [0x6b; 32], [0]).unwrap(),
        )
        .unwrap();
        let addr = server.addr();
        let mut client = CatalogClient::connect(addr, identity(), ClientConfig::default()).unwrap();
        assert_eq!(
            client
                .record_visit(1, 3, true, vec![0.1, 0.2])
                .unwrap()
                .version,
            1
        );
        drop(client);
        drop(server);
        addr
    };
    let server = CatalogServer::start(
        &first_addr.to_string(),
        ServerConfig::new("jcc-2026", "fault-ensemble", [0x6b; 32], [0]).unwrap(),
    )
    .unwrap();
    let mut client =
        CatalogClient::connect(server.addr(), identity(), ClientConfig::default()).unwrap();

    let snapshot = client.snapshot(1).unwrap();
    assert_eq!(snapshot.version, 0);
    assert_eq!(snapshot.census_visits, 0);
    assert!(server.header().empty_state_proof);
}
