#![cfg(feature = "bank-rpc")]

use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

use anneal_core::Catalog_capnp::{catalog_reply, catalog_request};
use anneal_core::catalog_rpc::client::{
    CatalogAccess, CatalogClient, CatalogClientEvent, ClientConfig, SyncSchedule,
};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{CatalogCandidate, CatalogIdentity, PROTOCOL_VERSION};
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;

fn identity() -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: "fault-ensemble".into(),
        replica: 0,
        signature_digest: [0x6b; 32],
    }
}

fn candidate() -> CatalogCandidate {
    CatalogCandidate {
        producer_replica: 0,
        coordinates: vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
        cell: None,
        energy: -1.0,
        forces: vec![0.0; 6],
        gradient_norm: 0.0,
        descriptor: vec![0.1, 0.2],
        descriptor_schema_version: 1,
        quench_converged: true,
        charged_work: 1,
        event_sequence: 1,
        seed: 100,
        census_basin: None,
    }
}

fn short_timeouts() -> ClientConfig {
    ClientConfig {
        connect_timeout: Duration::from_millis(50),
        io_timeout: Duration::from_millis(20),
    }
}

fn read_request_sequence(stream: &mut TcpStream) -> u64 {
    let message = serialize::read_message(stream, ReaderOptions::new()).unwrap();
    message
        .get_root::<catalog_request::Reader>()
        .unwrap()
        .get_event_sequence()
}

fn write_empty_snapshot(stream: &mut TcpStream, event_sequence: u64, duplicate: bool) {
    let mut message = Builder::new_default();
    let mut root = message.init_root::<catalog_reply::Builder>();
    root.set_protocol_version(PROTOCOL_VERSION);
    root.set_event_sequence(event_sequence);
    root.set_snapshot_version(7);
    let mut accepted = root.init_result().init_accepted();
    accepted.set_duplicate(duplicate);
    accepted.set_census_visits(3);
    accepted.set_active_entries(2);
    accepted.set_aggregate_charged(11);
    accepted.set_aggregate_budget(100);
    accepted.init_payload().set_none(());
    serialize::write_message(stream, &message).unwrap();
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
fn timed_out_request_reconnects_and_replays_before_returning() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let peer = thread::spawn(move || {
        let (mut first, _) = listener.accept().unwrap();
        let first_sequence = read_request_sequence(&mut first);
        thread::sleep(Duration::from_millis(80));
        drop(first);

        let (mut replay, _) = listener.accept().unwrap();
        let replay_sequence = read_request_sequence(&mut replay);
        write_empty_snapshot(&mut replay, replay_sequence, true);
        [first_sequence, replay_sequence]
    });
    let mut client = CatalogClient::connect(
        addr,
        identity(),
        ClientConfig {
            connect_timeout: Duration::from_millis(100),
            io_timeout: Duration::from_millis(60),
        },
    )
    .unwrap();

    let snapshot = client.snapshot(17).unwrap();

    assert_eq!(snapshot.version, 7);
    assert_eq!(snapshot.census_visits, 3);
    assert_eq!(peer.join().unwrap(), [17, 17]);
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
        assert_eq!(client.record_visit(1, candidate()).unwrap().version, 1);
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
