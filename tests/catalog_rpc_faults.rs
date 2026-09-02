#![cfg(feature = "bank-rpc")]

use std::cell::RefCell;
use std::net::TcpListener;
use std::rc::Rc;
use std::thread;
use std::time::Duration;

use anneal_core::Catalog_capnp::{coordinator, session};
use anneal_core::catalog_rpc::client::{
    CatalogAccess, CatalogClient, CatalogClientEvent, ClientConfig, SyncSchedule,
};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{CatalogCandidate, CatalogIdentity, PROTOCOL_VERSION};
use capnp::capability::Promise;
use capnp_rpc::rpc_twoparty_capnp::Side;
use capnp_rpc::twoparty::VatNetwork;
use capnp_rpc::RpcSystem;
use futures::AsyncReadExt;
use tokio_util::compat::TokioAsyncReadCompatExt;

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

struct CannedCoordinator {
    sequences: Rc<RefCell<Vec<u64>>>,
}

struct CannedSession {
    sequences: Rc<RefCell<Vec<u64>>>,
}

impl coordinator::Server for CannedCoordinator {
    fn attach(
        &mut self,
        _params: coordinator::AttachParams,
        mut results: coordinator::AttachResults,
    ) -> Promise<(), capnp::Error> {
        let mut reply = results.get();
        reply.set_session(capnp_rpc::new_client(CannedSession {
            sequences: Rc::clone(&self.sequences),
        }));
        let mut roster = reply.init_roster();
        roster.set_version(1);
        roster.set_spawn_requested(0);
        Promise::ok(())
    }
}

impl session::Server for CannedSession {
    fn call(
        &mut self,
        params: session::CallParams,
        mut results: session::CallResults,
    ) -> Promise<(), capnp::Error> {
        let sequence = match params.get().and_then(|params| params.get_request()) {
            Ok(request) => request.get_event_sequence(),
            Err(_) => 0,
        };
        self.sequences.borrow_mut().push(sequence);
        let mut reply = results.get().init_reply();
        reply.set_protocol_version(PROTOCOL_VERSION);
        reply.set_event_sequence(sequence);
        reply.set_snapshot_version(7);
        let mut accepted = reply.init_result().init_accepted();
        accepted.set_duplicate(true);
        accepted.set_census_visits(3);
        accepted.set_active_entries(2);
        accepted.set_aggregate_charged(11);
        accepted.set_aggregate_budget(100);
        accepted.init_payload().set_none(());
        Promise::ok(())
    }
}

fn serve_canned_snapshot(
    stream: std::net::TcpStream,
    sequences: Rc<RefCell<Vec<u64>>>,
) {
    stream.set_nonblocking(true).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, async move {
        let stream = tokio::net::TcpStream::from_std(stream).unwrap();
        let _ = stream.set_nodelay(true);
        let (reader, writer) = TokioAsyncReadCompatExt::compat(stream).split();
        let network = VatNetwork::new(
            futures::io::BufReader::new(reader),
            futures::io::BufWriter::new(writer),
            Side::Server,
            Default::default(),
        );
        let client: coordinator::Client = capnp_rpc::new_client(CannedCoordinator { sequences });
        let rpc = RpcSystem::new(Box::new(network), Some(client.client));
        let _ = rpc.await;
    });
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
    let sequences = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let peer_sequences = std::sync::Arc::clone(&sequences);
    let peer = thread::spawn(move || {
        let (first, _) = listener.accept().unwrap();
        thread::sleep(Duration::from_millis(80));
        drop(first);

        let (replay, _) = listener.accept().unwrap();
        let recorded = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        serve_canned_snapshot(replay, std::rc::Rc::clone(&recorded));
        *peer_sequences.lock().unwrap() = recorded.borrow().clone();
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
    peer.join().unwrap();
    assert_eq!(*sequences.lock().unwrap(), vec![17]);
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
