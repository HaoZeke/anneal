#![cfg(feature = "bank-rpc")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use anneal_core::Catalog_capnp::{RejectionKind, catalog_reply, catalog_request};
use anneal_core::catalog_rpc::client::{CatalogClient, CatalogClientError, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{
    CatalogCandidate, CatalogIdentity, CatalogLedgerEvent, PROTOCOL_VERSION, ProtocolRejection,
};
use anneal_core::cooperative_search::ledger::ChargeKind;
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;

fn identity(ensemble: &str, replica: u32) -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: ensemble.into(),
        replica,
        signature_digest: [0x5a; 32],
    }
}

fn candidate(replica: u32, sequence: u64, basin_marker: f64) -> CatalogCandidate {
    CatalogCandidate {
        producer_replica: replica,
        coordinates: vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
        cell: None,
        energy: -1.0,
        forces: vec![0.0; 6],
        gradient_norm: 0.0,
        descriptor: vec![basin_marker, 0.2],
        descriptor_schema_version: 1,
        quench_converged: true,
        charged_work: sequence,
        event_sequence: sequence,
        seed: 100 + u64::from(replica),
        census_basin: None,
    }
}

#[test]
fn isolated_server_starts_with_a_verifiable_empty_snapshot() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-07", [0x5a; 32], [0, 1, 2, 3]).unwrap(),
    )
    .unwrap();
    assert!(server.header().empty_state_proof);
    assert_eq!(server.header().initial_snapshot_version, 0);
    assert_eq!(server.header().replicas, vec![0, 1, 2, 3]);

    let mut client = CatalogClient::connect(
        server.addr(),
        identity("ensemble-07", 2),
        ClientConfig::default(),
    )
    .unwrap();
    let snapshot = client.snapshot(1).unwrap();
    assert_eq!(snapshot.version, 0);
    assert_eq!(snapshot.census_visits, 0);
    assert_eq!(snapshot.active_entries, 0);

    let mut foreign = CatalogClient::connect(
        server.addr(),
        identity("ensemble-08", 2),
        ClientConfig::default(),
    )
    .unwrap();
    assert_eq!(
        foreign.snapshot(1).unwrap_err(),
        CatalogClientError::Rejected(ProtocolRejection::EnsembleMismatch)
    );
}

#[test]
fn observer_status_is_isolated_by_system_signature() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "observer-system", [0x5a; 32], [0]).unwrap(),
    )
    .unwrap();
    let mut matching = CatalogClient::connect(
        server.addr(),
        CatalogIdentity {
            campaign: "jcc-2026".into(),
            ensemble: "observer-system".into(),
            replica: u32::MAX,
            signature_digest: [0x5a; 32],
        },
        ClientConfig::default(),
    )
    .unwrap();
    matching.observer_status(1).unwrap();

    let mut foreign = CatalogClient::connect(
        server.addr(),
        CatalogIdentity {
            campaign: "jcc-2026".into(),
            ensemble: "observer-system".into(),
            replica: u32::MAX,
            signature_digest: [0x6b; 32],
        },
        ClientConfig::default(),
    )
    .unwrap();
    assert_eq!(
        foreign.observer_status(1).unwrap_err(),
        CatalogClientError::Rejected(ProtocolRejection::SignatureMismatch)
    );
}

#[test]
fn accepted_connection_remains_open_between_requests() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-idle", [0x5a; 32], [0]).unwrap(),
    )
    .unwrap();
    let mut client = CatalogClient::connect(
        server.addr(),
        identity("ensemble-idle", 0),
        ClientConfig::default(),
    )
    .unwrap();

    assert_eq!(client.snapshot(1).unwrap().version, 0);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(client.snapshot(2).unwrap().version, 0);
}

#[test]
fn duplicate_mutation_is_idempotent_and_snapshot_versions_are_monotone() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-11", [0x5a; 32], [0]).unwrap(),
    )
    .unwrap();
    let mut client = CatalogClient::connect(
        server.addr(),
        identity("ensemble-11", 0),
        ClientConfig::default(),
    )
    .unwrap();

    let first = client.record_visit(1, candidate(0, 1, 0.1)).unwrap();
    let replay = client.record_visit(1, candidate(0, 1, 0.1)).unwrap();
    let snapshot = client.snapshot(2).unwrap();

    assert_eq!(first.version, 1);
    assert!(!first.duplicate);
    assert_eq!(replay.version, 1);
    assert!(replay.duplicate);
    assert_eq!(snapshot.version, 1);
    assert_eq!(snapshot.census_visits, 1);
}

#[test]
fn coordinator_aggregates_replayed_ledger_events_exactly_once() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-ledger", [0x5a; 32], [0, 1])
            .unwrap()
            .with_ledger_budget(100)
            .unwrap(),
    )
    .unwrap();
    let mut first = CatalogClient::connect(
        server.addr(),
        identity("ensemble-ledger", 0),
        ClientConfig::default(),
    )
    .unwrap();
    let recorded = first
        .record_ledger_event(1, ChargeKind::AcceptedQuench, 7, 7)
        .unwrap();
    let replayed = first
        .record_ledger_event(1, ChargeKind::AcceptedQuench, 7, 7)
        .unwrap();
    assert_eq!(recorded.snapshot.aggregate_charged, 7);
    assert_eq!(recorded.snapshot.aggregate_budget, 200);
    assert!(!recorded.duplicate);
    assert_eq!(replayed.snapshot.aggregate_charged, 7);
    assert!(replayed.duplicate);

    let mut second = CatalogClient::connect(
        server.addr(),
        identity("ensemble-ledger", 1),
        ClientConfig::default(),
    )
    .unwrap();
    let aggregate = second
        .record_ledger_event(1, ChargeKind::RejectedQuench, 11, 11)
        .unwrap();
    assert_eq!(aggregate.snapshot.aggregate_charged, 18);
    assert_eq!(aggregate.snapshot.aggregate_budget, 200);
}

#[test]
fn coordinator_applies_a_ledger_batch_as_exact_replay_safe_boundaries() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-ledger-batch", [0x5a; 32], [0])
            .unwrap()
            .with_ledger_budget(100)
            .unwrap(),
    )
    .unwrap();
    let mut client = CatalogClient::connect(
        server.addr(),
        identity("ensemble-ledger-batch", 0),
        ClientConfig::default(),
    )
    .unwrap();
    let events = vec![
        CatalogLedgerEvent {
            sequence: 1,
            kind: ChargeKind::AcceptedQuench.wire_code(),
            charged_calls: 7,
            cumulative_charged: 7,
        },
        CatalogLedgerEvent {
            sequence: 2,
            kind: ChargeKind::DescriptorEvaluation.wire_code(),
            charged_calls: 0,
            cumulative_charged: 7,
        },
        CatalogLedgerEvent {
            sequence: 3,
            kind: ChargeKind::RejectedQuench.wire_code(),
            charged_calls: 4,
            cumulative_charged: 11,
        },
    ];

    let recorded = client.record_ledger_batch(3, events.clone()).unwrap();
    let replayed = client.record_ledger_batch(3, events).unwrap();

    assert_eq!(recorded.snapshot.aggregate_charged, 11);
    assert_eq!(recorded.snapshot.version, 3);
    assert!(!recorded.duplicate);
    assert_eq!(replayed.snapshot.aggregate_charged, 11);
    assert_eq!(replayed.snapshot.version, 3);
    assert!(replayed.duplicate);
    assert_eq!(client.snapshot(4).unwrap().version, 3);
}

#[test]
fn rejected_ledger_batch_does_not_apply_a_valid_prefix() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-ledger-atomic", [0x5a; 32], [0])
            .unwrap()
            .with_ledger_budget(100)
            .unwrap(),
    )
    .unwrap();
    let mut client = CatalogClient::connect(
        server.addr(),
        identity("ensemble-ledger-atomic", 0),
        ClientConfig::default(),
    )
    .unwrap();

    assert_eq!(
        client
            .record_ledger_batch(
                2,
                vec![
                    CatalogLedgerEvent {
                        sequence: 1,
                        kind: ChargeKind::AcceptedQuench.wire_code(),
                        charged_calls: 7,
                        cumulative_charged: 7,
                    },
                    CatalogLedgerEvent {
                        sequence: 2,
                        kind: u16::MAX,
                        charged_calls: 0,
                        cumulative_charged: 7,
                    },
                ],
            )
            .unwrap_err(),
        CatalogClientError::Rejected(ProtocolRejection::ValidationRejected)
    );
    let recorded = client
        .record_ledger_event(1, ChargeKind::AcceptedQuench, 7, 7)
        .unwrap();
    assert_eq!(recorded.snapshot.aggregate_charged, 7);
    assert_eq!(recorded.snapshot.version, 1);
}

#[test]
fn durable_ledger_batch_is_one_frame_and_replays_every_boundary() {
    let directory = PathBuf::from(format!(
        "/tmp/anneal-catalog-batch-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let config = ServerConfig::new("jcc-2026", "ensemble-batch-durable", [0x5a; 32], [0])
        .unwrap()
        .with_ledger_budget(100)
        .unwrap()
        .with_state_directory(&directory)
        .unwrap();
    let events = vec![
        CatalogLedgerEvent {
            sequence: 1,
            kind: ChargeKind::AcceptedQuench.wire_code(),
            charged_calls: 7,
            cumulative_charged: 7,
        },
        CatalogLedgerEvent {
            sequence: 2,
            kind: ChargeKind::DescriptorEvaluation.wire_code(),
            charged_calls: 0,
            cumulative_charged: 7,
        },
        CatalogLedgerEvent {
            sequence: 3,
            kind: ChargeKind::RejectedQuench.wire_code(),
            charged_calls: 4,
            cumulative_charged: 11,
        },
    ];
    {
        let server = CatalogServer::start("127.0.0.1:0", config.clone()).unwrap();
        let mut client = CatalogClient::connect(
            server.addr(),
            identity("ensemble-batch-durable", 0),
            ClientConfig::default(),
        )
        .unwrap();
        assert_eq!(
            client
                .record_ledger_batch(3, events.clone())
                .unwrap()
                .snapshot
                .version,
            3
        );
    }

    let journal_path = directory.join("catalog-requests-v5.bin");
    let mut journal = std::fs::File::open(&journal_path).unwrap();
    let mut prefix = [0_u8; 8];
    journal.read_exact(&mut prefix).unwrap();
    let length = usize::try_from(u64::from_le_bytes(prefix)).unwrap();
    let mut payload = vec![0_u8; length];
    journal.read_exact(&mut payload).unwrap();
    assert_eq!(journal.read(&mut prefix).unwrap(), 0);

    let restarted = CatalogServer::start("127.0.0.1:0", config).unwrap();
    assert_eq!(restarted.header().initial_snapshot_version, 3);
    let mut replay = CatalogClient::connect(
        restarted.addr(),
        identity("ensemble-batch-durable", 0),
        ClientConfig::default(),
    )
    .unwrap();
    let receipt = replay.record_ledger_batch(3, events).unwrap();
    assert!(receipt.duplicate);
    assert_eq!(receipt.snapshot.aggregate_charged, 11);
    assert_eq!(receipt.snapshot.version, 3);

    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn coordinator_restart_replays_its_durable_request_journal() {
    let directory = PathBuf::from(format!(
        "/tmp/anneal-catalog-journal-{}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("catalog-rpc"),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let config = ServerConfig::new("jcc-2026", "ensemble-durable", [0x5a; 32], [0])
        .unwrap()
        .with_ledger_budget(100)
        .unwrap()
        .with_state_directory(&directory)
        .unwrap();
    {
        let server = CatalogServer::start("127.0.0.1:0", config.clone()).unwrap();
        let mut client = CatalogClient::connect(
            server.addr(),
            identity("ensemble-durable", 0),
            ClientConfig::default(),
        )
        .unwrap();
        let receipt = client
            .record_ledger_event(1, ChargeKind::AcceptedQuench, 7, 7)
            .unwrap();
        assert_eq!(receipt.snapshot.aggregate_charged, 7);
    }

    let restarted = CatalogServer::start("127.0.0.1:0", config).unwrap();
    assert!(!restarted.header().empty_state_proof);
    assert_eq!(restarted.header().initial_snapshot_version, 1);
    let mut replay = CatalogClient::connect(
        restarted.addr(),
        identity("ensemble-durable", 0),
        ClientConfig::default(),
    )
    .unwrap();
    let receipt = replay
        .record_ledger_event(1, ChargeKind::AcceptedQuench, 7, 7)
        .unwrap();
    assert!(receipt.duplicate);
    assert_eq!(receipt.snapshot.aggregate_charged, 7);
    assert_eq!(replay.snapshot(2).unwrap().version, 1);

    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn concurrent_replicas_observe_one_serialized_mutation_order() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-21", [0x5a; 32], 0..4).unwrap(),
    )
    .unwrap();
    let barrier = Arc::new(Barrier::new(4));
    let mut workers = Vec::new();
    for replica in 0..4 {
        let barrier = Arc::clone(&barrier);
        let addr = server.addr();
        workers.push(thread::spawn(move || {
            let mut client = CatalogClient::connect(
                addr,
                identity("ensemble-21", replica),
                ClientConfig::default(),
            )
            .unwrap();
            barrier.wait();
            client
                .record_visit(1, candidate(replica, 1, f64::from(replica)))
                .unwrap()
                .version
        }));
    }
    let mut versions = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    versions.sort_unstable();
    assert_eq!(versions, vec![1, 2, 3, 4]);

    let mut observer = CatalogClient::connect(
        server.addr(),
        identity("ensemble-21", 0),
        ClientConfig::default(),
    )
    .unwrap();
    let snapshot = observer.snapshot(2).unwrap();
    assert_eq!(snapshot.version, 4);
    assert_eq!(snapshot.census_visits, 4);
}

#[test]
fn reconnect_replays_exact_requests_and_rejects_conflicting_content() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-31", [0x5a; 32], [0]).unwrap(),
    )
    .unwrap();
    let mut first = CatalogClient::connect(
        server.addr(),
        identity("ensemble-31", 0),
        ClientConfig::default(),
    )
    .unwrap();
    assert_eq!(
        first.record_visit(1, candidate(0, 1, 0.1)).unwrap().version,
        1
    );
    drop(first);

    let mut replay = CatalogClient::connect(
        server.addr(),
        identity("ensemble-31", 0),
        ClientConfig::default(),
    )
    .unwrap();
    assert!(
        replay
            .record_visit(1, candidate(0, 1, 0.1))
            .unwrap()
            .duplicate
    );
    drop(replay);

    let mut conflict = CatalogClient::connect(
        server.addr(),
        identity("ensemble-31", 0),
        ClientConfig::default(),
    )
    .unwrap();
    assert_eq!(
        conflict.record_visit(1, candidate(0, 1, 0.3)).unwrap_err(),
        CatalogClientError::Rejected(ProtocolRejection::SequenceReplay)
    );
}

#[test]
fn parseable_wire_failures_are_structured_and_do_not_mutate_state() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-41", [0x5a; 32], [0]).unwrap(),
    )
    .unwrap();

    assert_eq!(
        raw_rejection(server.addr(), PROTOCOL_VERSION + 1, &[0x5a; 32]),
        (41, 0, RejectionKind::UnsupportedVersion)
    );
    assert_eq!(
        raw_rejection(server.addr(), PROTOCOL_VERSION, &[0x5a; 31]),
        (41, 0, RejectionKind::Malformed)
    );

    let mut observer = CatalogClient::connect(
        server.addr(),
        identity("ensemble-41", 0),
        ClientConfig::default(),
    )
    .unwrap();
    assert_eq!(observer.snapshot(1).unwrap().version, 0);
}

fn raw_rejection(
    addr: std::net::SocketAddr,
    version: u16,
    digest: &[u8],
) -> (u64, u64, RejectionKind) {
    let mut stream = TcpStream::connect(addr).unwrap();
    let mut message = Builder::new_default();
    let mut root = message.init_root::<catalog_request::Builder>();
    root.set_protocol_version(version);
    root.set_event_sequence(41);
    root.set_snapshot_version(0);
    {
        let mut wire_identity = root.reborrow().init_identity();
        wire_identity.set_campaign("jcc-2026");
        wire_identity.set_ensemble("ensemble-41");
        wire_identity.set_replica(0);
        wire_identity.set_signature_digest(digest);
    }
    root.init_operation().set_snapshot(());
    serialize::write_message(&mut stream, &message).unwrap();
    stream.flush().unwrap();

    let reply = serialize::read_message(&mut stream, ReaderOptions::new()).unwrap();
    let root = reply.get_root::<catalog_reply::Reader>().unwrap();
    let reason = match root.get_result().which().unwrap() {
        catalog_reply::result::Rejected(reason) => reason.unwrap(),
        catalog_reply::result::Accepted(_) => panic!("invalid request was accepted"),
    };
    (
        root.get_event_sequence(),
        root.get_snapshot_version(),
        reason,
    )
}

use anneal_core::catalog::{
    DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
};
use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorSchema, DescriptorSpace,
};
use anneal_core::pes_exploration::ExactStructureWitness;
use anneal_core::scaling::SuccessiveHalving;
use ndarray::ArrayView1;
use std::collections::BTreeMap;

struct SeparationWitness;

impl ExactStructureWitness for SeparationWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        let separation = |point: ArrayView1<'_, f64>| {
            (0..3)
                .map(|axis| (point[3 + axis] - point[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        (separation(left) - separation(right)).abs() < 1e-8
    }
}

fn roster_descriptor_space() -> DescriptorSpace {
    DescriptorSpace::new(
        DescriptorSchema::new(
            "cooperative-test-soap",
            1,
            vec![DescriptorBlockSpec::new(DescriptorBlockKind::SoapMean, 2, 2, 3.5).unwrap()],
        )
        .unwrap(),
    )
}

fn roster_signature() -> SystemSignature {
    SystemSignature {
        atomic_numbers: vec![18, 18],
        coordinate_dim: 6,
        group_labels: vec![0, 1],
        group_schema: "independent-atoms-v1".into(),
        frozen_mask: vec![false, false],
        cell: None,
        periodic: [false; 3],
        length_scale: 1.0,
        energy_scale: 1.0,
        engine: EngineSignature {
            kind: "fixture".into(),
            config_digest: [0x31; 32],
            external_inputs: BTreeMap::new(),
        },
        descriptor: DescriptorSignature {
            schema: "cooperative-test-soap".into(),
            version: 1,
            hyperparameters: BTreeMap::new(),
            species_channels: vec![18],
        },
        validation_schema_version: 1,
    }
}

fn roster_identity(ensemble: &str, replica: u32, digest: [u8; 32]) -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: ensemble.into(),
        replica,
        signature_digest: digest,
    }
}

fn roster_candidate(replica: u32, sequence: u64, separation: f64) -> CatalogCandidate {
    let coordinates = vec![0.0, 0.0, 0.0, separation, 0.0, 0.0];
    let descriptor = roster_descriptor_space()
        .describe(ArrayView1::from(&coordinates), Some(&[18, 18]))
        .unwrap()
        .values()
        .to_vec();
    CatalogCandidate {
        producer_replica: replica,
        coordinates,
        cell: None,
        energy: -separation,
        forces: vec![0.0; 6],
        gradient_norm: 0.0,
        descriptor,
        descriptor_schema_version: 1,
        quench_converged: true,
        charged_work: sequence * 5,
        event_sequence: sequence,
        seed: 1000 + u64::from(replica),
        census_basin: None,
    }
}

fn scientific_config(
    ensemble: &str,
    replicas: impl IntoIterator<Item = u32>,
) -> ServerConfig {
    let signature = roster_signature();
    let digest = signature.digest();
    ServerConfig::new("jcc-2026", ensemble, digest, replicas)
        .unwrap()
        .with_scientific_state(
            signature,
            roster_descriptor_space(),
            ValidatorConfig {
                reference_coordinates: vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
                descriptor_dim: 9,
                min_separation: 0.8,
                coordinate_tolerance: 1e-10,
                max_gradient_norm: 1e-8,
                energy_abs_tolerance: 1e-12,
                energy_rel_tolerance: 1e-12,
            },
            2,
            0.05,
            400,
            |coordinates| {
                Ok(FreshEvaluation {
                    energy: -coordinates[3],
                    forces: vec![0.0; coordinates.len()],
                })
            },
        )
        .unwrap()
        .with_exact_structure_witness(SeparationWitness)
        .unwrap()
}

#[test]
fn attach_admits_a_new_replica_and_status_lists_it() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "roster-attach", [0x5a; 32], [0]).unwrap(),
    )
    .unwrap();
    let mut walker = CatalogClient::connect(
        server.addr(),
        identity("roster-attach", 7),
        ClientConfig::default(),
    )
    .unwrap();
    let roster = walker.attach(1).unwrap();
    assert!(roster.live.contains(&7));
    assert!(roster.version >= 1);

    let mut observer = CatalogClient::connect(
        server.addr(),
        CatalogIdentity {
            campaign: "jcc-2026".into(),
            ensemble: "roster-attach".into(),
            replica: u32::MAX,
            signature_digest: [0x5a; 32],
        },
        ClientConfig::default(),
    )
    .unwrap();
    let status = observer.observer_status(1).unwrap();
    assert!(status.live_replicas.contains(&7));
}

#[test]
fn a_non_attached_unknown_replica_is_rejected_on_snapshot() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "roster-unknown", [0x5a; 32], [0]).unwrap(),
    )
    .unwrap();
    let mut stranger = CatalogClient::connect(
        server.addr(),
        identity("roster-unknown", 7),
        ClientConfig::default(),
    )
    .unwrap();
    assert_eq!(
        stranger.snapshot(1).unwrap_err(),
        CatalogClientError::Rejected(ProtocolRejection::ReplicaMismatch)
    );
}

#[test]
fn detach_leaves_the_barrier_so_a_two_replica_epoch_closes_on_one_submission() {
    let ensemble = "roster-detach";
    let config = scientific_config(ensemble, [0, 1]);
    let digest = roster_signature().digest();
    let server = CatalogServer::start("127.0.0.1:0", config).unwrap();
    let mut lead = CatalogClient::connect(
        server.addr(),
        roster_identity(ensemble, 0, digest),
        ClientConfig::default(),
    )
    .unwrap();
    let mut other = CatalogClient::connect(
        server.addr(),
        roster_identity(ensemble, 1, digest),
        ClientConfig::default(),
    )
    .unwrap();
    let offered = roster_candidate(0, 1, 1.2);
    lead.record_visit(1, offered.clone()).unwrap();
    let pending = lead.submit_population(2, 0, offered).unwrap();
    assert!(pending.plan.is_none());
    assert_eq!(pending.submitted, 1);
    other.detach(1, "done").unwrap();
    let closed = lead.population_plan(3, 0).unwrap();
    assert!(closed.plan.is_some());
}

#[test]
fn ticks_close_a_half_submitted_epoch_and_replay_to_the_same_open_epoch() {
    let ensemble = "roster-tick";
    let directory = PathBuf::from(format!(
        "/tmp/anneal-catalog-tick-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let digest = roster_signature().digest();
    let config = scientific_config(ensemble, [0, 1])
        .with_quorum(0.5, 3)
        .unwrap()
        .with_state_directory(&directory)
        .unwrap();
    let server = CatalogServer::start("127.0.0.1:0", config.clone()).unwrap();
    let mut lead = CatalogClient::connect(
        server.addr(),
        roster_identity(ensemble, 0, digest),
        ClientConfig::default(),
    )
    .unwrap();
    let mut clock = CatalogClient::connect(
        server.addr(),
        roster_identity(ensemble, u32::MAX, digest),
        ClientConfig::default(),
    )
    .unwrap();
    let offered = roster_candidate(0, 1, 1.2);
    lead.record_visit(1, offered.clone()).unwrap();
    let pending = lead.submit_population(2, 0, offered).unwrap();
    assert!(pending.plan.is_none());
    for sequence in 1..=3 {
        clock.tick(sequence, 1000).unwrap();
    }
    let closed = lead.population_plan(3, 0).unwrap();
    assert!(closed.plan.is_some());
    let mut observer = CatalogClient::connect(
        server.addr(),
        roster_identity(ensemble, u32::MAX, digest),
        ClientConfig::default(),
    )
    .unwrap();
    let open = observer.observer_status(4).unwrap().open_epoch;
    drop(server);

    let replayed = CatalogServer::start("127.0.0.1:0", config).unwrap();
    let mut replay_observer = CatalogClient::connect(
        replayed.addr(),
        roster_identity(ensemble, u32::MAX, digest),
        ClientConfig::default(),
    )
    .unwrap();
    assert_eq!(replay_observer.observer_status(1).unwrap().open_epoch, open);
}

#[test]
fn halving_retires_the_worst_of_three_replicas_and_requests_one_spawn() {
    let ensemble = "roster-halving";
    let digest = roster_signature().digest();
    let config = scientific_config(ensemble, [0, 1, 2])
        .with_ledger_budget(100)
        .unwrap()
        .with_halving(SuccessiveHalving::new(10, 2.0, 3).unwrap());
    let server = CatalogServer::start("127.0.0.1:0", config).unwrap();
    for (replica, separation) in [(0, 1.3), (1, 1.2), (2, 1.1)] {
        let mut client = CatalogClient::connect(
            server.addr(),
            roster_identity(ensemble, replica, digest),
            ClientConfig::default(),
        )
        .unwrap();
        client
            .record_visit(1, roster_candidate(replica, 1, separation))
            .unwrap();
        client
            .record_ledger_event(2, ChargeKind::AcceptedQuench, 10, 10)
            .unwrap();
    }
    let mut worst = CatalogClient::connect(
        server.addr(),
        roster_identity(ensemble, 2, digest),
        ClientConfig::default(),
    )
    .unwrap();
    let offered = roster_candidate(2, 3, 1.1);
    let policy = worst
        .policy_state(3, offered.descriptor, offered.energy)
        .unwrap();
    assert!(policy.retired);
    let mut observer = CatalogClient::connect(
        server.addr(),
        roster_identity(ensemble, u32::MAX, digest),
        ClientConfig::default(),
    )
    .unwrap();
    let status = observer.observer_status(1).unwrap();
    assert_eq!(status.spawn_requested, 1);
}
