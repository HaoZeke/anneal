#![cfg(feature = "bank-rpc")]

use anneal_core::catalog_rpc::client::{CatalogClient, CatalogClientError, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{CatalogIdentity, ProtocolRejection};

fn identity(ensemble: &str, replica: u32) -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: ensemble.into(),
        replica,
        signature_digest: [0x5a; 32],
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

    let first = client
        .record_visit(1, 17, true, vec![0.1, 0.2, 0.3])
        .unwrap();
    let replay = client
        .record_visit(1, 17, true, vec![0.1, 0.2, 0.3])
        .unwrap();
    let snapshot = client.snapshot(2).unwrap();

    assert_eq!(first.version, 1);
    assert!(!first.duplicate);
    assert_eq!(replay.version, 1);
    assert!(replay.duplicate);
    assert_eq!(snapshot.version, 1);
    assert_eq!(snapshot.census_visits, 1);
}
