#![cfg(feature = "bank-rpc")]

use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{
    CatalogIdentity, CatalogOperation, CatalogRequest, PROTOCOL_VERSION, decode_request,
    encode_request,
};
use anneal_core::coreclass::CoreVerdict;

fn identity(ensemble: &str, replica: u32) -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: ensemble.into(),
        replica,
        signature_digest: [0x5a; 32],
    }
}

#[test]
fn coordinator_core_class_report_round_trips_the_table_rule() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "ensemble-coreclass", [0x5a; 32], [0, 1, 2, 3])
            .unwrap()
            .with_core_class(10_000, 10),
    )
    .unwrap();

    let mut chains: Vec<CatalogClient> = (0..4)
        .map(|replica| {
            CatalogClient::connect(
                server.addr(),
                identity("ensemble-coreclass", replica),
                ClientConfig::default(),
            )
            .unwrap()
        })
        .collect();

    let enter = [1.0, 2.0, 3.0, 4.0];
    for (replica, energy) in enter.iter().enumerate() {
        assert_eq!(
            chains[replica]
                .report_core_class(1, 0, *energy, 0)
                .unwrap(),
            CoreVerdict::Continue
        );
    }
    assert_eq!(
        chains[0].report_core_class(2, 0, 1.0, 10).unwrap(),
        CoreVerdict::Continue
    );
    assert_eq!(
        chains[1].report_core_class(2, 0, 2.0, 10).unwrap(),
        CoreVerdict::Continue
    );
    assert_eq!(
        chains[2].report_core_class(2, 0, 3.0, 10).unwrap(),
        CoreVerdict::Continue
    );
    assert_eq!(
        chains[3].report_core_class(2, 0, 4.0, 10).unwrap(),
        CoreVerdict::Restart
    );
}

#[test]
fn report_core_class_request_round_trips_on_the_wire() {
    let request = CatalogRequest {
        protocol_version: PROTOCOL_VERSION,
        identity: identity("ensemble-coreclass", 2),
        event_sequence: 42,
        snapshot_version: 11,
        operation: CatalogOperation::ReportCoreClass {
            class: 3,
            energy: -173.928,
            charged: 10_000,
        },
    };
    assert_eq!(
        decode_request(&encode_request(&request).unwrap()).unwrap(),
        request
    );
}
