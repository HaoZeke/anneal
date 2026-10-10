//! Carrier RTT: 200 catalog snapshots on one live vat.
#![cfg(feature = "bank-rpc")]

use std::time::Instant;

use anneal_core::catalog_rpc::CatalogIdentity;
use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};

fn identity() -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: "carrier-rtt".into(),
        replica: 0,
        signature_digest: [0x5a; 32],
    }
}

#[test]
fn two_hundred_snapshots_print_carrier_rtt() {
    let server = CatalogServer::start(
        "127.0.0.1:0",
        ServerConfig::new("jcc-2026", "carrier-rtt", [0x5a; 32], [0]).unwrap(),
    )
    .unwrap();
    let mut client =
        CatalogClient::connect(server.addr(), identity(), ClientConfig::default()).unwrap();
    client.snapshot(1).unwrap();
    let started = Instant::now();
    for sequence in 2..202 {
        client.snapshot(sequence).unwrap();
    }
    let elapsed = started.elapsed();
    let per = elapsed / 200;
    eprintln!(
        "CARRIER_RTT snapshots=200 total={:?} per={:?} addr={}",
        elapsed,
        per,
        server.addr()
    );
}
