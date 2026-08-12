#![cfg(feature = "bank-rpc")]

use anneal_core::catalog_rpc::{
    CatalogIdentity, CatalogOperation, CatalogRequest, PROTOCOL_VERSION, ProtocolError,
    decode_request, encode_request, validate_identity,
};

fn identity(ensemble: &str) -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: ensemble.into(),
        replica: 2,
        signature_digest: [0x5a; 32],
    }
}

#[test]
fn every_catalog_operation_round_trips_all_identity_and_sequence_fields() {
    let operations = vec![
        CatalogOperation::Snapshot,
        CatalogOperation::RecordVisit {
            basin_id: 17,
            created: true,
            descriptor: vec![0.1, 0.2, 0.3],
        },
        CatalogOperation::OfferCandidate {
            basin_id: 17,
            energy: -396.282,
            coordinates: vec![0.0, 1.0, 2.0],
            descriptor: vec![0.1, 0.2, 0.3],
            provenance: "replica-2/quench-41".into(),
        },
        CatalogOperation::Sample { draw: 91 },
        CatalogOperation::DescriptorHole { samples: 4096 },
        CatalogOperation::LedgerEvent {
            kind: 5,
            charged_calls: 3,
            cumulative_charged: 81,
        },
    ];
    for operation in operations {
        let request = CatalogRequest {
            protocol_version: PROTOCOL_VERSION,
            identity: identity("ensemble-07"),
            event_sequence: 42,
            snapshot_version: 11,
            operation,
        };

        assert_eq!(
            decode_request(&encode_request(&request).unwrap()).unwrap(),
            request
        );
    }
}

#[test]
fn incompatible_protocol_versions_fail_explicitly() {
    let request = CatalogRequest {
        protocol_version: PROTOCOL_VERSION + 1,
        identity: identity("ensemble-07"),
        event_sequence: 1,
        snapshot_version: 0,
        operation: CatalogOperation::Snapshot,
    };

    assert_eq!(
        encode_request(&request).unwrap_err(),
        ProtocolError::UnsupportedVersion {
            received: PROTOCOL_VERSION + 1,
            supported: PROTOCOL_VERSION,
        }
    );
}

#[test]
fn cross_ensemble_and_cross_signature_requests_are_rejected() {
    let expected = identity("ensemble-07");
    let foreign_ensemble = identity("ensemble-08");
    let mut foreign_signature = expected.clone();
    foreign_signature.signature_digest[0] ^= 1;

    assert_eq!(
        validate_identity(&expected, &foreign_ensemble).unwrap_err(),
        ProtocolError::EnsembleMismatch
    );
    assert_eq!(
        validate_identity(&expected, &foreign_signature).unwrap_err(),
        ProtocolError::SignatureMismatch
    );
}
