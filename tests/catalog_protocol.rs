#![cfg(feature = "bank-rpc")]

use anneal_core::catalog_rpc::{
    decode_request, encode_request, validate_identity, CatalogCandidate, CatalogIdentity,
    CatalogOperation, CatalogRequest, ProtocolError, PROTOCOL_VERSION,
};

fn identity(ensemble: &str) -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: ensemble.into(),
        replica: 2,
        signature_digest: [0x5a; 32],
    }
}

fn candidate() -> CatalogCandidate {
    CatalogCandidate {
        producer_replica: 2,
        coordinates: vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
        cell: Some([8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]),
        energy: -396.282,
        forces: vec![0.0; 6],
        gradient_norm: 1.2e-9,
        descriptor: vec![0.1, 0.2, 0.3],
        descriptor_schema_version: 7,
        quench_converged: true,
        charged_work: 81,
        event_sequence: 42,
        seed: 91,
    }
}

#[test]
fn every_catalog_operation_round_trips_all_identity_and_sequence_fields() {
    let operations = vec![
        CatalogOperation::Snapshot,
        CatalogOperation::RecordVisit {
            candidate: candidate(),
        },
        CatalogOperation::OfferCandidate {
            candidate: candidate(),
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
