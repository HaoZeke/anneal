#![cfg(feature = "bank-rpc")]

use std::collections::BTreeMap;

use anneal_core::catalog::{
    DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
};
use anneal_core::catalog_rpc::client::{CatalogClient, CatalogClientError, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{CatalogCandidate, CatalogIdentity, ProtocolRejection};
use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorSchema, DescriptorSpace,
};
use ndarray::ArrayView1;

fn descriptor_space() -> DescriptorSpace {
    DescriptorSpace::new(
        DescriptorSchema::new(
            "cooperative-test-soap",
            1,
            vec![DescriptorBlockSpec::new(DescriptorBlockKind::SoapMean, 2, 2, 3.5).unwrap()],
        )
        .unwrap(),
    )
}

fn signature() -> SystemSignature {
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

fn identity(replica: u32, digest: [u8; 32]) -> CatalogIdentity {
    CatalogIdentity {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        replica,
        signature_digest: digest,
    }
}

fn candidate(replica: u32, sequence: u64, separation: f64) -> CatalogCandidate {
    let coordinates = vec![0.0, 0.0, 0.0, separation, 0.0, 0.0];
    let descriptor = descriptor_space()
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
    }
}

fn server() -> CatalogServer {
    let signature = signature();
    let digest = signature.digest();
    let config = ServerConfig::new("jcc-2026", "scientific-ensemble", digest, [0, 1, 2, 3])
        .unwrap()
        .with_scientific_state(
            signature,
            descriptor_space(),
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
        .unwrap();
    CatalogServer::start("127.0.0.1:0", config).unwrap()
}

#[test]
fn coordinator_validates_before_census_and_catalog_mutation() {
    let server = server();
    let digest = signature().digest();
    let mut client =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();

    let admitted = client.offer_candidate(1, candidate(0, 1, 1.2)).unwrap();
    assert_eq!(admitted.version, 1);
    let snapshot = client.snapshot(2).unwrap();
    assert_eq!(snapshot.census_visits, 1);
    assert_eq!(snapshot.active_entries, 1);

    let mut invalid = candidate(0, 3, 1.3);
    invalid.quench_converged = false;
    assert_eq!(
        client.offer_candidate(3, invalid).unwrap_err(),
        CatalogClientError::Rejected(ProtocolRejection::ValidationRejected)
    );
    let unchanged = client.snapshot(4).unwrap();
    assert_eq!(unchanged.version, 1);
    assert_eq!(unchanged.census_visits, 1);
    assert_eq!(unchanged.active_entries, 1);
}

#[test]
fn candidate_identity_must_match_the_authenticated_replica_and_event() {
    let server = server();
    let digest = signature().digest();
    let mut client =
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap();

    assert_eq!(
        client.offer_candidate(1, candidate(2, 1, 1.2)).unwrap_err(),
        CatalogClientError::Rejected(ProtocolRejection::ValidationRejected)
    );
    assert_eq!(
        client.offer_candidate(2, candidate(1, 8, 1.2)).unwrap_err(),
        CatalogClientError::Rejected(ProtocolRejection::ValidationRejected)
    );
    let snapshot = client.snapshot(3).unwrap();
    assert_eq!(snapshot.version, 0);
    assert_eq!(snapshot.census_visits, 0);
    assert_eq!(snapshot.active_entries, 0);
}
