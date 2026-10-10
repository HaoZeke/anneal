#![cfg(feature = "bank-rpc")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::catalog::{
    DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
};
use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{
    AcceptedPayload, AcceptedReply, CatalogCandidate, CatalogIdentity, CatalogMutationKind,
    CatalogOperation, CatalogReply, CatalogRequest, PROTOCOL_VERSION, ProtocolRejection,
};
use anneal_core::cooperative_search::ledger::ChargeKind;
use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorSchema, DescriptorSpace,
};
use anneal_core::pes_exploration::ExactStructureWitness;
use ndarray::ArrayView1;

struct ExactCoordinates;

impl ExactStructureWitness for ExactCoordinates {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

struct Fixture {
    client: CatalogClient,
    server: CatalogServer,
    request: CatalogRequest,
    accepted: AcceptedReply,
    fresh_evaluations: Arc<AtomicUsize>,
}

fn fixture() -> Fixture {
    let coordinates = vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
    let space = DescriptorSpace::new(
        DescriptorSchema::new(
            "replay-guards-soap",
            1,
            vec![DescriptorBlockSpec::new(DescriptorBlockKind::SoapMean, 2, 2, 3.5).unwrap()],
        )
        .unwrap(),
    );
    let signature = SystemSignature {
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
            kind: "harmonic-fixture".into(),
            config_digest: [0x31; 32],
            external_inputs: BTreeMap::new(),
        },
        descriptor: DescriptorSignature {
            schema: "replay-guards-soap".into(),
            version: 1,
            hyperparameters: BTreeMap::new(),
            species_channels: vec![18],
        },
        validation_schema_version: 1,
    };
    let digest = signature.digest();
    let descriptor = space
        .describe(
            ArrayView1::from(&coordinates),
            Some(&signature.atomic_numbers),
        )
        .unwrap()
        .values()
        .to_vec();
    let fresh_evaluations = Arc::new(AtomicUsize::new(0));
    let evaluator_calls = Arc::clone(&fresh_evaluations);
    let reference = coordinates.clone();
    let config = ServerConfig::new("replay-guards", "one-candidate", digest, [0])
        .unwrap()
        .with_scientific_state(
            signature,
            space,
            ValidatorConfig {
                reference_coordinates: coordinates.clone(),
                descriptor_dim: descriptor.len(),
                min_separation: 0.8,
                coordinate_tolerance: 1e-10,
                max_gradient_norm: 1e-8,
                energy_abs_tolerance: 1e-12,
                energy_rel_tolerance: 1e-12,
            },
            2,
            0.05,
            32,
            move |point| {
                evaluator_calls.fetch_add(1, Ordering::SeqCst);
                let displacement: Vec<_> = point
                    .iter()
                    .zip(reference.iter())
                    .map(|(x, center)| x - center)
                    .collect();
                Ok(FreshEvaluation {
                    energy: -1.0 + 0.5 * displacement.iter().map(|x| x * x).sum::<f64>(),
                    forces: displacement.iter().map(|x| -x).collect(),
                })
            },
        )
        .unwrap()
        .with_exact_structure_witness(ExactCoordinates)
        .unwrap();
    let server = CatalogServer::start("127.0.0.1:0", config).unwrap();
    let identity = CatalogIdentity {
        campaign: "replay-guards".into(),
        ensemble: "one-candidate".into(),
        replica: 0,
        signature_digest: digest,
    };
    let mut client =
        CatalogClient::connect(server.addr(), identity.clone(), ClientConfig::default()).unwrap();
    let ledger = client
        .record_ledger_event(1, ChargeKind::AcceptedQuench, 7, 7)
        .unwrap();
    assert_eq!(fresh_evaluations.load(Ordering::SeqCst), 0);
    let request = CatalogRequest {
        protocol_version: PROTOCOL_VERSION,
        identity,
        event_sequence: 2,
        snapshot_version: ledger.snapshot.version,
        operation: CatalogOperation::OfferCandidate {
            candidate: CatalogCandidate {
                producer_replica: 0,
                coordinates,
                cell: None,
                energy: -1.0,
                forces: vec![0.0; 6],
                gradient_norm: 0.0,
                descriptor,
                descriptor_schema_version: 1,
                quench_converged: true,
                charged_work: 7,
                event_sequence: 2,
                seed: 7,
                census_basin: None,
            },
        },
    };
    let CatalogReply::Accepted(accepted) = client.session_call(request.clone()).unwrap() else {
        panic!("the stationary harmonic candidate must be accepted");
    };
    assert!(!accepted.duplicate);
    let AcceptedPayload::CatalogMutation(mutation) = &accepted.payload else {
        panic!("an accepted offer must return its catalog admission");
    };
    assert_eq!(mutation.kind, CatalogMutationKind::Added);
    assert!(mutation.new_basin);
    assert_eq!(mutation.basin_visits, 1);
    assert_eq!(accepted.snapshot.census_visits, 1);
    assert_eq!(accepted.snapshot.active_entries, 1);
    assert_eq!(accepted.snapshot.aggregate_charged, 7);
    assert_eq!(accepted.snapshot.aggregate_budget, 32);
    assert_eq!(fresh_evaluations.load(Ordering::SeqCst), 1);
    Fixture {
        client,
        server,
        request,
        accepted,
        fresh_evaluations,
    }
}

#[test]
fn foreign_identities_cannot_reach_a_cached_request_or_repeat_validation() {
    let mut fixture = fixture();
    let mut wrong_campaign = fixture.request.identity.clone();
    wrong_campaign.campaign = "foreign-campaign".into();
    let mut wrong_ensemble = fixture.request.identity.clone();
    wrong_ensemble.ensemble = "foreign-ensemble".into();
    let mut wrong_signature = fixture.request.identity.clone();
    wrong_signature.signature_digest[0] ^= 1;

    for (identity, expected_reason) in [
        (wrong_campaign, ProtocolRejection::CampaignMismatch),
        (wrong_ensemble, ProtocolRejection::EnsembleMismatch),
        (wrong_signature, ProtocolRejection::SignatureMismatch),
    ] {
        let mut foreign = CatalogClient::connect(
            fixture.server.addr(),
            identity.clone(),
            ClientConfig::default(),
        )
        .unwrap();
        let mut request = fixture.request.clone();
        request.identity = identity;
        let CatalogReply::Rejected { reason, .. } = foreign.session_call(request).unwrap() else {
            panic!("a foreign identity must not receive the cached admission");
        };
        assert_eq!(reason, expected_reason);
        assert_eq!(fixture.fresh_evaluations.load(Ordering::SeqCst), 1);
    }
    assert_eq!(
        fixture.client.snapshot(3).unwrap(),
        fixture.accepted.snapshot
    );
}

#[test]
fn conflicting_cached_request_is_rejected_without_fresh_validation() {
    let mut fixture = fixture();
    let mut conflict = fixture.request.clone();
    let CatalogOperation::OfferCandidate { candidate } = &mut conflict.operation else {
        panic!("the fixture caches a candidate offer");
    };
    candidate.seed += 1;
    let CatalogReply::Rejected { reason, .. } = fixture.client.session_call(conflict).unwrap()
    else {
        panic!("changed request content must not receive the cached admission");
    };
    assert_eq!(reason, ProtocolRejection::SequenceReplay);
    assert_eq!(
        fixture.client.snapshot(3).unwrap(),
        fixture.accepted.snapshot
    );
    assert_eq!(fixture.fresh_evaluations.load(Ordering::SeqCst), 1);
}

#[test]
fn replay_preserves_the_original_payload_and_reports_current_ledger_state() {
    let mut fixture = fixture();
    let advanced = fixture
        .client
        .record_ledger_event(3, ChargeKind::RejectedQuench, 3, 10)
        .unwrap();
    assert!(!advanced.duplicate);
    assert_eq!(
        advanced.snapshot.version,
        fixture.accepted.snapshot.version + 1
    );
    assert_eq!(advanced.snapshot.aggregate_charged, 10);
    assert_eq!(advanced.snapshot.aggregate_budget, 32);
    assert_eq!(advanced.snapshot.census_visits, 1);
    assert_eq!(advanced.snapshot.active_entries, 1);
    assert_eq!(fixture.fresh_evaluations.load(Ordering::SeqCst), 1);

    let replay = fixture
        .client
        .session_call(fixture.request.clone())
        .unwrap();
    let mut expected = fixture.accepted.clone();
    expected.duplicate = true;
    expected.snapshot = advanced.snapshot;
    assert_eq!(replay, CatalogReply::Accepted(expected));
    assert_eq!(fixture.client.snapshot(4).unwrap(), advanced.snapshot);
    assert_eq!(fixture.fresh_evaluations.load(Ordering::SeqCst), 1);
}
