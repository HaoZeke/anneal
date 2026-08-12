#![cfg(feature = "bank-rpc")]

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::thread;

use anneal_core::catalog::{
    BasinCensus, DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature,
    ValidatorConfig,
};
use anneal_core::catalog_policy::{
    ActiveCatalogRelation, AggregateProgress, CatalogPolicyInput, CensusEvidence, PolicyAction,
    ValidationState,
};
use anneal_core::catalog_rpc::CatalogRelation;
use anneal_core::catalog_rpc::client::{CatalogClient, CatalogClientError, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{CatalogCandidate, CatalogIdentity, ProtocolRejection};
use anneal_core::cooperative_search::ledger::ChargeKind;
use anneal_core::cooperative_search::{
    CatalogOfferOutcome, CooperativeRun, PolicyEvidenceOutcome, RunManifest,
    SynchronizationOutcome, TraceKind,
};
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
fn catalog_outputs_are_actionable_and_seeded() {
    let server = server();
    let digest = signature().digest();
    let mut client =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let admitted = candidate(0, 1, 1.2);
    client.offer_candidate(1, admitted.clone()).unwrap();

    let sampled = client.sample_candidate(2, 91).unwrap().unwrap();
    assert_eq!(sampled.coordinates, admitted.coordinates);
    assert_eq!(sampled.descriptor, admitted.descriptor);

    let first = client
        .descriptor_hole(3, admitted.descriptor.clone(), 128, 73)
        .unwrap();
    let second = client
        .descriptor_hole(4, admitted.descriptor.clone(), 128, 73)
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.target.len(), admitted.descriptor.len());
    assert_eq!(first.increment.len(), admitted.descriptor.len());
    assert!(first.nearest_catalog_distance.is_finite());

    let policy = client
        .policy_state(5, admitted.descriptor.clone(), admitted.energy)
        .unwrap();
    assert_eq!(policy.total_visits, 1);
    assert_eq!(policy.singleton_basins, 1);
    assert_eq!(policy.local_basin_visits, 1);
    assert!(!policy.globally_saturated);
    assert_eq!(policy.relation, CatalogRelation::Incumbent);
}

#[test]
fn cooperative_run_builds_policy_input_from_exact_remote_evidence() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0, 1, 2, 3], 100).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    let admitted = candidate(0, 1, 1.2);
    assert_eq!(
        run.offer_candidate(0, admitted.clone()).unwrap(),
        CatalogOfferOutcome::Admitted
    );

    let outcome = run
        .policy_input(
            0,
            admitted.descriptor,
            admitted.energy,
            AggregateProgress::new(20, 400).unwrap(),
            3,
            false,
        )
        .unwrap();
    let PolicyEvidenceOutcome::Remote(input) = outcome else {
        panic!("scientific coordinator must return exact policy evidence")
    };
    assert_eq!(input.validation, ValidationState::Validated);
    assert_eq!(input.relation, ActiveCatalogRelation::Incumbent);
    assert_eq!(input.census.total_visits(), 1);
    assert_eq!(input.census.singleton_basins(), 1);
    assert_eq!(input.census.local_basin_visits(), 1);
    assert!(!input.census.globally_saturated());
    assert_eq!(input.local_stall_slices, 3);
    assert!(!input.local_deepened);
}

#[test]
fn candidate_replica_must_match_while_producer_sequence_remains_independent() {
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
        client
            .offer_candidate(2, candidate(1, 8, 1.2))
            .unwrap()
            .version,
        1
    );
    let snapshot = client.snapshot(3).unwrap();
    assert_eq!(snapshot.version, 1);
    assert_eq!(snapshot.census_visits, 1);
    assert_eq!(snapshot.active_entries, 1);
}

#[test]
fn four_replica_trace_covers_policy_ingress_refresh_and_fallback() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0, 1, 2, 3], 100).unwrap();
    for replica in 0..4 {
        run.attach_client(
            replica,
            CatalogClient::connect(
                server.addr(),
                identity(replica, digest),
                ClientConfig::default(),
            )
            .unwrap(),
        )
        .unwrap();
        run.record_work(replica, ChargeKind::AcceptedQuench, 5)
            .unwrap();
    }

    assert_eq!(
        run.offer_candidate(0, candidate(0, 1, 1.2)).unwrap(),
        CatalogOfferOutcome::Admitted
    );
    let mut invalid = candidate(1, 1, 1.3);
    invalid.quench_converged = false;
    assert_eq!(
        run.offer_candidate(1, invalid).unwrap(),
        CatalogOfferOutcome::Rejected
    );
    assert!(matches!(
        run.synchronize(2).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));

    let mut census = BasinCensus::new(9, 0.05).unwrap();
    let descriptor = candidate(0, 1, 1.2).descriptor;
    let mut local_basin = None;
    for _ in 0..8 {
        local_basin = Some(census.observe(&descriptor).unwrap().basin_id);
    }
    let evidence = CensusEvidence::from_census(&census, local_basin);
    let progress = AggregateProgress::new(20, 400).unwrap();
    let input = |relation| CatalogPolicyInput {
        validation: ValidationState::Validated,
        relation,
        census: evidence,
        progress,
        local_stall_slices: 0,
        local_deepened: false,
    };
    assert!(matches!(
        run.decide(
            0,
            input(ActiveCatalogRelation::Unrelated {
                lower_energy_anchor: true,
            })
        )
        .unwrap()
        .action,
        PolicyAction::Exploit { .. }
    ));
    assert_eq!(
        run.decide(
            1,
            input(ActiveCatalogRelation::Unrelated {
                lower_energy_anchor: false,
            })
        )
        .unwrap()
        .action,
        PolicyAction::Explore
    );
    assert_eq!(
        run.decide(2, input(ActiveCatalogRelation::SameBasin))
            .unwrap()
            .action,
        PolicyAction::Leave
    );
    assert_eq!(
        run.decide(3, input(ActiveCatalogRelation::Incumbent))
            .unwrap()
            .action,
        PolicyAction::ContinueLocal
    );

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let peer = thread::spawn(move || drop(listener.accept().unwrap()));
    let failed =
        CatalogClient::connect(addr, identity(3, digest), ClientConfig::default()).unwrap();
    peer.join().unwrap();
    run.attach_client(3, failed).unwrap();
    assert_eq!(
        run.synchronize(3).unwrap(),
        SynchronizationOutcome::LocalFallback
    );

    assert_eq!(run.ledger().ensemble_total(), 20);
    for required in [
        TraceKind::LocalWork,
        TraceKind::Admission,
        TraceKind::Rejection,
        TraceKind::SnapshotRefresh,
        TraceKind::PolicyExploit,
        TraceKind::PolicyExplore,
        TraceKind::PolicyLeave,
        TraceKind::PolicyLocal,
        TraceKind::RpcFallback,
    ] {
        assert!(run.events().iter().any(|event| event.kind == required));
    }
    let lines = run.json_lines(&RunManifest {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        sharing: true,
    });
    assert!(lines.lines().next().unwrap().contains("manifest_header"));
    assert_eq!(lines.lines().count(), run.events().len() + 1);
}

#[test]
fn no_sharing_run_executes_without_a_server_and_preserves_local_accounting() {
    let mut run = CooperativeRun::new([0, 1, 2, 3], 100).unwrap();
    for replica in 0..4 {
        run.record_work(replica, ChargeKind::AcceptedQuench, 5)
            .unwrap();
    }
    assert_eq!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::SharingDisabled
    );
    assert_eq!(
        run.policy_input(
            0,
            vec![0.0; 9],
            -1.0,
            AggregateProgress::new(20, 400).unwrap(),
            0,
            false,
        )
        .unwrap(),
        PolicyEvidenceOutcome::SharingDisabled
    );
    assert_eq!(run.ledger().ensemble_total(), 20);
    assert!(
        !run.events()
            .iter()
            .any(|event| event.kind == TraceKind::RpcFallback)
    );
}
