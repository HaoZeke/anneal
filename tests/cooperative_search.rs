#![cfg(feature = "bank-rpc")]

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::thread;

use anneal_core::catalog::{
    BasinCensus, DescriptorSignature, EngineSignature, FreshEvaluation, MixingEvidence,
    SystemSignature, ValidatorConfig,
};
use anneal_core::catalog_policy::{
    ActiveCatalogRelation, AggregateProgress, CatalogPolicyInput, CensusEvidence, PolicyAction,
    ValidationState,
};
use anneal_core::catalog_rpc::CatalogRelation;
use anneal_core::catalog_rpc::client::{CatalogClient, CatalogClientError, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{
    CatalogCandidate, CatalogIdentity, CatalogMutationKind, CatalogRideConnection,
    CatalogRideOutcome, CatalogRideReport, ProtocolRejection, SPARSE_SAMPLE_DRAW,
    TransitionDestination,
};
use anneal_core::cooperative_search::ledger::ChargeKind;
use anneal_core::cooperative_search::{
    CatalogBoundaryOutcome, CatalogHoleOutcome, CatalogOfferOutcome, CatalogSampleOutcome,
    CatalogSamplesOutcome, CooperativeRun, PolicyEvidenceOutcome, PolicyRole,
    PopulationSynchronizationOutcome, ProposalFamily, RideClaimOutcome, RideReportOutcome,
    RunManifest, SliceAdoption, SliceQuench, SliceTrace, SliceValidation, SynchronizationOutcome,
    TraceKind, TransitionRecordOutcome,
};
use anneal_core::descriptor_space::{
    DescriptorBlockKind, DescriptorBlockSpec, DescriptorGeometry, DescriptorSchema,
    DescriptorSpace, universal_descriptor_space,
};
use anneal_core::discovery_roster::DiscoveryRole;
use anneal_core::pes_exploration::ExactStructureWitness;
use anneal_core::ride_ledger::RideFailure;
use anneal_core::transition_graph::AttractionRegionConfig;
use ndarray::ArrayView1;

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
        census_basin: None,
    }
}

fn server() -> CatalogServer {
    server_with_capacity(2)
}

fn server_with_capacity(capacity: usize) -> CatalogServer {
    server_with_region_evidence(capacity, 8)
}

fn server_with_region_evidence(capacity: usize, minimum_probes: u64) -> CatalogServer {
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
            capacity,
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
        .with_attraction_region_config(AttractionRegionConfig {
            probe_action: "probe".into(),
            concentration: 0.5,
            diffusion_steps: 2,
            maximum_distance: 0.35,
            minimum_probes,
        })
        .unwrap();
    CatalogServer::start("127.0.0.1:0", config).unwrap()
}

fn ride_candidate(replica: u32, sequence: u64, separation: f64) -> CatalogCandidate {
    let mut record = candidate(replica, sequence, separation);
    let reaction = (separation - 1.6) / 0.4;
    record.energy = (reaction * reaction - 1.0).powi(2);
    record.forces[3] = -4.0 * reaction * (reaction * reaction - 1.0) / 0.4;
    record.gradient_norm = record.forces[3].abs();
    record
}

fn ride_server() -> CatalogServer {
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
                max_gradient_norm: 1e-7,
                energy_abs_tolerance: 1e-10,
                energy_rel_tolerance: 1e-10,
            },
            4,
            0.05,
            400,
            |coordinates| {
                let separation = coordinates[3];
                let reaction = (separation - 1.6) / 0.4;
                let mut forces = vec![0.0; coordinates.len()];
                forces[3] = -4.0 * reaction * (reaction * reaction - 1.0) / 0.4;
                Ok(FreshEvaluation {
                    energy: (reaction * reaction - 1.0).powi(2),
                    forces,
                })
            },
        )
        .unwrap()
        .with_exact_structure_witness(SeparationWitness)
        .unwrap();
    CatalogServer::start("127.0.0.1:0", config).unwrap()
}

#[test]
fn coordinator_segments_live_replicas_by_shared_pes_coverage() {
    let server = ride_server();
    let digest = signature().digest();
    let query = ride_candidate(0, 1, 1.2);
    let mut clients = (0..4)
        .map(|replica| {
            CatalogClient::connect(
                server.addr(),
                identity(replica, digest),
                ClientConfig::default(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    clients[0].offer_candidate(1, query.clone()).unwrap();

    let states = clients
        .iter_mut()
        .enumerate()
        .map(|(replica, client)| {
            client
                .policy_state(
                    if replica == 0 { 2 } else { 1 },
                    query.descriptor.clone(),
                    query.energy,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(
        states
            .iter()
            .filter(|state| state.discovery_role == DiscoveryRole::BasinEscape)
            .count(),
        2
    );
    assert_eq!(
        states
            .iter()
            .filter(|state| state.discovery_role == DiscoveryRole::SaddleRide)
            .count(),
        2
    );
    assert!(
        states
            .iter()
            .all(|state| state.discovery_epoch == states[0].discovery_epoch)
    );
    assert_eq!(states[0].basin_unseen_mass_upper, 1.0);
    assert_eq!(states[0].saddle_unseen_mass_upper, 1.0);
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
fn coordinator_credits_exact_basin_novelty_only_once_across_replicas() {
    let server = server();
    let digest = signature().digest();
    let mut first =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let mut second =
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap();

    let discovery = first
        .offer_candidate(1, candidate(0, 1, 1.2))
        .unwrap()
        .catalog
        .unwrap();
    let revisit = second
        .offer_candidate(1, candidate(1, 1, 1.2))
        .unwrap()
        .catalog
        .unwrap();

    assert!(discovery.new_basin);
    assert!(!revisit.new_basin);
    assert_eq!(discovery.basin_id, revisit.basin_id);
}

#[test]
fn replicas_share_exclusive_ride_arms_and_coordinator_computes_edge_novelty() {
    let server = ride_server();
    let digest = signature().digest();
    let mut first =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let mut second =
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap();
    let source_a = first
        .offer_candidate(1, ride_candidate(0, 1, 1.2))
        .unwrap()
        .catalog
        .unwrap()
        .basin_id;
    let source_b = second
        .offer_candidate(1, ride_candidate(1, 1, 2.0))
        .unwrap()
        .catalog
        .unwrap()
        .basin_id;

    let first_work = first.claim_ride(2, 8001).unwrap().unwrap();
    let second_work = second.claim_ride(2, 8002).unwrap().unwrap();

    assert_ne!(first_work.order.arm, second_work.order.arm);
    assert!(
        [source_a, source_b].contains(&first_work.order.arm.source_basin),
        "ride must start from a coordinator-certified minimum"
    );
    assert_eq!(
        first_work.source.census_basin,
        Some(first_work.order.arm.source_basin)
    );
    let connection = CatalogRideConnection {
        saddle: ride_candidate(0, 2, 1.6),
        endpoints: [ride_candidate(0, 2, 1.2), ride_candidate(0, 2, 2.0)],
    };
    let first_credit = first
        .report_ride(
            3,
            CatalogRideReport {
                work: first_work.order.id,
                charged_evaluations: 144,
                outcome: CatalogRideOutcome::Certified(connection.clone()),
            },
        )
        .unwrap();
    let second_connection = CatalogRideConnection {
        saddle: ride_candidate(1, 2, 1.6),
        endpoints: [ride_candidate(1, 2, 2.0), ride_candidate(1, 2, 1.2)],
    };
    let duplicate_credit = second
        .report_ride(
            3,
            CatalogRideReport {
                work: second_work.order.id,
                charged_evaluations: 133,
                outcome: CatalogRideOutcome::Certified(second_connection),
            },
        )
        .unwrap();

    assert!(first_credit.novel_saddle);
    assert!(!duplicate_credit.novel_saddle);
    assert!(first_credit.novel_edge);
    assert!(!duplicate_credit.novel_edge);
    assert_eq!(first_credit.total_charged_evaluations, 159);
    assert_eq!(duplicate_credit.total_charged_evaluations, 148);
}

#[test]
fn coordinator_schedules_each_universal_water_environment() {
    let coordinates = vec![0.0, 0.0, 0.0, 0.7572, 0.5865, 0.0, -0.7572, 0.5865, 0.0];
    let species = vec![8, 1, 1];
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.32).unwrap());
    let descriptor = descriptor_space
        .describe(ArrayView1::from(&coordinates), Some(&species))
        .unwrap();
    let signature = SystemSignature {
        atomic_numbers: species.clone(),
        coordinate_dim: coordinates.len().try_into().unwrap(),
        group_labels: vec![0, 1, 2],
        group_schema: "independent-atoms-v1".into(),
        frozen_mask: vec![false; 3],
        cell: None,
        periodic: [false; 3],
        length_scale: 1.32,
        energy_scale: 1.0,
        engine: EngineSignature {
            kind: "water-site-fixture".into(),
            config_digest: [0x71; 32],
            external_inputs: BTreeMap::new(),
        },
        descriptor: DescriptorSignature {
            schema: descriptor.schema_name().into(),
            version: descriptor.schema_version(),
            hyperparameters: BTreeMap::new(),
            species_channels: vec![1, 8],
        },
        validation_schema_version: 1,
    };
    let digest = signature.digest();
    let candidate = CatalogCandidate {
        producer_replica: 0,
        coordinates: coordinates.clone(),
        cell: None,
        energy: -1.0,
        forces: vec![0.0; coordinates.len()],
        gradient_norm: 0.0,
        descriptor: descriptor.values().to_vec(),
        descriptor_schema_version: descriptor.schema_version(),
        quench_converged: true,
        charged_work: 1,
        event_sequence: 1,
        seed: 17,
        census_basin: None,
    };
    let config = ServerConfig::new("jcc-2026", "scientific-ensemble", digest, [0])
        .unwrap()
        .with_scientific_state(
            signature,
            descriptor_space,
            ValidatorConfig {
                reference_coordinates: coordinates,
                descriptor_dim: descriptor.values().len(),
                min_separation: 0.5,
                coordinate_tolerance: 1e-10,
                max_gradient_norm: 1e-8,
                energy_abs_tolerance: 1e-12,
                energy_rel_tolerance: 1e-12,
            },
            4,
            0.05,
            1_000,
            |coordinates| {
                Ok(FreshEvaluation {
                    energy: -1.0,
                    forces: vec![0.0; coordinates.len()],
                })
            },
        )
        .unwrap()
        .with_exact_structure_witness(SeparationWitness)
        .unwrap();
    let server = CatalogServer::start("127.0.0.1:0", config).unwrap();
    let mut client =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    client.offer_candidate(1, candidate).unwrap();

    let mut atoms = std::collections::BTreeSet::new();
    let mut arms = std::collections::BTreeSet::new();
    let mut sequence = 2;
    for attempt in 0..16 {
        let work = client
            .claim_ride(sequence, 10_000 + attempt)
            .unwrap()
            .unwrap();
        sequence += 1;
        atoms.insert(work.order.representative_atom);
        arms.insert(work.order.arm.clone());
        client
            .report_ride(
                sequence,
                CatalogRideReport {
                    work: work.order.id,
                    charged_evaluations: 1,
                    outcome: CatalogRideOutcome::Failed(RideFailure::ActivationNotEscaped),
                },
            )
            .unwrap();
        sequence += 1;
    }

    assert_eq!(atoms, [0, 1].into_iter().collect());
    assert_eq!(arms.len(), 16);
}

#[test]
fn cooperative_run_routes_certified_ride_work_through_the_live_mailbox() {
    let server = ride_server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        run.offer_candidate(0, ride_candidate(0, 1, 1.2)).unwrap(),
        CatalogOfferOutcome::Admitted
    );
    let RideClaimOutcome::Work(work) = run.claim_ride(0, 8003).unwrap() else {
        panic!("a validated source must produce live ride work")
    };
    let report = CatalogRideReport {
        work: work.order.id,
        charged_evaluations: 144,
        outcome: CatalogRideOutcome::Certified(CatalogRideConnection {
            saddle: ride_candidate(0, 2, 1.6),
            endpoints: [ride_candidate(0, 2, 1.2), ride_candidate(0, 2, 2.0)],
        }),
    };
    let RideReportOutcome::Credited(credit) = run.report_ride(0, report).unwrap() else {
        panic!("a receiving-certified index-one connection must be credited")
    };

    assert!(credit.novel_saddle);
    assert!(credit.novel_edge);
    assert_eq!(credit.total_charged_evaluations, 159);
    let claim = run
        .events()
        .iter()
        .find(|event| event.kind == TraceKind::RideClaim)
        .and_then(|event| event.ride.as_ref())
        .expect("a claimed experiment must carry its complete arm identity");
    assert_eq!(claim.work, work.order.id);
    assert_eq!(claim.source_basin, Some(work.order.arm.source_basin));
    assert_eq!(
        claim.environment_class,
        Some(work.order.arm.environment_class)
    );
    assert_eq!(claim.mode_rank, Some(work.order.arm.mode_rank));
    assert_eq!(claim.direction, Some(work.order.arm.direction));
    assert_eq!(claim.method, Some(work.order.arm.method));
    assert_eq!(
        claim.representative_atom,
        Some(work.order.representative_atom)
    );
    assert_eq!(claim.attempt, Some(work.order.attempt));
    assert_eq!(claim.seed, Some(work.order.seed));

    let reported = run
        .events()
        .iter()
        .find(|event| event.kind == TraceKind::RideReport)
        .and_then(|event| event.ride.as_ref())
        .expect("a credited experiment must expose producer and receiver evidence");
    assert_eq!(reported.work, work.order.id);
    assert_eq!(reported.producer_charged_evaluations, Some(144));
    assert_eq!(reported.receiver_charged_evaluations, Some(15));
    assert_eq!(reported.total_charged_evaluations, Some(159));
    assert_eq!(reported.producer_certified_connection, Some(true));
    assert_eq!(reported.receiver_certified_connection, Some(true));
    assert_eq!(reported.producer_failure, None);
    assert_eq!(reported.receiver_failure, None);
    assert_eq!(reported.novel_saddle, Some(true));
    assert_eq!(reported.novel_edge, Some(true));

    let trace = run.json_lines(&RunManifest {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        sharing: true,
        engine: anneal_core::compatibility::EngineDescriptor::default(),
    });
    assert!(trace.contains(&format!("\"ride_work\":{}", work.order.id)));
    assert!(trace.contains("\"ride_method\":\"dimer\""));
    assert!(trace.contains("\"ride_receiver_charged\":15"));
    assert!(trace.contains("\"ride_novel_saddle\":true"));
    assert!(trace.contains("\"ride_novel_edge\":true"));
}

#[test]
fn visit_merges_the_posted_leftover_soap_without_a_recompute() {
    let server = server();
    let digest = signature().digest();
    let mut client =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();

    let mut posted = candidate(0, 1, 1.2);
    for value in &mut posted.descriptor {
        *value += 0.05;
    }
    let accepted = client.record_visit(1, posted).unwrap();
    assert_eq!(accepted.version, 1);
    assert_eq!(client.snapshot(2).unwrap().census_visits, 1);
}

#[test]
fn registered_policy_query_assigns_the_live_chain_to_a_census_basin() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0, 1, 2, 3], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();

    let live = candidate(0, 1, 1.2);
    assert!(matches!(
        run.registered_policy_input(0, live, 0, false).unwrap(),
        PolicyEvidenceOutcome::Remote(_)
    ));
    let policy = run
        .events()
        .iter()
        .rev()
        .find_map(|event| event.policy)
        .expect("registered policy query did not emit policy evidence");
    assert!(policy.local_basin.is_some());
    assert_eq!(policy.total_visits, 1);
    assert_eq!(policy.local_basin_visits, 1);
    assert_eq!(policy.local_basin_distance, 0.0);
}

#[test]
fn try_policy_input_does_not_block_the_hop() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    let admitted = candidate(0, 1, 1.2);
    let started = std::time::Instant::now();
    let first = run
        .try_policy_input(0, admitted.descriptor.clone(), admitted.energy, 0, false)
        .unwrap();
    assert!(
        started.elapsed() < std::time::Duration::from_millis(50),
        "try_policy_input waited {:?}",
        started.elapsed()
    );
    assert!(matches!(
        first,
        PolicyEvidenceOutcome::LocalFallback | PolicyEvidenceOutcome::Remote(_)
    ));
    let mut remote = matches!(first, PolicyEvidenceOutcome::Remote(_));
    for _ in 0..200 {
        if remote {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        let next = run
            .try_policy_input(0, admitted.descriptor.clone(), admitted.energy, 0, false)
            .unwrap();
        remote = matches!(next, PolicyEvidenceOutcome::Remote(_));
    }
    assert!(remote, "mailbox never delivered a policy");
}

#[test]
fn pending_policy_input_is_not_an_rpc_failure() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    let admitted = candidate(0, 1, 1.2);

    assert_eq!(
        run.try_policy_input(0, admitted.descriptor, admitted.energy, 0, false)
            .unwrap(),
        PolicyEvidenceOutcome::LocalFallback
    );
    assert!(
        !run.events()
            .iter()
            .any(|event| event.kind == TraceKind::RpcFallback)
    );
}

#[test]
fn try_policy_input_delivers_each_rpc_result_once() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    let admitted = candidate(0, 1, 1.2);

    let mut delivered = false;
    for _ in 0..200 {
        match run
            .try_policy_input(0, admitted.descriptor.clone(), admitted.energy, 0, false)
            .unwrap()
        {
            PolicyEvidenceOutcome::Remote(_) => {
                delivered = true;
                break;
            }
            PolicyEvidenceOutcome::LocalFallback => {
                thread::sleep(std::time::Duration::from_millis(5));
            }
            outcome => panic!("unexpected policy outcome: {outcome:?}"),
        }
    }
    assert!(delivered, "mailbox never delivered a policy");
    assert_eq!(
        run.try_policy_input(0, admitted.descriptor, admitted.energy, 1, false)
            .unwrap(),
        PolicyEvidenceOutcome::LocalFallback,
        "a completed policy must not be replayed while another RPC is pending"
    );
}

#[test]
fn try_policy_input_requires_the_request_state() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    let first = candidate(0, 1, 1.2);
    let second = candidate(0, 2, 2.0);

    assert_eq!(
        run.try_policy_input(0, first.descriptor, first.energy, 0, false)
            .unwrap(),
        PolicyEvidenceOutcome::LocalFallback
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert_eq!(
        run.try_policy_input(0, second.descriptor.clone(), second.energy, 0, false)
            .unwrap(),
        PolicyEvidenceOutcome::LocalFallback,
        "policy evidence for another state must not drive this checkpoint"
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert!(matches!(
        run.try_policy_input(0, second.descriptor, second.energy, 0, false)
            .unwrap(),
        PolicyEvidenceOutcome::Remote(_)
    ));
}

#[test]
fn try_descriptor_hole_does_not_block_the_hop() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
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
    let started = std::time::Instant::now();
    let first = run
        .try_descriptor_hole(0, admitted.descriptor.clone(), 32, 7)
        .unwrap();
    assert!(
        started.elapsed() < std::time::Duration::from_millis(50),
        "try_descriptor_hole waited {:?}",
        started.elapsed()
    );
    assert!(matches!(
        first,
        CatalogHoleOutcome::LocalFallback | CatalogHoleOutcome::Proposal(_)
    ));
    let mut got = matches!(first, CatalogHoleOutcome::Proposal(_));
    for _ in 0..200 {
        if got {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        let next = run
            .try_descriptor_hole(0, admitted.descriptor.clone(), 32, 7)
            .unwrap();
        got = matches!(next, CatalogHoleOutcome::Proposal(_));
    }
    assert!(got, "mailbox never delivered a catalog hole");
}

#[test]
fn try_descriptor_hole_delivers_each_rpc_result_once() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
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

    let mut delivered = false;
    for _ in 0..200 {
        match run
            .try_descriptor_hole(0, admitted.descriptor.clone(), 32, 7)
            .unwrap()
        {
            CatalogHoleOutcome::Proposal(_) => {
                delivered = true;
                break;
            }
            CatalogHoleOutcome::LocalFallback => {
                thread::sleep(std::time::Duration::from_millis(5));
            }
            outcome => panic!("unexpected descriptor-hole outcome: {outcome:?}"),
        }
    }
    assert!(delivered, "mailbox never delivered a catalog hole");
    assert_eq!(
        run.try_descriptor_hole(0, admitted.descriptor, 32, 8)
            .unwrap(),
        CatalogHoleOutcome::LocalFallback,
        "a completed descriptor hole must not be replayed while another RPC is pending"
    );
}

#[test]
fn try_descriptor_hole_requires_the_request_parameters() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    let admitted = candidate(0, 1, 1.2);
    let other = candidate(0, 2, 2.0);
    assert_eq!(
        run.offer_candidate(0, admitted.clone()).unwrap(),
        CatalogOfferOutcome::Admitted
    );

    assert_eq!(
        run.try_descriptor_hole(0, admitted.descriptor.clone(), 32, 7)
            .unwrap(),
        CatalogHoleOutcome::LocalFallback
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert_eq!(
        run.try_descriptor_hole(0, admitted.descriptor.clone(), 64, 7)
            .unwrap(),
        CatalogHoleOutcome::LocalFallback,
        "a proposal computed with another sample count must be discarded"
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert_eq!(
        run.try_descriptor_hole(0, admitted.descriptor.clone(), 64, 8)
            .unwrap(),
        CatalogHoleOutcome::LocalFallback,
        "a proposal computed with another draw must be discarded"
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert_eq!(
        run.try_descriptor_hole(0, other.descriptor.clone(), 64, 8)
            .unwrap(),
        CatalogHoleOutcome::LocalFallback,
        "a proposal computed from another descriptor must be discarded"
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert!(matches!(
        run.try_descriptor_hole(0, other.descriptor, 64, 8).unwrap(),
        CatalogHoleOutcome::Proposal(_)
    ));
}

#[test]
fn try_sample_candidate_delivers_each_rpc_result_once() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
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
    let expected = CatalogCandidate {
        census_basin: Some(0),
        ..admitted
    };

    let mut delivered = None;
    for _ in 0..200 {
        match run.try_sample_candidate(0, 91).unwrap() {
            CatalogSampleOutcome::Candidate(sampled) => {
                delivered = Some(sampled);
                break;
            }
            CatalogSampleOutcome::LocalFallback => {
                thread::sleep(std::time::Duration::from_millis(5));
            }
            outcome => panic!("unexpected sample outcome: {outcome:?}"),
        }
    }
    assert_eq!(
        delivered.expect("mailbox never delivered a catalog sample"),
        expected
    );
    assert_eq!(
        run.try_sample_candidate(0, 92).unwrap(),
        CatalogSampleOutcome::LocalFallback,
        "a completed sample must not be replayed while another RPC is pending"
    );
}

#[test]
fn try_sample_candidate_keeps_sample_roles_separate() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    let admitted = candidate(0, 1, 1.2);
    assert_eq!(
        run.offer_candidate(0, admitted).unwrap(),
        CatalogOfferOutcome::Admitted
    );

    assert_eq!(
        run.try_sample_candidate(0, 91).unwrap(),
        CatalogSampleOutcome::LocalFallback
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert_eq!(
        run.try_sample_candidate(0, SPARSE_SAMPLE_DRAW).unwrap(),
        CatalogSampleOutcome::LocalFallback,
        "a generic catalog result must not satisfy a sparse-family request"
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert!(matches!(
        run.try_sample_candidate(0, SPARSE_SAMPLE_DRAW).unwrap(),
        CatalogSampleOutcome::Candidate(_)
    ));
}

#[test]
fn try_sample_candidates_pipelines_every_reference_draw() {
    let server = server_with_capacity(2);
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 400).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    for admitted in [candidate(0, 1, 1.2), candidate(0, 2, 2.0)] {
        assert_eq!(
            run.offer_candidate(0, admitted).unwrap(),
            CatalogOfferOutcome::Admitted
        );
    }

    assert_eq!(
        run.try_sample_candidates(0, [0, 1]).unwrap(),
        CatalogSamplesOutcome::LocalFallback
    );
    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    let CatalogSamplesOutcome::Candidates(first) = run.try_sample_candidates(0, [2, 3]).unwrap()
    else {
        panic!("the completed two-draw batch must be delivered")
    };
    let mut first_separations = first
        .iter()
        .map(|candidate| candidate.coordinates[3])
        .collect::<Vec<_>>();
    first_separations.sort_by(f64::total_cmp);
    assert_eq!(first_separations, vec![1.2, 2.0]);

    assert!(matches!(
        run.synchronize(0).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    let CatalogSamplesOutcome::Candidates(second) = run.try_sample_candidates(0, [4, 5]).unwrap()
    else {
        panic!("consuming one batch must pipeline the succeeding draws")
    };
    let mut second_separations = second
        .iter()
        .map(|candidate| candidate.coordinates[3])
        .collect::<Vec<_>>();
    second_separations.sort_by(f64::total_cmp);
    assert_eq!(second_separations, vec![1.2, 2.0]);
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
    assert_eq!(sampled.census_basin, Some(0));

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
fn catalog_trace_records_admission_eviction_and_incumbent_identity() {
    let server = server_with_capacity(1);
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 100).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();

    assert_eq!(
        run.offer_candidate(0, candidate(0, 1, 1.2)).unwrap(),
        CatalogOfferOutcome::Admitted
    );
    let added = run.events().last().unwrap().catalog.as_ref().unwrap();
    assert_eq!(added.basin_id, 0);
    assert_eq!(added.kind, CatalogMutationKind::Added);
    assert!(added.evicted.is_empty());
    assert_eq!(added.incumbent_basin, Some(0));

    assert_eq!(
        run.offer_candidate(0, candidate(0, 2, 2.0)).unwrap(),
        CatalogOfferOutcome::Admitted
    );
    let replacement = run.events().last().unwrap().catalog.as_ref().unwrap();
    assert_eq!(replacement.basin_id, 1);
    assert_eq!(replacement.kind, CatalogMutationKind::ReplacedCapacity);
    assert_eq!(replacement.evicted, vec![0]);
    assert_eq!(replacement.incumbent_basin, Some(1));

    let CatalogSampleOutcome::Candidate(sampled) = run.sample_candidate(0, 0).unwrap() else {
        panic!("active catalog must return its sole representative")
    };
    assert_eq!(sampled.census_basin, Some(1));

    let trace = run.json_lines(&RunManifest {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        sharing: true,
        engine: anneal_core::compatibility::EngineDescriptor::default(),
    });
    assert!(trace.contains("\"catalog_basin\":1"));
    assert!(trace.contains("\"catalog_mutation\":\"replaced_capacity\""));
    assert!(trace.contains("\"catalog_evicted\":[0]"));
    assert!(trace.contains("\"catalog_incumbent\":1"));
}

#[test]
fn policy_state_exposes_hard_lj_diagnostic_boundaries() {
    let server = server();
    let digest = signature().digest();
    let mut client =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let first = candidate(0, 1, 1.2);
    let first_descriptor = first.descriptor.clone();

    client.offer_candidate(1, first).unwrap();
    client.offer_candidate(2, candidate(0, 2, 2.0)).unwrap();
    let state = client.policy_state(3, first_descriptor, -1.2).unwrap();

    assert_eq!(state.local_basin, Some(0));
    assert_eq!(state.local_basin_distance, 0.0);
    assert!(state.novelty.is_finite() && state.novelty > 0.0);
    assert!(state.transition_uncertainty.is_finite() && state.transition_uncertainty > 0.0);
}

#[test]
fn only_explicit_probe_transitions_update_transition_uncertainty() {
    let server = server();
    let digest = signature().digest();
    let mut client =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let current = candidate(0, 1, 1.2);
    let descriptor = current.descriptor.clone();

    client.record_visit(1, current.clone()).unwrap();
    let initialized = client
        .policy_state(2, descriptor.clone(), current.energy)
        .unwrap();
    client.offer_candidate(3, candidate(0, 2, 2.0)).unwrap();
    let after_offer = client
        .policy_state(4, descriptor.clone(), current.energy)
        .unwrap();
    assert_eq!(
        after_offer.transition_uncertainty,
        initialized.transition_uncertainty
    );

    client
        .record_transition(
            5,
            "probe",
            TransitionDestination::Resolved(candidate(0, 3, 1.2)),
            false,
        )
        .unwrap();
    let after_probe = client
        .policy_state(6, descriptor.clone(), current.energy)
        .unwrap();
    assert!(after_probe.transition_uncertainty < after_offer.transition_uncertainty);

    client
        .record_transition(7, "probe", TransitionDestination::Unresolved, false)
        .unwrap();
    let after_unresolved = client.policy_state(8, descriptor, current.energy).unwrap();
    assert!(after_unresolved.transition_uncertainty < after_probe.transition_uncertainty);
    assert_eq!(after_unresolved.local_basin, initialized.local_basin);
}

#[test]
fn policy_relation_uses_fixed_probe_attraction_regions() {
    let server = server();
    let digest = signature().digest();
    let mut first =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let mut second =
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap();
    let first_state = candidate(0, 1, 1.2);
    let second_state = candidate(1, 1, 2.0);

    first.record_visit(1, first_state.clone()).unwrap();
    second.record_visit(1, second_state.clone()).unwrap();
    for sequence in 2..10 {
        first
            .record_transition(
                sequence,
                "probe",
                TransitionDestination::Resolved(candidate(0, sequence, 2.0)),
                false,
            )
            .unwrap();
        second
            .record_transition(
                sequence,
                "probe",
                TransitionDestination::Resolved(candidate(1, sequence, 2.0)),
                false,
            )
            .unwrap();
    }

    let relation = first
        .policy_state(10, first_state.descriptor, first_state.energy)
        .unwrap()
        .relation;
    assert_eq!(
        relation,
        CatalogRelation::SameBasin,
        "replicas with the same fixed-probe return dynamics must share an attraction region"
    );
}

#[test]
fn attraction_region_evidence_threshold_is_explicit() {
    let server = server_with_region_evidence(2, 1);
    let digest = signature().digest();
    let mut first =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let mut second =
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap();
    let first_state = candidate(0, 1, 1.2);
    let second_state = candidate(1, 1, 2.0);
    first.record_visit(1, first_state.clone()).unwrap();
    second.record_visit(1, second_state.clone()).unwrap();
    first
        .record_transition(
            2,
            "probe",
            TransitionDestination::Resolved(candidate(0, 2, 2.0)),
            false,
        )
        .unwrap();
    second
        .record_transition(
            2,
            "probe",
            TransitionDestination::Resolved(candidate(1, 2, 2.0)),
            false,
        )
        .unwrap();

    assert_eq!(
        first
            .policy_state(3, first_state.descriptor, first_state.energy)
            .unwrap()
            .relation,
        CatalogRelation::SameBasin
    );
}

#[test]
fn observed_adopted_crossing_is_available_to_another_replica() {
    let server = server();
    let digest = signature().digest();
    let mut producer =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let mut consumer =
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap();
    let source = candidate(0, 1, 1.2);
    let source_descriptor = source.descriptor.clone();
    let destination = candidate(0, 2, 4.0);

    producer.record_visit(1, source.clone()).unwrap();
    producer
        .record_transition(
            2,
            "surface_relocate",
            TransitionDestination::Resolved(destination.clone()),
            true,
        )
        .unwrap();
    let crossing = consumer
        .boundary_crossing(1, source_descriptor, 71)
        .unwrap()
        .expect("an adopted inter-basin edge must be shareable");

    assert_eq!(crossing.action, "surface_relocate");
    assert_eq!(crossing.from, source.coordinates);
    assert_eq!(crossing.to, destination.coordinates);
    assert_ne!(crossing.source_basin, crossing.destination_basin);
}

#[test]
fn cooperative_run_exposes_an_observed_boundary_crossing() {
    let server = server();
    let digest = signature().digest();
    let mut producer =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let source = candidate(0, 1, 1.2);
    let source_descriptor = source.descriptor.clone();
    producer.record_visit(1, source).unwrap();
    producer
        .record_transition(
            2,
            "surface_relocate",
            TransitionDestination::Resolved(candidate(0, 2, 4.0)),
            true,
        )
        .unwrap();
    let mut run = CooperativeRun::new([1], 100).unwrap();
    run.attach_client(
        1,
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();

    let CatalogBoundaryOutcome::Crossing(crossing) =
        run.boundary_crossing(1, source_descriptor, 71).unwrap()
    else {
        panic!("cooperative run must expose the shared crossing")
    };

    assert_eq!(crossing.action, "surface_relocate");
    assert_ne!(crossing.source_basin, crossing.destination_basin);
}

#[test]
fn try_boundary_crossing_delivers_each_rpc_result_once() {
    let server = server();
    let digest = signature().digest();
    let mut producer =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let source = candidate(0, 1, 1.2);
    let source_descriptor = source.descriptor.clone();
    producer.record_visit(1, source).unwrap();
    producer
        .record_transition(
            2,
            "surface_relocate",
            TransitionDestination::Resolved(candidate(0, 2, 4.0)),
            true,
        )
        .unwrap();
    let mut run = CooperativeRun::new([1], 100).unwrap();
    run.attach_client(
        1,
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();

    let mut delivered = false;
    for _ in 0..200 {
        match run
            .try_boundary_crossing(1, source_descriptor.clone(), 71)
            .unwrap()
        {
            CatalogBoundaryOutcome::Crossing(_) => {
                delivered = true;
                break;
            }
            CatalogBoundaryOutcome::LocalFallback => {
                thread::sleep(std::time::Duration::from_millis(5));
            }
            outcome => panic!("unexpected boundary-crossing outcome: {outcome:?}"),
        }
    }
    assert!(delivered, "mailbox never delivered a boundary crossing");
    assert_eq!(
        run.try_boundary_crossing(1, source_descriptor, 72).unwrap(),
        CatalogBoundaryOutcome::LocalFallback,
        "a completed boundary crossing must not be replayed while another RPC is pending"
    );
}

#[test]
fn try_boundary_crossing_requires_the_request_parameters() {
    let server = server();
    let digest = signature().digest();
    let mut producer =
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap();
    let source = candidate(0, 1, 1.2);
    let source_descriptor = source.descriptor.clone();
    let destination = candidate(0, 2, 4.0);
    let destination_descriptor = destination.descriptor.clone();
    producer.record_visit(1, source).unwrap();
    producer
        .record_transition(
            2,
            "surface_relocate",
            TransitionDestination::Resolved(destination),
            true,
        )
        .unwrap();
    let mut run = CooperativeRun::new([1], 100).unwrap();
    run.attach_client(
        1,
        CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();

    assert_eq!(
        run.try_boundary_crossing(1, source_descriptor.clone(), 71)
            .unwrap(),
        CatalogBoundaryOutcome::LocalFallback
    );
    assert!(matches!(
        run.synchronize(1).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert_eq!(
        run.try_boundary_crossing(1, source_descriptor, 72).unwrap(),
        CatalogBoundaryOutcome::LocalFallback,
        "a crossing selected with another draw must be discarded"
    );
    assert!(matches!(
        run.synchronize(1).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert_eq!(
        run.try_boundary_crossing(1, destination_descriptor.clone(), 72)
            .unwrap(),
        CatalogBoundaryOutcome::LocalFallback,
        "a crossing selected from another descriptor must be discarded"
    );
    assert!(matches!(
        run.synchronize(1).unwrap(),
        SynchronizationOutcome::Refreshed(_)
    ));
    assert_eq!(
        run.try_boundary_crossing(1, destination_descriptor, 72)
            .unwrap(),
        CatalogBoundaryOutcome::Empty
    );
}

#[test]
fn cooperative_run_traces_explicit_transition_records() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 100).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();

    assert_eq!(
        run.record_current(0, candidate(0, 1, 1.2)).unwrap(),
        TransitionRecordOutcome::Recorded
    );
    assert_eq!(
        run.record_transition(
            0,
            "probe",
            TransitionDestination::Resolved(candidate(0, 2, 1.2)),
            false,
        )
        .unwrap(),
        TransitionRecordOutcome::Recorded
    );

    let event = run.events().last().unwrap();
    assert_eq!(event.kind, TraceKind::Transition);
    let transition = event.transition.as_ref().unwrap();
    assert_eq!(transition.action, "probe");
    assert!(transition.resolved);
    assert!(!transition.adopted);
    let trace = run.json_lines(&RunManifest {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        sharing: true,
        engine: anneal_core::compatibility::EngineDescriptor::default(),
    });
    assert!(trace.contains("\"transition_action\":\"probe\""));
    assert!(trace.contains("\"transition_resolved\":true"));
    assert!(trace.contains("\"transition_adopted\":false"));
}

#[test]
fn cooperative_run_traces_local_execution_without_a_coordinator() {
    let mut run = CooperativeRun::new([0], 100).unwrap();

    run.record_executed_transition(
        0,
        17,
        "boundary_transport",
        -170.713_101,
        -172.877_736,
        true,
    )
    .unwrap();

    let event = run.events().last().unwrap();
    assert_eq!(event.kind, TraceKind::TransitionExecution);
    let transition = event.transition.as_ref().unwrap();
    assert_eq!(transition.action, "boundary_transport");
    assert_eq!(transition.hop, Some(17));
    assert_eq!(transition.from_energy, Some(-170.713_101));
    assert_eq!(transition.to_energy, Some(-172.877_736));
    assert!(transition.resolved);
    assert!(transition.adopted);

    let trace = run.json_lines(&RunManifest {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        sharing: true,
        engine: anneal_core::compatibility::EngineDescriptor::default(),
    });
    assert!(trace.contains("\"kind\":\"transition_execution\""));
    assert!(trace.contains("\"transition_hop\":17"));
    assert!(trace.contains("\"transition_from_energy\":-170.713101"));
    assert!(trace.contains("\"transition_to_energy\":-172.877736"));
}

#[test]
fn cooperative_trace_records_policy_diagnostic_evidence() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 100).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();
    let first = candidate(0, 1, 1.2);
    let first_descriptor = first.descriptor.clone();

    assert_eq!(
        run.offer_candidate(0, first).unwrap(),
        CatalogOfferOutcome::Admitted
    );
    assert_eq!(
        run.offer_candidate(0, candidate(0, 2, 1.0)).unwrap(),
        CatalogOfferOutcome::Admitted
    );
    assert!(matches!(
        run.policy_input(0, first_descriptor, -1.2, 3, false)
            .unwrap(),
        PolicyEvidenceOutcome::Remote(_)
    ));

    let event = run.events().last().unwrap();
    assert_eq!(event.kind, TraceKind::SnapshotRefresh);
    let evidence = event
        .policy
        .expect("policy refresh must retain the exact coordinator evidence");
    assert_eq!(evidence.local_basin, Some(0));
    assert_eq!(evidence.relation, CatalogRelation::Incumbent);
    assert_eq!(evidence.total_visits, 2);
    assert_eq!(evidence.singleton_basins, 2);
    assert_eq!(evidence.local_basin_visits, 1);
    assert!(!evidence.globally_saturated);
    assert_eq!(evidence.local_basin_distance, 0.0);
    assert!(evidence.novelty.is_finite() && evidence.novelty > 0.0);
    assert!(evidence.transition_uncertainty.is_finite() && evidence.transition_uncertainty > 0.0);
    assert_eq!(evidence.query_energy, -1.2);

    let trace = run.json_lines(&RunManifest {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        sharing: true,
        engine: anneal_core::compatibility::EngineDescriptor::default(),
    });
    assert!(trace.contains("\"policy_local_basin\":0"));
    assert!(trace.contains("\"policy_relation\":\"incumbent\""));
    assert!(trace.contains("\"policy_total_visits\":2"));
    assert!(trace.contains("\"policy_transition_uncertainty\":"));
    assert!(trace.contains("\"policy_query_energy\":-1.2"));
}

#[test]
fn every_slice_has_one_complete_transition_diagnostic() {
    let mut run = CooperativeRun::new([0], 100).unwrap();
    run.record_work(0, ChargeKind::AcceptedQuench, 7).unwrap();
    run.record_slice(
        0,
        SliceTrace {
            slice: 1,
            current_basin: Some(3),
            active_relation: Some(CatalogRelation::SameBasin),
            policy_role: PolicyRole::Explore,
            policy_reason: "observed_boundary_crossing",
            proposal_family: ProposalFamily::BoundaryTransport,
            sampled_basin: Some(7),
            descriptor_step_norm: None,
            cartesian_step_norm: Some(0.5),
            validation: SliceValidation::Accepted,
            quench: SliceQuench::Converged,
            adoption: SliceAdoption::Adopted,
            novelty: Some(0.75),
            energy: Some(-397.492),
            charged_work: 7,
        },
    )
    .unwrap();

    let event = run.events().last().unwrap();
    assert_eq!(event.kind, TraceKind::Slice);
    let slice = event
        .slice
        .expect("a slice event must retain its complete diagnostic");
    assert_eq!(slice.slice, 1);
    assert_eq!(slice.current_basin, Some(3));
    assert_eq!(slice.policy_role, PolicyRole::Explore);
    assert_eq!(slice.proposal_family, ProposalFamily::BoundaryTransport);
    assert_eq!(slice.validation, SliceValidation::Accepted);
    assert_eq!(slice.quench, SliceQuench::Converged);
    assert_eq!(slice.adoption, SliceAdoption::Adopted);
    assert_eq!(slice.charged_work, 7);

    let trace = run.json_lines(&RunManifest {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        sharing: true,
        engine: anneal_core::compatibility::EngineDescriptor::default(),
    });
    assert!(trace.contains("\"kind\":\"slice\""));
    assert!(trace.contains("\"slice\":1"));
    assert!(trace.contains("\"slice_current_basin\":3"));
    assert!(trace.contains("\"slice_policy_role\":\"explore\""));
    assert!(trace.contains("\"slice_proposal_family\":\"boundary_transport\""));
    assert!(trace.contains("\"slice_validation\":\"accepted\""));
    assert!(trace.contains("\"slice_quench\":\"converged\""));
    assert!(trace.contains("\"slice_adoption\":\"adopted\""));
    assert!(trace.contains("\"slice_novelty\":0.75"));
    assert!(trace.contains("\"slice_energy\":-397.492"));
    assert!(trace.contains("\"slice_charged_work\":7"));
}

#[test]
fn coordinator_closes_population_epoch_only_after_all_replicas_submit() {
    let server = server();
    let digest = signature().digest();
    let mut clients = (0..4)
        .map(|replica| {
            CatalogClient::connect(
                server.addr(),
                identity(replica, digest),
                ClientConfig::default(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    for (replica, client) in clients.iter_mut().enumerate() {
        client
            .offer_candidate(1, candidate(replica as u32, 1, 1.2))
            .unwrap();
    }
    for (replica, client) in clients.iter_mut().enumerate() {
        let state = client
            .submit_population(2, 0, candidate(replica as u32, 2, 1.2))
            .unwrap();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.required, 4);
        if replica < 3 {
            assert_eq!(state.submitted, replica as u32 + 1);
            assert!(state.plan.is_none());
        } else {
            let plan = state.plan.expect("fourth submission closes the epoch");
            assert_eq!(plan.destinations, vec![0, 1, 2, 3]);
            assert_eq!(plan.parents, vec![0, 1, 2, 3]);
            assert_eq!(plan.parent_candidates.len(), 4);
            assert_eq!(plan.unique_parents, 4);
            assert_eq!(plan.max_family_size, 1);
        }
    }

    for client in clients.iter_mut().take(3) {
        let plan = client
            .population_plan(3, 0)
            .unwrap()
            .plan
            .expect("completed epoch must be pollable by every replica");
        assert_eq!(plan.destinations, vec![0, 1, 2, 3]);
        assert_eq!(plan.parents, vec![0, 1, 2, 3]);
    }
    assert_eq!(clients[0].snapshot(4).unwrap().census_visits, 4);
}

#[test]
fn population_barrier_covers_distinct_regions_before_duplicate_families() {
    let server = server();
    let digest = signature().digest();
    let mut clients = (0..4)
        .map(|replica| {
            CatalogClient::connect(
                server.addr(),
                identity(replica, digest),
                ClientConfig::default(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let separations = [4.0, 4.0, 1.0, 1.2];

    for (replica, client) in clients.iter_mut().enumerate() {
        let _ = client
            .offer_candidate(1, candidate(replica as u32, 1, separations[replica]))
            .unwrap();
    }
    let mut completed = None;
    for (replica, client) in clients.iter_mut().enumerate() {
        let state = client
            .submit_population(2, 0, candidate(replica as u32, 2, separations[replica]))
            .unwrap();
        if let Some(plan) = state.plan {
            completed = Some(plan);
        }
    }
    let plan = completed.expect("complete barrier must return a plan");
    let represented = plan
        .parent_candidates
        .iter()
        .map(|candidate| candidate.census_basin.unwrap())
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(represented.len(), 3);
    assert!(plan.max_family_size <= 2);
}

#[test]
fn cooperative_run_exposes_population_barrier_and_assigned_parent() {
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
        assert!(matches!(
            run.synchronize(replica).unwrap(),
            SynchronizationOutcome::Refreshed(_)
        ));
        assert_eq!(
            run.offer_candidate(replica, candidate(replica, 1, 1.2))
                .unwrap(),
            if replica == 0 {
                CatalogOfferOutcome::Admitted
            } else {
                CatalogOfferOutcome::Rejected
            }
        );
    }

    for replica in 0..3 {
        assert_eq!(
            run.submit_population(replica, 0, candidate(replica, 2, 1.2))
                .unwrap(),
            PopulationSynchronizationOutcome::Pending {
                submitted: replica + 1,
                required: 4,
            }
        );
    }
    let PopulationSynchronizationOutcome::Ready { parent, plan } =
        run.submit_population(3, 0, candidate(3, 2, 1.2)).unwrap()
    else {
        panic!("complete population must return the destination parent")
    };
    assert_eq!(parent.producer_replica, 3);
    assert_eq!(plan.parents, vec![0, 1, 2, 3]);

    let PopulationSynchronizationOutcome::Ready { parent, plan } =
        run.poll_population(0, 0).unwrap()
    else {
        panic!("closed population epoch must remain pollable")
    };
    assert_eq!(parent.producer_replica, 0);
    assert_eq!(plan.unique_parents, 4);
    assert_eq!(plan.max_family_size, 1);

    let ahead = run.join_population(0, 5).unwrap();
    assert!(
        matches!(
            ahead,
            PopulationSynchronizationOutcome::Pending { .. }
                | PopulationSynchronizationOutcome::Unaddressed
                | PopulationSynchronizationOutcome::Rejected
        ),
        "a join on a non-open epoch must not kill the walk: {ahead:?}"
    );
    assert!(
        run.events()
            .iter()
            .any(|event| event.kind == TraceKind::PopulationPending)
    );
    assert!(
        run.events()
            .iter()
            .any(|event| event.kind == TraceKind::PopulationReady)
    );
    let ready = run
        .events()
        .iter()
        .rev()
        .find(|event| event.kind == TraceKind::PopulationReady)
        .unwrap()
        .population
        .unwrap();
    assert_eq!(ready.epoch, 0);
    assert_eq!(ready.parent, 0);
    assert_eq!(ready.family_ordinal, 0);
    assert_eq!(ready.family_size, 1);
    assert!((ready.effective_sample_size - 4.0).abs() < 1e-12);

    let trace = run.json_lines(&RunManifest {
        campaign: "jcc-2026".into(),
        ensemble: "scientific-ensemble".into(),
        sharing: true,
        engine: anneal_core::compatibility::EngineDescriptor::default(),
    });
    assert!(trace.contains("\"population_epoch\":0"));
    assert!(trace.contains("\"population_parent\":0"));
    assert!(trace.contains("\"population_family_size\":1"));
}

#[test]
fn cooperative_run_builds_policy_input_from_exact_remote_evidence() {
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
        run.record_work(replica, ChargeKind::AcceptedQuench, 60)
            .unwrap();
        // The aggregate this run reads below is the coordinator's, and
        // record_work only posts toward it.
        run.flush(replica).unwrap();
    }
    let admitted = candidate(0, 1, 1.2);
    assert_eq!(
        run.offer_candidate(0, admitted.clone()).unwrap(),
        CatalogOfferOutcome::Admitted
    );

    let outcome = run
        .policy_input(0, admitted.descriptor.clone(), admitted.energy, 3, false)
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
    assert!(input.progress.win_only());
    assert_eq!(input.local_stall_slices, 3);
    assert!(!input.local_deepened);

    let CatalogSampleOutcome::Candidate(sampled) = run.sample_candidate(0, 91).unwrap() else {
        panic!("active scientific catalog must return a sampled candidate")
    };
    assert_eq!(sampled.coordinates, admitted.coordinates);
    let CatalogHoleOutcome::Proposal(hole) = run
        .descriptor_hole(0, admitted.descriptor, 128, 73)
        .unwrap()
    else {
        panic!("active scientific catalog must return a descriptor-hole proposal")
    };
    assert!(hole.nearest_catalog_distance.is_finite());
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
fn distributed_trace_uses_the_coordinator_aggregate_counter() {
    let server = server();
    let digest = signature().digest();
    let mut first = CooperativeRun::new([0], 100).unwrap();
    let mut second = CooperativeRun::new([1], 100).unwrap();
    first
        .attach_client(
            0,
            CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
                .unwrap(),
        )
        .unwrap();
    second
        .attach_client(
            1,
            CatalogClient::connect(server.addr(), identity(1, digest), ClientConfig::default())
                .unwrap(),
        )
        .unwrap();
    first
        .record_work(0, ChargeKind::AcceptedQuench, 60)
        .unwrap();
    second
        .record_work(1, ChargeKind::AcceptedQuench, 40)
        .unwrap();
    // record_work posts and returns, so the aggregate the coordinator
    // reports next is only guaranteed to include replica 1 once its
    // mailbox has drained.
    second.flush(1).unwrap();
    first.synchronize(0).unwrap();

    assert_eq!(first.ledger().ensemble_total(), 60);
    assert_eq!(first.events().last().unwrap().aggregate_charged, 100);
}

#[test]
fn work_batch_keeps_every_local_boundary_and_one_remote_aggregate() {
    let server = server();
    let digest = signature().digest();
    let mut run = CooperativeRun::new([0], 100).unwrap();
    run.attach_client(
        0,
        CatalogClient::connect(server.addr(), identity(0, digest), ClientConfig::default())
            .unwrap(),
    )
    .unwrap();

    run.record_work_batch(
        0,
        [
            (ChargeKind::AcceptedQuench, 7),
            (ChargeKind::DescriptorEvaluation, 0),
            (ChargeKind::RejectedQuench, 4),
        ],
    )
    .unwrap();
    run.flush(0).unwrap();
    run.synchronize(0).unwrap();

    assert_eq!(run.ledger().event_count(), 3);
    assert_eq!(run.ledger().ensemble_total(), 11);
    assert_eq!(run.events().last().unwrap().aggregate_charged, 11);
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
        run.flush(replica).unwrap();
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
        mixing: MixingEvidence::default(),
        leftover_lambda: 0.0,
        interface_rank: u32::MAX,
        interface_threshold: 0.0,
        occupied_family_count: 0,
        packing_saturated: false,
        leftover_dwell: true,
        ei_exhausted: false,
        min_families: 1,
        on_published_prize: false,
    };
    assert_eq!(
        run.decide(
            0,
            input(ActiveCatalogRelation::Unrelated {
                lower_energy_anchor: true,
            })
        )
        .unwrap()
        .action,
        PolicyAction::Exploit { win_only: false }
    );
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
        engine: anneal_core::compatibility::EngineDescriptor::default(),
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
        run.policy_input(0, vec![0.0; 9], -1.0, 0, false,).unwrap(),
        PolicyEvidenceOutcome::SharingDisabled
    );
    assert_eq!(
        run.sample_candidate(0, 91).unwrap(),
        CatalogSampleOutcome::SharingDisabled
    );
    assert_eq!(
        run.descriptor_hole(0, vec![0.0; 9], 128, 73).unwrap(),
        CatalogHoleOutcome::SharingDisabled
    );
    assert_eq!(run.ledger().ensemble_total(), 20);
    assert!(
        !run.events()
            .iter()
            .any(|event| event.kind == TraceKind::RpcFallback)
    );
}

#[test]
fn server_loss_preserves_independent_local_trajectory_and_ledger() {
    let mut independent = CooperativeRun::new([0], 100).unwrap();
    let mut disconnected = CooperativeRun::new([0], 100).unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let peer = thread::spawn(move || drop(listener.accept().unwrap()));
    let client = CatalogClient::connect(
        address,
        identity(0, signature().digest()),
        ClientConfig::default(),
    )
    .unwrap();
    peer.join().unwrap();
    disconnected.attach_client(0, client).unwrap();

    let mut independent_state = 1.0_f64;
    let mut disconnected_state = 1.0_f64;
    for (step, charged_calls) in [3_u64, 5, 7].into_iter().enumerate() {
        let local_transition = |state: f64| state.mul_add(0.5, -(step as f64 + 1.0));
        independent_state = local_transition(independent_state);
        disconnected_state = local_transition(disconnected_state);

        independent
            .record_work(0, ChargeKind::AcceptedQuench, charged_calls)
            .unwrap();
        disconnected
            .record_work(0, ChargeKind::AcceptedQuench, charged_calls)
            .unwrap();

        assert_eq!(
            independent.synchronize(0).unwrap(),
            SynchronizationOutcome::SharingDisabled
        );
        assert_eq!(
            disconnected.synchronize(0).unwrap(),
            SynchronizationOutcome::LocalFallback
        );
    }

    assert_eq!(disconnected_state, independent_state);
    assert_eq!(
        disconnected.ledger().replica_total(0),
        independent.ledger().replica_total(0)
    );
    assert_eq!(
        disconnected.ledger().ensemble_total(),
        independent.ledger().ensemble_total()
    );
    assert_eq!(
        disconnected.ledger().event_count(),
        independent.ledger().event_count()
    );
    assert!(
        disconnected
            .events()
            .iter()
            .any(|event| event.kind == TraceKind::RpcFallback)
    );
}
