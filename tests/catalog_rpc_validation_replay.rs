#![cfg(feature = "bank-rpc")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::catalog::{
    DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
};
use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{CatalogCandidate, CatalogIdentity, CatalogMutationKind};
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

#[test]
fn exact_accepted_candidate_replay_reuses_fresh_validation() {
    let coordinates = vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
    let space = DescriptorSpace::new(
        DescriptorSchema::new(
            "validation-replay-soap",
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
            schema: "validation-replay-soap".into(),
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
    let config = ServerConfig::new("validation-replay", "one-candidate", digest, [0])
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
    let mut client = CatalogClient::connect(
        server.addr(),
        CatalogIdentity {
            campaign: "validation-replay".into(),
            ensemble: "one-candidate".into(),
            replica: 0,
            signature_digest: digest,
        },
        ClientConfig::default(),
    )
    .unwrap();
    let ledger = client
        .record_ledger_event(1, ChargeKind::AcceptedQuench, 7, 7)
        .unwrap();
    assert_eq!(ledger.snapshot.aggregate_charged, 7);
    assert_eq!(fresh_evaluations.load(Ordering::SeqCst), 0);

    let candidate = CatalogCandidate {
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
    };
    let accepted = client.offer_candidate(2, candidate.clone()).unwrap();
    assert!(!accepted.duplicate);
    let mutation = accepted
        .catalog
        .as_ref()
        .expect("scientific catalog admission");
    assert_eq!(mutation.kind, CatalogMutationKind::Added);
    assert!(mutation.new_basin);
    assert_eq!(mutation.basin_visits, 1);
    assert_eq!(accepted.snapshot.census_visits, 1);
    assert_eq!(accepted.snapshot.active_entries, 1);
    assert_eq!(accepted.snapshot.aggregate_charged, 7);
    assert_eq!(accepted.snapshot.aggregate_budget, 32);
    assert_eq!(fresh_evaluations.load(Ordering::SeqCst), 1);

    let replayed = client.offer_candidate(2, candidate).unwrap();
    let mut expected_replay = accepted.clone();
    expected_replay.duplicate = true;
    assert_eq!(replayed, expected_replay);
    assert_eq!(client.snapshot(3).unwrap(), accepted.snapshot);
    assert_eq!(
        fresh_evaluations.load(Ordering::SeqCst),
        1,
        "an exact accepted request replay must not repeat the fresh engine evaluation",
    );
}
