use anneal_core::catalog::{
    AdmissionOutcome, AdmissionRejection, BasinCatalog, BasinId, CandidateRecord,
    DescriptorSignature, EngineSignature, FreshEvaluation, QuenchStatus, SystemSignature,
    ValidatedCandidate,
};
use std::collections::BTreeMap;

fn signature() -> SystemSignature {
    SystemSignature {
        atomic_numbers: vec![6],
        coordinate_dim: 3,
        group_labels: vec![0],
        group_schema: "independent-atoms-v1".to_owned(),
        frozen_mask: vec![false],
        cell: None,
        periodic: [false, false, false],
        length_scale: 1.0,
        energy_scale: 1.0,
        engine: EngineSignature {
            kind: "lj".to_owned(),
            config_digest: [0x11; 32],
            external_inputs: BTreeMap::new(),
        },
        descriptor: DescriptorSignature {
            schema: "soap".to_owned(),
            version: 1,
            hyperparameters: BTreeMap::new(),
            species_channels: vec![6],
        },
        validation_schema_version: 1,
    }
}

fn candidate(descriptor: f64, energy: f64, producer: u32, sequence: u64) -> ValidatedCandidate {
    ValidatedCandidate {
        candidate: CandidateRecord {
            signature: signature(),
            producer_replica: producer,
            coordinates: vec![descriptor, 0.0, 0.0],
            cell: None,
            energy,
            forces: vec![0.0; 3],
            gradient_norm: 0.0,
            descriptor: vec![descriptor],
            descriptor_schema_version: 1,
            quench_status: QuenchStatus::Converged,
            charged_work: sequence * 10,
            event_sequence: sequence,
            seed: 100 + u64::from(producer),
        },
        fresh: FreshEvaluation {
            energy,
            forces: vec![0.0; 3],
        },
    }
}

fn catalog(capacity: usize) -> BasinCatalog {
    BasinCatalog::new(capacity, 0.2, 100).unwrap()
}

#[test]
fn same_census_basin_keeps_only_its_lowest_energy_provenance() {
    let basin = BasinId::from_raw(8);
    let mut catalog = catalog(3);
    assert_eq!(
        catalog.admit(basin, 1, candidate(0.0, -1.0, 2, 1)),
        AdmissionOutcome::Added { basin_id: basin }
    );

    assert_eq!(
        catalog.admit(basin, 2, candidate(0.1, -0.5, 3, 2)),
        AdmissionOutcome::Rejected {
            reason: AdmissionRejection::SameBasinNotLower,
        }
    );
    assert_eq!(catalog.entry(basin).unwrap().energy(), -1.0);
    assert_eq!(catalog.entry(basin).unwrap().event_sequence(), 1);

    assert_eq!(
        catalog.admit(basin, 3, candidate(0.1, -1.5, 3, 3)),
        AdmissionOutcome::ReplacedSameBasin { basin_id: basin }
    );
    let entry = catalog.entry(basin).unwrap();
    assert_eq!(entry.energy(), -1.5);
    assert_eq!(entry.producer_replica(), 3);
    assert_eq!(entry.event_sequence(), 3);
    assert_eq!(entry.census_visits_at_admission(), 3);
}

#[test]
fn a_lower_candidate_replaces_its_complete_conflict_set() {
    let mut catalog = catalog(3);
    let a = BasinId::from_raw(0);
    let b = BasinId::from_raw(1);
    let c = BasinId::from_raw(2);
    catalog.admit(a, 1, candidate(0.0, -1.0, 0, 1));
    catalog.admit(b, 1, candidate(2.0, -2.0, 1, 2));
    catalog.admit(c, 1, candidate(4.0, -0.8, 2, 3));
    assert_eq!(catalog.initial_threshold(), Some(4.0 / 3.0));

    let replacement = BasinId::from_raw(3);
    assert_eq!(
        catalog.admit(replacement, 1, candidate(3.0, -3.0, 3, 4)),
        AdmissionOutcome::ReplacedConflicts {
            basin_id: replacement,
            evicted: vec![b, c],
        }
    );
    assert_eq!(catalog.len(), 2);
    assert!(catalog.entry(a).is_some());
    assert!(catalog.entry(b).is_none());
    assert!(catalog.entry(c).is_none());
    assert_eq!(catalog.incumbent().unwrap().census_id(), replacement);
    assert_eq!(catalog.incumbent().unwrap().energy(), -3.0);
}

#[test]
fn conflict_rejection_leaves_every_active_entry_unchanged() {
    let mut catalog = catalog(3);
    let a = BasinId::from_raw(0);
    let b = BasinId::from_raw(1);
    let c = BasinId::from_raw(2);
    catalog.admit(a, 1, candidate(0.0, -3.0, 0, 1));
    catalog.admit(b, 1, candidate(2.0, -2.0, 1, 2));
    catalog.admit(c, 1, candidate(4.0, -1.0, 2, 3));
    let before = catalog.entries().to_vec();

    assert_eq!(
        catalog.admit(BasinId::from_raw(3), 1, candidate(3.0, -1.5, 3, 4)),
        AdmissionOutcome::Rejected {
            reason: AdmissionRejection::ConflictNotLower,
        }
    );
    assert_eq!(catalog.entries(), before);
}

#[test]
fn capacity_eviction_never_discards_the_global_incumbent_for_a_worse_candidate() {
    let mut catalog = catalog(3);
    let incumbent = BasinId::from_raw(0);
    let middle = BasinId::from_raw(1);
    let worst = BasinId::from_raw(2);
    catalog.admit(incumbent, 1, candidate(0.0, -3.0, 0, 1));
    catalog.admit(middle, 1, candidate(2.0, -1.0, 1, 2));
    catalog.admit(worst, 1, candidate(4.0, -0.5, 2, 3));

    let replacement = BasinId::from_raw(4);
    assert_eq!(
        catalog.admit(replacement, 1, candidate(8.0, -0.75, 3, 4)),
        AdmissionOutcome::ReplacedCapacity {
            basin_id: replacement,
            evicted: worst,
        }
    );
    assert!(catalog.entry(incumbent).is_some());
    assert_eq!(catalog.incumbent().unwrap().energy(), -3.0);

    let before = catalog.entries().to_vec();
    assert_eq!(
        catalog.admit(BasinId::from_raw(5), 1, candidate(12.0, -0.4, 4, 5)),
        AdmissionOutcome::Rejected {
            reason: AdmissionRejection::CapacityNotLower,
        }
    );
    assert_eq!(catalog.entries(), before);
}

#[test]
fn packing_threshold_is_initialized_safely_and_never_increases() {
    let mut catalog = catalog(3);
    catalog.admit(BasinId::from_raw(0), 1, candidate(0.0, -3.0, 0, 1));
    catalog.admit(BasinId::from_raw(1), 1, candidate(2.0, -2.0, 1, 2));
    catalog.admit(BasinId::from_raw(2), 1, candidate(4.0, -1.0, 2, 3));

    let initial = catalog.packing_threshold().unwrap();
    let middle = catalog.update_threshold(40).unwrap();
    let floor = catalog.update_threshold(80).unwrap();
    let stale_update = catalog.update_threshold(20).unwrap();

    assert_eq!(initial, 4.0 / 3.0);
    assert!(middle < initial);
    assert!(floor < middle);
    assert_eq!(floor, (0.4 * initial).max(0.2));
    assert_eq!(stale_update, floor);
    for left in catalog.entries() {
        for right in catalog.entries() {
            if left.census_id() < right.census_id() {
                let distance = (left.descriptor()[0] - right.descriptor()[0]).abs();
                assert!(distance >= catalog.packing_threshold().unwrap());
            }
        }
    }
}
