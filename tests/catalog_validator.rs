use anneal_core::catalog::{
    CandidateRecord, CandidateValidator, DescriptorSignature, EngineSignature, FreshEvaluation,
    QuenchStatus, SystemSignature, ValidationFailure, ValidatorConfig,
};
use std::cell::Cell;
use std::collections::BTreeMap;

fn signature() -> SystemSignature {
    SystemSignature {
        atomic_numbers: vec![6, 6],
        coordinate_dim: 6,
        group_labels: vec![0, 1],
        group_schema: "independent-atoms-v1".to_owned(),
        frozen_mask: vec![false, false],
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

fn candidate(signature: SystemSignature) -> CandidateRecord {
    CandidateRecord {
        signature,
        producer_replica: 2,
        coordinates: vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
        cell: None,
        energy: -1.0,
        forces: vec![0.0; 6],
        gradient_norm: 0.0,
        descriptor: vec![0.2, 0.8],
        descriptor_schema_version: 1,
        quench_status: QuenchStatus::Converged,
        charged_work: 19,
        event_sequence: 7,
        seed: 41,
    }
}

fn validator(expected: SystemSignature) -> CandidateValidator {
    CandidateValidator::new(
        expected,
        ValidatorConfig {
            reference_coordinates: vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
            descriptor_dim: 2,
            min_separation: 0.8,
            coordinate_tolerance: 1e-10,
            max_gradient_norm: 1e-6,
            energy_abs_tolerance: 1e-10,
            energy_rel_tolerance: 1e-10,
        },
    )
}

fn assert_rejected_before_fresh(record: CandidateRecord, expected: ValidationFailure) {
    let called = Cell::new(false);
    let result = validator(signature()).validate(&record, |_| {
        called.set(true);
        Ok(FreshEvaluation {
            energy: -1.0,
            forces: vec![0.0; 6],
        })
    });

    assert_eq!(result.unwrap_err(), expected);
    assert!(!called.get());
}

#[test]
fn signature_mismatch_is_rejected_before_engine_evaluation() {
    let expected = signature();
    let mut foreign = expected.clone();
    foreign.engine.config_digest = [0x22; 32];
    let record = candidate(foreign);
    let called = Cell::new(false);

    let result = validator(expected).validate(&record, |_| {
        called.set(true);
        Ok(FreshEvaluation {
            energy: -1.0,
            forces: vec![0.0; 6],
        })
    });

    assert_eq!(result.unwrap_err(), ValidationFailure::SignatureMismatch);
    assert!(!called.get());
}

#[test]
fn coordinate_dimension_is_checked_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.coordinates.pop();

    assert_rejected_before_fresh(
        record,
        ValidationFailure::CoordinateDimension {
            expected: 6,
            actual: 5,
        },
    );
}

#[test]
fn force_dimension_is_checked_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.forces.pop();

    assert_rejected_before_fresh(
        record,
        ValidationFailure::ForceDimension {
            expected: 6,
            actual: 5,
        },
    );
}

#[test]
fn descriptor_dimension_is_checked_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.descriptor.pop();

    assert_rejected_before_fresh(
        record,
        ValidationFailure::DescriptorDimension {
            expected: 2,
            actual: 1,
        },
    );
}

#[test]
fn descriptor_schema_version_is_checked_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.descriptor_schema_version = 2;

    assert_rejected_before_fresh(
        record,
        ValidationFailure::DescriptorSchemaVersion {
            expected: 1,
            actual: 2,
        },
    );
}

#[test]
fn fresh_force_dimension_is_rejected() {
    let result = validator(signature()).validate(&candidate(signature()), |_| {
        Ok(FreshEvaluation {
            energy: -1.0,
            forces: vec![0.0; 5],
        })
    });

    assert_eq!(
        result.unwrap_err(),
        ValidationFailure::FreshForceDimension {
            expected: 6,
            actual: 5,
        }
    );
}
