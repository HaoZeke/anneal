use anneal_core::catalog::{
    CandidateRecord, CandidateValidator, DescriptorSignature, EngineSignature, FreshEvaluation,
    GradientSource, NumericField, QuenchStatus, SystemSignature, ValidationFailure,
    ValidatorConfig, euclidean_gradient_norm,
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

#[test]
fn nonfinite_candidate_arrays_are_rejected_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.coordinates[2] = f64::NAN;
    assert_rejected_before_fresh(
        record,
        ValidationFailure::NonFinite {
            field: NumericField::Coordinates,
            index: Some(2),
        },
    );

    let mut record = candidate(signature());
    record.forces[4] = f64::INFINITY;
    assert_rejected_before_fresh(
        record,
        ValidationFailure::NonFinite {
            field: NumericField::Forces,
            index: Some(4),
        },
    );

    let mut record = candidate(signature());
    record.descriptor[1] = f64::NEG_INFINITY;
    assert_rejected_before_fresh(
        record,
        ValidationFailure::NonFinite {
            field: NumericField::Descriptor,
            index: Some(1),
        },
    );

    let mut record = candidate(signature());
    record.cell = Some([10.0, 0.0, 0.0, f64::NAN, 10.0, 0.0, 0.0, 0.0, 10.0]);
    assert_rejected_before_fresh(
        record,
        ValidationFailure::NonFinite {
            field: NumericField::Cell,
            index: Some(3),
        },
    );
}

#[test]
fn nonfinite_candidate_scalars_are_rejected_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.energy = f64::NAN;
    assert_rejected_before_fresh(
        record,
        ValidationFailure::NonFinite {
            field: NumericField::Energy,
            index: None,
        },
    );

    let mut record = candidate(signature());
    record.gradient_norm = f64::INFINITY;
    assert_rejected_before_fresh(
        record,
        ValidationFailure::NonFinite {
            field: NumericField::GradientNorm,
            index: None,
        },
    );
}

#[test]
fn nonfinite_fresh_evaluation_is_rejected() {
    let record = candidate(signature());
    let result = validator(signature()).validate(&record, |_| {
        Ok(FreshEvaluation {
            energy: f64::NAN,
            forces: vec![0.0; 6],
        })
    });
    assert_eq!(
        result.unwrap_err(),
        ValidationFailure::NonFinite {
            field: NumericField::FreshEnergy,
            index: None,
        }
    );

    let result = validator(signature()).validate(&record, |_| {
        let mut forces = vec![0.0; 6];
        forces[3] = f64::INFINITY;
        Ok(FreshEvaluation {
            energy: -1.0,
            forces,
        })
    });
    assert_eq!(
        result.unwrap_err(),
        ValidationFailure::NonFinite {
            field: NumericField::FreshForces,
            index: Some(3),
        }
    );
}

#[test]
fn overlapping_atoms_are_rejected_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.coordinates[3] = 0.4;

    assert_rejected_before_fresh(
        record,
        ValidationFailure::MinimumSeparation {
            first_atom: 0,
            second_atom: 1,
        },
    );
}

#[test]
fn frozen_coordinates_are_rejected_before_engine_evaluation() {
    let mut expected = signature();
    expected.frozen_mask[0] = true;
    let mut record = candidate(expected.clone());
    record.coordinates[1] = 1e-5;
    let called = Cell::new(false);

    let result = validator(expected).validate(&record, |_| {
        called.set(true);
        Ok(FreshEvaluation {
            energy: -1.0,
            forces: vec![0.0; 6],
        })
    });

    assert_eq!(
        result.unwrap_err(),
        ValidationFailure::FrozenCoordinate { atom: 0, axis: 1 }
    );
    assert!(!called.get());
}

#[test]
fn rigid_group_distances_are_rejected_before_engine_evaluation() {
    let mut expected = signature();
    expected.group_labels = vec![4, 4];
    expected.group_schema = "rigid-groups-v1".to_owned();
    let mut record = candidate(expected.clone());
    record.coordinates[3] = 1.3;
    let called = Cell::new(false);

    let result = validator(expected).validate(&record, |_| {
        called.set(true);
        Ok(FreshEvaluation {
            energy: -1.0,
            forces: vec![0.0; 6],
        })
    });

    assert_eq!(
        result.unwrap_err(),
        ValidationFailure::RigidGroupDistance {
            first_atom: 0,
            second_atom: 1,
        }
    );
    assert!(!called.get());
}

#[test]
fn cell_presence_and_values_are_rejected_before_engine_evaluation() {
    let mut expected = signature();
    expected.cell = Some([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]);
    expected.periodic = [true, true, true];

    let record = candidate(expected.clone());
    let called = Cell::new(false);
    let result = validator(expected.clone()).validate(&record, |_| {
        called.set(true);
        Ok(FreshEvaluation {
            energy: -1.0,
            forces: vec![0.0; 6],
        })
    });
    assert_eq!(result.unwrap_err(), ValidationFailure::CellMismatch);
    assert!(!called.get());

    let mut record = candidate(expected.clone());
    record.cell = expected.cell;
    record.cell.as_mut().unwrap()[8] = 10.000_001;
    let called = Cell::new(false);
    let result = validator(expected).validate(&record, |_| {
        called.set(true);
        Ok(FreshEvaluation {
            energy: -1.0,
            forces: vec![0.0; 6],
        })
    });
    assert_eq!(result.unwrap_err(), ValidationFailure::CellMismatch);
    assert!(!called.get());
}

#[test]
fn unconverged_quench_is_rejected_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.quench_status = QuenchStatus::Unconverged;

    assert_rejected_before_fresh(record, ValidationFailure::UnconvergedQuench);
}

#[test]
fn producer_gradient_above_threshold_is_rejected_before_engine_evaluation() {
    let mut record = candidate(signature());
    record.gradient_norm = 1e-5;

    assert_rejected_before_fresh(
        record,
        ValidationFailure::GradientThreshold {
            source: GradientSource::Producer,
        },
    );
}

#[test]
fn fresh_gradient_above_threshold_is_rejected() {
    let result = validator(signature()).validate(&candidate(signature()), |_| {
        let mut forces = vec![0.0; 6];
        forces[0] = 1e-5;
        Ok(FreshEvaluation {
            energy: -1.0,
            forces,
        })
    });

    assert_eq!(
        result.unwrap_err(),
        ValidationFailure::GradientThreshold {
            source: GradientSource::Fresh,
        }
    );
}

#[test]
fn convergence_uses_the_full_euclidean_gradient_norm() {
    let gradient = [0.9e-5, 0.9e-5];

    assert!(gradient.iter().all(|component| component.abs() < 1e-5));
    assert!(euclidean_gradient_norm(&gradient) > 1e-5);
}

#[test]
fn producer_and_fresh_energies_must_agree() {
    let result = validator(signature()).validate(&candidate(signature()), |_| {
        Ok(FreshEvaluation {
            energy: -0.9,
            forces: vec![0.0; 6],
        })
    });

    assert_eq!(result.unwrap_err(), ValidationFailure::EnergyMismatch);
}

#[test]
fn engine_errors_are_classified_without_losing_the_message() {
    let result = validator(signature()).validate(&candidate(signature()), |_| {
        Err("potential unavailable".to_owned())
    });

    assert_eq!(
        result.unwrap_err(),
        ValidationFailure::EngineEvaluation("potential unavailable".to_owned())
    );
}

#[test]
fn declared_numeric_boundaries_are_accepted() {
    let expected = signature();
    let mut record = candidate(expected.clone());
    record.gradient_norm = 1.0;
    let validator = CandidateValidator::new(
        expected,
        ValidatorConfig {
            reference_coordinates: vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
            descriptor_dim: 2,
            min_separation: 1.2,
            coordinate_tolerance: 1e-10,
            max_gradient_norm: 1.0,
            energy_abs_tolerance: 0.25,
            energy_rel_tolerance: 0.0,
        },
    );

    let accepted = validator
        .validate(&record, |_| {
            let mut forces = vec![0.0; 6];
            forces[0] = 1.0;
            Ok(FreshEvaluation {
                energy: -0.75,
                forces,
            })
        })
        .unwrap();

    assert_eq!(accepted.candidate.event_sequence, 7);
    assert_eq!(accepted.fresh.energy, -0.75);
}
