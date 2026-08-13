//! Validation boundary for candidate records entering or leaving a catalog.

use super::SystemSignature;

/// Outcome reported by the quench that produced a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuenchStatus {
    /// The quench met its convergence contract.
    Converged,
    /// The quench stopped without satisfying its convergence contract.
    Unconverged,
}

/// Immutable candidate offered to a descriptor-basin catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRecord {
    /// Complete system and descriptor identity.
    pub signature: SystemSignature,
    /// Replica that produced the record within its ensemble.
    pub producer_replica: u32,
    /// Cartesian coordinates in signature order.
    pub coordinates: Vec<f64>,
    /// Row-major cell recorded with the candidate.
    pub cell: Option<[f64; 9]>,
    /// Quenched energy reported by the producer.
    pub energy: f64,
    /// Forces reported by the producer.
    pub forces: Vec<f64>,
    /// Euclidean gradient norm reported by the producer.
    pub gradient_norm: f64,
    /// Descriptor vector under the declared schema.
    pub descriptor: Vec<f64>,
    /// Descriptor schema version used for this vector.
    pub descriptor_schema_version: u32,
    /// Producer-side quench outcome.
    pub quench_status: QuenchStatus,
    /// Local charged-work counter at production.
    pub charged_work: u64,
    /// Monotone producer event sequence.
    pub event_sequence: u64,
    /// Random-seed identity for provenance.
    pub seed: u64,
}

/// Fresh potential result evaluated at candidate coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct FreshEvaluation {
    /// Fresh energy.
    pub energy: f64,
    /// Fresh forces in coordinate order.
    pub forces: Vec<f64>,
}

/// Euclidean norm used by producer and receiver convergence contracts.
pub fn euclidean_gradient_norm(gradient: &[f64]) -> f64 {
    gradient
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

/// Numeric and geometric thresholds for candidate validation.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Reference coordinates for frozen and grouped atoms.
    pub reference_coordinates: Vec<f64>,
    /// Required descriptor-vector length.
    pub descriptor_dim: usize,
    /// Smallest allowed interatomic distance.
    pub min_separation: f64,
    /// Absolute tolerance for frozen, grouped, and cell coordinates.
    pub coordinate_tolerance: f64,
    /// Largest accepted producer and fresh gradient norm.
    pub max_gradient_norm: f64,
    /// Absolute term in the producer/fresh energy comparison.
    pub energy_abs_tolerance: f64,
    /// Relative term in the producer/fresh energy comparison.
    pub energy_rel_tolerance: f64,
}

/// Numeric field inspected by candidate validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericField {
    /// Candidate Cartesian coordinates.
    Coordinates,
    /// Candidate producer forces.
    Forces,
    /// Candidate descriptor vector.
    Descriptor,
    /// Candidate cell matrix.
    Cell,
    /// Candidate producer energy.
    Energy,
    /// Candidate producer gradient norm.
    GradientNorm,
    /// Receiving-side energy.
    FreshEnergy,
    /// Receiving-side forces.
    FreshForces,
}

/// Origin of gradient evidence checked by candidate validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GradientSource {
    /// Norm reported by the candidate producer.
    #[error("producer")]
    Producer,
    /// Norm computed from receiving-side fresh forces.
    #[error("fresh evaluation")]
    Fresh,
}

/// Structured reason a candidate cannot cross the catalog boundary.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ValidationFailure {
    /// Candidate and catalog signatures differ.
    #[error("candidate system signature does not match the catalog")]
    SignatureMismatch,
    /// Cartesian coordinate length differs from the signature.
    #[error("coordinate dimension is {actual}, expected {expected}")]
    CoordinateDimension {
        /// Dimension declared by the signature.
        expected: u64,
        /// Dimension carried by the candidate.
        actual: u64,
    },
    /// Producer force length differs from the signature.
    #[error("force dimension is {actual}, expected {expected}")]
    ForceDimension {
        /// Dimension declared by the signature.
        expected: u64,
        /// Dimension carried by the candidate.
        actual: u64,
    },
    /// Fresh force length differs from the signature.
    #[error("fresh force dimension is {actual}, expected {expected}")]
    FreshForceDimension {
        /// Dimension declared by the signature.
        expected: u64,
        /// Dimension returned by the fresh evaluation.
        actual: u64,
    },
    /// Descriptor length differs from the validator contract.
    #[error("descriptor dimension is {actual}, expected {expected}")]
    DescriptorDimension {
        /// Dimension required by the validator.
        expected: usize,
        /// Dimension carried by the candidate.
        actual: usize,
    },
    /// Candidate descriptor version differs from the signature.
    #[error("descriptor schema version is {actual}, expected {expected}")]
    DescriptorSchemaVersion {
        /// Version declared by the signature.
        expected: u32,
        /// Version carried by the candidate.
        actual: u32,
    },
    /// A candidate or receiving-side numeric field contains NaN or infinity.
    #[error("nonfinite value in {field:?} at {index:?}")]
    NonFinite {
        /// Field containing the invalid value.
        field: NumericField,
        /// Element index for arrays, or no index for a scalar.
        index: Option<usize>,
    },
    /// Two atoms are closer than the configured physical floor.
    #[error("atoms {first_atom} and {second_atom} violate the minimum separation")]
    MinimumSeparation {
        /// Lower atom index in the first violating pair.
        first_atom: usize,
        /// Higher atom index in the first violating pair.
        second_atom: usize,
    },
    /// A frozen atom differs from its reference coordinate.
    #[error("frozen atom {atom} differs on Cartesian axis {axis}")]
    FrozenCoordinate {
        /// Atom whose frozen coordinate differs.
        atom: usize,
        /// Cartesian axis whose frozen coordinate differs.
        axis: usize,
    },
    /// A rigid-group pair differs from its reference distance.
    #[error("rigid-group distance differs between atoms {first_atom} and {second_atom}")]
    RigidGroupDistance {
        /// Lower atom index in the first violating pair.
        first_atom: usize,
        /// Higher atom index in the first violating pair.
        second_atom: usize,
    },
    /// The candidate cell is absent, unexpected, or differs from the signature.
    #[error("candidate cell does not match the system signature")]
    CellMismatch,
    /// The producer quench did not satisfy its convergence contract.
    #[error("candidate producer quench is unconverged")]
    UnconvergedQuench,
    /// Producer or receiving-side gradient evidence exceeds the threshold.
    #[error("{source:?} gradient exceeds the validation threshold")]
    GradientThreshold {
        /// Side of the validation boundary that supplied the gradient.
        source: GradientSource,
    },
    /// Producer and receiving-side energies disagree outside tolerance.
    #[error("producer and fresh energies disagree")]
    EnergyMismatch,
    /// The fresh potential evaluation failed.
    #[error("fresh engine evaluation failed: {0}")]
    EngineEvaluation(String),
}

/// Candidate paired with receiving-side potential evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedCandidate {
    /// Candidate record that passed validation.
    pub candidate: CandidateRecord,
    /// Fresh potential evidence obtained by the receiver.
    pub fresh: FreshEvaluation,
}

/// Receiving-side candidate validator for one system signature.
#[derive(Debug, Clone)]
pub struct CandidateValidator {
    expected: SystemSignature,
    config: ValidatorConfig,
}

impl CandidateValidator {
    /// Bind a validator to one system signature and threshold set.
    pub fn new(expected: SystemSignature, config: ValidatorConfig) -> Self {
        Self { expected, config }
    }

    /// Validate a candidate and attach a fresh engine evaluation.
    pub fn validate<F>(
        &self,
        candidate: &CandidateRecord,
        evaluate: F,
    ) -> Result<ValidatedCandidate, ValidationFailure>
    where
        F: FnOnce(&[f64]) -> Result<FreshEvaluation, String>,
    {
        if candidate.signature != self.expected {
            return Err(ValidationFailure::SignatureMismatch);
        }
        let coordinate_dim = u64::try_from(candidate.coordinates.len()).unwrap_or(u64::MAX);
        if coordinate_dim != self.expected.coordinate_dim {
            return Err(ValidationFailure::CoordinateDimension {
                expected: self.expected.coordinate_dim,
                actual: coordinate_dim,
            });
        }
        let force_dim = u64::try_from(candidate.forces.len()).unwrap_or(u64::MAX);
        if force_dim != self.expected.coordinate_dim {
            return Err(ValidationFailure::ForceDimension {
                expected: self.expected.coordinate_dim,
                actual: force_dim,
            });
        }
        if candidate.descriptor.len() != self.config.descriptor_dim {
            return Err(ValidationFailure::DescriptorDimension {
                expected: self.config.descriptor_dim,
                actual: candidate.descriptor.len(),
            });
        }
        if candidate.descriptor_schema_version != self.expected.descriptor.version {
            return Err(ValidationFailure::DescriptorSchemaVersion {
                expected: self.expected.descriptor.version,
                actual: candidate.descriptor_schema_version,
            });
        }
        require_finite_slice(&candidate.coordinates, NumericField::Coordinates)?;
        require_finite_slice(&candidate.forces, NumericField::Forces)?;
        require_finite_slice(&candidate.descriptor, NumericField::Descriptor)?;
        if let Some(cell) = candidate.cell.as_ref() {
            require_finite_slice(cell, NumericField::Cell)?;
        }
        require_finite_scalar(candidate.energy, NumericField::Energy)?;
        require_finite_scalar(candidate.gradient_norm, NumericField::GradientNorm)?;
        validate_cell(
            self.expected.cell.as_ref(),
            candidate.cell.as_ref(),
            self.config.coordinate_tolerance,
        )?;
        validate_minimum_separation(&candidate.coordinates, self.config.min_separation)?;
        validate_frozen_coordinates(
            &candidate.coordinates,
            &self.config.reference_coordinates,
            &self.expected.frozen_mask,
            self.config.coordinate_tolerance,
        )?;
        validate_rigid_groups(
            &candidate.coordinates,
            &self.config.reference_coordinates,
            &self.expected.group_labels,
            self.config.coordinate_tolerance,
        )?;
        if candidate.quench_status != QuenchStatus::Converged {
            return Err(ValidationFailure::UnconvergedQuench);
        }
        if candidate.gradient_norm > self.config.max_gradient_norm {
            return Err(ValidationFailure::GradientThreshold {
                source: GradientSource::Producer,
            });
        }
        let fresh =
            evaluate(&candidate.coordinates).map_err(ValidationFailure::EngineEvaluation)?;
        let fresh_force_dim = u64::try_from(fresh.forces.len()).unwrap_or(u64::MAX);
        if fresh_force_dim != self.expected.coordinate_dim {
            return Err(ValidationFailure::FreshForceDimension {
                expected: self.expected.coordinate_dim,
                actual: fresh_force_dim,
            });
        }
        require_finite_scalar(fresh.energy, NumericField::FreshEnergy)?;
        require_finite_slice(&fresh.forces, NumericField::FreshForces)?;
        let fresh_gradient_norm = euclidean_gradient_norm(&fresh.forces);
        if fresh_gradient_norm > self.config.max_gradient_norm {
            return Err(ValidationFailure::GradientThreshold {
                source: GradientSource::Fresh,
            });
        }
        let energy_tolerance = self.config.energy_abs_tolerance
            + self.config.energy_rel_tolerance * candidate.energy.abs().max(fresh.energy.abs());
        if (candidate.energy - fresh.energy).abs() > energy_tolerance {
            return Err(ValidationFailure::EnergyMismatch);
        }
        Ok(ValidatedCandidate {
            candidate: candidate.clone(),
            fresh,
        })
    }
}

fn validate_cell(
    expected: Option<&[f64; 9]>,
    actual: Option<&[f64; 9]>,
    tolerance: f64,
) -> Result<(), ValidationFailure> {
    match (expected, actual) {
        (None, None) => Ok(()),
        (Some(expected), Some(actual))
            if expected
                .iter()
                .zip(actual)
                .all(|(expected, actual)| (expected - actual).abs() <= tolerance) =>
        {
            Ok(())
        }
        _ => Err(ValidationFailure::CellMismatch),
    }
}

fn validate_minimum_separation(coordinates: &[f64], minimum: f64) -> Result<(), ValidationFailure> {
    let minimum_squared = minimum * minimum;
    for first_atom in 0..coordinates.len() / 3 {
        for second_atom in first_atom + 1..coordinates.len() / 3 {
            if squared_distance(coordinates, first_atom, second_atom) < minimum_squared {
                return Err(ValidationFailure::MinimumSeparation {
                    first_atom,
                    second_atom,
                });
            }
        }
    }
    Ok(())
}

fn validate_frozen_coordinates(
    coordinates: &[f64],
    reference: &[f64],
    frozen_mask: &[bool],
    tolerance: f64,
) -> Result<(), ValidationFailure> {
    for (atom, &frozen) in frozen_mask.iter().enumerate() {
        if !frozen {
            continue;
        }
        for axis in 0..3 {
            let index = 3 * atom + axis;
            if (coordinates[index] - reference[index]).abs() > tolerance {
                return Err(ValidationFailure::FrozenCoordinate { atom, axis });
            }
        }
    }
    Ok(())
}

fn validate_rigid_groups(
    coordinates: &[f64],
    reference: &[f64],
    group_labels: &[u32],
    tolerance: f64,
) -> Result<(), ValidationFailure> {
    for first_atom in 0..group_labels.len() {
        for second_atom in first_atom + 1..group_labels.len() {
            if group_labels[first_atom] != group_labels[second_atom] {
                continue;
            }
            let actual = squared_distance(coordinates, first_atom, second_atom).sqrt();
            let expected = squared_distance(reference, first_atom, second_atom).sqrt();
            if (actual - expected).abs() > tolerance {
                return Err(ValidationFailure::RigidGroupDistance {
                    first_atom,
                    second_atom,
                });
            }
        }
    }
    Ok(())
}

fn squared_distance(coordinates: &[f64], first_atom: usize, second_atom: usize) -> f64 {
    (0..3)
        .map(|axis| {
            let delta = coordinates[3 * first_atom + axis] - coordinates[3 * second_atom + axis];
            delta * delta
        })
        .sum()
}

fn require_finite_scalar(value: f64, field: NumericField) -> Result<(), ValidationFailure> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ValidationFailure::NonFinite { field, index: None })
    }
}

fn require_finite_slice(values: &[f64], field: NumericField) -> Result<(), ValidationFailure> {
    match values.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(ValidationFailure::NonFinite {
            field,
            index: Some(index),
        }),
        None => Ok(()),
    }
}
