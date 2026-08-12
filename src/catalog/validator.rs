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
#[derive(Debug, Clone)]
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
    /// The fresh potential evaluation failed.
    #[error("fresh engine evaluation failed: {0}")]
    EngineEvaluation(String),
}

/// Candidate paired with receiving-side potential evidence.
#[derive(Debug, Clone)]
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
        Ok(ValidatedCandidate {
            candidate: candidate.clone(),
            fresh,
        })
    }
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
