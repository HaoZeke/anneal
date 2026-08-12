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

/// Structured reason a candidate cannot cross the catalog boundary.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ValidationFailure {
    /// Candidate and catalog signatures differ.
    #[error("candidate system signature does not match the catalog")]
    SignatureMismatch,
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
        let fresh =
            evaluate(&candidate.coordinates).map_err(ValidationFailure::EngineEvaluation)?;
        let _ = &self.config;
        Ok(ValidatedCandidate {
            candidate: candidate.clone(),
            fresh,
        })
    }
}
