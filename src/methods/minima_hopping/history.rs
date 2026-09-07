//! Validated-minimum history for cooperating optimization chains.

use crate::descriptor_space::{DescriptorError, DescriptorVector};
use crate::methods::cluster_hopping::QuenchBoundary;
use crate::pes_exploration::{
    ExactStructureWitness, MinimumAdmission, PesNetwork, StructureContext,
};

/// Exact identity and accumulated visits returned to an escape controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoryObservation {
    /// Descriptor-ordered, exact-witness identity decision.
    pub minimum: MinimumAdmission,
    /// Visits to this minimum, including the submitted observation.
    pub visits: u64,
}

/// A history observation could not be admitted.
#[derive(Debug, thiserror::Error)]
pub enum MinimumHistoryError {
    /// The gradient certificate needs a finite positive tolerance.
    #[error("minimum history requires a finite positive gradient tolerance")]
    InvalidTolerance,
    /// The charged quench has no sufficient fresh minimum certificate.
    #[error("minimum history requires a validated converged quench")]
    RejectedMinimum,
    /// Incompatible descriptor metadata prevents identity retrieval.
    #[error(transparent)]
    Descriptor(#[from] DescriptorError),
    /// The observation count cannot be represented.
    #[error("minimum history visit counter overflow")]
    CounterOverflow,
}

/// Shared search memory without state adoption or an acceptance policy.
///
/// A private instance gives the isolated-chain comparison; serializing access
/// to one instance shares the same identity and visit rules across replicas.
/// Descriptors order the exact witness checks in [`PesNetwork`]. Descriptor
/// distance and packing-family membership never decide minimum identity.
pub struct MinimumHistory {
    network: PesNetwork,
    visits: Vec<u64>,
    total_visits: u64,
    gradient_tolerance: f64,
}

impl MinimumHistory {
    /// Empty history with the caller's minimum-certificate tolerance.
    pub fn new(gradient_tolerance: f64) -> Result<Self, MinimumHistoryError> {
        if !gradient_tolerance.is_finite() || gradient_tolerance <= 0.0 {
            return Err(MinimumHistoryError::InvalidTolerance);
        }
        Ok(Self {
            network: PesNetwork::new(),
            visits: Vec::new(),
            total_visits: 0,
            gradient_tolerance,
        })
    }

    /// Number of exact minimum identities in the shared history.
    pub fn minimum_count(&self) -> usize {
        self.network.minimum_count()
    }

    /// Number of validated observations, including replica initializations.
    pub fn total_visits(&self) -> u64 {
        self.total_visits
    }

    /// Admit one charged minimum certificate and return only search history.
    pub fn observe<W: ExactStructureWitness + ?Sized>(
        &mut self,
        quench: &QuenchBoundary,
        descriptor: DescriptorVector,
        context: StructureContext,
        witness: &W,
    ) -> Result<HistoryObservation, MinimumHistoryError> {
        let gradient = quench
            .gradient()
            .ok_or(MinimumHistoryError::RejectedMinimum)?;
        let max_gradient = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_gradient >= self.gradient_tolerance {
            return Err(MinimumHistoryError::RejectedMinimum);
        }
        let total_visits = self
            .total_visits
            .checked_add(1)
            .ok_or(MinimumHistoryError::CounterOverflow)?;
        let minimum = self.network.admit_minimum_with_context(
            quench.energy(),
            quench.state().to_owned(),
            max_gradient,
            descriptor,
            context,
            witness,
        )?;
        if minimum.is_new {
            self.visits.push(0);
        }
        // Every basin count is bounded by the checked total observation count.
        self.visits[minimum.id] += 1;
        self.total_visits = total_visits;
        Ok(HistoryObservation {
            minimum,
            visits: self.visits[minimum.id],
        })
    }
}
