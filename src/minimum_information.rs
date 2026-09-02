//! Same-PES allocation by information about the lowest reachable energy.
//!
//! A search mechanism maps a source structure to a terminal quenched energy.
//! Separate Gaussian processes model the energy change produced by basin
//! escape and minimum-mode riding. Their only shared random variable is the
//! lowest terminal energy on the current finite action set. GIBBON supplies a
//! lower bound on the mutual information with that minimum; division by the
//! action's charged PES cost gives the common allocation currency.
//!
//! Stationary-point counts, graph edges, transition rates, committors, and
//! network-completeness estimates are deliberately absent. They remain valid
//! output data for a downstream landscape database, but do not define a
//! global-minimum search reward.

use ndarray::ArrayView1;

use crate::funnel_bo::{FunnelCompression, FunnelModel};

const BASIN_DRAW_SALT: u64 = 0x6a09_e667_f3bc_c909;
const RIDE_DRAW_SALT: u64 = 0xbb67_ae85_84ca_a73b;
const DEFAULT_MAXIMUM_MODEL_RANK: usize = 64;

/// Proposal operator whose terminal quenched energy is modelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SearchMechanism {
    /// Perturb a minimum and quench the resulting structure.
    BasinEscape,
    /// Follow a minimum mode and quench the downhill branch or branches.
    SaddleRide,
}

impl SearchMechanism {
    fn index(self) -> usize {
        match self {
            Self::BasinEscape => 0,
            Self::SaddleRide => 1,
        }
    }
}

/// One finite action whose result would update a terminal-energy model.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchActionCandidate {
    /// Proposal operator.
    pub mechanism: SearchMechanism,
    /// Operator-specific, same-system feature vector.
    pub feature: Vec<f64>,
    /// Exact energy of the source minimum.
    pub source_energy: f64,
    /// Expected PES calls charged by this action.
    pub expected_charged_evaluations: f64,
}

/// Minimum-value information attached to one input candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchActionScore {
    /// Candidate index in the input slice.
    pub candidate: usize,
    /// Proposal operator.
    pub mechanism: SearchMechanism,
    /// GIBBON lower bound on mutual information with the minimum energy.
    pub information: f64,
    /// Information divided by expected charged PES evaluations.
    pub information_per_charged_evaluation: f64,
}

/// Invalid action-model configuration or observation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MinimumInformationError {
    /// GP scales must be finite and positive.
    #[error("minimum-information GP scales must be finite and positive")]
    InvalidModelScale,
    /// Kernel approximation rank must be positive.
    #[error("minimum-information GP rank must be positive")]
    InvalidModelRank,
    /// Candidate data must be finite and cost positive.
    #[error("minimum-information action contains invalid numeric data")]
    InvalidAction,
    /// One operator always consumes a fixed feature dimension.
    #[error("{mechanism:?} feature dimension is {actual}, expected {expected}")]
    FeatureDimension {
        /// Operator whose feature shape changed.
        mechanism: SearchMechanism,
        /// Dimension fixed by its first observation.
        expected: usize,
        /// Supplied dimension.
        actual: usize,
    },
}

/// Independent operator models coupled only through the target minimum value.
#[derive(Debug, Clone)]
pub struct MinimumInformationSearch {
    models: [FunnelModel; 2],
    feature_dimensions: [Option<usize>; 2],
    observations: [u64; 2],
    maximum_model_rank: usize,
    compression: [FunnelCompression; 2],
    incumbent_terminal_energy: Option<f64>,
    version: u64,
}

impl MinimumInformationSearch {
    /// Construct two zero-change-prior GP models with identical scales.
    pub fn new(
        length_scale: f64,
        amplitude: f64,
        noise: f64,
    ) -> Result<Self, MinimumInformationError> {
        Self::new_with_maximum_model_rank(
            length_scale,
            amplitude,
            noise,
            DEFAULT_MAXIMUM_MODEL_RANK,
        )
    }

    /// Construct bounded zero-change-prior GP models with identical scales.
    pub fn new_with_maximum_model_rank(
        length_scale: f64,
        amplitude: f64,
        noise: f64,
        maximum_model_rank: usize,
    ) -> Result<Self, MinimumInformationError> {
        if [length_scale, amplitude, noise]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(MinimumInformationError::InvalidModelScale);
        }
        if maximum_model_rank == 0 {
            return Err(MinimumInformationError::InvalidModelRank);
        }
        let mut basin_model = FunnelModel::new_euclidean(length_scale, amplitude, noise);
        basin_model.set_prior_mean(0.0);
        let mut ride_model = FunnelModel::new_euclidean(length_scale, amplitude, noise);
        ride_model.set_prior_mean(0.0);
        Ok(Self {
            models: [basin_model, ride_model],
            feature_dimensions: [None, None],
            observations: [0, 0],
            maximum_model_rank,
            compression: [FunnelCompression::default(); 2],
            incumbent_terminal_energy: None,
            version: 0,
        })
    }

    /// Observe the terminal energy returned by one charged action.
    ///
    /// An unsuccessful action is represented by `terminal_energy ==
    /// source_energy`: the live minimum remains available and the observation
    /// is finite rather than censored away.
    pub fn observe(
        &mut self,
        mechanism: SearchMechanism,
        feature: &[f64],
        source_energy: f64,
        terminal_energy: f64,
    ) -> Result<(), MinimumInformationError> {
        validate_feature(feature)?;
        if !source_energy.is_finite() || !terminal_energy.is_finite() {
            return Err(MinimumInformationError::InvalidAction);
        }
        self.ensure_dimension(mechanism, feature.len())?;
        let index = mechanism.index();
        self.models[index].observe(ArrayView1::from(feature), terminal_energy - source_energy);
        self.observations[index] = self.observations[index].saturating_add(1);
        let compressed = self.models[index].compress(self.maximum_model_rank);
        self.compression[index] = FunnelCompression {
            input_count: usize::try_from(self.observations[index]).unwrap_or(usize::MAX),
            retained_rank: compressed.retained_rank,
            residual_fraction: self.compression[index]
                .residual_fraction
                .max(compressed.residual_fraction),
            rank_limited: self.compression[index].rank_limited || compressed.rank_limited,
        };
        self.incumbent_terminal_energy = Some(
            self.incumbent_terminal_energy
                .map_or(terminal_energy, |held| held.min(terminal_energy)),
        );
        self.version = self.version.wrapping_add(1);
        Ok(())
    }

    /// Score a finite action set by minimum-value information per PES call.
    pub fn score(
        &mut self,
        candidates: &[SearchActionCandidate],
        minimum_samples: usize,
    ) -> Result<Vec<SearchActionScore>, MinimumInformationError> {
        if candidates.is_empty() || minimum_samples == 0 {
            return Ok(Vec::new());
        }
        for candidate in candidates {
            validate_candidate(candidate)?;
            self.check_dimension(candidate.mechanism, candidate.feature.len())?;
        }

        let mut minima_by_mechanism = [Vec::new(), Vec::new()];
        for mechanism in [SearchMechanism::BasinEscape, SearchMechanism::SaddleRide] {
            let shifted = candidates
                .iter()
                .filter(|candidate| candidate.mechanism == mechanism)
                .map(|candidate| {
                    (
                        ArrayView1::from(candidate.feature.as_slice()),
                        candidate.source_energy,
                    )
                })
                .collect::<Vec<_>>();
            if !shifted.is_empty() {
                minima_by_mechanism[mechanism.index()] = self.models[mechanism.index()]
                    .sample_shifted_joint_minima(
                        &shifted,
                        minimum_samples,
                        self.incumbent_terminal_energy,
                        match mechanism {
                            SearchMechanism::BasinEscape => BASIN_DRAW_SALT,
                            SearchMechanism::SaddleRide => RIDE_DRAW_SALT,
                        },
                    );
            }
        }
        let global_minima = (0..minimum_samples)
            .filter_map(|sample| {
                minima_by_mechanism
                    .iter()
                    .filter_map(|draws| draws.get(sample).copied())
                    .min_by(f64::total_cmp)
            })
            .collect::<Vec<_>>();

        Ok(candidates
            .iter()
            .enumerate()
            .map(|(candidate_index, candidate)| {
                let information = self.models[candidate.mechanism.index()]
                    .gibbon_information_with_offset(
                        ArrayView1::from(candidate.feature.as_slice()),
                        candidate.source_energy,
                        &global_minima,
                    );
                SearchActionScore {
                    candidate: candidate_index,
                    mechanism: candidate.mechanism,
                    information,
                    information_per_charged_evaluation: information
                        / candidate.expected_charged_evaluations,
                }
            })
            .collect())
    }

    /// Number of terminal outcomes learned for one operator.
    pub fn observations(&self, mechanism: SearchMechanism) -> u64 {
        self.observations[mechanism.index()]
    }

    /// Retained rank and cumulative covariance-loss evidence for one operator.
    pub fn compression(&self, mechanism: SearchMechanism) -> FunnelCompression {
        self.compression[mechanism.index()]
    }

    /// Lowest terminal energy seen through either operator.
    pub fn incumbent_terminal_energy(&self) -> Option<f64> {
        self.incumbent_terminal_energy
    }

    /// Monotonic evidence version used as a role-assignment epoch.
    pub fn version(&self) -> u64 {
        self.version
    }

    fn ensure_dimension(
        &mut self,
        mechanism: SearchMechanism,
        actual: usize,
    ) -> Result<(), MinimumInformationError> {
        self.check_dimension(mechanism, actual)?;
        self.feature_dimensions[mechanism.index()].get_or_insert(actual);
        Ok(())
    }

    fn check_dimension(
        &self,
        mechanism: SearchMechanism,
        actual: usize,
    ) -> Result<(), MinimumInformationError> {
        if let Some(expected) = self.feature_dimensions[mechanism.index()]
            && expected != actual
        {
            return Err(MinimumInformationError::FeatureDimension {
                mechanism,
                expected,
                actual,
            });
        }
        Ok(())
    }
}

fn validate_feature(feature: &[f64]) -> Result<(), MinimumInformationError> {
    if feature.is_empty() || feature.iter().any(|value| !value.is_finite()) {
        Err(MinimumInformationError::InvalidAction)
    } else {
        Ok(())
    }
}

fn validate_candidate(candidate: &SearchActionCandidate) -> Result<(), MinimumInformationError> {
    validate_feature(&candidate.feature)?;
    if !candidate.source_energy.is_finite()
        || !candidate.expected_charged_evaluations.is_finite()
        || candidate.expected_charged_evaluations <= 0.0
    {
        return Err(MinimumInformationError::InvalidAction);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_truncation_at_the_mean_has_the_half_normal_variance() {
        let variance = lower_truncated_variance(0.0, 1.0, 0.0);

        assert!((variance - (1.0 - 2.0 / std::f64::consts::PI)).abs() < 1e-10);
    }

    #[test]
    fn a_sampled_optimum_location_reduces_correlated_query_variance() {
        let (_, correlated_variance) = condition_on_noiseless_pair(0.2, 1.0, -0.5, 2.0, 0.8, -1.0);
        let (_, independent_variance) = condition_on_noiseless_pair(0.2, 1.0, -0.5, 2.0, 0.0, -1.0);

        assert!((correlated_variance - 0.68).abs() < 1e-12);
        assert!((independent_variance - 1.0).abs() < 1e-12);
    }
}
