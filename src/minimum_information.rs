//! Same-PES allocation by information about the lowest reachable minimum.
//!
//! A search mechanism maps a source structure to a terminal quenched energy.
//! Separate Gaussian processes model the energy change produced by basin
//! escape and minimum-mode riding. Their only shared random variable is the
//! identity-and-energy pair of the lowest terminal minimum on the current
//! finite action set. A moment-matched joint entropy search acquisition
//! measures information about that pair; division by the action's charged PES
//! cost gives the common allocation currency.
//!
//! Stationary-point counts, graph edges, transition rates, committors, and
//! network-completeness estimates are deliberately absent. They remain valid
//! output data for a downstream landscape database, but do not define a
//! global-minimum search reward.

use ndarray::{Array2, ArrayView1};

use crate::funnel_bo::{
    FunnelCompression, FunnelModel, inverse_mills_lower, positive_definite_log_determinant,
};

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

/// Joint minimum information attached to one input candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchActionScore {
    /// Candidate index in the input slice.
    pub candidate: usize,
    /// Proposal operator.
    pub mechanism: SearchMechanism,
    /// Moment-matched information about the minimum identity and energy.
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
    /// A batch family label is required for every candidate action.
    #[error("minimum-information batch has {families} families for {candidates} candidates")]
    BatchFamilyDimension {
        /// Number of candidate actions.
        candidates: usize,
        /// Number of supplied family labels.
        families: usize,
    },
}

/// Independent operator models coupled only through the target minimum pair.
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

#[derive(Debug)]
struct ActionPosterior {
    candidate_indices: Vec<usize>,
    means: Vec<f64>,
    covariance: Array2<f64>,
    draws: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy)]
struct SampledOptimum {
    mechanism: Option<SearchMechanism>,
    local_candidate: Option<usize>,
    energy: f64,
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

    /// Score a finite action set by joint minimum information per PES call.
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

        let mut posteriors: [Option<ActionPosterior>; 2] = [None, None];
        let mut local_candidates = vec![0usize; candidates.len()];
        for mechanism in [SearchMechanism::BasinEscape, SearchMechanism::SaddleRide] {
            let candidate_indices = candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| candidate.mechanism == mechanism)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let shifted = candidate_indices
                .iter()
                .map(|index| {
                    let candidate = &candidates[*index];
                    (
                        ArrayView1::from(candidate.feature.as_slice()),
                        candidate.source_energy,
                    )
                })
                .collect::<Vec<_>>();
            for (local, global) in candidate_indices.iter().copied().enumerate() {
                local_candidates[global] = local;
            }
            if shifted.is_empty() {
                continue;
            }
            let (means, covariance, draws) = self.models[mechanism.index()]
                .sample_shifted_joint_values(
                    &shifted,
                    minimum_samples,
                    match mechanism {
                        SearchMechanism::BasinEscape => BASIN_DRAW_SALT,
                        SearchMechanism::SaddleRide => RIDE_DRAW_SALT,
                    },
                );
            posteriors[mechanism.index()] = Some(ActionPosterior {
                candidate_indices,
                means,
                covariance,
                draws,
            });
        }
        let sampled_optima = (0..minimum_samples)
            .filter_map(|sample| {
                let mut optimum = self.incumbent_terminal_energy.map(|energy| SampledOptimum {
                    mechanism: None,
                    local_candidate: None,
                    energy,
                });
                for mechanism in [SearchMechanism::BasinEscape, SearchMechanism::SaddleRide] {
                    let Some(posterior) = &posteriors[mechanism.index()] else {
                        continue;
                    };
                    let Some(draw) = posterior.draws.get(sample) else {
                        continue;
                    };
                    for (local_candidate, energy) in draw.iter().copied().enumerate() {
                        if optimum.is_none_or(|held| energy.total_cmp(&held.energy).is_lt()) {
                            optimum = Some(SampledOptimum {
                                mechanism: Some(mechanism),
                                local_candidate: Some(local_candidate),
                                energy,
                            });
                        }
                    }
                }
                optimum
            })
            .collect::<Vec<_>>();

        Ok(candidates
            .iter()
            .enumerate()
            .map(|(candidate_index, candidate)| {
                let mechanism_index = candidate.mechanism.index();
                let posterior = posteriors[mechanism_index]
                    .as_ref()
                    .expect("each validated candidate belongs to one posterior block");
                let information = joint_optimum_information(
                    candidate.mechanism,
                    local_candidates[candidate_index],
                    posterior,
                    &sampled_optima,
                    self.models[mechanism_index].noise,
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

    /// Select a parallel action batch by conditional joint-optimum information.
    ///
    /// Singleton terms are JES information about the optimum identity and
    /// value. The predictive-observation log determinant discounts redundant
    /// actions without a diversity coefficient. Greedy increments are divided
    /// by each action's charged PES cost. `families` supplies the partition
    /// constrained by `max_family_size`; a candidate remains selectable until
    /// its family reaches that capacity.
    pub fn assign_batch(
        &mut self,
        candidates: &[SearchActionCandidate],
        families: &[usize],
        batch_size: usize,
        max_family_size: usize,
        minimum_samples: usize,
    ) -> Result<Vec<usize>, MinimumInformationError> {
        if candidates.len() != families.len() {
            return Err(MinimumInformationError::BatchFamilyDimension {
                candidates: candidates.len(),
                families: families.len(),
            });
        }
        if candidates.is_empty() || batch_size == 0 || max_family_size == 0 || minimum_samples == 0
        {
            return Ok(Vec::new());
        }
        let singleton_scores = self.score(candidates, minimum_samples)?;
        let mut selected = Vec::<usize>::with_capacity(batch_size);
        let mut family_sizes = std::collections::BTreeMap::<usize, usize>::new();
        let mut batch_information = 0.0;

        while selected.len() < batch_size {
            let mut best: Option<(f64, usize, f64)> = None;
            for candidate_index in 0..candidates.len() {
                let family = families[candidate_index];
                let family_size = family_sizes.get(&family).copied().unwrap_or(0);
                if family_size >= max_family_size {
                    continue;
                }
                let mut enlarged = selected.clone();
                enlarged.push(candidate_index);
                let Some(enlarged_information) =
                    self.batch_information(candidates, &singleton_scores, enlarged.as_slice())
                else {
                    continue;
                };
                let marginal_rate = (enlarged_information - batch_information)
                    / candidates[candidate_index].expected_charged_evaluations;
                if !marginal_rate.is_finite() {
                    continue;
                }
                let replace = best.as_ref().is_none_or(|(held_rate, held_index, _)| {
                    match marginal_rate.total_cmp(held_rate) {
                        std::cmp::Ordering::Greater => true,
                        std::cmp::Ordering::Equal => {
                            let held_family_size = family_sizes
                                .get(&families[*held_index])
                                .copied()
                                .unwrap_or(0);
                            family_size < held_family_size
                                || (family_size == held_family_size
                                    && candidate_index < *held_index)
                        }
                        std::cmp::Ordering::Less => false,
                    }
                });
                if replace {
                    best = Some((marginal_rate, candidate_index, enlarged_information));
                }
            }
            let Some((_, candidate_index, enlarged_information)) = best else {
                break;
            };
            selected.push(candidate_index);
            *family_sizes.entry(families[candidate_index]).or_default() += 1;
            batch_information = enlarged_information;
        }
        Ok(selected)
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

    fn batch_information(
        &mut self,
        candidates: &[SearchActionCandidate],
        singleton_scores: &[SearchActionScore],
        selected: &[usize],
    ) -> Option<f64> {
        if selected.is_empty() {
            return Some(0.0);
        }
        let singleton_information = selected
            .iter()
            .map(|index| singleton_scores[*index].information)
            .sum::<f64>();
        let mut correlation = Array2::<f64>::eye(selected.len());
        for row in 0..selected.len() {
            let left = &candidates[selected[row]];
            for column in 0..row {
                let right = &candidates[selected[column]];
                let value = if left.mechanism == right.mechanism {
                    self.models[left.mechanism.index()].predictive_observation_correlation(
                        ArrayView1::from(left.feature.as_slice()),
                        ArrayView1::from(right.feature.as_slice()),
                    )
                } else {
                    0.0
                };
                correlation[[row, column]] = value;
                correlation[[column, row]] = value;
            }
        }
        let log_determinant = positive_definite_log_determinant(&correlation)?;
        Some(singleton_information + 0.5 * log_determinant)
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

fn joint_optimum_information(
    mechanism: SearchMechanism,
    local_candidate: usize,
    posterior: &ActionPosterior,
    sampled_optima: &[SampledOptimum],
    observation_noise: f64,
) -> f64 {
    if sampled_optima.is_empty() {
        return 0.0;
    }
    debug_assert_eq!(posterior.candidate_indices.len(), posterior.means.len());
    let mean = posterior.means[local_candidate];
    let variance = posterior.covariance[[local_candidate, local_candidate]].max(0.0);
    let observation_variance = variance + observation_noise * observation_noise;
    if !mean.is_finite() || !observation_variance.is_finite() || observation_variance <= 0.0 {
        return 0.0;
    }

    let mut information = 0.0;
    let mut accepted_samples = 0usize;
    for optimum in sampled_optima {
        if !optimum.energy.is_finite() {
            continue;
        }
        let (conditioned_mean, conditioned_variance) = if optimum.mechanism == Some(mechanism) {
            let optimum_candidate = optimum
                .local_candidate
                .expect("a sampled action optimum has a candidate index");
            condition_on_noiseless_pair(
                mean,
                variance,
                posterior.means[optimum_candidate],
                posterior.covariance[[optimum_candidate, optimum_candidate]].max(0.0),
                posterior.covariance[[local_candidate, optimum_candidate]],
                optimum.energy,
            )
        } else {
            (mean, variance)
        };
        let truncated_variance =
            lower_truncated_variance(conditioned_mean, conditioned_variance, optimum.energy);
        let conditional_observation_variance =
            truncated_variance + observation_noise * observation_noise;
        if conditional_observation_variance.is_finite() && conditional_observation_variance > 0.0 {
            information += 0.5 * (observation_variance / conditional_observation_variance).ln();
            accepted_samples += 1;
        }
    }
    if accepted_samples == 0 {
        0.0
    } else {
        (information / accepted_samples as f64).max(0.0)
    }
}

fn condition_on_noiseless_pair(
    query_mean: f64,
    query_variance: f64,
    optimum_mean: f64,
    optimum_variance: f64,
    covariance: f64,
    sampled_optimum: f64,
) -> (f64, f64) {
    if !optimum_variance.is_finite() || optimum_variance <= f64::EPSILON {
        return (query_mean, query_variance.max(0.0));
    }
    let coefficient = covariance / optimum_variance;
    (
        query_mean + coefficient * (sampled_optimum - optimum_mean),
        (query_variance - covariance * coefficient).max(0.0),
    )
}

fn lower_truncated_variance(mean: f64, variance: f64, lower_bound: f64) -> f64 {
    if !mean.is_finite()
        || !variance.is_finite()
        || !lower_bound.is_finite()
        || variance <= f64::EPSILON
    {
        return variance.max(0.0);
    }
    let alpha = (lower_bound - mean) / variance.sqrt();
    let inverse_mills = if alpha.abs() <= f64::EPSILON {
        (2.0 / std::f64::consts::PI).sqrt()
    } else {
        inverse_mills_lower(-alpha)
    };
    let retained = (1.0 + alpha * inverse_mills - inverse_mills * inverse_mills).clamp(0.0, 1.0);
    variance * retained
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
