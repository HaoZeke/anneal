//! Target-free Feynman--Kac reconfiguration for cooperative search chains.
//!
//! A search epoch supplies one validated representative per chain. The
//! selection potential combines within-epoch energy rank, descriptor novelty,
//! and census scarcity. Systematic resampling keeps the chain population
//! fixed, while a family cap prevents a single observed funnel from occupying
//! every slot. This is a population-management operator, not a Green-function
//! approximation and not an electronic-structure convergence claim.

use thiserror::Error;

/// Invalid evidence or reconfiguration parameters.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ReconfigurationError {
    /// An evidence rank is not finite or lies outside the unit interval.
    #[error("{field} rank must be finite and lie in [0, 1], got {value}")]
    InvalidRank {
        /// Name of the invalid evidence component.
        field: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// A scalar parameter is outside its admissible domain.
    #[error("invalid reconfiguration parameter {field}: {value}")]
    InvalidParameter {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// Reconfiguration requires at least one chain.
    #[error("reconfiguration requires a nonempty population")]
    EmptyPopulation,
}

/// Target-free evidence attached to one chain at a synchronization epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasinEvidence {
    energy_rank: f64,
    novelty_rank: f64,
    scarcity_rank: f64,
}

impl BasinEvidence {
    /// Construct evidence from ranks in `[0, 1]`.
    ///
    /// Lower energy rank is better. Higher novelty and scarcity ranks are
    /// better. Ranks keep selection pressure independent of energy units and
    /// cluster-size-dependent energy scale.
    pub fn new(
        energy_rank: f64,
        novelty_rank: f64,
        scarcity_rank: f64,
    ) -> Result<Self, ReconfigurationError> {
        validate_rank("energy", energy_rank)?;
        validate_rank("novelty", novelty_rank)?;
        validate_rank("scarcity", scarcity_rank)?;
        Ok(Self {
            energy_rank,
            novelty_rank,
            scarcity_rank,
        })
    }

    /// Within-epoch energy rank, where zero is best.
    pub fn energy_rank(self) -> f64 {
        self.energy_rank
    }

    /// Descriptor novelty rank, where one is most novel.
    pub fn novelty_rank(self) -> f64 {
        self.novelty_rank
    }

    /// Census scarcity rank, where one is least sampled.
    pub fn scarcity_rank(self) -> f64 {
        self.scarcity_rank
    }
}

fn validate_rank(field: &'static str, value: f64) -> Result<(), ReconfigurationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ReconfigurationError::InvalidRank { field, value })
    }
}

/// Coefficients of the bounded logarithmic selection potential.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionCoefficients {
    /// Pressure toward lower within-epoch energy rank.
    pub energy: f64,
    /// Pressure toward descriptor novelty.
    pub novelty: f64,
    /// Pressure toward census scarcity.
    pub scarcity: f64,
    /// Maximum log-weight difference retained before exponentiation.
    pub log_weight_clip: f64,
}

impl SelectionCoefficients {
    fn validate(self) -> Result<(), ReconfigurationError> {
        for (field, value) in [
            ("energy", self.energy),
            ("novelty", self.novelty),
            ("scarcity", self.scarcity),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ReconfigurationError::InvalidParameter { field, value });
            }
        }
        if !self.log_weight_clip.is_finite() || self.log_weight_clip <= 0.0 {
            return Err(ReconfigurationError::InvalidParameter {
                field: "log_weight_clip",
                value: self.log_weight_clip,
            });
        }
        Ok(())
    }
}

/// Population and genealogy diagnostics at one synchronization epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenealogyDiagnostics {
    /// Kish effective sample size of the normalized selection weights.
    pub effective_sample_size: f64,
    /// Number of source chains represented among the offspring.
    pub unique_parents: usize,
    /// Largest number of offspring assigned to one source chain.
    pub max_family_size: usize,
    /// Population variance of source-chain offspring counts.
    pub offspring_variance: f64,
}

/// Replayable fixed-population assignment for one synchronization epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct ReconfigurationPlan {
    weights: Vec<f64>,
    parents: Vec<usize>,
    diagnostics: GenealogyDiagnostics,
}

impl ReconfigurationPlan {
    /// Normalized selection weights in source-chain order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Parent source index for every destination chain.
    pub fn parents(&self) -> &[usize] {
        &self.parents
    }

    /// Selection and realized-genealogy diagnostics.
    pub fn diagnostics(&self) -> GenealogyDiagnostics {
        self.diagnostics
    }
}

/// Fractional ascending ranks with average ranks for ties.
///
/// The minimum has rank zero and the maximum has rank one when at least two
/// distinct positions are present. Positive affine changes of units preserve
/// the result.
pub fn ascending_fractional_ranks(values: &[f64]) -> Result<Vec<f64>, ReconfigurationError> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    for &value in values {
        if !value.is_finite() {
            return Err(ReconfigurationError::InvalidParameter {
                field: "rank_input",
                value,
            });
        }
    }
    if values.len() == 1 {
        return Ok(vec![0.0]);
    }

    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let scale = (values.len() - 1) as f64;
    let mut ranks = vec![0.0; values.len()];
    let mut begin = 0;
    while begin < order.len() {
        let mut end = begin + 1;
        while end < order.len() && values[order[end]] == values[order[begin]] {
            end += 1;
        }
        let average_position = 0.5 * (begin + end - 1) as f64;
        let rank = average_position / scale;
        for &index in &order[begin..end] {
            ranks[index] = rank;
        }
        begin = end;
    }
    Ok(ranks)
}

/// Build a fixed-size, family-capped reconfiguration plan.
///
/// `systematic_offset` lies in `[0, 1)` and is stored by the caller as part of
/// the coordinator event. Replaying the same snapshot and offset produces the
/// same parent assignment.
pub fn reconfiguration_plan(
    evidence: &[BasinEvidence],
    coefficients: SelectionCoefficients,
    systematic_offset: f64,
    max_offspring: usize,
) -> Result<ReconfigurationPlan, ReconfigurationError> {
    if evidence.is_empty() {
        return Err(ReconfigurationError::EmptyPopulation);
    }
    coefficients.validate()?;
    if !systematic_offset.is_finite() || !(0.0..1.0).contains(&systematic_offset) {
        return Err(ReconfigurationError::InvalidParameter {
            field: "systematic_offset",
            value: systematic_offset,
        });
    }
    if max_offspring == 0 {
        return Err(ReconfigurationError::InvalidParameter {
            field: "max_offspring",
            value: 0.0,
        });
    }

    let mut log_weights: Vec<f64> = evidence
        .iter()
        .map(|item| {
            -coefficients.energy * item.energy_rank
                + coefficients.novelty * item.novelty_rank
                + coefficients.scarcity * item.scarcity_rank
        })
        .collect();
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    for value in &mut log_weights {
        *value = (*value - maximum).max(-coefficients.log_weight_clip);
    }
    let normalizer: f64 = log_weights.iter().map(|value| value.exp()).sum();
    let weights: Vec<f64> = log_weights
        .iter()
        .map(|value| value.exp() / normalizer)
        .collect();

    let mut parents = systematic_parents(&weights, systematic_offset);
    cap_families(&mut parents, &weights, max_offspring);
    let diagnostics = genealogy_diagnostics(&weights, &parents);
    Ok(ReconfigurationPlan {
        weights,
        parents,
        diagnostics,
    })
}

fn systematic_parents(weights: &[f64], offset: f64) -> Vec<usize> {
    let population = weights.len();
    let mut parents = Vec::with_capacity(population);
    let mut parent = 0;
    let mut cumulative = weights[0];
    for destination in 0..population {
        let threshold = (offset + destination as f64) / population as f64;
        while parent + 1 < population && threshold >= cumulative {
            parent += 1;
            cumulative += weights[parent];
        }
        parents.push(parent);
    }
    parents
}

fn cap_families(parents: &mut [usize], weights: &[f64], max_offspring: usize) {
    let mut counts = vec![0usize; weights.len()];
    for &parent in parents.iter() {
        counts[parent] += 1;
    }
    while let Some(donor) = counts.iter().position(|&count| count > max_offspring) {
        let receiver = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count < max_offspring)
            .max_by(|(left, left_count), (right, right_count)| {
                let left_deficit = weights[*left] * parents.len() as f64 - **left_count as f64;
                let right_deficit = weights[*right] * parents.len() as f64 - **right_count as f64;
                left_deficit
                    .total_cmp(&right_deficit)
                    .then_with(|| right.cmp(left))
            })
            .map(|(index, _)| index)
            .expect("a positive family cap has enough total capacity");
        let slot = parents
            .iter()
            .rposition(|&parent| parent == donor)
            .expect("the donor family has an offspring slot");
        parents[slot] = receiver;
        counts[donor] -= 1;
        counts[receiver] += 1;
    }
}

fn genealogy_diagnostics(weights: &[f64], parents: &[usize]) -> GenealogyDiagnostics {
    let effective_sample_size = 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>();
    let mut counts = vec![0usize; weights.len()];
    for &parent in parents {
        counts[parent] += 1;
    }
    let unique_parents = counts.iter().filter(|&&count| count > 0).count();
    let max_family_size = counts.iter().copied().max().unwrap_or(0);
    let mean = parents.len() as f64 / counts.len() as f64;
    let offspring_variance = counts
        .iter()
        .map(|&count| {
            let residual = count as f64 - mean;
            residual * residual
        })
        .sum::<f64>()
        / counts.len() as f64;
    GenealogyDiagnostics {
        effective_sample_size,
        unique_parents,
        max_family_size,
        offspring_variance,
    }
}
