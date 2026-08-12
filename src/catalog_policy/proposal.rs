//! Target-free descriptor increments for catalog exploration and exploitation.

use crate::descriptor_space::pullback::{
    PullbackConfig, PullbackConstraints, PullbackError, PullbackResult, regularized_pullback,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::Rng;
use std::f64::consts::PI;

/// Invalid catalog geometry or proposal parameter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProposalError {
    /// At least one descriptor is required.
    #[error("catalog proposal requires at least one descriptor")]
    EmptyCatalog,
    /// Descriptor vectors must be nonempty and share one dimension.
    #[error("descriptor dimension is {actual}, expected {expected}")]
    DescriptorDimension {
        /// Required descriptor dimension.
        expected: usize,
        /// Supplied descriptor dimension.
        actual: usize,
    },
    /// Descriptor vectors must contain finite, nonzero values where normalized.
    #[error("catalog proposal contains an invalid descriptor value")]
    InvalidDescriptor,
    /// D27 requires a finite scale in `(0, 1]`.
    #[error("differential scale must lie in (0, 1]")]
    InvalidDifferentialScale,
    /// Attraction weight must be finite and nonnegative.
    #[error("attraction weight must be finite and nonnegative")]
    InvalidAttraction,
    /// Descriptor-increment bound must be finite and nonnegative.
    #[error("descriptor increment bound must be finite and nonnegative")]
    InvalidIncrementBound,
    /// Farthest-hole search requires at least one sampled candidate.
    #[error("farthest-hole sample count must be positive")]
    ZeroHoleSamples,
    /// D21 rejected or could not solve the Cartesian pullback.
    #[error("descriptor pullback failed: {0}")]
    Pullback(#[from] PullbackError),
}

/// One D27 increment and the independent catalog draws that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentialProposal {
    left_index: usize,
    right_index: usize,
    increment: Array1<f64>,
}

impl DifferentialProposal {
    /// Index of `p_b` in the admissible catalog distribution.
    pub fn left_index(&self) -> usize {
        self.left_index
    }

    /// Index of `p_c` in the admissible catalog distribution.
    pub fn right_index(&self) -> usize {
        self.right_index
    }

    /// Requested centered descriptor increment `F (p_b - p_c)`.
    pub fn increment(&self) -> &Array1<f64> {
        &self.increment
    }
}

/// Seeded farthest-hole target and its descriptor increment.
#[derive(Debug, Clone, PartialEq)]
pub struct HoleProposal {
    target: Array1<f64>,
    increment: Array1<f64>,
    nearest_catalog_distance: f64,
}

impl HoleProposal {
    /// Unit-sphere candidate maximizing sampled nearest-catalog distance.
    pub fn target(&self) -> &Array1<f64> {
        &self.target
    }

    /// Descriptor increment from the current point to the sampled hole.
    pub fn increment(&self) -> &Array1<f64> {
        &self.increment
    }

    /// Distance from the target to its nearest normalized catalog entry.
    pub fn nearest_catalog_distance(&self) -> f64 {
        self.nearest_catalog_distance
    }
}

/// Explicit attraction-plus-differential descriptor increment.
#[derive(Debug, Clone, PartialEq)]
pub struct CombinedProposal {
    increment: Array1<f64>,
    clipped: bool,
}

impl CombinedProposal {
    /// Bounded sum of attraction and differential terms.
    pub fn increment(&self) -> &Array1<f64> {
        &self.increment
    }

    /// Whether the explicit descriptor-space bound shortened the sum.
    pub fn clipped(&self) -> bool {
        self.clipped
    }
}

/// Draw `p_b` and `p_c` independently with replacement and return D27.
pub fn catalog_differential<R: Rng + ?Sized>(
    catalog: &[Array1<f64>],
    scale: f64,
    rng: &mut R,
) -> Result<DifferentialProposal, ProposalError> {
    let dimension = validate_catalog(catalog)?;
    validate_scale(scale)?;
    let left_index = rng.random_range(0..catalog.len());
    let right_index = rng.random_range(0..catalog.len());
    let increment = scale * (&catalog[left_index] - &catalog[right_index]);
    debug_assert_eq!(increment.len(), dimension);
    Ok(DifferentialProposal {
        left_index,
        right_index,
        increment,
    })
}

/// Select a seeded unit-sphere candidate farthest from the occupied cloud.
pub fn farthest_hole<R: Rng + ?Sized>(
    current: &Array1<f64>,
    catalog: &[Array1<f64>],
    samples: usize,
    rng: &mut R,
) -> Result<HoleProposal, ProposalError> {
    let dimension = validate_catalog(catalog)?;
    validate_descriptor(current, dimension)?;
    if samples == 0 {
        return Err(ProposalError::ZeroHoleSamples);
    }
    let normalized_catalog = catalog
        .iter()
        .map(unit_vector)
        .collect::<Result<Vec<_>, _>>()?;
    let normalized_current = unit_vector(current)?;
    let mut centroid = normalized_catalog
        .iter()
        .fold(Array1::zeros(dimension), |sum, point| sum + point);
    centroid = unit_vector(&centroid).unwrap_or_else(|_| normalized_current.clone());
    let mut target = normalized_current.clone();
    let mut best_score = nearest_distance(&target, &normalized_catalog);
    for _ in 0..samples {
        let mut candidate = gaussian_unit_vector(dimension, rng);
        if candidate.dot(&centroid) > 0.0 {
            candidate *= -1.0;
        }
        let score = nearest_distance(&candidate, &normalized_catalog);
        if score > best_score {
            target = candidate;
            best_score = score;
        }
    }
    let increment = &target - current;
    Ok(HoleProposal {
        target,
        increment,
        nearest_catalog_distance: best_score,
    })
}

/// Add bounded attraction to an explicit D27 differential increment.
pub fn attraction_differential(
    current: &Array1<f64>,
    anchor: &Array1<f64>,
    left: &Array1<f64>,
    right: &Array1<f64>,
    scale: f64,
    attraction: f64,
    maximum_norm: f64,
) -> Result<CombinedProposal, ProposalError> {
    let dimension = current.len();
    if dimension == 0 {
        return Err(ProposalError::DescriptorDimension {
            expected: 1,
            actual: 0,
        });
    }
    for descriptor in [current, anchor, left, right] {
        validate_descriptor(descriptor, dimension)?;
    }
    validate_scale(scale)?;
    if !attraction.is_finite() || attraction < 0.0 {
        return Err(ProposalError::InvalidAttraction);
    }
    if !maximum_norm.is_finite() || maximum_norm < 0.0 {
        return Err(ProposalError::InvalidIncrementBound);
    }
    let mut increment = attraction * (anchor - current) + scale * (left - right);
    let norm = l2_norm(&increment);
    let clipped = norm > maximum_norm;
    if clipped && norm > 0.0 {
        increment *= maximum_norm / norm;
    }
    Ok(CombinedProposal { increment, clipped })
}

/// Apply D21 while retaining its structured failure at the proposal boundary.
pub fn pullback_increment(
    jacobian: ArrayView2<f64>,
    increment: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    reference_coordinates: Option<ArrayView1<f64>>,
    constraints: &PullbackConstraints,
    config: PullbackConfig,
) -> Result<PullbackResult, ProposalError> {
    regularized_pullback(
        jacobian,
        increment,
        weights,
        reference_coordinates,
        constraints,
        config,
    )
    .map_err(ProposalError::Pullback)
}

fn validate_catalog(catalog: &[Array1<f64>]) -> Result<usize, ProposalError> {
    let dimension = catalog.first().ok_or(ProposalError::EmptyCatalog)?.len();
    if dimension == 0 {
        return Err(ProposalError::DescriptorDimension {
            expected: 1,
            actual: 0,
        });
    }
    for descriptor in catalog {
        validate_descriptor(descriptor, dimension)?;
    }
    Ok(dimension)
}

fn validate_descriptor(descriptor: &Array1<f64>, dimension: usize) -> Result<(), ProposalError> {
    if descriptor.len() != dimension {
        return Err(ProposalError::DescriptorDimension {
            expected: dimension,
            actual: descriptor.len(),
        });
    }
    if descriptor.iter().any(|value| !value.is_finite()) {
        return Err(ProposalError::InvalidDescriptor);
    }
    Ok(())
}

fn validate_scale(scale: f64) -> Result<(), ProposalError> {
    if !scale.is_finite() || scale <= 0.0 || scale > 1.0 {
        Err(ProposalError::InvalidDifferentialScale)
    } else {
        Ok(())
    }
}

fn unit_vector(vector: &Array1<f64>) -> Result<Array1<f64>, ProposalError> {
    let norm = l2_norm(vector);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(ProposalError::InvalidDescriptor);
    }
    Ok(vector / norm)
}

fn gaussian_unit_vector<R: Rng + ?Sized>(dimension: usize, rng: &mut R) -> Array1<f64> {
    loop {
        let mut vector = Array1::zeros(dimension);
        for value in &mut vector {
            let radius = (-2.0 * rng.random::<f64>().max(f64::MIN_POSITIVE).ln()).sqrt();
            *value = radius * (2.0 * PI * rng.random::<f64>()).cos();
        }
        let norm = l2_norm(&vector);
        if norm > 0.0 && norm.is_finite() {
            return vector / norm;
        }
    }
}

fn nearest_distance(point: &Array1<f64>, catalog: &[Array1<f64>]) -> f64 {
    catalog
        .iter()
        .map(|entry| l2_norm(&(point - entry)))
        .fold(f64::INFINITY, f64::min)
}

fn l2_norm(vector: &Array1<f64>) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}
