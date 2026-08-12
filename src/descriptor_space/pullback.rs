//! Weighted regularized pullback from descriptor increments to Cartesian steps.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

const CONSTRAINT_EPSILON: f64 = 1e-12;

/// Numerical scales controlling one regularized pullback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PullbackConfig {
    /// Positive Tikhonov damping parameter `lambda`.
    pub damping: f64,
    /// Largest accepted dimensionless norm `||dx|| / length_scale`.
    pub trust_radius: f64,
    /// Canonical Cartesian length scale.
    pub length_scale: f64,
}

/// Linear Cartesian modes excluded from a pullback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullbackConstraints {
    /// Coordinate-level mask; true coordinates remain exactly zero.
    pub frozen_coordinates: Vec<bool>,
    /// Per-atom rigid-group labels; equal labels preserve pair distances to first order.
    pub rigid_group_labels: Vec<u32>,
    /// Remove the three global translational modes.
    pub remove_translation: bool,
}

/// Classified input or numerical failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PullbackError {
    /// Desired increment and weights must match the Jacobian row count.
    #[error(
        "descriptor dimensions are Jacobian={jacobian_rows}, desired={desired}, weights={weights}"
    )]
    DescriptorDimension {
        /// Jacobian row count.
        jacobian_rows: usize,
        /// Desired-increment length.
        desired: usize,
        /// Weight-vector length.
        weights: usize,
    },
    /// Frozen mask must match the Jacobian column count.
    #[error("frozen mask has length {actual}, expected {expected}")]
    FrozenDimension {
        /// Jacobian column count.
        expected: usize,
        /// Supplied mask length.
        actual: usize,
    },
    /// Translation and group constraints require complete Cartesian triples.
    #[error("coordinate dimension {actual} is not a positive multiple of three")]
    CartesianDimension {
        /// Jacobian column count.
        actual: usize,
    },
    /// Rigid-group labels must match the Cartesian atom count.
    #[error("group label count is {actual}, expected {expected}")]
    GroupDimension {
        /// Cartesian atom count.
        expected: usize,
        /// Supplied label count.
        actual: usize,
    },
    /// Rigid groups require reference coordinates matching the Jacobian columns.
    #[error("reference coordinate length is {actual}, expected {expected}")]
    ReferenceDimension {
        /// Jacobian column count.
        expected: usize,
        /// Supplied reference-coordinate length.
        actual: usize,
    },
    /// Damping must be finite and strictly positive.
    #[error("pullback damping must be finite and positive")]
    InvalidDamping,
    /// Trust radius must be finite and nonnegative.
    #[error("pullback trust radius must be finite and nonnegative")]
    InvalidTrustRadius,
    /// Length scale must be finite and strictly positive.
    #[error("pullback length scale must be finite and positive")]
    InvalidLengthScale,
    /// A Jacobian, desired increment, weight, or coordinate is NaN or infinite.
    #[error("nonfinite pullback input")]
    NonFiniteInput,
    /// Descriptor weights must be strictly positive.
    #[error("descriptor weights must be positive")]
    NonPositiveWeight,
    /// A grouped atom pair has coincident reference coordinates.
    #[error("rigid group contains coincident atoms {first_atom} and {second_atom}")]
    CoincidentGroupedAtoms {
        /// Lower atom index.
        first_atom: usize,
        /// Higher atom index.
        second_atom: usize,
    },
    /// The damped normal equations could not be solved stably.
    #[error("regularized pullback linear solve failed")]
    LinearSolve,
}

/// Cartesian step and diagnostics from one regularized pullback.
#[derive(Debug, Clone, PartialEq)]
pub struct PullbackResult {
    step: Array1<f64>,
    requested_weighted_norm: f64,
    realized_weighted_residual: f64,
    clipped: bool,
}

impl PullbackResult {
    /// Constrained and trust-radius-limited Cartesian step.
    pub fn step(&self) -> &Array1<f64> {
        &self.step
    }

    /// Weighted norm of the requested descriptor increment.
    pub fn requested_weighted_norm(&self) -> f64 {
        self.requested_weighted_norm
    }

    /// Weighted norm of `J dx - Delta p` after clipping.
    pub fn realized_weighted_residual(&self) -> f64 {
        self.realized_weighted_residual
    }

    /// Whether the unconstrained numerical solution exceeded the trust radius.
    pub fn clipped(&self) -> bool {
        self.clipped
    }
}

/// Solve the weighted D21 pullback after projecting forbidden Cartesian modes.
pub fn regularized_pullback(
    jacobian: ArrayView2<f64>,
    desired: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    reference_coordinates: Option<ArrayView1<f64>>,
    constraints: &PullbackConstraints,
    config: PullbackConfig,
) -> Result<PullbackResult, PullbackError> {
    let descriptor_dim = jacobian.nrows();
    let coordinate_dim = jacobian.ncols();
    if desired.len() != descriptor_dim || weights.len() != descriptor_dim {
        return Err(PullbackError::DescriptorDimension {
            jacobian_rows: descriptor_dim,
            desired: desired.len(),
            weights: weights.len(),
        });
    }
    if constraints.frozen_coordinates.len() != coordinate_dim {
        return Err(PullbackError::FrozenDimension {
            expected: coordinate_dim,
            actual: constraints.frozen_coordinates.len(),
        });
    }
    if !config.damping.is_finite() || config.damping <= 0.0 {
        return Err(PullbackError::InvalidDamping);
    }
    if !config.trust_radius.is_finite() || config.trust_radius < 0.0 {
        return Err(PullbackError::InvalidTrustRadius);
    }
    if !config.length_scale.is_finite() || config.length_scale <= 0.0 {
        return Err(PullbackError::InvalidLengthScale);
    }
    if jacobian.iter().any(|value| !value.is_finite())
        || desired.iter().any(|value| !value.is_finite())
        || weights.iter().any(|value| !value.is_finite())
    {
        return Err(PullbackError::NonFiniteInput);
    }
    if weights.iter().any(|&weight| weight <= 0.0) {
        return Err(PullbackError::NonPositiveWeight);
    }

    let geometric_constraints =
        constraints.remove_translation || !constraints.rigid_group_labels.is_empty();
    if geometric_constraints && (coordinate_dim == 0 || coordinate_dim % 3 != 0) {
        return Err(PullbackError::CartesianDimension {
            actual: coordinate_dim,
        });
    }
    let atoms = coordinate_dim / 3;
    if !constraints.rigid_group_labels.is_empty() && constraints.rigid_group_labels.len() != atoms {
        return Err(PullbackError::GroupDimension {
            expected: atoms,
            actual: constraints.rigid_group_labels.len(),
        });
    }
    let reference = if constraints.rigid_group_labels.is_empty() {
        None
    } else {
        let reference = reference_coordinates.ok_or(PullbackError::ReferenceDimension {
            expected: coordinate_dim,
            actual: 0,
        })?;
        if reference.len() != coordinate_dim {
            return Err(PullbackError::ReferenceDimension {
                expected: coordinate_dim,
                actual: reference.len(),
            });
        }
        if reference.iter().any(|value| !value.is_finite()) {
            return Err(PullbackError::NonFiniteInput);
        }
        Some(reference)
    };

    let forbidden = forbidden_basis(
        coordinate_dim,
        constraints,
        reference.as_ref().map(|coordinates| coordinates.view()),
    )?;
    let projector = projector(coordinate_dim, &forbidden);
    let projected_jacobian = jacobian.dot(&projector);
    let mut normal = Array2::<f64>::zeros((coordinate_dim, coordinate_dim));
    let mut rhs = Array1::<f64>::zeros(coordinate_dim);
    for row in 0..descriptor_dim {
        for left in 0..coordinate_dim {
            let weighted_left = weights[row] * projected_jacobian[[row, left]];
            rhs[left] += weighted_left * desired[row];
            for right in 0..coordinate_dim {
                normal[[left, right]] += weighted_left * projected_jacobian[[row, right]];
            }
        }
    }
    let damping_squared = config.damping * config.damping;
    for coordinate in 0..coordinate_dim {
        normal[[coordinate, coordinate]] += damping_squared;
    }
    let unconstrained = cholesky_solve(&normal, &rhs).ok_or(PullbackError::LinearSolve)?;
    let mut step = projector.dot(&unconstrained);
    let norm = step.iter().map(|value| value * value).sum::<f64>().sqrt();
    let maximum_norm = config.trust_radius * config.length_scale;
    let clipped = norm > maximum_norm;
    if clipped && norm > 0.0 {
        step *= maximum_norm / norm;
    }
    let requested_weighted_norm = weighted_norm(desired, weights);
    let realized = jacobian.dot(&step) - desired;
    let realized_weighted_residual = weighted_norm(realized.view(), weights);
    Ok(PullbackResult {
        step,
        requested_weighted_norm,
        realized_weighted_residual,
        clipped,
    })
}

fn forbidden_basis(
    coordinate_dim: usize,
    constraints: &PullbackConstraints,
    reference: Option<ArrayView1<f64>>,
) -> Result<Vec<Array1<f64>>, PullbackError> {
    let mut basis = Vec::new();
    for (coordinate, &frozen) in constraints.frozen_coordinates.iter().enumerate() {
        if frozen {
            let mut mode = Array1::zeros(coordinate_dim);
            mode[coordinate] = 1.0;
            add_orthonormal(mode, &mut basis);
        }
    }

    if let Some(reference) = reference {
        let atoms = coordinate_dim / 3;
        for first_atom in 0..atoms {
            for second_atom in first_atom + 1..atoms {
                if constraints.rigid_group_labels[first_atom]
                    != constraints.rigid_group_labels[second_atom]
                {
                    continue;
                }
                let mut difference = [0.0; 3];
                for axis in 0..3 {
                    difference[axis] =
                        reference[3 * first_atom + axis] - reference[3 * second_atom + axis];
                }
                let distance = difference
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if distance <= CONSTRAINT_EPSILON {
                    return Err(PullbackError::CoincidentGroupedAtoms {
                        first_atom,
                        second_atom,
                    });
                }
                let mut stretch = Array1::zeros(coordinate_dim);
                for axis in 0..3 {
                    let direction = difference[axis] / distance;
                    stretch[3 * first_atom + axis] = direction;
                    stretch[3 * second_atom + axis] = -direction;
                }
                add_orthonormal(stretch, &mut basis);
            }
        }
    }

    if constraints.remove_translation {
        for axis in 0..3 {
            let mut translation = Array1::zeros(coordinate_dim);
            for atom in 0..coordinate_dim / 3 {
                translation[3 * atom + axis] = 1.0;
            }
            add_orthonormal(translation, &mut basis);
        }
    }
    Ok(basis)
}

fn add_orthonormal(mut candidate: Array1<f64>, basis: &mut Vec<Array1<f64>>) {
    for mode in basis.iter() {
        let coefficient = candidate.dot(mode);
        candidate.scaled_add(-coefficient, mode);
    }
    let norm = candidate
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > CONSTRAINT_EPSILON {
        candidate /= norm;
        basis.push(candidate);
    }
}

fn projector(coordinate_dim: usize, forbidden: &[Array1<f64>]) -> Array2<f64> {
    let mut projector = Array2::eye(coordinate_dim);
    for mode in forbidden {
        for row in 0..coordinate_dim {
            for column in 0..coordinate_dim {
                projector[[row, column]] -= mode[row] * mode[column];
            }
        }
    }
    projector
}

fn weighted_norm(values: ArrayView1<f64>, weights: ArrayView1<f64>) -> f64 {
    values
        .iter()
        .zip(weights)
        .map(|(value, weight)| weight * value * value)
        .sum::<f64>()
        .sqrt()
}

fn cholesky_solve(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Option<Array1<f64>> {
    let dimension = matrix.nrows();
    if matrix.ncols() != dimension || rhs.len() != dimension {
        return None;
    }
    let mut lower = Array2::<f64>::zeros((dimension, dimension));
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[[row, column]];
            for inner in 0..column {
                value -= lower[[row, inner]] * lower[[column, inner]];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return None;
                }
                lower[[row, column]] = value.sqrt();
            } else {
                lower[[row, column]] = value / lower[[column, column]];
            }
        }
    }
    let mut intermediate = Array1::<f64>::zeros(dimension);
    for row in 0..dimension {
        let mut value = rhs[row];
        for column in 0..row {
            value -= lower[[row, column]] * intermediate[column];
        }
        intermediate[row] = value / lower[[row, row]];
    }
    let mut solution = Array1::<f64>::zeros(dimension);
    for row in (0..dimension).rev() {
        let mut value = intermediate[row];
        for column in row + 1..dimension {
            value -= lower[[column, row]] * solution[column];
        }
        solution[row] = value / lower[[row, row]];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}
