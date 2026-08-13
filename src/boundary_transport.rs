//! Transport proposals from observed attraction-region boundary crossings.
//!
//! A crossing displacement is empirical target-blind evidence. It is aligned
//! from its recorded source frame to the current structure, combined with a
//! centred perturbation, constrained, and returned for the caller's ordinary
//! quench and fresh physical validation.

use ndarray::{Array1, ArrayView1};

/// Invalid crossing evidence or geometry constraints.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BoundaryTransportError {
    /// Coordinates must contain at least one complete Cartesian triple.
    #[error("boundary transport coordinates must be a positive multiple of three")]
    CartesianDimension,
    /// Recorded source and destination dimensions differ.
    #[error("observed crossing endpoints have different dimensions")]
    CrossingDimension,
    /// Current state, crossing, and perturbation dimensions differ.
    #[error("boundary transport input dimensions differ")]
    InputDimension,
    /// A coordinate contains NaN or infinity.
    #[error("boundary transport input contains a nonfinite coordinate")]
    NonFiniteInput,
    /// Noise scale or trust radius lies outside its domain.
    #[error("boundary transport scale is invalid")]
    InvalidScale,
    /// A nonempty frozen mask must match the coordinate dimension.
    #[error("boundary transport frozen mask has the wrong dimension")]
    FrozenDimension,
    /// A rigid group contains an atom outside the current structure.
    #[error("boundary transport rigid group contains an invalid atom")]
    InvalidRigidAtom,
    /// A rigid group cannot mix frozen and mobile atoms.
    #[error("boundary transport rigid group mixes frozen and mobile atoms")]
    PartiallyFrozenRigidGroup,
}

/// One observed source-to-destination crossing on the quenched landscape.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedCrossing {
    from: Array1<f64>,
    to: Array1<f64>,
}

impl ObservedCrossing {
    /// Validate and retain a crossing in its recorded Cartesian frame.
    pub fn new(from: Array1<f64>, to: Array1<f64>) -> Result<Self, BoundaryTransportError> {
        if from.len() != to.len() {
            return Err(BoundaryTransportError::CrossingDimension);
        }
        if from.is_empty() || !from.len().is_multiple_of(3) {
            return Err(BoundaryTransportError::CartesianDimension);
        }
        if from.iter().chain(to.iter()).any(|value| !value.is_finite()) {
            return Err(BoundaryTransportError::NonFiniteInput);
        }
        Ok(Self { from, to })
    }

    /// Recorded source coordinates.
    pub fn from(&self) -> ArrayView1<'_, f64> {
        self.from.view()
    }

    /// Recorded destination coordinates.
    pub fn to(&self) -> ArrayView1<'_, f64> {
        self.to.view()
    }
}

/// Perturbation and physical-constraint parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryTransportConfig {
    /// Multiplier for the supplied zero-information stochastic draw.
    pub noise_scale: f64,
    /// Maximum Cartesian norm of the transported step.
    pub trust_radius: f64,
    /// Coordinate-level mask; true coordinates remain fixed.
    pub frozen_coordinates: Vec<bool>,
    /// Atom-index groups retracted to their nearest finite rigid motions.
    pub rigid_groups: Vec<Vec<usize>>,
}

impl BoundaryTransportConfig {
    /// Construct an unconstrained transport configuration.
    pub fn unconstrained(trust_radius: f64) -> Self {
        Self {
            noise_scale: 1.0,
            trust_radius,
            frozen_coordinates: Vec::new(),
            rigid_groups: Vec::new(),
        }
    }
}

/// Align, perturb, and constrain an observed boundary-crossing displacement.
pub fn boundary_transport(
    current: ArrayView1<f64>,
    crossing: &ObservedCrossing,
    noise: ArrayView1<f64>,
    config: &BoundaryTransportConfig,
) -> Result<Array1<f64>, BoundaryTransportError> {
    let dimension = current.len();
    if dimension == 0 || !dimension.is_multiple_of(3) {
        return Err(BoundaryTransportError::CartesianDimension);
    }
    if crossing.from.len() != dimension || noise.len() != dimension {
        return Err(BoundaryTransportError::InputDimension);
    }
    if current
        .iter()
        .chain(noise.iter())
        .any(|value| !value.is_finite())
    {
        return Err(BoundaryTransportError::NonFiniteInput);
    }
    if !config.noise_scale.is_finite()
        || config.noise_scale < 0.0
        || !config.trust_radius.is_finite()
        || config.trust_radius < 0.0
    {
        return Err(BoundaryTransportError::InvalidScale);
    }
    if !config.frozen_coordinates.is_empty() && config.frozen_coordinates.len() != dimension {
        return Err(BoundaryTransportError::FrozenDimension);
    }
    let frozen = if config.frozen_coordinates.is_empty() {
        vec![false; dimension]
    } else {
        config.frozen_coordinates.clone()
    };
    validate_rigid_groups(dimension / 3, &frozen, &config.rigid_groups)?;

    let rotation = alignment_rotation(crossing.from.view(), current);
    let mut step = Array1::zeros(dimension);
    for atom in 0..dimension / 3 {
        let displacement = [
            crossing.to[3 * atom] - crossing.from[3 * atom],
            crossing.to[3 * atom + 1] - crossing.from[3 * atom + 1],
            crossing.to[3 * atom + 2] - crossing.from[3 * atom + 2],
        ];
        for axis in 0..3 {
            step[3 * atom + axis] = rotation[axis][0] * displacement[0]
                + rotation[axis][1] * displacement[1]
                + rotation[axis][2] * displacement[2];
        }
    }
    let centred_noise = centred_noise(noise, &frozen);
    for coordinate in 0..dimension {
        if frozen[coordinate] {
            step[coordinate] = 0.0;
        } else {
            step[coordinate] += config.noise_scale * centred_noise[coordinate];
        }
    }
    clip_norm(&mut step, config.trust_radius);
    if !config.rigid_groups.is_empty() {
        crate::soap::project_rigid_groups(current, &mut step, &config.rigid_groups);
    }
    for coordinate in 0..dimension {
        if frozen[coordinate] {
            step[coordinate] = 0.0;
        }
    }
    Ok(current.to_owned() + step)
}

fn alignment_rotation(from: ArrayView1<f64>, current: ArrayView1<f64>) -> [[f64; 3]; 3] {
    let atoms = from.len() / 3;
    let mut from_points = Vec::with_capacity(atoms);
    let mut current_points = Vec::with_capacity(atoms);
    let mut from_centroid = [0.0; 3];
    let mut current_centroid = [0.0; 3];
    for atom in 0..atoms {
        for axis in 0..3 {
            from_centroid[axis] += from[3 * atom + axis];
            current_centroid[axis] += current[3 * atom + axis];
        }
    }
    for axis in 0..3 {
        from_centroid[axis] /= atoms as f64;
        current_centroid[axis] /= atoms as f64;
    }
    for atom in 0..atoms {
        from_points.push([
            from[3 * atom] - from_centroid[0],
            from[3 * atom + 1] - from_centroid[1],
            from[3 * atom + 2] - from_centroid[2],
        ]);
        current_points.push([
            current[3 * atom] - current_centroid[0],
            current[3 * atom + 1] - current_centroid[1],
            current[3 * atom + 2] - current_centroid[2],
        ]);
    }
    crate::soap::horn_rotation(&from_points, &current_points)
}

fn centred_noise(noise: ArrayView1<f64>, frozen: &[bool]) -> Array1<f64> {
    let mut centred = noise.to_owned();
    for axis in 0..3 {
        let movable = (0..noise.len() / 3)
            .map(|atom| 3 * atom + axis)
            .filter(|coordinate| !frozen[*coordinate])
            .collect::<Vec<_>>();
        let mean = if movable.is_empty() {
            0.0
        } else {
            movable
                .iter()
                .map(|coordinate| noise[*coordinate])
                .sum::<f64>()
                / movable.len() as f64
        };
        for coordinate in movable {
            centred[coordinate] -= mean;
        }
    }
    centred
}

fn clip_norm(step: &mut Array1<f64>, trust_radius: f64) {
    let norm = step.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > trust_radius && norm > 0.0 {
        *step *= trust_radius / norm;
    }
}

fn validate_rigid_groups(
    atoms: usize,
    frozen: &[bool],
    groups: &[Vec<usize>],
) -> Result<(), BoundaryTransportError> {
    for group in groups {
        if group.iter().any(|atom| *atom >= atoms) {
            return Err(BoundaryTransportError::InvalidRigidAtom);
        }
        let frozen_coordinates = group
            .iter()
            .flat_map(|atom| (0..3).map(move |axis| 3 * *atom + axis))
            .filter(|coordinate| frozen[*coordinate])
            .count();
        if frozen_coordinates != 0 && frozen_coordinates != 3 * group.len() {
            return Err(BoundaryTransportError::PartiallyFrozenRigidGroup);
        }
    }
    Ok(())
}
