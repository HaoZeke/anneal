//! Minimum--saddle--minimum exploration with rgmin and rgsaddle.
//!
//! The descriptor orders structural comparisons but never certifies identity.
//! A caller-supplied exact witness makes the final equivalence decision. This
//! keeps catalog admission independent of a fixed descriptor radius while the
//! same universal descriptor remains available for novelty and acquisition.

use std::fmt::Display;

use ndarray::{Array1, Array2, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rgmin::{GradNorm, Lbfgs};
use rgsaddle::{
    IrcConfig, IrcDirection, IrcSession, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus,
    PointSurface, SaddleError, SellaSaddleConfig, SellaSaddleSession, exact_eigh,
};

use crate::curvature::{project_rigid_with, rigid_basis};
use crate::descriptor_space::{
    DescriptorError, DescriptorGeometry, DescriptorSpace, DescriptorVector,
};

/// Energy and Cartesian-gradient evaluator used by all exploration stages.
pub trait PesSurface: Sync {
    /// Surface-specific evaluation failure.
    type Error: Display;

    /// Return energy and a gradient matching the Cartesian input dimension.
    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error>;
}

/// Species, boundary geometry, and caller domain carried by exact identity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StructureContext {
    species: Option<Vec<u32>>,
    masses: Option<Vec<f64>>,
    geometry: Option<DescriptorGeometry>,
    identity_domain: Option<String>,
}

impl StructureContext {
    /// Construct an owned identity context for a stationary structure.
    pub fn new(
        species: Option<Vec<u32>>,
        geometry: Option<DescriptorGeometry>,
        identity_domain: Option<String>,
    ) -> Self {
        Self {
            species,
            masses: None,
            geometry,
            identity_domain,
        }
    }

    /// Ordered atomic numbers, when chemical identities are known.
    pub fn species(&self) -> Option<&[u32]> {
        self.species.as_deref()
    }

    /// Per-atom masses in the native PES unit system, when known.
    pub fn masses(&self) -> Option<&[f64]> {
        self.masses.as_deref()
    }

    /// Attach per-atom native masses to the identity and persistence context.
    pub fn with_masses(mut self, masses: Option<Vec<f64>>) -> Self {
        self.masses = masses;
        self
    }

    /// Descriptor geometry, including cell and periodic axes.
    pub fn geometry(&self) -> Option<DescriptorGeometry> {
        self.geometry
    }

    /// Caller-defined namespace preventing cross-system identity merges.
    pub fn identity_domain(&self) -> Option<&str> {
        self.identity_domain.as_deref()
    }
}

/// Borrowed structure and identity metadata presented to an exact witness.
#[derive(Debug, Clone, Copy)]
pub struct StructureView<'a> {
    /// Cartesian coordinates.
    pub coordinates: ArrayView1<'a, f64>,
    /// Species, boundary geometry, and caller identity namespace.
    pub context: &'a StructureContext,
}

/// Symmetry-aware final witness for structural identity.
pub trait ExactStructureWitness {
    /// Whether two Cartesian structures represent the same stationary point.
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool;

    /// Whether two fully contextualized structures represent one identity.
    ///
    /// The default rejects species, cell/PBC, or identity-domain mismatches
    /// before applying the coordinate witness. Implementations may override
    /// this method to canonicalize symmetry-equivalent cells or domains.
    fn equivalent_structures(&self, left: StructureView<'_>, right: StructureView<'_>) -> bool {
        left.context == right.context && self.equivalent(left.coordinates, right.coordinates)
    }
}

impl<F> ExactStructureWitness for F
where
    F: for<'a, 'b> Fn(ArrayView1<'a, f64>, ArrayView1<'b, f64>) -> bool,
{
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        self(left, right)
    }
}

/// Lowest-mode ride used to approach a saddle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RideMethod {
    /// Jónsson dimer rotation with finite-difference Hessian actions.
    Dimer,
    /// Matrix-free Lanczos estimate of the lowest Hessian mode.
    Lanczos,
}

impl RideMethod {
    fn min_mode(self) -> MinModeKind {
        match self {
            Self::Dimer => MinModeKind::Dimer,
            Self::Lanczos => MinModeKind::Lanczos,
        }
    }
}

/// Sign of a deterministic transition-search initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RideModeDirection {
    /// Reverse the generated mode.
    Negative,
    /// Use the generated mode as sampled.
    Positive,
}

impl RideModeDirection {
    fn sign(self) -> f64 {
        match self {
            Self::Negative => -1.0,
            Self::Positive => 1.0,
        }
    }
}

fn ranked_seed(seed: u64, rank: u16) -> u64 {
    let mut mixed = seed ^ (u64::from(rank) + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn normalized_gaussian(
    dimension: usize,
    seed: u64,
    rank: u16,
) -> Result<Array1<f64>, PesExplorationError> {
    if dimension == 0 {
        return Err(PesExplorationError::InvalidShape(
            "ride dimension must be nonempty",
        ));
    }
    let mut rng = StdRng::seed_from_u64(ranked_seed(seed, rank));
    let mode = Array1::from_shape_simple_fn(dimension, || {
        let sample: f64 = StandardNormal.sample(&mut rng);
        sample
    });
    normalize_mode(mode.view())
}

/// Generate a deterministic normalized Gaussian mode for an arbitrary-N surface.
pub fn gaussian_nd_mode(
    dimension: usize,
    seed: u64,
    rank: u16,
    direction: RideModeDirection,
) -> Result<Array1<f64>, PesExplorationError> {
    Ok(normalized_gaussian(dimension, seed, rank)? * direction.sign())
}

fn translation_basis(atom_count: usize) -> Vec<Array1<f64>> {
    let mut basis = Vec::with_capacity(3);
    let scale = (atom_count as f64).sqrt().recip();
    for axis in 0..3 {
        let mut translation = Array1::zeros(3 * atom_count);
        for atom in 0..atom_count {
            translation[3 * atom + axis] = scale;
        }
        basis.push(translation);
    }
    basis
}

/// Generate an atom-local Gaussian mode with physical constraints applied.
///
/// Gaussian components are attenuated by scaled minimum-image distance from
/// `representative_atom`. Fully mobile finite clusters lose translations and
/// rotations; periodic structures lose translations only. Frozen atoms remain
/// exactly stationary and make the external constraints authoritative instead
/// of applying whole-structure rigid-motion projection.
#[allow(clippy::too_many_arguments)]
pub fn localized_cartesian_mode(
    coordinates: ArrayView1<f64>,
    representative_atom: usize,
    frozen_atoms: &[bool],
    geometry: DescriptorGeometry,
    localization_radius: f64,
    seed: u64,
    rank: u16,
    direction: RideModeDirection,
) -> Result<Array1<f64>, PesExplorationError> {
    if coordinates.is_empty() || !coordinates.len().is_multiple_of(3) {
        return Err(PesExplorationError::InvalidShape(
            "coordinates must be nonempty 3N Cartesian",
        ));
    }
    if coordinates.iter().any(|value| !value.is_finite()) {
        return Err(PesExplorationError::InvalidShape(
            "coordinates are nonfinite",
        ));
    }
    let atom_count = coordinates.len() / 3;
    if frozen_atoms.len() != atom_count {
        return Err(PesExplorationError::InvalidShape(
            "frozen mask must contain one value per atom",
        ));
    }
    if representative_atom >= atom_count {
        return Err(PesExplorationError::InvalidShape(
            "representative atom is outside the structure",
        ));
    }
    if frozen_atoms[representative_atom] {
        return Err(PesExplorationError::InvalidShape(
            "representative atom is frozen",
        ));
    }
    if !localization_radius.is_finite() || localization_radius <= 0.0 {
        return Err(PesExplorationError::InvalidConfig(
            "mode localization radius",
        ));
    }

    let mut mode = normalized_gaussian(coordinates.len(), seed, rank)?;
    let target = [
        coordinates[3 * representative_atom],
        coordinates[3 * representative_atom + 1],
        coordinates[3 * representative_atom + 2],
    ];
    let radius_squared = localization_radius * localization_radius;
    for atom in 0..atom_count {
        if frozen_atoms[atom] {
            mode[3 * atom] = 0.0;
            mode[3 * atom + 1] = 0.0;
            mode[3 * atom + 2] = 0.0;
            continue;
        }
        let displacement = geometry.displacement([
            coordinates[3 * atom] - target[0],
            coordinates[3 * atom + 1] - target[1],
            coordinates[3 * atom + 2] - target[2],
        ]);
        let distance_squared = displacement
            .iter()
            .map(|component| component * component)
            .sum::<f64>();
        let weight = (-0.5 * distance_squared / radius_squared).exp();
        for axis in 0..3 {
            mode[3 * atom + axis] *= weight;
        }
    }

    if frozen_atoms.iter().all(|frozen| !frozen) {
        let basis = if geometry.periodic().iter().any(|periodic| *periodic) {
            translation_basis(atom_count)
        } else {
            rigid_basis(coordinates)
        };
        project_rigid_with(&mut mode, &basis);
    }
    let mode = normalize_mode(mode.view())?;
    Ok(mode * direction.sign())
}

/// Controls for one minimum--saddle--minimum connection attempt.
#[derive(Debug, Clone)]
pub struct PesExplorationConfig {
    /// Lowest-mode algorithm.
    pub ride_method: RideMethod,
    /// Maximum rgmin L-BFGS iterations for each minimum certification.
    pub quench_steps: usize,
    /// Maximum rgsaddle lowest-mode steps.
    pub saddle_steps: usize,
    /// Maximum rgsaddle IRC outer points per direction.
    pub irc_steps: usize,
    /// Maximum Sella P-RFO refinement steps.
    pub prfo_steps: usize,
    /// Number of expanding hyperspheres probed before minimum-mode following.
    pub activation_attempts: usize,
    /// Multiplicative radius increase while leaving the convex basin.
    pub activation_growth: f64,
    /// Infinity-norm gradient threshold for a certified minimum.
    pub quench_gradient_tolerance: f64,
    /// Force threshold for the saddle sessions.
    pub saddle_force_tolerance: f64,
    /// Displacement from the quenched minimum along the supplied mode.
    pub saddle_displacement: f64,
    /// Curvature must be below the negative of this value.
    pub negative_curvature_tolerance: f64,
    /// Cartesian finite-difference step used to certify stationary index.
    pub hessian_step: f64,
    /// Maximum Cartesian move used by minimum-mode and IRC steppers.
    pub maximum_move: f64,
    /// IRC mass-weighted outer radius.
    pub irc_step: f64,
    /// IRC force threshold before endpoint quenching.
    pub irc_force_tolerance: f64,
    /// Refine the lowest-mode candidate with Sella order-1 P-RFO.
    pub refine_with_prfo: bool,
}

impl Default for PesExplorationConfig {
    fn default() -> Self {
        Self {
            ride_method: RideMethod::Dimer,
            quench_steps: 1_000,
            saddle_steps: 1_000,
            irc_steps: 200,
            prfo_steps: 300,
            activation_attempts: 4,
            activation_growth: 2.0,
            quench_gradient_tolerance: 1e-6,
            saddle_force_tolerance: 1e-3,
            saddle_displacement: 0.1,
            negative_curvature_tolerance: 1e-6,
            hessian_step: 1e-4,
            maximum_move: 0.2,
            irc_step: 0.1,
            irc_force_tolerance: 0.05,
            refine_with_prfo: true,
        }
    }
}

impl PesExplorationConfig {
    fn validate(&self) -> Result<(), PesExplorationError> {
        if self.quench_steps == 0
            || self.saddle_steps == 0
            || self.irc_steps == 0
            || self.prfo_steps == 0
            || self.activation_attempts == 0
        {
            return Err(PesExplorationError::InvalidConfig(
                "iteration limits must be positive",
            ));
        }
        for (name, value) in [
            ("quench gradient tolerance", self.quench_gradient_tolerance),
            ("saddle force tolerance", self.saddle_force_tolerance),
            ("saddle displacement", self.saddle_displacement),
            ("activation growth", self.activation_growth),
            ("Hessian finite-difference step", self.hessian_step),
            ("maximum move", self.maximum_move),
            ("IRC step", self.irc_step),
            ("IRC force tolerance", self.irc_force_tolerance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PesExplorationError::InvalidConfig(name));
            }
        }
        if self.activation_growth <= 1.0 {
            return Err(PesExplorationError::InvalidConfig(
                "activation growth must exceed one",
            ));
        }
        if !self.negative_curvature_tolerance.is_finite() || self.negative_curvature_tolerance < 0.0
        {
            return Err(PesExplorationError::InvalidConfig(
                "negative curvature tolerance",
            ));
        }
        Ok(())
    }
}

/// Failure that prevents one connection from becoming catalog evidence.
#[derive(Debug, thiserror::Error)]
pub enum PesExplorationError {
    /// Coordinates, masses, species, or a mode have incompatible dimensions.
    #[error("invalid PES exploration shape: {0}")]
    InvalidShape(&'static str),
    /// A configuration field violates its numerical domain.
    #[error("invalid PES exploration configuration: {0}")]
    InvalidConfig(&'static str),
    /// The user surface rejected an evaluation.
    #[error("PES evaluation failed: {0}")]
    Surface(String),
    /// A surface returned a malformed or nonfinite value/gradient pair.
    #[error("PES evaluation returned {0}")]
    InvalidEvaluation(&'static str),
    /// rgmin stopped without satisfying the minimum force condition.
    #[error("rgmin quench stopped at gradient infinity norm {max_gradient}")]
    QuenchNotConverged {
        /// Final infinity norm.
        max_gradient: f64,
    },
    /// rgsaddle stopped before the saddle force condition.
    #[error("rgsaddle {stage} stopped before convergence")]
    SaddleNotConverged {
        /// Saddle stage that stopped.
        stage: &'static str,
    },
    /// The activation ray did not reach negative directional curvature.
    #[error(
        "activation ray did not escape the convex basin; lowest directional curvature {lowest_curvature}"
    )]
    ActivationNotEscaped {
        /// Lowest directional curvature observed across all activation radii.
        lowest_curvature: f64,
    },
    /// Minimum-mode following ended without retaining an unstable mode.
    #[error("minimum-mode ride lost negative curvature; final curvature {curvature}")]
    MinimumModeLostCurvature {
        /// Curvature reported at the force-converged minimum-mode candidate.
        curvature: f64,
    },
    /// The stationary candidate is not an index-one saddle.
    #[error(
        "stationary candidate has index {negative_modes} and lowest curvature {lowest_curvature}"
    )]
    NotFirstOrder {
        /// Number of eigenvalues below the negative-curvature tolerance.
        negative_modes: usize,
        /// Lowest certified Hessian eigenvalue.
        lowest_curvature: f64,
    },
    /// Both IRC branches resolved to one exact minimum.
    #[error("forward and reverse IRC branches resolved to one minimum")]
    CollapsedConnection,
    /// A one-sided minimum-mode ride did not reconnect to its source basin.
    #[error("saddle descent endpoints do not contain the source minimum")]
    DisconnectedConnection,
    /// Universal descriptor evaluation or comparison failed.
    #[error(transparent)]
    Descriptor(#[from] DescriptorError),
    /// rgsaddle session failure.
    #[error("rgsaddle failed: {0}")]
    Saddle(String),
}

/// One rgmin-certified local minimum.
#[derive(Debug, Clone)]
pub struct MinimumRecord {
    /// Stable index in the owning [`PesNetwork`].
    pub id: usize,
    /// Receiving-side energy evaluated at the stored coordinates.
    pub energy: f64,
    /// Cartesian stationary point.
    pub coordinates: Array1<f64>,
    /// Species, cell/PBC, and system namespace used for exact identity.
    pub context: StructureContext,
    /// Final gradient infinity norm.
    pub max_gradient: f64,
    /// Universal descriptor used for retrieval and novelty.
    pub descriptor: DescriptorVector,
}

/// Result of exact-witness minimum admission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimumAdmission {
    /// Stable minimum index.
    pub id: usize,
    /// Whether a new exact structural identity entered the network.
    pub is_new: bool,
    /// Nearest descriptor distance before the exact witness decision.
    pub nearest_descriptor_distance: Option<f64>,
}

/// One index-one saddle and its two quenched IRC endpoints.
#[derive(Debug, Clone)]
pub struct SaddleConnection {
    /// Stable index in the owning [`PesNetwork`].
    pub id: usize,
    /// Minimum from which the mode ride started.
    pub origin: usize,
    /// Exact-witness endpoint minimum indices, forward then reverse.
    pub endpoints: [usize; 2],
    /// Saddle energy.
    pub saddle_energy: f64,
    /// Cartesian saddle coordinates.
    pub saddle_coordinates: Array1<f64>,
    /// Species, cell/PBC, and system namespace used for exact identity.
    pub context: StructureContext,
    /// Lowest curvature reported at the force-converged candidate.
    pub curvature: f64,
    /// Receiving-side lowest Hessian eigenvector at the certified saddle.
    pub lowest_mode: Array1<f64>,
    /// Number of certified negative Hessian eigenvalues.
    pub negative_modes: usize,
    /// Saddle gradient infinity norm.
    pub saddle_max_gradient: f64,
    /// Descriptor retained for saddle deduplication and novelty.
    pub descriptor: DescriptorVector,
    /// Lowest-mode ride used for this connection.
    pub ride_method: RideMethod,
    /// Whether each IRC session met its own endpoint force condition.
    pub irc_at_minimum: [bool; 2],
}

/// Exact stationary-point graph accumulated across ride attempts.
#[derive(Debug, Clone, Default)]
pub struct PesNetwork {
    minima: Vec<MinimumRecord>,
    saddles: Vec<SaddleConnection>,
}

impl PesNetwork {
    /// Empty stationary-point graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of exact-witness minimum identities.
    pub fn minimum_count(&self) -> usize {
        self.minima.len()
    }

    /// Number of exact-witness saddle identities.
    pub fn saddle_count(&self) -> usize {
        self.saddles.len()
    }

    /// Stored minima in stable admission order.
    pub fn minima(&self) -> &[MinimumRecord] {
        &self.minima
    }

    /// Stored saddle connections in stable admission order.
    pub fn saddles(&self) -> &[SaddleConnection] {
        &self.saddles
    }

    pub(crate) fn from_records(minima: Vec<MinimumRecord>, saddles: Vec<SaddleConnection>) -> Self {
        Self { minima, saddles }
    }

    /// Admit a certified minimum without applying a descriptor cutoff.
    pub fn admit_minimum<W: ExactStructureWitness + ?Sized>(
        &mut self,
        energy: f64,
        coordinates: Array1<f64>,
        max_gradient: f64,
        descriptor: DescriptorVector,
        witness: &W,
    ) -> Result<MinimumAdmission, DescriptorError> {
        self.admit_minimum_with_context(
            energy,
            coordinates,
            max_gradient,
            descriptor,
            StructureContext::default(),
            witness,
        )
    }

    /// Admit a certified minimum with its complete exact-identity context.
    pub fn admit_minimum_with_context<W: ExactStructureWitness + ?Sized>(
        &mut self,
        energy: f64,
        coordinates: Array1<f64>,
        max_gradient: f64,
        descriptor: DescriptorVector,
        context: StructureContext,
        witness: &W,
    ) -> Result<MinimumAdmission, DescriptorError> {
        let mut ordered = self
            .minima
            .iter()
            .enumerate()
            .map(|(index, minimum)| {
                minimum
                    .descriptor
                    .distance(&descriptor)
                    .map(|distance| (distance, index))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ordered.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let nearest_descriptor_distance = ordered.first().map(|entry| entry.0);
        for (_, index) in ordered {
            if witness.equivalent_structures(
                StructureView {
                    coordinates: self.minima[index].coordinates.view(),
                    context: &self.minima[index].context,
                },
                StructureView {
                    coordinates: coordinates.view(),
                    context: &context,
                },
            ) {
                if energy < self.minima[index].energy {
                    self.minima[index].energy = energy;
                    self.minima[index].coordinates = coordinates;
                    self.minima[index].context = context;
                    self.minima[index].max_gradient = max_gradient;
                    self.minima[index].descriptor = descriptor;
                }
                return Ok(MinimumAdmission {
                    id: index,
                    is_new: false,
                    nearest_descriptor_distance,
                });
            }
        }
        let id = self.minima.len();
        self.minima.push(MinimumRecord {
            id,
            energy,
            coordinates,
            context,
            max_gradient,
            descriptor,
        });
        Ok(MinimumAdmission {
            id,
            is_new: true,
            nearest_descriptor_distance,
        })
    }

    fn admit_saddle<W: ExactStructureWitness + ?Sized>(
        &mut self,
        mut candidate: SaddleConnection,
        witness: &W,
    ) -> Result<SaddleConnection, DescriptorError> {
        let mut ordered = self
            .saddles
            .iter()
            .enumerate()
            .map(|(index, saddle)| {
                saddle
                    .descriptor
                    .distance(&candidate.descriptor)
                    .map(|distance| (distance, index))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
        for (_, index) in ordered {
            let existing = &self.saddles[index];
            let same_endpoints = existing.endpoints == candidate.endpoints
                || existing.endpoints == [candidate.endpoints[1], candidate.endpoints[0]];
            if same_endpoints
                && witness.equivalent_structures(
                    StructureView {
                        coordinates: existing.saddle_coordinates.view(),
                        context: &existing.context,
                    },
                    StructureView {
                        coordinates: candidate.saddle_coordinates.view(),
                        context: &candidate.context,
                    },
                )
            {
                return Ok(existing.clone());
            }
        }
        candidate.id = self.saddles.len();
        self.saddles.push(candidate.clone());
        Ok(candidate)
    }
}

/// One rgmin-certified minimum on an arbitrary-dimensional point surface.
#[derive(Debug, Clone)]
pub struct NdMinimumRecord {
    /// Stable index in the owning [`NdPesNetwork`].
    pub id: usize,
    /// Receiving-side energy evaluated at the stored point.
    pub energy: f64,
    /// Point in the native coordinates of the optimization problem.
    pub coordinates: Array1<f64>,
    /// Final gradient infinity norm.
    pub max_gradient: f64,
}

/// One certified index-one connection on an arbitrary-dimensional surface.
#[derive(Debug, Clone)]
pub struct NdSaddleConnection {
    /// Stable index in the owning [`NdPesNetwork`].
    pub id: usize,
    /// Minimum from which the mode ride started.
    pub origin: usize,
    /// Exact-witness endpoint minimum indices, positive then negative branch.
    pub endpoints: [usize; 2],
    /// Receiving-side saddle energy.
    pub saddle_energy: f64,
    /// Saddle point in the native coordinates of the optimization problem.
    pub saddle_coordinates: Array1<f64>,
    /// Lowest certified Hessian eigenvalue.
    pub curvature: f64,
    /// Receiving-side lowest Hessian eigenvector.
    pub lowest_mode: Array1<f64>,
    /// Number of certified negative Hessian eigenvalues.
    pub negative_modes: usize,
    /// Saddle gradient infinity norm.
    pub saddle_max_gradient: f64,
    /// Lowest-mode solver used for this connection.
    pub ride_method: RideMethod,
}

/// Exact stationary-point graph for a single arbitrary-dimensional surface.
///
/// Point coordinates are used only to order exact-witness checks. The graph
/// has no atom, species, Cartesian-geometry, or universal-descriptor contract,
/// and it is never comparable with a graph belonging to another surface.
#[derive(Debug, Clone, Default)]
pub struct NdPesNetwork {
    minima: Vec<NdMinimumRecord>,
    saddles: Vec<NdSaddleConnection>,
}

impl NdPesNetwork {
    /// Empty stationary-point graph for one point surface.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of exact-witness minimum identities.
    pub fn minimum_count(&self) -> usize {
        self.minima.len()
    }

    /// Number of exact-witness saddle identities.
    pub fn saddle_count(&self) -> usize {
        self.saddles.len()
    }

    /// Stored minima in stable admission order.
    pub fn minima(&self) -> &[NdMinimumRecord] {
        &self.minima
    }

    /// Stored saddle connections in stable admission order.
    pub fn saddles(&self) -> &[NdSaddleConnection] {
        &self.saddles
    }

    fn admit_minimum<W: ExactStructureWitness + ?Sized>(
        &mut self,
        minimum: Quenched,
        witness: &W,
    ) -> usize {
        let mut ordered = self
            .minima
            .iter()
            .enumerate()
            .map(|(index, existing)| {
                let distance = existing
                    .coordinates
                    .iter()
                    .zip(&minimum.coordinates)
                    .map(|(left, right)| (left - right).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (distance, index)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        for (_, index) in ordered {
            if witness.equivalent(
                self.minima[index].coordinates.view(),
                minimum.coordinates.view(),
            ) {
                return index;
            }
        }
        let id = self.minima.len();
        self.minima.push(NdMinimumRecord {
            id,
            energy: minimum.energy,
            coordinates: minimum.coordinates,
            max_gradient: minimum.max_gradient,
        });
        id
    }

    fn admit_saddle<W: ExactStructureWitness + ?Sized>(
        &mut self,
        mut candidate: NdSaddleConnection,
        witness: &W,
    ) -> NdSaddleConnection {
        for existing in &self.saddles {
            let same_endpoints = existing.endpoints == candidate.endpoints
                || existing.endpoints == [candidate.endpoints[1], candidate.endpoints[0]];
            if same_endpoints
                && witness.equivalent(
                    existing.saddle_coordinates.view(),
                    candidate.saddle_coordinates.view(),
                )
            {
                return existing.clone();
            }
        }
        candidate.id = self.saddles.len();
        self.saddles.push(candidate.clone());
        candidate
    }
}

struct SaddleSurface<'a, S>(&'a S);

impl<S> PointSurface for SaddleSurface<'_, S>
where
    S: PesSurface,
{
    fn eval(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        checked_evaluate(self.0, coordinates)
            .map_err(|error| SaddleError::Surface(error.to_string()))
    }
}

#[derive(Debug, Clone)]
struct Quenched {
    energy: f64,
    coordinates: Array1<f64>,
    max_gradient: f64,
}

fn checked_evaluate<S: PesSurface + ?Sized>(
    surface: &S,
    coordinates: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>), PesExplorationError> {
    let (energy, gradient) = surface
        .evaluate(coordinates)
        .map_err(|error| PesExplorationError::Surface(error.to_string()))?;
    if !energy.is_finite() {
        return Err(PesExplorationError::InvalidEvaluation("a nonfinite energy"));
    }
    if gradient.len() != coordinates.len() {
        return Err(PesExplorationError::InvalidEvaluation(
            "a gradient with the wrong dimension",
        ));
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(PesExplorationError::InvalidEvaluation(
            "a nonfinite gradient",
        ));
    }
    Ok((energy, gradient))
}

fn max_abs(values: ArrayView1<f64>) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

/// Dense receiving-side Hessian eigensystem and stationary-point index.
#[derive(Debug, Clone)]
pub struct StationaryIndex {
    /// Hessian eigenvalues in ascending order.
    pub eigenvalues: Array1<f64>,
    /// Normalized eigenvector belonging to the lowest eigenvalue.
    pub lowest_mode: Array1<f64>,
    /// Number of eigenvalues below `-negative_tolerance`.
    pub negative_modes: usize,
}

fn finite_difference_hessian<S: PesSurface + ?Sized>(
    surface: &S,
    coordinates: ArrayView1<f64>,
    step: f64,
) -> Result<Array2<f64>, PesExplorationError> {
    if coordinates.is_empty() {
        return Err(PesExplorationError::InvalidShape(
            "stationary coordinates must be nonempty",
        ));
    }
    if coordinates.iter().any(|value| !value.is_finite()) {
        return Err(PesExplorationError::InvalidShape(
            "stationary coordinates are nonfinite",
        ));
    }
    if !step.is_finite() || step <= 0.0 {
        return Err(PesExplorationError::InvalidConfig(
            "Hessian finite-difference step",
        ));
    }

    let dimension = coordinates.len();
    let mut hessian = Array2::zeros((dimension, dimension));
    for column in 0..dimension {
        let mut plus = coordinates.to_owned();
        let mut minus = coordinates.to_owned();
        plus[column] += step;
        minus[column] -= step;
        let (_, plus_gradient) = checked_evaluate(surface, plus.view())?;
        let (_, minus_gradient) = checked_evaluate(surface, minus.view())?;
        for row in 0..dimension {
            hessian[(row, column)] = (plus_gradient[row] - minus_gradient[row]) / (2.0 * step);
        }
    }
    for row in 0..dimension {
        for column in 0..row {
            let symmetric = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            hessian[(row, column)] = symmetric;
            hessian[(column, row)] = symmetric;
        }
    }
    Ok(hessian)
}

fn index_hessian(
    hessian: Array2<f64>,
    negative_tolerance: f64,
) -> Result<StationaryIndex, PesExplorationError> {
    if !negative_tolerance.is_finite() || negative_tolerance < 0.0 {
        return Err(PesExplorationError::InvalidConfig(
            "negative curvature tolerance",
        ));
    }

    let (eigenvalues, eigenvectors) = exact_eigh(hessian.view())
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let mut order = (0..eigenvalues.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| eigenvalues[left].total_cmp(&eigenvalues[right]));
    let eigenvalues = Array1::from_iter(order.iter().map(|&index| eigenvalues[index]));
    let lowest_mode = match order.first().copied() {
        Some(index) => eigenvectors.column(index).to_owned(),
        None => {
            return Err(PesExplorationError::InvalidShape(
                "stationary Hessian has no eigenvalues",
            ));
        }
    };
    let negative_modes = eigenvalues
        .iter()
        .filter(|eigenvalue| **eigenvalue < -negative_tolerance)
        .count();
    Ok(StationaryIndex {
        lowest_mode,
        eigenvalues,
        negative_modes,
    })
}

/// Certify a stationary-point index from central differences of PES gradients.
///
/// This arbitrary-dimensional form counts every native coordinate. Atomistic
/// callers with rigid or frozen directions use [`stationary_index_cartesian`].
pub fn stationary_index<S: PesSurface + ?Sized>(
    surface: &S,
    coordinates: ArrayView1<f64>,
    step: f64,
    negative_tolerance: f64,
) -> Result<StationaryIndex, PesExplorationError> {
    index_hessian(
        finite_difference_hessian(surface, coordinates, step)?,
        negative_tolerance,
    )
}

/// Certify the index on the free Cartesian tangent space of one atomic system.
///
/// Frozen coordinates are removed exactly. An unconstrained finite system also
/// removes translation and rotation, while an unconstrained periodic system
/// removes translation. The projected directions remain as numerical zero
/// eigenvalues and therefore cannot masquerade as unstable physical modes.
pub fn stationary_index_cartesian<S: PesSurface + ?Sized>(
    surface: &S,
    coordinates: ArrayView1<f64>,
    frozen_atoms: &[bool],
    periodic: [bool; 3],
    step: f64,
    negative_tolerance: f64,
) -> Result<StationaryIndex, PesExplorationError> {
    if !coordinates.len().is_multiple_of(3) {
        return Err(PesExplorationError::InvalidShape(
            "Cartesian coordinates must have dimension 3N",
        ));
    }
    let atom_count = coordinates.len() / 3;
    if frozen_atoms.len() != atom_count {
        return Err(PesExplorationError::InvalidShape(
            "frozen mask must contain one value per atom",
        ));
    }

    let mut constraints = Vec::new();
    for (atom, frozen) in frozen_atoms.iter().copied().enumerate() {
        if frozen {
            for axis in 0..3 {
                let mut coordinate = Array1::zeros(coordinates.len());
                coordinate[3 * atom + axis] = 1.0;
                constraints.push(coordinate);
            }
        }
    }
    if constraints.is_empty() {
        constraints = if periodic.iter().any(|axis| *axis) {
            translation_basis(atom_count)
        } else {
            rigid_basis(coordinates)
        };
    }

    let mut hessian = finite_difference_hessian(surface, coordinates, step)?;
    for column in 0..hessian.ncols() {
        let mut projected = hessian.column(column).to_owned();
        project_rigid_with(&mut projected, &constraints);
        hessian.column_mut(column).assign(&projected);
    }
    for row in 0..hessian.nrows() {
        let mut projected = hessian.row(row).to_owned();
        project_rigid_with(&mut projected, &constraints);
        hessian.row_mut(row).assign(&projected);
    }
    for row in 0..hessian.nrows() {
        for column in 0..row {
            let symmetric = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            hessian[(row, column)] = symmetric;
            hessian[(column, row)] = symmetric;
        }
    }
    index_hessian(hessian, negative_tolerance)
}

fn quench<S: PesSurface + ?Sized>(
    surface: &S,
    start: ArrayView1<f64>,
    config: &PesExplorationConfig,
) -> Result<Quenched, PesExplorationError> {
    let mut optimizer = Lbfgs::default();
    optimizer.gtol = config.quench_gradient_tolerance;
    optimizer.norm = GradNorm::Infinity;
    let mut failure = None;
    let (_, coordinates, _) = optimizer.minimize(start, config.quench_steps, |trial| {
        match checked_evaluate(surface, trial) {
            Ok(value_gradient) => Some(value_gradient),
            Err(error) => {
                if failure.is_none() {
                    failure = Some(error);
                }
                None
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let (energy, gradient) = checked_evaluate(surface, coordinates.view())?;
    let max_gradient = max_abs(gradient.view());
    if max_gradient >= config.quench_gradient_tolerance {
        return Err(PesExplorationError::QuenchNotConverged { max_gradient });
    }
    Ok(Quenched {
        energy,
        coordinates,
        max_gradient,
    })
}

fn descriptor(
    descriptor_space: &DescriptorSpace,
    stationary: &Quenched,
    species: Option<&[u32]>,
) -> Result<DescriptorVector, PesExplorationError> {
    Ok(descriptor_space.describe(stationary.coordinates.view(), species)?)
}

fn min_mode_config(config: &PesExplorationConfig) -> MinModeConfig {
    MinModeConfig {
        kind: config.ride_method.min_mode(),
        force_tol: config.saddle_force_tolerance,
        max_move: config.maximum_move,
        ..MinModeConfig::default()
    }
}

fn irc_config(config: &PesExplorationConfig) -> IrcConfig {
    IrcConfig {
        dx: config.irc_step,
        force_tol: config.irc_force_tolerance,
        max_move: config.maximum_move,
        ..IrcConfig::default()
    }
}

fn normalize_mode(mode: ArrayView1<f64>) -> Result<Array1<f64>, PesExplorationError> {
    if mode.iter().any(|value| !value.is_finite()) {
        return Err(PesExplorationError::InvalidShape("mode is nonfinite"));
    }
    let norm = mode.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= 1e-14 {
        return Err(PesExplorationError::InvalidShape("mode has zero norm"));
    }
    Ok(mode.mapv(|value| value / norm))
}

fn directional_curvature<S: PesSurface + ?Sized>(
    surface: &S,
    coordinates: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    step: f64,
) -> Result<f64, PesExplorationError> {
    let mode = normalize_mode(mode)?;
    let plus = &coordinates + &(&mode * step);
    let minus = &coordinates - &(&mode * step);
    let (_, plus_gradient) = checked_evaluate(surface, plus.view())?;
    let (_, minus_gradient) = checked_evaluate(surface, minus.view())?;
    let hessian_mode = (&plus_gradient - &minus_gradient) / (2.0 * step);
    let curvature = mode.dot(&hessian_mode);
    if !curvature.is_finite() {
        return Err(PesExplorationError::InvalidEvaluation(
            "a nonfinite directional curvature",
        ));
    }
    Ok(curvature)
}

fn bowl_breakout<S: PesSurface + ?Sized>(
    surface: &S,
    origin: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    config: &PesExplorationConfig,
) -> Result<Array1<f64>, PesExplorationError> {
    let mode = normalize_mode(mode)?;
    let mut radius = config.saddle_displacement;
    let mut lowest_curvature = f64::INFINITY;
    for _ in 0..config.activation_attempts {
        let trial = &origin + &(&mode * radius);
        let curvature =
            directional_curvature(surface, trial.view(), mode.view(), config.hessian_step)?;
        lowest_curvature = lowest_curvature.min(curvature);
        if curvature < -config.negative_curvature_tolerance {
            return Ok(trial);
        }
        radius *= config.activation_growth;
        if !radius.is_finite() {
            return Err(PesExplorationError::InvalidEvaluation(
                "a nonfinite activation radius",
            ));
        }
    }
    Err(PesExplorationError::ActivationNotEscaped { lowest_curvature })
}

fn roll_branch<S: PesSurface>(
    surface: &S,
    saddle: &Array1<f64>,
    masses: &Array1<f64>,
    mode: &Array1<f64>,
    direction: IrcDirection,
    config: &PesExplorationConfig,
) -> Result<(Quenched, bool), PesExplorationError> {
    let adapter = SaddleSurface(surface);
    let mut session = IrcSession::from_surface(
        irc_config(config),
        saddle.clone(),
        masses.clone(),
        mode.clone(),
        direction,
        &adapter,
    )
    .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let report = session
        .run(&adapter, config.irc_steps)
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let endpoint = session.position().to_owned();
    Ok((quench(surface, endpoint.view(), config)?, report.at_minimum))
}

/// Quench a point, ride one supplied mode to an index-one saddle, and quench
/// both downhill branches on an arbitrary-dimensional surface.
///
/// This path deliberately has no atomistic descriptor, mass, species, or 3N
/// requirement. The two branches use small signed displacements along the
/// receiving-side unstable eigenvector followed by rgmin certification. A
/// molecular caller needing a mass-weighted IRC uses
/// [`discover_mode_connection`] instead.
pub fn discover_nd_connection<S, W>(
    surface: &S,
    network: &mut NdPesNetwork,
    start: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    config: &PesExplorationConfig,
    witness: &W,
) -> Result<NdSaddleConnection, PesExplorationError>
where
    S: PesSurface,
    W: ExactStructureWitness + ?Sized,
{
    config.validate()?;
    if start.is_empty() {
        return Err(PesExplorationError::InvalidShape(
            "point coordinates must be nonempty",
        ));
    }
    if mode.len() != start.len() {
        return Err(PesExplorationError::InvalidShape(
            "mode must match the point dimension",
        ));
    }

    let origin_minimum = quench(surface, start, config)?;
    let origin = network.admit_minimum(origin_minimum.clone(), witness);
    let mode = normalize_mode(mode)?;
    let saddle_start = bowl_breakout(
        surface,
        origin_minimum.coordinates.view(),
        mode.view(),
        config,
    )?;
    let adapter = SaddleSurface(surface);
    let mut min_mode = MinModeSession::new(min_mode_config(config), saddle_start, mode)
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let min_mode_report = min_mode
        .run(&adapter, config.saddle_steps)
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    if min_mode_report.status != MinModeStatus::Converged {
        return Err(PesExplorationError::SaddleNotConverged {
            stage: "minimum-mode ride",
        });
    }
    if min_mode_report.curvature >= -config.negative_curvature_tolerance {
        return Err(PesExplorationError::MinimumModeLostCurvature {
            curvature: min_mode_report.curvature,
        });
    }

    let saddle_coordinates = min_mode.position().to_owned();
    let (saddle_energy, saddle_gradient) = checked_evaluate(surface, saddle_coordinates.view())?;
    let saddle_max_gradient = max_abs(saddle_gradient.view());
    if saddle_max_gradient > config.saddle_force_tolerance {
        return Err(PesExplorationError::SaddleNotConverged {
            stage: "receiving-side force certification",
        });
    }
    let index = stationary_index(
        surface,
        saddle_coordinates.view(),
        config.hessian_step,
        config.negative_curvature_tolerance,
    )?;
    let curvature = index.eigenvalues[0];
    if index.negative_modes != 1 {
        return Err(PesExplorationError::NotFirstOrder {
            negative_modes: index.negative_modes,
            lowest_curvature: curvature,
        });
    }
    let lowest_mode = index.lowest_mode;
    let positive_start = &saddle_coordinates + &(&lowest_mode * config.irc_step);
    let negative_start = &saddle_coordinates - &(&lowest_mode * config.irc_step);
    let positive = quench(surface, positive_start.view(), config)?;
    let negative = quench(surface, negative_start.view(), config)?;
    let positive_id = network.admit_minimum(positive, witness);
    let negative_id = network.admit_minimum(negative, witness);
    if positive_id == negative_id {
        return Err(PesExplorationError::CollapsedConnection);
    }
    if positive_id != origin && negative_id != origin {
        return Err(PesExplorationError::DisconnectedConnection);
    }

    Ok(network.admit_saddle(
        NdSaddleConnection {
            id: usize::MAX,
            origin,
            endpoints: [positive_id, negative_id],
            saddle_energy,
            saddle_coordinates,
            curvature,
            lowest_mode,
            negative_modes: index.negative_modes,
            saddle_max_gradient,
            ride_method: config.ride_method,
        },
        witness,
    ))
}

/// Quench a basin, ride one supplied mode, roll both IRC branches, and admit
/// the resulting exact minimum and saddle identities.
#[allow(clippy::too_many_arguments)]
pub fn discover_mode_connection<S, W>(
    surface: &S,
    descriptor_space: &DescriptorSpace,
    network: &mut PesNetwork,
    start: ArrayView1<f64>,
    masses: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    species: Option<&[u32]>,
    config: &PesExplorationConfig,
    witness: &W,
) -> Result<SaddleConnection, PesExplorationError>
where
    S: PesSurface,
    W: ExactStructureWitness + ?Sized,
{
    discover_mode_connection_impl(
        surface,
        descriptor_space,
        network,
        start,
        masses,
        mode,
        species,
        config,
        None,
        witness,
    )
}

/// Discover an atomistic connection and certify its index only on free modes.
///
/// The descriptor geometry determines whether rotations are rigid symmetries;
/// `frozen_atoms` removes externally fixed coordinates. This is the production
/// path for finite clusters, molecules, and periodic surfaces. The native
/// [`discover_mode_connection`] path remains available for Cartesian-shaped
/// mathematical surfaces whose coordinates do not obey atomistic symmetries.
#[allow(clippy::too_many_arguments)]
pub fn discover_cartesian_mode_connection<S, W>(
    surface: &S,
    descriptor_space: &DescriptorSpace,
    network: &mut PesNetwork,
    start: ArrayView1<f64>,
    masses: ArrayView1<f64>,
    frozen_atoms: &[bool],
    mode: ArrayView1<f64>,
    species: Option<&[u32]>,
    config: &PesExplorationConfig,
    witness: &W,
) -> Result<SaddleConnection, PesExplorationError>
where
    S: PesSurface,
    W: ExactStructureWitness + ?Sized,
{
    let periodic = descriptor_space
        .geometry()
        .ok_or(PesExplorationError::InvalidConfig(
            "atomistic descriptor geometry",
        ))?
        .periodic();
    discover_mode_connection_impl(
        surface,
        descriptor_space,
        network,
        start,
        masses,
        mode,
        species,
        config,
        Some((frozen_atoms, periodic)),
        witness,
    )
}

#[allow(clippy::too_many_arguments)]
fn discover_mode_connection_impl<S, W>(
    surface: &S,
    descriptor_space: &DescriptorSpace,
    network: &mut PesNetwork,
    start: ArrayView1<f64>,
    masses: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    species: Option<&[u32]>,
    config: &PesExplorationConfig,
    cartesian_index: Option<(&[bool], [bool; 3])>,
    witness: &W,
) -> Result<SaddleConnection, PesExplorationError>
where
    S: PesSurface,
    W: ExactStructureWitness + ?Sized,
{
    config.validate()?;
    if start.is_empty() || !start.len().is_multiple_of(3) {
        return Err(PesExplorationError::InvalidShape(
            "coordinates must be nonempty 3N Cartesian",
        ));
    }
    if masses.len() * 3 != start.len() {
        return Err(PesExplorationError::InvalidShape(
            "masses must contain one value per atom",
        ));
    }
    if mode.len() != start.len() {
        return Err(PesExplorationError::InvalidShape(
            "mode must match the Cartesian dimension",
        ));
    }
    if species.is_some_and(|species| species.len() * 3 != start.len()) {
        return Err(PesExplorationError::InvalidShape(
            "species must contain one value per atom",
        ));
    }
    if cartesian_index.is_some_and(|(frozen_atoms, _)| frozen_atoms.len() * 3 != start.len()) {
        return Err(PesExplorationError::InvalidShape(
            "frozen mask must contain one value per atom",
        ));
    }

    let origin_minimum = quench(surface, start, config)?;
    let context = StructureContext::new(
        species.map(<[u32]>::to_vec),
        descriptor_space.geometry(),
        None,
    )
    .with_masses(Some(masses.to_vec()));
    let origin_descriptor = descriptor(descriptor_space, &origin_minimum, context.species())?;
    let origin = network.admit_minimum_with_context(
        origin_minimum.energy,
        origin_minimum.coordinates.clone(),
        origin_minimum.max_gradient,
        origin_descriptor,
        context.clone(),
        witness,
    )?;

    let mode = normalize_mode(mode)?;
    let saddle_start = bowl_breakout(
        surface,
        origin_minimum.coordinates.view(),
        mode.view(),
        config,
    )?;
    let adapter = SaddleSurface(surface);
    let mut min_mode = MinModeSession::new(min_mode_config(config), saddle_start, mode.clone())
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let min_mode_report = min_mode
        .run(&adapter, config.saddle_steps)
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    if min_mode_report.status != MinModeStatus::Converged {
        return Err(PesExplorationError::SaddleNotConverged {
            stage: "minimum-mode ride",
        });
    }
    if min_mode_report.curvature >= -config.negative_curvature_tolerance {
        return Err(PesExplorationError::MinimumModeLostCurvature {
            curvature: min_mode_report.curvature,
        });
    }
    let mut saddle_coordinates = min_mode.position().to_owned();

    if config.refine_with_prfo {
        let prfo_config = SellaSaddleConfig {
            force_tol: config.saddle_force_tolerance,
            ..SellaSaddleConfig::default()
        };
        let mut prfo =
            SellaSaddleSession::new(prfo_config, saddle_coordinates.clone(), masses.to_owned())
                .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
        let report = prfo
            .run(&adapter, config.prfo_steps)
            .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
        if !report.at_saddle {
            return Err(PesExplorationError::SaddleNotConverged {
                stage: "Sella P-RFO refinement",
            });
        }
        saddle_coordinates = prfo.position().to_owned();
    }

    let (saddle_energy, saddle_gradient) = checked_evaluate(surface, saddle_coordinates.view())?;
    let saddle_max_gradient = max_abs(saddle_gradient.view());
    if saddle_max_gradient > config.saddle_force_tolerance {
        return Err(PesExplorationError::SaddleNotConverged {
            stage: "receiving-side force certification",
        });
    }
    let index = match cartesian_index {
        Some((frozen_atoms, periodic)) => stationary_index_cartesian(
            surface,
            saddle_coordinates.view(),
            frozen_atoms,
            periodic,
            config.hessian_step,
            config.negative_curvature_tolerance,
        )?,
        None => stationary_index(
            surface,
            saddle_coordinates.view(),
            config.hessian_step,
            config.negative_curvature_tolerance,
        )?,
    };
    let curvature = index.eigenvalues[0];
    if index.negative_modes != 1 {
        return Err(PesExplorationError::NotFirstOrder {
            negative_modes: index.negative_modes,
            lowest_curvature: curvature,
        });
    }
    let saddle_mode = index.lowest_mode;

    let (forward, forward_irc) = roll_branch(
        surface,
        &saddle_coordinates,
        &masses.to_owned(),
        &saddle_mode,
        IrcDirection::Forward,
        config,
    )?;
    let (reverse, reverse_irc) = roll_branch(
        surface,
        &saddle_coordinates,
        &masses.to_owned(),
        &saddle_mode,
        IrcDirection::Reverse,
        config,
    )?;
    let forward_descriptor = descriptor(descriptor_space, &forward, context.species())?;
    let reverse_descriptor = descriptor(descriptor_space, &reverse, context.species())?;
    let forward_id = network.admit_minimum_with_context(
        forward.energy,
        forward.coordinates,
        forward.max_gradient,
        forward_descriptor,
        context.clone(),
        witness,
    )?;
    let reverse_id = network.admit_minimum_with_context(
        reverse.energy,
        reverse.coordinates,
        reverse.max_gradient,
        reverse_descriptor,
        context.clone(),
        witness,
    )?;
    if forward_id.id == reverse_id.id {
        return Err(PesExplorationError::CollapsedConnection);
    }
    if forward_id.id != origin.id && reverse_id.id != origin.id {
        return Err(PesExplorationError::DisconnectedConnection);
    }

    let saddle_descriptor =
        descriptor_space.describe(saddle_coordinates.view(), context.species())?;
    network
        .admit_saddle(
            SaddleConnection {
                id: usize::MAX,
                origin: origin.id,
                endpoints: [forward_id.id, reverse_id.id],
                saddle_energy,
                saddle_coordinates,
                context,
                curvature,
                lowest_mode: saddle_mode,
                negative_modes: index.negative_modes,
                saddle_max_gradient,
                descriptor: saddle_descriptor,
                ride_method: config.ride_method,
                irc_at_minimum: [forward_irc, reverse_irc],
            },
            witness,
        )
        .map_err(PesExplorationError::from)
}
