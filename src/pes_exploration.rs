//! Minimum--saddle--minimum exploration with rgmin and rgsaddle.
//!
//! The descriptor orders structural comparisons but never certifies identity.
//! A caller-supplied exact witness makes the final equivalence decision. This
//! keeps catalog admission independent of a fixed descriptor radius while the
//! same universal descriptor remains available for novelty and acquisition.

use std::fmt::{Display, Formatter};
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rgmin::{
    ApplyHessian, Control, EigenParams, EigensolverKind, FireKind, GradNorm, Lbfgs, Method, Oracle,
    Solver, lowest_mode,
};
use rgsaddle::geom::update_trust;
use rgsaddle::{
    HessUpdate, IrcConfig, IrcDirection, IrcSession, MinModeConfig, MinModeKind, MinModeSession,
    MinModeStatus, PointSurface, SaddleError, TrustSchedule, exact_eigh, prfo_trust_region,
    update_h,
};

use crate::curvature::{project_rigid_with, rigid_basis};
use crate::descriptor_space::{
    DescriptorError, DescriptorGeometry, DescriptorSpace, DescriptorVector,
};

/// Minimum ART downhill displacement as a fraction of source--saddle distance.
const ART_BRANCH_FRACTION: f64 = 0.15;
/// Lowest root whose overlap reaches this fraction of the best candidate.
const MODE_HOMING_OVERLAP_FRACTION: f64 = 0.7;
/// A transition-state secant model is rebuilt when its measured step has
/// opposite sign to the predicted energy change.
const PRFO_MODEL_MIN_RHO: f64 = 0.0;
/// A nearly orthogonal homed root is not reliable enough to transport.
const PRFO_MODEL_MIN_OVERLAP: f64 = 0.2;

/// Energy and Cartesian-gradient evaluator used by all exploration stages.
pub trait PesSurface: Sync {
    /// Surface-specific evaluation failure.
    type Error: Display;

    /// Return energy and a gradient matching the Cartesian input dimension.
    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error>;
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ndarray::{Array1, Array2, ArrayView1, array};

    use super::{
        IrcDirection, PesExplorationConfig, PesSurface, activation_basis, deflate_cartesian_mode,
        finest_irc_step, home_uphill_mode, refine_cartesian_with_prfo, roll_branch,
    };
    use crate::curvature::project_rigid_with;
    use crate::descriptor_space::DescriptorGeometry;

    struct FailingIrcSurface;

    struct CountingQuarticSaddle {
        evaluations: AtomicUsize,
    }

    impl PesSurface for CountingQuarticSaddle {
        type Error = &'static str;

        fn evaluate(
            &self,
            coordinates: ArrayView1<f64>,
        ) -> Result<(f64, Array1<f64>), Self::Error> {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
            let reaction = coordinates[0];
            let mut energy = (reaction * reaction - 1.0).powi(2);
            let mut gradient = Array1::zeros(coordinates.len());
            gradient[0] = 4.0 * reaction * (reaction * reaction - 1.0);
            for index in 1..coordinates.len() {
                let stiffness = 1.0 + index as f64 / coordinates.len() as f64;
                energy += 0.5 * stiffness * coordinates[index] * coordinates[index];
                gradient[index] = stiffness * coordinates[index];
            }
            Ok((energy, gradient))
        }
    }

    impl PesSurface for FailingIrcSurface {
        type Error = &'static str;

        fn evaluate(
            &self,
            coordinates: ArrayView1<f64>,
        ) -> Result<(f64, Array1<f64>), Self::Error> {
            if coordinates[0] > 0.05 {
                return Err("branch left the valid domain");
            }
            let mut energy = -0.5 * coordinates[0] * coordinates[0];
            let mut gradient = Array1::zeros(coordinates.len());
            gradient[0] = -coordinates[0];
            for index in 1..coordinates.len() {
                energy += 0.5 * coordinates[index] * coordinates[index];
                gradient[index] = coordinates[index];
            }
            Ok((energy, gradient))
        }
    }

    #[test]
    fn mode_homing_stays_in_the_negative_subspace_after_instability_appears() {
        let eigenvalues = array![-2.0, 0.4, 3.0];
        let eigenvectors = Array2::eye(3);
        let reference = array![0.2, 0.0, -0.98];

        let homed = home_uphill_mode(&eigenvalues, &eigenvectors, reference.view(), 1e-6).unwrap();

        assert_eq!(homed.source_index, 0);
        assert!(homed.eigenvalues[0] < 0.0);
        assert!(homed.eigenvectors.column(0).dot(&reference) > 0.0);
    }

    #[test]
    fn fuzzy_mode_homing_prefers_the_lower_overlapping_root_before_instability() {
        let eigenvalues = array![0.1, 0.2, 3.0];
        let eigenvectors = Array2::eye(3);
        let reference = array![0.65, 0.0, -0.76];

        let homed = home_uphill_mode(&eigenvalues, &eigenvectors, reference.view(), 1e-6).unwrap();

        assert_eq!(homed.source_index, 0);
        assert_eq!(homed.eigenvalues, array![0.1, 0.2, 3.0]);
        assert!(homed.eigenvectors.column(0).dot(&reference) > 0.0);
    }

    #[test]
    fn cartesian_prfo_reuses_secant_hessian_between_exact_refreshes() {
        let surface = CountingQuarticSaddle {
            evaluations: AtomicUsize::new(0),
        };
        let mut start = Array1::from_elem(18, 0.08);
        start[0] = 0.6;
        let mut mode = Array1::zeros(18);
        mode[0] = 1.0;
        let config = PesExplorationConfig {
            prfo_steps: 80,
            saddle_force_tolerance: 1e-8,
            saddle_displacement: 0.08,
            maximum_move: 0.08,
            hessian_step: 1e-5,
            ..PesExplorationConfig::default()
        };

        let saddle =
            refine_cartesian_with_prfo(&surface, start.view(), mode.view(), None, &config).unwrap();

        assert!(saddle.iter().all(|coordinate| coordinate.abs() < 1e-7));
        assert!(surface.evaluations.load(Ordering::Relaxed) < 100);
    }

    #[test]
    fn known_saddle_deflation_removes_the_previous_escape_direction() {
        let source = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let avoided = array![0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mode = array![0.8, 0.3, 0.0, -0.4, -0.2, 0.0, -0.4, -0.1, 0.0];
        let deflated = deflate_cartesian_mode(
            source.view(),
            mode.view(),
            &[avoided.to_vec()],
            &[false; 3],
            DescriptorGeometry::finite(1.0).unwrap(),
        )
        .unwrap();
        let mut escape = &avoided - &source;
        project_rigid_with(&mut escape, &activation_basis(source.view(), None));
        escape /= escape.dot(&escape).sqrt();

        assert!(deflated.dot(&escape).abs() < 1e-10);
        assert!((deflated.dot(&deflated) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn irc_surface_failure_identifies_the_branch_and_outer_step() {
        let error = roll_branch(
            &FailingIrcSurface,
            &Array1::zeros(6),
            &Array1::ones(2),
            &array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            IrcDirection::Forward,
            0.1,
            &PesExplorationConfig::default(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("IRC Forward step 0"));
    }

    #[test]
    fn molecular_irc_starts_at_the_finest_configured_resolution() {
        let config = PesExplorationConfig {
            branch_attempts: 4,
            branch_growth: 2.5,
            ..PesExplorationConfig::default()
        };

        let refined = finest_irc_step(0.2, &config).unwrap();

        assert!((refined - 0.0128).abs() < 1e-15);
    }
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

/// Generate one direction from a seeded orthonormal basis block in arbitrary N.
///
/// Ranks `0..dimension` cover one Gaussian-distributed orthonormal basis before
/// the next independently seeded block begins. This preserves isotropy while
/// preventing a finite ride portfolio from repeatedly sampling nearly parallel
/// directions. The direction sign is applied after orthogonalization.
pub fn orthonormal_nd_mode(
    dimension: usize,
    seed: u64,
    rank: u16,
    direction: RideModeDirection,
) -> Result<Array1<f64>, PesExplorationError> {
    if dimension == 0 {
        return Err(PesExplorationError::InvalidShape(
            "ride dimension must be nonempty",
        ));
    }
    let rank = usize::from(rank);
    let block = rank / dimension;
    let selected = rank % dimension;
    let block_rank = u16::try_from(block).map_err(|_| {
        PesExplorationError::InvalidShape("ride mode block exceeds the supported rank")
    })?;
    let mut rng = StdRng::seed_from_u64(ranked_seed(seed ^ 0xa076_1d64_78bd_642f, block_rank));
    let mut basis = Vec::<Array1<f64>>::with_capacity(selected + 1);
    for basis_index in 0..=selected {
        let mut candidate = Array1::from_shape_simple_fn(dimension, || {
            let sample: f64 = StandardNormal.sample(&mut rng);
            sample
        });
        for _ in 0..2 {
            for previous in &basis {
                let projection = candidate.dot(previous);
                candidate.scaled_add(-projection, previous);
            }
        }
        let norm = candidate.dot(&candidate).sqrt();
        if !norm.is_finite() || norm <= 1e-12 {
            candidate.fill(0.0);
            candidate[(basis_index + block) % dimension] = 1.0;
            for previous in &basis {
                let projection = candidate.dot(previous);
                candidate.scaled_add(-projection, previous);
            }
        }
        basis.push(normalize_mode(candidate.view())?);
    }
    Ok(basis.pop().expect("selected basis direction exists") * direction.sign())
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

/// Remove escape directions leading to already certified saddles.
///
/// Every avoided saddle is expressed in the same labelled Cartesian frame as
/// `coordinates`. Minimum-image displacements are projected onto the live
/// free-coordinate tangent space and orthonormalized before they are removed
/// from `mode`. If the supplied mode lies entirely in the known subspace, the
/// largest remaining canonical tangent direction provides a deterministic
/// replacement.
pub fn deflate_cartesian_mode(
    coordinates: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    avoided_saddles: &[Vec<f64>],
    frozen_atoms: &[bool],
    geometry: DescriptorGeometry,
) -> Result<Array1<f64>, PesExplorationError> {
    if coordinates.is_empty()
        || !coordinates.len().is_multiple_of(3)
        || mode.len() != coordinates.len()
        || frozen_atoms.len() * 3 != coordinates.len()
    {
        return Err(PesExplorationError::InvalidShape(
            "known-saddle deflation needs matching 3N coordinates",
        ));
    }
    let constraints = activation_basis(coordinates, Some((frozen_atoms, geometry.periodic())));
    let mut deflated = mode.to_owned();
    project_rigid_with(&mut deflated, &constraints);
    let mut known_directions = Vec::<Array1<f64>>::new();
    for saddle in avoided_saddles {
        if saddle.len() != coordinates.len() || saddle.iter().any(|value| !value.is_finite()) {
            return Err(PesExplorationError::InvalidShape(
                "known saddle must match the source coordinates",
            ));
        }
        let mut direction = Array1::zeros(coordinates.len());
        for atom in 0..frozen_atoms.len() {
            let displacement = geometry.displacement([
                saddle[3 * atom] - coordinates[3 * atom],
                saddle[3 * atom + 1] - coordinates[3 * atom + 1],
                saddle[3 * atom + 2] - coordinates[3 * atom + 2],
            ]);
            for axis in 0..3 {
                direction[3 * atom + axis] = displacement[axis];
            }
        }
        project_rigid_with(&mut direction, &constraints);
        for known in &known_directions {
            direction -= &(known * direction.dot(known));
        }
        let norm = direction.dot(&direction).sqrt();
        if norm > 1e-10 {
            known_directions.push(direction / norm);
        }
    }
    for known in &known_directions {
        deflated -= &(known * deflated.dot(known));
    }
    let mut norm = deflated.dot(&deflated).sqrt();
    if norm <= 1e-10 {
        let mut replacement = None::<(f64, Array1<f64>)>;
        for coordinate in 0..coordinates.len() {
            let mut candidate = Array1::zeros(coordinates.len());
            candidate[coordinate] = 1.0;
            project_rigid_with(&mut candidate, &constraints);
            for known in &known_directions {
                candidate -= &(known * candidate.dot(known));
            }
            let candidate_norm = candidate.dot(&candidate).sqrt();
            if replacement
                .as_ref()
                .is_none_or(|(best_norm, _)| candidate_norm > *best_norm)
            {
                replacement = Some((candidate_norm, candidate));
            }
        }
        let Some((replacement_norm, candidate)) = replacement else {
            return Err(PesExplorationError::InvalidShape(
                "known saddles span every free Cartesian direction",
            ));
        };
        if replacement_norm <= 1e-10 {
            return Err(PesExplorationError::InvalidShape(
                "known saddles span every free Cartesian direction",
            ));
        }
        deflated = candidate;
        norm = replacement_norm;
    }
    Ok(deflated / norm)
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
    /// Force threshold for handing minimum-mode following to P-RFO.
    pub minimum_mode_force_tolerance: f64,
    /// Maximum rgsaddle IRC outer points per direction.
    pub irc_steps: usize,
    /// Maximum Sella P-RFO refinement steps.
    pub prfo_steps: usize,
    /// Number of expanding activation shells probed before minimum-mode following.
    pub activation_attempts: usize,
    /// Multiplicative radius increase while leaving the convex basin.
    pub activation_growth: f64,
    /// rgmin FIRE steps in the perpendicular hyperplane at each shell.
    pub activation_relaxation_steps: usize,
    /// Infinity-norm gradient threshold for a certified minimum.
    pub quench_gradient_tolerance: f64,
    /// Optional Euclidean gradient-norm certification threshold.
    ///
    /// The infinity-norm threshold remains the rgmin stopping target. This
    /// second gate lets a receiving-side force contract certify the measured
    /// endpoint when numerical force noise prevents the optimizer from
    /// reaching its stricter component target.
    pub quench_gradient_norm_tolerance: Option<f64>,
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
    /// Resolution levels used to choose the molecular IRC launch radius and
    /// the maximum N-D branch shells used to escape a collapsed quench.
    pub branch_attempts: usize,
    /// N-D branch-shell growth and reciprocal molecular IRC refinement factor.
    pub branch_growth: f64,
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
            minimum_mode_force_tolerance: 1e-3,
            irc_steps: 200,
            prfo_steps: 300,
            activation_attempts: 4,
            activation_growth: 2.0,
            activation_relaxation_steps: 3,
            quench_gradient_tolerance: 1e-6,
            quench_gradient_norm_tolerance: None,
            saddle_force_tolerance: 1e-3,
            saddle_displacement: 0.1,
            negative_curvature_tolerance: 1e-6,
            hessian_step: 1e-4,
            maximum_move: 0.2,
            irc_step: 0.1,
            branch_attempts: 4,
            branch_growth: 2.0,
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
            || self.activation_relaxation_steps == 0
            || self.branch_attempts == 0
        {
            return Err(PesExplorationError::InvalidConfig(
                "iteration limits must be positive",
            ));
        }
        for (name, value) in [
            ("quench gradient tolerance", self.quench_gradient_tolerance),
            (
                "minimum-mode force tolerance",
                self.minimum_mode_force_tolerance,
            ),
            ("saddle force tolerance", self.saddle_force_tolerance),
            ("saddle displacement", self.saddle_displacement),
            ("activation growth", self.activation_growth),
            ("Hessian finite-difference step", self.hessian_step),
            ("maximum move", self.maximum_move),
            ("IRC step", self.irc_step),
            ("branch growth", self.branch_growth),
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
        if self.branch_growth <= 1.0 {
            return Err(PesExplorationError::InvalidConfig(
                "branch growth must exceed one",
            ));
        }
        if !self.negative_curvature_tolerance.is_finite() || self.negative_curvature_tolerance < 0.0
        {
            return Err(PesExplorationError::InvalidConfig(
                "negative curvature tolerance",
            ));
        }
        if self
            .quench_gradient_norm_tolerance
            .is_some_and(|tolerance| !tolerance.is_finite() || tolerance <= 0.0)
        {
            return Err(PesExplorationError::InvalidConfig(
                "quench gradient-norm tolerance",
            ));
        }
        Ok(())
    }
}

/// Numerical stage that must converge before a saddle can be certified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaddleConvergenceStage {
    /// Minimum-mode following along the activated unstable direction.
    MinimumMode,
    /// Sella partitioned rational-function refinement.
    Prfo,
    /// Independent force check at the proposed saddle coordinates.
    ForceCertification,
}

impl Display for SaddleConvergenceStage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MinimumMode => "minimum-mode ride",
            Self::Prfo => "Sella P-RFO refinement",
            Self::ForceCertification => "receiving-side force certification",
        })
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
    /// A saddle-search or certification stage stopped before its force condition.
    #[error("{stage} stopped before convergence")]
    SaddleNotConverged {
        /// Saddle stage that stopped.
        stage: SaddleConvergenceStage,
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
    unresolved_saddles: Vec<SaddleConnection>,
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

    /// Index-one saddles whose downhill branches did not define a new edge.
    pub fn unresolved_saddles(&self) -> &[SaddleConnection] {
        &self.unresolved_saddles
    }

    pub(crate) fn from_records(minima: Vec<MinimumRecord>, saddles: Vec<SaddleConnection>) -> Self {
        Self {
            minima,
            saddles,
            unresolved_saddles: Vec::new(),
        }
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

    fn retain_unresolved_saddle(&mut self, mut candidate: SaddleConnection) {
        candidate.id = self.unresolved_saddles.len();
        self.unresolved_saddles.push(candidate);
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

/// Result of exact-witness minimum admission on one N-dimensional surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NdMinimumAdmission {
    /// Stable minimum index.
    pub id: usize,
    /// Whether a new exact point identity entered the network.
    pub is_new: bool,
    /// Nearest coordinate distance before the exact witness decision.
    pub nearest_coordinate_distance: Option<f64>,
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

/// Result and exact potential charge of one budgeted N-dimensional ridge ride.
#[derive(Debug)]
pub struct NdConnectionAttempt {
    /// Certified connection or the scientific failure returned by the ride.
    pub connection: Result<NdSaddleConnection, PesExplorationError>,
    /// PES evaluations accepted by the hard counter.
    pub charged_evaluations: u64,
    /// Whether the ride attempted an evaluation beyond its assigned slice.
    pub budget_exhausted: bool,
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
    unresolved_saddles: Vec<NdSaddleConnection>,
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

    /// Index-one saddles whose downhill branches collapsed to one minimum.
    pub fn unresolved_saddles(&self) -> &[NdSaddleConnection] {
        &self.unresolved_saddles
    }

    /// Admit a force-certified source minimum using the network's exact witness.
    pub fn admit_minimum<W: ExactStructureWitness + ?Sized>(
        &mut self,
        minimum: QuenchedMinimum,
        witness: &W,
    ) -> NdMinimumAdmission {
        self.admit_quenched(
            Quenched {
                energy: minimum.energy,
                coordinates: minimum.coordinates,
                gradient: minimum.gradient,
                max_gradient: minimum.max_gradient,
            },
            witness,
        )
    }

    fn admit_quenched<W: ExactStructureWitness + ?Sized>(
        &mut self,
        minimum: Quenched,
        witness: &W,
    ) -> NdMinimumAdmission {
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
        let nearest_coordinate_distance = ordered.first().map(|entry| entry.0);
        for (_, index) in ordered {
            if witness.equivalent(
                self.minima[index].coordinates.view(),
                minimum.coordinates.view(),
            ) {
                if minimum.energy < self.minima[index].energy {
                    self.minima[index].energy = minimum.energy;
                    self.minima[index].coordinates = minimum.coordinates;
                    self.minima[index].max_gradient = minimum.max_gradient;
                }
                return NdMinimumAdmission {
                    id: index,
                    is_new: false,
                    nearest_coordinate_distance,
                };
            }
        }
        let id = self.minima.len();
        self.minima.push(NdMinimumRecord {
            id,
            energy: minimum.energy,
            coordinates: minimum.coordinates,
            max_gradient: minimum.max_gradient,
        });
        NdMinimumAdmission {
            id,
            is_new: true,
            nearest_coordinate_distance,
        }
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

    fn retain_unresolved_saddle(&mut self, mut candidate: NdSaddleConnection) {
        candidate.id = self.unresolved_saddles.len();
        self.unresolved_saddles.push(candidate);
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

#[derive(Debug)]
enum EvaluationBudgetError {
    Exhausted,
    Surface(String),
}

impl Display for EvaluationBudgetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("PES-call budget exhausted"),
            Self::Surface(error) => formatter.write_str(error),
        }
    }
}

struct EvaluationBudgetSurface<'a, S: ?Sized> {
    surface: &'a S,
    maximum_evaluations: u64,
    state: Mutex<(u64, bool)>,
}

impl<'a, S: ?Sized> EvaluationBudgetSurface<'a, S> {
    fn new(surface: &'a S, maximum_evaluations: u64) -> Self {
        Self {
            surface,
            maximum_evaluations,
            state: Mutex::new((0, false)),
        }
    }

    fn state(&self) -> (u64, bool) {
        *self.state.lock().unwrap_or_else(|error| error.into_inner())
    }
}

impl<S> PesSurface for EvaluationBudgetSurface<'_, S>
where
    S: PesSurface + ?Sized,
{
    type Error = EvaluationBudgetError;

    fn evaluate(
        &self,
        coordinates: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), Self::Error> {
        {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.0 >= self.maximum_evaluations {
                state.1 = true;
                return Err(EvaluationBudgetError::Exhausted);
            }
            state.0 += 1;
        }
        self.surface
            .evaluate(coordinates)
            .map_err(|error| EvaluationBudgetError::Surface(error.to_string()))
    }
}

#[derive(Debug, Clone)]
struct Quenched {
    energy: f64,
    coordinates: Array1<f64>,
    gradient: Array1<f64>,
    max_gradient: f64,
}

/// A force-converged local minimum produced by the rgmin quench contract.
#[derive(Debug, Clone, PartialEq)]
pub struct QuenchedMinimum {
    /// Potential energy at the converged coordinates.
    pub energy: f64,
    /// Coordinates satisfying the requested force condition.
    pub coordinates: Array1<f64>,
    /// Fresh Cartesian gradient used to certify the returned coordinates.
    pub gradient: Array1<f64>,
    /// Infinity norm of the final Cartesian gradient.
    pub max_gradient: f64,
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

fn emit_pes_trace(event: serde_json::Value) {
    if std::env::var_os("ANNEAL_PES_TRACE").is_some() {
        eprintln!("{event}");
    }
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
    quench_with_certification(
        surface,
        start,
        config.quench_steps,
        config.quench_gradient_tolerance,
        config.quench_gradient_norm_tolerance,
    )
}

/// Quench an arbitrary-dimensional PES point with rgmin L-BFGS.
///
/// The returned point must satisfy the requested Cartesian infinity-norm
/// gradient condition; exhausting the iteration limit is an explicit error.
pub fn quench_minimum<S: PesSurface + ?Sized>(
    surface: &S,
    start: ArrayView1<f64>,
    steps: usize,
    gradient_tolerance: f64,
) -> Result<QuenchedMinimum, PesExplorationError> {
    if steps == 0 || !gradient_tolerance.is_finite() || gradient_tolerance <= 0.0 {
        return Err(PesExplorationError::InvalidConfig(
            "minimum quench controls",
        ));
    }
    let quenched = quench_with_tolerance(surface, start, steps, gradient_tolerance)?;
    Ok(QuenchedMinimum {
        energy: quenched.energy,
        coordinates: quenched.coordinates,
        gradient: quenched.gradient,
        max_gradient: quenched.max_gradient,
    })
}

/// Quench an arbitrary-dimensional PES point with a strict rgmin target and
/// an independent Euclidean force-norm certification gate.
///
/// The component tolerance controls rgmin. The returned point is accepted if
/// its measured gradient satisfies either that infinity-norm target or the
/// supplied Euclidean norm contract.
pub fn quench_minimum_with_norm<S: PesSurface + ?Sized>(
    surface: &S,
    start: ArrayView1<f64>,
    steps: usize,
    gradient_tolerance: f64,
    gradient_norm_tolerance: f64,
) -> Result<QuenchedMinimum, PesExplorationError> {
    if steps == 0
        || !gradient_tolerance.is_finite()
        || gradient_tolerance <= 0.0
        || !gradient_norm_tolerance.is_finite()
        || gradient_norm_tolerance <= 0.0
    {
        return Err(PesExplorationError::InvalidConfig(
            "minimum quench controls",
        ));
    }
    let quenched = quench_with_certification(
        surface,
        start,
        steps,
        gradient_tolerance,
        Some(gradient_norm_tolerance),
    )?;
    Ok(QuenchedMinimum {
        energy: quenched.energy,
        coordinates: quenched.coordinates,
        gradient: quenched.gradient,
        max_gradient: quenched.max_gradient,
    })
}

fn quench_with_tolerance<S: PesSurface + ?Sized>(
    surface: &S,
    start: ArrayView1<f64>,
    steps: usize,
    gradient_tolerance: f64,
) -> Result<Quenched, PesExplorationError> {
    quench_with_certification(surface, start, steps, gradient_tolerance, None)
}

fn quench_with_certification<S: PesSurface + ?Sized>(
    surface: &S,
    start: ArrayView1<f64>,
    steps: usize,
    gradient_tolerance: f64,
    gradient_norm_tolerance: Option<f64>,
) -> Result<Quenched, PesExplorationError> {
    let mut optimizer = Lbfgs::default();
    optimizer.gtol = gradient_tolerance;
    optimizer.norm = GradNorm::Infinity;
    let mut failure = None;
    let (_, coordinates, _) = optimizer.minimize(start, steps, |trial| {
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
    if !gradient_certified(
        gradient.view(),
        max_gradient,
        gradient_tolerance,
        gradient_norm_tolerance,
    ) {
        return Err(PesExplorationError::QuenchNotConverged { max_gradient });
    }
    Ok(Quenched {
        energy,
        coordinates,
        gradient,
        max_gradient,
    })
}

fn gradient_certified(
    gradient: ArrayView1<f64>,
    max_gradient: f64,
    gradient_tolerance: f64,
    gradient_norm_tolerance: Option<f64>,
) -> bool {
    max_gradient < gradient_tolerance
        || gradient_norm_tolerance
            .is_some_and(|tolerance| gradient.dot(&gradient).sqrt() <= tolerance)
}

fn quench_recognizing<W, S>(
    surface: &S,
    start: ArrayView1<f64>,
    known: &Quenched,
    witness: &W,
    context: Option<&StructureContext>,
    steps: usize,
    gradient_tolerance: f64,
    gradient_norm_tolerance: Option<f64>,
) -> Result<Quenched, PesExplorationError>
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
{
    if equivalent_with_context(witness, known.coordinates.view(), start, context) {
        return Ok(known.clone());
    }
    let mut optimizer = Lbfgs::default();
    optimizer.gtol = gradient_tolerance;
    optimizer.norm = GradNorm::Infinity;
    let mut failure = None;
    let (_, coordinates, _, recognized) = optimizer.minimize_recognized(
        start,
        steps,
        |trial| match checked_evaluate(surface, trial) {
            Ok(value_gradient) => Some(value_gradient),
            Err(error) => {
                if failure.is_none() {
                    failure = Some(error);
                }
                None
            }
        },
        |_, _, trial| {
            equivalent_with_context(witness, known.coordinates.view(), trial, context)
                .then(|| (known.energy, known.coordinates.clone()))
        },
    );
    if let Some(error) = failure {
        return Err(error);
    }
    if recognized {
        return Ok(known.clone());
    }
    let (energy, gradient) = checked_evaluate(surface, coordinates.view())?;
    let max_gradient = max_abs(gradient.view());
    if !gradient_certified(
        gradient.view(),
        max_gradient,
        gradient_tolerance,
        gradient_norm_tolerance,
    ) {
        return Err(PesExplorationError::QuenchNotConverged { max_gradient });
    }
    Ok(Quenched {
        energy,
        coordinates,
        gradient,
        max_gradient,
    })
}

fn equivalent_with_context<W: ExactStructureWitness + ?Sized>(
    witness: &W,
    left: ArrayView1<f64>,
    right: ArrayView1<f64>,
    context: Option<&StructureContext>,
) -> bool {
    context.map_or_else(
        || witness.equivalent(left, right),
        |context| {
            witness.equivalent_structures(
                StructureView {
                    coordinates: left,
                    context,
                },
                StructureView {
                    coordinates: right,
                    context,
                },
            )
        },
    )
}

fn reconcile_source_connection<W, S>(
    surface: &S,
    origin: Quenched,
    positive: Quenched,
    negative: Quenched,
    config: &PesExplorationConfig,
    witness: &W,
    context: Option<&StructureContext>,
) -> Result<(Quenched, Quenched, Quenched), PesExplorationError>
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
{
    if equivalent_with_context(
        witness,
        origin.coordinates.view(),
        positive.coordinates.view(),
        context,
    ) || equivalent_with_context(
        witness,
        origin.coordinates.view(),
        negative.coordinates.view(),
        context,
    ) {
        return Ok((origin, positive, negative));
    }

    let identity_tolerance =
        (config.quench_gradient_tolerance * f64::EPSILON.sqrt()).max(f64::MIN_POSITIVE);
    let origin = quench_with_certification(
        surface,
        origin.coordinates.view(),
        config.quench_steps,
        identity_tolerance,
        config.quench_gradient_norm_tolerance,
    )?;
    let positive = quench_recognizing(
        surface,
        positive.coordinates.view(),
        &origin,
        witness,
        context,
        config.quench_steps,
        identity_tolerance,
        config.quench_gradient_norm_tolerance,
    )?;
    let negative = quench_recognizing(
        surface,
        negative.coordinates.view(),
        &origin,
        witness,
        context,
        config.quench_steps,
        identity_tolerance,
        config.quench_gradient_norm_tolerance,
    )?;
    Ok((origin, positive, negative))
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
        force_tol: if config.refine_with_prfo {
            config.minimum_mode_force_tolerance
        } else {
            config.saddle_force_tolerance
        },
        max_move: config.maximum_move,
        ..MinModeConfig::default()
    }
}

fn sorted_eigensystem(
    hessian: ArrayView2<'_, f64>,
) -> Result<(Array1<f64>, Array2<f64>), PesExplorationError> {
    if hessian.nrows() != hessian.ncols() || hessian.is_empty() {
        return Err(PesExplorationError::InvalidShape(
            "Hessian must be nonempty and square",
        ));
    }
    let (eigenvalues, eigenvectors) =
        exact_eigh(hessian).map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let mut order = (0..eigenvalues.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| eigenvalues[left].total_cmp(&eigenvalues[right]));
    let sorted_values = Array1::from_iter(order.iter().map(|&index| eigenvalues[index]));
    let mut sorted_vectors = Array2::zeros(eigenvectors.raw_dim());
    for (column, &index) in order.iter().enumerate() {
        sorted_vectors
            .column_mut(column)
            .assign(&eigenvectors.column(index));
    }
    Ok((sorted_values, sorted_vectors))
}

struct HomedEigensystem {
    eigenvalues: Array1<f64>,
    eigenvectors: Array2<f64>,
    source_index: usize,
    overlap: f64,
}

fn home_uphill_mode(
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
    reference: ArrayView1<f64>,
    negative_tolerance: f64,
) -> Result<HomedEigensystem, PesExplorationError> {
    if eigenvalues.is_empty()
        || eigenvectors.nrows() != eigenvalues.len()
        || eigenvectors.ncols() != eigenvalues.len()
        || reference.len() != eigenvalues.len()
    {
        return Err(PesExplorationError::InvalidShape(
            "mode homing requires one square eigensystem and matching reference",
        ));
    }
    if !negative_tolerance.is_finite() || negative_tolerance < 0.0 {
        return Err(PesExplorationError::InvalidConfig(
            "mode-homing negative tolerance",
        ));
    }
    let reference = normalize_mode(reference)?;
    let negative = (0..eigenvalues.len())
        .filter(|&index| eigenvalues[index] < -negative_tolerance)
        .collect::<Vec<_>>();
    let candidates = if negative.is_empty() {
        (0..eigenvalues.len()).collect::<Vec<_>>()
    } else {
        negative
    };
    let overlaps = candidates
        .iter()
        .map(|&index| (index, eigenvectors.column(index).dot(&reference)))
        .collect::<Vec<_>>();
    let maximum_overlap = overlaps
        .iter()
        .map(|(_, overlap)| overlap.abs())
        .max_by(f64::total_cmp)
        .ok_or(PesExplorationError::InvalidShape(
            "mode homing eigensystem is empty",
        ))?;
    let overlap_gate = MODE_HOMING_OVERLAP_FRACTION * maximum_overlap;
    let (source_index, signed_overlap) = overlaps
        .into_iter()
        .find(|(_, overlap)| overlap.abs() >= overlap_gate)
        .ok_or(PesExplorationError::InvalidShape(
            "mode homing found no admissible root",
        ))?;
    let mut order = Vec::with_capacity(eigenvalues.len());
    order.push(source_index);
    order.extend((0..eigenvalues.len()).filter(|index| *index != source_index));
    let eigenvalues = Array1::from_iter(order.iter().map(|&index| eigenvalues[index]));
    let mut homed_vectors = Array2::zeros(eigenvectors.raw_dim());
    for (column, &index) in order.iter().enumerate() {
        homed_vectors
            .column_mut(column)
            .assign(&eigenvectors.column(index));
    }
    if signed_overlap < 0.0 {
        homed_vectors.column_mut(0).mapv_inplace(|value| -value);
    }
    Ok(HomedEigensystem {
        eigenvalues,
        eigenvectors: homed_vectors,
        source_index,
        overlap: signed_overlap.abs(),
    })
}

fn orthonormal_complement(
    dimension: usize,
    constraints: &[Array1<f64>],
) -> Result<Array2<f64>, PesExplorationError> {
    let mut columns = Vec::<Array1<f64>>::new();
    for coordinate in 0..dimension {
        let mut direction = Array1::zeros(dimension);
        direction[coordinate] = 1.0;
        project_rigid_with(&mut direction, constraints);
        for _ in 0..2 {
            for basis in &columns {
                direction -= &(basis * direction.dot(basis));
            }
        }
        let norm = direction.dot(&direction).sqrt();
        if norm > 1e-10 {
            columns.push(direction / norm);
        }
    }
    if columns.is_empty() {
        return Err(PesExplorationError::InvalidShape(
            "Cartesian constraints remove every coordinate",
        ));
    }
    let mut basis = Array2::zeros((dimension, columns.len()));
    for (column, direction) in columns.iter().enumerate() {
        basis.column_mut(column).assign(direction);
    }
    Ok(basis)
}

fn finite_difference_free_hessian<S: PesSurface + ?Sized>(
    surface: &S,
    coordinates: ArrayView1<f64>,
    basis: &Array2<f64>,
    step: f64,
) -> Result<Array2<f64>, PesExplorationError> {
    let mut hessian = Array2::zeros((basis.ncols(), basis.ncols()));
    for column in 0..basis.ncols() {
        let direction = basis.column(column);
        let plus = &coordinates + &(&direction * step);
        let minus = &coordinates - &(&direction * step);
        let (_, plus_gradient) = checked_evaluate(surface, plus.view())?;
        let (_, minus_gradient) = checked_evaluate(surface, minus.view())?;
        let action = (plus_gradient - minus_gradient) / (2.0 * step);
        for row in 0..basis.ncols() {
            hessian[(row, column)] = basis.column(row).dot(&action);
        }
    }
    for row in 0..hessian.nrows() {
        for column in 0..row {
            let symmetric = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            hessian[(row, column)] = symmetric;
            hessian[(column, row)] = symmetric;
        }
    }
    Ok(hessian)
}

fn refine_nd_with_prfo<S: PesSurface + ?Sized>(
    surface: &S,
    start: ArrayView1<f64>,
    config: &PesExplorationConfig,
) -> Result<Array1<f64>, PesExplorationError> {
    let mut coordinates = start.to_owned();
    for _ in 0..config.prfo_steps {
        let (_, gradient) = checked_evaluate(surface, coordinates.view())?;
        if max_abs(gradient.view()) <= config.saddle_force_tolerance {
            return Ok(coordinates);
        }
        let hessian = finite_difference_hessian(surface, coordinates.view(), config.hessian_step)?;
        let (sorted_values, sorted_vectors) = sorted_eigensystem(hessian.view())?;
        let step = prfo_trust_region(
            &sorted_values,
            &sorted_vectors,
            &gradient,
            1,
            config.maximum_move,
        );
        if step.len() != coordinates.len() || step.iter().any(|value| !value.is_finite()) {
            return Err(PesExplorationError::Saddle(
                "P-RFO returned an invalid N-D step".into(),
            ));
        }
        coordinates += &step;
    }
    Err(PesExplorationError::SaddleNotConverged {
        stage: SaddleConvergenceStage::Prfo,
    })
}

fn refine_cartesian_with_prfo<S: PesSurface + ?Sized>(
    surface: &S,
    start: ArrayView1<f64>,
    initial_mode: ArrayView1<f64>,
    cartesian_index: Option<(&[bool], [bool; 3])>,
    config: &PesExplorationConfig,
) -> Result<Array1<f64>, PesExplorationError> {
    let mut coordinates = start.to_owned();
    let mut uphill_mode = normalize_mode(initial_mode)?;
    let mut trust_radius = config.saddle_displacement.min(config.maximum_move);
    let trust_schedule = TrustSchedule::saddle();
    let (mut energy, mut gradient) = checked_evaluate(surface, coordinates.view())?;
    let mut hessian_model: Option<(Array2<f64>, Array2<f64>)> = None;
    let mut refresh_hessian = true;
    let mut exact_refreshes = 0usize;
    let mut secant_updates = 0usize;
    for iteration in 0..config.prfo_steps {
        let constraints = activation_basis(coordinates.view(), cartesian_index);
        let basis = orthonormal_complement(coordinates.len(), &constraints)?;
        let free_gradient = basis.t().dot(&gradient);
        let mut tangent_gradient = gradient.clone();
        project_rigid_with(&mut tangent_gradient, &constraints);
        let maximum_gradient = max_abs(tangent_gradient.view());
        if maximum_gradient <= config.saddle_force_tolerance {
            emit_pes_trace(serde_json::json!({
                "kind": "pes_stage",
                "stage": "prfo",
                "status": "converged",
                "iteration": iteration,
                "energy": energy,
                "maximum_gradient": maximum_gradient,
                "free_dimension": basis.ncols(),
                "exact_hessian_refreshes": exact_refreshes,
                "ts_bfgs_updates": secant_updates,
            }));
            return Ok(coordinates);
        }
        let (mut hessian, hessian_source) = match hessian_model.take() {
            Some((model, old_basis))
                if !refresh_hessian
                    && model.nrows() == basis.ncols()
                    && old_basis.nrows() == basis.nrows()
                    && old_basis.ncols() == basis.ncols() =>
            {
                let transport = basis.t().dot(&old_basis);
                let mut transported = transport.dot(&model).dot(&transport.t());
                for row in 0..transported.nrows() {
                    for column in 0..row {
                        let symmetric =
                            0.5 * (transported[(row, column)] + transported[(column, row)]);
                        transported[(row, column)] = symmetric;
                        transported[(column, row)] = symmetric;
                    }
                }
                (transported, "ts-bfgs")
            }
            _ => {
                exact_refreshes += 1;
                (
                    finite_difference_free_hessian(
                        surface,
                        coordinates.view(),
                        &basis,
                        config.hessian_step,
                    )?,
                    "finite-difference",
                )
            }
        };
        let (eigenvalues, eigenvectors) = sorted_eigensystem(hessian.view())?;
        let negative_modes = eigenvalues
            .iter()
            .filter(|value| **value < -config.negative_curvature_tolerance)
            .count();
        let free_reference = basis.t().dot(&uphill_mode);
        let homed = home_uphill_mode(
            &eigenvalues,
            &eigenvectors,
            free_reference.view(),
            config.negative_curvature_tolerance,
        )?;
        let free_step = prfo_trust_region(
            &homed.eigenvalues,
            &homed.eigenvectors,
            &free_gradient,
            1,
            trust_radius,
        );
        let step = basis.dot(&free_step);
        let step_norm = step.dot(&step).sqrt();
        if step.len() != coordinates.len() || step.iter().any(|value| !value.is_finite()) {
            return Err(PesExplorationError::Saddle(
                "P-RFO returned an invalid Cartesian step".into(),
            ));
        }
        let candidate = &coordinates + &step;
        let (candidate_energy, candidate_gradient) = checked_evaluate(surface, candidate.view())?;
        let prediction =
            free_gradient.dot(&free_step) + 0.5 * free_step.dot(&hessian.dot(&free_step));
        let rho = if prediction.abs() >= 1e-14 {
            (candidate_energy - energy) / prediction
        } else {
            1.0
        };
        let next_trust_radius = if rho.is_finite() {
            update_trust(trust_radius, rho, step_norm, &trust_schedule).min(config.maximum_move)
        } else {
            trust_schedule.delta_min.min(config.maximum_move)
        };
        let secant = &candidate_gradient - &gradient;
        let free_secant = basis.t().dot(&secant);
        update_h(&mut hessian, &free_step, &free_secant, HessUpdate::TsBfgs);
        secant_updates += 1;
        let refresh_next =
            !rho.is_finite() || rho < PRFO_MODEL_MIN_RHO || homed.overlap < PRFO_MODEL_MIN_OVERLAP;
        emit_pes_trace(serde_json::json!({
            "kind": "pes_stage",
            "stage": "prfo",
            "status": "running",
            "iteration": iteration,
            "energy": energy,
            "candidate_energy": candidate_energy,
            "maximum_gradient": maximum_gradient,
            "lowest_curvature": eigenvalues[0],
            "followed_curvature": homed.eigenvalues[0],
            "mode_overlap": homed.overlap,
            "followed_mode_index": homed.source_index,
            "negative_modes": negative_modes,
            "step_norm": step_norm,
            "rho": rho,
            "trust_radius": trust_radius,
            "next_trust_radius": next_trust_radius,
            "free_dimension": basis.ncols(),
            "hessian_source": hessian_source,
            "exact_hessian_refreshes": exact_refreshes,
            "ts_bfgs_updates": secant_updates,
            "refresh_hessian": refresh_next,
        }));
        uphill_mode = normalize_mode(basis.dot(&homed.eigenvectors.column(0)).view())?;
        hessian_model = Some((hessian, basis));
        refresh_hessian = refresh_next;
        coordinates = candidate;
        energy = candidate_energy;
        gradient = candidate_gradient;
        trust_radius = next_trust_radius;
    }
    Err(PesExplorationError::SaddleNotConverged {
        stage: SaddleConvergenceStage::Prfo,
    })
}

fn cartesian_max_gradient(
    coordinates: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    cartesian_index: Option<(&[bool], [bool; 3])>,
) -> Result<f64, PesExplorationError> {
    let constraints = activation_basis(coordinates, cartesian_index);
    let mut tangent_gradient = gradient.to_owned();
    project_rigid_with(&mut tangent_gradient, &constraints);
    Ok(max_abs(tangent_gradient.view()))
}

fn irc_config(config: &PesExplorationConfig, branch_step: f64) -> IrcConfig {
    IrcConfig {
        dx: branch_step,
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

fn source_scaled_branch_step(
    origin: ArrayView1<f64>,
    saddle: ArrayView1<f64>,
    configured_step: f64,
) -> f64 {
    let source_distance = origin
        .iter()
        .zip(saddle)
        .map(|(origin, saddle)| (saddle - origin).powi(2))
        .sum::<f64>()
        .sqrt();
    configured_step.max(ART_BRANCH_FRACTION * source_distance)
}

fn grow_branch_step(step: f64, config: &PesExplorationConfig) -> Result<f64, PesExplorationError> {
    let next = step * config.branch_growth;
    if !next.is_finite() {
        return Err(PesExplorationError::InvalidEvaluation(
            "a nonfinite downhill branch radius",
        ));
    }
    Ok(next)
}

fn quench_branch_shells<S, W>(
    surface: &S,
    mut origin: Quenched,
    saddle: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    initial_step: f64,
    config: &PesExplorationConfig,
    witness: &W,
) -> Result<(Quenched, Quenched, Quenched), PesExplorationError>
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
{
    let mut step = initial_step;
    for shell in 0..config.branch_attempts {
        let positive_start = &saddle + &(&mode * step);
        let negative_start = &saddle - &(&mode * step);
        let positive = quench(surface, positive_start.view(), config)?;
        let negative = quench(surface, negative_start.view(), config)?;
        let (resolved_origin, positive, negative) = reconcile_source_connection(
            surface, origin, positive, negative, config, witness, None,
        )?;
        let collapsed =
            witness.equivalent(positive.coordinates.view(), negative.coordinates.view());
        if !collapsed || shell + 1 == config.branch_attempts {
            return Ok((resolved_origin, positive, negative));
        }
        origin = resolved_origin;
        step = grow_branch_step(step, config)?;
    }
    unreachable!("a positive branch-shell count returns inside the loop")
}

fn refine_irc_step(step: f64, config: &PesExplorationConfig) -> Result<f64, PesExplorationError> {
    let next = step / config.branch_growth;
    if !next.is_finite() || next <= 0.0 {
        return Err(PesExplorationError::InvalidEvaluation(
            "an invalid refined IRC radius",
        ));
    }
    Ok(next)
}

fn finest_irc_step(
    initial_step: f64,
    config: &PesExplorationConfig,
) -> Result<f64, PesExplorationError> {
    (1..config.branch_attempts).try_fold(initial_step, |step, _| refine_irc_step(step, config))
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

fn activation_basis(
    coordinates: ArrayView1<f64>,
    cartesian_index: Option<(&[bool], [bool; 3])>,
) -> Vec<Array1<f64>> {
    let Some((frozen_atoms, periodic)) = cartesian_index else {
        return Vec::new();
    };
    let mut basis = Vec::new();
    for (atom, frozen) in frozen_atoms.iter().copied().enumerate() {
        if frozen {
            for axis in 0..3 {
                let mut coordinate = Array1::zeros(coordinates.len());
                coordinate[3 * atom + axis] = 1.0;
                basis.push(coordinate);
            }
        }
    }
    if basis.is_empty() {
        basis = if periodic.iter().any(|axis| *axis) {
            translation_basis(coordinates.len() / 3)
        } else {
            rigid_basis(coordinates)
        };
    }
    basis
}

fn project_activation_vector(values: &mut Array1<f64>, basis: &[Array1<f64>]) {
    if !basis.is_empty() {
        project_rigid_with(values, basis);
    }
}

struct ActivationHessian<'a, S: PesSurface + ?Sized> {
    surface: &'a S,
    gradient: Array1<f64>,
    step: f64,
    basis: Vec<Array1<f64>>,
    failure: Mutex<Option<PesExplorationError>>,
}

impl<S: PesSurface + ?Sized> ApplyHessian for ActivationHessian<'_, S> {
    fn apply_hessian(&self, coordinates: ArrayView1<f64>, mode: ArrayView1<f64>) -> Array1<f64> {
        let mut mode = mode.to_owned();
        project_activation_vector(&mut mode, &self.basis);
        let mode = match normalize_mode(mode.view()) {
            Ok(mode) => mode,
            Err(error) => {
                *self.failure.lock().expect("activation failure lock") = Some(error);
                return Array1::zeros(coordinates.len());
            }
        };
        let shifted = &coordinates + &(&mode * self.step);
        let (_, shifted_gradient) = match checked_evaluate(self.surface, shifted.view()) {
            Ok(value) => value,
            Err(error) => {
                *self.failure.lock().expect("activation failure lock") = Some(error);
                return Array1::zeros(coordinates.len());
            }
        };
        let mut hessian_mode = (&shifted_gradient - &self.gradient) / self.step;
        project_activation_vector(&mut hessian_mode, &self.basis);
        hessian_mode
    }
}

fn lowest_activation_mode<S: PesSurface + ?Sized>(
    surface: &S,
    coordinates: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    cartesian_index: Option<(&[bool], [bool; 3])>,
    config: &PesExplorationConfig,
) -> Result<(f64, Array1<f64>), PesExplorationError> {
    let (_, gradient) = checked_evaluate(surface, coordinates)?;
    let basis = activation_basis(coordinates, cartesian_index);
    let mut seed = seed.to_owned();
    project_activation_vector(&mut seed, &basis);
    let seed = normalize_mode(seed.view())?;
    let hessian = ActivationHessian {
        surface,
        gradient,
        step: config.hessian_step,
        basis,
        failure: Mutex::new(None),
    };
    let params = EigenParams {
        kind: EigensolverKind::Lanczos,
        krylov: coordinates.len().min(12),
        ..EigenParams::default()
    };
    let result = lowest_mode(&hessian, coordinates, seed.view(), &params);
    let failure = hessian
        .failure
        .lock()
        .map_err(|_| PesExplorationError::Saddle("activation failure lock poisoned".into()))?
        .take();
    if let Some(error) = failure {
        return Err(error);
    }
    let result = result.map_err(|error| {
        PesExplorationError::Saddle(format!("activation Lanczos failed: {error}"))
    })?;
    let mut mode = result.vector;
    project_activation_vector(&mut mode, &hessian.basis);
    let mode = normalize_mode(mode.view())?;
    let curvature = directional_curvature(surface, coordinates, mode.view(), config.hessian_step)?;
    Ok((curvature, mode))
}

fn relax_activation_shell<S: PesSurface + ?Sized>(
    surface: &S,
    origin: ArrayView1<f64>,
    push_mode: ArrayView1<f64>,
    target_radius: f64,
    mut coordinates: Array1<f64>,
    cartesian_index: Option<(&[bool], [bool; 3])>,
    config: &PesExplorationConfig,
) -> Result<Array1<f64>, PesExplorationError> {
    let failure = Mutex::new(None);
    let push_mode = push_mode.to_owned();
    let oracle =
        Oracle::unbounded(
            coordinates.len(),
            |position: ArrayView1<f64>| match checked_evaluate(surface, position) {
                Ok((energy, mut gradient)) => {
                    let basis = activation_basis(position, cartesian_index);
                    project_activation_vector(&mut gradient, &basis);
                    let parallel = gradient.dot(&push_mode);
                    gradient -= &(&push_mode * parallel);
                    (energy, gradient)
                }
                Err(error) => {
                    *failure.lock().expect("activation failure lock") = Some(error);
                    (f64::INFINITY, Array1::zeros(position.len()))
                }
            },
        );
    let control = Control {
        maxiter: usize::MAX,
        gtol: 0.0,
        istep: 1.0,
        maxmove: Some(0.25 * config.maximum_move),
    };
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        control,
        coordinates.len(),
    );
    for _ in 0..config.activation_relaxation_steps {
        let step = solver.step(&oracle, &mut coordinates);
        if let Some(error) = failure.lock().expect("activation failure lock").take() {
            return Err(error);
        }
        step.map_err(|error| {
            PesExplorationError::Saddle(format!("activation relaxation failed: {error}"))
        })?;
        let axial = (&coordinates - &origin).dot(&push_mode);
        coordinates += &(&push_mode * (target_radius - axial));
    }
    Ok(coordinates)
}

struct ActivationStart {
    coordinates: Array1<f64>,
    mode: Array1<f64>,
}

fn bowl_breakout<S: PesSurface + ?Sized>(
    surface: &S,
    origin: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    cartesian_index: Option<(&[bool], [bool; 3])>,
    config: &PesExplorationConfig,
) -> Result<ActivationStart, PesExplorationError> {
    let push_mode = normalize_mode(mode)?;
    let mut eigenmode = push_mode.clone();
    let mut trial = origin.to_owned();
    let mut radius = config.saddle_displacement;
    let mut lowest_curvature = f64::INFINITY;
    for _ in 0..config.activation_attempts {
        let axial = (&trial - &origin).dot(&push_mode);
        trial += &(&push_mode * (radius - axial));
        trial = relax_activation_shell(
            surface,
            origin,
            push_mode.view(),
            radius,
            trial,
            cartesian_index,
            config,
        )?;
        let (curvature, mode) = lowest_activation_mode(
            surface,
            trial.view(),
            eigenmode.view(),
            cartesian_index,
            config,
        )?;
        lowest_curvature = lowest_curvature.min(curvature);
        if curvature < -config.negative_curvature_tolerance {
            return Ok(ActivationStart {
                coordinates: trial,
                mode,
            });
        }
        eigenmode = mode;
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
    branch_step: f64,
    config: &PesExplorationConfig,
) -> Result<(Quenched, bool), PesExplorationError> {
    let adapter = SaddleSurface(surface);
    let mut session = IrcSession::from_surface(
        irc_config(config, branch_step),
        saddle.clone(),
        masses.clone(),
        mode.clone(),
        direction,
        &adapter,
    )
    .map_err(|error| {
        PesExplorationError::Saddle(format!("IRC {direction:?} initialization: {error}"))
    })?;
    let mut final_report = None;
    for step in 0..config.irc_steps {
        let report = match session.step(&adapter) {
            Ok(report) => report,
            Err(error) => {
                let position = session.position();
                emit_pes_trace(serde_json::json!({
                    "kind": "pes_stage",
                    "stage": "irc",
                    "status": "failed",
                    "direction": format!("{direction:?}"),
                    "iteration": step,
                    "branch_step": branch_step,
                    "displacement_from_saddle": cartesian_distance(position, saddle.view()),
                    "minimum_pair_distance": minimum_pair_distance(position),
                    "error": error.to_string(),
                }));
                return Err(PesExplorationError::Saddle(format!(
                    "IRC {direction:?} step {step}: {error}"
                )));
            }
        };
        let position = session.position();
        emit_pes_trace(serde_json::json!({
            "kind": "pes_stage",
            "stage": "irc",
            "status": if report.at_minimum { "converged" } else { "running" },
            "direction": format!("{direction:?}"),
            "iteration": step,
            "branch_step": branch_step,
            "energy": report.energy,
            "maximum_gradient": report.max_force,
            "arc": report.arc,
            "inner_steps": report.inner_steps,
            "displacement_from_saddle": cartesian_distance(position, saddle.view()),
            "minimum_pair_distance": minimum_pair_distance(position),
        }));
        let at_minimum = report.at_minimum;
        final_report = Some(report);
        if at_minimum {
            break;
        }
    }
    let report = final_report.ok_or(PesExplorationError::InvalidConfig("IRC steps"))?;
    let endpoint = session.position().to_owned();
    Ok((quench(surface, endpoint.view(), config)?, report.at_minimum))
}

fn cartesian_distance(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
        .sqrt()
}

fn minimum_pair_distance(coordinates: ArrayView1<f64>) -> Option<f64> {
    let atom_count = coordinates.len() / 3;
    (coordinates.len().is_multiple_of(3) && atom_count >= 2).then(|| {
        (0..atom_count)
            .flat_map(|first| (first + 1..atom_count).map(move |second| (first, second)))
            .map(|(first, second)| {
                (0..3)
                    .map(|axis| {
                        let delta = coordinates[3 * first + axis] - coordinates[3 * second + axis];
                        delta * delta
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    })
}

#[allow(clippy::too_many_arguments)]
fn roll_branch_shells<S, W>(
    surface: &S,
    origin: Quenched,
    saddle: &Array1<f64>,
    masses: &Array1<f64>,
    mode: &Array1<f64>,
    initial_step: f64,
    config: &PesExplorationConfig,
    context: &StructureContext,
    witness: &W,
) -> Result<(Quenched, Quenched, Quenched, [bool; 2]), PesExplorationError>
where
    S: PesSurface,
    W: ExactStructureWitness + ?Sized,
{
    let step = finest_irc_step(initial_step, config)?;
    emit_pes_trace(serde_json::json!({
        "kind": "pes_stage",
        "stage": "irc_resolution",
        "initial_step": initial_step,
        "selected_step": step,
        "resolution_levels": config.branch_attempts,
    }));
    let (forward, forward_irc) = roll_branch(
        surface,
        saddle,
        masses,
        mode,
        IrcDirection::Forward,
        step,
        config,
    )?;
    let (reverse, reverse_irc) = roll_branch(
        surface,
        saddle,
        masses,
        mode,
        IrcDirection::Reverse,
        step,
        config,
    )?;
    let (resolved_origin, forward, reverse) = reconcile_source_connection(
        surface,
        origin,
        forward,
        reverse,
        config,
        witness,
        Some(context),
    )?;
    Ok((
        resolved_origin,
        forward,
        reverse,
        [forward_irc, reverse_irc],
    ))
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
    let mode = normalize_mode(mode)?;
    let activation = bowl_breakout(
        surface,
        origin_minimum.coordinates.view(),
        mode.view(),
        None,
        config,
    )?;
    let adapter = SaddleSurface(surface);
    let mut min_mode = MinModeSession::new(
        min_mode_config(config),
        activation.coordinates,
        activation.mode,
    )
    .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let min_mode_report = min_mode
        .run(&adapter, config.saddle_steps)
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    emit_pes_trace(serde_json::json!({
        "kind": "pes_stage",
        "stage": "minimum_mode",
        "status": format!("{:?}", min_mode_report.status),
        "method": format!("{:?}", config.ride_method),
        "iteration": min_mode_report.iteration,
        "maximum_gradient": min_mode_report.max_force,
        "lowest_curvature": min_mode_report.curvature,
        "rotations": min_mode_report.rotations,
    }));
    if min_mode_report.status != MinModeStatus::Converged {
        return Err(PesExplorationError::SaddleNotConverged {
            stage: SaddleConvergenceStage::MinimumMode,
        });
    }
    if min_mode_report.curvature >= -config.negative_curvature_tolerance {
        return Err(PesExplorationError::MinimumModeLostCurvature {
            curvature: min_mode_report.curvature,
        });
    }

    let mut saddle_coordinates = min_mode.position().to_owned();
    if config.refine_with_prfo {
        saddle_coordinates = refine_nd_with_prfo(surface, saddle_coordinates.view(), config)?;
    }
    let (saddle_energy, saddle_gradient) = checked_evaluate(surface, saddle_coordinates.view())?;
    let saddle_max_gradient = max_abs(saddle_gradient.view());
    if saddle_max_gradient > config.saddle_force_tolerance {
        return Err(PesExplorationError::SaddleNotConverged {
            stage: SaddleConvergenceStage::ForceCertification,
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
    let branch_step = source_scaled_branch_step(
        origin_minimum.coordinates.view(),
        saddle_coordinates.view(),
        config.irc_step,
    );
    let (origin_minimum, positive, negative) = quench_branch_shells(
        surface,
        origin_minimum,
        saddle_coordinates.view(),
        lowest_mode.view(),
        branch_step,
        config,
        witness,
    )?;
    let origin = network.admit_quenched(origin_minimum, witness).id;
    let positive_id = network.admit_quenched(positive, witness).id;
    let negative_id = network.admit_quenched(negative, witness).id;
    let candidate = NdSaddleConnection {
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
    };
    if positive_id == negative_id {
        network.retain_unresolved_saddle(candidate);
        return Err(PesExplorationError::CollapsedConnection);
    }

    Ok(network.admit_saddle(candidate, witness))
}

/// Run one N-dimensional ridge ride under a hard PES-evaluation boundary.
///
/// Calls refused at the boundary are not charged. The supplied network keeps
/// any certified minima or unresolved saddle evidence accumulated before a
/// scientific failure, so a hybrid explorer can reuse the evidence.
#[allow(clippy::too_many_arguments)]
pub fn discover_nd_connection_with_budget<S, W>(
    surface: &S,
    network: &mut NdPesNetwork,
    start: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    config: &PesExplorationConfig,
    witness: &W,
    maximum_evaluations: u64,
) -> NdConnectionAttempt
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
{
    let budgeted = EvaluationBudgetSurface::new(surface, maximum_evaluations);
    let connection = discover_nd_connection(&budgeted, network, start, mode, config, witness);
    let (charged_evaluations, budget_exhausted) = budgeted.state();
    NdConnectionAttempt {
        connection,
        charged_evaluations,
        budget_exhausted,
    }
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
    let mode = normalize_mode(mode)?;
    let activation = bowl_breakout(
        surface,
        origin_minimum.coordinates.view(),
        mode.view(),
        cartesian_index,
        config,
    )?;
    let adapter = SaddleSurface(surface);
    let mut min_mode = MinModeSession::new(
        min_mode_config(config),
        activation.coordinates,
        activation.mode,
    )
    .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let min_mode_report = min_mode
        .run(&adapter, config.saddle_steps)
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    emit_pes_trace(serde_json::json!({
        "kind": "pes_stage",
        "stage": "minimum_mode",
        "status": format!("{:?}", min_mode_report.status),
        "method": format!("{:?}", config.ride_method),
        "iteration": min_mode_report.iteration,
        "maximum_gradient": min_mode_report.max_force,
        "lowest_curvature": min_mode_report.curvature,
        "rotations": min_mode_report.rotations,
    }));
    if min_mode_report.status != MinModeStatus::Converged {
        return Err(PesExplorationError::SaddleNotConverged {
            stage: SaddleConvergenceStage::MinimumMode,
        });
    }
    if min_mode_report.curvature >= -config.negative_curvature_tolerance {
        return Err(PesExplorationError::MinimumModeLostCurvature {
            curvature: min_mode_report.curvature,
        });
    }
    let mut saddle_coordinates = min_mode.position().to_owned();
    let saddle_mode = min_mode.mode().to_owned();

    if config.refine_with_prfo {
        saddle_coordinates = refine_cartesian_with_prfo(
            surface,
            saddle_coordinates.view(),
            saddle_mode.view(),
            cartesian_index,
            config,
        )?;
    }

    let (saddle_energy, saddle_gradient) = checked_evaluate(surface, saddle_coordinates.view())?;
    let saddle_max_gradient = cartesian_max_gradient(
        saddle_coordinates.view(),
        saddle_gradient.view(),
        cartesian_index,
    )?;
    if saddle_max_gradient > config.saddle_force_tolerance {
        return Err(PesExplorationError::SaddleNotConverged {
            stage: SaddleConvergenceStage::ForceCertification,
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
    let branch_step = config.irc_step;

    let (origin_minimum, forward, reverse, irc_at_minimum) = roll_branch_shells(
        surface,
        origin_minimum,
        &saddle_coordinates,
        &masses.to_owned(),
        &saddle_mode,
        branch_step,
        config,
        &context,
        witness,
    )?;
    let origin_descriptor = descriptor(descriptor_space, &origin_minimum, context.species())?;
    let forward_descriptor = descriptor(descriptor_space, &forward, context.species())?;
    let reverse_descriptor = descriptor(descriptor_space, &reverse, context.species())?;
    let origin = network.admit_minimum_with_context(
        origin_minimum.energy,
        origin_minimum.coordinates,
        origin_minimum.max_gradient,
        origin_descriptor,
        context.clone(),
        witness,
    )?;
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
    let saddle_descriptor =
        descriptor_space.describe(saddle_coordinates.view(), context.species())?;
    let candidate = SaddleConnection {
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
        irc_at_minimum,
    };
    if forward_id.id == reverse_id.id {
        network.retain_unresolved_saddle(candidate);
        return Err(PesExplorationError::CollapsedConnection);
    }

    network
        .admit_saddle(candidate, witness)
        .map_err(PesExplorationError::from)
}
