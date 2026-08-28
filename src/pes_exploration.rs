//! Minimum--saddle--minimum exploration with rgmin and rgsaddle.
//!
//! The descriptor orders structural comparisons but never certifies identity.
//! A caller-supplied exact witness makes the final equivalence decision. This
//! keeps catalog admission independent of a fixed descriptor radius while the
//! same universal descriptor remains available for novelty and acquisition.

use std::fmt::Display;

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{GradNorm, Lbfgs};
use rgsaddle::{
    IrcConfig, IrcDirection, IrcSession, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus,
    PointSurface, SaddleError, SellaSaddleConfig, SellaSaddleSession, exact_eigh,
};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        {
            return Err(PesExplorationError::InvalidConfig(
                "iteration limits must be positive",
            ));
        }
        for (name, value) in [
            ("quench gradient tolerance", self.quench_gradient_tolerance),
            ("saddle force tolerance", self.saddle_force_tolerance),
            ("saddle displacement", self.saddle_displacement),
            ("Hessian finite-difference step", self.hessian_step),
            ("maximum move", self.maximum_move),
            ("IRC step", self.irc_step),
            ("IRC force tolerance", self.irc_force_tolerance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PesExplorationError::InvalidConfig(name));
            }
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

/// Certify a stationary-point index from central differences of PES gradients.
pub fn stationary_index<S: PesSurface + ?Sized>(
    surface: &S,
    coordinates: ArrayView1<f64>,
    step: f64,
    negative_tolerance: f64,
) -> Result<StationaryIndex, PesExplorationError> {
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
    if !negative_tolerance.is_finite() || negative_tolerance < 0.0 {
        return Err(PesExplorationError::InvalidConfig(
            "negative curvature tolerance",
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

    let (eigenvalues, eigenvectors) = exact_eigh(hessian.view())
        .map_err(|error| PesExplorationError::Saddle(error.to_string()))?;
    let negative_modes = eigenvalues
        .iter()
        .filter(|eigenvalue| **eigenvalue < -negative_tolerance)
        .count();
    Ok(StationaryIndex {
        lowest_mode: eigenvectors.column(0).to_owned(),
        eigenvalues,
        negative_modes,
    })
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
    let saddle_start = &origin_minimum.coordinates + &(&mode * config.saddle_displacement);
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
