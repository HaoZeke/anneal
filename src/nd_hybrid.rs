//! Same-surface hybrid exploration for arbitrary-dimensional energy landscapes.
//!
//! Basin escapes generate force-certified sources and minimum-mode rides turn
//! those sources into index-one connections. Separate action-outcome GPs rank
//! the next finite experiment by joint information about the identity and
//! energy of the lowest reachable minimum per charged PES evaluation. Catalog
//! admission is a producer--consumer barrier: an escape-discovered source is
//! acknowledged by the ridge segment before escapes resume. The returned
//! network belongs to one caller-supplied surface and witness; no identity,
//! energy model, or evidence crosses between systems.

use std::collections::{HashMap, HashSet, VecDeque};

use ndarray::{Array1, ArrayView1};
use rand::{rngs::StdRng, SeedableRng};

use crate::allocate::DiscoveryAccounting;
use crate::catalog::{
    leftover_esty_stable, leftover_esty_upper, PRODUCTION_MAX_UNSEEN_MASS,
    PRODUCTION_MINIMUM_VISITS,
};
use crate::descriptor_space::{DescriptorError, DescriptorSpace};
use crate::methods::minima_hopping::EscapeFeedback;
use crate::minimum_information::{
    MinimumInformationError, MinimumInformationSearch, SearchActionCandidate, SearchActionScore,
    SearchMechanism,
};
use crate::movekernel::{Gaussian, MoveKernel, TsallisVisit};
use crate::pes_exploration::{
    discover_nd_connection_with_budget, orthonormal_nd_mode, ExactStructureWitness, NdPesNetwork,
    PesExplorationConfig, PesExplorationError, PesSurface, RideModeDirection,
};
use crate::source_escape::{quench_source_escape, SourceEscapeConfig, SourceEscapeOutcome};

const RIDGE_ARM: usize = 0;
const ESCAPE_ARM: usize = 1;
const GAUSSIAN_MOVE: usize = 0;
const TSALLIS_MOVE: usize = 1;
const MINIMUM_INFORMATION_SAMPLES: usize = 128;

/// Failure to construct a stable action coordinate for the outcome model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ActionFeatureError {
    /// Generic action points must contain at least one coordinate.
    #[error("action point must be nonempty")]
    EmptyPoint,
    /// Generic action points cannot contain NaN or infinity.
    #[error("nonfinite action coordinate at index {index}")]
    NonFiniteCoordinate {
        /// Index of the first invalid coordinate.
        index: usize,
    },
    /// An invariant atomistic descriptor rejected the point or species.
    #[error(transparent)]
    Descriptor(#[from] DescriptorError),
}

/// Maps a concrete search point to the stable coordinates of an action GP.
pub trait ActionFeatureMap {
    /// Encode one proposed or ridge-displaced point.
    fn encode(&self, point: ArrayView1<'_, f64>) -> Result<Vec<f64>, ActionFeatureError>;
}

/// Identity feature map for generic N-dimensional objective functions.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoordinateActionFeatures;

impl ActionFeatureMap for CoordinateActionFeatures {
    fn encode(&self, point: ArrayView1<'_, f64>) -> Result<Vec<f64>, ActionFeatureError> {
        if point.is_empty() {
            return Err(ActionFeatureError::EmptyPoint);
        }
        if let Some(index) = point.iter().position(|coordinate| !coordinate.is_finite()) {
            return Err(ActionFeatureError::NonFiniteCoordinate { index });
        }
        Ok(point.to_vec())
    }
}

/// Rotation-, translation-, and like-species-permutation-invariant atomistic map.
#[derive(Debug, Clone)]
pub struct DescriptorActionFeatures {
    descriptor_space: DescriptorSpace,
    species: Vec<u32>,
}

impl DescriptorActionFeatures {
    /// Bind one immutable descriptor geometry and ordered species vector.
    pub fn new(descriptor_space: DescriptorSpace, species: Vec<u32>) -> Self {
        Self {
            descriptor_space,
            species,
        }
    }
}

impl ActionFeatureMap for DescriptorActionFeatures {
    fn encode(&self, point: ArrayView1<'_, f64>) -> Result<Vec<f64>, ActionFeatureError> {
        Ok(self
            .descriptor_space
            .describe(point, Some(&self.species))?
            .values()
            .to_vec())
    }
}

/// Controls for one system-local generic PES exploration campaign.
#[derive(Debug, Clone)]
pub struct NdHybridConfig {
    /// Hard PES-evaluation budget, including the initial source quench.
    pub evaluation_budget: u64,
    /// Largest call slice assigned to one ridge ride.
    pub ride_evaluation_cap: u64,
    /// Largest call slice assigned to one perturb--quench escape.
    pub escape_evaluation_cap: u64,
    /// Seeded orthonormal mode blocks explored per source.
    pub ride_mode_blocks: u16,
    /// Initial scale of Gaussian and Tsallis source proposals.
    pub initial_escape_scale: f64,
    /// Initial Goedecker acceptance threshold on quenched energy rises.
    pub initial_acceptance_threshold: f64,
    /// Tsallis visiting index in `(1, 3)` for the heavy-tailed escape arm.
    pub visiting_q: f64,
    /// rgmin, rgsaddle, P-RFO, and branch-certification controls.
    pub exploration: PesExplorationConfig,
}

/// Discovery mechanism responsible for one charged exploration event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdHybridMechanism {
    /// Minimum-mode ridge following from a shared exact minimum.
    Ridge,
    /// Gaussian or heavy-tailed proposal followed by an rgmin quench.
    BasinEscape,
}

/// Proposal kernel drawn for a generic basin-escape event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdEscapeKernel {
    /// Finite-variance local displacement.
    Gaussian,
    /// Heavy-tailed generalized-simulated-annealing displacement.
    Tsallis,
}

/// Mechanism policy used for matched-budget comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdHybridPolicy {
    /// Allocate every concrete action by minimum-energy information per cost.
    Adaptive,
    /// Explore only the finite signed mode portfolio of discovered sources.
    RidgeOnly,
    /// Generate exact minima by perturb--quench escapes without riding them.
    BasinEscapeOnly,
}

/// One observable mechanism decision and its stationary-point yield.
#[derive(Debug)]
pub struct NdHybridEvent {
    /// Monotonic event index after the initial source quench.
    pub attempt: u64,
    /// Mechanism selected by the declared discovery policy.
    pub mechanism: NdHybridMechanism,
    /// Best ridge-action minimum-information rate at this decision.
    pub ridge_information_rate: Option<f64>,
    /// Best basin-action minimum-information rate at this decision.
    pub escape_information_rate: Option<f64>,
    /// GIBBON information of the concrete action that was evaluated.
    pub selected_information: f64,
    /// Selected information divided by its expected charged PES cost.
    pub selected_information_rate: f64,
    /// Exact source basin for the proposal or ridge.
    pub source_basin: Option<usize>,
    /// Exact energy of the source minimum used by the action model.
    pub source_energy: f64,
    /// Lowest terminal minimum energy returned by this action.
    pub terminal_energy: f64,
    /// Mode rank for ridge events.
    pub mode_rank: Option<u16>,
    /// Signed initialization for ridge events.
    pub direction: Option<RideModeDirection>,
    /// Proposal kernel for basin-escape events.
    pub escape_kernel: Option<NdEscapeKernel>,
    /// Exact minimum identities introduced by this event.
    pub new_minimum_ids: Vec<usize>,
    /// Certified saddle identities introduced by this event.
    pub new_saddle_ids: Vec<usize>,
    /// Unresolved index-one saddle identities introduced by this event.
    pub new_unresolved_saddle_ids: Vec<usize>,
    /// Successful source quenches included in the exact-basin census.
    pub escape_observations: u64,
    /// Esty one-sided upper bound on unseen exact-basin mass.
    pub escape_unseen_mass_upper: Option<f64>,
    /// Whether the source census meets its visit and unseen-mass criteria.
    pub escape_coverage_saturated: bool,
    /// PES evaluations charged to this event.
    pub charged_evaluations: u64,
    /// Whether the selected numerical path returned its certified object.
    pub converged: bool,
    /// Whether the event attempted to step beyond its assigned slice.
    pub budget_exhausted: bool,
    /// Stable numerical failure text, when no certified object was returned.
    pub error: Option<String>,
}

/// Why a hybrid campaign stopped issuing PES evaluations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdHybridTermination {
    /// The matched potential-call budget was consumed.
    BudgetConsumed,
    /// A numerical path returned without issuing a PES evaluation.
    NoPesProgress,
    /// Every configured signed ridge mode has been attempted.
    RidePortfolioExhausted,
}

/// Exact stationary network and allocation evidence from one surface.
#[derive(Debug)]
pub struct NdHybridReport {
    /// System-local exact minimum--saddle graph.
    pub network: NdPesNetwork,
    /// Adaptive or fixed mechanism policy used by the campaign.
    pub policy: NdHybridPolicy,
    /// Total PES evaluations, including the initial source quench.
    pub charged_evaluations: u64,
    /// Ordered mechanism evidence.
    pub events: Vec<NdHybridEvent>,
    /// Ridge then basin-escape exposure counts.
    pub mechanism_pulls: Vec<usize>,
    /// Empirical distinct-stationary-object discoveries per PES evaluation.
    pub mechanism_discovery_rates: Vec<f64>,
    /// Gaussian then Tsallis proposal exposure counts.
    pub move_pulls: Vec<usize>,
    /// Empirical probability that each escape proposal lowers source energy.
    pub move_success_rates: Vec<f64>,
    /// Successful source quenches included in the exact-basin census.
    pub escape_observations: u64,
    /// Exact basins represented by one source-quench observation.
    pub escape_singletons: u64,
    /// Esty one-sided upper bound on unseen exact-basin mass.
    pub escape_unseen_mass_upper: Option<f64>,
    /// Whether the source census meets its visit and unseen-mass criteria.
    pub escape_coverage_saturated: bool,
    /// Terminal budget condition.
    pub termination: NdHybridTermination,
}

/// Configuration or initial-source failure that prevents a campaign.
#[derive(Debug, thiserror::Error)]
pub enum NdHybridError {
    /// A hybrid control lies outside its numerical domain.
    #[error("invalid N-dimensional hybrid configuration: {0}")]
    InvalidConfig(&'static str),
    /// The initial point did not produce a force-certified source minimum.
    #[error("initial N-dimensional source failed after {charged_evaluations} PES calls: {error}")]
    InitialSource {
        /// Stable quench failure text.
        error: String,
        /// PES calls consumed before the failure.
        charged_evaluations: u64,
    },
    /// A deterministic mode could not be constructed.
    #[error(transparent)]
    Mode(#[from] PesExplorationError),
    /// The action-outcome model rejected an invalid feature or score.
    #[error(transparent)]
    MinimumInformation(#[from] MinimumInformationError),
    /// The action representation rejected a concrete point.
    #[error(transparent)]
    ActionFeature(#[from] ActionFeatureError),
}

fn validate(config: &NdHybridConfig, dimension: usize) -> Result<(), NdHybridError> {
    if dimension == 0 {
        return Err(NdHybridError::InvalidConfig(
            "the point dimension must be positive",
        ));
    }
    if config.evaluation_budget == 0
        || config.ride_evaluation_cap == 0
        || config.escape_evaluation_cap == 0
    {
        return Err(NdHybridError::InvalidConfig(
            "evaluation budgets must be positive",
        ));
    }
    if config.ride_mode_blocks == 0 {
        return Err(NdHybridError::InvalidConfig(
            "at least one ride mode block is required",
        ));
    }
    if dimension
        .checked_mul(usize::from(config.ride_mode_blocks))
        .is_none_or(|ranks| ranks > usize::from(u16::MAX) + 1)
    {
        return Err(NdHybridError::InvalidConfig(
            "the mode portfolio exceeds the rank domain",
        ));
    }
    if !config.initial_escape_scale.is_finite() || config.initial_escape_scale <= 0.0 {
        return Err(NdHybridError::InvalidConfig(
            "the initial escape scale must be positive and finite",
        ));
    }
    if !config.initial_acceptance_threshold.is_finite()
        || config.initial_acceptance_threshold <= 0.0
    {
        return Err(NdHybridError::InvalidConfig(
            "the initial acceptance threshold must be positive and finite",
        ));
    }
    if !config.visiting_q.is_finite() || config.visiting_q <= 1.0 || config.visiting_q >= 3.0 {
        return Err(NdHybridError::InvalidConfig(
            "the Tsallis visiting index must lie in (1, 3)",
        ));
    }
    if !config.exploration.quench_gradient_tolerance.is_finite()
        || config.exploration.quench_gradient_tolerance <= 0.0
    {
        return Err(NdHybridError::InvalidConfig(
            "the quench gradient tolerance must be positive and finite",
        ));
    }
    Ok(())
}

fn ride_tasks(
    minima: usize,
    attempted: &HashSet<(usize, u16, RideModeDirection)>,
    source_cursor: usize,
    ranks: usize,
) -> Vec<(usize, u16, RideModeDirection)> {
    if minima == 0 {
        return Vec::new();
    }
    let mut tasks = Vec::new();
    for offset in 0..minima {
        let basin = (source_cursor + offset) % minima;
        for rank in 0..ranks {
            let rank = u16::try_from(rank).expect("validated mode portfolio fits u16");
            for direction in [RideModeDirection::Positive, RideModeDirection::Negative] {
                let task = (basin, rank, direction);
                if !attempted.contains(&task) {
                    tasks.push(task);
                }
            }
        }
    }
    tasks
}

fn new_ids(before: usize, after: usize) -> Vec<usize> {
    (before..after).collect()
}

#[derive(Debug, Clone, Copy)]
struct EscapeCoverageEvidence {
    observations: u64,
    singletons: u64,
    unseen_mass_upper: Option<f64>,
    saturated: bool,
}

#[derive(Debug, Default)]
struct EscapeCoverage {
    visits: HashMap<usize, u64>,
    observations: u64,
}

impl EscapeCoverage {
    fn observe(&mut self, basin: usize) {
        self.observations = self.observations.saturating_add(1);
        let visits = self.visits.entry(basin).or_insert(0);
        *visits = visits.saturating_add(1);
    }

    fn evidence(&self) -> EscapeCoverageEvidence {
        let singletons = self.visits.values().filter(|visits| **visits == 1).count() as u64;
        let doubletons = self.visits.values().filter(|visits| **visits == 2).count() as u64;
        let unseen_mass_upper = leftover_esty_upper(self.observations, singletons, doubletons);
        let saturated = self.observations >= PRODUCTION_MINIMUM_VISITS
            && leftover_esty_stable(
                self.observations,
                singletons,
                doubletons,
                PRODUCTION_MAX_UNSEEN_MASS,
            );
        EscapeCoverageEvidence {
            observations: self.observations,
            singletons,
            unseen_mass_upper,
            saturated,
        }
    }
}

fn basin_action_feature<F>(
    action_features: &F,
    proposal: ArrayView1<'_, f64>,
    kernel: NdEscapeKernel,
) -> Result<Vec<f64>, ActionFeatureError>
where
    F: ActionFeatureMap + ?Sized,
{
    let mut feature = action_features.encode(proposal)?;
    feature.extend(match kernel {
        NdEscapeKernel::Gaussian => [1.0, 0.0],
        NdEscapeKernel::Tsallis => [0.0, 1.0],
    });
    Ok(feature)
}

fn ridge_action_feature<F>(
    action_features: &F,
    source: ArrayView1<'_, f64>,
    mode: ArrayView1<'_, f64>,
    displacement: f64,
) -> Result<Vec<f64>, ActionFeatureError>
where
    F: ActionFeatureMap + ?Sized,
{
    let displaced = source
        .iter()
        .zip(mode.iter())
        .map(|(coordinate, component)| coordinate + displacement * component)
        .collect::<Array1<_>>();
    action_features.encode(displaced.view())
}

fn empirical_cost(charged: u64, pulls: usize) -> Option<f64> {
    (charged > 0 && pulls > 0).then(|| charged as f64 / pulls as f64)
}

fn expected_cost(charged: u64, pulls: usize, fallback: Option<f64>, maximum: u64) -> f64 {
    empirical_cost(charged, pulls)
        .or(fallback)
        .unwrap_or(maximum.max(1) as f64)
        .clamp(1.0, maximum.max(1) as f64)
}

fn best_information_rate(scores: &[SearchActionScore], mechanism: SearchMechanism) -> Option<f64> {
    scores
        .iter()
        .filter(|score| score.mechanism == mechanism)
        .map(|score| score.information_per_charged_evaluation)
        .max_by(f64::total_cmp)
}

#[derive(Debug)]
enum PlannedAction {
    Ridge {
        source_basin: usize,
        source_energy: f64,
        mode_rank: u16,
        direction: RideModeDirection,
        source: Array1<f64>,
        mode: Array1<f64>,
        feature: Vec<f64>,
    },
    Escape {
        source_basin: usize,
        source_energy: f64,
        move_index: usize,
        kernel: NdEscapeKernel,
        proposal: Array1<f64>,
        feature: Vec<f64>,
    },
}

impl PlannedAction {
    fn candidate(&self, expected_charged_evaluations: f64) -> SearchActionCandidate {
        match self {
            Self::Ridge {
                source_energy,
                feature,
                ..
            } => SearchActionCandidate {
                mechanism: SearchMechanism::SaddleRide,
                feature: feature.clone(),
                source_energy: *source_energy,
                expected_charged_evaluations,
            },
            Self::Escape {
                source_energy,
                feature,
                ..
            } => SearchActionCandidate {
                mechanism: SearchMechanism::BasinEscape,
                feature: feature.clone(),
                source_energy: *source_energy,
                expected_charged_evaluations,
            },
        }
    }

    fn admissible(&self, active_ride_source: Option<usize>) -> bool {
        match self {
            Self::Ridge { source_basin, .. } => Some(*source_basin) == active_ride_source,
            Self::Escape { .. } => true,
        }
    }
}

/// Explore one arbitrary-dimensional PES with cooperative basin and ridge arms.
///
/// Every discovered minimum enters one exact-witness network immediately. The
/// ridge action domain dovetails over source-minimum segments and uses joint
/// minimum information to order the signed modes inside each segment. This
/// makes every admitted source reachable without assigning graph novelty or a
/// stationary-point reward. A new invocation creates new action models and a
/// new network, so two surfaces cannot share evidence.
pub fn explore_nd_hybrid<S, W>(
    surface: &S,
    initial: ArrayView1<'_, f64>,
    config: &NdHybridConfig,
    witness: &W,
    seed: u64,
) -> Result<NdHybridReport, NdHybridError>
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
{
    explore_nd_with_policy(
        surface,
        initial,
        config,
        witness,
        seed,
        NdHybridPolicy::Adaptive,
    )
}

/// Explore one PES under an adaptive or fixed matched-budget mechanism policy.
///
/// Fixed policies use the same quench, ridge, witness, and accounting paths as
/// the adaptive campaign, isolating the contribution of mechanism allocation.
pub fn explore_nd_with_policy<S, W>(
    surface: &S,
    initial: ArrayView1<'_, f64>,
    config: &NdHybridConfig,
    witness: &W,
    seed: u64,
    policy: NdHybridPolicy,
) -> Result<NdHybridReport, NdHybridError>
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
{
    explore_nd_with_policy_and_features(
        surface,
        initial,
        config,
        witness,
        seed,
        policy,
        &CoordinateActionFeatures,
    )
}

/// Explore one PES with an explicit action representation.
///
/// Generic objectives use [`CoordinateActionFeatures`]. Atomistic callers use
/// [`DescriptorActionFeatures`] so the outcome posterior compares physical
/// structures independently of rigid coordinates and like-species labels.
pub fn explore_nd_with_policy_and_features<S, W, F>(
    surface: &S,
    initial: ArrayView1<'_, f64>,
    config: &NdHybridConfig,
    witness: &W,
    seed: u64,
    policy: NdHybridPolicy,
    action_features: &F,
) -> Result<NdHybridReport, NdHybridError>
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
    F: ActionFeatureMap + ?Sized,
{
    validate(config, initial.len())?;
    action_features.encode(initial)?;
    let dimension = initial.len();
    let norm_tolerance = config
        .exploration
        .quench_gradient_norm_tolerance
        .unwrap_or(config.exploration.quench_gradient_tolerance * (dimension as f64).sqrt());
    let initial_config = SourceEscapeConfig {
        maximum_evaluations: config.evaluation_budget.min(config.escape_evaluation_cap),
        quench_steps: config.exploration.quench_steps,
        gradient_tolerance: config.exploration.quench_gradient_tolerance,
        gradient_norm_tolerance: norm_tolerance,
    };
    let initial_record = match quench_source_escape(surface, initial, &initial_config) {
        SourceEscapeOutcome::Converged(record) => record,
        SourceEscapeOutcome::Failed(failure) => {
            return Err(NdHybridError::InitialSource {
                error: failure.error,
                charged_evaluations: failure.charged_evaluations,
            });
        }
    };
    let mut charged_evaluations = initial_record.charged_evaluations;
    let mut network = NdPesNetwork::new();
    let initial_admission = network.admit_minimum(initial_record.minimum, witness);
    let mut live_basin = initial_admission.id;
    let mut live_coordinates = network.minima()[live_basin].coordinates.clone();
    let mut live_energy = network.minima()[live_basin].energy;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut mechanism_accounting = DiscoveryAccounting::new(2);
    mechanism_accounting.observe(ESCAPE_ARM, 1, initial_record.charged_evaluations);
    let information_amplitude = config.initial_acceptance_threshold;
    let information_noise = (information_amplitude * f64::EPSILON.sqrt()).max(f64::MIN_POSITIVE);
    let mut minimum_information = MinimumInformationSearch::new(
        config.initial_escape_scale * (dimension as f64).sqrt(),
        information_amplitude,
        information_noise,
    )?;
    let mut move_pulls = [0usize; 2];
    let mut move_improvements = [0u64; 2];
    let mut move_charged = [0u64; 2];
    let mut escape_feedback = EscapeFeedback::new(
        config.initial_escape_scale,
        config.initial_acceptance_threshold,
    );
    escape_feedback.observe(None, live_basin);
    let mut escape_coverage = EscapeCoverage::default();
    escape_coverage.observe(live_basin);
    let tsallis = TsallisVisit::new(config.visiting_q);
    let ranks = dimension * usize::from(config.ride_mode_blocks);
    let mut attempted = HashSet::new();
    let mut pending_ridge_sources = VecDeque::new();
    let mut source_cursor = 0usize;
    let mut events = Vec::new();
    let mut attempt = 0u64;
    let termination = loop {
        let remaining = config.evaluation_budget.saturating_sub(charged_evaluations);
        if remaining == 0 {
            break NdHybridTermination::BudgetConsumed;
        }
        let ride_tasks = ride_tasks(network.minimum_count(), &attempted, source_cursor, ranks);
        if policy == NdHybridPolicy::RidgeOnly && ride_tasks.is_empty() {
            break NdHybridTermination::RidePortfolioExhausted;
        }
        let required_ride_source = (policy == NdHybridPolicy::Adaptive)
            .then(|| pending_ridge_sources.front().copied())
            .flatten();
        let active_ride_source =
            required_ride_source.or_else(|| ride_tasks.first().map(|task| task.0));

        let mut plans = Vec::<PlannedAction>::new();
        if policy != NdHybridPolicy::BasinEscapeOnly {
            for (source_basin, mode_rank, direction) in ride_tasks {
                if required_ride_source.is_some_and(|required| source_basin != required) {
                    continue;
                }
                let source = network.minima()[source_basin].coordinates.clone();
                let source_energy = network.minima()[source_basin].energy;
                let mode_seed =
                    seed ^ (source_basin as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                let mode = orthonormal_nd_mode(dimension, mode_seed, mode_rank, direction)?;
                let feature = ridge_action_feature(
                    action_features,
                    source.view(),
                    mode.view(),
                    config.exploration.saddle_displacement,
                )?;
                plans.push(PlannedAction::Ridge {
                    source_basin,
                    source_energy,
                    mode_rank,
                    direction,
                    source,
                    mode,
                    feature,
                });
            }
        }
        if policy != NdHybridPolicy::RidgeOnly && required_ride_source.is_none() {
            let escape_scale = escape_feedback.escape();
            let gaussian = Gaussian::new(escape_scale).propose(
                live_coordinates.view(),
                escape_scale,
                &mut rng,
            );
            let tsallis_proposal = tsallis.propose(live_coordinates.view(), escape_scale, &mut rng);
            for (move_index, kernel, proposal) in [
                (GAUSSIAN_MOVE, NdEscapeKernel::Gaussian, gaussian),
                (TSALLIS_MOVE, NdEscapeKernel::Tsallis, tsallis_proposal),
            ] {
                plans.push(PlannedAction::Escape {
                    source_basin: live_basin,
                    source_energy: live_energy,
                    move_index,
                    kernel,
                    feature: basin_action_feature(action_features, proposal.view(), kernel)?,
                    proposal,
                });
            }
        }
        let pooled_cost = empirical_cost(
            mechanism_accounting.charged_calls().iter().copied().sum(),
            mechanism_accounting.pulls().iter().copied().sum(),
        );
        let ride_cost = expected_cost(
            mechanism_accounting.charged_calls()[RIDGE_ARM],
            mechanism_accounting.pulls()[RIDGE_ARM],
            pooled_cost,
            remaining.min(config.ride_evaluation_cap),
        );
        let escape_cost = expected_cost(
            mechanism_accounting.charged_calls()[ESCAPE_ARM],
            mechanism_accounting.pulls()[ESCAPE_ARM],
            pooled_cost,
            remaining.min(config.escape_evaluation_cap),
        );
        let candidates = plans
            .iter()
            .map(|plan| match plan {
                PlannedAction::Ridge { .. } => plan.candidate(ride_cost),
                PlannedAction::Escape { move_index, .. } => plan.candidate(expected_cost(
                    move_charged[*move_index],
                    move_pulls[*move_index],
                    Some(escape_cost),
                    remaining.min(config.escape_evaluation_cap),
                )),
            })
            .collect::<Vec<_>>();
        let scores = minimum_information.score(&candidates, MINIMUM_INFORMATION_SAMPLES)?;
        let selected_index = scores
            .iter()
            .enumerate()
            .filter(|(index, _)| plans[*index].admissible(active_ride_source))
            .max_by(|(left_index, left), (right_index, right)| {
                left.information_per_charged_evaluation
                    .total_cmp(&right.information_per_charged_evaluation)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("every admissible policy supplies an action");
        let selected_score = scores[selected_index];
        let ridge_information_rate = scores
            .iter()
            .enumerate()
            .filter(|(index, score)| {
                score.mechanism == SearchMechanism::SaddleRide
                    && plans[*index].admissible(active_ride_source)
            })
            .map(|(_, score)| score.information_per_charged_evaluation)
            .max_by(f64::total_cmp);
        let escape_information_rate = best_information_rate(&scores, SearchMechanism::BasinEscape);
        let selected = plans.swap_remove(selected_index);
        attempt += 1;

        if let PlannedAction::Ridge {
            source_basin,
            source_energy,
            mode_rank,
            direction,
            source,
            mode,
            feature,
        } = selected
        {
            if pending_ridge_sources.front().copied() == Some(source_basin) {
                pending_ridge_sources.pop_front();
            }
            source_cursor = (source_basin + 1) % network.minimum_count();
            attempted.insert((source_basin, mode_rank, direction));
            let minima_before = network.minimum_count();
            let saddles_before = network.saddle_count();
            let unresolved_before = network.unresolved_saddles().len();
            let ridge = discover_nd_connection_with_budget(
                surface,
                &mut network,
                source.view(),
                mode.view(),
                &config.exploration,
                witness,
                remaining.min(config.ride_evaluation_cap),
            );
            let new_minimum_ids = new_ids(minima_before, network.minimum_count());
            let new_saddle_ids = new_ids(saddles_before, network.saddle_count());
            let new_unresolved_saddle_ids =
                new_ids(unresolved_before, network.unresolved_saddles().len());
            let terminal_energy = ridge
                .connection
                .as_ref()
                .ok()
                .into_iter()
                .flat_map(|connection| connection.endpoints)
                .chain(new_minimum_ids.iter().copied())
                .map(|id| network.minima()[id].energy)
                .fold(source_energy, f64::min);
            minimum_information.observe(
                SearchMechanism::SaddleRide,
                &feature,
                source_energy,
                terminal_energy,
            )?;
            let stationary_discoveries = new_minimum_ids
                .len()
                .saturating_add(new_saddle_ids.len())
                .saturating_add(new_unresolved_saddle_ids.len());
            mechanism_accounting.observe(
                RIDGE_ARM,
                u64::try_from(stationary_discoveries).unwrap_or(u64::MAX),
                ridge.charged_evaluations,
            );
            charged_evaluations = charged_evaluations
                .saturating_add(ridge.charged_evaluations)
                .min(config.evaluation_budget);
            let converged = ridge.connection.is_ok();
            let error = ridge.connection.err().map(|error| error.to_string());
            let event_charge = ridge.charged_evaluations;
            let coverage = escape_coverage.evidence();
            events.push(NdHybridEvent {
                attempt,
                mechanism: NdHybridMechanism::Ridge,
                ridge_information_rate,
                escape_information_rate,
                selected_information: selected_score.information,
                selected_information_rate: selected_score.information_per_charged_evaluation,
                source_basin: Some(source_basin),
                source_energy,
                terminal_energy,
                mode_rank: Some(mode_rank),
                direction: Some(direction),
                escape_kernel: None,
                new_minimum_ids,
                new_saddle_ids,
                new_unresolved_saddle_ids,
                escape_observations: coverage.observations,
                escape_unseen_mass_upper: coverage.unseen_mass_upper,
                escape_coverage_saturated: coverage.saturated,
                charged_evaluations: event_charge,
                converged,
                budget_exhausted: ridge.budget_exhausted,
                error,
            });
            if event_charge == 0 {
                break NdHybridTermination::NoPesProgress;
            }
            continue;
        }
        let PlannedAction::Escape {
            source_basin,
            source_energy,
            move_index,
            kernel: escape_kernel,
            proposal,
            feature,
        } = selected
        else {
            unreachable!("the ridge action returns from its branch")
        };
        let escape_config = SourceEscapeConfig {
            maximum_evaluations: remaining.min(config.escape_evaluation_cap),
            quench_steps: config.exploration.quench_steps,
            gradient_tolerance: config.exploration.quench_gradient_tolerance,
            gradient_norm_tolerance: norm_tolerance,
        };
        let escape = quench_source_escape(surface, proposal.view(), &escape_config);
        match escape {
            SourceEscapeOutcome::Failed(failure) => {
                mechanism_accounting.observe(ESCAPE_ARM, 0, failure.charged_evaluations);
                if failure.charged_evaluations > 0 {
                    minimum_information.observe(
                        SearchMechanism::BasinEscape,
                        &feature,
                        source_energy,
                        source_energy,
                    )?;
                }
                move_pulls[move_index] = move_pulls[move_index].saturating_add(1);
                move_charged[move_index] =
                    move_charged[move_index].saturating_add(failure.charged_evaluations);
                charged_evaluations = charged_evaluations
                    .saturating_add(failure.charged_evaluations)
                    .min(config.evaluation_budget);
                let event_charge = failure.charged_evaluations;
                let budget_exhausted = failure.error.contains("budget exhausted");
                let coverage = escape_coverage.evidence();
                events.push(NdHybridEvent {
                    attempt,
                    mechanism: NdHybridMechanism::BasinEscape,
                    ridge_information_rate,
                    escape_information_rate,
                    selected_information: selected_score.information,
                    selected_information_rate: selected_score.information_per_charged_evaluation,
                    source_basin: Some(source_basin),
                    source_energy,
                    terminal_energy: source_energy,
                    mode_rank: None,
                    direction: None,
                    escape_kernel: Some(escape_kernel),
                    new_minimum_ids: Vec::new(),
                    new_saddle_ids: Vec::new(),
                    new_unresolved_saddle_ids: Vec::new(),
                    escape_observations: coverage.observations,
                    escape_unseen_mass_upper: coverage.unseen_mass_upper,
                    escape_coverage_saturated: coverage.saturated,
                    charged_evaluations: event_charge,
                    converged: false,
                    budget_exhausted,
                    error: Some(failure.error),
                });
                if event_charge == 0 {
                    break NdHybridTermination::NoPesProgress;
                }
            }
            SourceEscapeOutcome::Converged(record) => {
                let candidate_energy = record.minimum.energy;
                let candidate_coordinates = record.minimum.coordinates.clone();
                let admission = network.admit_minimum(record.minimum, witness);
                let discovered = admission.is_new;
                let new_minimum_ids = discovered.then_some(admission.id).into_iter().collect();
                if discovered && policy == NdHybridPolicy::Adaptive {
                    pending_ridge_sources.push_back(admission.id);
                }
                minimum_information.observe(
                    SearchMechanism::BasinEscape,
                    &feature,
                    source_energy,
                    candidate_energy,
                )?;
                escape_feedback.observe(Some(live_basin), admission.id);
                escape_coverage.observe(admission.id);
                let accepted = escape_feedback.accept(candidate_energy - live_energy);
                if accepted {
                    live_basin = admission.id;
                    live_energy = candidate_energy;
                    live_coordinates = candidate_coordinates;
                }
                mechanism_accounting.observe(
                    ESCAPE_ARM,
                    u64::from(discovered),
                    record.charged_evaluations,
                );
                move_pulls[move_index] = move_pulls[move_index].saturating_add(1);
                move_charged[move_index] =
                    move_charged[move_index].saturating_add(record.charged_evaluations);
                if candidate_energy < source_energy {
                    move_improvements[move_index] = move_improvements[move_index].saturating_add(1);
                }
                charged_evaluations = charged_evaluations
                    .saturating_add(record.charged_evaluations)
                    .min(config.evaluation_budget);
                let event_charge = record.charged_evaluations;
                let coverage = escape_coverage.evidence();
                events.push(NdHybridEvent {
                    attempt,
                    mechanism: NdHybridMechanism::BasinEscape,
                    ridge_information_rate,
                    escape_information_rate,
                    selected_information: selected_score.information,
                    selected_information_rate: selected_score.information_per_charged_evaluation,
                    source_basin: Some(source_basin),
                    source_energy,
                    terminal_energy: candidate_energy,
                    mode_rank: None,
                    direction: None,
                    escape_kernel: Some(escape_kernel),
                    new_minimum_ids,
                    new_saddle_ids: Vec::new(),
                    new_unresolved_saddle_ids: Vec::new(),
                    escape_observations: coverage.observations,
                    escape_unseen_mass_upper: coverage.unseen_mass_upper,
                    escape_coverage_saturated: coverage.saturated,
                    charged_evaluations: event_charge,
                    converged: true,
                    budget_exhausted: false,
                    error: None,
                });
                if event_charge == 0 {
                    break NdHybridTermination::NoPesProgress;
                }
            }
        }
    };

    let coverage = escape_coverage.evidence();
    Ok(NdHybridReport {
        network,
        policy,
        charged_evaluations,
        events,
        mechanism_pulls: mechanism_accounting.pulls().to_vec(),
        mechanism_discovery_rates: mechanism_accounting.rates(),
        move_pulls: move_pulls.to_vec(),
        move_success_rates: move_improvements
            .iter()
            .zip(move_pulls)
            .map(|(improvements, pulls)| {
                if pulls == 0 {
                    0.0
                } else {
                    *improvements as f64 / pulls as f64
                }
            })
            .collect(),
        escape_observations: coverage.observations,
        escape_singletons: coverage.singletons,
        escape_unseen_mass_upper: coverage.unseen_mass_upper,
        escape_coverage_saturated: coverage.saturated,
        termination,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unmeasured_action_inherits_observed_same_pes_cost() {
        assert_eq!(expected_cost(0, 0, Some(37.0), 3_000), 37.0);
        assert_eq!(expected_cost(900, 3, Some(37.0), 3_000), 300.0);
        assert_eq!(expected_cost(0, 0, None, 3_000), 3_000.0);
        assert_eq!(expected_cost(9_000, 2, Some(37.0), 3_000), 3_000.0);
    }

    #[test]
    fn the_finite_ridge_domain_contains_every_unattempted_signed_mode() {
        let attempted = HashSet::from([(0, 0, RideModeDirection::Positive)]);

        let tasks = ride_tasks(2, &attempted, 1, 3);

        assert_eq!(tasks.len(), 2 * 3 * 2 - 1);
        assert_eq!(tasks[0], (1, 0, RideModeDirection::Positive));
        assert!(!tasks.contains(&(0, 0, RideModeDirection::Positive)));
        assert!(tasks.contains(&(0, 0, RideModeDirection::Negative)));
        assert!(tasks.contains(&(1, 2, RideModeDirection::Positive)));
        assert!(tasks.contains(&(1, 2, RideModeDirection::Negative)));
    }

    #[test]
    fn dovetail_marks_one_source_segment_admissible_at_a_time() {
        let tasks = ride_tasks(3, &HashSet::new(), 1, 2);
        let active_source = tasks.first().map(|task| task.0);
        let plans = [
            PlannedAction::Ridge {
                source_basin: 1,
                source_energy: 0.0,
                mode_rank: 0,
                direction: RideModeDirection::Positive,
                source: Array1::zeros(2),
                mode: Array1::zeros(2),
                feature: vec![0.0, 0.0],
            },
            PlannedAction::Ridge {
                source_basin: 2,
                source_energy: 0.0,
                mode_rank: 0,
                direction: RideModeDirection::Positive,
                source: Array1::zeros(2),
                mode: Array1::zeros(2),
                feature: vec![0.0, 0.0],
            },
        ];

        assert_eq!(active_source, Some(1));
        assert!(plans[0].admissible(active_source));
        assert!(!plans[1].admissible(active_source));
    }
}
