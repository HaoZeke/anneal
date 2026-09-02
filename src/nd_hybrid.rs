//! Same-surface hybrid exploration for arbitrary-dimensional energy landscapes.
//!
//! Basin escapes generate force-certified sources and minimum-mode rides turn
//! those sources into index-one connections. Good-UCB assigns each successive
//! experiment from exact basin and saddle singleton occupancy. The returned
//! network belongs to one caller-supplied surface and witness; no identity,
//! energy model, or evidence crosses between systems.

use std::collections::{HashMap, HashSet};

use ndarray::{Array1, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};

use crate::allocate::{DiscoveryAccounting, Exp3Ix};
use crate::catalog::{
    PRODUCTION_MAX_UNSEEN_MASS, PRODUCTION_MINIMUM_VISITS, leftover_esty_stable,
    leftover_esty_upper,
};
use crate::discovery_roster::good_ucb_missing_mass_index;
use crate::methods::minima_hopping::EscapeFeedback;
use crate::movekernel::{Gaussian, MoveKernel, TsallisVisit};
use crate::pes_exploration::{
    ExactStructureWitness, NdPesNetwork, PesExplorationConfig, PesExplorationError, PesSurface,
    RideModeDirection, discover_nd_connection_with_budget, orthonormal_nd_mode,
};
use crate::source_escape::{SourceEscapeConfig, SourceEscapeOutcome, quench_source_escape};

const RIDGE_ARM: usize = 0;
const ESCAPE_ARM: usize = 1;
const GAUSSIAN_MOVE: usize = 0;
const TSALLIS_MOVE: usize = 1;

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
    /// Allocate between ridge and escape by exact-species Good-UCB.
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
    /// Good-UCB exact-saddle missing-mass index at this decision.
    pub ridge_discovery_index: Option<f64>,
    /// Good-UCB exact-basin missing-mass index at this decision.
    pub escape_discovery_index: Option<f64>,
    /// Exact source basin for the proposal or ridge.
    pub source_basin: Option<usize>,
    /// Mode rank for ridge events.
    pub mode_rank: Option<u16>,
    /// Signed initialization for ridge events.
    pub direction: Option<RideModeDirection>,
    /// Proposal kernel for basin-escape events.
    pub escape_kernel: Option<NdEscapeKernel>,
    /// EXP3-IX draw probability for the selected proposal kernel.
    pub kernel_probability: Option<f64>,
    /// EXP3-IX anytime learning rate `eta_t`.
    pub kernel_learning_rate: Option<f64>,
    /// EXP3-IX implicit-exploration term `gamma_t`.
    pub kernel_implicit_exploration: Option<f64>,
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
    /// Empirical exact-basin discovery rates for the escape proposals.
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

fn next_ride_task(
    network: &NdPesNetwork,
    attempted: &HashSet<(usize, u16, RideModeDirection)>,
    source_cursor: usize,
    ranks: usize,
) -> Option<(usize, u16, RideModeDirection)> {
    let minima = network.minimum_count();
    if minima == 0 {
        return None;
    }
    for offset in 0..minima {
        let basin = (source_cursor + offset) % minima;
        for rank in 0..ranks {
            let rank = u16::try_from(rank).ok()?;
            for direction in [RideModeDirection::Positive, RideModeDirection::Negative] {
                let task = (basin, rank, direction);
                if !attempted.contains(&task) {
                    return Some(task);
                }
            }
        }
    }
    None
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

fn good_ucb_mechanism(
    escape: EscapeCoverageEvidence,
    network: &NdPesNetwork,
    pulls: [usize; 2],
) -> (usize, [f64; 2]) {
    let ridge_pulls = u64::try_from(pulls[RIDGE_ARM]).unwrap_or(u64::MAX);
    let escape_pulls = u64::try_from(pulls[ESCAPE_ARM]).unwrap_or(u64::MAX);
    let total = ridge_pulls.saturating_add(escape_pulls);
    let ridge_index = good_ucb_missing_mass_index(network.saddle_singletons(), ridge_pulls, total);
    let escape_index = good_ucb_missing_mass_index(escape.singletons, escape_pulls, total);
    let selected = match ridge_index.total_cmp(&escape_index) {
        std::cmp::Ordering::Greater => RIDGE_ARM,
        std::cmp::Ordering::Less => ESCAPE_ARM,
        std::cmp::Ordering::Equal if ridge_pulls <= escape_pulls => RIDGE_ARM,
        std::cmp::Ordering::Equal => ESCAPE_ARM,
    };
    (selected, [ridge_index, escape_index])
}

/// Explore one arbitrary-dimensional PES with cooperative basin and ridge arms.
///
/// Every discovered minimum enters one exact-witness network immediately. The
/// round-robin ridge scheduler can therefore consume a basin found by any
/// escape event, while exact basin and saddle occupancy determines the next
/// Good-UCB mechanism. A new invocation creates a new network, so two different
/// surfaces cannot share basins, saddles, energies, or calibration.
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
    validate(config, initial.len())?;
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
    let mut move_allocator = Exp3Ix::new(2);
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
    let mut source_cursor = 0usize;
    let mut events = Vec::new();
    let mut attempt = 0u64;
    let termination = loop {
        let remaining = config.evaluation_budget.saturating_sub(charged_evaluations);
        if remaining == 0 {
            break NdHybridTermination::BudgetConsumed;
        }
        let ride_task = next_ride_task(&network, &attempted, source_cursor, ranks);
        let coverage = escape_coverage.evidence();
        let (selected, discovery_indices) = match policy {
            NdHybridPolicy::Adaptive if ride_task.is_some() => {
                let pulls = mechanism_accounting.pulls();
                let (selected, indices) =
                    good_ucb_mechanism(coverage, &network, [pulls[RIDGE_ARM], pulls[ESCAPE_ARM]]);
                (selected, Some(indices))
            }
            NdHybridPolicy::Adaptive => (ESCAPE_ARM, None),
            NdHybridPolicy::RidgeOnly if ride_task.is_some() => (RIDGE_ARM, None),
            NdHybridPolicy::RidgeOnly => {
                break NdHybridTermination::RidePortfolioExhausted;
            }
            NdHybridPolicy::BasinEscapeOnly => (ESCAPE_ARM, None),
        };
        attempt += 1;

        if selected == RIDGE_ARM {
            let (source_basin, mode_rank, direction) = ride_task.expect("ride task is available");
            source_cursor = (source_basin + 1) % network.minimum_count();
            attempted.insert((source_basin, mode_rank, direction));
            let source = network.minima()[source_basin].coordinates.clone();
            let mode_seed = seed ^ (source_basin as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let mode = orthonormal_nd_mode(dimension, mode_seed, mode_rank, direction)?;
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
                ridge_discovery_index: discovery_indices.map(|indices| indices[RIDGE_ARM]),
                escape_discovery_index: discovery_indices.map(|indices| indices[ESCAPE_ARM]),
                source_basin: Some(source_basin),
                mode_rank: Some(mode_rank),
                direction: Some(direction),
                escape_kernel: None,
                kernel_probability: None,
                kernel_learning_rate: None,
                kernel_implicit_exploration: None,
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

        let kernel_selection = move_allocator.select(&mut rng);
        let move_index = kernel_selection.arm();
        let escape_kernel = match move_index {
            GAUSSIAN_MOVE => NdEscapeKernel::Gaussian,
            TSALLIS_MOVE => NdEscapeKernel::Tsallis,
            _ => unreachable!("the escape portfolio has exactly two moves"),
        };
        let escape_scale = escape_feedback.escape();
        let proposal: Array1<f64> = match move_index {
            GAUSSIAN_MOVE => {
                Gaussian::new(escape_scale).propose(live_coordinates.view(), escape_scale, &mut rng)
            }
            TSALLIS_MOVE => tsallis.propose(live_coordinates.view(), escape_scale, &mut rng),
            _ => unreachable!("the escape portfolio has exactly two moves"),
        };
        let escape_config = SourceEscapeConfig {
            maximum_evaluations: remaining.min(config.escape_evaluation_cap),
            quench_steps: config.exploration.quench_steps,
            gradient_tolerance: config.exploration.quench_gradient_tolerance,
            gradient_norm_tolerance: norm_tolerance,
        };
        let source_basin = live_basin;
        let escape = quench_source_escape(surface, proposal.view(), &escape_config);
        match escape {
            SourceEscapeOutcome::Failed(failure) => {
                mechanism_accounting.observe(ESCAPE_ARM, 0, failure.charged_evaluations);
                move_allocator.update(1.0);
                charged_evaluations = charged_evaluations
                    .saturating_add(failure.charged_evaluations)
                    .min(config.evaluation_budget);
                let event_charge = failure.charged_evaluations;
                let budget_exhausted = failure.error.contains("budget exhausted");
                let coverage = escape_coverage.evidence();
                events.push(NdHybridEvent {
                    attempt,
                    mechanism: NdHybridMechanism::BasinEscape,
                    ridge_discovery_index: discovery_indices.map(|indices| indices[RIDGE_ARM]),
                    escape_discovery_index: discovery_indices.map(|indices| indices[ESCAPE_ARM]),
                    source_basin: Some(source_basin),
                    mode_rank: None,
                    direction: None,
                    escape_kernel: Some(escape_kernel),
                    kernel_probability: Some(kernel_selection.probability()),
                    kernel_learning_rate: Some(kernel_selection.learning_rate()),
                    kernel_implicit_exploration: Some(kernel_selection.implicit_exploration()),
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
                move_allocator.update(if discovered { 0.0 } else { 1.0 });
                charged_evaluations = charged_evaluations
                    .saturating_add(record.charged_evaluations)
                    .min(config.evaluation_budget);
                let event_charge = record.charged_evaluations;
                let coverage = escape_coverage.evidence();
                events.push(NdHybridEvent {
                    attempt,
                    mechanism: NdHybridMechanism::BasinEscape,
                    ridge_discovery_index: discovery_indices.map(|indices| indices[RIDGE_ARM]),
                    escape_discovery_index: discovery_indices.map(|indices| indices[ESCAPE_ARM]),
                    source_basin: Some(source_basin),
                    mode_rank: None,
                    direction: None,
                    escape_kernel: Some(escape_kernel),
                    kernel_probability: Some(kernel_selection.probability()),
                    kernel_learning_rate: Some(kernel_selection.learning_rate()),
                    kernel_implicit_exploration: Some(kernel_selection.implicit_exploration()),
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
        move_pulls: move_allocator.pulls().to_vec(),
        move_success_rates: move_allocator.success_rates(),
        escape_observations: coverage.observations,
        escape_singletons: coverage.singletons,
        escape_unseen_mass_upper: coverage.unseen_mass_upper,
        escape_coverage_saturated: coverage.saturated,
        termination,
    })
}
