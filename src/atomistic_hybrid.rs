//! Same-PES atomistic global-minimum exploration by finite-action information.
//!
//! Each invocation owns its descriptor geometry, chemical context, exact
//! witness, action posterior, and stationary network. Basin actions are
//! isotropic Gaussian perturbations followed by rgmin quenches. Ridge actions
//! are physically projected local modes executed by rgsaddle minimum-mode,
//! P-RFO, and IRC sessions. A GP for each action family is coupled only through
//! the identity and value of the lowest reachable minimum. The selected action
//! maximizes joint-optimum information divided by posterior mean PES cost.

use std::collections::HashSet;

use ndarray::{Array1, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};

use crate::curvature::{project_rigid_with, rigid_basis};
use crate::descriptor_space::{DescriptorError, DescriptorGeometry, DescriptorSpace};
use crate::methods::cluster_search::Encounter;
use crate::minimum_information::{
    MinimumInformationError, MinimumInformationSearch, SearchActionCandidate, SearchMechanism,
};
use crate::movekernel::{Gaussian, MoveKernel};
use crate::nd_hybrid::{ActionFeatureError, ActionFeatureMap, DescriptorActionFeatures};
use crate::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesExplorationError, PesNetwork, PesSurface,
    RideModeDirection, StructureContext, discover_cartesian_mode_connection_in_context_with_budget,
    localized_cartesian_mode,
};
use crate::source_escape::{SourceEscapeConfig, SourceEscapeOutcome, quench_source_escape};

/// Chemical and mechanical identity of one atomistic PES invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomisticSystem {
    /// Atomic numbers in coordinate order.
    pub species: Vec<u32>,
    /// Native masses, one per atom.
    pub masses: Vec<f64>,
    /// Coordinates constrained by the caller's PES adapter.
    pub frozen_atoms: Vec<bool>,
    /// Stable caller namespace retained with the report.
    pub identity_domain: String,
}

/// Controls for one system-local atomistic exploration campaign.
#[derive(Debug, Clone)]
pub struct AtomisticHybridConfig {
    /// Hard PES-call budget including the initial source quench.
    pub evaluation_budget: u64,
    /// Largest PES-call slice assigned to one ridge action.
    pub ride_evaluation_cap: u64,
    /// Largest PES-call slice assigned to one perturb-and-quench action.
    pub escape_evaluation_cap: u64,
    /// Deterministic localized mode ranks exposed per mobile atom and source.
    pub ride_modes_per_atom: u16,
    /// Gaussian locality radius for rgsaddle launch modes.
    pub localization_radius: f64,
    /// Physical standard deviations in the finite basin-action portfolio.
    pub escape_scales: Vec<f64>,
    /// Monte Carlo draws used by joint entropy search.
    pub minimum_information_samples: usize,
    /// Descriptor-space GP length scale.
    pub information_length_scale: f64,
    /// Terminal-energy-change GP amplitude.
    pub information_amplitude: f64,
    /// Terminal-energy-change GP observation noise.
    pub information_noise: f64,
    /// Prior mean PES calls for a ridge action.
    pub expected_ride_cost: f64,
    /// Prior mean PES calls for a basin action.
    pub expected_escape_cost: f64,
    /// Pseudo-observation weight of each cost prior.
    pub cost_prior_strength: f64,
    /// rgmin, rgsaddle, P-RFO, IRC, and certification controls.
    pub exploration: PesExplorationConfig,
}

/// Mechanism responsible for one charged atomistic action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomisticHybridMechanism {
    /// Projected minimum-mode ride and IRC endpoint quenches.
    Ridge,
    /// Isotropic perturbation followed by an rgmin quench.
    BasinEscape,
}

/// Mechanism restriction used by matched-PES-call comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomisticHybridPolicy {
    /// Select across both action families by joint-optimum information per cost.
    Adaptive,
    /// Evaluate only the finite projected ridge portfolio.
    RidgeOnly,
    /// Evaluate only perturb-and-quench actions.
    BasinEscapeOnly,
}

/// One observable action and its exact stationary-structure yield.
#[derive(Debug, Clone)]
pub struct AtomisticHybridEvent {
    /// Monotonic charged-action index.
    pub attempt: u64,
    /// Selected action family.
    pub mechanism: AtomisticHybridMechanism,
    /// Exact source minimum in this report's network.
    pub source_basin: usize,
    /// Source energy used by the action model.
    pub source_energy: f64,
    /// Lowest terminal energy returned by the action.
    pub terminal_energy: f64,
    /// Joint optimum identity-and-value information.
    pub selected_information: f64,
    /// Selected information divided by posterior mean PES cost.
    pub selected_information_rate: f64,
    /// Mobile atom localizing a ridge launch.
    pub representative_atom: Option<usize>,
    /// Deterministic localized mode rank.
    pub mode_rank: Option<u16>,
    /// Signed ridge direction.
    pub direction: Option<RideModeDirection>,
    /// Gaussian basin displacement scale.
    pub escape_scale: Option<f64>,
    /// Exact minimum identities introduced by this action.
    pub new_minimum_ids: Vec<usize>,
    /// Certified index-one saddle identities introduced by this action.
    pub new_saddle_ids: Vec<usize>,
    /// Unresolved index-one saddle observations introduced by this action.
    pub new_unresolved_saddle_ids: Vec<usize>,
    /// PES calls charged to this action.
    pub charged_evaluations: u64,
    /// Whether the action returned its certified object.
    pub converged: bool,
    /// Whether the action attempted to exceed its assigned PES-call slice.
    pub budget_exhausted: bool,
    /// Stable numerical failure text.
    pub error: Option<String>,
}

/// Terminal condition of one atomistic campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomisticHybridTermination {
    /// The matched PES-call budget was consumed.
    BudgetConsumed,
    /// A numerical action returned without issuing a PES call.
    NoPesProgress,
    /// Every finite ridge action was attempted.
    RidePortfolioExhausted,
}

/// System-local stationary network and action-allocation evidence.
#[derive(Debug)]
pub struct AtomisticHybridReport {
    /// Caller namespace for the single PES represented by this report.
    pub identity_domain: String,
    /// Exact minima and certified or unresolved saddle observations.
    pub network: PesNetwork,
    /// Adaptive or fixed mechanism policy.
    pub policy: AtomisticHybridPolicy,
    /// Total PES calls, including the initial source quench.
    pub charged_evaluations: u64,
    /// Ordered charged-action evidence.
    pub events: Vec<AtomisticHybridEvent>,
    /// Ridge then basin-action exposure counts.
    pub mechanism_pulls: [usize; 2],
    /// Posterior mean cost for ridge then pooled basin actions.
    pub expected_mechanism_costs: [f64; 2],
    /// Terminal budget condition.
    pub termination: AtomisticHybridTermination,
}

impl AtomisticHybridReport {
    /// Lowest force-certified minimum retained by this same-PES invocation.
    pub fn best_energy(&self) -> Option<f64> {
        self.network
            .minima()
            .iter()
            .map(|minimum| minimum.energy)
            .min_by(f64::total_cmp)
    }

    /// Charged work at the first encounter with a target energy.
    ///
    /// A missing encounter remains censored at the campaign's actual charged
    /// count, so fixed-budget failures participate in Kaplan--Meier summaries.
    pub fn first_encounter(&self, target: f64, tolerance: f64) -> Encounter {
        let total_charged = usize::try_from(self.charged_evaluations).unwrap_or(usize::MAX);
        if !target.is_finite() || !tolerance.is_finite() || tolerance < 0.0 {
            return Encounter::Censored {
                charged: total_charged,
            };
        }
        let action_charged = self
            .events
            .iter()
            .map(|event| event.charged_evaluations)
            .fold(0u64, u64::saturating_add);
        let mut charged = self.charged_evaluations.saturating_sub(action_charged);
        if self
            .network
            .minima()
            .first()
            .is_some_and(|minimum| minimum.energy < target + tolerance)
        {
            return Encounter::Found {
                charged: usize::try_from(charged).unwrap_or(usize::MAX),
                hops: 0,
            };
        }
        for (hop, event) in self.events.iter().enumerate() {
            charged = charged.saturating_add(event.charged_evaluations);
            if event.terminal_energy < target + tolerance {
                return Encounter::Found {
                    charged: usize::try_from(charged).unwrap_or(usize::MAX),
                    hops: hop + 1,
                };
            }
        }
        Encounter::Censored {
            charged: total_charged,
        }
    }
}

/// Invalid system, action model, or initial-source failure.
#[derive(Debug, thiserror::Error)]
pub enum AtomisticHybridError {
    /// A system or campaign control lies outside its domain.
    #[error("invalid atomistic hybrid configuration: {0}")]
    InvalidConfig(&'static str),
    /// The initial source did not reach a force-certified minimum.
    #[error("initial atomistic source failed after {charged_evaluations} PES calls: {error}")]
    InitialSource {
        /// Stable rgmin or surface failure.
        error: String,
        /// PES calls consumed before the failure.
        charged_evaluations: u64,
    },
    /// Descriptor construction or exact-network comparison failed.
    #[error(transparent)]
    Descriptor(#[from] DescriptorError),
    /// An invariant action representation rejected a point.
    #[error(transparent)]
    ActionFeature(#[from] ActionFeatureError),
    /// A minimum-information model rejected its inputs.
    #[error(transparent)]
    MinimumInformation(#[from] MinimumInformationError),
    /// A projected mode could not be constructed.
    #[error(transparent)]
    Exploration(#[from] PesExplorationError),
}

#[derive(Debug, Clone, Copy)]
struct CostPosterior {
    prior_mean: f64,
    prior_strength: f64,
    charged: u64,
    pulls: usize,
}

impl CostPosterior {
    fn new(prior_mean: f64, prior_strength: f64) -> Self {
        Self {
            prior_mean,
            prior_strength,
            charged: 0,
            pulls: 0,
        }
    }

    fn observe(&mut self, charged: u64) {
        self.charged = self.charged.saturating_add(charged);
        self.pulls = self.pulls.saturating_add(1);
    }

    fn mean(&self, maximum: u64) -> f64 {
        let posterior = (self.prior_strength * self.prior_mean + self.charged as f64)
            / (self.prior_strength + self.pulls as f64);
        posterior.clamp(1.0, maximum.max(1) as f64)
    }
}

#[derive(Debug)]
enum PlannedAction {
    Ridge {
        source_basin: usize,
        source_energy: f64,
        representative_atom: usize,
        mode_rank: u16,
        direction: RideModeDirection,
        source: Array1<f64>,
        mode: Array1<f64>,
        feature: Vec<f64>,
    },
    Escape {
        source_basin: usize,
        source_energy: f64,
        scale_index: usize,
        scale: f64,
        proposal: Array1<f64>,
        feature: Vec<f64>,
    },
}

impl PlannedAction {
    fn mechanism(&self) -> SearchMechanism {
        match self {
            Self::Ridge { .. } => SearchMechanism::SaddleRide,
            Self::Escape { .. } => SearchMechanism::BasinEscape,
        }
    }

    fn feature(&self) -> &[f64] {
        match self {
            Self::Ridge { feature, .. } | Self::Escape { feature, .. } => feature,
        }
    }

    fn source_energy(&self) -> f64 {
        match self {
            Self::Ridge { source_energy, .. } | Self::Escape { source_energy, .. } => {
                *source_energy
            }
        }
    }

    fn candidate(&self, expected_charged_evaluations: f64) -> SearchActionCandidate {
        SearchActionCandidate {
            mechanism: self.mechanism(),
            feature: self.feature().to_vec(),
            source_energy: self.source_energy(),
            expected_charged_evaluations,
        }
    }
}

fn validate(
    initial: ArrayView1<'_, f64>,
    system: &AtomisticSystem,
    config: &AtomisticHybridConfig,
    descriptor_space: &DescriptorSpace,
) -> Result<DescriptorGeometry, AtomisticHybridError> {
    if initial.is_empty() || !initial.len().is_multiple_of(3) {
        return Err(AtomisticHybridError::InvalidConfig(
            "coordinates must be nonempty 3N Cartesian",
        ));
    }
    let atoms = initial.len() / 3;
    if system.species.len() != atoms
        || system.masses.len() != atoms
        || system.frozen_atoms.len() != atoms
    {
        return Err(AtomisticHybridError::InvalidConfig(
            "species, masses, and frozen mask must match the atom count",
        ));
    }
    if system.species.contains(&0)
        || system
            .masses
            .iter()
            .any(|mass| !mass.is_finite() || *mass <= 0.0)
        || system.frozen_atoms.iter().all(|frozen| *frozen)
        || system.identity_domain.is_empty()
    {
        return Err(AtomisticHybridError::InvalidConfig(
            "system identity, species, masses, and mobile atoms must be valid",
        ));
    }
    if config.evaluation_budget == 0
        || config.ride_evaluation_cap == 0
        || config.escape_evaluation_cap == 0
        || config.ride_modes_per_atom == 0
        || config.minimum_information_samples == 0
    {
        return Err(AtomisticHybridError::InvalidConfig(
            "budgets, mode count, and information samples must be positive",
        ));
    }
    if !config.localization_radius.is_finite()
        || config.localization_radius <= 0.0
        || config.escape_scales.is_empty()
        || config
            .escape_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(AtomisticHybridError::InvalidConfig(
            "localization radius and escape scales must be positive and finite",
        ));
    }
    if [
        config.information_length_scale,
        config.information_amplitude,
        config.information_noise,
        config.expected_ride_cost,
        config.expected_escape_cost,
        config.cost_prior_strength,
    ]
    .into_iter()
    .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(AtomisticHybridError::InvalidConfig(
            "information and cost priors must be positive and finite",
        ));
    }
    descriptor_space
        .geometry()
        .ok_or(AtomisticHybridError::InvalidConfig(
            "atomistic descriptor geometry is required",
        ))
}

fn structure_context(system: &AtomisticSystem, geometry: DescriptorGeometry) -> StructureContext {
    StructureContext::new(
        Some(system.species.clone()),
        Some(geometry),
        Some(system.identity_domain.clone()),
    )
    .with_masses(Some(system.masses.clone()))
}

fn action_seed(seed: u64, attempt: u64, source: usize, family: usize, rank: u16) -> u64 {
    let mut mixed = seed
        ^ attempt.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (source as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (family as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ u64::from(rank);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn constrained_gaussian_proposal(
    source: ArrayView1<'_, f64>,
    scale: f64,
    frozen_atoms: &[bool],
    geometry: DescriptorGeometry,
    seed: u64,
) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let raw = Gaussian::new(scale).propose(source, scale, &mut rng);
    let mut displacement = &raw - &source;
    for (atom, frozen) in frozen_atoms.iter().copied().enumerate() {
        if frozen {
            for axis in 0..3 {
                displacement[3 * atom + axis] = 0.0;
            }
        }
    }
    if frozen_atoms.iter().all(|frozen| !frozen) {
        if geometry.periodic().iter().any(|periodic| *periodic) {
            let atoms = frozen_atoms.len();
            for axis in 0..3 {
                let translation = (0..atoms)
                    .map(|atom| displacement[3 * atom + axis])
                    .sum::<f64>()
                    / atoms as f64;
                for atom in 0..atoms {
                    displacement[3 * atom + axis] -= translation;
                }
            }
        } else {
            project_rigid_with(&mut displacement, &rigid_basis(source));
        }
    }
    &source + &displacement
}

fn new_ids(before: usize, after: usize) -> Vec<usize> {
    (before..after).collect()
}

/// Explore one atomistic PES under an adaptive or fixed matched-cost policy.
#[allow(clippy::too_many_arguments)]
pub fn explore_atomistic_with_policy<S, W>(
    surface: &S,
    descriptor_space: &DescriptorSpace,
    initial: ArrayView1<'_, f64>,
    system: &AtomisticSystem,
    config: &AtomisticHybridConfig,
    witness: &W,
    seed: u64,
    policy: AtomisticHybridPolicy,
) -> Result<AtomisticHybridReport, AtomisticHybridError>
where
    S: PesSurface + ?Sized,
    W: ExactStructureWitness + ?Sized,
{
    let geometry = validate(initial, system, config, descriptor_space)?;
    let action_features =
        DescriptorActionFeatures::new(descriptor_space.clone(), system.species.clone());
    action_features.encode(initial)?;
    let norm_tolerance = config
        .exploration
        .quench_gradient_norm_tolerance
        .unwrap_or(config.exploration.quench_gradient_tolerance * (initial.len() as f64).sqrt());
    let initial_config = SourceEscapeConfig {
        maximum_evaluations: config.evaluation_budget.min(config.escape_evaluation_cap),
        quench_steps: config.exploration.quench_steps,
        gradient_tolerance: config.exploration.quench_gradient_tolerance,
        gradient_norm_tolerance: norm_tolerance,
    };
    let initial_record = match quench_source_escape(surface, initial, &initial_config) {
        SourceEscapeOutcome::Converged(record) => record,
        SourceEscapeOutcome::Failed(failure) => {
            return Err(AtomisticHybridError::InitialSource {
                error: failure.error,
                charged_evaluations: failure.charged_evaluations,
            });
        }
    };
    let mut charged_evaluations = initial_record.charged_evaluations;
    let mut network = PesNetwork::new();
    let initial_descriptor = descriptor_space.describe(
        initial_record.minimum.coordinates.view(),
        Some(&system.species),
    )?;
    network.admit_minimum_with_context(
        initial_record.minimum.energy,
        initial_record.minimum.coordinates,
        initial_record.minimum.max_gradient,
        initial_descriptor,
        structure_context(system, geometry),
        witness,
    )?;

    let mut information = MinimumInformationSearch::new(
        config.information_length_scale,
        config.information_amplitude,
        config.information_noise,
    )?;
    let mut ridge_cost = CostPosterior::new(config.expected_ride_cost, config.cost_prior_strength);
    let mut escape_costs =
        vec![
            CostPosterior::new(config.expected_escape_cost, config.cost_prior_strength,);
            config.escape_scales.len()
        ];
    let mut attempted_rides = HashSet::<(usize, usize, u16, RideModeDirection)>::new();
    let mut mechanism_pulls = [0usize; 2];
    let mut events = Vec::new();
    let mut attempt = 0u64;

    let termination = loop {
        let remaining = config.evaluation_budget.saturating_sub(charged_evaluations);
        if remaining == 0 {
            break AtomisticHybridTermination::BudgetConsumed;
        }
        let mut plans = Vec::<PlannedAction>::new();
        if policy != AtomisticHybridPolicy::BasinEscapeOnly {
            for minimum in network.minima() {
                for representative_atom in 0..system.species.len() {
                    if system.frozen_atoms[representative_atom] {
                        continue;
                    }
                    for mode_rank in 0..config.ride_modes_per_atom {
                        for direction in [RideModeDirection::Positive, RideModeDirection::Negative]
                        {
                            let task = (minimum.id, representative_atom, mode_rank, direction);
                            if attempted_rides.contains(&task) {
                                continue;
                            }
                            let mode = localized_cartesian_mode(
                                minimum.coordinates.view(),
                                representative_atom,
                                &system.frozen_atoms,
                                geometry,
                                config.localization_radius,
                                action_seed(
                                    seed,
                                    attempt,
                                    minimum.id,
                                    representative_atom,
                                    mode_rank,
                                ),
                                mode_rank,
                                direction,
                            )?;
                            let displaced = &minimum.coordinates
                                + &(mode.clone() * config.exploration.saddle_displacement);
                            plans.push(PlannedAction::Ridge {
                                source_basin: minimum.id,
                                source_energy: minimum.energy,
                                representative_atom,
                                mode_rank,
                                direction,
                                source: minimum.coordinates.clone(),
                                mode,
                                feature: action_features.encode(displaced.view())?,
                            });
                        }
                    }
                }
            }
        }
        if policy != AtomisticHybridPolicy::RidgeOnly {
            for minimum in network.minima() {
                for (scale_index, scale) in config.escape_scales.iter().copied().enumerate() {
                    let proposal = constrained_gaussian_proposal(
                        minimum.coordinates.view(),
                        scale,
                        &system.frozen_atoms,
                        geometry,
                        action_seed(seed, attempt, minimum.id, scale_index, u16::MAX),
                    );
                    let mut feature = action_features.encode(proposal.view())?;
                    feature.push(scale / geometry.length_scale());
                    plans.push(PlannedAction::Escape {
                        source_basin: minimum.id,
                        source_energy: minimum.energy,
                        scale_index,
                        scale,
                        proposal,
                        feature,
                    });
                }
            }
        }
        if plans.is_empty() {
            break AtomisticHybridTermination::RidePortfolioExhausted;
        }
        let candidates = plans
            .iter()
            .map(|plan| match plan {
                PlannedAction::Ridge { .. } => {
                    plan.candidate(ridge_cost.mean(remaining.min(config.ride_evaluation_cap)))
                }
                PlannedAction::Escape { scale_index, .. } => plan.candidate(
                    escape_costs[*scale_index].mean(remaining.min(config.escape_evaluation_cap)),
                ),
            })
            .collect::<Vec<_>>();
        let scores = information.score(&candidates, config.minimum_information_samples)?;
        let selected_index = scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.information_per_charged_evaluation
                    .total_cmp(&right.information_per_charged_evaluation)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("a nonempty finite action set has a score");
        let selected_score = scores[selected_index];
        let selected = plans.swap_remove(selected_index);
        attempt = attempt.saturating_add(1);

        match selected {
            PlannedAction::Ridge {
                source_basin,
                source_energy,
                representative_atom,
                mode_rank,
                direction,
                source,
                mode,
                feature,
            } => {
                attempted_rides.insert((source_basin, representative_atom, mode_rank, direction));
                let minima_before = network.minimum_count();
                let saddles_before = network.saddle_count();
                let unresolved_before = network.unresolved_saddles().len();
                let context = structure_context(system, geometry);
                let ride = discover_cartesian_mode_connection_in_context_with_budget(
                    surface,
                    descriptor_space,
                    &mut network,
                    source.view(),
                    &system.frozen_atoms,
                    mode.view(),
                    &context,
                    &config.exploration,
                    witness,
                    remaining.min(config.ride_evaluation_cap),
                );
                let new_minimum_ids = new_ids(minima_before, network.minimum_count());
                let new_saddle_ids = new_ids(saddles_before, network.saddle_count());
                let new_unresolved_saddle_ids =
                    new_ids(unresolved_before, network.unresolved_saddles().len());
                let terminal_energy = ride
                    .connection
                    .as_ref()
                    .ok()
                    .into_iter()
                    .flat_map(|connection| connection.endpoints)
                    .chain(new_minimum_ids.iter().copied())
                    .map(|id| network.minima()[id].energy)
                    .fold(source_energy, f64::min);
                if ride.charged_evaluations > 0 {
                    information.observe(
                        SearchMechanism::SaddleRide,
                        &feature,
                        source_energy,
                        terminal_energy,
                    )?;
                    ridge_cost.observe(ride.charged_evaluations);
                }
                charged_evaluations = charged_evaluations
                    .saturating_add(ride.charged_evaluations)
                    .min(config.evaluation_budget);
                mechanism_pulls[0] = mechanism_pulls[0].saturating_add(1);
                let converged = ride.connection.is_ok();
                let error = ride.connection.as_ref().err().map(ToString::to_string);
                events.push(AtomisticHybridEvent {
                    attempt,
                    mechanism: AtomisticHybridMechanism::Ridge,
                    source_basin,
                    source_energy,
                    terminal_energy,
                    selected_information: selected_score.information,
                    selected_information_rate: selected_score.information_per_charged_evaluation,
                    representative_atom: Some(representative_atom),
                    mode_rank: Some(mode_rank),
                    direction: Some(direction),
                    escape_scale: None,
                    new_minimum_ids,
                    new_saddle_ids,
                    new_unresolved_saddle_ids,
                    charged_evaluations: ride.charged_evaluations,
                    converged,
                    budget_exhausted: ride.budget_exhausted,
                    error,
                });
                if ride.charged_evaluations == 0 {
                    break AtomisticHybridTermination::NoPesProgress;
                }
            }
            PlannedAction::Escape {
                source_basin,
                source_energy,
                scale_index,
                scale,
                proposal,
                feature,
            } => {
                let escape_config = SourceEscapeConfig {
                    maximum_evaluations: remaining.min(config.escape_evaluation_cap),
                    quench_steps: config.exploration.quench_steps,
                    gradient_tolerance: config.exploration.quench_gradient_tolerance,
                    gradient_norm_tolerance: norm_tolerance,
                };
                let escape = quench_source_escape(surface, proposal.view(), &escape_config);
                match escape {
                    SourceEscapeOutcome::Converged(record) => {
                        let descriptor = descriptor_space
                            .describe(record.minimum.coordinates.view(), Some(&system.species))?;
                        let admission = network.admit_minimum_with_context(
                            record.minimum.energy,
                            record.minimum.coordinates,
                            record.minimum.max_gradient,
                            descriptor,
                            structure_context(system, geometry),
                            witness,
                        )?;
                        let new_minimum_ids = admission
                            .is_new
                            .then_some(admission.id)
                            .into_iter()
                            .collect();
                        let terminal_energy = network.minima()[admission.id].energy;
                        information.observe(
                            SearchMechanism::BasinEscape,
                            &feature,
                            source_energy,
                            terminal_energy,
                        )?;
                        escape_costs[scale_index].observe(record.charged_evaluations);
                        charged_evaluations = charged_evaluations
                            .saturating_add(record.charged_evaluations)
                            .min(config.evaluation_budget);
                        mechanism_pulls[1] = mechanism_pulls[1].saturating_add(1);
                        events.push(AtomisticHybridEvent {
                            attempt,
                            mechanism: AtomisticHybridMechanism::BasinEscape,
                            source_basin,
                            source_energy,
                            terminal_energy,
                            selected_information: selected_score.information,
                            selected_information_rate: selected_score
                                .information_per_charged_evaluation,
                            representative_atom: None,
                            mode_rank: None,
                            direction: None,
                            escape_scale: Some(scale),
                            new_minimum_ids,
                            new_saddle_ids: Vec::new(),
                            new_unresolved_saddle_ids: Vec::new(),
                            charged_evaluations: record.charged_evaluations,
                            converged: true,
                            budget_exhausted: false,
                            error: None,
                        });
                        if record.charged_evaluations == 0 {
                            break AtomisticHybridTermination::NoPesProgress;
                        }
                    }
                    SourceEscapeOutcome::Failed(failure) => {
                        if failure.charged_evaluations > 0 {
                            information.observe(
                                SearchMechanism::BasinEscape,
                                &feature,
                                source_energy,
                                source_energy,
                            )?;
                            escape_costs[scale_index].observe(failure.charged_evaluations);
                        }
                        charged_evaluations = charged_evaluations
                            .saturating_add(failure.charged_evaluations)
                            .min(config.evaluation_budget);
                        mechanism_pulls[1] = mechanism_pulls[1].saturating_add(1);
                        let budget_exhausted = failure.error.contains("budget exhausted");
                        events.push(AtomisticHybridEvent {
                            attempt,
                            mechanism: AtomisticHybridMechanism::BasinEscape,
                            source_basin,
                            source_energy,
                            terminal_energy: source_energy,
                            selected_information: selected_score.information,
                            selected_information_rate: selected_score
                                .information_per_charged_evaluation,
                            representative_atom: None,
                            mode_rank: None,
                            direction: None,
                            escape_scale: Some(scale),
                            new_minimum_ids: Vec::new(),
                            new_saddle_ids: Vec::new(),
                            new_unresolved_saddle_ids: Vec::new(),
                            charged_evaluations: failure.charged_evaluations,
                            converged: false,
                            budget_exhausted,
                            error: Some(failure.error),
                        });
                        if failure.charged_evaluations == 0 {
                            break AtomisticHybridTermination::NoPesProgress;
                        }
                    }
                }
            }
        }
    };

    let pooled_escape_cost = escape_costs
        .iter()
        .map(|posterior| posterior.mean(config.escape_evaluation_cap))
        .sum::<f64>()
        / escape_costs.len() as f64;
    Ok(AtomisticHybridReport {
        identity_domain: system.identity_domain.clone(),
        network,
        policy,
        charged_evaluations,
        events,
        mechanism_pulls,
        expected_mechanism_costs: [
            ridge_cost.mean(config.ride_evaluation_cap),
            pooled_escape_cost,
        ],
        termination,
    })
}

#[cfg(test)]
mod tests {
    use super::CostPosterior;

    #[test]
    fn cost_posterior_updates_from_charged_calls_without_discarding_its_prior() {
        let mut cost = CostPosterior::new(100.0, 2.0);
        cost.observe(40);
        cost.observe(80);

        assert!((cost.mean(1_000) - 80.0).abs() < 1e-12);
    }
}
