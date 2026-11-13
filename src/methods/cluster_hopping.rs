// Copyright (c) anneal developers
//! Basin hopping over quenched minima with a bias keyed on basin identity.
//!
//! Cluster global optimisation happens on the quenched landscape,
//! `E_q(x) = E(local_min(x))`, not on the raw surface: the funnel structure is
//! only visible after relaxation, and a search on the raw surface finds
//! nothing on a 38-atom Lennard-Jones cluster.
//!
//! Three things then decide whether a funnel can be left.
//!
//! The relaxation is where the budget goes. A full one costs a few hundred
//! charged evaluations and most trials land nowhere near the incumbent, so
//! trials are screened by a short relaxation first and only promoted when they
//! land within [`Config::screen_margin`] of the incumbent. Measured on LJ38,
//! screening took basin discovery from 27 to 327 at a fixed charge.
//!
//! The bias is keyed on basin identity rather than on a collective variable.
//! A variable has to be chosen correctly or it cannot see the competition: on
//! LJ75 the Marks decahedron and the structures a search settles into differ by
//! 0.023 in the fourth Steinhardt parameter, narrower than any usable
//! deposition width, so biasing on it fills both competitors at once.
//!
//! The moves have to change the packing rather than perturb it, which is what
//! [`crate::movekernel::SurfaceRelocate`], [`crate::movekernel::ShellRotate`]
//! and [`crate::movekernel::Symmetrise`] do, and which of them pays at a given
//! stage is decided online rather than fixed.

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::allocate::{BudgetWindowTemperature, FlooredThompson};
use crate::bias::{
    AdaptiveHeight, BasinBias, BasinIndex, Bias, Fingerprint, SiteEnergies, SortedPairs,
};
use crate::calibrate::StepCalibrator;
use crate::contextual::ContextualAllocator;
use crate::diversity::DiversityAnnealer;
use crate::exchange::MetropolisExchange;
use crate::methods::activation::{Activation, activate};
use crate::methods::minima_hopping::{EscapeFeedback, HistoryHook, HistoryReport, Visit};
use crate::movekernel::{
    HollowFill, HollowRelocate, MoveKernel, ShellRotate, SurfaceRelocate, Symmetrise,
};
use crate::path::{StallDetector, interpolate_path};
use crate::screen::Screen;

mod binary_geometry;
mod config;
mod cycles;
mod general;
#[cfg(test)]
mod mackay_cap_tests;
mod moves;
mod packing_bank;
mod preset;

pub use crate::methods::minima_hopping::SharedVisitPolicy;
pub use config::{
    ArmSelection, Config, ContinuousSymmetry, Keying, LadderMode, OrbitCore, PairRange,
    SoapProposalMode,
};
pub use moves::*;

#[cfg(test)]
use preset::LennardJonesPreset;

/// Work ledger: every objective or gradient evaluation is charged.
///
/// A relaxation inside a move spends the same budget as a proposal does, which
/// is the accounting that makes methods with different internal structure
/// comparable. Published cluster success rates are quoted per hopping step,
/// with the relaxation inside each step uncounted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuenchStatus {
    /// The relaxation ended at a freshly checked minimum.
    Validated,
    /// The relaxation was screened, capped, or failed its fresh gradient check.
    Rejected,
}

/// One relaxation boundary and the exact potential work it consumed.
#[derive(Debug, Clone)]
pub struct QuenchBoundary {
    status: QuenchStatus,
    charged_calls: usize,
    energy: f64,
    state: Array1<f64>,
    gradient: Option<Array1<f64>>,
}

/// One perturb--quench outcome from the live chain or a diagnostic probe.
///
/// The record is deliberately separate from the best-minimum ledger. Funnel
/// entry commonly requires accepted uphill motion, so a history containing
/// only record improvements cannot reconstruct the region the chain occupies
/// or the boundary crossings it has demonstrated.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedTransition {
    /// Completed hop index within this run.
    pub hop: usize,
    /// Target-blind proposal mechanism that generated the trial.
    pub action: String,
    /// Quenched energy at the source occupied by the chain.
    pub from_energy: f64,
    /// Trial energy; non-finite when a diagnostic quench fails.
    pub to_energy: f64,
    /// Quenched source coordinates.
    pub from_state: Array1<f64>,
    /// Fresh source gradient retained when validation requires one.
    pub from_gradient: Option<Array1<f64>>,
    /// Quenched destination coordinates.
    pub to_state: Array1<f64>,
    /// Fresh destination gradient retained when validation requires one.
    pub to_gradient: Option<Array1<f64>>,
    /// Whether the destination met the run's fresh quench-validity contract.
    pub validated: bool,
    /// Whether the destination became the state occupied by the live chain.
    pub adopted: bool,
}

/// Coordinates and their paid evidence travel together through adoption and
/// ladder scheduling. A shared identity belongs only to this exact occupancy.
struct OccupiedMinimum {
    energy: f64,
    coordinates: Array1<f64>,
    gradient: Option<Array1<f64>>,
    history_minimum: Option<usize>,
    local_minimum: Option<usize>,
    generation: u64,
    pending_initial: bool,
}

impl OccupiedMinimum {
    fn into_live(
        self,
    ) -> (
        f64,
        Array1<f64>,
        Option<Array1<f64>>,
        Option<usize>,
        u64,
        Option<usize>,
    ) {
        (
            self.energy,
            self.coordinates,
            self.gradient,
            self.history_minimum,
            self.generation,
            self.local_minimum,
        )
    }
}

enum AdoptionHistory {
    Fresh,
    Observed(Option<HistoryReport>),
}

/// Read-only scientific state exposed at a charged-work checkpoint.
///
/// A checkpoint observes the live chain in place. It does not end the run,
/// rebuild an adaptive controller, or consume the chain's random stream.
pub struct ChainCheckpoint<'a> {
    current_state: ArrayView1<'a, f64>,
    current_energy: f64,
    current_gradient: Option<ArrayView1<'a, f64>>,
    best_state: Option<ArrayView1<'a, f64>>,
    best_energy: f64,
    quench_boundaries: &'a [QuenchBoundary],
    accepted_transitions: &'a [AcceptedTransition],
    charged: usize,
    remaining: usize,
    hops: usize,
    bias: Option<&'a BasinBias<ClusterFingerprint>>,
}

impl<'a> ChainCheckpoint<'a> {
    /// The live chain's per-basin bias, for a caller that shares deposits.
    ///
    /// Read-only: centres and visit counts are what a multiple-walker
    /// scheme exports. Foreign deposits come back through
    /// [`CheckpointAction::DepositDescriptors`], never by writing here.
    pub fn bias(&self) -> Option<&'a BasinBias<ClusterFingerprint>> {
        self.bias
    }

    /// Quenched state occupied by the live chain.
    pub fn current_state(&self) -> ArrayView1<'a, f64> {
        self.current_state
    }

    /// Quenched energy of the state occupied by the live chain.
    pub fn current_energy(&self) -> f64 {
        self.current_energy
    }

    /// Fresh validation gradient retained for the occupied state, when any.
    pub fn current_gradient(&self) -> Option<ArrayView1<'a, f64>> {
        self.current_gradient
    }

    /// Lowest state found under this ledger, when one is recordable.
    pub fn best_state(&self) -> Option<ArrayView1<'a, f64>> {
        self.best_state
    }

    /// Lowest recordable energy found under this ledger.
    pub fn best_energy(&self) -> f64 {
        self.best_energy
    }

    /// Relaxation boundaries completed since the preceding checkpoint.
    pub fn quench_boundaries(&self) -> &'a [QuenchBoundary] {
        self.quench_boundaries
    }

    /// Accepted transitions completed since the preceding checkpoint.
    pub fn accepted_transitions(&self) -> &'a [AcceptedTransition] {
        self.accepted_transitions
    }

    /// Charged objective work completed by this checkpoint.
    pub fn charged(&self) -> usize {
        self.charged
    }

    /// Charged objective work still available after this checkpoint.
    pub fn remaining(&self) -> usize {
        self.remaining
    }

    /// Perturb--quench hops completed by this checkpoint.
    pub fn hops(&self) -> usize {
        self.hops
    }
}

/// A bias-only communication update, independent of the chain's state action.
#[derive(Debug, Clone, PartialEq)]
pub enum BiasUpdate {
    /// Import other walkers' descriptor visits without publishing them as local.
    DepositDescriptors {
        /// Descriptor centres and their foreign visit counts.
        deposits: Vec<(Array1<f64>, u64)>,
        /// Height relative to one local deposit.
        weight: f64,
    },
    /// Average local wells toward a peer's table.
    MergeWells {
        /// Peer descriptor centres and well depths.
        wells: Vec<(Array1<f64>, f64)>,
        /// Step toward the peer's depth, in `(0, 1]`.
        weight: f64,
        /// Whether omitted wells have zero depth in a complete peer table.
        complete: bool,
    },
}

impl BiasUpdate {
    fn apply(
        self,
        bias: &mut BasinBias<ClusterFingerprint>,
        temperature: f64,
        shared_deposits: &mut usize,
        gossip_rounds: &mut usize,
    ) {
        match self {
            Self::DepositDescriptors { deposits, weight } => {
                for (centre, count) in &deposits {
                    bias.deposit_scaled_n(centre.view(), temperature, weight, *count);
                    *shared_deposits += usize::try_from(*count).unwrap_or(usize::MAX);
                }
            }
            Self::MergeWells {
                wells,
                weight,
                complete,
            } => {
                bias.merge_wells(&wells, weight, complete);
                *gossip_rounds += 1;
            }
        }
    }
}

/// Action returned after observing a live-chain checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointAction {
    /// Leave the live chain and every adaptive controller unchanged.
    Continue,
    /// Apply communication in order, then process one state action.
    ///
    /// Bias updates cost no objective calls and do not require a gradient or
    /// a minimum certificate. Wrappers compose from outside inward, so a
    /// caller can add communication without replacing a proposal or retirement.
    WithBiasUpdates {
        /// Ordered updates to the receiving chain's own bias.
        updates: Vec<BiasUpdate>,
        /// The action to process after every update in this wrapper.
        action: Box<CheckpointAction>,
    },
    /// Quench and adopt a target-blind boundary perturbation.
    BoundaryProposal {
        /// Cartesian proposal produced from shared region-boundary evidence.
        state: Array1<f64>,
        /// General proposal-family label retained on the trajectory edge.
        action: String,
    },
    /// Quench a fixed diagnostic perturbation without moving the live chain.
    ProbeProposal {
        /// Cartesian proposal generated by the declared probe operator.
        state: Array1<f64>,
        /// Probe action label retained separately from adaptive moves.
        action: String,
    },
    /// Settle potential evaluations consumed by an external search that did
    /// not produce a proposal.
    ///
    /// A failed saddle search, disconnected ride, or rejected certificate
    /// still expends objective work. Charging it without a Cartesian state
    /// keeps the matched-work ledger exact without inventing a quench or a
    /// trajectory edge.
    ExternalWork {
        /// Potential calls the external search consumed.
        external_calls: usize,
    },
    /// Quench an externally produced perturbation whose construction
    /// burned potential evaluations outside the chain's oracle.
    ///
    /// An external engine (a molecular-dynamics burst, a foreign
    /// sampler) does real force work the ledger never saw; the calls
    /// settle here before the proposal is quenched, so an engine-driven
    /// move can never be cheaper than the work it did.
    ExternalProposal {
        /// Cartesian proposal produced by the external engine.
        state: Array1<f64>,
        /// Engine action label retained on the trajectory edge.
        action: String,
        /// Potential calls the external engine consumed.
        external_calls: usize,
    },
    /// Charge external potential work, quench the proposal, and move the
    /// live chain onto it.
    ///
    /// A checkpoint that constructs and pre-relaxes a candidate outside the
    /// chain's oracle, and has already applied its own acceptance test to it,
    /// settles that work here and hands the accepted state over; the
    /// candidate is then adopted the way a [`CheckpointAction::BoundaryProposal`]
    /// is. Recombination between chains is the case: the child of two live
    /// structures is quenched and judged at the checkpoint, and only a child
    /// that passed is worth the chain's relocation.
    ExternalAdopt {
        /// Cartesian candidate the checkpoint already relaxed and accepted.
        state: Array1<f64>,
        /// Action label retained on the trajectory edge.
        action: String,
        /// Potential calls the checkpoint consumed constructing the candidate.
        external_calls: usize,
    },
    /// Deposit bias at remote minima so the chain's own acceptance feels
    /// what the ensemble has already visited.
    ///
    /// Collective knowledge that only arrives as an occasional steering
    /// decision leaves the walk between checkpoints blind to it; a deposit
    /// reshapes the biased landscape every subsequent move and every
    /// Metropolis test walks on, which is continuous repulsion from
    /// visited territory rather than an occasional redirect. The states
    /// are deposited exactly as the chain's own visits are, so the
    /// well-tempered accounting and the merge radius apply unchanged.
    DepositRemote {
        /// Cartesian minima received from the ensemble.
        states: Vec<Array1<f64>>,
    },
    /// Deposit other walkers' visits, by descriptor centre and count.
    ///
    /// Multiple-walker sharing of one bias (Raiteri et al. 2006), with the
    /// synchronisation lag of the checkpoint interval: every walker
    /// deposits its own hops as it makes them and the others' at the next
    /// checkpoint, under the same well-tempered weight and merge radius.
    DepositDescriptors {
        /// Descriptor centres with the number of visits to deposit at each.
        deposits: Vec<(Array1<f64>, u64)>,
        /// Height of each foreign deposit relative to the chain's own; 1/N
        /// holds the total deposition rate of N walkers at one walker's.
        weight: f64,
    },
    /// One gossip step: move this chain's wells toward another walker's.
    ///
    /// DeGroot averaging over whatever graph the caller draws peers from;
    /// see [`BasinBias::merge_wells`]. Conserves the population's bias
    /// mass where [`CheckpointAction::DepositDescriptors`] multiplies it.
    MergeBias {
        /// The peer's wells as `(centre, depth)`.
        wells: Vec<(Array1<f64>, f64)>,
        /// Step toward the peer, one half for a pairwise average.
        weight: f64,
        /// Whether `wells` is the peer's whole table; see
        /// [`BasinBias::merge_wells`].
        complete: bool,
    },
    /// Occupancy certificate: stop this replica.
    ///
    /// Occupancy MixingCertified conjuncts. CatalogSaturated is
    /// not this action. A published energy is not this action.
    Retire {
        /// `mixing` or `saturated`.
        reason: String,
    },
}

fn continue_without_checkpoint(_: ChainCheckpoint<'_>) -> CheckpointAction {
    CheckpointAction::Continue
}

impl QuenchBoundary {
    /// Scientific status assigned by the caller's fresh convergence check.
    pub fn status(&self) -> QuenchStatus {
        self.status
    }

    /// Potential calls consumed from entry through the fresh check.
    pub fn charged_calls(&self) -> usize {
        self.charged_calls
    }

    /// Fresh energy when validated, or the relaxation's terminal energy.
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// Terminal Cartesian state.
    pub fn state(&self) -> ArrayView1<'_, f64> {
        self.state.view()
    }

    /// Fresh gradient evidence, present only for a validated minimum.
    pub fn gradient(&self) -> Option<ArrayView1<'_, f64>> {
        self.gradient.as_ref().map(|gradient| gradient.view())
    }

    /// A validated boundary assembled from fresh evidence the caller holds.
    ///
    /// For a history that receives energy, state and validation gradient
    /// directly from the live chain rather than reading the ledger. The same
    /// finiteness and dimension rules as [`Ledger::record_quench_boundary`]
    /// apply; anything failing them is not a certificate and returns `None`.
    /// No charged work is attributed to it.
    pub fn validated(energy: f64, state: Array1<f64>, gradient: Array1<f64>) -> Option<Self> {
        let certified = energy.is_finite()
            && !state.is_empty()
            && state.iter().all(|v| v.is_finite())
            && gradient.len() == state.len()
            && gradient.iter().all(|v| v.is_finite());
        certified.then_some(Self {
            status: QuenchStatus::Validated,
            charged_calls: 0,
            energy,
            state,
            gradient: Some(gradient),
        })
    }
}

/// A search stopped because no initial quench was finite.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SearchError {
    /// The initial evaluation was non-finite on every attempt.
    #[error("no finite start after {attempts} attempts")]
    NoFiniteStart {
        /// Quenches tried before the search stopped.
        attempts: usize,
    },
}

/// Why a search returned an [`Outcome`].
///
/// A ledger that is closed for more than one reason reports the target
/// first, then the caller stop, then the budget.
///
/// A later release may add a reason, so a `match` outside this crate needs
/// a wildcard arm. [`HopStop::code`] names every reason.
///
/// ```text
/// match stop {
///     HopStop::Budget => "out of budget",
///     HopStop::Target => "reached the target",
///     _ => stop.code(),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HopStop {
    /// The charged budget was spent.
    Budget,
    /// The incumbent reached the [`Ledger::stop_below`] threshold.
    Target,
    /// The caller raised the ledger's stop flag.
    CallerStop,
    /// A checkpoint returned [`CheckpointAction::Retire`].
    Retired {
        /// The reason the checkpoint gave.
        reason: String,
    },
    /// The hop needs the NVE escape and the `rgsaddle` feature is off.
    NveUnavailable,
    /// The run took [`Config::max_hops`] hops.
    MaxHops,
    /// The Hamiltonian proposal returned no trajectory while budget was left.
    NoTrajectory,
}

impl HopStop {
    /// Stable lower-case name for logs and run records.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Budget => "budget",
            Self::Target => "target",
            Self::CallerStop => "caller_stop",
            Self::Retired { .. } => "retired",
            Self::NveUnavailable => "nve_unavailable",
            Self::MaxHops => "max_hops",
            Self::NoTrajectory => "no_trajectory",
        }
    }

    fn of_closed(ledger: &Ledger) -> Self {
        if ledger.target_reached() {
            Self::Target
        } else if ledger.stopped() {
            Self::CallerStop
        } else {
            Self::Budget
        }
    }
}

/// Charged objective-work ledger with optional per-relaxation evidence.
pub struct Ledger {
    budget: usize,
    spent: usize,
    /// Fractional debt accumulated by partial-system evaluations, settled
    /// into whole charged units as it crosses one.
    fract: f64,
    /// Lowest objective value seen.
    pub best: f64,
    /// State attaining [`Ledger::best`].
    pub best_state: Option<Array1<f64>>,
    quench_boundaries: Vec<QuenchBoundary>,
    diagnostic_quench: bool,
    /// Energy at or below which the hop may stop.
    target: Option<f64>,
    stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// How many initial quenches to try before [`SearchError::NoFiniteStart`].
    start_attempts: usize,
    /// Geometries drawn after a non-finite initial quench, in order.
    start_redraws: Vec<Array1<f64>>,
    /// Minima file named by `ANNEAL_MIN_DUMP`, opened on the first finite record.
    dump_file: Option<std::fs::File>,
    /// The environment variable was unset, so later records skip the lookup.
    dump_disabled: bool,
    /// Byte offset of the acceptance mark on the line just written.
    dump_accept_at: Option<u64>,
}

struct DiagnosticQuench<'a> {
    ledger: &'a mut Ledger,
    previous: bool,
}

impl std::ops::Deref for DiagnosticQuench<'_> {
    type Target = Ledger;

    fn deref(&self) -> &Self::Target {
        self.ledger
    }
}

impl std::ops::DerefMut for DiagnosticQuench<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ledger
    }
}

impl Drop for DiagnosticQuench<'_> {
    fn drop(&mut self) {
        self.ledger.diagnostic_quench = self.previous;
    }
}

impl Ledger {
    /// Creates a ledger with `budget` charged evaluations.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            spent: 0,
            fract: 0.0,
            best: f64::INFINITY,
            best_state: None,
            quench_boundaries: Vec::new(),
            diagnostic_quench: false,
            target: None,
            stop: None,
            start_attempts: Self::DEFAULT_START_ATTEMPTS,
            start_redraws: Vec::new(),
            dump_file: None,
            dump_disabled: false,
            dump_accept_at: None,
        }
    }

    /// Initial quenches tried when the first evaluation is non-finite.
    pub const DEFAULT_START_ATTEMPTS: usize = 8;

    /// Caps the initial-quench redraws. Values below 1 are raised to 1.
    pub fn set_start_attempts(&mut self, attempts: usize) {
        self.start_attempts = attempts.max(1);
    }

    /// Starts drawn after a refused initial evaluation, oldest first.
    pub fn start_redraws(&self) -> &[Array1<f64>] {
        &self.start_redraws
    }

    fn note_start_redraw(&mut self, state: ArrayView1<f64>) {
        self.start_redraws.push(state.to_owned());
    }

    /// Stop once a recorded energy is at or below `energy`.
    ///
    /// The value lives on the ledger rather than on [`Config`], so a stock
    /// preset's resolved JSON is unchanged. A non-finite threshold is ignored.
    pub fn stop_below(&mut self, energy: f64) {
        if energy.is_finite() {
            self.target = Some(energy);
        }
    }

    /// Threshold installed by [`Ledger::stop_below`], if any.
    pub fn target(&self) -> Option<f64> {
        self.target
    }

    /// Whether `energy` has reached the installed threshold.
    pub fn below_target(&self, energy: f64) -> bool {
        self.target.is_some_and(|target| energy <= target)
    }

    /// Whether the incumbent has reached the installed threshold.
    pub fn target_reached(&self) -> bool {
        self.below_target(self.best)
    }

    /// A ledger that also closes when `stop` is raised.
    ///
    /// The caller owns the flag. Once it is set, [`Ledger::charge`] refuses
    /// and [`Ledger::remaining`] reads zero, so a search returns at once
    /// with the best state found instead of spending what is left. A harness
    /// that stops a run on a target, or a callback that raised, sets it.
    /// [`Ledger::slice`] keeps the same flag on a child budget.
    pub fn with_stop(budget: usize, stop: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        let mut ledger = Self::new(budget);
        ledger.stop = Some(stop);
        ledger
    }

    /// A child budget that closes on the same caller stop as this ledger.
    ///
    /// The child has its own spend. Handing it to an inner search keeps that
    /// search's budget arithmetic on the slice, and a stop raised on either
    /// ledger ends both.
    pub fn slice(&self, budget: usize) -> Self {
        let mut child = Self::new(budget);
        child.stop = self.stop.clone();
        child
    }

    /// Binds this ledger's stop on the running thread, if it has one.
    ///
    /// Worker threads do not see the binding. Give them [`Ledger::slice`]
    /// or [`Ledger::with_stop`] so the flag itself travels.
    pub fn bind_stop(&self) -> Option<crate::caller_stop::Bind> {
        self.stop.clone().map(crate::caller_stop::Bind::new)
    }

    /// Whether the caller has asked this ledger to close.
    ///
    /// True when this ledger's flag is raised or a stop is bound on this
    /// thread. A down flag is a relaxed load and does not change a charge.
    pub fn stopped(&self) -> bool {
        self.stop
            .as_ref()
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
            || crate::caller_stop::raised()
    }

    /// Whether the relaxation serves a non-adopting diagnostic probe.
    ///
    /// Diagnostic callbacks must use the declared, fixed relaxation kernel:
    /// no adaptive surface, search bias, noise, or portfolio reward credit.
    /// Objective work and discovered values still belong to this ledger.
    pub fn is_diagnostic_quench(&self) -> bool {
        self.diagnostic_quench
    }

    fn diagnostic_scope(&mut self) -> DiagnosticQuench<'_> {
        let previous = std::mem::replace(&mut self.diagnostic_quench, true);
        DiagnosticQuench {
            ledger: self,
            previous,
        }
    }

    /// Charges one unit, returning `false` when the budget is gone.
    pub fn charge(&mut self) -> bool {
        if self.spent >= self.budget || self.stopped() {
            return false;
        }
        self.spent += 1;
        true
    }

    /// Charges a fraction of one evaluation, for work that touches a subset
    /// of the system.
    ///
    /// A k-atom partial evaluation computes k of the n(n-1)/2 pair rows, so
    /// its honest price is a fraction of a full evaluation. The fractions
    /// accumulate and are charged as a whole unit each time their sum crosses
    /// one, which keeps the count deterministic and auditable. A residue
    /// under one unit is never charged, so [`Ledger::spent`] can fall up to
    /// one evaluation short of the work done, mid-run and when the run ends.
    /// Charging stops at the budget and the rest of a charge that reaches it
    /// goes unpaid, so work charged after it is done can fall more than one
    /// unit short when the ledger runs dry inside one charge.
    pub fn charge_frac(&mut self, frac: f64) -> bool {
        if self.stopped() {
            return false;
        }
        if !(frac > 0.0) || !frac.is_finite() {
            return self.spent < self.budget;
        }
        self.fract += frac;
        while self.fract >= 1.0 {
            self.fract -= 1.0;
            if !self.charge() {
                return false;
            }
        }
        !self.stopped() && self.spent < self.budget
    }

    /// Charges `n` units at once, returning `false` when the budget ran out
    /// partway.
    ///
    /// For a caller that ran work against a sub-ledger and is settling up. The
    /// alternative, handing the real ledger to the inner run, makes any budget
    /// arithmetic inside it see the whole campaign's budget rather than the
    /// slice it was given.
    pub fn charge_many(&mut self, n: usize) -> bool {
        let room = self.remaining();
        self.spent += n.min(room);
        n <= room
    }

    /// Records a value and its state when it improves the incumbent.
    ///
    /// A non-finite value is not an improvement. −∞ compares below every
    /// finite incumbent, and adopting it leaves the run with no geometry.
    /// Every finite value passed here is also appended to the minima file
    /// when `ANNEAL_MIN_DUMP` names one. The acceptance column starts at 0.
    pub fn record(&mut self, value: f64, state: ArrayView1<f64>) {
        if value.is_finite() && value < self.best {
            self.best = value;
            self.best_state = Some(state.to_owned());
        }
        self.note_recorded_minimum(value, state);
    }

    /// Appends one recorded minimum. The acceptance column is 0 until
    /// [`Ledger::mark_last_recorded`] sets it.
    fn note_recorded_minimum(&mut self, value: f64, state: ArrayView1<f64>) {
        self.dump_accept_at = None;
        if !value.is_finite() {
            return;
        }
        let Some(file) = self.dump_out() else {
            return;
        };
        use std::io::{Seek, SeekFrom, Write};
        let Some(end) = file.seek(SeekFrom::End(0)).ok() else {
            return;
        };
        let mut line = format!("{value:.8} 0");
        for coordinate in state.iter() {
            line.push(' ');
            line.push_str(&format!("{coordinate:.6}"));
        }
        line.push('\n');
        if file.write_all(line.as_bytes()).is_err() || file.flush().is_err() {
            return;
        }
        let marker = format!("{value:.8} ").len() as u64;
        self.dump_accept_at = Some(end + marker);
    }

    fn dump_out(&mut self) -> Option<&mut std::fs::File> {
        if self.dump_file.is_some() {
            return self.dump_file.as_mut();
        }
        if self.dump_disabled {
            return None;
        }
        let Ok(path) = std::env::var("ANNEAL_MIN_DUMP") else {
            self.dump_disabled = true;
            return None;
        };
        match std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
        {
            Ok(file) => {
                self.dump_file = Some(file);
                self.dump_file.as_mut()
            }
            Err(_) => {
                self.dump_disabled = true;
                None
            }
        }
    }

    /// Sets the acceptance column on the minimum [`Ledger::record`] just wrote.
    fn mark_last_recorded(&mut self, accepted: bool) {
        let Some(offset) = self.dump_accept_at.take() else {
            return;
        };
        if !accepted {
            return;
        }
        let Some(file) = self.dump_file.as_mut() else {
            return;
        };
        use std::io::{Seek, SeekFrom, Write};
        let _ = file
            .seek(SeekFrom::Start(offset))
            .and_then(|_| file.write_all(b"1"))
            .and_then(|_| file.flush());
    }

    /// Charged evaluations the ledger was created with.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Charged evaluations remaining.
    pub fn remaining(&self) -> usize {
        if self.stopped() {
            return 0;
        }
        self.budget.saturating_sub(self.spent)
    }

    /// Charged evaluations spent.
    pub fn spent(&self) -> usize {
        self.spent
    }

    /// Record one charged relaxation after its caller-owned convergence check.
    ///
    /// `gradient` is fresh validated evidence. Its absence classifies the
    /// boundary as rejected. Nonfinite energy, coordinates, or gradient and a
    /// mismatched gradient dimension also produce a rejected boundary while
    /// retaining its charged work. An invocation that consumed no potential
    /// calls is not a quench boundary and returns `false`.
    pub fn record_quench_boundary(
        &mut self,
        charged_before: usize,
        energy: f64,
        state: Array1<f64>,
        gradient: Option<Array1<f64>>,
    ) -> bool {
        let Some(charged_calls) = self.spent.checked_sub(charged_before) else {
            return false;
        };
        if charged_calls == 0 {
            return false;
        }
        let gradient = gradient.filter(|values| {
            energy.is_finite()
                && !state.is_empty()
                && state.iter().all(|v| v.is_finite())
                && values.len() == state.len()
                && values.iter().all(|v| v.is_finite())
        });
        let status = if gradient.is_some() {
            QuenchStatus::Validated
        } else {
            QuenchStatus::Rejected
        };
        self.quench_boundaries.push(QuenchBoundary {
            status,
            charged_calls,
            energy,
            state,
            gradient,
        });
        true
    }

    /// Relaxation boundaries recorded against this ledger in execution order.
    pub fn quench_boundaries(&self) -> &[QuenchBoundary] {
        &self.quench_boundaries
    }
}

/// How often one move type was drawn, accepted, and below the incumbent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArmCount {
    /// Short name of the proposal kernel.
    pub move_type: String,
    /// Proposals handed to the quench.
    pub proposed: usize,
    /// Proposals the acceptance test kept, after a tabu veto.
    pub accepted: usize,
    /// Proposals whose quenched energy was below the incumbent.
    pub improved: usize,
}

/// What a run produced.
#[derive(Debug, Clone, Default)]
pub struct Outcome {
    /// Lowest quenched value found.
    pub best: f64,
    /// State attaining it.
    pub best_state: Option<Array1<f64>>,
    /// Live chain at the end of the run, which a later hop can continue.
    pub final_state: Option<Array1<f64>>,
    /// Energy belonging to [`Outcome::final_state`].
    pub final_energy: f64,
    /// Accepted live-chain edges in execution order.
    pub accepted_transitions: Vec<AcceptedTransition>,
    /// Hops taken.
    pub hops: usize,
    /// Trials rejected by screening before a full relaxation.
    pub screened_out: usize,
    /// Distinct basins registered.
    pub basins: usize,
    /// Charged evaluations spent.
    pub charged: usize,
    /// Proposals closer than [`OVERLAP_SEPARATION`] before any energy call.
    ///
    /// Each one was separated, or dropped when separation failed. Neither the
    /// raw geometry nor a dropped geometry is charged to [`Outcome::charged`].
    pub overlap_rejected: usize,
    /// Trials abandoned because their partial relaxation was going home.
    pub returned: usize,
    /// Escape scale at the end of the run, when the controller is used.
    pub escape_scale: f64,
    /// Acceptance threshold at the end of the run.
    pub escape_threshold: f64,
    /// Quenches classified as a return, a known basin and a new one.
    pub visit_counts: (usize, usize, usize),
    /// Soft-subspace perturbations proposed.
    pub soft_perturbs: usize,
    /// Soft-subspace recomputations paid for.
    pub soft_subspaces: usize,
    /// Proposals made along the softest mode.
    pub soft_escapes: usize,
    /// Of those, the ones whose climb reached a saddle.
    pub soft_crossed: usize,
    /// Hop, charged evaluations spent, basin count and value at each new
    /// global best.
    ///
    /// This is what a first-encounter time is computed from, and it is the
    /// statistic worth reporting. A success rate at a fixed budget is the same
    /// quantity through an arbitrary threshold: above the budget it saturates
    /// and says nothing about the margin, below it censors and says nothing
    /// about how near the failures came. The work to first reach a target is a
    /// property of the method rather than of a budget someone chose, which is
    /// why the literature quotes mean first encounter times.
    ///
    /// The charged count is the part that makes it comparable. Hops are not:
    /// two arms with different screening spend different amounts per hop, and
    /// this campaign has arms ranging from 26 to 637 charged evaluations per
    /// hop.
    ///
    /// Capped on the number of *records* rather than on the hops: a run that
    /// improves ten thousand times is descending, and the tail of that is not
    /// what anyone is asking about.
    pub improvements: Vec<(usize, usize, usize, f64)>,
    /// Charged count, basin count, energy and convergence flag at every hop
    /// that ran a full relaxation.
    ///
    /// Empty unless [`Config::trace_quenched`] is set. Ordered by hop, so a
    /// prefix of it is what the run had seen at a given point in its budget,
    /// which is what makes a call read off it a prediction rather than a
    /// summary.
    ///
    /// Hops the screen or the return test stopped are absent, because their
    /// energy comes off a 25-step partial descent and is not a draw from the
    /// distribution of minima at all; no threshold keeps such a value out of
    /// the exceedances, since it lands wherever the descent stopped. That
    /// exclusion is itself a selection on energy, since the screen refuses
    /// exactly the trials whose partial energy sits above `best +
    /// screen_margin`, and the note on [`Config::trace_quenched`] says what
    /// that costs.
    ///
    /// The flag is the gradient guard the ledger records under. False means
    /// the relaxation stopped at its iteration cap, which leaves the energy a
    /// little above the minimum it was heading for rather than somewhere else
    /// entirely.
    pub quenched: Vec<(usize, usize, f64, bool)>,
    /// The superbasin hierarchy the transitions imply, and what the escape
    /// did with it; present whenever the graph was recorded.
    pub superbasin: Option<crate::superbasin::SuperbasinReport>,
    /// The recorded transition counts, when the quotient analysis is asked
    /// for.
    pub superbasin_counts: Option<crate::superbasin::HopCounts>,
    /// The archived structures, as `(basin, energy, state)`.
    pub superbasin_archive: Option<Vec<(usize, f64, Array1<f64>)>>,
    /// Every fully quenched energy the run produced, in the order produced.
    pub energy_trace: Option<Vec<f64>>,
    /// Merge radius at the end of the run, calibrated or as configured.
    pub merge_radius: f64,
    /// Mean accepted-hop step length, which the radius is a quantile of.
    pub mean_step: f64,
    /// Angular moves attempted, and the ratio they settled at.
    pub angular: (usize, usize, f64),
    /// Picks per move under the contextual allocator, and choices it forced.
    pub contextual: (Vec<usize>, usize),
    /// Screen decisions: made, relaxed, forced by the exploration floor, and
    /// observations the model was fitted on.
    pub screen: (usize, usize, usize, usize),
    /// Funnels quarantined, and proposals refused for landing in one.
    pub tabu: (usize, usize),
    /// The funnel partition at the end of the run: parts, and how separated.
    ///
    /// A connectivity near zero means the search's transitions split into two
    /// nearly disconnected sets, which is what a funnel boundary looks like
    /// from the inside.
    pub funnel: Option<(usize, usize, f64)>,
    /// Symmetrisations attempted, and the energy they gained.
    pub symmetrised: (usize, f64),
    /// Orbit completions attempted, and the energy they gained.
    pub orbits: (usize, f64),
    /// Continuous-symmetry quenches attempted, and downhill energy gained.
    pub continuous_symmetry: (usize, f64),
    /// Restarts triggered by a stall.
    pub restarts: usize,
    /// Heard structures refused by the exchange acceptance.
    pub exchanges_refused: usize,
    /// Occasional jumps taken on stagnation.
    pub jumps: usize,
    /// Certified quenches reported to the minimum history, and how many of
    /// them the history called new under its membership policy.
    pub history_visits: (usize, usize),
    /// Bias deposits made on behalf of other chains' visits.
    pub shared_deposits: usize,
    /// Gossip averaging steps applied to the bias.
    pub gossip_rounds: usize,
    /// MD escapes attempted, integration steps taken, and attempts that
    /// found no second potential minimum or ran out of budget.
    ///
    /// An attempt that never started because the `rgsaddle` feature is off
    /// is not counted here. That refusal is [`Outcome::nve_unavailable`].
    pub md_escape: (usize, usize, usize),
    /// The hop asked for a constant-energy (NVE) escape and the `rgsaddle`
    /// feature is off, so the run stopped instead of substituting a kick.
    pub nve_unavailable: bool,
    /// Why the run returned. A search always sets it; only an outcome
    /// built by hand leaves it `None`.
    pub stop: Option<HopStop>,
    /// Seconds the chain spent inside the history, lock waits included.
    pub history_seconds: f64,
    /// Climbs triggered by a stall.
    pub stall_escapes: usize,
    /// Stall exits taken through the recorded basin entry.
    pub trail_escapes: usize,
    /// Energy gained by those that landed lower than where they left.
    pub stall_escape_gain: f64,
    /// Mean softest eigenvalue over those proposals.
    pub soft_lambda: f64,
    /// Per-rung temperature, basin count and the energy the rung ends on.
    ///
    /// The temperature is `temperature` times the rung's ratio. Under
    /// [`Config::budget_window`] or [`Config::statistical_temperature`] a rung
    /// hops at that ratio times the adaptive temperature instead, and this is
    /// the ladder's nominal value.
    ///
    /// What says whether a ladder is doing its job rather than merely swapping:
    /// a hot rung should register many basins and a poor energy, a cold rung
    /// few basins and a deep one. A ladder where every rung looks alike is a
    /// ladder whose spread is too narrow to be worth its cost.
    pub rungs: Vec<(f64, usize, f64)>,
    /// Swap attempts, each from the active rung to the next one up, the
    /// hottest offering to the coldest.
    pub swaps_tried: usize,
    /// What the ladder transported: round trips, sweeps, and the communication
    /// barrier at the end of the run. `None` for a single chain.
    ///
    /// A round trip is one tagged configuration reaching the hottest rung from
    /// the coldest and returning, counted across every tag. Swap counts do not
    /// say this: a ladder can accept swaps at any rate and still shuffle the
    /// same two rungs forever, which is a different failure from a ladder that
    /// never swaps and looks the same in the solve count.
    pub transport: Option<(usize, usize, f64)>,
    /// Hops the acceptance rule took, before any veto.
    pub accepted: usize,
    /// Structures barred from the ledger for not being minima.
    pub unconverged_records: usize,
    /// Per-arm draws, accepts and best quenched value, in library order.
    pub arms: Vec<(String, usize, usize, f64)>,
    /// Every quenched proposal, by move type, in first-seen order.
    pub proposal_counts: Vec<ArmCount>,
    /// Delayed acceptance: first stages run, first-stage rejections (each a
    /// quench not paid), second stages run, and second-stage rejections (the
    /// surrogate's mistakes).
    pub delayed: Option<(usize, usize, usize, usize)>,
    /// Swaps accepted.
    pub swaps_accepted: usize,
    /// Paths attempted after a stall.
    pub paths: usize,
    /// Paths that produced a structure outside the starting basin.
    ///
    /// Nearly always all of them, and so not worth much on its own: an image
    /// interpolated towards a different structure differs from the start by
    /// construction. The useful count is `path_improvements`.
    pub path_escapes: usize,
    /// Paths that produced a structure lower than the chain was standing on.
    pub path_improvements: usize,
    /// Total depth gained from paths, in energy units.
    pub path_gain: f64,
    /// Sampler diagnostics per rung, when the Hamiltonian proposal is used.
    ///
    /// One entry per replica, because each rung adapts its own step size and
    /// its own metric. The counters are the result rather than an error path:
    /// a divergence rate says which configurations the sampler cannot traverse,
    /// a cap rate near one says the no-U-turn criterion is being truncated, and
    /// a metric condition near one says an adapted metric is carrying no
    /// anisotropy at all.
    pub hmc: Vec<crate::hmc::hop::HopDiagnostics>,

    /// The energy bias as the run left it, under [`Config::energy_bias`] once
    /// its first sample has filled.
    pub energy_bias: Option<crate::dos::EnergyBias>,
}

/// Relaxes `x`, charging every evaluation, and stopping when the budget ends.
///
/// The relaxation is supplied by the caller because the objective, its
/// gradient and the minimiser are the caller's: this module owns the search,
/// not the numerics under it.
pub type Relax<'a> =
    &'a mut (dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + Send);

/// Partial relaxation of the listed atoms in the frozen environment.
///
/// Arguments: ledger, structure, moved atom indices, descent steps. Returns
/// the settled structure. The callee owns the objective and therefore the
/// honest fractional price; it charges through [`Ledger::charge_frac`].
pub type Settle<'a> =
    &'a mut dyn FnMut(&mut Ledger, ArrayView1<f64>, &[usize], usize) -> Array1<f64>;

/// Gradient of the objective, charged to the ledger by the caller.
///
/// A supplied callback certifies candidate answers and supports soft-mode
/// escape. Gradient-free callers own their relaxation convergence contract.
pub type GradFn<'g> = dyn FnMut(&mut Ledger, ArrayView1<f64>) -> Option<Array1<f64>> + Send + 'g;

/// Value and gradient together, charged to the ledger by the caller.
///
/// Separate from [`GradFn`] because the Hamiltonian proposal needs both at
/// every leapfrog leaf and, on a pairwise potential, both come out of one pass
/// over the pairs. Charging twice for one pass would make the arm look half as
/// efficient as it is, which is exactly the kind of miscalibrated comparison
/// this campaign has already made once.
pub type EnergyGradFn<'g> =
    dyn FnMut(&mut Ledger, ArrayView1<f64>) -> Option<(f64, Array1<f64>)> + Send + 'g;

/// A borrow of one, for a caller that has a gradient to lend.
///
/// The two lifetimes are separate on purpose. Tying the trait object's lifetime
/// to the borrow makes the pair invariant, so a caller that holds a gradient and
/// wants to lend it to a sequence of inner runs cannot reborrow it: it has one
/// gradient and can hand it over once.
pub type Grad<'a> = &'a mut GradFn<'a>;

/// Moves the centre of mass to the origin.
pub(crate) fn recentre(x: &mut Array1<f64>, n: usize) {
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in c.iter_mut() {
        *v /= n as f64;
    }
    for i in 0..n {
        for k in 0..3 {
            x[3 * i + k] -= c[k];
        }
    }
}

/// Pulls points outside the container back onto its surface.
///
/// Applied when a move is generated and never inside a relaxation: a cluster
/// relaxes in free space, and constraining the minimiser makes it stop at its
/// own starting point.
pub(crate) fn contain(x: &mut Array1<f64>, n: usize, radius: f64) {
    for i in 0..n {
        let r = (0..3)
            .map(|k| x[3 * i + k] * x[3 * i + k])
            .sum::<f64>()
            .sqrt();
        if r > radius && r > 0.0 {
            let s = radius / r;
            for k in 0..3 {
                x[3 * i + k] *= s;
            }
        }
    }
}

/// Moves every connected part of `x` out of `reach` of the largest part
/// back to `bond` from it.
///
/// Parts are connected at `reach`. Each other part is translated along the
/// line from its atom nearest the largest part to that atom's partner there,
/// until the two are `bond` apart, and the parts are formed again. The
/// largest part does not move; of equal parts, the one holding the lowest
/// index stays.
pub(crate) fn gather_into_reach(x: &mut Array1<f64>, n: usize, reach: f64, bond: f64) {
    if !(reach.is_finite() && bond.is_finite() && bond > 0.0 && bond < reach) {
        return;
    }
    for _ in 0..n {
        let groups = connectivity_groups(x.view(), n, reach);
        if groups.len() < 2 {
            return;
        }
        let anchor = (0..groups.len())
            .max_by(|&a, &b| groups[a].len().cmp(&groups[b].len()).then(b.cmp(&a)))
            .unwrap_or(0);
        let mut nearest: Option<(f64, usize, usize, usize)> = None;
        for (g, group) in groups.iter().enumerate() {
            if g == anchor {
                continue;
            }
            for &i in group {
                for &j in &groups[anchor] {
                    let d2: f64 = (0..3).map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2)).sum();
                    if nearest.is_none_or(|(best, ..)| d2 < best) {
                        nearest = Some((d2, g, i, j));
                    }
                }
            }
        }
        let Some((d2, g, i, j)) = nearest else {
            return;
        };
        let d = d2.sqrt();
        if d <= 0.0 {
            return;
        }
        let shift: [f64; 3] =
            std::array::from_fn(|k| (x[3 * j + k] - x[3 * i + k]) * (d - bond) / d);
        for &a in &groups[g] {
            for (k, step) in shift.iter().enumerate() {
                x[3 * a + k] += step;
            }
        }
    }
}

/// Keeps a trial where the potential can still hold it together.
///
/// The pair's measured reach when there is one, so a ranged potential sets
/// its own bound, and the container sphere otherwise.
fn contain_trial(cfg: &Config, x: &mut Array1<f64>, n: usize) {
    match cfg.pair_range {
        Some(range) => {
            gather_into_reach(x, n, range.reach, range.bond);
            recentre(x, n);
        }
        None => contain(x, n, cfg.container),
    }
}

/// Pushes overlapping points apart to `min_sep`.
///
/// A configuration with two points on top of each other has an enormous value
/// under any repulsive potential, and a quasi-Newton relaxation started there
/// fails on its first line search and returns the configuration unchanged.
#[cfg(test)]
fn repair(x: &mut Array1<f64>, n: usize, min_sep: f64) {
    for _ in 0..40 {
        let mut moved = false;
        for a in 0..n {
            for b in (a + 1)..n {
                let mut d = [0.0; 3];
                let mut r2 = 0.0;
                for k in 0..3 {
                    d[k] = x[3 * a + k] - x[3 * b + k];
                    r2 += d[k] * d[k];
                }
                let r = r2.sqrt();
                if r < min_sep && r > 1e-9 {
                    let push = 0.5 * (min_sep - r) / r;
                    for k in 0..3 {
                        x[3 * a + k] += push * d[k];
                        x[3 * b + k] -= push * d[k];
                    }
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
}

/// Shortest pair distance in a 3N Cartesian state.
pub fn min_pair_distance(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            best = best.min((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    best
}

/// Whether a Cartesian state contains only finite, separated atoms.
pub fn structure_is_sane(x: ArrayView1<f64>, min_sep: f64) -> bool {
    x.iter().all(|value| value.is_finite()) && min_pair_distance(x) >= min_sep
}

/// Pair distance below which an external molecular potential is observing an
/// overlap artifact rather than a physical bond.
pub const OVERLAP_SEPARATION: f64 = 0.35;

/// Distances this close still abort xtb's SCC, and the line search walks the
/// gate one probe at a time. Each probe runs a full SCC that fails. 0.90 Å
/// is below a covalent O–H and above that failing shell, so the first probe
/// the potential sees is a bond, not a contact xtb cannot converge.
pub const OVERLAP_CALL_LIMIT: f64 = 0.90;

/// Fraction of the summed Cordero radii inside which a declared bond is not
/// handed to the potential. One fraction for every element. The radii, not
/// the species name, set the length. A contact that is not a declared bond
/// is refused inside the full covalent sum: that sum is a single-bond length.
pub const COVALENT_CALL_FRACTION: f64 = 0.95;

/// Distance at or below which a molecular potential is not called.
///
/// Species-free searches keep [`OVERLAP_CALL_LIMIT`]. With species, a
/// declared bond uses [`COVALENT_CALL_FRACTION`] of the covalent sum and any
/// other pair uses the sum itself. The absolute limit remains a floor so a
/// hydrogen pair is not treated more loosely than before.
pub fn molecular_call_floor(za: u32, zb: u32, bonded: bool) -> f64 {
    let sum = moves::covalent_radius(za) + moves::covalent_radius(zb);
    if sum <= 0.0 {
        return OVERLAP_CALL_LIMIT;
    }
    let scaled = if bonded {
        COVALENT_CALL_FRACTION * sum
    } else {
        sum
    };
    scaled.max(OVERLAP_CALL_LIMIT)
}

/// Whether any pair is inside [`molecular_call_floor`].
///
/// `groups` marks declared bonds. Atoms that share a group are bonded; every
/// other pair is a contact. This is geometry only. It does not call a potential.
pub fn inside_molecular_call_wall(
    x: ArrayView1<f64>,
    species: &[u32],
    groups: Option<&[Vec<usize>]>,
) -> bool {
    let n = x.len() / 3;
    if n < 2 || species.len() < n {
        return min_pair_distance(x) <= OVERLAP_CALL_LIMIT;
    }
    let mut owner = vec![usize::MAX; n];
    if let Some(groups) = groups {
        for (group, atoms) in groups.iter().enumerate() {
            for &atom in atoms {
                if atom < n {
                    owner[atom] = group;
                }
            }
        }
    }
    for a in 0..n {
        for b in (a + 1)..n {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = x[3 * a + axis] - x[3 * b + axis];
                square += delta * delta;
            }
            let distance = square.sqrt();
            let bonded = owner[a] != usize::MAX && owner[a] == owner[b];
            if distance <= molecular_call_floor(species[a], species[b], bonded) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod call_wall_tests {
    use super::{inside_molecular_call_wall, molecular_call_floor};
    use ndarray::Array1;

    #[test]
    fn a_nonbonded_contact_inside_the_covalent_sum_is_refused() {
        let mut coordinates = Array1::zeros(6);
        coordinates[3] = 0.90;
        let species = [7u32, 1];
        assert!(inside_molecular_call_wall(
            coordinates.view(),
            &species,
            None
        ));
        coordinates[3] = 1.05;
        assert!(!inside_molecular_call_wall(
            coordinates.view(),
            &species,
            None
        ));
    }

    #[test]
    fn a_declared_bond_near_its_covalent_length_is_allowed() {
        let mut coordinates = Array1::zeros(6);
        coordinates[3] = 1.012;
        let species = [7u32, 1];
        let groups = vec![vec![0usize, 1]];
        assert!(!inside_molecular_call_wall(
            coordinates.view(),
            &species,
            Some(&groups)
        ));
        coordinates[3] = 0.80;
        assert!(inside_molecular_call_wall(
            coordinates.view(),
            &species,
            Some(&groups)
        ));
    }

    #[test]
    fn hydrogen_pairs_keep_the_absolute_floor() {
        assert!((molecular_call_floor(1, 1, false) - 0.90).abs() < 1.0e-12);
        let oh = molecular_call_floor(8, 1, true);
        let expected = (0.95_f64 * (0.66 + 0.31)).max(0.90);
        assert!((oh - expected).abs() < 1.0e-12);
        assert!((molecular_call_floor(7, 1, false) - 1.02).abs() < 1.0e-12);
    }
}

/// How far past [`OVERLAP_CALL_LIMIT`] a non-bonded spike is opened.
///
/// Only pairs closer than [`OVERLAP_SPIKE`] move. 1.30 Å past the call limit
/// puts such a pair at 2.20 Å, where the force is a contact the line search
/// can follow. A hydrogen bond already longer than the spike is not moved.
pub const OVERLAP_CLASH_CLEARANCE: f64 = 1.30;

/// A non-bonded pair closer than this is a repulsive spike. Pairs already
/// past it, including a hydrogen bond, are left where they stand.
const OVERLAP_SPIKE: f64 = 1.40;

/// Whether `a` and `b` are a covalent bond inside one declared group.
///
/// Judged on the geometry before any repair. A missing species treats the
/// whole group as bonded, which is the rigid-body repair.
fn pair_is_covalent_bond(
    a: usize,
    b: usize,
    distance: f64,
    left: usize,
    right: usize,
    species: Option<&[u32]>,
    bond_tolerance: f64,
) -> bool {
    if left == usize::MAX || left != right {
        return false;
    }
    // No species: a declared group is treated as bonded, which is what the
    // rigid-body repair has always done.
    let Some(species) = species else {
        return true;
    };
    let za = species.get(a).copied().unwrap_or(0);
    let zb = species.get(b).copied().unwrap_or(0);
    let reach = bond_tolerance
        * (crate::methods::cluster_hopping::moves::covalent_radius(za)
            + crate::methods::cluster_hopping::moves::covalent_radius(zb));
    reach > 0.0 && distance <= reach
}

fn open_target(
    a: usize,
    b: usize,
    distance: f64,
    left: usize,
    right: usize,
    bonded: bool,
    species: Option<&[u32]>,
    bond_target: f64,
    clash_target: f64,
) -> f64 {
    if bonded {
        // Classified once, at the geometry we were handed. A later stretch
        // must not turn the bond into a clash or the repair dissociates it.
        // A bond already near the covalent length is left alone, so a healthy
        // molecule is not rewritten on the way past an unrelated spike.
        let Some(species) = species else {
            return if distance >= bond_target {
                distance
            } else {
                bond_target
            };
        };
        let reach = crate::methods::cluster_hopping::moves::covalent_radius(
            species.get(a).copied().unwrap_or(0),
        ) + crate::methods::cluster_hopping::moves::covalent_radius(
            species.get(b).copied().unwrap_or(0),
        );
        let restore = bond_target.max(reach);
        return if distance + 0.02 >= restore {
            distance
        } else {
            restore
        };
    }
    // A same-group pair that is not a bond is an angle, not a contact
    // between molecules. Open it off the gate and no further.
    if left != usize::MAX && left == right {
        let angle = bond_target + 0.30;
        return if distance >= angle { distance } else { angle };
    }
    if distance >= OVERLAP_SPIKE {
        distance
    } else {
        clash_target
    }
}

fn contacts_are_open(
    x: ArrayView1<f64>,
    owner: Option<&[usize]>,
    species: Option<&[u32]>,
    bonded: &[bool],
    bond_target: f64,
    clash_target: f64,
) -> bool {
    let n = x.len() / 3;
    for a in 0..n {
        for b in (a + 1)..n {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = x[3 * a + axis] - x[3 * b + axis];
                square += delta * delta;
            }
            let distance = square.sqrt();
            let left = owner.map(|owner| owner[a]).unwrap_or(usize::MAX);
            let right = owner.map(|owner| owner[b]).unwrap_or(usize::MAX);
            let target = open_target(
                a,
                b,
                distance,
                left,
                right,
                bonded[a * n + b],
                species,
                bond_target,
                clash_target,
            );
            if distance < target {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
pub(crate) fn separate_to_overlap_rule(
    x: &mut Array1<f64>,
    groups: Option<&[Vec<usize>]>,
    min_sep: f64,
    frozen: Option<&[bool]>,
) -> bool {
    separate_contacts(x, groups, min_sep, 1.0e-8, frozen, None, 1.25)
}

/// Push clashing pairs apart.
///
/// A pair inside one declared group is a bond and is opened only to
/// `min_sep`. Every other pair is opened to `min_sep + clash_clearance`,
/// so a quench is not handed a contact sitting on the call limit. The
/// clearance is `1e-8` for a repair that only has to clear the gate.
pub(crate) fn separate_contacts(
    x: &mut Array1<f64>,
    groups: Option<&[Vec<usize>]>,
    min_sep: f64,
    clash_clearance: f64,
    frozen: Option<&[bool]>,
    species: Option<&[u32]>,
    bond_tolerance: f64,
) -> bool {
    if !(min_sep.is_finite() && min_sep > 0.0) || x.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let n = x.len() / 3;
    if n < 2 {
        return true;
    }
    let owner = groups.map(|groups| {
        let mut owner = vec![usize::MAX; n];
        for (group, atoms) in groups.iter().enumerate() {
            for &atom in atoms {
                if atom < n {
                    owner[atom] = group;
                }
            }
        }
        owner
    });
    let n_groups = groups.map(|groups| groups.len()).unwrap_or(0);
    let held = |atom: usize| frozen.is_some_and(|mask| mask.get(atom).copied().unwrap_or(false));
    // A group that is entirely frozen is the frame. The repair pushes the
    // other side by the whole gap and leaves the frame where it stands.
    let group_held: Vec<bool> = groups
        .map(|groups| {
            groups
                .iter()
                .map(|atoms| !atoms.is_empty() && atoms.iter().copied().all(held))
                .collect()
        })
        .unwrap_or_default();
    let side_held = |atom: usize, group: usize| {
        if group != usize::MAX {
            group_held.get(group).copied().unwrap_or(false)
        } else {
            held(atom)
        }
    };
    // A fraction of an angstrom past the rule, so a later strict comparison
    // does not reject the same pair on a rounding residue. Bonds stay at
    // the call limit. A clash between groups is opened by the clearance,
    // which is the room a unit line-search step needs.
    let bond_target = min_sep + 1.0e-8;
    let clash_target = min_sep + clash_clearance.max(1.0e-8);
    // Bond identity is fixed on the geometry we were handed. Reclassifying
    // after a push turns a stretched O–H into a clash and dissociates it.
    let mut bonded = vec![false; n * n];
    for a in 0..n {
        for b in (a + 1)..n {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = x[3 * a + axis] - x[3 * b + axis];
                square += delta * delta;
            }
            let left = owner.as_ref().map(|owner| owner[a]).unwrap_or(usize::MAX);
            let right = owner.as_ref().map(|owner| owner[b]).unwrap_or(usize::MAX);
            bonded[a * n + b] =
                pair_is_covalent_bond(a, b, square.sqrt(), left, right, species, bond_tolerance);
        }
    }
    for _ in 0..64 {
        if contacts_are_open(
            x.view(),
            owner.as_deref(),
            species,
            &bonded,
            bond_target,
            clash_target,
        ) {
            return x.iter().all(|value| value.is_finite())
                && min_pair_distance(x.view()) >= min_sep;
        }
        let mut group_shift = vec![[0.0_f64; 3]; n_groups];
        let mut atom_shift = vec![[0.0_f64; 3]; n];
        let mut moved = false;
        for a in 0..n {
            for b in (a + 1)..n {
                let mut delta = [0.0_f64; 3];
                let mut distance_squared = 0.0;
                for axis in 0..3 {
                    delta[axis] = x[3 * a + axis] - x[3 * b + axis];
                    distance_squared += delta[axis] * delta[axis];
                }
                let distance = distance_squared.sqrt();
                let left = owner.as_ref().map(|owner| owner[a]).unwrap_or(usize::MAX);
                let right = owner.as_ref().map(|owner| owner[b]).unwrap_or(usize::MAX);
                let target = open_target(
                    a,
                    b,
                    distance,
                    left,
                    right,
                    bonded[a * n + b],
                    species,
                    bond_target,
                    clash_target,
                );
                if distance >= target {
                    continue;
                }
                let (direction, scale) = if distance < 1.0e-12 {
                    ([1.0, 0.0, 0.0], 0.5 * target)
                } else {
                    let push = 0.5 * (target - distance) / distance;
                    (delta, push)
                };
                let left_held = side_held(a, left);
                let right_held = side_held(b, right);
                if left_held && right_held {
                    continue;
                }
                // `scale` is half the gap. A frozen side stays put, so the
                // free side takes the whole gap.
                let left_weight = if right_held {
                    2.0
                } else if left_held {
                    0.0
                } else {
                    1.0
                };
                let right_weight = if left_held {
                    2.0
                } else if right_held {
                    0.0
                } else {
                    1.0
                };
                if left != usize::MAX && right != usize::MAX && left != right {
                    for axis in 0..3 {
                        group_shift[left][axis] += left_weight * scale * direction[axis];
                        group_shift[right][axis] -= right_weight * scale * direction[axis];
                    }
                } else {
                    for axis in 0..3 {
                        atom_shift[a][axis] += left_weight * scale * direction[axis];
                        atom_shift[b][axis] -= right_weight * scale * direction[axis];
                    }
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
        if let Some(owner) = owner.as_ref() {
            for atom in 0..n {
                if held(atom) {
                    continue;
                }
                let group = owner[atom];
                if group != usize::MAX {
                    for axis in 0..3 {
                        x[3 * atom + axis] += group_shift[group][axis];
                    }
                }
            }
        }
        for atom in 0..n {
            if held(atom) {
                continue;
            }
            for axis in 0..3 {
                x[3 * atom + axis] += atom_shift[atom][axis];
            }
        }
    }
    x.iter().all(|value| value.is_finite()) && min_pair_distance(x.view()) >= min_sep
}

fn quench_is_sane(cfg: &Config, energy: f64, x: ArrayView1<f64>) -> bool {
    let minimum = if cfg.species.is_some() {
        OVERLAP_SEPARATION
    } else {
        0.0
    };
    energy.is_finite() && structure_is_sane(x, minimum)
}

/// Whether a quenched point cluster has shed an atom past the potential's range.
///
/// The range is the neighbour cutoff. An atom with no neighbour inside it
/// has left that range. It is shed when the potential has stopped acting on
/// it and is still acting on the rest: its force is exactly zero while some
/// other atom still feels one. A stationary point has no force anywhere, and
/// that is not an atom past the range. With no gradient the geometry is the
/// whole test, so a chain that already holds a free atom is not frozen.
/// On Tersoff carbon half the accepted hops of a ten-atom walk held such an
/// atom. A tail that never reaches zero, as in Lennard-Jones and Morse,
/// leaves some force on every atom and never meets the test. Rigid bodies
/// and labelled species keep their own notion of contact and are not judged
/// here.
fn sheds_free_atom(cfg: &Config, x: ArrayView1<f64>, gradient: Option<ArrayView1<f64>>) -> bool {
    let n = cfg.n_points;
    if cfg.move_library.is_rigid_body()
        || cfg.species.is_some()
        || n < 2
        || x.len() != 3 * n
        || cfg.neighbour_cutoff <= 0.0
    {
        return false;
    }
    let force = |g: ArrayView1<f64>, atom: usize| (0..3).any(|axis| g[3 * atom + axis] != 0.0);
    let free = |atom: usize| match gradient {
        None => true,
        Some(g) if g.len() == 3 * n => {
            !force(g, atom) && (0..n).any(|other| other != atom && force(g, other))
        }
        Some(_) => false,
    };
    connectivity_groups(x, n, cfg.neighbour_cutoff)
        .iter()
        .any(|group| group.len() == 1 && free(group[0]))
}

fn gradient_is_converged(gradient: ArrayView1<f64>, dimensions: usize, tolerance: f64) -> bool {
    dimensions > 0
        && gradient.len() == dimensions
        && gradient
            .iter()
            .all(|v| v.is_finite() && v.abs() < tolerance)
}

/// Whether a quenched gradient may be recorded.
///
/// A rigid record accepts the root-mean-square force at the ranked
/// tolerance. The root-mean-square is at most the largest entry, so a
/// quench that met the preset infinity norm is accepted, and so is a
/// spread-out gradient whose largest entry is still above that norm.
fn answer_gradient_ok(cfg: &Config, gradient: ArrayView1<f64>, dimensions: usize) -> bool {
    if cfg.move_library.is_rigid_body() {
        crate::methods::rigid_quench::rms_is_converged(
            gradient,
            dimensions,
            crate::methods::rigid_quench::SLOPPY_RMS,
        )
    } else {
        gradient_is_converged(gradient, dimensions, cfg.record_gradient)
    }
}

/// Records an auxiliary quench without restricting exploratory adoption.
fn record_quenched_answer(
    cfg: &Config,
    ledger: &mut Ledger,
    grad: &mut Option<&mut GradFn<'_>>,
    energy: f64,
    state: ArrayView1<f64>,
    unresolved: &mut usize,
) -> Option<Array1<f64>> {
    let sane = !state.is_empty() && quench_is_sane(cfg, energy, state);
    let required = grad.is_some();
    let gradient = if sane {
        grad.as_deref_mut().and_then(|evaluate| {
            evaluate(ledger, state)
                .filter(|values| answer_gradient_ok(cfg, values.view(), state.len()))
        })
    } else {
        None
    };
    if sane && (!required || gradient.is_some()) {
        ledger.record(energy, state);
    } else {
        *unresolved += 1;
    }
    gradient
}

/// A failed stall growth holds the next growth quench until the incumbent drops.
///
/// The margin is the one the adoption uses. The test reads the incumbent
/// energy and nothing about the number of atoms, so one recommended preset
/// keeps it at every size.
fn stall_growth_waits(held: Option<f64>, incumbent: f64) -> bool {
    held.is_some_and(|energy| incumbent >= energy - 1e-9)
}

fn arm_row<'a>(arms: &'a mut Vec<ArmCount>, name: &str) -> &'a mut ArmCount {
    if let Some(index) = arms.iter().position(|arm| arm.move_type == name) {
        return &mut arms[index];
    }
    arms.push(ArmCount {
        move_type: name.to_owned(),
        ..ArmCount::default()
    });
    arms.last_mut().expect("the row was just pushed")
}

fn begin_arm(arms: &mut Vec<ArmCount>, name: &str) {
    arm_row(arms, name).proposed += 1;
}

fn finish_arm(arms: &mut Vec<ArmCount>, name: &str, accepted: bool, improved: bool) {
    let row = arm_row(arms, name);
    if accepted {
        row.accepted += 1;
    }
    if improved {
        row.improved += 1;
    }
}

/// Leaves skipped while the incumbent stays at the held energy before the
/// crossing growth is allowed to run again.
///
/// One miss is not a proof that the other packing is absent. The paper leave
/// is thousands of force calls apart, so eight skips is one retry about every
/// forty thousand calls. A five-hop test never reaches the eighth skip, and
/// the single growth it counts stays single.
const GROWTH_RETRY_LEAVES: u32 = 8;

/// True while a failed growth should stay idle.
///
/// The energy test is [`stall_growth_waits`]. At 98 atoms and above, after
/// [`GROWTH_RETRY_LEAVES`] idle leaves the pentagonal order is due again.
/// That is the leave the plateau was not taking: the live chain's 555
/// fraction sits near 0.06, under the 0.12 icosahedral cut, so that branch
/// never opened. Below 98 the displacement library still leaves the floor,
/// and a retry would move the call clock those hits were measured on.
fn growth_held(held: Option<f64>, incumbent: f64, skipped: u32, n_points: usize) -> bool {
    stall_growth_waits(held, incumbent) && (n_points < 98 || skipped < GROWTH_RETRY_LEAVES)
}

/// The arm a hop plays.
///
/// A contextual decision, when one was built, stays in front of this choice:
/// the paper preset does not build one. [`ArmSelection::Uniform`] replaces
/// the depth posterior and the floored Thompson draw with one uniform index
/// and consumes nothing else from the generator.
fn choose_arm<R: Rng + ?Sized>(
    cfg: &Config,
    context: Option<ArrayView1<f64>>,
    contextual: &mut ContextualAllocator,
    depth_allocator: &crate::allocate::DepthAllocator,
    allocator: &mut FlooredThompson,
    n_arms: usize,
    rng: &mut R,
) -> usize {
    if let Some(features) = context {
        return contextual.select(features, rng);
    }
    if cfg.arm_selection == ArmSelection::Uniform {
        return rng.random_range(0..n_arms);
    }
    if cfg.allocate_moves && cfg.depth_reward {
        return depth_allocator.select(rng);
    }
    if cfg.allocate_moves {
        return allocator.select(rng);
    }
    rng.random_range(0..n_arms)
}

/// Kernels the chain actually draws, including the energy-lattice arm.
fn search_kernels(cfg: &Config) -> Vec<ClusterMove> {
    let mut kernels = cfg.move_library.kernels(cfg);
    if cfg.displacement_only {
        kernels = vec![ClusterMove::AllPoints { step: 0.38 }];
    } else if cfg.energy_lattice && cfg.packing_cna_applies() {
        kernels.push(ClusterMove::EnergyLattice);
    }
    if !cfg.prior_soap {
        kernels.retain(|kernel| !matches!(kernel, ClusterMove::Soap { .. }));
    }
    if cfg.general_collective {
        kernels.push(ClusterMove::Plane {
            n_points: cfg.n_points,
        });
    }
    kernels
}

/// One charged dynamic-lattice exchange, or a single-point step when the
/// lattice has nothing to offer.
fn energy_lattice_proposal<R: Rng + ?Sized>(
    cfg: &Config,
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
    rng: &mut R,
) -> Array1<f64> {
    let reduced = cfg.temperature / cfg.energy_scale.max(1e-12);
    let lattice = crate::methods::lattice_search::Lattice::default();
    let coords = x.to_vec();
    if let Some(y) = lattice.propose_energy_exchange(&coords, ledger, reduced, rng) {
        return Array1::from(y);
    }
    ClusterMove::SinglePoint {
        n_points: cfg.n_points.max(1),
        step: preset::LennardJonesPreset::SINGLE_POINT_STEP * cfg.length_scale,
    }
    .propose(x, cfg.temperature, rng)
}

/// Runs the driver until the ledger is spent.
///
/// `start` is a starting configuration and `relax` performs a relaxation of the
/// requested number of steps, charging the ledger.
pub fn run<R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    rng: &mut R,
) -> Result<Outcome, SearchError> {
    run_with_gradient(cfg, start, ledger, relax, None, rng)
}

/// As [`run`], with a gradient for the soft-mode escape.
///
/// Without one, [`Config::minima_hopping`] falls back to scaling the ordinary
/// displacement, which carries the same feedback law and is what Goedecker
/// reports as strictly weaker.
pub fn run_with_gradient_settle<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    settle: Option<Settle<'_>>,
    rng: &mut R,
) -> Result<Outcome, SearchError> {
    let mut checkpoint = continue_without_checkpoint;
    run_full(
        cfg,
        start,
        ledger,
        relax,
        grad,
        None,
        None,
        settle,
        None,
        None,
        &mut checkpoint,
        rng,
    )
}

/// Runs from `start` with an optional charged gradient.
pub fn run_with_gradient<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    rng: &mut R,
) -> Result<Outcome, SearchError> {
    let mut checkpoint = continue_without_checkpoint;
    run_full(
        cfg,
        start,
        ledger,
        relax,
        grad,
        None,
        None,
        None,
        None,
        None,
        &mut checkpoint,
        rng,
    )
}

/// As [`run_with_gradient`], with value and gradient together, which is
/// what [`Config::hmc`] needs: one charge per leapfrog leaf.
pub fn run_with_energy_gradient<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    energy_grad: Option<&mut EnergyGradFn<'g>>,
    rng: &mut R,
) -> Result<Outcome, SearchError> {
    let mut checkpoint = continue_without_checkpoint;
    run_full(
        cfg,
        start,
        ledger,
        relax,
        grad,
        energy_grad,
        None,
        None,
        None,
        None,
        &mut checkpoint,
        rng,
    )
}

/// As [`run_with_energy_gradient`], with periodic observations.
///
/// [`CheckpointAction::Retire`] ends the run with the best state found.
/// The interval counts charged evaluations. It must be positive.
pub fn run_with_energy_gradient_at_checkpoints<'g, R, H>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    energy_grad: Option<&mut EnergyGradFn<'g>>,
    rng: &mut R,
    checkpoint_interval: usize,
    checkpoint: &mut H,
) -> Result<Outcome, SearchError>
where
    R: Rng + ?Sized,
    H: for<'a> FnMut(ChainCheckpoint<'a>) -> CheckpointAction,
{
    assert!(
        checkpoint_interval > 0,
        "checkpoint interval must be positive"
    );
    run_full(
        cfg,
        start,
        ledger,
        relax,
        grad,
        energy_grad,
        None,
        None,
        None,
        Some(checkpoint_interval),
        checkpoint,
        rng,
    )
}

/// As [`run_with_gradient`], with a bias supplied by the caller and left
/// behind when the run ends.
///
/// For a caller running several chains under one budget. The well-tempered
/// bias is a memory of the landscape rather than of the chain that walked it,
/// and a funnel one chain has filled is filled for the next one. Rebuilding it
/// per chain throws that away, which is not a small effect: at 75 points the
/// crossing takes on the order of a hundred thousand hops of accumulation, and
/// a bank of sixteen chains each starting from an empty bias solved 2 seeds in
/// 8 where one long chain solved 9 in 16.
///
/// Only for a single-rung run. Replica exchange gives each rung its own bias by
/// construction and there is nothing for one external bias to be.
pub fn run_with_bias<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    bias: &mut BasinBias<ClusterFingerprint>,
    rng: &mut R,
) -> Result<Outcome, SearchError> {
    assert!(
        cfg.replicas <= 1,
        "a shared bias and a replica ladder are different things: \
         each rung owns its own bias"
    );
    let mut checkpoint = continue_without_checkpoint;
    run_full(
        cfg,
        start,
        ledger,
        relax,
        grad,
        None,
        Some(bias),
        None,
        None,
        None,
        &mut checkpoint,
        rng,
    )
}

/// As [`run_with_bias`], exposing periodic observations without ending the
/// live chain or rebuilding any adaptive state.
pub fn run_with_bias_at_checkpoints<'g, R, H>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    bias: &mut BasinBias<ClusterFingerprint>,
    rng: &mut R,
    checkpoint_interval: usize,
    checkpoint: &mut H,
) -> Result<Outcome, SearchError>
where
    R: Rng + ?Sized,
    H: for<'a> FnMut(ChainCheckpoint<'a>) -> CheckpointAction,
{
    assert!(
        checkpoint_interval > 0,
        "checkpoint interval must be positive"
    );
    assert!(
        cfg.replicas <= 1,
        "a shared bias and a replica ladder are different things: \
         each rung owns its own bias"
    );
    run_full(
        cfg,
        start,
        ledger,
        relax,
        grad,
        None,
        Some(bias),
        None,
        None,
        Some(checkpoint_interval),
        checkpoint,
        rng,
    )
}

/// As [`run_with_gradient_settle`], reporting every certified quench and
/// every adoption to a minimum history, with periodic observations.
///
/// The history is the communication channel of a cooperating ensemble: a
/// private one is the isolated control, and one shared behind a lock is the
/// communicating arm. The chain keeps its coordinates, random stream,
/// allocator and bias; what it receives is exact identity and visit counts,
/// which feed the escape controller under [`Config::minima_hopping`] and the
/// per-basin deposits otherwise ([`Config::shared_deposits`]).
pub fn run_with_history_at_checkpoints<'g, R, H>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    settle: Option<Settle<'_>>,
    history: Option<&mut dyn HistoryHook>,
    rng: &mut R,
    checkpoint_interval: usize,
    checkpoint: &mut H,
) -> Result<Outcome, SearchError>
where
    R: Rng + ?Sized,
    H: for<'a> FnMut(ChainCheckpoint<'a>) -> CheckpointAction,
{
    assert!(
        checkpoint_interval > 0,
        "checkpoint interval must be positive"
    );
    run_full(
        cfg,
        start,
        ledger,
        relax,
        grad,
        None,
        None,
        settle,
        history,
        Some(checkpoint_interval),
        checkpoint,
        rng,
    )
}

/// Learned stall response. Each enabled escape is one arm under the
/// same depth-rewarded Thompson allocation the move set uses: a stall
/// draws one response and the draw is rewarded with the deepening the
/// escape realized, a signed number every stall produces, under a
/// Normal-Gamma posterior whose predictive needs no reward scale. A
/// success bit was the first version and is exactly the sparse reward
/// DepthAllocator documents as failing to switch a wrong arm off.
/// Every arm is domain-free: the trail is the objective's own descent
/// history, the climb needs a gradient alone, the restart draws fresh
/// coordinates. With one arm enabled the allocator is never consulted,
/// so single-flag behaviour is unchanged.
fn stall_arms(cfg: &Config) -> Vec<&'static str> {
    [
        cfg.trail_on_stall.then_some("trail"),
        cfg.escape_on_stall.then_some("climb"),
        // RESTART_STALL_ARM=0 keeps the restart out of the stall allocator,
        // so a restart happens only on the charged-patience rule. Measured:
        // with the stall arm in, a chain restarted 9 to 16 times a run
        // whatever RESTART_PATIENCE said, every 5 to 8k hops, shorter than
        // the descent to the shelf. A rigid chain never takes it: its
        // stall moves within the chain's own archive instead.
        (cfg.restart_on_stall
            && !cfg.move_library.is_rigid_body()
            && !std::env::var("RESTART_STALL_ARM").is_ok_and(|v| v == "0"))
        .then_some("restart"),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Each rung's temperature as a multiple of the one a single chain would hop
/// at: `ladder_top^(k/(R-1))` for rung `k` of `R`, and exactly one for a
/// single chain.
///
/// Geometric, so swap acceptance is spaced evenly along the ladder rather than
/// bunched at one end.
fn rung_ratios(replicas: usize, ladder_top: f64) -> Vec<f64> {
    let n_rep = replicas.max(1);
    (0..n_rep)
        .map(|k| {
            if n_rep == 1 {
                1.0
            } else {
                ladder_top.powf(k as f64 / (n_rep - 1) as f64)
            }
        })
        .collect()
}

/// The temperature a single chain hops at from a state at energy `e`, given the
/// one the budget-window law or the configuration holds it at.
///
/// Under [`Config::statistical_temperature`] the entropy's slope where the
/// chain stands takes over once the density of states has been refreshed,
/// clamped to a band around the configured value so a slope estimated from few
/// counts cannot freeze the chain or boil it. The band is wide enough that the
/// adaptation has somewhere to go and narrow enough that a bad estimate is
/// survivable.
fn single_chain_temperature(
    cfg: &Config,
    held: f64,
    dos: Option<&crate::dos::DensityOfStates>,
    e: f64,
) -> f64 {
    let mut single = held;
    if cfg.statistical_temperature
        && let Some(d) = dos
        && d.refreshes > 0
    {
        let (t, _) = d.temperature(e);
        if t.is_finite() && t > 0.0 {
            single = t.clamp(0.2 * cfg.temperature, 5.0 * cfg.temperature);
        }
    }
    single
}

/// The temperature a rung at `ratio` hops at from a state at energy `e`, given
/// the one the budget-window law or the configuration holds a single chain at.
///
/// The ratio multiplies what a single chain standing there would hop at, so
/// the coldest rung is that chain whichever temperature rule is in force. The
/// statistical temperature's clamp comes before the ratio, so each rung's band
/// sits around its own ladder temperature.
fn rung_temperature(
    cfg: &Config,
    ratio: f64,
    held: f64,
    dos: Option<&crate::dos::DensityOfStates>,
    e: f64,
) -> f64 {
    single_chain_temperature(cfg, held, dos, e) * ratio
}

/// The temperature a rung at `ratio` would hop at from a state at energy `e`,
/// by the rule a hop uses and without counting a step of the law.
///
/// For a rung that is not hopping: the swap weighs each state at the
/// temperature its rung would hop at, and the energy bias is built and filled
/// at the coldest rung's.
fn standing_temperature(
    cfg: &Config,
    law: &BudgetWindowTemperature,
    ledger: &Ledger,
    ratio: f64,
    dos: Option<&crate::dos::DensityOfStates>,
    e: f64,
) -> f64 {
    let held = if cfg.budget_window {
        law.peek((e - ledger.best).abs().max(1e-12), ledger.remaining())
    } else {
        cfg.temperature
    };
    rung_temperature(cfg, ratio, held, dos, e)
}

/// Deposit height on a rung at `ratio` of a ladder of `n_rep`.
///
/// Under [`Config::bias_by_rung`] it follows the rung's temperature, so a
/// deposit weighs the same against the temperature each rung hops at and the
/// hottest carries the configured height.
fn rung_height(cfg: &Config, n_rep: usize, ratio: f64) -> f64 {
    if cfg.bias_by_rung && n_rep > 1 {
        cfg.bias_height * ratio / cfg.ladder_top
    } else {
        cfg.bias_height
    }
}

/// Log acceptance of rungs `k` and `j` exchanging the states they hold.
///
/// Rung `k` hops on `exp(-(E + V_k + S) / T_k)`, with `V_k` its own bias and
/// `S` the biases every rung shares. `own_k` is `E + V_k` at the state rung
/// `k` holds and at the one rung `j` holds, `own_j` is `E + V_j` at the state
/// rung `j` holds and at rung `k`'s, and `shared` is `S` at rung `k`'s state
/// and at rung `j`'s. `S` is one function on every rung, so it enters only
/// through the difference of the inverse temperatures and drops out of a flat
/// ladder.
#[cfg(test)]
fn swap_log_acceptance(
    own_k: [f64; 2],
    own_j: [f64; 2],
    shared: [f64; 2],
    t_k: f64,
    t_j: f64,
) -> f64 {
    (own_k[0] - own_k[1]) / t_k
        + (own_j[0] - own_j[1]) / t_j
        + (shared[0] - shared[1]) * (1.0 / t_k - 1.0 / t_j)
}

/// What a rung's acceptance can read at one state besides the rung's own bias.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct HopTerms {
    energy: f64,
    funnel: f64,
    pile: f64,
    energy_bias: f64,
}

/// What every rung's acceptance weighs a state by, and so what a swap between
/// two of them has to exchange.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
enum RungWeight {
    /// `exp(-(E + V_k + F + P + B) / T_k)`, with `V_k` the rung's own bias,
    /// `F` the funnel bias, `P` the packing pile and `B` the energy bias: the
    /// Metropolis rule. Minima hopping's threshold has no weight of its own,
    /// and its ladder is swapped by this one.
    Boltzmann,
    /// `exp(-(E + V_k + F + P) / T_k)`: the flat-histogram rule's first sweep,
    /// before its cost exists, which does not read the energy bias.
    FirstSweep,
    /// `exp(-c(E) - (V_k + F) / T_k)`, with `c` the flat-histogram cost, one
    /// function of the energy that every rung reads and none divides by its
    /// temperature.
    Flat,
}

#[cfg(test)]
impl RungWeight {
    /// The weight the acceptance test applies, given whether the
    /// flat-histogram cost exists yet.
    fn of(cfg: &Config, flat_cost: bool) -> Self {
        if !cfg.flat_histogram || cfg.minima_hopping {
            Self::Boltzmann
        } else if flat_cost {
            Self::Flat
        } else {
            Self::FirstSweep
        }
    }
}

/// Log acceptance of rungs `k` and `j` exchanging the states they hold, for
/// rungs that weigh a state by `weight`.
///
/// `at_k` is what the weight can read at the state rung `k` holds and `at_j`
/// at the one rung `j` holds; `own_k` is `V_k` at those two states and `own_j`
/// is `V_j` at rung `j`'s state and at rung `k`'s. The flat cost is the same
/// on both rungs at either state, so it cancels from the factor and takes the
/// energy with it: under that weight a swap exchanges the biases alone.
#[cfg(test)]
fn exchange_log_acceptance(
    weight: RungWeight,
    at_k: HopTerms,
    at_j: HopTerms,
    own_k: [f64; 2],
    own_j: [f64; 2],
    t_k: f64,
    t_j: f64,
) -> f64 {
    let energy = |at: HopTerms| match weight {
        RungWeight::Flat => 0.0,
        RungWeight::Boltzmann | RungWeight::FirstSweep => at.energy,
    };
    let shared = |at: HopTerms| match weight {
        RungWeight::Boltzmann => at.funnel + at.pile + at.energy_bias,
        RungWeight::FirstSweep => at.funnel + at.pile,
        RungWeight::Flat => at.funnel,
    };
    swap_log_acceptance(
        [energy(at_k) + own_k[0], energy(at_j) + own_k[1]],
        [energy(at_j) + own_j[0], energy(at_k) + own_j[1]],
        [shared(at_k), shared(at_j)],
        t_k,
        t_j,
    )
}

/// Lowest unrelaxed exchange of an inner large atom with an outer small one.
///
/// Sixteen pair trials, each one evaluation. The geometry is the incumbent
/// minimum with those two coordinates exchanged. A trial whose unrelaxed
/// energy sits above `ceiling` is discarded. `None` when every trial does,
/// or the budget runs out before a keeper.
fn best_wide_exchange<F>(
    x: ArrayView1<f64>,
    sigma: &[f64],
    ceiling: f64,
    mut evaluate: F,
) -> Option<Array1<f64>>
where
    F: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let n = sigma.len();
    if x.len() != 3 * n || n < 4 {
        return None;
    }
    let mean = sigma.iter().copied().sum::<f64>() / n as f64;
    let mut centre = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            centre[k] += x[3 * i + k];
        }
    }
    for value in &mut centre {
        *value /= n as f64;
    }
    let radii: Vec<f64> = (0..n)
        .map(|i| {
            let mut radius = 0.0;
            for k in 0..3 {
                let delta = x[3 * i + k] - centre[k];
                radius += delta * delta;
            }
            radius
        })
        .collect();
    let mut large: Vec<usize> = (0..n).filter(|&i| sigma[i] > mean + 1e-12).collect();
    let mut small: Vec<usize> = (0..n).filter(|&i| sigma[i] <= mean + 1e-12).collect();
    if large.is_empty() || small.is_empty() {
        return None;
    }
    large.sort_by(|&left, &right| radii[left].total_cmp(&radii[right]));
    small.sort_by(|&left, &right| radii[right].total_cmp(&radii[left]));
    large.truncate(4);
    small.truncate(4);
    let mut best_energy = f64::INFINITY;
    let mut best: Option<Array1<f64>> = None;
    for &large_i in &large {
        for &small_i in &small {
            if large_i == small_i {
                continue;
            }
            let mut y = x.to_owned();
            for k in 0..3 {
                y.swap(3 * large_i + k, 3 * small_i + k);
            }
            let Some(energy) = evaluate(y.view()) else {
                break;
            };
            if energy < best_energy {
                best_energy = energy;
                best = Some(y);
            }
        }
    }
    best.filter(|_| best_energy <= ceiling)
}

/// Body count at and below which the rigid factor may climb to 24.
///
/// An all-body kick on a ten-water chart still falls into the target
/// basin at that factor. A larger chart caps the factor at
/// [`RIGID_FACTOR_CAP_LARGE`] and can still cool when the acceptance
/// rate is high.
const RIGID_HOT_BODIES: usize = 10;
/// Highest rigid factor on a chart with more than [`RIGID_HOT_BODIES`] bodies.
const RIGID_FACTOR_CAP_LARGE: f64 = 4.0;
/// Metropolis acceptance the orientation amplitude aims at.
///
/// The ratio a Cartesian basin-hopping step is adjusted toward, and the
/// default of GMIN's acceptance-ratio update. A chart with more than ten
/// bodies keeps its acceptance temperature at the factor cap and moves this
/// amplitude instead, so the two controls do not chase the same ratio.
const RIGID_ACCEPT_TARGET: f64 = 0.5;
/// Quenched trials between orientation-amplitude updates.
const RIGID_ACCEPT_BLOCK: usize = 50;
/// Patience windows with no new lowest minimum before a second rigid chain
/// walks another oxygen cage.
///
/// The main walker is not moved and its generator is not advanced. Forty
/// windows is a quiet stretch long enough to be a stalled cage, and the
/// walk itself is capped at the same length so the main chain resumes.
const RIGID_CAGE_WINDOWS: usize = 40;

/// Seats in a rigid basin index.
///
/// The lookup walks every seat the triangle test cannot skip, so an open
/// list makes every later hop slower. Past this many seats a new region
/// reuses the latest one. The cap is not applied to an atomic cluster.
const RIGID_BASIN_CAP: usize = 2_048;
/// Orientation steps in the shell turn that opens the second walk.
///
/// Two steps still quench back into the cage. Three steps, the preset
/// orientation amplitude taken three times, land in a different basin.
const RIGID_SHELL_STEPS: f64 = 3.0;
/// Multiplier applied when a block misses [`RIGID_ACCEPT_TARGET`].
const RIGID_STEP_FACTOR: f64 = 1.05;

/// Next orientation amplitude after one acceptance block.
///
/// Above the target the step grows, below it the step shrinks, and the
/// result stays inside `(1e-3, 1]`. One is the orientation cap already
/// stored on the move: a larger kick is the regime the factor cap exists
/// to keep a large water chart out of.
/// True when some rigid centre has moved by more than `tol`.
///
/// A cycle reversal turns bodies about their own centres. That is a new
/// proton arrangement of the same cage, not a new oxygen framework.
fn rigid_centres_parted(
    before: ArrayView1<f64>,
    after: ArrayView1<f64>,
    n: usize,
    tol: f64,
) -> bool {
    if n == 0 || before.len() < 3 * n || after.len() < 3 * n {
        return false;
    }
    let tol2 = tol * tol;
    (0..n).any(|i| {
        let mut d2 = 0.0;
        for k in 0..3 {
            let d = before[3 * i + k] - after[3 * i + k];
            d2 += d * d;
        }
        d2 > tol2
    })
}

fn rigid_cage_kernel(mv: &ClusterMove) -> bool {
    matches!(
        mv,
        ClusterMove::RigidRelocate { .. }
            | ClusterMove::RigidReorient { .. }
            | ClusterMove::RigidTranslateOne { .. }
            | ClusterMove::RigidKick { .. }
    )
}

/// Quantised oxygen-oxygen distances, so proton isomers of one cage share a key.
///
/// The bin width is a fraction of the preset contact length. A rearrangement
/// that changes a contact moves at least one distance by more than that bin.
fn rigid_cage_key(state: ArrayView1<f64>, n: usize, quantum: f64) -> u64 {
    let quantum = if quantum > 0.0 { quantum } else { 1.0 };
    if n < 2 || state.len() < 3 * n {
        return 0;
    }
    let mut bins = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut distance = 0.0;
            for k in 0..3 {
                let delta = state[3 * i + k] - state[3 * j + k];
                distance += delta * delta;
            }
            bins.push((distance.sqrt() / quantum).round() as i64);
        }
    }
    bins.sort_unstable();
    let mut hash = 0xcbf29ce484222325u64;
    hash ^= n as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    for bin in bins {
        hash ^= bin as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Turn the outer half of a rigid cage about its core.
///
/// The angle is a multiple of the preset orientation step. The inner half
/// stays, so the turn is a rearrangement of the surface about the hollow
/// rather than a rigid rotation of the whole cluster.
fn rigid_shell_turn(state: ArrayView1<f64>, n: usize, axis: usize, angle: f64) -> Array1<f64> {
    let mut y = state.to_owned();
    if n < 4 || y.len() < 6 * n || !angle.is_finite() {
        return y;
    }
    let mut com = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            com[k] += y[3 * i + k];
        }
    }
    for value in &mut com {
        *value /= n as f64;
    }
    let mut radial = vec![0.0; n];
    for i in 0..n {
        let mut r = 0.0;
        for k in 0..3 {
            let d = y[3 * i + k] - com[k];
            r += d * d;
        }
        radial[i] = r;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| radial[a].total_cmp(&radial[b]));
    let (sin, cos) = angle.sin_cos();
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    for &i in &order[n / 2..] {
        let mut p = [0.0; 3];
        for k in 0..3 {
            p[k] = y[3 * i + k] - com[k];
        }
        let pa = cos * p[a] - sin * p[b];
        let pb = sin * p[a] + cos * p[b];
        p[a] = pa;
        p[b] = pb;
        for k in 0..3 {
            y[3 * i + k] = com[k] + p[k];
        }
    }
    y
}

/// True when every pair of rigid centres is at least `min_d` apart.
fn rigid_centres_clear(state: ArrayView1<f64>, n: usize, min_d: f64) -> bool {
    if n < 2 || state.len() < 3 * n || !(min_d > 0.0) {
        return false;
    }
    let min2 = min_d * min_d;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for k in 0..3 {
                let d = state[3 * i + k] - state[3 * j + k];
                d2 += d * d;
            }
            if d2 < min2 {
                return false;
            }
        }
    }
    true
}

fn rigid_step_scale(scale: f64, accepted: usize, trials: usize) -> f64 {
    if trials == 0 {
        return scale;
    }
    let rate = accepted as f64 / trials as f64;
    let next = if rate > RIGID_ACCEPT_TARGET {
        scale * RIGID_STEP_FACTOR
    } else if rate < RIGID_ACCEPT_TARGET {
        scale / RIGID_STEP_FACTOR
    } else {
        scale
    };
    next.clamp(1.0e-3, 1.0)
}

#[cfg(test)]
mod rigid_step_tests {
    use super::rigid_step_scale;

    #[test]
    fn a_low_acceptance_shrinks_the_orientation_step_and_a_high_one_grows_it() {
        let shrunk = rigid_step_scale(1.0, 10, 50);
        assert!(
            (shrunk - 1.0 / 1.05).abs() < 1e-12,
            "low acceptance left the step at {shrunk}"
        );
        let grown = rigid_step_scale(0.8, 40, 50);
        assert!(
            (grown - 0.84).abs() < 1e-12,
            "high acceptance left the step at {grown}"
        );
        assert_eq!(
            rigid_step_scale(1.0, 40, 50),
            1.0,
            "the step grew past the orientation cap"
        );
        assert!(
            (rigid_step_scale(0.9, 25, 50) - 0.9).abs() < 1e-12,
            "an acceptance of one half moved the step"
        );
    }
}

fn rigid_factor_cap(n_bodies: usize) -> f64 {
    if n_bodies > RIGID_HOT_BODIES {
        RIGID_FACTOR_CAP_LARGE
    } else {
        24.0
    }
}

/// True when two species differ in diameter by at least a fifth.
///
/// At that mismatch an exchange of every misplaced label leaves the
/// low-energy funnel. The hop proposes a soft constant-energy (NVE)
/// escape instead. Without the `rgsaddle` feature that escape is not
/// compiled, the run stops, and [`Outcome::nve_unavailable`] is set.
/// The library kick is not substituted.
fn species_count(species: &[u32]) -> usize {
    let mut labels = species.to_vec();
    labels.sort_unstable();
    labels.dedup();
    labels.len()
}

/// The all-pairs quench is an atomic binary homotop.
///
/// Water carries oxygen and hydrogen labels too, and so does a rigid
/// body. Swapping those atoms breaks the molecule. The polish belongs
/// on an atomic library, where the labels are the sites being ordered.
fn polishes_unlike_pairs(cfg: &Config) -> bool {
    if cfg.move_library.is_molecular() || cfg.move_library.is_rigid_body() {
        return false;
    }
    cfg.species
        .as_deref()
        .is_some_and(|species| species_count(species) >= 2)
}

fn species_groups(species: &[u32]) -> (Vec<usize>, Vec<usize>) {
    let mut labels = Vec::new();
    for label in species {
        if !labels.contains(label) {
            labels.push(*label);
        }
    }
    let a = labels.first().copied().unwrap_or(0);
    let b = labels.get(1).copied().unwrap_or(a);
    let group_a: Vec<usize> = species
        .iter()
        .enumerate()
        .filter_map(|(index, label)| (*label == a).then_some(index))
        .collect();
    let group_b: Vec<usize> = species
        .iter()
        .enumerate()
        .filter_map(|(index, label)| (*label == b).then_some(index))
        .collect();
    (group_a, group_b)
}

fn large_diameter_ratio(cfg: &Config) -> bool {
    let Some(sigma) = cfg.species_sigma.as_ref() else {
        return false;
    };
    if sigma.len() < 2 {
        return false;
    }
    let mut smallest = f64::INFINITY;
    let mut largest = 0.0_f64;
    for value in sigma {
        if value.is_finite() && *value > 0.0 {
            smallest = smallest.min(*value);
            largest = largest.max(*value);
        }
    }
    smallest.is_finite() && largest / smallest >= 1.2
}
/// How far a partial relaxation still had to descend, over the recent trials
/// that went on to a full relaxation.
///
/// The energy screen compares a partially relaxed trial with the incumbent, so
/// the margin it needs is the descent the screening pass leaves undone, and
/// that is a property of the potential and the screening length rather than a
/// constant. On rigid TIP4P a 15-step screen leaves tens of kJ/mol; on LJ a
/// 25-step screen leaves a few epsilon.
///
/// Trials the screen passes are the ones that had little left to descend, so
/// a window of those alone underestimates the descent of the trials it
/// refuses and the screen tightens on itself: on TIP4P20 it refused 99.7 per
/// cent of hops. Every [`ScreenDescent::AUDIT`]th refusal is relaxed in full
/// instead, so the window also samples the population the screen judges.
#[derive(Debug, Default)]
struct ScreenDescent {
    drops: std::collections::VecDeque<f64>,
    allowance: f64,
    refusals: usize,
}

impl ScreenDescent {
    const WINDOW: usize = 256;
    const WARMUP: usize = 8;
    const QUANTILE: f64 = 0.9;
    const AUDIT: usize = 8;

    /// Called on a refusal; true when this one is audited instead.
    fn audit(&mut self) -> bool {
        self.refusals += 1;
        self.refusals.is_multiple_of(Self::AUDIT)
    }

    fn observe(&mut self, drop: f64) {
        if !drop.is_finite() {
            return;
        }
        if self.drops.len() == Self::WINDOW {
            self.drops.pop_front();
        }
        self.drops.push_back(drop.max(0.0));
        if self.drops.len() >= Self::WARMUP {
            let mut sorted: Vec<f64> = self.drops.iter().copied().collect();
            let rank = ((sorted.len() - 1) as f64 * Self::QUANTILE).round() as usize;
            let (_, q, _) = sorted.select_nth_unstable_by(rank, f64::total_cmp);
            self.allowance = *q;
        }
    }

    /// The descent a screened trial is credited with: zero until the warmup
    /// has been observed, then the upper decile of the window.
    fn allowance(&self) -> f64 {
        self.allowance
    }
}

/// Continue a full relaxation that ended below the incumbent until it stops
/// descending, `steps` at a time and at most [`FINISH_ROUNDS`] more times.
///
/// Only a converged structure can be recorded, so a relaxation that stopped
/// on its step cap below the incumbent is the one trial whose convergence
/// decides whether the run has found anything. A continuation from a
/// converged point stops at its first gradient.
fn finish_improving(
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut out: (f64, Array1<f64>),
    steps: usize,
) -> (f64, Array1<f64>) {
    for _ in 0..FINISH_ROUNDS {
        if !(out.0 < ledger.best - 1e-10) || ledger.remaining() == 0 {
            break;
        }
        let more = relax(ledger, out.1.view(), steps);
        let descended = more.0 < out.0 - 1e-10 * (1.0 + out.0.abs());
        if more.0 <= out.0 {
            out = more;
        }
        if !descended {
            break;
        }
    }
    out
}

/// Continuations [`finish_improving`] grants one relaxation.
const FINISH_ROUNDS: usize = 8;

/// Keep relaxing a rigid trial until the root-mean-square force is at the
/// record tolerance, or the block stops descending.
///
/// A fixed step cap on a rigid chart converges a small fraction of the
/// quenches. A trial that has reached the global basin is then not a
/// record, and the chain walks on. The continuation is the same block
/// length, up to [`FINISH_ROUNDS`] more times, whether or not the trial
/// is below the incumbent.
fn finish_rigid<'g>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: &mut Option<&mut GradFn<'g>>,
    mut out: (f64, Array1<f64>),
) -> (f64, Array1<f64>) {
    if !cfg.move_library.is_rigid_body() {
        return out;
    }
    for _ in 0..FINISH_ROUNDS {
        if ledger.remaining() == 0 {
            break;
        }
        let rms = match grad.as_deref_mut() {
            Some(evaluate) => match evaluate(ledger, out.1.view()) {
                Some(gradient) => crate::methods::rigid_quench::rms_force(gradient.view()),
                None => break,
            },
            None => break,
        };
        if rms <= cfg.record_gradient {
            break;
        }
        let more = relax(ledger, out.1.view(), cfg.relax_steps);
        let descended = more.0 < out.0 - 1e-10 * (1.0 + out.0.abs());
        if more.0 <= out.0 {
            out = more;
        }
        if !descended {
            break;
        }
    }
    out
}

/// A caller stop during the opening quench is not a refused potential.
///
/// The ledger may hold no finite minimum yet. The search still returns
/// that record, and it does not draw another start.
fn outcome_when_start_was_stopped(ledger: &Ledger, x: &Array1<f64>, energy: f64) -> Outcome {
    Outcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        final_state: Some(x.clone()),
        final_energy: energy,
        charged: ledger.spent(),
        stop: Some(HopStop::CallerStop),
        ..Outcome::default()
    }
}

fn run_full<'g, R, H>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    mut energy_grad: Option<&mut EnergyGradFn<'g>>,
    external_bias: Option<&mut BasinBias<ClusterFingerprint>>,
    mut settle: Option<Settle<'_>>,
    mut history: Option<&mut dyn HistoryHook>,
    checkpoint_interval: Option<usize>,
    checkpoint: &mut H,
    rng: &mut R,
) -> Result<Outcome, SearchError>
where
    R: Rng + ?Sized,
    H: for<'a> FnMut(ChainCheckpoint<'a>) -> CheckpointAction,
{
    let _caller_stop = ledger.bind_stop();
    // A target on the configuration is the same stop a caller can raise
    // on the ledger. A ledger target already set is left alone, so an
    // explicit `stop_below` still wins.
    if ledger.target().is_none()
        && let Some(energy) = cfg.target_energy
    {
        ledger.stop_below(energy);
    }
    let n = cfg.n_points;
    let (continuous_symmetry_classes, csm_reseed_hops) = match cfg.continuous_symmetry {
        ContinuousSymmetry::Off => (None, 0),
        ContinuousSymmetry::Inversion { interval }
        | ContinuousSymmetry::Measure { interval, .. } => {
            assert!(
                interval > 0,
                "continuous-symmetry interval must be positive"
            );
            assert!(
                cfg.active_region.is_none() && cfg.frozen.is_none(),
                "continuous-symmetry projection requires a free non-periodic point set"
            );
            let classes = cfg.species.clone().unwrap_or_else(|| vec![0; n]);
            assert_eq!(
                classes.len(),
                n,
                "continuous-symmetry classes must match n_points"
            );
            let reseed = match cfg.continuous_symmetry {
                ContinuousSymmetry::Measure { reseed_hops, .. } => reseed_hops,
                _ => 0,
            };
            (Some(classes), reseed)
        }
    };
    // The descriptor and the metric have to agree. A shape distance is
    // computed from coordinates, so keying on it means passing coordinates
    // through rather than reducing them to a sorted distance spectrum first;
    // handing the spectrum to a shape metric would compare the wrong objects
    // and quietly return distances that mean nothing.
    // One chain per rung, each with its own bias, alive for the whole run.
    //
    // The bias has to persist per replica. Advancing a rung by calling this
    // function again would construct a fresh one every slice, and a
    // well-tempered bias rebuilt every fifty hops has nothing to accumulate:
    // measured that way an LJ38 run registered 18 basins instead of about 200.
    let n_rep = cfg.replicas.max(1);
    let ratios = rung_ratios(n_rep, cfg.ladder_top);
    assert!(
        !(cfg.delayed_acceptance && n_rep > 1),
        "delayed acceptance needs a single chain: a hop its surrogate decides \
         weighs the bare energy and one it abstains on weighs the biases, so \
         no swap between rungs can be balanced"
    );
    // The first quench supplies a stable canonical reference before the bias
    // is built. Reporting it as a minimum additionally requires the same
    // geometry and gradient contract as every subsequent quench.
    let attempts = ledger.start_attempts.max(1);
    let mut x = start.to_owned();
    let mut e = f64::NAN;
    let mut initial_validation_gradient: Option<Array1<f64>> = None;
    let mut initial_recordable = false;
    let gradient_required = grad.is_some();
    for attempt in 0..attempts {
        // `remaining` is also zero when the caller has stopped. That is a
        // finished search, not a potential that refused every start.
        if ledger.stopped() {
            return Ok(outcome_when_start_was_stopped(ledger, &x, e));
        }
        if attempt > 0 {
            if ledger.remaining() == 0 {
                return Err(SearchError::NoFiniteStart { attempts: attempt });
            }
            let mut fresh = random_cluster_in_radius(
                n,
                cfg.start_radius().max(cfg.min_separation),
                cfg.min_separation,
                rng,
            );
            if fresh.len() != x.len() {
                fresh = x.clone();
                if let Some(coord) = fresh.first_mut() {
                    *coord += 0.25;
                }
            }
            ledger.note_start_redraw(fresh.view());
            x = fresh;
        }
        let quenched = relax(ledger, x.view(), cfg.relax_steps);
        e = quenched.0;
        x = quenched.1;
        if !e.is_finite() {
            if ledger.stopped() {
                return Ok(outcome_when_start_was_stopped(ledger, &x, e));
            }
            if attempt + 1 == attempts {
                return Err(SearchError::NoFiniteStart {
                    attempts: attempt + 1,
                });
            }
            continue;
        }
        let initial_sane = quench_is_sane(cfg, e, x.view());
        let nested = initial_sane.then(|| {
            grad.as_deref_mut().and_then(|g| {
                g(ledger, x.view()).filter(|values| answer_gradient_ok(cfg, values.view(), x.len()))
            })
        });
        initial_validation_gradient = nested.flatten();
        initial_recordable = initial_sane
            && (ledger.below_target(e)
                || !gradient_required
                || initial_validation_gradient.is_some());
        break;
    }
    if !e.is_finite() {
        if ledger.stopped() {
            return Ok(outcome_when_start_was_stopped(ledger, &x, e));
        }
        return Err(SearchError::NoFiniteStart { attempts });
    }
    if initial_recordable {
        ledger.record(e, x.view());
        ledger.mark_last_recorded(true);
    }
    let mut current_validation_gradient = initial_validation_gradient;
    let canonical_reference = x.clone();
    let mut biases: Vec<BasinBias<ClusterFingerprint>> = ratios
        .iter()
        .map(|&ratio| {
            BasinBias::new(
                ClusterFingerprint::of_config(cfg, &canonical_reference),
                cfg.merge_radius,
                rung_height(cfg, n_rep, ratio),
                cfg.bias_gamma,
            )
        })
        .collect();
    // Geometric ladder, so swap acceptance is spaced evenly rather than
    // bunched at one end.
    let mut temps: Vec<f64> = ratios.iter().map(|r| cfg.temperature * r).collect();
    // Under `LadderMode::NonReversible` the geometric schedule is a
    // placeholder that the pilot replaces with a ladder built from the
    // measured energy fluctuation, and the barrier estimator moves afterwards.
    let mut ladder = (n_rep > 1 && cfg.ladder_mode.sweeps())
        .then(|| crate::tempering::Ladder::from_temperatures(&temps, cfg.ladder_mode.scheme()));
    // The index process runs under every mode, so the arms are compared on
    // transport and not only on swap counts.
    let mut transport = (n_rep > 1).then(|| crate::tempering::IndexProcess::new(n_rep));
    let mut sweeps = 0usize;
    let mut cyclic_offers = 0usize;
    let mut adaptations = 0usize;
    // Welford accumulator over the cold rung's energies, which the first
    // ladder is built from.
    let mut pilot: (u64, f64, f64) = (0, 0.0, 0.0);
    let mut ladder_built = !cfg.ladder_mode.adapts();
    let _exchange = MetropolisExchange;
    let mut swaps_tried = 0usize;
    let mut swaps_accepted = 0usize;
    let mut rep = 0usize;
    let mut since_swap = 0usize;
    // A caller's bias is used in place of the first rung's, and put back when
    // the run ends, so the next chain starts where this one left off.
    let (mut bias, carried) = match external_bias {
        Some(b) => {
            let fresh = biases.remove(0);
            (std::mem::replace(b, fresh), Some(b))
        }
        None => (biases.remove(0), None),
    };
    let mut chains: Vec<OccupiedMinimum> = Vec::new();
    // One sampler per rung, parked and taken alongside the bias.
    //
    // Per chain and not global. A hot rung crosses barriers a cold rung cannot
    // and its trajectories see a differently conditioned region, so the two
    // converge to different step sizes and different metric estimates. A swap
    // moves configurations between rungs; the adaptation stays with the
    // temperature it was learned at, which is why this is parked with the bias
    // rather than carried with the state.
    let mut hop_parked: Vec<crate::hmc::hop::HopChain> = Vec::new();
    let mut hop = cfg.hmc.as_ref().map(crate::hmc::hop::HopChain::new);
    if let Some(hmc) = cfg.hmc.as_ref() {
        for _ in 1..n_rep {
            hop_parked.push(crate::hmc::hop::HopChain::new(hmc));
        }
    }
    #[cfg(feature = "ira")]
    if cfg.shape_keyed {
        bias = bias.with_metric(Box::new(crate::shape::IraMetric::default()));
    }
    #[cfg(not(feature = "ira"))]
    assert!(
        !cfg.shape_keyed,
        "shape keying needs the `ira` feature; without it the threshold would \
         silently remain a descriptor-space number"
    );

    let mut kernels = search_kernels(cfg);
    let mut general_taboo = general::FingerprintTaboo::default();
    let mut collective = general::CollectiveShare::default();
    let mut regrow_share = general::CollectiveShare::patient();
    let mut regrow_temperature = general::MoveTemperature::default();
    // Which kernel to propose from is learned rather than drawn uniformly. The
    // useful move changes as the search moves through the landscape, so the
    // evidence is discounted and a decaying floor keeps every kernel reachable.
    let mut allocator = FlooredThompson::new(kernels.len());
    // A water rewire is the move that improves the incumbent. The prior
    // starts it ahead of the shake; the improvement rate takes over.
    if cfg.improvement_rewards {
        for (index, kernel) in kernels.iter().enumerate() {
            if matches!(kernel, ClusterMove::GroupRewire { .. }) {
                allocator.credit(index, crate::catalog::molecular::WATER_REWIRE_PRIOR);
            }
        }
    }
    // Rewarded by the depth a move reaches rather than by whether it was
    // accepted. See [`crate::allocate::DepthAllocator`].
    let mut depth_allocator = crate::allocate::DepthAllocator::new(kernels.len());
    // Per-arm draws and accepts. The crate's own methods note says a solve
    // count cannot tell a mechanism that works poorly from one that does not
    // run, and a move set is exactly where that applies: an arm the allocator
    // learns to avoid and an arm that fires and is always rejected produce the
    // same total.
    let mut arm_draws = vec![0usize; kernels.len()];
    let mut arm_accepts = vec![0usize; kernels.len()];
    let mut arm_best = vec![f64::INFINITY; kernels.len()];
    // Delayed acceptance: a surrogate for the quenched energy decides the
    // first stage for one evaluation, and only survivors are quenched. The
    // incumbent's surrogate value travels with the chain, because reversibility
    // needs the same number on both sides of the ratio.
    let mut surrogate = if cfg.delayed_acceptance {
        Some(crate::delayed::Surrogate::new())
    } else {
        None
    };
    let mut surrogate_here: Option<f64> = None;
    let mut unconverged_records = usize::from(!initial_recordable);
    let mut pending_raw: Option<(f64, f64)> = None;
    let mut pending_design: Option<Array1<f64>> = None;
    // A posterior over what to build, consulted when a growth move is drawn.
    // The allocator decides which move; this decides the move's parameters,
    // which is the one place a model can change what a hop reaches rather than
    // where it goes. Costs no charged evaluations: candidates are built and
    // featured without calling the objective.
    let mut constructor = if cfg.move_library.learns_construction() {
        Some(crate::construct::Constructor::new(cfg.construct_width))
    } else {
        None
    };
    let mut pending_features: Option<Array1<f64>> = None;
    // The temperature is the law rather than a setting: the design point
    // clamped between the sphere-model descent ceiling and the birth-death
    // escape floor, with the barrier estimated from the uphill steps the chain
    // declines. On a funnelled landscape the window is routinely empty, which
    // is counted rather than hidden.
    let mut law = BudgetWindowTemperature::new(3 * n, cfg.theta);
    // The deposit height is set from the escape gaps observed rather than
    // fixed, since a height above the gap empties a basin on one revisit and
    // the gap is a property of the landscape.
    let mut height = AdaptiveHeight::new(0.1, cfg.height_revisits, cfg.bias_height);
    // Starts at the configured radius and narrows. The paper's rule takes the
    // start from the spread of an initial population; here the configured value
    // is the start, so a run that does not anneal is unchanged and one that does
    // begins where the fixed version sat rather than somewhere new.
    let mut diversity =
        DiversityAnnealer::from_initial(cfg.merge_radius).with_final_fraction(cfg.diversity_floor);
    let mut stall = StallDetector::new(cfg.stall_patience);
    let mut quenched: Vec<(usize, usize, f64, bool)> = Vec::new();
    // The transition graph the absorbing-chain escape solves, and the archive
    // it lands on; allocated only when asked for.
    let mut superbasin = if cfg.superbasin_report || cfg.superbasin_escape {
        Some(crate::superbasin::SuperbasinEscape::new())
    } else {
        None
    };
    let mut sb_last_jump = 0usize;
    let mut trace: Option<Vec<f64>> = if cfg.energy_trace {
        Some(Vec::new())
    } else {
        None
    };
    let mut improvements: Vec<(usize, usize, usize, f64)> = Vec::new();
    if initial_recordable {
        improvements.push((0, ledger.spent(), 0, e));
    }
    let mut soft_escapes = 0usize;
    let mut soft_crossed = 0usize;
    // Kept here rather than in a StallDetector because the threshold is not a
    // constant: it is set from the longest quiet stretch this chain has already
    // survived.
    let mut radius = StepCalibrator::new(
        cfg.calibrate_quantile,
        cfg.calibrate_warmup,
        cfg.merge_radius,
    );
    // The pair-energy ratio, tuned to the paper's acceptance target. Started
    // at their reported converged value so a short run is not spent finding it.
    let mut angular_ratio = 0.42_f64;
    let mut angular_tried = 0usize;
    let mut angular_accepted = 0usize;
    // Intercept, screened energy, how far the partial relaxation moved in
    // descriptor space, and the incumbent's distance from it. All cheap, all
    // already computed by the margin screen.
    let mut contextual = ContextualAllocator::new(kernels.len(), 3, cfg.contextual_floor);
    let mut screen = Screen::new(
        4,
        cfg.bayes_warmup,
        cfg.bayes_exploration,
        cfg.bayes_threshold,
    );
    let mut tabu: Vec<Array1<f64>> = Vec::new();
    let mut tabu_hits = 0usize;
    let mut funnels = crate::funnel_spectral::FunnelSpectrum::new();
    let mut funnel_split: Option<crate::funnel_spectral::Partition> = None;
    // Spectral funnel bias: same fingerprint as the discrete basin bias, CV =
    // Fiedler coordinate of the accepted-hop graph. Only allocated when
    // track_funnels is on so a plain run pays nothing.
    let mut spectral: Option<crate::spectral::SpectralBias<ClusterFingerprint>> =
        if cfg.track_funnels {
            let mut sb = crate::spectral::SpectralBias::new(
                ClusterFingerprint::of_config(cfg, &canonical_reference),
                cfg.merge_radius,
                cfg.bias_height,
                cfg.bias_gamma,
                0.35,
            );
            sb.refit_every = cfg.funnel_period.max(8);
            sb.min_nodes = 8;
            Some(sb)
        } else {
            None
        };
    // The packing pile. Off unless asked for, because it costs a single
    // linkage test the first time each basin is seen, and on unless the
    // caller has a reason: the cell-grain pile alone has never taken an
    // LJ75 chain out of the icosahedral funnel.
    let mut pave: Option<crate::catalog::PackingPave> = std::env::var("CATALOG_PACKING_PAVE")
        .is_ok_and(|value| value == "1")
        .then(crate::catalog::PackingPave::new);
    // The seam bank. Shares partial progress along the road out of the
    // occupied packing: every accepted quench is offered by its DECAF
    // distance from the run's floor, and a stuck chain restarts from the
    // furthest banked structure instead of from the floor. Nothing here
    // touches the acceptance rule -- that is what Hop.road_priced
    // requires and what the community pile violated.
    let mut seam: Option<crate::catalog::SeamBank> = std::env::var("CATALOG_SEAM_LADDER")
        .is_ok_and(|value| value == "1")
        .then(crate::catalog::SeamBank::new);
    // How long a chain walks without improvement before it reseeds from
    // the frontier. The splitting gain is per round: n p arrivals per
    // stage only accumulates over repeated episodes, and a patience tied
    // to the escape machinery's 5000 gives one or two episodes in a
    // twelve-thousand-hop run, which is no ensemble at all. Basins here
    // are about 8.5 accepted hops apart, so 800 quiet hops is roughly a
    // hundred basins without improvement: locally exhausted, cheap to
    // abandon for the frontier.
    let seam_patience: usize = std::env::var("CATALOG_SEAM_PATIENCE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(800);
    // Print the seam coordinate of every accepted hop, so whether the
    // gap from the floor orders progress toward the other funnel is a
    // measurement rather than an assumption: icosahedral isomers reach
    // L1 0.56 from the floor while Marks sits at 0.4267, so a bank keyed
    // on this coordinate may be holding sideways isomers as its
    // frontier. Tracing a winning plain seed answers it.
    let seam_trace = std::env::var("CATALOG_SEAM_TRACE").is_ok_and(|value| value == "1");
    // Minima other chains have validated, held as (descriptor, energy,
    // structure) so the return screen can recognise a descent into any of
    // them. Recognition is monotone in this set and the acceptance still
    // sees a real quenched energy, so widening it can only refund cost.
    // The env var is the catalog exchange path. Recognition also fills
    // this bank from certified quenches on this chain, which is the
    // Pedersen skip analog: abort a descent into a basin already on file.
    let recognition_skip = matches!(cfg.shared_visit_policy, SharedVisitPolicy::Recognition);
    let shared_screen =
        recognition_skip || std::env::var("CATALOG_SHARED_SCREEN").is_ok_and(|value| value == "1");
    // The ensemble frontier ladder: raw doorway states ship out to the
    // coordinator, and other chains' posts fold into this chain's seam
    // bank, so the ladder holds population at every occupied stage of
    // the road instead of one chain's private progress.
    let frontier_exchange =
        std::env::var("CATALOG_FRONTIER_EXCHANGE").is_ok_and(|value| value == "1");
    let mut screen_bank: Vec<(Array1<f64>, f64, Array1<f64>)> = Vec::new();
    // The run's own environment codebook, grown from every accepted
    // recordable structure, so the trace can report each arrival's
    // unseen-environment share before the arrival is added.
    let mut trace_book = seam_trace.then(crate::catalog::PackingBook::default);
    let mut restarts = 0usize;
    let mut exchanges_refused = 0usize;
    let mut jumps = 0usize;
    let mut symmetrised = 0usize;
    let mut symmetry_gain = 0.0_f64;
    let mut continuous_symmetry_attempts = 0usize;
    // Project when the interval is due and the ledger has a new lowest
    // energy. Repeating the projection on a basin that is not the
    // incumbent spends the force budget and, on the six-seed panel,
    // finds fewer Cambridge minima.
    let mut new_incumbent = true;
    let mut continuous_symmetry_quenches = 0usize;
    let mut continuous_symmetry_gain = 0.0_f64;
    let mut quiet = 0usize;
    let mut longest_quiet = 0usize;
    let mut stall_escapes = 0usize;
    let mut stall_escape_gain = 0.0_f64;
    let mut trail_escapes = 0usize;
    // Evaluated screen state of the trial that entered the current basin,
    // kept as the exit point a stalled chain leaves through.
    let mut basin_entry: Option<Array1<f64>> = None;
    let stall_arms = stall_arms(cfg);
    let mut stall_allocator = crate::allocate::DepthAllocator::new(stall_arms.len().max(1));
    let mut last_polish_spent = 0usize;
    let mut soft_lambda = 0.0_f64;
    // The escape scale starts at the move library's own amplitude, so a run
    // without feedback and one with it begin identically.
    let mut feedback = EscapeFeedback::new(1.0, cfg.temperature.max(1e-6));
    if cfg.move_library.is_rigid_body() {
        // The shrink on every accept otherwise closes the window below
        // the barriers between rigid cages. The preset temperature is
        // the floor, so a rise of that size stays acceptable.
        feedback.set_threshold_floor(cfg.temperature.max(1e-6));
    }
    // The entropy the chain accepts against, and the weight frozen for the
    // sweep in progress. Both stay empty until the first sweep has said what
    // range of quenched energies this problem occupies: the window is read off
    // the run rather than supplied, so nothing here is specific to Lennard-Jones
    // or to a cluster size.
    let mut dos: Option<crate::dos::DensityOfStates> = None;
    let mut flat_weight: Option<crate::dos::CutWeight> = None;
    let mut flat_seen: Vec<f64> = Vec::new();
    let mut flat_since = 0usize;
    let flat_sweep = cfg.flat_sweep.max(32);
    let mut ebias: Option<crate::dos::EnergyBias> = None;
    let mut soft_cache: Option<(f64, Vec<f64>, Vec<Array1<f64>>)> = None;
    // Ring buffer of accepted quenched displacements for the learned proposal.
    let mut cov_buf: Vec<Array1<f64>> = Vec::new();
    let mut cov_next = 0usize;
    let mut soft_fired = 0usize;
    let mut soft_recomputes = 0usize;
    // The basin the *chain* stands in, not the one the last quench produced.
    // A rejected trial leaves the chain where it was, so keying "same" on the
    // previous quench counts a rejected excursion as a departure and the
    // controller escalates against the wrong history.
    let mut here: Option<usize> = None;
    // Basin identity for the controller, kept apart from the bias.
    //
    // The two mechanisms share a notion of "the same basin" and nothing else,
    // and this map is never deposited into. Reading identity off the bias
    // instead would break under replica exchange, where each rung owns its own
    // bias and the indices of one rung mean nothing in another.
    let mut identity = BasinIndex::new(
        ClusterFingerprint::of_config(cfg, &canonical_reference),
        cfg.merge_radius,
    );
    if cfg.move_library.is_rigid_body() {
        identity.set_cap(RIGID_BASIN_CAP);
        bias.cap_basins(RIGID_BASIN_CAP);
        for parked in &mut biases {
            parked.cap_basins(RIGID_BASIN_CAP);
        }
    }
    if cfg.minima_hopping {
        let initial_basin = identity.basin_of(x.view());
        feedback.register_driver_local_initial(initial_basin);
        here = Some(initial_basin);
    }
    // The history's identity of the occupied minimum, numbered by the history
    // and not by `identity`: under a shared history the numbering is the
    // population's, and this chain's local index means nothing to it.
    let mut history_here: Option<usize> = None;
    let mut occupancy_generation = 0u64;
    // Global visit count of each history minimum at this chain's last look,
    // so a later look deposits only what other chains added in between.
    let mut history_seen: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
    // Deposit cursors belong to bias owners, not to the coordinates that
    // exchange between rungs. Parked maps have the same ordering as biases.
    let mut parked_history_seen: Vec<std::collections::HashMap<usize, u64>> = (1..n_rep)
        .map(|_| std::collections::HashMap::new())
        .collect();
    let mut history_observations = 0usize;
    let mut history_new = 0usize;
    let mut shared_deposits = 0usize;
    let mut gossip_rounds = 0usize;
    let mut md_attempts = 0usize;
    let mut md_steps = 0usize;
    let mut md_failed = 0usize;
    let mut nve_unavailable = false;
    // Wide size mismatch. Random identity exchange leaves the funnel, so the
    // ordinary proposal is a softened microcanonical escape whose total
    // kinetic energy grows when the quench returns and shrinks when the
    // energy falls. Every eighth hop tries one inner-large/outer-small
    // exchange instead, and only if the unrelaxed energy stays near the
    // minimum. The preset temperature is the Metropolis temperature.
    let species_wide = large_diameter_ratio(cfg);
    // A build without the escape stops at the first hop that asks for it.
    // The labelled side probes would otherwise spend the budget first and
    // the run would end without saying so.
    let escape_refused = !cfg!(feature = "rgsaddle")
        && ((species_wide && energy_grad.is_some())
            || (cfg.minima_hopping && cfg.md_escape && grad.is_some()));
    let mut species_kinetic = (16.0 * cfg.energy_scale).max(f64::MIN_POSITIVE);
    let species_kinetic_floor = (4.0 * cfg.energy_scale).max(f64::MIN_POSITIVE);
    let species_kinetic_cap = (1.5 * n as f64 * cfg.energy_scale).max(species_kinetic);
    let mut orbits_completed = 0usize;
    let mut orbit_gain = 0.0_f64;
    let mut accepted_transitions = Vec::new();
    // All live-state replacement consumes one complete destination record.
    // Ordinary hops supply their paid history report; auxiliary adoptions
    // observe their paid certificate once without issuing another PES call.
    macro_rules! adopt_minimum {
        ($energy:expr, $state:expr, $gradient:expr, $local_minimum:expr, $evidence:expr,
         $local_feedback:expr, $action:expr, $hop:expr) => {{
            let mut destination = OccupiedMinimum {
                energy: $energy,
                coordinates: $state,
                gradient: $gradient,
                history_minimum: None,
                local_minimum: $local_minimum,
                generation: occupancy_generation
                    .checked_add(1)
                    .expect("occupancy generation"),
                pending_initial: false,
            };
            let evidence = $evidence;
            let local_feedback: Option<(Option<usize>, usize)> = $local_feedback;
            let action: Option<&str> = $action;
            let fresh = matches!(&evidence, AdoptionHistory::Fresh);
            let report = match evidence {
                AdoptionHistory::Fresh => {
                    match (history.as_deref_mut(), destination.gradient.as_ref()) {
                        (Some(h), Some(g)) => {
                            h.observe(destination.energy, destination.coordinates.view(), g.view())
                        }
                        _ => None,
                    }
                }
                AdoptionHistory::Observed(report) => report,
            };
            if fresh {
                if let Some(report) = report {
                    history_observations += 1;
                    history_new += usize::from(report.is_new);
                    if cfg.minima_hopping {
                        let reached = match destination.local_minimum {
                            Some(reached) => reached,
                            None => identity.basin_of(destination.coordinates.view()),
                        };
                        destination.local_minimum = Some(reached);
                        match cfg.shared_visit_policy {
                            SharedVisitPolicy::Tabu => {
                                feedback.remember_driver_local(reached);
                                feedback.observe_shared(
                                    history_here,
                                    report.minimum,
                                    report.is_new,
                                    report.visits,
                                );
                            }
                            SharedVisitPolicy::Recognition => {
                                feedback.observe_driver_local(here, reached);
                            }
                        }
                    } else if let Some((from, reached)) = local_feedback {
                        feedback.observe_driver_local(from, reached);
                    }
                } else if let Some((from, reached)) = local_feedback {
                    feedback.observe_driver_local(from, reached);
                }
            }
            if let (Some(h), Some(report)) = (history.as_deref_mut(), report) {
                h.mark_accepted(report.minimum);
                destination.history_minimum = Some(report.minimum);
                if fresh {
                    let seen = history_seen.entry(report.minimum).or_insert(0);
                    *seen = seen.saturating_add(1).min(report.visits.max(1));
                } else if report.visits == 0 {
                    history_seen.insert(report.minimum, 1);
                }
            }
            if let Some(action) = action {
                if std::env::var("ANNEAL_ACC_TRACE").is_ok() {
                    eprintln!(
                        "ACCTRACE hop {} e {:.6} arm {action} ord 0/0/0",
                        $hop, destination.energy
                    );
                }
                accepted_transitions.push(AcceptedTransition {
                    hop: $hop,
                    action: String::from(action),
                    from_energy: e,
                    from_state: x.clone(),
                    from_gradient: current_validation_gradient.clone(),
                    to_energy: destination.energy,
                    to_state: destination.coordinates.clone(),
                    to_gradient: destination.gradient.clone(),
                    validated: quench_is_sane(
                        cfg,
                        destination.energy,
                        destination.coordinates.view(),
                    ) && (!gradient_required || destination.gradient.is_some()),
                    adopted: true,
                });
            }
            (
                e,
                x,
                current_validation_gradient,
                history_here,
                occupancy_generation,
                here,
            ) = destination.into_live();
        }};
    }
    if let (Some(h), Some(g)) = (history.as_deref_mut(), current_validation_gradient.as_ref()) {
        // The start is part of the history even though no hop reached it,
        // exactly as the controller registers it: a later return to it must
        // classify as known.
        if let Some(report) = h.observe(e, x.view(), g.view()) {
            h.mark_accepted(report.minimum);
            history_here = Some(report.minimum);
            // Own start visit is seen; every other chain's visit to this
            // minimum is owed and paid at the first look that reaches it.
            history_seen.insert(report.minimum, 1);
        }
    }
    // Structures kept for path endpoints. Only ones far from every member are
    // added, because interpolating between two structures in one funnel lands
    // back in it, which is what archive-based escape moves holding a single
    // funnel's structures already showed.
    let mut archive: Vec<(f64, Array1<f64>)> = Vec::new();
    let mut paths_run = 0usize;
    let mut path_escapes = 0usize;
    let mut path_improvements = 0usize;
    let mut path_gain = 0.0_f64;

    // Every rung's start is held to the first one's contract before it is
    // recorded, and keeps its validation gradient for the first step the rung
    // takes from it.
    for _ in 1..n_rep {
        let s0 = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, rng);
        let (e0, x0) = relax(ledger, s0.view(), cfg.relax_steps);
        let gradient = record_quenched_answer(
            cfg,
            ledger,
            &mut grad,
            e0,
            x0.view(),
            &mut unconverged_records,
        );
        chains.push(OccupiedMinimum {
            energy: e0,
            coordinates: x0,
            gradient,
            history_minimum: None,
            local_minimum: None,
            generation: 0,
            pending_initial: true,
        });
    }
    let mut screened_out = 0usize;
    let mut returned = 0usize;
    let mut descent = ScreenDescent::default();
    let mut accepted = 0usize;
    let mut arm_counts: Vec<ArmCount> = Vec::new();
    // A rigid chart is not in the preset's reduced energy unit. Charts of
    // ten bodies or fewer start at the top of the acceptance window and let
    // the rate pull the factor down. A chart with more bodies stays at the
    // lower cap: the orientation amplitude, not the temperature, is what
    // tracks the acceptance ratio. Letting that factor climb back to 24
    // leaves the hydrogen-bond funnel.
    let mut rigid_factor: f64 = if cfg.move_library.is_rigid_body() {
        rigid_factor_cap(cfg.n_points)
    } else {
        1.0
    };
    let mut rigid_window = 0usize;
    let mut rigid_hits = 0usize;
    // Large rigid charts hold the acceptance temperature at the factor cap.
    // The orientation amplitude is what moves toward a one-half acceptance
    // ratio. Charts of ten bodies and fewer leave both untouched.
    let mut orient_scale = 1.0_f64;
    let mut orient_window = 0usize;
    let mut orient_yes = 0usize;
    let mut hops = 0usize;
    let mut proposal_overlaps = 0usize;
    let mut checkpoint_hops = 0usize;
    let mut checkpoint_quench_start = 0usize;
    let mut checkpoint_transition_start = 0usize;
    let mut next_checkpoint = checkpoint_interval;
    let mut next_reoccupy = cfg.reoccupy_interval;
    let mut best_seen = ledger.best;
    // Energy last polished by an unlike-pair quench. `NaN` until the first
    // binary incumbent, so a one-species chain never enters the sweep.
    let mut polished_best = f64::NAN;
    let mut charged_at_best = ledger.spent();
    // Minima separated by common-neighbour fractions. A stall in the
    // icosahedral packing can return to a lower minimum of another packing
    // the chain already quenched, instead of redrawing inside the same one.
    let mut packing = packing_bank::PackingBank::new();
    let packing_cutoff = 1.2 * cfg.length_scale;
    // Hops elapsed when the ledger last recorded a lower minimum.
    // A rigid stall measures 1000 hops from that hop count.
    let mut hops_at_lowest = 0usize;
    // Water restarts from a stored basin on its own clock, and then
    // stays there for a few hops before the incumbent gap applies.
    let mut last_basin_restart = 0usize;
    let mut basin_hold = 0usize;
    // Set after a rigid stall when the preset relocates one molecule.
    // Cleared when a new lowest minimum is recorded.
    let mut cage_edit = false;
    let mut csm_reseed_stamp = 0usize;
    // Calls spent when a labelled motif reseed last ran. Half a budget
    // since the incumbent last improved, so a search that is still
    // descending does not pay for it.
    let mut motif_at_spent = 0usize;
    // Set when a stall growth fails to lower the incumbent. Later growth
    // quenches wait until that energy drops. Only the leave whose growth
    // was skipped is an all-points displacement. The hops between leaves
    // keep the move library: replacing every one of them locks the chain
    // on the icosahedral floor (LJ75 about 1.21 above Marks, LJ98 about
    // 0.022 above the Leary tetrahedron).
    let mut stall_growth_hold: Option<f64> = None;
    let mut stall_growth_skips: u32 = 0;
    let hop_room = |hops: usize| cfg.max_hops.is_none_or(|cap| hops < cap);
    // Distinct minima a rigid chain has quenched. Its stall moves to one.
    let mut stall_archive = crate::methods::stall_archive::StallArchive::default();
    // One seat per oxygen cage. The energy archive fills with proton
    // isomers of the lowest cage and then has nowhere else to walk.
    let mut cage_archive = crate::methods::stall_archive::StallArchive::new(
        crate::methods::stall_archive::CAGE_SEATS,
        crate::methods::stall_archive::MEMORY,
    );
    let cage_quantum = cfg.min_separation / 50.0;
    let mut cage_walked = false;
    let mut archive_walked = false;
    let mut binary_core_offered = false;
    // Energy of the incumbent whose close-packed sheet has already been slid.
    let mut layer_slide_bits: Option<u64> = None;
    // Held until the walk finishes. Accepting a packing cut at the raw
    // start replaces the chain with a local minimum the later hops would
    // have left. The cut is kept when it still beats that incumbent.
    let mut pending_cut: Option<(f64, Array1<f64>)> = None;
    if cfg.move_library.is_rigid_body() {
        stall_archive.offer(e, x.view());
        cage_archive.offer_cage(
            e,
            x.view(),
            rigid_cage_key(x.view(), cfg.n_points, cage_quantum),
        );
    } else if cfg.improvement_rewards {
        stall_archive.offer(e, x.view());
    }

    // Descriptor and seat of the geometry the chain is standing on.
    // A rejected proposal leaves both in place, so the next hop does
    // not scan for them again. Opening any other seat drops the seat
    // count out of agreement and the scan runs.
    let mut held_bits: Vec<u64> = Vec::new();
    let mut held_cv: Option<Array1<f64>> = None;
    let mut held_seat: Option<usize> = None;
    let mut held_count: usize = 0;
    // A long-ranged well binds the second neighbour, and the Mackay
    // icosahedron is then the minimum at the complete shell. One quench,
    // charged like any other. A short-ranged run does not enter. The walk
    // keeps the start it already has: the cut is only the recorded best
    // when it is lower, and a target stops the loop before the first hop.
    if cfg.long_ranged && !ledger.target_reached() && ledger.remaining() > 0 {
        let mackay = binary_geometry::mackay_cap(n, cfg.length_scale.max(1e-6));
        if mackay.len() == x.len() {
            let (mackay_energy, mackay_state) = relax(ledger, mackay.view(), cfg.relax_steps);
            if mackay_energy.is_finite() && quench_is_sane(cfg, mackay_energy, mackay_state.view())
            {
                let before = ledger.best;
                let _gradient = record_quenched_answer(
                    cfg,
                    ledger,
                    &mut grad,
                    mackay_energy,
                    mackay_state.view(),
                    &mut unconverged_records,
                );
                if ledger.best < before - 1e-12 && improvements.len() < 512 {
                    improvements.push((0, ledger.spent(), bias.n_basins(), ledger.best));
                }
            }
        }
    }
    let stop = loop {
        // A finished checkpoint interval is delivered before the budget
        // exit. The loop otherwise returns here with nothing left to
        // charge, and the callback that the last hop completed never runs.
        let interval_finished = matches!(
            (checkpoint_interval, next_checkpoint),
            (Some(_), Some(threshold))
                if hops > checkpoint_hops && ledger.spent() >= threshold
        );
        if ledger.stopped()
            || ledger.target_reached()
            || (ledger.remaining() == 0 && !interval_finished)
        {
            break HopStop::of_closed(ledger);
        }
        // The cap is read here because every early `continue` returns here,
        // and most of them have just taken a hop.
        if !hop_room(hops) {
            break HopStop::MaxHops;
        }
        // A labelled single-species cluster that has settled into a
        // close-packed sheet, rather than a fivefold plateau, can still be
        // one lattice step inside that sheet away from the lower packing.
        // The sheet that stands clear of the others is slid once for this
        // incumbent, and a stacking fault in a smaller sheet is slipped in
        // the plane. Two species, a rigid body, and an unlabelled cluster
        // do not enter, so their call counts are unchanged.
        if cfg.species.is_some()
            && cfg.monoatomic()
            && cfg.packing_cna_applies()
            && layer_slide_bits != Some(ledger.best.to_bits())
            && let Some(incumbent) = ledger.best_state.clone()
        {
            let (frac_555, frac_421, frac_422) = crate::soap::motif_fractions(incumbent.view());
            let close_packed = !crate::soap::fivefold_plateau(incumbent.view())
                && frac_421 > frac_422
                && frac_421 > frac_555;
            if close_packed {
                layer_slide_bits = Some(ledger.best.to_bits());
                for slide in crate::twin::dominant_layer_slides(incumbent.view()) {
                    if ledger.remaining() == 0 || ledger.target_reached() {
                        break;
                    }
                    let (candidate_energy, candidate) =
                        relax(ledger, slide.view(), cfg.relax_steps);
                    if ledger.stopped() {
                        break;
                    }
                    let candidate_sane = quench_is_sane(cfg, candidate_energy, candidate.view());
                    let candidate_gradient = candidate_sane.then(|| {
                        grad.as_deref_mut().and_then(|gradient| {
                            gradient(ledger, candidate.view()).filter(|values| {
                                answer_gradient_ok(cfg, values.view(), candidate.len())
                            })
                        })
                    });
                    let candidate_gradient = candidate_gradient.flatten();
                    let recordable =
                        candidate_sane && (!gradient_required || candidate_gradient.is_some());
                    let improved = recordable && candidate_energy < ledger.best - 1e-10;
                    if recordable {
                        ledger.record(candidate_energy, candidate.view());
                    } else {
                        unconverged_records += 1;
                    }
                    let adopted = recordable && candidate_energy < e - 1e-10;
                    if recordable {
                        ledger.mark_last_recorded(adopted);
                    }
                    if adopted {
                        let from_basin = here.unwrap_or_else(|| identity.basin_of(x.view()));
                        let reached = identity.basin_of(candidate.view());
                        adopt_minimum!(
                            candidate_energy,
                            candidate.clone(),
                            candidate_gradient.clone(),
                            Some(reached),
                            AdoptionHistory::Fresh,
                            Some((Some(from_basin), reached)),
                            Some("layer-slide"),
                            hops
                        );
                        soft_cache = None;
                        basin_entry = None;
                        quiet = 0;
                        if improved && improvements.len() < 512 {
                            improvements.push((
                                hops,
                                ledger.spent(),
                                bias.n_basins(),
                                candidate_energy,
                            ));
                        }
                    }
                    if ledger.target_reached() {
                        break;
                    }
                }
                // The edge slide has its own funnel. What it does not move
                // is a fault that slips inside the plane: a patch of a
                // smaller sheet, or the sheet itself, by a Shockley partial
                // and by a full lattice step. The first slip that actually
                // lowers the incumbent is kept. The rest of that list is
                // the same fault in other directions, and the next pass of
                // the edge slide is what finishes a packing the partial has
                // opened. A numerical quiver of the quench is not that.
                if !ledger.target_reached() && !ledger.stopped() {
                    let before = ledger.best;
                    // One edge-slide follow-up per quenched energy. A later
                    // partial can land above the best slip so far and still
                    // be the packing the edge slide finishes.
                    let mut followed: Vec<f64> = Vec::new();
                    for slip in crate::twin::stacking_fault_slips(incumbent.view()) {
                        if ledger.remaining() == 0 || ledger.target_reached() {
                            break;
                        }
                        let (candidate_energy, candidate) =
                            relax(ledger, slip.view(), cfg.relax_steps);
                        if ledger.stopped() {
                            break;
                        }
                        let candidate_sane =
                            quench_is_sane(cfg, candidate_energy, candidate.view());
                        let candidate_gradient = candidate_sane.then(|| {
                            grad.as_deref_mut().and_then(|gradient| {
                                gradient(ledger, candidate.view()).filter(|values| {
                                    answer_gradient_ok(cfg, values.view(), candidate.len())
                                })
                            })
                        });
                        let candidate_gradient = candidate_gradient.flatten();
                        let recordable =
                            candidate_sane && (!gradient_required || candidate_gradient.is_some());
                        let improved = recordable && candidate_energy < before - 1e-4;
                        if !improved {
                            if !recordable {
                                unconverged_records += 1;
                            }
                            continue;
                        }
                        if followed
                            .iter()
                            .any(|prior| (prior - candidate_energy).abs() < 1e-4)
                        {
                            continue;
                        }
                        followed.push(candidate_energy);
                        if candidate_energy < ledger.best - 1e-10 {
                            ledger.record(candidate_energy, candidate.view());
                            let adopted = candidate_energy < e - 1e-10;
                            ledger.mark_last_recorded(adopted);
                            if adopted {
                                let from_basin =
                                    here.unwrap_or_else(|| identity.basin_of(x.view()));
                                let reached = identity.basin_of(candidate.view());
                                adopt_minimum!(
                                    candidate_energy,
                                    candidate.clone(),
                                    candidate_gradient.clone(),
                                    Some(reached),
                                    AdoptionHistory::Fresh,
                                    Some((Some(from_basin), reached)),
                                    Some("stacking-slip"),
                                    hops
                                );
                                soft_cache = None;
                                basin_entry = None;
                                quiet = 0;
                                if improvements.len() < 512 {
                                    improvements.push((
                                        hops,
                                        ledger.spent(),
                                        bias.n_basins(),
                                        candidate_energy,
                                    ));
                                }
                            }
                        }
                        if ledger.target_reached() {
                            break;
                        }
                        // The opened sheet can sit so close to another face
                        // that the factor-of-two test calls them a tie. The
                        // most isolated sheet is still the one to move.
                        let mut slides = crate::twin::dominant_layer_slides(candidate.view());
                        if slides.is_empty() {
                            slides = crate::twin::most_isolated_layer_slides(candidate.view());
                        }
                        for slide in slides {
                            if ledger.remaining() == 0 || ledger.target_reached() {
                                break;
                            }
                            let (slide_energy, slid) = relax(ledger, slide.view(), cfg.relax_steps);
                            if ledger.stopped() {
                                break;
                            }
                            let slide_sane = quench_is_sane(cfg, slide_energy, slid.view());
                            let slide_gradient = slide_sane.then(|| {
                                grad.as_deref_mut().and_then(|gradient| {
                                    gradient(ledger, slid.view()).filter(|values| {
                                        answer_gradient_ok(cfg, values.view(), slid.len())
                                    })
                                })
                            });
                            let slide_gradient = slide_gradient.flatten();
                            let slide_recordable =
                                slide_sane && (!gradient_required || slide_gradient.is_some());
                            let slide_improved =
                                slide_recordable && slide_energy < ledger.best - 1e-10;
                            if slide_recordable {
                                ledger.record(slide_energy, slid.view());
                            } else {
                                unconverged_records += 1;
                            }
                            let slide_adopted = slide_recordable && slide_energy < e - 1e-10;
                            if slide_recordable {
                                ledger.mark_last_recorded(slide_adopted);
                            }
                            if slide_adopted {
                                let from_basin =
                                    here.unwrap_or_else(|| identity.basin_of(x.view()));
                                let reached = identity.basin_of(slid.view());
                                adopt_minimum!(
                                    slide_energy,
                                    slid.clone(),
                                    slide_gradient.clone(),
                                    Some(reached),
                                    AdoptionHistory::Fresh,
                                    Some((Some(from_basin), reached)),
                                    Some("layer-slide"),
                                    hops
                                );
                                soft_cache = None;
                                basin_entry = None;
                                quiet = 0;
                                if slide_improved && improvements.len() < 512 {
                                    improvements.push((
                                        hops,
                                        ledger.spent(),
                                        bias.n_basins(),
                                        slide_energy,
                                    ));
                                }
                            }
                            if ledger.target_reached() {
                                break;
                            }
                        }
                    }
                }
            }
        }
        if ledger.target_reached() {
            break HopStop::of_closed(ledger);
        }
        // Labelled atomic clusters are offered close-packed cuts once:
        // the n sites of an fcc lattice nearest an atom, an octahedral
        // hole and a tetrahedral hole, interpenetrating icosahedra, a
        // Mackay icosahedron, and the packings the motif growth already
        // knows. Two species are
        // ordered by homonuclear distance
        // and by the collective swap ranking. The quenched cut is held
        // and compared with the incumbent the walk actually reaches.
        // Every evaluation is charged. Unlabelled and rigid runs never enter.
        if !binary_core_offered
            && !escape_refused
            && cfg.prior_core
            && !cfg.move_library.is_molecular()
            && !cfg.move_library.is_rigid_body()
            && let Some(species) = cfg.species.as_deref()
            && species.len() == x.len() / 3
            && species.len() >= 2
        {
            binary_core_offered = true;
            let n_atoms = species.len();
            let sigma = cfg.species_sigma.as_deref();
            let mut radius_sum = 0.0;
            let mut radius_count = 0.0;
            let mut seen_labels = Vec::new();
            for label in species {
                if !seen_labels.contains(label) {
                    seen_labels.push(*label);
                    radius_sum += binary_geometry::label_radius(species, sigma, *label);
                    radius_count += 1.0;
                }
            }
            let nearest = if radius_count > 0.0 {
                (radius_sum / radius_count).max(1e-6)
            } else {
                crate::lattice::nearest_neighbour_scale(x.view()).max(1e-6)
            };
            let mut grown = Vec::new();
            for centre in [
                binary_geometry::FccCentre::Atom,
                binary_geometry::FccCentre::Octahedral,
                binary_geometry::FccCentre::Tetrahedral,
            ] {
                grown.push(binary_geometry::fcc_cut(n_atoms, centre, nearest));
            }
            for join in [
                binary_geometry::IcosahedronJoin::Compact,
                binary_geometry::IcosahedronJoin::Linear,
            ] {
                grown.push(binary_geometry::polyicosahedral_cut(n_atoms, join, nearest));
            }
            grown.push(binary_geometry::mackay_cut(n_atoms, nearest));
            let mut side_rng = StdRng::seed_from_u64(0x0C10_5ED0);
            for source in [
                crate::lattice::Source::Named(crate::structure::Template::FaceCentredCubic),
                crate::lattice::Source::Named(crate::structure::Template::Icosahedral),
                crate::lattice::Source::Pentagonal,
            ] {
                grown.push(crate::lattice::candidate(
                    source,
                    x.view(),
                    n_atoms,
                    &mut side_rng,
                ));
            }
            let mut probe = Ledger::new(ledger.remaining().max(1));
            struct Side<'a> {
                probe: &'a mut Ledger,
                relax: Relax<'a>,
                steps: usize,
            }
            impl binary_geometry::QuenchProbe for Side<'_> {
                fn energy(&mut self, point: ArrayView1<f64>) -> Option<f64> {
                    if self.probe.remaining() == 0 {
                        return None;
                    }
                    let (value, _) = (self.relax)(self.probe, point, 0);
                    value.is_finite().then_some(value)
                }

                fn quench(&mut self, point: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
                    if self.probe.remaining() == 0 {
                        return None;
                    }
                    // The descent can record a lower physical point than the
                    // one it returns when a later pass leaves that basin.
                    let seen = self.probe.best;
                    let (value, relaxed) = (self.relax)(self.probe, point, self.steps);
                    if self.probe.best < seen
                        && self.probe.best < value
                        && let Some(state) = self.probe.best_state.clone()
                    {
                        return Some((self.probe.best, state));
                    }
                    value.is_finite().then_some((value, relaxed))
                }
            }
            let mut best_cut: Option<(f64, Array1<f64>)> = None;
            {
                let mut side = Side {
                    probe: &mut probe,
                    relax,
                    steps: cfg.relax_steps,
                };
                let mut arranged = Vec::new();
                if species_count(species) >= 2
                    && let Some(spine) = binary_geometry::axial_spine_cut(species, |label| {
                        binary_geometry::label_radius(species, sigma, label)
                    })
                    && let Some(seeded) = binary_geometry::quench_as_placed(spine, &mut side)
                {
                    arranged.push(seeded);
                }
                for candidate in grown {
                    if side.probe.remaining() == 0 {
                        break;
                    }
                    let placed =
                        binary_geometry::place_smaller_inside(candidate.view(), species, |label| {
                            binary_geometry::label_radius(species, sigma, label)
                        });
                    if let Some(seeded) = binary_geometry::arrange(placed, species, &mut side) {
                        arranged.push(seeded);
                    }
                }
                arranged.sort_by(|left, right| left.0.total_cmp(&right.0));
                // Ordering can lower a worse packing by more than it lowers
                // the best one, so every cut is polished while the probe lasts.
                for (energy, state) in arranged {
                    if side.probe.remaining() == 0 {
                        if best_cut.as_ref().is_none_or(|(held, _)| energy < *held) {
                            best_cut = Some((energy, state));
                        }
                        break;
                    }
                    if let Some(refined) =
                        binary_geometry::refine(energy, state, species, &mut side)
                        && refined.0.is_finite()
                        && best_cut.as_ref().is_none_or(|(held, _)| refined.0 < *held)
                    {
                        best_cut = Some(refined);
                    }
                }
            }
            let before = ledger.spent();
            ledger.charge_many(probe.spent());
            // The probe is not a quiet stretch of the chain. The stall
            // clock stays where the walker's own quenches left it.
            charged_at_best = charged_at_best.saturating_add(ledger.spent().saturating_sub(before));
            pending_cut = best_cut;
            if std::env::var("H2H_PRIOR_TRACE").ok().as_deref() == Some("1")
                && let Some((energy, state)) = pending_cut.as_ref()
            {
                let (before_555, before_421, before_422) = crate::soap::motif_fractions(x.view());
                let (frac_555, frac_421, frac_422) = crate::soap::motif_fractions(state.view());
                eprintln!(
                    "prior core before 555 {before_555:.3} 421 {before_421:.3} 422 {before_422:.3} after {energy:.6} 555 {frac_555:.3} 421 {frac_421:.3} 422 {frac_422:.3}"
                );
            }
        }
        // A rigid quench creeps by far less than an isomer gap. Treating
        // that creep as a new lowest restarts the stall clock forever, so
        // the chain never edits the cage.
        let new_lowest = if cfg.move_library.is_rigid_body() {
            5.0e-2
        } else {
            1e-9
        };
        if ledger.best < best_seen - new_lowest {
            best_seen = ledger.best;
            charged_at_best = ledger.spent();
            hops_at_lowest = hops;
            csm_reseed_stamp = hops;
            cage_edit = false;
            new_incumbent = true;
        }
        // Every unlike pair, quenched, kept when it is lower. One random
        // exchange every fourth hop does not finish the ordering. Absent
        // unless two labels are present, so an unlabelled chain is unchanged.
        // A molecular or rigid library keeps its atoms inside a body.
        if polishes_unlike_pairs(cfg)
            && !escape_refused
            && ledger.best.is_finite()
            && (polished_best.is_nan() || ledger.best < polished_best - 1e-8)
            && let Some(best_state) = ledger.best_state.clone()
        {
            let species = cfg.species.as_deref().expect("a binary polish has species");
            let mut current = best_state;
            let mut current_e = ledger.best;
            let (group_a, group_b) = species_groups(species);
            for _round in 0..3 {
                if ledger.remaining() == 0 {
                    break;
                }
                let mut gained = false;
                let mut tried = 0usize;
                'pairs: for &i in &group_a {
                    for &j in &group_b {
                        if ledger.remaining() == 0 || tried >= 800 {
                            break 'pairs;
                        }
                        tried += 1;
                        let mut swapped = current.clone();
                        for axis in 0..3 {
                            swapped.swap(3 * i + axis, 3 * j + axis);
                        }
                        let (eq, xq) = relax(ledger, swapped.view(), cfg.relax_steps);
                        if !quench_is_sane(cfg, eq, xq.view()) || !(eq < current_e - 1e-6) {
                            continue;
                        }
                        let gradient = record_quenched_answer(
                            cfg,
                            ledger,
                            &mut grad,
                            eq,
                            xq.view(),
                            &mut unconverged_records,
                        );
                        if gradient.is_none() && grad.is_some() {
                            continue;
                        }
                        let from_basin = identity.basin_of(current.view());
                        let reached = identity.basin_of(xq.view());
                        adopt_minimum!(
                            eq,
                            xq,
                            gradient,
                            Some(reached),
                            AdoptionHistory::Fresh,
                            Some((Some(from_basin), reached)),
                            Some("homotop"),
                            hops
                        );
                        current = x.clone();
                        current_e = e;
                        gained = true;
                    }
                }
                if !gained {
                    break;
                }
            }
            polished_best = ledger.best;
        }
        // The main walker stays put and its generator is not advanced.
        // After many patience windows with no new lowest minimum, the calls
        // that remain are spent on another cage the chain already quenched.
        // A chain that is still descending never enters, so its proposals
        // and its evaluation count stay what they were.
        let cage_patience = cfg
            .restart_patience
            .saturating_mul(RIGID_CAGE_WINDOWS)
            .max(1);
        if cfg.move_library.is_rigid_body()
            && !cage_walked
            && !kernels.is_empty()
            && ledger.spent().saturating_sub(charged_at_best) >= cage_patience
        {
            cage_walked = true;
            let shell_mark = ledger.spent();
            // Start from the walker itself. Touching the shared cage archive
            // here would change the later walk's destination.
            let key = rigid_cage_key(x.view(), cfg.n_points, cage_quantum);
            let (mut side_energy, mut side_state) = (e, x.clone());
            // A fresh generator, not a draw on the walker's.
            let mut side_rng = StdRng::seed_from_u64(
                (hops as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(key)
                    .wrapping_add(side_energy.to_bits()),
            );
            let side_scale = orient_scale;
            macro_rules! hop_side {
                ($rng:ident, $energy:ident, $state:ident) => {{
                    let mut going = false;
                    if ledger.remaining() > 0 && !ledger.stopped() {
                        let arm = $rng.random_range(0..kernels.len());
                        let mut trial = kernels[arm].propose_scaled(
                            $state.view(),
                            cfg.temperature,
                            side_scale,
                            &mut $rng,
                        );
                        recentre(&mut trial, cfg.n_points);
                        contain_trial(cfg, &mut trial, cfg.n_points);
                        let (quenched, relaxed) = relax(ledger, trial.view(), cfg.relax_steps);
                        if quenched.is_finite() && relaxed.len() == $state.len() {
                            let rise = (quenched - $energy) / cfg.temperature.max(1e-12);
                            if rise <= 0.0 || $rng.random::<f64>() < (-rise).exp() {
                                $energy = quenched;
                                $state = relaxed;
                            }
                            going = true;
                        }
                    }
                    going
                }};
            }
            // The outer half turns about the core by three orientation
            // steps. A single-molecule move into a hollow quenches back
            // into the same cage; this turn does not.
            if !ledger.stopped() && ledger.remaining() > 0 {
                let angle = match cfg.move_library {
                    MoveLibrary::RigidBody { rotate_step, .. } => RIGID_SHELL_STEPS * rotate_step,
                    _ => 0.0,
                };
                let mut turned: Option<(f64, Array1<f64>)> = None;
                if angle > 0.0 {
                    for axis in 0..3 {
                        let trial = rigid_shell_turn(side_state.view(), cfg.n_points, axis, angle);
                        let (quenched, relaxed) = relax(ledger, trial.view(), cfg.relax_steps);
                        if quenched.is_finite()
                            && rigid_centres_clear(
                                relaxed.view(),
                                cfg.n_points,
                                0.5 * cfg.min_separation,
                            )
                            && turned.as_ref().is_none_or(|(held, _)| quenched < *held)
                        {
                            turned = Some((quenched, relaxed));
                        }
                    }
                }
                if let Some((energy, state)) = turned
                    && (energy - side_energy).abs() > 1.0e-3
                {
                    side_energy = energy;
                    side_state = state;
                    let twist_end = ledger.spent().saturating_add(cage_patience);
                    while ledger.spent() < twist_end && hop_side!(side_rng, side_energy, side_state)
                    {
                    }
                }
            }
            // The shell of the walker is one leave. When it comes up short,
            // walk a stored cage and turn that shell. The stored-cage turn
            // is what reaches the contact the walker's own shell does not.
            if !ledger.stopped() && ledger.remaining() > 0 {
                cage_archive.offer_cage(e, x.view(), key);
                if let Some(best) = ledger.best_state.as_ref() {
                    cage_archive.offer_cage(
                        ledger.best,
                        best.view(),
                        rigid_cage_key(best.view(), cfg.n_points, cage_quantum),
                    );
                }
                if let Some((energy, state)) = cage_archive.escape_other_cage(key)
                    && state.len() == x.len()
                {
                    side_energy = energy;
                    side_state = state;
                }
                side_rng = StdRng::seed_from_u64(
                    (hops as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(key)
                        .wrapping_add(side_energy.to_bits()),
                );
                let stored_end = ledger.spent().saturating_add(cage_patience);
                let mut stored_hops = 0usize;
                while ledger.spent() < stored_end && hop_side!(side_rng, side_energy, side_state) {
                    if stored_hops > 0
                        && stored_hops.is_multiple_of(crate::methods::rigid_quench::STAGNATION_HOPS)
                    {
                        let next_key =
                            rigid_cage_key(side_state.view(), cfg.n_points, cage_quantum);
                        if let Some((energy, state)) = cage_archive.escape_other_cage(next_key)
                            && state.len() == side_state.len()
                        {
                            side_energy = energy;
                            side_state = state;
                        }
                    }
                    stored_hops += 1;
                }
                if !ledger.stopped() && ledger.remaining() > 0 {
                    let angle = match cfg.move_library {
                        MoveLibrary::RigidBody { rotate_step, .. } => {
                            RIGID_SHELL_STEPS * rotate_step
                        }
                        _ => 0.0,
                    };
                    let mut turned: Option<(f64, Array1<f64>)> = None;
                    if angle > 0.0 {
                        for axis in 0..3 {
                            let trial =
                                rigid_shell_turn(side_state.view(), cfg.n_points, axis, angle);
                            let (quenched, relaxed) = relax(ledger, trial.view(), cfg.relax_steps);
                            if quenched.is_finite()
                                && rigid_centres_clear(
                                    relaxed.view(),
                                    cfg.n_points,
                                    0.5 * cfg.min_separation,
                                )
                                && turned.as_ref().is_none_or(|(held, _)| quenched < *held)
                            {
                                turned = Some((quenched, relaxed));
                            }
                        }
                    }
                    if let Some((energy, state)) = turned
                        && (energy - side_energy).abs() > 1.0e-3
                    {
                        side_energy = energy;
                        side_state = state;
                        let twist_end = ledger.spent().saturating_add(cage_patience);
                        while ledger.spent() < twist_end
                            && hop_side!(side_rng, side_energy, side_state)
                        {
                        }
                    }
                }
            }
            // The calls just spent are not a new lowest of the main chain.
            // Leaving them on the quiet clock would start the later walk
            // before the main chain has actually stalled.
            charged_at_best =
                charged_at_best.saturating_add(ledger.spent().saturating_sub(shell_mark));
        }
        // The shell turn above is the early leave. A longer walk on a stored
        // cage is what reaches the minimum when the shell lands short of it,
        // and it waits out a longer quiet stretch so a chain that is still
        // descending is not spent here.
        let archive_patience = cfg
            .restart_patience
            .saturating_mul(RIGID_CAGE_WINDOWS.saturating_mul(4))
            .max(1);
        if cfg.move_library.is_rigid_body()
            && !archive_walked
            && !kernels.is_empty()
            && ledger.spent().saturating_sub(charged_at_best) >= archive_patience
        {
            archive_walked = true;
            let key = rigid_cage_key(x.view(), cfg.n_points, cage_quantum);
            cage_archive.offer_cage(e, x.view(), key);
            if let Some(best) = ledger.best_state.as_ref() {
                cage_archive.offer_cage(
                    ledger.best,
                    best.view(),
                    rigid_cage_key(best.view(), cfg.n_points, cage_quantum),
                );
            }
            let (mut side_energy, mut side_state) = match cage_archive.escape_other_cage(key) {
                Some((energy, state)) if state.len() == x.len() => (energy, state),
                _ => (e, x.clone()),
            };
            // A fresh generator, not a draw on the walker's.
            let mut side_rng = StdRng::seed_from_u64(
                (hops as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(key)
                    .wrapping_add(side_energy.to_bits()),
            );
            let side_scale = orient_scale;
            let mut side_hops = 0usize;
            let archive_end = ledger.spent().saturating_add(archive_patience);
            while ledger.spent() < archive_end && ledger.remaining() > 0 && !ledger.stopped() {
                if side_hops > 0
                    && side_hops.is_multiple_of(crate::methods::rigid_quench::STAGNATION_HOPS)
                {
                    let next_key = rigid_cage_key(side_state.view(), cfg.n_points, cage_quantum);
                    if let Some((energy, state)) = cage_archive.escape_other_cage(next_key)
                        && state.len() == side_state.len()
                    {
                        side_energy = energy;
                        side_state = state;
                    }
                }
                side_hops += 1;
                let arm = side_rng.random_range(0..kernels.len());
                let mut trial = kernels[arm].propose_scaled(
                    side_state.view(),
                    cfg.temperature,
                    side_scale,
                    &mut side_rng,
                );
                recentre(&mut trial, cfg.n_points);
                contain_trial(cfg, &mut trial, cfg.n_points);
                let (quenched, relaxed) = relax(ledger, trial.view(), cfg.relax_steps);
                if !quenched.is_finite() || relaxed.len() != side_state.len() {
                    break;
                }
                let rise = (quenched - side_energy) / cfg.temperature.max(1e-12);
                if rise <= 0.0 || side_rng.random::<f64>() < (-rise).exp() {
                    side_energy = quenched;
                    side_state = relaxed;
                }
            }
        }
        if cfg.move_library.is_rigid_body()
            && crate::methods::rigid_quench::stalled(hops, hops_at_lowest)
        {
            // The stop target is never read here, and no cluster is drawn.
            // The walker moves to the lowest minimum it has already quenched
            // that is neither its own nor a recent stall destination. The
            // ledger best is offered at every stall, so a walker away from
            // it goes back there unless it did so in the last few stalls.
            hops_at_lowest = hops;
            if let Some(state) = ledger.best_state.as_ref() {
                stall_archive.offer(ledger.best, state.view());
            }
            if let Some((energy, state)) = stall_archive.escape(e)
                && state.len() == x.len()
            {
                let gradient = grad.as_deref_mut().and_then(|g| g(ledger, state.view()));
                let ok = gradient
                    .as_ref()
                    .is_some_and(|values| answer_gradient_ok(cfg, values.view(), state.len()));
                e = energy;
                x = state;
                current_validation_gradient = gradient.filter(|_| ok);
            }
            // Translation, rotation and cycle reversal leave the oxygen cage
            // where it is. After a thousand hops with no new lowest minimum,
            // the proposals that follow move one molecule onto a missing contact.
            if matches!(
                cfg.move_library,
                MoveLibrary::RigidBody { relocate: true, .. }
            ) {
                cage_edit = true;
            }
        }
        if cfg.improvement_rewards
            && hops.saturating_sub(hops_at_lowest)
                >= crate::catalog::molecular::WATER_BASIN_STALL_HOPS
            && hops.saturating_sub(last_basin_restart)
                >= crate::catalog::molecular::WATER_BASIN_STALL_HOPS
        {
            // The proposals have stopped lowering the best. Leave from
            // another stored minimum instead of the basin the chain is in.
            last_basin_restart = hops;
            if let Some(state) = ledger.best_state.as_ref() {
                stall_archive.offer(ledger.best, state.view());
            }
            if let Some((energy, state)) = stall_archive.escape(e)
                && state.len() == x.len()
            {
                let gradient = grad.as_deref_mut().and_then(|g| g(ledger, state.view()));
                let ok = gradient
                    .as_ref()
                    .is_some_and(|values| answer_gradient_ok(cfg, values.view(), state.len()));
                e = energy;
                x = state;
                current_validation_gradient = gradient.filter(|_| ok);
                basin_hold = crate::catalog::molecular::WATER_BASIN_HOLD;
            }
        }
        // The reoccupation move runs inside the chain and charges the
        // ledger directly; its result enters the same adoption path as an
        // external proposal.
        let mut internal_action: Option<CheckpointAction> = None;
        if let Some(lattice) = cfg.reoccupy.as_ref()
            && hops > 0
            && ledger.spent() >= next_reoccupy
        {
            next_reoccupy = ledger.spent().saturating_add(cfg.reoccupy_interval.max(1));
            let rebuilt = crate::methods::lattice_search::reoccupy(lattice, ledger, x.view());
            let (energy, relaxed) = relax(ledger, rebuilt.view(), cfg.relax_steps);
            if energy.is_finite() && energy < e - 1e-9 && relaxed.len() == x.len() {
                internal_action = Some(CheckpointAction::ExternalAdopt {
                    state: relaxed,
                    action: "reoccupy".to_owned(),
                    external_calls: 0,
                });
            }
        }
        // A fresh cluster on this clock returns to the Mackay icosahedron
        // (Doye, J. Chem. Phys. 111, 8417 (1999)). When the climb is on,
        // the same clock is one directed leave, not that redraw. The hop
        // patience is thousands of hops and does not fall inside a paper
        // force budget, so this call plateau is the leave that runs.
        let call_due = hops > 0
            && ledger.spent().saturating_sub(charged_at_best) >= cfg.restart_patience.max(1);
        let mut directed_leave = false;
        if cfg.general_collective
            && !cfg.move_library.is_rigid_body()
            && call_due
            && internal_action.is_none()
            && collective.stride() <= 16
        {
            let mut best_plane: Option<(f64, Array1<f64>)> = None;
            for _ in 0..4 {
                if ledger.remaining() == 0 {
                    break;
                }
                let trial = moves::plane_rearrangement(x.view(), x.len() / 3, rng);
                let (energy, relaxed) = relax(ledger, trial.view(), cfg.relax_steps);
                if energy.is_finite()
                    && relaxed.len() == x.len()
                    && quench_is_sane(cfg, energy, relaxed.view())
                    && best_plane.as_ref().is_none_or(|(held, _)| energy < *held)
                {
                    best_plane = Some((energy, relaxed));
                }
            }
            if let Some((energy, relaxed)) = best_plane {
                let recorded = energy < ledger.best - 1e-6;
                if recorded {
                    ledger.record(energy, relaxed.view());
                }
                // A cut that loses to the chain used to replace it anyway.
                // That spends the plateau on a worse basin. The chain moves
                // only when the quenched cut beats the incumbent.
                if energy < e {
                    collective.observe(true, recorded);
                    x = relaxed;
                    e = energy;
                    here = None;
                    soft_cache = None;
                    current_validation_gradient = None;
                } else {
                    collective.observe(false, recorded);
                }
            } else {
                collective.observe(false, false);
            }
        }
        if internal_action.is_none()
            && cfg.restart_on_stall
            && !cfg.escape_on_stall
            && !cfg.move_library.is_rigid_body()
            && call_due
        {
            let fresh = random_cluster(x.len() / 3, 0.7, cfg.min_separation, rng);
            let (energy, relaxed) = relax(ledger, fresh.view(), cfg.relax_steps);
            charged_at_best = ledger.spent();
            best_seen = ledger.best;
            if energy.is_finite() && relaxed.len() == x.len() {
                internal_action = Some(CheckpointAction::ExternalAdopt {
                    state: relaxed,
                    action: "restart".to_owned(),
                    external_calls: 0,
                });
            }
        }
        if internal_action.is_none() && cfg.escape_on_stall && call_due {
            // One response per plateau. Without the reset the clock stays
            // due and the leave runs on every following hop.
            charged_at_best = ledger.spent();
            directed_leave = true;
        }
        let checkpoint_due = matches!(
            (checkpoint_interval, next_checkpoint),
            (Some(_), Some(threshold)) if hops > checkpoint_hops && ledger.spent() >= threshold
        );
        if checkpoint_due || internal_action.is_some() {
            let mut checkpoint_action = if let Some(action) = internal_action.take() {
                action
            } else {
                let interval = checkpoint_interval.expect("a due checkpoint has an interval");
                let snapshot = ChainCheckpoint {
                    current_state: x.view(),
                    current_energy: e,
                    current_gradient: current_validation_gradient.as_ref().map(|g| g.view()),
                    best_state: ledger.best_state.as_ref().map(|state| state.view()),
                    best_energy: ledger.best,
                    quench_boundaries: &ledger.quench_boundaries[checkpoint_quench_start..],
                    accepted_transitions: &accepted_transitions[checkpoint_transition_start..],
                    charged: ledger.spent(),
                    remaining: ledger.remaining(),
                    hops,
                    bias: Some(&bias),
                };
                let checkpoint_action = checkpoint(snapshot);
                checkpoint_hops = hops;
                checkpoint_quench_start = ledger.quench_boundaries.len();
                checkpoint_transition_start = accepted_transitions.len();
                next_checkpoint = ledger
                    .spent()
                    .checked_div(interval)
                    .and_then(|completed| completed.checked_add(1))
                    .and_then(|next| next.checked_mul(interval));
                checkpoint_action
            };
            while let CheckpointAction::WithBiasUpdates { updates, action } = checkpoint_action {
                for update in updates {
                    update.apply(
                        &mut bias,
                        cfg.temperature,
                        &mut shared_deposits,
                        &mut gossip_rounds,
                    );
                }
                checkpoint_action = *action;
            }
            if let CheckpointAction::Retire { reason } = checkpoint_action {
                break HopStop::Retired { reason };
            }
            let diagnostic_quench =
                matches!(&checkpoint_action, CheckpointAction::ProbeProposal { .. });
            let proposal = match checkpoint_action {
                CheckpointAction::Continue => None,
                CheckpointAction::Retire { .. } => unreachable!("retire breaks before this match"),
                CheckpointAction::WithBiasUpdates { .. } => {
                    unreachable!("bias wrappers are applied before the state action")
                }
                CheckpointAction::DepositRemote { states } => {
                    for remote in &states {
                        if remote.len() == x.len() {
                            bias.deposit_scaled_n(
                                bias.cv(remote.view()).view(),
                                cfg.temperature,
                                1.0,
                                1,
                            );
                        }
                    }
                    None
                }
                CheckpointAction::MergeBias {
                    wells,
                    weight,
                    complete,
                } => {
                    BiasUpdate::MergeWells {
                        wells,
                        weight,
                        complete,
                    }
                    .apply(
                        &mut bias,
                        cfg.temperature,
                        &mut shared_deposits,
                        &mut gossip_rounds,
                    );
                    None
                }
                CheckpointAction::DepositDescriptors { deposits, weight } => {
                    BiasUpdate::DepositDescriptors { deposits, weight }.apply(
                        &mut bias,
                        cfg.temperature,
                        &mut shared_deposits,
                        &mut gossip_rounds,
                    );
                    None
                }
                CheckpointAction::BoundaryProposal { state, action } => {
                    // The boundary start is the Leave. Re-walking from
                    // the live well is a function of the occupied
                    // tangent and quenches back onto the occupied family.
                    Some((state, action, true))
                }
                CheckpointAction::ProbeProposal { state, action } => Some((state, action, false)),
                CheckpointAction::ExternalWork { external_calls } => {
                    ledger.charge_many(external_calls);
                    None
                }
                CheckpointAction::ExternalProposal {
                    state,
                    action,
                    external_calls,
                } => {
                    if ledger.charge_many(external_calls) {
                        Some((state, action, false))
                    } else {
                        // The engine's own work drained the budget; the
                        // proposal is unaffordable and the debt stands.
                        None
                    }
                }
                CheckpointAction::ExternalAdopt {
                    state,
                    action,
                    external_calls,
                } => {
                    if ledger.charge_many(external_calls) {
                        Some((state, action, true))
                    } else {
                        None
                    }
                }
            };
            if let Some((state, action, adopt)) = proposal {
                assert_eq!(
                    state.len(),
                    x.len(),
                    "boundary proposal must match the live Cartesian state"
                );
                let from_energy = e;
                let from_state = x.clone();
                let mut from_gradient = current_validation_gradient.clone();
                let published_prize = action == "catalog_incumbent";
                let soap_push = action == "soap_push";
                if from_gradient.is_none() && (published_prize || soap_push || !adopt) {
                    from_gradient = grad.as_deref_mut().and_then(|g| {
                        g(ledger, from_state.view()).filter(|values| {
                            answer_gradient_ok(cfg, values.view(), from_state.len())
                        })
                    });
                }
                let leave_action = crate::catalog::is_occupancy_leave_action(&action);
                let references = if leave_action {
                    // The well being left is a cell of the packing being
                    // left, and the accept needs the cloud those cells chain
                    // into, not one representative of it.
                    if let Some(here) = from_state.as_slice() {
                        crate::catalog::remember_packing_reference(here);
                    }
                    crate::catalog::packing_references()
                } else {
                    Vec::new()
                };
                // OtherFamily is a catalog draw. Ridge is APE: classify
                // local environments, seed a dimer on a highlighted
                // atom, climb under an energy ceiling. An offered
                // residual is the seed when the checkpoint moved.
                let quenched = if diagnostic_quench {
                    let mut scope = ledger.diagnostic_scope();
                    relax(&mut scope, state.view(), cfg.relax_steps)
                } else if action == "catalog_ridge" {
                    let atoms = from_state.len() / 3;
                    let depth = from_energy.abs() / atoms.max(1) as f64;
                    let ceiling = from_energy + crate::known_basin::LEAVE_WALK_CLIMB * depth;
                    let offered = state
                        .iter()
                        .zip(from_state.iter())
                        .any(|(a, b)| (a - b).abs() > 1e-12);
                    let seed = if offered {
                        state.clone()
                    } else {
                        let queue = from_state
                            .as_slice()
                            .map(crate::catalog::ape_highlight_queue)
                            .unwrap_or_default();
                        let atom = crate::known_basin::take_leave_cover().unwrap_or(0);
                        let atom = if queue.iter().any(|(held, _)| *held == atom) {
                            atom
                        } else {
                            queue.first().map(|(held, _)| *held).unwrap_or(0)
                        };
                        from_state.as_slice().map_or_else(
                            || from_state.to_owned(),
                            |here| {
                                Array1::from(crate::catalog::ape_local_seed(
                                    here,
                                    atom,
                                    cfg.escape_amplitude.max(0.2),
                                    hops,
                                ))
                            },
                        )
                    };
                    if let Some(g) = grad.as_deref_mut() {
                        let act = crate::methods::activation::Activation {
                            step: cfg.escape_amplitude.max(0.15),
                            max_steps: cfg.escape_max_climb.max(48),
                            overshoot: cfg.escape_overshoot.max(1.0),
                            ..crate::methods::activation::Activation::default()
                        };
                        crate::methods::activation::activate_from_origin(
                            seed.view(),
                            from_state.view(),
                            |y| {
                                let (energy, _) = relax(ledger, y, 0);
                                if energy.is_finite() && energy <= ceiling {
                                    g(ledger, y)
                                } else {
                                    None
                                }
                            },
                            &act,
                        )
                        .and_then(|o| {
                            if !o.crossed {
                                return None;
                            }
                            let (rise, _) = relax(ledger, o.state.view(), 0);
                            if !(rise.is_finite() && rise <= ceiling) {
                                return None;
                            }
                            let quenched = relax(ledger, o.state.view(), cfg.relax_steps);
                            let left = from_state
                                .as_slice()
                                .zip(quenched.1.as_slice())
                                .is_some_and(|(origin, trial)| {
                                    crate::catalog::leaves_packing(origin, trial, &references)
                                });
                            if left {
                                Some(quenched)
                            } else if o.state.as_slice().is_some_and(|overshoot| {
                                crate::catalog::occupied_unseen_share(overshoot) > 0.0
                            }) {
                                // Quench is the occupied packing projector.
                                // Unseen local classes on the overshoot are
                                // the Leave. The destination packing is not
                                // a target.
                                Some(relax(ledger, o.state.view(), 0))
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                    .unwrap_or_else(|| relax(ledger, from_state.view(), 0))
                } else if action == "soap_push" {
                    // Off the occupied packing mean, then follow the
                    // ridge until the mode force flips. A quench of the
                    // SOAP step alone is the occupied packing projector.
                    if let Some(g) = grad.as_deref_mut() {
                        let act = crate::methods::activation::Activation {
                            step: cfg.escape_amplitude.max(0.15),
                            max_steps: cfg.escape_max_climb.max(16),
                            overshoot: cfg.escape_overshoot.max(1.0),
                            ..crate::methods::activation::Activation::default()
                        };
                        crate::methods::activation::activate(
                            state.view(),
                            |y| g(ledger, y),
                            &act,
                            1.0,
                        )
                        .and_then(|o| {
                            if !o.crossed {
                                return None;
                            }
                            let quenched = relax(ledger, o.state.view(), cfg.relax_steps);
                            let home = from_state
                                .as_slice()
                                .zip(quenched.1.as_slice())
                                .is_some_and(|(origin, trial)| {
                                    !crate::catalog::leaves_packing(
                                        origin,
                                        trial,
                                        &crate::catalog::packing_references(),
                                    )
                                });
                            if home {
                                Some(relax(ledger, o.state.view(), 0))
                            } else {
                                Some(quenched)
                            }
                        })
                    } else {
                        None
                    }
                    .unwrap_or_else(|| relax(ledger, state.view(), 0))
                } else {
                    relax(ledger, state.view(), cfg.relax_steps)
                };
                let (mut candidate_energy, mut candidate) = if leave_action {
                    crate::known_basin::with_disarmed(|| {
                        relax(ledger, quenched.1.view(), cfg.relax_steps)
                    })
                } else {
                    quenched
                };
                let left_packing = |trial: &Array1<f64>| {
                    from_state
                        .as_slice()
                        .zip(trial.as_slice())
                        .is_some_and(|(origin, trial)| {
                            crate::catalog::leaves_packing(origin, trial, &references)
                        })
                };
                // SEAKMC: several searches in the active volume, then
                // flush. The first quench is the offered start. Further
                // starts are packing leftovers restricted to the atoms
                // that carry \(\|h_i-\mu\|\), compacted then polished.
                // Unused starts are discarded. This is not Xu confidence
                // and not a campaign stop.
                if leave_action
                    && !left_packing(&candidate)
                    && let Some((energy, landed, _)) = crate::known_basin::leave_av_walk(
                        from_state.view(),
                        cfg.neighbour_cutoff,
                        crate::known_basin::LEAVE_WALK_HOPS,
                        from_energy,
                        rng,
                        |trial| relax(ledger, trial, cfg.relax_steps),
                    )
                {
                    candidate_energy = energy;
                    candidate = landed;
                }
                let novel = candidate
                    .as_slice()
                    .is_some_and(|trial| crate::catalog::occupied_unseen_share(trial) > 0.0);
                let walked_off = leave_action && (left_packing(&candidate) || novel);
                if leave_action {
                    if let Some(trial) = candidate.as_slice() {
                        let mut book = crate::catalog::PackingBook::default();
                        if let Some(origin) = from_state.as_slice() {
                            book.observe(origin);
                        }
                        if let Some(histogram) = book.histogram(trial) {
                            crate::catalog::observe_leave(
                                &histogram,
                                candidate_energy,
                                None,
                                walked_off,
                            );
                        }
                    }
                    crate::catalog::credit_action(
                        crate::catalog::ACTION_LEAVE,
                        walked_off || candidate_energy < from_energy - 1e-6,
                    );
                }
                let (proposal_energy, proposal_state) = (candidate_energy, candidate);
                #[cfg(not(feature = "featomic"))]
                let walked_off = walked_off || leave_action;
                let leave = crate::catalog::occupancy_leave_adopt(&action, walked_off);
                if leave == Some(crate::catalog::OccupancyLeaveAdopt::Refuse) {
                    hops += 1;
                    continue;
                }
                if leave_action
                    && walked_off
                    && let Some(installed) = proposal_state.as_slice()
                {
                    crate::catalog::remember_packing_reference(installed);
                }
                if leave == Some(crate::catalog::OccupancyLeaveAdopt::HoleStep) {
                    // Zero descent steps: the chain takes the hole geometry
                    // itself, so what it needs is that geometry's energy. A
                    // relaxed energy carried onto the unrelaxed coordinates
                    // biases every later Metropolis test in the chain.
                    let (hole_energy, hole_state) = relax(ledger, state.view(), 0);
                    if !quench_is_sane(cfg, hole_energy, hole_state.view()) {
                        // An exhausted budget answers with an infinite
                        // energy. Moving the chain onto that would accept
                        // every later trial, so the Leave fails instead.
                        continue;
                    }
                    hops += 1;
                    let hole_gradient = record_quenched_answer(
                        cfg,
                        ledger,
                        &mut grad,
                        hole_energy,
                        hole_state.view(),
                        &mut unconverged_records,
                    );
                    let reached = identity.basin_of(hole_state.view());
                    let from = here.unwrap_or_else(|| identity.basin_of(from_state.view()));
                    adopt_minimum!(
                        hole_energy,
                        hole_state,
                        hole_gradient,
                        Some(reached),
                        AdoptionHistory::Fresh,
                        Some((Some(from), reached)),
                        None,
                        hops
                    );
                    accepted += 1;
                    bias.deposit(x.view(), cfg.temperature);
                    accepted_transitions.push(AcceptedTransition {
                        hop: hops,
                        action,
                        from_energy,
                        to_energy: hole_energy,
                        from_state,
                        from_gradient,
                        to_state: x.clone(),
                        to_gradient: current_validation_gradient.clone(),
                        validated: current_validation_gradient.is_some(),
                        adopted: true,
                    });
                    continue;
                }
                let adopt = match leave {
                    Some(crate::catalog::OccupancyLeaveAdopt::Quench) => true,
                    Some(crate::catalog::OccupancyLeaveAdopt::HoleStep) => {
                        unreachable!("hole-step leave relocates before the quench path")
                    }
                    Some(crate::catalog::OccupancyLeaveAdopt::Refuse) => {
                        unreachable!("same-family leave is refused before the quench path")
                    }
                    None => adopt,
                };
                let proposal_sane = quench_is_sane(cfg, proposal_energy, proposal_state.view());
                let gradient_required = grad.is_some();
                let validation_gradient = if proposal_sane {
                    grad.as_deref_mut().and_then(|g| {
                        g(ledger, proposal_state.view()).filter(|values| {
                            answer_gradient_ok(cfg, values.view(), proposal_state.len())
                        })
                    })
                } else {
                    None
                };
                // A published catalog prize is already a validated
                // minimum. Mid-hop from_gradient is often missing, and
                // that gate left every hear of Marks on the ico chain.
                let recordable = proposal_sane
                    && (published_prize
                        || soap_push
                        || !gradient_required
                        || from_gradient.is_some())
                    && (published_prize
                        || soap_push
                        || !gradient_required
                        || validation_gradient.is_some());
                // Catalog hearing relocates one chain; it does not swap a
                // pair of replica states. This receiving-field Metropolis
                // filter is a global-search heuristic, not an equilibrium
                // witness: catalog selection need not be symmetric, and the
                // rule has neither proposal-density nor sender-field terms.
                // Its utility is measured by ensemble discovery and retained
                // search diversity, not by the number of copied minima.
                let exchange_accept = if published_prize && cfg.exchange_metropolis && proposal_sane
                {
                    let v_here = bias.potential(bias.cv(from_state.view()).view());
                    let v_there = bias.potential(bias.cv(proposal_state.view()).view());
                    let d = ((proposal_energy + v_there) - (from_energy + v_here))
                        / cfg.temperature.max(1e-12);
                    d <= 0.0 || rng.random::<f64>() < (-d).exp()
                } else {
                    true
                };
                if published_prize && !exchange_accept {
                    exchanges_refused += 1;
                }
                let adopt = adopt && exchange_accept;
                if recordable {
                    // A validated discovery belongs to the answer even when
                    // the live chain does not adopt the diagnostic endpoint.
                    let improved = proposal_energy < ledger.best - 1e-10;
                    ledger.record(proposal_energy, proposal_state.view());
                    if adopt {
                        hops += 1;
                        let reached = identity.basin_of(proposal_state.view());
                        let from = here.unwrap_or_else(|| identity.basin_of(from_state.view()));
                        adopt_minimum!(
                            proposal_energy,
                            proposal_state.clone(),
                            validation_gradient.clone(),
                            Some(reached),
                            AdoptionHistory::Fresh,
                            Some((Some(from), reached)),
                            None,
                            hops
                        );
                        accepted += 1;
                        bias.deposit(x.view(), cfg.temperature);
                    }
                    if improved && improvements.len() < 512 {
                        improvements.push((hops, ledger.spent(), bias.n_basins(), proposal_energy));
                    }
                    accepted_transitions.push(AcceptedTransition {
                        hop: hops,
                        action,
                        from_energy,
                        to_energy: proposal_energy,
                        from_state,
                        from_gradient,
                        to_state: proposal_state,
                        to_gradient: validation_gradient,
                        validated: true,
                        adopted: adopt,
                    });
                } else if adopt {
                    unconverged_records += 1;
                    bias.deposit(x.view(), cfg.temperature);
                } else if action == "probe" {
                    // Failed diagnostics are part of the probe denominator.
                    // They carry no validated destination and cannot move the
                    // live chain or certify that its region is exhausted.
                    accepted_transitions.push(AcceptedTransition {
                        hop: hops,
                        action,
                        from_energy,
                        to_energy: proposal_energy,
                        from_state,
                        from_gradient,
                        to_state: proposal_state,
                        to_gradient: validation_gradient,
                        validated: false,
                        adopted: false,
                    });
                }
            }
        }
        if ledger.remaining() == 0 {
            break HopStop::of_closed(ledger);
        }
        // An adoption above can take the last hop the cap allows.
        if !hop_room(hops) {
            break HopStop::MaxHops;
        }
        let continuous_symmetry_due = match cfg.continuous_symmetry {
            ContinuousSymmetry::Off => false,
            ContinuousSymmetry::Inversion { interval }
            | ContinuousSymmetry::Measure { interval, .. } => {
                let quench_index = 1usize
                    .saturating_add(hops)
                    .saturating_add(continuous_symmetry_quenches);
                quench_index.is_multiple_of(interval)
            }
        };
        if continuous_symmetry_due {
            continuous_symmetry_attempts += 1;
            let classes = continuous_symmetry_classes
                .as_deref()
                .expect("enabled continuous symmetry has equivalence classes");
            let projection = match cfg.continuous_symmetry {
                ContinuousSymmetry::Inversion { .. } => {
                    crate::continuous_symmetry::project_inversion(x.view(), classes)
                        .map(|projected| projected.coordinates)
                }
                ContinuousSymmetry::Measure { .. } => {
                    if !new_incumbent {
                        None
                    } else {
                        new_incumbent = false;
                        crate::csm::propose(x.view(), classes)
                            .map(|projected| projected.coordinates)
                    }
                }
                ContinuousSymmetry::Off => None,
            };
            let symmetry_action = match cfg.continuous_symmetry {
                ContinuousSymmetry::Measure { .. } => "continuous-symmetry-csm",
                _ => "continuous-symmetry-ci",
            };
            if let Some(projection) = projection {
                continuous_symmetry_quenches += 1;
                let from_energy = e;
                let from_state = x.clone();
                let from_gradient = current_validation_gradient.clone();
                let (candidate_energy, candidate) =
                    relax(ledger, projection.view(), cfg.relax_steps);
                let candidate_sane = quench_is_sane(cfg, candidate_energy, candidate.view());
                let gradient_required = grad.is_some();
                let candidate_gradient = candidate_sane.then(|| {
                    grad.as_deref_mut().and_then(|gradient| {
                        gradient(ledger, candidate.view()).filter(|values| {
                            answer_gradient_ok(cfg, values.view(), candidate.len())
                        })
                    })
                });
                let candidate_gradient = candidate_gradient.flatten();
                let recordable =
                    candidate_sane && (!gradient_required || candidate_gradient.is_some());
                let improved = recordable && candidate_energy < ledger.best - 1e-10;
                if recordable {
                    ledger.record(candidate_energy, candidate.view());
                } else {
                    unconverged_records += 1;
                }
                let adopted = recordable && candidate_energy < e - 1e-10;
                if recordable {
                    ledger.mark_last_recorded(adopted);
                }
                if adopted {
                    continuous_symmetry_gain += e - candidate_energy;
                    let from_basin = here.unwrap_or_else(|| identity.basin_of(x.view()));
                    let reached = identity.basin_of(candidate.view());
                    adopt_minimum!(
                        candidate_energy,
                        candidate.clone(),
                        candidate_gradient.clone(),
                        Some(reached),
                        AdoptionHistory::Fresh,
                        Some((Some(from_basin), reached)),
                        None,
                        hops
                    );
                    soft_cache = None;
                    basin_entry = None;
                    quiet = 0;
                    if improved && improvements.len() < 512 {
                        improvements.push((
                            hops,
                            ledger.spent(),
                            bias.n_basins(),
                            candidate_energy,
                        ));
                    }
                }
                if recordable {
                    accepted_transitions.push(AcceptedTransition {
                        hop: hops,
                        action: symmetry_action.to_owned(),
                        from_energy,
                        to_energy: candidate_energy,
                        from_state,
                        from_gradient,
                        to_state: candidate,
                        to_gradient: candidate_gradient,
                        validated: true,
                        adopted,
                    });
                }
            }
            if ledger.remaining() == 0 {
                break HopStop::of_closed(ledger);
            }
        }
        // A new incumbent, including one just produced by the projection
        // above, restarts the short clock. The clock is not
        // `restart_patience`: that one is thousands of force calls and does
        // not match the five-step restart in the 2013 symmetry paper.
        if ledger.best < best_seen - 1e-9 {
            best_seen = ledger.best;
            csm_reseed_stamp = hops;
        }
        // Unlabelled clusters keep the icosahedral test alone. Counting
        // common neighbours here would run on every hop of a labelled
        // metal and was the Au38 slowdown: a fivefold pause of a few hops
        // spent a reseed before the close-packed minimum.
        let mono_ih = cfg.packing_cna_applies()
            && crate::soap::ih_dominated(x.view(), crate::soap::SoapSpec::default());
        if csm_reseed_hops > 0
            && cfg.prior_reseed
            && hops.saturating_sub(csm_reseed_stamp) >= csm_reseed_hops
            && mono_ih
            && ledger.remaining() > 0
        {
            csm_reseed_stamp = hops;
            // Orbit completion stays. The fresh cluster keeps its atom order,
            // so species labels stay put. Projection is the stall gate below,
            // not every reseed.
            let classes = continuous_symmetry_classes
                .as_deref()
                .expect("a CSM reseed has equivalence classes");
            let mut fresh = random_cluster(n, 0.7, cfg.min_separation, rng);
            if let Some(closed) = crate::packing::on_new_basin(
                fresh.view(),
                n,
                cfg.symmetry_tolerance,
                cfg.symmetry_merge_radius,
                cfg.symmetrise_core_fraction,
                cfg.min_separation,
            ) {
                fresh = closed;
            }
            // The unprojected reseed is the incumbent path. Projecting every
            // reseed leaves the LJ75 icosahedral floor and also pulls LJ104
            // off seeds that already hit. Project only when the best state
            // is still icosahedral and the plain quench did not beat it.
            let best_before = ledger.best;
            let best_ih = ledger.best_state.as_ref().is_some_and(|state| {
                crate::soap::ih_dominated(state.view(), crate::soap::SoapSpec::default())
            });
            // Successful LJ104 paths leave an icosahedral best within about
            // thirty hops. The LJ75 Marks search sits on that best for the
            // rest of the budget. Project only after the longer stall, and
            // only when the plain quench did not already beat it.
            let stalled = hops.saturating_sub(hops_at_lowest) >= csm_reseed_hops.saturating_mul(8);
            let (plain_energy, plain_state) = relax(ledger, fresh.view(), cfg.relax_steps);
            let (reseed_energy, reseed_state) = if best_ih
                && stalled
                && plain_energy >= best_before - 1e-10
                && let Some(projected) = crate::csm::propose(fresh.view(), classes)
            {
                let (projected_energy, projected_state) =
                    relax(ledger, projected.coordinates.view(), cfg.relax_steps);
                if projected_energy < best_before - 1e-10 {
                    (projected_energy, projected_state)
                } else {
                    (plain_energy, plain_state)
                }
            } else {
                (plain_energy, plain_state)
            };
            let reseed_gradient = record_quenched_answer(
                cfg,
                ledger,
                &mut grad,
                reseed_energy,
                reseed_state.view(),
                &mut unconverged_records,
            );
            let reseed_ok = quench_is_sane(cfg, reseed_energy, reseed_state.view())
                && (grad.is_none() || reseed_gradient.is_some());
            if reseed_ok {
                let reached = identity.basin_of(reseed_state.view());
                adopt_minimum!(
                    reseed_energy,
                    reseed_state,
                    reseed_gradient,
                    Some(reached),
                    AdoptionHistory::Fresh,
                    Some((here, reached)),
                    Some("csm-reseed"),
                    hops
                );
                soft_cache = None;
                basin_entry = None;
                quiet = 0;
            }
            if ledger.remaining() == 0 {
                break HopStop::of_closed(ledger);
            }
        }
        // A labelled incumbent whose energy has stood for half the budget
        // is offered the motifs its common-neighbour signature lacks:
        // close packing, a pentagonal (decahedral) seed, and an
        // icosahedron. Smaller atoms take the inner sites, then one
        // unlike-pair quench pass. The chain generator is not drawn.
        // Unlabelled clusters never enter, so their evaluation counts stay.
        // A lattice cut replaces every atom, so a molecular or rigid
        // library, which keeps its atoms inside a body, never enters either.
        let motif_gap = ledger.budget() / 2;
        if motif_gap > 0
            && !cfg.move_library.is_molecular()
            && !cfg.move_library.is_rigid_body()
            && cfg.prior_motif
            && ledger.spent().saturating_sub(charged_at_best) >= motif_gap
            && ledger.spent().saturating_sub(motif_at_spent) >= motif_gap
            && ledger.remaining() > 0
            && let Some(species) = cfg.species.as_deref()
            && let Some(best_state) = ledger.best_state.clone()
        {
            motif_at_spent = ledger.spent();
            let (frac_555, frac_421, frac_422) = crate::soap::motif_fractions(best_state.view());
            let mut sources = Vec::new();
            if frac_421 < 0.20 {
                sources.push(crate::lattice::Source::Named(
                    crate::structure::Template::FaceCentredCubic,
                ));
            }
            if frac_422 < 0.15 {
                sources.push(crate::lattice::Source::Pentagonal);
            }
            if frac_555 < 0.20 {
                sources.push(crate::lattice::Source::Named(
                    crate::structure::Template::Icosahedral,
                ));
            }
            if !sources.is_empty() {
                let mut side = StdRng::seed_from_u64(
                    (hops as u64).wrapping_mul(0x9E37_79B9) ^ ledger.best.to_bits() ^ 0xFCCC_0517,
                );
                let n_atoms = best_state.len() / 3;
                let two = species_count(species) >= 2;
                let (group_a, group_b) = if two {
                    species_groups(species)
                } else {
                    (Vec::new(), Vec::new())
                };
                let mut best_alt: Option<(f64, Array1<f64>, Option<Array1<f64>>)> = None;
                // The probe is not the chain ledger. A higher seed must not
                // spend the budget the descent is still using, or the hops
                // after the miss are a different search.
                let mut probe = Ledger::new(65_536);
                for source in sources {
                    if probe.remaining() == 0 {
                        break;
                    }
                    let grown =
                        crate::lattice::candidate(source, best_state.view(), n_atoms, &mut side);
                    let grown = if two {
                        moves::place_smaller_inside(grown, species)
                    } else {
                        grown
                    };
                    let (mut eq, mut xq) = relax(&mut probe, grown.view(), cfg.relax_steps);
                    if two && quench_is_sane(cfg, eq, xq.view()) {
                        let mut gained = true;
                        let mut passes = 0;
                        while gained && passes < 2 && probe.remaining() > 0 {
                            gained = false;
                            passes += 1;
                            let mut tried = 0usize;
                            'pairs: for &i in &group_a {
                                for &j in &group_b {
                                    if probe.remaining() == 0 || tried >= 400 {
                                        break 'pairs;
                                    }
                                    tried += 1;
                                    let mut swapped = xq.clone();
                                    for axis in 0..3 {
                                        swapped.swap(3 * i + axis, 3 * j + axis);
                                    }
                                    let (alt_e, alt_x) =
                                        relax(&mut probe, swapped.view(), cfg.relax_steps);
                                    if quench_is_sane(cfg, alt_e, alt_x.view()) && alt_e < eq - 1e-6
                                    {
                                        eq = alt_e;
                                        xq = alt_x;
                                        gained = true;
                                    }
                                }
                            }
                        }
                    }
                    if !quench_is_sane(cfg, eq, xq.view()) {
                        continue;
                    }
                    let gradient = record_quenched_answer(
                        cfg,
                        &mut probe,
                        &mut grad,
                        eq,
                        xq.view(),
                        &mut unconverged_records,
                    );
                    if gradient.is_none() && grad.is_some() {
                        continue;
                    }
                    let replace = best_alt.as_ref().is_none_or(|(energy, _, _)| eq < *energy);
                    if replace {
                        best_alt = Some((eq, xq, gradient));
                    }
                }
                // A decahedral signature is also offered a stacking twin of
                // the incumbent. The twin is a reflection in one dense
                // plane, which is how fivefold packing becomes close
                // packed. Adopted only when the quench is lower, so a
                // miss leaves the chain and its later calls where they were.
                if frac_422 >= 0.15 && frac_421 < 0.20 {
                    for _ in 0..4 {
                        if probe.remaining() == 0 {
                            break;
                        }
                        let twisted =
                            crate::packing::propose_twin(best_state.view(), n_atoms, &mut side);
                        let (eq, xq) = relax(&mut probe, twisted.view(), cfg.relax_steps);
                        if !quench_is_sane(cfg, eq, xq.view()) {
                            continue;
                        }
                        let gradient = record_quenched_answer(
                            cfg,
                            &mut probe,
                            &mut grad,
                            eq,
                            xq.view(),
                            &mut unconverged_records,
                        );
                        if gradient.is_none() && grad.is_some() {
                            continue;
                        }
                        let replace = best_alt.as_ref().is_none_or(|(energy, _, _)| eq < *energy);
                        if replace {
                            best_alt = Some((eq, xq, gradient));
                        }
                    }
                }
                if let Some((eq, xq, gradient)) = best_alt
                    && eq < ledger.best - 1e-6
                {
                    if std::env::var("H2H_PRIOR_TRACE").ok().as_deref() == Some("1") {
                        let (frac_555, frac_421, frac_422) =
                            crate::soap::motif_fractions(xq.view());
                        eprintln!(
                            "prior motif {eq:.6} 555 {frac_555:.3} 421 {frac_421:.3} 422 {frac_422:.3}"
                        );
                    }
                    ledger.charge_many(probe.spent());
                    let from_basin = identity.basin_of(best_state.view());
                    let reached = identity.basin_of(xq.view());
                    adopt_minimum!(
                        eq,
                        xq,
                        gradient,
                        Some(reached),
                        AdoptionHistory::Fresh,
                        Some((Some(from_basin), reached)),
                        Some("motif-reseed"),
                        hops
                    );
                    soft_cache = None;
                    basin_entry = None;
                    quiet = 0;
                }
            }
            if ledger.remaining() == 0 {
                break HopStop::of_closed(ledger);
            }
        }
        // Gap to the incumbent, which is what the law scales the window by.
        let gap = (e - ledger.best).abs().max(1e-12);
        let held = if cfg.budget_window {
            law.temperature(gap, ledger.remaining())
        } else {
            cfg.temperature
        };
        let single = single_chain_temperature(cfg, held, dos.as_ref(), e);
        let scale = if n_rep > 1 && cfg.ladder_mode.tempers() {
            temps[rep] / cfg.temperature.max(1e-12)
        } else {
            ratios[rep]
        };
        let mut temperature = single * scale;
        if cfg.general_collective
            && let Some(spread) = general_taboo.spread()
        {
            let floor = 0.25 * cfg.temperature;
            let cap = 4.0 * cfg.temperature;
            temperature = spread.clamp(floor, cap);
        }
        if cfg.move_library.is_rigid_body() {
            // The preset temperature stays in its own energy unit. A rigid
            // water chart is not that unit, so a local factor moves the
            // acceptance rate toward one half without editing the preset.
            temperature *= rigid_factor;
        }
        // The coldest rung's temperature from the state it holds, which the
        // energy bias is built and filled at. Read here, where that rung's own
        // hop reads its temperature, so the reading is the same whichever rung
        // is hopping.
        let bias_temperature = if rep == 0 || !cfg.energy_bias || chains.is_empty() {
            single
        } else {
            standing_temperature(cfg, &law, ledger, ratios[0], dos.as_ref(), chains[0].energy)
        };

        if cfg.anneal_diversity {
            let progress = 1.0 - (ledger.remaining() as f64 / ledger.budget() as f64);
            bias.set_merge_radius(diversity.threshold(progress));
        }
        // What the chain is standing on, for an allocator that conditions on it.
        //
        // Pair-energy statistics, because that is what distinguishes the
        // situations the moves are for: a structure with one badly bound point
        // wants that point relocated, an evenly bound one does not. The same
        // quantity Wales and Doye use for the angular criterion, read as a
        // continuous context rather than a threshold.
        let context = if cfg.contextual_moves {
            let e = pair_energies_scaled(x.view(), n, cfg.length_scale, cfg.energy_scale);
            let lo = e.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = e.iter().sum::<f64>() / n.max(1) as f64;
            let spread = if lo.abs() > 1e-12 { hi / lo } else { 0.0 };
            let depth = if lo.abs() > 1e-12 { mean / lo } else { 0.0 };
            Some(Array1::from(vec![1.0, spread, depth]))
        } else {
            None
        };
        // The molecules the incumbent actually contains, re-read each hop.
        // A quench that changed the bond graph changes the groups, and the
        // move library follows the structure rather than the declaration.
        // The frozen frame this hop: the static mask, or its dynamic form,
        // everything outside the active region around the seeds.
        let hop_frozen: Option<Vec<bool>> = match (&cfg.active_region, &cfg.species) {
            (Some((seeds, shells)), Some(z)) => Some(
                active_mask(x.view(), z, seeds, *shells, cfg.bond_tolerance)
                    .into_iter()
                    .map(|a| !a)
                    .collect(),
            ),
            _ => cfg.frozen.clone(),
        };
        if let MoveLibrary::Molecular {
            reactive,
            reverse_cycles,
            ..
        } = &cfg.move_library
        {
            let fresh = match cfg.species.as_ref() {
                Some(z) => connectivity_groups_z(x.view(), z, cfg.bond_tolerance),
                None => connectivity_groups(x.view(), n, cfg.covalent_cutoff),
            };
            let movable: Vec<Vec<usize>> = match hop_frozen.as_ref() {
                Some(f) => fresh
                    .into_iter()
                    .map(|g| {
                        g.into_iter()
                            .filter(|&a| !f.get(a).copied().unwrap_or(false))
                            .collect::<Vec<usize>>()
                    })
                    .filter(|g: &Vec<usize>| !g.is_empty())
                    .collect(),
                None => fresh,
            };
            if !movable.is_empty() {
                // Same shape as the initial pool: the allocator's arm
                // statistics stay aligned across the per-hop rebuild.
                // A connectivity change can drop the cycle arm. That
                // shorter list would make the next draw index off the end.
                let rebuilt = MoveLibrary::Molecular {
                    groups: movable,
                    reactive: *reactive,
                    reverse_cycles: *reverse_cycles,
                }
                .kernels(cfg);
                if rebuilt.len() == allocator.len() {
                    kernels = rebuilt;
                }
            }
        }
        // Extra fixed-composition proposal. The wide-ratio escape and the
        // inner-outer exchange below still run on the chain generator. This
        // draw uses its own generator, and the chain moves only when the
        // quenched exchange is strictly downhill, so an uphill homotop cannot
        // throw away the minimum the escape is standing on.
        if species_wide
            && let Some(species) = cfg.species.as_deref()
            && species.len() == n
        {
            let mut side = StdRng::seed_from_u64(
                e.to_bits() ^ (hops as u64).wrapping_mul(0x9E37_79B9) ^ 0xD07E_5A4D,
            );
            if let Some(swapped) = doye_unlike_exchange(x.view(), species, &mut side) {
                let ceiling = e + 15.0 * cfg.energy_scale;
                let raw = energy_grad
                    .as_deref_mut()
                    .and_then(|evaluate| evaluate(ledger, swapped.view()))
                    .map(|(energy, _)| energy);
                if let Some(raw) = raw
                    && raw <= ceiling
                    && ledger.remaining() > 0
                {
                    let (eq, xq) = relax(ledger, swapped.view(), cfg.relax_steps);
                    if quench_is_sane(cfg, eq, xq.view()) && eq < e - 1e-8 {
                        let gradient = record_quenched_answer(
                            cfg,
                            ledger,
                            &mut grad,
                            eq,
                            xq.view(),
                            &mut unconverged_records,
                        );
                        if gradient.is_some() || !gradient_required {
                            let from_basin = here.unwrap_or_else(|| identity.basin_of(x.view()));
                            let reached = identity.basin_of(xq.view());
                            adopt_minimum!(
                                eq,
                                xq,
                                gradient,
                                Some(reached),
                                AdoptionHistory::Fresh,
                                Some((Some(from_basin), reached)),
                                Some("doye-exchange"),
                                hops
                            );
                            soft_cache = None;
                        }
                    }
                }
            }
            if ledger.remaining() == 0 || ledger.target_reached() {
                break HopStop::of_closed(ledger);
            }
        }
        // The labelled cut wins by handing the quench a compact body and an
        // ordering the energy itself prefers. The same two steps, without a
        // named body: grow onto the holes of the best-coordinated atoms, then
        // put one species on the sites an unrelaxed swap ranks lowest. The
        // family waits, and a miss widens the wait, so a search that was
        // already descending is not spent on it.
        if cfg.general_collective
            && !cfg.move_library.is_rigid_body()
            && !cfg.move_library.is_molecular()
            && hops as u64 >= u64::from(regrow_share.stride())
            && regrow_share.due(hops as u64)
            && ledger.remaining() > 32
        {
            let species = cfg.species.clone().unwrap_or_else(|| vec![0; n]);
            let mut score = |point: ArrayView1<f64>| -> Option<f64> {
                if ledger.remaining() == 0 {
                    return None;
                }
                let (value, _) = relax(ledger, point, 0);
                value.is_finite().then_some(value)
            };
            if let Some(grown) = general::regrow_by_energy(x.view(), &mut score) {
                let (group_a, group_b) = species_groups(&species);
                let ordered = if group_a.len().saturating_mul(group_b.len()) > 1
                    && group_a.len() * group_b.len() <= 320
                {
                    binary_geometry::reassign_by_site_energy(
                        grown.view(),
                        &group_a,
                        &group_b,
                        &mut score,
                    )
                    .unwrap_or(grown)
                } else {
                    let mut best = grown.clone();
                    let mut best_value = score(best.view()).unwrap_or(f64::INFINITY);
                    for candidate in general::homotop_candidates(grown.view(), &species) {
                        let Some(value) = score(candidate.view()) else {
                            break;
                        };
                        if value < best_value {
                            best_value = value;
                            best = candidate;
                        }
                    }
                    best
                };
                if ledger.remaining() > 0 {
                    let (energy, state) = relax(ledger, ordered.view(), cfg.relax_steps);
                    if energy.is_finite()
                        && state.len() == x.len()
                        && quench_is_sane(cfg, energy, state.view())
                    {
                        let recorded = energy < ledger.best - 1e-6;
                        if recorded {
                            ledger.record(energy, state.view());
                        }
                        let temperature = regrow_temperature.value(cfg.temperature);
                        let draw = rng.random::<f64>();
                        let accepted = regrow_temperature.accept(energy - e, temperature, draw);
                        if recorded || (accepted && energy < e) {
                            general_taboo.remember(energy, state.view(), &species);
                            x = state;
                            e = energy;
                            here = None;
                            soft_cache = None;
                            current_validation_gradient = None;
                        }
                        regrow_share.observe(energy < e, recorded);
                    } else {
                        regrow_share.observe(false, false);
                    }
                }
            } else {
                regrow_share.observe(false, false);
            }
            if ledger.remaining() == 0 || ledger.target_reached() {
                break HopStop::of_closed(ledger);
            }
        }
        if cfg.general_collective
            && call_due
            && let Some((energy, state)) = general_taboo.lowest()
            && energy < e - 1e-6
            && state.len() == x.len()
        {
            x = state;
            e = energy;
            here = None;
            soft_cache = None;
            current_validation_gradient = None;
        }
        let mut k = choose_arm(
            cfg,
            context.as_ref().map(|value| value.view()),
            &mut contextual,
            &depth_allocator,
            &mut allocator,
            kernels.len(),
            rng,
        );
        // One unlike-pair exchange on a fixed cadence. The depth allocator
        // drops the homotop once geometry moves pay more, and a binary
        // minimum is a chemical ordering as well as a geometry. Absent on a
        // one-species library, so a monoatomic chain draws exactly as before.
        if let Some(swap_at) = kernels.iter().position(|kernel| kernel.name() == "swap")
            && hops.is_multiple_of(4)
        {
            k = swap_at;
        }
        if cfg.general_collective
            && let Some(plane_at) = kernels.iter().position(|kernel| kernel.name() == "plane")
            && collective.due(hops as u64)
        {
            k = plane_at;
        }
        // The move scale stays the configured temperature. The law's
        // temperature is an acceptance temperature: it governs which uphill
        // steps are taken, not how far a proposal reaches. Feeding it to the
        // kernel makes a correctly small temperature shrink the proposals to
        // nothing and freeze the chain, which took LJ38 from 1 seed in 8 to 0.
        // The escape scale multiplies the move amplitude. A chain that keeps
        // returning proposes further each time until it leaves.
        // The angular move takes the step when a point is loose enough for the
        // criterion to fire, whatever the allocator picked.
        let angular = cfg.angular_moves
            && worst_bound_scaled(
                x.view(),
                n,
                angular_ratio,
                cfg.length_scale,
                cfg.energy_scale,
            )
            .is_some();
        // The soft-subspace arm competes uniformly with the library's arms.
        // The subspace is cached per incumbent and recomputed when the chain
        // moves, since it is a property of the point the chain stands on.
        // The learned-covariance arm competes uniformly like the others.
        let cov_fire = cfg.cov_perturb && !angular && rng.random_range(0..kernels.len() + 1) == 0;
        let soft_fire = cfg.soft_perturb
            && !angular
            && grad.is_some()
            && rng.random_range(0..kernels.len() + 1) == 0;
        if soft_fire {
            let stale = soft_cache
                .as_ref()
                .map(|(ce, _, _)| *ce != e)
                .unwrap_or(true);
            if stale && let Some(g) = grad.as_deref_mut() {
                let got = crate::curvature::soft_subspace(
                    x.view(),
                    |p| g(ledger, p),
                    cfg.soft_steps,
                    1e-4,
                    cfg.soft_modes,
                );
                if let Some((l, v, _ev)) = got {
                    soft_recomputes += 1;
                    soft_cache = Some((e, l, v));
                }
            }
        }
        // A rigid chart takes the preset step. The minima-hopping escape
        // scale is in a reduced energy unit and either freezes the kick
        // or throws the bodies apart.
        let escape = if cfg.minima_hopping && !cfg.move_library.is_rigid_body() {
            feedback.escape()
        } else {
            1.0
        };
        let proposal_scale = escape * orient_scale;
        // Ordinary hops: scale the library move by the escape feedback. Soft
        // mode climbs live under `escape_on_stall` below; they are a few per
        // cent of the budget when the chain has stopped improving, not the
        // default proposal.
        // The Hamiltonian proposal replaces the displacement rather than
        // competing with it as one arm among many: the comparison is between
        // a trajectory and a kick at equal charge.
        let hamiltonian = cfg.hmc.is_some() && energy_grad.is_some() && !angular;
        let mut hmc_trial: Option<Array1<f64>> = None;
        let mut species_escape_trial = false;
        // Goedecker's escape: an NVE trajectory from the occupied minimum,
        // launched with the controller's escape scale as kinetic energy and
        // stopped after two potential minima, replaces the kick. Its end
        // point enters the same screen and quench as any trial, so the
        // comparison is trajectory against kick at equal charge.
        if cfg.minima_hopping
            && cfg.md_escape
            && !angular
            && hmc_trial.is_none()
            && let Some(g) = grad.as_deref_mut()
        {
            let md_config = crate::methods::minima_hopping::MdEscapeConfig {
                dt: cfg.md_escape_dt,
                potential_minima: cfg.md_escape_minima.max(1),
                maximum_steps: cfg.md_escape_max_steps.max(1),
                softening: (cfg.md_escape_soften > 0).then_some(
                    crate::methods::minima_hopping::VelocitySofteningConfig {
                        steps: cfg.md_escape_soften,
                        ..Default::default()
                    },
                ),
                ..Default::default()
            };
            let kinetic = (cfg.md_escape_kinetic * feedback.escape()).max(f64::MIN_POSITIVE);
            let mut evaluate = |p: ArrayView1<f64>| -> Option<(f64, Array1<f64>)> {
                let (energy, _) = relax(ledger, p, 0);
                let gradient = g(ledger, p)?;
                energy.is_finite().then_some((energy, gradient))
            };
            match crate::methods::minima_hopping::nve_escape_with_frozen(
                x.view(),
                kinetic,
                &md_config,
                &mut evaluate,
                rng,
                hop_frozen.as_deref(),
            ) {
                Ok(report) if report.potential_minima >= md_config.potential_minima => {
                    md_attempts += 1;
                    md_steps += report.steps;
                    if md_attempts <= 8 && std::env::var("MD_TRACE").is_ok_and(|v| v == "1") {
                        let displacement = (&report.position - &x).mapv(|d| d * d).sum().sqrt();
                        eprintln!(
                            "md_escape attempt {md_attempts}: kinetic {kinetic:.3} steps {} minima {} \
                             endpoint energy {:.4} from {e:.4} span {:.4} conservation {:.3} displacement {displacement:.3}",
                            report.steps,
                            report.potential_minima,
                            report.energy,
                            report.potential_energy_span,
                            report.energy_conservation_ratio()
                        );
                    }
                    hmc_trial = Some(report.position);
                }
                Ok(report) => {
                    md_attempts += 1;
                    md_steps += report.steps;
                    md_failed += 1;
                }
                Err(crate::methods::minima_hopping::NveError::FeatureDisabled) => {
                    nve_unavailable = true;
                    break HopStop::NveUnavailable;
                }
                Err(_) => {
                    md_attempts += 1;
                    md_failed += 1;
                }
            }
            if ledger.remaining() == 0 {
                break HopStop::of_closed(ledger);
            }
        }
        // Every eighth hop, exchange one inner large atom with one outer
        // small atom and quench that geometry. The pair is kept only when
        // its unrelaxed energy stays within a fixed window of the minimum.
        if hmc_trial.is_none()
            && !angular
            && species_wide
            && hops.is_multiple_of(8)
            && let (Some(sigma), Some(eg)) =
                (cfg.species_sigma.as_deref(), energy_grad.as_deref_mut())
        {
            let ceiling = e + 15.0 * cfg.energy_scale;
            if let Some(swapped) = best_wide_exchange(x.view(), sigma, ceiling, |p| {
                eg(ledger, p).map(|(energy, _)| energy)
            }) {
                hmc_trial = Some(swapped);
                species_escape_trial = true;
            }
        }
        // A large diameter ratio makes an all-pairs identity exchange climb
        // out of the funnel (Sicher, Mohr, Goedecker). The ordinary proposal
        // is a softened constant-energy (NVE) escape that has to travel
        // before it counts, then a full quench and the preset Metropolis
        // test. A trajectory that stays in the well falls through to the
        // library and the next launch is hotter. An escape that is not
        // compiled stops the run instead of falling through.
        if hmc_trial.is_none()
            && !angular
            && species_wide
            && let Some(eg) = energy_grad.as_deref_mut()
        {
            let escape_floor = 0.80 * cfg.length_scale;
            let md_config = crate::methods::minima_hopping::MdEscapeConfig {
                dt: 0.02,
                potential_minima: 2,
                maximum_steps: 800,
                minimum_rise: 0.50 * cfg.energy_scale,
                max_rms: 2.5 * cfg.length_scale,
                min_well_rms: 0.0,
                softening: Some(crate::methods::minima_hopping::VelocitySofteningConfig {
                    steps: 5,
                    displacement: 0.10 * cfg.length_scale,
                    mixing: 0.25,
                }),
                ..Default::default()
            };
            let kinetic = species_kinetic;
            let mut evaluate = |p: ArrayView1<f64>| eg(ledger, p);
            match crate::methods::minima_hopping::nve_escape_with_frozen(
                x.view(),
                kinetic,
                &md_config,
                &mut evaluate,
                rng,
                hop_frozen.as_deref(),
            ) {
                Ok(report) if report.potential_minima >= 2 || report.far_rms >= escape_floor => {
                    md_attempts += 1;
                    md_steps += report.steps;
                    let position = if report.potential_minima >= 2 {
                        report.position
                    } else {
                        report.far_position
                    };
                    hmc_trial = Some(position);
                    species_escape_trial = true;
                }
                Ok(report) => {
                    md_attempts += 1;
                    md_steps += report.steps;
                    md_failed += 1;
                    species_kinetic = (species_kinetic * 1.3).min(species_kinetic_cap);
                }
                Err(crate::methods::minima_hopping::NveError::FeatureDisabled) => {
                    nve_unavailable = true;
                    break HopStop::NveUnavailable;
                }
                Err(_) => {
                    md_attempts += 1;
                    md_failed += 1;
                    species_kinetic = (species_kinetic * 1.3).min(species_kinetic_cap);
                }
            }
            if ledger.remaining() == 0 {
                break HopStop::of_closed(ledger);
            }
        }
        if hamiltonian {
            let hc = cfg.hmc.as_ref().expect("hamiltonian implies a config");
            let chain = hop.as_mut().expect("hamiltonian implies a sampler");
            let eg = energy_grad
                .as_deref_mut()
                .expect("hamiltonian implies energies");
            let mut eg_ref: crate::hmc::hop::Energy<'_> = eg;
            hmc_trial = chain
                .propose(hc, ledger, x.view(), e, &mut eg_ref, rng)
                .map(|p| p.x);
            if hmc_trial.is_none() {
                // The ledger ran out inside the trajectory, or the sampler
                // could not start one from this state.
                break if ledger.remaining() == 0 {
                    HopStop::of_closed(ledger)
                } else {
                    HopStop::NoTrajectory
                };
            }
        }
        // A high accepted minimum is a poor place to spend the next
        // proposal. Water sets `incumbent_gap` so the hop leaves from
        // the best minimum once the chain has climbed past that gap.
        if basin_hold > 0 {
            basin_hold -= 1;
        } else if cfg.incumbent_gap > 0.0
            && e > ledger.best + cfg.incumbent_gap
            && let Some(state) = ledger.best_state.clone()
            && state.len() == x.len()
        {
            x = state;
            e = ledger.best;
        }
        let mut trial = if let Some(t) = hmc_trial {
            t
        } else if cov_fire {
            let dim = x.len();
            let m = cov_buf.len();
            // Evidence weight: nothing at a cold start, most of the draw once
            // the buffer holds a history.
            let gamma = m as f64 / (m as f64 + 8.0);
            let sigma0 = 0.22 * escape;
            let mut t = x.to_owned();
            let mut gauss = || {
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            };
            let iso = (1.0 - gamma).sqrt() * sigma0;
            for i in 0..dim {
                t[i] += iso * gauss();
            }
            if m > 0 {
                let w = (gamma / m as f64).sqrt();
                for d in cov_buf.iter() {
                    let z = gauss();
                    for i in 0..dim {
                        t[i] += w * z * d[i];
                    }
                }
            }
            t
        } else if soft_fire && soft_cache.is_some() {
            let (_, lambdas, modes) = soft_cache.as_ref().expect("checked");
            soft_fired += 1;
            let mut t = x.to_owned();
            for (lam, mode) in lambdas.iter().zip(modes.iter()) {
                // Thermal amplitude in the quadratic model: c ~ N(0, T/lambda),
                // floored where a shoulder sends an eigenvalue to zero, so the
                // draw is the truncated N(0, T H^{-1}) rather than a blow-up.
                let z = {
                    let u1: f64 = rng.random::<f64>().max(1e-12);
                    let u2: f64 = rng.random::<f64>();
                    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
                };
                let c = z * (cfg.temperature / lam.max(0.05)).sqrt() * escape;
                for i in 0..t.len() {
                    t[i] += c * mode[i];
                }
            }
            t
        } else if angular {
            // The criterion decides this is the right move, so it takes the
            // step rather than competing as one arm among many.
            angular_tried += 1;
            ClusterMove::Angular {
                n_points: n,
                length_scale: cfg.length_scale,
                energy_scale: cfg.energy_scale,
            }
            .propose(x.view(), cfg.temperature, rng)
        } else {
            match (&mut constructor, &kernels[k]) {
                (Some(c), ClusterMove::Reseed { n_points, .. }) => {
                    let (cand, f) = c.propose(x.view(), *n_points, rng);
                    pending_features = Some(f);
                    cand
                }
                _ => {
                    if matches!(kernels[k], ClusterMove::EnergyLattice) {
                        energy_lattice_proposal(cfg, ledger, x.view(), rng)
                    } else {
                        kernels[k].propose_scaled(x.view(), cfg.temperature, proposal_scale, rng)
                    }
                }
            }
        };
        // Identity SOAP is not an arm. Thompson must not buy a no-op:
        // reselect a real kernel, or skip the hop.
        // A cycle reversal that finds no cycle returns the state unchanged too.
        let may_idle = |name: &str| name == "soap" || name == "rcycle" || name == "hbrewire";
        if !species_escape_trial
            && !cov_fire
            && !soft_fire
            && !angular
            && may_idle(&kernels[k].name())
            && hop_is_identity(x.view(), trial.view())
        {
            let mut swapped = false;
            for _ in 0..kernels.len() {
                let k2 = choose_arm(
                    cfg,
                    context.as_ref().map(|value| value.view()),
                    &mut contextual,
                    &depth_allocator,
                    &mut allocator,
                    kernels.len(),
                    rng,
                );
                if may_idle(&kernels[k2].name()) {
                    continue;
                }
                k = k2;
                trial = match (&mut constructor, &kernels[k]) {
                    (Some(c), ClusterMove::Reseed { n_points, .. }) => {
                        let (cand, f) = c.propose(x.view(), *n_points, rng);
                        pending_features = Some(f);
                        cand
                    }
                    _ => {
                        if matches!(kernels[k], ClusterMove::EnergyLattice) {
                            energy_lattice_proposal(cfg, ledger, x.view(), rng)
                        } else {
                            kernels[k].propose_scaled(
                                x.view(),
                                cfg.temperature,
                                proposal_scale,
                                rng,
                            )
                        }
                    }
                };
                swapped = true;
                break;
            }
            if !swapped {
                continue;
            }
        }
        if cfg.general_collective
            && collective.due(hops as u64)
            && matches!(kernels.get(k), Some(ClusterMove::Plane { .. }))
        {
            let species = cfg.species.clone().unwrap_or_else(|| vec![0; x.len() / 3]);
            let cuts = moves::plane_cuts(x.view(), x.len() / 3, rng);
            let mut best_cut: Option<(f64, Array1<f64>)> = None;
            for cut in cuts {
                if ledger.remaining() == 0 {
                    break;
                }
                let (energy, relaxed) = relax(ledger, cut.view(), cfg.relax_steps);
                if !(energy.is_finite()
                    && relaxed.len() == x.len()
                    && quench_is_sane(cfg, energy, relaxed.view()))
                {
                    continue;
                }
                if general_taboo.forbids(energy, relaxed.view(), &species) {
                    continue;
                }
                if best_cut.as_ref().is_none_or(|(held, _)| energy < *held) {
                    best_cut = Some((energy, relaxed));
                }
            }
            if let Some((mut energy, mut state)) = best_cut {
                let accepted = energy < e;
                if accepted {
                    let mut best_swap: Option<(f64, Array1<f64>)> = None;
                    for candidate in general::homotop_candidates(state.view(), &species) {
                        if ledger.remaining() == 0 {
                            break;
                        }
                        let (raw, _) = relax(ledger, candidate.view(), 0);
                        if raw.is_finite() && best_swap.as_ref().is_none_or(|(held, _)| raw < *held)
                        {
                            best_swap = Some((raw, candidate));
                        }
                    }
                    if let Some((_, candidate)) = best_swap
                        && ledger.remaining() > 0
                    {
                        let (swapped, swapped_state) =
                            relax(ledger, candidate.view(), cfg.relax_steps);
                        if swapped.is_finite()
                            && swapped < energy
                            && swapped_state.len() == x.len()
                            && quench_is_sane(cfg, swapped, swapped_state.view())
                            && !general_taboo.forbids(swapped, swapped_state.view(), &species)
                        {
                            energy = swapped;
                            state = swapped_state;
                        }
                    }
                }
                let recorded = energy < ledger.best - 1e-6;
                if recorded {
                    ledger.record(energy, state.view());
                }
                if accepted {
                    general_taboo.remember(energy, state.view(), &species);
                    x = state;
                    e = energy;
                    here = None;
                    soft_cache = None;
                    current_validation_gradient = None;
                }
                collective.observe(accepted, recorded);
            } else {
                collective.observe(false, false);
            }
            hops += 1;
            continue;
        }
        // A failed stall growth does not quench another lattice until the
        // incumbent drops. The leave that would have grown is one all-points
        // displacement. Other proposals keep the kernel they drew. The test
        // is the incumbent energy, not the atom count.
        let held_displacement = directed_leave
            && growth_held(
                stall_growth_hold,
                ledger.best,
                stall_growth_skips,
                cfg.n_points,
            )
            && cfg.packing_cna_applies();
        if held_displacement {
            let step = preset::LennardJonesPreset::ALL_POINTS_STEP * cfg.length_scale;
            trial = ClusterMove::AllPoints { step }.propose(x.view(), cfg.temperature, rng);
        }
        let editing_cage = cage_edit;
        if editing_cage {
            // One edit, then the library again. Replacing every later
            // proposal locks the chain on the relocation and it never
            // returns to the kick that was deepening the cage.
            trial = ClusterMove::RigidRelocate {
                n_molecules: cfg.n_points,
                min_separation: cfg.min_separation,
            }
            .propose(x.view(), cfg.temperature, rng);
            cage_edit = false;
            hops_at_lowest = hops;
        }
        let proposal_action = if held_displacement {
            "all".to_owned()
        } else if cov_fire {
            "covariance".to_owned()
        } else if soft_fire && soft_cache.is_some() {
            "soft_mode".to_owned()
        } else if angular {
            "angular".to_owned()
        } else {
            kernels[k].name().to_owned()
        };
        begin_arm(&mut arm_counts, &proposal_action);
        // Stage one of the staged quench: settle the moved atoms against the
        // frozen environment at fractional price, before recentring shifts
        // every coordinate and hides which atoms the move touched.
        if cfg.staged_quench
            && let Some(st) = settle.as_deref_mut()
        {
            let moved = crate::neighbors::NeighborTable::moved_between(x.view(), trial.view());
            if !moved.is_empty() && moved.len() * 8 <= n {
                trial = st(ledger, trial.view(), &moved, cfg.settle_iters);
            }
        }
        // A frozen frame is the frame: no recentring, since that drags the
        // free atoms relative to the substrate. Containment stays, taken
        // relative to the frozen atoms' bounding box: without it a desorbed
        // molecule drifts into the infinite flat region where every energy is
        // the separated limit, and the run measured exactly that, an intact
        // H2 four billion Angstrom above its slab. The free atoms are clamped
        // to within `container` of the frame.
        match hop_frozen.as_ref() {
            None => {
                recentre(&mut trial, n);
                contain_trial(cfg, &mut trial, n);
            }
            Some(f) => {
                let mut lo = [f64::INFINITY; 3];
                let mut hi = [f64::NEG_INFINITY; 3];
                for i in 0..n {
                    if f.get(i).copied().unwrap_or(false) {
                        for k in 0..3 {
                            lo[k] = lo[k].min(trial[3 * i + k]);
                            hi[k] = hi[k].max(trial[3 * i + k]);
                        }
                    }
                }
                if lo[0].is_finite() {
                    for i in 0..n {
                        if !f.get(i).copied().unwrap_or(false) {
                            for k in 0..3 {
                                trial[3 * i + k] = trial[3 * i + k]
                                    .clamp(lo[k] - cfg.container, hi[k] + cfg.container);
                            }
                        }
                    }
                }
            }
        }
        if cfg.move_library.is_rigid_body()
            && let Some(slice) = trial.as_slice_mut()
        {
            crate::potentials::fold_rotation_vectors(n, slice);
        }

        // A molecular potential refuses a pair closer than the overlap rule
        // before it will compute an energy. The hop has to hand that potential
        // a geometry the rule already accepts. Separation is geometry only.
        // The raw overlapping proposal is not an energy call.
        if cfg.species.is_some() && !structure_is_sane(trial.view(), OVERLAP_CALL_LIMIT) {
            proposal_overlaps += 1;
            let separated = separate_contacts(
                &mut trial,
                cfg.move_library.declared_groups(),
                OVERLAP_CALL_LIMIT,
                OVERLAP_CLASH_CLEARANCE,
                hop_frozen.as_deref(),
                cfg.species.as_deref(),
                cfg.bond_tolerance,
            );
            if !separated {
                hops += 1;
                bias.deposit(x.view(), temperature);
                if proposal_overlaps > ledger.budget().saturating_mul(4) {
                    break HopStop::Budget;
                }
                continue;
            }
        }
        // Screen cheaply, then carry on regardless. A screened trial does not
        // leave the chain: it goes through the acceptance test on its screened
        // energy and, whether accepted or not, a hill is deposited on wherever
        // the chain now stands.
        //
        // Skipping the rest of the iteration on a screened trial was the port
        // error behind this driver scoring 2 seeds in 8 on LJ38 where the
        // reference implementation scores 8. Around three quarters of trials
        // are screened out, so returning early deposited bias about four times
        // less often and the basins filled at a quite different rate. The
        // screen is there to avoid paying for a full relaxation, not to remove
        // the step from the chain.
        // First stage. One evaluation of the raw energy, one surrogate lookup,
        // and a Metropolis test against the surrogate target. A rejection here
        // costs the chain a hop and costs the ledger nothing beyond that
        // evaluation, which is the whole saving: the quench is never paid.
        let mut stage_one_reject = false;
        let mut pending_surrogate = None;
        if let Some(sur) = surrogate.as_mut() {
            // A zero-step relaxation is the raw energy for the price of the
            // evaluation the stage is allowed.
            let (raw_y, _) = relax(ledger, trial.view(), 0);
            // Recorded whatever the posterior can say, because the training
            // pair is produced by the quench this hop is about to pay for
            // anyway. Keeping it inside the branch that consults the posterior
            // was a deadlock: the surrogate is trained only where it is used,
            // it is used only after a warmup, and the warmup could never
            // arrive. Measured, the first stage ran 0 times in 5561 hops while
            // costing two evaluations each.
            // The gradient at the unrelaxed point, which is what says how far
            // it will fall: in a locally quadratic basin the depth goes as
            // |g|^2 / 2 lambda. One evaluation against the twenty-five a
            // quench costs.
            let gvec = match grad.as_deref_mut().and_then(|g| g(ledger, trial.view())) {
                Some(v) => v.to_owned(),
                None => Array1::zeros(trial.len()),
            };
            let gnorm = gvec.iter().fold(0.0_f64, |a, q| a + q * q).sqrt();
            pending_raw = Some((raw_y, gnorm));
            let design =
                crate::delayed::features_with_depth(trial.view(), n, raw_y, gvec.view(), x.view());
            pending_design = Some(design.clone());
            // The first stage speaks only where the posterior is sharp against
            // the temperature that scales the acceptance ratio.
            let tol = cfg.surrogate_tolerance * temperature;
            match sur.predict_features(design.view(), raw_y, tol) {
                None => sur.abstained += 1,
                Some(pred_y) => {
                    let pred_x = match surrogate_here {
                        Some(v) => v,
                        None => {
                            // The incumbent has no surrogate value yet, so give it
                            // one from its own raw energy rather than comparing
                            // against nothing.
                            let (raw_x, _) = relax(ledger, x.view(), 0);
                            let v = sur.predict(x.view(), n, raw_x).unwrap_or(e);
                            surrogate_here = Some(v);
                            v
                        }
                    };
                    sur.stage_one += 1;
                    let a1 = sur.stage_one_probability(pred_x, pred_y, temperature);
                    if rng.random::<f64>() >= a1 {
                        sur.stage_one_rejected += 1;
                        stage_one_reject = true;
                    }
                    pending_surrogate = Some((pred_y, raw_y));
                }
            }
        }
        let best_before_iteration = ledger.best;
        let mut moved_basin = false;
        if stage_one_reject {
            // The chain stays. A rejected proposal still deposits on where the
            // chain stands, exactly as a rejection through the ordinary
            // acceptance test does, so the bias accumulates at the same rate.
            hops += 1;
            bias.deposit(x.view(), temperature);
            continue;
        }
        // A rigid partial quench is not a basin. The return test and the
        // Metropolis step both need the minimum, so the rigid trial is
        // relaxed once, to the full iteration cap, and that result is the
        // trial. The Cartesian screen stays a short prefix of a second pass.
        let rigid_quench = cfg.move_library.is_rigid_body();
        // The species escape is already a point past a barrier. A 25-step
        // screen of that point falls back into the well it left, and the
        // return test then skips the quench. Relax it once, in full.
        let quenched_in_place = rigid_quench || species_escape_trial || cfg.quench_once;
        let (e_screen, x_screen) = relax(
            ledger,
            trial.view(),
            if quenched_in_place {
                cfg.relax_steps
            } else {
                cfg.screen_steps
            },
        );
        if ledger.stopped() {
            break HopStop::of_closed(ledger);
        }
        let entry_snapshot = if cfg.trail_on_stall {
            Some(x_screen.clone())
        } else {
            None
        };
        // Two reasons to stop before the full relaxation. The trial is going
        // nowhere useful by energy, or it is going back where the chain already
        // is, which the energy screen cannot see because a returning trial
        // carries the incumbent's energy.
        // SOAP leftover on a packing shell is a local reconstruction.
        // Skipping the return and energy screens for every non-identity
        // packing SOAP paid a full quench that polished back onto the
        // icosahedral shelf and cut LJ75 Marks from 10/48 to 4/48.
        // The ordinary screens stay on. Molecule and slab leftover
        // never hit this packing path.
        if frontier_exchange && let Some(seam) = seam.as_mut() {
            for (gap, energy, coordinates) in crate::catalog::drain_frontier_arrivals() {
                seam.offer(gap, energy, &coordinates);
            }
        }
        if shared_screen {
            for (energy, coordinates) in crate::catalog::take_known_minima() {
                if coordinates.len() == x.len() {
                    let held = Array1::from(coordinates);
                    screen_bank.push((bias.cv(held.view()), energy, held));
                }
            }
        }
        // A trial descending into a basin any chain has already validated
        // is recognised here the same way a return to the incumbent is,
        // and the stored minimum stands in for the rest of the descent:
        // the Metropolis test still sees a real quenched energy for that
        // basin, so the per-attempt rate is untouched and the remaining
        // force calls are refunded.
        let ds = bias.cv(x_screen.view());
        let dc = bias.cv(x.view());
        // SOAP projection learns its basis while descriptors arrive. Every
        // other key is a pure function, so its incumbent descriptor can be
        // carried through the hop instead of sorted again.
        let reusable_descriptor = effective_keying(cfg) != Keying::Soap;
        let known_stand_in: Option<(f64, Array1<f64>)> = if shared_screen && !screen_bank.is_empty()
        {
            let radius2 = cfg.merge_radius * cfg.merge_radius;
            screen_bank
                .iter()
                .filter_map(|(cv, energy, coords)| {
                    let mut d2 = 0.0;
                    for (p, q) in ds.iter().zip(cv.iter()) {
                        d2 += (p - q) * (p - q);
                        if d2 >= radius2 {
                            return None;
                        }
                    }
                    Some((d2, energy, coords))
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_, energy, coords)| (*energy, coords.clone()))
        } else {
            None
        };
        let drift: f64 = ds
            .iter()
            .zip(dc.iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        let returning = cfg.return_screen && drift < cfg.merge_radius;
        if returning {
            returned += 1;
        }
        // Screen under every mode, including minima hopping. Turning the screen
        // off under MH paid full quenches for every scatter: ~228 force per hop
        // against ~37 with the screen. A screened trial still goes through
        // Metropolis on the plain path (deposit rate and chain motion); under
        // MH it is a rejection and a same-basin observation, because the
        // controller needs a quenched minimum to classify.
        // Features of the partial relaxation: an intercept, the screened
        // energy, how far the partial relaxation moved in descriptor space, and
        // their product. All of it is already computed above by the return
        // screen, so consulting the posterior costs no evaluations.
        let feats = Array1::from(vec![1.0, e_screen, drift, e_screen * drift]);
        let screened_this = if cfg.bayes_screen {
            // Refusing is what "screened" means here: the posterior says
            // finishing this relaxation is unlikely to improve on the
            // incumbent, so the evaluations go elsewhere.
            //
            // Never refused while the partial energy already beats the
            // incumbent, whatever the posterior says. A screened structure has
            // not been relaxed, so recording one as the run's best reports a
            // point that is not a minimum. The margin screen cannot do this,
            // because it refuses only above `best + margin`; a posterior has no
            // such guarantee, and a run of this arm returned a structure whose
            // gradient was 0.31 where every other arm returns about 1e-6.
            e_screen >= ledger.best
                && !screen.decide(feats.view(), ledger.best, rng.random::<f64>())
        } else if cfg.calibrated_screen {
            // A cage-changing proposal starts far above the incumbent. A
            // short screen then refuses it, and the centres never move.
            // That move is quenched in full.
            let relocating = editing_cage || kernels.get(k).is_some_and(rigid_cage_kernel);
            !relocating
                && e_screen - descent.allowance() > ledger.best + cfg.screen_margin
                && !descent.audit()
        } else {
            // A rigid partial quench sits far above the incumbent by an
            // absolute margin copied from a reduced-unit cluster. Skipping
            // that test still honours the return screen, and the full quench
            // runs unless the trial is returning.
            !cfg.move_library.is_rigid_body() && e_screen > ledger.best + cfg.screen_margin
        };
        let (e_new, x_new) = if quenched_in_place && known_stand_in.is_none() {
            let mut out = (e_screen, x_screen);
            // The rigid path stops at its cap. A water one-pass quench
            // still continues while a point below the incumbent is
            // descending, which is what the split path's second stage
            // does through `finish_improving`.
            if cfg.quench_once && cfg.finish_improving {
                out = finish_improving(ledger, relax, out, cfg.relax_steps);
            }
            out
        } else if let Some((known_e, known_x)) = known_stand_in {
            // The refund: the basin is on file, its minimum stands in for
            // the descent, and the force calls the relaxation would have
            // spent go to the next attempt instead.
            (known_e, known_x)
        } else if returning
            && cfg.return_polish > 0
            && (cfg.return_polish_after == 0 || ledger.spent() >= cfg.return_polish_after)
        {
            // `return_polish_after == 0` finishes every returning trial.
            // A positive threshold keeps the early hop as skip-return.
            relax(ledger, x_screen.view(), cfg.return_polish)
        } else if screened_this || returning {
            if screened_this && !returning {
                screened_out += 1;
            }
            (e_screen, x_screen)
        } else {
            let mut out = relax(ledger, x_screen.view(), cfg.relax_steps);
            if cfg.finish_improving {
                out = finish_improving(ledger, relax, out, cfg.relax_steps);
            }
            out = finish_rigid(cfg, ledger, relax, &mut grad, out);
            if cfg.bayes_screen {
                // The answer to the question the posterior was asked.
                screen.observe(feats.view(), out.0);
            }
            descent.observe(e_screen - out.0);
            out
        };
        if ledger.stopped() {
            break HopStop::of_closed(ledger);
        }
        if !quench_is_sane(cfg, e_new, x_new.view()) {
            unconverged_records += 1;
            hops += 1;
            bias.deposit(x.view(), temperature);
            continue;
        }
        // A screened or returning structure has only a partial relaxation.
        // It can train screens and count as a rejected attempt, but the live
        // chain remains on the quenched landscape under every acceptance law.
        // A rigid trial was relaxed to the iteration cap before this test.
        // The Cartesian screen still hands back a partial descent, and that
        // partial descent is not a recordable minimum.
        let unquenched = !quenched_in_place && (screened_this || returning);
        if species_escape_trial {
            let same = (e_new - e).abs() <= 5.0e-2;
            if same {
                species_kinetic = (species_kinetic * 1.3).min(species_kinetic_cap);
            } else if e_new < e - 0.05 {
                species_kinetic = (species_kinetic / 1.15).max(species_kinetic_floor);
            }
        }
        // Every quench trains the surrogate, including the ones taken before
        // it had an opinion. This is where its training data comes from.
        if let (Some(sur), Some((raw_y, gnorm))) = (surrogate.as_mut(), pending_raw.take()) {
            if let Some(design) = pending_design.take() {
                sur.observe_features(design.view(), raw_y, e_new);
            } else {
                sur.observe_full(trial.view(), n, raw_y, gnorm, e_new);
            }
            // And the relaxed end of the same quench, whose depth is zero.
            //
            // Without it the model sees only unrelaxed structures and is asked
            // about a relaxed one at every first stage, because the chain state
            // is a quenched minimum: its true value is its own energy and its
            // depth is zero by definition. Trained on one regime and consulted
            // in another it extrapolated, so every acceptance ratio was formed
            // against a meaningless incumbent estimate. Measured, the second
            // stage rejected 98 per cent of what the first passed.
            //
            // The pair is free: it is the other end of a quench already paid
            // for, and it is what teaches the model that a vanishing gradient
            // means a vanishing depth.
            sur.observe_full(x_new.view(), n, e_new, 0.0, e_new);
        }
        let improved = e_new < ledger.best - 1e-10;
        // A structure the ledger records can be reported as the run's answer,
        // so it has to be a minimum. Two paths reach here with one that is not.
        //
        // A trial leaving through the screen carries a 25-step partial quench.
        // The posterior screen guards against recording one, by refusing to
        // screen anything already below the incumbent; the return screen has no
        // such guard and is on in every arm measured. Instrumenting the record
        // sites on a failing seed found three successive new bests tagged as
        // returning, at maximum absolute gradients of 1.23, 0.87 and 0.87.
        //
        // The full relaxation is not safe either. It stops on its iteration cap
        // or on a line-search failure, and the driver takes the result
        // regardless: one recorded best came back at 2.02e-1, where re-relaxing
        // the same structure for 4000 steps reaches 1.7e-6 at an energy 1.1e-3
        // lower.
        //
        // Both are guarded here rather than at either branch, so a path added
        // later inherits the guarantee. Anything not converged still moves the
        // chain and still deposits bias; it is only barred from being recorded
        // as an answer, which is the one thing it cannot be.
        let hit_target = ledger.below_target(e_new);
        let gradient_required = grad.is_some();
        let quenched_gradient = if unquenched {
            None
        } else {
            grad.as_deref_mut().and_then(|g| g(ledger, x_new.view()))
        };
        // A chain already holding a free atom, as a random start can, still
        // moves, so it is not held where every hop is refused. Without a
        // gradient the current state is judged by its geometry alone.
        if let Some(g) = quenched_gradient.as_ref()
            && sheds_free_atom(cfg, x_new.view(), Some(g.view()))
            && !sheds_free_atom(cfg, x.view(), None)
        {
            hops += 1;
            continue;
        }
        let validation_gradient =
            quenched_gradient.filter(|values| answer_gradient_ok(cfg, values.view(), x_new.len()));
        let recordable =
            hit_target || (!unquenched && (!gradient_required || validation_gradient.is_some()));
        // Every certified quench is reported, adopted or not: the history's
        // membership policy decides what counts, not the chain.
        let history_report = match (history.as_deref_mut(), validation_gradient.as_ref()) {
            (Some(h), Some(g)) if recordable => h.observe(e_new, x_new.view(), g.view()),
            _ => None,
        };
        // The reached minimum's bias key, taken before an acceptance moves
        // the coordinates into the chain.
        let history_cv = history_report.map(|_| bias.cv(x_new.view()));
        if let Some(report) = history_report {
            history_observations += 1;
            if report.is_new {
                history_new += 1;
            }
        }
        // The floor the fivefold channel reads is the incumbent before this
        // record. Gap and energy are not that comparison.
        let channel_floor_energy = ledger.best;
        let channel_floor_state = seam.as_ref().and_then(|_| ledger.best_state.clone());
        if recordable {
            ledger.record(e_new, x_new.view());
            if ledger.target_reached() {
                hops += 1;
                if improved && improvements.len() < 512 {
                    improvements.push((hops, ledger.spent(), bias.n_basins(), e_new));
                }
                break HopStop::Target;
            }
            // Recognition refunds the rest of a later descent into this
            // basin. HistoryHook still returns identity only; the bank
            // holds coordinates the loop already paid to certify.
            if recognition_skip && screen_bank.len() < 512 {
                screen_bank.push((bias.cv(x_new.view()), e_new, x_new.clone()));
                if let Some(coords) = x_new.as_slice() {
                    crate::catalog::offer_known_minimum(e_new, coords);
                }
            }
        } else {
            unconverged_records += 1;
        }
        hops += 1;
        // Only where a full relaxation actually ran: a screened or returning
        // trial carries the energy of a partial descent.
        if cfg.trace_quenched && !unquenched && !screened_this && !returning {
            quenched.push((ledger.spent(), bias.n_basins(), e_new, recordable));
        }
        if recordable && improved && improvements.len() < 512 {
            improvements.push((hops, ledger.spent(), bias.n_basins(), e_new));
            // The anatomy of the draw that produced a new best, for the
            // question no mechanism arm has answered: what does a crossing
            // perturbation look like? Written to stderr under an environment
            // switch so a campaign can collect it without an API change.
            if std::env::var("ANNEAL_IMP_TRACE").is_ok() {
                let pnorm: f64 = trial
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                let dnorm: f64 = x_new
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                // Participation of the realised displacement over atoms: one
                // when every atom moves equally, 1/n when one atom carries it.
                let n_at = x.len() / 3;
                let mut tot = 0.0_f64;
                let mut p2 = 0.0_f64;
                for i in 0..n_at {
                    let a = (0..3)
                        .map(|c| {
                            let d = x_new[3 * i + c] - x[3 * i + c];
                            d * d
                        })
                        .sum::<f64>();
                    tot += a;
                    p2 += a * a;
                }
                let part = if tot > 0.0 {
                    tot * tot / (n_at as f64 * p2)
                } else {
                    0.0
                };
                let arm = if soft_fire {
                    "soft".to_string()
                } else if cov_fire {
                    "cov".to_string()
                } else if angular {
                    "angular".to_string()
                } else {
                    kernels[k].name()
                };
                // Local order of the structure this improvement produced and
                // of the incumbent it came from, so the trace shows where in
                // the improvement chain a funnel entry actually happens rather
                // than only the finishing step. Classification only: nothing
                // downstream steers by it.
                let count = |v: ArrayView1<f64>| {
                    let m = crate::structure::ptm(v, n, 0.12);
                    let mut fcc = 0usize;
                    let mut hcp = 0usize;
                    let mut ico = 0usize;
                    for t in &m {
                        match t.template {
                            crate::structure::Template::FaceCentredCubic => fcc += 1,
                            crate::structure::Template::HexagonalClosePacked => hcp += 1,
                            crate::structure::Template::Icosahedral => ico += 1,
                            _ => {}
                        }
                    }
                    (fcc, hcp, ico)
                };
                let (f_new, h_new, i_new) = count(x_new.view());
                let (f_old, h_old, i_old) = count(x.view());
                eprintln!(
                    "IMPTRACE hop {hops} e {e_new:.6} arm {arm} pnorm {pnorm:.4} dnorm {dnorm:.4} part {part:.4} new {f_new}/{h_new}/{i_new} old {f_old}/{h_old}/{i_old}"
                );
            }
        }
        // Kept before the acceptance branch, which may move `x_new` into the
        // chain. The archive wants the structure this hop produced whether or
        // not the chain took it: a rejected structure in a different funnel is
        // exactly the path endpoint that is otherwise never seen again.
        let produced = if cfg.path_on_stall {
            Some((e_new, x_new.clone()))
        } else {
            None
        };

        // Where the chain stands before the acceptance test, so an accepted
        // hop can be recorded as an edge from here to there. Taken only when
        // the tracker is on, since it costs a descriptor and a lookup.
        let here_before = if cfg.track_funnels || superbasin.is_some() {
            Some(*here.get_or_insert_with(|| identity.basin_of(x.view())))
        } else {
            None
        };
        // Paving at the packing grain, beside the paving at the cell grain.
        //
        // BasinBias opens an account per cluster fingerprint, so every
        // icosahedral isomer gets its own, and the icosahedral shelf of
        // LJ75 holds hundreds of minima that the book splits into about
        // thirty-two families. A quarter of an eps a visit, fragmented
        // that far, never approaches the 8.69 eps between the funnels.
        // The packing pile holds one account for the whole shelf, so
        // arriving anywhere on it is expensive to the acceptance test.
        //
        // No walk out of the icosahedral minimum leaves it, measured; what
        // reaches Marks is plain hopping, over a sequence of ordinary
        // moves. Tilting that sequence is the lever the measurements leave
        // standing.
        let (pave_old, pave_new) = if let Some(pave) = pave.as_mut() {
            let from = *here.get_or_insert_with(|| identity.basin_of(x.view()));
            let reached = identity.basin_of(x_new.view());
            let old = x
                .as_slice()
                .map(|slice| {
                    pave.community(from as u64, slice);
                    pave.potential(from as u64)
                })
                .unwrap_or(0.0);
            let new = x_new
                .as_slice()
                .map(|slice| {
                    pave.community(reached as u64, slice);
                    pave.potential(reached as u64)
                })
                .unwrap_or(0.0);
            (old, new)
        } else {
            (0.0, 0.0)
        };
        // Uphill in the bare energy, and not a stride step. The basin
        // scan is what those proposals were paying for. A downhill
        // proposal is still described, because a filled destination can
        // turn it back, and one step in `basin_stride` is exact either way.
        let stride = cfg.basin_stride.max(1);
        let skip_basin = stride > 1 && !hops.is_multiple_of(stride) && e_new > e;
        let held_live = held_cv.as_ref().is_some_and(|_| {
            held_count == bias.n_basins()
                && held_bits.len() == x.len()
                && held_bits
                    .iter()
                    .zip(x.iter())
                    .all(|(bits, value)| *bits == value.to_bits())
        });
        let (s_old, v_old_held, seat_old) = if skip_basin {
            (Array1::zeros(0), 0.0, None)
        } else if held_live {
            let cv = held_cv.clone().expect("held");
            let height = held_seat.map(|i| bias.height_at(i)).unwrap_or(0.0);
            (cv, height, held_seat)
        } else if reusable_descriptor {
            let cv = dc;
            let (height, seat) = bias.locate(cv.view());
            (cv, height, seat)
        } else {
            let cv = bias.cv(x.view());
            let (height, seat) = bias.locate(cv.view());
            (cv, height, seat)
        };
        let (s_new, v_new_held, seat_new) = if skip_basin {
            (Array1::zeros(0), 0.0, None)
        } else {
            let cv = bias.cv(x_new.view());
            let (height, seat) = bias.locate(cv.view());
            (cv, height, seat)
        };
        // A draw from the region's density of states, recorded before the
        // acceptance rule reweights it towards the low tail.
        if let (Some(t), false) = (trace.as_mut(), unquenched) {
            t.push(e_new);
        }
        // The transition this proposal represents for the unbiased chain,
        // weighted by the acceptance it would have had with no deposits: the
        // move kernel never sees the deposits, so only the acceptance is
        // bias-dependent, and this replaces it rather than corrects it.
        if let (Some(sb), Some(from)) = (superbasin.as_mut(), here_before) {
            if unquenched {
                sb.observe(from, Some(from), 0.0);
            } else {
                let a = if e_new <= e {
                    1.0
                } else {
                    (-(e_new - e) / temperature.max(1e-12)).exp()
                };
                sb.observe(from, identity.lookup(x_new.view()), a);
            }
        }
        // Biased rise. The bias is part of the landscape the chain walks; a
        // threshold or Metropolis on raw energy alone ignores the deposits and
        // re-enters filled basins freely. Measured: MH accepting on raw delta
        // solved 1 of 4 LJ38 seeds at 400k where the biased Metropolis path
        // solved the same seed in ~10k hops.
        //
        // When track_funnels is on, the spectral term is well-tempered MetaD
        // on the Fiedler coordinate of the hop graph (SpectralBias): it fills
        // the *funnel* the chain is stuck in, not only the current basin.
        let spectral_old = if skip_basin {
            0.0
        } else {
            spectral
                .as_ref()
                .map(|sp| sp.potential(sp.cv(x.view()).view()))
                .unwrap_or(0.0)
        };
        let spectral_new = if skip_basin {
            0.0
        } else {
            spectral
                .as_ref()
                .map(|sp| sp.potential(sp.cv(x_new.view()).view()))
                .unwrap_or(0.0)
        };
        let v_old = v_old_held + spectral_old;
        let v_new = v_new_held + spectral_new;
        let delta = (e_new + v_new + pave_new) - (e + v_old + pave_old);
        // Unquenched trials are not answers. They still face Metropolis,
        // which is how the measured LJ38 run discovers basins from a
        // 25-step screen. Gating accept on recordable froze that path
        // at fifteen basins.
        let mut trial_local_minimum = here;
        let accept = if cfg.minima_hopping {
            let from = *here.get_or_insert_with(|| identity.basin_of(x.view()));
            if unquenched {
                feedback.observe_driver_local(Some(from), from);
                false
            } else {
                // Threshold on the *biased* rise. Adapts like Goedecker's E_diff
                // while still feeling the per-basin deposits. A self-return
                // updates escape strength but is not an acceptance trial.
                let reached = identity.basin_of(x_new.view());
                let ok = if let Some(report) = history_report {
                    // The population's identity and counts drive the
                    // controller: a minimum first found by another chain is
                    // known here too, and only a minimum new under the
                    // membership policy is offered to the threshold.
                    let visit = match cfg.shared_visit_policy {
                        SharedVisitPolicy::Tabu => {
                            feedback.remember_driver_local(reached);
                            feedback.observe_shared(
                                history_here,
                                report.minimum,
                                report.is_new,
                                report.visits,
                            )
                        }
                        SharedVisitPolicy::Recognition => {
                            feedback.observe_driver_local(Some(from), reached)
                        }
                    };
                    visit == Visit::New && feedback.accept(delta)
                } else {
                    // A rigid basin above the incumbent by more than the
                    // preset temperature is not a discovery, and neither is
                    // a proton rearrangement that leaves the centres put.
                    // Counting either as a discovery shrinks the kick.
                    let cage_changed = cfg.move_library.is_rigid_body()
                        && rigid_centres_parted(
                            x.view(),
                            x_new.view(),
                            cfg.n_points,
                            0.5 * cfg.min_separation,
                        );
                    let competitive = !cfg.move_library.is_rigid_body()
                        || (cage_changed && e_new <= ledger.best + cfg.temperature)
                        || e_new < ledger.best - 5.0e-2;
                    feedback.observe_driver_local_scored(Some(from), reached, competitive);
                    reached != from && feedback.accept(delta)
                };
                if ok {
                    trial_local_minimum = Some(reached);
                }
                ok
            }
        } else if let (Some(sur), Some((pred_y, _raw_y))) = (surrogate.as_mut(), pending_surrogate)
        {
            // Second stage. The surrogate difference is subtracted back out, so
            // what is tested is the error the surrogate made on this pair, and
            // the composite step is reversible with respect to the true target
            // whatever that error was.
            let pred_x = surrogate_here.unwrap_or(e);
            sur.stage_two += 1;
            let a2 = sur.stage_two_probability(pred_x, pred_y, e, e_new, temperature);
            let ok = rng.random::<f64>() < a2;
            if !ok {
                sur.stage_two_rejected += 1;
            } else {
                // The accepted state carries its surrogate value forward,
                // because the next first stage needs the same number this one
                // used or the ratio does not telescope.
                surrogate_here = Some(pred_y);
            }
            ok
        } else if cfg.flat_histogram {
            // Metropolis against 1/g rather than against exp(-E~/T). The bias
            // deposits still enter, so a mechanism that pushes the chain out of
            // a basin it has already paid for keeps working; what changes is
            // that the multiplicity of the destination no longer decides.
            // The density of states is held over the bare quenched energy.
            // Binning the biased energy instead puts the histogram on a
            // coordinate that drifts as deposits accumulate, so the bias enters
            // the exponent additively the way it does under Metropolis rather
            // than moving the axis.

            match flat_weight.as_ref() {
                Some(w) => {
                    let bias_delta = (v_new - v_old) / temperature.max(1e-12);
                    rng.random::<f64>() < w.accept_prob(e, e_new, bias_delta)
                }
                // Before the window exists there is no entropy to accept
                // against, so the first sweep runs the rule it is replacing and
                // its energies are what set the window.
                None => {
                    delta < 0.0 || rng.random::<f64>() < (-delta / temperature.max(1e-12)).exp()
                }
            }
        } else {
            // The energy bias enters the exponent alongside the per-basin one,
            // so the two compose rather than one replacing the other.
            let eb = ebias
                .as_ref()
                .map(|b| b.delta(e, e_new, temperature))
                .unwrap_or(0.0);
            let d = delta / temperature.max(1e-12) + eb;
            d < 0.0 || rng.random::<f64>() < (-d).exp()
        };
        let mut accept = accept;
        if cfg.move_library.is_rigid_body() && !unquenched {
            // Metropolis at the preset temperature. The threshold rule
            // collapses under a run of small accepts and then cannot
            // cross the gap between rigid cages.
            let rise = (e_new - e) / cfg.temperature.max(1e-12);
            accept = rise <= 0.0 || rng.random::<f64>() < (-rise).exp();
        }
        if cfg.move_library.is_rigid_body() && cfg.n_points > RIGID_HOT_BODIES {
            orient_window += 1;
            if accept {
                orient_yes += 1;
            }
            if orient_window >= RIGID_ACCEPT_BLOCK {
                orient_scale = rigid_step_scale(orient_scale, orient_yes, orient_window);
                orient_window = 0;
                orient_yes = 0;
            }
        }
        // AS-KMC (Chatterjee–Voter 2010): after N_f sightings the
        // intra-packing hop is a frequent process. Raise its barrier
        // (α = 2, their usual superbasin tolerance) so the rare exit
        // is selected. Not another deposit: that is well-tempered
        // filling. This is rate scaling.
        if accept && cfg.adaptive_height {
            let cap = cfg.bias_height * cfg.height_revisits.max(1.0);
            if bias.frequent_superbasin(s_old.view(), s_new.view(), cap)
                && rng.random::<f64>() * 2.0 > 1.0
            {
                accept = false;
            }
        }
        // Counted before the tabu veto, so the figure describes the acceptance
        // rule rather than the rule plus whatever the veto happens to remove.
        // White and Mayne report plain basin hopping running near a half, and
        // the temperature that produces it is the parameter every other
        // mechanism here sits downstream of; it has never been measured in this
        // driver.
        if cfg.move_library.is_rigid_body() && returning && (e_new - e).abs() <= 5.0e-2 {
            // Same quenched energy and the same distance key. Adopting it
            // does not leave the basin, so it is not an acceptance the
            // temperature window should see.
            accept = false;
        }
        if cfg.move_library.is_rigid_body()
            && cfg.n_points <= RIGID_HOT_BODIES
            && (!returning || (e_new - e).abs() > 5.0e-2)
        {
            // Same-basin accepts have no energy change, and counting them
            // keeps a rigid chart cold. Only a move that lands on a
            // different minimum says whether the temperature is high enough.
            let changed = accept && (e_new - e).abs() > 5.0e-2;
            rigid_window += 1;
            if changed {
                rigid_hits += 1;
            }
            if rigid_window >= 8 {
                let rate = rigid_hits as f64 / rigid_window as f64;
                let cap = rigid_factor_cap(cfg.n_points);
                if rate < 0.25 {
                    rigid_factor = (rigid_factor * 2.0).min(cap);
                } else if rate > 0.6 {
                    rigid_factor = (rigid_factor / 1.5).max(1.0);
                }
                rigid_window = 0;
                rigid_hits = 0;
            }
        }
        // An uphill step the bare rule accepted is checked with the
        // deposits before the chain moves. A filled destination can
        // still refuse it. The random draw already happened.
        if skip_basin && accept && !cfg.move_library.is_rigid_body() {
            let old_cv = bias.cv(x.view());
            let new_cv = bias.cv(x_new.view());
            let biased =
                (e_new + bias.potential(new_cv.view())) - (e + bias.potential(old_cv.view()));
            if biased > e_new - e {
                accept = false;
            }
        }
        if accept {
            accepted += 1;
            // Every accepted quench within the window is a candidate stage
            // of the road out of the floor. Offered before the floor can
            // move: a new incumbent redraws every gap, and the bank tolerates
            // stale bands because only the frontier is ever read.
            if seam_trace
                && recordable
                && let (Some(floor), Some(trial)) = (
                    ledger
                        .best_state
                        .as_ref()
                        .and_then(|b| b.as_slice().map(<[f64]>::to_vec)),
                    x_new.as_slice(),
                )
            {
                let gap = crate::catalog::packing_seam_gap(&floor, trial);
                // The two template shares that separate the funnels on the
                // sealed pair: FCC 0.013 against 0.120 and icosahedral
                // 0.307 against 0.120. Whether the isomers smear across
                // that separation the way they smeared across the DECAF
                // gap is what the trace is for.
                let novel = trace_book
                    .as_mut()
                    .and_then(|book| {
                        let share = book.unseen_share(trial);
                        book.observe(trial);
                        share
                    })
                    .unwrap_or(f64::NAN);
                println!(
                    "{{\"kind\":\"seam_trace\",\"hop\":{hops},\"gap\":{gap:.4},\"e\":{e_new:.6},\"novel\":{novel:.4}}}"
                );
            }
            if let Some(seam) = seam.as_mut()
                && recordable
                && let (Some(floor_state), Some(trial)) =
                    (channel_floor_state.as_ref(), x_new.as_slice())
            {
                let share = crate::catalog::icosahedral_share(x_new.view());
                let floor_share = crate::catalog::icosahedral_share(floor_state.view());
                if crate::catalog::fivefold_channel(e_new, channel_floor_energy, share, floor_share)
                    && seam.offer(share, e_new, trial)
                    && frontier_exchange
                {
                    crate::catalog::offer_frontier_post(share, e_new, trial);
                }
            }
            // One arrival on the packing the chain has just moved onto.
            // The pile is what makes returning to the icosahedral shelf
            // expensive, and it only means anything if it counts arrivals
            // the chain actually made.
            if let Some(pave) = pave.as_mut()
                && let Some(slice) = x_new.as_slice()
            {
                let reached = identity.basin_of(x_new.view());
                pave.deposit(
                    reached as u64,
                    slice,
                    cfg.bias_height,
                    cfg.bias_gamma,
                    cfg.temperature,
                );
            }
        }
        // A quarantined funnel is refused whatever the energy. Checked after
        // the acceptance test rather than instead of it, so the veto is
        // visible as a veto rather than folded into the rule.
        if !tabu.is_empty() && accept {
            let d = s_new.view();
            if tabu.iter().any(|t| {
                t.len() == d.len()
                    && t.iter()
                        .zip(d.iter())
                        .map(|(p, q)| (p - q) * (p - q))
                        .sum::<f64>()
                        .sqrt()
                        <= bias.merge_radius()
            }) {
                accept = false;
                tabu_hits += 1;
            }
        }
        finish_arm(&mut arm_counts, &proposal_action, accept, improved);
        // The quench was recorded before this test. The minima file already
        // has the line; this sets the acceptance column.
        if recordable {
            ledger.mark_last_recorded(accept);
        }
        // The energy bias is one function on every rung, so its factor and its
        // deposits read one temperature whichever rung fills the sample or
        // deposits: the one the coldest rung hops at from the state it holds,
        // at which (gamma - 1) T is the sample's spread. Each rung reads the
        // bias over its own temperature.
        if cfg.energy_bias {
            let occupied = if accept { e_new } else { e };
            if occupied.is_finite() {
                match ebias.as_mut() {
                    Some(b) => {
                        b.deposit(occupied, bias_temperature);
                        if std::env::var("EBIAS_TRACE").is_ok() && b.deposits % 200 == 0 {
                            eprintln!("ebias deposits {} peak {:.4}", b.deposits, b.peak());
                        }
                    }
                    None => {
                        flat_seen.push(occupied);
                        if flat_seen.len() >= flat_sweep {
                            ebias = crate::dos::EnergyBias::from_sample(
                                &flat_seen,
                                bias_temperature,
                                crate::dos::BINS,
                            );
                            flat_seen.clear();
                        }
                    }
                }
            }
        }
        // Under the statistical temperature only the ratio-one rungs record. A
        // rung walking at r times the entropy's slope stands with weight
        // exp(-S/r), which is not the occupancy the estimator fits, so its
        // counts bias the estimate. Measured on a two-rung LJ13 ladder ten
        // times hotter at the top, they heated every rung: the hot rung's
        // energies widened the window the first sweep sets, to bins about twice
        // as wide, and the fitted entropy came out flatter where the coldest
        // rung stands, its estimate 0.27 against 0.11 after one refresh.
        if (cfg.flat_histogram || cfg.statistical_temperature)
            && (ratios[rep] == 1.0 || !cfg.statistical_temperature)
        {
            // The histogram is over where the chain *stands*, so a rejected
            // trial records the state it stayed in. Recording the proposal
            // instead would measure the move library rather than the
            // occupancy, and the occupancy is what the weight has to flatten.
            let occupied = if accept { e_new } else { e };
            if occupied.is_finite() {
                flat_seen.push(occupied);
            }
            if let Some(d) = dos.as_mut() {
                d.observe(occupied);
            }
            flat_since += 1;
            if flat_since >= flat_sweep {
                flat_since = 0;
                match dos.as_mut() {
                    Some(d) => {
                        d.refresh();
                        // The cut walks down with the energies this sweep
                        // actually reached, so the schedule follows the run's
                        // own progress rather than a cooling curve.
                        if let Some((cut, width)) =
                            crate::dos::cut_from(&flat_seen, cfg.flat_quantile)
                        {
                            flat_weight = Some(crate::dos::CutWeight {
                                weight: d.draw(rng),
                                cut,
                                width,
                            });
                        }
                        flat_seen.clear();
                    }
                    None => {
                        // The window comes from the energies the first sweep
                        // reached, padded below by the range it covered so the
                        // chain has somewhere to go that it has not yet been.
                        // Anything past the padding is handled by the linear
                        // extrapolation rather than by clamping.
                        let lo = flat_seen.iter().cloned().fold(f64::INFINITY, f64::min);
                        let hi = flat_seen.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        if lo.is_finite() && hi > lo {
                            let span = hi - lo;
                            let mut d = crate::dos::DensityOfStates::new(
                                lo - span,
                                hi + 0.3 * span,
                                crate::dos::BINS,
                            );
                            for v in flat_seen.iter() {
                                d.observe(*v);
                            }
                            d.refresh();
                            if let Some((cut, width)) =
                                crate::dos::cut_from(&flat_seen, cfg.flat_quantile)
                            {
                                flat_weight = Some(crate::dos::CutWeight {
                                    weight: d.draw(rng),
                                    cut,
                                    width,
                                });
                            }
                            dos = Some(d);
                        }
                        flat_seen.clear();
                    }
                }
            }
        }
        if angular {
            if accept {
                angular_accepted += 1;
            }
            // R adjusted toward the target acceptance, and the sign is the
            // part that has to be right.
            //
            // A low R is the strict criterion: it fires only when some point is
            // very loosely bound, and relocating such a point almost always
            // helps, so acceptance is high. A high R fires on ordinary surface
            // points, where relocation usually hurts. Acceptance therefore
            // falls as R rises, and accepting too often calls for a larger R.
            //
            // Coupled the other way it is positive feedback and runs to a
            // bound: measured at 75 points, five seeds settled near R = 0.11
            // and two ran to the 0.95 ceiling, firing the move on 30000 and
            // 63000 hops of a hundred thousand and ending at -386.30 and
            // -394.99. Wales and Doye report R converging to between 0.40 and
            // 0.44.
            //
            // Robbins-Monro on the acceptance indicator rather than on a
            // cumulative rate. A cumulative rate stops responding once the run
            // is long, so an early transient is never corrected.
            let hit = if accept { 1.0 } else { 0.0 };
            let step = 0.02 / (1.0 + angular_tried as f64 / 500.0).sqrt();
            angular_ratio = (angular_ratio + step * (hit - cfg.angular_target)).clamp(0.05, 0.95);
        } else if let Some(c) = &context {
            // The context is the one the move was chosen in, not the one the
            // chain now stands in: the reward belongs to the decision.
            contextual.update(k, c.view(), if improved || accept { 1.0 } else { 0.0 });
        } else if cfg.allocate_moves {
            // An angular step is not the allocator's, so it does not carry a
            // reward for whichever arm the allocator happened to pick.
            if cfg.depth_reward {
                // How close the move brought the chain to the best it knows.
                // Dense, because every hop produces one, and informative,
                // because its size says how deep rather than merely whether.
                depth_allocator.update(k, -(e_new - ledger.best));
            }
            // A rigid accept that only moves centres, or only turns a
            // hydrogen-bond cycle, is not evidence the arm is useful.
            // Paying for it fills the draw with that arm. An improvement
            // of the incumbent is the payment.
            let reward = if cfg.move_library.is_rigid_body() || cfg.improvement_rewards {
                improved
            } else {
                improved || (accept && !(cfg.novel_reward && returning))
            };
            allocator.update(k, reward);
            arm_draws[k] += 1;
            if accept {
                arm_accepts[k] += 1;
            }
            arm_best[k] = arm_best[k].min(e_new);
            // The quench the construction actually reached, which is the only
            // feedback the posterior gets and the reason it costs nothing
            // extra: this relaxation was paid for by the move regardless.
            if let (Some(c), Some(f)) = (&mut constructor, pending_features.take()) {
                c.observe(f.view(), e_new, e);
            }
        }
        // Every hop, accepted or not, so a basin census can tell a minimum
        // that was never proposed from one that was proposed and then lost
        // to the return screen, the record gate or the acceptance test.
        if std::env::var("ANNEAL_HOP_TRACE").is_ok() {
            let max_gradient = validation_gradient
                .as_ref()
                .map(|g| g.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
                .unwrap_or(f64::NAN);
            let coordinates: Vec<String> = x_new.iter().map(|v| format!("{v:.6}")).collect();
            eprintln!(
                "HOPTRACE {{\"hop\":{hops},\"spent\":{},\"arm\":\"{proposal_action}\",\"e\":{e:.8},\"e_screen\":{e_screen:.8},\"e_new\":{e_new:.8},\"best\":{:.8},\"drift\":{drift:.6},\"returning\":{returning},\"screened\":{screened_this},\"unquenched\":{unquenched},\"recordable\":{recordable},\"max_gradient\":{max_gradient:.6e},\"accept\":{accept},\"x\":[{}]}}",
                ledger.spent(),
                ledger.best,
                coordinates.join(",")
            );
        }
        if accept && cfg.calibrate_radius {
            // How far this hop actually moved, in the metric the bias keys on.
            let d: f64 = s_old
                .iter()
                .zip(s_new.iter())
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                .sqrt();
            radius.observe(d);
            if radius.warm() {
                let r = radius.threshold();
                if (r - bias.merge_radius()).abs() > 1e-12 {
                    bias.set_merge_radius(r);
                    identity.set_merge_radius(r);
                }
            }
        }
        if cfg.general_collective && accept && !unquenched {
            let species = cfg
                .species
                .clone()
                .unwrap_or_else(|| vec![0; x_new.len() / 3]);
            if general_taboo.forbids(e_new, x_new.view(), &species) {
                accept = false;
            } else {
                general_taboo.remember(e_new, x_new.view(), &species);
            }
        }
        if accept {
            if !unquenched && cfg.packing_cna_applies() {
                packing.offer(e_new, x_new.view(), packing_cutoff);
            }
            if !unquenched && (cfg.move_library.is_rigid_body() || cfg.improvement_rewards) {
                stall_archive.offer(e_new, x_new.view());
                cage_archive.offer_cage(
                    e_new,
                    x_new.view(),
                    rigid_cage_key(x_new.view(), cfg.n_points, cage_quantum),
                );
            }
            // An accepted uphill step to a different basin samples the escape
            // distribution, which is the quantity the deposit height has to be
            // commensurate with.
            if cfg.adaptive_height && e_new > e {
                height.observe(e_new - e);
                bias.set_height(height.height());
            }
            // The chain carries the *quenched* geometry, not the perturbed one
            // that produced it. That is White and Mayne's distinction between
            // same-structure and random-structure basin hopping, and they
            // report the first as the better operator: the next proposal is
            // made from a minimum rather than from a point part-way down a
            // slope, so a rejected step does not leave the chain somewhere the
            // landscape was never sampled at.
            //
            // Stated here because "plain basin hopping" is ambiguous without
            // it, and a baseline that is quietly the weaker operator flatters
            // everything measured against it.
            if cfg.cov_perturb {
                // The displacement that was actually accepted, minimum to
                // minimum, with the incumbent's rigid components projected out
                // so net drift of the free cluster does not masquerade as a
                // useful direction.
                let mut d = &x_new - &x;
                let rb = crate::curvature::rigid_basis(x.view());
                crate::curvature::project_rigid_with(&mut d, &rb);
                let norm: f64 = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 1e-9 {
                    if cov_buf.len() < 32 {
                        cov_buf.push(d);
                    } else {
                        cov_buf[cov_next] = d;
                        cov_next = (cov_next + 1) % 32;
                    }
                }
            }
            // The chain's own trajectory, not only its record: funnel entry
            // happens through accepted moves that are worse than the best so
            // far, which the improvement trace cannot see.
            if std::env::var("ANNEAL_ACC_TRACE").is_ok() {
                let m = crate::structure::ptm(x_new.view(), n, 0.12);
                let mut fcc = 0usize;
                let mut hcp = 0usize;
                let mut ico = 0usize;
                for t in &m {
                    match t.template {
                        crate::structure::Template::FaceCentredCubic => fcc += 1,
                        crate::structure::Template::HexagonalClosePacked => hcp += 1,
                        crate::structure::Template::Icosahedral => ico += 1,
                        _ => {}
                    }
                }
                let arm = if soft_fire {
                    "soft".to_string()
                } else if cov_fire {
                    "cov".to_string()
                } else if angular {
                    "angular".to_string()
                } else {
                    kernels[k].name()
                };
                eprintln!("ACCTRACE hop {hops} e {e_new:.6} arm {arm} ord {fcc}/{hcp}/{ico}");
            }
            // Convergence curves: charged evaluations against the accepted
            // energy, one line per accept; the reader takes the running
            // minimum. ANNEAL_CURVE names the file.
            if let Ok(path) = std::env::var("ANNEAL_CURVE") {
                use std::io::Write as _;
                if let Ok(mut fh) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                {
                    let _ = writeln!(fh, "{} {e_new:.8}", ledger.spent());
                }
            }
            accepted_transitions.push(AcceptedTransition {
                hop: hops,
                action: proposal_action,
                from_energy: e,
                to_energy: e_new,
                from_state: x.clone(),
                from_gradient: current_validation_gradient.clone(),
                to_state: x_new.clone(),
                to_gradient: validation_gradient.clone(),
                validated: recordable,
                adopted: true,
            });
            if !returning && let Some(snapshot) = entry_snapshot {
                basin_entry = Some(snapshot);
            }
            moved_basin = !returning;
            adopt_minimum!(
                e_new,
                x_new,
                validation_gradient,
                trial_local_minimum,
                AdoptionHistory::Observed(history_report),
                None,
                None,
                hops
            );
        } else if cfg.budget_window {
            // The biased delta, which is what the chain actually declined, not
            // the raw energy difference. The bias is part of the barrier the
            // chain faces, and estimating the barrier without it measures a
            // landscape the chain is not walking on.
            //
            // Divided by the rung's ratio, because the law sets the coldest
            // rung's temperature: a rise declined at `r T` weighs what one `r`
            // times smaller does at `T`, so the escape floor asks each rung to
            // clear, at its own temperature, what it is failing to cross. The
            // larger rises a hot rung declines would otherwise heat every rung.
            law.observe_rejection(delta / ratios[rep]);
        }
        if !skip_basin {
            let (occupied, seat) = if accept {
                (s_new.view(), seat_new)
            } else {
                (s_old.view(), seat_old)
            };
            let landed_seat = bias.deposit_at(occupied, seat, temperature);
            held_bits.clear();
            held_bits.extend(x.iter().map(|value| value.to_bits()));
            held_cv = Some(if accept { s_new.clone() } else { s_old.clone() });
            held_seat = Some(landed_seat);
            held_count = bias.n_basins();
        }
        // Visits other chains made to the minimum this quench reached, paid
        // into this chain's bias as if it had made them. The count is under
        // the history's membership policy, so with accepted membership only
        // minima some chain stood in are filled. Capped per look, because a
        // shelf the population has sat on for a hundred thousand hops is
        // filled well before the cap and an uncapped loop is pure cost.
        if !cfg.minima_hopping
            && cfg.shared_deposits > 0
            && let (Some(report), Some(cv)) = (history_report, history_cv.as_ref())
        {
            let seen = history_seen.entry(report.minimum).or_insert(0);
            // This chain's own observation is already in its hop deposits.
            let foreign = report
                .visits
                .saturating_sub(*seen)
                .saturating_sub(1)
                .min(cfg.shared_deposits as u64);
            for _ in 0..foreign {
                // Counted as this chain's own visits: the rung owes what the
                // other rung recorded while this bias was parked.
                bias.deposit(cv.view(), temperature);
                shared_deposits += 1;
            }
            *seen = report.visits;
            // Under accepted membership this chain's own adoption raises the
            // count from zero to one after the look; that visit is its own.
            if accept && report.visits == 0 {
                *seen = 1;
            }
        }
        // Core symmetrisation of a newly entered basin (Oakley, Johnston,
        // Wales 2013), quenched and offered to the same acceptance rule as a
        // hop. Once per new basin: the arm form of this move was measured to
        // take half of all draws at 83 per cent acceptance on 75 points,
        // because a symmetric minimum re-symmetrises to itself and an
        // accepted return is still an accept.
        if cfg.point_symmetrise_on_new
            && accept
            && hop_room(hops)
            && (moved_basin || cfg.point_symmetrise_every_accept)
            && let Some(y) = crate::symmetrise::symmetrise_core(
                x.view(),
                n,
                cfg.symmetry_tolerance,
                cfg.symmetry_merge_radius,
                cfg.symmetrise_core_fraction,
            )
        {
            let (es, xs) = relax(ledger, y.view(), cfg.relax_steps);
            if es.is_finite() && xs.len() == x.len() {
                let sym_gradient = record_quenched_answer(
                    cfg,
                    ledger,
                    &mut grad,
                    es,
                    xs.view(),
                    &mut unconverged_records,
                );
                hops += 1;
                symmetrised += 1;
                let d = (es - e) / temperature.max(1e-12);
                if d < 0.0 || rng.random::<f64>() < (-d).exp() {
                    if es < e {
                        symmetry_gain += e - es;
                    }
                    adopt_minimum!(
                        es,
                        xs,
                        sym_gradient,
                        None,
                        AdoptionHistory::Fresh,
                        None,
                        Some("point-symmetrise"),
                        hops
                    );
                }
            }
        }
        // Orbit completion of a newly entered basin: the core's point group
        // applied to the whole cluster, so surface atoms move onto the empty
        // orbit positions the core implies. Once per new basin, every
        // candidate the core asks for is quenched and recorded, and the
        // lowest is offered to the acceptance rule.
        let mut orbit_offer = None;
        if cfg.packing_surface().orbit_on_new
            && accept
            && hop_room(hops)
            && (moved_basin || cfg.point_symmetrise_every_accept)
        {
            for y in crate::packing::orbit_candidates(
                x.view(),
                n,
                cfg.orbit_core,
                cfg.symmetry_tolerance,
                cfg.symmetry_merge_radius,
                cfg.symmetrise_core_fraction,
                cfg.min_separation,
            ) {
                let (es, xs) = relax(ledger, y.view(), cfg.relax_steps);
                if es.is_finite() && xs.len() == x.len() {
                    let sym_gradient = record_quenched_answer(
                        cfg,
                        ledger,
                        &mut grad,
                        es,
                        xs.view(),
                        &mut unconverged_records,
                    );
                    hops += 1;
                    orbits_completed += 1;
                    if orbit_offer
                        .as_ref()
                        .is_none_or(|(lowest, _, _)| es < *lowest)
                    {
                        orbit_offer = Some((es, xs, sym_gradient));
                    }
                }
            }
        }
        if let Some((es, xs, sym_gradient)) = orbit_offer {
            let d = (es - e) / temperature.max(1e-12);
            if d < 0.0 || rng.random::<f64>() < (-d).exp() {
                if es < e {
                    orbit_gain += e - es;
                }
                adopt_minimum!(
                    es,
                    xs,
                    sym_gradient,
                    None,
                    AdoptionHistory::Fresh,
                    None,
                    Some("orbit-completion"),
                    hops
                );
            }
        }
        // Graph edge + Fiedler deposit at the chain's current basin. Called
        // every hop (accepted or not) so the coordinate tracks occupation;
        // only accepted moves grow the graph (visit records last→current).
        if let Some(sp) = spectral.as_mut() {
            sp.visit(x.view(), temperature);
        }

        // A climb out of the funnel, when nothing else is working.
        //
        // Under minima hopping the escape scale also multiplies the overshoot:
        // a chain that has been revisiting is thrown further past the ridge it
        // just crossed, which is the same feedback law on a quantity that can
        // actually leave a basin.
        if improved {
            longest_quiet = longest_quiet.max(quiet);
            quiet = 0;
        } else {
            quiet += 1;
        }
        // Stuck means stuck for longer than this chain has ever been stuck
        // before, not stuck for some number someone chose. Traced on 75
        // points, a run that succeeds goes tens of thousands of hops between
        // improvements on its way to the crossing, so a fixed patience of 400
        // fires about 180 climbs into a healthy search.
        let stuck = quiet
            >= cfg
                .escape_stall_patience
                .max((cfg.escape_stall_factor * longest_quiet as f64) as usize);
        // Basin hopping with occasional jumping (Iwamatsu and Okabe, Chem.
        // Phys. Lett. 399, 396 (2004)): after `jump_patience` hops without
        // improvement, take `jump_steps` collective displacements with no
        // quench and no acceptance test, then quench and continue from
        // wherever that lands. The walk keeps most of the structure, so the
        // landing is a neighbouring region rather than a random start,
        // which is what the measured shelf needs: the control at ten times
        // the budget left the icosahedral shelf about once per several
        // hundred thousand hops, and a fresh random start pays the descent
        // again and lands on the shelf with the same odds.
        if cfg.jump_on_stall
            && hop_room(hops)
            && cfg.jump_patience > 0
            && quiet >= cfg.jump_patience
            && quiet.is_multiple_of(cfg.jump_patience)
        {
            let mut y = x.clone();
            for _ in 0..cfg.jump_steps.max(1) {
                for v in y.iter_mut() {
                    *v += rng.random_range(-cfg.jump_step..cfg.jump_step);
                }
            }
            let (ej, xj) = relax(ledger, y.view(), cfg.relax_steps);
            if ej.is_finite() && xj.len() == x.len() && quench_is_sane(cfg, ej, xj.view()) {
                let jump_gradient = record_quenched_answer(
                    cfg,
                    ledger,
                    &mut grad,
                    ej,
                    xj.view(),
                    &mut unconverged_records,
                );
                hops += 1;
                jumps += 1;
                adopt_minimum!(
                    ej,
                    xj,
                    jump_gradient,
                    None,
                    AdoptionHistory::Fresh,
                    None,
                    Some("stall-jump"),
                    hops
                );
                longest_quiet = longest_quiet.max(quiet);
                quiet = 0;
            }
        }
        // Where the chain stands after the acceptance test, computed once for
        // every consumer that wants it.
        let landed = if accept && here_before.is_some() {
            let now = identity.basin_of(x.view());
            here = Some(now);
            Some(now)
        } else {
            None
        };
        if let (Some(sb), Some(from), Some(now)) = (superbasin.as_mut(), here_before, landed) {
            sb.observe_accepted(from, now);
            // Every accepted state is a landing point: the chain is standing
            // on it and continuing from it.
            if !unquenched {
                sb.keep(now, e, x.view());
            }
        }
        if cfg.track_funnels {
            // Accepted hops only. A rejected proposal says the chain declined
            // to move, which is a statement about the acceptance rule rather
            // than about reachability.
            if let (Some(prev), Some(now)) = (here_before, landed) {
                funnels.record(prev, now);
            }
            if funnels.pending() >= cfg.funnel_period && funnels.len() >= 8 {
                funnel_split = funnels.split().ok();
            }
        }
        // A stuck chain restarts from the shared frontier rather than
        // symmetrising or escaping in place. This is the cloning step of
        // the seam ladder: the stage the ensemble has already reached is
        // restored to full population before the next attempt, so the
        // frontier arrival rate stays constant in the depth of the road
        // instead of decaying by one factor per stage
        // (Hop.cloning_dominates). The jump replaces the state and its
        // energy and nothing else; Metropolis continues raw.
        if quiet >= seam_patience
            && let Some(bank) = seam.as_ref()
            && let Some((_, frontier_energy, frontier_state)) =
                bank.restart(0.25, rng.random::<f64>())
            && frontier_state.len() == x.len()
        {
            adopt_minimum!(
                frontier_energy,
                Array1::from(frontier_state.to_vec()),
                None,
                None,
                AdoptionHistory::Fresh,
                None,
                Some("seam-frontier"),
                hops
            );
            quiet = 0;
            longest_quiet = 0;
            restarts += 1;
            continue;
        }
        if cfg.symmetrise_on_stall && stuck && hop_room(hops) {
            let soap_spec = crate::soap::SoapSpec {
                rcut_nn: 3.5 * cfg.length_scale,
                ..Default::default()
            };
            let withhold = (cfg.packing_cna_applies()
                && crate::soap::ih_dominated(x.view(), soap_spec))
                || cfg.active_region.is_some()
                || cfg.frozen.is_some();
            if withhold {
                // Withhold Ih-preserving symmetrise on a packing cluster,
                // and never run CNA or point-group symmetrise on a slab.
                // A molecule without a frozen frame still uses the branch
                // below. The 555 fraction is an observation, not a hop
                // target.
                quiet = 0;
                longest_quiet = 0;
            } else {
                // The stall counter is cleared here, not only in the escape
                // branches below. Without it a stuck chain satisfies the condition
                // on every subsequent hop and symmetrises on every one: measured at
                // 98 points, 57989 firings in a single run, about one hop in seven,
                // and the seed came back at -539.81 against a target of -543.67.
                quiet = 0;
                longest_quiet = 0;
                // The structure is pushed onto whatever approximate symmetry it
                // has and quenched. Taken only when it improves: unlike a funnel
                // escape, this is a guess about where the answer is, not a way out
                // of where the chain is.
                // The whole point group, not one axis. A tetrahedral structure is
                // not produced by averaging orbits under a single three-fold
                // rotation; the tetrahedral group is generated by a three-fold and
                // a two-fold together, and the 98-point global minimum is
                // tetrahedral. Detecting several axes and closing them into a group
                // is what makes this the published scheme rather than an axial
                // constraint wearing its name.
                let mut cands: Vec<crate::symmetrise::Candidate> = Vec::new();
                for order in [2usize, 3, 4, 5, 6] {
                    if let Some(c) =
                        crate::symmetrise::detect(x.view(), n, &[order], cfg.symmetry_tolerance)
                    {
                        cands.push(c);
                    }
                }
                let group = crate::symmetrise::generate_group(&cands, 60);
                let symmetrised_state = if group.len() > 1 {
                    Some(match cfg.species.as_deref() {
                        Some(species) if species.len() == n => {
                            crate::symmetrise::symmetrise_group_within(
                                x.view(),
                                n,
                                &group,
                                cfg.symmetry_merge_radius,
                                species,
                            )
                        }
                        _ => crate::symmetrise::symmetrise_group(
                            x.view(),
                            n,
                            &group,
                            cfg.symmetry_merge_radius,
                        ),
                    })
                } else {
                    crate::symmetrise::symmetrise_detected(
                        x.view(),
                        n,
                        &[2, 3, 4, 5, 6],
                        cfg.symmetry_tolerance,
                        cfg.symmetry_merge_radius,
                    )
                    .map(|(y, _)| y)
                };
                if let Some(y) = symmetrised_state {
                    let (es, xs) = relax(ledger, y.view(), cfg.relax_steps);
                    let sym_gradient = record_quenched_answer(
                        cfg,
                        ledger,
                        &mut grad,
                        es,
                        xs.view(),
                        &mut unconverged_records,
                    );
                    hops += 1;
                    symmetrised += 1;
                    if es < e {
                        symmetry_gain += e - es;
                        adopt_minimum!(
                            es,
                            xs,
                            sym_gradient,
                            None,
                            AdoptionHistory::Fresh,
                            None,
                            Some("stall-symmetrise"),
                            hops
                        );
                    }
                }
            }
        }
        if cfg.tabu_on_stall && stuck {
            // The funnel the chain has been unable to leave, named by where it
            // is standing.
            let d = bias.cv(x.view());
            if !tabu.iter().any(|t| {
                t.len() == d.len()
                    && t.iter()
                        .zip(d.iter())
                        .map(|(p, q)| (p - q) * (p - q))
                        .sum::<f64>()
                        .sqrt()
                        <= bias.merge_radius()
            }) {
                if tabu.len() >= cfg.tabu_capacity {
                    tabu.remove(0);
                }
                tabu.push(d);
            }
        }
        // Tabu names the basin. It does not redraw the cluster.
        // Coupling the two sampled Doye's wide icosahedral funnel
        // (J. Chem. Phys. 111, 8417 (1999)): a random restart lands
        // back on Mackay. The call plateau above is the directed leave
        // when the climb is on, and it is not a draw among the stall arms.
        // Each stall arm can take a hop, and the quench above may have taken
        // the last one the cap allows. The climb can take up to three, one
        // per lattice growth and one for its activation quench, so each of
        // those rechecks.
        let stall_response = if !hop_room(hops) {
            None
        } else if directed_leave {
            stall_arms.iter().position(|arm| *arm == "climb")
        } else if stuck && !stall_arms.is_empty() {
            Some(if stall_arms.len() > 1 {
                stall_allocator.select(rng)
            } else {
                0
            })
        } else {
            None
        };
        let stalled_from = e;
        let mut stall_outcome: Option<f64> = None;
        if stall_response.map(|arm| stall_arms[arm] == "restart") == Some(true) {
            quiet = 0;
            longest_quiet = 0;
            let fresh = if let Some(groups) = cfg.move_library.declared_groups() {
                // Keep each rigid group's internal geometry; only the
                // packing is redrawn. An atomic redraw overlaps waters.
                repack_rigid_groups(start, groups, cfg.length_scale, rng)
            } else {
                random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, rng)
            };
            let (ef, xf) = relax(ledger, fresh.view(), cfg.relax_steps);
            let restart_gradient = record_quenched_answer(
                cfg,
                ledger,
                &mut grad,
                ef,
                xf.view(),
                &mut unconverged_records,
            );
            hops += 1;
            restarts += 1;
            adopt_minimum!(
                ef,
                xf,
                restart_gradient,
                None,
                AdoptionHistory::Fresh,
                None,
                Some("stall-restart"),
                hops
            );
            stall_outcome = Some(stalled_from - e);
        }
        if stall_response.map(|arm| stall_arms[arm] == "trail") == Some(true)
            && let Some(x_entry) = basin_entry.take()
        {
            quiet = 0;
            longest_quiet = 0;
            // Outward is from the minimum toward the entry, continued past it
            // with noise, so the restart leans toward the ridge the chain
            // crossed on the way in rather than back down the funnel.
            let mut exit = x_entry.clone();
            let outward: Vec<f64> = x_entry
                .iter()
                .zip(x.iter())
                .map(|(entry, minimum)| entry - minimum)
                .collect();
            let norm = outward.iter().map(|v| v * v).sum::<f64>().sqrt();
            for (index, value) in exit.iter_mut().enumerate() {
                let along = if norm > 0.0 {
                    outward[index] / norm
                } else {
                    0.0
                };
                *value += cfg.escape_amplitude * (0.5 * along + rng.random::<f64>() - 0.5);
            }
            let (ee, xe) = relax(ledger, exit.view(), cfg.relax_steps);
            let trail_gradient = record_quenched_answer(
                cfg,
                ledger,
                &mut grad,
                ee,
                xe.view(),
                &mut unconverged_records,
            );
            hops += 1;
            stall_escapes += 1;
            trail_escapes += 1;
            if ee < e {
                stall_escape_gain += e - ee;
            }
            // Taken whatever its energy, exactly as the climb below: the
            // chain has shown it cannot improve from where it stands.
            adopt_minimum!(
                ee,
                xe,
                trail_gradient,
                None,
                AdoptionHistory::Fresh,
                None,
                Some("stall-trail"),
                hops
            );
            stall_outcome = Some(stalled_from - e);
        }
        if stall_response.map(|arm| stall_arms[arm] == "climb") == Some(true) {
            quiet = 0;
            longest_quiet = 0;
            // An icosahedral plateau returns to a lower minimum of another
            // packing when the bank holds one. The structure was already
            // quenched, so this does not run another relaxation and it draws
            // no random numbers. A face-centred-cubic plateau does not take
            // it: that is the 38-point minimum, and leaving it is the wrong
            // move.
            if directed_leave
                && cfg.packing_cna_applies()
                && crate::soap::ih_dominated(x.view(), crate::soap::SoapSpec::default())
                && let Some((other_energy, other_state)) =
                    packing.better_other(x.view(), e, packing_cutoff)
            {
                let other_gradient = record_quenched_answer(
                    cfg,
                    ledger,
                    &mut grad,
                    other_energy,
                    other_state.view(),
                    &mut unconverged_records,
                );
                if grad.is_none() || other_gradient.is_some() {
                    e = other_energy;
                    x = other_state;
                    current_validation_gradient = other_gradient;
                }
            }
            // The budget window only cools a derived search. It does not
            // decide whether a non-icosahedral plateau may leave. Recommended
            // leaves that flag off, and the icosahedral test below is false
            // for the LJ104 minima, so neither growth ran and the hold
            // stayed unset.
            // The chain takes a growth only when it deepens the incumbent.
            // A miss sets the hold. The leave that then skips growth is the
            // all-points displacement above. The library keeps proposing
            // until the incumbent drops.
            if directed_leave
                && cfg.prior_growth
                && hop_room(hops)
                && cfg.packing_cna_applies()
                && !crate::soap::ih_dominated(x.view(), crate::soap::SoapSpec::default())
            {
                let best_before_growth = ledger.best;
                if growth_held(
                    stall_growth_hold,
                    best_before_growth,
                    stall_growth_skips,
                    cfg.n_points,
                ) {
                    // The lattice quench already failed at this incumbent.
                    // Further growth waits. This leave's proposal is the
                    // all-points displacement. After several idle leaves the
                    // pentagonal order is tried again below.
                    stall_growth_skips = stall_growth_skips.saturating_add(1);
                } else {
                    let retry = stall_growth_waits(stall_growth_hold, best_before_growth);
                    stall_growth_skips = 0;
                    let mut best_growth: Option<(f64, Array1<f64>, Option<Array1<f64>>)> = None;
                    // A flat incumbent whose live chain is under the
                    // icosahedral cut never opened the pentagonal branch.
                    // The retry writes that order, plus the two close
                    // packings, from a side stream so a miss leaves the
                    // chain's own draws where they were.
                    let mut side = StdRng::seed_from_u64(
                        (hops as u64).wrapping_mul(0x9E37_79B9)
                            ^ best_before_growth.to_bits()
                            ^ 0x0C05_11CE,
                    );
                    let crossing = [
                        crate::lattice::Source::Named(crate::structure::Template::FaceCentredCubic),
                        crate::lattice::Source::Named(
                            crate::structure::Template::HexagonalClosePacked,
                        ),
                        crate::lattice::Source::Pentagonal,
                    ];
                    if retry {
                        for source in crossing {
                            if ledger.remaining() == 0 {
                                break;
                            }
                            let grown =
                                crate::lattice::candidate(source, x.view(), x.len() / 3, &mut side);
                            let (grown_energy, grown_state) =
                                relax(ledger, grown.view(), cfg.relax_steps);
                            if grown_state.len() != x.len() || !grown_energy.is_finite() {
                                continue;
                            }
                            let grown_gradient = record_quenched_answer(
                                cfg,
                                ledger,
                                &mut grad,
                                grown_energy,
                                grown_state.view(),
                                &mut unconverged_records,
                            );
                            let replace = best_growth
                                .as_ref()
                                .is_none_or(|(energy, _, _)| grown_energy < *energy);
                            if replace {
                                best_growth = Some((grown_energy, grown_state, grown_gradient));
                            }
                        }
                    } else {
                        for source in crate::lattice::Source::library() {
                            if ledger.remaining() == 0 {
                                break;
                            }
                            let grown =
                                crate::lattice::candidate(source, x.view(), x.len() / 3, rng);
                            let (grown_energy, grown_state) =
                                relax(ledger, grown.view(), cfg.relax_steps);
                            if grown_state.len() != x.len() || !grown_energy.is_finite() {
                                continue;
                            }
                            let grown_gradient = record_quenched_answer(
                                cfg,
                                ledger,
                                &mut grad,
                                grown_energy,
                                grown_state.view(),
                                &mut unconverged_records,
                            );
                            let replace = best_growth
                                .as_ref()
                                .is_none_or(|(energy, _, _)| grown_energy < *energy);
                            if replace {
                                best_growth = Some((grown_energy, grown_state, grown_gradient));
                            }
                        }
                    }
                    if let Some((grown_energy, grown_state, grown_gradient)) = best_growth
                        && grown_energy < best_before_growth - 1e-9
                    {
                        hops += 1;
                        adopt_minimum!(
                            grown_energy,
                            grown_state,
                            grown_gradient,
                            None,
                            AdoptionHistory::Fresh,
                            None,
                            Some("stall-growth"),
                            hops
                        );
                    }
                    if ledger.best < best_before_growth - 1e-9 {
                        stall_growth_hold = None;
                    } else {
                        stall_growth_hold = Some(best_before_growth);
                    }
                    if ledger.best < best_before_growth - 1e-12 && improvements.len() < 512 {
                        improvements.push((hops, ledger.spent(), bias.n_basins(), ledger.best));
                    }
                }
            }
            // The same growth while the occupied structure is icosahedral.
            // The chain is then inside the funnel that contains most of the
            // minima, and the other orders, including the pentagonal one a
            // decahedron is built from, are the leave. The draws come from a
            // stream seeded by the hop and the incumbent, so a growth that
            // does not win leaves the chain's own stream where it was. The
            // quench is kept only when it deepens the incumbent. A growth
            // that fails holds the next one until the incumbent changes.
            // The skipped leave is one all-points displacement.
            if directed_leave
                && hop_room(hops)
                && cfg.packing_cna_applies()
                && crate::soap::ih_dominated(x.view(), crate::soap::SoapSpec::default())
            {
                let best_before_growth = ledger.best;
                if growth_held(
                    stall_growth_hold,
                    best_before_growth,
                    stall_growth_skips,
                    cfg.n_points,
                ) {
                    // Same wait as the other lattice growth. This leave's
                    // proposal is the all-points displacement.
                    stall_growth_skips = stall_growth_skips.saturating_add(1);
                } else {
                    stall_growth_skips = 0;
                    let mut side = StdRng::seed_from_u64(
                        (hops as u64).wrapping_mul(0x9E37_79B9)
                            ^ best_before_growth.to_bits()
                            ^ 0x11A7_11CE,
                    );
                    // Close packing and the pentagonal order. Regrowing the
                    // icosahedron the chain is already in does not leave it.
                    let sources = [
                        crate::lattice::Source::Named(crate::structure::Template::FaceCentredCubic),
                        crate::lattice::Source::Named(
                            crate::structure::Template::HexagonalClosePacked,
                        ),
                        crate::lattice::Source::Pentagonal,
                    ];
                    let mut best_growth: Option<(f64, Array1<f64>, Option<Array1<f64>>)> = None;
                    for source in sources {
                        if ledger.remaining() == 0 {
                            break;
                        }
                        let grown =
                            crate::lattice::candidate(source, x.view(), x.len() / 3, &mut side);
                        let (grown_energy, grown_state) =
                            relax(ledger, grown.view(), cfg.relax_steps);
                        if grown_state.len() != x.len() || !grown_energy.is_finite() {
                            continue;
                        }
                        let grown_gradient = record_quenched_answer(
                            cfg,
                            ledger,
                            &mut grad,
                            grown_energy,
                            grown_state.view(),
                            &mut unconverged_records,
                        );
                        let replace = best_growth
                            .as_ref()
                            .is_none_or(|(energy, _, _)| grown_energy < *energy);
                        if replace {
                            best_growth = Some((grown_energy, grown_state, grown_gradient));
                        }
                    }
                    if let Some((grown_energy, grown_state, grown_gradient)) = best_growth
                        && grown_energy < best_before_growth - 1e-9
                    {
                        hops += 1;
                        adopt_minimum!(
                            grown_energy,
                            grown_state,
                            grown_gradient,
                            None,
                            AdoptionHistory::Fresh,
                            None,
                            Some("stall-growth"),
                            hops
                        );
                        packing.offer(e, x.view(), packing_cutoff);
                    }
                    if ledger.best < best_before_growth - 1e-9 {
                        stall_growth_hold = None;
                    } else {
                        stall_growth_hold = Some(best_before_growth);
                    }
                    if ledger.best < best_before_growth - 1e-12 && improvements.len() < 512 {
                        improvements.push((hops, ledger.spent(), bias.n_basins(), ledger.best));
                    }
                }
            }
            if hop_room(hops)
                && let Some(g) = grad.as_deref_mut()
            {
                let scale = if cfg.minima_hopping {
                    feedback.escape()
                } else {
                    1.0
                };
                let act = Activation {
                    step: cfg.escape_amplitude,
                    overshoot: cfg.escape_overshoot * scale,
                    max_steps: cfg.escape_max_climb,
                    lanczos_steps: cfg.escape_lanczos_steps,
                    epsilon: cfg.escape_epsilon,
                    ..Activation::default()
                };
                let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
                if let Some(o) = activate(x.view(), |y| g(ledger, y), &act, sign) {
                    soft_escapes += 1;
                    if o.crossed {
                        soft_crossed += 1;
                    }
                    soft_lambda += o.lambda;
                    let (ee, xe) = relax(ledger, o.state.view(), cfg.relax_steps);
                    let escape_gradient = record_quenched_answer(
                        cfg,
                        ledger,
                        &mut grad,
                        ee,
                        xe.view(),
                        &mut unconverged_records,
                    );
                    hops += 1;
                    stall_escapes += 1;
                    if ee < e {
                        stall_escape_gain += e - ee;
                    }
                    // Taken whatever its energy. The chain has already shown it
                    // cannot improve from where it is, so the value of the new
                    // structure is that it is somewhere else.
                    let local_feedback = if cfg.minima_hopping {
                        let from = here.unwrap_or_else(|| identity.basin_of(x.view()));
                        let reached = identity.basin_of(xe.view());
                        Some((Some(from), reached))
                    } else {
                        None
                    };
                    adopt_minimum!(
                        ee,
                        xe,
                        escape_gradient,
                        local_feedback.map(|(_, reached)| reached),
                        AdoptionHistory::Fresh,
                        local_feedback,
                        Some("stall-climb"),
                        hops
                    );
                    stall_outcome = Some(stalled_from - e);
                }
            }
        }
        if let (Some(arm), Some(gain)) = (stall_response, stall_outcome) {
            stall_allocator.update(arm, gain);
        }
        // A new best is the one state worth sharing, and the census a shared
        // run identifies basins in was calibrated at the share tolerance,
        // which neither the ordinary record gate nor an escape's capped
        // relaxation reaches. One choke point covers every route a best can
        // arrive by, hop records and stall escapes alike: whenever this
        // iteration deepened the ledger, its best state is polished through
        // the relax closure, which owns boundary recording and share-grade
        // validation, so the polished minimum enters the boundary record the
        // checkpoint offer loop already reads. Bounded by the number of
        // improvement events and charged like any other relaxation.
        // Two triggers, one mechanism. A deepened record is worth sharing on
        // its own; a move into a fresh basin is worth registering because a
        // census that only knows record-breakers cannot tell a chain it is
        // walking into a basin someone has already searched, and telling it
        // exactly that is what the shared catalog is for. The cooldown
        // bounds the visit trigger so polishing can never take more than
        // about a fifth of the budget on a chain that hops basins every
        // slice.
        let improved_record = ledger.best < best_before_iteration - 1e-10;
        let visit_due = moved_basin
            && ledger.spent().saturating_sub(last_polish_spent)
                >= cfg.polish_records.saturating_mul(4);
        if cfg.polish_records > 0
            && (improved_record || visit_due)
            && let Some(best_state) = if improved_record {
                ledger.best_state.clone()
            } else {
                Some(x.clone())
            }
        {
            last_polish_spent = ledger.spent();
            let (polished_energy, polished_state) =
                relax(ledger, best_state.view(), cfg.polish_records);
            if std::env::var("ANNEAL_POLISH_TRACE").is_ok() {
                eprintln!(
                    "POLISH hop {hops} from {best} to {polished_energy} spent {spent}",
                    best = best_before_iteration,
                    spent = ledger.spent(),
                );
            }
            if polished_energy < ledger.best
                && quench_is_sane(cfg, polished_energy, polished_state.view())
            {
                let validated = match grad.as_deref_mut() {
                    Some(gradient) => gradient(ledger, polished_state.view())
                        .is_some_and(|g| answer_gradient_ok(cfg, g.view(), polished_state.len())),
                    None => true,
                };
                if validated {
                    ledger.record(polished_energy, polished_state.view());
                    if improvements.len() < 512 {
                        improvements.push((hops, ledger.spent(), bias.n_basins(), polished_energy));
                    }
                } else {
                    unconverged_records += 1;
                }
            }
        }

        // Offered on a period rather than behind the energy stall the other
        // escapes use, because this mechanism carries its own trapping test and
        // that test is sharper. A stall detector says the chain has stopped
        // improving; the absorbing chain says the chain revisits its own states
        // ten times more than crossing them would need, which is the condition
        // being escaped. Stacking the two makes the move rare for a reason
        // unrelated to whether it applies, and this crate has one mechanism
        // already catalogued as inert rather than ineffective for that shape of
        // reason.
        if cfg.superbasin_escape && hops >= sb_last_jump + cfg.superbasin_period {
            sb_last_jump = hops;
            let from = *here.get_or_insert_with(|| identity.basin_of(x.view()));
            if let Some(sb) = superbasin.as_mut() {
                // Refusal is the normal outcome and is not a failure: the
                // algebra declines when the graph is too small, too well mixed,
                // or has no exit with a structure stored, and each of those is
                // a case where jumping would push the chain out of a region it
                // has not finished.
                if let Ok(j) = sb.propose(from, rng) {
                    quiet = 0;
                    longest_quiet = 0;
                    // No charged evaluations and no hop. The structure was
                    // quenched and recorded when the run first reached it, so
                    // the ledger has already paid for it, and counting a hop
                    // here would make the charged-per-hop figure describe a
                    // move that costs nothing.
                    if j.energy < e {
                        sb.observe_gain(e - j.energy);
                    }
                    adopt_minimum!(
                        j.energy,
                        j.state,
                        None,
                        Some(j.basin),
                        AdoptionHistory::Fresh,
                        None,
                        Some("superbasin-exit"),
                        hops
                    );
                }
            }
        }

        if n_rep > 1 {
            if !ladder_built && rep == 0 {
                // Welford over the cold rung's energies. This is the whole
                // input to the first ladder: a spread in the units of the
                // objective, rather than a multiple of the cold temperature
                // that means something different for every potential.
                pilot.0 += 1;
                let d = e - pilot.1;
                pilot.1 += d / pilot.0 as f64;
                pilot.2 += d * (e - pilot.1);
            }
            since_swap += 1;
            let slice = if ladder_built {
                cfg.swap_period
            } else {
                cfg.ladder_pilot.max(cfg.swap_period)
            };
            if since_swap >= slice {
                since_swap = 0;
                // Park the active rung, offer a swap with the next, then make
                // that one active. Each rung keeps its own bias and its own
                // temperature; only the states move, so a hot rung's crossing
                // lands in a cold rung that can polish it.
                chains.insert(
                    rep,
                    OccupiedMinimum {
                        energy: e,
                        coordinates: x.clone(),
                        gradient: current_validation_gradient.take(),
                        history_minimum: history_here,
                        local_minimum: here,
                        generation: occupancy_generation,
                        pending_initial: false,
                    },
                );
                // A placeholder only; the destination rung's own bias is taken
                // below, so this is never deposited into.
                biases.insert(
                    rep,
                    std::mem::replace(
                        &mut bias,
                        BasinBias::new(
                            ClusterFingerprint::of_config(cfg, &canonical_reference),
                            cfg.merge_radius,
                            cfg.bias_height,
                            cfg.bias_gamma,
                        ),
                    ),
                );
                parked_history_seen.insert(rep, std::mem::take(&mut history_seen));
                if let (Some(h), Some(hc)) = (hop.take(), cfg.hmc.as_ref()) {
                    // The adaptation stays with the rung, not the state; the
                    // destination rung's own sampler is taken below.
                    hop_parked.insert(rep.min(hop_parked.len()), h);
                    hop = Some(crate::hmc::hop::HopChain::new(hc));
                }
                if !ladder_built {
                    // The ladder the run's own energy scale implies, from the
                    // spread of the energies the cold chain visits.
                    let sigma = if pilot.0 > 1 {
                        (pilot.2 / (pilot.0 - 1) as f64).sqrt()
                    } else {
                        0.0
                    };
                    let built = crate::tempering::Ladder::from_fluctuation(
                        cfg.temperature,
                        sigma,
                        n_rep,
                        cfg.ladder_target_accept,
                        cfg.ladder_mode.scheme(),
                    );
                    temps = built.temperatures();
                    if cfg.bias_by_rung {
                        let top = temps[n_rep - 1];
                        for (k, b) in biases.iter_mut().enumerate() {
                            b.set_height(cfg.bias_height * temps[k] / top);
                        }
                    }
                    ladder = Some(built);
                    ladder_built = true;
                }
                if cfg.ladder_mode.sweeps() {
                    // Every rung is advanced by one slice before any pair is
                    // offered, so a sweep is a sweep.
                    rep += 1;
                    if rep >= n_rep {
                        rep = 0;
                        let l = ladder.as_mut().expect("a sweep mode builds a ladder");
                        let before = l.swap_counts();
                        let taken = {
                            let ch = &chains;
                            let bi = &biases;
                            let tp = &temps;
                            l.offer(&mut *rng, |k| {
                                let (ek, xk) = (ch[k].energy, ch[k].coordinates.view());
                                let (ej, xj) = (ch[k + 1].energy, ch[k + 1].coordinates.view());
                                let vk_xk = bi[k].potential(bi[k].cv(xk).view());
                                let vk_xj = bi[k].potential(bi[k].cv(xj).view());
                                let vj_xj = bi[k + 1].potential(bi[k + 1].cv(xj).view());
                                let vj_xk = bi[k + 1].potential(bi[k + 1].cv(xk).view());
                                let log_a = crate::tempering::biased_swap_log_ratio(
                                    tp[k],
                                    tp[k + 1],
                                    ek + vk_xk,
                                    ej + vk_xj,
                                    ej + vj_xj,
                                    ek + vj_xk,
                                );
                                if log_a >= 0.0 { 1.0 } else { log_a.exp() }
                            })
                        };
                        for k in taken {
                            chains.swap(k, k + 1);
                        }
                        let after = l.swap_counts();
                        swaps_tried += (after.0 - before.0) as usize;
                        swaps_accepted += (after.1 - before.1) as usize;
                        sweeps += 1;
                        if cfg.ladder_mode.adapts()
                            && sweeps.is_multiple_of(cfg.ladder_window.max(1))
                        {
                            adaptations += 1;
                            // The interior is moved by the barrier estimator;
                            // every third adaptation the endpoint controller
                            // gets a turn.
                            if adaptations.is_multiple_of(3) {
                                l.retune_top(1.0 - cfg.ladder_target_accept, 0.5);
                            } else {
                                l.equalise();
                            }
                            temps = l.temperatures();
                            if cfg.bias_by_rung {
                                let top = temps[n_rep - 1];
                                for (k, b) in biases.iter_mut().enumerate() {
                                    b.set_height(cfg.bias_height * temps[k] / top);
                                }
                            }
                        }
                    }
                } else {
                    let k = rep;
                    let j = (rep + 1) % n_rep;
                    if k != j && cfg.ladder_mode.exchanges() {
                        swaps_tried += 1;
                        // Bias exchange, not plain parallel tempering: each
                        // rung samples exp(-(E + V_k)/T_k), so the factor
                        // evaluates each rung's bias at both states (Piana and
                        // Laio), which is what biased_swap_log_ratio does.
                        let (ek, xk) = (chains[k].energy, chains[k].coordinates.clone());
                        let (ej, xj) = (chains[j].energy, chains[j].coordinates.clone());
                        let vk_xk = biases[k].potential(biases[k].cv(xk.view()).view());
                        let vk_xj = biases[k].potential(biases[k].cv(xj.view()).view());
                        let vj_xj = biases[j].potential(biases[j].cv(xj.view()).view());
                        let vj_xk = biases[j].potential(biases[j].cv(xk.view()).view());
                        let log_a = crate::tempering::biased_swap_log_ratio(
                            temps[k],
                            temps[j],
                            ek + vk_xk,
                            ej + vk_xj,
                            ej + vj_xj,
                            ek + vj_xk,
                        );
                        let p = if log_a >= 0.0 { 1.0 } else { log_a.exp() };
                        if rng.random::<f64>() < p {
                            swaps_accepted += 1;
                            chains.swap(k, j);
                            if let Some(t) = transport.as_mut() {
                                t.swap(k, j);
                            }
                        }
                        if let Some(t) = transport.as_mut() {
                            t.observe_ends();
                        }
                        // A sweep under this scheme is n_rep offers, one per
                        // rung, so round trips per sweep compare across schemes.
                        cyclic_offers += 1;
                        if cyclic_offers.is_multiple_of(n_rep) {
                            sweeps += 1;
                        }
                    }
                    rep = j;
                }
                history_seen = parked_history_seen.remove(rep);
                let mut next = chains.remove(rep);
                if next.pending_initial {
                    if let (Some(h), Some(g)) = (history.as_deref_mut(), next.gradient.as_ref())
                        && let Some(report) =
                            h.observe(next.energy, next.coordinates.view(), g.view())
                    {
                        h.mark_accepted(report.minimum);
                        next.history_minimum = Some(report.minimum);
                        let seen = history_seen.entry(report.minimum).or_insert(0);
                        *seen = seen.saturating_add(1);
                    }
                    next.pending_initial = false;
                }
                (
                    e,
                    x,
                    current_validation_gradient,
                    history_here,
                    occupancy_generation,
                    here,
                ) = next.into_live();
                bias = biases.remove(rep);
                if cfg.hmc.is_some() && rep < hop_parked.len() {
                    hop = Some(hop_parked.remove(rep));
                }
            }
        }

        if !hop_room(hops) {
            break HopStop::MaxHops;
        }

        if cfg.path_on_stall {
            // Diversity is judged on the descriptor, which is the same notion
            // of sameness the bias is keyed on, so an archive member is one the
            // bias would call a different basin.
            let (pe, px) = produced.expect("kept when path_on_stall is set");
            let d_new = bias.cv(px.view());
            let far = archive.iter().all(|(_, a)| {
                let da = bias.cv(a.view());
                da.iter()
                    .zip(d_new.iter())
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
                    .sqrt()
                    > 4.0 * cfg.merge_radius
            });
            if far && archive.len() < 32 {
                archive.push((pe, px));
            }
            if stall.observe(e) && archive.len() >= 2 {
                // The deepest member that is not where the chain already is.
                let target = archive
                    .iter()
                    .filter(|(_, a)| {
                        let da = bias.cv(a.view());
                        let dc = bias.cv(x.view());
                        da.iter()
                            .zip(dc.iter())
                            .map(|(p, q)| (p - q) * (p - q))
                            .sum::<f64>()
                            .sqrt()
                            > 4.0 * cfg.merge_radius
                    })
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(_, a)| a.clone());
                if let Some(t) = target {
                    paths_run += 1;
                    let start_cv = bias.cv(x.view());
                    let mut path_certificates = Vec::new();
                    let out = interpolate_path(
                        x.view(),
                        t.view(),
                        cfg.path_images,
                        |img| {
                            if ledger.remaining() == 0 {
                                return None;
                            }
                            let (ev, xv) = relax(ledger, img, cfg.relax_steps);
                            let certificate = record_quenched_answer(
                                cfg,
                                ledger,
                                &mut grad,
                                ev,
                                xv.view(),
                                &mut unconverged_records,
                            );
                            path_certificates.push((ev, xv.clone(), certificate));
                            Some((ev, xv))
                        },
                        |st| {
                            let d = bias.cv(st);
                            d.iter()
                                .zip(start_cv.iter())
                                .map(|(p, q)| (p - q) * (p - q))
                                .sum::<f64>()
                                .sqrt()
                                > cfg.merge_radius
                        },
                    );
                    // The deepest structure that actually left, not the deepest
                    // overall: the deepest is usually a relaxation back home.
                    if let Some(esc) = out.best_escape() {
                        path_escapes += 1;
                        if esc.energy < e {
                            path_improvements += 1;
                            path_gain += e - esc.energy;
                            let certificate = path_certificates.iter().rev().find_map(
                                |(energy, state, gradient)| {
                                    (*energy == esc.energy && *state == esc.state)
                                        .then(|| gradient.clone())
                                        .flatten()
                                },
                            );
                            adopt_minimum!(
                                esc.energy,
                                esc.state.clone(),
                                certificate,
                                None,
                                AdoptionHistory::Fresh,
                                None,
                                Some("path-escape"),
                                hops
                            );
                        }
                    }
                }
            }
        }
    };

    if let Some((energy, state)) = pending_cut
        && energy < ledger.best - 1e-6
        && quench_is_sane(cfg, energy, state.view())
    {
        let gradient = record_quenched_answer(
            cfg,
            ledger,
            &mut grad,
            energy,
            state.view(),
            &mut unconverged_records,
        );
        // The budget may already be spent, so the record tolerance cannot
        // be checked. The polished energy is still an improvement.
        if gradient.is_none() && energy < ledger.best {
            ledger.record(energy, state.view());
        }
        if energy <= ledger.best + 1e-12 && improvements.len() < 512 {
            improvements.push((hops, ledger.spent(), bias.n_basins(), energy));
        }
    }

    let n_basins = bias.n_basins();
    // Per-rung sampler diagnostics, with the active rung put back at its own
    // index so the report reads in ladder order.
    let hmc_diag: Vec<crate::hmc::hop::HopDiagnostics> = match hop {
        Some(active) => {
            let mut v: Vec<_> = hop_parked.iter().map(|h| h.diag.clone()).collect();
            v.insert(rep.min(v.len()), active.diag.clone());
            v
        }
        None => Vec::new(),
    };
    let final_radius = bias.merge_radius();
    if let Some(slot) = carried {
        // Handed back so the next chain inherits what this one learned.
        *slot = bias;
    }

    Ok(Outcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        final_state: Some(x.clone()),
        final_energy: e,
        accepted_transitions,
        hops,
        screened_out,
        basins: n_basins,
        charged: ledger.spent(),
        overlap_rejected: proposal_overlaps,
        returned,
        escape_scale: feedback.escape(),
        escape_threshold: feedback.threshold(),
        visit_counts: (feedback.n_same, feedback.n_known, feedback.n_new),
        soft_perturbs: soft_fired,
        soft_subspaces: soft_recomputes,
        soft_escapes,
        soft_crossed,
        improvements,
        quenched,
        angular: (angular_tried, angular_accepted, angular_ratio),
        contextual: (contextual.picks.clone(), contextual.forced),
        screen: (
            screen.decided,
            screen.relaxed,
            screen.explored,
            screen.observations(),
        ),
        hmc: hmc_diag,
        tabu: (tabu.len(), tabu_hits),
        funnel: funnel_split.as_ref().map(|p| {
            let (a, b) = p.sizes();
            (a, b, p.connectivity)
        }),
        symmetrised: (symmetrised, symmetry_gain),
        orbits: (orbits_completed, orbit_gain),
        continuous_symmetry: (continuous_symmetry_attempts, continuous_symmetry_gain),
        restarts,
        exchanges_refused,
        jumps,
        history_visits: (history_observations, history_new),
        shared_deposits,
        gossip_rounds,
        md_escape: (md_attempts, md_steps, md_failed),
        nve_unavailable,
        stop: Some(stop),
        history_seconds: history.as_deref().map_or(0.0, |h| h.cost().2),
        merge_radius: final_radius,
        mean_step: radius.mean_step(),
        stall_escapes,
        trail_escapes,
        stall_escape_gain,
        soft_lambda: if soft_escapes > 0 {
            soft_lambda / soft_escapes as f64
        } else {
            f64::NAN
        },
        rungs: {
            // The active rung is held outside the parked list, so it is put
            // back in place before reporting.
            let mut all: Vec<(f64, usize, f64)> = Vec::with_capacity(n_rep);
            let mut parked = biases.iter().map(|b| b.n_basins());
            let mut energies = chains.iter().map(|minimum| minimum.energy);
            for k in 0..n_rep {
                if k == rep {
                    all.push((temps[k], n_basins, e));
                } else {
                    all.push((
                        temps[k],
                        parked.next().unwrap_or(0),
                        energies.next().unwrap_or(f64::NAN),
                    ));
                }
            }
            all
        },
        swaps_tried,
        transport: match (&ladder, &transport) {
            (Some(l), _) => Some((l.index().round_trips(), l.sweeps(), l.barrier())),
            (None, Some(t)) => Some((
                t.round_trips(),
                sweeps,
                // The barrier a cyclic ladder implies from its one acceptance
                // rate; it does not resolve the profile per pair.
                (n_rep - 1) as f64 * (1.0 - swaps_accepted as f64 / swaps_tried.max(1) as f64),
            )),
            _ => None,
        },
        accepted,
        unconverged_records,
        delayed: surrogate.as_ref().map(|s| {
            (
                s.stage_one,
                s.stage_one_rejected,
                s.stage_two,
                s.stage_two_rejected,
            )
        }),
        arms: kernels
            .iter()
            .enumerate()
            .map(|(i, k)| (k.name(), arm_draws[i], arm_accepts[i], arm_best[i]))
            .collect(),
        proposal_counts: arm_counts,
        swaps_accepted,
        paths: paths_run,
        path_escapes,
        path_improvements,
        path_gain,
        energy_trace: trace,
        superbasin_counts: superbasin
            .as_ref()
            .filter(|_| cfg.superbasin_quotient)
            .map(|sb| sb.counts.clone()),
        superbasin_archive: superbasin
            .as_ref()
            .filter(|_| cfg.superbasin_quotient)
            .map(|sb| sb.archive_entries()),
        superbasin: superbasin.as_ref().map(|sb| {
            let mut r = sb.report();
            if cfg.superbasin_quotient {
                #[cfg(feature = "ira")]
                {
                    // Zero for a structure against its own relabelling and
                    // rotation, order one between different minima: measured
                    // 2.7e-16 for a relabelled copy at 13 points and 2.9e-16 at
                    // 38, against 1.58 for a different basin. The threshold sits
                    // in a gap of fifteen orders of magnitude, and the report
                    // carries the largest accepted and smallest rejected
                    // distance so it can be checked rather than trusted.
                    let metric = crate::shape::IraMetric::default();
                    // The energy filter is wide on purpose. An orbit is a
                    // level set of the energy for exact minima, but the archive
                    // holds accepted chain states and 181 of 12287 relaxations
                    // reach a gradient of 1e-3 within the step cap, so two
                    // members of one orbit can sit further apart in energy than
                    // a converged pair would. 1e-2 is fifty times below the
                    // 0.5 spacing between distinct minima on this landscape, so
                    // it cannot miss a true pair, and the shape distance does
                    // the discriminating.
                    r.quotient = Some(sb.quotient(|a, b| metric.distance(a, b), 1e-3, 1e-2, 16));
                }
                #[cfg(not(feature = "ira"))]
                {
                    // Without a shape distance the only usable test is exact
                    // energy degeneracy, which merges accidental degeneracies
                    // along with real orbits. Refused rather than reported as
                    // if it were the same measurement.
                    r.quotient = None;
                }
            }
            if cfg.superbasin_features {
                // Polyhedral template fractions, the same descriptor the
                // benchmark reports a run's morphology with, so a separability
                // measured here and a morphology quoted there mean the same
                // thing.
                r.separability = sb.separability(|st| crate::structure::ptm_fractions(st, n, 0.12));
            }
            r
        }),
        energy_bias: ebias,
    })
}

/// Seeds a non-overlapping configuration at liquid-like density.
///
/// Uniform draws over a container overlap almost surely at the sizes of
/// interest, and a relaxation cannot recover from that.
pub fn random_cluster<R: Rng + ?Sized>(
    n: usize,
    density: f64,
    min_sep: f64,
    rng: &mut R,
) -> Array1<f64> {
    let radius = (3.0 * n as f64 / (4.0 * std::f64::consts::PI * density)).cbrt();
    random_cluster_in_radius(n, radius, min_sep, rng)
}

/// Seeds a non-overlapping configuration inside a declared sphere radius.
/// Re-place rigid groups on a new sphere, keeping each group's internals.
///
/// The shell dimensions are dimensionless molecular-preset coefficients
/// multiplied by the caller's declared `length_scale`.
pub fn repack_rigid_groups<R: Rng + ?Sized>(
    template: ArrayView1<f64>,
    groups: &[Vec<usize>],
    length_scale: f64,
    rng: &mut R,
) -> Array1<f64> {
    assert!(
        length_scale.is_finite() && length_scale > 0.0,
        "length_scale must be finite and positive"
    );
    let mut y = template.to_owned();
    let r0 = (preset::MolecularPreset::REPACK_RADIUS
        + rng.random::<f64>() * preset::MolecularPreset::REPACK_RADIAL_JITTER)
        * length_scale;
    for (g, atoms) in groups.iter().enumerate() {
        if atoms.is_empty() {
            continue;
        }
        let r = r0 + (g as f64) * preset::MolecularPreset::REPACK_GROUP_SPACING * length_scale;
        let th = rng.random::<f64>() * std::f64::consts::TAU;
        let ct = 2.0 * rng.random::<f64>() - 1.0;
        let st = (1.0 - ct * ct).sqrt();
        let new_c = [r * st * th.cos(), r * st * th.sin(), r * ct];
        let n = atoms.len() as f64;
        let mut com = [0.0; 3];
        for &i in atoms {
            if 3 * i + 2 < y.len() {
                for d in 0..3 {
                    com[d] += y[3 * i + d];
                }
            }
        }
        for d in 0..3 {
            com[d] /= n;
        }
        for &i in atoms {
            if 3 * i + 2 < y.len() {
                for d in 0..3 {
                    y[3 * i + d] += new_c[d] - com[d];
                }
            }
        }
    }
    y
}

fn hop_is_identity(x: ArrayView1<f64>, y: ArrayView1<f64>) -> bool {
    if x.len() != y.len() || x.is_empty() {
        return false;
    }
    let n = (x.len() / 3).max(1) as f64;
    let mut s = 0.0;
    for i in 0..x.len() {
        let d = y[i] - x[i];
        s += d * d;
    }
    (s / n).sqrt() < 1e-8
}

/// Samples a non-overlapping cluster uniformly inside a sphere.
pub fn random_cluster_in_radius<R: Rng + ?Sized>(
    n: usize,
    radius: f64,
    min_sep: f64,
    rng: &mut R,
) -> Array1<f64> {
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut tries = 0;
    while pts.len() < n && tries < 20_000 {
        tries += 1;
        let mut v = [0.0; 3];
        let mut norm = 0.0;
        for k in 0..3 {
            // Box-Muller from two uniforms, avoiding a distribution dependency.
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            v[k] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            norm += v[k] * v[k];
        }
        let norm = norm.sqrt().max(1e-12);
        let r = radius * rng.random::<f64>().cbrt();
        let p = [v[0] / norm * r, v[1] / norm * r, v[2] / norm * r];
        if pts.iter().all(|q| {
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
                >= min_sep
        }) {
            pts.push(p);
        }
    }
    let mut out = Array1::zeros(3 * pts.len());
    for (i, p) in pts.iter().enumerate() {
        for k in 0..3 {
            out[3 * i + k] = p[k];
        }
    }
    out
}

/// Convenience entry point seeding its own start.
pub fn optimize(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    seed: u64,
) -> Result<Outcome, SearchError> {
    optimize_with_gradient(cfg, ledger, relax, None, seed)
}

/// As [`optimize_with_gradient`], with a settle stage for the staged quench.
pub fn optimize_with_settle<'g>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    settle: Option<Settle<'_>>,
    seed: u64,
) -> Result<Outcome, SearchError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        cfg.n_points,
        cfg.start_radius(),
        cfg.min_separation,
        &mut rng,
    );
    run_with_gradient_settle(cfg, start.view(), ledger, relax, grad, settle, &mut rng)
}

/// As [`optimize`], with a gradient for the soft-mode escape.
pub fn optimize_with_gradient<'g>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    seed: u64,
) -> Result<Outcome, SearchError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        cfg.n_points,
        cfg.start_radius(),
        cfg.min_separation,
        &mut rng,
    );
    run_with_gradient(cfg, start.view(), ledger, relax, grad, &mut rng)
}

/// As [`optimize_with_gradient`], with value and gradient together for
/// [`Config::hmc`].
pub fn optimize_with_energy_gradient<'g>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    energy_grad: Option<&mut EnergyGradFn<'g>>,
    seed: u64,
) -> Result<Outcome, SearchError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    run_with_energy_gradient(
        cfg,
        start.view(),
        ledger,
        relax,
        grad,
        energy_grad,
        &mut rng,
    )
}

#[cfg(test)]
mod bond_matrix_tests {
    use super::*;

    /// One length cannot serve hydrogen and copper together; the radii-sum
    /// rule must bond H-H at 0.75, Cu-Cu at 2.5 and keep a 2.0 separation
    /// between two hydrogens unbonded, all under one tolerance.
    #[test]
    fn radii_sums_bond_what_a_single_length_cannot() {
        // H2 at 0.75, a Cu pair at 2.5, far apart; and a stray H at 2.0 from
        // the molecule.
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 10.0, 0.0, 0.0, 12.5, 0.0, 0.0, 2.75, 0.0, 0.0,
        ]);
        let z = [1u32, 1, 29, 29, 1];
        let g = connectivity_groups_z(x.view(), &z, 1.25);
        assert_eq!(g, vec![vec![0, 1], vec![2, 3], vec![4]]);
        // A flat cutoff wide enough for the copper bond swallows the stray
        // hydrogen into the molecule: the failure the species rule removes.
        let flat = connectivity_groups(x.view(), 5, 2.6);
        assert_ne!(flat, g);
    }
}

#[cfg(test)]
mod connectivity_tests {
    use super::*;

    /// Two intact triatomics read as two groups; after an atom migrates to
    /// bond with the other molecule, the groups must follow the new bond
    /// graph. This is the defect that stranded a reacted walker: moves kept
    /// the declared molecules while the structure had different ones.
    #[test]
    fn groups_follow_the_bond_graph() {
        // O at origin with two H at 1.0; second molecule 4.0 away.
        let intact = Array1::from(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.2, 0.0,
            0.0,
        ]);
        let g = connectivity_groups(intact.view(), 6, 1.3);
        assert_eq!(g, vec![vec![0, 1, 2], vec![3, 4, 5]]);
        // Atom 1 migrates next to atom 3: the bond graph now joins it to the
        // second molecule, leaving the first as a diatomic.
        let mut reacted = intact.clone();
        reacted[3] = 3.6;
        let g2 = connectivity_groups(reacted.view(), 6, 1.3);
        assert_eq!(g2, vec![vec![0, 2], vec![1, 3, 4, 5]]);
    }
}

#[cfg(test)]
mod group_move_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Six rigid three-atom groups in a blob.
    fn waterish() -> (Array1<f64>, Vec<Vec<usize>>) {
        let mut x = Array1::zeros(3 * 18);
        let mut groups = Vec::new();
        for g in 0..6 {
            let base = [
                (g % 3) as f64 * 2.0,
                (g / 3) as f64 * 2.0,
                (g % 2) as f64 * 1.5,
            ];
            let local = [[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]];
            let mut idx = Vec::new();
            for (a, l) in local.iter().enumerate() {
                let i = 3 * g + a;
                idx.push(i);
                for k in 0..3 {
                    x[3 * i + k] = base[k] + l[k];
                }
            }
            groups.push(idx);
        }
        (x, groups)
    }

    /// The move must preserve every intra-group distance exactly and move
    /// exactly one group.
    #[test]
    fn the_group_moves_rigidly_and_alone() {
        let (x, groups) = waterish();
        let mut rng = StdRng::seed_from_u64(3);
        let y = group_relocate(x.view(), &groups, 1.6, &mut rng);
        let d = |v: &Array1<f64>, a: usize, b: usize| -> f64 {
            (0..3)
                .map(|k| (v[3 * a + k] - v[3 * b + k]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mut moved_groups = 0;
        for atoms in &groups {
            let moved = atoms
                .iter()
                .any(|&a| (0..3).any(|k| (x[3 * a + k] - y[3 * a + k]).abs() > 1e-9));
            if moved {
                moved_groups += 1;
            }
            for i in 0..atoms.len() {
                for j in (i + 1)..atoms.len() {
                    assert!(
                        (d(&x, atoms[i], atoms[j]) - d(&y, atoms[i], atoms[j])).abs() < 1e-9,
                        "intra-group distance changed"
                    );
                }
            }
        }
        assert_eq!(moved_groups, 1, "moved {moved_groups} groups, wanted 1");
    }

    /// A majority-ungrouped frame is a slab: the adsorbate is placed
    /// above the substrate, not on a sphere about the all-atom centroid
    /// (which sits inside the metal).
    #[test]
    fn group_relocate_on_a_slab_stays_above_the_substrate() {
        let n_cu = 8usize;
        let n_h = 2usize;
        let n = n_cu + n_h;
        let mut x = Array1::zeros(3 * n);
        for i in 0..n_cu {
            x[3 * i] = (i % 4) as f64;
            x[3 * i + 1] = (i / 4) as f64;
            x[3 * i + 2] = 0.0;
        }
        x[3 * n_cu] = 0.5;
        x[3 * n_cu + 1] = 0.5;
        x[3 * n_cu + 2] = 2.3;
        x[3 * (n_cu + 1)] = 2.5;
        x[3 * (n_cu + 1) + 1] = 0.5;
        x[3 * (n_cu + 1) + 2] = 2.3;
        let groups = vec![vec![n_cu], vec![n_cu + 1]];
        let mut rng = StdRng::seed_from_u64(3);
        let y = group_relocate(x.view(), &groups, 1.6, &mut rng);
        for i in 0..n_cu {
            for k in 0..3 {
                assert_eq!(y[3 * i + k], x[3 * i + k], "substrate atom {i} moved");
            }
        }
        let z_top = (0..n_cu)
            .map(|i| x[3 * i + 2])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut moved_h = 0;
        for i in n_cu..n {
            let dz = y[3 * i + 2] - z_top;
            assert!(
                dz >= 1.5,
                "hydrogen {i} at z={} is not above the substrate (z_top={z_top})",
                y[3 * i + 2]
            );
            if (0..3).any(|k| (y[3 * i + k] - x[3 * i + k]).abs() > 1e-9) {
                moved_h += 1;
            }
        }
        assert_eq!(moved_h, 1, "moved {moved_h} hydrogens, wanted 1");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_molecular_water_search_is_not_a_binary_homotop() {
        let species = crate::catalog::molecular::water_species(4).expect("species");
        let groups = crate::catalog::molecular::water_groups(4).expect("groups");
        let mut water = Config::recommended_molecular(species, groups.clone(), 1.0);
        water.move_library = MoveLibrary::Molecular {
            groups,
            reactive: false,
            reverse_cycles: true,
        };
        assert!(!polishes_unlike_pairs(&water));
        assert!(!polishes_unlike_pairs(&Config::for_tip4p(4)));
        let binary = Config::recommended(6).with_species(vec![29, 29, 29, 79, 79, 79], None);
        assert!(polishes_unlike_pairs(&binary));
        assert!(!polishes_unlike_pairs(&Config::recommended(6)));
    }

    #[test]
    fn an_arm_count_separates_proposals_from_accepts_and_improvements() {
        let mut arms = Vec::new();
        begin_arm(&mut arms, "gshake");
        finish_arm(&mut arms, "gshake", false, false);
        begin_arm(&mut arms, "gburst");
        finish_arm(&mut arms, "gburst", true, true);
        begin_arm(&mut arms, "gshake");
        finish_arm(&mut arms, "gshake", true, false);
        assert_eq!(arms[0].move_type, "gshake");
        assert_eq!(arms[0].proposed, 2);
        assert_eq!(arms[0].accepted, 1);
        assert_eq!(arms[0].improved, 0);
        assert_eq!(arms[1].proposed, 1);
        assert_eq!(arms[1].accepted, 1);
        assert_eq!(arms[1].improved, 1);
    }

    /// Uniform selection is one `random_range` over the preset's arms, and
    /// the untouched preset still draws from the depth posterior.
    #[test]
    fn uniform_arm_selection_matches_a_uniform_draw() {
        let preset = Config::recommended(38);
        assert_eq!(preset.arm_selection, ArmSelection::Posterior);
        assert!(preset.depth_reward && preset.allocate_moves);
        let n_arms = search_kernels(&preset).len();
        assert!(n_arms >= 2, "the paper preset needs more than one arm");
        let mut depth = crate::allocate::DepthAllocator::new(n_arms);
        for _ in 0..40 {
            depth.update(0, 8.0);
            for arm in 1..n_arms {
                depth.update(arm, -8.0);
            }
        }
        let mut thompson = FlooredThompson::new(n_arms);
        let mut contextual = ContextualAllocator::new(n_arms, 3, preset.contextual_floor);
        let mut uniform_cfg = preset.clone();
        uniform_cfg.arm_selection = ArmSelection::Uniform;
        assert!(uniform_cfg.depth_reward && uniform_cfg.allocate_moves);
        let mut rng = StdRng::seed_from_u64(20261111);
        let mut twin = StdRng::seed_from_u64(20261111);
        let mut counts = vec![0usize; n_arms];
        for _ in 0..4_000 {
            let got = choose_arm(
                &uniform_cfg,
                None,
                &mut contextual,
                &depth,
                &mut thompson,
                n_arms,
                &mut rng,
            );
            let expect = twin.random_range(0..n_arms);
            assert_eq!(got, expect);
            counts[got] += 1;
        }
        let low = 4_000 / n_arms / 2;
        let high = 4_000 / n_arms + 4_000 / n_arms / 2;
        for count in &counts {
            assert!(
                *count > low && *count < high,
                "uniform draws were {counts:?}, expected each arm near {}",
                4_000 / n_arms
            );
        }
        let mut posterior_rng = StdRng::seed_from_u64(7);
        let mut direct_rng = posterior_rng.clone();
        let mut posterior_counts = vec![0usize; n_arms];
        for _ in 0..400 {
            let via_preset = choose_arm(
                &preset,
                None,
                &mut contextual,
                &depth,
                &mut thompson,
                n_arms,
                &mut posterior_rng,
            );
            let via_posterior = depth.select(&mut direct_rng);
            assert_eq!(via_preset, via_posterior);
            posterior_counts[via_preset] += 1;
        }
        assert!(
            posterior_counts[0] > 300,
            "the depth posterior did not keep the deep arm: {posterior_counts:?}"
        );
        let mut uniform_seq = Vec::new();
        let mut posterior_seq = Vec::new();
        let mut left = StdRng::seed_from_u64(99);
        let mut right = StdRng::seed_from_u64(99);
        for _ in 0..64 {
            uniform_seq.push(choose_arm(
                &uniform_cfg,
                None,
                &mut contextual,
                &depth,
                &mut thompson,
                n_arms,
                &mut left,
            ));
            posterior_seq.push(depth.select(&mut right));
        }
        assert_ne!(uniform_seq, posterior_seq);
    }

    #[test]
    fn recommended_offers_the_energy_lattice_and_the_baseline_does_not() {
        let rec = search_kernels(&Config::recommended(104));
        assert!(
            rec.iter().any(|kernel| kernel.name() == "dls-kernel"),
            "recommended kernels miss the DLS-style exchange kernel: {:?}",
            rec.iter().map(|kernel| kernel.name()).collect::<Vec<_>>()
        );
        let base = search_kernels(&Config::for_cluster(104));
        assert!(base.iter().all(|kernel| kernel.name() != "dls-kernel"));
        let pure = search_kernels(&Config::recommended(8).with_species(vec![79; 8], None));
        assert!(
            pure.iter().any(|kernel| kernel.name() == "dls-kernel"),
            "one repeated label lost the unlabelled lattice exchange"
        );
        let mixed = search_kernels(
            &Config::recommended(8).with_species(vec![29, 29, 29, 29, 79, 79, 79, 79], None),
        );
        assert!(mixed.iter().all(|kernel| kernel.name() != "dls-kernel"));
    }

    #[test]
    fn repack_rigid_groups_keeps_internal_distances() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut x = Array1::zeros(9);
        x[0] = 0.0;
        x[1] = 0.0;
        x[2] = 0.0;
        x[3] = 0.96;
        x[4] = 0.0;
        x[5] = 0.0;
        x[6] = -0.24;
        x[7] = 0.93;
        x[8] = 0.0;
        let groups = vec![vec![0, 1, 2]];
        let y = repack_rigid_groups(x.view(), &groups, 1.0, &mut rng);
        let d = |a: &Array1<f64>, i: usize, j: usize| {
            let mut s = 0.0;
            for k in 0..3 {
                let t = a[3 * i + k] - a[3 * j + k];
                s += t * t;
            }
            s.sqrt()
        };
        assert!((d(&x, 0, 1) - d(&y, 0, 1)).abs() < 1e-12);
        assert!((d(&x, 0, 2) - d(&y, 0, 2)).abs() < 1e-12);
        assert!((d(&x, 1, 2) - d(&y, 1, 2)).abs() < 1e-12);
        let moved = (0..3).any(|k| (y[k] - x[k]).abs() > 1e-9);
        assert!(moved, "repack left the group on its original centre");
    }

    #[test]
    fn a_failed_stall_growth_waits_until_the_incumbent_drops() {
        assert!(!stall_growth_waits(None, -10.0));
        assert!(stall_growth_waits(Some(-10.0), -10.0));
        assert!(stall_growth_waits(Some(-10.0), -10.0 - 1e-9));
        assert!(!stall_growth_waits(Some(-10.0), -10.0 - 1e-8));
    }

    #[test]
    fn derived_replaces_the_two_hand_set_scalars() {
        let rec = Config::recommended(13);
        let der = Config::derived(13);
        assert!(matches!(der.move_library, MoveLibrary::LeanBurst));
        assert!(der.allocate_moves && der.depth_reward && der.tabu_on_stall);
        assert!(der.escape_on_stall && der.trail_on_stall && !der.restart_on_stall);
        assert!(rec.escape_on_stall && rec.trail_on_stall && !rec.restart_on_stall);
        assert!(rec.prior_soap && rec.prior_reseed && rec.prior_growth);
        assert!(rec.prior_core && rec.prior_motif && !rec.general_collective);
        assert!(der.budget_window);
        assert!(der.bayes_screen);
        assert!(!rec.budget_window);
        assert!(rec.bayes_screen);
        assert!(matches!(
            Config::recommended(38).move_library,
            MoveLibrary::LeanBurst
        ));
        assert!(matches!(
            Config::recommended(75).move_library,
            MoveLibrary::LeanBurst
        ));
        let want = crate::screen::cost_asymmetric_threshold(der.screen_steps, der.relax_steps);
        assert!((der.bayes_threshold - want).abs() < 1e-12);
        assert!((der.bayes_threshold - 7.0 / 15.0).abs() < 1e-12);
    }

    #[test]
    fn communicating_is_orbit_without_depth_reward() {
        let rec = Config::recommended(75);
        let comm = Config::communicating(75);
        assert!(rec.depth_reward);
        assert!(rec.orbit_complete_on_new);
        assert_eq!(rec.shared_visit_policy, SharedVisitPolicy::Tabu);
        assert!(!comm.depth_reward);
        assert!(comm.orbit_complete_on_new);
        assert_eq!(comm.shared_deposits, 0);
        assert_eq!(comm.shared_visit_policy, SharedVisitPolicy::Recognition);
        assert!(comm.allocate_moves && comm.return_screen);
        assert!(!comm.jump_on_stall);
        assert!(!comm.tabu_on_stall);
        assert!(!matches!(comm.move_library, MoveLibrary::LeanBurst));
        assert!(matches!(rec.move_library, MoveLibrary::LeanBurst));
        assert_eq!(comm.surfaces.len(), 1);
        assert!(comm.surfaces[0].is_active());
        assert_eq!(rec.surfaces, comm.surfaces);
        assert!(Config::communicating(98).surfaces.is_empty());
        assert!(matches!(
            rec.continuous_symmetry,
            ContinuousSymmetry::Measure {
                interval: 10,
                reseed_hops: 5
            }
        ));
        assert_eq!(rec.continuous_symmetry, comm.continuous_symmetry);
        assert!(matches!(
            Config::for_cluster(75).continuous_symmetry,
            ContinuousSymmetry::Off
        ));
        let rigid =
            Config::recommended(4).with_rigid_bodies(crate::potentials::gmin_water_geometry());
        assert!(matches!(rigid.continuous_symmetry, ContinuousSymmetry::Off));
    }

    #[test]
    fn a_rigid_chain_has_no_restart_among_its_stall_arms() {
        let mut atomic = Config::recommended(6);
        atomic.restart_on_stall = true;
        assert_eq!(stall_arms(&atomic), ["trail", "climb", "restart"]);
        let rigid = atomic.with_rigid_bodies(crate::potentials::gmin_water_geometry());
        assert!(rigid.restart_on_stall);
        assert!(!stall_arms(&rigid).contains(&"restart"));
    }

    #[test]
    fn packing_surface_names_twin_and_orbit_together() {
        let mut cfg = Config::for_cluster(13);
        assert!(!cfg.packing_surface().orbit_on_new);
        assert!(!cfg.packing_surface().twin_as_move);
        cfg.orbit_complete_on_new = true;
        cfg.move_library = MoveLibrary::Twin;
        let surface = cfg.packing_surface();
        assert!(surface.orbit_on_new);
        assert!(surface.twin_as_move);
    }

    #[test]
    fn the_dual_orbit_core_is_opt_in() {
        let plain = Config::for_cluster(13);
        assert_eq!(plain.orbit_core, OrbitCore::Fraction);
        assert!(!plain.resolved_json().unwrap().contains("orbit_core"));
        let dual = Config::for_cluster_dual_core(13);
        assert!(dual.packing_surface().orbit_on_new);
        assert_eq!(
            dual.orbit_core,
            OrbitCore::Both(crate::symmetrise::ShellCore::scaled(dual.length_scale))
        );
        assert!(dual.restart_on_stall);
        assert!(!dual.escape_on_stall);
        assert_eq!(dual.restart_patience, 4_000);
        assert!(
            dual.resolved_json()
                .unwrap()
                .contains(r#""orbit_core":{"both":"#)
        );
        assert_eq!(
            dual.resolved_json().unwrap(),
            Config::for_cluster(13)
                .with_dual_orbit_core()
                .resolved_json()
                .unwrap()
        );
        let rigid = dual.with_rigid_bodies(crate::potentials::gmin_water_geometry());
        assert!(!rigid.orbit_complete_on_new);
        assert_eq!(rigid.orbit_core, OrbitCore::Fraction);
    }

    /// The claim the bank rests on: a bias handed to one chain and then to the
    /// next carries what the first one learned. Without this each chain starts
    /// from an empty bias, and at 75 points the crossing takes on the order of
    /// a hundred thousand hops of accumulation to reach.
    #[test]
    fn a_supplied_bias_survives_the_run_that_used_it() {
        let cfg = Config::for_cluster(8);
        let mut bias = BasinBias::new(
            ClusterFingerprint::for_keying(8, false),
            cfg.merge_radius,
            cfg.bias_height,
            cfg.bias_gamma,
        );
        let mut rng = StdRng::seed_from_u64(4);
        let start = random_cluster(8, 0.7, cfg.min_separation, &mut rng);

        let mut l1 = Ledger::new(3_000);
        let mut r1 = |led: &mut Ledger, x: ArrayView1<f64>, n: usize| toy_relax(led, x, n);
        run_with_bias(
            &cfg,
            start.view(),
            &mut l1,
            &mut r1,
            None,
            &mut bias,
            &mut rng,
        )
        .expect("finite start");
        let after_first = bias.n_basins();
        assert!(after_first > 0, "the first chain deposited nothing");

        let mut l2 = Ledger::new(3_000);
        let mut r2 = |led: &mut Ledger, x: ArrayView1<f64>, n: usize| toy_relax(led, x, n);
        let out = run_with_bias(
            &cfg,
            start.view(),
            &mut l2,
            &mut r2,
            None,
            &mut bias,
            &mut rng,
        )
        .expect("finite start");
        assert!(
            out.basins >= after_first,
            "the second chain saw {} basins where the first left {after_first}",
            out.basins
        );
        assert!(
            bias.n_basins() >= after_first,
            "the bias came back smaller than it went in"
        );
    }

    /// The angular move has to actually be taken, and the criterion has to
    /// fire on a cluster that has a loose point. Both were wrong once: the
    /// proposal branch was absent so the flag was computed and discarded, and
    /// the attempt counter was never incremented so the acceptance rate read as
    /// `accepted / 1` and drove the ratio to its floor.
    #[test]
    fn the_angular_move_relocates_the_worst_bound_point() {
        // Twelve points on an icosahedron and one thrown far out.
        //
        // The geometry has to be relaxed, not merely spread out. Pair energy is
        // highest for an overlapping point, not a distant one, so a fixture
        // with two points on top of each other makes the criterion pick the
        // overlap, which is the right answer to the wrong question.
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        // Edge length 2 before scaling; 0.56 puts neighbours near the
        // Lennard-Jones minimum.
        let sc = 0.56;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, phi],
            [0.0, -1.0, -phi],
            [1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, -1.0],
        ];
        let n = 13;
        let mut x = Array1::<f64>::zeros(3 * n);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = sc * v[k];
            }
        }
        let last = n - 1;
        x[3 * last] = 9.0;

        let e = pair_energies(x.view(), n);
        let hi = (0..n)
            .max_by(|a, b| e[*a].partial_cmp(&e[*b]).unwrap())
            .unwrap();
        assert_eq!(hi, last, "the distant point should be the worst bound");
        assert_eq!(
            worst_bound(x.view(), n, 0.42),
            Some(last),
            "the criterion should fire on a point this loose"
        );

        let mut rng = StdRng::seed_from_u64(9);
        let y = ClusterMove::Angular {
            n_points: n,
            length_scale: LennardJonesPreset::REDUCED_SCALE,
            energy_scale: LennardJonesPreset::REDUCED_SCALE,
        }
        .propose(x.view(), 0.8, &mut rng);
        // Every other point is untouched: "with all other atoms fixed".
        for i in 0..last {
            for k in 0..3 {
                assert!(
                    (y[3 * i + k] - x[3 * i + k]).abs() < 1e-12,
                    "point {i} moved"
                );
            }
        }
        assert!(
            (0..3).any(|k| (y[3 * last + k] - x[3 * last + k]).abs() > 1e-9),
            "the worst-bound point did not move"
        );
        // It lands at the largest radius in the cluster, about the centre of
        // mass of the structure it was given.
        let mut c = [0.0_f64; 3];
        for i in 0..n {
            for k in 0..3 {
                c[k] += x[3 * i + k];
            }
        }
        for v in c.iter_mut() {
            *v /= n as f64;
        }
        let rmax = (0..n)
            .map(|i| {
                ((x[3 * i] - c[0]).powi(2)
                    + (x[3 * i + 1] - c[1]).powi(2)
                    + (x[3 * i + 2] - c[2]).powi(2))
                .sqrt()
            })
            .fold(0.0_f64, f64::max);
        let rnew = ((y[3 * last] - c[0]).powi(2)
            + (y[3 * last + 1] - c[1]).powi(2)
            + (y[3 * last + 2] - c[2]).powi(2))
        .sqrt();
        assert!(
            (rnew - rmax).abs() < 1e-9,
            "landed at radius {rnew} where the cluster's largest is {rmax}"
        );
    }

    /// The ratio has to settle where the acceptance target is met, not run to
    /// a bound. Driven by a process whose acceptance falls as the ratio rises,
    /// which is the coupling the real criterion has.
    #[test]
    fn the_angular_ratio_settles_rather_than_running_away() {
        let target = 0.5_f64;
        let mut ratio = 0.42_f64;
        let mut tried = 0usize;
        // Acceptance probability 1 - r, so the fixed point is r = 0.5.
        let mut seed = 12345u64;
        for _ in 0..20_000 {
            tried += 1;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((seed >> 33) as f64) / ((1u64 << 31) as f64);
            let accept = u < (1.0 - ratio);
            let hit = if accept { 1.0 } else { 0.0 };
            let step = 0.02 / (1.0 + tried as f64 / 500.0).sqrt();
            ratio = (ratio + step * (hit - target)).clamp(0.05, 0.95);
        }
        assert!(
            (0.35..0.65).contains(&ratio),
            "the ratio settled at {ratio}, not near the fixed point of 0.5"
        );
    }

    /// A compact cluster has no loose point, so the criterion must stay quiet.
    #[test]
    fn the_criterion_does_not_fire_on_an_even_cluster() {
        let n = 13;
        let mut x = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            let a = i as f64 * 0.48;
            x[3 * i] = 1.1 * a.cos();
            x[3 * i + 1] = 1.1 * a.sin();
            x[3 * i + 2] = 0.2 * (i % 4) as f64;
        }
        assert_eq!(worst_bound(x.view(), n, 0.05), None);
    }

    /// A positive `return_polish` finishes a returning trial for the whole hop,
    /// including while more than half of that hop's ledger remains. The first
    /// hop of a two-phase search opts out by setting the field to zero; the
    /// gate is that assignment, not a spent-versus-remaining test inside the
    /// driver.
    #[test]
    fn return_polish_fires_while_first_half_of_ledger_remains() {
        let mut cfg = Config::recommended(7);
        cfg.return_screen = true;
        cfg.return_polish = 8;
        cfg.screen_steps = 6;
        cfg.relax_steps = 16;
        // Dummy does not walk a trial home, so the return screen sees the
        // move where the kernel left it. A wide radius makes those trials
        // returning, which is the case this gate is about.
        cfg.merge_radius = 1.0e3;

        let mut rng = StdRng::seed_from_u64(1);
        let start = random_cluster(7, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(400);
        let polish = cfg.return_polish;
        let mut first_half_polish = 0usize;
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
            if steps == polish && led.spent() < led.remaining() {
                first_half_polish += 1;
            }
            for _ in 0..steps {
                if !led.charge() {
                    break;
                }
            }
            let e = x.iter().map(|v| v * v).sum::<f64>();
            (e, x.to_owned())
        };
        let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
        assert!(
            out.returned >= 1,
            "expected returning trials, got {}",
            out.returned
        );
        assert!(
            first_half_polish >= 1,
            "return_polish={polish} never fired while spent < remaining; returned {}",
            out.returned
        );
    }

    /// A positive `return_polish_after` keeps the early hop as skip-return.
    #[test]
    fn return_polish_after_skips_until_the_threshold() {
        let mut cfg = Config::recommended(7);
        cfg.return_screen = true;
        cfg.return_polish = 8;
        cfg.return_polish_after = 200;
        cfg.screen_steps = 6;
        cfg.relax_steps = 16;
        cfg.merge_radius = 1.0e3;

        let mut rng = StdRng::seed_from_u64(1);
        let start = random_cluster(7, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(400);
        let polish = cfg.return_polish;
        let after = cfg.return_polish_after;
        let mut before = 0usize;
        let mut after_n = 0usize;
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
            if steps == polish {
                if led.spent() < after {
                    before += 1;
                } else {
                    after_n += 1;
                }
            }
            for _ in 0..steps {
                if !led.charge() {
                    break;
                }
            }
            let e = x.iter().map(|v| v * v).sum::<f64>();
            (e, x.to_owned())
        };
        let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
        assert!(
            out.returned >= 1,
            "expected returning trials, got {}",
            out.returned
        );
        assert_eq!(before, 0, "polished {before} times before {after}");
        assert!(
            after_n >= 1,
            "return_polish never fired after {after}; returned {}",
            out.returned
        );
    }

    /// A separable quadratic in the point coordinates: its minimum is every
    /// point at the origin, so a relaxation is a step toward zero. Enough to
    /// exercise the driver's accounting and control flow without a potential.
    fn toy_relax(ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
        let mut cur = x.to_owned();
        for _ in 0..steps {
            if !ledger.charge() {
                break;
            }
            cur.mapv_inplace(|v| v * 0.85);
        }
        let e = cur.iter().map(|v| v * v).sum::<f64>();
        (e, cur)
    }

    /// One well at a fixed tetrahedron; every relaxation converges onto it
    /// and certifies the result with the analytic gradient.
    fn well_relax(
        target: &Array1<f64>,
        ledger: &mut Ledger,
        x: ArrayView1<f64>,
        steps: usize,
    ) -> (f64, Array1<f64>) {
        let before = ledger.spent();
        let mut cur = x.to_owned();
        for _ in 0..steps {
            if !ledger.charge() {
                break;
            }
            cur = target + &((&cur - target) * 0.85);
        }
        let d = &cur - target;
        let e = d.iter().map(|v| v * v).sum::<f64>();
        if steps > 1 {
            ledger.record_quench_boundary(before, e, cur.clone(), Some(2.0 * d));
        }
        (e, cur)
    }

    /// Two chains over one history: the second pays deposits for the visits
    /// the first made, and both report their observations. A chain alone
    /// pays nothing on anyone's behalf, and one well is one identity.
    #[test]
    fn a_shared_history_pays_deposits_for_the_other_chains_visits() {
        use crate::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
        use crate::methods::minima_hopping::{
            HistoryHook, HistoryMembership, MinimumHistory, SerializedWitness, SharedMinimumHistory,
        };
        use crate::pes_exploration::StructureContext;
        use std::sync::Mutex;

        let n = 4;
        let mut cfg = Config::for_cluster(n);
        cfg.max_hops = Some(40);
        cfg.screen_steps = 1;
        // Enough contraction that the certified gradient meets the record
        // tolerance, which is what makes a quench observable.
        cfg.relax_steps = 120;
        cfg.screen_margin = f64::INFINITY;
        cfg.return_screen = false;
        cfg.shared_deposits = 8;
        let target = Array1::from(vec![
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        ]);
        let descriptor = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
        let context = StructureContext::new(Some(vec![18; n]), None, Some("well".into()));
        let witness = SerializedWitness(Mutex::new(|l: ArrayView1<f64>, r: ArrayView1<f64>| {
            l.iter()
                .zip(r.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
                < 0.5
        }));
        let history = Mutex::new(MinimumHistory::new(10.0).unwrap());
        let mut rng = StdRng::seed_from_u64(7);
        let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
        let run = |seed: u64, hook: &mut dyn HistoryHook| {
            let mut ledger = Ledger::new(12_000);
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
                well_relax(&target, led, x, steps)
            };
            let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
                led.charge().then(|| 2.0 * (&x - &target))
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let mut checkpoint = |_: ChainCheckpoint<'_>| CheckpointAction::Continue;
            run_with_history_at_checkpoints(
                &cfg,
                start.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                None,
                Some(hook),
                &mut rng,
                500,
                &mut checkpoint,
            )
        };
        let mut first = SharedMinimumHistory::new(
            &history,
            &descriptor,
            context.clone(),
            &witness,
            HistoryMembership::Accepted,
        );
        let out1 = run(1, &mut first).expect("finite start");
        assert!(
            out1.history_visits.0 > 0,
            "the first chain reported nothing"
        );
        assert_eq!(
            out1.shared_deposits, 0,
            "alone, nothing is paid on another chain's behalf"
        );
        let prior = history.lock().unwrap().total_visits();
        let mut second = SharedMinimumHistory::new(
            &history,
            &descriptor,
            context,
            &witness,
            HistoryMembership::Accepted,
        );
        let out2 = run(2, &mut second).expect("finite start");
        assert!(
            out2.history_visits.0 > 0,
            "the second chain reported nothing"
        );
        assert!(
            out2.shared_deposits > 0,
            "the second chain paid nothing for {prior} prior visits"
        );
        assert_eq!(
            history.lock().unwrap().minimum_count(),
            1,
            "one well is one identity"
        );
    }

    #[test]
    fn separated_groups_clear_the_overlap_rule_without_stretching_a_bond() {
        let mut x = Array1::from(vec![
            0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 0.10, 0.0, 0.0, 1.06, 0.0, 0.0, -0.14,
            0.93, 0.0,
        ]);
        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let bond = |state: &Array1<f64>| {
            let dx = state[3] - state[0];
            let dy = state[4] - state[1];
            let dz = state[5] - state[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let before = bond(&x);
        assert!(min_pair_distance(x.view()) < OVERLAP_SEPARATION);
        assert!(separate_to_overlap_rule(
            &mut x,
            Some(&groups),
            OVERLAP_SEPARATION,
            None
        ));
        assert!(min_pair_distance(x.view()) >= OVERLAP_SEPARATION);
        assert!((bond(&x) - before).abs() < 1.0e-9);
    }

    #[test]
    fn recognition_does_not_count_a_peer_well_as_known_escape() {
        use crate::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
        use crate::methods::minima_hopping::{
            HistoryHook, HistoryMembership, MinimumHistory, SerializedWitness, SharedMinimumHistory,
        };
        use crate::pes_exploration::StructureContext;
        use std::sync::Mutex;

        let n = 4;
        let mut cfg = Config::for_cluster(n);
        cfg.max_hops = Some(20);
        cfg.screen_steps = 1;
        cfg.relax_steps = 120;
        cfg.screen_margin = f64::INFINITY;
        cfg.return_screen = false;
        cfg.minima_hopping = true;
        cfg.shared_deposits = 0;
        cfg.shared_visit_policy = SharedVisitPolicy::Recognition;
        let target = Array1::from(vec![
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        ]);
        let descriptor = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
        let context = StructureContext::new(Some(vec![18; n]), None, Some("well".into()));
        let witness = SerializedWitness(Mutex::new(|l: ArrayView1<f64>, r: ArrayView1<f64>| {
            l.iter()
                .zip(r.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
                < 0.5
        }));
        let history = Mutex::new(MinimumHistory::new(10.0).unwrap());
        let mut rng = StdRng::seed_from_u64(7);
        let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
        let run = |seed: u64, hook: &mut dyn HistoryHook| {
            let mut ledger = Ledger::new(12_000);
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
                well_relax(&target, led, x, steps)
            };
            let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
                led.charge().then(|| 2.0 * (&x - &target))
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let mut checkpoint = |_: ChainCheckpoint<'_>| CheckpointAction::Continue;
            run_with_history_at_checkpoints(
                &cfg,
                start.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                None,
                Some(hook),
                &mut rng,
                500,
                &mut checkpoint,
            )
        };
        let mut first = SharedMinimumHistory::new(
            &history,
            &descriptor,
            context.clone(),
            &witness,
            HistoryMembership::Accepted,
        );
        let _ = run(1, &mut first).expect("finite start");
        let mut second = SharedMinimumHistory::new(
            &history,
            &descriptor,
            context,
            &witness,
            HistoryMembership::Accepted,
        );
        let out2 = run(2, &mut second).expect("finite start");
        assert_eq!(
            out2.visit_counts.1, 0,
            "recognition must not inherit the peer well as known: {:?}",
            out2.visit_counts
        );
        assert_eq!(out2.shared_deposits, 0);
        assert_eq!(second.cost().1, 0, "no certified quench was refused");
    }

    #[test]
    fn recognition_refunds_a_second_descent_into_the_same_well() {
        let n = 4;
        let mut cfg = Config::for_cluster(n);
        cfg.max_hops = Some(6);
        cfg.screen_steps = 1;
        cfg.relax_steps = 80;
        cfg.screen_margin = f64::INFINITY;
        cfg.return_screen = false;
        cfg.merge_radius = 1.0e3;
        cfg.shared_visit_policy = SharedVisitPolicy::Recognition;
        let target = Array1::from(vec![
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        ]);
        let mut full = 0usize;
        let mut screen = 0usize;
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
            if steps >= 80 {
                full += 1;
            } else {
                screen += 1;
            }
            well_relax(&target, led, x, steps)
        };
        let mut rng = StdRng::seed_from_u64(7);
        let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(8_000);
        let _ = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
        assert!(full >= 1, "the first descent must certify the well");
        assert!(
            full < 6,
            "later descents into the same well must stand in: full={full} screen={screen}"
        );
    }

    #[test]
    fn overlapping_quench_is_not_recorded_or_reported_as_an_improvement() {
        let mut cfg = Config::for_cluster(2);
        cfg.species = Some(vec![29, 1]);
        cfg.max_hops = Some(1);
        cfg.screen_steps = 1;
        cfg.relax_steps = 1;
        cfg.screen_margin = f64::INFINITY;
        cfg.return_screen = false;

        let start = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let overlap = Array1::zeros(6);
        let mut calls = 0usize;
        let mut relax = |ledger: &mut Ledger, _x: ArrayView1<f64>, _steps: usize| {
            assert!(ledger.charge());
            calls += 1;
            if calls == 1 {
                (0.0, start.clone())
            } else {
                (-298_303.448_809, overlap.clone())
            }
        };
        let mut ledger = Ledger::new(16);
        let mut rng = StdRng::seed_from_u64(41);
        let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");

        assert_eq!(ledger.best, 0.0);
        assert_eq!(out.best, 0.0);
        assert!(
            out.improvements
                .iter()
                .all(|(_, _, _, energy)| *energy > -1_000.0),
            "overlap catastrophe entered the trace: {:?}",
            out.improvements
        );
    }

    #[test]
    fn unconverged_initial_quench_is_not_a_reported_minimum() {
        let mut cfg = Config::for_cluster(2);
        cfg.max_hops = Some(1);
        let start = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
            assert!(ledger.charge());
            (-10.0, x.to_owned())
        };
        let mut grad = |_ledger: &mut Ledger, x: ArrayView1<f64>| Some(Array1::ones(x.len()));
        let mut ledger = Ledger::new(1);
        let mut rng = StdRng::seed_from_u64(43);

        let out = run_with_gradient(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            Some(&mut grad),
            &mut rng,
        )
        .expect("finite start");

        assert!(out.best.is_infinite());
        assert!(out.best_state.is_none());
        assert!(out.improvements.is_empty());
    }

    #[test]
    fn unavailable_initial_gradient_is_not_validation() {
        let mut cfg = Config::for_cluster(2);
        cfg.max_hops = Some(1);
        let start = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
            assert!(ledger.charge());
            (-10.0, x.to_owned())
        };
        let mut grad = |_ledger: &mut Ledger, _x: ArrayView1<f64>| None;
        let mut ledger = Ledger::new(1);
        let mut rng = StdRng::seed_from_u64(45);

        let out = run_with_gradient(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            Some(&mut grad),
            &mut rng,
        )
        .expect("finite start");

        assert!(out.best.is_infinite());
        assert!(out.best_state.is_none());
        assert!(out.improvements.is_empty());
    }

    #[test]
    fn a_refused_start_redraws_until_a_finite_quench() {
        let mut cfg = Config::for_cluster(4);
        cfg.max_hops = Some(0);
        let start = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2,
        ]);
        {
            let mut calls = 0usize;
            let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
                assert!(ledger.charge());
                calls += 1;
                if calls <= 3 {
                    (f64::NAN, x.to_owned())
                } else {
                    (-1.0, x.to_owned())
                }
            };
            let mut ledger = Ledger::new(32);
            ledger.set_start_attempts(4);
            let mut rng = StdRng::seed_from_u64(9);
            let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng)
                .expect("the fourth quench is finite");
            assert_eq!(out.best, -1.0);
            assert_eq!(ledger.start_redraws().len(), 3);
            assert!(ledger.best.is_finite());
        }
        {
            let mut calls = 0usize;
            let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
                assert!(ledger.charge());
                calls += 1;
                if calls <= 3 {
                    (f64::NAN, x.to_owned())
                } else {
                    (-1.0, x.to_owned())
                }
            };
            let mut ledger = Ledger::new(32);
            ledger.set_start_attempts(3);
            let mut rng = StdRng::seed_from_u64(9);
            let err = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng)
                .expect_err("three refusals exhaust the attempts");
            assert_eq!(err, SearchError::NoFiniteStart { attempts: 3 });
            assert_eq!(ledger.start_redraws().len(), 2);
            assert!(ledger.best.is_infinite());
        }
    }

    #[test]
    fn a_caller_stop_during_a_refused_start_returns_the_outcome() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let mut cfg = Config::for_cluster(4);
        cfg.max_hops = Some(0);
        let start = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2,
        ]);
        let stop = Arc::new(AtomicBool::new(false));
        let mut calls = 0usize;
        let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
            assert!(ledger.charge());
            calls += 1;
            stop.store(true, Ordering::Relaxed);
            (f64::NAN, x.to_owned())
        };
        let mut ledger = Ledger::with_stop(32, Arc::clone(&stop));
        ledger.set_start_attempts(4);
        let mut rng = StdRng::seed_from_u64(9);
        let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng)
            .expect("a caller stop is not a refused start");
        assert_eq!(calls, 1);
        assert!(ledger.start_redraws().is_empty());
        assert!(out.best.is_infinite());
        assert_eq!(out.charged, 1);

        let raised = Arc::new(AtomicBool::new(true));
        let mut calls = 0usize;
        let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
            let _ = ledger.charge();
            calls += 1;
            (-1.0, x.to_owned())
        };
        let mut ledger = Ledger::with_stop(32, raised);
        let mut rng = StdRng::seed_from_u64(9);
        let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng)
            .expect("a raised stop returns before the first quench");
        assert_eq!(calls, 0);
        assert_eq!(out.charged, 0);
        assert!(out.best.is_infinite());
    }

    #[test]
    fn unconverged_trial_is_not_an_improvement_or_first_encounter() {
        let mut cfg = Config::for_cluster(2);
        cfg.max_hops = Some(1);
        cfg.screen_steps = 1;
        cfg.relax_steps = 1;
        cfg.screen_margin = f64::INFINITY;
        cfg.return_screen = false;

        let start = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let trial = Array1::from(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let mut calls = 0usize;
        let mut relax = |ledger: &mut Ledger, _x: ArrayView1<f64>, _steps: usize| {
            assert!(ledger.charge());
            calls += 1;
            if calls == 1 {
                (0.0, start.clone())
            } else {
                (-10.0, trial.clone())
            }
        };
        let mut grad = |_ledger: &mut Ledger, x: ArrayView1<f64>| {
            if x[3] < 1.5 {
                Some(Array1::zeros(x.len()))
            } else {
                Some(Array1::ones(x.len()))
            }
        };
        let mut ledger = Ledger::new(16);
        let mut rng = StdRng::seed_from_u64(47);

        let out = run_with_gradient(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            Some(&mut grad),
            &mut rng,
        )
        .expect("finite start");

        assert_eq!(out.best, 0.0);
        assert!(
            out.improvements
                .iter()
                .all(|(_, _, _, energy)| *energy >= 0.0),
            "unconverged trial entered the encounter trace: {:?}",
            out.improvements
        );
    }

    /// Each rung adapts its own step size and metric, and a swap moves
    /// configurations without moving the adaptation.
    #[test]
    fn every_rung_adapts_its_own_sampler() {
        let mut cfg = Config::for_cluster(6);
        cfg.replicas = 3;
        cfg.swap_period = 5;
        let mut h = crate::hmc::hop::HopConfig::new(6, crate::hmc::metric::MetricKind::Identity);
        h.warmup_hops = 20;
        h.max_depth = 2;
        cfg.hmc = Some(h);
        let mut ledger = Ledger::new(40_000);
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, n: usize| toy_relax(led, x, n);
        let mut eg = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<(f64, Array1<f64>)> {
            if !led.charge() {
                return None;
            }
            let e = x.iter().map(|v| v * v).sum::<f64>();
            Some((e, x.mapv(|v| 2.0 * v)))
        };
        let mut rng = StdRng::seed_from_u64(11);
        let start = random_cluster(6, 0.7, cfg.min_separation, &mut rng);
        let out = run_with_energy_gradient(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            Some(&mut eg),
            &mut rng,
        )
        .expect("finite start");
        assert_eq!(
            out.hmc.len(),
            3,
            "a three-rung ladder reported {} samplers",
            out.hmc.len()
        );
        for (k, d) in out.hmc.iter().enumerate() {
            assert!(d.proposals > 0, "rung {k} made no proposals");
            assert!(
                d.epsilon_final > 0.0 && d.epsilon_final.is_finite(),
                "rung {k} froze at a step size of {}",
                d.epsilon_final
            );
        }
        let eps: Vec<f64> = out.hmc.iter().map(|d| d.epsilon_final).collect();
        assert!(
            eps.iter().any(|v| (v - eps[0]).abs() > 0.0),
            "all three rungs froze at exactly {}",
            eps[0]
        );
    }

    #[test]
    fn respects_the_ledger() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(500);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 0).expect("finite start");
        assert!(out.charged <= 500, "spent {} of 500", out.charged);
        assert_eq!(ledger.remaining(), 0, "a run should spend its budget");
    }

    #[test]
    fn a_raised_stop_closes_the_ledger() {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut ledger = Ledger::with_stop(500, std::sync::Arc::clone(&stop));
        assert!(ledger.charge());
        assert_eq!(ledger.remaining(), 499);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(!ledger.charge(), "a stopped ledger charged");
        assert_eq!(ledger.remaining(), 0);
        assert_eq!(ledger.spent(), 1);
        let cfg = Config::for_cluster(6);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 0).expect("finite start");
        assert_eq!(out.charged, 1, "a stopped run spent its budget");
    }

    #[test]
    fn a_slice_closes_with_the_parent_stop() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let stop = Arc::new(AtomicBool::new(false));
        let parent = Ledger::with_stop(100, Arc::clone(&stop));
        let mut child = parent.slice(40);
        let mut plain = Ledger::new(40);
        assert!(child.charge());
        assert!(plain.charge());
        assert!(child.charge_frac(1.5));
        assert!(plain.charge_frac(1.5));
        assert_eq!(child.spent(), plain.spent());
        stop.store(true, Ordering::Relaxed);
        assert!(child.stopped());
        assert!(!child.charge());
        assert!(!child.charge_frac(0.5));
        assert_eq!(child.remaining(), 0);
        assert_eq!(child.spent(), plain.spent());
    }

    #[test]
    fn outcome_keeps_live_energy_and_accepted_trajectory_edges() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 19).expect("finite start");

        assert!(out.final_energy.is_finite());
        let final_state = out
            .final_state
            .as_ref()
            .expect("a completed hopping run must retain its live state");
        let last = out
            .accepted_transitions
            .last()
            .expect("the accepted trajectory must not be reduced to its best minimum");
        assert_eq!(last.to_energy, out.final_energy);
        assert_eq!(&last.to_state, final_state);
        assert!(!last.action.is_empty());
        assert!(last.validated);

        for pair in out.accepted_transitions.windows(2) {
            assert_eq!(pair[0].to_energy, pair[1].from_energy);
            assert_eq!(pair[0].to_state, pair[1].from_state);
        }
    }

    #[test]
    fn validated_accepted_transitions_retain_their_fresh_gradient() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let mut grad = |ledger: &mut Ledger, x: ArrayView1<f64>| {
            ledger.charge().then(|| Array1::zeros(x.len()))
        };
        let mut rng = StdRng::seed_from_u64(19);

        let out = run_with_gradient(
            &cfg,
            random_cluster(6, 0.7, cfg.min_separation, &mut rng).view(),
            &mut ledger,
            &mut relax,
            Some(&mut grad),
            &mut rng,
        )
        .expect("finite start");

        let validated = out
            .accepted_transitions
            .iter()
            .filter(|transition| transition.validated)
            .collect::<Vec<_>>();
        assert!(!validated.is_empty());
        assert!(validated.iter().all(|transition| {
            transition
                .from_gradient
                .as_ref()
                .is_some_and(|gradient| gradient.len() == transition.from_state.len())
                && transition
                    .to_gradient
                    .as_ref()
                    .is_some_and(|gradient| gradient.len() == transition.to_state.len())
        }));
    }

    #[test]
    fn spectral_funnel_bias_runs_under_the_ledger() {
        // track_funnels must not change the charge contract: SpectralBias is an
        // extra term on the Metropolis delta and a graph update on hop identity,
        // not a second force evaluation.
        let mut cfg = Config::for_cluster(6);
        cfg.track_funnels = true;
        cfg.funnel_period = 8;
        let mut ledger = Ledger::new(1500);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 11).expect("finite start");
        assert!(out.charged <= 1500, "spent {} of 1500", out.charged);
        assert_eq!(
            ledger.remaining(),
            0,
            "a funnel-biased run must still empty the ledger"
        );
        assert!(out.hops > 0, "no hop completed under spectral bias");
        assert!(
            out.accepted <= out.hops,
            "accepted {} > hops {}",
            out.accepted,
            out.hops
        );
    }

    #[test]
    fn screen_descent_credits_the_upper_decile_after_warmup() {
        let mut descent = ScreenDescent::default();
        for drop in 1..ScreenDescent::WARMUP {
            descent.observe(drop as f64);
            assert_eq!(descent.allowance(), 0.0, "credited before the warmup");
        }
        descent.observe(f64::NAN);
        assert_eq!(descent.allowance(), 0.0, "a non-finite drop was counted");
        for drop in ScreenDescent::WARMUP..=100 {
            descent.observe(drop as f64);
        }
        assert_eq!(descent.allowance(), 90.0);
        descent.observe(-5.0);
        assert_eq!(descent.drops.back(), Some(&0.0), "a rise is no descent");
        for _ in 0..ScreenDescent::WINDOW {
            descent.observe(2.0);
        }
        assert_eq!(descent.drops.len(), ScreenDescent::WINDOW);
        assert_eq!(descent.allowance(), 2.0, "the window forgets old drops");
        assert!(Config::for_cluster(13).calibrated_screen);
        assert!(Config::for_cluster(13).finish_improving);
        assert!(Config::for_tip4p(4).calibrated_screen);
    }

    #[test]
    fn screening_skips_full_relaxation_but_completes_the_hop() {
        let mut cfg = Config::for_cluster(6);
        // The audit below is the margin screen. `for_cluster` now starts
        // with the Bayes screen, which would decide instead.
        cfg.bayes_screen = false;
        cfg.screen_margin = -1.0e12;
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 1).expect("finite start");
        assert!(out.hops > 0, "no hop completed");
        assert_eq!(
            out.screened_out,
            out.hops - out.hops / ScreenDescent::AUDIT,
            "the calibrated screen must audit every eighth refusal"
        );

        cfg.calibrated_screen = false;
        let mut ledger = Ledger::new(4000);
        let mut relaxation_steps = Vec::new();
        let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
            relaxation_steps.push(steps);
            toy_relax(ledger, x, steps)
        };
        let out = optimize(&cfg, &mut ledger, &mut relax, 1).expect("finite start");
        assert!(out.hops > 0, "no hop completed");
        assert_eq!(out.screened_out, out.hops);
        assert!(
            relaxation_steps.contains(&cfg.screen_steps),
            "the short screen never ran"
        );
        assert_eq!(
            relaxation_steps
                .iter()
                .filter(|&&steps| steps == cfg.relax_steps)
                .count(),
            1,
            "a screened hop paid for a full relaxation"
        );
    }

    #[test]
    fn registers_basins_and_reports_them() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 2).expect("finite start");
        assert!(out.basins >= 1, "at least the starting basin must register");
    }

    /// A single chain has no ladder, so there is nothing to report about
    /// transport and the field says so rather than reporting a zero that reads
    /// like a ladder that failed.
    #[test]
    fn a_single_chain_reports_no_transport() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 3).expect("finite start");
        assert!(out.transport.is_none());
    }

    /// The instrument that says whether the ladder does its job. A swap count
    /// cannot: a ladder that shuffles one pair for the whole run and a ladder
    /// that carries configurations from the hottest rung to the coldest report
    /// the same swaps and the same solve count.
    #[test]
    fn the_ladder_reports_what_it_transported() {
        let mut cfg = Config::for_cluster(6);
        cfg.replicas = 4;
        cfg.ladder_mode = LadderMode::NonReversible;
        cfg.swap_period = 2;
        cfg.ladder_pilot = 20;
        let mut ledger = Ledger::new(60_000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 5).expect("finite start");
        let (trips, sweeps, barrier) = out.transport.expect("a ladder reports transport");
        assert!(sweeps > 20, "only {sweeps} sweeps; the ladder never ran");
        assert!(trips > 0, "no round trip in {sweeps} sweeps");
        assert!(
            (0.0..=3.0).contains(&barrier),
            "barrier {barrier} outside [0, rungs - 1]"
        );
        assert_eq!(out.rungs.len(), 4);
    }

    /// The adapted ladder is placed by the run and not by the schedule, so its
    /// rungs have to differ from the geometric ones it started at.
    #[test]
    fn the_adapted_ladder_leaves_the_geometric_schedule() {
        let mut base = Config::for_cluster(6);
        base.replicas = 4;
        base.swap_period = 2;
        base.ladder_pilot = 20;
        let run = |mode: LadderMode| {
            let mut c = base.clone();
            c.ladder_mode = mode;
            let mut ledger = Ledger::new(60_000);
            let mut relax = toy_relax;
            let out = optimize(&c, &mut ledger, &mut relax, 5).expect("finite start");
            out.rungs.iter().map(|(t, _, _)| *t).collect::<Vec<f64>>()
        };
        let geometric = run(LadderMode::Reversible);
        let adapted = run(LadderMode::NonReversible);
        assert!(
            (geometric[0] - base.temperature).abs() < 1e-12
                && (adapted[0] - base.temperature).abs() < 1e-12,
            "the cold rung is the configured temperature under both"
        );
        let moved = geometric
            .iter()
            .zip(&adapted)
            .any(|(g, a)| (g - a).abs() > 1e-6 * g.abs().max(1.0));
        assert!(
            moved,
            "adapted ladder {adapted:?} is the geometric one {geometric:?}; \
             nothing was derived from the run"
        );
        assert!(
            adapted.windows(2).all(|w| w[1] > w[0]),
            "adapted ladder {adapted:?} is not ordered by temperature"
        );
    }

    #[test]
    fn seeds_are_reproducible() {
        let cfg = Config::for_cluster(6);
        let run_once = |seed| {
            let mut ledger = Ledger::new(2000);
            let mut relax = toy_relax;
            optimize(&cfg, &mut ledger, &mut relax, seed)
                .expect("finite start")
                .best
        };
        assert_eq!(run_once(7), run_once(7), "same seed must give same result");
    }

    #[test]
    fn rung_ratios_run_geometrically_from_one_to_the_top() {
        assert_eq!(rung_ratios(1, 4.0), vec![1.0]);
        assert_eq!(rung_ratios(0, 4.0), vec![1.0]);
        let ratios = rung_ratios(4, 4.0);
        assert_eq!(ratios.len(), 4);
        assert_eq!(ratios[0], 1.0);
        assert_eq!(ratios[3], 4.0);
        for pair in ratios.windows(2) {
            assert!((pair[1] / pair[0] - 4.0_f64.cbrt()).abs() < 1e-12);
        }
    }

    /// Whatever a single chain would hop at, the coldest rung hops at it and
    /// the others at their ratio times it. The statistical estimate is clamped
    /// to its band before the ratio, so the band moves up the ladder with the
    /// rung rather than cutting the hot end off.
    #[test]
    fn each_rung_hops_at_its_ratio_times_the_single_chain_temperature() {
        let mut cfg = Config::for_cluster(13);
        let ratios = rung_ratios(4, 4.0);
        for held in [cfg.temperature, 0.037] {
            for &r in &ratios {
                assert_eq!(rung_temperature(&cfg, r, held, None, -40.0), held * r);
            }
        }

        // An entropy rising half a nat per unit energy, so the estimate is
        // finite and positive.
        let mut dos = crate::dos::DensityOfStates::new(-45.0, -30.0, 30);
        for k in 0..30 {
            let visits = (2000.0 * (0.5 * (dos.centre(k) + 30.0)).exp()).round() as usize;
            for _ in 0..visits.max(1) {
                dos.observe(dos.centre(k));
            }
        }
        let e = -36.0;
        cfg.statistical_temperature = true;
        assert_eq!(
            rung_temperature(&cfg, 2.0, 0.037, Some(&dos), e),
            0.037 * 2.0
        );
        assert!(dos.refresh());
        let (estimate, _) = dos.temperature(e);
        assert!(
            estimate.is_finite() && estimate > 0.0,
            "estimate {estimate}"
        );

        for (temperature, single) in [
            (estimate, estimate),
            (estimate / 50.0, 5.0 * estimate / 50.0),
            (estimate * 50.0, 0.2 * estimate * 50.0),
        ] {
            cfg.temperature = temperature;
            for &r in &ratios {
                let t = rung_temperature(&cfg, r, cfg.temperature, Some(&dos), e);
                assert!(
                    (t - single * r).abs() <= 1e-12 * single * r,
                    "rung at {r} hopped at {t}, wanted {}",
                    single * r
                );
            }
        }
        cfg.statistical_temperature = false;
        assert_eq!(
            rung_temperature(&cfg, 2.0, 0.037, Some(&dos), e),
            0.037 * 2.0
        );
    }

    /// Scaled by rung, a deposit weighs the same against the temperature each
    /// rung hops at, and the hottest carries the configured height.
    #[test]
    fn rung_heights_follow_the_rung_temperatures() {
        let mut cfg = Config::for_cluster(13);
        let ratios = rung_ratios(4, cfg.ladder_top);
        assert!(
            ratios
                .iter()
                .all(|&r| rung_height(&cfg, 4, r) == cfg.bias_height)
        );
        cfg.bias_by_rung = true;
        assert_eq!(rung_height(&cfg, 1, 1.0), cfg.bias_height);
        let weight = |r: f64| {
            rung_height(&cfg, 4, r) / rung_temperature(&cfg, r, cfg.temperature, None, 0.0)
        };
        for &r in &ratios {
            assert!((weight(r) / weight(1.0) - 1.0).abs() < 1e-12);
        }
        assert!((rung_height(&cfg, 4, ratios[3]) - cfg.bias_height).abs() < 1e-12);
        assert!(rung_height(&cfg, 4, ratios[0]) < cfg.bias_height);
    }

    /// With one bias on both rungs and nothing shared the swap is the
    /// Metropolis exchange. What every rung shares drops out of a flat ladder
    /// and otherwise weighs as if it were part of each rung's own bias.
    #[test]
    fn the_swap_weighs_shared_biases_by_the_temperature_difference() {
        use crate::exchange::{Exchange, MetropolisExchange};
        let (t_k, t_j) = (0.8, 2.0);
        let (u_k, u_j) = (-40.0, -38.5);
        let log_a = swap_log_acceptance([u_k, u_j], [u_j, u_k], [0.0, 0.0], t_k, t_j);
        let metropolis: f64 = MetropolisExchange.swap_accept_prob(u_k, t_k, u_j, t_j);
        assert!(metropolis < 1.0);
        assert!((log_a.exp() - metropolis).abs() < 1e-12);

        let own_k = [-40.0, -38.0];
        let own_j = [-37.5, -39.0];
        let shared = [3.0, -2.0];
        assert_eq!(
            swap_log_acceptance(own_k, own_j, shared, 1.3, 1.3),
            swap_log_acceptance(own_k, own_j, [0.0, 0.0], 1.3, 1.3)
        );
        let folded = swap_log_acceptance(
            [own_k[0] + shared[0], own_k[1] + shared[1]],
            [own_j[0] + shared[1], own_j[1] + shared[0]],
            [0.0, 0.0],
            t_k,
            t_j,
        );
        let split = swap_log_acceptance(own_k, own_j, shared, t_k, t_j);
        assert!((folded - split).abs() < 1e-12 * folded.abs().max(1.0));
        assert!(
            (split - swap_log_acceptance(own_k, own_j, [0.0, 0.0], t_k, t_j)).abs() > 1.0,
            "the shared terms should matter between two temperatures"
        );
    }

    /// The swap is the ratio of the weights the rungs' acceptance applies, at
    /// the states exchanged and at the states held. Under the flat-histogram
    /// cost the energies leave it, so states whose biases agree swap freely
    /// between two temperatures however far apart their energies are.
    #[test]
    fn the_swap_exchanges_the_weight_each_rung_hops_by() {
        let mut cfg = Config::recommended(13);
        assert_eq!(RungWeight::of(&cfg, true), RungWeight::Boltzmann);
        cfg.flat_histogram = true;
        assert_eq!(RungWeight::of(&cfg, false), RungWeight::FirstSweep);
        assert_eq!(RungWeight::of(&cfg, true), RungWeight::Flat);
        cfg.minima_hopping = true;
        assert_eq!(RungWeight::of(&cfg, true), RungWeight::Boltzmann);

        let mut dos = crate::dos::DensityOfStates::new(-45.0, -30.0, 30);
        for k in 0..30 {
            let visits = (2000.0 * (0.5 * (dos.centre(k) + 30.0)).exp()).round() as usize;
            for _ in 0..visits.max(1) {
                dos.observe(dos.centre(k));
            }
        }
        assert!(dos.refresh());
        let flat = crate::dos::CutWeight {
            weight: dos.mean_weight(),
            cut: -38.0,
            width: 0.7,
        };
        let mut rng = StdRng::seed_from_u64(11);
        let draw = |rng: &mut StdRng| HopTerms {
            energy: rng.random_range(-44.0..-31.0),
            funnel: rng.random_range(0.0..2.0),
            pile: rng.random_range(0.0..3.0),
            energy_bias: rng.random_range(0.0..1.5),
        };
        for _ in 0..200 {
            let (at_k, at_j) = (draw(&mut rng), draw(&mut rng));
            let own_k = [rng.random_range(0.0..2.0), rng.random_range(0.0..2.0)];
            let own_j = [rng.random_range(0.0..2.0), rng.random_range(0.0..2.0)];
            let (t_k, t_j) = (rng.random_range(0.1..1.0), rng.random_range(1.0..4.0));
            for weight in [
                RungWeight::Boltzmann,
                RungWeight::FirstSweep,
                RungWeight::Flat,
            ] {
                // Minus the log of the weight a rung at temperature `t` with
                // its own bias `v` applies at a state.
                let cost = |at: HopTerms, v: f64, t: f64| match weight {
                    RungWeight::Boltzmann => {
                        (at.energy + v + at.funnel + at.pile + at.energy_bias) / t
                    }
                    RungWeight::FirstSweep => (at.energy + v + at.funnel + at.pile) / t,
                    RungWeight::Flat => flat.cost(at.energy) + (v + at.funnel) / t,
                };
                let expected = cost(at_k, own_k[0], t_k) + cost(at_j, own_j[0], t_j)
                    - cost(at_j, own_k[1], t_k)
                    - cost(at_k, own_j[1], t_j);
                let log_a = exchange_log_acceptance(weight, at_k, at_j, own_k, own_j, t_k, t_j);
                assert!(
                    (log_a - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                    "{weight:?}: {log_a} against {expected}"
                );
            }
        }

        let low = HopTerms {
            energy: -44.0,
            funnel: 0.5,
            pile: 1.0,
            energy_bias: 0.2,
        };
        let high = HopTerms {
            energy: -32.0,
            ..low
        };
        let swap =
            |weight| exchange_log_acceptance(weight, low, high, [0.3; 2], [0.6; 2], 0.2, 2.0);
        assert_eq!(swap(RungWeight::Flat), 0.0);
        assert!(swap(RungWeight::Boltzmann) < -40.0);
    }

    #[test]
    fn seeded_cluster_has_no_overlapping_points() {
        let mut rng = StdRng::seed_from_u64(3);
        let n = 38;
        let x = random_cluster(n, 0.7, 0.85, &mut rng);
        assert_eq!(x.len(), 3 * n, "seeding fell short of the requested size");
        for a in 0..n {
            for b in (a + 1)..n {
                let d = ((x[3 * a] - x[3 * b]).powi(2)
                    + (x[3 * a + 1] - x[3 * b + 1]).powi(2)
                    + (x[3 * a + 2] - x[3 * b + 2]).powi(2))
                .sqrt();
                assert!(d >= 0.85 - 1e-9, "points {a} and {b} overlap at {d}");
            }
        }
    }

    #[test]
    fn containment_pulls_strays_back() {
        let n = 3;
        let mut x = Array1::from(vec![10.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -0.3]);
        contain(&mut x, n, 1.0);
        for i in 0..n {
            let r = (0..3).map(|k| x[3 * i + k].powi(2)).sum::<f64>().sqrt();
            assert!(r <= 1.0 + 1e-9, "point {i} left the container at {r}");
        }
    }

    #[test]
    fn an_atom_past_the_cutoff_with_no_force_is_shed() {
        let cfg = Config::recommended(3);
        let cut = cfg.neighbour_cutoff;
        let bonded = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.1, 0.0]);
        assert!(!sheds_free_atom(&cfg, bonded.view(), None));
        let apart = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.1 + cut, 0.0]);
        assert!(sheds_free_atom(&cfg, apart.view(), None));
        let mut gradient = Array1::from(vec![0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(sheds_free_atom(&cfg, apart.view(), Some(gradient.view())));
        gradient[7] = 1.0e-300;
        assert!(!sheds_free_atom(&cfg, apart.view(), Some(gradient.view())));
        let mut labelled = cfg.clone();
        labelled.species = Some(vec![29, 29, 79]);
        assert!(!sheds_free_atom(&labelled, apart.view(), None));
    }

    #[test]
    fn orbit_arm_is_offered_only_without_completion_on_new_basins() {
        let names = |cfg: &Config| -> Vec<String> {
            search_kernels(cfg)
                .iter()
                .map(|kernel| kernel.name())
                .collect()
        };
        let recommended = Config::recommended(13);
        let mut armed = recommended.clone();
        armed.orbit_arm = true;
        assert_eq!(names(&recommended), names(&armed));
        assert!(
            armed
                .resolved_json()
                .unwrap()
                .contains("\"orbit_arm\":true")
        );
        assert!(!recommended.resolved_json().unwrap().contains("orbit_arm"));
        armed.orbit_complete_on_new = false;
        let offered = names(&armed);
        assert_eq!(offered.iter().filter(|name| *name == "orbit").count(), 1);
        armed.orbit_arm = false;
        assert!(!names(&armed).iter().any(|name| name == "orbit"));
    }

    #[test]
    fn the_orbit_arm_steps_one_point_when_no_group_moves_anyone() {
        let cfg = Config::for_cluster(3);
        let kernel = ClusterMove::OrbitComplete {
            n_points: 3,
            tolerance: cfg.symmetry_tolerance,
            merge_radius: cfg.symmetry_merge_radius,
            core_fraction: cfg.symmetrise_core_fraction,
            min_separation: cfg.min_separation,
            fallback_step: 0.1,
        };
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.1, 0.0]);
        let mut rng = StdRng::seed_from_u64(7);
        let y = kernel.propose(x.view(), cfg.temperature, &mut rng);
        let moved: Vec<usize> = (0..3)
            .filter(|&i| (0..3).any(|k| y[3 * i + k] != x[3 * i + k]))
            .collect();
        assert_eq!(moved.len(), 1);
    }

    #[test]
    fn a_part_out_of_reach_comes_back_to_one_bond() {
        let start = Array1::from(vec![
            0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.4, 0.0, 9.0, 0.0, 0.0, 9.0, 1.4, 0.0,
        ]);
        let mut x = start.clone();
        gather_into_reach(&mut x, 5, 2.1, 1.45);
        assert_eq!(connectivity_groups(x.view(), 5, 2.1).len(), 1);
        let d = |a: usize, b: usize| {
            (0..3)
                .map(|k| (x[3 * a + k] - x[3 * b + k]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        assert!((d(1, 3) - 1.45).abs() < 1e-12, "moved to {}", d(1, 3));
        assert!((d(3, 4) - 1.4).abs() < 1e-12);
        assert_eq!(x.slice(ndarray::s![0..9]), start.slice(ndarray::s![0..9]));
    }

    #[test]
    fn a_measured_reach_replaces_the_sphere_at_any_length() {
        let mut cfg = Config::recommended(3);
        cfg.container = 1.5;
        let start = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 6.0, 0.0, 0.0]);
        let mut sphere = start.clone();
        contain(&mut sphere, 3, cfg.container);
        let mut plain = start.clone();
        contain_trial(&cfg, &mut plain, 3);
        assert_eq!(plain, sphere);
        for reach in [2.0, 4.0 * cfg.container] {
            cfg.pair_range = Some(PairRange { reach, bond: 1.1 });
            let mut gathered = start.clone();
            contain_trial(&cfg, &mut gathered, 3);
            assert_ne!(gathered, sphere);
            assert_eq!(connectivity_groups(gathered.view(), 3, reach).len(), 1);
        }
        assert!(cfg.resolved_json().unwrap().contains("\"pair_range\""));
    }

    #[test]
    fn a_cutoff_has_a_reach_and_a_tail_does_not() {
        let range = PairRange::measure(1.45, 1.0e3, |r| Some(r >= 2.1)).expect("a reach");
        assert!((range.reach - 2.1).abs() < 1e-5);
        assert_eq!(range.bond, 1.45);
        assert!(PairRange::measure(1.12, 1.0e3, |_| Some(false)).is_none());
        let mut left = 3;
        let starved = PairRange::measure(1.45, 1.0e3, |r| {
            if left == 0 {
                return None;
            }
            left -= 1;
            Some(r >= 2.1)
        });
        assert!(starved.is_none());
    }

    #[test]
    fn repair_enforces_the_separation() {
        let n = 2;
        let mut x = Array1::from(vec![0.0, 0.0, 0.0, 0.01, 0.0, 0.0]);
        repair(&mut x, n, 0.85);
        let d = ((x[0] - x[3]).powi(2) + (x[1] - x[4]).powi(2) + (x[2] - x[5]).powi(2)).sqrt();
        assert!(d >= 0.85 - 1e-6, "overlap survived repair at {d}");
    }
}

/// Descriptor for basin keying, matched to the metric that will compare it.
///
/// A sorted distance spectrum is permutation and rotation invariant already, so
/// Euclidean distance on it is a usable if scale-broken notion of sameness.
/// A shape metric quotients out those symmetries itself and needs the
/// coordinates, so the two cannot be mixed.
pub enum ClusterFingerprint {
    /// Actual rigid-body site distances, separated by chemical species.
    RigidBodies {
        /// Number of centers and rotation vectors in the state.
        molecules: usize,
        /// Body-frame geometry used by the objective.
        geometry: crate::rigid_body::RigidBodyGeometry,
    },
    /// Sorted pairwise distances, compared by Euclidean distance.
    Spectrum(SortedPairs),
    /// Coordinates, for a metric that does its own matching.
    Coordinates,
    /// Sorted per-point pair energies, keying on how well each point is bound
    /// rather than on how far apart the points are.
    Sites(SiteEnergies),
    /// Sorted distances with the two-body and three-body kernel spectra
    /// appended, compared by Euclidean distance.
    Triplet(Box<crate::tensor_id::TripletSpectrum>),
    /// Per-site coordination numbers, smoothed into a kernel density estimate.
    Coordination(Box<crate::morphology::CoordinationKde>),
    /// Steinhardt bond-order parameters of the whole cluster.
    #[cfg(feature = "featomic")]
    Steinhardt(Box<crate::morphology::SteinhardtQ>),
    /// Leading principal component of the SOAP power spectrum, fitted online.
    #[cfg(feature = "featomic")]
    SoapProjection(Box<crate::morphology::SoapProjection>),
    /// Coordinates put in a canonical order against a fixed reference, so
    /// Euclidean distance between two of them is a shape distance.
    #[cfg(feature = "ira")]
    Canonical(Box<crate::shape::CanonicalOrder>),
    /// Coloured core ring-graph key: one point per superbasin, spread over
    /// four coordinates at least one apart, so any merge radius below one
    /// keys exactly on the superbasin.
    Core {
        /// Optional atomic numbers, one per point.
        species: Option<Vec<u32>>,
    },
    /// Unit high-`l` mean SOAP. Packing superbasin, not an isomer.
    #[cfg(feature = "featomic")]
    SoapMean {
        /// SOAP neighbour cutoff in the coordinate units.
        rcut: f64,
        /// Optional atomic numbers, one per point.
        species: Option<Vec<u32>>,
    },
}

/// The keying a config asks for, honouring the older boolean.
fn effective_keying(cfg: &Config) -> Keying {
    if cfg.shape_keyed && cfg.keying == Keying::Distances {
        Keying::Shape
    } else {
        cfg.keying
    }
}

impl ClusterFingerprint {
    /// The descriptor a given keying requires.
    pub fn for_keying(n_points: usize, shape_keyed: bool) -> Self {
        Self::of(
            n_points,
            if shape_keyed {
                Keying::Shape
            } else {
                Keying::Distances
            },
        )
    }

    /// The descriptor for a named keying, without a reference.
    ///
    /// [`Keying::Canonical`] needs one and falls back to the distance spectrum
    /// here; callers that want it should use [`ClusterFingerprint::of_with`].
    pub fn of(n_points: usize, keying: Keying) -> Self {
        Self::of_with(n_points, keying, &Array1::zeros(0))
    }

    /// The descriptor for a named keying, against `reference`, with the
    /// [`Keying::Triplet`] kernel width at its Lennard-Jones default.
    pub fn of_with(n_points: usize, keying: Keying, reference: &Array1<f64>) -> Self {
        Self::of_tuned(n_points, keying, reference, 2.5)
    }

    /// The descriptor for a named keying, against `reference`, with the kernel
    /// width `sigma` for [`Keying::Triplet`]. `sigma` carries the length units
    /// of the coordinates.
    pub fn of_tuned(n_points: usize, keying: Keying, reference: &Array1<f64>, sigma: f64) -> Self {
        Self::of_full(n_points, keying, reference, sigma, 1.0)
    }

    /// The descriptor for a named keying at a length scale of `scale`, the
    /// potential's `r_min` over `2^(1/6)`, which moves the neighbour shells
    /// the morphology descriptors cut between.
    pub fn of_scaled(n_points: usize, keying: Keying, reference: &Array1<f64>, scale: f64) -> Self {
        Self::of_full(n_points, keying, reference, 2.5, scale)
    }

    /// Every named keying, with the triplet kernel width and the length scale.
    pub fn of_full(
        n_points: usize,
        keying: Keying,
        reference: &Array1<f64>,
        sigma: f64,
        scale: f64,
    ) -> Self {
        #[cfg(not(feature = "ira"))]
        let _ = reference;
        #[cfg(not(feature = "featomic"))]
        let _ = scale;
        match keying {
            Keying::Coordination => {
                let mut kde = crate::morphology::CoordinationKde::for_lj(n_points, scale);
                // The bin centres are a measurement; COORD_BINS and
                // COORD_SIGMA override them without a rebuild.
                if let Ok(v) = std::env::var("COORD_BINS") {
                    let bins: Vec<f64> = v
                        .split(',')
                        .filter_map(|t| t.trim().parse::<f64>().ok())
                        .collect();
                    if !bins.is_empty() {
                        kde.bins = bins;
                    }
                }
                if let Ok(v) = std::env::var("COORD_SIGMA")
                    && let Ok(sg) = v.parse::<f64>()
                    && sg > 0.0
                {
                    kde.sigma = sg;
                }
                ClusterFingerprint::Coordination(Box::new(kde))
            }
            #[cfg(feature = "featomic")]
            Keying::Q4 => ClusterFingerprint::Steinhardt(Box::new(
                crate::morphology::SteinhardtQ::q4(n_points, scale),
            )),
            #[cfg(feature = "featomic")]
            Keying::Q4Q6 => ClusterFingerprint::Steinhardt(Box::new(
                crate::morphology::SteinhardtQ::q4q6(n_points, scale),
            )),
            #[cfg(feature = "featomic")]
            Keying::Soap => ClusterFingerprint::SoapProjection(Box::new(
                crate::morphology::SoapProjection::new(n_points, scale),
            )),
            // Without featomic there is no spherical expansion; falling back
            // to the distance spectrum would run an arm that reports itself as
            // a Q4 bias and is not one.
            #[cfg(not(feature = "featomic"))]
            Keying::Q4 | Keying::Q4Q6 | Keying::Soap => {
                panic!("keying {keying:?} needs the `featomic` feature")
            }
            Keying::Shape => ClusterFingerprint::Coordinates,
            Keying::Core => ClusterFingerprint::Core { species: None },
            Keying::Distances => ClusterFingerprint::Spectrum(SortedPairs { n_points }),
            Keying::Sites => ClusterFingerprint::Sites(SiteEnergies { n_points }),
            Keying::Triplet => ClusterFingerprint::Triplet(Box::new(
                crate::tensor_id::TripletSpectrum::new(n_points).with_sigma(sigma),
            )),
            Keying::SoapPacking => ClusterFingerprint::Spectrum(SortedPairs { n_points }),
            #[cfg(feature = "ira")]
            Keying::Canonical => {
                if reference.len() == 3 * n_points {
                    ClusterFingerprint::Canonical(Box::new(crate::shape::CanonicalOrder::new(
                        reference.clone(),
                        1.8,
                    )))
                } else {
                    ClusterFingerprint::Spectrum(SortedPairs { n_points })
                }
            }
            #[cfg(not(feature = "ira"))]
            Keying::Canonical => ClusterFingerprint::Spectrum(SortedPairs { n_points }),
        }
    }

    /// Descriptor from the live config, so SOAP packing carries cutoff
    /// and species.
    pub fn of_config(cfg: &Config, reference: &Array1<f64>) -> Self {
        if cfg.move_library.is_rigid_body()
            && effective_keying(cfg) == Keying::Distances
            && let Some(geometry) = cfg.rigid_body_geometry.as_ref()
        {
            return Self::RigidBodies {
                molecules: cfg.n_points,
                geometry: geometry.clone(),
            };
        }
        match effective_keying(cfg) {
            Keying::Core => ClusterFingerprint::Core {
                species: cfg.species.clone(),
            },
            Keying::SoapPacking => {
                #[cfg(feature = "featomic")]
                {
                    ClusterFingerprint::SoapMean {
                        rcut: 3.5 * cfg.length_scale,
                        species: cfg.species.clone(),
                    }
                }
                #[cfg(not(feature = "featomic"))]
                {
                    ClusterFingerprint::Spectrum(SortedPairs {
                        n_points: cfg.n_points,
                    })
                }
            }
            other => Self::of_full(
                cfg.n_points,
                other,
                reference,
                cfg.keying_sigma,
                cfg.morphology_scale,
            ),
        }
    }
}

impl Fingerprint for ClusterFingerprint {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        match self {
            ClusterFingerprint::RigidBodies {
                molecules,
                geometry,
            } => geometry.describe(*molecules, x),
            ClusterFingerprint::Spectrum(s) => s.describe(x),
            ClusterFingerprint::Coordinates => x.to_owned(),
            ClusterFingerprint::Core { species } => {
                let key = crate::corekey::core_key_nn(
                    x,
                    species.as_deref().unwrap_or(&[]),
                    crate::corekey::CoreRule::default(),
                );
                Array1::from(key.coordinates().to_vec())
            }
            ClusterFingerprint::Sites(s) => s.describe(x),
            ClusterFingerprint::Triplet(t) => t.describe(x),
            ClusterFingerprint::Coordination(c) => c.describe(x),
            #[cfg(feature = "featomic")]
            ClusterFingerprint::Steinhardt(s) => s.describe(x),
            #[cfg(feature = "featomic")]
            ClusterFingerprint::SoapProjection(s) => s.describe(x),
            #[cfg(feature = "ira")]
            ClusterFingerprint::Canonical(c) => c.describe(x),
            #[cfg(feature = "featomic")]
            ClusterFingerprint::SoapMean { rcut, species } => {
                crate::featomic_hop::soap_cloud_mean(x, *rcut, species.as_deref(), None)
            }
        }
    }
}

#[cfg(test)]
mod descriptor_reuse_tests {
    use super::*;
    use crate::methods::warm_lbfgs::WarmLbfgs;
    use crate::potentials::{PairPotential, random_tip4p_cluster};

    const POINTS: usize = 13;
    const WATERS: usize = 10;

    /// Three quenched Lennard-Jones minima and one unquenched start, the
    /// kinds of state a hop describes.
    fn lennard_jones_states() -> Vec<Array1<f64>> {
        let cfg = Config::for_cluster(POINTS);
        let potential = PairPotential::lennard_jones(POINTS);
        let mut rng = StdRng::seed_from_u64(29);
        [true, true, true, false]
            .into_iter()
            .map(|quench| {
                let start = random_cluster_in_radius(
                    POINTS,
                    cfg.start_radius(),
                    cfg.min_separation,
                    &mut rng,
                );
                if quench {
                    let (_, minimum, _) = WarmLbfgs::default()
                        .minimize(start.view(), 500, |x| Some(potential.value_and_gradient(x)));
                    minimum
                } else {
                    start
                }
            })
            .collect()
    }

    fn assert_same_bits(label: &str, observed: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(observed.len(), expected.len(), "{label}: length changed");
        for (component, (a, b)) in observed.iter().zip(expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: component {component} changed"
            );
        }
    }

    /// The hop computes `bias.cv(x)` for the incumbent before the quench and
    /// carries that descriptor to the acceptance test and the deposit. This
    /// is exact only if a later call on the same coordinates returns the same
    /// bits, after calls on other states and after deposits, and if a bias
    /// built the same way returns them too.
    fn assert_repeat_calls_bit_identical(cfg: &Config, states: &[Array1<f64>]) {
        let keying = effective_keying(cfg);
        assert_ne!(keying, Keying::Soap, "the hop recomputes SOAP descriptors");
        let build = || {
            BasinBias::new(
                ClusterFingerprint::of_config(cfg, &states[0]),
                cfg.merge_radius,
                cfg.bias_height,
                cfg.bias_gamma,
            )
        };
        let mut bias = build();
        let first: Vec<Array1<f64>> = states.iter().map(|x| bias.cv(x.view())).collect();
        assert!(
            first.iter().all(|descriptor| !descriptor.is_empty()),
            "{keying:?}: an empty descriptor checks nothing"
        );
        for descriptor in &first {
            bias.deposit(descriptor.view(), cfg.temperature);
        }
        for (x, expected) in states.iter().zip(&first).rev() {
            assert_same_bits(&format!("{keying:?} repeat"), &bias.cv(x.view()), expected);
        }
        let fresh = build();
        for (x, expected) in states.iter().zip(&first) {
            assert_same_bits(&format!("{keying:?} fresh"), &fresh.cv(x.view()), expected);
        }
    }

    fn assert_keying_repeats(keying: Keying, builds: fn(&ClusterFingerprint) -> bool) {
        let mut cfg = Config::for_cluster(POINTS);
        cfg.keying = keying;
        let states = lennard_jones_states();
        assert!(
            builds(&ClusterFingerprint::of_config(&cfg, &states[0])),
            "{keying:?} built another descriptor"
        );
        assert_repeat_calls_bit_identical(&cfg, &states);
    }

    #[test]
    fn distances_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Distances, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Spectrum(_))
        });
    }

    #[test]
    fn sites_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Sites, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Sites(_))
        });
    }

    #[test]
    fn triplet_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Triplet, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Triplet(_))
        });
    }

    #[test]
    fn core_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Core, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Core { .. })
        });
    }

    #[test]
    fn coordination_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Coordination, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Coordination(_))
        });
    }

    #[test]
    fn shape_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Shape, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Coordinates)
        });
    }

    /// With `featomic`, SOAP packing keys on the mean SOAP descriptor.
    #[cfg(feature = "featomic")]
    #[test]
    fn soap_packing_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::SoapPacking, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::SoapMean { .. })
        });
    }

    /// Without `featomic`, SOAP packing keys on the distance spectrum.
    #[cfg(not(feature = "featomic"))]
    #[test]
    fn soap_packing_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::SoapPacking, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Spectrum(_))
        });
    }

    /// With `ira`, the canonical keying orders and aligns each structure
    /// against the reference.
    #[cfg(feature = "ira")]
    #[test]
    fn canonical_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Canonical, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Canonical(_))
        });
    }

    /// Without `ira`, the canonical keying falls back to the distance spectrum.
    #[cfg(not(feature = "ira"))]
    #[test]
    fn canonical_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Canonical, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Spectrum(_))
        });
    }

    #[cfg(feature = "featomic")]
    #[test]
    fn q4_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Q4, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Steinhardt(_))
        });
    }

    #[cfg(feature = "featomic")]
    #[test]
    fn q4q6_descriptor_repeats_bit_for_bit() {
        assert_keying_repeats(Keying::Q4Q6, |fingerprint| {
            matches!(fingerprint, ClusterFingerprint::Steinhardt(_))
        });
    }

    #[test]
    fn rigid_body_descriptor_repeats_bit_for_bit() {
        let cfg = Config::for_tip4p(WATERS);
        let mut rng = StdRng::seed_from_u64(31);
        let states: Vec<Array1<f64>> = (0..4)
            .map(|_| random_tip4p_cluster(WATERS, &mut rng))
            .collect();
        assert!(matches!(
            ClusterFingerprint::of_config(&cfg, &states[0]),
            ClusterFingerprint::RigidBodies { .. }
        ));
        assert_repeat_calls_bit_identical(&cfg, &states);
    }
}

#[cfg(test)]
mod geometry_guards {
    use super::{
        contain, min_pair_distance, recentre, separate_to_overlap_rule, structure_is_sane,
    };
    use ndarray::Array1;

    #[test]
    fn recentre_puts_the_centre_of_mass_at_the_origin() {
        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]);
        recentre(&mut x, 2);
        for axis in 0..3 {
            let centre = 0.5 * (x[axis] + x[3 + axis]);
            assert!(centre.abs() < 1e-12, "{centre}");
        }
    }

    #[test]
    fn contain_pulls_an_outside_point_onto_the_sphere() {
        let mut x = Array1::from_vec(vec![3.0, 0.0, 0.0, 0.2, 0.0, 0.0]);
        contain(&mut x, 2, 1.0);
        let radius = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        assert!((radius - 1.0).abs() < 1e-12);
        assert!((x[3] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn one_atom_has_no_pair_and_a_non_finite_state_is_not_sane() {
        assert!(min_pair_distance(Array1::from_vec(vec![1.0, 0.0, 0.0]).view()).is_infinite());
        let pair = Array1::from_vec(vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
        assert!((min_pair_distance(pair.view()) - 3.0).abs() < 1e-12);
        assert!(structure_is_sane(pair.view(), 1.0));
        assert!(!structure_is_sane(pair.view(), 4.0));
        let mut broken = pair.clone();
        broken[0] = f64::NAN;
        assert!(!structure_is_sane(broken.view(), 1.0));
    }

    #[test]
    fn overlap_repair_refuses_a_bad_rule_and_separates_a_clash() {
        let mut x = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.1, 0.0, 0.0]);
        let before = x.clone();
        assert!(!separate_to_overlap_rule(&mut x, None, 0.0, None));
        assert!(!separate_to_overlap_rule(&mut x, None, f64::NAN, None));
        assert_eq!(x, before);
        assert!(separate_to_overlap_rule(&mut x, None, 0.5, None));
        assert!(min_pair_distance(x.view()) >= 0.5);
        let mut one = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(separate_to_overlap_rule(&mut one, None, 0.5, None));
        let groups = vec![vec![0], vec![1]];
        let mut grouped = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.05, 0.0, 0.0]);
        assert!(separate_to_overlap_rule(
            &mut grouped,
            Some(&groups),
            0.4,
            None
        ));
        assert!(min_pair_distance(grouped.view()) >= 0.4);
    }
}
