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
use crate::methods::minima_hopping::EscapeFeedback;
use crate::movekernel::{MoveKernel, ShellRotate, SurfaceRelocate, Symmetrise};
use crate::path::{StallDetector, interpolate_path};
use crate::screen::Screen;

mod config;
mod moves;
mod preset;

pub use config::{Config, Keying, SoapProposalMode};
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

/// One state-changing perturb--quench step taken by the live chain.
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
    /// Quenched energy adopted by the chain.
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
}

impl<'a> ChainCheckpoint<'a> {
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

/// Action returned after observing a live-chain checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointAction {
    /// Leave the live chain and every adaptive controller unchanged.
    Continue,
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
        }
    }

    /// Charges one unit, returning `false` when the budget is gone.
    pub fn charge(&mut self) -> bool {
        if self.spent >= self.budget {
            return false;
        }
        self.spent += 1;
        true
    }

    /// Charges a fraction of one evaluation, for work that touches a subset
    /// of the system.
    ///
    /// A k-atom partial evaluation computes k of the n(n-1)/2 pair rows, so
    /// its honest price is a fraction of a full evaluation. The fraction
    /// accumulates as exact debt and is settled into whole charged units as it
    /// crosses one: deterministic, auditable, and never cheaper than the work
    /// done because the residue is still owed when the run ends.
    pub fn charge_frac(&mut self, frac: f64) -> bool {
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
        self.spent < self.budget
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
    pub fn record(&mut self, value: f64, state: ArrayView1<f64>) {
        if value < self.best {
            self.best = value;
            self.best_state = Some(state.to_owned());
        }
    }

    /// Charged evaluations the ledger was created with.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Charged evaluations remaining.
    pub fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.spent)
    }

    /// Charged evaluations spent.
    pub fn spent(&self) -> usize {
        self.spent
    }

    /// Record one charged relaxation after its caller-owned convergence check.
    ///
    /// `gradient` is fresh validated evidence. Its absence classifies the
    /// boundary as rejected. An invocation that consumed no potential calls is
    /// not a quench boundary and returns `false`.
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
    /// Restarts triggered by a stall.
    pub restarts: usize,
    /// Climbs triggered by a stall.
    pub stall_escapes: usize,
    /// Stall exits taken through the recorded basin entry.
    pub trail_escapes: usize,
    /// Energy gained by those that landed lower than where they left.
    pub stall_escape_gain: f64,
    /// Mean softest eigenvalue over those proposals.
    pub soft_lambda: f64,
    /// Per-rung temperature, basin count and best energy.
    ///
    /// What says whether a ladder is doing its job rather than merely swapping:
    /// a hot rung should register many basins and a poor energy, a cold rung
    /// few basins and a deep one. A ladder where every rung looks alike is a
    /// ladder whose spread is too narrow to be worth its cost.
    pub rungs: Vec<(f64, usize, f64)>,
    /// Swap attempts between adjacent replicas.
    pub swaps_tried: usize,
    /// Hops the acceptance rule took, before any veto.
    pub accepted: usize,
    /// Structures barred from the ledger for not being minima.
    pub unconverged_records: usize,
    /// Per-arm draws, accepts and best quenched value, in library order.
    pub arms: Vec<(String, usize, usize, f64)>,
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
}

/// Relaxes `x`, charging every evaluation, and stopping when the budget ends.
///
/// The relaxation is supplied by the caller because the objective, its
/// gradient and the minimiser are the caller's: this module owns the search,
/// not the numerics under it.
pub type Relax<'a> = &'a mut dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>);

/// Partial relaxation of the listed atoms in the frozen environment.
///
/// Arguments: ledger, structure, moved atom indices, descent steps. Returns
/// the settled structure. The callee owns the objective and therefore the
/// honest fractional price; it charges through [`Ledger::charge_frac`].
pub type Settle<'a> =
    &'a mut dyn FnMut(&mut Ledger, ArrayView1<f64>, &[usize], usize) -> Array1<f64>;

/// Gradient of the objective, charged to the ledger by the caller.
///
/// Optional because only the soft-mode escape needs it: everything else in this
/// driver works from relaxations alone.
pub type GradFn<'g> = dyn FnMut(&mut Ledger, ArrayView1<f64>) -> Option<Array1<f64>> + 'g;

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

fn quench_is_sane(cfg: &Config, energy: f64, x: ArrayView1<f64>) -> bool {
    let minimum = if cfg.species.is_some() {
        OVERLAP_SEPARATION
    } else {
        0.0
    };
    energy.is_finite() && structure_is_sane(x, minimum)
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
) -> Outcome {
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
) -> Outcome {
    let mut checkpoint = continue_without_checkpoint;
    run_full(
        cfg,
        start,
        ledger,
        relax,
        grad,
        None,
        settle,
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
) -> Outcome {
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
        &mut checkpoint,
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
) -> Outcome {
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
        Some(bias),
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
) -> Outcome
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
        Some(bias),
        None,
        Some(checkpoint_interval),
        checkpoint,
        rng,
    )
}

fn run_full<'g, R, H>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    external_bias: Option<&mut BasinBias<ClusterFingerprint>>,
    mut settle: Option<Settle<'_>>,
    checkpoint_interval: Option<usize>,
    checkpoint: &mut H,
    rng: &mut R,
) -> Outcome
where
    R: Rng + ?Sized,
    H: for<'a> FnMut(ChainCheckpoint<'a>) -> CheckpointAction,
{
    let n = cfg.n_points;
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
    let rung_temp = |k: usize| -> f64 {
        if n_rep == 1 {
            cfg.temperature
        } else {
            cfg.temperature * cfg.ladder_top.powf(k as f64 / (n_rep - 1) as f64)
        }
    };
    // The first quench supplies a stable canonical reference before the bias
    // is built. Reporting it as a minimum additionally requires the same
    // geometry and gradient contract as every subsequent quench.
    let (mut e, mut x) = relax(ledger, start, cfg.relax_steps);
    let initial_sane = quench_is_sane(cfg, e, x.view());
    let gradient_required = grad.is_some();
    let initial_validation_gradient = initial_sane.then(|| {
        grad.as_deref_mut().and_then(|g| {
            g(ledger, x.view()).filter(|values| {
                values.iter().fold(0.0_f64, |a, q| a.max(q.abs())) < cfg.record_gradient
            })
        })
    });
    let initial_validation_gradient = initial_validation_gradient.flatten();
    let initial_recordable =
        initial_sane && (!gradient_required || initial_validation_gradient.is_some());
    if initial_recordable {
        ledger.record(e, x.view());
    }
    let mut current_validation_gradient = initial_validation_gradient;
    let canonical_reference = x.clone();
    let mut biases: Vec<BasinBias<ClusterFingerprint>> = (0..n_rep)
        .map(|k| {
            // The coldest rung keeps a token bias so it still recognises
            // revisits; the hottest carries the configured height.
            let h = if cfg.bias_by_rung && n_rep > 1 {
                cfg.bias_height * (rung_temp(k) / cfg.temperature) / cfg.ladder_top
            } else {
                cfg.bias_height
            };
            BasinBias::new(
                ClusterFingerprint::of_config(cfg, &canonical_reference),
                cfg.merge_radius,
                h,
                cfg.bias_gamma,
            )
        })
        .collect();
    // Geometric ladder, so swap acceptance is spaced evenly rather than
    // bunched at one end.
    let temps: Vec<f64> = (0..n_rep).map(rung_temp).collect();
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
    let mut chains: Vec<(f64, Array1<f64>)> = Vec::new();
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

    let mut kernels = cfg.move_library.kernels(cfg);
    // Which kernel to propose from is learned rather than drawn uniformly. The
    // useful move changes as the search moves through the landscape, so the
    // evidence is discounted and a decaying floor keeps every kernel reachable.
    let mut allocator = FlooredThompson::new(kernels.len());
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
    let mut restarts = 0usize;
    let mut symmetrised = 0usize;
    let mut symmetry_gain = 0.0_f64;
    let mut quiet = 0usize;
    let mut longest_quiet = 0usize;
    let mut stall_escapes = 0usize;
    let mut stall_escape_gain = 0.0_f64;
    let mut trail_escapes = 0usize;
    // Evaluated screen state of the trial that entered the current basin,
    // kept as the exit point a stalled chain leaves through.
    let mut basin_entry: Option<Array1<f64>> = None;
    // Learned stall response. Each enabled escape is one arm under the
    // same depth-rewarded Thompson allocation the move set uses: a stall
    // draws one response and the draw is rewarded with the deepening the
    // escape realized, a signed number every stall produces, under a
    // Normal-Gamma posterior whose predictive needs no reward scale. A
    // success bit was the first version and is exactly the sparse reward
    // DepthAllocator documents as failing to switch a wrong arm off.
    // Every arm is domain-free: the trail is the objective's own descent
    // history, the climb needs a gradient alone, the restart draws fresh
    // coordinates. With one arm enabled the allocator is never consulted,
    // so single-flag behaviour is unchanged.
    let stall_arms: Vec<&'static str> = [
        cfg.trail_on_stall.then_some("trail"),
        cfg.escape_on_stall.then_some("climb"),
        cfg.restart_on_stall.then_some("restart"),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut stall_allocator = crate::allocate::DepthAllocator::new(stall_arms.len().max(1));
    let mut soft_lambda = 0.0_f64;
    // The escape scale starts at the move library's own amplitude, so a run
    // without feedback and one with it begin identically.
    let mut feedback = EscapeFeedback::new(1.0, cfg.temperature.max(1e-6));
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
    // Structures kept for path endpoints. Only ones far from every member are
    // added, because interpolating between two structures in one funnel lands
    // back in it, which is what archive-based escape moves holding a single
    // funnel's structures already showed.
    let mut archive: Vec<(f64, Array1<f64>)> = Vec::new();
    let mut paths_run = 0usize;
    let mut path_escapes = 0usize;
    let mut path_improvements = 0usize;
    let mut path_gain = 0.0_f64;

    for _ in 1..n_rep {
        let s0 = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, rng);
        let (e0, x0) = relax(ledger, s0.view(), cfg.relax_steps);
        ledger.record(e0, x0.view());
        chains.push((e0, x0));
    }
    let mut screened_out = 0usize;
    let mut returned = 0usize;
    let mut accepted = 0usize;
    let mut accepted_transitions = Vec::new();
    let mut hops = 0usize;
    let mut checkpoint_hops = 0usize;
    let mut checkpoint_quench_start = 0usize;
    let mut checkpoint_transition_start = 0usize;
    let mut next_checkpoint = checkpoint_interval;

    loop {
        if let (Some(interval), Some(threshold)) = (checkpoint_interval, next_checkpoint)
            && hops > checkpoint_hops
            && ledger.spent() >= threshold
        {
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
            let proposal = match checkpoint_action {
                CheckpointAction::Continue => None,
                CheckpointAction::BoundaryProposal { state, action } => Some((state, action, true)),
                CheckpointAction::ProbeProposal { state, action } => Some((state, action, false)),
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
                if !adopt && from_gradient.is_none() {
                    from_gradient = grad.as_deref_mut().and_then(|g| {
                        g(ledger, from_state.view()).filter(|values| {
                            values.iter().fold(0.0_f64, |a, q| a.max(q.abs())) < cfg.record_gradient
                        })
                    });
                }
                let (proposal_energy, proposal_state) =
                    relax(ledger, state.view(), cfg.relax_steps);
                let proposal_sane = quench_is_sane(cfg, proposal_energy, proposal_state.view());
                let gradient_required = grad.is_some();
                let validation_gradient = if proposal_sane {
                    grad.as_deref_mut().and_then(|g| {
                        g(ledger, proposal_state.view()).filter(|values| {
                            values.iter().fold(0.0_f64, |a, q| a.max(q.abs())) < cfg.record_gradient
                        })
                    })
                } else {
                    None
                };
                let recordable = proposal_sane
                    && (!gradient_required || from_gradient.is_some())
                    && (!gradient_required || validation_gradient.is_some());
                if recordable {
                    if adopt {
                        hops += 1;
                        let improved = proposal_energy < ledger.best - 1e-10;
                        ledger.record(proposal_energy, proposal_state.view());
                        let reached = identity.basin_of(proposal_state.view());
                        let from = here.unwrap_or_else(|| identity.basin_of(from_state.view()));
                        feedback.observe(Some(from), reached);
                        here = Some(reached);
                        e = proposal_energy;
                        x = proposal_state.clone();
                        current_validation_gradient = validation_gradient.clone();
                        accepted += 1;
                        bias.deposit(x.view(), cfg.temperature);
                        if improved && improvements.len() < 512 {
                            improvements.push((
                                hops,
                                ledger.spent(),
                                bias.n_basins(),
                                proposal_energy,
                            ));
                        }
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
                }
            }
        }
        if ledger.remaining() == 0 {
            break;
        }
        // Gap to the incumbent, which is what the law scales the window by.
        let gap = (e - ledger.best).abs().max(1e-12);
        let mut temperature = if cfg.budget_window {
            law.temperature(gap, ledger.remaining())
        } else {
            cfg.temperature
        };
        // The entropy's slope where the chain stands, clamped to a band around
        // the configured value so a slope estimated from few counts cannot
        // freeze the chain or boil it. The band is wide enough that the
        // adaptation has somewhere to go and narrow enough that a bad estimate
        // is survivable.
        if cfg.statistical_temperature
            && let Some(d) = dos.as_ref()
            && d.refreshes > 0
        {
            let (t, _) = d.temperature(e);
            if t.is_finite() && t > 0.0 {
                temperature = t.clamp(0.2 * cfg.temperature, 5.0 * cfg.temperature);
            }
        }

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
        if let MoveLibrary::Molecular { reactive, .. } = &cfg.move_library {
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
                kernels = MoveLibrary::Molecular {
                    groups: movable,
                    reactive: *reactive,
                }
                .kernels(cfg);
            }
        }
        let mut k = match (&context, cfg.allocate_moves) {
            (Some(c), _) => contextual.select(c.view(), rng),
            (None, true) if cfg.depth_reward => depth_allocator.select(rng),
            (None, true) => allocator.select(rng),
            (None, false) => rng.random_range(0..kernels.len()),
        };
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
        let escape = if cfg.minima_hopping {
            feedback.escape()
        } else {
            1.0
        };
        // Ordinary hops: scale the library move by the escape feedback. Soft
        // mode climbs live under `escape_on_stall` below; they are a few per
        // cent of the budget when the chain has stopped improving, not the
        // default proposal.
        let mut trial = if cov_fire {
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
                _ => kernels[k].propose_scaled(x.view(), cfg.temperature, escape, rng),
            }
        };
        // Identity SOAP is not an arm. Thompson must not buy a no-op:
        // reselect a real kernel, or skip the hop.
        if !cov_fire
            && !soft_fire
            && !angular
            && kernels[k].name() == "soap"
            && hop_is_identity(x.view(), trial.view())
        {
            let mut swapped = false;
            for _ in 0..kernels.len() {
                let k2 = match (&context, cfg.allocate_moves) {
                    (Some(c), _) => contextual.select(c.view(), rng),
                    (None, true) if cfg.depth_reward => depth_allocator.select(rng),
                    (None, true) => allocator.select(rng),
                    (None, false) => rng.random_range(0..kernels.len()),
                };
                if kernels[k2].name() == "soap" {
                    continue;
                }
                k = k2;
                trial = match (&mut constructor, &kernels[k]) {
                    (Some(c), ClusterMove::Reseed { n_points, .. }) => {
                        let (cand, f) = c.propose(x.view(), *n_points, rng);
                        pending_features = Some(f);
                        cand
                    }
                    _ => kernels[k].propose_scaled(x.view(), cfg.temperature, escape, rng),
                };
                swapped = true;
                break;
            }
            if !swapped {
                continue;
            }
        }
        let proposal_action = if cov_fire {
            "covariance".to_owned()
        } else if soft_fire && soft_cache.is_some() {
            "soft_mode".to_owned()
        } else if angular {
            "angular".to_owned()
        } else {
            kernels[k].name().to_owned()
        };
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
                contain(&mut trial, n, cfg.container);
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
            let gnorm = match grad.as_deref_mut().and_then(|g| g(ledger, trial.view())) {
                Some(v) => v.iter().fold(0.0_f64, |a, q| a + q * q).sqrt(),
                None => 0.0,
            };
            pending_raw = Some((raw_y, gnorm));
            // The first stage speaks only where the posterior is sharp against
            // the temperature that scales the acceptance ratio.
            let tol = cfg.surrogate_tolerance * temperature;
            match sur.predict_full(trial.view(), n, raw_y, gnorm, tol) {
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
        if stage_one_reject {
            // The chain stays. A rejected proposal still deposits on where the
            // chain stands, exactly as a rejection through the ordinary
            // acceptance test does, so the bias accumulates at the same rate.
            hops += 1;
            bias.deposit(x.view(), temperature);
            continue;
        }
        let (e_screen, x_screen) = relax(ledger, trial.view(), cfg.screen_steps);
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
        let returning = cfg.return_screen && {
            let ds = bias.cv(x_screen.view());
            let dc = bias.cv(x.view());
            let d: f64 = ds
                .iter()
                .zip(dc.iter())
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                .sqrt();
            d < cfg.merge_radius
        };
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
        let feats = {
            let ds = bias.cv(x_screen.view());
            let dc = bias.cv(x.view());
            let drift: f64 = ds
                .iter()
                .zip(dc.iter())
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                .sqrt();
            Array1::from(vec![1.0, e_screen, drift, e_screen * drift])
        };
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
        } else {
            e_screen > ledger.best + cfg.screen_margin
        };
        let (e_new, x_new) = if returning
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
            let out = relax(ledger, x_screen.view(), cfg.relax_steps);
            if cfg.bayes_screen {
                // The answer to the question the posterior was asked.
                screen.observe(feats.view(), out.0);
            }
            out
        };
        if !quench_is_sane(cfg, e_new, x_new.view()) {
            unconverged_records += 1;
            hops += 1;
            bias.deposit(x.view(), temperature);
            if cfg.max_hops.is_some_and(|cap| hops >= cap) {
                break;
            }
            continue;
        }
        // A screened or returning structure has only a partial relaxation.
        // It can train screens and count as a rejected attempt, but the live
        // chain remains on the quenched landscape under every acceptance law.
        let unquenched = screened_this || returning;
        // Every quench trains the surrogate, including the ones taken before
        // it had an opinion. This is where its training data comes from.
        if let (Some(sur), Some((raw_y, gnorm))) = (surrogate.as_mut(), pending_raw.take()) {
            sur.observe_full(trial.view(), n, raw_y, gnorm, e_new);
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
        let gradient_required = grad.is_some();
        let validation_gradient = if unquenched {
            None
        } else {
            grad.as_deref_mut().and_then(|g| {
                g(ledger, x_new.view()).filter(|values| {
                    values.iter().fold(0.0_f64, |a, q| a.max(q.abs())) < cfg.record_gradient
                })
            })
        };
        let recordable = !unquenched && (!gradient_required || validation_gradient.is_some());
        if recordable {
            let improved_record = e_new < ledger.best - 1e-10;
            ledger.record(e_new, x_new.view());
            // A new best is the one state worth sharing, and the census a
            // shared run identifies basins in was calibrated from quenches
            // at the share tolerance, which the ordinary record tolerance
            // does not reach. Polishing only improvements bounds the cost
            // to the number of improvement events, every evaluation charged
            // to this ledger, and the polished state enters the boundary
            // record the checkpoint offer loop already reads.
            if cfg.polish_records > 0 && improved_record {
                // The relax closure owns boundary recording and share-grade
                // validation; recording a second boundary here counted the
                // polish twice and tripped the checkpoint accounting. The
                // library's whole contribution is the extra descent, bounded
                // by the number of improvement events and charged like any
                // other relaxation.
                let (polished_energy, polished_state) =
                    relax(ledger, x_new.view(), cfg.polish_records);
                if polished_energy <= e_new {
                    ledger.record(polished_energy, polished_state.view());
                }
            }
        } else {
            unconverged_records += 1;
        }
        hops += 1;
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
        let here_before = if cfg.track_funnels {
            Some(*here.get_or_insert_with(|| identity.basin_of(x.view())))
        } else {
            None
        };
        let s_old = bias.cv(x.view());
        let s_new = bias.cv(x_new.view());
        // Biased rise. The bias is part of the landscape the chain walks; a
        // threshold or Metropolis on raw energy alone ignores the deposits and
        // re-enters filled basins freely. Measured: MH accepting on raw delta
        // solved 1 of 4 LJ38 seeds at 400k where the biased Metropolis path
        // solved the same seed in ~10k hops.
        //
        // When track_funnels is on, the spectral term is well-tempered MetaD
        // on the Fiedler coordinate of the hop graph (SpectralBias): it fills
        // the *funnel* the chain is stuck in, not only the current basin.
        let v_old = bias.potential(s_old.view())
            + spectral
                .as_ref()
                .map(|sp| sp.potential(sp.cv(x.view()).view()))
                .unwrap_or(0.0);
        let v_new = bias.potential(s_new.view())
            + spectral
                .as_ref()
                .map(|sp| sp.potential(sp.cv(x_new.view()).view()))
                .unwrap_or(0.0);
        let delta = (e_new + v_new) - (e + v_old);
        let accept = if !recordable {
            if cfg.minima_hopping {
                let from = *here.get_or_insert_with(|| identity.basin_of(x.view()));
                feedback.observe(Some(from), from);
            }
            false
        } else if cfg.minima_hopping {
            let from = *here.get_or_insert_with(|| identity.basin_of(x.view()));
            // Threshold on the *biased* rise. Adapts like Goedecker's E_diff
            // while still feeling the per-basin deposits.
            let reached = identity.basin_of(x_new.view());
            feedback.observe(Some(from), reached);
            let ok = feedback.accept(delta);
            if ok {
                here = Some(reached);
            }
            ok
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
        if accept {
            accepted += 1;
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
        if cfg.energy_bias {
            let occupied = if accept { e_new } else { e };
            if occupied.is_finite() {
                match ebias.as_mut() {
                    Some(b) => {
                        b.deposit(occupied, temperature);
                        if std::env::var("EBIAS_TRACE").is_ok() && b.deposits % 200 == 0 {
                            eprintln!("ebias deposits {} peak {:.4}", b.deposits, b.peak());
                        }
                    }
                    None => {
                        flat_seen.push(occupied);
                        if flat_seen.len() >= flat_sweep {
                            ebias = crate::dos::EnergyBias::from_sample(
                                &flat_seen,
                                temperature,
                                crate::dos::BINS,
                            );
                            flat_seen.clear();
                        }
                    }
                }
            }
        }
        if cfg.flat_histogram || cfg.statistical_temperature {
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
            allocator.update(k, improved || accept);
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
        if accept {
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
            // Every accepted minimum, appended as one line of energy then
            // flat coordinates, for landscape projections. A campaign sets
            // ANNEAL_MIN_DUMP to a path prefix; the driver never reads it
            // back.
            if let Ok(prefix) = std::env::var("ANNEAL_MIN_DUMP") {
                use std::io::Write as _;
                if let Ok(mut fh) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&prefix)
                {
                    let mut line = format!("{e_new:.8}");
                    for v in x_new.iter() {
                        line.push_str(&format!(" {v:.6}"));
                    }
                    line.push('\n');
                    let _ = fh.write_all(line.as_bytes());
                }
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
            e = e_new;
            x = x_new;
            current_validation_gradient = validation_gradient;
        } else if cfg.budget_window {
            // The biased delta, which is what the chain actually declined, not
            // the raw energy difference. The bias is part of the barrier the
            // chain faces, and estimating the barrier without it measures a
            // landscape the chain is not walking on.
            law.observe_rejection(delta);
        }
        bias.deposit(bias.cv(x.view()).view(), temperature);
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
        if cfg.track_funnels {
            // Accepted hops only. A rejected proposal says the chain declined
            // to move, which is a statement about the acceptance rule rather
            // than about reachability.
            if accept && let Some(prev) = here_before {
                let now = identity.basin_of(x.view());
                funnels.record(prev, now);
                here = Some(now);
            }
            if funnels.pending() >= cfg.funnel_period && funnels.len() >= 8 {
                funnel_split = funnels.split().ok();
            }
        }
        if cfg.symmetrise_on_stall && stuck {
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
                    Some(crate::symmetrise::symmetrise_group(
                        x.view(),
                        n,
                        &group,
                        cfg.symmetry_merge_radius,
                    ))
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
                    ledger.record(es, xs.view());
                    hops += 1;
                    symmetrised += 1;
                    if es < e {
                        symmetry_gain += e - es;
                        e = es;
                        x = xs;
                        here = None;
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
        // back on Mackay. The manuscript stall step is tabu only.
        // A directed leave is `escape_on_stall` (Goedecker / Schönborn).
        let stall_response = if stuck && !stall_arms.is_empty() {
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
            ledger.record(ef, xf.view());
            hops += 1;
            restarts += 1;
            e = ef;
            x = xf;
            here = None;
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
            ledger.record(ee, xe.view());
            hops += 1;
            stall_escapes += 1;
            trail_escapes += 1;
            if ee < e {
                stall_escape_gain += e - ee;
            }
            // Taken whatever its energy, exactly as the climb below: the
            // chain has shown it cannot improve from where it stands.
            e = ee;
            x = xe;
            here = None;
            stall_outcome = Some(stalled_from - e);
        }
        if stall_response.map(|arm| stall_arms[arm] == "climb") == Some(true) {
            quiet = 0;
            longest_quiet = 0;
            if let Some(g) = grad.as_deref_mut() {
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
                    ledger.record(ee, xe.view());
                    hops += 1;
                    stall_escapes += 1;
                    if ee < e {
                        stall_escape_gain += e - ee;
                    }
                    // Taken whatever its energy. The chain has already shown it
                    // cannot improve from where it is, so the value of the new
                    // structure is that it is somewhere else.
                    if cfg.minima_hopping {
                        let from = *here.get_or_insert_with(|| identity.basin_of(x.view()));
                        let reached = identity.basin_of(xe.view());
                        feedback.observe(Some(from), reached);
                        here = Some(reached);
                    }
                    e = ee;
                    x = xe;
                    if !cfg.minima_hopping {
                        here = None;
                    }
                    stall_outcome = Some(stalled_from - e);
                }
            }
        }
        if let (Some(arm), Some(gain)) = (stall_response, stall_outcome) {
            stall_allocator.update(arm, gain);
        }

        if n_rep > 1 {
            since_swap += 1;
            if since_swap >= cfg.swap_period {
                since_swap = 0;
                // Park the active rung, offer a swap with the next, then make
                // that one active. Each rung keeps its own bias and its own
                // temperature; only the states move, so a hot rung's crossing
                // lands in a cold rung that can polish it.
                chains.insert(rep, (e, x.clone()));
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
                let k = rep;
                let j = (rep + 1) % n_rep;
                if k != j {
                    swaps_tried += 1;
                    // Bias exchange, not plain parallel tempering.
                    //
                    // Each rung carries its own accumulating bias, so each
                    // samples exp(-(E + V_k)/T_k) rather than exp(-E/T_k), and
                    // a swap acceptance built from raw energies is exchanging
                    // between distributions neither chain is sampling. The
                    // measured symptom was a ladder that never stratified:
                    // four rungs with 141 to 179 basins each and energies not
                    // ordered by temperature at all.
                    //
                    // The correct factor evaluates each rung's bias at both
                    // states (Piana and Laio):
                    //
                    //   ln a = (1/T_k)[U_k(x_k) - U_k(x_j)]
                    //        + (1/T_j)[U_j(x_j) - U_j(x_k)]
                    //
                    // with U_k(x) = E(x) + V_k(x). It reduces to the plain
                    // Metropolis swap when the biases are equal, which is what
                    // Exchange supplies and what this generalises.
                    let (ek, xk) = (chains[k].0, chains[k].1.clone());
                    let (ej, xj) = (chains[j].0, chains[j].1.clone());
                    let vk_xk = biases[k].potential(biases[k].cv(xk.view()).view());
                    let vk_xj = biases[k].potential(biases[k].cv(xj.view()).view());
                    let vj_xj = biases[j].potential(biases[j].cv(xj.view()).view());
                    let vj_xk = biases[j].potential(biases[j].cv(xk.view()).view());
                    let log_a = ((ek + vk_xk) - (ej + vk_xj)) / temps[k].max(1e-12)
                        + ((ej + vj_xj) - (ek + vj_xk)) / temps[j].max(1e-12);
                    let p = if log_a >= 0.0 { 1.0 } else { log_a.exp() };
                    if rng.random::<f64>() < p {
                        swaps_accepted += 1;
                        chains.swap(k, j);
                    }
                }
                rep = j;
                let (ne, nx) = chains.remove(rep);
                e = ne;
                x = nx;
                bias = biases.remove(rep);
            }
        }

        if let Some(cap) = cfg.max_hops
            && hops >= cap
        {
            break;
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
                    let out = interpolate_path(
                        x.view(),
                        t.view(),
                        cfg.path_images,
                        |img| {
                            if ledger.remaining() == 0 {
                                return None;
                            }
                            let (ev, xv) = relax(ledger, img, cfg.relax_steps);
                            ledger.record(ev, xv.view());
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
                            e = esc.energy;
                            x = esc.state.clone();
                        }
                    }
                }
            }
        }
    }

    if checkpoint_interval.is_some() && hops > checkpoint_hops {
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
        };
        let _ = checkpoint(snapshot);
    }

    let n_basins = bias.n_basins();
    let final_radius = bias.merge_radius();
    if let Some(slot) = carried {
        // Handed back so the next chain inherits what this one learned.
        *slot = bias;
    }

    Outcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        final_state: Some(x.clone()),
        final_energy: e,
        accepted_transitions,
        hops,
        screened_out,
        basins: n_basins,
        charged: ledger.spent(),
        returned,
        escape_scale: feedback.escape(),
        escape_threshold: feedback.threshold(),
        visit_counts: (feedback.n_same, feedback.n_known, feedback.n_new),
        soft_perturbs: soft_fired,
        soft_subspaces: soft_recomputes,
        soft_escapes,
        soft_crossed,
        improvements,
        angular: (angular_tried, angular_accepted, angular_ratio),
        contextual: (contextual.picks.clone(), contextual.forced),
        screen: (
            screen.decided,
            screen.relaxed,
            screen.explored,
            screen.observations(),
        ),
        tabu: (tabu.len(), tabu_hits),
        funnel: funnel_split.as_ref().map(|p| {
            let (a, b) = p.sizes();
            (a, b, p.connectivity)
        }),
        symmetrised: (symmetrised, symmetry_gain),
        restarts,
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
            let mut energies = chains.iter().map(|(en, _)| *en);
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
        swaps_accepted,
        paths: paths_run,
        path_escapes,
        path_improvements,
        path_gain,
    }
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
pub fn optimize(cfg: &Config, ledger: &mut Ledger, relax: Relax<'_>, seed: u64) -> Outcome {
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
) -> Outcome {
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
) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        cfg.n_points,
        cfg.start_radius(),
        cfg.min_separation,
        &mut rng,
    );
    run_with_gradient(cfg, start.view(), ledger, relax, grad, &mut rng)
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn derived_replaces_the_two_hand_set_scalars() {
        let rec = Config::recommended(13);
        let der = Config::derived(13);
        assert!(matches!(der.move_library, MoveLibrary::LeanBurst));
        assert!(der.allocate_moves && der.depth_reward && der.tabu_on_stall);
        assert!(der.escape_on_stall);
        assert!(!der.restart_on_stall);
        assert!(der.budget_window);
        assert!(der.bayes_screen);
        assert!(!rec.budget_window);
        assert!(!rec.bayes_screen);
        let want = crate::screen::cost_asymmetric_threshold(der.screen_steps, der.relax_steps);
        assert!((der.bayes_threshold - want).abs() < 1e-12);
        assert!((der.bayes_threshold - 7.0 / 15.0).abs() < 1e-12);
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
        );
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
        );
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
        let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng);
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
        let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng);
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
        let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng);

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
        );

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
        );

        assert!(out.best.is_infinite());
        assert!(out.best_state.is_none());
        assert!(out.improvements.is_empty());
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
        );

        assert_eq!(out.best, 0.0);
        assert!(
            out.improvements
                .iter()
                .all(|(_, _, _, energy)| *energy >= 0.0),
            "unconverged trial entered the encounter trace: {:?}",
            out.improvements
        );
    }

    #[test]
    fn respects_the_ledger() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(500);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 0);
        assert!(out.charged <= 500, "spent {} of 500", out.charged);
        assert_eq!(ledger.remaining(), 0, "a run should spend its budget");
    }

    #[test]
    fn outcome_keeps_live_energy_and_accepted_trajectory_edges() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 19);

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
        );

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
        let out = optimize(&cfg, &mut ledger, &mut relax, 11);
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
    fn screening_skips_full_relaxation_but_completes_the_hop() {
        let mut cfg = Config::for_cluster(6);
        cfg.screen_margin = -1.0e12;
        let mut ledger = Ledger::new(4000);
        let mut relaxation_steps = Vec::new();
        let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
            relaxation_steps.push(steps);
            toy_relax(ledger, x, steps)
        };
        let out = optimize(&cfg, &mut ledger, &mut relax, 1);
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
        let out = optimize(&cfg, &mut ledger, &mut relax, 2);
        assert!(out.basins >= 1, "at least the starting basin must register");
    }

    #[test]
    fn seeds_are_reproducible() {
        let cfg = Config::for_cluster(6);
        let run_once = |seed| {
            let mut ledger = Ledger::new(2000);
            let mut relax = toy_relax;
            optimize(&cfg, &mut ledger, &mut relax, seed).best
        };
        assert_eq!(run_once(7), run_once(7), "same seed must give same result");
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
    /// Sorted pairwise distances, compared by Euclidean distance.
    Spectrum(SortedPairs),
    /// Coordinates, for a metric that does its own matching.
    Coordinates,
    /// Sorted per-point pair energies, keying on how well each point is bound
    /// rather than on how far apart the points are.
    Sites(SiteEnergies),
    /// Coordinates put in a canonical order against a fixed reference, so
    /// Euclidean distance between two of them is a shape distance.
    #[cfg(feature = "ira")]
    Canonical(Box<crate::shape::CanonicalOrder>),
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

    /// The descriptor for a named keying, against `reference`.
    pub fn of_with(n_points: usize, keying: Keying, reference: &Array1<f64>) -> Self {
        #[cfg(not(feature = "ira"))]
        let _ = reference;
        match keying {
            Keying::Shape => ClusterFingerprint::Coordinates,
            Keying::Distances => ClusterFingerprint::Spectrum(SortedPairs { n_points }),
            Keying::Sites => ClusterFingerprint::Sites(SiteEnergies { n_points }),
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
        match effective_keying(cfg) {
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
            other => Self::of_with(cfg.n_points, other, reference),
        }
    }
}

impl Fingerprint for ClusterFingerprint {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        match self {
            ClusterFingerprint::Spectrum(s) => s.describe(x),
            ClusterFingerprint::Coordinates => x.to_owned(),
            ClusterFingerprint::Sites(s) => s.describe(x),
            #[cfg(feature = "ira")]
            ClusterFingerprint::Canonical(c) => c.describe(x),
            #[cfg(feature = "featomic")]
            ClusterFingerprint::SoapMean { rcut, species } => {
                crate::featomic_hop::soap_cloud_mean(x, *rcut, species.as_deref(), None)
            }
        }
    }
}
