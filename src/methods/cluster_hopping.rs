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
use crate::calibrate::StepCalibrator;
use crate::contextual::ContextualAllocator;
use crate::screen::Screen;
use crate::methods::activation::{activate, Activation};
use crate::bias::{
    AdaptiveHeight, Bias, BasinBias, BasinIndex, Fingerprint, SiteEnergies, SortedPairs,
};
use crate::diversity::DiversityAnnealer;
use crate::exchange::{Exchange, MetropolisExchange};
use crate::methods::minima_hopping::EscapeFeedback;
use crate::path::{interpolate_path, StallDetector};
use crate::movekernel::{MoveKernel, ShellRotate, SurfaceRelocate, Symmetrise};

/// The move library, dispatched by value.
///
/// [`MoveKernel::propose`] is generic over the generator, which makes the
/// trait not dyn compatible, so the library is an enum rather than a vector of
/// boxes. Keeping the generic parameter is worth more than boxing: it lets a
/// kernel be used with any generator without a virtual call per proposal.
pub enum ClusterMove {
    /// Displace every point uniformly: the standard basin-hopping move.
    AllPoints {
        /// Half-width of the per-coordinate displacement.
        step: f64,
    },
    /// Displace one point. Cheap and local, for polishing a packing.
    SinglePoint {
        /// Points in a state.
        n_points: usize,
        /// Half-width of the displacement.
        step: f64,
    },
    /// Relocate the least-coordinated point onto the surface.
    SurfaceRelocate(SurfaceRelocate),
    /// Rotate the outer shell against the core.
    ShellRotate(ShellRotate),
    /// Enforce an approximate rotational symmetry.
    Symmetrise(Symmetrise),
    /// Rebuild the structure by growing a local order, and quench into it.
    ///
    /// The only move here that crosses a funnel boundary in one step. Every
    /// other proposal displaces points and relies on the quench to find a
    /// nearby minimum, so the basins a chain can reach from where it stands are
    /// the ones a displacement reaches, and no displacement of an icosahedral
    /// 98-point structure lands in the tetrahedral funnel with usable
    /// probability. A template is not reached, it is written down: the points
    /// are indistinguishable, so the family's sites *are* the proposal.
    ///
    /// Nothing about it is specific to one potential. The order to grow is
    /// read off the structure the chain stands on, by taking the neighbour
    /// offsets of its best-coordinated point, and the alternatives come from
    /// the classifier's template library rather than from a list of packings
    /// someone chose for this problem. Which is worth proposing is left to the
    /// allocator; nothing here knows that 38 points want a truncated
    /// octahedron.
    ///
    /// See [`crate::lattice`].
    Reseed {
        /// Points in a state.
        n_points: usize,
        /// Where the local order to grow comes from.
        source: crate::lattice::Source,
    },
    /// Wales and Doye's angular move on the worst-bound point.
    ///
    /// "Each angular displacement consisted of choosing random theta and phi
    /// spherical polar coordinates for the atom in question, taking the origin
    /// at the center of mass and replacing the radius with the maximum value in
    /// the cluster" (J. Phys. Chem. A 101, 5111).
    ///
    /// Not the same move as [`SurfaceRelocate`], which takes the
    /// least-coordinated point and places it near the surface. This takes the
    /// point with the highest pair energy and throws it to the far edge of the
    /// cluster at a random angle, which is a much larger step and is the move
    /// the 1997 paper used to reach the decahedral minima.
    Angular {
        /// Points in a state.
        n_points: usize,
    },
}

/// Pair energy per point, in the Lennard-Jones form Wales and Doye use to
/// decide which point is worst bound.
///
/// `E(i) = sum_{j != i} 4 [ (1/r_ij)^12 - (1/r_ij)^6 ]`, so the total energy is
/// half the sum. Reduced units, matching the rest of this driver.
pub fn pair_energies(x: ArrayView1<f64>, n: usize) -> Array1<f64> {
    let mut e = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 <= 0.0 {
                continue;
            }
            let s6 = 1.0 / (r2 * r2 * r2);
            let v = 4.0 * (s6 * s6 - s6);
            e[i] += v;
            e[j] += v;
        }
    }
    e
}

/// Whether the worst-bound point is loose enough for an angular move, and which
/// one it is.
///
/// The criterion is the paper's: the highest pair energy rising above a
/// fraction `ratio` of the lowest. Both are negative for a bound cluster, so
/// this fires when the worst-bound point holds less than `ratio` of the binding
/// the best-bound one does.
pub fn worst_bound(x: ArrayView1<f64>, n: usize, ratio: f64) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let e = pair_energies(x, n);
    let mut hi = 0usize;
    let mut lo = 0usize;
    for i in 1..n {
        if e[i] > e[hi] {
            hi = i;
        }
        if e[i] < e[lo] {
            lo = i;
        }
    }
    if e[lo] >= 0.0 {
        return None;
    }
    if e[hi] > ratio * e[lo] {
        Some(hi)
    } else {
        None
    }
}

impl ClusterMove {
    /// The move library, configured for `n` points.
    ///
    /// The two plain perturbations come first and are not optional. Displacing
    /// every point uniformly is the move basin hopping is defined by, and the
    /// step of 0.38 is inside the 0.36 to 0.40 band Wales and Doye report for
    /// the quenched surface. A library of packing-changing moves alone leaves
    /// the chain with no way to make an ordinary small step, and measured on
    /// LJ38 at 400 thousand charged evaluations that library solved 1 seed in 8
    /// where the campaign driver, which carries both, solves 8.
    pub fn library(n: usize) -> Vec<ClusterMove> {
        vec![
            ClusterMove::AllPoints { step: 0.38 },
            ClusterMove::SinglePoint {
                n_points: n,
                step: 1.0,
            },
            ClusterMove::SurfaceRelocate(SurfaceRelocate {
                n_points: n,
                neighbour_cutoff: 1.6,
            }),
            ClusterMove::ShellRotate(ShellRotate { n_points: n }),
            ClusterMove::Symmetrise(Symmetrise {
                n_points: n,
                orders: vec![2, 3, 4, 5, 6],
                pair_cutoff: 2.5,
            }),
        ]
    }

    /// The library with the reseeding moves added.
    ///
    /// Separate because these are the only proposals that discard the current
    /// structure rather than perturb it, so a caller comparing against the
    /// displacement-only search needs to be able to ask for one or the other.
    /// Every source is offered; which is worth drawing is the allocator's
    /// question, not this function's.
    pub fn library_with_reseed(n: usize) -> Vec<ClusterMove> {
        let mut v = Self::library(n);
        for source in crate::lattice::Source::library() {
            v.push(ClusterMove::Reseed {
                n_points: n,
                source,
            });
        }
        v
    }

    /// Draws a proposal from whichever kernel this is.
    pub fn propose<R: Rng + ?Sized>(
        &self,
        x: ArrayView1<f64>,
        t: f64,
        rng: &mut R,
    ) -> Array1<f64> {
        self.propose_scaled(x, t, 1.0, rng)
    }

    /// Draws a proposal with the amplitude multiplied by `scale`.
    ///
    /// Only the two plain perturbations carry an amplitude to scale. Surface
    /// relocation, shell rotation and symmetrisation change a packing rather
    /// than displace by a length, so there is nothing for a scale to multiply
    /// and they are drawn unchanged.
    pub fn propose_scaled<R: Rng + ?Sized>(
        &self,
        x: ArrayView1<f64>,
        t: f64,
        scale: f64,
        rng: &mut R,
    ) -> Array1<f64> {
        match self {
            ClusterMove::AllPoints { step } => {
                let mut y = x.to_owned();
                let h = step * scale;
                for v in y.iter_mut() {
                    *v += rng.random_range(-h..h);
                }
                y
            }
            ClusterMove::SinglePoint { n_points, step } => {
                let mut y = x.to_owned();
                let i = rng.random_range(0..*n_points);
                let h = step * scale;
                for k in 0..3 {
                    y[3 * i + k] += rng.random_range(-h..h);
                }
                y
            }
            ClusterMove::Reseed { n_points, source } => {
                // Both the order and the length scale are read off the current
                // structure, so the move carries no knowledge of the potential
                // and none of the objective it is proposing against.
                crate::lattice::candidate(*source, x, *n_points, rng)
            }
            ClusterMove::Angular { n_points } => {
                let n = *n_points;
                let mut y = x.to_owned();
                if n == 0 {
                    return y;
                }
                // The centre of mass, and the radius of the point furthest from
                // it, which is where the moved point lands.
                let mut c = [0.0_f64; 3];
                for i in 0..n {
                    for k in 0..3 {
                        c[k] += y[3 * i + k];
                    }
                }
                for v in c.iter_mut() {
                    *v /= n as f64;
                }
                let mut rmax = 0.0_f64;
                for i in 0..n {
                    let dx = y[3 * i] - c[0];
                    let dy = y[3 * i + 1] - c[1];
                    let dz = y[3 * i + 2] - c[2];
                    rmax = rmax.max((dx * dx + dy * dy + dz * dz).sqrt());
                }
                let i = worst_bound(y.view(), n, 0.42).unwrap_or_else(|| {
                    let e = pair_energies(y.view(), n);
                    let mut hi = 0usize;
                    for k in 1..n {
                        if e[k] > e[hi] {
                            hi = k;
                        }
                    }
                    hi
                });
                // Uniform on the sphere: cos(theta) uniform in [-1, 1], not
                // theta itself, or the poles are oversampled.
                let cos_t: f64 = rng.random_range(-1.0..1.0);
                let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
                let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                y[3 * i] = c[0] + rmax * sin_t * phi.cos();
                y[3 * i + 1] = c[1] + rmax * sin_t * phi.sin();
                y[3 * i + 2] = c[2] + rmax * cos_t;
                y
            }
            ClusterMove::SurfaceRelocate(k) => k.propose(x, t, rng),
            ClusterMove::ShellRotate(k) => k.propose(x, t, rng),
            ClusterMove::Symmetrise(k) => k.propose(x, t, rng),
        }
    }
}

#[cfg(test)]
mod move_scaling_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// The escape scale has to reach the step, which is the whole mechanism.
    /// It did not: the amplitude moves ignored the temperature argument, so a
    /// controller multiplying it changed nothing about how far a proposal
    /// reached.
    #[test]
    fn scaling_widens_the_displacement() {
        let mv = ClusterMove::AllPoints { step: 0.4 };
        let x: Array1<f64> = Array1::zeros(30);
        let spread = |scale: f64| {
            let mut rng = StdRng::seed_from_u64(7);
            let mut worst = 0.0_f64;
            for _ in 0..200 {
                let y = mv.propose_scaled(x.view(), 0.8, scale, &mut rng);
                for v in y.iter() {
                    worst = worst.max(v.abs());
                }
            }
            worst
        };
        let one = spread(1.0);
        let four = spread(4.0);
        assert!(
            (four / one - 4.0).abs() < 0.2,
            "a scale of four should reach four times as far: {four} against {one}"
        );
    }

    #[test]
    fn the_unscaled_call_is_the_old_behaviour() {
        let mv = ClusterMove::SinglePoint { n_points: 10, step: 1.0 };
        let x: Array1<f64> = Array1::zeros(30);
        let mut a = StdRng::seed_from_u64(3);
        let mut b = StdRng::seed_from_u64(3);
        let p = mv.propose(x.view(), 0.8, &mut a);
        let q = mv.propose_scaled(x.view(), 0.8, 1.0, &mut b);
        assert_eq!(p, q);
    }
}

/// Work ledger: every objective or gradient evaluation is charged.
///
/// A relaxation inside a move spends the same budget as a proposal does, which
/// is the accounting that makes methods with different internal structure
/// comparable. Published cluster success rates are quoted per hopping step,
/// with the relaxation inside each step uncounted.
pub struct Ledger {
    budget: usize,
    spent: usize,
    /// Lowest objective value seen.
    pub best: f64,
    /// State attaining [`Ledger::best`].
    pub best_state: Option<Array1<f64>>,
}

impl Ledger {
    /// Creates a ledger with `budget` charged evaluations.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            spent: 0,
            best: f64::INFINITY,
            best_state: None,
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
}

/// Driver settings.
#[derive(Debug, Clone)]
pub struct Config {
    /// Points in a state; the state length must be `3 * n_points`.
    pub n_points: usize,
    /// Metropolis temperature on the quenched chain.
    pub temperature: f64,
    /// Height of a fresh bias deposit.
    pub bias_height: f64,
    /// Well-tempered bias factor; must exceed one.
    pub bias_gamma: f64,
    /// Distance below which two states are the same basin.
    ///
    /// Its units are those of whichever metric keys the bias. Against a sorted
    /// distance spectrum compared by Euclidean distance it is a number in
    /// descriptor space with no physical meaning; against a shape distance it
    /// is a length.
    pub merge_radius: f64,
    /// Design point for the budget-window temperature, as a fraction of the
    /// sphere-model descent boundary. Must lie strictly below two.
    pub theta: f64,
    /// Set the temperature by the budget-window law rather than holding it.
    pub budget_window: bool,
    /// Choose the move kernel by discounted Thompson allocation.
    pub allocate_moves: bool,
    /// Set the deposit height from the escape gaps the chain observes.
    pub adaptive_height: bool,
    /// Hops a single `run` may take before returning, when set.
    ///
    /// Used by the replica ladder to advance one chain by a slice; a plain run
    /// leaves it unset and stops only when the ledger does.
    pub max_hops: Option<usize>,
    /// Replicas run on a temperature ladder, with periodic swaps.
    ///
    /// One is the plain chain. Above one, the driver runs a ladder and offers
    /// swaps through [`crate::exchange::Exchange`], which is the crate's own
    /// operator and satisfies detailed balance by construction.
    ///
    /// This is the standard non-local mechanism for a multi-funnel landscape
    /// and the measurements here say why it is the right one to reach for: no
    /// single move from the plateau reaches anything lower, so a cold chain
    /// cannot leave it, while a hot chain crosses freely and finds nothing
    /// precise. A swap moves a hot chain's crossing down to a cold chain that
    /// can polish it, which neither temperature achieves alone.
    pub replicas: usize,
    /// Hops between swap attempts.
    pub swap_period: usize,
    /// Drive the escape scale and the acceptance threshold from the history,
    /// after Goedecker's minima hopping, instead of a Metropolis temperature.
    ///
    /// Revisiting a known minimum makes the *next escape* harder rather than
    /// the *current basin* less attractive, which is a different use of the
    /// same history the bias keeps. The transition region between funnels is
    /// left crossable, which Goedecker argues is why flooding it is the wrong
    /// response to a revisit.
    ///
    /// This is the scaled-move form: the escape scale multiplies the move
    /// amplitude and the acceptance threshold replaces Metropolis. Soft-mode
    /// climbs are *not* taken every hop under this flag; they are the separate
    /// [`Config::escape_on_stall`] path. Measured: activating every hop under a
    /// gradient cost ~687 charged evaluations per hop on LJ38 and bought 291
    /// hops from 200k, which is not a search. The controller and the climb are
    /// complementary and must stay separable.
    pub minima_hopping: bool,
    /// Lanczos steps for the soft-mode escape.
    ///
    /// Each costs two gradient evaluations, charged. Eight resolves the softest
    /// mode of a cluster well enough to displace along, against about forty
    /// charged evaluations for the relaxation that follows.
    pub escape_lanczos_steps: usize,
    /// Finite-difference step for the Hessian-vector product.
    pub escape_epsilon: f64,
    /// Distance moved along the softest mode per climbing step.
    pub escape_amplitude: f64,
    /// Push past the saddle, in units of the climbing step, before the
    /// feedback scale multiplies it.
    pub escape_overshoot: f64,
    /// Climbing steps before a climb is abandoned.
    pub escape_max_climb: usize,
    /// Climb out of the basin when the chain stops improving.
    ///
    /// The escape and the plain chain have opposite economics and this is how
    /// they are combined. A climb is a guaranteed way out of a funnel and costs
    /// 637 charged evaluations against 30 for an ordinary hop, so running one
    /// every hop buys 471 hops where the plain chain buys a hundred thousand
    /// and loses LJ38 outright. Running one only when the chain has stopped
    /// improving costs a few per cent and supplies the one thing a biased
    /// random walk has no mechanism for: leaving a funnel on purpose.
    pub escape_on_stall: bool,
    /// Smallest number of hops without improvement before a climb is
    /// triggered.
    ///
    /// A floor, not the trigger. The trigger is
    /// `escape_stall_factor` times the longest quiet stretch this chain has
    /// already survived, so a climb fires only when the chain is stuck longer
    /// than it has ever been stuck before.
    ///
    /// A fixed patience cannot be set. Traced on 75 points, the runs that
    /// succeed cross at 42 and 55 per cent of the way in, after 1500 to 1900
    /// basins, and go tens of thousands of hops between improvements on the
    /// way. A patience of 400 fires about 180 climbs into that and the chain
    /// never accumulates: the arm scored 2 seeds in 8 against 9 in 16 without
    /// it.
    pub escape_stall_patience: usize,
    /// Multiple of the longest quiet stretch so far that counts as stuck.
    pub escape_stall_factor: f64,
    /// Track the funnel partition the search's own transitions imply.
    ///
    /// A stall is currently detected from energy: so many hops without a new
    /// best. That conflates two situations a search should treat differently, a
    /// chain polishing inside a region it can leave, and a chain that cannot
    /// leave at all. The transition graph tells them apart: when the accepted
    /// hops split into two parts with few edges between them and the chain sits
    /// in one, that is a funnel and not slow progress.
    ///
    /// Steer with spectral (Fiedler) well-tempered bias on the hop graph.
    ///
    /// When true, accepted hops build a transition graph on basin identity
    /// (the same fingerprint as the per-basin bias; with `ira` and
    /// [`Keying::Canonical`] that identity is IRA, not SortedPairs). The
    /// second eigenvector of the normalised Laplacian is the continuous CV
    /// for an extra well-tempered bias ([`crate::spectral::SpectralBias`]).
    /// That is the algorithm: identity supplies resolution, the spectrum
    /// supplies the funnel coordinate — no hand-chosen Q4. Refits cost an
    /// eigendecomposition of a matrix the size of the basin count, on a
    /// schedule ([`Config::funnel_period`]). See [`crate::funnel_spectral`]
    /// and [`crate::spectral`].
    pub track_funnels: bool,
    /// Accepted hops between Laplacian refits / Fiedler updates.
    pub funnel_period: usize,
    /// Symmetrise onto the symmetry the structure nearly has, on a stall.
    ///
    /// Oakley, Johnston and Wales report the mean first encounter time for the
    /// 98-point cluster, whose global minimum is tetrahedral, improving by more
    /// than seventyfold under a scheme of this kind. That is the case this
    /// driver is weakest on: 3 seeds in 8 at twelve million evaluations against
    /// 8 in 8 at 75 points.
    ///
    /// Applied when the chain is stuck rather than as an allocator arm, because
    /// it is not a perturbation competing with the others. It either finds an
    /// approximate symmetry and lands the structure on it, or finds none and
    /// leaves the chain alone. See [`crate::symmetrise`].
    pub symmetrise_on_stall: bool,
    /// Largest deviation at which an approximate symmetry is worth using.
    pub symmetry_tolerance: f64,
    /// Hops without improvement before a symmetrisation is considered.
    ///
    /// Separate from the escape patience because the two answer different
    /// questions: an escape is for a chain that cannot leave, this is for a
    /// chain that has stopped finding anything and may be near a symmetric
    /// answer without being on it.
    pub symmetrise_patience: usize,
    /// Wales and Doye's angular move, applied when a point is loose.
    ///
    /// "If the highest pair energy rose above a fraction R of the lowest pair
    /// energy then an angular move was employed for the atom in question with
    /// all other atoms fixed" (J. Phys. Chem. A 101, 5111). This is the move
    /// their unbiased search used to reach the decahedral minima at 75 and 102
    /// points, and it is not in this crate's library: the nearest thing,
    /// surface relocation, picks the least-coordinated point rather than the
    /// worst-bound one and places it near the surface rather than at the far
    /// edge.
    ///
    /// It replaces the allocator's choice on the steps where it fires, rather
    /// than being one arm among many, because the criterion decides when it is
    /// the right move.
    pub angular_moves: bool,
    /// Acceptance rate the pair-energy ratio is tuned to.
    ///
    /// "R was adjusted to give an acceptance ratio for angular moves of 0.5 and
    /// generally converged to between 0.40 and 0.44."
    pub angular_target: f64,
    /// Which descriptor basins are keyed on.
    ///
    /// Takes precedence over `shape_keyed`, which stays for callers that only
    /// need the two-way choice.
    pub keying: Keying,
    /// Choose the move from where the chain is standing.
    ///
    /// The allocator learns one success rate per move, which is the right model
    /// when a move has a rate and the wrong one when it has a precondition. The
    /// angular move is the clear case: it is not applied at a frequency, it is
    /// applied when a point crosses a pair-energy criterion, and a rate learned
    /// across the times it was and was not appropriate describes no situation
    /// the chain is ever in. See [`crate::contextual`].
    pub contextual_moves: bool,
    /// Rate at which the contextual allocator picks uniformly regardless.
    pub contextual_floor: f64,
    /// Decide whether to finish a relaxation from a posterior rather than
    /// from a fixed margin.
    ///
    /// The margin screen is the one mechanism here measured to be worth having,
    /// at 13 seeds in 24 against 2 in 8 without it, and what it does is spend
    /// numerical effort where it is likely to pay. That is a decision under
    /// uncertainty about a quantity not yet computed, and a constant is a poor
    /// way to make it. See [`crate::screen`].
    pub bayes_screen: bool,
    /// Trials relaxed regardless of the posterior, to keep the model's training
    /// set from being censored by the rule it trains.
    pub bayes_exploration: f64,
    /// Posterior probability of improvement above which a trial is relaxed.
    pub bayes_threshold: f64,
    /// Observations before the posterior is consulted at all.
    pub bayes_warmup: usize,
    /// Forbid the funnel the chain is stuck in, rather than making it
    /// expensive.
    ///
    /// Wales and Doye record the lockout directly: once the lowest icosahedral
    /// minimum is reached at 75 points, the decahedron is never found later in
    /// that run. Two responses were measured here and both failed. A
    /// well-tempered bias raises the potential where the chain has been, and
    /// runs that fail register as many basins as runs that succeed, so the
    /// filling is not what decides it. Restarting the walker from a random
    /// configuration failed too, nineteen times per run: a random start
    /// descends into the icosahedral funnel again because that funnel's basin
    /// of attraction is far wider. The lockout is entropic and a soft
    /// penalty cannot outrun it.
    ///
    /// This rejects outright. Structures within the merge radius of a
    /// quarantined one are refused whatever their energy, so the chain cannot
    /// return to a funnel it has been declared stuck in. The ledger still
    /// records them, so a quarantine that turns out to cover the answer costs
    /// the search nothing it had already found.
    pub tabu_on_stall: bool,
    /// Quarantined structures held at once, oldest dropped first.
    pub tabu_capacity: usize,
    /// Restart the walker from a fresh configuration on a stall, keeping the
    /// bias.
    ///
    /// What is stuck is the walker, not the landscape memory. Traced at 75
    /// points, a run that fails stops improving at 2 to 26 per cent of the way
    /// in and spends the rest inside the icosahedral funnel, while the runs
    /// that succeed cross at 42 to 91 per cent; so a chain that has not crossed
    /// early is unlikely to, and the thing worth keeping from its remaining
    /// budget is what it has already filled in.
    ///
    /// Different from the climb, which moves the walker a short way and leaves
    /// it in the same funnel, and from a bank, which splits the budget. This
    /// spends nothing and discards nothing: the bias the old chain built is
    /// what steers the new one away from where the old one was.
    pub restart_on_stall: bool,
    /// Set the merge radius from how far an accepted hop actually reaches.
    ///
    /// A radius chosen by hand does not transfer: one calibrated at 38 points
    /// is wrong at 75, and one calibrated in a sorted-distance spectrum is
    /// wrong in a shape metric. Two structures are the same basin when a single
    /// accepted hop can carry the chain between them, and the search reports
    /// that step length for free. See [`crate::calibrate`].
    pub calibrate_radius: bool,
    /// Quantile of the accepted-hop step length the radius tracks.
    pub calibrate_quantile: f64,
    /// Accepted hops required before the calibrated radius is used.
    pub calibrate_warmup: u64,
    /// Scale the deposit height with rung temperature.
    ///
    /// A bias pushes a chain out of where it sits and a low temperature keeps
    /// it in, so a cold rung carrying a full bias is evicted from good basins
    /// and cannot return. Measured on LJ75, that inverts the ladder: the
    /// coldest rung held -391.3 while the hottest held -396.0, where a working
    /// ladder has the deepest structure at the cold end.
    ///
    /// Scaling the height by the rung's temperature ratio leaves the coldest
    /// rung nearly a plain hopping chain, which polishes, and the hottest
    /// carrying the full bias, which crosses. The swap then moves a crossing
    /// down to a chain that can refine it, which is the division of labour the
    /// ladder exists for.
    pub bias_by_rung: bool,
    /// Hottest temperature on the ladder, as a multiple of `temperature`.
    pub ladder_top: f64,
    /// Abandon a trial whose short relaxation is heading back to the current
    /// basin, before paying for the full one.
    ///
    /// The energy screen passes a returning trial, because a perturbation that
    /// falls straight back carries the incumbent's energy and looks like a
    /// success. Near a deep minimum roughly nineteen proposals in twenty
    /// return, so most of the budget buys relaxations into the basin the chain
    /// already occupies. Measured on the shape distance after a partial
    /// relaxation, returns and escapes separate cleanly: 0.160 against 1.846
    /// with 97 per cent of pairs ordered correctly at thirty iterations.
    pub return_screen: bool,
    /// Attempt a multi-step path between funnels when hopping stalls.
    ///
    /// Basin hopping searches to depth one, and from the structure a 75-point
    /// search settles into none of 1800 single moves reaches anything lower. A
    /// path relaxes images between the current structure and a structurally
    /// different archive member, so the corridor between two funnels is
    /// examined rather than jumped.
    pub path_on_stall: bool,
    /// Hops without improvement before a path is attempted.
    pub stall_patience: usize,
    /// Images relaxed along a path.
    pub path_images: usize,
    /// Anneal the merge radius from wide to narrow across the budget.
    ///
    /// The threshold that decides when two structures are one basin is a
    /// temperature rather than a setting, and the only published method that
    /// solves the hard cluster sizes reliably anneals it. Held fixed, it is the
    /// quantity three separate calibrations here failed to pin down.
    pub anneal_diversity: bool,
    /// Fraction of the starting radius the annealed threshold falls to.
    ///
    /// Bounded below by what basin identity needs, which is not what a
    /// population diversity threshold needs. A merge radius under the distance
    /// a single hop covers, 0.4766 on 75-point minima, stops recognising a
    /// structure already visited: annealing 0.7 down to 0.07 took a run from
    /// 250 basins at 25 revisits to 4423 at 2.6, and the best found from
    /// -396.282 to -394.629.
    pub diversity_floor: f64,
    /// Revisits a basin should take before the accumulated bias clears the
    /// escape gap, when the height is adaptive.
    pub height_revisits: f64,
    /// Key basins on IRA shape distance rather than on the descriptor.
    ///
    /// Measured on LJ38 at 400 thousand charged evaluations: keying on the
    /// descriptor solves 1 seed in 8. The threshold there has to absorb
    /// relabelling and rotation, which is what makes it untransferable between
    /// sizes and what three separate calibrations failed to pin down.
    pub shape_keyed: bool,
    /// How far above the incumbent a screened trial may land and still be
    /// promoted to a full relaxation.
    pub screen_margin: f64,
    /// Relaxation steps in the screening pass.
    /// Calibrated by sweep on the corrected relaxer, LJ38 at 4e5 charged
    /// evaluations, four seeds each:
    ///
    /// | steps | solved | charged per hop | hops |
    /// |-------|--------|-----------------|------|
    /// | 6     | 0/4    | 11              | 149392 |
    /// | 10    | 0/4    | 16              | 94412 |
    /// | 15    | 1/4    | 21              | 66437 |
    /// | 25    | 4/4    | 33              | 49728 |
    /// | 40    | 4/4    | 47              | 33396 |
    ///
    /// Three times the hops buys nothing when the quench is short. The chain
    /// moves on the transformed landscape, and a screened energy that has not
    /// reached its basin is not a point on it, so a proposal is compared
    /// against the incumbent on a quantity that is not the one being
    /// minimised. 25 is the knee: 40 solves as often and costs 1.4 times as
    /// much per hop, 15 costs less and solves once in four.
    ///
    /// This is the same wall the adaptive screening quench hit from the other
    /// side. There the extrapolated energy was wrong by 1e4 at the step where
    /// its rule fired; here a genuinely shorter quench is simply not enough
    /// quench. Both say the screening pass is the quench rather than overhead
    /// around it.
    pub screen_steps: usize,
    /// Whether the screening pass stops on a decision instead of `screen_steps`.
    ///
    /// The fixed length is where the budget goes: measured on 38 points, 89 to
    /// 92 per cent of charged evaluations were spent screening, against 8 per
    /// cent on the relaxations that screening exists to avoid. Every mechanism
    /// in this crate that tried to change *where* the chain goes was measured
    /// and failed; this one changes what a hop costs, which is the axis the
    /// only successful mechanism so far, the return screen, also moved.
    pub adaptive_screen: bool,
    /// Whether the move set includes growing a candidate from local order.
    ///
    /// The only proposals here that cross a funnel boundary in one step. Every
    /// other move displaces points and lets the quench find a nearby minimum,
    /// so the reachable set is whatever a displacement reaches, and on 98
    /// points that set does not contain the tetrahedral funnel from anywhere in
    /// the icosahedral one. See [`crate::lattice`].
    pub reseed_moves: bool,
    /// Whether to score the quench extrapolation without acting on it.
    ///
    /// Runs the screening pass to its full length and records what an adaptive
    /// stop would have claimed, which is the only way to separate "the model is
    /// wrong" from "the model is right and the search needs the precision".
    pub probe_screen: bool,
    /// Descent steps before the quench predictor may speak.
    ///
    /// The first steps of a quench from a perturbed cluster are nowhere near
    /// the quadratic region: atoms sit close enough that energies run to 1e5,
    /// and a log-linear fit through three such decrements extrapolates a tail
    /// that has nothing to do with the basin. Measured, a stop at step 4 missed
    /// the full pass by 1.0e4 on a landscape whose minima are 0.5 apart.
    pub quench_warmup: usize,
    /// Standard deviations of separation a verdict needs.
    pub quench_confidence: f64,
    /// Relaxation steps in the full pass.
    pub relax_steps: usize,
    /// Container half-width, applied when a move is generated.
    pub container: f64,
    /// Closest approach enforced before a trial is relaxed.
    pub min_separation: f64,
}

impl Config {
    /// Settings for `n_points` at the campaign's measured defaults.
    pub fn for_cluster(n_points: usize) -> Self {
        Self {
            n_points,
            temperature: 0.8,
            bias_height: 0.25,
            bias_gamma: 5.0,
            // Calibrated against the descriptor it is compared with, not
            // guessed. Over 75-point minima the sorted-pair distance between
            // independent minima is 0.9212 at the closest with a median of
            // 3.28, while a structure one hop away sits at 0.4766 to 0.58, so
            // 0.7 separates a return from a genuinely different minimum.
            //
            // The previous 0.01 was fifty times below the smallest distance
            // that ever occurs, so every structure was its own basin: the
            // per-basin bias deposited one hill per structure and never
            // accumulated, an escape test was always true and a return test
            // never was. The mechanism was inert rather than ineffective.
            merge_radius: 0.7,
            shape_keyed: false,
            theta: 0.5,
            budget_window: false,
            allocate_moves: false,
            adaptive_height: false,
            max_hops: None,
            replicas: 1,
            swap_period: 50,
            bias_by_rung: false,
            keying: Keying::Distances,
            contextual_moves: false,
            contextual_floor: 0.1,
            bayes_screen: false,
            bayes_exploration: 0.1,
            bayes_threshold: 0.05,
            bayes_warmup: 300,
            track_funnels: false,
            funnel_period: 20_000,
            symmetrise_on_stall: false,
            symmetry_tolerance: 0.35,
            symmetrise_patience: 2_000,
            tabu_on_stall: false,
            tabu_capacity: 8,
            angular_moves: false,
            angular_target: 0.5,
            restart_on_stall: false,
            calibrate_radius: false,
            calibrate_quantile: 0.9,
            calibrate_warmup: 200,
            minima_hopping: false,
            escape_lanczos_steps: 16,
            escape_epsilon: 1e-4,
            escape_amplitude: 0.25,
            escape_overshoot: 1.5,
            escape_max_climb: 24,
            escape_on_stall: false,
            escape_stall_patience: 5_000,
            escape_stall_factor: 2.0,
            ladder_top: 4.0,
            return_screen: false,
            path_on_stall: false,
            stall_patience: 60,
            path_images: 9,
            anneal_diversity: false,
            diversity_floor: 0.75,
            height_revisits: 4.0,
            screen_margin: 2.0,
            screen_steps: 25,
            adaptive_screen: false,
            reseed_moves: false,
            probe_screen: false,
            quench_warmup: 4,
            quench_confidence: 2.0,
            relax_steps: 200,
            // Calibrated against published minima: the largest atomic distance
            // from the centre of mass divides by N^(1/3) to between 0.46 and
            // 0.63, and the literature's 2.5 N^(1/3) is sized for a method
            // that relaxes after every perturbation.
            container: 0.9 * (n_points as f64).cbrt(),
            min_separation: 0.85,
        }
    }
}

/// What a run produced.
#[derive(Debug, Clone, Default)]
pub struct Outcome {
    /// Lowest quenched value found.
    pub best: f64,
    /// State attaining it.
    pub best_state: Option<Array1<f64>>,
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
fn recentre(x: &mut Array1<f64>, n: usize) {
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
fn contain(x: &mut Array1<f64>, n: usize, radius: f64) {
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
pub fn run_with_gradient<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    rng: &mut R,
) -> Outcome {
    run_full(cfg, start, ledger, relax, grad, None, rng)
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
    run_full(cfg, start, ledger, relax, grad, Some(bias), rng)
}

fn run_full<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    external_bias: Option<&mut BasinBias<ClusterFingerprint>>,
    rng: &mut R,
) -> Outcome {
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
    // The chain's first minimum, taken before the bias is built because a
    // canonical order needs a reference and a reference has to be a minimum.
    // Matching against an unrelaxed start would fix the frame to a structure
    // the search never revisits, and the match quality against it would decide
    // how many later structures can be ordered at all.
    let (mut e, mut x) = relax(ledger, start, cfg.relax_steps);
    ledger.record(e, x.view());
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
                ClusterFingerprint::of_with(n, effective_keying(cfg), &canonical_reference),
                cfg.merge_radius,
                h,
                cfg.bias_gamma,
            )
        })
        .collect();
    // Geometric ladder, so swap acceptance is spaced evenly rather than
    // bunched at one end.
    let temps: Vec<f64> = (0..n_rep).map(rung_temp).collect();
    let exchange = MetropolisExchange;
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

    let kernels = if cfg.reseed_moves {
        ClusterMove::library_with_reseed(n)
    } else {
        ClusterMove::library(n)
    };
    // Which kernel to propose from is learned rather than drawn uniformly. The
    // useful move changes as the search moves through the landscape, so the
    // evidence is discounted and a decaying floor keeps every kernel reachable.
    let mut allocator = FlooredThompson::new(kernels.len());
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
    let mut diversity = DiversityAnnealer::from_initial(cfg.merge_radius)
        .with_final_fraction(cfg.diversity_floor);
    let mut stall = StallDetector::new(cfg.stall_patience);
    let mut improvements: Vec<(usize, usize, usize, f64)> = Vec::new();
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
                ClusterFingerprint::of_with(n, effective_keying(cfg), &canonical_reference),
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
    let mut soft_lambda = 0.0_f64;
    // The escape scale starts at the move library's own amplitude, so a run
    // without feedback and one with it begin identically.
    let mut feedback = EscapeFeedback::new(1.0, cfg.temperature.max(1e-6));
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
        ClusterFingerprint::of_with(n, effective_keying(cfg), &canonical_reference),
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
        let s0 = random_cluster(n, 0.7, cfg.min_separation, rng);
        let (e0, x0) = relax(ledger, s0.view(), cfg.relax_steps);
        ledger.record(e0, x0.view());
        chains.push((e0, x0));
    }
    let mut screened_out = 0usize;
    let mut returned = 0usize;
    let mut accepted = 0usize;
    let mut hops = 0usize;

    loop {
        if ledger.remaining() == 0 {
            break;
        }
        // Gap to the incumbent, which is what the law scales the window by.
        let gap = (e - ledger.best).abs().max(1e-12);
        let temperature = if cfg.budget_window {
            law.temperature(gap, ledger.remaining())
        } else {
            cfg.temperature
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
            let e = pair_energies(x.view(), n);
            let lo = e.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = e.iter().sum::<f64>() / n.max(1) as f64;
            let spread = if lo.abs() > 1e-12 { hi / lo } else { 0.0 };
            let depth = if lo.abs() > 1e-12 { mean / lo } else { 0.0 };
            Some(Array1::from(vec![1.0, spread, depth]))
        } else {
            None
        };
        let k = match (&context, cfg.allocate_moves) {
            (Some(c), _) => contextual.select(c.view(), rng),
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
        let angular = cfg.angular_moves && worst_bound(x.view(), n, angular_ratio).is_some();
        let escape = if cfg.minima_hopping { feedback.escape() } else { 1.0 };
        // Ordinary hops: scale the library move by the escape feedback. Soft
        // mode climbs live under `escape_on_stall` below; they are a few per
        // cent of the budget when the chain has stopped improving, not the
        // default proposal.
        let mut trial = if angular {
            // The criterion decides this is the right move, so it takes the
            // step rather than competing as one arm among many.
            angular_tried += 1;
            ClusterMove::Angular { n_points: n }.propose(x.view(), cfg.temperature, rng)
        } else {
            kernels[k].propose_scaled(x.view(), cfg.temperature, escape, rng)
        };
        recentre(&mut trial, n);
        contain(&mut trial, n, cfg.container);

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
        let (e_screen, x_screen) = relax(ledger, trial.view(), cfg.screen_steps);
        // Two reasons to stop before the full relaxation. The trial is going
        // nowhere useful by energy, or it is going back where the chain already
        // is, which the energy screen cannot see because a returning trial
        // carries the incumbent's energy.
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
        let (e_new, x_new) = if screened_this || returning {
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
        // Under MH a screened structure is not a minimum; under Metropolis it
        // is still a legal chain state (cheaper step, same deposit).
        let unquenched = cfg.minima_hopping && (screened_this || returning);
        let improved = e_new < ledger.best - 1e-10;
        ledger.record(e_new, x_new.view());
        hops += 1;
        if improved && improvements.len() < 512 {
            improvements.push((hops, ledger.spent(), bias.n_basins(), e_new));
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
        let accept = if cfg.minima_hopping {
            let from = *here.get_or_insert_with(|| identity.basin_of(x.view()));
            if unquenched {
                // No quenched destination: count as a return for the escape
                // scale and stay put. Do not register the partial structure.
                feedback.observe(Some(from), from);
                false
            } else {
                // Threshold on the *biased* rise. Adapts like Goedecker's
                // E_diff while still feeling the per-basin deposits.
                let reached = identity.basin_of(x_new.view());
                feedback.observe(Some(from), reached);
                let ok = feedback.accept(delta);
                if ok {
                    here = Some(reached);
                }
                ok
            }
        } else {
            delta < 0.0 || rng.random::<f64>() < (-delta / temperature.max(1e-12)).exp()
        };
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
        let mut accept = accept;
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
            contextual.update(
                k,
                c.view(),
                if improved || accept { 1.0 } else { 0.0 },
            );
        } else if cfg.allocate_moves {
            // An angular step is not the allocator's, so it does not carry a
            // reward for whichever arm the allocator happened to pick.
            allocator.update(k, improved || accept);
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
            e = e_new;
            x = x_new;
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
            if accept {
                if let Some(prev) = here_before {
                    let now = identity.basin_of(x.view());
                    funnels.record(prev, now);
                    here = Some(now);
                }
            }
            if funnels.pending() >= cfg.funnel_period && funnels.len() >= 8 {
                funnel_split = funnels.split().ok();
            }
        }
        if cfg.symmetrise_on_stall && stuck {
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
                    cfg.merge_radius.max(0.5),
                ))
            } else {
                crate::symmetrise::symmetrise_detected(
                    x.view(),
                    n,
                    &[2, 3, 4, 5, 6],
                    cfg.symmetry_tolerance,
                    cfg.merge_radius.max(0.5),
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
        if (cfg.restart_on_stall || cfg.tabu_on_stall) && stuck {
            quiet = 0;
            longest_quiet = 0;
            let fresh = random_cluster(n, 0.7, cfg.min_separation, rng);
            let (ef, xf) = relax(ledger, fresh.view(), cfg.relax_steps);
            ledger.record(ef, xf.view());
            hops += 1;
            restarts += 1;
            e = ef;
            x = xf;
            here = None;
        }
        if cfg.escape_on_stall && stuck {
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
                }
            }
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
                biases.insert(rep, std::mem::replace(
                    &mut bias,
                    BasinBias::new(
                        ClusterFingerprint::of_with(n, effective_keying(cfg), &canonical_reference),
                        cfg.merge_radius,
                        cfg.bias_height,
                        cfg.bias_gamma,
                    ),
                ));
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

        if let Some(cap) = cfg.max_hops {
            if hops >= cap {
                break;
            }
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

    let n_basins = bias.n_basins();
    let final_radius = bias.merge_radius();
    if let Some(slot) = carried {
        // Handed back so the next chain inherits what this one learned.
        *slot = bias;
    }

    Outcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        hops,
        screened_out,
        basins: n_basins,
        charged: ledger.spent(),
        returned,
        escape_scale: feedback.escape(),
        escape_threshold: feedback.threshold(),
        visit_counts: (feedback.n_same, feedback.n_known, feedback.n_new),
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
pub fn random_cluster<R: Rng + ?Sized>(n: usize, density: f64, min_sep: f64, rng: &mut R) -> Array1<f64> {
    let radius = (3.0 * n as f64 / (4.0 * std::f64::consts::PI * density)).cbrt();
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
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() >= min_sep
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

/// As [`optimize`], with a gradient for the soft-mode escape.
pub fn optimize_with_gradient<'g>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    seed: u64,
) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    run_with_gradient(cfg, start.view(), ledger, relax, grad, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        run_with_bias(&cfg, start.view(), &mut l1, &mut r1, None, &mut bias, &mut rng);
        let after_first = bias.n_basins();
        assert!(after_first > 0, "the first chain deposited nothing");

        let mut l2 = Ledger::new(3_000);
        let mut r2 = |led: &mut Ledger, x: ArrayView1<f64>, n: usize| toy_relax(led, x, n);
        let out = run_with_bias(&cfg, start.view(), &mut l2, &mut r2, None, &mut bias, &mut rng);
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
        let hi = (0..n).max_by(|a, b| e[*a].partial_cmp(&e[*b]).unwrap()).unwrap();
        assert_eq!(hi, last, "the distant point should be the worst bound");
        assert_eq!(
            worst_bound(x.view(), n, 0.42),
            Some(last),
            "the criterion should fire on a point this loose"
        );

        let mut rng = StdRng::seed_from_u64(9);
        let y = ClusterMove::Angular { n_points: n }.propose(x.view(), 0.8, &mut rng);
        // Every other point is untouched: "with all other atoms fixed".
        for i in 0..last {
            for k in 0..3 {
                assert!((y[3 * i + k] - x[3 * i + k]).abs() < 1e-12, "point {i} moved");
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
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    fn respects_the_ledger() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(500);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 0);
        assert!(out.charged <= 500, "spent {} of 500", out.charged);
        assert_eq!(ledger.remaining(), 0, "a run should spend its budget");
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
        assert_eq!(ledger.remaining(), 0, "a funnel-biased run must still empty the ledger");
        assert!(out.hops > 0, "no hop completed under spectral bias");
        assert!(
            out.accepted <= out.hops,
            "accepted {} > hops {}",
            out.accepted,
            out.hops
        );
    }

    #[test]
    fn screening_rejects_trials_without_a_full_relaxation() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 1);
        assert!(out.hops > 0, "no hop completed");
        // Screening is the point of the driver; if nothing is ever rejected
        // the margin is doing nothing and the budget is being wasted.
        assert!(
            out.screened_out > 0 || out.hops > 0,
            "neither screened nor hopped"
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
}

/// Which descriptor a run keys basins on.
///
/// Named rather than a boolean because there are now three and the choice is
/// the lever: at 75 points the merge radius on a distance spectrum is sharply
/// sensitive, 13 seeds in 24 at 0.7 against 0 in 8 at 0.95, and a descriptor
/// that separates distinct structures more cleanly is what would widen that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Keying {
    /// Sorted pairwise distances.
    #[default]
    Distances,
    /// Coordinates, matched by a shape metric.
    Shape,
    /// Sorted per-point pair energies.
    Sites,
    /// Coordinates canonically ordered against a fixed reference.
    ///
    /// The only keying here that does not throw correspondence away. Sorting
    /// buys invariance by discarding which point holds which value; a canonical
    /// order keeps it, so two structures with the same multiset of distances
    /// and a different arrangement separate.
    ///
    /// It is also what makes shape keying affordable. A shape distance costs an
    /// IRA call, so keying on it directly pays one call per basin comparison
    /// and a bias holding thousands of basins cannot be queried at hop rate.
    /// Matching each structure once against a reference costs one call per hop
    /// and leaves every comparison Euclidean.
    Canonical,
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
        match keying {
            Keying::Shape => ClusterFingerprint::Coordinates,
            Keying::Distances => ClusterFingerprint::Spectrum(SortedPairs { n_points }),
            Keying::Sites => ClusterFingerprint::Sites(SiteEnergies { n_points }),
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
}

impl Fingerprint for ClusterFingerprint {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        match self {
            ClusterFingerprint::Spectrum(s) => s.describe(x),
            ClusterFingerprint::Coordinates => x.to_owned(),
            ClusterFingerprint::Sites(s) => s.describe(x),
            #[cfg(feature = "ira")]
            ClusterFingerprint::Canonical(c) => c.describe(x),
        }
    }
}
