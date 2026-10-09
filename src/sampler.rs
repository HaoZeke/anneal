//! `trait Sampler<T>`: a single-step interface for the SA driver loop.
//!
//! Stan's `base_mcmc::transition` (`stan/mcmc/base_mcmc.hpp:21`) factors
//! the per-step logic into one virtual call, with the driver loop in
//! `services/util/generate_transitions.hpp:42` independent of which
//! sampler is wired. Our `run_rs` previously monomorphised on five type
//! parameters `<O, C, N, M, A>`; this trait gives the driver a single
//! type bound and a single dispatch point.

use eindir_core::{Bounds, FPair};
use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use rand::Rng;
use std::sync::Mutex;

use crate::accept::AcceptRule;
use crate::cool::Cooling;
use crate::history::State;
use crate::movekernel::MoveKernel;
use crate::neigh::Neighborhood;
use crate::variant::SaVariant;

/// One step of an SA driver: take the current state at the given epoch,
/// produce a new state, and report whether the proposal was accepted.
///
/// The trait is intentionally minimal: state, temperature schedule, and
/// proposal+accept logic all live behind `step`. This lets the driver
/// loop (`run_rs`) treat preset, custom, and adaptive variants through a
/// single dispatch point.
pub trait Sampler<T: Float>: Send + Sync {
    /// Draws an initial state from the sampler's prior (uniform on the
    /// objective's bounds for the shipped impl).
    fn initial_state<R: Rng>(&self, rng: &mut R) -> State;

    /// Bounds for low-discrepancy starts when the sampler can construct
    /// a state from an externally supplied position.
    fn qmc_bounds(&self) -> Option<&Bounds<f64>> {
        None
    }

    /// Constructs an initial state from a bounded design point.
    fn initial_state_from_position(&self, _pos: Array1<f64>) -> Option<State> {
        None
    }

    /// One proposal + accept cycle at the given epoch. Mutates `state`
    /// in place and returns `true` iff the proposal was accepted.
    fn step<R: Rng>(&self, state: &mut State, epoch: usize, rng: &mut R) -> bool;

    /// Best-seen pair for the given state. Default impl returns `state.best`.
    fn best_pair(&self, state: &State) -> FPair<f64> {
        state.best.clone()
    }
}

/// Basin-hopping sampler with interchangeable proposal and quench kernels.
///
/// `run_rs` owns the hop loop. This type supplies the basin-hopping step:
/// propose from the current local minimum, quench the proposal, then apply the
/// configured acceptance rule to the two quenched objective values. Cluster
/// moves and ordinary continuous-space kernels therefore use the same driver.
pub struct HoppingSampler<C, M, A, Q> {
    initial_position: Array1<f64>,
    mover: M,
    cooling: C,
    accept: A,
    quench_steps: usize,
    quench: Mutex<Q>,
}

impl<C, M, A, Q> HoppingSampler<C, M, A, Q>
where
    Q: for<'a> FnMut(ArrayView1<'a, f64>, usize) -> FPair<f64> + Send,
{
    /// Constructs a basin-hopping sampler from general sampler components.
    pub fn new(
        initial_position: Array1<f64>,
        mover: M,
        cooling: C,
        accept: A,
        quench_steps: usize,
        quench: Q,
    ) -> Self {
        assert!(
            !initial_position.is_empty(),
            "initial position must not be empty"
        );
        assert!(quench_steps > 0, "quench_steps must be positive");
        Self {
            initial_position,
            mover,
            cooling,
            accept,
            quench_steps,
            quench: Mutex::new(quench),
        }
    }
}

impl<C, M, A, Q> Sampler<f64> for HoppingSampler<C, M, A, Q>
where
    C: Cooling<f64>,
    M: MoveKernel<f64>,
    A: AcceptRule<f64>,
    Q: for<'a> FnMut(ArrayView1<'a, f64>, usize) -> FPair<f64> + Send,
{
    fn initial_state<R: Rng>(&self, _rng: &mut R) -> State {
        let pair = self.quench.lock().expect("quench mutex poisoned")(
            self.initial_position.view(),
            self.quench_steps,
        );
        State {
            cur: pair.clone(),
            best: pair,
        }
    }

    fn initial_state_from_position(&self, pos: Array1<f64>) -> Option<State> {
        let pair =
            self.quench.lock().expect("quench mutex poisoned")(pos.view(), self.quench_steps);
        Some(State {
            cur: pair.clone(),
            best: pair,
        })
    }

    fn step<R: Rng>(&self, state: &mut State, epoch: usize, rng: &mut R) -> bool {
        let temperature = self.cooling.temperature(epoch);
        let proposal = self.mover.propose(state.cur.pos.view(), temperature, rng);
        let quenched =
            self.quench.lock().expect("quench mutex poisoned")(proposal.view(), self.quench_steps);
        let probability = self
            .accept
            .accept_prob(quenched.val - state.cur.val, temperature);
        if rng.random::<f64>() >= probability {
            return false;
        }
        state.cur = quenched;
        if state.cur.val < state.best.val {
            state.best = state.cur.clone();
        }
        true
    }
}

// ---------------------------------------------------------------------------
// SaVariant impl: glues the typed component algebra to the Sampler trait.
// ---------------------------------------------------------------------------

/// A NaN objective value is scored as the worst value. Left as NaN it makes
/// every later energy difference NaN, so the chain never accepts a move and
/// reports NaN as its best.
fn nan_as_worst(value: f64) -> f64 {
    if value.is_nan() { f64::INFINITY } else { value }
}

/// A uniform draw on `bounds`. An axis with `low == high` is that value, and
/// an axis with an infinite wall is its finite wall, or 0 when both walls are
/// infinite; neither draws a number. An axis whose width overflows is drawn
/// at half scale and doubled. `Bounds::mkpoint` panics on these axes, and it
/// is the draw whenever every axis has a positive finite width, so the seed
/// stream of such a box stays the one `mkpoint` already used.
fn initial_position<R: Rng>(bounds: &Bounds<f64>, rng: &mut R) -> Array1<f64> {
    let drawable = |lo: f64, hi: f64| lo != hi && (hi - lo).is_finite();
    if bounds
        .low
        .iter()
        .zip(bounds.high.iter())
        .all(|(&lo, &hi)| drawable(lo, hi))
    {
        return bounds.mkpoint(rng);
    }
    Array1::from_iter(bounds.low.iter().zip(bounds.high.iter()).map(|(&lo, &hi)| {
        if lo == hi {
            lo
        } else if drawable(lo, hi) {
            rng.random_range(lo..hi)
        } else if lo.is_finite() && hi.is_finite() {
            2.0 * rng.random_range(0.5 * lo..0.5 * hi)
        } else if lo.is_finite() {
            lo
        } else if hi.is_finite() {
            hi
        } else {
            0.0
        }
    }))
}

impl<O, C, N, M, A> Sampler<f64> for SaVariant<f64, O, C, N, M, A>
where
    O: eindir_core::Objective<f64> + Send + Sync,
    C: Cooling<f64>,
    N: Neighborhood<f64>,
    M: MoveKernel<f64>,
    A: AcceptRule<f64>,
{
    fn initial_state<R: Rng>(&self, rng: &mut R) -> State {
        let pos = initial_position(self.obj.bounds(), rng);
        let val = nan_as_worst(self.obj.eval(pos.view()));
        let pair = FPair { pos, val };
        State {
            cur: pair.clone(),
            best: pair,
        }
    }

    fn qmc_bounds(&self) -> Option<&Bounds<f64>> {
        Some(self.obj.bounds())
    }

    fn initial_state_from_position(&self, pos: Array1<f64>) -> Option<State> {
        let pos = self.obj.bounds().clip(pos.view());
        let val = nan_as_worst(self.obj.eval(pos.view()));
        let pair = FPair { pos, val };
        Some(State {
            cur: pair.clone(),
            best: pair,
        })
    }

    fn step<R: Rng>(&self, state: &mut State, epoch: usize, rng: &mut R) -> bool {
        let temp = self.cool.temperature(epoch);
        let proposal_pos = self.mover.propose(state.cur.pos.view(), temp, rng);
        if !self
            .neigh
            .contains(state.cur.pos.view(), proposal_pos.view())
        {
            return false;
        }
        let proposal_val = nan_as_worst(self.obj.eval(proposal_pos.view()));
        // Two infeasible points sit on one flat plateau. `inf - inf` is NaN,
        // which every acceptance rule rejects, so a walk started there could
        // never reach the feasible region; the plateau is walked as level.
        let delta = if proposal_val == f64::INFINITY && state.cur.val == f64::INFINITY {
            0.0
        } else {
            proposal_val - state.cur.val
        };
        let p = self.accept.accept_prob(delta, temp);
        let u: f64 = rng.random();
        if u < p {
            state.cur = FPair {
                pos: proposal_pos,
                val: proposal_val,
            };
            if state.cur.val < state.best.val {
                state.best = state.cur.clone();
            }
            true
        } else {
            false
        }
    }
}
