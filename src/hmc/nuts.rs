//! NUTS Phase 3a: No-U-Turn Sampler trajectory builder.
//!
//! Mirrors Stan's recursive doubling tree from
//! `stan/mcmc/hmc/nuts/base_nuts.hpp:122-265`. Each NUTS step:
//!   1. Draws fresh momentum from the `Momentum` kernel.
//!   2. Builds a binary tree by random doubling, each leaf one
//!      leapfrog step in `+/-` direction.
//!   3. Terminates on U-turn (`Momentum::uturn`) or divergence
//!      (`|delta_H| > max_delta_h`).
//!   4. Multinomial-samples a candidate from the visited leaves
//!      weighted by `exp(-H_i)`.
//!   5. Accepts the candidate with the standard HMC-Metropolis
//!      probability `min(1, exp(H_old - H_new))`.
//!
//! Phase 3a ships Gaussian + q-Gaussian momentum (both inherit the
//! Gaussian U-turn predicate via the `Momentum` trait default).
//! Phase 3b will refine the U-turn for q-Gaussian when the
//! `Z_q(p) > 0` guard becomes meaningful at higher q.
//!
//! See `~/Git/Gitlab/obsidian-notes/Software/anneal/design_pass_11_pt_nuts_lit.org`
//! Section "NUTS Phase 3 design" for the full pseudocode.

use eindir_core::{FPair, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;

use crate::cool::Cooling;
use crate::grad::Gradient;
use crate::history::State;
use crate::hmc::momentum::Momentum;
use crate::sampler::Sampler;

/// Diagnostic for a single NUTS step.
#[derive(Clone, Debug)]
pub struct NutsTransition {
    /// New position (the multinomial-sampled candidate).
    pub x: Array1<f64>,
    /// Number of leapfrog leaves visited (= 2^depth).
    pub n_leapfrog: usize,
    /// Tree depth reached (0..max_depth).
    pub tree_depth: u32,
    /// `true` if any leaf diverged (`|delta_H| > max_delta_h`).
    pub diverged: bool,
    /// `true` if the candidate was accepted (always true in vanilla
    /// NUTS: candidate selection is *itself* the Metropolis step).
    pub accepted: bool,
}

const MAX_DELTA_H: f64 = 1000.0;

/// Single leapfrog step at temperature `temp`. Returns the new
/// `(x, p)` plus the resulting Hamiltonian.
fn leapfrog_leaf<G, M, Obj>(
    x: &Array1<f64>,
    p: &Array1<f64>,
    eps: f64,
    temp: f64,
    grad_fn: &G,
    momentum: &M,
    obj_fn: &Obj,
) -> (Array1<f64>, Array1<f64>, f64)
where
    G: Gradient<f64>,
    M: Momentum + ?Sized,
    Obj: Fn(&Array1<f64>) -> f64,
{
    let n = x.len();
    let grad_old = grad_fn.grad(x.view());
    let mut p_new = p.clone();
    for i in 0..n {
        p_new[i] -= 0.5 * eps * grad_old[i] / temp;
    }
    let dk = momentum.dk_dp(&p_new);
    let mut x_new = x.clone();
    for i in 0..n {
        x_new[i] += eps * dk[i];
    }
    let grad_new = grad_fn.grad(x_new.view());
    for i in 0..n {
        p_new[i] -= 0.5 * eps * grad_new[i] / temp;
    }
    let u = obj_fn(&x_new);
    let h = u / temp + momentum.kinetic(&p_new);
    (x_new, p_new, h)
}

/// One sub-tree node: the leapfrog state at the leaf.
#[derive(Clone, Debug)]
struct Leaf {
    x: Array1<f64>,
    p: Array1<f64>,
    #[allow(dead_code)]
    h: f64,
}

/// Output of a `build_tree` recursion step.
#[derive(Clone, Debug)]
struct SubTree {
    /// Leftmost (most-negative-direction) leaf.
    left: Leaf,
    /// Rightmost (most-positive-direction) leaf.
    right: Leaf,
    /// Multinomial-sampled candidate from this sub-tree.
    candidate: Leaf,
    /// `log sum_i exp(-H_i)` accumulator over the sub-tree's leaves.
    log_w_sum: f64,
    /// Total leapfrog leaves under this sub-tree.
    n_leapfrog: usize,
    /// `true` if a U-turn was detected within the sub-tree.
    terminated: bool,
    /// `true` if any leaf diverged.
    divergent: bool,
}

/// log_sum_exp(a, b) = max(a, b) + ln(1 + exp(-|a - b|)).
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    let d = (-((a - b).abs())).exp();
    m + d.ln_1p()
}

/// Recursive build_tree mirroring Stan `base_nuts.hpp:122-265`.
#[allow(clippy::too_many_arguments)]
fn build_tree<G, M, Obj, R: Rng>(
    leaf: Leaf,
    h0: f64,
    direction: i8,
    depth: u32,
    eps: f64,
    temp: f64,
    grad_fn: &G,
    momentum: &M,
    obj_fn: &Obj,
    rng: &mut R,
) -> SubTree
where
    G: Gradient<f64>,
    M: Momentum + ?Sized,
    Obj: Fn(&Array1<f64>) -> f64,
{
    if depth == 0 {
        let signed_eps = if direction < 0 { -eps } else { eps };
        let (x_new, p_new, h_new) = leapfrog_leaf(
            &leaf.x, &leaf.p, signed_eps, temp, grad_fn, momentum, obj_fn,
        );
        let new_leaf = Leaf {
            x: x_new,
            p: p_new,
            h: h_new,
        };
        let divergent = (h_new - h0).abs() > MAX_DELTA_H || !h_new.is_finite();
        let log_w = -h_new;
        return SubTree {
            left: new_leaf.clone(),
            right: new_leaf.clone(),
            candidate: new_leaf,
            log_w_sum: log_w,
            n_leapfrog: 1,
            terminated: divergent,
            divergent,
        };
    }
    let sub1 = build_tree(
        leaf,
        h0,
        direction,
        depth - 1,
        eps,
        temp,
        grad_fn,
        momentum,
        obj_fn,
        rng,
    );
    if sub1.terminated {
        return sub1;
    }
    let start = if direction < 0 {
        sub1.left.clone()
    } else {
        sub1.right.clone()
    };
    let sub2 = build_tree(
        start,
        h0,
        direction,
        depth - 1,
        eps,
        temp,
        grad_fn,
        momentum,
        obj_fn,
        rng,
    );
    let (left, right) = if direction < 0 {
        (sub2.left.clone(), sub1.right.clone())
    } else {
        (sub1.left.clone(), sub2.right.clone())
    };
    let log_w_sum = log_sum_exp(sub1.log_w_sum, sub2.log_w_sum);
    // Progressive multinomial: pick sub2's candidate with prob exp(lw2 - log_w_sum).
    let p_pick_sub2 = (sub2.log_w_sum - log_w_sum).exp();
    let u: f64 = rng.random();
    let candidate = if u < p_pick_sub2 {
        sub2.candidate
    } else {
        sub1.candidate
    };
    let terminated =
        sub1.terminated || sub2.terminated || momentum.uturn(&left.x, &left.p, &right.x, &right.p);
    let divergent = sub1.divergent || sub2.divergent;
    SubTree {
        left,
        right,
        candidate,
        log_w_sum,
        n_leapfrog: sub1.n_leapfrog + sub2.n_leapfrog,
        terminated,
        divergent,
    }
}

/// One NUTS step. Builds the binary tree by doubling up to
/// `max_depth`, then accepts the multinomial-sampled candidate.
#[allow(clippy::too_many_arguments)]
pub fn nuts_step<G, M, Obj, R: Rng>(
    x: &Array1<f64>,
    u: f64,
    eps: f64,
    temp: f64,
    grad_fn: &G,
    momentum: &M,
    obj_fn: &Obj,
    max_depth: u32,
    rng: &mut R,
) -> NutsTransition
where
    G: Gradient<f64>,
    M: Momentum + ?Sized,
    Obj: Fn(&Array1<f64>) -> f64,
{
    let dim = x.len();
    let p0 = momentum.sample(dim, rng);
    let h0 = u / temp + momentum.kinetic(&p0);
    let mut left = Leaf {
        x: x.clone(),
        p: p0.clone(),
        h: h0,
    };
    let mut right = left.clone();
    let mut candidate = left.clone();
    let mut log_w_sum = -h0;
    let mut n_leapfrog: usize = 0;
    let mut diverged = false;
    let mut depth = 0;

    while depth < max_depth {
        let direction: i8 = if rng.random::<f64>() < 0.5 { -1 } else { 1 };
        let start = if direction < 0 {
            left.clone()
        } else {
            right.clone()
        };
        let sub = build_tree(
            start, h0, direction, depth, eps, temp, grad_fn, momentum, obj_fn, rng,
        );
        if sub.divergent {
            diverged = true;
            break;
        }
        // Progressive multinomial across the doubling.
        let p_pick_new = (sub.log_w_sum - log_w_sum).exp().min(1.0);
        let u_pick: f64 = rng.random();
        if u_pick < p_pick_new {
            candidate = sub.candidate.clone();
        }
        log_w_sum = log_sum_exp(log_w_sum, sub.log_w_sum);
        n_leapfrog += sub.n_leapfrog;
        if direction < 0 {
            left = sub.left;
        } else {
            right = sub.right;
        }
        if sub.terminated || momentum.uturn(&left.x, &left.p, &right.x, &right.p) {
            break;
        }
        depth += 1;
    }

    NutsTransition {
        x: candidate.x,
        n_leapfrog,
        tree_depth: depth,
        diverged,
        accepted: !diverged,
    }
}

/// NUTS-driven SA sampler. Drops into `run_rs` and `MultiChainSampler`
/// like `HmcSaSampler` since it impls `Sampler<f64>`.
pub struct NutsSaSampler<O, G, C, M>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
{
    /// The objective.
    pub obj: O,
    /// The gradient (analytic or finite-difference).
    pub gradient: G,
    /// The cooling schedule.
    pub cool: C,
    /// The momentum kernel.
    pub momentum: M,
    /// Base leapfrog step size; rescaled by `sqrt(temp/temp_ref)` per step.
    pub epsilon: f64,
    /// Reference temperature for the cooling rescaling.
    pub temp_ref: f64,
    /// Maximum doubling depth (`max_n_leapfrog = 2^max_depth`).
    pub max_depth: u32,
}

impl<O, G, C, M> NutsSaSampler<O, G, C, M>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
{
    /// Constructs a NUTS-SA sampler. `temp_ref` should typically equal
    /// `cool.temperature(0)`.
    pub fn new(
        obj: O,
        gradient: G,
        cool: C,
        momentum: M,
        epsilon: f64,
        temp_ref: f64,
        max_depth: u32,
    ) -> Self {
        Self {
            obj,
            gradient,
            cool,
            momentum,
            epsilon,
            temp_ref,
            max_depth,
        }
    }
}

impl<O, G, C, M> Sampler<f64> for NutsSaSampler<O, G, C, M>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
{
    fn initial_state<R: Rng>(&self, rng: &mut R) -> State {
        let pos = self.obj.bounds().mkpoint(rng);
        let val = self.obj.eval(pos.view());
        let pair = FPair { pos, val };
        State {
            cur: pair.clone(),
            best: pair,
        }
    }

    fn step<R: Rng>(&self, state: &mut State, epoch: usize, rng: &mut R) -> bool {
        let temp = self.cool.temperature(epoch);
        let eps_eff = self.epsilon * (temp / self.temp_ref).sqrt();
        let obj = &self.obj;
        let result = nuts_step(
            &state.cur.pos,
            state.cur.val,
            eps_eff,
            temp,
            &self.gradient,
            &self.momentum,
            &|x: &Array1<f64>| obj.eval(ArrayView1::from(x.as_slice().unwrap())),
            self.max_depth,
            rng,
        );
        if result.diverged {
            return false;
        }
        let new_val = obj.eval(result.x.view());
        // Accept the candidate if it improved or via Metropolis on the
        // objective alone (NUTS already accepted via the trajectory's
        // multinomial weighting; we only need to track best/cur here).
        state.cur = FPair {
            pos: result.x,
            val: new_val,
        };
        if new_val < state.best.val {
            state.best = state.cur.clone();
        }
        result.accepted
    }
}
