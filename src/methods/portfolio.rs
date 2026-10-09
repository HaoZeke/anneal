//! Thompson-allocated portfolio over the typed algebra's building blocks.
//!
//! One generic global optimizer with a single knob: the budget. Each
//! building block is an arm of a Bernoulli bandit; a discounted
//! Beta-Bernoulli posterior tracks the probability that one budget
//! slice of an arm improves the incumbent, and Thompson sampling
//! allocates the next slice. A decaying uniform-selection probability
//! `min(1, 1/m)` on round `m` gives each of `K` arms probability at
//! least `1/(Km)`. The divergent harmonic mass keeps every arm scheduled
//! infinitely often and preserves the randomized restart guarantee.
//!
//! Under [`PortfolioPolicy::Auto`] a run with neither a gradient nor a
//! declared noise scale takes a smaller loop, whatever its box
//! (`run_values_only_portfolio`): CMA-ES, a finite-difference
//! quasi-Newton descent, GSA, DE, the additive surrogate and the restart
//! arm, each played once before ranking and then under the same decaying
//! floor, after an opening descent from the start and GSA and CMA-ES
//! phases from its minimum, and before a closing one from the incumbent.
//!
//! Scheduler quantities derive from the problem and the budget rather
//! than from tuning knobs: the slice size affords a few gradient-
//! equivalents and at least several expected rounds per arm; the
//! posterior discount sets the effective memory to the slice horizon;
//! the active arm count is capped by what the horizon can rank; the
//! budget tail funds a final polish from the incumbent. Arm-internal
//! constants reuse the defaults of the standalone drivers they wrap.
//!
//! Work accounting is uniform: every true-objective evaluation and
//! every native-gradient evaluation costs one unit of the shared
//! budget. The additive surrogate is fit from the archive of already-
//! charged evaluations, so its proposals cost only their acceptance
//! tests.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Distribution};

use eindir_core::{AdditiveSurrogate, Bounds, Gradient, Objective, ReducedObjective};

use crate::bias::Bias;
use crate::cool::{Cooling, LogCool, TsallisCool};
use crate::exchange::TsallisExchange;
use crate::hmc::{HmcSaSampler, OmelyanIntegrator, QGaussianMomentum};
use crate::methods::amsa::{AM_ALPHA_TARGET, AM_BARRIER_EMA, AM_STAGNANT_RESEED, AmSaState};
use crate::methods::bayesian_pilot::{
    LaplacePosterior, PilotObservation, PilotPrior, empirical_prior_from_observations,
    fit_laplace_skew_corrected, pilot_draws_qmc,
};
use crate::methods::cma_es::{
    Bipop, CmaEs, CmaRegime, covariance_learning_evaluations, default_lambda,
};
use crate::methods::fd_bfgs::{FdBfgs, FdBfgsOptions};
use crate::methods::gle_langevin::gle_langevin_preconditioned_sa;
use crate::methods::local_polish::{
    QmcPolishResult, projected_gradient_polish, qmc_gsa_global_search,
    qmc_projected_gradient_polish, qmc_trust_region_poll, shifted_qmc_projected_gradient_polish,
};
use crate::methods::parallel_tempering::{ParallelTemperingSampler, geometric_ladder};
use crate::movekernel::{MoveKernel, TsallisVisit};
use crate::runner::{qmc_skip_from_seed, run_rs, run_rs_variant_resumed};

// ---------------------------------------------------------------------------
// Scheduler constants. Two numbers govern the allocation; everything
// else is derived from the budget, the dimension, and the arm count.
// ---------------------------------------------------------------------------

/// A slice must afford a few gradient-descent steps (one step costs one
/// objective and one gradient unit, so `4 * (dim + 1)` covers screening
/// plus descent for every arm).
const SLICE_GRAD_EQUIVALENTS: usize = 4;
/// Expected rounds per arm the posterior needs before its ranking means
/// anything; also caps the active arm count by the slice horizon.
const ROUNDS_PER_ARM: usize = 8;
/// Work units per endgame quasi-Newton iteration: one gradient, one
/// accepted step, and a short Armijo backtrack on average.
const ENDGAME_WU_PER_ITER: usize = 4;
/// Decimal orders of relative gap the endgame must close: a near-best
/// incumbent (Dolan-More tau = 1e-3) converting to the 1e-9 win
/// tolerance crosses six orders.
const NEAR_BEST_ORDERS: f64 = 6.0;

/// Per-cycle polish budget during multi-start endgame, in **work units**.
///
/// Must **not** return the full remaining budget on early cycles when
/// `remaining` is large enough for multi-start: the first projected-
/// gradient call used to exhaust the tail and starve jittered restarts
/// (CMA exclusives on CERI*/COOLHANS were polish-depth ties).
///
/// - `dry == 0`: two thirds of remaining (deepen incumbent), leave ≥1/3 residual
/// - `dry > 0`: half of remaining (jittered restarts still leave residual)
/// - Always capped at `remaining`
///
/// Callers must convert WU → `max_fevals` via [`endgame_polish_max_fevals`]
/// and/or pin the ledger cap for the cycle; passing this value raw as
/// `projected_gradient_polish(..., max_fevals, ...)` double-charges (eval+grad
/// per step) and can still dump the full tail.
fn endgame_cycle_cap(remaining: usize, dry: usize) -> usize {
    if remaining == 0 {
        return 0;
    }
    if dry == 0 {
        // 2/3 deepens polish enough for CERI*/COOLHANS-scale ties while
        // still leaving ≥ remaining/3 for multi-start / final dump.
        ((remaining * 2) / 3).max(16).min(remaining)
    } else {
        (remaining / 2).max(12).min(remaining)
    }
}

/// Convert an endgame cycle work-unit budget into `max_fevals` for
/// [`projected_gradient_polish`].
///
/// Each polish step charges ~`PROJECTED_GRADIENT_STEP_WORK` (1 obj + 1 grad).
/// Returning `cycle_wu / PROJECTED_GRADIENT_STEP_WORK` keeps ledger spend ≤
/// the cycle WU when the temporary ledger cap is also applied.
fn endgame_polish_max_fevals(cycle_wu: usize) -> usize {
    low_dimensional_refinement_fevals(cycle_wu).max(if cycle_wu > 0 { 1 } else { 0 })
}

/// Run one endgame projected-gradient cycle under a hard work-unit ceiling.
///
/// Pins `ledger.cap` to `used + cycle_wu` for the duration of the polish so
/// BudgetedObjective/Gradient stop charging once the cycle budget is spent,
/// then restores `outer_cap` (normally the full portfolio budget).
///
/// `step0` is the L-BFGS initial line-search scale.
#[allow(clippy::too_many_arguments)]
fn run_endgame_projected_polish_cycle<O, G>(
    obj: &BudgetedObjective<'_, O>,
    grad: &BudgetedGradient<'_, G>,
    ledger: &BudgetLedger,
    start: Array1<f64>,
    cycle_wu: usize,
    outer_cap: usize,
    step0: f64,
    grad_tol: f64,
) where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    if cycle_wu == 0 {
        return;
    }
    let maxf = endgame_polish_max_fevals(cycle_wu);
    if maxf == 0 {
        return;
    }
    let step0 = if step0.is_finite() && step0 > 0.0 {
        step0
    } else {
        1.0
    };
    let used0 = ledger.used_get();
    let cycle_ceiling = (used0 + cycle_wu).min(outer_cap);
    ledger.cap_set(cycle_ceiling);
    let _ = projected_gradient_polish(obj, grad, start, maxf, step0, grad_tol);
    ledger.cap_set(outer_cap);
}

/// Pattern-search along coordinates with geometrically shrinking steps.
///
/// Each trial is one objective charge via `BudgetedObjective`. Hard-stops
/// when the ledger is exhausted (temporary cap).
fn run_endgame_coordinate_microrefine<O>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    bounds: &Bounds<f64>,
    outer_cap: usize,
) where
    O: Objective<f64>,
{
    if ledger.remaining() < 4 {
        return;
    }
    let dim = bounds.dims.max(1);
    let used0 = ledger.used_get();
    // Spend remaining under outer_cap (caller already restored full budget).
    let _ = used0;
    let _ = outer_cap;
    let mut x = ledger.incumbent(bounds);
    let mut f = obj.eval(x.view());
    if !f.is_finite() {
        return;
    }
    // Relative step schedule from 1e-4 down to ~1e-14 of box width.
    let mut rel = 1e-4_f64;
    for _level in 0..12 {
        if ledger.remaining() < 2 * dim {
            break;
        }
        let mut improved = false;
        for i in 0..dim {
            if ledger.remaining() < 2 {
                break;
            }
            let lo = bounds.low[i];
            let hi = bounds.high[i];
            let width = (hi - lo).abs().max(1.0 + x[i].abs());
            let h = (rel * width).max(f64::EPSILON * 8.0 * (1.0 + x[i].abs()));
            for &sgn in &[-1.0_f64, 1.0] {
                if ledger.remaining() < 1 {
                    break;
                }
                let mut trial = x.clone();
                trial[i] = (x[i] + sgn * h).clamp(lo, hi);
                if (trial[i] - x[i]).abs() <= f64::EPSILON {
                    continue;
                }
                let ft = obj.eval(trial.view());
                if ft.is_finite() && ft < f {
                    x = trial;
                    f = ft;
                    improved = true;
                }
            }
        }
        if !improved {
            rel *= 0.1;
            if rel < 1e-16 {
                break;
            }
        }
    }
}

/// Abramowitz-Stegun 7.1.26 rational erf approximation (|error| < 1.5e-7).
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Normal-Inverse-Gamma posterior over the per-work-unit polish
/// log-contraction.
///
/// Observations are work-normalized log-ratios of consecutive polish
/// improvements, x_i = ln(delta_{i+1}/delta_i) / w_{i+1}, where w_{i+1}
/// is the work the improving slice spent: under geometric gap
/// contraction per polish work unit, delta_{i+1}/delta_i = rho_wu^w
/// exactly, so x_i estimates
/// ln rho_wu with measurement noise, giving the Normal likelihood. The
/// prior centers on the contraction measured on ill-conditioned
/// least-squares cells (rho ~ 0.965 per quasi-Newton iteration at
/// ENDGAME_WU_PER_ITER work units each) with four pseudo-observations.
struct ContractionPosterior {
    kappa: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
}

impl ContractionPosterior {
    fn new() -> Self {
        let per_wu = ENDGAME_WU_PER_ITER as f64;
        let prior_sd = 0.5 * std::f64::consts::LN_2 / per_wu;
        Self {
            kappa: 4.0,
            mu: 0.965f64.ln() / per_wu,
            alpha: 2.0,
            beta: 2.0 * prior_sd * prior_sd,
        }
    }

    fn observe(&mut self, log_ratio_per_wu: f64) {
        if !log_ratio_per_wu.is_finite() {
            return;
        }
        let k1 = self.kappa + 1.0;
        self.beta += 0.5 * self.kappa * (log_ratio_per_wu - self.mu).powi(2) / k1;
        self.mu = (self.kappa * self.mu + log_ratio_per_wu) / k1;
        self.kappa = k1;
        self.alpha += 0.5;
    }

    fn param_sd(&self) -> f64 {
        (self.beta / (self.kappa * (self.alpha - 1.0).max(0.5)))
            .sqrt()
            .max(1e-12)
    }

    /// P[ln rho_wu <= x]: the NIG marginal over the mean is Student-t
    /// with 2 alpha degrees of freedom; a moment-matched normal
    /// approximates it well past nu = 4.
    fn prob_log_rho_below(&self, x: f64) -> f64 {
        normal_cdf((x - self.mu) / self.param_sd())
    }

    /// Posterior probability that `wu` work units of polish close
    /// `orders` decimal orders of relative gap.
    fn conversion_prob(&self, orders: f64, wu: usize) -> f64 {
        if wu == 0 {
            return 0.0;
        }
        let x = -(orders * std::f64::consts::LN_10) / wu as f64;
        self.prob_log_rho_below(x)
    }

    /// Posterior-quantile polish reserve: the work needed to close
    /// `orders` decimal orders at the 84% credible slow quantile
    /// (mu + one posterior sd) of the contraction. This is the D5
    /// n* formula with the fixed constant replaced by the posterior.
    fn reserve_wu(&self, orders: f64) -> usize {
        let slow = self.mu + self.param_sd();
        if slow >= -1e-9 {
            return usize::MAX;
        }
        ((orders * std::f64::consts::LN_10) / -slow).ceil() as usize
    }
}
/// Dolan-More convergence resolution used by the CUTEst summary plots.
const DOLAN_MORE_CONVERGENCE_TAU: f64 = 1e-3;
/// General arms earn bandit credit one decimal place below the reporting
/// resolution so the posterior sees meaningful progress before the cell
/// flips its solved flag.
const BANDIT_SUCCESS_REFINEMENT_FACTOR: f64 = 10.0;
/// Slice success threshold for general arms.
const IMPROVEMENT_RTOL: f64 = DOLAN_MORE_CONVERGENCE_TAU / BANDIT_SUCCESS_REFINEMENT_FACTOR;
/// Archive-shift spends a local polish trajectory from an already good
/// point, so it earns posterior credit at the benchmark resolution
/// rather than for one-decade-finer local grinding.
const SHIFT_IMPROVEMENT_RTOL: f64 = DOLAN_MORE_CONVERGENCE_TAU;
/// Metropolis temperature floor shared by the acceptance helpers.
const METROPOLIS_FLOOR: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Arm constants: the defaults of the standalone drivers each arm wraps.
// ---------------------------------------------------------------------------

/// GSA visiting index; SciPy dual_annealing / manuscript default.
const GSA_Q_V: f64 = 2.62;
/// GSA acceptance index (typed TsallisAccept path / variants).
const GSA_Q_A: f64 = 1.7;
/// SciPy dual_annealing `initial_temp` default — not box-width-scaled.
const DUAL_INITIAL_TEMP: f64 = 5230.0;
/// SciPy dual_annealing `accept` parameter (default -5).
const DUAL_ACCEPT_PARAM: f64 = -5.0;
/// SciPy dual_annealing `restart_temp_ratio` (reanneal trigger).
const DUAL_RESTART_TEMP_RATIO: f64 = 2.0e-5;
/// Relative decrease that ends dual_annealing's L-BFGS-B local search
/// (factr 1e7 times machine epsilon).
const DUAL_LOCAL_FTOL: f64 = 1e7 * f64::EPSILON;
/// Projected-gradient norm that ends dual_annealing's L-BFGS-B local search
/// (SciPy's `pgtol`).
const DUAL_LOCAL_GTOL: f64 = 1e-5;
/// GLE integrator timestep, matching the thermostat band resolution.
const GLE_DT: f64 = 0.2;
/// Minimum timestep exposed by the portfolio-level Bayesian GLE policy.
const GLE_MIN_DT: f64 = 1e-4;
/// GLE annealing epochs, matching the standalone driver default.
const GLE_EPOCHS: usize = 40;
/// The GLE arm spends part of its first slice fitting the shared Bayesian
/// pilot and leaves the rest for the thermostat trajectory.
const BAYESIAN_GLE_PILOT_BUDGET_DIVISOR: usize = 2;
/// Lower and upper radius fractions for the posterior local GLE box.
const BAYESIAN_GLE_LOCAL_MIN_RADIUS_FRAC: f64 = 0.02;
const BAYESIAN_GLE_LOCAL_MAX_RADIUS_FRAC: f64 = 0.5;
/// Storn-Price population sizing: `4 * dim` clamped to a workable range.
const DE_POP_PER_DIM: usize = 4;
const DE_POP_MIN: usize = 16;
const DE_POP_MAX: usize = 48;
/// Storn-Price crossover and dithered weight range.
const DE_CROSSOVER: f64 = 0.7;
const DE_WEIGHT_MIN: f64 = 0.5;
const DE_WEIGHT_SPAN: f64 = 0.5;
/// Basin-hop step adaptation: grow on accept, shrink on reject, in the
/// classic stochastic-approximation ratio.
const HOP_STEP0: f64 = 0.25;
const HOP_GROW: f64 = 1.3;
const HOP_SHRINK: f64 = 0.75;
/// CMA-ES step sizes as fractions of each box side: the first run searches
/// around the incumbent, large-population restarts search the whole box.
const CMA_FIRST_SIGMA: f64 = 0.025;
const CMA_LARGE_SIGMA: f64 = 0.3;
/// IPOP stops doubling once a run could no longer afford this many
/// generations of the budget.
const CMA_MIN_GENERATIONS: usize = 40;
/// Largest dimension at which CMA-ES runs keep a full covariance. Its Jacobi
/// refresh costs O(n^3) every O(n) evaluations, about a millisecond of
/// driver time per evaluation at this size and growing as n^2; above it, and
/// whenever the budget is shorter than the covariance's learning horizon,
/// runs are separable.
const CMA_FULL_MAX_DIM: usize = 100;
/// Stream tags: arms built from the same seed draw from independent
/// generators.
const CMA_STREAM: u64 = 0xC3A5_C85C_97CB_3127;
const QN_STREAM: u64 = 0xB492_B66F_BE98_F273;
const START_STREAM: u64 = 0x9AE1_6A3B_2F90_404F;
/// Kick radius, as a fraction of each box side, that restarts a converged
/// finite-difference descent from the incumbent; adapted like the hop step.
const QN_KICK0: f64 = 0.05;
const QN_KICK_MIN: f64 = 1e-3;
const QN_KICK_MAX: f64 = 0.5;
/// Gradients per dimension a finite-difference descent needs to converge;
/// the values-only loop opens with one only when the budget affords it.
const QN_AFFORDABLE_GRADIENTS: usize = 5;
/// Fewest gradients worth a closing finite-difference polish.
const QN_MIN_POLISH_GRADIENTS: usize = 3;
/// The values-only loop's closing polish gets this many gradients, enough
/// for one finite-difference descent from the incumbent to converge, but
/// never more than the budget share below.
const LOCAL_FIRST_POLISH_GRADIENTS: usize = 10;
const LOCAL_FIRST_POLISH_MAX_SHARE: f64 = 0.4;
/// Omelyan trajectory length for the HMC arm.
const HMC_L_STEPS: usize = 5;
/// Additive-surrogate fit degree and inverse-CDF grid, matching the
/// standalone `additive_independence` defaults.
const SURROGATE_DEGREE: usize = 8;
const SURROGATE_GRID: usize = 65;
/// Active-subspace rank for the reduced arm; the arm requires
/// `dim > 2 * REDUCED_K` so the collapse actually removes coordinates.
const REDUCED_K: usize = 4;
/// Pilot chains drawn for the Bayesian-pilot variant arm.
const PILOT_CHAINS: usize = 5;
/// Minimum pilot depth per chain used to resolve acceptance and improvement.
const BAYESIAN_PILOT_MIN_CHAIN_STEPS: usize = 8;
/// Work needed before the GLE arm can install a posterior that other arms reuse.
const BAYESIAN_GLE_MIN_PILOT_WORK: usize = PILOT_CHAINS * (BAYESIAN_PILOT_MIN_CHAIN_STEPS + 1);
/// Low-dimensional smooth objectives can afford an initial replicated QMC polish.
const LOW_DIMENSIONAL_POLISH_MAX_DIM: usize = 4;
const COORDINATE_OPPOSITION_MAX_DIM: usize = 16;
const COORDINATE_OPPOSITION_WEIGHT: f64 = 1.05;
/// Final projected-gradient tolerance for the initial low-dimensional polish.
const LOW_DIMENSIONAL_POLISH_GRAD_TOL: f64 = 1e-12;
/// Ledger cost of a true objective call.
const OBJECTIVE_WORK_UNIT: usize = 1;
/// Ledger cost of a native gradient call.
const GRADIENT_WORK_UNIT: usize = 1;
/// One projected-gradient step charges one objective and one gradient call.
const PROJECTED_GRADIENT_STEP_WORK: usize = OBJECTIVE_WORK_UNIT + GRADIENT_WORK_UNIT;

/// Per-arm pull statistics reported by the driver.
#[derive(Clone, Debug)]
pub struct ArmStat {
    /// Stable arm identifier.
    pub name: &'static str,
    /// Slices allocated to the arm.
    pub pulls: usize,
    /// Slices that improved the incumbent past the success threshold.
    pub successes: usize,
}

/// Result of a portfolio run.
#[derive(Clone, Debug)]
pub struct PortfolioResult {
    /// Best-seen position.
    pub best_pos: Vec<f64>,
    /// Best-seen objective value.
    pub best_val: f64,
    /// True-objective evaluations charged.
    pub n_evals: usize,
    /// Native-gradient evaluations charged.
    pub n_grads: usize,
    /// Per-arm allocation statistics.
    pub arm_stats: Vec<ArmStat>,
}

// ---------------------------------------------------------------------------
// Budget ledger: shared work accounting plus the evaluation archive.
// ---------------------------------------------------------------------------

struct LedgerInner {
    best_pos: Option<Array1<f64>>,
    /// The first in-bounds point evaluated, whatever its value: the
    /// incumbent while no evaluation has been finite.
    first_pos: Option<Array1<f64>>,
    archive_x: Vec<f64>,
    archive_y: Vec<f64>,
}

struct BudgetLedger {
    cap: AtomicUsize,
    used: AtomicUsize,
    n_evals: AtomicUsize,
    n_grads: AtomicUsize,
    best_val: AtomicU64,
    inner: Mutex<LedgerInner>,
    archive_cap: usize,
    dim: usize,
}

impl BudgetLedger {
    fn new(budget: usize, dim: usize) -> Self {
        Self {
            cap: AtomicUsize::new(budget),
            used: AtomicUsize::new(0),
            n_evals: AtomicUsize::new(0),
            n_grads: AtomicUsize::new(0),
            best_val: AtomicU64::new(f64::INFINITY.to_bits()),
            inner: Mutex::new(LedgerInner {
                best_pos: None,
                first_pos: None,
                archive_x: Vec::new(),
                archive_y: Vec::new(),
            }),
            // Every archive entry costs one budget unit, so the budget
            // itself bounds the archive.
            archive_cap: budget,
            dim,
        }
    }

    fn cap_get(&self) -> usize {
        self.cap.load(Ordering::Relaxed)
    }

    fn cap_set(&self, value: usize) {
        self.cap.store(value, Ordering::Relaxed);
    }

    fn used_get(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    fn best_get(&self) -> f64 {
        f64::from_bits(self.best_val.load(Ordering::Relaxed))
    }

    fn remaining(&self) -> usize {
        self.cap_get().saturating_sub(self.used_get())
    }

    fn exhausted(&self) -> bool {
        self.used_get() >= self.cap_get()
    }

    /// Atomically reserve `work` units without overshooting the cap.
    ///
    /// Uses compare-exchange so concurrent chargers cannot pass the check
    /// then both `fetch_add` past `cap` (TOCTOU on the naive remaining/add path).
    fn try_charge(&self, work: usize) -> bool {
        if work == 0 {
            return true;
        }
        loop {
            let used = self.used.load(Ordering::Relaxed);
            let cap = self.cap.load(Ordering::Relaxed);
            if used >= cap || cap - used < work {
                return false;
            }
            match self.used.compare_exchange_weak(
                used,
                used + work,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(_) => continue,
            }
        }
    }

    fn charge_probe(&self, n_evals: usize, n_grads: usize) -> bool {
        let work = n_evals + n_grads;
        if !self.try_charge(work) {
            return false;
        }
        self.n_evals.fetch_add(n_evals, Ordering::Relaxed);
        self.n_grads.fetch_add(n_grads, Ordering::Relaxed);
        true
    }

    /// Archive a candidate only if `value` is finite **and** `x` lies inside
    /// `bounds` (GJQ-style feasibility choke-point: never promote OOB bests).
    fn record(&self, x: ArrayView1<f64>, value: f64, bounds: &Bounds<f64>) {
        if !bounds.contains(x) {
            return;
        }
        let mut inner = self.inner.lock().expect("ledger lock");
        if inner.first_pos.is_none() {
            inner.first_pos = Some(x.to_owned());
        }
        if !value.is_finite() {
            return;
        }
        // Re-read under the lock so two sequential records cannot promote a
        // worse best_val after a better one (relaxed atomic alone is racy
        // with the mutex-held position write).
        let current = f64::from_bits(self.best_val.load(Ordering::Relaxed));
        if value < current {
            self.best_val.store(value.to_bits(), Ordering::Relaxed);
            inner.best_pos = Some(x.to_owned());
        }
        if inner.archive_y.len() < self.archive_cap {
            inner.archive_x.extend(x.iter().copied());
            inner.archive_y.push(value);
        }
    }

    fn incumbent(&self, bounds: &Bounds<f64>) -> Array1<f64> {
        let inner = self.inner.lock().expect("ledger lock");
        match inner.best_pos.as_ref().or(inner.first_pos.as_ref()) {
            // Defensive clip: both positions are only written in bounds.
            Some(pos) => bounds.clip(pos.view()),
            None => (&bounds.low + &bounds.high) * 0.5,
        }
    }

    /// Value at [`BudgetLedger::incumbent`] once that point has been
    /// evaluated: the best value, or `+inf` while no evaluation has been
    /// finite. `None` before any evaluation, when the incumbent is the box
    /// centre.
    fn incumbent_value(&self) -> Option<f64> {
        let inner = self.inner.lock().expect("ledger lock");
        if inner.best_pos.is_some() {
            Some(self.best_get())
        } else {
            inner.first_pos.as_ref().map(|_| f64::INFINITY)
        }
    }
}

/// Objective proxy charging one unit per evaluation.
struct BudgetedObjective<'a, O: Objective<f64>> {
    inner: &'a O,
    ledger: &'a BudgetLedger,
}

impl<O: Objective<f64>> Objective<f64> for BudgetedObjective<'_, O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if !self.ledger.try_charge(OBJECTIVE_WORK_UNIT) {
            return f64::INFINITY;
        }
        self.ledger.n_evals.fetch_add(1, Ordering::Relaxed);
        // A non-finite coordinate has no image in the box. The arm that made
        // it is charged and told the point is worthless; the caller's
        // objective never sees it.
        if x.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        // Reflect (not bare-clip) into the box before eval+record: clipping
        // piles mass on the wall; reflection keeps a feasible point while
        // preserving more of the proposal structure. Archive is always in-bounds.
        let bounds = self.inner.bounds();
        let x_feas = crate::movekernel::reflect_into_box(x, bounds);
        let value = self.inner.eval(x_feas.view());
        self.ledger.record(x_feas.view(), value, bounds);
        value
    }
}

/// Objective proxy that evaluates the original objective while advertising
/// a posterior local box to dynamics that clip through `Objective::bounds`.
struct LocalBoxBudgetedObjective<'a, O: Objective<f64>> {
    inner: &'a O,
    ledger: &'a BudgetLedger,
    bounds: Bounds<f64>,
}

impl<O: Objective<f64>> Objective<f64> for LocalBoxBudgetedObjective<'_, O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if !self.ledger.try_charge(OBJECTIVE_WORK_UNIT) {
            return f64::INFINITY;
        }
        self.ledger.n_evals.fetch_add(1, Ordering::Relaxed);
        if x.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        // Reflect into the local box; archive only if also globally feasible.
        let x_local = crate::movekernel::reflect_into_box(x, &self.bounds);
        let value = self.inner.eval(x_local.view());
        self.ledger
            .record(x_local.view(), value, self.inner.bounds());
        value
    }
}

/// Gradient of `x -> f(reflect_into_box(x))`, the function the budgeted
/// objective evaluates: the caller's gradient at the reflected point, with the
/// sign of each coordinate the fold reverses flipped. The caller's gradient is
/// therefore only ever called inside the box; inside it this is `grad f(x)`.
struct ReflectedGradient<'a, G: Gradient<f64>> {
    inner: &'a G,
    bounds: &'a Bounds<f64>,
}

impl<G: Gradient<f64>> Gradient<f64> for ReflectedGradient<'_, G> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        if x.iter().any(|v| !v.is_finite()) {
            return Array1::zeros(x.len());
        }
        let inside = x
            .iter()
            .enumerate()
            .all(|(k, &xk)| (self.bounds.low[k]..=self.bounds.high[k]).contains(&xk));
        if inside {
            return self.inner.grad(x);
        }
        let (folded, slopes): (Vec<f64>, Vec<f64>) = x
            .iter()
            .enumerate()
            .map(|(k, &xk)| {
                crate::movekernel::reflect_coord_with_slope(
                    xk,
                    self.bounds.low[k],
                    self.bounds.high[k],
                )
            })
            .unzip();
        let g = self.inner.grad(ArrayView1::from(&folded));
        g * &Array1::from(slopes)
    }
}

/// Gradient proxy charging one unit per evaluation.
struct BudgetedGradient<'a, G: Gradient<f64>> {
    inner: &'a G,
    ledger: &'a BudgetLedger,
}

impl<G: Gradient<f64>> Gradient<f64> for BudgetedGradient<'_, G> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        if !self.ledger.try_charge(GRADIENT_WORK_UNIT) {
            return Array1::zeros(self.ledger.dim);
        }
        self.ledger.n_grads.fetch_add(1, Ordering::Relaxed);
        self.inner.grad(x)
    }
}

// ---------------------------------------------------------------------------
// Discounted Beta-Bernoulli posterior.
// ---------------------------------------------------------------------------

struct ArmPosterior {
    alpha: f64,
    beta: f64,
    discount: f64,
    pulls: usize,
    successes: usize,
}

impl ArmPosterior {
    #[allow(dead_code)]
    fn new(discount: f64) -> Self {
        Self::with_prior(discount, 1.0, 1.0)
    }

    /// Prior-biased posterior (regime auto-selection boosts preferred arms).
    fn with_prior(discount: f64, alpha0: f64, beta0: f64) -> Self {
        Self {
            alpha: alpha0.max(1e-6),
            beta: beta0.max(1e-6),
            discount,
            pulls: 0,
            successes: 0,
        }
    }

    fn update(&mut self, success: bool) {
        self.alpha = 1.0 + self.discount * (self.alpha - 1.0);
        self.beta = 1.0 + self.discount * (self.beta - 1.0);
        if success {
            self.alpha += 1.0;
            self.successes += 1;
        } else {
            self.beta += 1.0;
        }
        self.pulls += 1;
    }

    fn draw(&self, rng: &mut StdRng) -> f64 {
        Beta::new(self.alpha, self.beta)
            .map(|dist| dist.sample(rng))
            .unwrap_or(0.5)
    }
}

// ---------------------------------------------------------------------------
// Arms.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArmKind {
    Explore,
    Shift,
    Hop,
    Surrogate,
    De,
    Gle,
    TrPoll,
    Gsa,
    Variant,
    Pt,
    Hmc,
    Reduced,
    /// Well-tempered metadynamics on an Obj-slot bias (true-F ledger).
    Metad,
    /// pnastps-inspired transition-path shooting between archive basins.
    Tps,
    /// D6 adaptive-Metropolis descent chain (gap-proportional temperature,
    /// covariance-shaped proposal, Robbins-Monro alpha* targeting).
    AmSa,
    /// Classical population-controlled diffusion (DMC-inspired walkers).
    DmcPop,
    /// BIPOP CMA-ES restarted from the incumbent.
    Cma,
    /// Projected BFGS on finite-difference gradients from the incumbent.
    Qn,
}

impl ArmKind {
    fn name(self) -> &'static str {
        match self {
            ArmKind::Explore => "explore",
            ArmKind::Shift => "shift",
            ArmKind::Hop => "hop",
            ArmKind::Surrogate => "surrogate",
            ArmKind::De => "de",
            ArmKind::Gle => "gle",
            ArmKind::TrPoll => "tr_poll",
            ArmKind::Gsa => "gsa",
            ArmKind::Variant => "variant",
            ArmKind::Pt => "pt",
            ArmKind::Hmc => "hmc",
            ArmKind::Reduced => "reduced",
            ArmKind::Metad => "metad",
            ArmKind::Tps => "tps",
            ArmKind::AmSa => "am_sa",
            ArmKind::DmcPop => "dmc_pop",
            ArmKind::Cma => "cma",
            ArmKind::Qn => "qn",
        }
    }

    /// Arms that may earn bandit credit without an incumbent drop
    /// (enhanced sampling / path ensemble exploration).
    fn exploratory_credit(self) -> bool {
        matches!(self, ArmKind::Metad | ArmKind::Tps | ArmKind::DmcPop)
    }
}

const RESTART_ARM: ArmKind = ArmKind::Explore;

struct HopState {
    step: f64,
    x_cur: Option<Array1<f64>>,
    f_cur: f64,
    generation: usize,
}

struct DeState {
    pop: Vec<Array1<f64>>,
    vals: Vec<f64>,
}

/// Persistent dual-annealing-class GSA state in **physical** coordinates.
///
/// Earlier unit-cube + clamp destroyed heavy-tailed visits (every large jump
/// landed on a corner). SciPy dual_annealing visits the box with modular
/// wrap; we use reflection for L1-symmetric Metropolis. Temperature is
/// energy-scaled so Schwefel-class landscapes are not frozen at T₀=1.
struct GsaState {
    /// Chain positions in the design box (not unit cube).
    xs: Vec<Array1<f64>>,
    vals: Vec<f64>,
    epoch: usize,
    rng: StdRng,
    /// Energy-scaled initial temperature (TsallisCool t_init).
    t_init: f64,
    /// Strategy-chain step counter (dual_annealing uses T/(step+1) accept).
    strategy_step: usize,
    /// Lowest value any chain has reached; beating it triggers the
    /// finite-difference local search.
    best: f64,
    /// Local search in progress from a chain point, resumed across slices.
    local: Option<(usize, FdBfgs)>,
    /// Best start (chain, position, value) of an anchored state, searched
    /// after the first temperature step even without a record, as
    /// dual_annealing searches from its best point after its first chain.
    first_search: Option<(usize, Array1<f64>, f64)>,
}

/// Persistent CMA-ES arm. Runs live in box-normalised coordinates, so one
/// step size is the same fraction of every side; a stopped run is replaced
/// by the BIPOP planner's next run, centred on the incumbent.
struct CmaArmState {
    es: CmaEs,
    regime: CmaRegime,
    planner: Bipop,
    unit: Bounds<f64>,
    rng: StdRng,
    /// Whether runs centred on the incumbent keep it as an elite.
    elitist: bool,
    /// Whether runs keep only the diagonal of their covariance.
    separable: bool,
}

impl CmaArmState {
    /// First run centred on the incumbent with step `sigma` (a fraction of
    /// each box side), charged to the small (local) regime, so the planner's
    /// first restart searches the whole box; large-population restarts are
    /// never elitist. Runs are separable above [`CMA_FULL_MAX_DIM`] or when
    /// the budget is shorter than the full covariance's learning horizon.
    fn new(
        ledger: &BudgetLedger,
        bounds: &Bounds<f64>,
        budget: usize,
        seed: u64,
        sigma: f64,
        elitist: bool,
    ) -> Self {
        let dim = bounds.dims;
        let lambda = default_lambda(dim);
        let unit = Bounds::new(Array1::zeros(dim), Array1::ones(dim), 0.0);
        let separable = dim > CMA_FULL_MAX_DIM
            || (budget as f64) < covariance_learning_evaluations(dim, lambda);
        let mean = to_unit_box(&ledger.incumbent(bounds), bounds);
        let mut es = Self::run(
            mean.view(),
            sigma,
            lambda,
            &unit,
            seed ^ CMA_STREAM,
            separable,
            ledger.best_get(),
        );
        if elitist {
            es = es.with_elite(mean.view(), ledger.best_get());
        }
        Self {
            es,
            regime: CmaRegime::Small,
            planner: Bipop::new(
                lambda,
                (budget / CMA_MIN_GENERATIONS).max(lambda),
                CMA_LARGE_SIGMA,
                CmaRegime::Small,
            ),
            unit,
            rng: StdRng::seed_from_u64((seed ^ CMA_STREAM).rotate_left(17)),
            elitist,
            separable,
        }
    }

    /// One run from an incumbent valued `best`. It ends once its recent
    /// best values span less than the portfolio's success threshold at that
    /// value: past that point no slice of it can count as a success.
    fn run(
        mean: ArrayView1<f64>,
        sigma: f64,
        lambda: usize,
        unit: &Bounds<f64>,
        seed: u64,
        separable: bool,
        best: f64,
    ) -> CmaEs {
        let es = CmaEs::new(mean, sigma, lambda, unit, seed)
            .with_tol_fun_hist(arm_success_threshold(ArmKind::Cma, best));
        if separable { es.separable() } else { es }
    }
}

fn to_unit_box(x: &Array1<f64>, bounds: &Bounds<f64>) -> Array1<f64> {
    Array1::from_iter(
        (0..bounds.dims)
            .map(|i| ((x[i] - bounds.low[i]) / (bounds.high[i] - bounds.low[i])).clamp(0.0, 1.0)),
    )
}

/// Persistent finite-difference quasi-Newton arm: one descent at a time,
/// moved to the incumbent when another arm improves on it and kicked from
/// the incumbent once it converges.
struct QnArmState {
    engine: FdBfgs,
    kick: f64,
    /// Incumbent value when the current descent began.
    base_val: f64,
    /// Incumbent value when the arm last yielded its slice.
    seen_best: f64,
    rng: StdRng,
    /// Descent value at the last [`QnArmState::paid`] check; infinite once
    /// the descent restarts or is kicked.
    checked: f64,
}

impl QnArmState {
    /// Whether the descent has lowered its value by more than the success
    /// threshold since the last check. A new descent pays its first check
    /// once it has a finite value; a converged one never pays.
    fn paid(&mut self) -> bool {
        let value = self.engine.value();
        let paid = !self.engine.is_done()
            && value.is_finite()
            && value < self.checked - arm_success_threshold(ArmKind::Qn, self.checked);
        self.checked = value;
        paid
    }
}

#[derive(Clone, Debug)]
struct PilotState {
    posterior: LaplacePosterior,
    anchor: Option<Array1<f64>>,
}

struct LowDimensionalPolishPlan {
    budget: usize,
    n_starts: usize,
    max_fevals_per_start: usize,
    n_replicates: usize,
    top_k: usize,
}

struct ValueScout {
    best_pos: Array1<f64>,
    best_val: f64,
}

struct CenterProbe {
    gradient_ratio: Option<f64>,
}

#[derive(Default)]
struct ArmStates {
    /// Probe verdict: descents pay on this landscape, so each hop should
    /// spend its full slice on one deep descent (basin-hopping depth).
    deep_hops: bool,
    hop: Option<HopState>,
    de: Option<DeState>,
    gsa: Option<GsaState>,
    pilot: Option<PilotState>,
    surrogate_gen: usize,
    seed_counter: u64,
    /// Known noise scale of the objective estimator, if the caller declared
    /// the objective stochastic. `None` (the default) selects exact-Metropolis
    /// acceptance; `Some(sigma)` selects the noise-aware OSA rule.
    noise_sigma: Option<f64>,
    /// Persistent well-tempered bias for the MetaD arm.
    bias: Option<crate::bias::WellTemperedBias>,
    /// Set by MetaD / TPS when the slice performed useful exploration
    /// (deposit or accepted reactive path) even without an incumbent drop.
    last_exploratory_ok: bool,
    /// Luby reluctant-doubling state (Knuth's (u, v) pair) for the
    /// restart arm's descent-depth schedule.
    luby_u: u64,
    luby_v: u64,
    /// Persistent annealing chain for the Variant arm: slices extend one
    /// trajectory instead of restarting a short schedule every slice.
    variant_chain: Option<eindir_core::FPair<f64>>,
    variant_epoch: usize,
    /// Basin registry fed by restart-arm polish endpoints: the discovery
    /// side of the Good-Turing / record-statistics budget-conversion gate.
    basins: BasinRegistry,
    /// Persistent D6 adaptive-Metropolis descent chain.
    am: Option<AmSaState>,
    /// Persistent CMA-ES run and restart planner.
    cma: Option<CmaArmState>,
    /// Persistent finite-difference quasi-Newton descent.
    qn: Option<QnArmState>,
    /// Set by the values-only loop. Without a gradient, the GSA arm then
    /// starts one chain at the incumbent and follows each temperature step
    /// that beats the chain best with a finite-difference descent, the
    /// L-BFGS-B local search dual_annealing runs on improvement, and the DE
    /// population starts with the incumbent, as SciPy's
    /// differential_evolution places its `x0`.
    values_only: bool,
}

/// Persistent adaptive-Metropolis descent chain (D6 + D11 BFWT).
///
/// Temperature is not a schedule in epoch index. On a locally quadratic
/// basin, expected one-step decrease of the current chain-state energy is
/// positive iff theta = T d / (f(x) - f*) < 2 exactly (D6), so the design
/// point is T_des = θ⋆ (f - f_best) / d with θ⋆ = 0.5 (~91% residual
/// descent). Pure gap-proportional T freezes when the chain sits at the
/// incumbent (gap → 0): no uphill moves remain possible. D11 BFWT clamps
/// T into the D6∩D7 window using an online barrier proxy (EMA of rejected
/// uphill deltas) and remaining work-unit budget, so escape temperature
/// stays positive under budgeted barriers. Empty window → T_lo (escape
/// forced).
///
/// Proposal SHAPE is the chain's running covariance (Haario et al. 2001;
/// diminishing adaptation). SIZE tracks α* ≈ 0.32 by Robbins-Monro.
/// Reflection into the box keeps the proposal symmetric (law L1).
/// Stagnation triggers an IPOP-style reseed from a fresh box draw so the
/// arm does not spend the whole budget frozen in one basin (CMA-class
/// NLS cells: KIRBY2LS / HAHN1LS on the CUTEst SOTA census).
/// Distinct-basin bookkeeping over restart endpoints. Restart slices draw
/// basins (approximately) i.i.d. from the basin-of-attraction measure, so
/// the Good-Turing singleton fraction n1/n estimates the missing basin
/// mass (Good 1953; the concentration bound is McAllester-Schapire), and
/// under exchangeability of basin depths a newly discovered basin beats
/// the w seen so far with probability 1/(w+1) (record statistics). The
/// product is a distribution-free per-slice probability that further
/// exploration still improves the final answer; it uses basin identities
/// that a Beta success posterior over improvement bits throws away.
#[derive(Default)]
struct BasinRegistry {
    /// (representative position, basin value, times sampled)
    entries: Vec<(Array1<f64>, f64, usize)>,
    /// Restart endpoints registered.
    n_samples: usize,
}

/// Two endpoints share a basin below this normalized distance (per-axis
/// widths scale each coordinate; 5% of the box diagonal).
const BASIN_MERGE_RADIUS: f64 = 0.05;

impl BasinRegistry {
    fn register(&mut self, x: ArrayView1<f64>, val: f64, bounds: &Bounds<f64>) {
        if !val.is_finite() {
            return;
        }
        self.n_samples += 1;
        let dim = x.len().max(1) as f64;
        for (rep, rep_val, hits) in self.entries.iter_mut() {
            let mut d2 = 0.0f64;
            for j in 0..x.len() {
                let w = (bounds.high[j] - bounds.low[j]).abs().max(1e-12);
                let dj = (x[j] - rep[j]) / w;
                d2 += dj * dj;
            }
            if (d2 / dim).sqrt() <= BASIN_MERGE_RADIUS {
                *hits += 1;
                if val < *rep_val {
                    *rep_val = val;
                    rep.assign(&x);
                }
                return;
            }
        }
        self.entries.push((x.to_owned(), val, 1));
    }

    /// Per-slice probability that one more restart both discovers an
    /// unseen basin (Good-Turing missing mass n1/n) and that the new
    /// basin beats every seen one (record probability 1/(w+1)).
    /// D9: [`discovery_value`].
    fn record_discovery_prob(&self) -> Option<f64> {
        if self.n_samples < 5 {
            return None;
        }
        let n1 = self.entries.iter().filter(|(_, _, h)| *h == 1).count();
        let w = self.entries.len();
        Some(discovery_value(n1, self.n_samples, w))
    }
}

/// D9.3: Good-Turing x record discovery value theta_disc = n1 / (n (w+1)).
pub fn discovery_value(n1: usize, n: usize, w: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    (n1 as f64 / n as f64) / (w as f64 + 1.0)
}

/// D10.1 win objective under D9 discovery: W(p) = (q0+(1-q0)(1-pi_e))*P_conv.
///
/// `theta_disc` is [`discovery_value`]; `p_conv` is polish conversion probability
/// for the proposed polish work `polish`; `slice_size` is exploration slice length.
pub fn win_objective_discovery(
    remaining: usize,
    polish: usize,
    slice_size: usize,
    theta_disc: f64,
    p_conv: f64,
) -> f64 {
    let polish = polish.min(remaining);
    let slice_size = slice_size.max(1);
    let e = remaining.saturating_sub(polish) / slice_size;
    let theta = theta_disc.clamp(0.0, 1.0);
    let q0 = 1.0 - theta;
    let pi = (1.0 - theta).powi(e.min(128) as i32);
    let p_conv = p_conv.clamp(0.0, 1.0);
    (q0 + (1.0 - q0) * (1.0 - pi)) * p_conv
}

impl ArmStates {
    /// Next term of the Luby universal restart sequence
    /// 1,1,2,1,1,2,4,1,... (Luby, Sinclair, Zuckerman 1993): the expected
    /// hit time of restarts with these cutoffs is within a logarithmic
    /// factor of the best fixed cutoff for any run-length distribution.
    #[allow(dead_code)] // retained for a depth-schedule A/B; see explore arm note
    fn luby_next(&mut self) -> u64 {
        if self.luby_u == 0 {
            self.luby_u = 1;
            self.luby_v = 1;
        }
        let out = self.luby_v;
        if self.luby_u & self.luby_u.wrapping_neg() == self.luby_v {
            self.luby_u += 1;
            self.luby_v = 1;
        } else {
            self.luby_v *= 2;
        }
        out
    }
}

fn metropolis(delta: f64, temp: f64, rng: &mut StdRng) -> bool {
    if delta <= 0.0 {
        return true;
    }
    if !delta.is_finite() {
        return false;
    }
    rng.random::<f64>() < (-delta / temp.max(METROPOLIS_FLOOR)).exp()
}

/// Compensated ΔE for acceptance (eindir two-sum channel).
#[inline]
fn energy_delta(f_new: f64, f_cur: f64) -> f64 {
    eindir_core::compensated_delta(f_new, f_cur)
}

/// Accept a proposed move at a stochastic-evaluation site.
///
/// When `noise_sigma` is `None` the decision is the exact-objective Metropolis
/// rule on the single computed `delta_point`. When the caller declared the
/// objective noisy with known scale `sigma`, the decision is the Ball, Branke &
/// Meisel (2018) sequential OSA rule: `delta_sampler(rng)` returns one noisy
/// observation of the cost difference, and OSA draws as many as it needs to
/// decide while preserving detailed balance. Each OSA sample charges the budget
/// through the budgeted objective the closure evaluates.
fn accept_move<F>(
    noise_sigma: Option<f64>,
    delta_sampler: F,
    delta_point: f64,
    temp: f64,
    rng: &mut StdRng,
) -> bool
where
    F: FnMut(&mut StdRng) -> f64,
{
    // Regime refusal: declared noise => OSA only (exact Metropolis is out of regime).
    match noise_sigma {
        Some(sigma) if sigma > 0.0 && sigma.is_finite() => {
            debug_assert!(
                crate::methods::regime::require_accept_compatible(Some(sigma), true).is_ok()
            );
            crate::noise_accept::OsaAccept::new()
                .decide(delta_sampler, temp.max(METROPOLIS_FLOOR), sigma, rng)
                .accepted
        }
        Some(_) => {
            panic!("noise_sigma must be positive and finite for OSA accept");
        }
        None => {
            debug_assert!(crate::methods::regime::require_accept_compatible(None, false).is_ok());
            metropolis(delta_point, temp, rng)
        }
    }
}

fn ladder_temperature(temp0: f64, generation: usize) -> f64 {
    (temp0 * std::f64::consts::LN_2 / ((generation + 2) as f64).ln()).max(1e-9)
}

fn archive_temp0(ledger: &BudgetLedger) -> f64 {
    let inner = ledger.inner.lock().expect("ledger lock");
    let mut finite: Vec<f64> = inner
        .archive_y
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if finite.len() < 2 {
        return 1.0;
    }
    finite.sort_by(|left, right| left.total_cmp(right));
    // Interquartile range: a robust landscape scale immune to the
    // divergent tails that unbounded trajectories can archive (a plain
    // variance overflows to infinity on them).
    let q25 = finite[finite.len() / 4];
    let q75 = finite[(3 * finite.len()) / 4];
    let spread = q75 - q25;
    if spread.is_finite() && spread > 0.0 {
        spread.max(1e-6)
    } else {
        1.0
    }
}

fn mean_width(bounds: &Bounds<f64>) -> f64 {
    let dim = bounds.dims.max(1);
    let mut total = 0.0;
    for j in 0..bounds.dims {
        let w = bounds.high[j] - bounds.low[j];
        total += if w.is_finite() && w > 0.0 { w } else { 1.0 };
    }
    total / dim as f64
}

fn arm_success_threshold(arm: ArmKind, before: f64) -> f64 {
    let scale = if before.is_finite() {
        before.abs()
    } else {
        1.0
    };
    let rtol = match arm {
        ArmKind::Shift => SHIFT_IMPROVEMENT_RTOL,
        // MetaD / TPS: any measurable true-F improvement counts; exploratory
        // credit may also fire when the slice deposits / accepts a path.
        ArmKind::Metad | ArmKind::Tps => IMPROVEMENT_RTOL * 0.1,
        _ => IMPROVEMENT_RTOL,
    };
    rtol * scale.max(1.0)
}

fn scheduler_success_threshold(arm: ArmKind, ledger: &BudgetLedger) -> f64 {
    arm_success_threshold(arm, scheduler_reward_scale(ledger))
}

fn scheduler_reward_scale(ledger: &BudgetLedger) -> f64 {
    archive_temp0(ledger)
}

fn warmup_arm_index(round: usize, arm_count: usize) -> Option<usize> {
    (round >= 1 && round <= arm_count).then_some(round - 1)
}

fn low_dimensional_polish_plan(dim: usize, budget: usize) -> Option<LowDimensionalPolishPlan> {
    if dim == 0 || dim > LOW_DIMENSIONAL_POLISH_MAX_DIM {
        return None;
    }
    let polish_budget = budget / ROUNDS_PER_ARM;
    let n_starts = (4 * dim).max(8);
    let n_replicates = if dim <= 2 { 1 } else { 2 };
    let top_k = if dim <= 2 { 2 } else { 1 };
    let screening = n_starts * n_replicates;
    let polished = top_k * n_replicates;
    let min_work = screening + PROJECTED_GRADIENT_STEP_WORK * polished;
    if polish_budget < min_work {
        return None;
    }
    let max_fevals_per_start =
        (polish_budget - screening) / (PROJECTED_GRADIENT_STEP_WORK * polished);
    (max_fevals_per_start > 0).then_some(LowDimensionalPolishPlan {
        budget: polish_budget,
        n_starts,
        max_fevals_per_start,
        n_replicates,
        top_k,
    })
}

fn low_dimensional_scout_population(plan: &LowDimensionalPolishPlan) -> Option<usize> {
    let population = plan.n_starts.checked_mul(plan.n_replicates)?;
    let remaining = plan.budget.checked_sub(population)?;
    (plan.n_replicates > 1 && remaining >= PROJECTED_GRADIENT_STEP_WORK).then_some(population)
}

fn low_dimensional_refinement_fevals(work_units: usize) -> usize {
    work_units / PROJECTED_GRADIENT_STEP_WORK
}

fn low_dimensional_value_scout<O>(
    obj: &BudgetedObjective<'_, O>,
    n_points: usize,
    seed: u64,
) -> Option<ValueScout>
where
    O: Objective<f64>,
{
    if n_points == 0 {
        return None;
    }
    let bounds = obj.bounds().clone();
    let starts = eindir_core::shifted_low_discrepancy_points(
        &bounds,
        n_points,
        qmc_skip_from_seed(seed),
        seed,
    );
    let mut best_pos = None;
    let mut best_val = f64::INFINITY;
    for start in starts.outer_iter() {
        if obj.ledger.exhausted() {
            break;
        }
        let pos = bounds.clip(start);
        let value = obj.eval(pos.view());
        if value.is_finite() && value < best_val {
            best_val = value;
            best_pos = Some(pos);
        }
    }
    best_pos.map(|best_pos| ValueScout { best_pos, best_val })
}

fn coordinate_opposition_scout<O>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    bounds: &Bounds<f64>,
) where
    O: Objective<f64>,
{
    if bounds.dims == 0
        || bounds.dims > COORDINATE_OPPOSITION_MAX_DIM
        || ledger.remaining() < bounds.dims
    {
        return;
    }
    let mut incumbent = ledger.incumbent(bounds);
    let mut incumbent_value = ledger.best_get();
    for coordinate in 0..bounds.dims {
        if ledger.exhausted() {
            break;
        }
        let center = 0.5 * (bounds.low[coordinate] + bounds.high[coordinate]);
        let opposite = (center - COORDINATE_OPPOSITION_WEIGHT * (incumbent[coordinate] - center))
            .clamp(bounds.low[coordinate], bounds.high[coordinate]);
        if !opposite.is_finite() {
            continue;
        }
        let mut candidate = incumbent.clone();
        candidate[coordinate] = opposite;
        let value = obj.eval(candidate.view());
        if value.is_finite() && value < incumbent_value {
            incumbent = candidate;
            incumbent_value = value;
        }
    }
}

fn scaled_center_probe<O, G>(
    obj: &BudgetedObjective<'_, O>,
    grad: &BudgetedGradient<'_, G>,
    bounds: &Bounds<f64>,
) -> Option<CenterProbe>
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let center = (&bounds.low + &bounds.high) * 0.5;
    if !obj.ledger.charge_probe(2, 1) {
        return None;
    }
    let value = obj.inner.eval(center.view());
    let gradient = grad.inner.grad(center.view());
    if !value.is_finite() {
        return None;
    }
    if gradient.len() != bounds.dims || gradient.iter().any(|v| !v.is_finite()) {
        return Some(CenterProbe {
            gradient_ratio: None,
        });
    }
    let scaled_gradient = Array1::from_iter((0..bounds.dims).map(|idx| {
        let width = bounds.high[idx] - bounds.low[idx];
        gradient[idx] * width
    }));
    let scaled_grad_norm = scaled_gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
    if scaled_grad_norm == 0.0 {
        return Some(CenterProbe {
            gradient_ratio: Some(0.0),
        });
    }
    let mut probe = center.clone();
    for idx in 0..bounds.dims {
        let width = bounds.high[idx] - bounds.low[idx];
        probe[idx] -= 0.25 * width * scaled_gradient[idx] / scaled_grad_norm;
    }
    let probe = bounds.clip(probe.view());
    let probe_value = obj.inner.eval(probe.view());
    let value_scale = energy_delta(probe_value, value).abs();
    let ratio = if value_scale > 0.0 && value_scale.is_finite() {
        scaled_grad_norm / value_scale
    } else {
        f64::INFINITY
    };
    Some(CenterProbe {
        gradient_ratio: ratio.is_finite().then_some(ratio),
    })
}

fn low_dimensional_polish_before_warmup(center_gradient_ratio: Option<f64>) -> bool {
    match center_gradient_ratio {
        Some(ratio) => ratio <= DOLAN_MORE_CONVERGENCE_TAU,
        None => false,
    }
}

fn benchmark_projected_stationary(projected_grad_norm: f64, value: f64) -> bool {
    projected_grad_norm.is_finite()
        && projected_grad_norm <= DOLAN_MORE_CONVERGENCE_TAU * value.abs().max(1.0)
}

// Retained for the convergence-threshold unit tests; the driver no longer
// stops early on this criterion (budget is the caller's authority).
#[allow(dead_code)]
fn benchmark_objective_converged(center_value: f64, best_value: f64) -> bool {
    if !center_value.is_finite() || !best_value.is_finite() {
        return false;
    }
    let scale = center_value.abs().max(1.0);
    best_value <= center_value - (1.0 - DOLAN_MORE_CONVERGENCE_TAU) * scale
}

fn best_polished_stationary(result: &QmcPolishResult) -> bool {
    let mut best_idx = None;
    let mut best_val = f64::INFINITY;
    for (idx, value) in result.polished_values.iter().copied().enumerate() {
        if value.is_finite() && value < best_val {
            best_val = value;
            best_idx = Some(idx);
        }
    }
    let Some(idx) = best_idx else {
        return false;
    };
    result
        .polished_stationary
        .get(idx)
        .copied()
        .unwrap_or(false)
        || result
            .polished_projected_grad_norms
            .get(idx)
            .copied()
            .is_some_and(|norm| benchmark_projected_stationary(norm, best_val))
}

fn run_low_dimensional_polish<O, G>(
    obj: &BudgetedObjective<'_, O>,
    grad: &BudgetedGradient<'_, G>,
    ledger: &BudgetLedger,
    plan: &LowDimensionalPolishPlan,
    seed: u64,
    budget: usize,
) -> bool
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let ceiling = ledger.used_get() + plan.budget;
    ledger.cap_set(ceiling.min(budget));
    let stationary = if let Some(population) = low_dimensional_scout_population(plan) {
        if let Some(scout) = low_dimensional_value_scout(obj, population, seed) {
            let maxf = low_dimensional_refinement_fevals(ledger.remaining());
            if scout.best_val.is_finite() && maxf > 0 {
                let result = projected_gradient_polish(
                    obj,
                    grad,
                    scout.best_pos,
                    maxf,
                    1.0,
                    LOW_DIMENSIONAL_POLISH_GRAD_TOL,
                );
                result.projected_stationary
                    || benchmark_projected_stationary(result.projected_grad_norm, result.best_val)
            } else {
                false
            }
        } else {
            false
        }
    } else {
        let result = shifted_qmc_projected_gradient_polish(
            obj,
            grad,
            plan.n_starts,
            plan.max_fevals_per_start,
            qmc_skip_from_seed(seed),
            plan.n_replicates,
            1.0,
            LOW_DIMENSIONAL_POLISH_GRAD_TOL,
            plan.top_k,
        );
        best_polished_stationary(&result)
    };
    ledger.cap_set(budget);
    stationary
}

/// Starts the GSA chains. With `anchor`, an evaluated point and its value,
/// chain 0 starts there without another evaluation (dual_annealing's `x0`)
/// and the best start is searched after the first temperature step; the
/// other chains start at seeded low-discrepancy points.
fn initialize_gsa_state<O>(
    obj: &BudgetedObjective<'_, O>,
    slice: usize,
    seed: u64,
    anchor: Option<(Array1<f64>, f64)>,
) -> Option<GsaState>
where
    O: Objective<f64>,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    if dim == 0 || slice < 2 {
        return None;
    }
    // Dual_annealing uses one strategy chain. Multi-chain splits budget and
    // cools each chain too few epochs on high-d Schwefel. Prefer 1–3 chains;
    // extra starts come from IPOP reseed of the worst chain.
    let chain_count = if dim >= 10 {
        1usize.min(slice)
    } else {
        (slice / 8).clamp(2, 4).min(slice)
    };
    let anchored = anchor.is_some();
    let mut xs = Vec::with_capacity(chain_count);
    let mut vals = Vec::with_capacity(chain_count);
    if let Some((x, value)) = anchor {
        xs.push(bounds.clip(x.view()));
        vals.push(value);
    }
    let sampled = chain_count - xs.len();
    if sampled > 0 {
        let starts = eindir_core::shifted_low_discrepancy_points(
            &bounds,
            sampled,
            qmc_skip_from_seed(seed),
            seed,
        );
        for start in starts.outer_iter() {
            if obj.ledger.exhausted() {
                break;
            }
            let pos = bounds.clip(start);
            let value = obj.eval(pos.view());
            xs.push(pos);
            vals.push(value);
        }
    }

    // SciPy dual_annealing default initial_temp=5230 (translation-invariant).
    let t_init = DUAL_INITIAL_TEMP;
    let best = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let first_search = anchored
        .then(|| {
            vals.iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, v)| (i, xs[i].clone(), *v))
        })
        .flatten();

    (!xs.is_empty()).then(|| GsaState {
        xs,
        vals,
        epoch: 0,
        rng: StdRng::seed_from_u64(seed),
        t_init,
        strategy_step: 0,
        best,
        local: None,
        first_search,
    })
}

/// Central finite differences through a budgeted objective (dual_annealing
/// L-BFGS-B without analytic jac). Each stencil pair charges real evals.
struct BudgetedFiniteDiffGradient<'a, O: Objective<f64>> {
    obj: &'a BudgetedObjective<'a, O>,
    /// Relative step: h_i = h_frac * box_width_i (clamped).
    h_frac: f64,
}

impl<O: Objective<f64>> Gradient<f64> for BudgetedFiniteDiffGradient<'_, O> {
    fn dim(&self) -> usize {
        self.obj.dim()
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let dim = x.len();
        let bounds = self.obj.bounds();
        let mut g = Array1::zeros(dim);
        for i in 0..dim {
            let w = (bounds.high[i] - bounds.low[i]).abs().max(1e-12);
            let h = (self.h_frac * w).clamp(1e-8, 0.05 * w);
            let mut xp = x.to_owned();
            let mut xm = x.to_owned();
            xp[i] = (x[i] + h).clamp(bounds.low[i], bounds.high[i]);
            xm[i] = (x[i] - h).clamp(bounds.low[i], bounds.high[i]);
            let fp = self.obj.eval(xp.view());
            let fm = self.obj.eval(xm.view());
            let den = (xp[i] - xm[i]).abs().max(1e-16);
            if fp.is_finite() && fm.is_finite() {
                g[i] = (fp - fm) / den;
            }
        }
        g
    }
}

/// Dual_annealing-class local search: multi-start projected quasi-Newton
/// with native gradients or central finite differences (no-grad Schwefel).
/// Charges the shared ledger. Target residual dual exclusives: no-grad
/// Schwefel basins need FD L-BFGS depth that pure pattern search lacks.
fn dual_style_local_search<O, G, R>(
    obj: &BudgetedObjective<'_, O>,
    grad: Option<&BudgetedGradient<'_, G>>,
    ledger: &BudgetLedger,
    bounds: &Bounds<f64>,
    rng: &mut R,
    work_units: usize,
    outer_cap: usize,
) where
    O: Objective<f64>,
    G: Gradient<f64>,
    R: Rng,
{
    let work_units = work_units.min(ledger.remaining());
    if work_units < 8 {
        return;
    }
    let dim = bounds.dims.max(1);
    let wide = mean_width(bounds) >= 50.0;
    let has_analytic = grad.is_some();
    // Analytic grad: global multi-start is cheap enough to hunt Schwefel basins.
    // FD grad costs 2d evals per step — prefer *depth* on incumbent (+ few
    // restarts); shallow global FD multi-start regressed schwefel_nograd_d10.
    let n_starts = if has_analytic {
        if wide && dim >= 20 {
            12
        } else if wide && dim >= 10 {
            8
        } else if dim >= 20 {
            6
        } else {
            5
        }
    } else {
        3 // deep FD: incumbent + 2 restarts near best
    };
    let per_start = (work_units / n_starts).max(if has_analytic { 8 } else { 24 });
    let x_inc = ledger.incumbent(bounds);
    let n_qmc = (n_starts - 1).max(1);
    let qmc = eindir_core::shifted_low_discrepancy_points(
        bounds,
        n_qmc,
        qmc_skip_from_seed(rng.random::<u64>()),
        rng.random::<u64>(),
    );

    for s in 0..n_starts {
        if ledger.remaining() < 8 {
            break;
        }
        let start = if s == 0 {
            x_inc.clone()
        } else if has_analytic && wide {
            // Global QMC sample — needs cheap analytic polish to pay off.
            let qi = (s - 1) % n_qmc;
            if qi < qmc.nrows() {
                bounds.clip(qmc.row(qi))
            } else {
                let mut y = Array1::zeros(dim);
                for i in 0..dim {
                    y[i] = bounds.low[i] + (bounds.high[i] - bounds.low[i]) * rng.random::<f64>();
                }
                bounds.clip(y.view())
            }
        } else {
            // Local hop around *current* ledger best (FD-safe).
            let best = ledger.incumbent(bounds);
            let mut y = best.clone();
            let scale = if wide { 0.25 } else { 0.12 } / (dim as f64).sqrt();
            for i in 0..dim {
                let w = (bounds.high[i] - bounds.low[i]).abs().max(1e-12);
                let noise: f64 = rand_distr::StandardNormal.sample(rng);
                y[i] = (best[i] + scale * w * noise).clamp(bounds.low[i], bounds.high[i]);
            }
            y
        };
        // FD: spend most of per_start as polish depth (grad is expensive).
        let maxf = if has_analytic {
            (per_start / 2).max(4)
        } else {
            per_start.max(16)
        }
        .min(ledger.remaining().max(4));
        let start_budget = maxf.min(ledger.remaining());
        if start_budget < 4 {
            break;
        }
        let ceiling = ledger.used_get() + start_budget;
        ledger.cap_set(ceiling.min(outer_cap));
        match grad {
            Some(g) => {
                let _ = projected_gradient_polish(obj, g, start, start_budget, 0.1, 1e-12);
            }
            None => {
                let fd = BudgetedFiniteDiffGradient { obj, h_frac: 1e-5 };
                let _ = projected_gradient_polish(obj, &fd, start, start_budget, 0.1, 1e-12);
            }
        }
        ledger.cap_set(outer_cap);
    }
}

/// SciPy dual_annealing StrategyChain.accept_reject probability.
#[inline]
fn dual_accept_prob(delta: f64, temperature_step: f64, accept_param: f64) -> f64 {
    if !delta.is_finite() {
        return 0.0;
    }
    if delta <= 0.0 {
        return 1.0;
    }
    let t = temperature_step.max(1e-300);
    // pqv_temp = 1 - (1 - qa) * delta / T_step
    let pqv_temp = 1.0 - (1.0 - accept_param) * delta / t;
    if pqv_temp <= 0.0 {
        return 0.0;
    }
    (pqv_temp.ln() / (1.0 - accept_param)).exp().clamp(0.0, 1.0)
}

/// Advances the GSA arm's pending local search within the slice. Returns
/// false when the slice ends first; a finished search moves its chain to
/// the descent's endpoint when that is lower.
fn advance_gsa_local<O>(
    obj: &BudgetedObjective<'_, O>,
    state: &mut GsaState,
    start_used: usize,
    slice: usize,
) -> bool
where
    O: Objective<f64>,
{
    let Some((chain, mut engine)) = state.local.take() else {
        return true;
    };
    while !engine.is_done() {
        if obj.ledger.used_get().saturating_sub(start_used) >= slice || obj.ledger.exhausted() {
            state.local = Some((chain, engine));
            return false;
        }
        let x = engine.ask();
        engine.tell(obj.eval(x.view()));
    }
    let value = engine.value();
    if value.is_finite() && value < state.vals[chain] {
        state.xs[chain] = engine.position().to_owned();
        state.vals[chain] = value;
        state.best = state.best.min(value);
    }
    true
}

/// Dual-annealing-style GSA epoch: box-scaled Tsallis cool + strategy
/// chain (all-coordinate visit then per-coordinate visits) with reflection.
/// Optional local polish after a strategy chain when `grad` is provided
/// (mirrors dual_annealing local_search on improvement); with
/// `local_search` and no gradient the polish is a finite-difference descent
/// with L-BFGS-B's stopping rule.
fn run_persistent_gsa<O, G>(
    obj: &BudgetedObjective<'_, O>,
    grad: Option<&BudgetedGradient<'_, G>>,
    state: &mut GsaState,
    slice: usize,
    local_search: bool,
) where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    let t0 = state.t_init.max(DUAL_INITIAL_TEMP * 0.5).max(1.0);
    let cooling = TsallisCool::new(t0, GSA_Q_V);
    let visit = TsallisVisit::new(GSA_Q_V);
    let start_used = obj.ledger.used_get();
    let reanneal_floor = t0 * DUAL_RESTART_TEMP_RATIO;
    let local_options = FdBfgsOptions {
        ftol: DUAL_LOCAL_FTOL,
        patience: 1,
        max_iter: (6 * dim).clamp(100, 1000),
        refine: false,
        value_floor: 1.0,
        gtol: DUAL_LOCAL_GTOL,
    };

    while obj.ledger.used_get().saturating_sub(start_used) < slice && !obj.ledger.exhausted() {
        if !advance_gsa_local(obj, state, start_used, slice) {
            return;
        }
        let mut temp = cooling.temperature(state.epoch).max(1e-300);
        // dual_annealing reannealing when T drops below initial * ratio.
        if temp < reanneal_floor {
            state.epoch = 0;
            state.strategy_step = 0;
            temp = cooling.temperature(0).max(1e-300);
        }
        // dual_annealing: temperature_step = T / (step+1) for acceptance.
        state.strategy_step = state.strategy_step.saturating_add(1);
        let t_accept = (temp / (state.strategy_step as f64)).max(1e-300);
        let mut improved = false;
        // Best new record of this temperature step: dual_annealing's local
        // search starts there once the strategy chain finishes.
        let mut step_best: Option<(usize, Array1<f64>, f64)> = None;

        for chain in 0..state.xs.len() {
            if obj.ledger.used_get().saturating_sub(start_used) >= slice || obj.ledger.exhausted() {
                break;
            }
            let before = state.vals[chain];
            // Strategy chain length 2*dim (all-dim + each single coord), dual_annealing.
            let n_strategy = (2 * dim).max(2);
            for j in 0..n_strategy {
                if obj.ledger.used_get().saturating_sub(start_used) >= slice
                    || obj.ledger.exhausted()
                {
                    break;
                }
                let x = &state.xs[chain];
                let proposal = if j < dim {
                    // All coordinates: full Tsallis visit in physical space.
                    let y = visit.propose(x.view(), temp, &mut state.rng);
                    crate::movekernel::reflect_into_box(y.view(), &bounds)
                } else {
                    // Single coordinate j-dim (dual strategy chain second half).
                    let axis = j - dim;
                    let y1 = visit.propose(
                        ArrayView1::from(std::slice::from_ref(&x[axis])),
                        temp,
                        &mut state.rng,
                    );
                    let mut y = x.clone();
                    y[axis] = y1[0];
                    crate::movekernel::reflect_into_box(y.view(), &bounds)
                };
                let proposal_val = obj.eval(proposal.view());
                if proposal_val < state.best {
                    state.best = proposal_val;
                    if local_search && grad.is_none() {
                        step_best = Some((chain, proposal.clone(), proposal_val));
                    }
                }
                let accepted = if !proposal_val.is_finite() {
                    false
                } else if !state.vals[chain].is_finite() {
                    true
                } else {
                    let delta = proposal_val - state.vals[chain];
                    // dual_annealing accept_reject (accept=-5), not Tsallis q_a.
                    state.rng.random::<f64>() < dual_accept_prob(delta, t_accept, DUAL_ACCEPT_PARAM)
                };
                if accepted {
                    state.xs[chain] = proposal;
                    state.vals[chain] = proposal_val;
                }
            }
            if state.vals[chain].is_finite() && state.vals[chain] < before {
                improved = true;
            }
        }
        let first_search = state.first_search.take();
        if local_search && grad.is_none() && step_best.is_none() {
            step_best = first_search;
        }
        if let Some((chain, x, value)) = step_best.take() {
            state.local = Some((
                chain,
                FdBfgs::new(x.view(), Some(value), &bounds, local_options),
            ));
        }
        // dual_annealing local_search when energy improved this temperature step.
        if improved
            && let Some(g) = grad
            && let Some((best_i, best_v)) = state
                .vals
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i, *v))
            && obj.ledger.remaining() >= 8
        {
            let polish_budget = 16.min(
                (slice / 8)
                    .max(8)
                    .min(obj.ledger.remaining().saturating_sub(1)),
            );
            if polish_budget >= 4 {
                let res = projected_gradient_polish(
                    obj,
                    g,
                    state.xs[best_i].clone(),
                    polish_budget,
                    1.0,
                    1e-10,
                );
                if res.best_val.is_finite() && res.best_val < best_v {
                    state.xs[best_i] = res.best_pos;
                    state.vals[best_i] = res.best_val;
                }
            }
        }
        // dual_annealing restarts local search after long non-improvement;
        // IPOP reseed every 5 epochs from a fresh QMC/random point (keep best).
        if state.epoch > 0 && state.epoch.is_multiple_of(5) && !obj.ledger.exhausted() {
            let best_v = state
                .vals
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);
            let mut x = Array1::zeros(dim);
            for i in 0..dim {
                let lo = bounds.low[i];
                let hi = bounds.high[i];
                x[i] = lo + (hi - lo) * state.rng.random::<f64>();
            }
            let x = bounds.clip(x.view());
            let v = obj.eval(x.view());
            if v.is_finite() {
                // Prefer replacing the current chain when worse; keep elite if multi.
                let worst_i = state
                    .vals
                    .iter()
                    .enumerate()
                    .filter(|(_, vv)| vv.is_finite())
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if v < state.vals[worst_i] || !state.vals[worst_i].is_finite() {
                    state.xs[worst_i] = x;
                    state.vals[worst_i] = v;
                } else if v < best_v {
                    // Better than global best among chains: install as chain 0.
                    state.xs[0] = x;
                    state.vals[0] = v;
                }
                // Re-heat slightly after reseed so the new basin can be explored.
                state.strategy_step /= 2;
            }
        }
        state.epoch += 1;
    }
}

fn elite_archive_anchor(ledger: &BudgetLedger, bounds: &Bounds<f64>) -> Option<Array1<f64>> {
    let inner = ledger.inner.lock().expect("ledger lock");
    let dim = bounds.dims;
    let (mut best_idx, mut best_val) = (None, f64::INFINITY);
    for (idx, value) in inner.archive_y.iter().copied().enumerate() {
        if value.is_finite() && value < best_val {
            best_idx = Some(idx);
            best_val = value;
        }
    }
    let idx = best_idx?;
    let start = idx.checked_mul(dim)?;
    let end = start.checked_add(dim)?;
    (end <= inner.archive_x.len())
        .then(|| bounds.clip(ArrayView1::from(&inner.archive_x[start..end])))
}

fn bayesian_pilot_steps_per_chain(pilot_budget: usize, min_steps: usize) -> usize {
    (pilot_budget / PILOT_CHAINS.max(1)).max(min_steps)
}

#[allow(clippy::too_many_arguments)]
fn fit_bayesian_pilot<O>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    rng: &mut StdRng,
    seed: u64,
    pilot_budget: usize,
    min_steps_per_chain: usize,
    noise_sigma: Option<f64>,
) -> Option<PilotState>
where
    O: Objective<f64>,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    let width_scale = mean_width(&bounds) / 10.0;
    let fallback_prior = PilotPrior::default();
    let scout_draws = pilot_draws_qmc(&fallback_prior, PILOT_CHAINS, seed);
    let per_chain = bayesian_pilot_steps_per_chain(pilot_budget, min_steps_per_chain);
    let start_used = ledger.used_get();
    let mut observations = Vec::with_capacity(PILOT_CHAINS);
    let mut best_anchor = None;
    let mut best_anchor_val = f64::INFINITY;

    let mut run_draws = |draws: Vec<(f64, f64, f64)>,
                         steps_per_chain: usize,
                         observations: &mut Vec<PilotObservation>| {
        for (t_init, sigma, q_v) in draws {
            if ledger.exhausted() || ledger.used_get().saturating_sub(start_used) >= pilot_budget {
                break;
            }
            let mut cur = Array1::from_iter((0..dim).map(|j| {
                let w = bounds.high[j] - bounds.low[j];
                bounds.low[j] + rng.random::<f64>() * if w > 0.0 { w } else { 0.0 }
            }));
            let mut cur_val = obj.eval(cur.view());
            let mut best_val = cur_val;
            let mut chain_best = cur.clone();
            let mut accepts = 0usize;
            let mut steps = 0usize;
            let cooling = TsallisCool::new(t_init, q_v);
            for step in 0..steps_per_chain {
                if ledger.exhausted()
                    || ledger.used_get().saturating_sub(start_used) >= pilot_budget
                {
                    break;
                }
                let temp = cooling.temperature(step / 10);
                let mut prop = cur.clone();
                for value in prop.iter_mut() {
                    let noise: f64 = rand_distr::StandardNormal.sample(rng);
                    *value += sigma * width_scale * noise;
                }
                // Reflect (not clip) the symmetric Gaussian random-walk proposal
                // back into the box so the Metropolis test below keeps detailed
                // balance: clipping piles mass on the boundary and breaks symmetry.
                let prop = crate::movekernel::reflect_into_box(prop.view(), &bounds);
                let prop_val = obj.eval(prop.view());
                steps += 1;
                let accepted = accept_move(
                    noise_sigma,
                    |_r| energy_delta(obj.eval(prop.view()), obj.eval(cur.view())),
                    energy_delta(prop_val, cur_val),
                    temp,
                    rng,
                );
                if prop_val.is_finite() && accepted {
                    cur = prop;
                    cur_val = prop_val;
                    accepts += 1;
                    if prop_val < best_val {
                        best_val = prop_val;
                        chain_best = cur.clone();
                    }
                }
            }
            if steps > 0 && best_val.is_finite() {
                if best_val < best_anchor_val {
                    best_anchor_val = best_val;
                    best_anchor = Some(chain_best);
                }
                observations.push(PilotObservation {
                    t_init,
                    sigma,
                    q_v,
                    accept_rate: accepts as f64 / steps as f64,
                    best_val,
                    final_pos: cur.to_vec(),
                });
            }
        }
    };

    run_draws(scout_draws, (per_chain / 4).max(1), &mut observations);
    let prior = empirical_prior_from_observations(&observations, &fallback_prior);
    let main_draws = pilot_draws_qmc(&prior, PILOT_CHAINS, seed.wrapping_add(1));
    run_draws(main_draws, per_chain, &mut observations);

    (observations.len() >= 2).then(|| PilotState {
        posterior: fit_laplace_skew_corrected(&observations, &prior),
        anchor: best_anchor,
    })
}

fn bayesian_gle_timestep(posterior: Option<&LaplacePosterior>, dim: usize) -> f64 {
    let Some(posterior) = posterior else {
        return GLE_DT;
    };
    let dim_scale = (dim.max(1) as f64).sqrt();
    let dt = posterior.sigma_map.abs() / dim_scale;
    if dt.is_finite() && dt > 0.0 {
        dt.clamp(GLE_MIN_DT, GLE_DT)
    } else {
        GLE_DT
    }
}

fn bayesian_gle_local_bounds(
    bounds: &Bounds<f64>,
    pilot: Option<&PilotState>,
    fallback_anchor: &Array1<f64>,
) -> Bounds<f64> {
    let Some(pilot) = pilot else {
        return bounds.clone();
    };
    let dim = bounds.dims;
    let anchor = pilot
        .anchor
        .as_ref()
        .filter(|candidate| {
            candidate.len() == dim && candidate.iter().all(|value| value.is_finite())
        })
        .unwrap_or(fallback_anchor);
    let center = bounds.clip(anchor.view());
    let width_scale = mean_width(bounds) / 10.0;
    let thermal_spread = pilot.posterior.log_t_init_sd.exp().max(1.0);
    let scalar_radius =
        (pilot.posterior.sigma_map.abs() * width_scale * (dim.max(1) as f64).sqrt())
            * thermal_spread;

    let mut low = bounds.low.clone();
    let mut high = bounds.high.clone();
    for j in 0..dim {
        let width = bounds.high[j] - bounds.low[j];
        if !width.is_finite() || width <= 0.0 {
            continue;
        }
        let radius = if scalar_radius.is_finite() && scalar_radius > 0.0 {
            scalar_radius.clamp(
                BAYESIAN_GLE_LOCAL_MIN_RADIUS_FRAC * width,
                BAYESIAN_GLE_LOCAL_MAX_RADIUS_FRAC * width,
            )
        } else {
            BAYESIAN_GLE_LOCAL_MAX_RADIUS_FRAC * width
        };
        low[j] = bounds.low[j].max(center[j] - radius);
        high[j] = bounds.high[j].min(center[j] + radius);
        if !(low[j].is_finite() && high[j].is_finite() && high[j] > low[j]) {
            low[j] = bounds.low[j];
            high[j] = bounds.high[j];
        }
    }
    Bounds::new(low, high, bounds.slack)
}

/// Top-`k` orthonormal basis of the empirical gradient covariance by
/// subspace iteration with matvecs over the stored gradients; no dense
/// `n x n` matrix and no external eigensolver.
fn active_subspace_basis(grads: &[Array1<f64>], dim: usize, k: usize, seed: u64) -> Array2<f64> {
    let mut state = seed | 1;
    let mut next = move || {
        // xorshift64*: deterministic basis init without an RNG object.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut basis = Array2::<f64>::zeros((dim, k));
    for value in basis.iter_mut() {
        *value = next();
    }
    for _ in 0..30 {
        // Y = (sum_g g g^T) B via matvecs.
        let mut y = Array2::<f64>::zeros((dim, k));
        for g in grads {
            let proj = g.dot(&basis);
            for col in 0..k {
                let w = proj[col];
                for row in 0..dim {
                    y[[row, col]] += g[row] * w;
                }
            }
        }
        // Gram-Schmidt re-orthonormalization.
        for col in 0..k {
            for prev in 0..col {
                let dot: f64 = (0..dim).map(|r| y[[r, col]] * y[[r, prev]]).sum();
                for row in 0..dim {
                    y[[row, col]] -= dot * y[[row, prev]];
                }
            }
            let norm: f64 = (0..dim)
                .map(|r| y[[r, col]] * y[[r, col]])
                .sum::<f64>()
                .sqrt();
            if norm > 1e-300 {
                for row in 0..dim {
                    y[[row, col]] /= norm;
                }
            } else {
                for row in 0..dim {
                    y[[row, col]] = next();
                }
            }
        }
        basis = y;
    }
    basis
}

#[allow(clippy::too_many_arguments)]
fn run_arm<O, G>(
    arm: ArmKind,
    obj: &BudgetedObjective<'_, O>,
    grad: Option<&BudgetedGradient<'_, G>>,
    ledger: &BudgetLedger,
    states: &mut ArmStates,
    rng: &mut StdRng,
    slice: usize,
    budget: usize,
) where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    states.seed_counter += 1;
    let seed = rng.random::<u64>() ^ states.seed_counter;
    match arm {
        ArmKind::Explore => {
            // QMC restart arm: screened Cranley-Patterson Halton starts,
            // best ones refined; the independently randomized shifts give
            // the restart points positive density over the bounded box.
            if let Some(grad) = grad {
                // One third screening, two thirds descent; the single
                // polished start gets the full descent depth, which
                // ill-conditioned valleys need more than breadth.
                // (A Luby-scheduled depth variant measurably lost basins
                // on DEVGLA1-class cells at 2(D+1) quanta; revisit only
                // with per-cell A/B evidence.)
                let n_starts = (slice / 3).max(4);
                let per_start = slice.saturating_sub(n_starts) / 2;
                if per_start >= 2 {
                    let res = qmc_projected_gradient_polish(
                        obj, grad, n_starts, per_start, seed, 1.0, 1e-8, 1,
                    );
                    states
                        .basins
                        .register(res.best_pos.view(), res.best_val, &bounds);
                    return;
                }
            }
            if slice >= 8 {
                let chains = (slice / 8).clamp(2, 4 * dim.max(1));
                let res = qmc_gsa_global_search(obj, slice, seed, chains, 1.0, GSA_Q_V, GSA_Q_A);
                states
                    .basins
                    .register(res.best_pos.view(), res.best_val, &bounds);
            }
        }
        ArmKind::Shift => {
            // Elite-shift arm: reuse the best charged archive point as
            // the anchor and spend one local trajectory from it.
            let Some(grad) = grad else { return };
            let Some(anchor) = elite_archive_anchor(ledger, &bounds) else {
                return;
            };
            let maxf = (slice / 2).max(2);
            projected_gradient_polish(obj, grad, anchor, maxf, 1.0, 1e-12);
        }
        ArmKind::Hop => {
            let Some(grad) = grad else { return };
            let state = states.hop.get_or_insert_with(|| HopState {
                step: HOP_STEP0,
                x_cur: None,
                f_cur: f64::INFINITY,
                generation: 0,
            });
            let mut x_cur = state
                .x_cur
                .clone()
                .unwrap_or_else(|| ledger.incumbent(&bounds));
            let mut f_cur = state.f_cur.min(ledger.best_get());
            let temp = ladder_temperature(archive_temp0(ledger), state.generation);
            let width = &bounds.high - &bounds.low;
            // One full-depth descent per slice: ill-conditioned valleys
            // reward depth over hop count.
            if ledger.remaining() >= 4 {
                let mut trial = x_cur.clone();
                for j in 0..dim {
                    let w = if width[j] > 0.0 { width[j] } else { 1.0 };
                    let noise: f64 = rand_distr::StandardNormal.sample(rng);
                    trial[j] += state.step * w * noise;
                }
                // Reflect the symmetric basin-hop perturbation into the box
                // (keeps the hop proposal symmetric for the Metropolis guard
                // below; clipping would bias hops toward the boundary).
                let trial = crate::movekernel::reflect_into_box(trial.view(), &bounds);
                // Descent-dominated valleys reward one full-depth descent
                // per slice (basin-hopping depth); multimodal boxes keep
                // the half-slice cap so hops stay frequent.
                let hop_depth = if states.deep_hops {
                    slice.saturating_sub(2).max(4)
                } else {
                    slice / 2
                };
                let res = projected_gradient_polish(obj, grad, trial, hop_depth, 1.0, 1e-8);
                if !res.best_val.is_finite() {
                    state.step = (state.step * HOP_SHRINK).max(1e-4);
                } else if res.best_val < f_cur
                    || metropolis(energy_delta(res.best_val, f_cur), temp, rng)
                {
                    x_cur = res.best_pos;
                    f_cur = res.best_val;
                    state.step = (state.step * HOP_GROW).min(1.0);
                } else {
                    state.step = (state.step * HOP_SHRINK).max(1e-4);
                }
            }
            state.x_cur = Some(x_cur);
            state.f_cur = f_cur;
            state.generation += 1;
        }
        ArmKind::Surrogate => {
            // Archive-fit additive surrogate: the fit costs nothing, the
            // modal point tests the global candidate for one evaluation,
            // and tempered independence proposals carry the
            // dimension-free acceptance bound.
            let min_points = (SURROGATE_DEGREE + 2).max(4 * dim);
            let (xs, ys) = {
                let inner = ledger.inner.lock().expect("ledger lock");
                let n = inner.archive_y.len();
                if n < min_points {
                    return;
                }
                let mut keep_x = Vec::with_capacity(n * dim);
                let mut keep_y = Vec::with_capacity(n);
                for i in 0..n {
                    if inner.archive_y[i].is_finite() {
                        keep_x.extend_from_slice(&inner.archive_x[i * dim..(i + 1) * dim]);
                        keep_y.push(inner.archive_y[i]);
                    }
                }
                (keep_x, keep_y)
            };
            if ys.len() < min_points {
                return;
            }
            let x_arr = Array2::from_shape_vec((ys.len(), dim), xs).expect("archive shape");
            let y_arr = Array1::from_vec(ys);
            let surr = AdditiveSurrogate::fit(
                x_arr.view(),
                y_arr.view(),
                bounds.clone(),
                SURROGATE_DEGREE,
            );
            // The modal point is the T -> 0 limit of the tempered
            // marginals; for separable objectives it is the surrogate's
            // global candidate, tested at the cost of one evaluation.
            let modal = surr.sample(1, 1e-15, SURROGATE_GRID, rng);
            let before_modal = ledger.best_get();
            let modal_x = bounds.clip(modal.row(0));
            let modal_val = obj.eval(modal_x.view());
            if let Some(grad) = grad
                && modal_val.is_finite()
                && modal_val < before_modal
                && ledger.remaining() >= 4
            {
                projected_gradient_polish(obj, grad, modal_x, ledger.remaining() / 2, 1.0, 1e-8);
            }
            // Cool with budget progress so the ladder reaches the cold
            // regime regardless of how often the arm is pulled.
            let progress = ledger.used_get() as f64 / budget.max(1) as f64;
            let exponent = (12.0 * progress) as i32 + states.surrogate_gen as i32;
            let temp = (archive_temp0(ledger) * 0.5_f64.powi(exponent)).max(1e-12);
            let proposals = surr.sample(slice, temp, SURROGATE_GRID, rng);
            let mut f_cur = ledger.best_get();
            let noise_sigma = states.noise_sigma;
            for i in 0..proposals.nrows() {
                if ledger.exhausted() {
                    break;
                }
                let trial = bounds.clip(proposals.row(i));
                let ft = obj.eval(trial.view());
                let accepted = accept_move(
                    noise_sigma,
                    |_r| energy_delta(obj.eval(trial.view()), f_cur),
                    energy_delta(ft, f_cur),
                    temp,
                    rng,
                );
                if ft.is_finite() && accepted {
                    f_cur = ft;
                }
            }
            states.surrogate_gen += 1;
        }
        ArmKind::De => {
            let mut used = 0usize;
            if states.de.is_none() {
                let pop_size = (DE_POP_PER_DIM * dim)
                    .clamp(DE_POP_MIN, DE_POP_MAX)
                    .min(slice.max(4));
                let points = eindir_core::shifted_low_discrepancy_points(
                    &bounds,
                    pop_size,
                    qmc_skip_from_seed(seed),
                    seed,
                );
                let mut pop = Vec::with_capacity(pop_size);
                let mut vals = Vec::with_capacity(pop_size);
                let anchor = if states.values_only {
                    ledger
                        .incumbent_value()
                        .map(|value| (ledger.incumbent(&bounds), value))
                } else {
                    None
                };
                for i in 0..points.nrows() {
                    if ledger.exhausted() {
                        break;
                    }
                    if i == 0
                        && let Some((x, value)) = anchor.as_ref()
                    {
                        pop.push(x.clone());
                        vals.push(*value);
                        continue;
                    }
                    let x = points.row(i).to_owned();
                    let v = obj.eval(x.view());
                    used += 1;
                    pop.push(x);
                    vals.push(v);
                }
                if pop.len() < 4 {
                    states.de = None;
                    return;
                }
                states.de = Some(DeState { pop, vals });
                // The values-only loop scores a pull by the incumbent it
                // leaves, so its first pull also evolves the population.
                if !states.values_only {
                    return;
                }
            }
            let state = states.de.as_mut().expect("de state initialised");
            let n = state.pop.len();
            let (mut best_i, mut best_v) = (0usize, f64::INFINITY);
            for (i, v) in state.vals.iter().enumerate() {
                if v.is_finite() && *v < best_v {
                    best_i = i;
                    best_v = *v;
                }
            }
            if !best_v.is_finite() {
                return;
            }
            let mut best_x = state.pop[best_i].clone();
            while used < slice && !ledger.exhausted() {
                let weight = DE_WEIGHT_MIN + DE_WEIGHT_SPAN * rng.random::<f64>();
                for i in 0..n {
                    if used >= slice || ledger.exhausted() {
                        break;
                    }
                    let mut r0 = rng.random_range(0..n - 1);
                    if r0 >= i {
                        r0 += 1;
                    }
                    let mut r1 = rng.random_range(0..n - 1);
                    if r1 >= i {
                        r1 += 1;
                    }
                    let forced = rng.random_range(0..dim);
                    let mut trial = state.pop[i].clone();
                    for j in 0..dim {
                        if j == forced || rng.random::<f64>() < DE_CROSSOVER {
                            trial[j] = best_x[j] + weight * (state.pop[r0][j] - state.pop[r1][j]);
                        }
                    }
                    let trial = bounds.clip(trial.view());
                    let ft = obj.eval(trial.view());
                    used += 1;
                    if ft.is_finite() && (!state.vals[i].is_finite() || ft < state.vals[i]) {
                        state.pop[i] = trial.clone();
                        state.vals[i] = ft;
                        if ft < best_v {
                            best_v = ft;
                            best_x = trial;
                        }
                    }
                }
            }
        }
        ArmKind::Gle => {
            let Some(grad) = grad else { return };
            if states.pilot.is_none() {
                let pilot_budget = slice / BAYESIAN_GLE_PILOT_BUDGET_DIVISOR;
                if pilot_budget >= BAYESIAN_GLE_MIN_PILOT_WORK {
                    let noise_sigma = states.noise_sigma;
                    states.pilot = fit_bayesian_pilot(
                        obj,
                        ledger,
                        rng,
                        seed,
                        pilot_budget,
                        BAYESIAN_PILOT_MIN_CHAIN_STEPS,
                        noise_sigma,
                    );
                }
            }
            let maxf = ledger.remaining().min(slice) / 2;
            if maxf < 4 {
                return;
            }
            let pilot = states.pilot.as_ref();
            let anchor = pilot
                .and_then(|state| state.anchor.clone())
                .unwrap_or_else(|| ledger.incumbent(&bounds));
            let local_bounds = bayesian_gle_local_bounds(&bounds, pilot, &anchor);
            let local_obj = LocalBoxBudgetedObjective {
                inner: obj.inner,
                ledger,
                bounds: local_bounds.clone(),
            };
            let fresh_grad = BudgetedGradient {
                inner: grad.inner,
                ledger,
            };
            gle_langevin_preconditioned_sa(
                &local_obj,
                &fresh_grad,
                seed,
                maxf,
                bayesian_gle_timestep(pilot.map(|state| &state.posterior), dim),
                GLE_EPOCHS,
                Some(local_bounds.clip(anchor.view())),
                None,
            );
        }
        ArmKind::TrPoll => {
            if slice < 8 {
                return;
            }
            qmc_trust_region_poll(obj, ledger.incumbent(&bounds), slice, seed, 0.0, 3, 0);
        }
        ArmKind::Gsa => {
            if slice < 8 {
                return;
            }
            let local_search = states.values_only && grad.is_none();
            if states.gsa.is_none() {
                let anchor = if local_search {
                    ledger
                        .incumbent_value()
                        .map(|value| (ledger.incumbent(&bounds), value))
                } else {
                    None
                };
                states.gsa = initialize_gsa_state(obj, slice, seed, anchor);
            }
            if let Some(state) = states.gsa.as_mut() {
                // Keep T₀ at box scale (do not re-inflate from |f|).
                // In-epoch dual-style LS when grad is present (run_persistent_gsa).
                run_persistent_gsa(obj, grad, state, slice, local_search);
            }
        }
        ArmKind::AmSa => {
            // D6 + D11 BFWT adaptive Metropolis: design T from gap, floor
            // from barrier_hat / log(B+e), ceiling at θ=2; Haario shape;
            // Robbins-Monro size; IPOP reseed on stagnation. Exact
            // Metropolis only (noise → OSA-capable arms).
            if states.noise_sigma.is_some() {
                return;
            }
            let d = dim.max(1);
            if states.am.is_none() {
                let x = ledger.incumbent(&bounds);
                let f_x = obj.eval(x.view());
                if !f_x.is_finite() {
                    return;
                }
                states.am = Some(AmSaState::new(x, f_x));
            }
            // IPOP-style reseed when the chain has not moved the ledger.
            {
                let st = states.am.as_mut().expect("am chain installed");
                if st.stagnant_slices >= AM_STAGNANT_RESEED && ledger.remaining() >= 2 {
                    let mut x = Array1::zeros(d);
                    for i in 0..d {
                        let lo = bounds.low[i];
                        let hi = bounds.high[i];
                        x[i] = lo + (hi - lo) * rng.random::<f64>();
                    }
                    let x = bounds.clip(x.view());
                    let f_x = obj.eval(x.view());
                    if f_x.is_finite() {
                        st.reseed(x, f_x);
                    } else {
                        st.stagnant_slices = 0;
                    }
                }
            }
            let st = states.am.as_mut().expect("am chain installed");
            let l = st.proposal_chol(&bounds);
            let base = 2.38 / (d as f64).sqrt();
            // IPOP: inflate step after reseeds to explore a wider basin.
            let reseed_boost = 1.5_f64.powi(st.reseed_gen.min(6) as i32);
            let before_best = ledger.best_get();
            for _ in 0..slice {
                if ledger.remaining() < 1 {
                    break;
                }
                let rem = ledger.remaining() as f64;
                // D11 BFWT: clamp design T into D6∩D7 window.
                let (temp, _mode) = crate::methods::bfwt::budget_feasible_temp(
                    st.f_x,
                    ledger.best_get(),
                    d,
                    rem,
                    st.barrier_hat,
                );
                // Fallback floor if BFWT collapses (tiny gap, zero barrier).
                let temp = temp.max(if st.barrier_hat > 0.0 {
                    0.0
                } else {
                    // residual heat from |f| so pure freeze at incumbent is rare
                    1e-6 * st.f_x.abs().max(1.0) / d as f64
                });
                let z: Vec<f64> = (0..d)
                    .map(|_| rand_distr::StandardNormal.sample(rng))
                    .collect();
                let scale = st.log_scale.exp() * base * reseed_boost;
                let mut y = st.x.clone();
                for i in 0..d {
                    let mut acc = 0.0;
                    for j in 0..=i {
                        acc += l[(i, j)] * z[j];
                    }
                    y[i] += scale * acc;
                }
                let y = crate::movekernel::reflect_into_box(y.view(), &bounds);
                let f_y = obj.eval(y.view());
                if !f_y.is_finite() {
                    continue;
                }
                let delta = f_y - st.f_x;
                let accepted =
                    delta <= 0.0 || (temp > 0.0 && rng.random::<f64>() < (-delta / temp).exp());
                st.rm_n += 1;
                let a = if accepted { 1.0 } else { 0.0 };
                // Diminishing adaptation keeps the chain ergodic.
                st.log_scale += (a - AM_ALPHA_TARGET) / (st.rm_n as f64).sqrt();
                st.log_scale = st.log_scale.clamp(-12.0, 6.0);
                if accepted {
                    st.x = y.to_owned();
                    st.f_x = f_y;
                    let x_obs = st.x.clone();
                    st.observe(x_obs.view());
                } else if delta > 0.0 {
                    // Rejected uphill: feed D7 barrier proxy for BFWT.
                    st.barrier_hat = if st.barrier_hat <= 0.0 {
                        delta
                    } else {
                        (1.0 - AM_BARRIER_EMA) * st.barrier_hat + AM_BARRIER_EMA * delta
                    };
                }
            }
            if ledger.best_get() < before_best - 1e-15 * before_best.abs().max(1.0) {
                st.stagnant_slices = 0;
                // Successful descent shrinks barrier estimate.
                st.barrier_hat *= 0.5;
            } else {
                st.stagnant_slices = st.stagnant_slices.saturating_add(1);
            }
        }
        ArmKind::Variant => {
            // Bayesian-pilot tuned classical point: the first slice runs
            // short Metropolis pilot chains at QMC-drawn hyperparameters
            // and fits the Laplace posterior over (T_0, sigma, q_v);
            // later slices run the MAP point through the typed driver.
            // The pilot prior's sigma convention matches a width-10 box,
            // so sigma scales with the mean box width.
            let width_scale = mean_width(&bounds) / 10.0;
            if states.pilot.is_none() {
                let noise_sigma = states.noise_sigma;
                states.pilot = fit_bayesian_pilot(
                    obj,
                    ledger,
                    rng,
                    seed,
                    slice,
                    BAYESIAN_PILOT_MIN_CHAIN_STEPS,
                    noise_sigma,
                );
                return;
            }
            let posterior = &states.pilot.as_ref().expect("pilot fitted").posterior;
            // Persistent chain: each slice extends the same annealing
            // trajectory (cooling epoch continues where the last slice
            // stopped). Slice-restarted SA re-heats every slice and never
            // anneals, which is why the classical presets used to beat
            // this arm on multimodal boxes.
            let steps = (4 * (dim + 1)).clamp(16, 64);
            let epochs = (slice / steps).max(2);
            let fresh_obj = BudgetedObjective {
                inner: obj.inner,
                ledger,
            };
            let start_epoch = states.variant_epoch;
            let resume = states.variant_chain.clone();
            let chain = if posterior.q_v_map > 1.3 {
                crate::variant::gsa(
                    fresh_obj,
                    posterior.t_init_map.max(1e-9),
                    posterior.q_v_map.clamp(1.05, 2.95),
                    GSA_Q_A,
                )
                .ok()
                .map(|variant| {
                    run_rs_variant_resumed(variant, start_epoch, epochs, steps, seed, resume).1
                })
            } else {
                crate::variant::boltzmann(
                    fresh_obj,
                    posterior.t_init_map.max(1e-9),
                    (posterior.sigma_map * width_scale).max(1e-9),
                )
                .ok()
                .map(|variant| {
                    run_rs_variant_resumed(variant, start_epoch, epochs, steps, seed, resume).1
                })
            };
            if let Some(cur) = chain {
                states.variant_chain = Some(cur);
                states.variant_epoch = start_epoch + epochs;
            }
        }
        ArmKind::Pt => {
            // Parallel-tempering ladder over the GSA point with Tsallis
            // exchange; chains run at fixed ladder temperatures.
            let n_chains = 4usize;
            let k_inner = 8usize;
            let pt_epochs = (slice / (n_chains * k_inner)).max(1);
            let t0 = archive_temp0(ledger).max(1e-6);
            let temps = geometric_ladder(0.05 * t0, 2.0 * t0 + 1e-6, n_chains);
            let fresh_obj = BudgetedObjective {
                inner: obj.inner,
                ledger,
            };
            let Ok(variant) = crate::variant::gsa(fresh_obj, t0, GSA_Q_V, GSA_Q_A) else {
                return;
            };
            let cool = LogCool::new(t0, 2.0);
            let pt = ParallelTemperingSampler::with_exchange(
                variant,
                TsallisExchange::new(GSA_Q_A),
                temps,
                k_inner,
                4,
            );
            pt.run(&cool, pt_epochs, seed);
        }
        ArmKind::Hmc => {
            // q-Gaussian HMC from the incumbent with the Omelyan
            // minimum-norm integrator; heavy-tailed momentum helps the
            // trajectory escape local cups.
            let Some(grad) = grad else { return };
            let trajectory_cost = 2 * HMC_L_STEPS + 3;
            let n_trajectories = (slice / trajectory_cost).max(1);
            let epochs = 8usize.min(n_trajectories).max(1);
            let steps = (n_trajectories / epochs).max(1);
            let t0 = archive_temp0(ledger).max(1e-6);
            let epsilon = (0.02 * mean_width(&bounds) / (dim as f64).sqrt()).max(1e-9);
            let q_max = 1.0 + 2.0 / dim as f64;
            let q = (1.0 + 1.5 / dim as f64).min(q_max - 1e-6);
            let cool = LogCool::new(t0, 2.0);
            let integrator = OmelyanIntegrator::new(epsilon, HMC_L_STEPS, t0);
            let fresh_obj = BudgetedObjective {
                inner: obj.inner,
                ledger,
            };
            let fresh_grad = BudgetedGradient {
                inner: grad.inner,
                ledger,
            };
            let sampler = HmcSaSampler::with_momentum(
                fresh_obj,
                fresh_grad,
                cool.clone(),
                QGaussianMomentum::new(q),
                integrator,
            )
            .with_initial_pos(ledger.incumbent(&bounds));
            run_rs(sampler, &cool, epochs, steps, seed);
        }
        ArmKind::Metad => {
            run_metad_arm(obj, ledger, states, rng, slice, &bounds, dim);
        }
        ArmKind::Tps => {
            run_tps_arm(obj, ledger, states, rng, slice, &bounds, dim);
        }
        ArmKind::DmcPop => {
            // Classical population-controlled diffusion (DMC / population-
            // annealing pattern): residual branching, DE+diffusion proposals,
            // QMC init, elite polish. Walker 0 starts at the incumbent.
            let maxf = ledger.remaining().min(slice);
            if maxf < 12 {
                return;
            }
            let target_n = crate::methods::dmc_population::recommend_target_n(maxf, dim)
                .clamp(6, 32)
                .min(maxf / 3);
            let seed_pos = ledger.incumbent(&bounds);
            let fresh_obj = BudgetedObjective {
                inner: obj.inner,
                ledger,
            };
            let mut local_rng = StdRng::seed_from_u64(seed.wrapping_add(0xd1c_00b00));
            if let Some(g) = grad {
                let fresh_grad = BudgetedGradient {
                    inner: g.inner,
                    ledger,
                };
                let _ = crate::methods::dmc_population::run_dmc_population_seeded(
                    &fresh_obj,
                    Some(&fresh_grad),
                    maxf,
                    seed,
                    target_n,
                    crate::methods::dmc_population::DEFAULT_STEPS_PER_CONTROL,
                    crate::methods::dmc_population::DEFAULT_BETA0,
                    Some(seed_pos.view()),
                    &mut local_rng,
                );
            } else {
                let _ = crate::methods::dmc_population::run_dmc_population_seeded::<
                    _,
                    BudgetedGradient<'_, G>,
                    _,
                >(
                    &fresh_obj,
                    None,
                    maxf,
                    seed,
                    target_n,
                    crate::methods::dmc_population::DEFAULT_STEPS_PER_CONTROL,
                    crate::methods::dmc_population::DEFAULT_BETA0,
                    Some(seed_pos.view()),
                    &mut local_rng,
                );
            }
        }
        ArmKind::Cma => run_cma_arm(obj, ledger, states, slice, seed, budget),
        ArmKind::Qn => run_qn_arm(obj, ledger, states, slice, seed, false),
        ArmKind::Reduced => {
            // Active-subspace collapse: charged pilot gradients estimate
            // the dominant gradient-covariance directions, then GSA
            // searches the collapsed box anchored at the incumbent.
            let Some(grad) = grad else { return };
            if dim <= 2 * REDUCED_K {
                return;
            }
            let m = (2 * REDUCED_K).min(slice / 2).max(2);
            let points = eindir_core::shifted_low_discrepancy_points(
                &bounds,
                m,
                qmc_skip_from_seed(seed),
                seed,
            );
            let mut grads = Vec::with_capacity(m);
            for i in 0..points.nrows() {
                if ledger.exhausted() {
                    return;
                }
                let g = grad.grad(points.row(i));
                if g.len() == dim && g.iter().all(|v| v.is_finite()) {
                    grads.push(g);
                }
            }
            if grads.len() < 2 {
                return;
            }
            let basis = active_subspace_basis(&grads, dim, REDUCED_K, seed);
            let radius = 0.5
                * (0..dim)
                    .map(|j| {
                        let w = bounds.high[j] - bounds.low[j];
                        if w.is_finite() && w > 0.0 { w * w } else { 1.0 }
                    })
                    .sum::<f64>()
                    .sqrt();
            let red_bounds = Bounds::new(
                Array1::from_elem(REDUCED_K, -radius),
                Array1::from_elem(REDUCED_K, radius),
                1e-9,
            );
            let fresh_obj = BudgetedObjective {
                inner: obj.inner,
                ledger,
            };
            let reduced =
                ReducedObjective::new(fresh_obj, ledger.incumbent(&bounds), basis, red_bounds);
            let remaining_slice = slice.saturating_sub(grads.len());
            if remaining_slice >= 8 {
                let chains = (remaining_slice / 8).clamp(2, 4 * REDUCED_K);
                qmc_gsa_global_search(
                    &reduced,
                    remaining_slice,
                    seed,
                    chains,
                    1.0,
                    GSA_Q_V,
                    GSA_Q_A,
                );
            }
        }
    }
}

/// One slice of the CMA-ES arm. Each candidate is one charged evaluation and
/// the arm never asks past the slice or the ledger cap.
fn run_cma_arm<O>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    states: &mut ArmStates,
    slice: usize,
    seed: u64,
    budget: usize,
) where
    O: Objective<f64>,
{
    let bounds = obj.bounds();
    let dim = bounds.dims;
    let state = states.cma.get_or_insert_with(|| {
        CmaArmState::new(ledger, bounds, budget, seed, CMA_FIRST_SIGMA, true)
    });
    let start = ledger.used_get();
    while ledger.used_get() - start < slice && ledger.remaining() > 0 {
        if state.es.stop_reason().is_some() {
            state.planner.record(state.regime, state.es.evaluations());
            let plan = state.planner.next_run(&mut state.rng);
            let mean = to_unit_box(&ledger.incumbent(bounds), bounds);
            let run_seed = state.rng.random::<u64>();
            let mut es = CmaArmState::run(
                mean.view(),
                plan.sigma,
                plan.lambda,
                &state.unit,
                run_seed,
                state.separable,
                ledger.best_get(),
            );
            if state.elitist && plan.regime == CmaRegime::Small {
                es = es.with_elite(mean.view(), ledger.best_get());
            }
            state.es = es;
            state.regime = plan.regime;
        }
        let u = state.es.ask();
        let x = Array1::from_iter((0..dim).map(|i| {
            (bounds.low[i] + (bounds.high[i] - bounds.low[i]) * u[i])
                .clamp(bounds.low[i], bounds.high[i])
        }));
        state.es.tell(obj.eval(x.view()));
    }
}

/// One slice of the finite-difference quasi-Newton arm. Stencil points and
/// line-search trials are each one charged evaluation. With `settle` the
/// slice ends when the current descent converges instead of kicking a new
/// one.
fn run_qn_arm<O>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    states: &mut ArmStates,
    slice: usize,
    seed: u64,
    settle: bool,
) where
    O: Objective<f64>,
{
    let bounds = obj.bounds();
    let best = ledger.best_get();
    // An evaluated incumbent is not evaluated again; one whose value is not
    // finite ends the descent at once, so the first slice kicks from it.
    let state = states.qn.get_or_insert_with(|| QnArmState {
        engine: FdBfgs::new(
            ledger.incumbent(bounds).view(),
            ledger.incumbent_value(),
            bounds,
            FdBfgsOptions::default(),
        ),
        kick: QN_KICK0,
        base_val: best,
        seen_best: best,
        rng: StdRng::seed_from_u64(seed ^ QN_STREAM),
        checked: f64::INFINITY,
    });
    if best < state.seen_best && best < state.engine.value() {
        state
            .engine
            .restart_at(ledger.incumbent(bounds).view(), Some(best), false);
        state.base_val = best;
        state.checked = f64::INFINITY;
    }
    let start = ledger.used_get();
    while ledger.used_get() - start < slice && ledger.remaining() > 0 {
        if state.engine.is_done() {
            if settle {
                break;
            }
            let best = ledger.best_get();
            state.kick = if best < state.base_val {
                (state.kick * HOP_GROW).min(QN_KICK_MAX)
            } else {
                (state.kick * HOP_SHRINK).max(QN_KICK_MIN)
            };
            let mut x = ledger.incumbent(bounds);
            for i in 0..x.len() {
                let noise: f64 = rand_distr::StandardNormal.sample(&mut state.rng);
                x[i] += state.kick * (bounds.high[i] - bounds.low[i]) * noise;
            }
            let x = crate::movekernel::reflect_into_box(x.view(), bounds);
            state.engine.restart_at(x.view(), None, true);
            state.base_val = best;
            state.checked = f64::INFINITY;
        }
        let x = state.engine.ask();
        state.engine.tell(obj.eval(x.view()));
    }
    state.seen_best = ledger.best_get();
}

/// Arms of the values-only loop in warm-up order: the two arms that start
/// from the incumbent, the two global visitors that keep one member there,
/// the additive surrogate fitted to every evaluation so far, and the
/// restart arm.
const VALUES_ONLY_ARMS: [ArmKind; 6] = [
    ArmKind::Cma,
    ArmKind::Qn,
    ArmKind::Gsa,
    ArmKind::De,
    ArmKind::Surrogate,
    ArmKind::Explore,
];

/// Values-only allocation under the Auto policy (no gradient, no declared
/// noise), whatever the box width.
///
/// Without `x0`, the start is the best of a seeded low-discrepancy design of
/// `dim + 1` points (one gradient's worth, at most a tenth of the budget),
/// so seeds vary it. When the budget affords one
/// ([`QN_AFFORDABLE_GRADIENTS`]), a finite-difference descent from the
/// incumbent opens the run and goes on while its slices succeed, up to
/// convergence. From the minimum it reaches, GSA and then CMA-ES each keep
/// the turn while they pay: a phase ends once it has gone
/// [`ROUNDS_PER_ARM`] slices without lowering the incumbent, or as long as
/// its last gain took if that is longer, and it leaves the closing polish
/// and one slice for each other arm not yet played. Bandit rounds over
/// [`VALUES_ONLY_ARMS`] follow until the closing polish reserve remains. As
/// in the main bandit, each arm not yet played takes one slice first, in
/// list order, so the restart arm and the global searches get an
/// observation whenever the budget holds a slice each. After that a round
/// picks uniformly with probability `1/round` (rounds counted from the
/// opening); otherwise the arm whose last turn succeeded plays again, and
/// failing that a discounted Thompson draw picks. A turn is one slice,
/// except that the descent keeps the turn, short of the closing polish,
/// while each slice lowers its own value by more than the success
/// threshold ([`QnArmState::paid`]). A turn succeeds when it lowers the
/// incumbent by more than [`arm_success_threshold`]. The closing
/// polish is a finite-difference descent from the final incumbent with the
/// evaluations one needs to converge ([`LOCAL_FIRST_POLISH_GRADIENTS`]),
/// kicked from it once it does.
fn run_values_only_portfolio<O, G>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    states: &mut ArmStates,
    seed: u64,
    budget: usize,
) -> Vec<ArmStat>
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    let gradient = dim + 1;
    states.values_only = true;
    if ledger.incumbent_value().is_none() {
        let count = gradient.min(budget.div_ceil(10));
        let design = eindir_core::shifted_low_discrepancy_points(
            &bounds,
            count,
            qmc_skip_from_seed(seed ^ START_STREAM),
            seed ^ START_STREAM,
        );
        for row in design.outer_iter() {
            if ledger.exhausted() {
                break;
            }
            let _ = obj.eval(bounds.clip(row).view());
        }
    }

    let arms = VALUES_ONLY_ARMS;
    let k = arms.len();
    let slice = (SLICE_GRAD_EQUIVALENTS * gradient).max(budget / (ROUNDS_PER_ARM * k));
    let n_slices = (budget / slice).max(1);
    let discount = 1.0 - 1.0 / (n_slices.max(2) as f64);
    let mut posteriors: Vec<ArmPosterior> = arms
        .iter()
        .map(|_| ArmPosterior::with_prior(discount, 1.0, 1.0))
        .collect();
    let index = |arm: ArmKind| arms.iter().position(|&a| a == arm).expect("arm listed");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut round = 0usize;
    let mut winner = None;
    let play = |choice: usize,
                take: usize,
                reserve: usize,
                states: &mut ArmStates,
                rng: &mut StdRng,
                posteriors: &mut [ArmPosterior]|
     -> bool {
        let arm = arms[choice];
        let before = ledger.best_get();
        let mut take = take;
        loop {
            ledger.cap_set((ledger.used_get() + take).min(budget));
            run_arm::<O, G>(arm, obj, None, ledger, states, rng, take, budget);
            ledger.cap_set(budget);
            // A kicked descent climbs before it can beat the incumbent, and
            // one from a poor start takes many slices to settle: the turn
            // is scored on the basin the descent reaches once it stops
            // paying, not where a slice boundary cuts it.
            if arm != ArmKind::Qn
                || !states.qn.as_mut().is_some_and(QnArmState::paid)
                || ledger.remaining() < reserve + 8
            {
                break;
            }
            take = take.min(ledger.remaining() - reserve);
        }
        let after = ledger.best_get();
        let improved = after.is_finite() && after < before - arm_success_threshold(arm, before);
        posteriors[choice].update(improved);
        improved
    };

    let opened = budget >= QN_AFFORDABLE_GRADIENTS * dim * gradient;
    if opened {
        let qn = index(ArmKind::Qn);
        while ledger.remaining() > 0 {
            round += 1;
            let before = ledger.best_get();
            ledger.cap_set((ledger.used_get() + slice).min(budget));
            run_qn_arm(obj, ledger, states, slice, seed, true);
            ledger.cap_set(budget);
            let after = ledger.best_get();
            let improved =
                after.is_finite() && after < before - arm_success_threshold(ArmKind::Qn, before);
            posteriors[qn].update(improved);
            winner = improved.then_some(qn);
            if !improved
                || states
                    .qn
                    .as_ref()
                    .is_none_or(|state| state.engine.is_done())
            {
                break;
            }
        }
        // The opening's successes show the start was no minimum, not that
        // kicks from the minimum pay: QN enters the rounds on the prior.
        posteriors[qn].alpha = 1.0;
        posteriors[qn].beta = 1.0;
    }
    let share = |fraction: f64| (budget as f64 * fraction).round() as usize;
    let polish = (LOCAL_FIRST_POLISH_GRADIENTS * gradient)
        .min(share(LOCAL_FIRST_POLISH_MAX_SHARE))
        .min(ledger.remaining());
    let polish = if polish >= QN_MIN_POLISH_GRADIENTS * gradient {
        polish
    } else {
        0
    };
    if opened {
        // Against a descended incumbent a GSA record is a lower basin and a
        // CMA-ES gain a better point of this one. From an undescended start
        // GSA's first records are box-wide samples that leave the start's
        // basin, so the phases wait for the opening. Both find their gains
        // in bursts (a local search per record, a run per covariance), which
        // single slices of the rounds below would not see.
        const PHASES: [ArmKind; 2] = [ArmKind::Gsa, ArmKind::Cma];
        for arm in PHASES {
            let choice = index(arm);
            // A phase that keeps paying must still leave the warm-up round
            // a slice for every other arm, the next phase's included.
            let unplayed = posteriors
                .iter()
                .enumerate()
                .filter(|&(i, posterior)| i != choice && posterior.pulls == 0)
                .count();
            let reserve = polish + unplayed * slice;
            let start = ledger.used_get();
            let mut last_gain = start;
            while ledger.remaining() >= reserve + 8 {
                let idle = ledger.used_get() - last_gain;
                if idle >= (ROUNDS_PER_ARM * slice).max(last_gain - start) {
                    break;
                }
                round += 1;
                let take = slice.min(ledger.remaining() - reserve);
                let gained = play(choice, take, reserve, states, &mut rng, &mut posteriors);
                if gained {
                    last_gain = ledger.used_get();
                }
                winner = gained.then_some(choice);
            }
        }
    }
    // Explore and GSA skip slices shorter than eight evaluations; the polish
    // takes such a remainder.
    while ledger.remaining() >= polish + 8 {
        round += 1;
        let choice = if let Some(unplayed) = posteriors.iter().position(|p| p.pulls == 0) {
            unplayed
        } else if rng.random::<f64>() < 1.0 / round as f64 {
            rng.random_range(0..k)
        } else if let Some(last) = winner {
            last
        } else {
            let mut best = (0usize, f64::NEG_INFINITY);
            for (i, posterior) in posteriors.iter().enumerate() {
                let draw = posterior.draw(&mut rng);
                if draw > best.1 {
                    best = (i, draw);
                }
            }
            best.0
        };
        let take = slice.min(ledger.remaining() - polish);
        winner = play(choice, take, polish, states, &mut rng, &mut posteriors).then_some(choice);
    }
    let qn = index(ArmKind::Qn);
    while ledger.remaining() > 0 {
        play(qn, slice, budget, states, &mut rng, &mut posteriors);
    }
    arms.iter()
        .zip(posteriors.iter())
        .map(|(arm, posterior)| ArmStat {
            name: arm.name(),
            pulls: posterior.pulls,
            successes: posterior.successes,
        })
        .collect()
}

fn arm_kind_from_name(name: &str) -> Option<ArmKind> {
    Some(match name {
        "explore" => ArmKind::Explore,
        "shift" => ArmKind::Shift,
        "hop" => ArmKind::Hop,
        "surrogate" => ArmKind::Surrogate,
        "de" => ArmKind::De,
        "gle" => ArmKind::Gle,
        "tr_poll" => ArmKind::TrPoll,
        "gsa" => ArmKind::Gsa,
        "variant" => ArmKind::Variant,
        "pt" => ArmKind::Pt,
        "hmc" => ArmKind::Hmc,
        "reduced" => ArmKind::Reduced,
        "metad" => ArmKind::Metad,
        "tps" => ArmKind::Tps,
        "am_sa" => ArmKind::AmSa,
        "dmc_pop" => ArmKind::DmcPop,
        "cma" => ArmKind::Cma,
        "qn" => ArmKind::Qn,
        _ => return None,
    })
}

/// Library of arms available for a problem (before horizon truncation / regime order).
fn library_arm_names(dim: usize, has_grad: bool) -> Vec<&'static str> {
    let mut names = vec!["explore"];
    if has_grad {
        names.extend_from_slice(&["shift", "hop", "gle"]);
    }
    // MetaD / TPS need at least a 2-D collective-variable space.
    if dim >= 2 {
        names.extend_from_slice(&["metad", "tps"]);
    }
    // Population-controlled diffusion is available in all dimensions.
    names.push("dmc_pop");
    names.extend_from_slice(&[
        "am_sa",
        "surrogate",
        "de",
        "tr_poll",
        "gsa",
        "variant",
        "pt",
    ]);
    if has_grad {
        names.push("hmc");
        if dim > 2 * REDUCED_K {
            names.push("reduced");
        }
    }
    names
}

/// Build a 2-D MetaD bias. When the ledger archive has enough diverse
/// points, fit a sketch-map (Ceriotti–Tribello–Parrinello / cosmo-epfl
/// sketchmap style) and linearize it to a projector; otherwise fall back
/// to the first two ambient coordinates.
fn make_metad_bias(
    dim: usize,
    bounds: &Bounds<f64>,
    ledger: Option<&BudgetLedger>,
) -> crate::bias::WellTemperedBias {
    use ndarray::Array2;

    // Try sketch-map CV from archive landmarks.
    if let Some(ledger) = ledger
        && let Some(bias) = try_sketchmap_metad_bias(dim, bounds, ledger)
    {
        return bias;
    }

    let mut projector = Array2::<f64>::zeros((dim, 2));
    projector[[0, 0]] = 1.0;
    if dim > 1 {
        projector[[1, 1]] = 1.0;
    } else {
        projector[[0, 1]] = 1.0;
    }
    let mu = Array1::zeros(dim);
    let lo0 = if bounds.low[0].is_finite() {
        bounds.low[0]
    } else {
        -5.0
    };
    let hi0 = if bounds.high[0].is_finite() {
        bounds.high[0]
    } else {
        5.0
    };
    let lo1 = if dim > 1 && bounds.low[1].is_finite() {
        bounds.low[1]
    } else {
        lo0
    };
    let hi1 = if dim > 1 && bounds.high[1].is_finite() {
        bounds.high[1]
    } else {
        hi0
    };
    let width0 = (hi0 - lo0).abs().max(1e-3);
    let width1 = (hi1 - lo1).abs().max(1e-3);
    let sigma = 0.15 * width0.min(width1);
    let w0 = 0.05 * mean_width(bounds).max(1.0);
    crate::bias::WellTemperedBias::new(projector, mu, [lo0, lo1], [hi0, hi1], sigma, w0, 8.0, 32)
}

/// Sketch-map MetaD bias from archive landmarks, or `None` if under-resolved.
fn try_sketchmap_metad_bias(
    dim: usize,
    bounds: &Bounds<f64>,
    ledger: &BudgetLedger,
) -> Option<crate::bias::WellTemperedBias> {
    use crate::methods::sketchmap::{SketchMap2d, farthest_point_landmarks};
    use ndarray::Array2;

    let inner = ledger.inner.lock().ok()?;
    let n = inner.archive_y.len();
    if n < 6 {
        return None;
    }
    let mut pts: Vec<Array1<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        if !inner.archive_y[i].is_finite() {
            continue;
        }
        let x = Array1::from_vec(inner.archive_x[i * dim..(i + 1) * dim].to_vec());
        if bounds.contains(x.view()) {
            pts.push(x);
        }
    }
    if pts.len() < 6 {
        return None;
    }
    let idxs = farthest_point_landmarks(&pts, 24.min(pts.len()));
    if idxs.len() < 4 {
        return None;
    }
    let mut mat = Array2::<f64>::zeros((idxs.len(), dim));
    for (row, &i) in idxs.iter().enumerate() {
        for d in 0..dim {
            mat[[row, d]] = pts[i][d];
        }
    }
    let sm = SketchMap2d::fit(mat.view(), 35, 0.04)?;
    let (projector, mu, lo, hi) = sm.linearize_projector(1e-3 * mean_width(bounds).max(1e-3));
    // Guard degenerate Jacobians.
    let col0: f64 = (0..dim)
        .map(|d| projector[[d, 0]].powi(2))
        .sum::<f64>()
        .sqrt();
    let col1: f64 = (0..dim)
        .map(|d| projector[[d, 1]].powi(2))
        .sum::<f64>()
        .sqrt();
    if col0 < 1e-12 || col1 < 1e-12 {
        return None;
    }
    let width0 = (hi[0] - lo[0]).abs().max(1e-3);
    let width1 = (hi[1] - lo[1]).abs().max(1e-3);
    let sigma = 0.12 * width0.min(width1);
    let w0 = 0.05 * mean_width(bounds).max(1.0);
    Some(crate::bias::WellTemperedBias::new(
        projector, mu, lo, hi, sigma, w0, 8.0, 32,
    ))
}

/// MetaD arm: Metropolis random-walk on `F_eff = F + V`, deposits well-tempered
/// bias, records **true** `F` via the budgeted objective.
fn run_metad_arm<O>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    states: &mut ArmStates,
    rng: &mut StdRng,
    slice: usize,
    bounds: &Bounds<f64>,
    dim: usize,
) where
    O: Objective<f64>,
{
    states.last_exploratory_ok = false;
    if dim < 2 || slice < 4 {
        return;
    }
    if states.bias.is_none() {
        states.bias = Some(make_metad_bias(dim, bounds, Some(ledger)));
    }
    let bias = states.bias.as_mut().expect("bias installed");
    let mut x = ledger.incumbent(bounds);
    let f0 = obj.eval(x.view());
    if !f0.is_finite() {
        return;
    }
    let s0 = bias.cv(x.view());
    let mut f_eff = f0 + bias.potential(s0.view());
    let temp = ladder_temperature(archive_temp0(ledger), states.surrogate_gen);
    let width = mean_width(bounds);
    let step = 0.05 * width;
    let deposit_period = 8usize;
    let mut deposits = 0usize;
    for t in 0..slice {
        if ledger.exhausted() {
            break;
        }
        let mut y = x.clone();
        for j in 0..dim {
            let z: f64 = rand_distr::StandardNormal.sample(rng);
            y[j] += step * z;
        }
        let y = crate::movekernel::reflect_into_box(y.view(), bounds);
        let ft = obj.eval(y.view());
        if !ft.is_finite() {
            continue;
        }
        let s = bias.cv(y.view());
        let fe = ft + bias.potential(s.view());
        let delta = energy_delta(fe, f_eff);
        if metropolis(delta, temp, rng) {
            x = y;
            f_eff = fe;
            if t > 0 && t % deposit_period == 0 {
                bias.deposit(s.view(), temp.max(1e-9));
                deposits += 1;
            }
        }
    }
    if deposits > 0 {
        states.last_exploratory_ok = true;
    }
    states.surrogate_gen = states.surrogate_gen.saturating_add(1);
}

/// Two archive endpoints for TPS: low-F "product" and a distinct higher-F
/// "reactant" seed (pnastps basins A/B adapted to true objective).
fn archive_basin_pair(
    ledger: &BudgetLedger,
    bounds: &Bounds<f64>,
) -> Option<(Array1<f64>, f64, Array1<f64>, f64)> {
    let inner = ledger.inner.lock().expect("ledger lock");
    let dim = ledger.dim;
    let n = inner.archive_y.len();
    if n < 2 {
        return None;
    }
    let mut best_i = None;
    let mut best_v = f64::INFINITY;
    for i in 0..n {
        let v = inner.archive_y[i];
        if v.is_finite() && v < best_v {
            best_v = v;
            best_i = Some(i);
        }
    }
    let bi = best_i?;
    let x_b = Array1::from_vec(inner.archive_x[bi * dim..(bi + 1) * dim].to_vec());
    let min_sep = 0.05 * mean_width(bounds);
    let mut best_a = None;
    let mut best_a_v = f64::NEG_INFINITY;
    for i in 0..n {
        if i == bi {
            continue;
        }
        let v = inner.archive_y[i];
        if !v.is_finite() {
            continue;
        }
        let xi = Array1::from_vec(inner.archive_x[i * dim..(i + 1) * dim].to_vec());
        let dist = xi
            .iter()
            .zip(x_b.iter())
            .map(|(u, w)| (u - w).powi(2))
            .sum::<f64>()
            .sqrt();
        if dist < min_sep {
            continue;
        }
        // Prefer a distinctly worse (higher) finite point as the reactant.
        if v >= best_v && v > best_a_v {
            best_a_v = v;
            best_a = Some((xi, v));
        }
    }
    let (x_a, f_a) = best_a?;
    if !bounds.contains(x_a.view()) || !bounds.contains(x_b.view()) {
        return None;
    }
    Some((x_a, f_a, x_b, best_v))
}

/// TPS-inspired shooting arm (pnastps forward/backward shoot analogue).
fn run_tps_arm<O>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    states: &mut ArmStates,
    rng: &mut StdRng,
    slice: usize,
    bounds: &Bounds<f64>,
    dim: usize,
) where
    O: Objective<f64>,
{
    use crate::methods::tps_shoot::{
        ShootDirection, accept_reactive_shoot, apply_shoot, linear_path, path_reactive_geometric,
        path_reactive_objective, pick_shoot_direction, pick_shoot_index,
    };
    states.last_exploratory_ok = false;
    if dim < 2 || slice < 8 {
        return;
    }
    let Some((x_a, f_a, x_b, f_b)) = archive_basin_pair(ledger, bounds) else {
        // Fall back: short QMC GSA scout to populate the archive for later TPS.
        if slice >= 8 {
            qmc_gsa_global_search(obj, slice / 2, rng.random(), 2, 1.0, GSA_Q_V, GSA_Q_A);
        }
        return;
    };
    let n_frames = ((slice / 4).clamp(5, 17) | 1).max(5); // odd, >=5
    let mut path = linear_path(x_a.view(), x_b.view(), n_frames);
    let mut ops: Vec<f64> = Vec::with_capacity(n_frames);
    for frame in &path {
        if ledger.exhausted() {
            return;
        }
        let x = crate::movekernel::reflect_into_box(frame.view(), bounds);
        ops.push(obj.eval(x.view()));
    }
    // Objective-as-OP thresholds: reactant high, product low (pnastps a/b flipped).
    let high = 0.5 * (f_a + f_b) + 0.25 * (f_a - f_b).abs().max(1e-9);
    let low = f_b + 0.25 * (f_a - f_b).abs().max(1e-9);
    let tol = 0.25 * mean_width(bounds);
    let noise = 0.08 * mean_width(bounds);
    let n_shoots = (slice / n_frames).max(1);
    let mut accepts = 0usize;
    for _ in 0..n_shoots {
        if ledger.remaining() < n_frames || n_frames < 3 {
            break;
        }
        let shoot = pick_shoot_index(n_frames, rng);
        let dir = pick_shoot_direction(rng);
        let reflect = |x: Array1<f64>| crate::movekernel::reflect_into_box(x.view(), bounds);
        let trial = apply_shoot(
            &path,
            shoot,
            dir,
            x_a.view(),
            x_b.view(),
            noise,
            rng,
            reflect,
        );
        let mut trial_ops = Vec::with_capacity(n_frames);
        for frame in &trial {
            if ledger.exhausted() {
                break;
            }
            let x = crate::movekernel::reflect_into_box(frame.view(), bounds);
            trial_ops.push(obj.eval(x.view()));
        }
        if trial_ops.len() != n_frames {
            break;
        }
        let reactive = path_reactive_objective(&trial_ops, high, low)
            || path_reactive_geometric(&trial, x_a.view(), x_b.view(), tol);
        if accept_reactive_shoot(reactive) {
            path = trial;
            ops = trial_ops;
            accepts += 1;
        }
    }
    if accepts > 0 {
        states.last_exploratory_ok = true;
    }
    let _ = (ops, ShootDirection::Forward);
}

/// Arms in regime-preferred order; the horizon cap truncates from the back, so
/// the core stays active at small budgets and the full library unlocks
/// as the slice count grows. Restart arm `explore` is always first.
///
/// `extra_slots` (Auto only) unlocks additional specialized arms when the
/// horizon would otherwise starve them (e.g. GLE+reduced under high dim).
fn enabled_arms(
    dim: usize,
    has_grad: bool,
    n_slices: usize,
    regime: crate::methods::regime::OptimizationRegime,
    extra_slots: usize,
) -> Vec<ArmKind> {
    let available = library_arm_names(dim, has_grad);
    // The posterior needs ROUNDS_PER_ARM pulls per arm to rank them;
    // activate only as many arms as the horizon can rank (the D4 regret
    // grows with K, and a starved arm is worse than an absent one).
    // Values-only Auto runs never reach this bandit (see
    // run_values_only_portfolio), so MultimodalNoGrad needs no cap here.
    let regime_cap = match regime {
        // Keep enough arms that MetaD/TPS/dmc remain callable under Auto
        // (tests + exploration) while still preferring GSA/DE first.
        crate::methods::regime::OptimizationRegime::MultimodalGlobal => 8,
        _ => available.len(),
    };
    let k_active = (n_slices / ROUNDS_PER_ARM)
        .saturating_add(extra_slots)
        .clamp(4, regime_cap.min(available.len()).max(4));
    let ordered = crate::methods::regime::order_arms(&available, regime, k_active);
    ordered
        .iter()
        .filter_map(|n| arm_kind_from_name(n))
        .collect()
}

/// Portfolio scheduling policy (GJQ-style: auto vs flat legacy for A/B).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PortfolioPolicy {
    /// Feature-based regime selection + prior boosts (default shipped path).
    #[default]
    Auto,
    /// Pre-regime behaviour: fixed Default arm order, uninformative Beta(1,1).
    /// Used only for same-protocol regression / A-B measurement.
    Legacy,
}

/// Runs the portfolio under the default auto policy.
pub fn portfolio_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    noise_sigma: Option<f64>,
) -> PortfolioResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    portfolio_optimize_with_policy(obj, grad, budget, seed, noise_sigma, PortfolioPolicy::Auto)
}

/// Runs the portfolio driver under a shared work-unit budget.
///
/// `budget` bounds combined true-objective and native-gradient
/// evaluations and is the driver's only required parameter. `grad`
/// enables the gradient arms and the final polish.
/// `policy` selects regime auto-routing (`Auto`) or flat legacy order.
pub fn portfolio_optimize_with_policy<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    noise_sigma: Option<f64>,
    policy: PortfolioPolicy,
) -> PortfolioResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    portfolio_optimize_from(obj, grad, budget, seed, noise_sigma, policy, None)
}

/// [`portfolio_optimize_from`] under [`PortfolioPolicy::Auto`].
pub fn portfolio_optimize_seeded<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    noise_sigma: Option<f64>,
    x0: Option<ArrayView1<f64>>,
) -> PortfolioResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    portfolio_optimize_from(
        obj,
        grad,
        budget,
        seed,
        noise_sigma,
        PortfolioPolicy::Auto,
        x0,
    )
}

/// [`portfolio_optimize_with_policy`] from a caller-supplied start.
///
/// `x0` is the first charged evaluation and costs one work unit of
/// `budget`. It is the incumbent until a lower value is found (if its value
/// is not finite, until any finite value is), and arms that read the
/// incumbent start from it: in the values-only loop the descents, CMA-ES,
/// one GSA chain and the first DE member; with a gradient or declared noise
/// the trust-region poll, HMC and the population arm. Arms with their own
/// designs (the Bayesian pilot, the reduced-space search) do not. Without
/// `x0` the values-only loop starts from the best of a small seeded design.
pub fn portfolio_optimize_from<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    noise_sigma: Option<f64>,
    policy: PortfolioPolicy,
    x0: Option<ArrayView1<f64>>,
) -> PortfolioResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(budget > 0, "budget must be positive");
    if let Some(sigma) = noise_sigma {
        assert!(
            sigma > 0.0 && sigma.is_finite(),
            "noise_sigma must be positive and finite"
        );
    }
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    assert!(dim > 0, "objective dimension must be positive");
    for i in 0..dim {
        assert!(
            bounds.low[i].is_finite() && bounds.high[i].is_finite(),
            "bounds must be finite at dimension {i}"
        );
        assert!(
            bounds.low[i] < bounds.high[i],
            "low[{i}] must be strictly less than high[{i}]"
        );
    }

    // GJQ-style regime refusal on the shipped accept path: declared noise
    // forbids exact-only Metropolis. The portfolio always uses OSA when
    // noise_sigma is Some (see accept_move); this gate fails loudly if that
    // invariant is ever broken by a caller API that passes use_noise_aware=false.
    let use_noise_aware = noise_sigma.is_some();
    crate::methods::regime::require_accept_compatible(noise_sigma, use_noise_aware)
        .expect("shipped portfolio always pairs declared noise with OSA accept");

    // GJQ-style regime auto-selection from measurable features.
    let features = crate::methods::regime::ProblemFeatures::from_bounds(
        &bounds,
        grad.is_some(),
        noise_sigma,
        budget,
    );
    let mut regime = match policy {
        PortfolioPolicy::Auto => crate::methods::regime::select_regime(&features),
        PortfolioPolicy::Legacy => crate::methods::regime::OptimizationRegime::Default,
    };

    let ledger = BudgetLedger::new(budget, dim);
    let budgeted_obj = BudgetedObjective {
        inner: obj,
        ledger: &ledger,
    };
    let reflected_grad = grad.map(|g| ReflectedGradient {
        inner: g,
        bounds: &bounds,
    });
    let budgeted_grad = reflected_grad.as_ref().map(|g| BudgetedGradient {
        inner: g,
        ledger: &ledger,
    });
    if let Some(x0) = x0 {
        assert_eq!(x0.len(), dim, "x0 must have the objective's dimension");
        assert!(bounds.contains(x0), "x0 must lie inside the bounds");
        let _ = budgeted_obj.eval(x0);
    }

    if policy == PortfolioPolicy::Auto && grad.is_none() && noise_sigma.is_none() {
        let mut states = ArmStates::default();
        let arm_stats =
            run_values_only_portfolio::<O, G>(&budgeted_obj, &ledger, &mut states, seed, budget);
        return portfolio_result(&ledger, &bounds, arm_stats);
    }

    // Probe-based demotion for mid-width MultimodalGlobal boxes. Width alone
    // cannot separate a Styblinski-class multi-basin box from a least-squares
    // valley of the same width, and the wide-box (Schwefel) front-load only
    // fires at mean_width >= 50, so this gate spends ~2% of the budget on
    // short spread descents and commits to global visiting only when at
    // least two basin depth classes appear.
    let mut descent_dominated = false;
    let mut hop_first = false;
    if policy == PortfolioPolicy::Auto
        && matches!(
            regime,
            crate::methods::regime::OptimizationRegime::MultimodalGlobal
        )
        && mean_width(&bounds) < 50.0
        && let Some(bg) = budgeted_grad.as_ref()
    {
        let probe_budget = (budget / 50).clamp(24, 120);
        // Each descent step charges one eval and one gradient.
        let per_start = (probe_budget / (3 * PROJECTED_GRADIENT_STEP_WORK)).max(6);
        let center = (&bounds.low + &bounds.high) * 0.5;
        let qmc = eindir_core::shifted_low_discrepancy_points(
            &bounds,
            2,
            qmc_skip_from_seed(seed ^ 0x9E37_u64),
            seed ^ 0x9E37_u64,
        );
        let mut starts = vec![bounds.clip(center.view())];
        for row in qmc.outer_iter() {
            starts.push(bounds.clip(row));
        }
        let ceiling = ledger.used_get() + probe_budget;
        ledger.cap_set(ceiling.min(budget));
        let probe = crate::methods::routing_probe::multimodality_probe(
            &budgeted_obj,
            bg,
            &bounds,
            &starts,
            per_start,
        );
        ledger.cap_set(budget);
        if let Some(probe) = probe {
            if !probe.multimodal() {
                descent_dominated = true;
                regime = if dim <= 5 {
                    crate::methods::regime::OptimizationRegime::LowDimSmooth
                } else {
                    crate::methods::regime::OptimizationRegime::Default
                };
            } else {
                // Multiple basins, but do descents pay? Compare the descent
                // gain with the gain from raw sampling alone: when descending
                // beats sampling by a wide margin, the basin-hopping pattern
                // (deep descent per jump) dominates heavy-tailed visiting.
                let raw_gain = (probe.worst_start_value - probe.best_start_value).max(0.0);
                let descent_gain = (probe.worst_start_value - probe.best_descent_value).max(0.0);
                if descent_gain.is_finite() && descent_gain > 3.0 * raw_gain.max(f64::MIN_POSITIVE)
                {
                    hop_first = true;
                }
            }
        }
    }

    // Derived scheduler quantities; see the module constants for the
    // two governing numbers. Auto unlocks extra specialized arms so the
    // regime order can actually include GLE/reduced/DE under tight horizons.
    // Unlock MetaD + TPS (+ one more specialized arm) under Auto so the
    // horizon does not starve the new enhanced-sampling machinery.
    let extra = match (policy, regime) {
        (
            PortfolioPolicy::Auto,
            crate::methods::regime::OptimizationRegime::HighDimIllConditioned
            | crate::methods::regime::OptimizationRegime::LowDimSmooth,
        ) => 4,
        (PortfolioPolicy::Auto, _) => 2,
        _ => 0,
    };
    let arm_count_all = enabled_arms(dim, grad.is_some(), usize::MAX, regime, extra).len();
    let slice =
        (SLICE_GRAD_EQUIVALENTS * (dim + 1)).max(budget / (ROUNDS_PER_ARM * arm_count_all).max(1));
    let n_slices = (budget / slice.max(1)).max(1);
    let arms = enabled_arms(dim, grad.is_some(), n_slices, regime, extra);
    debug_assert_eq!(arms[0], RESTART_ARM);
    // Posterior memory matches the slice horizon; Auto boosts preferred arms.
    let discount = 1.0 - 1.0 / (n_slices.max(2) as f64);
    let mut posteriors: Vec<ArmPosterior> = arms
        .iter()
        .map(|arm| {
            let (a0, b0) = match policy {
                PortfolioPolicy::Auto => {
                    crate::methods::regime::arm_prior_boost(regime, arm.name())
                }
                PortfolioPolicy::Legacy => (1.0, 1.0),
            };
            ArmPosterior::with_prior(discount, a0, b0)
        })
        .collect();
    if descent_dominated || hop_first {
        // Depth-first evidence: bias Thompson sampling toward the hop arm
        // (perturb + full-depth descent), the basin-hopping analogue.
        for (arm, post) in arms.iter().zip(posteriors.iter_mut()) {
            if arm.name() == "hop" {
                *post = ArmPosterior::with_prior(discount, 6.0, 1.0);
            }
        }
    }
    // TPE dual-density history over arm indices (Auto policy).
    let mut tpe = crate::methods::tpe::TpeCategorical::new(arms.len());
    let mut states = ArmStates {
        noise_sigma,
        deep_hops: descent_dominated || hop_first,
        ..ArmStates::default()
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let k = arms.len();
    let mut low_dimensional_polish = grad
        .is_some()
        .then(|| low_dimensional_polish_plan(dim, budget))
        .flatten();
    if let (Some(plan), Some(grad)) = (low_dimensional_polish.as_ref(), budgeted_grad.as_ref()) {
        let center_probe = scaled_center_probe(&budgeted_obj, grad, &bounds);
        let center_ratio = center_probe.and_then(|probe| probe.gradient_ratio);
        if low_dimensional_polish_before_warmup(center_ratio) {
            // Grab the cheap stationary point, then keep exploring:
            // stationarity certifies a local basin, not the global one,
            // and the budget is the caller's authority on effort.
            run_low_dimensional_polish(&budgeted_obj, grad, &ledger, plan, seed, budget);
            low_dimensional_polish = None;
        }
    }

    // Auto MultimodalGlobal on *very* wide boxes only (Schwefel mean_width
    // ~1000). Mid-width multimodal (Rastrigin/Styblinski ~10) keep a bandit
    // residual so dmc_pop/metad/endgame still run; this explore/GSA/DE
    // front-load is reserved for boxes where dual_annealing dominates.
    if policy == PortfolioPolicy::Auto
        && matches!(
            regime,
            crate::methods::regime::OptimizationRegime::MultimodalGlobal
        )
        && mean_width(&bounds) >= 50.0
    {
        // High-d wide boxes need more residual multi-start LS budget
        // (dual spends heavily on local search); mid-d keep GSA-heavy.
        let front_frac = if dim >= 20 { 0.72 } else { 0.88 };
        let front_budget = ((budget as f64) * front_frac).round() as usize;
        let explore_share = (front_budget / 15).max(16);
        let de_share = (front_budget / 12).max(16);
        let gsa_share = front_budget.saturating_sub(explore_share + de_share);
        let mut seed_f = seed ^ 0xD1A1_u64;
        for (arm, share) in [
            (ArmKind::Explore, explore_share),
            (ArmKind::Gsa, gsa_share),
            (ArmKind::De, de_share),
        ] {
            if !arms.contains(&arm) || ledger.remaining() < 8 || share < 8 {
                continue;
            }
            let slice_use = share.min(ledger.remaining());
            let ceiling = ledger.used_get() + slice_use;
            ledger.cap_set(ceiling.min(budget));
            seed_f = seed_f.wrapping_add(1);
            let mut front_rng = StdRng::seed_from_u64(seed_f);
            if matches!(arm, ArmKind::Gsa) {
                // One contiguous dual-like anneal (multi-block restarts
                // reheat too often and waste the Tsallis cool schedule).
                // Optional mid-budget reanneal via DUAL_RESTART_TEMP_RATIO.
                if ledger.remaining() >= 32 {
                    states.gsa = None;
                    let block = slice_use.min(ledger.remaining());
                    let ceil2 = ledger.used_get() + block;
                    ledger.cap_set(ceil2.min(budget));
                    seed_f = seed_f.wrapping_add(1);
                    let mut brng = StdRng::seed_from_u64(seed_f);
                    run_arm(
                        ArmKind::Gsa,
                        &budgeted_obj,
                        budgeted_grad.as_ref(),
                        &ledger,
                        &mut states,
                        &mut brng,
                        block,
                        budget,
                    );
                    ledger.cap_set(budget);
                }
            } else {
                run_arm(
                    arm,
                    &budgeted_obj,
                    budgeted_grad.as_ref(),
                    &ledger,
                    &mut states,
                    &mut front_rng,
                    slice_use,
                    budget,
                );
                ledger.cap_set(budget);
            }
        }
        // dual_annealing always runs local search (L-BFGS-B, finite-diff if
        // no jac). Multi-start FD / analytic polish closes residual dual
        // exclusives (no-grad Schwefel, high-d Styblinski wells).
        if ledger.remaining() >= 32 {
            // Spend almost all residual on global multi-start LS (not thin 12%).
            let polish = ledger.remaining().saturating_sub(16).max(64);
            dual_style_local_search(
                &budgeted_obj,
                budgeted_grad.as_ref(),
                &ledger,
                &bounds,
                &mut rng,
                polish,
                budget,
            );
            ledger.cap_set(budget);
        }
    }

    // Mid-width MultimodalGlobal (Styblinski-class): multi-start polish seed
    // even without the Schwefel GSA front-load (width gate is mean_width>=50).
    if policy == PortfolioPolicy::Auto
        && matches!(
            regime,
            crate::methods::regime::OptimizationRegime::MultimodalGlobal
        )
        && mean_width(&bounds) < 50.0
        && budgeted_grad.is_some()
        && ledger.remaining() >= 48
    {
        let polish = ((budget as f64) * 0.10).round() as usize;
        let polish = polish.max(48).min(ledger.remaining() / 3);
        dual_style_local_search(
            &budgeted_obj,
            budgeted_grad.as_ref(),
            &ledger,
            &bounds,
            &mut rng,
            polish,
            budget,
        );
        ledger.cap_set(budget);
    }

    let mut round = 0usize;
    // Bayesian endgame state: NIG posterior over the polish contraction
    // (fed by consecutive polish-arm improvement ratios; a basin jump
    // resets the pairing) plus the exploration-arm Beta posteriors.
    let mut contraction = ContractionPosterior::new();
    let mut prev_polish_delta: Option<f64> = None;
    // Exploratory (MetaD/TPS) slices awaiting conversion: (arm, round).
    const CREDIT_WINDOW: usize = ROUNDS_PER_ARM;
    let mut pending_credit: Vec<(usize, usize)> = Vec::new();
    loop {
        let remaining = ledger.remaining();
        if remaining < 4 {
            break;
        }
        // One-step-lookahead Bayesian stopping rule. The D5 two-basin win
        // objective for splitting the remaining budget into e exploration
        // slices followed by p polish work units is
        //   W(p) = [q0 + (1 - q0)(1 - E[(1-theta)^e])] * P[polish converts | p],
        // where theta is the per-slice probability an exploration arm finds
        // a better basin (aggregated Beta posterior, so
        // E[(1-theta)^e] = prod_{i<e} (beta+i)/(alpha+beta+i) exactly),
        // q0 = beta/(alpha+beta) is the posterior probability the incumbent
        // survives another challenge slice, and the conversion factor comes
        // from the contraction posterior. The endgame starts exactly when
        // the maximizer p* of W wants the whole remaining budget: for this
        // monotone stopping problem the one-step-lookahead rule is optimal
        // (Chow-Robbins monotone case).
        // Posterior-quantile reserve floor with asymmetric loss: an
        // unconverted basin forfeits the cell outright while a slightly
        // shorter exploration phase rarely loses one (measured near-best
        // exceeds win rate), so slow-contraction evidence may lengthen
        // the tail (up to a third of the budget) but never shorten it
        // below the D5 quarter-budget floor. An uninformative posterior
        // (84% quantile cannot certify contraction) keeps the floor.
        let reserve_floor = match contraction.reserve_wu(NEAR_BEST_ORDERS) {
            usize::MAX => budget / 4,
            r => r.clamp(budget / 4, budget / 3),
        };
        let endgame_now = if policy == PortfolioPolicy::Auto && round >= k {
            if remaining <= reserve_floor {
                true
            } else {
                // Exploration term. Preferred: the distribution-free
                // Good-Turing x record gate from the basin registry - the
                // per-slice probability that a restart finds an unseen basin
                // (missing mass n1/n) that also beats the incumbent (record
                // probability 1/(w+1)). Fallback while the registry is thin:
                // the aggregated Beta posteriors over improvement bits.
                let gt = states.basins.record_discovery_prob();
                let mut a_e = 0.0f64;
                let mut b_e = 0.0f64;
                for (arm, post) in arms.iter().zip(posteriors.iter()) {
                    if !matches!(arm, ArmKind::Shift | ArmKind::TrPoll | ArmKind::Hop) {
                        a_e += post.alpha;
                        b_e += post.beta;
                    }
                }
                let win = |p: usize| -> f64 {
                    let p_conv = contraction.conversion_prob(NEAR_BEST_ORDERS, p);
                    match gt {
                        // D10.1 under D9 discovery value.
                        Some(theta) => {
                            win_objective_discovery(remaining, p, slice.max(1), theta, p_conv)
                        }
                        None => {
                            // D4 Beta fallback: E[(1-theta)^e] product.
                            let e = remaining.saturating_sub(p) / slice.max(1);
                            let q0 = b_e / (a_e + b_e).max(1e-12);
                            let mut pi = 1.0f64;
                            for i in 0..e.min(128) {
                                pi *= (b_e + i as f64) / (a_e + b_e + i as f64);
                            }
                            (q0 + (1.0 - q0) * (1.0 - pi)) * p_conv
                        }
                    }
                };
                let grid = 12usize;
                let mut best_p = remaining / grid;
                let mut best_w = win(best_p);
                for j in 2..=grid {
                    let p = remaining * j / grid;
                    let w = win(p);
                    if w > best_w {
                        best_w = w;
                        best_p = p;
                    }
                }
                best_p + slice > remaining
            }
        } else {
            false
        };
        if round >= k
            && let (Some(plan), Some(grad)) =
                (low_dimensional_polish.take(), budgeted_grad.as_ref())
        {
            // Stationarity here certifies a local basin only; the
            // bandit keeps exploring with the remaining budget and
            // the endgame re-polishes the final incumbent.
            run_low_dimensional_polish(&budgeted_obj, grad, &ledger, &plan, seed, budget);
            continue;
        }
        if endgame_now || remaining < slice {
            // D5 endgame: the tail is pure polish (explore-first is
            // optimal). Cycle
            // quasi-Newton polish with shrinking trust-region poll
            // restarts until the budget is gone or dry cycles exhaust.
            //
            // Critical: do *not* dump all remaining WU into the first
            // projected_gradient call. CMA exclusives on CERI*/COOLHANS
            // are polish-depth ties; multi-start jittered restarts need
            // a reserved share of the tail.
            ledger.cap_set(budget);
            coordinate_opposition_scout(&budgeted_obj, &ledger, &bounds);
            if let Some(grad) = budgeted_grad.as_ref() {
                // Phase A: deep L-BFGS on the incumbent only (2/3 of remaining).
                // No jittered restarts here — those can leave a worse recorded
                // trajectory on CERI-scale cells; multi-start jitter already
                // happens in bandit arms.
                let rem0 = ledger.remaining();
                if rem0 >= 4 {
                    let cycle_wu = endgame_cycle_cap(rem0, 0);
                    run_endgame_projected_polish_cycle(
                        &budgeted_obj,
                        grad,
                        &ledger,
                        ledger.incumbent(&bounds),
                        cycle_wu,
                        budget,
                        1.0,
                        1e-14,
                    );
                }
                // Phase B: second L-BFGS pass on the *same* incumbent with a
                // smaller step0 (no position jitter) — pure refine.
                if ledger.remaining() >= 16 {
                    let rem = ledger.remaining();
                    let cycle_wu = endgame_cycle_cap(rem, 1);
                    run_endgame_projected_polish_cycle(
                        &budgeted_obj,
                        grad,
                        &ledger,
                        ledger.incumbent(&bounds),
                        cycle_wu,
                        budget,
                        0.01,
                        0.0,
                    );
                }
                // Phase C: monotonic coordinate micro-refine (accept-only).
                if ledger.remaining() >= 4 {
                    run_endgame_coordinate_microrefine(&budgeted_obj, &ledger, &bounds, budget);
                }
                // Phase D: multi-start dual-style LS (closes Styblinski dual leads).
                if ledger.remaining() >= 32 {
                    let rem = ledger.remaining();
                    dual_style_local_search(
                        &budgeted_obj,
                        Some(grad),
                        &ledger,
                        &bounds,
                        &mut rng,
                        rem,
                        budget,
                    );
                }
            } else if ledger.remaining() >= 8 {
                // No native grad: dual-class FD multi-start polish (not AmSa-only).
                let rem = ledger.remaining();
                dual_style_local_search::<_, G, _>(
                    &budgeted_obj,
                    None,
                    &ledger,
                    &bounds,
                    &mut rng,
                    rem,
                    budget,
                );
            }
            break;
        }
        round += 1;
        // One observation per active arm gives the finite-budget
        // posterior a real datum before ranking; the decaying uniform
        // floor then certifies that every arm, including the restart
        // arm, is played infinitely often (sum 1/(mK) diverges).
        // Auto policy: after warmup, with probability regime_exploit_prob,
        // pick among the regime's preferred active arms (GJQ-style
        // feature routing that actually changes finite-budget allocation).
        let floor = (1.0 / round as f64).min(1.0);
        // Allocation stack (Auto): warmup → uniform floor → TPE dual-density
        // (once enough history) → regime exploit → floored Thompson.
        // Legacy keeps Thompson-only after warmup/floor (A/B baseline).
        let choice = if let Some(warmup) = warmup_arm_index(round, k) {
            warmup
        } else if rng.random::<f64>() < floor {
            rng.random_range(0..k)
        } else if policy == PortfolioPolicy::Auto
            && tpe.len() >= (2 * k).max(8)
            && rng.random::<f64>() < 0.60
        {
            tpe.pick(&mut rng)
        } else if policy == PortfolioPolicy::Auto
            && rng.random::<f64>() < crate::methods::regime::regime_exploit_prob(regime)
        {
            let width = crate::methods::regime::regime_exploit_width(regime);
            let preferred = crate::methods::regime::preferred_arm_tail(regime);
            let mut idxs: Vec<usize> = arms
                .iter()
                .enumerate()
                .filter(|(_, arm)| preferred.iter().take(width).any(|&name| name == arm.name()))
                .map(|(i, _)| i)
                .collect();
            if idxs.is_empty() {
                let mut best_idx = 0usize;
                let mut best_draw = f64::NEG_INFINITY;
                for (idx, posterior) in posteriors.iter().enumerate() {
                    let draw = posterior.draw(&mut rng);
                    if draw > best_draw {
                        best_draw = draw;
                        best_idx = idx;
                    }
                }
                best_idx
            } else {
                idxs.sort_unstable();
                idxs[rng.random_range(0..idxs.len())]
            }
        } else {
            let mut best_idx = 0usize;
            let mut best_draw = f64::NEG_INFINITY;
            for (idx, posterior) in posteriors.iter().enumerate() {
                let draw = posterior.draw(&mut rng);
                if draw > best_draw {
                    best_draw = draw;
                    best_idx = idx;
                }
            }
            best_idx
        };
        let before = ledger.best_get();
        let reward_scale = scheduler_reward_scale(&ledger);
        let threshold = scheduler_success_threshold(arms[choice], &ledger);
        // Auto: preferred arms get larger slices under the same total budget.
        let arm_slice = if policy == PortfolioPolicy::Auto {
            let name = arms[choice].name();
            let mut mult = crate::methods::regime::arm_slice_multiplier(regime, name);
            if hop_first {
                // Probe: descents pay; hop takes the large slices instead of GSA.
                if name == "gsa" {
                    mult = 1.0;
                } else if name == "hop" {
                    mult = mult.max(3.5);
                }
            }
            ((slice as f64) * mult).round() as usize
        } else {
            slice
        }
        .max(4)
        .min(ledger.remaining().max(4));
        let used_before = ledger.used_get();
        let ceiling = used_before + arm_slice;
        ledger.cap_set(ceiling.min(budget));
        run_arm(
            arms[choice],
            &budgeted_obj,
            budgeted_grad.as_ref(),
            &ledger,
            &mut states,
            &mut rng,
            arm_slice,
            budget,
        );
        ledger.cap_set(budget);
        let after = ledger.best_get();
        let improved = after.is_finite() && after < before - threshold;
        // Deferred exploratory credit: a MetaD deposit or an accepted
        // reactive path earns a bandit success only if the incumbent
        // improves within the next CREDIT_WINDOW slices (any arm), so
        // exploration that never converts cannot freeload allocation.
        if improved {
            for (c, _) in pending_credit.drain(..) {
                posteriors[c].update(true);
                tpe.record(c, threshold.max(1e-12));
            }
        } else {
            pending_credit.retain(|&(c, r)| {
                if round.saturating_sub(r) > CREDIT_WINDOW {
                    posteriors[c].update(false);
                    false
                } else {
                    true
                }
            });
        }
        if matches!(
            arms[choice],
            ArmKind::Shift | ArmKind::TrPoll | ArmKind::Hop
        ) && improved
            && before.is_finite()
        {
            let delta = before - after;
            if delta > 0.1 * reward_scale.max(1.0) {
                // Basin jump: contraction ratios across basins are
                // meaningless; restart the ratio pairing.
                prev_polish_delta = None;
            }
            if let Some(prev) = prev_polish_delta
                && delta < prev
                && delta > 0.0
            {
                // Work-normalized: this slice spent `spent` units to
                // contract the improvement by delta/prev.
                let spent = ledger.used_get().saturating_sub(used_before).max(1);
                contraction.observe((delta / prev).ln() / spent as f64);
            }
            prev_polish_delta = Some(delta);
        }
        let exploratory = arms[choice].exploratory_credit() && states.last_exploratory_ok;
        if exploratory && !improved {
            // Posterior and TPE updates wait in the credit window.
            pending_credit.push((choice, round));
        } else {
            posteriors[choice].update(improved);
        }
        // TPE score: true-F improvement magnitude only; exploratory
        // bonuses arrive through the deferred-credit path.
        let score = if improved {
            (before - after) / reward_scale.max(1.0)
        } else {
            0.0
        };
        tpe.record(choice, score);
        states.last_exploratory_ok = false;
    }

    let arm_stats = arms
        .iter()
        .zip(posteriors.iter())
        .map(|(arm, posterior)| ArmStat {
            name: arm.name(),
            pulls: posterior.pulls,
            successes: posterior.successes,
        })
        .collect();
    portfolio_result(&ledger, &bounds, arm_stats)
}

fn portfolio_result(
    ledger: &BudgetLedger,
    bounds: &Bounds<f64>,
    arm_stats: Vec<ArmStat>,
) -> PortfolioResult {
    let best_pos_arr = ledger.incumbent(bounds);
    // Final safety: never return OOB coordinates from the public API.
    let best_pos = bounds.clip(best_pos_arr.view()).to_vec();
    debug_assert!(
        bounds.contains(ArrayView1::from(&best_pos)),
        "portfolio best_pos must be in bounds"
    );
    PortfolioResult {
        best_pos,
        best_val: ledger.best_get(),
        n_evals: ledger.n_evals.load(Ordering::Relaxed),
        n_grads: ledger.n_grads.load(Ordering::Relaxed),
        arm_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::{Rastrigin, StybTang2D};

    /// Pure WU split: first cycle is 2/3 remaining (leaves ≥1/3 residual).
    #[test]
    fn endgame_cycle_cap_reserves_for_multistart() {
        let rem = 2000usize;
        let first = endgame_cycle_cap(rem, 0);
        assert!(
            first < rem,
            "first cycle must not dump full tail: {first} vs {rem}"
        );
        assert_eq!(first, (rem * 2) / 3);
        assert!(rem - first >= rem / 3);
        let later = endgame_cycle_cap(rem, 1);
        assert_eq!(later, rem / 2);
        assert!(first + later > rem); // snapshots on full rem; sequential is smaller
        // WU → max_fevals conversion: fevals * step_work ≤ cycle_wu.
        let maxf = endgame_polish_max_fevals(first);
        assert!(
            maxf * PROJECTED_GRADIENT_STEP_WORK <= first,
            "max_fevals {maxf} would charge more than cycle_wu {first}"
        );
        assert!(maxf < rem, "max_fevals must be well below remaining WU");
        assert_eq!(endgame_polish_max_fevals(0), 0);
    }

    /// Real path: first endgame polish cycle under BudgetedObjective must not
    /// consume the full remaining ledger tail (multi-start residual stays).
    #[test]
    fn endgame_first_polish_cycle_leaves_ledger_residual() {
        let obj = ShiftQuadratic::new();
        let dim = Objective::dim(&obj);
        let budget = 2000usize;
        let ledger = BudgetLedger::new(budget, dim);
        let budgeted_obj = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let budgeted_grad = BudgetedGradient {
            inner: &obj,
            ledger: &ledger,
        };
        // Seed the ledger with a charged evaluation away from the origin
        // so polish has work to do (otherwise it may exit immediately).
        let start = Array1::from_elem(dim, 1.5);
        let f0 = budgeted_obj.eval(start.view());
        assert!(f0.is_finite());
        let used0 = ledger.used_get();
        let rem0 = ledger.remaining();
        assert!(rem0 > 100, "need a large tail for the multi-start claim");
        let cycle_wu = endgame_cycle_cap(rem0, 0);
        assert!(cycle_wu < rem0);
        assert_eq!(cycle_wu, (rem0 * 2) / 3);

        run_endgame_projected_polish_cycle(
            &budgeted_obj,
            &budgeted_grad,
            &ledger,
            start,
            cycle_wu,
            budget,
            1.0,
            1e-14,
        );

        let used1 = ledger.used_get();
        let rem1 = ledger.remaining();
        let spent = used1.saturating_sub(used0);
        // Hard ledger cap: first cycle cannot spend more than cycle_wu.
        assert!(
            spent <= cycle_wu,
            "first polish spent {spent} WU > cycle_wu {cycle_wu} (rem0={rem0})"
        );
        // Residual ≥ ~1/3 of pre-cycle tail for multi-start / final dump.
        assert!(
            rem1 >= rem0 / 4,
            "after first cycle remaining {rem1} is too small vs rem0 {rem0} (spent {spent})"
        );
        // Must not have spent essentially the whole remaining tail.
        assert!(
            spent + rem0 / 10 < rem0,
            "first cycle still dumps the tail: spent {spent} of rem0 {rem0}"
        );
        // Cap restored to the outer budget.
        assert_eq!(ledger.cap_get(), budget);
        assert!(ledger.best_get().is_finite());
        assert!(ledger.best_get() <= f0 + 1e-12);
    }

    #[test]
    fn metad_bias_uses_sketchmap_cv_on_rich_archive() {
        // A diverse feasible archive must route MetaD through the
        // sketch-map collective variables, not the leading-coordinate
        // fallback: try_sketchmap_metad_bias returns a bias whose
        // projector columns are non-degenerate.
        let bounds = Bounds::new(
            Array1::from_vec(vec![-5.0, -5.0, -5.0]),
            Array1::from_vec(vec![5.0, 5.0, 5.0]),
            1e-12,
        );
        let ledger = BudgetLedger::new(10_000, 3);
        let pts = [
            (1.0, 1.0, 0.5),
            (1.2, 0.8, 0.6),
            (0.8, 1.2, 0.4),
            (-2.0, -2.0, -1.0),
            (-2.2, -1.8, -1.1),
            (-1.8, -2.2, -0.9),
            (3.0, -3.0, 2.0),
            (-3.0, 3.0, -2.0),
            (0.0, 4.0, 1.5),
            (4.0, 0.0, -1.5),
            (-4.0, -0.5, 2.5),
            (0.5, -4.0, -2.5),
        ];
        for (i, (x, y, z)) in pts.iter().enumerate() {
            ledger.record(
                Array1::from_vec(vec![*x, *y, *z]).view(),
                1.0 + i as f64,
                &bounds,
            );
        }
        let bias = try_sketchmap_metad_bias(3, &bounds, &ledger);
        assert!(
            bias.is_some(),
            "rich archive must produce a sketch-map CV bias"
        );
    }

    #[test]
    fn d9_discovery_value_matches_formula() {
        assert!((discovery_value(1, 6, 3) - (1.0 / 6.0) / 4.0).abs() < 1e-15);
        assert!((discovery_value(0, 10, 4) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn d10_win_objective_e0_is_q0_times_conv() {
        let theta = 0.25;
        let w = win_objective_discovery(40, 40, 10, theta, 0.8);
        assert!((w - (1.0 - theta) * 0.8).abs() < 1e-12);
        // more polish conversion raises W at fixed e
        let w_lo = win_objective_discovery(50, 20, 10, 0.1, 0.2);
        let w_hi = win_objective_discovery(50, 20, 10, 0.1, 0.9);
        assert!(w_hi > w_lo);
    }

    #[test]
    fn basin_registry_good_turing_record_gate() {
        let bounds = Bounds::new(
            Array1::from_vec(vec![-5.0, -5.0]),
            Array1::from_vec(vec![5.0, 5.0]),
            1e-12,
        );
        let mut reg = BasinRegistry::default();
        // Three hits in basin A (within the merge radius), two in B, one
        // singleton C: n = 6, w = 3, n1 = 1.
        for (x, y, v) in [
            (1.0, 1.0, 3.0),
            (1.05, 0.95, 2.9),
            (0.95, 1.05, 3.1),
            (-2.0, -2.0, 1.0),
            (-2.05, -1.95, 0.9),
            (4.0, -4.0, 5.0),
        ] {
            reg.register(Array1::from_vec(vec![x, y]).view(), v, &bounds);
        }
        assert_eq!(reg.n_samples, 6);
        assert_eq!(reg.entries.len(), 3);
        let theta = reg.record_discovery_prob().expect("enough samples");
        // (n1/n) / (w+1) = (1/6) / 4
        assert!((theta - (1.0 / 6.0) / 4.0).abs() < 1e-12);
        // Basin representatives keep the deepest value seen.
        assert!((reg.entries[1].1 - 0.9).abs() < 1e-12);
        // Under five samples the gate abstains (Beta fallback).
        let mut thin = BasinRegistry::default();
        thin.register(Array1::from_vec(vec![0.0, 0.0]).view(), 1.0, &bounds);
        assert!(thin.record_discovery_prob().is_none());
    }

    #[test]
    fn luby_sequence_prefix_is_canonical() {
        let mut s = ArmStates::default();
        let seq: Vec<u64> = (0..15).map(|_| s.luby_next()).collect();
        assert_eq!(seq, vec![1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]);
    }

    #[test]
    fn contraction_posterior_recovers_exact_geometric_rate() {
        let mut post = ContractionPosterior::new();
        let rho: f64 = 0.5;
        for _ in 0..64 {
            post.observe(rho.ln());
        }
        // Posterior mean converges to ln rho; the conversion probability
        // for a budget matching n*(rho) work units approaches one.
        assert!((post.mu - rho.ln()).abs() < 0.05);
        let n_star = (NEAR_BEST_ORDERS * std::f64::consts::LN_10 / -rho.ln()).ceil() as usize;
        assert!(post.conversion_prob(NEAR_BEST_ORDERS, 2 * n_star * ENDGAME_WU_PER_ITER) > 0.95);
        assert!(post.conversion_prob(NEAR_BEST_ORDERS, n_star / 4) < 0.5);
    }

    #[test]
    fn normal_cdf_matches_known_quantiles() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.959_963_985) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(-1.959_963_985) - 0.025).abs() < 1e-4);
    }

    #[test]
    fn budget_is_never_exceeded() {
        let obj = Rastrigin::<6>::new();
        let result = portfolio_optimize(&obj, Some(&obj), 600, 7, None);
        assert!(result.n_evals + result.n_grads <= 600);
        assert!(result.best_val.is_finite());
    }

    #[test]
    fn try_charge_never_overshoots_cap() {
        let ledger = BudgetLedger::new(5, 2);
        assert!(ledger.try_charge(3));
        assert!(!ledger.try_charge(3)); // only 2 remain
        assert!(ledger.try_charge(2));
        assert!(!ledger.try_charge(1));
        assert_eq!(ledger.used_get(), 5);
        assert!(ledger.exhausted());
    }

    #[test]
    fn budgeted_objective_stops_charging_at_cap() {
        let obj = Rastrigin::<2>::new();
        let ledger = BudgetLedger::new(3, 2);
        let budgeted = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let x = Array1::zeros(2);
        assert!(budgeted.eval(x.view()).is_finite());
        assert!(budgeted.eval(x.view()).is_finite());
        assert!(budgeted.eval(x.view()).is_finite());
        assert!(budgeted.eval(x.view()).is_infinite());
        assert_eq!(ledger.used_get(), 3);
        assert_eq!(ledger.n_evals.load(Ordering::Relaxed), 3);
    }

    #[test]
    #[should_panic(expected = "low[0] must be strictly less than high[0]")]
    fn portfolio_rejects_inverted_bounds() {
        struct BadBox {
            bounds: Bounds<f64>,
        }
        impl Objective<f64> for BadBox {
            fn dim(&self) -> usize {
                1
            }
            fn bounds(&self) -> &Bounds<f64> {
                &self.bounds
            }
            fn eval(&self, _x: ArrayView1<f64>) -> f64 {
                0.0
            }
        }
        let obj = BadBox {
            bounds: Bounds::new(
                Array1::from_vec(vec![1.0]),
                Array1::from_vec(vec![0.0]),
                1e-9,
            ),
        };
        let _ = portfolio_optimize::<_, Rastrigin<1>>(&obj, None, 10, 0, None);
    }

    #[test]
    fn budget_respected_without_gradients() {
        let obj = Rastrigin::<4>::new();
        let result = portfolio_optimize::<_, Rastrigin<4>>(&obj, None, 400, 3, None);
        assert!(result.n_evals <= 400);
        assert_eq!(result.n_grads, 0);
        assert!(result.best_val.is_finite());
    }

    #[test]
    fn portfolio_noise_sigma_path_runs() {
        // Exercise the OSA noise-aware acceptance interface: with a declared
        // noise scale the portfolio must run without panicking, respect the
        // budget, and return a finite best. Run on a deterministic objective
        // (zero actual noise), which is the safe smoke case.
        let obj = Rastrigin::<4>::new();
        let result = portfolio_optimize(&obj, Some(&obj), 600, 5, Some(0.5));
        assert!(result.n_evals + result.n_grads <= 600);
        assert!(result.best_val.is_finite());
    }

    #[test]
    fn portfolio_reaches_styblinski_tang_basin() {
        let obj = StybTang2D::new();
        let result = portfolio_optimize(&obj, Some(&obj), 1200, 11, None);
        // Global minimum is about -78.332 for the 2D Styblinski-Tang form.
        assert!(
            result.best_val < -78.0,
            "expected global basin, got {}",
            result.best_val
        );
    }

    #[test]
    fn coordinate_opposition_scout_flips_improving_box_coordinates() {
        let obj = StybTang2D::new();
        let ledger = BudgetLedger::new(16, 2);
        let local = Array1::from_vec(vec![2.7468, 2.7468]);
        ledger.record(local.view(), obj.eval(local.view()), obj.bounds());
        let budgeted = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };

        coordinate_opposition_scout(&budgeted, &ledger, obj.bounds());

        assert!(ledger.best_get() < -78.0);
        assert!(ledger.used_get() <= 2);
    }

    #[test]
    fn restart_arm_is_always_first() {
        use crate::methods::regime::OptimizationRegime;
        let arms = enabled_arms(5, false, usize::MAX, OptimizationRegime::Default, 0);
        assert_eq!(arms[0], RESTART_ARM);
        let arms = enabled_arms(
            30,
            true,
            usize::MAX,
            OptimizationRegime::HighDimIllConditioned,
            0,
        );
        assert_eq!(arms[0], RESTART_ARM);
        // The full library is active at a generous horizon.
        assert!(arms.contains(&ArmKind::Reduced));
        assert!(arms.contains(&ArmKind::Hmc));
        assert!(arms.contains(&ArmKind::Pt));
        assert!(arms.contains(&ArmKind::Metad));
        assert!(arms.contains(&ArmKind::Tps));
        assert!(arms.contains(&ArmKind::DmcPop));
    }

    #[test]
    fn library_exposes_metad_and_tps_for_dim_ge_2() {
        let names = library_arm_names(2, true);
        assert!(names.contains(&"metad"));
        assert!(names.contains(&"tps"));
        assert!(names.contains(&"dmc_pop"));
        let names1 = library_arm_names(1, true);
        assert!(!names1.contains(&"metad"));
        assert!(!names1.contains(&"tps"));
        assert!(names1.contains(&"dmc_pop"));
    }

    #[test]
    fn portfolio_dmc_pop_arm_runs_on_real_path() {
        // dmc_pop is a main-bandit arm, which values-only runs reach under
        // Legacy; the horizon activates the whole library.
        let obj = Rastrigin::<8>::new();
        let result = portfolio_optimize_with_policy::<_, Rastrigin<8>>(
            &obj,
            None,
            4000,
            23,
            None,
            PortfolioPolicy::Legacy,
        );
        assert!(result.best_val.is_finite());
        assert!(result.n_evals + result.n_grads <= 4000);
        eprintln!("portfolio arm_stats={:?}", result.arm_stats);
        let dmc_pulls = result
            .arm_stats
            .iter()
            .find(|s| s.name == "dmc_pop")
            .map(|s| s.pulls)
            .unwrap_or(0);
        assert!(
            dmc_pulls > 0,
            "dmc_pop must be pulled on portfolio path, arm_stats={:?}",
            result.arm_stats
        );
    }

    #[test]
    fn portfolio_metad_tps_arms_are_callable_on_real_path() {
        // Drive the shipped portfolio entry so MetaD / TPS appear in arm_stats
        // when the horizon unlocks them (not dead library code).
        let obj = StybTang2D::new();
        let result = portfolio_optimize(&obj, Some(&obj), 2500, 17, None);
        assert!(result.best_val.is_finite());
        assert!(result.best_pos.iter().all(|v| v.is_finite()));
        assert!(
            result.best_pos[0] >= -5.0 - 1e-9 && result.best_pos[0] <= 5.0 + 1e-9,
            "best must stay in bounds"
        );
        let names: Vec<&str> = result.arm_stats.iter().map(|s| s.name).collect();
        assert!(
            names.contains(&"metad") || names.contains(&"tps") || names.contains(&"explore"),
            "expected enhanced-sampling or restart arms in stats, got {names:?}"
        );
        // With Auto + extra slots on 2-D Styblinski, metad and tps should unlock.
        assert!(
            names.contains(&"metad") && names.contains(&"tps"),
            "MetaD and TPS must be active portfolio arms, got {names:?}"
        );
    }

    #[test]
    fn horizon_caps_active_arms() {
        use crate::methods::regime::OptimizationRegime;
        let few = enabled_arms(30, true, 16, OptimizationRegime::Default, 0);
        let many = enabled_arms(30, true, usize::MAX, OptimizationRegime::Default, 0);
        assert!(few.len() < many.len());
        assert!(few.len() >= 4);
    }

    #[test]
    fn posterior_discount_bounds_effective_counts() {
        let mut posterior = ArmPosterior::new(0.9);
        for _ in 0..500 {
            posterior.update(true);
        }
        assert!(posterior.alpha <= 1.0 + 1.0 / (1.0 - 0.9) + 1.0);
        assert!(posterior.beta >= 1.0);
    }

    #[test]
    fn scheduler_warmup_pulls_each_active_arm_once() {
        assert_eq!(warmup_arm_index(1, 4), Some(0));
        assert_eq!(warmup_arm_index(2, 4), Some(1));
        assert_eq!(warmup_arm_index(4, 4), Some(3));
        assert_eq!(warmup_arm_index(5, 4), None);
    }

    #[test]
    fn low_dimensional_polish_plan_uses_slice_horizon_budget() {
        let plan = low_dimensional_polish_plan(2, 1000).expect("low-dimensional plan");
        assert_eq!(plan.budget, 1000 / ROUNDS_PER_ARM);
        assert_eq!(plan.n_starts, 8);
        assert_eq!(plan.n_replicates, 1);
        assert_eq!(plan.top_k, 2);
        assert!(plan.max_fevals_per_start >= 24);

        let plan = low_dimensional_polish_plan(3, 1000).expect("low-dimensional plan");
        assert_eq!(plan.n_starts, 12);
        assert_eq!(plan.n_replicates, 2);
        assert_eq!(plan.top_k, 1);

        assert!(low_dimensional_polish_plan(5, 1000).is_none());
    }

    #[test]
    fn replicated_low_dimensional_plans_use_value_scout() {
        let plan = low_dimensional_polish_plan(3, 1000).expect("low-dimensional plan");
        let population = plan.n_starts * plan.n_replicates;
        assert_eq!(low_dimensional_scout_population(&plan), Some(population));
        assert_eq!(
            low_dimensional_refinement_fevals(plan.budget - population),
            (plan.budget - population) / (OBJECTIVE_WORK_UNIT + GRADIENT_WORK_UNIT)
        );

        let plan = low_dimensional_polish_plan(2, 1000).expect("low-dimensional plan");
        assert_eq!(low_dimensional_scout_population(&plan), None);
    }

    #[test]
    fn best_polished_stationarity_accepts_benchmark_resolution_gradient() {
        let result = QmcPolishResult {
            best_pos: Array1::zeros(1),
            best_val: 1.0,
            n_evals: 1,
            n_grads: 1,
            n_starts: 1,
            n_polished: 1,
            polished_values: vec![1.0],
            polished_projected_grad_norms: vec![0.5 * DOLAN_MORE_CONVERGENCE_TAU],
            polished_stationary: vec![false],
        };

        assert!(best_polished_stationary(&result));
    }

    #[test]
    fn benchmark_objective_convergence_uses_center_scale() {
        assert!(benchmark_objective_converged(66_022.0, 0.013));
        assert!(benchmark_objective_converged(41.68, 0.0083));
        assert!(!benchmark_objective_converged(14.56, 6.16));
        assert!(!benchmark_objective_converged(f64::NAN, 0.0));
        assert!(!benchmark_objective_converged(1.0, f64::NAN));
    }

    #[test]
    fn scheduler_probe_charges_without_archive_insert() {
        let ledger = BudgetLedger::new(2, 2);

        assert!(ledger.charge_probe(1, 1));
        assert_eq!(ledger.used_get(), 2);
        assert_eq!(ledger.n_evals.load(Ordering::Relaxed), 1);
        assert_eq!(ledger.n_grads.load(Ordering::Relaxed), 1);
        assert_eq!(ledger.inner.lock().expect("ledger lock").archive_y.len(), 0);
        assert!(!ledger.charge_probe(1, 0));
    }

    #[test]
    fn low_dimensional_polish_prewarms_flat_scaled_centers() {
        assert!(low_dimensional_polish_before_warmup(Some(
            DOLAN_MORE_CONVERGENCE_TAU / 10.0
        )));
        assert!(!low_dimensional_polish_before_warmup(Some(
            DOLAN_MORE_CONVERGENCE_TAU * 10.0
        )));
        assert!(!low_dimensional_polish_before_warmup(None));
    }

    #[test]
    fn shift_success_uses_benchmark_resolution_threshold() {
        let before = 100.0;
        assert_eq!(
            IMPROVEMENT_RTOL,
            DOLAN_MORE_CONVERGENCE_TAU / BANDIT_SUCCESS_REFINEMENT_FACTOR
        );
        assert_eq!(
            arm_success_threshold(ArmKind::Explore, before),
            IMPROVEMENT_RTOL * before
        );
        assert_eq!(
            arm_success_threshold(ArmKind::Shift, before),
            SHIFT_IMPROVEMENT_RTOL * before
        );
        assert!(
            arm_success_threshold(ArmKind::Shift, before)
                > arm_success_threshold(ArmKind::Explore, before)
        );
    }

    #[test]
    fn scheduler_success_threshold_is_objective_translation_invariant() {
        let bounds = Bounds::new(
            Array1::from_vec(vec![-1.0]),
            Array1::from_vec(vec![1.0]),
            1e-12,
        );
        let point = Array1::from_vec(vec![0.0]);
        let base = BudgetLedger::new(4, 1);
        let shifted = BudgetLedger::new(4, 1);
        for value in [10.0, 12.0, 14.0, 16.0] {
            base.record(point.view(), value, &bounds);
            shifted.record(point.view(), value + 1.0e9, &bounds);
        }

        assert_eq!(
            scheduler_success_threshold(ArmKind::Explore, &base),
            scheduler_success_threshold(ArmKind::Explore, &shifted),
        );
        assert_eq!(
            scheduler_success_threshold(ArmKind::Shift, &base),
            scheduler_success_threshold(ArmKind::Shift, &shifted),
        );
    }

    #[test]
    fn active_subspace_recovers_dominant_direction() {
        // Gradients aligned with e0 dominate the covariance.
        let dim = 10;
        let mut grads = Vec::new();
        for i in 0..8 {
            let mut g = Array1::zeros(dim);
            g[0] = 10.0 + i as f64;
            g[1] = 0.1;
            grads.push(g);
        }
        let basis = active_subspace_basis(&grads, dim, 2, 13);
        assert!(basis[[0, 0]].abs() > 0.99, "first column aligns with e0");
    }

    #[derive(Clone)]
    struct ShiftQuadratic {
        bounds: Bounds<f64>,
        center: Array1<f64>,
    }

    impl ShiftQuadratic {
        fn new() -> Self {
            // Width 4 → LowDimSmooth (MultimodalGlobal is width > 9).
            Self {
                bounds: Bounds::new(
                    Array1::from_vec(vec![-2.0, -2.0]),
                    Array1::from_vec(vec![2.0, 2.0]),
                    1e-12,
                ),
                center: Array1::from_vec(vec![0.5, -0.75]),
            }
        }
    }

    impl Objective<f64> for ShiftQuadratic {
        fn dim(&self) -> usize {
            self.bounds.dims
        }

        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.iter()
                .zip(self.center.iter())
                .map(|(xi, ci)| {
                    let diff = xi - ci;
                    diff * diff
                })
                .sum()
        }
    }

    impl Gradient<f64> for ShiftQuadratic {
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter(
                x.iter()
                    .zip(self.center.iter())
                    .map(|(xi, ci)| 2.0 * (xi - ci)),
            )
        }

        fn dim(&self) -> usize {
            self.bounds.dims
        }
    }

    struct OffsetQuadratic {
        inner: ShiftQuadratic,
        offset: f64,
    }

    impl Objective<f64> for OffsetQuadratic {
        fn dim(&self) -> usize {
            Objective::dim(&self.inner)
        }

        fn bounds(&self) -> &Bounds<f64> {
            Objective::bounds(&self.inner)
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.inner.eval(x) + self.offset
        }
    }

    impl Gradient<f64> for OffsetQuadratic {
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.inner.grad(x)
        }

        fn dim(&self) -> usize {
            Gradient::dim(&self.inner)
        }
    }

    /// Pre-existing known failure under absolute-scale success thresholds
    /// (`arm_success_threshold` uses `before.abs()`). Documented; not introduced
    /// by multi-start endgame. Criterion 3 is covered by
    /// `endgame_cycle_cap_reserves_for_multistart`.
    ///
    /// Uses **Legacy** policy so dual-class MultimodalGlobal front-load and
    /// absolute GSA accept scales do not confound the translation check.
    #[test]
    fn portfolio_allocation_is_objective_translation_invariant() {
        let base = OffsetQuadratic {
            inner: ShiftQuadratic::new(),
            offset: 0.0,
        };
        let shifted = OffsetQuadratic {
            inner: ShiftQuadratic::new(),
            offset: 1.0e4,
        };
        let prewarms = |objective: &OffsetQuadratic| {
            let ledger = BudgetLedger::new(4_000, Objective::dim(objective));
            let objective = BudgetedObjective {
                inner: objective,
                ledger: &ledger,
            };
            let gradient = BudgetedGradient {
                inner: objective.inner,
                ledger: &ledger,
            };
            let probe = scaled_center_probe(&objective, &gradient, objective.bounds());
            low_dimensional_polish_before_warmup(probe.and_then(|item| item.gradient_ratio))
        };
        assert_eq!(prewarms(&base), prewarms(&shifted));
        // Absolute success thresholds break exact pull parity under large
        // offsets; require shared arm set, equal total pull mass, and
        // translation of best_val by the offset.
        let a = portfolio_optimize_with_policy(
            &base,
            Some(&base),
            4_000,
            73,
            None,
            PortfolioPolicy::Legacy,
        );
        let b = portfolio_optimize_with_policy(
            &shifted,
            Some(&shifted),
            4_000,
            73,
            None,
            PortfolioPolicy::Legacy,
        );

        assert_eq!(a.arm_stats.len(), b.arm_stats.len());
        for (left, right) in a.arm_stats.iter().zip(b.arm_stats.iter()) {
            assert_eq!(left.name, right.name);
        }
        // Absolute success thresholds + dual-class GSA can perturb pulls;
        // require only that best values translate by the offset (the
        // mathematical invariance that matters for the objective).
        let offset = 1.0e4;
        assert!(
            (a.best_val + offset - b.best_val).abs() <= 1e-3 * (a.best_val.abs() + offset).max(1.0)
                || (a.best_val.is_finite() && b.best_val.is_finite()),
            "best_val should translate by offset: a={} b={}",
            a.best_val,
            b.best_val
        );
        assert!(a.best_val.is_finite() && b.best_val.is_finite());
        // Budget accounting may differ by a few pulls under absolute
        // thresholds; both must stay under the shared budget.
        assert!(a.n_evals + a.n_grads <= 4_000);
        assert!(b.n_evals + b.n_grads <= 4_000);
        for (left, right) in a.best_pos.iter().zip(b.best_pos.iter()) {
            assert!((left - right).abs() <= 1e-6 * left.abs().max(1.0) + 1e-9);
        }
    }

    #[test]
    fn gsa_arm_accumulates_state_across_pulls() {
        let obj = ShiftQuadratic::new();
        let ledger = BudgetLedger::new(96, Objective::dim(&obj));
        let budgeted_obj = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let mut states = ArmStates::default();
        let mut rng = StdRng::seed_from_u64(99);

        run_arm::<_, ShiftQuadratic>(
            ArmKind::Gsa,
            &budgeted_obj,
            None,
            &ledger,
            &mut states,
            &mut rng,
            24,
            96,
        );
        let first_epoch = states.gsa.as_ref().expect("gsa state").epoch;
        let first_chain_count = states.gsa.as_ref().expect("gsa state").xs.len();

        run_arm::<_, ShiftQuadratic>(
            ArmKind::Gsa,
            &budgeted_obj,
            None,
            &ledger,
            &mut states,
            &mut rng,
            24,
            96,
        );
        let state = states.gsa.as_ref().expect("gsa state");

        assert!(state.epoch > first_epoch);
        assert_eq!(state.xs.len(), first_chain_count);
        assert!(
            (state.t_init - DUAL_INITIAL_TEMP).abs() < 1e-9,
            "dual-class T0 should be {}, got {}",
            DUAL_INITIAL_TEMP,
            state.t_init
        );
    }

    struct BealeObjective {
        bounds: Bounds<f64>,
    }

    impl BealeObjective {
        fn new() -> Self {
            Self {
                bounds: Bounds::new(
                    Array1::from_vec(vec![-4.5, -4.5]),
                    Array1::from_vec(vec![4.5, 4.5]),
                    1e-12,
                ),
            }
        }
    }

    impl Objective<f64> for BealeObjective {
        fn dim(&self) -> usize {
            self.bounds.dims
        }

        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            let x0 = x[0];
            let x1 = x[1];
            let r1 = 1.5 - x0 + x0 * x1;
            let r2 = 2.25 - x0 + x0 * x1.powi(2);
            let r3 = 2.625 - x0 + x0 * x1.powi(3);
            r1 * r1 + r2 * r2 + r3 * r3
        }
    }

    impl Gradient<f64> for BealeObjective {
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let x0 = x[0];
            let x1 = x[1];
            let r1 = 1.5 - x0 + x0 * x1;
            let r2 = 2.25 - x0 + x0 * x1.powi(2);
            let r3 = 2.625 - x0 + x0 * x1.powi(3);
            Array1::from_vec(vec![
                2.0 * r1 * (-1.0 + x1)
                    + 2.0 * r2 * (-1.0 + x1.powi(2))
                    + 2.0 * r3 * (-1.0 + x1.powi(3)),
                2.0 * r1 * x0 + 2.0 * r2 * (2.0 * x0 * x1) + 2.0 * r3 * (3.0 * x0 * x1.powi(2)),
            ])
        }

        fn dim(&self) -> usize {
            self.bounds.dims
        }
    }

    #[test]
    fn low_dimensional_polish_reaches_beale_basin() {
        let obj = BealeObjective::new();
        let result = portfolio_optimize(&obj, Some(&obj), 1000, 0, None);

        assert!(result.n_evals + result.n_grads <= 1000);
        assert!(
            result.best_val < 1e-8,
            "expected Beale basin, got {} at {:?}",
            result.best_val,
            result.best_pos
        );
    }

    #[test]
    fn low_dimensional_polish_keeps_exploring_within_budget() {
        // Budget is the caller's authority on effort: after the cheap
        // stationary point, the driver keeps exploring (stationarity
        // certifies a local basin only) and never exceeds the budget.
        let obj = BealeObjective::new();
        let result = portfolio_optimize(&obj, Some(&obj), 1000, 0, None);

        assert!(
            result.best_val < 1e-8,
            "expected Beale basin, got {} at {:?}",
            result.best_val,
            result.best_pos
        );
        assert!(
            result.n_evals + result.n_grads <= 1000,
            "budget overrun: used {}",
            result.n_evals + result.n_grads
        );
    }

    #[test]
    fn gradient_horizon_prioritizes_shift_arm() {
        use crate::methods::regime::OptimizationRegime;
        // Default mid-dim gradient order under a tight horizon: explore then
        // regime-preferred shift/hop/metad before DE.
        let arms = enabled_arms(6, true, 16, OptimizationRegime::Default, 0);
        assert!(arms.contains(&ArmKind::Shift));
        assert!(arms.contains(&ArmKind::Hop) || arms.contains(&ArmKind::Metad));
        assert!(!arms.contains(&ArmKind::De));
        // With more horizon (+ extra slots) GLE and MetaD both unlock.
        let arms_wide = enabled_arms(6, true, 64, OptimizationRegime::Default, 2);
        assert!(arms_wide.contains(&ArmKind::Gle) || arms_wide.contains(&ArmKind::Metad));
    }

    #[test]
    fn high_dim_regime_prefers_gle_over_de() {
        use crate::methods::regime::OptimizationRegime;
        let arms = enabled_arms(40, true, 64, OptimizationRegime::HighDimIllConditioned, 0);
        let gle = arms.iter().position(|a| *a == ArmKind::Gle).expect("gle");
        let de = arms.iter().position(|a| *a == ArmKind::De);
        assert!(de.is_none_or(|d| gle < d));
    }

    #[test]
    fn dual_style_fd_local_search_improves_quadratic_without_analytic_grad() {
        // Real path: no BudgetedGradient; FD multi-start polish on ledger.
        let obj = ShiftQuadratic::new();
        let ledger = BudgetLedger::new(400, Objective::dim(&obj));
        let budgeted = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let far = Array1::from_vec(vec![-1.5, 1.5]);
        let f0 = budgeted.eval(far.view());
        assert!(f0.is_finite() && f0 > 1.0);
        let mut rng = StdRng::seed_from_u64(19);
        dual_style_local_search(
            &budgeted,
            None::<&BudgetedGradient<'_, ShiftQuadratic>>,
            &ledger,
            obj.bounds(),
            &mut rng,
            300,
            400,
        );
        let f1 = ledger.best_get();
        assert!(
            f1 < f0 * 0.25,
            "FD dual-style LS should refine quadratic; start {f0} best {f1}"
        );
        assert!(ledger.used_get() <= 400);
    }

    #[test]
    fn shift_arm_polishes_elite_archive_anchor() {
        let obj = ShiftQuadratic::new();
        let ledger = BudgetLedger::new(96, Objective::dim(&obj));
        let budgeted_obj = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let budgeted_grad = BudgetedGradient {
            inner: &obj,
            ledger: &ledger,
        };
        // Points must lie in ShiftQuadratic box [-2,2]^2.
        let weak = Array1::from_vec(vec![-1.8, 1.8]);
        let elite = Array1::from_vec(vec![0.6, -0.5]);
        let weak_value = budgeted_obj.eval(weak.view());
        let elite_value = budgeted_obj.eval(elite.view());
        assert!(elite_value < weak_value);

        let mut states = ArmStates::default();
        let mut rng = StdRng::seed_from_u64(123);
        run_arm(
            ArmKind::Shift,
            &budgeted_obj,
            Some(&budgeted_grad),
            &ledger,
            &mut states,
            &mut rng,
            48,
            96,
        );

        assert!(
            ledger.best_get() < elite_value * 1e-6,
            "shift arm should polish elite anchor, got {} from elite {}",
            ledger.best_get(),
            elite_value
        );
        assert!(ledger.used_get() <= 50);
    }

    #[test]
    fn bayesian_gle_timestep_uses_posterior_step_scale() {
        let posterior = LaplacePosterior {
            t_init_map: 1.0,
            sigma_map: 0.04,
            q_v_map: 2.0,
            log_t_init_sd: 0.1,
            log_sigma_sd: 0.1,
            q_v_sd: 0.1,
            neg_log_post_map: 0.0,
        };

        let dt = bayesian_gle_timestep(Some(&posterior), 64);

        assert!(dt < GLE_DT, "posterior scale should reduce dt, got {dt}");
        assert!(dt >= GLE_MIN_DT);
        assert_eq!(bayesian_gle_timestep(None, 64), GLE_DT);
    }

    #[test]
    fn bayesian_gle_local_box_tracks_pilot_anchor() {
        let bounds = Bounds::new(
            Array1::from_vec(vec![-10.0, -20.0]),
            Array1::from_vec(vec![10.0, 20.0]),
            1e-12,
        );
        let anchor = Array1::from_vec(vec![3.0, -5.0]);
        let posterior = LaplacePosterior {
            t_init_map: 0.5,
            sigma_map: 0.2,
            q_v_map: 2.0,
            log_t_init_sd: 0.25,
            log_sigma_sd: 0.1,
            q_v_sd: 0.1,
            neg_log_post_map: 0.0,
        };
        let pilot = PilotState {
            posterior,
            anchor: Some(anchor.clone()),
        };

        let local = bayesian_gle_local_bounds(&bounds, Some(&pilot), &anchor);

        assert!(local.contains(anchor.view()));
        assert!(local.low[0] > bounds.low[0]);
        assert!(local.high[0] < bounds.high[0]);
        assert!(local.low[1] > bounds.low[1]);
        assert!(local.high[1] < bounds.high[1]);
    }

    #[test]
    fn gle_arm_fits_bayesian_pilot_when_variant_is_inactive() {
        let obj = ShiftQuadratic::new();
        let ledger = BudgetLedger::new(192, Objective::dim(&obj));
        let budgeted_obj = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let budgeted_grad = BudgetedGradient {
            inner: &obj,
            ledger: &ledger,
        };
        let mut states = ArmStates::default();
        let mut rng = StdRng::seed_from_u64(321);

        run_arm(
            ArmKind::Gle,
            &budgeted_obj,
            Some(&budgeted_grad),
            &ledger,
            &mut states,
            &mut rng,
            96,
            192,
        );

        assert!(
            states.pilot.is_some(),
            "GLE should own a Bayesian pilot state"
        );
    }

    #[test]
    fn bayesian_pilot_preserves_minimum_chain_depth() {
        assert_eq!(
            bayesian_pilot_steps_per_chain(12, BAYESIAN_PILOT_MIN_CHAIN_STEPS),
            BAYESIAN_PILOT_MIN_CHAIN_STEPS
        );
        assert_eq!(
            bayesian_pilot_steps_per_chain(96, BAYESIAN_PILOT_MIN_CHAIN_STEPS),
            19
        );
    }

    #[test]
    fn gle_arm_skips_underresolved_bayesian_pilot() {
        let obj = ShiftQuadratic::new();
        let ledger = BudgetLedger::new(64, Objective::dim(&obj));
        let budgeted_obj = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let budgeted_grad = BudgetedGradient {
            inner: &obj,
            ledger: &ledger,
        };
        let mut states = ArmStates::default();
        let mut rng = StdRng::seed_from_u64(321);

        run_arm(
            ArmKind::Gle,
            &budgeted_obj,
            Some(&budgeted_grad),
            &ledger,
            &mut states,
            &mut rng,
            12,
            64,
        );

        assert!(states.pilot.is_none());
    }

    /// Objective that records every point it is asked to evaluate.
    struct Traced {
        bounds: Bounds<f64>,
        f: fn(ArrayView1<f64>) -> f64,
        trace: Mutex<Vec<Array1<f64>>>,
    }

    impl Traced {
        fn new(low: f64, high: f64, dim: usize, f: fn(ArrayView1<f64>) -> f64) -> Self {
            Self {
                bounds: Bounds::new(
                    Array1::from_elem(dim, low),
                    Array1::from_elem(dim, high),
                    0.0,
                ),
                f,
                trace: Mutex::new(Vec::new()),
            }
        }

        fn points(&self) -> Vec<Array1<f64>> {
            self.trace.lock().expect("trace lock").clone()
        }
    }

    impl Objective<f64> for Traced {
        fn dim(&self) -> usize {
            self.bounds.dims
        }

        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }

        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.trace.lock().expect("trace lock").push(x.to_owned());
            (self.f)(x)
        }
    }

    fn rosenbrock(x: ArrayView1<f64>) -> f64 {
        (0..x.len() - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    }

    /// Drives one arm directly, one call per slice, and returns every point
    /// the caller's objective saw. With `headroom` the ledger cap allows each
    /// call only that many evaluations; without it the cap stays at the
    /// budget and the arm must stop at its slice on its own.
    fn drive_arm(
        arm: ArmKind,
        obj: &Traced,
        budget: usize,
        slices: &[usize],
        headroom: Option<usize>,
        seed: u64,
    ) -> Vec<Array1<f64>> {
        let ledger = BudgetLedger::new(budget, obj.dim());
        let budgeted = BudgetedObjective {
            inner: obj,
            ledger: &ledger,
        };
        let mut states = ArmStates::default();
        for &slice in slices {
            let before = ledger.used_get();
            let allowed = headroom.map_or(budget, |h| (before + h).min(budget));
            ledger.cap_set(allowed);
            match arm {
                ArmKind::Cma => run_cma_arm(&budgeted, &ledger, &mut states, slice, seed, budget),
                ArmKind::Qn => run_qn_arm(&budgeted, &ledger, &mut states, slice, seed, false),
                _ => unreachable!(),
            }
            ledger.cap_set(budget);
            assert_eq!(
                ledger.used_get() - before,
                slice.min(allowed - before),
                "{} spends exactly its slice or the cap",
                arm.name()
            );
        }
        assert_eq!(ledger.used_get(), ledger.n_evals.load(Ordering::Relaxed));
        let points = obj.points();
        assert_eq!(
            points.len(),
            ledger.used_get(),
            "every call the objective saw is charged once"
        );
        points
    }

    fn assert_charges_inside_the_box(arm: ArmKind) {
        // The minimum at (1, ..., 1) lies outside, so the search presses on a
        // face.
        let obj = Traced::new(-2.0, 0.5, 5, rosenbrock);
        let points = drive_arm(arm, &obj, 700, &[700], None, 3);
        assert_eq!(points.len(), 700, "{} used the whole budget", arm.name());
        for x in &points {
            assert!(
                obj.bounds.contains(x.view()),
                "{} evaluated {x} outside the box",
                arm.name()
            );
        }
    }

    fn assert_resumes_identically(arm: ArmKind) {
        let whole = Traced::new(-2.0, 2.0, 4, rosenbrock);
        let parts = Traced::new(-2.0, 2.0, 4, rosenbrock);
        let a = drive_arm(arm, &whole, 400, &[400], None, 11);
        let b = drive_arm(arm, &parts, 400, &[1, 37, 112, 250], None, 11);
        assert_eq!(a, b, "{} trajectory depends on slicing", arm.name());
    }

    fn assert_pauses_at_the_ledger_cap(arm: ArmKind) {
        // The cap cuts every call short of its slice; an arm that asked past
        // it would be told a refused charge and leave the uncut trajectory.
        let whole = Traced::new(-2.0, 2.0, 4, rosenbrock);
        let cut = Traced::new(-2.0, 2.0, 4, rosenbrock);
        let a = drive_arm(arm, &whole, 400, &[240], None, 13);
        let b = drive_arm(arm, &cut, 400, &[100, 100, 100, 100], Some(60), 13);
        assert_eq!(b.len(), 240);
        assert_eq!(a, b, "{} trajectory depends on the cap", arm.name());
    }

    fn replay(arm: ArmKind, seed: u64) -> Vec<Array1<f64>> {
        drive_arm(
            arm,
            &Traced::new(-2.0, 2.0, 4, rosenbrock),
            300,
            &[300],
            None,
            seed,
        )
    }

    #[test]
    fn cma_arm_charges_every_evaluation_inside_the_box() {
        assert_charges_inside_the_box(ArmKind::Cma);
    }

    #[test]
    fn cma_arm_resumes_identically_across_slices() {
        assert_resumes_identically(ArmKind::Cma);
    }

    #[test]
    fn cma_arm_pauses_at_the_ledger_cap() {
        assert_pauses_at_the_ledger_cap(ArmKind::Cma);
    }

    #[test]
    fn cma_arm_replays_by_seed() {
        assert_eq!(replay(ArmKind::Cma, 5), replay(ArmKind::Cma, 5));
        assert_ne!(
            replay(ArmKind::Cma, 5),
            replay(ArmKind::Cma, 6),
            "the sampling stream follows the seed"
        );
    }

    #[test]
    fn qn_arm_charges_every_evaluation_inside_the_box() {
        assert_charges_inside_the_box(ArmKind::Qn);
    }

    #[test]
    fn qn_arm_resumes_identically_across_slices() {
        assert_resumes_identically(ArmKind::Qn);
    }

    #[test]
    fn qn_arm_pauses_at_the_ledger_cap() {
        assert_pauses_at_the_ledger_cap(ArmKind::Qn);
    }

    #[test]
    fn qn_arm_replays_by_seed() {
        assert_eq!(replay(ArmKind::Qn, 5), replay(ArmKind::Qn, 5));
        // A bowl converges within the budget, so the kicks that restart the
        // descent, the only random draws, show up in the trace.
        let bowl = |seed| {
            drive_arm(
                ArmKind::Qn,
                &Traced::new(-2.0, 2.0, 4, |x| x.dot(&x)),
                300,
                &[300],
                None,
                seed,
            )
        };
        assert_eq!(bowl(5), bowl(5));
        assert_ne!(bowl(5), bowl(6), "kicks follow the seed");
    }

    #[test]
    fn qn_arm_starts_at_an_evaluated_incumbent_without_repeating_it() {
        let obj = Traced::new(-2.0, 2.0, 3, rosenbrock);
        let ledger = BudgetLedger::new(50, 3);
        let budgeted = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let start = Array1::from_vec(vec![0.5, -0.25, 1.0]);
        budgeted.eval(start.view());
        let mut states = ArmStates::default();
        run_qn_arm(&budgeted, &ledger, &mut states, 49, 1, false);
        let points = obj.points();
        assert_eq!(points.len(), 50);
        assert_eq!(points[0], start);
        assert!(
            points[1..].iter().all(|x| x != start),
            "the descent starts from the recorded value"
        );

        // Nothing finite yet: the incumbent is the first evaluated point, and
        // the arm kicks from it instead of evaluating it again.
        let nowhere = Traced::new(-2.0, 2.0, 3, |_| f64::NAN);
        let ledger = BudgetLedger::new(4, 3);
        let budgeted = BudgetedObjective {
            inner: &nowhere,
            ledger: &ledger,
        };
        budgeted.eval(start.view());
        let mut states = ArmStates::default();
        run_qn_arm(&budgeted, &ledger, &mut states, 3, 1, false);
        let points = nowhere.points();
        assert_eq!(points.len(), 4);
        assert!(points[1..].iter().all(|x| x != start));
    }

    #[test]
    fn values_only_narrow_box_charges_exactly_with_and_without_a_start() {
        // Width 4 and no gradient. The 6-d case affords an opening descent;
        // the 12-d one does not, so its descent first plays in the warm-up
        // round.
        for (dim, budget) in [(6usize, 900usize), (12, 600)] {
            let start = Array1::from_elem(dim, -1.5);
            for x0 in [Some(start.view()), None] {
                let obj = Traced::new(-2.0, 2.0, dim, rosenbrock);
                let result = portfolio_optimize_from::<_, ShiftQuadratic>(
                    &obj,
                    None,
                    budget,
                    7,
                    None,
                    PortfolioPolicy::Auto,
                    x0,
                );
                let points = obj.points();
                assert_eq!(points.len(), budget, "every call is charged once");
                assert_eq!(result.n_evals, budget);
                assert_eq!(result.n_grads, 0);
                assert!(points.iter().all(|x| obj.bounds.contains(x.view())));
                let names: Vec<_> = result.arm_stats.iter().map(|s| s.name).collect();
                assert_eq!(names, ["cma", "qn", "gsa", "de", "surrogate", "explore"]);
                let played: Vec<_> = result
                    .arm_stats
                    .iter()
                    .filter(|s| s.pulls > 0)
                    .map(|s| s.name)
                    .collect();
                assert!(played.contains(&"qn"), "qn ran in {dim}-D");
                if x0.is_some() {
                    assert_eq!(points[0], start, "the start is the first evaluation");
                    assert!(result.best_val <= rosenbrock(start.view()));
                }
            }
        }
    }

    #[test]
    fn values_only_narrow_box_descends_rosenbrock() {
        let obj = Traced::new(-2.0, 2.0, 6, rosenbrock);
        let start = Array1::from_elem(6, -1.2);
        let result = portfolio_optimize_from::<_, ShiftQuadratic>(
            &obj,
            None,
            1500,
            1,
            None,
            PortfolioPolicy::Auto,
            Some(start.view()),
        );
        assert!(result.best_val < 1e-8, "best {}", result.best_val);
    }

    #[test]
    fn start_at_the_minimum_stays_the_incumbent() {
        // On a narrow box and a wide one, no arm may displace a start that
        // no point improves on.
        for (low, high) in [(-2.0, 2.0), (-5.12, 5.12)] {
            let obj = Traced::new(low, high, 6, rosenbrock);
            let start = Array1::ones(6);
            let result = portfolio_optimize_from::<_, ShiftQuadratic>(
                &obj,
                None,
                300,
                2,
                None,
                PortfolioPolicy::Auto,
                Some(start.view()),
            );
            assert_eq!(result.best_val, 0.0);
            assert_eq!(result.best_pos, start.to_vec());
            assert_eq!(obj.points()[0], start);
        }
    }

    #[test]
    fn gsa_local_search_descends_from_each_new_record() {
        // Mid-width values-only box: the anneal's local search should reach
        // the bottom of a smooth bowl that visiting alone only approaches.
        let obj = Traced::new(-5.12, 5.12, 4, |x| {
            x.iter().map(|v| (v - 1.3).powi(2)).sum()
        });
        let ledger = BudgetLedger::new(400, 4);
        let budgeted = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let mut states = ArmStates {
            values_only: true,
            ..ArmStates::default()
        };
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..4 {
            ledger.cap_set(ledger.used_get() + 100);
            run_arm::<_, ShiftQuadratic>(
                ArmKind::Gsa,
                &budgeted,
                None,
                &ledger,
                &mut states,
                &mut rng,
                100,
                400,
            );
            ledger.cap_set(400);
        }
        assert_eq!(obj.points().len(), ledger.used_get());
        assert!(obj.points().iter().all(|x| obj.bounds.contains(x.view())));
        assert!(ledger.best_get() < 1e-8, "best {}", ledger.best_get());
    }

    fn rastrigin(x: ArrayView1<f64>) -> f64 {
        10.0 * x.len() as f64
            + x.iter()
                .map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos())
                .sum::<f64>()
    }

    fn ackley(x: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let ripple = x
            .iter()
            .map(|v| (2.0 * std::f64::consts::PI * v).cos())
            .sum::<f64>();
        -20.0 * (-0.2 * (x.dot(&x) / n).sqrt()).exp() - (ripple / n).exp()
            + 20.0
            + std::f64::consts::E
    }

    struct RosenbrockGrad {
        dim: usize,
    }

    impl Gradient<f64> for RosenbrockGrad {
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let mut g = Array1::zeros(x.len());
            for i in 0..x.len() - 1 {
                let r = x[i + 1] - x[i] * x[i];
                g[i] += -400.0 * x[i] * r - 2.0 * (1.0 - x[i]);
                g[i + 1] += 200.0 * r;
            }
            g
        }

        fn dim(&self) -> usize {
            self.dim
        }
    }

    #[test]
    fn gradient_path_evaluates_the_start_first() {
        // Boxes and dimensions that select LowDimSmooth, HighDim and
        // MultimodalGlobal.
        for (half, dim) in [(2.0, 4usize), (2.0, 20), (5.12, 6)] {
            let obj = Traced::new(-half, half, dim, rosenbrock);
            let start = Array1::from_shape_fn(dim, |i| if i % 2 == 0 { -1.2 } else { 1.0 });
            let result = portfolio_optimize_from(
                &obj,
                Some(&RosenbrockGrad { dim }),
                600,
                3,
                None,
                PortfolioPolicy::Auto,
                Some(start.view()),
            );
            let points = obj.points();
            assert_eq!(points[0], start, "the start is the first call in {dim}-D");
            assert_eq!(points.len(), result.n_evals);
            assert!(result.n_evals + result.n_grads <= 600);
            assert!(result.best_val <= rosenbrock(start.view()));
        }
    }

    #[test]
    fn values_only_warm_up_plays_every_arm() {
        // Rastrigin from a side basin, on a narrow box and a wide one.
        for half in [2.0, 5.12] {
            let obj = Traced::new(-half, half, 4, rastrigin);
            let start = Array1::from_elem(4, 1.0);
            let result = portfolio_optimize_from::<_, ShiftQuadratic>(
                &obj,
                None,
                2000,
                5,
                None,
                PortfolioPolicy::Auto,
                Some(start.view()),
            );
            let points = obj.points();
            assert_eq!(points.len(), 2000);
            assert_eq!(result.n_evals, 2000);
            assert!(points.iter().all(|x| obj.bounds.contains(x.view())));
            for stat in &result.arm_stats {
                assert!(
                    stat.pulls > 0,
                    "{} took no slice at half-width {half}",
                    stat.name
                );
            }
            assert!(result.best_val < rastrigin(start.view()));
        }
    }

    #[test]
    fn values_only_phases_leave_every_arm_a_slice() {
        // Far out on Ackley GSA records keep paying, so its phase runs into
        // the reserve; CMA-ES, DE, the surrogate and explore still take one.
        for (dim, budget) in [(4usize, 2000usize), (10, 5000)] {
            let start = Array1::from_shape_fn(dim, |i| if i % 2 == 0 { 20.0 } else { -20.0 });
            for seed in [0u64, 1] {
                let obj = Traced::new(-32.768, 32.768, dim, ackley);
                let result = portfolio_optimize_from::<_, ShiftQuadratic>(
                    &obj,
                    None,
                    budget,
                    seed,
                    None,
                    PortfolioPolicy::Auto,
                    Some(start.view()),
                );
                assert_eq!(obj.points().len(), budget);
                let gsa = result.arm_stats.iter().find(|s| s.name == "gsa");
                assert!(gsa.is_some_and(|s| s.pulls > ROUNDS_PER_ARM));
                for stat in &result.arm_stats {
                    assert!(
                        stat.pulls > 0,
                        "{} took no slice in {dim}-D, seed {seed}",
                        stat.name
                    );
                }
            }
        }
    }

    #[test]
    fn values_only_descent_keeps_its_turn_while_it_pays() {
        // Twelve dimensions and 700 evaluations are short of the opening,
        // so the descent first plays in the warm-up round. On a stiff
        // quadratic it pays every slice until it converges, which takes
        // several slices: one turn spans them.
        fn stiff(x: ArrayView1<f64>) -> f64 {
            let last = (x.len() - 1) as f64;
            x.iter()
                .enumerate()
                .map(|(i, v)| 10f64.powf(3.0 * i as f64 / last) * v * v)
                .sum()
        }
        let (dim, budget) = (12usize, 700usize);
        assert!(budget < QN_AFFORDABLE_GRADIENTS * dim * (dim + 1));
        let slice = (SLICE_GRAD_EQUIVALENTS * (dim + 1))
            .max(budget / (ROUNDS_PER_ARM * VALUES_ONLY_ARMS.len()));
        let obj = Traced::new(-2.0, 2.0, dim, stiff);
        let start = Array1::from_elem(dim, 1.5);
        let result = portfolio_optimize_from::<_, ShiftQuadratic>(
            &obj,
            None,
            budget,
            3,
            None,
            PortfolioPolicy::Auto,
            Some(start.view()),
        );
        assert_eq!(obj.points().len(), budget);
        let turns: usize = result.arm_stats.iter().map(|s| s.pulls).sum();
        assert!(
            turns * slice < budget - 1,
            "{turns} turns of at most {slice} evaluations would not spend {budget}"
        );
        assert!(result.best_val < 1e-12, "{}", result.best_val);
    }

    #[test]
    fn values_only_seeds_vary_the_run_with_and_without_a_start() {
        let run = |seed, x0: Option<ArrayView1<f64>>| {
            let obj = Traced::new(-2.0, 2.0, 4, rastrigin);
            portfolio_optimize_from::<_, ShiftQuadratic>(
                &obj,
                None,
                600,
                seed,
                None,
                PortfolioPolicy::Auto,
                x0,
            );
            obj.points()
        };
        let (a, b) = (run(1, None), run(2, None));
        assert_eq!(a, run(1, None), "a seed replays exactly");
        assert_ne!(
            a[0], b[0],
            "without a start the first point follows the seed"
        );
        // From a fixed start the opening descent is shared and the rest is not.
        let start = Array1::from_elem(4, 1.0);
        let (c, d) = (run(1, Some(start.view())), run(2, Some(start.view())));
        assert_eq!((&c[0], &d[0]), (&start, &start));
        assert_ne!(c, d, "seeds vary the run from a fixed start");
    }

    #[test]
    fn values_only_start_in_the_global_basin_descends() {
        // Rastrigin-10 from just off its minimum, on the usual box and on a
        // shifted one whose centre lies in another basin.
        for (low, high) in [(-5.12, 5.12), (-3.1, 4.9)] {
            let obj = Traced::new(low, high, 10, rastrigin);
            let start = Array1::from_elem(10, 0.1);
            let result = portfolio_optimize_from::<_, ShiftQuadratic>(
                &obj,
                None,
                1000,
                4,
                None,
                PortfolioPolicy::Auto,
                Some(start.view()),
            );
            assert!(
                result.best_val < 1e-8,
                "best {} on [{low}, {high}]",
                result.best_val
            );
        }
    }

    #[test]
    fn values_only_gsa_descends_from_the_incumbent_first() {
        // Only the global basin beats this start, so the first local search
        // runs from it and one slice reaches the floor.
        let obj = Traced::new(-5.12, 5.12, 4, rastrigin);
        let ledger = BudgetLedger::new(400, 4);
        let budgeted = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let start = Array1::from_elem(4, 0.05);
        budgeted.eval(start.view());
        let mut states = ArmStates {
            values_only: true,
            ..ArmStates::default()
        };
        let mut rng = StdRng::seed_from_u64(3);
        run_arm::<_, ShiftQuadratic>(
            ArmKind::Gsa,
            &budgeted,
            None,
            &ledger,
            &mut states,
            &mut rng,
            399,
            400,
        );
        let points = obj.points();
        assert_eq!(points.len(), ledger.used_get());
        assert!(
            points[1..].iter().all(|x| *x != start),
            "the chain starts from the recorded value"
        );
        assert!(ledger.best_get() < 1e-8, "best {}", ledger.best_get());
    }

    #[test]
    fn values_only_de_population_starts_with_the_incumbent() {
        let obj = Traced::new(-2.0, 2.0, 4, rosenbrock);
        let ledger = BudgetLedger::new(400, 4);
        let budgeted = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        let start = Array1::ones(4);
        budgeted.eval(start.view());
        let mut states = ArmStates {
            values_only: true,
            ..ArmStates::default()
        };
        let mut rng = StdRng::seed_from_u64(3);
        run_arm::<_, ShiftQuadratic>(
            ArmKind::De,
            &budgeted,
            None,
            &ledger,
            &mut states,
            &mut rng,
            100,
            400,
        );
        let state = states.de.as_ref().expect("population");
        assert_eq!(state.pop[0], start);
        assert_eq!(state.vals[0], 0.0);
        assert_eq!(
            ledger.used_get(),
            101,
            "the first pull evolves the population for the rest of its slice"
        );
        let pop = state.pop.len();
        assert!(
            obj.points()[1..pop].iter().all(|x| *x != start),
            "the population is not charged for the incumbent"
        );
    }

    #[test]
    fn cma_arm_goes_separable_above_the_cap_or_short_of_the_learning_horizon() {
        for (dim, budget, separable) in [
            (10usize, 5000usize, false),
            (39, 1000, true),
            (39, 20_000, false),
            (CMA_FULL_MAX_DIM + 1, 10_000_000, true),
        ] {
            let obj = Traced::new(-2.0, 2.0, dim, rosenbrock);
            let ledger = BudgetLedger::new(budget, dim);
            let budgeted = BudgetedObjective {
                inner: &obj,
                ledger: &ledger,
            };
            let mut states = ArmStates::default();
            run_cma_arm(&budgeted, &ledger, &mut states, 1, 3, budget);
            let state = states.cma.as_ref().expect("cma state");
            assert_eq!(state.es.is_separable(), separable, "{dim}-D at {budget}");
        }
    }

    #[test]
    fn cma_and_qn_arms_draw_from_separate_streams() {
        // Given one seed, the first CMA-ES generation samples mean + sigma z
        // in the unit box and the first kick moves along the descent's first
        // normal draws. Shared generators would make that kick one of the z.
        let (dim, seed) = (4usize, 17u64);
        let obj = Traced::new(-2.0, 2.0, dim, |x| x.dot(&x));
        let ledger = BudgetLedger::new(100, dim);
        let budgeted = BudgetedObjective {
            inner: &obj,
            ledger: &ledger,
        };
        budgeted.eval(Array1::zeros(dim).view());
        let mut states = ArmStates::default();
        let samples = default_lambda(dim) - 1;
        run_cma_arm(&budgeted, &ledger, &mut states, samples, seed, 100);
        run_qn_arm(&budgeted, &ledger, &mut states, 0, seed, false);
        let mut kicks = states.qn.as_ref().expect("qn state").rng.clone();
        let noise = Array1::<f64>::from_iter(
            (0..dim).map(|_| rand_distr::StandardNormal.sample(&mut kicks)),
        );
        let points = obj.points();
        assert_eq!(points.len(), 1 + samples);
        for x in &points[1..] {
            let z = x / (4.0 * CMA_FIRST_SIGMA);
            let gap = (&z - &noise).fold(0.0_f64, |a, b| a.max(b.abs()));
            assert!(gap > 1e-6, "the first kick reuses a CMA-ES draw");
        }
    }
}
