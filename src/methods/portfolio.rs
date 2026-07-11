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

use crate::accept::{AcceptRule, TsallisAccept};
use crate::bias::Bias;
use crate::cool::{Cooling, LogCool, TsallisCool};
use crate::exchange::TsallisExchange;
use crate::hmc::{HmcSaSampler, OmelyanIntegrator, QGaussianMomentum};
use crate::methods::bayesian_pilot::{
    LaplacePosterior, PilotObservation, PilotPrior, fit_laplace, pilot_draws_qmc,
};
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
/// exactly (proofs/d5_endgame_switch.py, check 2), so x_i estimates
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

/// GSA visiting index; the manuscript's generalized-SA default.
const GSA_Q_V: f64 = 2.62;
/// GSA acceptance index; the manuscript's generalized-SA default.
const GSA_Q_A: f64 = 1.7;
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

    fn charge_probe(&self, n_evals: usize, n_grads: usize) -> bool {
        let work = n_evals + n_grads;
        if self.remaining() < work {
            return false;
        }
        self.used.fetch_add(work, Ordering::Relaxed);
        self.n_evals.fetch_add(n_evals, Ordering::Relaxed);
        self.n_grads.fetch_add(n_grads, Ordering::Relaxed);
        true
    }

    /// Archive a candidate only if `value` is finite **and** `x` lies inside
    /// `bounds` (GJQ-style feasibility choke-point: never promote OOB bests).
    fn record(&self, x: ArrayView1<f64>, value: f64, bounds: &Bounds<f64>) {
        if !value.is_finite() || !bounds.contains(x) {
            return;
        }
        let mut inner = self.inner.lock().expect("ledger lock");
        if value < self.best_get() {
            self.best_val.store(value.to_bits(), Ordering::Relaxed);
            inner.best_pos = Some(x.to_owned());
        }
        if inner.archive_y.len() < self.archive_cap {
            inner.archive_x.extend(x.iter().copied());
            inner.archive_y.push(value);
        }
    }

    fn incumbent(&self, bounds: &Bounds<f64>) -> Array1<f64> {
        match self.inner.lock().expect("ledger lock").best_pos.as_ref() {
            // Defensive clip: best_pos is only written for in-bounds points.
            Some(pos) => bounds.clip(pos.view()),
            None => (&bounds.low + &bounds.high) * 0.5,
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
        if self.ledger.exhausted() {
            return f64::INFINITY;
        }
        self.ledger.used.fetch_add(1, Ordering::Relaxed);
        self.ledger.n_evals.fetch_add(1, Ordering::Relaxed);
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
        if self.ledger.exhausted() {
            return f64::INFINITY;
        }
        self.ledger.used.fetch_add(1, Ordering::Relaxed);
        self.ledger.n_evals.fetch_add(1, Ordering::Relaxed);
        // Reflect into the local box; archive only if also globally feasible.
        let x_local = crate::movekernel::reflect_into_box(x, &self.bounds);
        let value = self.inner.eval(x_local.view());
        self.ledger
            .record(x_local.view(), value, self.inner.bounds());
        value
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
        if self.ledger.exhausted() {
            return Array1::zeros(self.ledger.dim);
        }
        self.ledger.used.fetch_add(1, Ordering::Relaxed);
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
        }
    }

    /// Arms that may earn bandit credit without an incumbent drop
    /// (enhanced sampling / path ensemble exploration).
    fn exploratory_credit(self) -> bool {
        matches!(self, ArmKind::Metad | ArmKind::Tps)
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

struct GsaState {
    units: Vec<Array1<f64>>,
    vals: Vec<f64>,
    epoch: usize,
    rng: StdRng,
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
}

/// Persistent adaptive-Metropolis descent chain: the D6 annealed-descent
/// scaling law productized (proofs/d6_annealed_descent_scaling.py).
///
/// Temperature is not a schedule in epoch index. On a locally quadratic
/// basin, expected one-step decrease of the current chain-state energy is
/// positive iff theta = T d / (f(x) - f*) < 2 exactly (D6, symbolic
/// paired-increment identity), so the chain runs at the gap-proportional law
///     T(x) = THETA_TILDE (f(x) - f_best) / d,
/// with THETA_TILDE = 0.5 keeping ~91% of the maximal descent rate:
/// cooling happens exactly as fast as measured progress, with no cooling
/// constant. The proposal SHAPE is the chain's running covariance
/// (Haario, Saksman, and Tamminen 2001; ergodic under diminishing
/// adaptation, Roberts and Rosenthal 2007); the proposal SIZE tracks the
/// D6 optimal acceptance alpha*(0.5) ~ 0.32 by Robbins-Monro - the
/// descent-side analogue of the 0.234 sampling rule, recovering
/// Rechenberg's 0.270 at theta -> 0. Reflection into the box keeps the
/// proposal symmetric (law L1), so Metropolis acceptance stays valid.
struct AmSaState {
    x: Array1<f64>,
    f_x: f64,
    mean: Array1<f64>,
    /// Unnormalized scatter matrix sum (x - mean) outer products.
    scatter: Array2<f64>,
    n_obs: usize,
    /// Robbins-Monro log proposal-scale multiplier.
    log_scale: f64,
    /// Robbins-Monro step counter (diminishing adaptation).
    rm_n: usize,
}

/// D6 operating temperature theta = T d / gap (91% of maximal descent).
const AM_THETA_TILDE: f64 = 0.5;
/// D6 optimal acceptance at AM_THETA_TILDE.
const AM_ALPHA_TARGET: f64 = 0.32;

impl AmSaState {
    fn new(x: Array1<f64>, f_x: f64) -> Self {
        let d = x.len();
        Self {
            mean: x.clone(),
            scatter: Array2::zeros((d, d)),
            n_obs: 1,
            x,
            f_x,
            log_scale: 0.0,
            rm_n: 0,
        }
    }

    /// Welford-style running mean/scatter update over chain states.
    fn observe(&mut self, x: ArrayView1<f64>) {
        self.n_obs += 1;
        let n = self.n_obs as f64;
        let delta = &x.to_owned() - &self.mean;
        self.mean = &self.mean + &delta.mapv(|v| v / n);
        let delta2 = &x.to_owned() - &self.mean;
        for i in 0..delta.len() {
            for j in 0..delta.len() {
                self.scatter[(i, j)] += delta[i] * delta2[j];
            }
        }
    }

    /// Covariance Cholesky factor with Haario-style regularization:
    /// Sigma_hat + eps diag(width^2). Falls back to the diagonal on
    /// factorization failure.
    fn proposal_chol(&self, bounds: &Bounds<f64>) -> Array2<f64> {
        let d = self.x.len();
        let mut cov = Array2::zeros((d, d));
        // Haario burn-in: until the chain has 2d accepted observations
        // the scatter is uninformative, so propose from a box-scaled
        // identity ((0.1 width_i)^2) rather than the near-zero scatter;
        // a frozen initial chain wastes its whole slice inside one basin.
        if self.n_obs < 2 * d {
            let mut l = Array2::zeros((d, d));
            for i in 0..d {
                let w = (bounds.high[i] - bounds.low[i]).abs().max(1e-12);
                l[(i, i)] = 0.1 * w;
            }
            return l;
        }
        let denom = (self.n_obs as f64 - 1.0).max(1.0);
        for i in 0..d {
            for j in 0..d {
                cov[(i, j)] = self.scatter[(i, j)] / denom;
            }
        }
        for i in 0..d {
            let w = (bounds.high[i] - bounds.low[i]).abs().max(1e-12);
            cov[(i, i)] += 1e-6 * w * w + 1e-12;
        }
        cholesky_lower(&cov).unwrap_or_else(|| {
            let mut l = Array2::zeros((d, d));
            for i in 0..d {
                l[(i, i)] = cov[(i, i)].sqrt();
            }
            l
        })
    }
}

/// Dense lower Cholesky for the small proposal covariances (d <= ~64).
fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let d = a.nrows();
    let mut l: Array2<f64> = Array2::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[(i, j)] = s.sqrt();
            } else {
                l[(i, j)] = s / l[(j, j)];
            }
        }
    }
    Some(l)
}

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
    fn record_discovery_prob(&self) -> Option<f64> {
        if self.n_samples < 5 {
            return None;
        }
        let n1 = self.entries.iter().filter(|(_, _, h)| *h == 1).count();
        let w = self.entries.len();
        Some((n1 as f64 / self.n_samples as f64) / (w as f64 + 1.0))
    }
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

fn unit_coordinate(pos: f64, low: f64, high: f64) -> f64 {
    let width = high - low;
    if width.is_finite() && width > 0.0 {
        ((pos - low) / width).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn unit_to_box(unit: &Array1<f64>, low: &Array1<f64>, high: &Array1<f64>) -> Array1<f64> {
    Array1::from_iter(
        unit.iter()
            .zip(low.iter().zip(high.iter()))
            .map(|(u, (lo, hi))| lo + u.clamp(0.0, 1.0) * (hi - lo)),
    )
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
    if !obj.ledger.charge_probe(1, 1) {
        return None;
    }
    let value = obj.inner.eval(center.view());
    let gradient = grad.inner.grad(center.view());
    if !value.is_finite() {
        return None;
    }
    let value_scale = value.abs().max(1.0);
    if gradient.len() != bounds.dims || gradient.iter().any(|v| !v.is_finite()) {
        return Some(CenterProbe {
            gradient_ratio: None,
        });
    }
    let width_norm = (0..bounds.dims)
        .map(|idx| {
            let width = bounds.high[idx] - bounds.low[idx];
            if width.is_finite() && width > 0.0 {
                width * width
            } else {
                1.0
            }
        })
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    let grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
    let ratio = grad_norm * width_norm / value_scale;
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

fn initialize_gsa_state<O>(
    obj: &BudgetedObjective<'_, O>,
    slice: usize,
    seed: u64,
) -> Option<GsaState>
where
    O: Objective<f64>,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    if dim == 0 || slice < 2 {
        return None;
    }
    let chain_count = (slice / 8).clamp(2, 4 * dim.max(1)).min(slice);
    let starts = eindir_core::shifted_low_discrepancy_points(
        &bounds,
        chain_count,
        qmc_skip_from_seed(seed),
        seed,
    );
    let mut units = Vec::with_capacity(chain_count);
    let mut vals = Vec::with_capacity(chain_count);

    for start in starts.outer_iter() {
        if obj.ledger.exhausted() {
            break;
        }
        let pos = bounds.clip(start);
        let unit = Array1::from_iter(
            (0..dim).map(|axis| unit_coordinate(pos[axis], bounds.low[axis], bounds.high[axis])),
        );
        let value = obj.eval(pos.view());
        units.push(unit);
        vals.push(value);
    }

    (!units.is_empty()).then(|| GsaState {
        units,
        vals,
        epoch: 0,
        rng: StdRng::seed_from_u64(seed),
    })
}

fn run_persistent_gsa<O>(obj: &BudgetedObjective<'_, O>, state: &mut GsaState, slice: usize)
where
    O: Objective<f64>,
{
    let bounds = obj.bounds().clone();
    let cooling = TsallisCool::new(1.0, GSA_Q_V);
    let visit = TsallisVisit::new(GSA_Q_V);
    let accept = TsallisAccept::new(GSA_Q_A);
    let start_used = obj.ledger.used_get();

    while obj.ledger.used_get().saturating_sub(start_used) < slice && !obj.ledger.exhausted() {
        let temp = cooling.temperature(state.epoch);
        for chain in 0..state.units.len() {
            if obj.ledger.used_get().saturating_sub(start_used) >= slice || obj.ledger.exhausted() {
                break;
            }
            let proposal_unit = visit
                .propose(state.units[chain].view(), temp, &mut state.rng)
                .mapv(|value| value.clamp(0.0, 1.0));
            let proposal_pos = unit_to_box(&proposal_unit, &bounds.low, &bounds.high);
            let proposal_val = obj.eval(proposal_pos.view());
            let accepted = if !proposal_val.is_finite() {
                false
            } else if !state.vals[chain].is_finite() {
                true
            } else {
                let probability = accept
                    .accept_prob(proposal_val - state.vals[chain], temp)
                    .clamp(0.0, 1.0);
                state.rng.random::<f64>() < probability
            };
            if accepted {
                state.units[chain] = proposal_unit;
                state.vals[chain] = proposal_val;
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
    let prior = PilotPrior::default();
    let draws = pilot_draws_qmc(&prior, PILOT_CHAINS, seed);
    let per_chain = bayesian_pilot_steps_per_chain(pilot_budget, min_steps_per_chain);
    let start_used = ledger.used_get();
    let mut observations = Vec::with_capacity(PILOT_CHAINS);
    let mut best_anchor = None;
    let mut best_anchor_val = f64::INFINITY;

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
        for step in 0..per_chain {
            if ledger.exhausted() || ledger.used_get().saturating_sub(start_used) >= pilot_budget {
                break;
            }
            let temp = ladder_temperature(t_init, step / 10);
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

    (observations.len() >= 2).then(|| PilotState {
        posterior: fit_laplace(&observations, &PilotPrior::default()),
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
                let res = projected_gradient_polish(obj, grad, trial, slice / 2, 1.0, 1e-8);
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
                for i in 0..points.nrows() {
                    if ledger.exhausted() {
                        break;
                    }
                    let x = points.row(i).to_owned();
                    let v = obj.eval(x.view());
                    pop.push(x);
                    vals.push(v);
                }
                if pop.len() < 4 {
                    states.de = None;
                    return;
                }
                states.de = Some(DeState { pop, vals });
                return;
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
            let mut used = 0usize;
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
            if states.gsa.is_none() {
                states.gsa = initialize_gsa_state(obj, slice, seed);
            }
            if let Some(state) = states.gsa.as_mut() {
                run_persistent_gsa(obj, state, slice);
            }
        }
        ArmKind::AmSa => {
            // D6 annealed-descent chain: gap-proportional temperature
            // T = 0.5 gap / d (progress requires theta < 2; see
            // proofs/d6_annealed_descent_scaling.py), Haario covariance
            // shape, Robbins-Monro size targeting alpha* ~ 0.32.
            // Exact Metropolis only: declared evaluation noise routes to
            // the OSA-capable arms instead.
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
            let st = states.am.as_mut().expect("am chain installed");
            let l = st.proposal_chol(&bounds);
            let base = 2.38 / (d as f64).sqrt();
            for _ in 0..slice {
                if ledger.remaining() < 1 {
                    break;
                }
                let gap = (st.f_x - ledger.best_get()).max(0.0);
                let temp = AM_THETA_TILDE * gap / d as f64;
                let z: Vec<f64> = (0..d)
                    .map(|_| rand_distr::StandardNormal.sample(rng))
                    .collect();
                let scale = st.log_scale.exp() * base;
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
                }
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
    // Gradient-free multimodal boxes are capped harder: ranking ten arms
    // leaves no arm the contiguous budget its chain or population needs,
    // which is how the flagship no-grad path lost to its own preset.
    let regime_cap = match regime {
        crate::methods::regime::OptimizationRegime::MultimodalNoGrad => 6,
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
    let regime = match policy {
        PortfolioPolicy::Auto => crate::methods::regime::select_regime(&features),
        PortfolioPolicy::Legacy => crate::methods::regime::OptimizationRegime::Default,
    };

    let ledger = BudgetLedger::new(budget, dim);
    let budgeted_obj = BudgetedObjective {
        inner: obj,
        ledger: &ledger,
    };
    let budgeted_grad = grad.map(|g| BudgetedGradient {
        inner: g,
        ledger: &ledger,
    });

    // Derived scheduler quantities; see the module constants for the
    // two governing numbers. Auto unlocks extra specialized arms so the
    // regime order can actually include GLE/reduced/DE under tight horizons.
    // Unlock MetaD + TPS (+ one more specialized arm) under Auto so the
    // horizon does not starve the new enhanced-sampling machinery.
    let extra = match (policy, regime) {
        (
            PortfolioPolicy::Auto,
            crate::methods::regime::OptimizationRegime::HighDimIllConditioned
            | crate::methods::regime::OptimizationRegime::MultimodalNoGrad
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
    // TPE dual-density history over arm indices (Auto policy).
    let mut tpe = crate::methods::tpe::TpeCategorical::new(arms.len());
    let mut states = ArmStates {
        noise_sigma,
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

    // Auto MultimodalNoGrad: short DE/GSA front-load (values-only problems
    // need global structure before Thompson). Kept small so restarts still fire.
    if policy == PortfolioPolicy::Auto
        && matches!(
            regime,
            crate::methods::regime::OptimizationRegime::MultimodalNoGrad
        )
    {
        // 28% front-load: DE needs population-scale budget on rugged boxes.
        let front_budget = ((budget as f64) * 0.28).round() as usize;
        let preferred = [
            ArmKind::De,
            ArmKind::Gsa,
            ArmKind::Surrogate,
            ArmKind::Explore,
        ];
        let per = (front_budget / preferred.len()).max(slice);
        let mut seed_f = seed ^ 0xBEEF_u64;
        for arm in preferred {
            if !arms.contains(&arm) || ledger.remaining() < 8 {
                continue;
            }
            let slice_use = per.min(ledger.remaining());
            if slice_use < 4 {
                break;
            }
            let ceiling = ledger.used_get() + slice_use;
            ledger.cap_set(ceiling.min(budget));
            seed_f = seed_f.wrapping_add(1);
            let mut front_rng = StdRng::seed_from_u64(seed_f);
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
                let q0 = match gt {
                    Some(theta) => 1.0 - theta.min(1.0),
                    None => b_e / (a_e + b_e).max(1e-12),
                };
                let win = |p: usize| -> f64 {
                    let e = remaining.saturating_sub(p) / slice.max(1);
                    let pi = match gt {
                        Some(theta) => (1.0 - theta.min(1.0)).powi(e.min(128) as i32),
                        None => {
                            let mut pi = 1.0f64;
                            for i in 0..e.min(128) {
                                pi *= (b_e + i as f64) / (a_e + b_e + i as f64);
                            }
                            pi
                        }
                    };
                    (q0 + (1.0 - q0) * (1.0 - pi))
                        * contraction.conversion_prob(NEAR_BEST_ORDERS, p)
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
            // optimal; see proofs/d5_endgame_switch.py). Cycle
            // quasi-Newton polish with shrinking trust-region poll
            // restarts until the budget is gone or two cycles go dry.
            ledger.cap_set(budget);
            coordinate_opposition_scout(&budgeted_obj, &ledger, &bounds);
            let mut dry = 0usize;
            while ledger.remaining() >= 4 && dry < 3 {
                let before = ledger.best_get();
                if let Some(grad) = budgeted_grad.as_ref() {
                    let mut start = ledger.incumbent(&bounds);
                    if dry > 0 {
                        // Jittered quasi-Newton restart: a relative-scale
                        // perturbation escapes line-search stall points
                        // while staying inside the incumbent basin.
                        for v in start.iter_mut() {
                            let u = 2.0 * rng.random::<f64>() - 1.0;
                            *v += 1e-7 * (1.0 + v.abs()) * u;
                        }
                        start = bounds.clip(start.view());
                    }
                    projected_gradient_polish(
                        &budgeted_obj,
                        grad,
                        start,
                        ledger.remaining(),
                        1.0,
                        1e-12,
                    );
                }
                if budgeted_grad.is_none() && ledger.remaining() >= 8 {
                    // Gradient-free polish: the D6 chain at gap ~ 0 is
                    // anisotropic greedy descent with a learned
                    // covariance - poll steps are isotropic and stall on
                    // ill-conditioned basin bottoms.
                    if let Some(st) = states.am.as_mut() {
                        st.x = ledger.incumbent(&bounds);
                        st.f_x = ledger.best_get();
                    }
                    let am_share = (ledger.remaining() / 2).max(8);
                    run_arm(
                        ArmKind::AmSa,
                        &budgeted_obj,
                        budgeted_grad.as_ref(),
                        &ledger,
                        &mut states,
                        &mut rng,
                        am_share,
                        budget,
                    );
                }
                let rem = ledger.remaining();
                if rem >= 8 {
                    qmc_trust_region_poll(
                        &budgeted_obj,
                        ledger.incumbent(&bounds),
                        (rem / 2).max(8).min(rem),
                        rng.random::<u64>(),
                        0.0,
                        3,
                        0,
                    );
                }
                let after = ledger.best_get();
                if after < before {
                    dry = 0;
                } else {
                    dry += 1;
                }
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
        // Auto: preferred arms get larger slices under the same total budget.
        let arm_slice = if policy == PortfolioPolicy::Auto {
            let mult = crate::methods::regime::arm_slice_multiplier(regime, arms[choice].name());
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
        let threshold = arm_success_threshold(arms[choice], before);
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
            if delta > 0.1 * before.abs().max(1.0) {
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
        let score = if after.is_finite() && before.is_finite() && after < before {
            before - after
        } else {
            0.0
        };
        tpe.record(choice, score);
        states.last_exploratory_ok = false;
    }

    let best_pos_arr = ledger.incumbent(&bounds);
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
        arm_stats: arms
            .iter()
            .zip(posteriors.iter())
            .map(|(arm, posterior)| ArmStat {
                name: arm.name(),
                pulls: posterior.pulls,
                successes: posterior.successes,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::{Rastrigin, StybTang2D};

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
    }

    #[test]
    fn library_exposes_metad_and_tps_for_dim_ge_2() {
        let names = library_arm_names(2, true);
        assert!(names.contains(&"metad"));
        assert!(names.contains(&"tps"));
        let names1 = library_arm_names(1, true);
        assert!(!names1.contains(&"metad"));
        assert!(!names1.contains(&"tps"));
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
            Self {
                bounds: Bounds::new(
                    Array1::from_vec(vec![-5.0, -5.0]),
                    Array1::from_vec(vec![5.0, 5.0]),
                    1e-12,
                ),
                center: Array1::from_vec(vec![1.25, -1.75]),
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
        let first_chain_count = states.gsa.as_ref().expect("gsa state").units.len();

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
        assert_eq!(state.units.len(), first_chain_count);
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
        let weak = Array1::from_vec(vec![-4.0, 4.0]);
        let elite = Array1::from_vec(vec![1.7, -1.1]);
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
}
