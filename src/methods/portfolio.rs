//! Thompson-allocated portfolio over the typed algebra's building blocks.
//!
//! One generic global optimizer with a single knob: the budget. Each
//! building block is an arm of a Bernoulli bandit; a discounted
//! Beta-Bernoulli posterior tracks the probability that one budget
//! slice of an arm improves the incumbent, and Thompson sampling
//! allocates the next slice. A decaying uniform floor `min(1, K/m)` on
//! round `m` keeps every arm scheduled infinitely often, which
//! preserves the QMC restart arm's global convergence guarantee and is
//! the order-optimal floor for the allocation regret.
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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Distribution};

use eindir_core::{AdditiveSurrogate, Bounds, Gradient, Objective, ReducedObjective};

use crate::cool::LogCool;
use crate::exchange::TsallisExchange;
use crate::hmc::{HmcSaSampler, OmelyanIntegrator, QGaussianMomentum};
use crate::methods::bayesian_pilot::{
    fit_laplace, pilot_draws_qmc, LaplacePosterior, PilotObservation, PilotPrior,
};
use crate::methods::gle_langevin::gle_langevin_preconditioned_sa;
use crate::methods::local_polish::{
    projected_gradient_polish, qmc_gsa_global_search, qmc_projected_gradient_polish,
    qmc_trust_region_poll,
};
use crate::methods::parallel_tempering::{geometric_ladder, ParallelTemperingSampler};
use crate::runner::{qmc_skip_from_seed, run_rs, run_rs_variant};

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

    fn record(&self, x: ArrayView1<f64>, value: f64) {
        let mut inner = self.inner.lock().expect("ledger lock");
        if value.is_finite() && value < self.best_get() {
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
            Some(pos) => pos.clone(),
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
        let value = self.inner.eval(x);
        self.ledger.record(x, value);
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
        let value = self.inner.eval(x);
        self.ledger.record(x, value);
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
    fn new(discount: f64) -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
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
        }
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

#[derive(Clone, Debug)]
struct PilotState {
    posterior: LaplacePosterior,
    anchor: Option<Array1<f64>>,
}

#[derive(Default)]
struct ArmStates {
    hop: Option<HopState>,
    de: Option<DeState>,
    pilot: Option<PilotState>,
    surrogate_gen: usize,
    seed_counter: u64,
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
        _ => IMPROVEMENT_RTOL,
    };
    rtol * scale.max(1.0)
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

fn fit_bayesian_pilot<O>(
    obj: &BudgetedObjective<'_, O>,
    ledger: &BudgetLedger,
    rng: &mut StdRng,
    seed: u64,
    pilot_budget: usize,
    min_steps_per_chain: usize,
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
            let prop = bounds.clip(prop.view());
            let prop_val = obj.eval(prop.view());
            steps += 1;
            if prop_val.is_finite() && metropolis(prop_val - cur_val, temp, rng) {
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
            // best ones refined; positive-density restarts carry the
            // global convergence guarantee.
            if let Some(grad) = grad {
                // One third screening, two thirds descent; the single
                // polished start gets the full descent depth, which
                // ill-conditioned valleys need more than breadth.
                let n_starts = (slice / 3).max(4);
                let per_start = slice.saturating_sub(n_starts) / 2;
                if per_start >= 2 {
                    qmc_projected_gradient_polish(
                        obj, grad, n_starts, per_start, seed, 1.0, 1e-8, 1,
                    );
                    return;
                }
            }
            if slice >= 8 {
                let chains = (slice / 8).clamp(2, 4 * dim.max(1));
                qmc_gsa_global_search(obj, slice, seed, chains, 1.0, GSA_Q_V, GSA_Q_A);
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
                let trial = bounds.clip(trial.view());
                let res = projected_gradient_polish(obj, grad, trial, slice / 2, 1.0, 1e-8);
                if !res.best_val.is_finite() {
                    state.step = (state.step * HOP_SHRINK).max(1e-4);
                } else if res.best_val < f_cur || metropolis(res.best_val - f_cur, temp, rng) {
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
            if let Some(grad) = grad {
                if modal_val.is_finite() && modal_val < before_modal && ledger.remaining() >= 4 {
                    projected_gradient_polish(
                        obj,
                        grad,
                        modal_x,
                        ledger.remaining() / 2,
                        1.0,
                        1e-8,
                    );
                }
            }
            // Cool with budget progress so the ladder reaches the cold
            // regime regardless of how often the arm is pulled.
            let progress = ledger.used_get() as f64 / budget.max(1) as f64;
            let exponent = (12.0 * progress) as i32 + states.surrogate_gen as i32;
            let temp = (archive_temp0(ledger) * 0.5_f64.powi(exponent)).max(1e-12);
            let proposals = surr.sample(slice, temp, SURROGATE_GRID, rng);
            let mut f_cur = ledger.best_get();
            for i in 0..proposals.nrows() {
                if ledger.exhausted() {
                    break;
                }
                let trial = bounds.clip(proposals.row(i));
                let ft = obj.eval(trial.view());
                if ft.is_finite() && metropolis(ft - f_cur, temp, rng) {
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
                    states.pilot = fit_bayesian_pilot(
                        obj,
                        ledger,
                        rng,
                        seed,
                        pilot_budget,
                        BAYESIAN_PILOT_MIN_CHAIN_STEPS,
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
            let chains = (slice / 8).clamp(2, 4 * dim.max(1));
            qmc_gsa_global_search(obj, slice, seed, chains, 1.0, GSA_Q_V, GSA_Q_A);
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
                states.pilot = fit_bayesian_pilot(
                    obj,
                    ledger,
                    rng,
                    seed,
                    slice,
                    BAYESIAN_PILOT_MIN_CHAIN_STEPS,
                );
                return;
            }
            let posterior = &states.pilot.as_ref().expect("pilot fitted").posterior;
            let epochs = 16usize;
            let steps = (slice / epochs).max(4);
            let fresh_obj = BudgetedObjective {
                inner: obj.inner,
                ledger,
            };
            if posterior.q_v_map > 1.3 {
                if let Ok(variant) = crate::variant::gsa(
                    fresh_obj,
                    posterior.t_init_map.max(1e-9),
                    posterior.q_v_map.clamp(1.05, 2.95),
                    GSA_Q_A,
                ) {
                    run_rs_variant(variant, epochs, steps, seed);
                }
            } else if let Ok(variant) = crate::variant::boltzmann(
                fresh_obj,
                posterior.t_init_map.max(1e-9),
                (posterior.sigma_map * width_scale).max(1e-9),
            ) {
                run_rs_variant(variant, epochs, steps, seed);
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
                        if w.is_finite() && w > 0.0 {
                            w * w
                        } else {
                            1.0
                        }
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

/// Arms in priority order; the horizon cap truncates from the back, so
/// the core stays active at small budgets and the full library unlocks
/// as the slice count grows.
fn enabled_arms(dim: usize, has_grad: bool, n_slices: usize) -> Vec<ArmKind> {
    let mut arms = vec![ArmKind::Explore];
    if has_grad {
        arms.push(ArmKind::Shift);
        arms.push(ArmKind::Hop);
        arms.push(ArmKind::Gle);
    }
    arms.push(ArmKind::Surrogate);
    arms.push(ArmKind::De);
    arms.push(ArmKind::TrPoll);
    arms.push(ArmKind::Gsa);
    arms.push(ArmKind::Variant);
    arms.push(ArmKind::Pt);
    if has_grad {
        arms.push(ArmKind::Hmc);
        if dim > 2 * REDUCED_K {
            arms.push(ArmKind::Reduced);
        }
    }
    // The posterior needs ROUNDS_PER_ARM pulls per arm to rank them;
    // activate only as many arms as the horizon can rank (the D4 regret
    // grows with K, and a starved arm is worse than an absent one).
    let k_active = (n_slices / ROUNDS_PER_ARM).clamp(4, arms.len());
    arms.truncate(k_active);
    arms
}

/// Runs the portfolio driver under a shared work-unit budget.
///
/// `budget` bounds combined true-objective and native-gradient
/// evaluations and is the driver's only required parameter. `grad`
/// enables the gradient arms and the final polish.
pub fn portfolio_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
) -> PortfolioResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(budget > 0, "budget must be positive");
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    assert!(dim > 0, "objective dimension must be positive");

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
    // two governing numbers.
    let arm_count_all = enabled_arms(dim, grad.is_some(), usize::MAX).len();
    let slice =
        (SLICE_GRAD_EQUIVALENTS * (dim + 1)).max(budget / (ROUNDS_PER_ARM * arm_count_all).max(1));
    let n_slices = (budget / slice.max(1)).max(1);
    let arms = enabled_arms(dim, grad.is_some(), n_slices);
    debug_assert_eq!(arms[0], RESTART_ARM);
    // Posterior memory matches the slice horizon.
    let discount = 1.0 - 1.0 / (n_slices.max(2) as f64);
    let mut posteriors: Vec<ArmPosterior> =
        arms.iter().map(|_| ArmPosterior::new(discount)).collect();
    let mut states = ArmStates::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let k = arms.len();

    let mut round = 0usize;
    loop {
        let remaining = ledger.remaining();
        if remaining < 4 {
            break;
        }
        if remaining < slice {
            // Budget tail: drive the incumbent to projected
            // stationarity at the measurement resolution.
            if let Some(grad) = budgeted_grad.as_ref() {
                projected_gradient_polish(
                    &budgeted_obj,
                    grad,
                    ledger.incumbent(&bounds),
                    (remaining / 2).max(2),
                    1.0,
                    1e-12,
                );
            } else if remaining >= 8 {
                qmc_trust_region_poll(
                    &budgeted_obj,
                    ledger.incumbent(&bounds),
                    remaining,
                    rng.random::<u64>(),
                    0.0,
                    3,
                    0,
                );
            }
            break;
        }
        round += 1;
        // Decaying uniform floor min(1, 1/m): the Beta(1, 1) priors
        // already explore, so the floor only certifies that every arm,
        // including the restart arm, is played infinitely often
        // (sum 1/(mK) diverges); its cumulative cost is ln(n) slices.
        let floor = (1.0 / round as f64).min(1.0);
        let choice = if rng.random::<f64>() < floor {
            rng.random_range(0..k)
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
        let ceiling = ledger.used_get() + slice;
        ledger.cap_set(ceiling.min(budget));
        run_arm(
            arms[choice],
            &budgeted_obj,
            budgeted_grad.as_ref(),
            &ledger,
            &mut states,
            &mut rng,
            slice,
            budget,
        );
        ledger.cap_set(budget);
        let threshold = arm_success_threshold(arms[choice], before);
        let after = ledger.best_get();
        posteriors[choice].update(after.is_finite() && after < before - threshold);
    }

    let best_pos = ledger.incumbent(&bounds).to_vec();
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
    fn budget_is_never_exceeded() {
        let obj = Rastrigin::<6>::new();
        let result = portfolio_optimize(&obj, Some(&obj), 600, 7);
        assert!(result.n_evals + result.n_grads <= 600);
        assert!(result.best_val.is_finite());
    }

    #[test]
    fn budget_respected_without_gradients() {
        let obj = Rastrigin::<4>::new();
        let result = portfolio_optimize::<_, Rastrigin<4>>(&obj, None, 400, 3);
        assert!(result.n_evals <= 400);
        assert_eq!(result.n_grads, 0);
        assert!(result.best_val.is_finite());
    }

    #[test]
    fn portfolio_reaches_styblinski_tang_basin() {
        let obj = StybTang2D::new();
        let result = portfolio_optimize(&obj, Some(&obj), 1200, 11);
        // Global minimum is about -78.332 for the 2D Styblinski-Tang form.
        assert!(
            result.best_val < -78.0,
            "expected global basin, got {}",
            result.best_val
        );
    }

    #[test]
    fn restart_arm_is_always_first() {
        let arms = enabled_arms(5, false, usize::MAX);
        assert_eq!(arms[0], RESTART_ARM);
        let arms = enabled_arms(30, true, usize::MAX);
        assert_eq!(arms[0], RESTART_ARM);
        // The full library is active at a generous horizon.
        assert!(arms.contains(&ArmKind::Reduced));
        assert!(arms.contains(&ArmKind::Hmc));
        assert!(arms.contains(&ArmKind::Pt));
    }

    #[test]
    fn horizon_caps_active_arms() {
        let few = enabled_arms(30, true, 16);
        let many = enabled_arms(30, true, usize::MAX);
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
    fn gradient_horizon_prioritizes_shift_arm() {
        let arms = enabled_arms(6, true, 16);
        assert!(arms.contains(&ArmKind::Shift));
        assert!(arms.contains(&ArmKind::Gle));
        assert!(!arms.contains(&ArmKind::De));
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
