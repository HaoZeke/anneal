//! Thompson-allocated portfolio over the typed algebra's building blocks.
//!
//! One generic global optimizer with a single budget knob. Each building
//! block (QMC-seeded multistart descent, adaptive basin hopping,
//! preconditioned GLE-Langevin, best/1/bin differential evolution,
//! generalized simulated annealing, archive-fit additive-surrogate
//! independence proposals, and shifted-QMC trust-region polls) is an arm
//! of a Bernoulli bandit. A discounted Beta-Bernoulli posterior tracks
//! the probability that one budget slice of an arm improves the
//! incumbent; Thompson sampling allocates the next slice. A probability
//! floor on the QMC restart arm keeps the restart measure scheduled
//! infinitely often, which preserves the restart arm's global
//! convergence guarantee no matter how the posterior concentrates.
//!
//! Work accounting is uniform: every true-objective evaluation and every
//! native-gradient evaluation costs one unit of the shared budget. The
//! additive surrogate is fit from the archive of already-charged
//! evaluations, so its proposals cost only their acceptance tests.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Distribution};

use eindir_core::{AdditiveSurrogate, Bounds, Gradient, Objective};

use crate::methods::gle_langevin::gle_langevin_preconditioned_sa;
use crate::methods::local_polish::{
    projected_gradient_polish, qmc_gsa_global_search, qmc_projected_gradient_polish,
    qmc_trust_region_poll,
};

/// Tuning surface for the portfolio driver. The defaults are uniform
/// across problems; the only per-run input is the budget.
#[derive(Clone, Debug)]
pub struct PortfolioConfig {
    /// Budget slice as a fraction of the total: `budget / slice_divisor`.
    pub slice_divisor: usize,
    /// Lower bound on the slice in units of `dim + 1`.
    pub slice_dim_multiplier: usize,
    /// Absolute lower bound on the slice.
    pub slice_min: usize,
    /// Probability floor for the QMC restart arm.
    pub restart_floor: f64,
    /// Relative incumbent improvement that counts as a slice success.
    pub improvement_rtol: f64,
    /// Absolute incumbent improvement that counts as a slice success.
    pub improvement_atol: f64,
    /// Discount factor for the Beta-Bernoulli posterior.
    pub discount: f64,
    /// Fraction of the budget reserved for the final gradient polish.
    pub final_polish_fraction: f64,
    /// Minimum reserved final-polish budget.
    pub final_polish_min: usize,
    /// Archive capacity for surrogate fitting.
    pub archive_cap: usize,
    /// Fraction of an explore slice spent screening QMC starts.
    pub explore_eval_fraction: f64,
    /// Initial basin-hop step as a fraction of the box width.
    pub hop_initial_step: f64,
    /// Multiplicative step growth after an accepted hop.
    pub hop_step_grow: f64,
    /// Multiplicative step shrink after a rejected hop.
    pub hop_step_shrink: f64,
    /// Chebyshev degree of the archive-fit additive surrogate.
    pub surrogate_degree: usize,
    /// Minimum archived evaluations before the surrogate arm activates.
    pub surrogate_min_archive: usize,
    /// Inverse-CDF grid resolution for surrogate sampling.
    pub surrogate_grid: usize,
    /// Minimum dimension for the GLE arm.
    pub gle_min_dim: usize,
    /// GLE integrator timestep.
    pub gle_dt: f64,
    /// GLE annealing epochs per slice.
    pub gle_n_epochs: usize,
    /// Differential-evolution population floor.
    pub de_pop_min: usize,
    /// Differential-evolution population per dimension.
    pub de_pop_dim_multiplier: usize,
    /// Differential-evolution population cap.
    pub de_pop_max: usize,
    /// Differential-evolution weight floor.
    pub de_weight_min: f64,
    /// Differential-evolution weight span above the floor.
    pub de_weight_span: f64,
    /// Differential-evolution crossover rate.
    pub de_crossover: f64,
    /// GSA initial temperature.
    pub gsa_t_init: f64,
    /// GSA visiting shape parameter.
    pub gsa_q_v: f64,
    /// GSA acceptance shape parameter.
    pub gsa_q_a: f64,
    /// Trust-region poll level count.
    pub tr_levels: usize,
    /// Metropolis temperature floor.
    pub metropolis_floor: f64,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            slice_divisor: 40,
            slice_dim_multiplier: 8,
            slice_min: 32,
            restart_floor: 0.12,
            improvement_rtol: 1e-4,
            improvement_atol: 1e-12,
            discount: 0.97,
            final_polish_fraction: 0.06,
            final_polish_min: 50,
            archive_cap: 8192,
            explore_eval_fraction: 0.35,
            hop_initial_step: 0.25,
            hop_step_grow: 1.3,
            hop_step_shrink: 0.75,
            surrogate_degree: 8,
            surrogate_min_archive: 64,
            surrogate_grid: 65,
            gle_min_dim: 2,
            gle_dt: 0.2,
            gle_n_epochs: 40,
            de_pop_min: 16,
            de_pop_dim_multiplier: 4,
            de_pop_max: 48,
            de_weight_min: 0.5,
            de_weight_span: 0.5,
            de_crossover: 0.7,
            gsa_t_init: 1.0,
            gsa_q_v: 2.62,
            gsa_q_a: 1.7,
            tr_levels: 3,
            metropolis_floor: 1e-12,
        }
    }
}

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
    fn new(budget: usize, archive_cap: usize, dim: usize) -> Self {
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
            archive_cap,
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
    Hop,
    Gle,
    De,
    Gsa,
    Surrogate,
    TrPoll,
}

impl ArmKind {
    fn name(self) -> &'static str {
        match self {
            ArmKind::Explore => "explore",
            ArmKind::Hop => "hop",
            ArmKind::Gle => "gle",
            ArmKind::De => "de",
            ArmKind::Gsa => "gsa",
            ArmKind::Surrogate => "surrogate",
            ArmKind::TrPoll => "tr_poll",
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

#[derive(Default)]
struct ArmStates {
    hop: Option<HopState>,
    de: Option<DeState>,
    surrogate_gen: usize,
    seed_counter: u64,
}

fn metropolis(delta: f64, temp: f64, rng: &mut StdRng, floor: f64) -> bool {
    if delta <= 0.0 {
        return true;
    }
    if !delta.is_finite() {
        return false;
    }
    rng.random::<f64>() < (-delta / temp.max(floor)).exp()
}

fn ladder_temperature(temp0: f64, generation: usize) -> f64 {
    (temp0 * std::f64::consts::LN_2 / ((generation + 2) as f64).ln()).max(1e-9)
}

fn archive_temp0(ledger: &BudgetLedger) -> f64 {
    let inner = ledger.inner.lock().expect("ledger lock");
    let finite: Vec<f64> = inner
        .archive_y
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if finite.len() < 2 {
        return 1.0;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / finite.len() as f64;
    var.sqrt().max(1e-6)
}

#[allow(clippy::too_many_arguments)]
fn run_arm<O, G>(
    arm: ArmKind,
    obj: &BudgetedObjective<'_, O>,
    grad: Option<&BudgetedGradient<'_, G>>,
    ledger: &BudgetLedger,
    states: &mut ArmStates,
    config: &PortfolioConfig,
    rng: &mut StdRng,
    slice: usize,
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
            if let Some(grad) = grad {
                let n_starts = ((slice as f64 * config.explore_eval_fraction) as usize).max(4);
                let top_k = 2usize.min(n_starts);
                let per_start = slice.saturating_sub(n_starts) / (2 * top_k).max(1);
                if per_start >= 2 {
                    qmc_projected_gradient_polish(
                        obj, grad, n_starts, per_start, seed, 1.0, 1e-8, top_k,
                    );
                    return;
                }
            }
            let chains = (slice / 8).clamp(2, 4 * dim.max(1));
            if slice >= 8 {
                qmc_gsa_global_search(
                    obj,
                    slice,
                    seed,
                    chains,
                    config.gsa_t_init,
                    config.gsa_q_v,
                    config.gsa_q_a,
                );
            }
        }
        ArmKind::Hop => {
            let Some(grad) = grad else { return };
            let state = states.hop.get_or_insert_with(|| HopState {
                step: config.hop_initial_step,
                x_cur: None,
                f_cur: f64::INFINITY,
                generation: 0,
            });
            let x_init = state
                .x_cur
                .clone()
                .unwrap_or_else(|| ledger.incumbent(&bounds));
            let mut x_cur = x_init;
            let mut f_cur = state.f_cur.min(ledger.best_get());
            let temp = ladder_temperature(archive_temp0(ledger), state.generation);
            let n_hops = 3usize;
            let per_hop = (slice / n_hops).max(4);
            let width = &bounds.high - &bounds.low;
            for _ in 0..n_hops {
                if ledger.remaining() < 4 {
                    break;
                }
                let mut trial = x_cur.clone();
                for j in 0..dim {
                    let w = if width[j] > 0.0 { width[j] } else { 1.0 };
                    let noise: f64 = rand_distr::StandardNormal.sample(rng);
                    trial[j] += state.step * w * noise;
                }
                let trial = bounds.clip(trial.view());
                let res = projected_gradient_polish(obj, grad, trial, per_hop / 2, 1.0, 1e-8);
                if !res.best_val.is_finite() {
                    state.step = (state.step * config.hop_step_shrink).max(1e-4);
                    continue;
                }
                if res.best_val < f_cur
                    || metropolis(res.best_val - f_cur, temp, rng, config.metropolis_floor)
                {
                    x_cur = res.best_pos;
                    f_cur = res.best_val;
                    state.step = (state.step * config.hop_step_grow).min(1.0);
                } else {
                    state.step = (state.step * config.hop_step_shrink).max(1e-4);
                }
            }
            state.x_cur = Some(x_cur);
            state.f_cur = f_cur;
            state.generation += 1;
        }
        ArmKind::Gle => {
            let Some(grad) = grad else { return };
            let maxf = slice / 2;
            if maxf < 4 {
                return;
            }
            gle_langevin_preconditioned_sa(
                obj,
                grad,
                seed,
                maxf,
                config.gle_dt,
                config.gle_n_epochs,
                Some(ledger.incumbent(&bounds)),
                None,
            );
        }
        ArmKind::De => {
            if states.de.is_none() {
                let pop_size = (config.de_pop_dim_multiplier * dim)
                    .clamp(config.de_pop_min, config.de_pop_max)
                    .min(slice.max(4));
                let points = eindir_core::shifted_low_discrepancy_points(
                    &bounds,
                    pop_size,
                    crate::runner::qmc_skip_from_seed(seed),
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
                let weight = config.de_weight_min + config.de_weight_span * rng.random::<f64>();
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
                        if j == forced || rng.random::<f64>() < config.de_crossover {
                            trial[j] =
                                best_x[j] + weight * (state.pop[r0][j] - state.pop[r1][j]);
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
        ArmKind::Gsa => {
            if slice < 8 {
                return;
            }
            let chains = (slice / 8).clamp(2, 4 * dim.max(1));
            qmc_gsa_global_search(
                obj,
                slice,
                seed,
                chains,
                config.gsa_t_init,
                config.gsa_q_v,
                config.gsa_q_a,
            );
        }
        ArmKind::Surrogate => {
            let min_points = config.surrogate_min_archive.max(4 * dim);
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
                config.surrogate_degree,
            );
            // The modal point (per-coordinate argmin, the T -> 0 limit of
            // the tempered marginals) tests the surrogate's global
            // candidate at the cost of one evaluation; for separable
            // objectives it is the global minimizer once the fit settles.
            let modal = surr.sample(1, 1e-15, config.surrogate_grid, rng);
            let before_modal = ledger.best_get();
            let modal_x = bounds.clip(modal.row(0));
            let modal_val = obj.eval(modal_x.view());
            if let Some(grad) = grad {
                if modal_val.is_finite()
                    && modal_val < before_modal
                    && ledger.remaining() >= 4
                {
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
            let progress = ledger.used_get() as f64 / ledger.cap_get().max(1) as f64;
            let exponent = (12.0 * progress) as i32 + states.surrogate_gen as i32;
            let temp = (archive_temp0(ledger) * 0.5_f64.powi(exponent)).max(1e-12);
            let proposals = surr.sample(slice, temp, config.surrogate_grid, rng);
            let mut f_cur = ledger.best_get();
            for i in 0..proposals.nrows() {
                if ledger.exhausted() {
                    break;
                }
                let trial = bounds.clip(proposals.row(i));
                let ft = obj.eval(trial.view());
                if ft.is_finite() && metropolis(ft - f_cur, temp, rng, config.metropolis_floor) {
                    f_cur = ft;
                }
            }
            states.surrogate_gen += 1;
        }
        ArmKind::TrPoll => {
            if slice < 8 {
                return;
            }
            qmc_trust_region_poll(
                obj,
                ledger.incumbent(&bounds),
                slice,
                seed,
                0.0,
                config.tr_levels,
                0,
            );
        }
    }
}

fn enabled_arms(dim: usize, has_grad: bool, config: &PortfolioConfig) -> Vec<ArmKind> {
    let mut arms = vec![ArmKind::Explore];
    if has_grad {
        arms.push(ArmKind::Hop);
    }
    arms.push(ArmKind::De);
    if has_grad && dim >= config.gle_min_dim {
        arms.push(ArmKind::Gle);
    }
    arms.push(ArmKind::Gsa);
    arms.push(ArmKind::Surrogate);
    arms.push(ArmKind::TrPoll);
    arms
}

fn slice_budget(budget: usize, dim: usize, config: &PortfolioConfig) -> usize {
    (budget / config.slice_divisor.max(1))
        .max(config.slice_dim_multiplier * (dim + 1))
        .max(config.slice_min)
}

/// Runs the portfolio driver under a shared work-unit budget.
///
/// `budget` bounds combined true-objective and native-gradient
/// evaluations. `grad` enables the gradient arms and the final polish.
pub fn portfolio_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    budget: usize,
    seed: u64,
    config: &PortfolioConfig,
) -> PortfolioResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(budget > 0, "budget must be positive");
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    assert!(dim > 0, "objective dimension must be positive");

    let ledger = BudgetLedger::new(budget, config.archive_cap, dim);
    let budgeted_obj = BudgetedObjective {
        inner: obj,
        ledger: &ledger,
    };
    let budgeted_grad = grad.map(|g| BudgetedGradient {
        inner: g,
        ledger: &ledger,
    });

    let final_polish = if grad.is_some() {
        ((budget as f64 * config.final_polish_fraction) as usize).max(config.final_polish_min)
    } else {
        0
    };
    let main_ceiling = budget.saturating_sub(final_polish);
    let slice = slice_budget(budget, dim, config);
    let arms = enabled_arms(dim, grad.is_some(), config);
    let mut posteriors: Vec<ArmPosterior> = arms
        .iter()
        .map(|_| ArmPosterior::new(config.discount))
        .collect();
    let mut states = ArmStates::default();
    let mut rng = StdRng::seed_from_u64(seed);

    let run_slice = |arm_idx: usize,
                         states: &mut ArmStates,
                         posteriors: &mut Vec<ArmPosterior>,
                         rng: &mut StdRng| {
        let before = ledger.best_get();
        let ceiling = (ledger.used_get() + slice).min(main_ceiling);
        ledger.cap_set(ceiling);
        let this_slice = ceiling.saturating_sub(ledger.used_get());
        if this_slice == 0 {
            ledger.cap_set(budget);
            return;
        }
        run_arm(
            arms[arm_idx],
            &budgeted_obj,
            budgeted_grad.as_ref(),
            &ledger,
            states,
            config,
            rng,
            this_slice,
        );
        ledger.cap_set(budget);
        let scale = if before.is_finite() { before.abs() } else { 1.0 };
        let threshold = config.improvement_atol + config.improvement_rtol * scale.max(1.0);
        let after = ledger.best_get();
        posteriors[arm_idx].update(after.is_finite() && after < before - threshold);
    };

    // Warm start: one slice per arm in declaration order.
    for idx in 0..arms.len() {
        if ledger.used_get() + 4 > main_ceiling {
            break;
        }
        run_slice(idx, &mut states, &mut posteriors, &mut rng);
    }

    // Thompson allocation with the restart-arm floor.
    let restart_idx = arms
        .iter()
        .position(|a| *a == RESTART_ARM)
        .expect("restart arm is always enabled");
    while ledger.used_get() + 4 <= main_ceiling {
        let choice = if rng.random::<f64>() < config.restart_floor {
            restart_idx
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
        run_slice(choice, &mut states, &mut posteriors, &mut rng);
    }

    // Final polish from the incumbent with the reserved budget.
    if let Some(grad) = budgeted_grad.as_ref() {
        let remaining = ledger.remaining();
        if remaining >= 4 {
            projected_gradient_polish(
                &budgeted_obj,
                grad,
                ledger.incumbent(&bounds),
                remaining / 2,
                1.0,
                1e-8,
            );
        }
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
        let config = PortfolioConfig::default();
        let result = portfolio_optimize(&obj, Some(&obj), 600, 7, &config);
        assert!(result.n_evals + result.n_grads <= 600);
        assert!(result.best_val.is_finite());
    }

    #[test]
    fn budget_respected_without_gradients() {
        let obj = Rastrigin::<4>::new();
        let config = PortfolioConfig::default();
        let result = portfolio_optimize::<_, Rastrigin<4>>(&obj, None, 400, 3, &config);
        assert!(result.n_evals <= 400);
        assert_eq!(result.n_grads, 0);
        assert!(result.best_val.is_finite());
    }

    #[test]
    fn portfolio_reaches_styblinski_tang_basin() {
        let obj = StybTang2D::new();
        let config = PortfolioConfig::default();
        let result = portfolio_optimize(&obj, Some(&obj), 1200, 11, &config);
        // Global minimum is about -78.332 for the 2D Styblinski-Tang form.
        assert!(
            result.best_val < -78.0,
            "expected global basin, got {}",
            result.best_val
        );
    }

    #[test]
    fn restart_arm_is_always_first() {
        let config = PortfolioConfig::default();
        let arms = enabled_arms(5, false, &config);
        assert_eq!(arms[0], RESTART_ARM);
        let arms = enabled_arms(5, true, &config);
        assert_eq!(arms[0], RESTART_ARM);
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
}
