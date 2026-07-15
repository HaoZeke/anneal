//! Bounded local refinement used as the deterministic polish stage after a
//! stochastic annealing or pilot phase.

use eindir_core::Bounds;
use eindir_core::Gradient;
use eindir_core::Objective;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::accept::{AcceptRule, TsallisAccept};
use crate::cool::{Cooling, TsallisCool};
use crate::movekernel::{MoveKernel, TsallisVisit};

const ARMIJO_SUFFICIENT_DECREASE: f64 = 1e-4;
const BACKTRACK_SHRINK: f64 = 0.5;
const MIN_BACKTRACK_STEP: f64 = f64::EPSILON;
const FULL_LINE_SEARCH_STEP: f64 = 1.0;
const TRUST_REGION_RADIUS_DIMENSION_SCALE: f64 = 2.0;

/// Result of bounded local refinement.
#[derive(Clone, Debug)]
pub struct LocalPolishResult {
    /// Best point found by the local refinement.
    pub best_pos: Array1<f64>,
    /// Objective value at `best_pos`.
    pub best_val: f64,
    /// Objective evaluations consumed by the refinement.
    pub n_evals: usize,
    /// Gradient evaluations consumed by the refinement.
    pub n_grads: usize,
    /// Norm of the projected gradient at `best_pos`.
    pub projected_grad_norm: f64,
    /// Whether the projected gradient satisfies the requested tolerance.
    pub projected_stationary: bool,
}

/// Result of QMC-seeded bounded local refinement.
#[derive(Clone, Debug)]
pub struct QmcPolishResult {
    /// Best point found across the screened low-discrepancy starts.
    pub best_pos: Array1<f64>,
    /// Objective value at `best_pos`.
    pub best_val: f64,
    /// Objective evaluations consumed by screening and refinement.
    pub n_evals: usize,
    /// Gradient evaluations consumed by refinement.
    pub n_grads: usize,
    /// Number of low-discrepancy starts screened.
    pub n_starts: usize,
    /// Number of screened starts sent to local refinement.
    pub n_polished: usize,
    /// Objective values returned by each local refinement, in polish order.
    pub polished_values: Vec<f64>,
    /// Projected-gradient norms returned by each local refinement, in polish order.
    pub polished_projected_grad_norms: Vec<f64>,
    /// Stationarity flags returned by each local refinement, in polish order.
    pub polished_stationary: Vec<bool>,
}

fn sample_index_excluding<R: Rng + ?Sized>(rng: &mut R, len: usize, excluded: usize) -> usize {
    let mut idx = rng.random_range(0..(len - 1));
    if idx >= excluded {
        idx += 1;
    }
    idx
}

fn sample_two_indices_excluding<R: Rng + ?Sized>(
    rng: &mut R,
    len: usize,
    excluded: usize,
) -> (usize, usize) {
    let first = sample_index_excluding(rng, len, excluded);
    let mut second = sample_index_excluding(rng, len, excluded);
    while second == first {
        second = sample_index_excluding(rng, len, excluded);
    }
    (first, second)
}

fn projected_gradient(
    x: &Array1<f64>,
    grad: &Array1<f64>,
    low: &Array1<f64>,
    high: &Array1<f64>,
) -> Array1<f64> {
    let mut out = grad.clone();
    for i in 0..out.len() {
        if (x[i] <= low[i] && grad[i] > 0.0) || (x[i] >= high[i] && grad[i] < 0.0) {
            out[i] = 0.0;
        }
    }
    out
}

/// Number of correction pairs retained by the limited-memory inverse-Hessian.
const LBFGS_MEMORY: usize = 10;

/// Limited-memory BFGS inverse-Hessian, stored as a ring of correction pairs.
///
/// A dense inverse Hessian costs `dim * dim` storage and `dim * dim` work per
/// step, which is 9 GB and a billion operations on a 34000-variable problem.
/// The two-loop recursion reproduces the same quasi-Newton direction from the
/// last `LBFGS_MEMORY` pairs in `O(dim * LBFGS_MEMORY)` time and storage, so
/// the polish scales to native CUTEst sizes.
#[derive(Default)]
struct LbfgsMemory {
    /// `(s, y, 1 / y.s)` triples, oldest first.
    pairs: std::collections::VecDeque<(Array1<f64>, Array1<f64>, f64)>,
    /// Initial-Hessian scale `gamma = (s.y) / (y.y)` from the newest pair.
    gamma: f64,
}

impl LbfgsMemory {
    fn new(step0: f64) -> Self {
        Self {
            pairs: std::collections::VecDeque::with_capacity(LBFGS_MEMORY),
            gamma: step0,
        }
    }

    fn reset(&mut self, step0: f64) {
        self.pairs.clear();
        self.gamma = step0;
    }

    /// Records a correction pair, rejecting those that violate the curvature
    /// condition. Returns whether the pair was accepted.
    fn push(&mut self, s: Array1<f64>, y: Array1<f64>) -> bool {
        let ys = y.dot(&s);
        let s_norm = vector_norm(&s);
        let y_norm = vector_norm(&y);
        let curvature_floor = f64::EPSILON.sqrt() * s_norm * y_norm;
        if !ys.is_finite() || ys <= curvature_floor {
            return false;
        }
        let yy = y.dot(&y);
        if yy > 0.0 && yy.is_finite() {
            self.gamma = ys / yy;
        }
        if self.pairs.len() == LBFGS_MEMORY {
            self.pairs.pop_front();
        }
        self.pairs.push_back((s, y, 1.0 / ys));
        true
    }

    /// Two-loop recursion: returns the inverse-Hessian times `-pgrad`.
    fn descent(&self, pgrad: &Array1<f64>) -> Array1<f64> {
        let mut q = pgrad.clone();
        let mut alphas = Vec::with_capacity(self.pairs.len());
        for (s, y, rho) in self.pairs.iter().rev() {
            let alpha = rho * s.dot(&q);
            q.scaled_add(-alpha, y);
            alphas.push(alpha);
        }
        q.mapv_inplace(|v| v * self.gamma);
        for ((s, y, rho), alpha) in self.pairs.iter().zip(alphas.iter().rev()) {
            let beta = rho * y.dot(&q);
            q.scaled_add(alpha - beta, s);
        }
        -q
    }
}

fn active_descent_direction(
    memory: &LbfgsMemory,
    x: &Array1<f64>,
    pgrad: &Array1<f64>,
    low: &Array1<f64>,
    high: &Array1<f64>,
) -> Array1<f64> {
    let mut direction = memory.descent(pgrad);
    apply_active_bounds(&mut direction, x, low, high);
    if pgrad.dot(&direction) < 0.0 {
        return direction;
    }

    let mut fallback = -pgrad;
    apply_active_bounds(&mut fallback, x, low, high);
    fallback
}

fn vector_norm(x: &Array1<f64>) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn projected_grad_norm<G>(
    gradient: &G,
    x: &Array1<f64>,
    low: &Array1<f64>,
    high: &Array1<f64>,
) -> Option<f64>
where
    G: Gradient<f64>,
{
    let grad = gradient.grad(x.view());
    if grad.len() != x.len() || grad.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let pgrad = projected_gradient(x, &grad, low, high);
    let norm = vector_norm(&pgrad);
    norm.is_finite().then_some(norm)
}

fn box_diagonal(low: &Array1<f64>, high: &Array1<f64>) -> f64 {
    low.iter()
        .zip(high.iter())
        .map(|(lo, hi)| {
            let width = hi - lo;
            width * width
        })
        .sum::<f64>()
        .sqrt()
}

fn unit_coordinate(pos: f64, low: f64, high: f64) -> f64 {
    let width = high - low;
    if width > 0.0 && width.is_finite() {
        ((pos - low) / width).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn unit_to_box(unit: &Array1<f64>, low: &Array1<f64>, high: &Array1<f64>) -> Array1<f64> {
    Array1::from_iter(
        unit.iter()
            .enumerate()
            .map(|(axis, value)| low[axis] + (high[axis] - low[axis]) * (*value).clamp(0.0, 1.0)),
    )
}

fn initial_line_search_step(direction: &Array1<f64>, low: &Array1<f64>, high: &Array1<f64>) -> f64 {
    let direction_norm = vector_norm(direction);
    let diagonal = box_diagonal(low, high);
    if direction_norm.is_finite() && diagonal.is_finite() && direction_norm > diagonal {
        (diagonal / direction_norm).max(MIN_BACKTRACK_STEP)
    } else {
        FULL_LINE_SEARCH_STEP
    }
}

fn line_search_trial_limit(dim: usize) -> usize {
    dim.saturating_add((f64::MANTISSA_DIGITS as usize).div_ceil(2))
        .max(1)
}

fn apply_active_bounds(
    direction: &mut Array1<f64>,
    x: &Array1<f64>,
    low: &Array1<f64>,
    high: &Array1<f64>,
) {
    for i in 0..direction.len() {
        if (x[i] <= low[i] && direction[i] < 0.0) || (x[i] >= high[i] && direction[i] > 0.0) {
            direction[i] = 0.0;
        }
    }
}

/// Refine a point with bounded quasi-Newton backtracking inside objective bounds.
pub fn projected_gradient_polish<O, G>(
    obj: &O,
    gradient: &G,
    x0: Array1<f64>,
    max_fevals: usize,
    step0: f64,
    grad_tol: f64,
) -> LocalPolishResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(max_fevals > 0, "max_fevals must be positive");
    assert!(step0 > 0.0, "step0 must be positive");
    assert!(grad_tol >= 0.0, "grad_tol must be non-negative");

    let bounds = obj.bounds();
    let low = &bounds.low;
    let high = &bounds.high;
    let mut x = bounds.clip(x0.view());
    let mut value = obj.eval(x.view());
    let mut n_evals = 1usize;
    let mut n_grads = 0usize;
    let mut best_pos = x.clone();
    let mut best_val = value;
    let mut memory = LbfgsMemory::new(step0);
    let mut prev_x = None;
    let mut prev_pgrad = None;
    let mut final_projected_grad_norm = f64::INFINITY;
    let mut final_grad_matches_x = false;

    while n_evals < max_fevals {
        let grad = gradient.grad(x.view());
        n_grads += 1;
        if grad.len() != x.len() || grad.iter().any(|v| !v.is_finite()) {
            break;
        }
        let pgrad = projected_gradient(&x, &grad, low, high);
        final_projected_grad_norm = vector_norm(&pgrad);
        final_grad_matches_x = true;
        if final_projected_grad_norm <= grad_tol {
            break;
        }
        if let (Some(px), Some(pg)) = (prev_x.take(), prev_pgrad.take()) {
            let s = &x - &px;
            let y = &pgrad - &pg;
            memory.push(s, y);
        }
        let mut accepted = false;
        let mut direction = active_descent_direction(&memory, &x, &pgrad, low, high);
        let mut fallback_attempted = false;
        loop {
            let directional_decrease = -pgrad.dot(&direction);
            if !directional_decrease.is_finite() || directional_decrease <= 0.0 {
                break;
            }

            let mut alpha = initial_line_search_step(&direction, low, high);
            let mut trials = 0usize;
            let max_trials = line_search_trial_limit(x.len());
            while n_evals < max_fevals && alpha > MIN_BACKTRACK_STEP && trials < max_trials {
                trials += 1;
                let trial = bounds.clip((&x + &(direction.mapv(|v| alpha * v))).view());
                if trial
                    .iter()
                    .zip(x.iter())
                    .all(|(a, b)| (*a - *b).abs() <= f64::EPSILON)
                {
                    alpha *= BACKTRACK_SHRINK;
                    continue;
                }
                let trial_value = obj.eval(trial.view());
                n_evals += 1;
                if trial_value.is_finite()
                    && trial_value
                        <= value - ARMIJO_SUFFICIENT_DECREASE * alpha * directional_decrease
                {
                    prev_x = Some(x.clone());
                    prev_pgrad = Some(pgrad.clone());
                    x = trial;
                    value = trial_value;
                    if trial_value < best_val {
                        best_val = trial_value;
                        best_pos = x.clone();
                    }
                    final_grad_matches_x = false;
                    accepted = true;
                    break;
                }
                alpha *= BACKTRACK_SHRINK;
            }
            if accepted || fallback_attempted {
                break;
            }
            let mut fallback = -&pgrad;
            apply_active_bounds(&mut fallback, &x, low, high);
            if fallback
                .iter()
                .zip(direction.iter())
                .all(|(a, b)| (*a - *b).abs() <= f64::EPSILON)
            {
                break;
            }
            memory.reset(step0);
            direction = fallback;
            fallback_attempted = true;
        }
        if !accepted {
            break;
        }
    }

    if !final_grad_matches_x {
        n_grads += 1;
        final_projected_grad_norm =
            projected_grad_norm(gradient, &best_pos, low, high).unwrap_or(f64::INFINITY);
    }
    let projected_stationary =
        final_projected_grad_norm.is_finite() && final_projected_grad_norm <= grad_tol;

    LocalPolishResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
        projected_grad_norm: final_projected_grad_norm,
        projected_stationary,
    }
}

/// Refine low-discrepancy starts with bounded quasi-Newton polish.
#[allow(clippy::too_many_arguments)]
pub fn qmc_projected_gradient_polish<O, G>(
    obj: &O,
    gradient: &G,
    n_starts: usize,
    max_fevals_per_start: usize,
    seed: u64,
    step0: f64,
    grad_tol: f64,
    top_k: usize,
) -> QmcPolishResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(n_starts > 0, "n_starts must be positive");
    assert!(
        max_fevals_per_start > 0,
        "max_fevals_per_start must be positive"
    );

    let bounds = obj.bounds();
    let starts = eindir_core::boundary_anchored_low_discrepancy_points(
        bounds,
        n_starts,
        crate::runner::qmc_skip_from_seed(seed),
    );
    // Clip starts into a dense matrix, then batch-evaluate in parallel.
    let mut clipped = Array2::<f64>::zeros((n_starts, bounds.dims));
    for (i, start) in starts.outer_iter().enumerate() {
        let pos = bounds.clip(start);
        clipped.row_mut(i).assign(&pos);
    }
    // Trait eval_batch: Python overrides for single-GIL / process-pool walkers;
    // native types may use Rayon in their own override. Never Rayon over
    // Python::attach here (GIL deadlock).
    let values = obj.eval_batch(clipped.view());
    let n_evals_screen = values.len();
    let mut screened: Vec<(f64, Array1<f64>)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, &value)| {
            if value.is_finite() {
                Some((value, clipped.row(i).to_owned()))
            } else {
                None
            }
        })
        .collect();
    screened.sort_by(|left, right| left.0.total_cmp(&right.0));
    let (mut best_pos, mut best_val) = if let Some((v, p)) = screened.first() {
        (p.clone(), *v)
    } else {
        (bounds.clip(starts.row(0)), f64::INFINITY)
    };

    let polish_limit = if top_k == 0 {
        screened.len()
    } else {
        top_k.min(screened.len())
    };
    // Serial polish of top-k: Python gradients cannot run under Rayon (GIL).
    let mut n_evals = n_evals_screen;
    let mut n_grads = 0usize;
    let mut n_polished = 0usize;
    let mut polished_values = Vec::with_capacity(polish_limit);
    let mut polished_projected_grad_norms = Vec::with_capacity(polish_limit);
    let mut polished_stationary = Vec::with_capacity(polish_limit);
    for (_value, start) in screened.into_iter().take(polish_limit) {
        let result =
            projected_gradient_polish(obj, gradient, start, max_fevals_per_start, step0, grad_tol);
        n_evals += result.n_evals;
        n_grads += result.n_grads;
        n_polished += 1;
        polished_values.push(result.best_val);
        polished_projected_grad_norms.push(result.projected_grad_norm);
        polished_stationary.push(result.projected_stationary);
        if result.best_val.is_finite() && result.best_val < best_val {
            best_val = result.best_val;
            best_pos = result.best_pos;
        }
    }

    QmcPolishResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
        n_starts,
        n_polished,
        polished_values,
        polished_projected_grad_norms,
        polished_stationary,
    }
}

/// Run a QMC-initialized `best/1/bin` differential-evolution scout.
pub fn qmc_best1bin_scout<O>(
    obj: &O,
    max_evals: usize,
    seed: u64,
    population_size: usize,
    weight_min: f64,
    weight_span: f64,
    crossover_rate: f64,
) -> QmcPolishResult
where
    O: Objective<f64>,
{
    assert!(max_evals > 0, "max_evals must be positive");
    assert!(population_size >= 4, "population_size must be at least 4");
    assert!(
        max_evals >= population_size,
        "max_evals must cover the initial population"
    );
    assert!(
        weight_min.is_finite() && weight_min >= 0.0,
        "weight_min must be finite and non-negative"
    );
    assert!(
        weight_span.is_finite() && weight_span >= 0.0,
        "weight_span must be finite and non-negative"
    );
    assert!(
        crossover_rate.is_finite() && (0.0..=1.0).contains(&crossover_rate),
        "crossover_rate must be in [0, 1]"
    );

    let bounds = obj.bounds();
    let dim = bounds.dims;
    assert!(dim > 0, "objective dimension must be positive");
    let mut rng = StdRng::seed_from_u64(seed);
    let starts = eindir_core::shifted_low_discrepancy_points(
        bounds,
        population_size,
        crate::runner::qmc_skip_from_seed(seed),
        seed,
    );
    let mut population: Vec<Array1<f64>> = starts
        .outer_iter()
        .map(|point| bounds.clip(point))
        .collect();
    let mut values = Vec::with_capacity(population_size);
    let mut best_pos = population[0].clone();
    let mut best_val = f64::INFINITY;
    let mut n_evals = 0usize;

    for point in &population {
        let value = obj.eval(point.view());
        n_evals += 1;
        values.push(value);
        if value.is_finite() && value < best_val {
            best_val = value;
            best_pos = point.clone();
        }
    }

    while n_evals < max_evals {
        let weight = weight_min + weight_span * rng.random::<f64>();
        for slot in 0..population.len() {
            if n_evals >= max_evals {
                break;
            }
            let (left, right) = sample_two_indices_excluding(&mut rng, population.len(), slot);
            let mut mutant = best_pos.clone();
            for axis in 0..dim {
                mutant[axis] += weight * (population[left][axis] - population[right][axis]);
            }
            let forced_axis = rng.random_range(0..dim);
            let mut trial = population[slot].clone();
            for axis in 0..dim {
                if axis == forced_axis || rng.random::<f64>() < crossover_rate {
                    trial[axis] = mutant[axis];
                }
            }
            let trial = bounds.clip(trial.view());
            let value = obj.eval(trial.view());
            n_evals += 1;
            if value.is_finite() && (!values[slot].is_finite() || value < values[slot]) {
                population[slot] = trial.clone();
                values[slot] = value;
                if value < best_val {
                    best_val = value;
                    best_pos = trial;
                }
            }
        }
    }

    QmcPolishResult {
        best_pos,
        best_val,
        n_evals,
        n_grads: 0,
        n_starts: population_size,
        n_polished: 0,
        polished_values: Vec::new(),
        polished_projected_grad_norms: Vec::new(),
        polished_stationary: Vec::new(),
    }
}

/// Run bounded QMC-initialized generalized simulated annealing.
pub fn qmc_gsa_global_search<O>(
    obj: &O,
    max_evals: usize,
    seed: u64,
    n_chains: usize,
    t_init: f64,
    q_v: f64,
    q_a: f64,
) -> QmcPolishResult
where
    O: Objective<f64>,
{
    assert!(max_evals > 0, "max_evals must be positive");
    assert!(n_chains > 0, "n_chains must be positive");
    assert!(
        t_init.is_finite() && t_init > 0.0,
        "t_init must be finite and positive"
    );
    assert!(
        q_v.is_finite() && q_v > 1.0 && q_v < 3.0,
        "q_v must lie in (1, 3)"
    );
    assert!(q_a.is_finite(), "q_a must be finite");

    let bounds = obj.bounds();
    let dim = bounds.dims;
    assert!(dim > 0, "objective dimension must be positive");
    let chain_count = n_chains.min(max_evals).max(1);
    let starts = eindir_core::shifted_low_discrepancy_points(
        bounds,
        chain_count,
        crate::runner::qmc_skip_from_seed(seed),
        seed,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let cooling = TsallisCool::new(t_init, q_v);
    let visit = TsallisVisit::new(q_v);
    let accept = TsallisAccept::new(q_a);
    let mut units = Vec::with_capacity(chain_count);
    let mut values = Vec::with_capacity(chain_count);
    let mut best_pos = bounds.clip(starts.row(0));
    let mut best_val = f64::INFINITY;
    let mut n_evals = 0usize;

    for start in starts.outer_iter() {
        let pos = bounds.clip(start);
        let unit = Array1::from_iter(
            (0..dim).map(|axis| unit_coordinate(pos[axis], bounds.low[axis], bounds.high[axis])),
        );
        let value = obj.eval(pos.view());
        n_evals += 1;
        if value.is_finite() && value < best_val {
            best_val = value;
            best_pos = pos.clone();
        }
        units.push(unit);
        values.push(value);
    }

    let refine_budget = max_evals / dim.saturating_add(1);
    let global_budget = chain_count.max(max_evals.saturating_sub(refine_budget));
    let mut epoch = 0usize;
    while n_evals < global_budget {
        let temp = cooling.temperature(epoch);
        for chain in 0..chain_count {
            if n_evals >= global_budget {
                break;
            }
            let proposal_unit = visit
                .propose(units[chain].view(), temp, &mut rng)
                .mapv(|value| value.clamp(0.0, 1.0));
            let proposal_pos = unit_to_box(&proposal_unit, &bounds.low, &bounds.high);
            let proposal_val = obj.eval(proposal_pos.view());
            n_evals += 1;
            if proposal_val.is_finite() && proposal_val < best_val {
                best_val = proposal_val;
                best_pos = proposal_pos.clone();
            }
            let accepted = if !proposal_val.is_finite() {
                false
            } else if !values[chain].is_finite() {
                true
            } else {
                let probability = accept
                    .accept_prob(proposal_val - values[chain], temp)
                    .clamp(0.0, 1.0);
                rng.random::<f64>() < probability
            };
            if accepted {
                units[chain] = proposal_unit;
                values[chain] = proposal_val;
            }
        }
        epoch += 1;
    }
    if n_evals < max_evals && best_val.is_finite() {
        let trust_seed = seed
            .wrapping_add(max_evals as u64)
            .wrapping_add((dim * chain_count) as u64);
        let trust = qmc_trust_region_poll(
            obj,
            best_pos.clone(),
            max_evals - n_evals,
            trust_seed,
            0.0,
            dim.max(1),
            0,
        );
        n_evals += trust.n_evals;
        if trust.best_val.is_finite() && trust.best_val < best_val {
            best_val = trust.best_val;
            best_pos = trust.best_pos;
        }
    }

    QmcPolishResult {
        best_pos,
        best_val,
        n_evals,
        n_grads: 0,
        n_starts: chain_count,
        n_polished: 0,
        polished_values: Vec::new(),
        polished_projected_grad_norms: Vec::new(),
        polished_stationary: Vec::new(),
    }
}

/// Poll a local trust region with shifted low-discrepancy batches.
pub fn qmc_trust_region_poll<O>(
    obj: &O,
    center: Array1<f64>,
    max_evals: usize,
    seed: u64,
    radius_fraction: f64,
    n_levels: usize,
    points_per_level: usize,
) -> QmcPolishResult
where
    O: Objective<f64>,
{
    assert!(max_evals > 0, "max_evals must be positive");
    assert!(
        radius_fraction.is_finite() && radius_fraction >= 0.0,
        "radius_fraction must be finite and non-negative"
    );
    let bounds = obj.bounds();
    assert_eq!(
        center.len(),
        bounds.dims,
        "center dimension must match objective bounds"
    );

    let levels = n_levels.max(1);
    let dim_scale = (bounds.dims * bounds.dims).max(1) as f64;
    let base_radius_fraction = if radius_fraction > 0.0 {
        radius_fraction
    } else {
        1.0 / (TRUST_REGION_RADIUS_DIMENSION_SCALE * dim_scale)
    };
    let inferred_points_per_level = max_evals.saturating_sub(1).div_ceil(levels).max(1);
    let batch_size = if points_per_level == 0 {
        inferred_points_per_level
    } else {
        points_per_level
    };

    let mut n_evals = 0usize;
    let mut current_center = bounds.clip(center.view());
    let mut best_pos = current_center.clone();
    let mut best_val = obj.eval(best_pos.view());
    n_evals += 1;

    for level in 0..levels {
        if n_evals >= max_evals {
            break;
        }
        let scale = base_radius_fraction * 2.0_f64.powi(level as i32);
        let radius = Array1::from_iter(
            (0..bounds.dims).map(|axis| ((bounds.high[axis] - bounds.low[axis]) * scale).max(0.0)),
        );
        let remaining = max_evals - n_evals;
        let n_batch = batch_size.min(remaining);
        let line_probes_per_axis = (n_batch / bounds.dims.max(1)).max(1);
        let mut level_improved = false;
        for axis in 0..bounds.dims {
            for probe in 1..=line_probes_per_axis {
                if n_evals >= max_evals {
                    break;
                }
                let scalar = 2.0 * eindir_core::radical_inverse(probe as u64, 2) - 1.0;
                if scalar.abs() <= f64::EPSILON {
                    continue;
                }
                let mut trial = current_center.clone();
                trial[axis] += radius[axis] * scalar;
                let trial = bounds.clip(trial.view());
                let value = obj.eval(trial.view());
                n_evals += 1;
                if value.is_finite() && value < best_val {
                    best_val = value;
                    best_pos = trial;
                    current_center = best_pos.clone();
                    level_improved = true;
                }
            }
        }
        if n_evals >= max_evals {
            break;
        }
        let zoom_low = Array1::from_iter(
            (0..bounds.dims)
                .map(|axis| (current_center[axis] - radius[axis]).max(bounds.low[axis])),
        );
        let zoom_high = Array1::from_iter(
            (0..bounds.dims)
                .map(|axis| (current_center[axis] + radius[axis]).min(bounds.high[axis])),
        );
        let zoom_bounds = Bounds::new(zoom_low, zoom_high, bounds.slack);
        let replica_count = bounds.dims.min(n_batch).max(1);
        let points_per_replica = n_batch.div_ceil(replica_count).max(1);
        let mut direct_points = 0usize;
        for replica in 0..replica_count {
            if n_evals >= max_evals || direct_points >= n_batch {
                break;
            }
            let remaining_direct = n_batch - direct_points;
            let replica_batch = points_per_replica.min(remaining_direct);
            let level_seed = seed.wrapping_add((level * replica_count + replica) as u64);
            let points = eindir_core::shifted_low_discrepancy_points(
                &zoom_bounds,
                replica_batch,
                crate::runner::qmc_skip_from_seed(level_seed),
                level_seed,
            );
            for point in points.outer_iter() {
                direct_points += 1;
                let pos = bounds.clip(point);
                let value = obj.eval(pos.view());
                n_evals += 1;
                if value.is_finite() && value < best_val {
                    best_val = value;
                    best_pos = pos;
                    level_improved = true;
                }
                if n_evals >= max_evals {
                    break;
                }
                let reflected = bounds.clip((&current_center * 2.0 - point).view());
                let reflected_value = obj.eval(reflected.view());
                n_evals += 1;
                if reflected_value.is_finite() && reflected_value < best_val {
                    best_val = reflected_value;
                    best_pos = reflected;
                    level_improved = true;
                }
                if n_evals >= max_evals {
                    break;
                }
            }
        }
        if level_improved {
            current_center = best_pos.clone();
        }
    }

    QmcPolishResult {
        best_pos,
        best_val,
        n_evals,
        n_grads: 0,
        n_starts: n_evals,
        n_polished: 0,
        polished_values: Vec::new(),
        polished_projected_grad_norms: Vec::new(),
        polished_stationary: Vec::new(),
    }
}

/// Refine replicated shifted low-discrepancy starts with bounded polish.
#[allow(clippy::too_many_arguments)]
pub fn shifted_qmc_projected_gradient_polish<O, G>(
    obj: &O,
    gradient: &G,
    n_starts: usize,
    max_fevals_per_start: usize,
    seed: u64,
    n_replicates: usize,
    step0: f64,
    grad_tol: f64,
    top_k: usize,
) -> QmcPolishResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(n_starts > 0, "n_starts must be positive");
    assert!(
        max_fevals_per_start > 0,
        "max_fevals_per_start must be positive"
    );
    assert!(n_replicates > 0, "n_replicates must be positive");

    let bounds = obj.bounds();
    let mut best_pos = bounds.low.clone();
    let mut best_val = f64::INFINITY;
    let mut n_evals = 0usize;
    let mut n_grads = 0usize;
    let mut n_polished = 0usize;
    let mut polished_values = Vec::new();
    let mut polished_projected_grad_norms = Vec::new();
    let mut polished_stationary = Vec::new();

    for replica in 0..n_replicates {
        let replica_seed = seed.wrapping_add(replica as u64);
        let starts = eindir_core::shifted_low_discrepancy_points(
            bounds,
            n_starts,
            crate::runner::qmc_skip_from_seed(replica_seed),
            replica_seed,
        );
        let mut screened = Vec::with_capacity(n_starts);
        for start in starts.outer_iter() {
            let pos = bounds.clip(start);
            let value = obj.eval(pos.view());
            n_evals += 1;
            if value.is_finite() {
                if value < best_val {
                    best_val = value;
                    best_pos = pos.clone();
                }
                screened.push((value, pos));
            }
        }
        screened.sort_by(|left, right| left.0.total_cmp(&right.0));

        let polish_limit = if top_k == 0 {
            screened.len()
        } else {
            top_k.min(screened.len())
        };
        for (_value, start) in screened.into_iter().take(polish_limit) {
            let result = projected_gradient_polish(
                obj,
                gradient,
                start,
                max_fevals_per_start,
                step0,
                grad_tol,
            );
            n_evals += result.n_evals;
            n_grads += result.n_grads;
            n_polished += 1;
            polished_values.push(result.best_val);
            polished_projected_grad_norms.push(result.projected_grad_norm);
            polished_stationary.push(result.projected_stationary);
            if result.best_val.is_finite() && result.best_val < best_val {
                best_val = result.best_val;
                best_pos = result.best_pos;
            }
        }
    }

    QmcPolishResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
        n_starts: n_starts * n_replicates,
        n_polished,
        polished_values,
        polished_projected_grad_norms,
        polished_stationary,
    }
}
