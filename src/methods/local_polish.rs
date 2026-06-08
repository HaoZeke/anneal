//! Bounded local refinement used as the deterministic polish stage after a
//! stochastic annealing or pilot phase.

use eindir_core::Gradient;
use eindir_core::Objective;
use ndarray::{Array1, Array2};

const ARMIJO_SUFFICIENT_DECREASE: f64 = 1e-4;
const BACKTRACK_SHRINK: f64 = 0.5;
const MIN_BACKTRACK_STEP: f64 = f64::EPSILON;
const FULL_LINE_SEARCH_STEP: f64 = 1.0;

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

fn scaled_identity(dim: usize, scale: f64) -> Array2<f64> {
    let mut matrix = Array2::zeros((dim, dim));
    for i in 0..dim {
        matrix[[i, i]] = scale;
    }
    matrix
}

fn active_descent_direction(
    inverse_hessian: &Array2<f64>,
    x: &Array1<f64>,
    pgrad: &Array1<f64>,
    low: &Array1<f64>,
    high: &Array1<f64>,
) -> Array1<f64> {
    let mut direction = -inverse_hessian.dot(pgrad);
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

fn update_inverse_hessian(
    inverse_hessian: &mut Array2<f64>,
    s: &Array1<f64>,
    y: &Array1<f64>,
) -> bool {
    let ys = y.dot(s);
    let s_norm = s.iter().map(|v| v * v).sum::<f64>().sqrt();
    let y_norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    let curvature_floor = f64::EPSILON.sqrt() * s_norm * y_norm;
    if !ys.is_finite() || ys <= curvature_floor {
        return false;
    }

    let hy = inverse_hessian.dot(y);
    let yhy = y.dot(&hy);
    if !yhy.is_finite() {
        return false;
    }

    let dim = s.len();
    let mut next = inverse_hessian.clone();
    let ss_coeff = (ys + yhy) / (ys * ys);
    for i in 0..dim {
        for j in 0..dim {
            next[[i, j]] += ss_coeff * s[i] * s[j] - (s[i] * hy[j] + hy[i] * s[j]) / ys;
        }
    }
    if next.iter().all(|v| v.is_finite()) {
        *inverse_hessian = next;
        true
    } else {
        false
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
    let mut inverse_hessian = scaled_identity(x.len(), step0);
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
            if !update_inverse_hessian(&mut inverse_hessian, &s, &y) {
                inverse_hessian = scaled_identity(x.len(), step0);
            }
        }
        let mut accepted = false;
        let mut direction = active_descent_direction(&inverse_hessian, &x, &pgrad, low, high);
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
            inverse_hessian = scaled_identity(x.len(), step0);
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
    let mut screened = Vec::with_capacity(n_starts);
    let mut n_evals = 0usize;
    let mut best_pos = bounds.clip(starts.row(0));
    let mut best_val = f64::INFINITY;

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

/// Refine replicated shifted low-discrepancy starts with bounded polish.
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
