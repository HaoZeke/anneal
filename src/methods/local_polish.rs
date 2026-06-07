//! Bounded local refinement used as the deterministic polish stage after a
//! stochastic annealing or pilot phase.

use crate::grad::Gradient;
use eindir_core::Objective;
use ndarray::Array1;

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

fn update_diagonal_spectral_scale(
    scale: &mut Array1<f64>,
    x: &Array1<f64>,
    pgrad: &Array1<f64>,
    prev_x: &Array1<f64>,
    prev_pgrad: &Array1<f64>,
) {
    for i in 0..scale.len() {
        let step_delta = x[i] - prev_x[i];
        let grad_delta = pgrad[i] - prev_pgrad[i];
        let curvature = step_delta * grad_delta;
        if curvature > 0.0 {
            let candidate = (step_delta / grad_delta).abs();
            if candidate.is_finite() && candidate > 0.0 {
                scale[i] = candidate;
            }
        }
    }
}

/// Refine a point with projected-gradient backtracking inside objective bounds.
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
    let mut scale = Array1::from_elem(x.len(), step0);
    let mut prev_x = None;
    let mut prev_pgrad = None;

    while n_evals < max_fevals {
        let grad = gradient.grad(x.view());
        n_grads += 1;
        if grad.len() != x.len() || grad.iter().any(|v| !v.is_finite()) {
            break;
        }
        let pgrad = projected_gradient(&x, &grad, low, high);
        let norm2 = pgrad.iter().map(|v| v * v).sum::<f64>();
        if norm2.sqrt() <= grad_tol {
            break;
        }
        if let (Some(px), Some(pg)) = (&prev_x, &prev_pgrad) {
            update_diagonal_spectral_scale(&mut scale, &x, &pgrad, px, pg);
        }
        let direction = &pgrad * &scale;
        let directional_decrease = pgrad.dot(&direction);
        if !directional_decrease.is_finite() || directional_decrease <= 0.0 {
            break;
        }

        let mut accepted = false;
        let mut alpha = FULL_LINE_SEARCH_STEP;
        while n_evals < max_fevals && alpha > MIN_BACKTRACK_STEP {
            let trial = bounds.clip((&x - &(direction.mapv(|v| alpha * v))).view());
            if trial
                .iter()
                .zip(x.iter())
                .all(|(a, b)| (*a - *b).abs() <= f64::EPSILON)
            {
                alpha *= 0.5;
                continue;
            }
            let trial_value = obj.eval(trial.view());
            n_evals += 1;
            if trial_value.is_finite()
                && trial_value <= value - ARMIJO_SUFFICIENT_DECREASE * alpha * directional_decrease
            {
                prev_x = Some(x.clone());
                prev_pgrad = Some(pgrad.clone());
                x = trial;
                value = trial_value;
                if trial_value < best_val {
                    best_val = trial_value;
                    best_pos = x.clone();
                }
                accepted = true;
                break;
            }
            alpha *= BACKTRACK_SHRINK;
        }
        if !accepted {
            break;
        }
    }

    LocalPolishResult {
        best_pos,
        best_val,
        n_evals,
        n_grads,
    }
}
