//! Values-only access to the portfolio's budgeted quasi-Newton refinement.

use std::sync::atomic::Ordering;

use eindir_core::Objective;
use ndarray::Array1;

use super::{BudgetLedger, BudgetedFiniteDiffGradient, BudgetedObjective};
use crate::methods::local_polish::{LocalPolishResult, projected_gradient_polish};

/// Refine a point using scalar values, without a user gradient capability.
///
/// `max_evals` includes every finite-difference probe. Rejected line-search
/// and stencil points remain eligible for the raw-objective incumbent.
/// `n_grads` is zero; an available `best_grad` is a numerical approximation,
/// not an analytic force or a global-optimality certificate.
pub fn values_local_polish<O: Objective<f64>>(
    obj: &O,
    start: Array1<f64>,
    max_evals: usize,
    step0: f64,
    grad_tol: f64,
) -> LocalPolishResult {
    assert!(max_evals > 0, "max_evals must be positive");
    let ledger = BudgetLedger::new(max_evals, obj.dim());
    let budgeted = BudgetedObjective { inner: obj, ledger: &ledger };
    refine_with_ledger(&budgeted, start, max_evals, step0, grad_tol)
}

pub(super) fn refine_with_ledger<O: Objective<f64>>(
    obj: &BudgetedObjective<'_, O>,
    start: Array1<f64>,
    max_evals: usize,
    step0: f64,
    grad_tol: f64,
) -> LocalPolishResult {
    let before = obj.ledger.n_evals.load(Ordering::Relaxed);
    let fd = BudgetedFiniteDiffGradient { obj, h_frac: 1e-5 };
    let mut result = projected_gradient_polish(obj, &fd, start, max_evals, step0, grad_tol);
    let best = obj.ledger.incumbent(obj.bounds());
    let value = obj.ledger.best_get();
    if result.best_pos != best || result.best_val != value {
        result.best_grad = None;
        result.projected_grad_norm = f64::INFINITY;
        result.projected_stationary = false;
    }
    result.best_pos = best;
    result.best_val = value;
    result.n_evals = obj.ledger.n_evals.load(Ordering::Relaxed) - before;
    result.n_grads = 0;
    result
}
