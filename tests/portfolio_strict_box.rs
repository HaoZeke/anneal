use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::{PortfolioPolicy, portfolio_optimize_with_policy};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

struct StrictObjective {
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}

impl Objective<f64> for StrictObjective {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), self.bounds.dims);
        for (i, &value) in x.iter().enumerate() {
            assert!(
                value >= self.bounds.low[i] && value <= self.bounds.high[i],
                "callback coordinate {i}={value} violates the strict objective domain"
            );
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        x.iter().map(|value| (value - 0.93).powi(2)).sum()
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

struct NoUserGradient;

impl Gradient<f64> for NoUserGradient {
    fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> {
        panic!("the scalar portfolio must not request a user gradient")
    }

    fn dim(&self) -> usize {
        panic!("the scalar portfolio must not inspect a user gradient")
    }
}

#[test]
fn feasibility_slack_does_not_expand_the_objective_domain() {
    for seed in 0..4 {
        let objective = StrictObjective {
            bounds: Bounds::new(Array1::zeros(3), Array1::ones(3), 0.25),
            calls: AtomicUsize::new(0),
        };
        let result = portfolio_optimize_with_policy::<_, NoUserGradient>(
            &objective,
            None,
            12_000,
            seed,
            None,
            PortfolioPolicy::Legacy,
        );
        assert_eq!(result.n_evals, objective.calls.load(Ordering::Relaxed));
        assert!(result.n_evals > 0 && result.n_evals <= 12_000);
        assert_eq!(result.n_grads, 0);
        assert!(result.best_pos.iter().all(|v| *v >= 0.0 && *v <= 1.0));
        assert!(result.best_val >= 0.0 && result.best_val.is_finite());
    }
}
