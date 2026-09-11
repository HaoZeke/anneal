//! Fixed-axis adapter for the values-only portfolio's positive-width domain.

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::methods::portfolio::{PortfolioPolicy, PortfolioResult, portfolio_optimize_seeded};

/// Search free axes without changing the dimension seen by the objective.
pub(super) fn values_portfolio<O, G>(
    obj: &O,
    budget: usize,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
) -> PortfolioResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let bounds = obj.bounds();
    let free: Vec<usize> = (0..bounds.dims)
        .filter(|&i| bounds.low[i] != bounds.high[i])
        .collect();
    if free.len() == bounds.dims {
        return portfolio_optimize_seeded::<_, G>(
            obj,
            None,
            budget,
            seed,
            None,
            PortfolioPolicy::Auto,
            x0,
        );
    }
    if let Some(x0) = x0 {
        assert_eq!(
            x0.len(),
            bounds.dims,
            "initial position must match the objective dimension"
        );
    }
    if free.is_empty() {
        let value = obj.eval(bounds.low.view());
        return PortfolioResult {
            best_pos: bounds.low.to_vec(),
            best_val: if value.is_finite() {
                value
            } else {
                f64::INFINITY
            },
            n_evals: 1,
            n_grads: 0,
            arm_stats: Vec::new(),
        };
    }
    let reduced = FreeCoordinates {
        inner: obj,
        bounds: Bounds::new(
            Array1::from_iter(free.iter().map(|&i| bounds.low[i])),
            Array1::from_iter(free.iter().map(|&i| bounds.high[i])),
            bounds.slack,
        ),
        free,
    };
    let start = x0.map(|x| Array1::from_iter(reduced.free.iter().map(|&i| x[i])));
    let mut result = portfolio_optimize_seeded::<_, G>(
        &reduced,
        None,
        budget,
        seed,
        None,
        PortfolioPolicy::Auto,
        start.as_ref().map(|x| x.view()),
    );
    result.best_pos = reduced.expand(ArrayView1::from(&result.best_pos)).to_vec();
    result
}

/// A coordinate-index map uses linear storage without a dense affine basis.
struct FreeCoordinates<'a, O> {
    inner: &'a O,
    bounds: Bounds<f64>,
    free: Vec<usize>,
}

impl<O: Objective<f64>> FreeCoordinates<'_, O> {
    fn expand(&self, reduced: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(reduced.len(), self.free.len());
        let mut full = self.inner.bounds().low.clone();
        for (&i, &coordinate) in self.free.iter().zip(reduced.iter()) {
            full[i] = coordinate;
        }
        full
    }
}

impl<O: Objective<f64>> Objective<f64> for FreeCoordinates<'_, O> {
    fn dim(&self) -> usize {
        self.bounds.dims
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.inner.eval(self.expand(x).view())
    }
}
