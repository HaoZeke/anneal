//! Feasibility choke-point: portfolio best must stay inside the declared box.
//!
//! Regression for OOB Schwefel archives (best_val ≪ domain min on the box).

use anneal_core::methods::{PortfolioPolicy, portfolio_optimize, portfolio_optimize_with_policy};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

/// Schwefel on [-500, 500]^d; global min ≈ 0 at x* ≈ (420.97,...).
/// Any best_val ≪ 0 is impossible on the box and indicates an OOB archive.
struct Schwefel {
    bounds: Bounds<f64>,
    dim: usize,
}

impl Schwefel {
    fn new(dim: usize) -> Self {
        let low = Array1::from_elem(dim, -500.0);
        let high = Array1::from_elem(dim, 500.0);
        Self {
            bounds: Bounds::new(low, high, 1e-9),
            dim,
        }
    }
}

impl Objective<f64> for Schwefel {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        // Intentionally evaluate OOB values if asked — the ledger must not
        // promote them. Domain formula (valid anywhere).
        let d = x.len() as f64;
        418.9829 * d
            - x.iter()
                .map(|&xi| xi * (xi.abs().sqrt()).sin())
                .sum::<f64>()
    }
}

impl Gradient<f64> for Schwefel {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        // Return zeros so polish is weak; feasibility is the property under test.
        Array1::zeros(x.len())
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

fn assert_feasible(pos: &[f64], val: f64, low: f64, high: f64) {
    for (i, &p) in pos.iter().enumerate() {
        assert!(
            p >= low - 1e-8 && p <= high + 1e-8,
            "best_pos[{i}]={p} outside [{low},{high}]"
        );
    }
    // On [-500,500]^d Schwefel is ≥ ~0; anything << -1 is an OOB artifact.
    assert!(
        val.is_finite() && val > -1.0,
        "best_val={val} is domain-impossible on the Schwefel box (OOB archive)"
    );
}

#[test]
fn portfolio_best_stays_in_bounds_schwefel() {
    let obj = Schwefel::new(2);
    let result = portfolio_optimize(&obj, Some(&obj), 1500, 7, None);
    assert_feasible(&result.best_pos, result.best_val, -500.0, 500.0);
}

#[test]
fn portfolio_legacy_best_stays_in_bounds_schwefel() {
    let obj = Schwefel::new(5);
    let result =
        portfolio_optimize_with_policy(&obj, Some(&obj), 1500, 11, None, PortfolioPolicy::Legacy);
    assert_feasible(&result.best_pos, result.best_val, -500.0, 500.0);
}

#[test]
fn portfolio_high_dim_schwefel_feasible() {
    let obj = Schwefel::new(20);
    let result = portfolio_optimize::<_, Schwefel>(&obj, None, 2000, 3, None);
    assert_feasible(&result.best_pos, result.best_val, -500.0, 500.0);
}
