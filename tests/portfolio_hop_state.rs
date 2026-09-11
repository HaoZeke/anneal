//! Occupied hop state and the cross-arm incumbent are distinct evaluated pairs.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::{PortfolioPolicy, portfolio_optimize_with_policy};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

struct Loss {
    bounds: Bounds<f64>,
    terraced: bool,
    observations: Mutex<Vec<(Vec<f64>, f64)>>,
    gradients: AtomicUsize,
}

impl Loss {
    fn new(terraced: bool) -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(8, -2.0), Array1::from_elem(8, 2.0), 0.0),
            terraced,
            observations: Mutex::new(Vec::new()),
            gradients: AtomicUsize::new(0),
        }
    }

    fn value(&self, x: ArrayView1<f64>) -> f64 {
        x.iter().enumerate().map(|(j, &coordinate)| {
            if self.terraced {
                ((2.0 * (coordinate + 2.0)).floor() - 3.0).powi(2)
            } else {
                let shifted = coordinate - 0.271 - 0.01 * j as f64;
                shifted * shifted + 10.0 * (1.0 - (std::f64::consts::TAU * shifted).cos())
            }
        }).sum()
    }
}

impl Objective<f64> for Loss {
    fn dim(&self) -> usize { self.bounds.dims }
    fn bounds(&self) -> &Bounds<f64> { &self.bounds }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert!(x.iter().all(|&v| (-2.0..=2.0).contains(&v)));
        let value = self.value(x);
        self.observations.lock().unwrap().push((x.to_vec(), value));
        value
    }
}

impl Gradient<f64> for Loss {
    fn dim(&self) -> usize { self.bounds.dims }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        Array1::from_iter(x.iter().enumerate().map(|(j, &coordinate)| {
            if self.terraced {
                0.0
            } else {
                let shifted = coordinate - 0.271 - 0.01 * j as f64;
                2.0 * shifted + 10.0 * std::f64::consts::TAU * (std::f64::consts::TAU * shifted).sin()
            }
        }))
    }
}

fn check_occupied_pair(terraced: bool) {
    const BUDGET: usize = 2_000;
    for seed in 0..16 {
        let loss = Loss::new(terraced);
        let result = portfolio_optimize_with_policy(
            &loss, Some(&loss), BUDGET, seed, None, PortfolioPolicy::Legacy,
        );
        let observations = loss.observations.lock().unwrap();
        assert_eq!(result.n_evals, observations.len());
        assert_eq!(result.n_grads, loss.gradients.load(Ordering::Relaxed));
        assert!(result.n_evals + result.n_grads <= BUDGET);
        assert_eq!(result.best_val, observations.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min));
        assert_eq!(result.best_val, loss.value(ArrayView1::from(&result.best_pos)));
        let allocated = result.arm_stats.iter().find(|arm| arm.name == "hop").unwrap();
        assert!(allocated.pulls >= 2, "seed {seed} must resume the hop arm");
        let occupied = result.hop_state.expect("the allocated hop arm retains its state");
        assert_eq!(
            occupied.val, loss.value(occupied.pos.view()),
            "seed {seed}, terraced {terraced}: cached hop energy belongs to its occupied position, not another arm's incumbent",
        );
        assert!(observations.iter().any(|(position, value)| {
            position.as_slice() == occupied.pos.as_slice().unwrap() && *value == occupied.val
        }), "seed {seed}: occupied pair must have been evaluated");
    }
}

#[test]
fn terraced_hops_retain_their_own_evaluated_energy() {
    check_occupied_pair(true);
}

#[test]
fn smooth_hops_retain_their_own_evaluated_energy() {
    check_occupied_pair(false);
}

struct NoGradient;
impl Gradient<f64> for NoGradient {
    fn dim(&self) -> usize { panic!("no gradient capability is supplied") }
    fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> { panic!("no gradient callback is supplied") }
}

#[test]
fn scalar_portfolio_does_not_fabricate_gradient_hop_state() {
    let loss = Loss::new(true);
    let result = portfolio_optimize_with_policy::<_, NoGradient>(
        &loss, None, 257, 7, None, PortfolioPolicy::Auto,
    );
    assert!(result.hop_state.is_none());
    assert_eq!(result.n_evals, loss.observations.lock().unwrap().len());
    assert_eq!(result.n_evals, 257);
    assert_eq!(result.n_grads, 0);
}
