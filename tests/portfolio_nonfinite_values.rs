//! A run under declared noise whose every value is not finite ends within its
//! budget, asks only points inside the box and reports every call it made.
//!
//! Once nothing finite has been seen, each arm of the bandit can return
//! without a call; the loop must then leave on what it can no longer spend.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use anneal_core::methods::{
    BoxCoverageConfig, PortfolioEnsembleConfig, PortfolioPolicy, portfolio_ensemble_optimize,
    portfolio_optimize_from,
};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

const NOISE: Option<f64> = Some(1e-3);
/// Runs of these budgets end within a second; one that spins never does.
const LIMIT: Duration = Duration::from_secs(30);
const VALUES: [f64; 3] = [f64::INFINITY, f64::NAN, f64::NEG_INFINITY];

struct Flat {
    bounds: Bounds<f64>,
    value: f64,
    calls: AtomicUsize,
}

impl Flat {
    fn new(dim: usize, value: f64) -> Self {
        Self {
            bounds: Bounds::new(Array1::zeros(dim), Array1::ones(dim), 1e-9),
            value,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Objective<f64> for Flat {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert!(in_unit_box(x.iter()), "{x} lies outside the box");
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.value
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

/// Never called: the runs here have no gradient.
struct NoGradient;

impl Gradient<f64> for NoGradient {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        unreachable!("no gradient was given, yet one was asked at {x}")
    }

    fn dim(&self) -> usize {
        0
    }
}

fn in_unit_box<'a>(mut x: impl Iterator<Item = &'a f64>) -> bool {
    x.all(|v| (0.0..=1.0).contains(v))
}

struct Ran {
    n_evals: usize,
    n_grads: usize,
    calls: usize,
    best_pos: Vec<f64>,
    best_val: f64,
}

fn within_limit(what: String, run: impl FnOnce() -> Ran + Send + 'static) -> Ran {
    let (send, receive) = mpsc::channel();
    std::thread::spawn(move || send.send(run()));
    match receive.recv_timeout(LIMIT) {
        Ok(ran) => ran,
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("{what}: no answer within {LIMIT:?}"),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{what}: the run panicked"),
    }
}

fn check(what: &str, ran: &Ran, dim: usize, budget: usize) {
    assert_eq!(ran.n_evals, ran.calls, "{what}: reported calls");
    assert_eq!(ran.n_grads, 0, "{what}: gradient calls");
    assert!(ran.n_evals <= budget, "{what}: {} calls", ran.n_evals);
    assert!(ran.n_evals > 0, "{what}: no call");
    assert_eq!(ran.best_pos.len(), dim, "{what}: best_pos");
    assert!(
        in_unit_box(ran.best_pos.iter()),
        "{what}: best_pos {:?} outside the box",
        ran.best_pos
    );
    assert_eq!(ran.best_val, f64::INFINITY, "{what}: best_val");
}

#[test]
fn a_single_controller_ends_when_no_value_is_finite() {
    for (dim, budget) in [(1usize, 60usize), (2, 80)] {
        for value in VALUES {
            for start in [true, false] {
                for seed in [0u64, 1] {
                    let what =
                        format!("d={dim} budget={budget} value={value} x0={start} seed={seed}");
                    let ran = within_limit(what.clone(), move || {
                        let obj = Flat::new(dim, value);
                        let x0 = Array1::from_elem(dim, 0.5);
                        let result = portfolio_optimize_from::<_, NoGradient>(
                            &obj,
                            None,
                            budget,
                            seed,
                            NOISE,
                            PortfolioPolicy::Auto,
                            start.then(|| x0.view()),
                        );
                        Ran {
                            n_evals: result.n_evals,
                            n_grads: result.n_grads,
                            calls: obj.calls.load(Ordering::Relaxed),
                            best_pos: result.best_pos,
                            best_val: result.best_val,
                        }
                    });
                    check(&what, &ran, dim, budget);
                }
            }
        }
    }
}

#[test]
fn persistent_replicas_end_when_no_value_is_finite() {
    let budget = 60;
    for value in VALUES {
        for start in [true, false] {
            let what = format!("replicas=3 budget={budget} value={value} x0={start}");
            let ran = within_limit(what.clone(), move || {
                let obj = Flat::new(1, value);
                let x0 = Array1::from_elem(1, 0.5);
                let config = PortfolioEnsembleConfig {
                    replicas: 3,
                    budget,
                    noise_sigma: NOISE,
                    policy: PortfolioPolicy::Auto,
                    coverage: BoxCoverageConfig::default(),
                };
                let result = portfolio_ensemble_optimize::<_, NoGradient>(
                    &obj,
                    None,
                    0,
                    start.then(|| x0.view()),
                    &config,
                );
                Ran {
                    n_evals: result.n_evals,
                    n_grads: result.n_grads,
                    calls: obj.calls.load(Ordering::Relaxed),
                    best_pos: result.best_pos,
                    best_val: result.best_val,
                }
            });
            check(&what, &ran, 1, budget);
        }
    }
}
