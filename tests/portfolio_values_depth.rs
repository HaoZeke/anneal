//! Scalar refinement funds a complete stencil before dividing work among starts.

use std::collections::HashMap;
use std::sync::Mutex;
use std::thread::ThreadId;

use anneal_core::methods::portfolio::{
    PortfolioEnsembleConfig, portfolio_ensemble_optimize, portfolio_optimize,
};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

const DIM: usize = 128;

struct ScalarQuadratic {
    bounds: Bounds<f64>,
    traces: Mutex<HashMap<ThreadId, Vec<Array1<f64>>>>,
}

impl ScalarQuadratic {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_elem(DIM, -5.12),
                Array1::from_elem(DIM, 5.12),
                0.0,
            ),
            traces: Mutex::new(HashMap::new()),
        }
    }

    fn value(x: ArrayView1<f64>) -> f64 {
        x.iter()
            .enumerate()
            .map(|(j, &x)| {
                let optimum = 0.7 + 0.3 * ((j + 1) as f64 * std::f64::consts::SQRT_2).sin();
                (x - optimum).powi(2)
            })
            .sum()
    }

    /// Whether `calls` open with the full central stencil about `anchor`
    /// and a trial off it with a finite value.
    fn funds_stencil(anchor: &Array1<f64>, calls: &[Array1<f64>]) -> bool {
        let h = 1e-5 * 10.24;
        for j in 0..DIM {
            let mut plus = anchor.clone();
            let mut minus = anchor.clone();
            plus[j] = (anchor[j] + h).max(anchor[j].next_up()).min(5.12);
            minus[j] = (anchor[j] - h).min(anchor[j].next_down()).max(-5.12);
            if calls[2 * j] != plus || calls[1 + 2 * j] != minus {
                return false;
            }
        }
        let trial = &calls[2 * DIM];
        trial != anchor && Self::value(trial.view()).is_finite()
    }

    /// Each replica's calls, in any order, against the traces.
    fn assert_funded_refinement(&self, calls: &[usize]) {
        let traces = self.traces.lock().unwrap();
        let mut lengths: Vec<_> = traces.values().map(Vec::len).collect();
        let mut calls = calls.to_vec();
        lengths.sort_unstable();
        calls.sort_unstable();
        assert_eq!(lengths, calls);
        for trace in traces.values() {
            // The stencil's anchor, the descent's iterate, was called
            // before it but not always just before: a descent that starts
            // from a point evaluated earlier asks for it again, which the
            // run answers without a call.
            let complete = (1..trace.len().saturating_sub(2 * DIM)).any(|k| {
                trace[..k].iter().any(|anchor| {
                    trace[k].iter().zip(anchor).skip(1).all(|(a, b)| a == b)
                        && Self::funds_stencil(anchor, &trace[k..])
                })
            });
            assert!(
                complete,
                "a 2000-evaluation chain must fund one full 256-call stencil and a trial, not only partial stencils"
            );
        }
    }
}

impl Objective<f64> for ScalarQuadratic {
    fn dim(&self) -> usize {
        DIM
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), DIM);
        assert!(
            x.iter()
                .all(|v| v.is_finite() && (-5.12..=5.12).contains(v))
        );
        self.traces
            .lock()
            .unwrap()
            .entry(std::thread::current().id())
            .or_default()
            .push(x.to_owned());
        Self::value(x)
    }
}

struct NoGradient;
impl Gradient<f64> for NoGradient {
    fn dim(&self) -> usize {
        panic!("the caller supplies values only")
    }
    fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> {
        panic!("the caller supplies values only")
    }
}

#[test]
fn scalar_portfolio_funds_complete_high_dimensional_refinement() {
    let loss = ScalarQuadratic::new();
    let result = portfolio_optimize::<_, NoGradient>(&loss, None, 2_000, 7, None);
    assert_eq!(result.n_evals + result.n_answered, 2_000);
    assert_eq!(result.n_grads, 0);
    assert_eq!(
        result.best_val,
        ScalarQuadratic::value(ArrayView1::from(&result.best_pos))
    );
    loss.assert_funded_refinement(&[result.n_evals]);
}

#[test]
fn each_scalar_replica_funds_complete_high_dimensional_refinement() {
    let loss = ScalarQuadratic::new();
    let mut config = PortfolioEnsembleConfig {
        budget: 8_000,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.shared = false;
    let result = portfolio_ensemble_optimize::<_, NoGradient>(&loss, None, 7, None, &config);
    assert_eq!(result.n_evals + result.n_answered, 8_000);
    assert_eq!(result.n_grads, 0);
    assert_eq!(
        result.best_val,
        ScalarQuadratic::value(ArrayView1::from(&result.best_pos))
    );
    assert!(
        result
            .replicas
            .iter()
            .all(|replica| replica.n_evals + replica.n_answered == 2_000)
    );
    let calls: Vec<_> = result
        .replicas
        .iter()
        .map(|replica| replica.n_evals)
        .collect();
    loss.assert_funded_refinement(&calls);
}
