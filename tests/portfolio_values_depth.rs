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
            bounds: Bounds::new(Array1::from_elem(DIM, -5.12), Array1::from_elem(DIM, 5.12), 0.0),
            traces: Mutex::new(HashMap::new()),
        }
    }

    fn value(x: ArrayView1<f64>) -> f64 {
        x.iter().enumerate().map(|(j, &x)| {
            let optimum = 0.7 + 0.3 * ((j + 1) as f64 * std::f64::consts::SQRT_2).sin();
            (x - optimum).powi(2)
        }).sum()
    }

    fn assert_funded_refinement(&self, replicas: usize) {
        let traces = self.traces.lock().unwrap();
        assert_eq!(traces.len(), replicas);
        for trace in traces.values() {
            assert_eq!(trace.len(), 2_000);
            let complete = trace.windows(2 * DIM + 2).any(|calls| {
                let anchor = &calls[0];
                let h = 1e-5 * 10.24;
                for j in 0..DIM {
                    let mut plus = anchor.clone();
                    let mut minus = anchor.clone();
                    plus[j] = (anchor[j] + h).max(anchor[j].next_up()).min(5.12);
                    minus[j] = (anchor[j] - h).min(anchor[j].next_down()).max(-5.12);
                    if calls[1 + 2 * j] != plus || calls[2 + 2 * j] != minus {
                        return false;
                    }
                }
                let trial = &calls[2 * DIM + 1];
                trial != anchor && Self::value(trial.view()).is_finite()
            });
            assert!(complete, "a 2000-call chain must fund one full 256-call stencil and a trial, not only partial stencils");
        }
    }
}

impl Objective<f64> for ScalarQuadratic {
    fn dim(&self) -> usize { DIM }
    fn bounds(&self) -> &Bounds<f64> { &self.bounds }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), DIM);
        assert!(x.iter().all(|v| v.is_finite() && (-5.12..=5.12).contains(v)));
        self.traces.lock().unwrap().entry(std::thread::current().id())
            .or_default().push(x.to_owned());
        Self::value(x)
    }
}

struct NoGradient;
impl Gradient<f64> for NoGradient {
    fn dim(&self) -> usize { panic!("the caller supplies values only") }
    fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> { panic!("the caller supplies values only") }
}

#[test]
fn scalar_portfolio_funds_complete_high_dimensional_refinement() {
    let loss = ScalarQuadratic::new();
    let result = portfolio_optimize::<_, NoGradient>(&loss, None, 2_000, 7, None);
    assert_eq!(result.n_evals, 2_000);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, ScalarQuadratic::value(ArrayView1::from(&result.best_pos)));
    loss.assert_funded_refinement(1);
}

#[test]
fn each_scalar_replica_funds_complete_high_dimensional_refinement() {
    let loss = ScalarQuadratic::new();
    let mut config = PortfolioEnsembleConfig { budget: 8_000, ..PortfolioEnsembleConfig::default() };
    config.coverage.shared = false;
    let result = portfolio_ensemble_optimize::<_, NoGradient>(&loss, None, 7, None, &config);
    assert_eq!(result.n_evals, 8_000);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, ScalarQuadratic::value(ArrayView1::from(&result.best_pos)));
    loss.assert_funded_refinement(4);
}
