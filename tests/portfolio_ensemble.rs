//! Cooperation retains each portfolio's controller and actual callback contract.

use std::collections::HashMap;
use std::sync::Mutex;
use std::thread::ThreadId;

use anneal_core::methods::portfolio::{
    PortfolioEnsembleConfig, PortfolioPolicy, portfolio_ensemble_optimize,
    portfolio_optimize_with_policy,
};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

struct Loss {
    bounds: Bounds<f64>,
    traces: Mutex<HashMap<ThreadId, Vec<Vec<u64>>>>,
    panic_at_start: bool,
}

impl Loss {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(Array1::from_elem(8, -2.0), Array1::from_elem(8, 2.0), 0.0),
            traces: Mutex::new(HashMap::new()),
            panic_at_start: false,
        }
    }

    fn value(&self, x: ArrayView1<f64>) -> f64 {
        x.iter()
            .enumerate()
            .map(|(j, &v)| {
                let d = v - 0.271 - 0.01 * j as f64;
                d * d + 10.0 * (1.0 - (std::f64::consts::TAU * d).cos())
            })
            .sum()
    }

    fn record(&self, kind: u64, x: ArrayView1<f64>) {
        assert_eq!(x.len(), 8);
        assert!(x.iter().all(|v| v.is_finite() && (-2.0..=2.0).contains(v)));
        let mut event = vec![kind];
        event.extend(x.iter().map(|v| v.to_bits()));
        self.traces
            .lock()
            .unwrap()
            .entry(std::thread::current().id())
            .or_default()
            .push(event);
    }

    fn trajectories(&self) -> Vec<Vec<Vec<u64>>> {
        let mut traces: Vec<_> = self.traces.lock().unwrap().values().cloned().collect();
        traces.sort();
        traces
    }
}

impl Objective<f64> for Loss {
    fn dim(&self) -> usize {
        8
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert!(!(self.panic_at_start && x.iter().all(|&v| v == -0.125)));
        self.record(0, x);
        self.value(x)
    }
}

impl Gradient<f64> for Loss {
    fn dim(&self) -> usize {
        8
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.record(1, x);
        Array1::from_iter(x.iter().enumerate().map(|(j, &v)| {
            let d = v - 0.271 - 0.01 * j as f64;
            2.0 * d + 10.0 * std::f64::consts::TAU * (std::f64::consts::TAU * d).sin()
        }))
    }
}

struct NoGradient;
impl Gradient<f64> for NoGradient {
    fn dim(&self) -> usize {
        panic!("no caller gradient is supplied")
    }
    fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> {
        panic!("no caller gradient is supplied")
    }
}

#[test]
fn one_replica_preserves_the_full_portfolio_trace() {
    for analytic in [false, true] {
        let reference = Loss::new();
        let expected = portfolio_optimize_with_policy(
            &reference,
            analytic.then_some(&reference),
            2_000,
            13,
            None,
            PortfolioPolicy::Auto,
        );
        let actual = Loss::new();
        let config = PortfolioEnsembleConfig {
            replicas: 1,
            budget: 2_000,
            ..PortfolioEnsembleConfig::default()
        };
        let result =
            portfolio_ensemble_optimize(&actual, analytic.then_some(&actual), 13, None, &config);
        assert_eq!(actual.trajectories(), reference.trajectories());
        assert_eq!(result.best_val, expected.best_val);
        assert_eq!(result.best_pos, expected.best_pos);
        assert_eq!(result.n_evals, expected.n_evals);
        assert_eq!(result.n_grads, expected.n_grads);
        assert_eq!(result.replicas.len(), 1);
        assert_eq!(result.coverage.published_samples, 0);
    }
}

#[test]
fn private_replicas_preserve_independent_portfolio_traces() {
    let mut expected = Vec::new();
    let mut expected_best = f64::INFINITY;
    for replica in 0..4 {
        let loss = Loss::new();
        let seed = 7 ^ (replica as u64).wrapping_mul(0x9E37_79B9);
        let result = portfolio_optimize_with_policy::<_, NoGradient>(
            &loss,
            None,
            1_000,
            seed,
            None,
            PortfolioPolicy::Auto,
        );
        expected_best = expected_best.min(result.best_val);
        expected.extend(loss.trajectories());
    }
    expected.sort();
    let loss = Loss::new();
    let mut config = PortfolioEnsembleConfig {
        replicas: 4,
        budget: 4_000,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.shared = false;
    let result = portfolio_ensemble_optimize::<_, NoGradient>(&loss, None, 7, None, &config);
    assert_eq!(loss.trajectories(), expected);
    assert_eq!(result.best_val, expected_best);
    assert_eq!(result.n_evals, config.budget);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.coverage.published_samples, 0);
    assert!(result.replicas.iter().all(|r| !r.arm_stats.is_empty()));
}

#[test]
fn seeded_private_replicas_match_the_convenience_portfolio_starts() {
    use anneal_core::methods::box_hopping::ensemble_hop_optimize;
    use anneal_core::methods::ensemble::HistoryMode;
    use anneal_core::methods::minima_hopping::HistoryMembership;

    let initial = Array1::from_elem(8, 0.375);
    let mut expected = Vec::new();
    for replica in 0..4 {
        let seed = 13 ^ (replica as u64).wrapping_mul(0x9E37_79B9);
        let mut rng = StdRng::seed_from_u64(seed);
        let start = if replica == 0 {
            initial.clone()
        } else {
            Array1::from_shape_fn(8, |_| -2.0 + 4.0 * rng.random::<f64>())
        };
        let loss = Loss::new();
        ensemble_hop_optimize::<_, NoGradient>(
            &loss, None, seed, Some(start.view()), 500, 1,
            HistoryMode::None, HistoryMembership::Accepted,
        );
        expected.extend(loss.trajectories());
    }
    expected.sort();
    let loss = Loss::new();
    let mut config = PortfolioEnsembleConfig {
        budget: 2_000,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.shared = false;
    portfolio_ensemble_optimize::<_, NoGradient>(
        &loss, None, 13, Some(initial.view()), &config,
    );
    assert_eq!(loss.trajectories(), expected);
}

#[test]
fn zero_height_sharing_preserves_private_controller_traces() {
    let private = Loss::new();
    let mut config = PortfolioEnsembleConfig {
        budget: 2_000,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.shared = false;
    let expected = portfolio_ensemble_optimize::<_, NoGradient>(&private, None, 19, None, &config);
    let disabled = Loss::new();
    config.coverage.shared = true;
    config.coverage.height = 0.0;
    let result = portfolio_ensemble_optimize::<_, NoGradient>(&disabled, None, 19, None, &config);
    assert_eq!(disabled.trajectories(), private.trajectories());
    assert_eq!(result.best_val, expected.best_val);
    assert_eq!(result.coverage.published_samples, 0);
    assert_eq!(result.coverage.repelled_proposals, 0);
}

fn check_shared(analytic: bool) -> Vec<Vec<Vec<u64>>> {
    let loss = Loss::new();
    let mut config = PortfolioEnsembleConfig {
        replicas: 4,
        budget: 8_003,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.radius = 0.8;
    let result = portfolio_ensemble_optimize(&loss, analytic.then_some(&loss), 17, None, &config);
    let traces = loss.trajectories();
    let evals = traces.iter().flatten().filter(|e| e[0] == 0).count();
    let grads = traces.iter().flatten().filter(|e| e[0] == 1).count();
    assert_eq!(result.n_evals, evals);
    assert_eq!(result.n_grads, grads);
    assert!(evals + grads <= config.budget);
    if analytic {
        assert!(grads > 0);
    } else {
        assert_eq!(grads, 0);
        assert_eq!(evals, config.budget);
    }
    assert_eq!(
        result.best_val,
        loss.value(ArrayView1::from(&result.best_pos))
    );
    let measured_best = traces
        .iter()
        .flatten()
        .filter(|e| e[0] == 0)
        .map(|e| {
            let x = Array1::from_iter(e[1..].iter().map(|&bits| f64::from_bits(bits)));
            loss.value(x.view())
        })
        .fold(f64::INFINITY, f64::min);
    assert_eq!(result.best_val, measured_best);
    assert!(result.coverage.published_samples > 0);
    assert!(result.coverage.applied_foreign_samples > 0);
    assert!(result.coverage.repelled_proposals > 0);
    assert_eq!(result.coverage.published_visits, 0);
    for (replica, out) in result.replicas.iter().enumerate() {
        let allowance = config.budget / 4 + usize::from(replica < config.budget % 4);
        assert!(out.n_evals + out.n_grads <= allowance);
        if let Some(hop) = &out.hop_state {
            assert_eq!(hop.val, loss.value(hop.pos.view()));
        }
    }
    traces
}

#[test]
fn shared_scalar_proposals_change_without_a_caller_gradient() {
    assert_eq!(check_shared(false), check_shared(false));
}

#[test]
fn shared_gradient_proposals_retain_raw_values_and_repeatability() {
    assert_eq!(check_shared(true), check_shared(true));
}

#[test]
fn unfunded_replicas_do_not_hold_the_checkpoint_barrier() {
    let loss = Loss::new();
    let config = PortfolioEnsembleConfig {
        replicas: 12,
        budget: 7,
        ..PortfolioEnsembleConfig::default()
    };
    let result = portfolio_ensemble_optimize::<_, NoGradient>(&loss, None, 9, None, &config);
    assert_eq!(result.n_evals, 7);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.replicas.len(), 7);
}

#[test]
fn panicking_replica_releases_waiting_peers() {
    let mut loss = Loss::new();
    loss.panic_at_start = true;
    let start = Array1::from_elem(8, -0.125);
    let config = PortfolioEnsembleConfig {
        budget: 4_000,
        ..PortfolioEnsembleConfig::default()
    };
    let result = std::panic::catch_unwind(|| {
        portfolio_ensemble_optimize::<_, NoGradient>(&loss, None, 9, Some(start.view()), &config)
    });
    assert!(result.is_err());
}
