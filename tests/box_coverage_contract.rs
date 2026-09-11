use std::sync::Mutex;

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEnsembleResult, box_ensemble_optimize_with_coverage,
    box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};

struct Plateau {
    bounds: Bounds<f64>,
    trace: Mutex<Vec<Array1<f64>>>,
    gradients: Mutex<usize>,
    finite: bool,
}

impl Plateau {
    fn new(finite: bool) -> Self {
        Self {
            bounds: Bounds::new(
                array![-1000.0, 4.0, -0.002],
                array![3000.0, 4.0, 0.006],
                0.0,
            ),
            trace: Mutex::new(Vec::new()),
            gradients: Mutex::new(0),
            finite,
        }
    }

    fn run(
        &self,
        gradient: bool,
        config: &BoxEnsembleConfig,
        coverage: &BoxCoverageConfig,
    ) -> BoxEnsembleResult {
        let result = if gradient {
            box_ensemble_optimize_with_coverage(self, self, 71, None, config, coverage)
        } else {
            box_values_ensemble_optimize_with_coverage(self, 71, None, config, coverage)
        };
        assert_eq!(result.n_evals, self.trace.lock().unwrap().len());
        assert_eq!(result.n_grads, *self.gradients.lock().unwrap());
        assert!(result.n_evals + result.n_grads <= config.budget);
        assert_eq!(result.history_minima, 0);
        assert_eq!(result.history_observations, 0);
        assert_eq!(
            result.shared_deposits,
            result.coverage.applied_foreign_visits
        );
        result
    }
}

impl Objective<f64> for Plateau {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 3);
        for (x, (lo, hi)) in x
            .iter()
            .zip(self.bounds.low.iter().zip(self.bounds.high.iter()))
        {
            assert!(x.is_finite() && x >= lo && x <= hi);
        }
        self.trace.lock().unwrap().push(x.to_owned());
        if self.finite { 0.0 } else { f64::INFINITY }
    }

    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

impl Gradient<f64> for Plateau {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        *self.gradients.lock().unwrap() += 1;
        Array1::zeros(x.len())
    }

    fn dim(&self) -> usize {
        3
    }
}

fn config() -> BoxEnsembleConfig {
    BoxEnsembleConfig {
        replicas: 4,
        budget: 512,
        history: HistoryMode::None,
        ..BoxEnsembleConfig::default()
    }
}

#[test]
fn coverage_shares_without_a_minimum_ledger_and_counts_only_search_boundaries() {
    for gradient in [false, true] {
        let objective = Plateau::new(true);
        let config = config();
        let result = objective.run(gradient, &config, &BoxCoverageConfig::default());
        assert!(result.hops >= config.replicas);
        assert_eq!(
            result.coverage.local_observations,
            config.replicas + result.hops
        );
        assert_eq!(
            result.coverage.published_visits,
            result.coverage.local_observations as u64
        );
        assert!(result.coverage.applied_foreign_visits > 0);
        assert!(
            result.coverage.applied_foreign_visits
                <= result.coverage.local_observations * (config.replicas - 1)
        );
        if gradient {
            assert_eq!(
                result.coverage.local_observations, result.n_evals,
                "a flat gradient quench evaluates only its search boundary"
            );
        } else {
            assert!(
                result.coverage.local_observations < result.n_evals,
                "pattern-search and certificate probes are not extra coverage visits"
            );
        }
        assert_eq!(result.coverage.per_chain_regions.len(), config.replicas);
    }
}

#[test]
fn zero_coverage_height_preserves_the_unbiased_search_path() {
    for gradient in [false, true] {
        let shared = Plateau::new(true);
        let private = Plateau::new(true);
        let coverage = BoxCoverageConfig {
            height: 0.0,
            ..BoxCoverageConfig::default()
        };
        let first = shared.run(gradient, &config(), &coverage);
        let second = private.run(
            gradient,
            &config(),
            &BoxCoverageConfig {
                shared: false,
                ..coverage
            },
        );
        assert_eq!(
            *shared.trace.lock().unwrap(),
            *private.trace.lock().unwrap()
        );
        assert_eq!(
            (first.n_evals, first.n_grads, first.hops),
            (second.n_evals, second.n_grads, second.hops)
        );
        assert_eq!(first.best_pos, second.best_pos);
        assert_eq!(first.best_val, second.best_val);
        assert!(first.coverage.applied_foreign_visits > 0);
        assert_eq!(second.coverage.applied_foreign_visits, 0);
        assert_eq!(second.coverage.published_visits, 0);
    }
}

#[test]
fn coverage_radius_is_normalized_and_peer_caps_apply_per_receiving_region() {
    for gradient in [false, true] {
        let objective = Plateau::new(true);
        let config = BoxEnsembleConfig {
            shared_deposits: 1,
            ..config()
        };
        let coverage = BoxCoverageConfig {
            radius: 2.0,
            ..BoxCoverageConfig::default()
        };
        let result = objective.run(gradient, &config, &coverage);
        assert_eq!(result.coverage.per_chain_regions, vec![1; config.replicas]);
        assert!(result.coverage.capped_foreign_visits > 0);
        assert!(result.coverage.applied_foreign_visits > 0);
        assert!(result.coverage.applied_foreign_visits <= result.hops);
    }
}

#[test]
fn minimum_match_tolerance_does_not_set_coverage_reach() {
    for gradient in [false, true] {
        let tight = Plateau::new(true);
        let loose = Plateau::new(true);
        let coverage = BoxCoverageConfig::default();
        let first = tight.run(
            gradient,
            &BoxEnsembleConfig {
                identity_tol: 1e-12,
                ..config()
            },
            &coverage,
        );
        let second = loose.run(
            gradient,
            &BoxEnsembleConfig {
                identity_tol: 1.0,
                ..config()
            },
            &coverage,
        );
        assert_eq!(*tight.trace.lock().unwrap(), *loose.trace.lock().unwrap());
        assert_eq!(first.coverage, second.coverage);
    }
}

#[test]
fn nonfinite_values_are_not_coverage_observations() {
    for gradient in [false, true] {
        let objective = Plateau::new(false);
        let config = config();
        let result = objective.run(gradient, &config, &BoxCoverageConfig::default());
        assert_eq!(result.coverage.local_observations, 0);
        assert_eq!(result.coverage.published_visits, 0);
        assert_eq!(result.coverage.applied_foreign_visits, 0);
        assert_eq!(result.coverage.per_chain_regions, vec![0; config.replicas]);
    }
}

#[test]
fn rejected_finite_trials_still_record_coverage() {
    struct Needle {
        bounds: Bounds<f64>,
        calls: Mutex<usize>,
    }

    impl Objective<f64> for Needle {
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            assert_eq!(x.len(), 1);
            assert!(x[0].is_finite() && (-1.0..=1.0).contains(&x[0]));
            *self.calls.lock().unwrap() += 1;
            if x[0] == 0.0 { 0.0 } else { 1e6 }
        }

        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
    }

    let objective = Needle {
        bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
        calls: Mutex::new(0),
    };
    let config = BoxEnsembleConfig {
        replicas: 1,
        budget: 256,
        history: HistoryMode::None,
        ..BoxEnsembleConfig::default()
    };
    let start = array![0.0];
    let result = box_values_ensemble_optimize_with_coverage(
        &objective,
        71,
        Some(start.view()),
        &config,
        &BoxCoverageConfig::default(),
    );
    assert_eq!(result.n_evals, *objective.calls.lock().unwrap());
    assert!(result.n_evals <= config.budget);
    assert_eq!(result.n_grads, 0);
    assert!(result.hops > 2);
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.best_pos, start);
    assert_eq!(
        result.coverage.local_observations,
        1 + result.hops,
        "finite uphill trials remain explored even though acceptance underflows to zero"
    );
    assert!(result.coverage.per_chain_regions[0] > 1);
    assert_eq!(result.coverage.applied_foreign_visits, 0);
}

#[test]
fn catalog_environment_cannot_enable_a_box_coverage_penalty() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "zero_coverage_height_preserves_the_unbiased_search_path",
            "--nocapture",
        ])
        .env("CATALOG_ENTROPIC_BIAS", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
