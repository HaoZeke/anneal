use std::sync::Mutex;

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEscape, GleEscapeConfig,
    box_ensemble_optimize_with_coverage, box_values_ensemble_optimize_with_coverage,
    ensemble_hop_optimize_with_config,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::gle_langevin::GleNoise;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};

struct LoggedBox {
    bounds: Bounds<f64>,
    values: Mutex<Vec<Array1<f64>>>,
    gradients: Mutex<Vec<Array1<f64>>>,
}

impl LoggedBox {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, 3.0, -1.0], array![1.0, 3.0, 1.0], 0.0),
            values: Mutex::new(Vec::new()),
            gradients: Mutex::new(Vec::new()),
        }
    }

    fn check_position(&self, x: ArrayView1<f64>) {
        assert_eq!(x.len(), 3);
        assert_eq!(x[1], 3.0);
        for axis in [0, 2] {
            assert!(x[axis].is_finite() && (-1.0..=1.0).contains(&x[axis]));
        }
    }
}

impl Objective<f64> for LoggedBox {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.check_position(x);
        self.values.lock().unwrap().push(x.to_owned());
        0.0
    }
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

impl Gradient<f64> for LoggedBox {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.check_position(x);
        self.gradients.lock().unwrap().push(x.to_owned());
        Array1::zeros(3)
    }
    fn dim(&self) -> usize {
        3
    }
}

fn matches_engine(gradient: bool, escape: BoxEscape, replicas: usize) {
    for shared in [false, true] {
        let direct = LoggedBox::new();
        let configured = LoggedBox::new();
        let config = BoxEnsembleConfig {
            replicas,
            budget: 512,
            history: HistoryMode::None,
            shared_deposits: 3,
            escape,
            ..BoxEnsembleConfig::default()
        };
        let coverage = BoxCoverageConfig {
            radius: 0.2,
            height: 10.0,
            peer_weight: 0.25,
            shared,
            ..BoxCoverageConfig::default()
        };
        let start = array![0.5, 3.0, -0.25];
        let expected = if gradient {
            box_ensemble_optimize_with_coverage(
                &direct,
                &direct,
                71,
                Some(start.view()),
                &config,
                &coverage,
            )
        } else {
            box_values_ensemble_optimize_with_coverage(
                &direct,
                71,
                Some(start.view()),
                &config,
                &coverage,
            )
        };
        let actual = ensemble_hop_optimize_with_config(
            &configured,
            gradient.then_some(&configured),
            71,
            Some(start.view()),
            &config,
            &coverage,
        );
        assert_eq!(
            *configured.values.lock().unwrap(),
            *direct.values.lock().unwrap()
        );
        assert_eq!(
            *configured.gradients.lock().unwrap(),
            *direct.gradients.lock().unwrap()
        );
        assert_eq!(actual.best_pos, expected.best_pos);
        assert_eq!(actual.best_val, expected.best_val);
        assert_eq!(actual.n_evals, configured.values.lock().unwrap().len());
        assert_eq!(actual.n_grads, configured.gradients.lock().unwrap().len());
        assert_eq!(
            (actual.n_evals, actual.n_grads),
            (expected.n_evals, expected.n_grads)
        );
        assert_eq!(actual.charged, actual.n_evals + actual.n_grads);
        assert!(actual.charged <= config.budget);
        assert_eq!(actual.hops, expected.hops);
        assert!(actual.hops > 0);
        assert_eq!(actual.history_observations, 0);
        assert_eq!(actual.history_minima, 0);
        assert_eq!(actual.history_cost, (0, 0, 0.0));
        assert_eq!(actual.coverage, expected.coverage);
        assert_eq!(actual.coverage_decisions, expected.coverage_decisions);
        assert!(actual.coverage.local_observations > 0);
        if shared && replicas > 1 {
            assert!(actual.coverage.applied_foreign_visits > 0);
        } else {
            assert_eq!(actual.coverage.applied_foreign_visits, 0);
        }
    }
}

#[test]
fn configured_gaussian_entry_matches_the_gradient_engine() {
    matches_engine(true, BoxEscape::Gaussian, 4);
}

#[test]
fn configured_white_entry_matches_the_gradient_engine() {
    matches_engine(
        true,
        BoxEscape::Langevin(GleEscapeConfig {
            noise: GleNoise::White { friction: 4.0 },
            ..GleEscapeConfig::default()
        }),
        4,
    );
}

#[test]
fn configured_colored_entry_matches_the_gradient_engine() {
    matches_engine(true, BoxEscape::Langevin(GleEscapeConfig::default()), 4);
}

#[test]
fn configured_values_entry_matches_the_replica_engine() {
    matches_engine(false, BoxEscape::Gaussian, 4);
}

#[test]
fn configured_single_values_replica_retains_hops_and_coverage() {
    matches_engine(false, BoxEscape::Gaussian, 1);
}

#[test]
fn configured_values_entry_rejects_langevin_before_callbacks() {
    for replicas in [1, 4] {
        let objective = LoggedBox::new();
        let config = BoxEnsembleConfig {
            replicas,
            escape: BoxEscape::Langevin(GleEscapeConfig::default()),
            ..BoxEnsembleConfig::default()
        };
        let rejected = std::panic::catch_unwind(|| {
            ensemble_hop_optimize_with_config::<_, LoggedBox>(
                &objective,
                None,
                71,
                None,
                &config,
                &BoxCoverageConfig::default(),
            )
        });
        assert!(rejected.is_err());
        assert!(objective.values.lock().unwrap().is_empty());
        assert!(objective.gradients.lock().unwrap().is_empty());
    }
}
