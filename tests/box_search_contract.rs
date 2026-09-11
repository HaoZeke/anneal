use std::sync::Mutex;

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEnsembleResult, BoxEscape, GleEscapeConfig,
    box_ensemble_optimize_with_coverage, box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::gle_langevin::GleNoise;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};

struct FlatObjective {
    offset: f64,
    bounds: Bounds<f64>,
    evaluations: Mutex<Vec<Array1<f64>>>,
    gradients: Mutex<Vec<Array1<f64>>>,
}

impl FlatObjective {
    fn new(offset: f64) -> Self {
        Self {
            offset,
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 0.0),
            evaluations: Mutex::new(Vec::new()),
            gradients: Mutex::new(Vec::new()),
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
        assert_eq!(result.n_evals, self.evaluations.lock().unwrap().len());
        assert_eq!(result.n_grads, self.gradients.lock().unwrap().len());
        assert!(result.n_evals + result.n_grads <= config.budget);
        assert_eq!(result.best_val, self.offset);
        result
    }
}

impl Objective<f64> for FlatObjective {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 2);
        assert!(x.iter().all(|x| x.is_finite() && (-1.0..=1.0).contains(x)));
        self.evaluations.lock().unwrap().push(x.to_owned());
        self.offset
    }

    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

impl Gradient<f64> for FlatObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.lock().unwrap().push(x.to_owned());
        Array1::zeros(x.len())
    }

    fn dim(&self) -> usize {
        2
    }
}

fn same_trace(left: &Mutex<Vec<Array1<f64>>>, right: &Mutex<Vec<Array1<f64>>>) {
    let left = left.lock().unwrap();
    let right = right.lock().unwrap();
    assert_eq!(left.len(), right.len(), "callback count differs");
    let difference = left.iter().zip(right.iter()).position(|(a, b)| a != b);
    assert_eq!(difference, None, "first differing callback index");
}

fn origin_contract(gradient: bool, escape: BoxEscape) {
    for shared in [false, true] {
        let config = BoxEnsembleConfig {
            replicas: 4,
            budget: 4096,
            history: if shared {
                HistoryMode::Shared
            } else {
                HistoryMode::Private
            },
            escape,
            ..BoxEnsembleConfig::default()
        };
        let coverage = BoxCoverageConfig {
            shared,
            ..BoxCoverageConfig::default()
        };
        let original = FlatObjective::new(0.0);
        let shifted = FlatObjective::new(1024.0);
        let a = original.run(gradient, &config, &coverage);
        let b = shifted.run(gradient, &config, &coverage);
        same_trace(&original.evaluations, &shifted.evaluations);
        same_trace(&original.gradients, &shifted.gradients);
        assert_eq!(a.best_pos, b.best_pos);
        assert_eq!(
            (a.n_evals, a.n_grads, a.hops),
            (b.n_evals, b.n_grads, b.hops)
        );
    }
}

#[test]
fn gradient_gaussian_search_is_independent_of_objective_origin() {
    origin_contract(true, BoxEscape::Gaussian);
}

#[test]
fn values_search_is_independent_of_objective_origin() {
    origin_contract(false, BoxEscape::Gaussian);
}

#[test]
fn white_langevin_search_is_independent_of_objective_origin() {
    origin_contract(
        true,
        BoxEscape::Langevin(GleEscapeConfig {
            noise: GleNoise::White { friction: 4.0 },
            ..GleEscapeConfig::default()
        }),
    );
}

#[test]
fn colored_langevin_search_is_independent_of_objective_origin() {
    origin_contract(true, BoxEscape::Langevin(GleEscapeConfig::default()));
}

#[test]
fn whole_box_coverage_delivers_visits_without_direct_height_influence() {
    for gradient in [false, true] {
        let objective = FlatObjective::new(0.0);
        let config = BoxEnsembleConfig {
            replicas: 4,
            budget: 512,
            history: HistoryMode::None,
            ..BoxEnsembleConfig::default()
        };
        let coverage = BoxCoverageConfig {
            radius: 2.0,
            height: 10.0,
            ..BoxCoverageConfig::default()
        };
        let result = objective.run(gradient, &config, &coverage);
        assert!(result.coverage.applied_foreign_visits > 0);
        assert_eq!(result.coverage.per_chain_regions, vec![1; config.replicas]);
        let decisions = result.coverage_decisions;
        assert_eq!(decisions.comparisons, result.hops);
        assert_eq!(decisions.unresolved, 0);
        assert!(decisions.peer_overlap > 0);
        assert_eq!(decisions.peer_delta_changes, 0);
        assert_eq!(decisions.probability_changes, 0);
        assert_eq!(decisions.drawn_disagreements, 0);
        assert_eq!(decisions.probability_change_sum, 0.0);
        assert_eq!(decisions.max_abs_peer_delta, 0.0);
    }
}

#[test]
fn overlapping_peer_coverage_can_change_acceptance_probabilities() {
    for gradient in [false, true] {
        let objective = FlatObjective::new(0.0);
        let config = BoxEnsembleConfig {
            replicas: 4,
            budget: 512,
            history: HistoryMode::None,
            ..BoxEnsembleConfig::default()
        };
        let coverage = BoxCoverageConfig {
            radius: 0.2,
            height: 10.0,
            ..BoxCoverageConfig::default()
        };
        let result = objective.run(gradient, &config, &coverage);
        let decisions = result.coverage_decisions;
        assert_eq!(decisions.comparisons, result.hops);
        assert_eq!(decisions.unresolved, 0);
        assert!(decisions.peer_overlap > 0);
        assert!(decisions.peer_delta_changes > 0);
        assert!(decisions.probability_changes > 0);
        assert!(decisions.probability_change_sum > 0.0);
        assert!(decisions.max_probability_change <= 1.0);
        assert!(decisions.drawn_disagreements <= decisions.drawn_comparisons);
    }
}
