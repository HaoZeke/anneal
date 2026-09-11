use std::sync::Mutex;

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEnsembleResult, BoxEscape, GleEscapeConfig,
    box_ensemble_optimize_with_coverage, box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::gle_langevin::GleNoise;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};

struct Bowl {
    bounds: Bounds<f64>,
    flat: bool,
    objectives: Mutex<Vec<Array1<f64>>>,
    gradients: Mutex<usize>,
}

impl Objective<f64> for Bowl {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 1);
        assert!(x[0].is_finite() && (-1.0..=1.0).contains(&x[0]));
        self.objectives.lock().unwrap().push(x.to_owned());
        if self.flat { 0.0 } else { 0.5 * x.dot(&x) }
    }

    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

impl Gradient<f64> for Bowl {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        *self.gradients.lock().unwrap() += 1;
        if self.flat { array![0.0] } else { x.to_owned() }
    }

    fn dim(&self) -> usize {
        1
    }
}

fn run(
    values: bool,
    flat: bool,
    escape: BoxEscape,
    history: HistoryMode,
    radius: f64,
    height: f64,
) -> (BoxEnsembleResult, Vec<Array1<f64>>) {
    let objective = Bowl {
        bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
        flat,
        objectives: Mutex::new(Vec::new()),
        gradients: Mutex::new(0),
    };
    let config = BoxEnsembleConfig {
        replicas: 1,
        budget: 512,
        history,
        escape,
        ..BoxEnsembleConfig::default()
    };
    let coverage = BoxCoverageConfig {
        shared: false,
        radius,
        height,
        ..BoxCoverageConfig::default()
    };
    let start = array![0.0];
    let result = if values {
        box_values_ensemble_optimize_with_coverage(
            &objective, 7, Some(start.view()), &config, &coverage,
        )
    } else {
        box_ensemble_optimize_with_coverage(
            &objective, &objective, 7, Some(start.view()), &config, &coverage,
        )
    };
    let trace = objective.objectives.into_inner().unwrap();
    assert_eq!(result.n_evals, trace.len());
    assert_eq!(result.n_grads, objective.gradients.into_inner().unwrap());
    assert!(result.n_evals + result.n_grads <= config.budget);
    assert!(result.hops >= 4);
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.best_pos, start);
    assert_eq!(result.coverage.local_observations, result.hops + 1);
    if matches!(history, HistoryMode::None) {
        assert_eq!(result.history_minima, 0);
        assert_eq!(result.history_observations, 0);
        assert_eq!(result.history_cost, (0, 0, 0.0));
    }
    (result, trace)
}

fn mechanisms() -> [BoxEscape; 3] {
    [
        BoxEscape::Gaussian,
        BoxEscape::Langevin(GleEscapeConfig::default()),
        BoxEscape::Langevin(GleEscapeConfig {
            noise: GleNoise::White { friction: 4.0 },
            ..GleEscapeConfig::default()
        }),
    ]
}

#[test]
fn gradient_recrossing_changes_escape_without_a_minimum_report() {
    for escape in mechanisms() {
        let (_, inactive) = run(false, false, escape, HistoryMode::None, 0.01, 0.0);
        let (result, active) = run(false, false, escape, HistoryMode::None, 0.01, 0.1);
        assert_eq!(active[1], inactive[1], "the first launch has no return feedback");
        assert_eq!(result.coverage_decisions.accepted, result.hops);
        assert_ne!(
            active, inactive,
            "{escape:?}: a raw launch followed by return to covered zero must inform escape"
        );
    }
}

#[test]
fn values_recrossing_changes_escape_without_a_minimum_report() {
    let (_, inactive) = run(true, false, BoxEscape::Gaussian, HistoryMode::None, 0.05, 0.0);
    let (result, active) = run(true, false, BoxEscape::Gaussian, HistoryMode::None, 0.05, 0.1);
    assert_eq!(result.coverage_decisions.accepted, result.hops);
    assert_ne!(active, inactive, "covered pattern-search returns must inform escape");
}

#[test]
fn coverage_without_a_departure_preserves_the_escape_trace() {
    for flat in [false, true] {
        for escape in mechanisms() {
            let (_, inactive) = run(false, flat, escape, HistoryMode::None, 2.0, 0.0);
            let (_, active) = run(false, flat, escape, HistoryMode::None, 2.0, 0.1);
            assert_eq!(active, inactive, "{escape:?}: no departure is no recrossing");
        }
    }
}

#[test]
fn certified_feedback_is_not_counted_twice_as_coverage_feedback() {
    for escape in mechanisms() {
        let (_, inactive) = run(false, false, escape, HistoryMode::Private, 0.01, 0.0);
        let (result, active) = run(false, false, escape, HistoryMode::Private, 0.01, 0.1);
        assert!(result.history_observations > 1);
        assert_eq!(active, inactive, "{escape:?}: one feedback update per return");
    }
}
