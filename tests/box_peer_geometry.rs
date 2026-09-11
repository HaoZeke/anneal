use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEnsembleResult,
    box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Objective};
use ndarray::{ArrayView1, array};
use serde_json::{Value, json};

struct ScalarSurface {
    bounds: Bounds<f64>,
    calls: AtomicUsize,
    flat: bool,
}

impl Objective<f64> for ScalarSurface {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 1);
        assert!(x[0].is_finite() && (-1.0..=1.0).contains(&x[0]));
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.flat { 0.0 } else { 0.5 * x[0] * x[0] }
    }

    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

fn run(flat: bool, shared: bool, radius: f64, height: f64) -> (BoxEnsembleResult, Value) {
    let objective = ScalarSurface {
        bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
        calls: AtomicUsize::new(0),
        flat,
    };
    let config = BoxEnsembleConfig {
        replicas: 2,
        budget: 512,
        history: HistoryMode::None,
        ..BoxEnsembleConfig::default()
    };
    let coverage = BoxCoverageConfig {
        shared,
        radius,
        height,
        ..BoxCoverageConfig::default()
    };
    let result = box_values_ensemble_optimize_with_coverage(
        &objective, 7, Some(array![0.0].view()), &config, &coverage,
    );
    assert_eq!(result.n_evals, objective.calls.load(Ordering::Relaxed));
    assert_eq!(result.n_evals, config.budget);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.history_observations, 0);
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.best_pos, array![0.0]);
    let statistics = json!(result.coverage);
    (result, statistics)
}

fn count(statistics: &Value, field: &str) -> u64 {
    statistics[field].as_u64().unwrap_or_else(|| panic!("missing peer geometry counter: {field}"))
}

#[test]
fn occupied_peer_overlap_is_distinguished_from_proposal_overlap() {
    let (result, statistics) = run(false, true, 0.05, 0.1);
    let checks = count(&statistics, "sample_peer_checks");
    let anchors = count(&statistics, "sample_anchor_overlaps");
    let anchor_only = count(&statistics, "sample_anchor_only_overlaps");
    assert_eq!(checks, result.hops as u64);
    assert!(anchors > 0);
    assert!(anchor_only > 0);
    assert!(anchor_only <= anchors && anchors <= checks);
    assert!(anchor_only + result.coverage.sample_overlaps as u64 <= checks);
}

#[test]
fn whole_box_interaction_counts_both_geometries() {
    let (result, statistics) = run(true, true, 2.0, 0.1);
    assert_eq!(count(&statistics, "sample_peer_checks"), result.hops as u64);
    assert_eq!(count(&statistics, "sample_anchor_overlaps"), result.hops as u64);
    assert_eq!(count(&statistics, "sample_anchor_only_overlaps"), 0);
    assert_eq!(result.coverage.sample_overlaps, result.hops);
}

#[test]
fn disabled_peer_interaction_reports_no_geometry_checks() {
    for (shared, height) in [(false, 0.1), (true, 0.0)] {
        let (result, statistics) = run(false, shared, 0.05, height);
        assert_eq!(count(&statistics, "sample_peer_checks"), 0);
        assert_eq!(count(&statistics, "sample_anchor_overlaps"), 0);
        assert_eq!(count(&statistics, "sample_anchor_only_overlaps"), 0);
        assert_eq!(result.coverage.sample_overlaps, 0);
        assert_eq!(result.coverage.repelled_proposals, 0);
    }
}
