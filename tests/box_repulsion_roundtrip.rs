use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEnsembleResult, box_ensemble_optimize_with_coverage,
    box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::portfolio::values_local_polish;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};

struct Flat {
    bounds: Bounds<f64>,
    points: Mutex<Vec<Array1<f64>>>,
    gradients: AtomicUsize,
}

impl Flat {
    fn new(bounds: Bounds<f64>) -> Self {
        Self {
            bounds,
            points: Mutex::new(Vec::new()),
            gradients: AtomicUsize::new(0),
        }
    }
}

impl Objective<f64> for Flat {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert!(self.bounds.contains(x));
        self.points.lock().unwrap().push(x.to_owned());
        0.0
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

impl Gradient<f64> for Flat {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        Array1::zeros(x.len())
    }
    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

struct Run {
    points: Vec<Array1<f64>>,
    result: BoxEnsembleResult,
}

fn run(bounds: &Bounds<f64>, start: &Array1<f64>, shared: bool, values: bool) -> Run {
    let objective = Flat::new(bounds.clone());
    let config = BoxEnsembleConfig {
        replicas: 4,
        budget: 64,
        history: HistoryMode::None,
        ..BoxEnsembleConfig::default()
    };
    let coverage = BoxCoverageConfig {
        shared,
        radius: 0.36,
        ..BoxCoverageConfig::default()
    };
    let result = if values {
        box_values_ensemble_optimize_with_coverage(
            &objective,
            133,
            Some(start.view()),
            &config,
            &coverage,
        )
    } else {
        box_ensemble_optimize_with_coverage(
            &objective,
            &objective,
            133,
            Some(start.view()),
            &config,
            &coverage,
        )
    };
    let points = objective.points.into_inner().unwrap();
    assert_eq!(result.n_evals, points.len());
    assert_eq!(result.n_grads, objective.gradients.load(Ordering::Relaxed));
    assert!(result.n_evals + result.n_grads <= config.budget);
    if values {
        assert_eq!(result.n_grads, 0);
    }
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.best_pos, *start);
    assert_eq!(result.history_observations, 0);
    Run { points, result }
}

fn distance(a: ArrayView1<f64>, b: ArrayView1<f64>, bounds: &Bounds<f64>) -> f64 {
    let widths = &bounds.high - &bounds.low;
    let free = widths.iter().filter(|w| **w > 0.0).count();
    let scale = 1.0 / (free as f64).sqrt();
    let normalized =
        |point: ArrayView1<f64>, j: usize| ((point[j] - bounds.low[j]) / widths[j]) * scale;
    widths
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > 0.0)
        .map(|(j, _)| (normalized(a, j) - normalized(b, j)).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn clearance(point: ArrayView1<f64>, peers: &[Array1<f64>], bounds: &Bounds<f64>) -> f64 {
    peers
        .iter()
        .map(|peer| distance(point, peer.view(), bounds))
        .fold(f64::INFINITY, f64::min)
}

fn rounded_case(
    fixed: bool,
    values: bool,
) -> (Array1<f64>, Array1<f64>, Vec<Array1<f64>>, Bounds<f64>, f64) {
    let low = 1e10;
    let spacing = 1.9073486328125e-6;
    let point = |a: f64, b: f64| {
        if fixed {
            array![low + a * spacing, low + b * spacing, 7.0]
        } else {
            array![low + a * spacing, low + b * spacing]
        }
    };
    let bounds = Bounds::new(point(0.0, 0.0), point(4.0, 6.0), 0.0);
    let start = point(3.0, 2.0);
    let reference = run(&bounds, &start, false, false);
    let starts = reference.points[..4].to_vec();
    let peers = starts[1..].to_vec();
    assert_eq!(
        peers,
        vec![point(0.0, 2.0), point(4.0, 6.0), point(3.0, 4.0)]
    );
    let private = if values {
        run(&bounds, &start, false, true)
    } else {
        reference
    };
    let shared = run(&bounds, &start, true, values);
    let initialization = if values {
        let mut initialization = Vec::new();
        for position in &starts {
            let objective = Flat::new(bounds.clone());
            let local = values_local_polish(&objective, position.clone(), 14, 0.1, 1e-12);
            let points = objective.points.into_inner().unwrap();
            assert_eq!(local.n_evals, points.len());
            initialization.extend(points);
        }
        initialization
    } else {
        starts
    };
    assert_eq!(
        &private.points[..initialization.len()],
        initialization.as_slice()
    );
    assert_eq!(
        &shared.points[..initialization.len()],
        initialization.as_slice()
    );
    let initial = private.points[initialization.len()].clone();
    let proposal = shared.points[initialization.len()].clone();
    assert_eq!(initial, point(2.0, 3.0));
    let radius = 0.36;
    let before = clearance(initial.view(), &peers, &bounds);
    let allowance = radius - before;
    let sideways = point(2.0, 2.0);
    assert!(clearance(sideways.view(), &peers, &bounds) > before);
    assert!(distance(initial.view(), sideways.view(), &bounds) <= allowance);
    assert!(shared.result.coverage.applied_foreign_samples >= peers.len());
    assert!(shared.result.coverage.repelled_proposals > 0);
    assert_eq!(private.result.coverage.applied_foreign_samples, 0);
    assert!(bounds.contains(proposal.view()));
    if fixed {
        assert_eq!(proposal[2], 7.0);
    }
    (initial, proposal, peers, bounds, allowance)
}

#[test]
fn representable_parameter_repulsion_increases_actual_peer_clearance() {
    for fixed in [false, true] {
        for values in [false, true] {
            let (initial, proposal, peers, bounds, _) = rounded_case(fixed, values);
            let before = clearance(initial.view(), &peers, &bounds);
            let after = clearance(proposal.view(), &peers, &bounds);
            assert!(
                after > before,
                "physical clearance decreased: {before} -> {after}"
            );
        }
    }
}

#[test]
fn representable_parameter_repulsion_keeps_the_geometric_step_cap() {
    for fixed in [false, true] {
        for values in [false, true] {
            let (initial, proposal, _, bounds, allowance) = rounded_case(fixed, values);
            let displacement = distance(initial.view(), proposal.view(), &bounds);
            assert!(
                displacement <= allowance + 16.0 * f64::EPSILON,
                "physical displacement {displacement} exceeds allowance {allowance}"
            );
        }
    }
}
