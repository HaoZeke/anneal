use std::sync::Mutex;

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEnsembleResult, BoxEscape, GleEscapeConfig,
    box_ensemble_optimize_with_coverage, box_values_ensemble_optimize_with_coverage,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::gle_langevin::GleNoise;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};

struct Surface {
    bounds: Bounds<f64>,
    flat: bool,
    samples: Mutex<Vec<Array1<f64>>>,
    gradients: Mutex<usize>,
}

impl Objective<f64> for Surface {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 1);
        assert!(x[0].is_finite() && (-1.0..=1.0).contains(&x[0]));
        self.samples.lock().unwrap().push(x.to_owned());
        if self.flat { 0.0 } else { 0.5 * x.dot(&x) }
    }

    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

impl Gradient<f64> for Surface {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        *self.gradients.lock().unwrap() += 1;
        if self.flat { array![0.0] } else { x.to_owned() }
    }

    fn dim(&self) -> usize {
        1
    }
}

fn run(
    seed: u64,
    values: bool,
    flat: bool,
    escape: BoxEscape,
    shared: bool,
    radius: f64,
) -> (BoxEnsembleResult, Vec<Array1<f64>>) {
    let surface = Surface {
        bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
        flat,
        samples: Mutex::new(Vec::new()),
        gradients: Mutex::new(0),
    };
    let config = BoxEnsembleConfig {
        replicas: 2,
        budget: 512,
        history: HistoryMode::None,
        escape,
        ..BoxEnsembleConfig::default()
    };
    let coverage = BoxCoverageConfig {
        radius,
        shared,
        ..BoxCoverageConfig::default()
    };
    let start = array![0.0];
    let result = if values {
        box_values_ensemble_optimize_with_coverage(
            &surface,
            seed,
            Some(start.view()),
            &config,
            &coverage,
        )
    } else {
        box_ensemble_optimize_with_coverage(
            &surface,
            &surface,
            seed,
            Some(start.view()),
            &config,
            &coverage,
        )
    };
    let samples = surface.samples.into_inner().unwrap();
    assert_eq!(result.n_evals, samples.len());
    assert_eq!(result.n_grads, surface.gradients.into_inner().unwrap());
    assert!(result.n_evals + result.n_grads <= config.budget);
    assert_eq!(result.best_val, 0.0);
    assert_eq!(result.best_pos, start);
    assert_eq!(result.history_minima, 0);
    assert_eq!(result.history_observations, 0);
    assert_eq!(result.coverage.local_observations, 2 + result.hops);
    (result, samples)
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
fn nearby_peer_displaces_the_paid_proposal_in_each_native_escape() {
    for escape in mechanisms() {
        for values in [false, true] {
            if values && !matches!(escape, BoxEscape::Gaussian) {
                continue;
            }
            let peer_index = if values { 19 } else { 1 };
            let proposal_index = if values { 38 } else { 2 };
            let radius = 0.1;
            let (seed, private, distance) = (0..64)
                .find_map(|seed| {
                    let (_, trace) = run(seed, values, true, escape, false, radius);
                    let distance = (trace[proposal_index][0] - trace[peer_index][0]).abs() / 2.0;
                    (distance > 1e-8 && distance < radius * 0.75).then_some((seed, trace, distance))
                })
                .expect("a reproducible near-peer proposal");
            let (_, shared) = run(seed, values, true, escape, true, radius);
            assert_eq!(shared[..proposal_index], private[..proposal_index]);
            let separation = (shared[proposal_index][0] - shared[peer_index][0]).abs() / 2.0;
            assert!(
                separation > distance,
                "{escape:?}, values={values}, seed={seed}: shared proposal must move away from the nearby peer; private={distance}, shared={separation}"
            );
        }
    }
}

#[test]
fn distant_peer_does_not_displace_the_first_proposal() {
    for escape in mechanisms() {
        let (_, private) = run(7, false, true, escape, false, 1e-12);
        let (_, shared) = run(7, false, true, escape, true, 1e-12);
        assert!((private[2][0] - private[1][0]).abs() > 1e-6);
        assert_eq!(shared[..3], private[..3]);
    }
}

#[test]
fn evaluated_launch_repels_even_when_both_quenches_return_to_the_same_point() {
    let radius = 0.1;
    let (seed, private, distance) = (0..128)
        .find_map(|seed| {
            let (_, trace) = run(seed, false, false, BoxEscape::Gaussian, false, radius);
            assert_eq!(trace[0], array![0.0]);
            assert_eq!(trace[2], array![0.0]);
            let distance = (trace[3][0] - trace[1][0]).abs() / 2.0;
            (distance > 1e-8 && distance < radius * 0.75 && trace[3][0].abs() / 2.0 > radius)
                .then_some((seed, trace, distance))
        })
        .expect("a launch near a peer excursion and away from the shared quench point");
    let (_, shared) = run(seed, false, false, BoxEscape::Gaussian, true, radius);
    assert_eq!(shared[..3], private[..3]);
    let separation = (shared[3][0] - shared[1][0]).abs() / 2.0;
    assert!(
        separation > distance,
        "paid sample history must survive quench collapse: private={distance}, shared={separation}"
    );
}
