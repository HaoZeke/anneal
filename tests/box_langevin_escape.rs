use std::sync::Mutex;

use anneal_core::methods::box_hopping::{
    BoxEnsembleConfig, BoxEscape, GleEscapeConfig, box_ensemble_optimize,
    box_values_ensemble_optimize,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::gle_langevin::GleNoise;
use eindir_core::{Bounds, GleThermostat, Gradient, Objective, optimal_sampling_drift};
use ndarray::{Array1, Array2, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

struct Surface {
    bounds: Bounds<f64>,
    linear: bool,
    objectives: Mutex<Vec<Array1<f64>>>,
    gradients: Mutex<Vec<Array1<f64>>>,
}

impl Surface {
    fn new(linear: bool) -> Self {
        Self {
            bounds: if linear {
                Bounds::new(array![0.0], array![1.0], 0.0)
            } else {
                Bounds::new(array![-1e6], array![1e6], 0.0)
            },
            linear,
            objectives: Mutex::new(Vec::new()),
            gradients: Mutex::new(Vec::new()),
        }
    }
}

impl Objective<f64> for Surface {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        assert_eq!(x.len(), 1);
        assert!(x[0].is_finite() && x[0] >= self.bounds.low[0] && x[0] <= self.bounds.high[0]);
        self.objectives.lock().unwrap().push(x.to_owned());
        if self.linear { x[0] } else { 0.0 }
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
        self.gradients.lock().unwrap().push(x.to_owned());
        array![if self.linear { 1.0 } else { 0.0 }]
    }

    fn dim(&self) -> usize {
        1
    }
}

fn drift(noise: GleNoise) -> Array2<f64> {
    match noise {
        GleNoise::Colored => optimal_sampling_drift(0.2),
        GleNoise::White { friction } => array![[friction]],
    }
}

fn config(noise: GleNoise, budget: usize) -> BoxEnsembleConfig {
    BoxEnsembleConfig {
        replicas: 1,
        budget,
        history: HistoryMode::None,
        shared_deposits: 0,
        escape: BoxEscape::Langevin(GleEscapeConfig {
            steps: 2,
            omega0: 0.2,
            dt: 0.2,
            noise,
        }),
        ..BoxEnsembleConfig::default()
    }
}

#[test]
fn box_langevin_keeps_noise_memory_and_quenches_terminal_endpoints() {
    for noise in [GleNoise::Colored, GleNoise::White { friction: 4.0 }] {
        let surface = Surface::new(false);
        let seed = 0x6c65;
        let start = array![0.0];
        let drift = drift(noise);
        let dt = 0.01;
        let mut rng = StdRng::seed_from_u64(seed);
        let hot = GleThermostat::canonical(&drift, dt, 5.0, 1.0);
        let covariance = Array2::eye(drift.nrows()) * 5.0;
        let mut state = hot.sample_stationary(&covariance, 1, 1.0, &mut rng);
        let mut x = start.clone();
        let mut expected = vec![start.clone()];
        for generation in 1..=2 {
            let temperature = 5.0 * std::f64::consts::LN_2 / (generation as f64 + 1.0).ln();
            if generation == 2 {
                state *= (temperature / 5.0).sqrt();
            }
            let thermostat = GleThermostat::canonical(&drift, dt, temperature, 1.0);
            for _ in 0..2 {
                x = &x + &(state.row(0).to_owned() * dt);
                expected.push(x.clone());
                thermostat.step(&mut state.view_mut(), &mut rng);
            }
            expected.push(x.clone());
        }

        let result = box_ensemble_optimize(
            &surface,
            &surface,
            seed,
            Some(start.view()),
            &config(noise, 20),
        );
        let objectives = surface.objectives.lock().unwrap();
        let gradients = surface.gradients.lock().unwrap();
        assert_eq!(result.hops, 2);
        assert_eq!((result.n_evals, result.n_grads), (7, 9));
        assert_eq!(objectives.len(), 7);
        assert_eq!(gradients.len(), 9);
        assert_eq!(
            *objectives, expected,
            "{noise:?}: persistent terminal trace"
        );
        assert_eq!(
            gradients[1], start,
            "fresh launch force for the first segment"
        );
        assert_eq!(gradients[5], expected[3], "fresh launch force after quench");
        assert_eq!(
            result.best_pos, start,
            "equal-value escapes do not replace the incumbent"
        );
        assert_eq!(result.history_observations, 0);
    }
}

#[test]
fn box_langevin_uses_raw_force_at_a_constrained_minimum() {
    let noise = GleNoise::Colored;
    let drift = drift(noise);
    let thermostat = GleThermostat::canonical(&drift, 0.01, 5.0, 1.0);
    let covariance = Array2::eye(drift.nrows()) * 5.0;
    let (seed, momentum) = (0..128)
        .find_map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let state = thermostat.sample_stationary(&covariance, 1, 1.0, &mut rng);
            (state[[0, 0]] > 0.5 && state[[0, 0]] < 20.0).then_some((seed, state[[0, 0]]))
        })
        .expect("a positive launch velocity");
    let surface = Surface::new(true);
    let result = box_ensemble_optimize(
        &surface,
        &surface,
        seed,
        Some(array![0.0].view()),
        &config(noise, 9),
    );
    let objectives = surface.objectives.lock().unwrap();
    let gradients = surface.gradients.lock().unwrap();
    let expected = (momentum - 0.005) * 0.01;
    assert_eq!(result.hops, 1);
    assert_eq!(
        objectives[1][0], expected,
        "raw gradient is one, not projected zero"
    );
    assert_eq!(gradients[0], array![0.0]);
    assert_eq!(gradients[1], array![0.0]);
    assert_eq!(gradients[2], array![expected]);
    assert_eq!(
        (result.n_evals, result.n_grads),
        (objectives.len(), gradients.len())
    );
    assert!(result.n_evals + result.n_grads <= 9);
}

#[test]
fn box_langevin_charges_segments_and_reserves_quench_work() {
    for noise in [GleNoise::Colored, GleNoise::White { friction: 4.0 }] {
        for history in [HistoryMode::None, HistoryMode::Private, HistoryMode::Shared] {
            for budget in 1..=64 {
                let surface = Surface::new(false);
                let mut config = config(noise, budget);
                config.replicas = 3;
                config.history = history;
                let result = box_ensemble_optimize(&surface, &surface, 13, None, &config);
                let counts = (
                    surface.objectives.lock().unwrap().len(),
                    surface.gradients.lock().unwrap().len(),
                );
                assert_eq!((result.n_evals, result.n_grads), counts);
                assert!(
                    counts.0 + counts.1 <= budget,
                    "{noise:?} {history:?} budget {budget}"
                );
                assert!(counts.0 > 0);
                if budget >= 48 {
                    assert!(result.hops >= 3, "every replica has a funded escape");
                    if !matches!(history, HistoryMode::None) {
                        assert!(result.history_observations >= 6);
                    }
                }
            }
        }
    }
}

#[test]
fn values_only_box_cannot_silently_ignore_langevin_selection() {
    let surface = Surface::new(false);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        box_values_ensemble_optimize(&surface, 3, None, &config(GleNoise::Colored, 20))
    }));
    assert!(result.is_err());
    assert!(surface.objectives.lock().unwrap().is_empty());
    assert!(surface.gradients.lock().unwrap().is_empty());
}
