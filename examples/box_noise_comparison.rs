//! Matched-budget noise/history controls from explicit nonoptimal box starts.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anneal_core::methods::box_hopping::{
    BoxEnsembleConfig, BoxEscape, GleEscapeConfig, box_ensemble_optimize,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::gle_langevin::GleNoise;
use anneal_core::methods::local_polish::{projected_gradient, projected_gradient_polish};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::json;

#[derive(Clone, Copy, Debug)]
enum Landscape {
    Rastrigin,
    ConditionedQuadratic,
}

struct Surface {
    landscape: Landscape,
    bounds: Bounds<f64>,
    evaluations: AtomicUsize,
    gradients: AtomicUsize,
}

impl Surface {
    fn value(&self, x: ArrayView1<f64>) -> f64 {
        match self.landscape {
            Landscape::Rastrigin => x
                .iter()
                .map(|v| v * v + 10.0 * (1.0 - (std::f64::consts::TAU * v).cos()))
                .sum(),
            Landscape::ConditionedQuadratic => x
                .iter()
                .enumerate()
                .map(|(j, v)| 0.5 * self.curvature(j) * v * v)
                .sum(),
        }
    }

    fn curvature(&self, j: usize) -> f64 {
        1000.0_f64.powf(j as f64 / self.bounds.dims.saturating_sub(1).max(1) as f64)
    }
}

impl Objective<f64> for Surface {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        assert_eq!(x.len(), self.bounds.dims);
        assert!(
            x.iter()
                .all(|v| v.is_finite() && (-5.12..=5.12).contains(v))
        );
        self.value(x)
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
}

impl Gradient<f64> for Surface {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        match self.landscape {
            Landscape::Rastrigin => x.mapv(|v| {
                2.0 * v + 10.0 * std::f64::consts::TAU * (std::f64::consts::TAU * v).sin()
            }),
            Landscape::ConditionedQuadratic => {
                Array1::from_iter(x.iter().enumerate().map(|(j, v)| self.curvature(j) * v))
            }
        }
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let number = |index: usize, default: usize| {
        args.get(index)
            .map(|s| s.parse::<usize>().expect("positive integer argument"))
            .unwrap_or(default)
    };
    let dim = number(1, 32);
    let budget = number(2, 8_000);
    let seeds = number(3, 4);
    let steps = number(4, 16);
    assert!(dim > 0 && budget > 0 && seeds > 0 && steps > 0);
    assert!(
        std::env::var_os("HISTORY_NNG").is_none(),
        "this control measures in-process shared history"
    );
    if let Some(mode) = args.get(5) {
        assert_eq!(mode, "quench", "the optional control mode is quench");
        quench_controls(dim, budget, seeds);
        return;
    }
    let settings = GleEscapeConfig {
        steps,
        ..GleEscapeConfig::default()
    };
    let modes = [
        ("gaussian", BoxEscape::Gaussian),
        (
            "white",
            BoxEscape::Langevin(GleEscapeConfig {
                noise: GleNoise::White { friction: 4.0 },
                ..settings
            }),
        ),
        ("colored", BoxEscape::Langevin(settings)),
    ];
    println!(
        "{}",
        json!({
            "record": "configuration", "dimension": dim, "budget": budget, "seeds": seeds,
            "replicas": 4, "steps": steps, "omega0": settings.omega0, "requested_dt": settings.dt,
            "white_friction": 4.0, "history_transport": "in-process",
        "start_protocol": "seeded-uniform; independent-first-replica-start-stream",
            "version": env!("CARGO_PKG_VERSION"),
        })
    );
    for landscape in [Landscape::Rastrigin, Landscape::ConditionedQuadratic] {
        for seed in 0..seeds as u64 {
            for (noise, escape) in modes {
                for (history_name, history) in [
                    ("private", HistoryMode::Private),
                    ("shared", HistoryMode::Shared),
                ] {
                    let surface = Surface {
                        landscape,
                        bounds: Bounds::new(
                            Array1::from_elem(dim, -5.12),
                            Array1::from_elem(dim, 5.12),
                            0.0,
                        ),
                        evaluations: AtomicUsize::new(0),
                        gradients: AtomicUsize::new(0),
                    };
                    let mut start_rng = StdRng::seed_from_u64(seed ^ 0x5354_4152_545f_424f);
                    let start =
                        Array1::from_shape_fn(dim, |_| -5.12 + 10.24 * start_rng.random::<f64>());
                    let initial_value = surface.value(start.view());
                    assert!(initial_value > 0.0);
                    let config = BoxEnsembleConfig {
                        budget,
                        history,
                        escape,
                        ..BoxEnsembleConfig::default()
                    };
                    let began = Instant::now();
                    let result = box_ensemble_optimize(
                        &surface,
                        &surface,
                        seed,
                        Some(start.view()),
                        &config,
                    );
                    let elapsed = began.elapsed().as_secs_f64();
                    let counts = (
                        surface.evaluations.load(Ordering::Relaxed),
                        surface.gradients.load(Ordering::Relaxed),
                    );
                    assert_eq!((result.n_evals, result.n_grads), counts);
                    assert!(counts.0 + counts.1 <= budget);
                    assert_eq!(result.best_val, surface.value(result.best_pos.view()));
                    println!(
                        "{}",
                        json!({
                            "record": "result", "landscape": format!("{landscape:?}"),
                        "dimension": dim, "seed": seed, "noise": noise, "history": history_name,
                        "initial_position": start.to_vec(),
                            "initial_value": initial_value, "best_value": result.best_val,
                            "n_evals": counts.0, "n_grads": counts.1, "budget": budget,
                        "hops": result.hops, "history_minima": result.history_minima,
                        "history_minima_scope": if matches!(history, HistoryMode::Private) {
                            "largest-private-table"
                        } else { "ensemble-shared-table" },
                            "history_observations": result.history_observations,
                            "history_refusals": result.history_cost.1, "history_seconds": result.history_cost.2,
                            "shared_deposits": result.shared_deposits, "elapsed_seconds": elapsed,
                        })
                    );
                }
            }
        }
    }
}

/// Resolve whether the dimension-only quench allowance supplies a certificate.
fn quench_controls(dim: usize, ensemble_budget: usize, seeds: usize) {
    for seed in 0..seeds as u64 {
        for multiplier in [1, 2, 4, 8] {
            let surface = Surface {
                landscape: Landscape::ConditionedQuadratic,
                bounds: Bounds::new(Array1::from_elem(dim, -5.12), Array1::from_elem(dim, 5.12), 0.0),
                evaluations: AtomicUsize::new(0),
                gradients: AtomicUsize::new(0),
            };
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5354_4152_545f_424f);
            let start = Array1::from_shape_fn(dim, |_| -5.12 + 10.24 * rng.random::<f64>());
            let allowance = (2 * dim + 8) * multiplier;
            let result = projected_gradient_polish(&surface, &surface, start, allowance, 1.0, 1e-8);
            let counts = (surface.evaluations.load(Ordering::Relaxed), surface.gradients.load(Ordering::Relaxed));
            assert_eq!((result.n_evals, result.n_grads), counts);
            let certificate = result.best_grad.as_ref().map(|gradient| {
                projected_gradient(&result.best_pos, gradient, &surface.bounds.low, &surface.bounds.high)
                    .iter().map(|g| g.abs()).fold(0.0, f64::max)
            });
            println!("{}", json!({
                "record": "quench-control", "dimension": dim, "seed": seed,
                "allowance": allowance, "multiplier": multiplier,
                "best_value": result.best_val, "max_projected_gradient": certificate,
                "history_admissible": certificate.is_some_and(|g| g < 1e-3),
                "n_evals": counts.0, "n_grads": counts.1,
                "within_replica_budget": counts.0 + counts.1 <= ensemble_budget / 4,
            }));
        }
    }
}
