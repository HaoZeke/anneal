//! Matched-budget noise/history controls from explicit nonoptimal box starts.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anneal_core::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEscape, EnsembleHopResult, GleEscapeConfig,
    box_ensemble_optimize_with_coverage, box_values_ensemble_optimize_with_coverage,
    ensemble_hop_optimize,
};
use anneal_core::methods::ensemble::HistoryMode;
use anneal_core::methods::gle_langevin::GleNoise;
use anneal_core::methods::local_polish::projected_gradient_polish;
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

/// Values-only diagnostic with an interior optimum distinct from the box centre.
struct ControllerSurface {
    surface: Surface,
    optimum: Array1<f64>,
}

impl ControllerSurface {
    fn new(landscape: Landscape, dim: usize) -> Self {
        Self {
            surface: Surface {
                landscape,
                bounds: Bounds::new(
                    Array1::from_elem(dim, -5.12),
                    Array1::from_elem(dim, 5.12),
                    0.0,
                ),
                evaluations: AtomicUsize::new(0),
                gradients: AtomicUsize::new(0),
            },
            optimum: Array1::from_shape_fn(dim, |j| {
                0.7 + 0.3 * ((j + 1) as f64 * std::f64::consts::SQRT_2).sin()
            }),
        }
    }

    fn value(&self, x: ArrayView1<f64>) -> f64 {
        let mut shifted = &x - &self.optimum;
        if matches!(self.surface.landscape, Landscape::ConditionedQuadratic) {
            let twice_mean = 2.0 * shifted.sum() / shifted.len() as f64;
            shifted.mapv_inplace(|v| v - twice_mean);
        }
        self.surface.value(shifted.view())
    }
}

impl Objective<f64> for ControllerSurface {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.surface.evaluations.fetch_add(1, Ordering::Relaxed);
        assert_eq!(x.len(), self.surface.bounds.dims);
        assert!(
            x.iter()
                .all(|v| v.is_finite() && (-5.12..=5.12).contains(v))
        );
        self.value(x)
    }

    fn dim(&self) -> usize {
        self.surface.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.surface.bounds
    }
}

fn values_controller_records(
    landscape: Landscape,
    dim: usize,
    budget: usize,
    seed: u64,
) -> Vec<serde_json::Value> {
    assert!(dim > 0 && budget > 0);
    let mut start_rng = StdRng::seed_from_u64(seed ^ 0x5354_4152_545f_424f);
    let start = Array1::from_shape_fn(dim, |_| -5.12 + 10.24 * start_rng.random::<f64>());
    let mut records = Vec::new();
    for (arm, portfolio, replicas, shared) in [
        ("portfolio_single", true, 1, false),
        ("portfolio_independent", true, 4, false),
        ("hopping_single", false, 1, false),
        ("hopping_independent", false, 4, false),
        ("hopping_shared", false, 4, true),
    ] {
        let surface = ControllerSurface::new(landscape, dim);
        let config = BoxEnsembleConfig {
            replicas,
            budget,
            history: HistoryMode::None,
            ..BoxEnsembleConfig::default()
        };
        let coverage = BoxCoverageConfig {
            shared,
            ..BoxCoverageConfig::default()
        };
        let budgets = config.budgets();
        let replica_seeds: Vec<_> = (0..replicas)
            .map(|index| seed ^ (index as u64).wrapping_mul(0x9E37_79B9))
            .collect();
        let starts: Vec<_> = replica_seeds
            .iter()
            .enumerate()
            .map(|(index, &replica_seed)| {
                if index == 0 {
                    start.clone()
                } else {
                    let mut rng = StdRng::seed_from_u64(replica_seed);
                    Array1::from_shape_fn(dim, |_| -5.12 + 10.24 * rng.random::<f64>())
                }
            })
            .collect();
        let began = Instant::now();
        let results: Vec<EnsembleHopResult> = if portfolio {
            (0..replicas)
                .filter(|&index| budgets[index] > 0)
                .map(|index| {
                    ensemble_hop_optimize::<_, Surface>(
                        &surface,
                        None,
                        replica_seeds[index],
                        Some(starts[index].view()),
                        budgets[index],
                        1,
                        HistoryMode::None,
                        config.membership,
                    )
                })
                .collect()
        } else {
            vec![
                box_values_ensemble_optimize_with_coverage(
                    &surface,
                    seed,
                    Some(start.view()),
                    &config,
                    &coverage,
                )
                .into(),
            ]
        };
        let elapsed = began.elapsed().as_secs_f64();
        let n_evals: usize = results.iter().map(|out| out.n_evals).sum();
        let n_grads: usize = results.iter().map(|out| out.n_grads).sum();
        let observed_calls = surface.surface.evaluations.load(Ordering::Relaxed);
        assert_eq!(n_evals, observed_calls);
        assert_eq!(n_grads, 0);
        assert_eq!(surface.surface.gradients.load(Ordering::Relaxed), 0);
        assert!(n_evals > 0 && n_evals <= budget);
        let best = results
            .iter()
            .min_by(|a, b| a.best_val.total_cmp(&b.best_val))
            .unwrap();
        assert!(best.best_val.is_finite());
        assert!(
            best.best_pos
                .iter()
                .all(|v| v.is_finite() && (-5.12..=5.12).contains(v))
        );
        let verified_value = surface.value(best.best_pos.view());
        assert_eq!(best.best_val, verified_value);
        records.push(json!({
            "record": "result", "comparison": "values-controllers", "arm": arm,
            "controller": if portfolio { "portfolio" } else { "values-quasi-newton-hopping" },
            "landscape": format!("{landscape:?}"), "dimension": dim, "seed": seed,
            "transformation": if matches!(landscape, Landscape::ConditionedQuadratic) {
                "shifted-householder"
            } else { "shifted" },
            "optimum": surface.optimum.to_vec(),
            "budget": budget, "replicas": replicas, "replica_budgets": budgets,
            "replica_seeds": replica_seeds,
            "initial_positions": starts.iter().map(|x| x.to_vec()).collect::<Vec<_>>(),
            "initial_values": starts.iter().map(|x| surface.value(x.view())).collect::<Vec<_>>(),
            "n_evals": n_evals, "observed_calls": observed_calls, "n_grads": n_grads,
            "best_position": best.best_pos.to_vec(), "best_value": best.best_val,
            "verified_value": verified_value, "elapsed_seconds": elapsed,
            "history": "none", "coverage": if shared { "shared" } else { "private" },
            "coverage_radius": coverage.radius, "coverage_height": coverage.height,
            "hops": results.iter().map(|out| out.hops).sum::<usize>(),
            "coverage_published_samples": results.iter().map(|out| out.coverage.published_samples).sum::<u64>(),
            "coverage_applied_foreign_samples": results.iter().map(|out| out.coverage.applied_foreign_samples).sum::<usize>(),
            "coverage_sample_peer_checks": results.iter().map(|out| out.coverage.sample_peer_checks).sum::<usize>(),
            "coverage_sample_anchor_overlaps": results.iter().map(|out| out.coverage.sample_anchor_overlaps).sum::<usize>(),
            "coverage_sample_anchor_only_overlaps": results.iter().map(|out| out.coverage.sample_anchor_only_overlaps).sum::<usize>(),
            "coverage_sample_overlaps": results.iter().map(|out| out.coverage.sample_overlaps).sum::<usize>(),
            "coverage_repelled_proposals": results.iter().map(|out| out.coverage.repelled_proposals).sum::<usize>(),
        }));
    }
    records
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
    let coverage_only = match args.get(5).map(String::as_str) {
        None => false,
        Some("coverage") => true,
        Some("quench") => {
            quench_controls(dim, budget, seeds);
            return;
        }
        Some("controllers") => {
            println!(
                "{}",
                json!({
                    "record": "configuration", "comparison": "values-controllers",
                    "dimension": dim, "budget": budget, "seeds": seeds,
                    "objective_capability": "values", "execution": "serial",
                    "history": "none", "coverage_transport": "in-process",
                    "coverage_metric": "RMS-scaled-free-box-coordinates",
                    "version": env!("CARGO_PKG_VERSION"),
                })
            );
            for landscape in [Landscape::Rastrigin, Landscape::ConditionedQuadratic] {
                for seed in 0..seeds as u64 {
                    for record in values_controller_records(landscape, dim, budget, seed) {
                        println!("{record}");
                    }
                }
            }
            return;
        }
        Some(_) => panic!("the optional control mode is coverage, quench or controllers"),
    };
    let mut coverage_settings = BoxCoverageConfig::default();
    if let Some(radius) = args.get(6) {
        coverage_settings.radius = radius.parse().expect("positive coverage radius");
    }
    if let Some(height) = args.get(7) {
        coverage_settings.height = height.parse().expect("nonnegative coverage height");
    }
    let values_only = match args.get(8).map(String::as_str) {
        None | Some("gradient") => false,
        Some("values") => true,
        Some(_) => panic!("the objective capability must be gradient or values"),
    };
    let settings = GleEscapeConfig {
        steps,
        ..GleEscapeConfig::default()
    };
    let modes = if values_only {
        vec![("values", BoxEscape::Gaussian)]
    } else {
        vec![
            ("gaussian", BoxEscape::Gaussian),
            (
                "white",
                BoxEscape::Langevin(GleEscapeConfig {
                    noise: GleNoise::White { friction: 4.0 },
                    ..settings
                }),
            ),
            ("colored", BoxEscape::Langevin(settings)),
        ]
    };
    println!(
        "{}",
        json!({
            "record": "configuration", "dimension": dim, "budget": budget, "seeds": seeds,
            "replicas": 4, "steps": steps, "omega0": settings.omega0, "requested_dt": settings.dt,
            "white_friction": 4.0, "history_transport": "in-process",
            "comparison": if coverage_only { "coverage-only" } else { "minimum-history-and-coverage" },
            "objective_capability": if values_only { "values" } else { "gradient" },
            "coverage_transport": "in-process",
            "coverage_metric": "RMS-scaled-free-box-coordinates",
            "coverage_radius": coverage_settings.radius,
            "coverage_height": coverage_settings.height,
            "coverage_well_tempering": coverage_settings.well_tempering,
            "coverage_peer_weight": coverage_settings.peer_weight,
        "start_protocol": "seeded-uniform; independent-first-replica-start-stream",
            "version": env!("CARGO_PKG_VERSION"),
        })
    );
    for landscape in [Landscape::Rastrigin, Landscape::ConditionedQuadratic] {
        for seed in 0..seeds as u64 {
            for &(noise, escape) in &modes {
                for (history_name, history) in [
                    ("private", HistoryMode::Private),
                    ("shared", HistoryMode::Shared),
                ] {
                    let coverage = BoxCoverageConfig {
                        shared: matches!(history, HistoryMode::Shared),
                        ..coverage_settings.clone()
                    };
                    let history = if coverage_only {
                        HistoryMode::None
                    } else {
                        history
                    };
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
                    let result = if values_only {
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
                        "dimension": dim, "seed": seed, "noise": noise,
                        "history": if coverage_only { "none" } else { history_name },
                        "coverage": if coverage.shared { "shared" } else { "private" },
                        "initial_position": start.to_vec(),
                            "initial_value": initial_value, "best_value": result.best_val,
                            "n_evals": counts.0, "n_grads": counts.1, "budget": budget,
                        "hops": result.hops, "history_minima": result.history_minima,
                        "history_minima_scope": match history {
                            HistoryMode::None => "disabled",
                            HistoryMode::Private => "largest-private-table",
                            HistoryMode::Shared => "ensemble-shared-table",
                        },
                            "history_observations": result.history_observations,
                            "history_refusals": result.history_cost.1, "history_seconds": result.history_cost.2,
                            "shared_deposits": result.shared_deposits, "elapsed_seconds": elapsed,
                            "coverage_observations": result.coverage.local_observations,
                            "coverage_published": result.coverage.published_visits,
                            "coverage_applied_foreign": result.coverage.applied_foreign_visits,
                            "coverage_capped_foreign": result.coverage.capped_foreign_visits,
                            "coverage_regions_per_chain": result.coverage.per_chain_regions,
                            "coverage_recrossings": result.coverage.recrossings,
                            "coverage_peer_recrossings": result.coverage.peer_recrossings,
                            "coverage_peer_only_recrossings": result.coverage.peer_only_recrossings,
                            "coverage_escape_updates": result.coverage.escape_updates,
                            "coverage_novel_arrivals": result.coverage.novel_arrivals,
                            "coverage_novelty_updates": result.coverage.novelty_updates,
                            "coverage_published_samples": result.coverage.published_samples,
                            "coverage_applied_foreign_samples": result.coverage.applied_foreign_samples,
                            "coverage_sample_peer_checks": result.coverage.sample_peer_checks,
                            "coverage_sample_anchor_overlaps": result.coverage.sample_anchor_overlaps,
                            "coverage_sample_anchor_only_overlaps": result.coverage.sample_anchor_only_overlaps,
                            "coverage_sample_overlaps": result.coverage.sample_overlaps,
                            "coverage_repelled_proposals": result.coverage.repelled_proposals,
                            "coverage_constrained_repulsions": result.coverage.constrained_repulsions,
                            "coverage_decisions": result.coverage_decisions,
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
                bounds: Bounds::new(
                    Array1::from_elem(dim, -5.12),
                    Array1::from_elem(dim, 5.12),
                    0.0,
                ),
                evaluations: AtomicUsize::new(0),
                gradients: AtomicUsize::new(0),
            };
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5354_4152_545f_424f);
            let start = Array1::from_shape_fn(dim, |_| -5.12 + 10.24 * rng.random::<f64>());
            let allowance = (2 * dim + 8) * multiplier;
            let result = projected_gradient_polish(&surface, &surface, start, allowance, 1.0, 1e-8);
            let counts = (
                surface.evaluations.load(Ordering::Relaxed),
                surface.gradients.load(Ordering::Relaxed),
            );
            assert_eq!((result.n_evals, result.n_grads), counts);
            // This quadratic's descent direction points inward at either
            // bound, so projection leaves every gradient component unchanged.
            let certificate = result
                .best_grad
                .as_ref()
                .map(|gradient| gradient.iter().map(|g| g.abs()).fold(0.0, f64::max));
            println!(
                "{}",
                json!({
                    "record": "quench-control", "dimension": dim, "seed": seed,
                    "allowance": allowance, "multiplier": multiplier,
                    "best_value": result.best_val, "max_projected_gradient": certificate,
                    "history_admissible": certificate.is_some_and(|g| g < 1e-3),
                    "n_evals": counts.0, "n_grads": counts.1,
                    "within_replica_budget": counts.0 + counts.1 <= ensemble_budget / 4,
                })
            );
        }
    }
}

#[cfg(test)]
mod controller_tests {
    use super::*;

    #[test]
    fn portfolio_peer_control_keeps_the_independent_controller_starts_and_work() {
        let baseline = values_controller_records(Landscape::Rastrigin, 2, 257, 13);
        let coverage = BoxCoverageConfig {
            radius: 0.4,
            height: 0.0,
            ..BoxCoverageConfig::default()
        };
        let peer = portfolio_peer_record(Landscape::Rastrigin, 2, 257, 13, &coverage);
        for field in [
            "initial_positions", "replica_seeds", "replica_budgets", "budget",
            "best_position", "best_value", "n_evals", "n_grads",
        ] {
            assert_eq!(peer[field], baseline[1][field], "field {field}");
        }
        assert_eq!(peer["coverage_published_samples"], 0);
    }

    #[test]
    fn portfolio_peer_record_reports_actual_scalar_work_and_delivery() {
        let coverage = BoxCoverageConfig {
            radius: 0.4,
            ..BoxCoverageConfig::default()
        };
        let peer = portfolio_peer_record(Landscape::ConditionedQuadratic, 2, 257, 13, &coverage);
        assert_eq!(peer["arm"], "portfolio_shared");
        assert_eq!(peer["controller"], "portfolio");
        assert_eq!(peer["transformation"], "shifted-householder");
        assert_eq!(peer["n_evals"], 257);
        assert_eq!(peer["observed_calls"], 257);
        assert_eq!(peer["n_grads"], 0);
        assert_eq!(peer["best_value"], peer["verified_value"]);
        assert!(peer["coverage_applied_foreign_samples"].as_u64().unwrap() > 0);
    }

    #[test]
    fn controller_surfaces_have_noncentral_optima() {
        for landscape in [Landscape::Rastrigin, Landscape::ConditionedQuadratic] {
            let surface = ControllerSurface::new(landscape, 8);
            assert_eq!(surface.value(surface.optimum.view()), 0.0);
            assert!(surface.value(Array1::zeros(8).view()) > 1.0);
            assert!(surface.optimum.iter().all(|x| (0.4..=1.0).contains(x)));
            assert_eq!(surface.surface.evaluations.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn controller_records_separate_policy_splitting_and_sharing() {
        let records = values_controller_records(Landscape::ConditionedQuadratic, 2, 256, 7);
        let names: Vec<_> = records
            .iter()
            .map(|row| row["arm"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "portfolio_single",
                "portfolio_independent",
                "hopping_single",
                "hopping_independent",
                "hopping_shared"
            ]
        );
        for row in &records {
            assert_eq!(row["budget"], 256);
            assert_eq!(row["n_grads"], 0);
            assert!(row["n_evals"].as_u64().unwrap() > 0);
            assert!(row["n_evals"].as_u64().unwrap() <= 256);
            assert_eq!(row["n_evals"], row["observed_calls"]);
            assert_eq!(row["best_value"], row["verified_value"]);
        }
        assert_eq!(records[0]["replicas"], 1);
        assert_eq!(records[2]["replicas"], 1);
        for index in [1, 3, 4] {
            assert_eq!(records[index]["replicas"], 4);
            assert_eq!(records[index]["replica_budgets"], json!([64, 64, 64, 64]));
        }
    }

    #[test]
    fn controller_comparisons_match_starts_and_isolate_sample_delivery() {
        let records = values_controller_records(Landscape::Rastrigin, 2, 257, 13);
        assert_eq!(
            records[1]["initial_positions"],
            records[3]["initial_positions"]
        );
        assert_eq!(
            records[1]["initial_positions"],
            records[4]["initial_positions"]
        );
        assert_eq!(
            records[0]["initial_positions"][0],
            records[1]["initial_positions"][0]
        );
        assert_eq!(
            records[0]["initial_positions"],
            records[2]["initial_positions"]
        );
        assert_eq!(records[1]["replica_budgets"], json!([65, 64, 64, 64]));
        for row in &records[..4] {
            assert_eq!(row["coverage_applied_foreign_samples"], 0);
        }
        assert!(
            records[4]["coverage_applied_foreign_samples"]
                .as_u64()
                .unwrap()
                > 0
        );
    }
}
