//! Matched-budget stationary-network benchmark on a rotated N-D multiwell.
//!
//! The landscape has `2^D` known minima and `D 2^(D-1)` index-one edges. A
//! deterministic orthogonal rotation removes coordinate-axis assistance while
//! preserving those exact counts. Adaptive hybrid, ridge-only, and basin-only
//! policies use the same surface, witness, numerical controls, and PES budget.

use std::convert::Infallible;
use std::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::nd_hybrid::{NdHybridConfig, NdHybridPolicy, explore_nd_with_policy};
use anneal_core::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesSurface, RideMethod,
};
use ndarray::{Array1, ArrayView1};
use serde_json::json;

struct RotatedProductWell {
    dimension: usize,
    calls: AtomicU64,
}

impl RotatedProductWell {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            calls: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    fn angle(first: usize, second: usize) -> f64 {
        0.071 * ((first + 1) * (second + 2)) as f64
    }

    fn rotate_forward(&self, point: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut rotated = point.to_owned();
        for first in 0..self.dimension {
            for second in first + 1..self.dimension {
                let (sin, cos) = Self::angle(first, second).sin_cos();
                let left = rotated[first];
                let right = rotated[second];
                rotated[first] = cos * left - sin * right;
                rotated[second] = sin * left + cos * right;
            }
        }
        rotated
    }

    fn rotate_transpose(&self, point: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut rotated = point.to_owned();
        for first in (0..self.dimension).rev() {
            for second in (first + 1..self.dimension).rev() {
                let (sin, cos) = Self::angle(first, second).sin_cos();
                let left = rotated[first];
                let right = rotated[second];
                rotated[first] = cos * left + sin * right;
                rotated[second] = -sin * left + cos * right;
            }
        }
        rotated
    }
}

impl PesSurface for RotatedProductWell {
    type Error = Infallible;

    fn evaluate(&self, point: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let rotated = self.rotate_forward(point);
        let mut energy = 0.0;
        let mut rotated_gradient = Array1::zeros(self.dimension);
        for (index, coordinate) in rotated.iter().copied().enumerate() {
            let weight = 1.0 + 0.23 * index as f64;
            let well = coordinate * coordinate - 1.0;
            energy += weight * well * well;
            rotated_gradient[index] = 4.0 * weight * coordinate * well;
        }
        Ok((energy, self.rotate_transpose(rotated_gradient.view())))
    }
}

struct PointWitness;

impl ExactStructureWitness for PointWitness {
    fn equivalent(&self, left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> bool {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt()
            < 1e-5
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Moments {
    count: u64,
    mean: f64,
    squared_deviation: f64,
}

impl Moments {
    fn observe(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let updated_delta = value - self.mean;
        self.squared_deviation += delta * updated_delta;
    }

    fn count(&self) -> u64 {
        self.count
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn sample_standard_deviation(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.squared_deviation / (self.count - 1) as f64).sqrt()
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PolicySummary {
    minima: Moments,
    edges: Moments,
    stationary_objects: Moments,
    full_minimum_runs: u64,
    full_edge_runs: u64,
}

impl PolicySummary {
    fn observe(
        &mut self,
        minima: usize,
        edges: usize,
        stationary_objects: usize,
        expected: [usize; 2],
    ) {
        self.minima.observe(minima as f64);
        self.edges.observe(edges as f64);
        self.stationary_objects.observe(stationary_objects as f64);
        self.full_minimum_runs += u64::from(minima == expected[0]);
        self.full_edge_runs += u64::from(edges == expected[1]);
    }
}

fn policy_name(policy: NdHybridPolicy) -> &'static str {
    match policy {
        NdHybridPolicy::Adaptive => "adaptive",
        NdHybridPolicy::RidgeOnly => "ridge_only",
        NdHybridPolicy::BasinEscapeOnly => "basin_escape_only",
    }
}

fn parse_argument<T: std::str::FromStr>(index: usize, default: T) -> T {
    std::env::args()
        .nth(index)
        .and_then(|argument| argument.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let dimension = parse_argument(1, 5usize);
    let budget = parse_argument(2, 50_000u64);
    let seeds = parse_argument(3, 8u64);
    if !(2..=12).contains(&dimension) || budget == 0 || seeds == 0 {
        return Err("usage: nd_pes_benchmark [dimension 2..12] [budget >0] [seeds >0]".into());
    }

    let expected_minima = 1usize << dimension;
    let expected_edges = dimension * (1usize << (dimension - 1));
    let exploration = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 400,
        saddle_steps: 700,
        prfo_steps: 120,
        activation_attempts: 8,
        activation_growth: 1.8,
        quench_gradient_tolerance: 1e-8,
        quench_gradient_norm_tolerance: Some(3e-8),
        minimum_mode_force_tolerance: 5e-2,
        saddle_force_tolerance: 1e-7,
        saddle_displacement: 0.15,
        irc_step: 0.08,
        refine_with_prfo: true,
        ..PesExplorationConfig::default()
    };
    let config = NdHybridConfig {
        evaluation_budget: budget,
        ride_evaluation_cap: 4_000,
        escape_evaluation_cap: 500,
        ride_mode_blocks: 4,
        initial_escape_scale: 0.45,
        initial_acceptance_threshold: 10.0,
        visiting_q: 2.0,
        exploration,
    };
    let policies = [
        NdHybridPolicy::Adaptive,
        NdHybridPolicy::RidgeOnly,
        NdHybridPolicy::BasinEscapeOnly,
    ];
    let mut summaries = [PolicySummary::default(); 3];

    for seed in 0..seeds {
        for (policy_index, policy) in policies.iter().copied().enumerate() {
            let surface = RotatedProductWell::new(dimension);
            let rotated_start =
                Array1::from_shape_fn(dimension, |index| -1.0 + 0.11 * ((index + 1) as f64).sin());
            let start = surface.rotate_transpose(rotated_start.view());
            let report = explore_nd_with_policy(
                &surface,
                start.view(),
                &config,
                &PointWitness,
                seed,
                policy,
            )?;
            if surface.calls() != report.charged_evaluations {
                return Err(format!(
                    "surface charged {} calls but report retained {}",
                    surface.calls(),
                    report.charged_evaluations
                )
                .into());
            }
            if report.charged_evaluations > budget {
                return Err("a benchmark policy exceeded the matched PES budget".into());
            }
            let discovery_events = report
                .events
                .iter()
                .filter(|event| {
                    !event.new_minimum_ids.is_empty()
                        || !event.new_saddle_ids.is_empty()
                        || !event.new_unresolved_saddle_ids.is_empty()
                })
                .count();
            let stationary_objects = report.network.minimum_count()
                + report.network.saddle_count()
                + report.network.unresolved_saddles().len();
            summaries[policy_index].observe(
                report.network.minimum_count(),
                report.network.saddle_count(),
                stationary_objects,
                [expected_minima, expected_edges],
            );
            println!(
                "{}",
                json!({
                    "kind": "nd_pes_benchmark",
                    "dimension": dimension,
                    "seed": seed,
                    "policy": policy_name(policy),
                    "budget": budget,
                    "charged_evaluations": report.charged_evaluations,
                    "expected_minima": expected_minima,
                    "exact_minima": report.network.minimum_count(),
                    "minimum_coverage": report.network.minimum_count() as f64 / expected_minima as f64,
                    "expected_edges": expected_edges,
                    "certified_edges": report.network.saddle_count(),
                    "edge_coverage": report.network.saddle_count() as f64 / expected_edges as f64,
                    "unresolved_index_one_saddles": report.network.unresolved_saddles().len(),
                    "stationary_objects": stationary_objects,
                    "events": report.events.len(),
                    "discovery_events": discovery_events,
                    "mechanism_pulls": report.mechanism_pulls,
                    "mechanism_discovery_rates": report.mechanism_discovery_rates,
                    "move_pulls": report.move_pulls,
                    "move_success_rates": report.move_success_rates,
                    "escape_observations": report.escape_observations,
                    "escape_singletons": report.escape_singletons,
                    "escape_unseen_mass_upper": report.escape_unseen_mass_upper,
                    "escape_coverage_saturated": report.escape_coverage_saturated,
                    "termination": format!("{:?}", report.termination),
                })
            );
        }
    }
    for (policy, summary) in policies.into_iter().zip(summaries) {
        println!(
            "{}",
            json!({
                "kind": "nd_pes_benchmark_summary",
                "dimension": dimension,
                "policy": policy_name(policy),
                "budget": budget,
                "seeds": summary.minima.count(),
                "expected_minima": expected_minima,
                "mean_exact_minima": summary.minima.mean(),
                "sd_exact_minima": summary.minima.sample_standard_deviation(),
                "full_minimum_runs": summary.full_minimum_runs,
                "expected_edges": expected_edges,
                "mean_certified_edges": summary.edges.mean(),
                "sd_certified_edges": summary.edges.sample_standard_deviation(),
                "full_edge_runs": summary.full_edge_runs,
                "mean_stationary_objects": summary.stationary_objects.mean(),
                "sd_stationary_objects": summary.stationary_objects.sample_standard_deviation(),
            })
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Moments;

    #[test]
    fn benchmark_moments_report_sample_uncertainty() {
        let mut moments = Moments::default();
        for value in [1.0, 2.0, 3.0] {
            moments.observe(value);
        }

        assert_eq!(moments.count(), 3);
        assert!((moments.mean() - 2.0).abs() < 1e-14);
        assert!((moments.sample_standard_deviation() - 1.0).abs() < 1e-14);
    }
}
