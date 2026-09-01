use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::nd_hybrid::{
    NdHybridConfig, NdHybridMechanism, NdHybridPolicy, explore_nd_hybrid, explore_nd_with_policy,
};
use anneal_core::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesSurface, RideMethod,
};
use ndarray::{Array1, ArrayView1, array};

struct ProductDoubleWell {
    calls: AtomicU64,
}

impl ProductDoubleWell {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl PesSurface for ProductDoubleWell {
    type Error = Infallible;

    fn evaluate(&self, point: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let weights = [1.0, 1.3, 1.7, 2.1];
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(point.len());
        for (index, coordinate) in point.iter().copied().enumerate() {
            let well = coordinate * coordinate - 1.0;
            energy += weights[index] * well * well;
            gradient[index] = 4.0 * weights[index] * coordinate * well;
        }
        Ok((energy, gradient))
    }
}

struct PointWitness;

impl ExactStructureWitness for PointWitness {
    fn equivalent(&self, left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> bool {
        (&left - &right).dot(&(&left - &right)).sqrt() < 1e-5
    }
}

#[test]
fn hybrid_nd_search_shares_escaped_minima_with_budgeted_ridge_rides() {
    let surface = ProductDoubleWell::new();
    let exploration = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 300,
        saddle_steps: 500,
        prfo_steps: 100,
        activation_attempts: 8,
        activation_growth: 1.8,
        quench_gradient_tolerance: 1e-8,
        quench_gradient_norm_tolerance: Some(2e-8),
        minimum_mode_force_tolerance: 5e-2,
        saddle_force_tolerance: 1e-7,
        saddle_displacement: 0.15,
        irc_step: 0.08,
        refine_with_prfo: true,
        ..PesExplorationConfig::default()
    };
    let config = NdHybridConfig {
        evaluation_budget: 30_000,
        ride_evaluation_cap: 3_000,
        escape_evaluation_cap: 500,
        ride_mode_blocks: 1,
        initial_escape_scale: 0.45,
        initial_acceptance_threshold: 10.0,
        visiting_q: 2.0,
        exploration,
    };

    let report = explore_nd_hybrid(
        &surface,
        array![-0.91, -1.08, -0.87, -1.12].view(),
        &config,
        &PointWitness,
        0xdecaf,
    )
    .unwrap();

    assert_eq!(report.charged_evaluations, surface.calls());
    assert!(report.charged_evaluations <= config.evaluation_budget);
    assert!(report.network.minimum_count() >= 3);
    assert!(report.network.saddle_count() >= 1);
    assert!(report.mechanism_pulls.iter().all(|pulls| *pulls > 0));
    assert!(report.move_pulls.iter().all(|pulls| *pulls > 0));

    let escaped = report
        .events
        .iter()
        .enumerate()
        .find_map(|(index, event)| {
            (event.mechanism == NdHybridMechanism::BasinEscape)
                .then(|| {
                    event
                        .new_minimum_ids
                        .first()
                        .copied()
                        .map(|basin| (index, basin))
                })
                .flatten()
        })
        .expect("a basin escape must discover an exact minimum");
    assert!(report.events[escaped.0 + 1..].iter().any(|event| {
        event.mechanism == NdHybridMechanism::Ridge && event.source_basin == Some(escaped.1)
    }));

    let saturated = report
        .events
        .iter()
        .position(|event| event.escape_coverage_saturated)
        .expect("the exact escape census must acquire a finite unseen-mass bound");
    let coverage_window = report.events[saturated + 1..].iter().take(20);
    let (ridge_after_saturation, observed_after_saturation) =
        coverage_window.fold((0usize, 0usize), |(ridge, observed), event| {
            (
                ridge + usize::from(event.mechanism == NdHybridMechanism::Ridge),
                observed + 1,
            )
        });
    assert_eq!(observed_after_saturation, 20);
    assert!(
        ridge_after_saturation >= 15,
        "saturated escape coverage issued only {ridge_after_saturation}/20 ridge events"
    );
    assert!(report.escape_coverage_saturated);
    assert!(report.escape_unseen_mass_upper.unwrap() < 0.2);

    for minimum in report.network.minima() {
        assert!(minimum.max_gradient < 2e-8);
        assert!(
            minimum
                .coordinates
                .iter()
                .all(|coordinate| (coordinate.abs() - 1.0).abs() < 1e-6)
        );
    }
    for saddle in report.network.saddles() {
        assert_eq!(saddle.negative_modes, 1);
        assert!(saddle.saddle_max_gradient < config.exploration.saddle_force_tolerance);
        assert!(
            saddle
                .endpoints
                .iter()
                .all(|endpoint| *endpoint < report.network.minimum_count())
        );
    }
}

#[test]
fn fixed_nd_policies_issue_only_the_named_mechanism() {
    let exploration = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 200,
        saddle_steps: 300,
        quench_gradient_tolerance: 1e-8,
        quench_gradient_norm_tolerance: Some(2e-8),
        saddle_force_tolerance: 1e-6,
        saddle_displacement: 0.15,
        irc_step: 0.08,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };
    let config = NdHybridConfig {
        evaluation_budget: 4_000,
        ride_evaluation_cap: 1_000,
        escape_evaluation_cap: 200,
        ride_mode_blocks: 4,
        initial_escape_scale: 0.45,
        initial_acceptance_threshold: 10.0,
        visiting_q: 2.0,
        exploration,
    };

    for (policy, expected) in [
        (NdHybridPolicy::RidgeOnly, NdHybridMechanism::Ridge),
        (
            NdHybridPolicy::BasinEscapeOnly,
            NdHybridMechanism::BasinEscape,
        ),
    ] {
        let surface = ProductDoubleWell::new();
        let report = explore_nd_with_policy(
            &surface,
            array![-0.91, -1.08, -0.87, -1.12].view(),
            &config,
            &PointWitness,
            0x51ce,
            policy,
        )
        .unwrap();

        assert!(!report.events.is_empty());
        assert!(
            report
                .events
                .iter()
                .all(|event| event.mechanism == expected)
        );
        assert_eq!(report.charged_evaluations, surface.calls());
        assert!(report.charged_evaluations <= config.evaluation_budget);
    }
}
