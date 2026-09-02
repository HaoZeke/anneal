use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::nd_hybrid::{
    NdEscapeKernel, NdHybridConfig, NdHybridMechanism, NdHybridPolicy, explore_nd_hybrid,
    explore_nd_with_policy,
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
    let (ridge_after_saturation, escape_after_saturation, observed_after_saturation) =
        coverage_window.fold(
            (0usize, 0usize, 0usize),
            |(ridge, escape, observed), event| {
                (
                    ridge + usize::from(event.mechanism == NdHybridMechanism::Ridge),
                    escape + usize::from(event.mechanism == NdHybridMechanism::BasinEscape),
                    observed + 1,
                )
            },
        );
    assert_eq!(observed_after_saturation, 20);
    assert!(ridge_after_saturation > 0);
    assert!(escape_after_saturation > 0);
    let indexed_decisions = report
        .events
        .iter()
        .filter_map(|event| {
            Some((
                event,
                event.ridge_discovery_index?,
                event.escape_discovery_index?,
            ))
        })
        .collect::<Vec<_>>();
    assert!(!indexed_decisions.is_empty());
    for (event, ridge_index, escape_index) in indexed_decisions {
        match ridge_index.total_cmp(&escape_index) {
            std::cmp::Ordering::Greater => {
                assert_eq!(event.mechanism, NdHybridMechanism::Ridge)
            }
            std::cmp::Ordering::Less => {
                assert_eq!(event.mechanism, NdHybridMechanism::BasinEscape)
            }
            std::cmp::Ordering::Equal => {}
        }
    }
    assert!(report.escape_coverage_saturated);
    assert!(report.escape_unseen_mass_upper.unwrap() < 0.2);

    let mut escape_round = 0usize;
    let mut reconstructed_pulls = [0usize; 2];
    let mut reconstructed_discoveries = [0usize; 2];
    for event in &report.events {
        if event.mechanism == NdHybridMechanism::BasinEscape {
            escape_round += 1;
            let kernel = event
                .escape_kernel
                .expect("escape event must name its kernel");
            let arm = match kernel {
                NdEscapeKernel::Gaussian => 0,
                NdEscapeKernel::Tsallis => 1,
            };
            reconstructed_pulls[arm] += 1;
            reconstructed_discoveries[arm] += usize::from(!event.new_minimum_ids.is_empty());
            let probability = event
                .kernel_probability
                .expect("escape event must record its draw probability");
            let eta = event
                .kernel_learning_rate
                .expect("escape event must record eta_t");
            let gamma = event
                .kernel_implicit_exploration
                .expect("escape event must record gamma_t");
            let expected_eta = (2.0_f64.ln() / (2.0 * escape_round as f64)).sqrt();
            assert!(probability > 0.0 && probability <= 1.0);
            assert!((eta - expected_eta).abs() < 1e-15);
            assert!((gamma - eta / 2.0).abs() < 1e-15);
        } else {
            assert_eq!(event.escape_kernel, None);
            assert_eq!(event.kernel_probability, None);
            assert_eq!(event.kernel_learning_rate, None);
            assert_eq!(event.kernel_implicit_exploration, None);
        }
    }
    assert_eq!(report.move_pulls, reconstructed_pulls);
    for arm in 0..2 {
        let expected = reconstructed_discoveries[arm] as f64 / reconstructed_pulls[arm] as f64;
        assert!((report.move_success_rates[arm] - expected).abs() < 1e-15);
    }

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
