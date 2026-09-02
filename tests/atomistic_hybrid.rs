use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::atomistic_hybrid::{
    AtomisticHybridConfig, AtomisticHybridMechanism, AtomisticHybridPolicy, AtomisticSystem,
    explore_atomistic_with_policy,
};
use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_search::Encounter;
use anneal_core::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesSurface, RideMethod,
};
use ndarray::{Array1, ArrayView1, array};

struct CountingRadialPairBarrier {
    calls: AtomicU64,
}

impl CountingRadialPairBarrier {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl PesSurface for CountingRadialPairBarrier {
    type Error = Infallible;

    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let displacement = [
            coordinates[3] - coordinates[0],
            coordinates[4] - coordinates[1],
            coordinates[5] - coordinates[2],
        ];
        let distance = displacement
            .iter()
            .map(|component| component * component)
            .sum::<f64>()
            .sqrt();
        let shifted = distance - 1.0;
        let energy = (shifted * shifted - 0.25).powi(2);
        let radial_gradient = 4.0 * shifted * (shifted * shifted - 0.25);
        let mut gradient = Array1::zeros(6);
        for axis in 0..3 {
            let component = radial_gradient * displacement[axis] / distance;
            gradient[axis] = -component;
            gradient[3 + axis] = component;
        }
        Ok((energy, gradient))
    }
}

struct PairDistanceWitness;

impl ExactStructureWitness for PairDistanceWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        fn distance(coordinates: ArrayView1<f64>) -> f64 {
            (0..3)
                .map(|axis| (coordinates[3 + axis] - coordinates[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        }
        (distance(left) - distance(right)).abs() < 1e-5
    }
}

fn system() -> AtomisticSystem {
    AtomisticSystem {
        species: vec![18, 18],
        masses: vec![1.0, 1.0],
        frozen_atoms: vec![false, false],
        identity_domain: "radial-pair".into(),
    }
}

fn config() -> AtomisticHybridConfig {
    AtomisticHybridConfig {
        evaluation_budget: 10_000,
        ride_evaluation_cap: 10_000,
        escape_evaluation_cap: 200,
        ride_modes_per_atom: 1,
        localization_radius: 2.0,
        escape_scales: vec![0.1, 0.4],
        minimum_information_samples: 128,
        information_length_scale: 1.0,
        information_amplitude: 1.0,
        information_noise: 1e-8,
        expected_ride_cost: 300.0,
        expected_escape_cost: 40.0,
        cost_prior_strength: 1.0,
        exploration: PesExplorationConfig {
            ride_method: RideMethod::Dimer,
            quench_steps: 300,
            saddle_steps: 600,
            irc_steps: 100,
            quench_gradient_tolerance: 1e-8,
            quench_gradient_norm_tolerance: Some(2e-8),
            saddle_force_tolerance: 1e-5,
            saddle_displacement: 0.1,
            negative_curvature_tolerance: 1e-7,
            hessian_step: 1e-5,
            maximum_move: 0.1,
            irc_step: 0.05,
            refine_with_prfo: false,
            ..PesExplorationConfig::default()
        },
    }
}

#[test]
fn ridge_only_atomistic_search_uses_the_budgeted_rgsaddle_path() {
    let surface = CountingRadialPairBarrier::new();
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let config = config();

    let report = explore_atomistic_with_policy(
        &surface,
        &descriptor_space,
        array![-0.25, 0.0, 0.0, 0.25, 0.0, 0.0].view(),
        &system(),
        &config,
        &PairDistanceWitness,
        0x5add1e,
        AtomisticHybridPolicy::RidgeOnly,
    )
    .unwrap();

    assert_eq!(report.charged_evaluations, surface.calls());
    assert!(report.charged_evaluations <= config.evaluation_budget);
    assert!(
        report
            .events
            .iter()
            .all(|event| event.mechanism == AtomisticHybridMechanism::Ridge)
    );
    assert_eq!(report.network.minimum_count(), 2);
    assert_eq!(report.network.saddle_count(), 1);
    assert_eq!(report.network.saddles()[0].negative_modes, 1);
    assert!(
        report
            .best_energy()
            .is_some_and(|energy| energy.abs() < 1e-8)
    );
    let action_calls = report
        .events
        .iter()
        .map(|event| event.charged_evaluations)
        .sum::<u64>();
    assert_eq!(
        report.first_encounter(0.0, 1e-8),
        Encounter::Found {
            charged: usize::try_from(report.charged_evaluations - action_calls).unwrap(),
            hops: 0,
        }
    );
    assert_eq!(
        report.first_encounter(-1.0, 1e-8),
        Encounter::Censored {
            charged: usize::try_from(report.charged_evaluations).unwrap(),
        }
    );
    assert!(
        report
            .network
            .minima()
            .iter()
            .all(|minimum| minimum.context.identity_domain() == Some("radial-pair"))
    );
    assert!(
        report
            .network
            .saddles()
            .iter()
            .all(|saddle| saddle.context.identity_domain() == Some("radial-pair"))
    );
}

#[test]
fn basin_only_atomistic_search_uses_invariant_minima_without_saddles() {
    let surface = CountingRadialPairBarrier::new();
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let mut config = config();
    config.evaluation_budget = 500;

    let report = explore_atomistic_with_policy(
        &surface,
        &descriptor_space,
        array![-0.25, 0.0, 0.0, 0.25, 0.0, 0.0].view(),
        &system(),
        &config,
        &PairDistanceWitness,
        0xdecaf,
        AtomisticHybridPolicy::BasinEscapeOnly,
    )
    .unwrap();

    assert_eq!(report.charged_evaluations, surface.calls());
    assert!(report.charged_evaluations <= config.evaluation_budget);
    assert!(!report.events.is_empty());
    assert!(
        report
            .events
            .iter()
            .all(|event| event.mechanism == AtomisticHybridMechanism::BasinEscape)
    );
    assert_eq!(report.network.saddle_count(), 0);
    assert!(report.network.minimum_count() >= 1);
}
