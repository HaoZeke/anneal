use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::pes_exploration::{
    ExactStructureRelation, ExactStructureWitness, NdPesNetwork, PesExplorationConfig, PesSurface,
    QuenchedMinimum, RideMethod, discover_nd_connection, discover_nd_connection_with_budget,
};
use ndarray::{Array1, ArrayView1};

/// One bistable coordinate coupled to four stable coordinates. Five is
/// deliberately not a Cartesian 3N dimension.
struct FiveDimensionalDoubleWell;

impl PesSurface for FiveDimensionalDoubleWell {
    type Error = Infallible;

    fn evaluate(&self, point: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        let reaction = point[0];
        let mut energy = (reaction * reaction - 1.0).powi(2);
        let mut gradient = Array1::zeros(point.len());
        gradient[0] = 4.0 * reaction * (reaction * reaction - 1.0);
        for index in 1..point.len() {
            let stiffness = index as f64 + 1.0;
            energy += stiffness * point[index] * point[index];
            gradient[index] = 2.0 * stiffness * point[index];
        }
        Ok((energy, gradient))
    }
}

struct CountingFiveDimensionalDoubleWell {
    calls: AtomicU64,
}

impl CountingFiveDimensionalDoubleWell {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl PesSurface for CountingFiveDimensionalDoubleWell {
    type Error = Infallible;

    fn evaluate(&self, point: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        FiveDimensionalDoubleWell.evaluate(point)
    }
}

struct ShallowDoubleWell;

impl PesSurface for ShallowDoubleWell {
    type Error = Infallible;

    fn evaluate(&self, point: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        let reaction = point[0];
        let energy = 0.025 * (reaction * reaction - 1.0).powi(2);
        let gradient = Array1::from_vec(vec![0.1 * reaction * (reaction * reaction - 1.0)]);
        Ok((energy, gradient))
    }
}

/// A shallow barrier satisfies the force gate while the local
/// unstable-coordinate error still exceeds the IRC path radius.
struct OffsetShallowDoubleWell;

impl PesSurface for OffsetShallowDoubleWell {
    type Error = Infallible;

    fn evaluate(&self, point: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        let reaction = point[0];
        let scale = 5e-4;
        let energy = scale * (reaction * reaction - 1.0).powi(2);
        let gradient = Array1::from_vec(vec![4.0 * scale * reaction * (reaction * reaction - 1.0)]);
        Ok((energy, gradient))
    }
}

/// The reaction valley bends away from the launch ray. Along `y = 0` the
/// coupling keeps every sampled x-direction curvature positive, while a
/// constrained y relaxation exposes the negative mode of the double well.
struct CurvedFiveDimensionalDoubleWell;

impl PesSurface for CurvedFiveDimensionalDoubleWell {
    type Error = Infallible;

    fn evaluate(&self, point: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        let reaction = point[0];
        let progress = reaction + 1.0;
        let valley_offset = point[1] - progress * progress;
        let coupling = 10.0;
        let mut energy =
            (reaction * reaction - 1.0).powi(2) + coupling * valley_offset * valley_offset;
        let mut gradient = Array1::zeros(point.len());
        gradient[0] = 4.0 * reaction * (reaction * reaction - 1.0)
            - 4.0 * coupling * progress * valley_offset;
        gradient[1] = 2.0 * coupling * valley_offset;
        for index in 2..point.len() {
            let stiffness = index as f64 + 1.0;
            energy += stiffness * point[index] * point[index];
            gradient[index] = 2.0 * stiffness * point[index];
        }
        Ok((energy, gradient))
    }
}

struct PointWitness;

impl ExactStructureWitness for PointWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt()
            < 1e-5
    }
}

struct PermutationDoubleWell;

impl PesSurface for PermutationDoubleWell {
    type Error = Infallible;

    fn evaluate(&self, point: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        let inverse_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
        let reaction = (point[0] - point[1]) * inverse_sqrt_two;
        let stable = (point[0] + point[1]) * inverse_sqrt_two;
        let energy = (reaction * reaction - 1.0).powi(2) + stable * stable;
        let reaction_gradient = 4.0 * reaction * (reaction * reaction - 1.0);
        let stable_gradient = 2.0 * stable;
        Ok((
            energy,
            Array1::from_vec(vec![
                (reaction_gradient + stable_gradient) * inverse_sqrt_two,
                (-reaction_gradient + stable_gradient) * inverse_sqrt_two,
            ]),
        ))
    }
}

struct SwapWitness;

impl ExactStructureWitness for SwapWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        self.relation(left, right).is_equivalent()
    }

    fn relation(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> ExactStructureRelation {
        let direct = ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)).sqrt();
        if direct < 1e-6 {
            return ExactStructureRelation::Equivalent;
        }
        let swapped = ((left[0] - right[1]).powi(2) + (left[1] - right[0]).powi(2)).sqrt();
        if swapped < 1e-6 {
            ExactStructureRelation::NontrivialPermutation(vec![1, 0])
        } else {
            ExactStructureRelation::Distinct
        }
    }
}

#[test]
fn generic_nd_ride_preserves_a_permutation_certified_self_loop() {
    let mut network = NdPesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 300,
        saddle_steps: 500,
        quench_gradient_tolerance: 1e-8,
        saddle_force_tolerance: 1e-6,
        saddle_displacement: 0.2,
        irc_step: 0.08,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };
    let scale = std::f64::consts::FRAC_1_SQRT_2;

    let connection = discover_nd_connection(
        &PermutationDoubleWell,
        &mut network,
        Array1::from_vec(vec![-scale, scale]).view(),
        Array1::from_vec(vec![scale, -scale]).view(),
        &config,
        &SwapWitness,
    )
    .unwrap();

    assert_eq!(connection.endpoints[0], connection.endpoints[1]);
    assert_eq!(network.minimum_count(), 1);
    assert_eq!(network.saddle_count(), 1);
    assert_eq!(connection.negative_modes, 1);
    assert_eq!(network.saddle_observations(), 1);
    assert_eq!(network.saddle_singletons(), 1);

    discover_nd_connection(
        &PermutationDoubleWell,
        &mut network,
        Array1::from_vec(vec![-scale, scale]).view(),
        Array1::from_vec(vec![scale, -scale]).view(),
        &config,
        &SwapWitness,
    )
    .unwrap();

    assert_eq!(network.saddle_count(), 1);
    assert_eq!(network.saddle_observations(), 2);
    assert_eq!(network.saddle_singletons(), 0);
}

#[test]
fn exact_source_admission_retains_isolated_nd_minima() {
    let mut network = NdPesNetwork::new();
    let first = QuenchedMinimum {
        energy: -1.0,
        coordinates: Array1::from_vec(vec![-1.0, 0.0]),
        gradient: Array1::zeros(2),
        max_gradient: 0.0,
    };

    let first_admission = network.admit_minimum(first, &PointWitness);
    assert!(first_admission.is_new);
    assert_eq!(first_admission.id, 0);
    assert_eq!(first_admission.nearest_coordinate_distance, None);

    let duplicate = QuenchedMinimum {
        energy: -1.1,
        coordinates: Array1::from_vec(vec![-1.0 + 1e-7, 0.0]),
        gradient: Array1::from_vec(vec![1e-10, 0.0]),
        max_gradient: 1e-10,
    };
    let duplicate_admission = network.admit_minimum(duplicate, &PointWitness);
    assert!(!duplicate_admission.is_new);
    assert_eq!(duplicate_admission.id, 0);
    assert!(duplicate_admission.nearest_coordinate_distance.unwrap() < 1e-6);
    assert_eq!(network.minimum_count(), 1);
    assert_eq!(network.minima()[0].energy, -1.1);

    let distinct = QuenchedMinimum {
        energy: -0.9,
        coordinates: Array1::from_vec(vec![1.0, 0.0]),
        gradient: Array1::zeros(2),
        max_gradient: 0.0,
    };
    let distinct_admission = network.admit_minimum(distinct, &PointWitness);
    assert!(distinct_admission.is_new);
    assert_eq!(distinct_admission.id, 1);
    assert!(distinct_admission.nearest_coordinate_distance.unwrap() > 1.9);
    assert_eq!(network.minimum_count(), 2);
}

#[test]
fn generic_nd_ride_has_a_hard_pes_call_boundary() {
    let surface = CountingFiveDimensionalDoubleWell::new();
    let mut network = NdPesNetwork::new();
    let config = PesExplorationConfig {
        quench_steps: 300,
        saddle_steps: 500,
        quench_gradient_tolerance: 1e-8,
        saddle_force_tolerance: 1e-6,
        saddle_displacement: 0.2,
        irc_step: 0.08,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };

    let attempt = discover_nd_connection_with_budget(
        &surface,
        &mut network,
        Array1::from_vec(vec![-0.83, 0.12, -0.07, 0.03, 0.09]).view(),
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]).view(),
        &config,
        &PointWitness,
        1,
    );

    assert!(attempt.connection.is_err());
    assert!(attempt.budget_exhausted);
    assert_eq!(attempt.charged_evaluations, 1);
    assert_eq!(surface.calls(), 1);
}

#[test]
fn generic_nd_ride_certifies_both_minima_without_atomistic_metadata() {
    let mut network = NdPesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 300,
        saddle_steps: 500,
        quench_gradient_tolerance: 1e-8,
        saddle_force_tolerance: 1e-6,
        saddle_displacement: 0.2,
        irc_step: 0.08,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };
    let start = Array1::from_vec(vec![-0.83, 0.12, -0.07, 0.03, 0.09]);
    let mode = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);

    let connection = discover_nd_connection(
        &FiveDimensionalDoubleWell,
        &mut network,
        start.view(),
        mode.view(),
        &config,
        &PointWitness,
    )
    .unwrap();

    assert_eq!(network.minimum_count(), 2);
    assert_eq!(network.saddle_count(), 1);
    assert_ne!(connection.endpoints[0], connection.endpoints[1]);
    assert_eq!(connection.saddle_coordinates.len(), 5);
    assert_eq!(connection.negative_modes, 1);
    assert!(connection.curvature < 0.0);
    assert!(connection.saddle_max_gradient < config.saddle_force_tolerance);
    for minimum in network.minima() {
        assert_eq!(minimum.coordinates.len(), 5);
        assert!(minimum.energy < 1e-12);
        assert!((minimum.coordinates[0].abs() - 1.0).abs() < 1e-6);
        assert!(minimum.max_gradient < config.quench_gradient_tolerance);
    }
}

#[test]
fn loose_minimum_mode_handoff_retains_strict_prfo_certification() {
    let mut network = NdPesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 300,
        saddle_steps: 100,
        prfo_steps: 100,
        minimum_mode_force_tolerance: 5e-2,
        quench_gradient_tolerance: 1e-8,
        saddle_force_tolerance: 1e-8,
        saddle_displacement: 0.2,
        irc_step: 0.08,
        refine_with_prfo: true,
        ..PesExplorationConfig::default()
    };

    let connection = discover_nd_connection(
        &FiveDimensionalDoubleWell,
        &mut network,
        Array1::from_vec(vec![-0.83, 0.12, -0.07, 0.03, 0.09]).view(),
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]).view(),
        &config,
        &PointWitness,
    )
    .unwrap();

    assert_eq!(connection.negative_modes, 1);
    assert!(connection.saddle_max_gradient < config.saddle_force_tolerance);
    assert_eq!(network.minimum_count(), 2);
}

#[test]
fn prfo_centering_separates_a_force_converged_shallow_saddle() {
    let mut network = NdPesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 300,
        saddle_steps: 500,
        quench_gradient_tolerance: 1e-10,
        saddle_force_tolerance: 1.2e-2,
        saddle_displacement: 0.25,
        irc_step: 0.05,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };
    let start = Array1::from_vec(vec![-0.83]);
    let mode = Array1::from_vec(vec![1.0]);

    let connection = discover_nd_connection(
        &ShallowDoubleWell,
        &mut network,
        start.view(),
        mode.view(),
        &config,
        &PointWitness,
    )
    .unwrap();

    assert_ne!(connection.endpoints[0], connection.endpoints[1]);
    assert_eq!(network.minimum_count(), 2);
    assert_eq!(network.saddle_count(), 1);
}

#[test]
fn prfo_centering_recovers_connectivity_from_an_offset_saddle() {
    let mut network = NdPesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 300,
        saddle_steps: 500,
        activation_attempts: 4,
        activation_growth: 2.0,
        quench_gradient_tolerance: 1e-10,
        saddle_force_tolerance: 1e-3,
        saddle_displacement: 0.25,
        irc_step: 0.01,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };

    let connection = discover_nd_connection(
        &OffsetShallowDoubleWell,
        &mut network,
        Array1::from_vec(vec![-0.83]).view(),
        Array1::from_vec(vec![1.0]).view(),
        &config,
        &PointWitness,
    )
    .unwrap();

    assert_ne!(connection.endpoints[0], connection.endpoints[1]);
    assert_eq!(network.minimum_count(), 2);
    assert_eq!(network.saddle_count(), 1);
}

#[test]
fn bowl_breakout_reaches_negative_curvature_before_minimum_mode_following() {
    let mut network = NdPesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 300,
        saddle_steps: 500,
        activation_attempts: 8,
        activation_growth: 2.0,
        quench_gradient_tolerance: 1e-8,
        saddle_force_tolerance: 0.1,
        saddle_displacement: 0.01,
        irc_step: 0.08,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };
    let start = Array1::from_vec(vec![-0.83, 0.12, -0.07, 0.03, 0.09]);
    let mode = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);

    let connection = discover_nd_connection(
        &FiveDimensionalDoubleWell,
        &mut network,
        start.view(),
        mode.view(),
        &config,
        &PointWitness,
    )
    .unwrap();

    assert_eq!(connection.negative_modes, 1);
    assert!(connection.curvature < 0.0);
    assert_eq!(network.minimum_count(), 2);
}

#[test]
fn perpendicular_relaxation_finds_a_curved_saddle_channel() {
    let mut network = NdPesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Lanczos,
        quench_steps: 500,
        saddle_steps: 800,
        activation_attempts: 4,
        activation_growth: 2.0,
        activation_relaxation_steps: 8,
        quench_gradient_tolerance: 1e-8,
        saddle_force_tolerance: 1e-6,
        saddle_displacement: 0.1,
        irc_step: 0.08,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };
    let start = Array1::from_vec(vec![-0.83, 0.04, -0.07, 0.03, 0.09]);
    let mode = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);

    let connection = discover_nd_connection(
        &CurvedFiveDimensionalDoubleWell,
        &mut network,
        start.view(),
        mode.view(),
        &config,
        &PointWitness,
    )
    .unwrap();

    assert_eq!(connection.negative_modes, 1);
    assert!(connection.curvature < 0.0);
    assert_eq!(network.minimum_count(), 2);
}
