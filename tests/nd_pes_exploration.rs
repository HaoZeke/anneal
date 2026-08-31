use std::convert::Infallible;

use anneal_core::pes_exploration::{
    ExactStructureWitness, NdPesNetwork, PesExplorationConfig, PesSurface, RideMethod,
    discover_nd_connection,
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
