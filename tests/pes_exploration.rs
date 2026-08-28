use std::convert::Infallible;

use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesNetwork, PesSurface, RideMethod,
    discover_mode_connection,
};
use ndarray::{Array1, ArrayView1};

struct DoubleWell;

impl PesSurface for DoubleWell {
    type Error = Infallible;

    fn evaluate(
        &self,
        coordinates: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), Self::Error> {
        let reaction = coordinates[0];
        let mut energy = (reaction * reaction - 1.0).powi(2);
        let mut gradient = Array1::zeros(coordinates.len());
        gradient[0] = 4.0 * reaction * (reaction * reaction - 1.0);
        for index in 1..coordinates.len() {
            energy += coordinates[index] * coordinates[index];
            gradient[index] = 2.0 * coordinates[index];
        }
        Ok((energy, gradient))
    }
}

struct CartesianWitness {
    tolerance: f64,
}

impl ExactStructureWitness for CartesianWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right) * (left - right))
            .sum::<f64>()
            .sqrt()
            <= self.tolerance
    }
}

#[test]
fn rgmin_rgsaddle_connection_finds_both_double_well_minima() {
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let mut network = PesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Dimer,
        quench_steps: 300,
        saddle_steps: 500,
        irc_steps: 80,
        quench_gradient_tolerance: 1e-7,
        saddle_force_tolerance: 1e-5,
        saddle_displacement: 0.25,
        refine_with_prfo: true,
        ..PesExplorationConfig::default()
    };
    let mut start = Array1::zeros(6);
    start[0] = -0.82;
    start[1] = 0.06;
    let mut mode = Array1::zeros(6);
    mode[0] = 1.0;

    let connection = discover_mode_connection(
        &DoubleWell,
        &descriptor_space,
        &mut network,
        start.view(),
        Array1::from_vec(vec![1.0, 1.0]).view(),
        mode.view(),
        Some(&[1, 1]),
        &config,
        &CartesianWitness { tolerance: 1e-4 },
    )
    .unwrap();

    assert_eq!(network.minimum_count(), 2);
    assert_eq!(network.saddle_count(), 1);
    assert_ne!(connection.endpoints[0], connection.endpoints[1]);
    assert!(connection.curvature < 0.0);
    assert!(connection.saddle_energy > 0.9);
    for minimum in network.minima() {
        assert!(minimum.energy < 1e-10);
        assert!((minimum.coordinates[0].abs() - 1.0).abs() < 1e-5);
        assert!(minimum.max_gradient < config.quench_gradient_tolerance);
    }
}

#[test]
fn descriptor_distance_orders_checks_but_exact_witness_decides_identity() {
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let mut network = PesNetwork::new();
    let witness = CartesianWitness { tolerance: 1e-6 };
    let first = network
        .admit_minimum(
            -1.0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            0.0,
            descriptor_space
                .describe(Array1::from_vec(vec![0.0, 0.0, 0.0]).view(), Some(&[1]))
                .unwrap(),
            &witness,
        )
        .unwrap();
    let distinct = network
        .admit_minimum(
            -1.0,
            Array1::from_vec(vec![1e-4, 0.0, 0.0]),
            0.0,
            descriptor_space
                .describe(Array1::from_vec(vec![1e-4, 0.0, 0.0]).view(), Some(&[1]))
                .unwrap(),
            &witness,
        )
        .unwrap();
    let duplicate = network
        .admit_minimum(
            -1.0,
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            0.0,
            descriptor_space
                .describe(Array1::from_vec(vec![0.0, 0.0, 0.0]).view(), Some(&[1]))
                .unwrap(),
            &witness,
        )
        .unwrap();

    assert!(first.is_new);
    assert!(distinct.is_new);
    assert_ne!(first.id, distinct.id);
    assert!(!duplicate.is_new);
    assert_eq!(duplicate.id, first.id);
    assert_eq!(distinct.nearest_descriptor_distance, Some(0.0));
}
