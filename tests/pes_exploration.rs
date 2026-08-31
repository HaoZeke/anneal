use std::convert::Infallible;

use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesNetwork, PesSurface, RideMethod,
    StructureContext, discover_cartesian_mode_connection, discover_mode_connection,
    stationary_index, stationary_index_cartesian,
};
use ndarray::{Array1, ArrayView1};

struct DoubleWell;

impl PesSurface for DoubleWell {
    type Error = Infallible;

    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
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

struct IndexTwoSaddle;

impl PesSurface for IndexTwoSaddle {
    type Error = Infallible;

    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        let mut energy = -coordinates[0] * coordinates[0] - 2.0 * coordinates[1] * coordinates[1];
        let mut gradient = Array1::zeros(coordinates.len());
        gradient[0] = -2.0 * coordinates[0];
        gradient[1] = -4.0 * coordinates[1];
        for index in 2..coordinates.len() {
            energy += coordinates[index] * coordinates[index];
            gradient[index] = 2.0 * coordinates[index];
        }
        Ok((energy, gradient))
    }
}

struct RadialPairBarrier;

impl PesSurface for RadialPairBarrier {
    type Error = Infallible;

    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
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

struct FrozenContaminatedIndex;

impl PesSurface for FrozenContaminatedIndex {
    type Error = Infallible;

    fn evaluate(&self, coordinates: ArrayView1<f64>) -> Result<(f64, Array1<f64>), Self::Error> {
        let signs = [-1.0, 1.0, 1.0, -2.0, 1.0, 1.0];
        let energy = coordinates
            .iter()
            .zip(signs)
            .map(|(coordinate, sign)| sign * coordinate * coordinate)
            .sum();
        let gradient = Array1::from_iter(
            coordinates
                .iter()
                .zip(signs)
                .map(|(coordinate, sign)| 2.0 * sign * coordinate),
        );
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
    assert_eq!(connection.negative_modes, 1);
    assert!(connection.curvature < 0.0);
    assert!(connection.saddle_energy > 0.9);
    let (_, saddle_gradient) = DoubleWell
        .evaluate(connection.saddle_coordinates.view())
        .unwrap();
    let receiving_force = saddle_gradient
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    assert!((connection.saddle_max_gradient - receiving_force).abs() < 1e-12);
    for minimum in network.minima() {
        assert!(minimum.energy < 1e-10);
        assert!((minimum.coordinates[0].abs() - 1.0).abs() < 1e-5);
        assert!(minimum.max_gradient < config.quench_gradient_tolerance);
    }
}

#[test]
fn constrained_atomistic_connection_certifies_only_free_modes() {
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let mut network = PesNetwork::new();
    let config = PesExplorationConfig {
        ride_method: RideMethod::Dimer,
        quench_steps: 300,
        saddle_steps: 600,
        irc_steps: 100,
        quench_gradient_tolerance: 1e-8,
        saddle_force_tolerance: 1e-5,
        saddle_displacement: 0.1,
        negative_curvature_tolerance: 1e-7,
        hessian_step: 1e-5,
        maximum_move: 0.1,
        irc_step: 0.05,
        refine_with_prfo: false,
        ..PesExplorationConfig::default()
    };
    let start = array![-0.25, 0.0, 0.0, 0.25, 0.0, 0.0];
    let mode = array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let connection = discover_cartesian_mode_connection(
        &RadialPairBarrier,
        &descriptor_space,
        &mut network,
        start.view(),
        array![1.0, 1.0].view(),
        &[false, false],
        mode.view(),
        Some(&[1, 1]),
        &config,
        &CartesianWitness { tolerance: 1e-4 },
    )
    .unwrap();

    assert_eq!(connection.negative_modes, 1);
    assert_eq!(network.minimum_count(), 2);
    assert_eq!(network.saddle_count(), 1);
}

#[test]
fn finite_difference_index_counts_every_unstable_mode() {
    let report = stationary_index(&IndexTwoSaddle, Array1::zeros(6).view(), 1e-4, 1e-7).unwrap();

    assert_eq!(report.negative_modes, 2);
    assert!((report.eigenvalues[0] + 4.0).abs() < 1e-8);
    assert!((report.eigenvalues[1] + 2.0).abs() < 1e-8);
    assert!(report.eigenvalues[2] > 1.9);
}

#[test]
fn cartesian_index_excludes_finite_cluster_rigid_modes() {
    let coordinates = Array1::from_vec(vec![-0.5, 0.0, 0.0, 0.5, 0.0, 0.0]);
    let report = stationary_index_cartesian(
        &RadialPairBarrier,
        coordinates.view(),
        &[false, false],
        [false; 3],
        1e-4,
        1e-7,
    )
    .unwrap();

    assert_eq!(report.negative_modes, 1);
    assert!((report.eigenvalues[0] + 2.0).abs() < 1e-6);
    assert!(report.lowest_mode[0] * report.lowest_mode[3] < 0.0);
}

#[test]
fn cartesian_index_excludes_frozen_surface_coordinates() {
    let report = stationary_index_cartesian(
        &FrozenContaminatedIndex,
        Array1::zeros(6).view(),
        &[true, false],
        [true; 3],
        1e-4,
        1e-7,
    )
    .unwrap();

    assert_eq!(report.negative_modes, 1);
    assert!(
        report
            .lowest_mode
            .iter()
            .take(3)
            .all(|component| component.abs() < 1e-12)
    );
    assert!(report.eigenvalues[0] < -3.9);
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

#[test]
fn exact_admission_preserves_species_and_identity_domain() {
    let geometry = DescriptorGeometry::finite(1.0).unwrap();
    let descriptor_space = universal_descriptor_space(geometry);
    let coordinates = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0]);
    let witness = CartesianWitness { tolerance: 1e-6 };
    let mut network = PesNetwork::new();

    let carbon_oxygen = StructureContext::new(
        Some(vec![6, 8]),
        Some(geometry),
        Some("molecular-test".into()),
    );
    let oxygen_carbon = StructureContext::new(
        Some(vec![8, 6]),
        Some(geometry),
        Some("molecular-test".into()),
    );
    let other_domain = StructureContext::new(
        Some(vec![6, 8]),
        Some(geometry),
        Some("surface-test".into()),
    );

    let first = network
        .admit_minimum_with_context(
            -1.0,
            coordinates.clone(),
            0.0,
            descriptor_space
                .describe(coordinates.view(), carbon_oxygen.species())
                .unwrap(),
            carbon_oxygen.clone(),
            &witness,
        )
        .unwrap();
    let swapped = network
        .admit_minimum_with_context(
            -1.0,
            coordinates.clone(),
            0.0,
            descriptor_space
                .describe(coordinates.view(), oxygen_carbon.species())
                .unwrap(),
            oxygen_carbon,
            &witness,
        )
        .unwrap();
    let different_system = network
        .admit_minimum_with_context(
            -1.0,
            coordinates.clone(),
            0.0,
            descriptor_space
                .describe(coordinates.view(), other_domain.species())
                .unwrap(),
            other_domain,
            &witness,
        )
        .unwrap();
    let duplicate = network
        .admit_minimum_with_context(
            -1.0,
            coordinates.clone(),
            0.0,
            descriptor_space
                .describe(coordinates.view(), carbon_oxygen.species())
                .unwrap(),
            carbon_oxygen,
            &witness,
        )
        .unwrap();

    assert!(first.is_new);
    assert!(swapped.is_new);
    assert!(different_system.is_new);
    assert!(!duplicate.is_new);
    assert_eq!(duplicate.id, first.id);
    assert_eq!(network.minimum_count(), 3);
}
