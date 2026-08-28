use std::collections::BTreeMap;
use std::convert::Infallible;

use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::pes_db::{PesNetworkDatabase, PesProvenance, PesUnits};
use anneal_core::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesNetwork, PesSurface, RideMethod,
    discover_mode_connection,
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

struct CartesianWitness;

impl ExactStructureWitness for CartesianWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt()
            < 1e-5
    }
}

fn network() -> (PesNetwork, anneal_core::descriptor_space::DescriptorSpace) {
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
    discover_mode_connection(
        &DoubleWell,
        &descriptor_space,
        &mut network,
        start.view(),
        Array1::from_vec(vec![1.0, 1.0]).view(),
        mode.view(),
        Some(&[1, 1]),
        &config,
        &CartesianWitness,
    )
    .unwrap();
    (network, descriptor_space)
}

fn provenance() -> PesProvenance {
    PesProvenance {
        campaign: "pes-db-contract".into(),
        replica: 3,
        event_sequence: 17,
        potential: serde_json::json!({"type": "analytic-double-well", "version": 1}),
        exploration: serde_json::json!({"ride": "dimer", "hessian_step": 1e-4}),
        software: BTreeMap::from([
            ("anneal-core".into(), env!("CARGO_PKG_VERSION").into()),
            (
                "rgsaddle".into(),
                "3015f68b12ebac9b7950469efa86d0f1ecfa001d".into(),
            ),
        ]),
        units: PesUnits::new(2.5, 0.0103, 1.0, "sigma", "epsilon", "reduced-mass").unwrap(),
    }
}

#[test]
fn network_descriptors_reproduce_from_exact_records() {
    let (network, descriptor_space) = network();
    for minimum in network.minima() {
        let reproduced = descriptor_space
            .describe(minimum.coordinates.view(), minimum.context.species())
            .unwrap();
        assert_eq!(
            reproduced, minimum.descriptor,
            "minimum {} descriptor is not bound to its retained coordinates",
            minimum.id
        );
    }
    for saddle in network.saddles() {
        let reproduced = descriptor_space
            .describe(
                saddle.saddle_coordinates.view(),
                saddle.context.species(),
            )
            .unwrap();
        assert_eq!(
            reproduced, saddle.descriptor,
            "saddle {} descriptor is not bound to its retained coordinates",
            saddle.id
        );
    }
}

#[test]
fn readcon_db_round_trips_minima_saddles_descriptors_and_provenance() {
    let (network, descriptor_space) = network();
    let directory = tempfile::tempdir().unwrap();
    let database = PesNetworkDatabase::open(directory.path()).unwrap();
    let receipt = database
        .write_snapshot(11, &network, &provenance())
        .unwrap();
    let loaded = database.read_snapshot(11, &descriptor_space).unwrap();

    assert_eq!(receipt.frame_hashes.len(), 3);
    assert_eq!(loaded.frame_hashes, receipt.frame_hashes);
    assert_eq!(loaded.provenance, provenance());
    assert_eq!(loaded.network.minimum_count(), network.minimum_count());
    assert_eq!(loaded.network.saddle_count(), network.saddle_count());

    for (stored, original) in loaded.network.minima().iter().zip(network.minima()) {
        assert_eq!(stored.id, original.id);
        assert_eq!(stored.energy.to_bits(), original.energy.to_bits());
        assert_eq!(stored.coordinates, original.coordinates);
        assert_eq!(stored.context, original.context);
        assert_eq!(stored.descriptor, original.descriptor);
    }
    let stored = &loaded.network.saddles()[0];
    let original = &network.saddles()[0];
    assert_eq!(stored.id, original.id);
    assert_eq!(stored.origin, original.origin);
    assert_eq!(stored.endpoints, original.endpoints);
    assert_eq!(stored.saddle_coordinates, original.saddle_coordinates);
    assert_eq!(stored.context, original.context);
    assert_eq!(stored.descriptor, original.descriptor);
    assert_eq!(stored.negative_modes, 1);
    assert_eq!(stored.curvature.to_bits(), original.curvature.to_bits());
    assert_eq!(stored.lowest_mode, original.lowest_mode);
}

#[test]
fn readcon_db_rejects_a_descriptor_geometry_mismatch() {
    let (network, _) = network();
    let directory = tempfile::tempdir().unwrap();
    let database = PesNetworkDatabase::open(directory.path()).unwrap();
    database.write_snapshot(9, &network, &provenance()).unwrap();
    let incompatible = universal_descriptor_space(DescriptorGeometry::finite(2.0).unwrap());

    let error = database.read_snapshot(9, &incompatible).unwrap_err();
    assert!(error.to_string().contains("descriptor payload"));
}
