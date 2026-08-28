#![cfg(feature = "bank-rpc")]

use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::catalog_rpc::{CatalogCandidate, CatalogRideOutcome, CatalogRideWork};
use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesSurface, RideMethod, RideModeDirection,
    localized_cartesian_mode,
};
use anneal_core::ride_execution::{CatalogRideExecutionConfig, execute_catalog_ride};
use anneal_core::ride_ledger::{RideArm, RideDirection, RideFailure, RideWorkOrder};
use ndarray::{Array1, ArrayView1, array};

struct RadialDoubleWell {
    calls: AtomicU64,
}

impl RadialDoubleWell {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl PesSurface for RadialDoubleWell {
    type Error = &'static str;

    fn evaluate(
        &self,
        coordinates: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if coordinates.len() != 6 {
            return Err("six coordinates required");
        }
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
        let reaction = (distance - 1.6) / 0.4;
        let energy = (reaction * reaction - 1.0).powi(2);
        let radial_gradient = 4.0 * reaction * (reaction * reaction - 1.0) / 0.4;
        let mut gradient = Array1::zeros(6);
        for axis in 0..3 {
            let component = radial_gradient * displacement[axis] / distance;
            gradient[axis] = -component;
            gradient[3 + axis] = component;
        }
        Ok((energy, gradient))
    }
}

struct SeparationWitness;

impl ExactStructureWitness for SeparationWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        let separation = |point: ArrayView1<'_, f64>| {
            (0..3)
                .map(|axis| (point[3 + axis] - point[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        (separation(left) - separation(right)).abs() < 1e-5
    }
}

fn exploration_config() -> PesExplorationConfig {
    PesExplorationConfig {
        ride_method: RideMethod::Dimer,
        quench_steps: 300,
        saddle_steps: 600,
        irc_steps: 200,
        prfo_steps: 100,
        quench_gradient_tolerance: 1e-8,
        saddle_force_tolerance: 1e-7,
        saddle_displacement: 0.12,
        negative_curvature_tolerance: 1e-6,
        hessian_step: 1e-5,
        maximum_move: 0.1,
        irc_step: 0.05,
        irc_force_tolerance: 0.05,
        refine_with_prfo: false,
    }
}

fn work(direction: RideDirection) -> CatalogRideWork {
    let coordinates = vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let descriptor = descriptor_space
        .describe(ArrayView1::from(&coordinates), Some(&[18, 18]))
        .unwrap();
    CatalogRideWork {
        order: RideWorkOrder {
            id: 41,
            replica: 7,
            arm: RideArm {
                source_basin: 3,
                environment_class: 0,
                mode_rank: 2,
                direction,
                method: RideMethod::Dimer,
            },
            representative_atom: 1,
            attempt: 1,
            seed: 0xdecaf,
        },
        source: CatalogCandidate {
            producer_replica: 2,
            coordinates,
            cell: None,
            energy: 0.0,
            forces: vec![0.0; 6],
            gradient_norm: 0.0,
            descriptor: descriptor.values().to_vec(),
            descriptor_schema_version: descriptor.schema_version(),
            quench_converged: true,
            charged_work: 81,
            event_sequence: 19,
            seed: 5,
            census_basin: Some(3),
        },
    }
}

fn direction_toward_saddle() -> RideDirection {
    let coordinates = array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
    let mode = localized_cartesian_mode(
        coordinates.view(),
        1,
        &[false; 2],
        DescriptorGeometry::finite(1.0).unwrap(),
        1.0,
        0xdecaf,
        2,
        RideModeDirection::Positive,
    )
    .unwrap();
    if mode[3] - mode[0] > 0.0 {
        RideDirection::Positive
    } else {
        RideDirection::Negative
    }
}

fn execution_config(maximum_evaluations: u64) -> CatalogRideExecutionConfig {
    CatalogRideExecutionConfig {
        exploration: exploration_config(),
        localization_radius: 1.0,
        maximum_evaluations,
        producer_event_sequence: 700,
        producer_charged_work: 200,
    }
}

#[test]
fn claimed_ride_executes_a_counted_minimum_saddle_minimum_connection() {
    let surface = RadialDoubleWell::new();
    let work = work(direction_toward_saddle());
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let report = execute_catalog_ride(
        &surface,
        &descriptor_space,
        &work,
        &[18, 18],
        array![1.0, 1.0].view(),
        &[false; 2],
        &execution_config(5_000),
        &SeparationWitness,
    );

    assert_eq!(report.work, work.order.id);
    assert_eq!(report.charged_evaluations, surface.calls());
    assert!(report.charged_evaluations > 0);
    let CatalogRideOutcome::Certified(connection) = report.outcome else {
        panic!("the analytic double well must yield a certified connection")
    };
    assert_eq!(connection.saddle.producer_replica, work.order.replica);
    assert!(
        connection
            .endpoints
            .iter()
            .all(|endpoint| endpoint.census_basin.is_none())
    );
    let mut separations = connection
        .endpoints
        .iter()
        .map(|endpoint| endpoint.coordinates[3] - endpoint.coordinates[0])
        .collect::<Vec<_>>();
    separations.sort_by(f64::total_cmp);
    assert!((separations[0] - 1.2).abs() < 1e-4);
    assert!((separations[1] - 2.0).abs() < 1e-4);
}

#[test]
fn ride_budget_is_a_hard_pes_call_boundary_and_shared_failure() {
    let surface = RadialDoubleWell::new();
    let work = work(direction_toward_saddle());
    let report = execute_catalog_ride(
        &surface,
        &universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap()),
        &work,
        &[18, 18],
        array![1.0, 1.0].view(),
        &[false; 2],
        &execution_config(1),
        &SeparationWitness,
    );

    assert_eq!(report.charged_evaluations, 1);
    assert_eq!(surface.calls(), 1);
    assert_eq!(
        report.outcome,
        CatalogRideOutcome::Failed(RideFailure::BudgetExhausted)
    );
}
