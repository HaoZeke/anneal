#![cfg(feature = "bank-rpc")]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use anneal_core::catalog::{
    DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
};
use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
use anneal_core::catalog_rpc::{
    CatalogCandidate, CatalogIdentity, CatalogRideOutcome, CatalogRideWork,
};
use anneal_core::cooperative_search::{
    CatalogOfferOutcome, CooperativeRun, RideClaimOutcome, RideReportOutcome,
};
use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::pes_exploration::{
    ExactStructureWitness, PesExplorationConfig, PesSurface, RideMethod, RideModeDirection,
    localized_cartesian_mode,
};
use anneal_core::ride_execution::{
    CatalogRideExecutionConfig, connected_destination, execute_catalog_ride,
};
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

struct CollapsedWitness;

impl ExactStructureWitness for CollapsedWitness {
    fn equivalent(&self, _left: ArrayView1<f64>, _right: ArrayView1<f64>) -> bool {
        true
    }
}

fn exploration_config() -> PesExplorationConfig {
    PesExplorationConfig {
        ride_method: RideMethod::Dimer,
        quench_steps: 300,
        saddle_steps: 600,
        minimum_mode_force_tolerance: 1e-8,
        irc_steps: 200,
        prfo_steps: 100,
        activation_attempts: 4,
        activation_growth: 2.0,
        activation_relaxation_steps: 3,
        quench_gradient_tolerance: 1e-8,
        quench_gradient_norm_tolerance: None,
        saddle_force_tolerance: 1e-8,
        saddle_displacement: 0.12,
        negative_curvature_tolerance: 1e-6,
        hessian_step: 1e-5,
        maximum_move: 0.1,
        irc_step: 0.05,
        branch_attempts: 4,
        branch_growth: 2.0,
        irc_force_tolerance: 0.05,
        certify_degenerate_rearrangements: false,
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
        avoid_saddles: Vec::new(),
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

fn opposite(direction: RideDirection) -> RideDirection {
    match direction {
        RideDirection::Negative => RideDirection::Positive,
        RideDirection::Positive => RideDirection::Negative,
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

fn live_signature() -> SystemSignature {
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    SystemSignature {
        atomic_numbers: vec![18, 18],
        coordinate_dim: 6,
        group_labels: vec![0, 1],
        group_schema: "independent-sites-v1".into(),
        frozen_mask: vec![false, false],
        cell: None,
        periodic: [false; 3],
        length_scale: 1.0,
        energy_scale: 1.0,
        engine: EngineSignature {
            kind: "radial-double-well".into(),
            config_digest: [0x52; 32],
            external_inputs: BTreeMap::new(),
        },
        descriptor: DescriptorSignature {
            schema: descriptor_space.schema().name().into(),
            version: descriptor_space.schema().version(),
            hyperparameters: BTreeMap::new(),
            species_channels: vec![18],
        },
        validation_schema_version: 1,
    }
}

fn live_source(replica: u32) -> CatalogCandidate {
    let mut source = work(RideDirection::Negative).source;
    source.producer_replica = replica;
    source.census_basin = None;
    source
}

fn live_server_with_limits(max_gradient_norm: f64, census_radius: f64) -> CatalogServer {
    let signature = live_signature();
    let digest = signature.digest();
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let descriptor_dim = descriptor_space
        .describe(
            ArrayView1::from(&live_source(7).coordinates),
            Some(&[18, 18]),
        )
        .unwrap()
        .values()
        .len();
    let config = ServerConfig::new("ride-live", "radial-double-well", digest, [7])
        .unwrap()
        .with_scientific_state(
            signature,
            descriptor_space,
            ValidatorConfig {
                reference_coordinates: live_source(7).coordinates,
                descriptor_dim,
                min_separation: 0.8,
                coordinate_tolerance: 1e-10,
                max_gradient_norm,
                energy_abs_tolerance: 1e-10,
                energy_rel_tolerance: 1e-10,
            },
            4,
            census_radius,
            20_000,
            |coordinates| {
                let surface = RadialDoubleWell::new();
                let (energy, gradient) = surface.evaluate(ArrayView1::from(coordinates))?;
                Ok(FreshEvaluation {
                    energy,
                    forces: gradient.iter().map(|component| -*component).collect(),
                })
            },
        )
        .unwrap()
        .with_exact_structure_witness(SeparationWitness)
        .unwrap();
    CatalogServer::start("127.0.0.1:0", config).unwrap()
}

fn live_run_with_gradient_tolerance(max_gradient_norm: f64) -> (CatalogServer, CooperativeRun) {
    live_run_with_limits(max_gradient_norm, 0.05)
}

fn live_run_with_limits(
    max_gradient_norm: f64,
    census_radius: f64,
) -> (CatalogServer, CooperativeRun) {
    let server = live_server_with_limits(max_gradient_norm, census_radius);
    let signature = live_signature();
    let mut run = CooperativeRun::new([7], 20_000).unwrap();
    run.attach_client(
        7,
        CatalogClient::connect(
            server.addr(),
            CatalogIdentity {
                campaign: "ride-live".into(),
                ensemble: "radial-double-well".into(),
                replica: 7,
                signature_digest: signature.digest(),
            },
            ClientConfig::default(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        run.offer_candidate(7, live_source(7)).unwrap(),
        CatalogOfferOutcome::Admitted
    );
    (server, run)
}

fn live_run() -> (CatalogServer, CooperativeRun) {
    live_run_with_gradient_tolerance(1e-7)
}

fn first_claim_seed_toward_saddle() -> u64 {
    for seed in 0..10_000 {
        let points_toward_saddle = [0, 1].iter().all(|&representative_atom| {
            let mode = localized_cartesian_mode(
                array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0].view(),
                representative_atom,
                &[false; 2],
                DescriptorGeometry::finite(1.0).unwrap(),
                1.0,
                seed,
                0,
                RideModeDirection::Negative,
            )
            .unwrap();
            mode[3] - mode[0] > 0.0
        });
        if points_toward_saddle {
            return seed;
        }
    }
    panic!("no deterministic first-arm seed points toward the analytic saddle")
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
fn collapsed_connection_preserves_its_certified_saddle_for_other_chains() {
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
        &CollapsedWitness,
    );

    let CatalogRideOutcome::Unresolved(evidence) = report.outcome else {
        panic!("a collapsed connection must retain its stationary saddle")
    };
    assert_eq!(evidence.failure, RideFailure::CollapsedConnection);
    assert!(evidence.saddle.gradient_norm < 1e-7);
    assert!((evidence.saddle.energy - 1.0).abs() < 1e-7);
    assert_eq!(report.charged_evaluations, surface.calls());
}

#[test]
fn symmetry_reduced_connection_is_a_certified_degenerate_rearrangement() {
    let surface = RadialDoubleWell::new();
    let work = work(direction_toward_saddle());
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let mut config = execution_config(5_000);
    config.exploration.certify_degenerate_rearrangements = true;
    let report = execute_catalog_ride(
        &surface,
        &descriptor_space,
        &work,
        &[18, 18],
        array![1.0, 1.0].view(),
        &[false; 2],
        &config,
        &CollapsedWitness,
    );

    let CatalogRideOutcome::Certified(connection) = &report.outcome else {
        panic!("a symmetry-reduced index-one path must remain certified KTN evidence")
    };
    assert!(CollapsedWitness.equivalent(
        ArrayView1::from(connection.endpoints[0].coordinates.as_slice()),
        ArrayView1::from(connection.endpoints[1].coordinates.as_slice())
    ));
    assert!(connection.saddle.gradient_norm < 1e-7);
    assert!(connected_destination(&work, &report, &CollapsedWitness).is_none());
    assert_eq!(report.charged_evaluations, surface.calls());
}

#[test]
fn coordinator_returns_a_collapsed_saddle_as_same_pes_avoidance_evidence() {
    let (_server, mut run) = live_run();
    let RideClaimOutcome::Work(work) = run.claim_ride(7, first_claim_seed_toward_saddle()).unwrap()
    else {
        panic!("the admitted source did not produce transition-search work")
    };
    let surface = RadialDoubleWell::new();
    let report = execute_catalog_ride(
        &surface,
        &universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap()),
        &work,
        &[18, 18],
        array![1.0, 1.0].view(),
        &[false; 2],
        &execution_config(5_000),
        &CollapsedWitness,
    );
    let CatalogRideOutcome::Unresolved(evidence) = &report.outcome else {
        panic!("the producer did not retain its collapsed saddle")
    };
    let saddle_coordinates = evidence.saddle.coordinates.clone();

    let RideReportOutcome::Credited(credit) = run.report_ride(7, report).unwrap() else {
        panic!("the coordinator rejected index-one unresolved evidence")
    };
    assert!(!credit.certified_connection);
    assert_eq!(credit.failure, Some(RideFailure::CollapsedConnection));
    let RideClaimOutcome::Work(next) = run.claim_ride(7, 0x5eed).unwrap() else {
        panic!("the unresolved saddle did not release the live claim")
    };
    assert_eq!(next.avoid_saddles.len(), 1);
    assert_eq!(next.avoid_saddles[0].coordinates, saddle_coordinates);
}

#[test]
fn convex_ray_failure_is_distinct_from_stationary_index_failure() {
    let surface = RadialDoubleWell::new();
    let work = work(opposite(direction_toward_saddle()));
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let mut config = execution_config(5_000);
    config.exploration.activation_attempts = 1;
    config.exploration.saddle_displacement = 0.01;

    let report = execute_catalog_ride(
        &surface,
        &descriptor_space,
        &work,
        &[18, 18],
        array![1.0, 1.0].view(),
        &[false; 2],
        &config,
        &SeparationWitness,
    );

    assert_eq!(
        report.outcome,
        CatalogRideOutcome::Failed(RideFailure::ActivationNotEscaped)
    );
}

#[test]
fn live_claim_executes_reports_and_returns_the_connected_minimum() {
    let (_server, mut run) = live_run();

    let RideClaimOutcome::Work(work) = run.claim_ride(7, first_claim_seed_toward_saddle()).unwrap()
    else {
        panic!("the admitted source did not produce transition-search work")
    };
    assert_eq!(work.order.arm.mode_rank, 0);
    assert_eq!(work.order.arm.direction, RideDirection::Negative);
    assert_eq!(work.order.arm.method, RideMethod::Dimer);
    let surface = RadialDoubleWell::new();
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
    let producer_calls = report.charged_evaluations;
    let destination = connected_destination(&work, &report, &SeparationWitness)
        .expect("certified connection must expose its non-source endpoint");
    let CatalogRideOutcome::Certified(connection) = &report.outcome else {
        unreachable!("a connected destination requires a certified report")
    };
    let saddle_gradient_norm = connection.saddle.gradient_norm;
    let endpoint_gradient_norms = connection
        .endpoints
        .each_ref()
        .map(|endpoint| endpoint.gradient_norm);
    let report_outcome = run.report_ride(7, report).unwrap();
    let RideReportOutcome::Credited(credit) = report_outcome else {
        panic!(
            "receiving-side index certification returned {report_outcome:?}; producer calls \
             {producer_calls}, saddle gradient norm {saddle_gradient_norm}, endpoint gradient norms \
             {endpoint_gradient_norms:?}"
        )
    };

    assert!(credit.certified_connection);
    assert_eq!(credit.failure, None);
    assert!(credit.novel_saddle);
    assert!(credit.novel_edge);
    assert_eq!(credit.total_charged_evaluations, producer_calls + 15);
    assert!((destination.coordinates[3] - destination.coordinates[0] - 2.0).abs() < 1e-4);
}

#[test]
fn exact_endpoint_identity_does_not_depend_on_descriptor_radius() {
    let (_server, mut run) = live_run_with_limits(1e-7, 0.0);
    let RideClaimOutcome::Work(work) = run.claim_ride(7, first_claim_seed_toward_saddle()).unwrap()
    else {
        panic!("the admitted source did not produce transition-search work")
    };
    let surface = RadialDoubleWell::new();
    let report = execute_catalog_ride(
        &surface,
        &universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap()),
        &work,
        &[18, 18],
        array![1.0, 1.0].view(),
        &[false; 2],
        &execution_config(5_000),
        &SeparationWitness,
    );

    let RideReportOutcome::Credited(credit) = run.report_ride(7, report).unwrap() else {
        panic!("an exact source witness must survive a zero descriptor radius")
    };
    assert!(credit.certified_connection);
    assert!(credit.novel_saddle);
    assert!(credit.novel_edge);
}

#[test]
fn receiver_excludes_rigid_curvature_from_saddle_index() {
    let (_server, mut run) = live_run_with_gradient_tolerance(1e-5);
    let RideClaimOutcome::Work(work) = run.claim_ride(7, first_claim_seed_toward_saddle()).unwrap()
    else {
        panic!("the admitted source did not produce transition-search work")
    };
    let surface = RadialDoubleWell::new();
    let descriptor_space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let mut report = execute_catalog_ride(
        &surface,
        &descriptor_space,
        &work,
        &[18, 18],
        array![1.0, 1.0].view(),
        &[false; 2],
        &execution_config(5_000),
        &SeparationWitness,
    );
    let CatalogRideOutcome::Certified(connection) = &mut report.outcome else {
        panic!("the analytic ride must reach producer certification")
    };
    connection.saddle.coordinates[3] += 1e-7;
    let (energy, gradient) = surface
        .evaluate(ArrayView1::from(&connection.saddle.coordinates))
        .unwrap();
    connection.saddle.energy = energy;
    connection.saddle.forces = gradient.iter().map(|component| -*component).collect();
    connection.saddle.gradient_norm = gradient.dot(&gradient).sqrt();
    let descriptor = descriptor_space
        .describe(
            ArrayView1::from(&connection.saddle.coordinates),
            Some(&[18, 18]),
        )
        .unwrap();
    connection.saddle.descriptor = descriptor.values().to_vec();

    let RideReportOutcome::Credited(credit) = run.report_ride(7, report).unwrap() else {
        panic!("a near-stationary radial barrier must produce receiver credit")
    };
    assert!(credit.certified_connection);
    assert_eq!(credit.failure, None);
}

#[test]
fn receiving_disagreement_is_charged_and_releases_the_live_claim() {
    let (_server, mut run) = live_run();
    let RideClaimOutcome::Work(work) = run.claim_ride(7, first_claim_seed_toward_saddle()).unwrap()
    else {
        panic!("the admitted source did not produce transition-search work")
    };
    let surface = RadialDoubleWell::new();
    let mut report = execute_catalog_ride(
        &surface,
        &universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap()),
        &work,
        &[18, 18],
        array![1.0, 1.0].view(),
        &[false; 2],
        &execution_config(5_000),
        &SeparationWitness,
    );
    let CatalogRideOutcome::Certified(connection) = &mut report.outcome else {
        panic!("the analytic ride must reach producer certification")
    };
    connection.saddle.energy += 1e-4;
    let producer_calls = report.charged_evaluations;

    let RideReportOutcome::Credited(credit) = run.report_ride(7, report).unwrap() else {
        panic!("fresh energy disagreement must become shared negative evidence")
    };
    assert!(!credit.certified_connection);
    assert_eq!(credit.failure, Some(RideFailure::Surface));
    assert!(!credit.novel_saddle);
    assert!(!credit.novel_edge);
    assert_eq!(credit.total_charged_evaluations, producer_calls + 1);
    let RideClaimOutcome::Work(next) = run.claim_ride(7, 0x5eed).unwrap() else {
        panic!("negative evidence did not release the replica's live claim")
    };
    assert_ne!(next.order.id, work.order.id);
    assert_ne!(next.order.arm, work.order.arm);
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
