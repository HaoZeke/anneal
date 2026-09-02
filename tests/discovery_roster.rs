use anneal_core::discovery_roster::{
    DiscoveryCoverage, DiscoveryRole, assign_discovery_roles, coverage_allocation_weight,
    good_ucb_missing_mass_index,
};

#[test]
fn coverage_allocation_keeps_full_uncertainty_before_the_observation_floor() {
    assert_eq!(coverage_allocation_weight(4, 20, Some(0.0)), 1.0);
    assert_eq!(coverage_allocation_weight(20, 20, Some(0.0)), 0.0);
    assert_eq!(coverage_allocation_weight(20, 20, None), 1.0);
}

#[test]
fn unresolved_saddle_coverage_receives_the_discovery_batch() {
    let assignments = assign_discovery_roles(
        &[7, 2, 9, 4],
        DiscoveryCoverage {
            basin_observations: 10,
            basin_singletons: 0,
            saddle_observations: 10,
            saddle_singletons: 8,
            ride_available: true,
        },
        11,
    )
    .unwrap();

    assert_eq!(
        assignments
            .iter()
            .filter(|assignment| assignment.role == DiscoveryRole::BasinEscape)
            .count(),
        0
    );
    assert_eq!(
        assignments
            .iter()
            .filter(|assignment| assignment.role == DiscoveryRole::SaddleRide)
            .count(),
        4
    );
    assert!(assignments.iter().all(|assignment| assignment.epoch == 11));
}

#[test]
fn discovery_roles_rotate_without_changing_the_coverage_allocation() {
    let coverage = DiscoveryCoverage {
        basin_observations: 10,
        basin_singletons: 5,
        saddle_observations: 10,
        saddle_singletons: 5,
        ride_available: true,
    };
    let first = assign_discovery_roles(&[7, 2, 9, 4], coverage, 11).unwrap();
    let second = assign_discovery_roles(&[7, 2, 9, 4], coverage, 12).unwrap();

    let roles = |assignments: &[anneal_core::discovery_roster::DiscoveryAssignment]| {
        assignments
            .iter()
            .map(|assignment| (assignment.replica, assignment.role))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        first
            .iter()
            .filter(|assignment| assignment.role == DiscoveryRole::BasinEscape)
            .count(),
        second
            .iter()
            .filter(|assignment| assignment.role == DiscoveryRole::BasinEscape)
            .count()
    );
    assert_ne!(roles(&first), roles(&second));
}

#[test]
fn absent_ride_work_assigns_every_replica_to_basin_discovery() {
    let assignments = assign_discovery_roles(
        &[0, 1, 2],
        DiscoveryCoverage {
            basin_observations: 100,
            basin_singletons: 1,
            saddle_observations: 0,
            saddle_singletons: 0,
            ride_available: false,
        },
        3,
    )
    .unwrap();

    assert!(
        assignments
            .iter()
            .all(|assignment| assignment.role == DiscoveryRole::BasinEscape)
    );
}

#[test]
fn good_ucb_index_is_the_good_turing_estimate_plus_the_distribution_free_bonus() {
    let actual = good_ucb_missing_mass_index(3, 10, 40);
    let expected = 0.3 + (1.0 + 2.0_f64.sqrt()) * (160.0_f64.ln() / 10.0).sqrt();

    assert!((actual - expected).abs() < 1e-12, "actual={actual}, expected={expected}");
    assert!(good_ucb_missing_mass_index(0, 0, 40).is_infinite());
}

#[test]
fn unobserved_mechanisms_split_a_parallel_discovery_batch() {
    let assignments = assign_discovery_roles(
        &[0, 1, 2, 3],
        DiscoveryCoverage {
            basin_observations: 0,
            basin_singletons: 0,
            saddle_observations: 0,
            saddle_singletons: 0,
            ride_available: true,
        },
        17,
    )
    .unwrap();

    assert_eq!(
        assignments
            .iter()
            .filter(|assignment| assignment.role == DiscoveryRole::BasinEscape)
            .count(),
        2
    );
    assert_eq!(
        assignments
            .iter()
            .filter(|assignment| assignment.role == DiscoveryRole::SaddleRide)
            .count(),
        2
    );
}
