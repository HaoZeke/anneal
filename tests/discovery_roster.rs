use anneal_core::discovery_roster::{
    DiscoveryCoverage, DiscoveryEffort, DiscoveryRole, assign_discovery_roles,
    coverage_allocation_weight,
};

fn effort(observations: u64, charged_calls: u64) -> DiscoveryEffort {
    DiscoveryEffort {
        observations,
        charged_calls,
    }
}

#[test]
fn coverage_allocation_keeps_full_uncertainty_before_the_observation_floor() {
    assert_eq!(coverage_allocation_weight(4, 20, Some(0.0)), 1.0);
    assert_eq!(coverage_allocation_weight(20, 20, Some(0.0)), 0.0);
    assert_eq!(coverage_allocation_weight(20, 20, None), 1.0);
}

#[test]
fn unresolved_saddle_coverage_receives_most_replica_seats() {
    let assignments = assign_discovery_roles(
        &[7, 2, 9, 4],
        DiscoveryCoverage {
            basin_unseen_mass_upper: 0.02,
            saddle_unseen_mass_upper: 0.80,
            basin_effort: effort(10, 2_000),
            saddle_effort: effort(10, 2_000),
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
        1
    );
    assert_eq!(
        assignments
            .iter()
            .filter(|assignment| assignment.role == DiscoveryRole::SaddleRide)
            .count(),
        3
    );
    assert!(assignments.iter().all(|assignment| assignment.epoch == 11));
}

#[test]
fn discovery_roles_rotate_without_changing_the_coverage_allocation() {
    let coverage = DiscoveryCoverage {
        basin_unseen_mass_upper: 0.4,
        saddle_unseen_mass_upper: 0.6,
        basin_effort: effort(10, 2_000),
        saddle_effort: effort(10, 2_000),
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
            basin_unseen_mass_upper: 0.01,
            saddle_unseen_mass_upper: 1.0,
            basin_effort: effort(10, 2_000),
            saddle_effort: effort(10, 2_000),
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
fn equal_missing_mass_prefers_the_cheaper_discovery_mechanism_per_pes_call() {
    let assignments = assign_discovery_roles(
        &[0, 1, 2, 3],
        DiscoveryCoverage {
            basin_unseen_mass_upper: 1.0,
            saddle_unseen_mass_upper: 1.0,
            basin_effort: effort(10, 1_000),
            saddle_effort: effort(10, 9_000),
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
        3
    );
    assert_eq!(
        assignments
            .iter()
            .filter(|assignment| assignment.role == DiscoveryRole::SaddleRide)
            .count(),
        1
    );
}
