use anneal_core::discovery_roster::{
    DiscoveryOpportunity, DiscoveryRole, assign_discovery_roles,
    assign_discovery_roles_with_minimum,
};

#[test]
fn roster_maximizes_total_minimum_information_under_ride_capacity() {
    let assignments = assign_discovery_roles(
        &[
            DiscoveryOpportunity::new(7, 0.8, Some(0.9)).unwrap(),
            DiscoveryOpportunity::new(2, 0.4, Some(1.1)).unwrap(),
            DiscoveryOpportunity::new(9, 0.7, Some(0.6)).unwrap(),
            DiscoveryOpportunity::new(4, 0.2, Some(0.5)).unwrap(),
        ],
        2,
        11,
    )
    .unwrap();

    let rides = assignments
        .iter()
        .filter(|assignment| assignment.role == DiscoveryRole::SaddleRide)
        .map(|assignment| assignment.replica)
        .collect::<Vec<_>>();
    assert_eq!(rides, vec![2, 4]);
}

#[test]
fn unavailable_rides_leave_every_replica_on_basin_search() {
    let assignments = assign_discovery_roles(
        &[
            DiscoveryOpportunity::new(7, 0.8, None).unwrap(),
            DiscoveryOpportunity::new(2, 0.4, None).unwrap(),
        ],
        4,
        11,
    )
    .unwrap();

    assert!(
        assignments
            .iter()
            .all(|assignment| assignment.role == DiscoveryRole::BasinEscape)
    );
}

#[test]
fn discovery_roles_stay_stable_when_only_the_evidence_epoch_changes() {
    let opportunities = [
        DiscoveryOpportunity::new(7, 0.8, Some(0.9)).unwrap(),
        DiscoveryOpportunity::new(2, 0.4, Some(1.1)).unwrap(),
        DiscoveryOpportunity::new(9, 0.7, Some(0.6)).unwrap(),
        DiscoveryOpportunity::new(4, 0.2, Some(0.5)).unwrap(),
    ];
    let first = assign_discovery_roles(&opportunities, 2, 11).unwrap();
    let second = assign_discovery_roles(&opportunities, 2, 12).unwrap();

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
    assert_eq!(roles(&first), roles(&second));
}

#[test]
fn exact_information_ties_split_without_changing_the_maximum() {
    let assignments = assign_discovery_roles(
        &[
            DiscoveryOpportunity::new(0, 1.0, Some(1.0)).unwrap(),
            DiscoveryOpportunity::new(1, 1.0, Some(1.0)).unwrap(),
            DiscoveryOpportunity::new(2, 1.0, Some(1.0)).unwrap(),
            DiscoveryOpportunity::new(3, 1.0, Some(1.0)).unwrap(),
        ],
        2,
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

#[test]
fn available_ride_arms_form_an_exact_consumer_capacity() {
    let assignments = assign_discovery_roles_with_minimum(
        &[
            DiscoveryOpportunity::new(0, 1.0, Some(0.2)).unwrap(),
            DiscoveryOpportunity::new(1, 0.9, Some(0.3)).unwrap(),
            DiscoveryOpportunity::new(2, 0.8, Some(0.1)).unwrap(),
            DiscoveryOpportunity::new(3, 0.7, Some(0.1)).unwrap(),
        ],
        2,
        2,
        17,
    )
    .unwrap();

    let rides = assignments
        .iter()
        .filter(|assignment| assignment.role == DiscoveryRole::SaddleRide)
        .map(|assignment| assignment.replica)
        .collect::<Vec<_>>();
    assert_eq!(rides, vec![1, 3]);
}

#[test]
fn invalid_information_rates_are_rejected() {
    assert!(DiscoveryOpportunity::new(0, f64::NAN, None).is_err());
    assert!(DiscoveryOpportunity::new(0, 1.0, Some(-1.0)).is_err());
}
