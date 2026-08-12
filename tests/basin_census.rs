use anneal_core::catalog::{BasinCensus, BasinId, CensusError};

fn census(radius: f64) -> BasinCensus {
    BasinCensus::new(2, radius).unwrap()
}

#[test]
fn empty_census_has_no_unseen_mass_or_saturation() {
    let census = census(0.25);

    assert!(census.is_empty());
    assert_eq!(census.len(), 0);
    assert_eq!(census.total_visits(), 0);
    assert_eq!(census.singleton_count(), 0);
    assert_eq!(census.unseen_mass(), None);
    assert!(!census.is_saturated());
}

#[test]
fn observations_use_immutable_medoids_and_exact_counts() {
    let mut census = census(0.25);

    let first = census.observe(&[0.0, 0.0]).unwrap();
    let revisit = census.observe(&[0.1, 0.0]).unwrap();
    let second = census.observe(&[1.0, 0.0]).unwrap();
    let second_revisit = census.observe(&[0.9, 0.0]).unwrap();

    assert_eq!(first.basin_id, BasinId::from_raw(0));
    assert!(first.created);
    assert_eq!(revisit.basin_id, first.basin_id);
    assert!(!revisit.created);
    assert_eq!(second.basin_id, BasinId::from_raw(1));
    assert_eq!(second_revisit.basin_id, second.basin_id);
    assert_eq!(census.entry(first.basin_id).unwrap().medoid(), &[0.0, 0.0]);
    assert_eq!(census.entry(first.basin_id).unwrap().visits(), 2);
    assert_eq!(census.entry(second.basin_id).unwrap().medoid(), &[1.0, 0.0]);
    assert_eq!(census.entry(second.basin_id).unwrap().visits(), 2);
    assert_eq!(census.total_visits(), 4);
    assert_eq!(
        census
            .entries()
            .iter()
            .map(|entry| entry.visits())
            .sum::<u64>(),
        census.total_visits()
    );
}

#[test]
fn nearest_assignment_breaks_exact_ties_by_basin_id() {
    let mut census = census(1.0);
    let lower = census.observe(&[0.0, 0.0]).unwrap().basin_id;
    let upper = census.observe(&[1.5, 0.0]).unwrap().basin_id;

    let tie = census.observe(&[0.75, 0.0]).unwrap();

    assert_eq!(lower, BasinId::from_raw(0));
    assert_eq!(upper, BasinId::from_raw(1));
    assert_eq!(tie.basin_id, lower);
    assert_eq!(census.entry(lower).unwrap().visits(), 2);
    assert_eq!(census.entry(upper).unwrap().visits(), 1);
}

#[test]
fn invalid_observations_do_not_change_the_census() {
    let mut census = census(0.25);
    census.observe(&[0.0, 0.0]).unwrap();

    assert_eq!(
        census.observe(&[1.0]).unwrap_err(),
        CensusError::DescriptorDimension {
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        census.observe(&[f64::NAN, 0.0]).unwrap_err(),
        CensusError::NonFiniteDescriptor { index: 0 }
    );
    assert_eq!(census.len(), 1);
    assert_eq!(census.total_visits(), 1);
}

#[test]
fn saturation_uses_only_exact_visit_and_singleton_counts() {
    let mut census = census(0.1);
    for _ in 0..16 {
        census.observe(&[0.0, 0.0]).unwrap();
    }
    for descriptor in [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]] {
        census.observe(&descriptor).unwrap();
    }

    assert_eq!(census.total_visits(), 20);
    assert_eq!(census.singleton_count(), 4);
    assert_eq!(census.unseen_mass(), Some(0.2));
    assert!(!census.is_saturated());

    census.observe(&[1.0, 0.0]).unwrap();
    assert_eq!(census.total_visits(), 21);
    assert_eq!(census.singleton_count(), 3);
    assert!(census.is_saturated());
}

#[test]
fn fewer_than_twenty_visits_never_report_saturation() {
    let mut census = census(0.25);
    for _ in 0..19 {
        census.observe(&[0.0, 0.0]).unwrap();
    }

    assert_eq!(census.unseen_mass(), Some(0.0));
    assert!(!census.is_saturated());
}
