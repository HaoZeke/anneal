use anneal_core::catalog::{BasinCensus, BasinId, CensusError, WitnessId};
use proptest::prelude::*;
use std::collections::BTreeMap;

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
    assert_eq!(census.doubleton_count(), 2);
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
fn a_witness_merges_near_duplicates_the_radius_splits() {
    let mut split = BasinCensus::new(1, 0.1).unwrap();
    for index in 0..20 {
        split.observe(&[f64::from(index) * 0.5]).unwrap();
    }
    assert_eq!(split.len(), 20);
    assert_eq!(split.singleton_count(), 20);
    assert!(!split.is_saturated());
    assert!(!split.target_found_with_confidence());

    let mut same = BasinCensus::new(1, 0.1).unwrap();
    for index in 0..20 {
        let observation = same
            .observe_witness(&[f64::from(index) * 0.5], Some(WitnessId::from_raw(1)))
            .unwrap();
        assert_eq!(observation.basin_id, BasinId::from_raw(0));
        assert_eq!(observation.created, index == 0);
    }
    assert_eq!(same.len(), 1);
    assert_eq!(same.singleton_count(), 0);
    assert!(same.unseen_mass().unwrap() < 0.2);
    assert!(same.is_saturated());
    assert!(same.target_found_with_confidence());

    let mut distinct = BasinCensus::new(1, 10.0).unwrap();
    distinct
        .observe_witness(&[0.0], Some(WitnessId::from_raw(1)))
        .unwrap();
    distinct
        .observe_witness(&[0.0], Some(WitnessId::from_raw(2)))
        .unwrap();
    assert_eq!(distinct.len(), 2);
    assert!(!distinct.target_found_with_confidence());

    let mut anonymous = BasinCensus::new(1, 0.1).unwrap();
    for _ in 0..20 {
        anonymous.observe(&[0.0]).unwrap();
    }
    assert!(anonymous.is_saturated());
    assert!(!anonymous.has_witness());
    assert!(!anonymous.target_found_with_confidence());
}

#[test]
fn witnessed_confidence_follows_the_esty_bound_not_a_fixed_count() {
    let witness = Some(WitnessId::from_raw(7));
    let mut early = BasinCensus::new(1, 0.1).unwrap();
    early.observe_witness(&[0.0], witness).unwrap();
    early.observe_witness(&[0.4], witness).unwrap();
    assert_eq!(early.total_visits(), 2);
    assert_eq!(early.unseen_mass(), Some(0.0));
    assert!(early.total_visits() < anneal_core::catalog::PRODUCTION_MINIMUM_VISITS);
    assert!(!early.is_saturated());
    assert!(!early.target_found_with_confidence());

    let mut repeated = BasinCensus::new(1, 0.1).unwrap();
    for index in 0..4 {
        repeated
            .observe_witness(&[f64::from(index)], witness)
            .unwrap();
        assert!(
            !repeated.target_found_with_confidence(),
            "visit {} is still short of the bound",
            index + 1
        );
    }
    repeated.observe_witness(&[4.0], witness).unwrap();
    assert_eq!(repeated.total_visits(), 5);
    assert_eq!(repeated.singleton_count(), 0);
    assert_eq!(repeated.doubleton_count(), 0);
    assert!(repeated.total_visits() < anneal_core::catalog::PRODUCTION_MINIMUM_VISITS);
    assert!(!repeated.is_saturated());
    assert!(repeated.target_found_with_confidence());

    let mut distinct = BasinCensus::new(1, 0.1).unwrap();
    distinct
        .observe_witness(&[0.0], Some(WitnessId::from_raw(1)))
        .unwrap();
    distinct
        .observe_witness(&[1.0], Some(WitnessId::from_raw(2)))
        .unwrap();
    assert_eq!(distinct.total_visits(), 2);
    assert!(!distinct.target_found_with_confidence());

    // Four visits of one witness and two of another: the hatch is under
    // the ceiling, and the doubleton keeps the Esty upper bound above it.
    let mut mixed = BasinCensus::new(1, 0.1).unwrap();
    for index in 0..4 {
        mixed
            .observe_witness(&[f64::from(index)], Some(WitnessId::from_raw(1)))
            .unwrap();
    }
    mixed
        .observe_witness(&[10.0], Some(WitnessId::from_raw(2)))
        .unwrap();
    mixed
        .observe_witness(&[10.4], Some(WitnessId::from_raw(2)))
        .unwrap();
    assert_eq!(mixed.total_visits(), 6);
    assert_eq!(mixed.singleton_count(), 0);
    assert_eq!(mixed.doubleton_count(), 1);
    assert!(!mixed.target_found_with_confidence());
    mixed
        .observe_witness(&[10.2], Some(WitnessId::from_raw(2)))
        .unwrap();
    assert_eq!(mixed.total_visits(), 7);
    assert_eq!(mixed.doubleton_count(), 0);
    assert!(mixed.target_found_with_confidence());
    assert!(!mixed.is_saturated());
}

#[test]
fn witness_tripleton_ceiling_retires_the_third_clean_visit() {
    use anneal_core::catalog::census::WITNESS_TRIPLETON_CEILING;

    let witness = Some(WitnessId::from_raw(7));
    let mut census = BasinCensus::new(1, 0.1).unwrap();
    for index in 0..2 {
        census
            .observe_witness(&[f64::from(index)], witness)
            .unwrap();
    }
    assert!(!census.target_found_below(WITNESS_TRIPLETON_CEILING));
    census.observe_witness(&[2.0], witness).unwrap();
    assert_eq!(census.total_visits(), 3);
    assert_eq!(census.singleton_count(), 0);
    assert_eq!(census.doubleton_count(), 0);
    assert!(!census.target_found_with_confidence());
    assert!(census.target_found_below(WITNESS_TRIPLETON_CEILING));

    let mut mixed = BasinCensus::new(1, 0.1).unwrap();
    for index in 0..3 {
        mixed
            .observe_witness(&[f64::from(index)], Some(WitnessId::from_raw(1)))
            .unwrap();
    }
    mixed
        .observe_witness(&[10.0], Some(WitnessId::from_raw(2)))
        .unwrap();
    mixed
        .observe_witness(&[10.4], Some(WitnessId::from_raw(2)))
        .unwrap();
    assert_eq!(mixed.total_visits(), 5);
    assert_eq!(mixed.singleton_count(), 0);
    assert_eq!(mixed.doubleton_count(), 1);
    assert!(!mixed.target_found_below(WITNESS_TRIPLETON_CEILING));
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

proptest! {
    #[test]
    fn arbitrary_observation_streams_conserve_counts_and_medoids(
        points in prop::collection::vec((-100i16..100, -100i16..100), 0..512)
    ) {
        let mut census = census(0.75);
        let mut opening_descriptors = BTreeMap::new();

        for &(x, y) in &points {
            let descriptor = [f64::from(x), f64::from(y)];
            let observation = census.observe(&descriptor).unwrap();
            opening_descriptors
                .entry(observation.basin_id)
                .or_insert(descriptor);
        }

        prop_assert_eq!(census.total_visits(), points.len() as u64);
        prop_assert_eq!(
            census.entries().iter().map(|entry| entry.visits()).sum::<u64>(),
            census.total_visits()
        );
        prop_assert_eq!(census.entries().len(), opening_descriptors.len());
        for entry in census.entries() {
            prop_assert_eq!(entry.medoid(), opening_descriptors[&entry.id()]);
        }
    }
}
