use std::collections::BTreeSet;

use anneal_core::region_assignment::{
    RegionCandidate, RegionUtility, diversity_constrained_assignment,
};

fn candidate(source: u32, region: usize, utility: f64) -> RegionCandidate {
    RegionCandidate::new(
        source,
        region,
        RegionUtility {
            transition_uncertainty: utility,
            inverse_occupancy: 0.0,
            outgoing_frontier: 0.0,
            geometry_compatibility: 0.0,
            access_cost: 0.0,
        },
    )
    .unwrap()
}

#[test]
fn distinct_regions_receive_coverage_before_duplicate_families() {
    let candidates = vec![
        candidate(0, 7, 100.0),
        candidate(1, 7, 90.0),
        candidate(2, 8, 1.0),
        candidate(3, 9, 0.5),
    ];
    let assigned = diversity_constrained_assignment(&candidates, 4, 2).unwrap();

    let represented = assigned
        .iter()
        .map(|parent| {
            candidates
                .iter()
                .find(|item| item.source() == *parent)
                .unwrap()
                .region()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(represented, BTreeSet::from([7, 8, 9]));
    assert!(candidates.iter().all(|candidate| {
        assigned
            .iter()
            .filter(|parent| **parent == candidate.source())
            .count()
            <= 2
    }));
}

#[test]
fn equivalent_region_candidates_use_unused_families_before_cloning() {
    let candidates = vec![
        candidate(0, 7, 1.0),
        candidate(1, 7, 1.0),
        candidate(2, 7, 1.0),
        candidate(3, 7, 1.0),
    ];

    let assigned = diversity_constrained_assignment(&candidates, 4, 2).unwrap();

    assert_eq!(assigned, vec![0, 1, 2, 3]);
}

#[test]
fn utility_combines_frontier_uncertainty_occupancy_compatibility_and_cost() {
    let preferred = RegionUtility {
        transition_uncertainty: 0.8,
        inverse_occupancy: 0.7,
        outgoing_frontier: 0.6,
        geometry_compatibility: 0.9,
        access_cost: 0.2,
    };
    let expensive = RegionUtility {
        access_cost: 4.0,
        ..preferred
    };

    assert!(preferred.score() > expensive.score());
    assert!(preferred.score().is_finite());
}

#[test]
fn inadmissible_geometry_is_never_assigned() {
    let candidates = vec![
        candidate(0, 0, 1.0),
        candidate(1, 1, 100.0).with_admissible(false),
    ];
    let assigned = diversity_constrained_assignment(&candidates, 2, 2).unwrap();

    assert_eq!(assigned, vec![0, 0]);
}
