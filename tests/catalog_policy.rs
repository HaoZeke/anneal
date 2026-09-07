use anneal_core::catalog::BasinCensus;
use anneal_core::catalog::MixingEvidence;
use anneal_core::catalog_policy::{
    ActiveCatalogRelation, AggregateProgress, CatalogPolicy, CatalogPolicyInput, CensusEvidence,
    PolicyAction, PolicyReason, ValidationState,
};

fn census_with_repeated_visits(visits: usize) -> (BasinCensus, anneal_core::catalog::BasinId) {
    let mut census = BasinCensus::new(2, 0.1).unwrap();
    let mut basin_id = None;
    for _ in 0..visits {
        basin_id = Some(census.observe(&[0.0, 0.0]).unwrap().basin_id);
    }
    (census, basin_id.unwrap())
}

fn input(
    relation: ActiveCatalogRelation,
    census: CensusEvidence,
    progress: AggregateProgress,
) -> CatalogPolicyInput {
    CatalogPolicyInput {
        validation: ValidationState::Validated,
        relation,
        census,
        progress,
        local_stall_slices: 0,
        local_deepened: false,
        mixing: MixingEvidence::default(),
        leftover_lambda: 0.0,
        interface_rank: u32::MAX,
        interface_threshold: 0.0,
        occupied_family_count: 0,
        packing_saturated: false,
        leftover_dwell: true,
        ei_exhausted: false,
        min_families: 1,
        on_published_prize: false,
    }
}

#[test]
fn published_reference_does_not_change_a_stalled_chains_work() {
    let (census, basin_id) = census_with_repeated_visits(4);
    let mut state = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(90, 100).unwrap(),
    );
    state.interface_rank = 1;
    state.local_stall_slices = anneal_core::catalog_policy::LOCAL_STALL_LEAVE;
    let without_reference = CatalogPolicy::decide(state);
    assert_eq!(without_reference.action, PolicyAction::Leave);
    state.on_published_prize = true;
    assert_eq!(CatalogPolicy::decide(state), without_reference);
}

#[test]
fn leftover_unsaturated_extra_walks_like_serial() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut extra = input(
        ActiveCatalogRelation::SameBasin,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    extra.leftover_dwell = false;
    let decision = CatalogPolicy::decide(extra);
    assert_eq!(decision.action, PolicyAction::Leave);
    assert_eq!(decision.reason, PolicyReason::OccupiedPackingLeave);
}

#[test]
fn leftover_unsaturated_walks_through_census_exhaustion() {
    let (census, basin_id) = census_with_repeated_visits(8);
    let mut state = input(
        ActiveCatalogRelation::SameBasin,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.leftover_dwell = false;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::Leave);
    assert_eq!(decision.reason, PolicyReason::LocalCensusExhausted);
}

#[test]
fn leftover_unsaturated_does_not_yank_to_a_remote_anchor() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut state = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: true,
        },
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.leftover_dwell = false;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::IsomerWalk);
}

#[test]
fn leftover_unsaturated_does_not_explore_off_the_walk() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut state = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: false,
        },
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.leftover_dwell = false;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::IsomerWalk);
}

#[test]
fn leftover_unsaturated_pruned_extra_still_walks() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut extra = input(
        ActiveCatalogRelation::SameBasin,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    extra.leftover_dwell = false;
    extra.mixing.pruned = true;
    let decision = CatalogPolicy::decide(extra);
    assert_eq!(decision.action, PolicyAction::Leave);
    assert_eq!(decision.reason, PolicyReason::HyperbandPruned);
}

#[test]
fn leftover_unsaturated_extra_leaves_when_another_family_is_on_file() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut extra = input(
        ActiveCatalogRelation::SameBasin,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    extra.leftover_dwell = false;
    extra.occupied_family_count = 2;
    let decision = CatalogPolicy::decide(extra);
    assert_eq!(decision.action, PolicyAction::Leave);
    assert_eq!(decision.reason, PolicyReason::OccupiedPackingLeave);
}

#[test]
fn leftover_unsaturated_incumbent_walks_when_another_family_is_on_file() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut incumbent = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    incumbent.leftover_dwell = false;
    incumbent.occupied_family_count = 2;
    let decision = CatalogPolicy::decide(incumbent);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::IsomerWalk);
}

#[test]
fn leftover_unsaturated_novel_family_still_walks() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut novel = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: false,
        },
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    novel.leftover_dwell = false;
    novel.occupied_family_count = 2;
    let decision = CatalogPolicy::decide(novel);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::IsomerWalk);
}

#[test]
fn extras_on_a_published_prize_keep() {
    let (census, basin_id) = census_with_repeated_visits(4);
    let mut extra = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(90, 100).unwrap(),
    );
    extra.interface_rank = 1;
    extra.on_published_prize = true;
    let decision = CatalogPolicy::decide(extra);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    extra.on_published_prize = false;
    let walk = CatalogPolicy::decide(extra);
    assert_eq!(walk.action, PolicyAction::ContinueLocal);
}

#[test]
fn incumbent_replica_stays_in_the_well_until_it_stalls() {
    let (census, basin_id) = census_with_repeated_visits(21);
    let relaxing = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(90, 100).unwrap(),
    );
    let mut stalled = relaxing;
    stalled.local_stall_slices = 8;

    let local = CatalogPolicy::decide(relaxing);
    assert_eq!(local.action, PolicyAction::ContinueLocal);
    assert_eq!(local.reason, PolicyReason::IsomerWalk);
    assert_eq!(local.reason.code(), "isomer_walk");

    let leave = CatalogPolicy::decide(stalled);
    assert_eq!(leave.action, PolicyAction::Leave);
    assert_eq!(leave.reason, PolicyReason::LocalStall);
}

#[test]
fn a_better_catalog_min_is_taken_before_same_packing_leave() {
    let (census, basin_id) = census_with_repeated_visits(8);
    let mut state = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: true,
        },
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.local_stall_slices = 8;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::Exploit { win_only: false });
    assert_eq!(decision.reason, PolicyReason::RemoteAnchorOpen);
}

#[test]
fn global_saturation_never_forces_an_unrelated_replica_to_leave() {
    let (census, _) = census_with_repeated_visits(21);
    let state = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: false,
        },
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(30, 100).unwrap(),
    );

    let decision = CatalogPolicy::decide(state);

    assert_eq!(decision.action, PolicyAction::Explore);
    assert_eq!(decision.reason, PolicyReason::GlobalCensusSaturatedExplore);
}

#[test]
fn a_family_extra_of_a_crowded_packing_leaves_it() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut extra = input(
        ActiveCatalogRelation::SameBasin,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    let unseated = CatalogPolicy::decide(extra);
    assert_eq!(unseated.action, PolicyAction::Leave);
    assert_eq!(unseated.reason, PolicyReason::OccupiedPackingLeave);
    extra.interface_rank = 1;
    extra.interface_threshold = 0.5;
    let leave = CatalogPolicy::decide(extra);
    assert_eq!(leave.action, PolicyAction::Leave);
    assert_eq!(leave.reason, PolicyReason::OccupiedPackingLeave);
    assert_eq!(leave.reason.code(), "occupied_packing_leave");
    extra.mixing.pruned = true;
    let pruned = CatalogPolicy::decide(extra);
    assert_eq!(pruned.action, PolicyAction::Leave);
    assert_eq!(pruned.reason, PolicyReason::HyperbandPruned);
}

#[test]
fn a_packing_champion_walks_isomers_rather_than_leaving() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let state = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::IsomerWalk);
}

#[test]
fn a_seated_extra_leaves_once_it_crosses_its_own_interface() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let mut extra = input(
        ActiveCatalogRelation::SameBasin,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    extra.interface_rank = 1;
    extra.interface_threshold = 0.5;

    extra.leftover_lambda = 0.4;
    let below = CatalogPolicy::decide(extra);
    assert_eq!(below.action, PolicyAction::Leave);
    assert_eq!(below.reason, PolicyReason::OccupiedPackingLeave);

    extra.leftover_lambda = 0.5;
    let crossed = CatalogPolicy::decide(extra);
    assert_eq!(crossed.action, PolicyAction::Leave);
    assert_eq!(crossed.reason, PolicyReason::InterfaceCrossed);
}

#[test]
fn exact_local_census_visits_trigger_leave_without_a_height_proxy() {
    let (census, basin_id) = census_with_repeated_visits(8);
    let state = input(
        ActiveCatalogRelation::SameBasin,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );

    let decision = CatalogPolicy::decide(state);

    assert_eq!(decision.action, PolicyAction::Leave);
    assert_eq!(decision.reason, PolicyReason::LocalCensusExhausted);
}

#[test]
fn local_stall_triggers_leave_only_from_the_related_basin() {
    let (census, basin_id) = census_with_repeated_visits(1);
    let evidence = CensusEvidence::from_census(&census, Some(basin_id));
    let mut related = input(
        ActiveCatalogRelation::SameBasin,
        evidence,
        AggregateProgress::new(20, 100).unwrap(),
    );
    related.local_stall_slices = 8;
    let mut unrelated = related;
    unrelated.relation = ActiveCatalogRelation::Unrelated {
        lower_energy_anchor: false,
    };

    assert_eq!(CatalogPolicy::decide(related).action, PolicyAction::Leave);
    assert_eq!(
        CatalogPolicy::decide(related).reason,
        PolicyReason::LocalStall
    );
    assert_eq!(
        CatalogPolicy::decide(unrelated).action,
        PolicyAction::Explore
    );
}

#[test]
fn aggregate_progress_tightens_catalog_exploitation() {
    let (census, _) = census_with_repeated_visits(2);
    let relation = ActiveCatalogRelation::Unrelated {
        lower_energy_anchor: true,
    };
    let early = input(
        relation,
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(49, 100).unwrap(),
    );
    let late = input(
        relation,
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(50, 100).unwrap(),
    );

    assert_eq!(
        CatalogPolicy::decide(early).action,
        PolicyAction::Exploit { win_only: false }
    );
    assert_eq!(
        CatalogPolicy::decide(late).action,
        PolicyAction::Exploit { win_only: true }
    );
    assert_eq!(
        CatalogPolicy::decide(early).reason,
        PolicyReason::RemoteAnchorOpen
    );
    assert_eq!(
        CatalogPolicy::decide(late).reason,
        PolicyReason::RemoteAnchorClosed
    );
}

#[test]
fn validation_failure_and_local_descent_preserve_local_search() {
    let (census, _) = census_with_repeated_visits(2);
    let mut rejected = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: true,
        },
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(10, 100).unwrap(),
    );
    rejected.validation = ValidationState::Rejected;
    let mut descending = rejected;
    descending.validation = ValidationState::Validated;
    descending.local_deepened = true;

    assert_eq!(
        CatalogPolicy::decide(rejected).reason,
        PolicyReason::ValidationRejected
    );
    assert_eq!(
        CatalogPolicy::decide(descending).reason,
        PolicyReason::LocalDescent
    );
    assert_eq!(
        CatalogPolicy::decide(descending).action,
        PolicyAction::ContinueLocal
    );
}

#[test]
fn explore_collapse_leaves_instead_of_adopting_a_lower_funnel() {
    let (census, _) = census_with_repeated_visits(2);
    let mut state = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: true,
        },
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.mixing.explore_collapsed = true;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::Leave);
    assert_eq!(decision.reason, PolicyReason::ExploreCollapsed);
}

#[test]
fn explore_collapse_does_not_yank_a_different_packing() {
    let (census, _) = census_with_repeated_visits(2);
    let mut state = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: false,
        },
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.mixing.explore_collapsed = true;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::Explore);
    assert_eq!(decision.reason, PolicyReason::UnrelatedCatalogExplore);
}

#[test]
fn a_mixed_uncertified_incumbent_keeps_the_isomer_walk() {
    let (census, basin_id) = census_with_repeated_visits(2);
    let mut state = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.mixing.explore_collapsed = true;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::IsomerWalk);
}

#[test]
fn hyperband_prune_reseeds_instead_of_adopting_a_lower_funnel() {
    let (census, _) = census_with_repeated_visits(2);
    let mut state = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: true,
        },
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.mixing.pruned = true;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::Leave);
    assert_eq!(decision.reason, PolicyReason::HyperbandPruned);
    assert_eq!(decision.reason.code(), "hyperband_pruned");
}

#[test]
fn a_crowded_packing_reseeds_even_when_the_walk_deepened() {
    let (census, _) = census_with_repeated_visits(2);
    let mut state = input(
        ActiveCatalogRelation::SameBasin,
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.local_deepened = true;
    state.mixing.pruned = true;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::Leave);
    assert_eq!(decision.reason, PolicyReason::HyperbandPruned);
}

#[test]
fn a_certified_attractor_is_not_pruned() {
    let (census, basin_id) = census_with_repeated_visits(21);
    let mut state = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(90, 100).unwrap(),
    );
    state.mixing.pruned = true;
    state.mixing.certified_attractor = true;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::CertifiedAttractor);
}

#[test]
fn a_certified_attractor_is_not_left_on_stall() {
    let (census, basin_id) = census_with_repeated_visits(21);
    let mut state = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(90, 100).unwrap(),
    );
    state.local_stall_slices = 8;
    state.mixing.certified_attractor = true;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::CertifiedAttractor);
    assert_eq!(decision.reason.code(), "certified_attractor");
}

#[test]
fn a_certificate_does_not_yank_an_unrelated_replica() {
    let (census, _) = census_with_repeated_visits(2);
    let mut state = input(
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: false,
        },
        CensusEvidence::from_census(&census, None),
        AggregateProgress::new(20, 100).unwrap(),
    );
    state.mixing.certified_attractor = true;

    let decision = CatalogPolicy::decide(state);
    assert_eq!(decision.action, PolicyAction::Explore);
    assert_eq!(decision.reason, PolicyReason::UnrelatedCatalogExplore);
}

#[test]
fn zero_aggregate_budget_is_rejected() {
    assert!(AggregateProgress::new(0, 0).is_err());
}

#[test]
fn exact_remote_census_counts_reject_impossible_evidence() {
    let evidence = CensusEvidence::from_exact_counts(5, 2, 4, false).unwrap();
    assert_eq!(evidence.total_visits(), 5);
    assert_eq!(evidence.singleton_basins(), 2);
    assert_eq!(evidence.local_basin_visits(), 4);
    assert!(!evidence.globally_saturated());

    assert!(CensusEvidence::from_exact_counts(5, 6, 0, false).is_err());
    assert!(CensusEvidence::from_exact_counts(5, 2, 6, false).is_err());
}

#[test]
fn decision_table_covers_every_discrete_input_state() {
    let (unsaturated, local_basin) = census_with_repeated_visits(1);
    let (saturated, _) = census_with_repeated_visits(21);
    let relations = [
        ActiveCatalogRelation::Empty,
        ActiveCatalogRelation::Incumbent,
        ActiveCatalogRelation::SameBasin,
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: false,
        },
        ActiveCatalogRelation::Unrelated {
            lower_energy_anchor: true,
        },
    ];
    let census_states = [
        CensusEvidence::from_census(&unsaturated, Some(local_basin)),
        CensusEvidence::from_census(&saturated, None),
    ];
    let progress_states = [
        AggregateProgress::new(0, 100).unwrap(),
        AggregateProgress::new(100, 100).unwrap(),
    ];

    for validation in [ValidationState::Validated, ValidationState::Rejected] {
        for relation in relations {
            for census in census_states {
                for progress in progress_states {
                    for local_stall_slices in [0, 8] {
                        for local_deepened in [false, true] {
                            let decision = CatalogPolicy::decide(CatalogPolicyInput {
                                validation,
                                relation,
                                census,
                                progress,
                                local_stall_slices,
                                local_deepened,
                                mixing: MixingEvidence::default(),
                                leftover_lambda: 0.0,
                                interface_rank: u32::MAX,
                                interface_threshold: 0.0,
                                occupied_family_count: 0,
                                packing_saturated: false,
                                leftover_dwell: false,
                                ei_exhausted: false,
                                min_families: 1,
                                on_published_prize: false,
                            });
                            assert!(!decision.reason.code().is_empty());
                            if relation == ActiveCatalogRelation::Incumbent
                                && validation == ValidationState::Validated
                                && !local_deepened
                            {
                                assert_eq!(decision.action, PolicyAction::ContinueLocal);
                            }
                        }
                    }
                }
            }
        }
    }
}
