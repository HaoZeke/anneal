use anneal_core::catalog::BasinCensus;
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
    }
}

#[test]
fn incumbent_replica_continues_despite_saturation_and_stall() {
    let (census, basin_id) = census_with_repeated_visits(21);
    let mut state = input(
        ActiveCatalogRelation::Incumbent,
        CensusEvidence::from_census(&census, Some(basin_id)),
        AggregateProgress::new(90, 100).unwrap(),
    );
    state.local_stall_slices = 100;

    let decision = CatalogPolicy::decide(state);

    assert_eq!(decision.action, PolicyAction::ContinueLocal);
    assert_eq!(decision.reason, PolicyReason::IncumbentLocalSearch);
    assert_eq!(decision.reason.code(), "incumbent_local_search");
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
fn zero_aggregate_budget_is_rejected() {
    assert!(AggregateProgress::new(0, 0).is_err());
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
                            });
                            assert!(!decision.reason.code().is_empty());
                            if relation == ActiveCatalogRelation::Incumbent
                                && validation == ValidationState::Validated
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
