use anneal_core::methods::feynman_kac::{
    BasinEvidence, EpochSubmissionOutcome, PopulationMember, SelectionCoefficients,
    SynchronousPopulation, ascending_fractional_ranks, population_family_position,
    population_rejuvenation_draw, rank_population, reconfiguration_plan,
};

fn coefficients() -> SelectionCoefficients {
    SelectionCoefficients {
        energy: 1.0,
        novelty: 0.8,
        scarcity: 0.6,
        uncertainty: 0.4,
        log_weight_clip: 4.0,
    }
}

#[test]
fn energy_ranks_are_invariant_to_positive_affine_units() {
    let energies = [-543.6, -541.2, -539.0, -530.5];
    let converted = energies.map(|energy| 7.5 * energy + 19.0);

    assert_eq!(
        ascending_fractional_ranks(&energies).unwrap(),
        ascending_fractional_ranks(&converted).unwrap()
    );
}

#[test]
fn novelty_and_scarcity_protect_a_productive_second_funnel() {
    let evidence = [
        BasinEvidence::new(0.0, 0.0, 0.0).unwrap(),
        BasinEvidence::new(0.25, 1.0, 1.0).unwrap(),
        BasinEvidence::new(0.75, 0.4, 0.5).unwrap(),
        BasinEvidence::new(1.0, 0.2, 0.1).unwrap(),
    ];
    let plan = reconfiguration_plan(&evidence, coefficients(), 0.125, 2).unwrap();

    assert!(plan.weights()[1] > plan.weights()[0]);
    assert_eq!(plan.parents().len(), evidence.len());
    assert!(plan.diagnostics().unique_parents >= 2);
    assert!(plan.diagnostics().max_family_size <= 2);
}

#[test]
fn systematic_reconfiguration_is_fixed_size_capped_and_replayable() {
    let evidence = [
        BasinEvidence::new(0.0, 0.0, 0.0).unwrap(),
        BasinEvidence::new(0.4, 0.2, 0.2).unwrap(),
        BasinEvidence::new(0.7, 0.8, 0.7).unwrap(),
        BasinEvidence::new(1.0, 1.0, 1.0).unwrap(),
    ];

    let first = reconfiguration_plan(&evidence, coefficients(), 0.375, 2).unwrap();
    let replay = reconfiguration_plan(&evidence, coefficients(), 0.375, 2).unwrap();

    assert_eq!(first.parents(), replay.parents());
    assert_eq!(first.parents().len(), 4);
    assert!(first.parents().iter().all(|parent| *parent < 4));
    assert!(first.diagnostics().max_family_size <= 2);
    assert!(first.diagnostics().effective_sample_size >= 1.0);
    assert!(first.diagnostics().effective_sample_size <= 4.0);
}

#[test]
fn equal_evidence_preserves_one_parent_per_chain() {
    let evidence = [BasinEvidence::new(0.5, 0.5, 0.5).unwrap(); 4];
    let plan = reconfiguration_plan(&evidence, coefficients(), 0.91, 2).unwrap();

    assert_eq!(plan.parents(), &[0, 1, 2, 3]);
    assert_eq!(plan.diagnostics().unique_parents, 4);
    assert_eq!(plan.diagnostics().max_family_size, 1);
    assert!((plan.diagnostics().effective_sample_size - 4.0).abs() < 1e-12);
    assert!(plan.diagnostics().offspring_variance.abs() < 1e-12);
}

#[test]
fn invalid_ranks_and_offsets_are_rejected() {
    assert!(BasinEvidence::new(-0.1, 0.5, 0.5).is_err());
    assert!(BasinEvidence::new(0.5, f64::NAN, 0.5).is_err());

    let evidence = [BasinEvidence::new(0.5, 0.5, 0.5).unwrap(); 4];
    assert!(reconfiguration_plan(&evidence, coefficients(), 1.0, 2).is_err());
    assert!(reconfiguration_plan(&evidence, coefficients(), 0.5, 0).is_err());
}

#[test]
fn coordinator_ranks_raw_population_evidence_once_per_epoch() {
    let members = [
        PopulationMember::new(11, -397.0, 0.2, 12.0).unwrap(),
        PopulationMember::new(17, -399.0, 0.9, 2.0).unwrap(),
        PopulationMember::new(23, -398.0, 0.6, 5.0).unwrap(),
        PopulationMember::new(29, -396.0, 0.1, 20.0).unwrap(),
    ];
    let ranked = rank_population(&members).unwrap();

    assert_eq!(ranked[0].replica(), 11);
    assert_eq!(ranked[1].evidence().energy_rank(), 0.0);
    assert_eq!(ranked[1].evidence().novelty_rank(), 1.0);
    assert_eq!(ranked[1].evidence().scarcity_rank(), 1.0);
    assert_eq!(ranked[3].evidence().energy_rank(), 1.0);
    assert_eq!(ranked[3].evidence().scarcity_rank(), 0.0);
}

#[test]
fn scarcity_is_inverse_visit_count_but_ranking_is_scale_free() {
    let first = PopulationMember::new(0, -10.0, 0.5, 1.0).unwrap();
    let second = PopulationMember::new(1, -10.0, 0.5, 50.0).unwrap();
    let ranked = rank_population(&[first, second]).unwrap();

    assert_eq!(ranked[0].evidence().scarcity_rank(), 1.0);
    assert_eq!(ranked[1].evidence().scarcity_rank(), 0.0);
}

#[test]
fn gmrf_uncertainty_augments_but_does_not_replace_exact_scarcity() {
    let certain = PopulationMember::new_with_uncertainty(0, -10.0, 0.5, 2.0, 0.1).unwrap();
    let uncertain = PopulationMember::new_with_uncertainty(1, -10.0, 0.5, 2.0, 0.9).unwrap();
    let ranked = rank_population(&[certain, uncertain]).unwrap();

    assert_eq!(ranked[0].evidence().scarcity_rank(), 0.5);
    assert_eq!(ranked[1].evidence().scarcity_rank(), 0.5);
    assert_eq!(ranked[0].evidence().uncertainty_rank(), 0.0);
    assert_eq!(ranked[1].evidence().uncertainty_rank(), 1.0);
    let evidence = [ranked[0].evidence(), ranked[1].evidence()];
    let plan = reconfiguration_plan(&evidence, coefficients(), 0.25, 1).unwrap();
    assert!(plan.weights()[1] > plan.weights()[0]);
}

#[test]
fn raw_population_rejects_duplicate_replicas_and_nonfinite_metrics() {
    assert!(PopulationMember::new(0, f64::NAN, 0.5, 1.0).is_err());
    assert!(PopulationMember::new(0, -1.0, -0.1, 1.0).is_err());
    assert!(PopulationMember::new(0, -1.0, 0.1, 0.0).is_err());

    let member = PopulationMember::new(3, -1.0, 0.1, 1.0).unwrap();
    assert!(rank_population(&[member, member]).is_err());
}

#[test]
fn synchronous_epoch_waits_for_every_replica_and_returns_replica_parents() {
    let mut population = SynchronousPopulation::new([0, 1, 2, 3], coefficients(), 2, 91).unwrap();
    let members = [
        PopulationMember::new(0, -10.0, 0.2, 8.0).unwrap(),
        PopulationMember::new(1, -11.0, 0.9, 1.0).unwrap(),
        PopulationMember::new(2, -9.0, 0.5, 3.0).unwrap(),
        PopulationMember::new(3, -8.0, 0.1, 13.0).unwrap(),
    ];

    for (submitted, member) in members.into_iter().enumerate() {
        let outcome = population.submit(0, member).unwrap();
        if submitted < 3 {
            assert_eq!(
                outcome,
                EpochSubmissionOutcome::Pending {
                    epoch: 0,
                    submitted: submitted + 1,
                    required: 4,
                }
            );
        } else {
            let EpochSubmissionOutcome::Ready(plan) = outcome else {
                panic!("fourth replica must close the epoch")
            };
            assert_eq!(plan.epoch(), 0);
            assert_eq!(plan.destinations(), &[0, 1, 2, 3]);
            assert!(plan.parents().iter().all(|parent| *parent < 4));
            assert!(plan.diagnostics().max_family_size <= 2);
        }
    }
}

#[test]
fn synchronous_epoch_replay_is_idempotent_and_conflicts_are_rejected() {
    let mut population = SynchronousPopulation::new([4, 9], coefficients(), 1, 71).unwrap();
    let first = PopulationMember::new(4, -4.0, 0.4, 2.0).unwrap();

    let pending = population.submit(0, first).unwrap();
    assert_eq!(population.submit(0, first).unwrap(), pending);
    let conflict = PopulationMember::new(4, -5.0, 0.4, 2.0).unwrap();
    assert!(population.submit(0, conflict).is_err());

    let second = PopulationMember::new(9, -3.0, 0.8, 1.0).unwrap();
    let ready = population.submit(0, second).unwrap();
    assert_eq!(population.submit(0, second).unwrap(), ready);
}

#[test]
fn synchronous_epoch_rejects_unknown_replicas_and_skipped_epochs() {
    let mut population = SynchronousPopulation::new([0, 1], coefficients(), 1, 5).unwrap();
    assert!(
        population
            .submit(0, PopulationMember::new(7, -1.0, 0.2, 1.0).unwrap())
            .is_err()
    );
    assert!(
        population
            .submit(1, PopulationMember::new(0, -1.0, 0.2, 1.0).unwrap())
            .is_err()
    );
}

#[test]
fn cloned_offspring_receive_stable_distinct_rejuvenation_draws() {
    let destinations = [0, 1, 2, 3];
    let parents = [2, 2, 1, 3];
    let first = population_family_position(&destinations, &parents, 0).unwrap();
    let second = population_family_position(&destinations, &parents, 1).unwrap();

    assert_eq!(first.parent(), 2);
    assert_eq!(first.ordinal(), 0);
    assert_eq!(first.family_size(), 2);
    assert_eq!(second.parent(), 2);
    assert_eq!(second.ordinal(), 1);
    assert_eq!(second.family_size(), 2);
    assert_ne!(
        population_rejuvenation_draw(91, 7, 0, first.ordinal()),
        population_rejuvenation_draw(91, 7, 1, second.ordinal())
    );
    assert_eq!(
        population_rejuvenation_draw(91, 7, 0, first.ordinal()),
        population_rejuvenation_draw(91, 7, 0, first.ordinal())
    );
    assert!(population_family_position(&destinations, &parents[..3], 0).is_none());
    assert!(population_family_position(&destinations, &parents, 9).is_none());
}

#[test]
fn an_abstaining_replica_releases_the_epoch_instead_of_holding_it_open() {
    // Three replicas submit and one cannot. Requiring the absent one holds
    // the barrier open until every budget drains, which is what stalled the
    // 2026-08-14 campaign: one replica polled sixty thousand times while
    // its peers finished and left.
    let mut population = SynchronousPopulation::new([0, 1, 2, 3], coefficients(), 2, 17).unwrap();
    for replica in 0..3u32 {
        let member = PopulationMember::new(replica, -10.0 + f64::from(replica), 0.3, 2.0).unwrap();
        assert!(matches!(
            population.submit(0, member).unwrap(),
            EpochSubmissionOutcome::Pending { .. }
        ));
    }

    let outcome = population.abstain(0, 3).unwrap();

    let EpochSubmissionOutcome::Ready(plan) = outcome else {
        panic!("an abstention completing the barrier must yield a plan");
    };
    assert_eq!(plan.epoch(), 0);
    // The abstaining replica is not a destination: it has nothing to be
    // reconfigured into and is not part of this population.
    assert_eq!(plan.destinations(), &[0, 1, 2]);
    for parent in plan.parents() {
        assert!(plan.destinations().contains(parent));
    }
}

#[test]
fn abstention_applies_to_one_epoch_and_not_to_the_next() {
    let mut population = SynchronousPopulation::new([0, 1], coefficients(), 1, 23).unwrap();
    population
        .submit(0, PopulationMember::new(0, -5.0, 0.4, 1.0).unwrap())
        .unwrap();
    assert!(matches!(
        population.abstain(0, 1).unwrap(),
        EpochSubmissionOutcome::Ready(_)
    ));

    // Replica 1 abstained because its own state offered nothing at that
    // barrier, which says nothing about the next one.
    let outcome = population
        .submit(1, PopulationMember::new(0, -6.0, 0.4, 1.0).unwrap())
        .unwrap();

    assert_eq!(
        outcome,
        EpochSubmissionOutcome::Pending {
            epoch: 1,
            submitted: 1,
            required: 2,
        }
    );
}

#[test]
fn an_epoch_every_replica_abstains_from_closes_vacantly() {
    // Leaving it open would wedge every epoch counter on a barrier nobody
    // can meet. Zero submitted of zero required is the vacant-close answer,
    // which no genuinely pending epoch can produce.
    let mut population = SynchronousPopulation::new([0, 1], coefficients(), 1, 41).unwrap();
    assert!(matches!(
        population.abstain(0, 0).unwrap(),
        EpochSubmissionOutcome::Pending {
            epoch: 0,
            submitted: 1,
            required: 1,
        } | EpochSubmissionOutcome::Pending {
            epoch: 0,
            submitted: 0,
            required: 1,
        }
    ));

    let close = population.abstain(0, 1).unwrap();

    assert_eq!(
        close,
        EpochSubmissionOutcome::Pending {
            epoch: 0,
            submitted: 0,
            required: 0,
        }
    );
    // The next epoch is open and both replicas are expected again.
    let outcome = population
        .submit(1, PopulationMember::new(0, -3.0, 0.2, 1.0).unwrap())
        .unwrap();
    assert_eq!(
        outcome,
        EpochSubmissionOutcome::Pending {
            epoch: 1,
            submitted: 1,
            required: 2,
        }
    );
    // A late look back at the vacant epoch answers vacantly, not with an error.
    assert_eq!(
        population.abstain(0, 0).unwrap(),
        EpochSubmissionOutcome::Pending {
            epoch: 0,
            submitted: 0,
            required: 0,
        }
    );
}
