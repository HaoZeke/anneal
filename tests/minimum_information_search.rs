use anneal_core::minimum_information::{
    MinimumInformationSearch, SearchActionCandidate, SearchMechanism,
};

fn candidate(mechanism: SearchMechanism, cost: f64) -> SearchActionCandidate {
    SearchActionCandidate {
        mechanism,
        feature: vec![0.25, -0.5],
        source_energy: -10.0,
        expected_charged_evaluations: cost,
    }
}

#[test]
fn information_rate_is_information_divided_by_charged_pes_cost() {
    let mut search = MinimumInformationSearch::new(1.0, 4.0, 1e-3).unwrap();
    let scores = search
        .score(
            &[
                candidate(SearchMechanism::BasinEscape, 20.0),
                candidate(SearchMechanism::BasinEscape, 80.0),
            ],
            256,
        )
        .unwrap();

    assert!((scores[0].information - scores[1].information).abs() < 1e-14);
    assert!(
        (scores[0].information_per_charged_evaluation
            - 4.0 * scores[1].information_per_charged_evaluation)
            .abs()
            < 1e-14
    );
}

#[test]
fn observing_one_operator_moves_minimum_information_to_the_unobserved_operator() {
    let mut search = MinimumInformationSearch::new(1.0, 4.0, 1e-3).unwrap();
    search
        .observe(SearchMechanism::BasinEscape, &[0.25, -0.5], -10.0, -11.0)
        .unwrap();

    let scores = search
        .score(
            &[
                candidate(SearchMechanism::BasinEscape, 40.0),
                candidate(SearchMechanism::SaddleRide, 40.0),
            ],
            256,
        )
        .unwrap();

    assert!(scores[1].information > scores[0].information);
    assert_eq!(search.observations(SearchMechanism::BasinEscape), 1);
    assert_eq!(search.observations(SearchMechanism::SaddleRide), 0);
}

#[test]
fn failed_operator_attempt_is_a_finite_no_improvement_observation() {
    let mut search = MinimumInformationSearch::new(1.0, 4.0, 1e-3).unwrap();
    search
        .observe(SearchMechanism::SaddleRide, &[0.25, -0.5], -10.0, -10.0)
        .unwrap();

    assert_eq!(search.observations(SearchMechanism::SaddleRide), 1);
    assert_eq!(search.incumbent_terminal_energy(), Some(-10.0));
}

#[test]
fn each_operator_has_one_immutable_feature_dimension() {
    let mut search = MinimumInformationSearch::new(1.0, 4.0, 1e-3).unwrap();
    search
        .observe(SearchMechanism::BasinEscape, &[0.0, 1.0], -2.0, -3.0)
        .unwrap();

    let error = search
        .observe(SearchMechanism::BasinEscape, &[0.0], -2.0, -3.0)
        .unwrap_err();
    assert!(error.to_string().contains("feature dimension"));
}

#[test]
fn action_models_bound_kernel_rank_without_losing_observation_counts() {
    let mut search =
        MinimumInformationSearch::new_with_maximum_model_rank(0.3, 2.0, 1e-3, 8).unwrap();
    for index in 0..40 {
        let coordinate = 0.1 * index as f64;
        search
            .observe(
                SearchMechanism::BasinEscape,
                &[coordinate, 0.0],
                -10.0,
                -10.0 - coordinate,
            )
            .unwrap();
    }

    let compression = search.compression(SearchMechanism::BasinEscape);
    assert_eq!(search.observations(SearchMechanism::BasinEscape), 40);
    assert!(compression.retained_rank <= 8);
    assert!(compression.residual_fraction.is_finite());
    assert!((0.0..=1.0).contains(&compression.residual_fraction));
    assert!(compression.rank_limited);
}

#[test]
fn batch_assignment_discounts_correlated_chain_actions() {
    let mut search = MinimumInformationSearch::new(1.0, 4.0, 1e-3).unwrap();
    let actions = [
        SearchActionCandidate {
            mechanism: SearchMechanism::BasinEscape,
            feature: vec![0.0, 0.0],
            source_energy: -10.0,
            expected_charged_evaluations: 20.0,
        },
        SearchActionCandidate {
            mechanism: SearchMechanism::BasinEscape,
            feature: vec![0.0, 0.0],
            source_energy: -10.0,
            expected_charged_evaluations: 20.0,
        },
        SearchActionCandidate {
            mechanism: SearchMechanism::BasinEscape,
            feature: vec![4.0, 0.0],
            source_energy: -10.0,
            expected_charged_evaluations: 20.0,
        },
    ];

    let selected = search
        .assign_batch(&actions, &[0, 1, 2], 2, 1, 256)
        .unwrap();

    assert_eq!(selected, vec![0, 2]);
}

#[test]
fn batch_assignment_maximizes_marginal_information_per_cost() {
    let mut search = MinimumInformationSearch::new(1.0, 4.0, 1e-3).unwrap();
    let actions = [
        SearchActionCandidate {
            mechanism: SearchMechanism::BasinEscape,
            feature: vec![0.0, 0.0],
            source_energy: -10.0,
            expected_charged_evaluations: 80.0,
        },
        SearchActionCandidate {
            mechanism: SearchMechanism::BasinEscape,
            feature: vec![4.0, 0.0],
            source_energy: -10.0,
            expected_charged_evaluations: 20.0,
        },
    ];

    let selected = search.assign_batch(&actions, &[0, 1], 1, 1, 256).unwrap();

    assert_eq!(selected, vec![1]);
}

#[test]
fn batch_assignment_fills_the_population_without_exceeding_family_capacity() {
    let mut search = MinimumInformationSearch::new(1.0, 4.0, 1e-3).unwrap();
    let actions = [
        SearchActionCandidate {
            mechanism: SearchMechanism::BasinEscape,
            feature: vec![0.0, 0.0],
            source_energy: -10.0,
            expected_charged_evaluations: 20.0,
        },
        SearchActionCandidate {
            mechanism: SearchMechanism::BasinEscape,
            feature: vec![4.0, 0.0],
            source_energy: -10.0,
            expected_charged_evaluations: 20.0,
        },
    ];

    let selected = search.assign_batch(&actions, &[7, 11], 4, 2, 256).unwrap();

    assert_eq!(selected.len(), 4);
    assert_eq!(selected.iter().filter(|&&index| index == 0).count(), 2);
    assert_eq!(selected.iter().filter(|&&index| index == 1).count(), 2);
}

#[test]
fn batch_assignment_fills_exact_duplicate_actions_at_tight_numeric_tolerance() {
    let mut search = MinimumInformationSearch::new(1.0, 1.0, 1e-12).unwrap();
    let actions = (0..4)
        .map(|_| SearchActionCandidate {
            mechanism: SearchMechanism::BasinEscape,
            feature: vec![0.0, 0.0],
            source_energy: -10.0,
            expected_charged_evaluations: 20.0,
        })
        .collect::<Vec<_>>();

    let selected = search
        .assign_batch(&actions, &[0, 1, 2, 3], 4, 1, 256)
        .unwrap();

    assert_eq!(selected, vec![0, 1, 2, 3]);
}
