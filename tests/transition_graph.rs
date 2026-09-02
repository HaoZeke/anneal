use anneal_core::transition_graph::{AttractionRegionConfig, TransitionGraph, TransitionOutcome};

#[test]
fn fixed_probe_matrix_does_not_pool_adaptive_actions() {
    let mut graph = TransitionGraph::new();
    graph
        .observe("adaptive", 0, TransitionOutcome::Resolved(0))
        .unwrap();
    graph
        .observe("adaptive", 0, TransitionOutcome::Resolved(1))
        .unwrap();
    graph
        .observe("adaptive", 0, TransitionOutcome::Resolved(1))
        .unwrap();
    graph
        .observe("probe", 0, TransitionOutcome::Unresolved)
        .unwrap();

    let matrix = graph.posterior_matrix("probe", 0.5).unwrap();
    assert_eq!(matrix.dim(), (2, 3));
    assert!((matrix[[0, 0]] - 0.2).abs() < 1e-12);
    assert!((matrix[[0, 1]] - 0.2).abs() < 1e-12);
    assert!((matrix[[0, 2]] - 0.6).abs() < 1e-12);
}

#[test]
fn every_probe_row_is_a_probability_distribution_with_unresolved_mass() {
    let mut graph = TransitionGraph::new();
    graph
        .observe("probe", 2, TransitionOutcome::Resolved(0))
        .unwrap();
    graph
        .observe("probe", 2, TransitionOutcome::Resolved(2))
        .unwrap();

    let matrix = graph.posterior_matrix("probe", 0.25).unwrap();
    assert_eq!(matrix.dim(), (3, 4));
    for row in matrix.rows() {
        assert!((row.sum() - 1.0).abs() < 1e-12);
        assert!(row.iter().all(|probability| *probability > 0.0));
    }
    assert!(matrix[[2, 3]] > 0.0);
}

#[test]
fn action_counts_remain_independent() {
    let mut graph = TransitionGraph::new();
    graph
        .observe("probe", 0, TransitionOutcome::Resolved(1))
        .unwrap();
    graph
        .observe("transport", 0, TransitionOutcome::Resolved(1))
        .unwrap();
    graph
        .observe("transport", 0, TransitionOutcome::Resolved(1))
        .unwrap();

    assert_eq!(graph.count("probe", 0, TransitionOutcome::Resolved(1)), 1);
    assert_eq!(
        graph.count("transport", 0, TransitionOutcome::Resolved(1)),
        2
    );
    assert_eq!(graph.observations("probe", 0), 1);
    assert_eq!(graph.observations("transport", 0), 2);
}

#[test]
fn equal_probe_return_dynamics_form_one_attraction_region() {
    let mut graph = TransitionGraph::new();
    for source in [0, 1] {
        for _ in 0..20 {
            graph
                .observe("probe", source, TransitionOutcome::Resolved(0))
                .unwrap();
            graph
                .observe("probe", source, TransitionOutcome::Resolved(1))
                .unwrap();
        }
    }
    for _ in 0..40 {
        graph
            .observe("probe", 2, TransitionOutcome::Resolved(2))
            .unwrap();
    }

    let regions = graph
        .attraction_regions(&AttractionRegionConfig {
            probe_action: "probe".into(),
            concentration: 0.1,
            diffusion_steps: 1,
            maximum_distance: 0.1,
            minimum_probes: 8,
        })
        .unwrap();

    assert_eq!(regions, vec![vec![0, 1], vec![2]]);
}

#[test]
fn insufficient_probe_evidence_stays_singleton_unresolved() {
    let mut graph = TransitionGraph::new();
    for _ in 0..12 {
        graph
            .observe("probe", 0, TransitionOutcome::Resolved(0))
            .unwrap();
        graph
            .observe("probe", 1, TransitionOutcome::Resolved(0))
            .unwrap();
    }
    graph
        .observe("adaptive", 2, TransitionOutcome::Resolved(0))
        .unwrap();

    let regions = graph
        .attraction_regions(&AttractionRegionConfig {
            probe_action: "probe".into(),
            concentration: 0.1,
            diffusion_steps: 2,
            maximum_distance: 0.1,
            minimum_probes: 8,
        })
        .unwrap();

    assert_eq!(regions, vec![vec![0, 1], vec![2]]);
    assert_eq!(graph.observations("probe", 2), 0);
}

#[test]
fn dirichlet_risk_forecast_matches_exhaustive_two_probe_predictive() {
    let mut graph = TransitionGraph::new();
    for _ in 0..2 {
        graph
            .observe("probe", 0, TransitionOutcome::Resolved(0))
            .unwrap();
    }
    graph
        .observe("probe", 0, TransitionOutcome::Resolved(1))
        .unwrap();

    let concentration = 0.5;
    let alpha = [2.5, 1.5, 0.5];
    let information = graph
        .dirichlet_information("probe", 0, concentration)
        .unwrap()
        .unwrap();

    let total = alpha.iter().sum::<f64>();
    let trace = |parameters: &[f64]| {
        let mass = parameters.iter().sum::<f64>();
        let squared_mean = parameters
            .iter()
            .map(|parameter| (parameter / mass).powi(2))
            .sum::<f64>();
        (1.0 - squared_mean) / (mass + 1.0)
    };
    let exhaustive = alpha
        .iter()
        .enumerate()
        .map(|(first, first_alpha)| {
            let first_probability = first_alpha / total;
            let mut after_first = alpha;
            after_first[first] += 1.0;
            after_first
                .iter()
                .enumerate()
                .map(|(second, second_alpha)| {
                    let second_probability = second_alpha / (total + 1.0);
                    let mut after_second = after_first;
                    after_second[second] += 1.0;
                    first_probability * second_probability * trace(&after_second)
                })
                .sum::<f64>()
        })
        .sum::<f64>();

    assert!((information.total_concentration() - total).abs() < 1e-12);
    assert!((information.covariance_trace() - trace(&alpha)).abs() < 1e-12);
    assert!((information.expected_covariance_trace(2) - exhaustive).abs() < 1e-12);
    assert!(information.marginal_risk_reduction(0) > information.marginal_risk_reduction(1));
}

#[test]
fn balanced_transition_rows_have_more_value_of_information() {
    let mut balanced = TransitionGraph::new();
    let mut concentrated = TransitionGraph::new();
    for _ in 0..4 {
        balanced
            .observe("probe", 0, TransitionOutcome::Resolved(0))
            .unwrap();
        balanced
            .observe("probe", 0, TransitionOutcome::Resolved(1))
            .unwrap();
    }
    for _ in 0..8 {
        concentrated
            .observe("probe", 0, TransitionOutcome::Resolved(0))
            .unwrap();
    }
    concentrated.register_node(1).unwrap();

    let balanced = balanced
        .dirichlet_information("probe", 0, 0.5)
        .unwrap()
        .unwrap();
    let concentrated = concentrated
        .dirichlet_information("probe", 0, 0.5)
        .unwrap()
        .unwrap();

    assert_eq!(
        balanced.total_concentration(),
        concentrated.total_concentration()
    );
    assert!(balanced.marginal_risk_reduction(0) > concentrated.marginal_risk_reduction(0));
}
