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
fn unresolved_probes_do_not_certify_shared_return_dynamics() {
    let mut graph = TransitionGraph::new();
    for source in [0, 1] {
        for _ in 0..16 {
            graph
                .observe("probe", source, TransitionOutcome::Unresolved)
                .unwrap();
        }
    }
    let regions = graph
        .attraction_regions(&AttractionRegionConfig {
            probe_action: "probe".into(),
            concentration: 0.1,
            diffusion_steps: 2,
            maximum_distance: 0.1,
            minimum_probes: 8,
        })
        .unwrap();
    assert_eq!(regions, vec![vec![0], vec![1]]);
}
