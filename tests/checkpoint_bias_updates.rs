use anneal_core::bias::{BasinBias, SortedPairs};
use anneal_core::methods::cluster_hopping::{
    BiasUpdate, ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, Outcome,
    run_with_bias_at_checkpoints,
};
use ndarray::{Array1, ArrayView1, array};
use rand::{SeedableRng, rngs::StdRng};

fn start() -> Array1<f64> {
    array![
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0
    ]
}

fn centre() -> Array1<f64> {
    Array1::from_elem(6, 1000.0)
}

fn merge() -> BiasUpdate {
    BiasUpdate::MergeWells {
        wells: vec![(centre(), 4.0)],
        weight: 0.5,
        complete: false,
    }
}

fn deposits() -> BiasUpdate {
    BiasUpdate::DepositDescriptors {
        deposits: vec![(centre(), 3)],
        weight: 1.0,
    }
}

fn retire() -> CheckpointAction {
    CheckpointAction::Retire {
        reason: "contract-boundary".into(),
    }
}

fn run(action: CheckpointAction) -> (Outcome, BasinBias<ClusterFingerprint>, Vec<Array1<f64>>) {
    let mut config = Config::for_cluster(4);
    config.screen_steps = 0;
    config.relax_steps = 0;
    config.screen_margin = f64::INFINITY;
    config.return_screen = false;
    config.temperature = 1.0;
    config.bias_height = 0.1;
    config.bias_gamma = 10.0;
    let mut bias = BasinBias::new(
        ClusterFingerprint::Spectrum(SortedPairs { n_points: 4 }),
        config.merge_radius,
        config.bias_height,
        config.bias_gamma,
    );
    let mut calls = Vec::new();
    let mut ledger = Ledger::new(128);
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _| {
        if ledger.charge() {
            calls.push(x.to_owned());
            (0.0, x.to_owned())
        } else {
            (f64::INFINITY, x.to_owned())
        }
    };
    let mut action = Some(action);
    let mut checkpoint = |_: ChainCheckpoint<'_>| action.take().unwrap_or_else(retire);
    let outcome = run_with_bias_at_checkpoints(
        &config,
        start().view(),
        &mut ledger,
        &mut relax,
        None,
        &mut bias,
        &mut StdRng::seed_from_u64(11),
        1,
        &mut checkpoint,
    );
    assert_eq!(ledger.spent(), calls.len());
    assert_eq!(outcome.charged, calls.len());
    assert!(calls.len() <= 128);
    assert_eq!(outcome.best, 0.0);
    assert_eq!(outcome.best_state.as_ref(), Some(&start()));
    (outcome, bias, calls)
}

fn assert_remote_bias(bias: &BasinBias<ClusterFingerprint>) {
    let index = bias.index();
    let remote = (0..index.n_basins())
        .find(|&i| index.centre(i) == centre())
        .expect("communicated descriptor is in the retained bias");
    assert_eq!(index.visits(remote), 0, "foreign visits must not echo");
    let mut expected: f64 = 2.0;
    for _ in 0..3 {
        expected += 0.1 * (-expected / 9.0).exp();
    }
    assert_eq!(bias.well_depth(remote), expected);
}

#[test]
fn ordered_updates_survive_a_coordinate_action_without_forces() {
    let proposal = start() * 1.2;
    let (outcome, bias, calls) = run(CheckpointAction::WithBiasUpdates {
        updates: vec![merge(), deposits()],
        action: Box::new(CheckpointAction::ExternalAdopt {
            state: proposal.clone(),
            action: "channel-adoption".into(),
            external_calls: 0,
        }),
    });
    assert_remote_bias(&bias);
    assert_eq!(outcome.gossip_rounds, 1);
    assert_eq!(outcome.shared_deposits, 3);
    assert!(calls.contains(&proposal));
    assert!(outcome.accepted_transitions.iter().any(|edge| {
        edge.action == "channel-adoption" && edge.adopted && edge.to_state == proposal
    }));
}

#[test]
fn retirement_applies_communication_without_another_objective_call() {
    let plain = run(retire());
    let (outcome, bias, calls) = run(CheckpointAction::WithBiasUpdates {
        updates: vec![merge(), deposits()],
        action: Box::new(retire()),
    });
    assert_remote_bias(&bias);
    assert_eq!(outcome.gossip_rounds, 1);
    assert_eq!(outcome.shared_deposits, 3);
    assert_eq!(outcome.hops, plain.0.hops);
    assert_eq!(calls, plain.2);
}

#[test]
fn nested_wrappers_preserve_update_order() {
    let (outcome, bias, _) = run(CheckpointAction::WithBiasUpdates {
        updates: vec![merge()],
        action: Box::new(CheckpointAction::WithBiasUpdates {
            updates: vec![deposits()],
            action: Box::new(retire()),
        }),
    });
    assert_remote_bias(&bias);
    assert_eq!(outcome.gossip_rounds, 1);
    assert_eq!(outcome.shared_deposits, 3);
}

#[test]
fn wrapping_a_standalone_update_preserves_its_effect_and_work() {
    let standalone = run(CheckpointAction::DepositDescriptors {
        deposits: vec![(centre(), 3)],
        weight: 1.0,
    });
    let wrapped = run(CheckpointAction::WithBiasUpdates {
        updates: vec![deposits()],
        action: Box::new(CheckpointAction::Continue),
    });
    assert_eq!(standalone.0.shared_deposits, wrapped.0.shared_deposits);
    assert_eq!(standalone.0.gossip_rounds, wrapped.0.gossip_rounds);
    assert_eq!(standalone.1.wells(), wrapped.1.wells());
    assert_eq!(standalone.2, wrapped.2);
}

#[test]
fn an_empty_update_wrapper_preserves_the_control() {
    let plain = run(retire());
    let wrapped = run(CheckpointAction::WithBiasUpdates {
        updates: Vec::new(),
        action: Box::new(retire()),
    });
    assert_eq!(plain.0.hops, wrapped.0.hops);
    assert_eq!(plain.0.shared_deposits, wrapped.0.shared_deposits);
    assert_eq!(plain.0.gossip_rounds, wrapped.0.gossip_rounds);
    assert_eq!(plain.1.wells(), wrapped.1.wells());
    assert_eq!(plain.2, wrapped.2);
}
