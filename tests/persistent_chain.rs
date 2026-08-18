use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, random_cluster,
    run_with_bias, run_with_bias_at_checkpoints, run_with_gradient,
};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn toy_relax(ledger: &mut Ledger, state: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
    let mut relaxed = state.to_owned();
    for _ in 0..steps {
        if !ledger.charge() {
            break;
        }
        relaxed.mapv_inplace(|value| value * 0.85);
    }
    let energy = relaxed.iter().map(|value| value * value).sum();
    (energy, relaxed)
}

fn fresh_bias(cfg: &Config) -> BasinBias<ClusterFingerprint> {
    BasinBias::new(
        ClusterFingerprint::for_keying(cfg.n_points, cfg.shape_keyed),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    )
}

#[test]
fn no_action_checkpoints_are_identical_to_an_uninterrupted_run() {
    let cfg = Config::recommended(6);
    let mut seeding_rng = StdRng::seed_from_u64(0x5eed);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut seeding_rng);
    let mut uninterrupted_rng = seeding_rng.clone();
    let mut checkpointed_rng = seeding_rng;
    let mut uninterrupted_ledger = Ledger::new(4_000);
    let mut checkpointed_ledger = Ledger::new(4_000);
    let mut uninterrupted_bias = fresh_bias(&cfg);
    let mut checkpointed_bias = fresh_bias(&cfg);
    let mut uninterrupted_relax = toy_relax;
    let mut checkpointed_relax = toy_relax;

    let uninterrupted = run_with_bias(
        &cfg,
        start.view(),
        &mut uninterrupted_ledger,
        &mut uninterrupted_relax,
        None,
        &mut uninterrupted_bias,
        &mut uninterrupted_rng,
    );

    let mut checkpoints = Vec::new();
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        checkpoints.push((snapshot.charged(), snapshot.hops()));
        CheckpointAction::Continue
    };
    let checkpointed = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut checkpointed_ledger,
        &mut checkpointed_relax,
        None,
        &mut checkpointed_bias,
        &mut checkpointed_rng,
        211,
        &mut checkpoint,
    );

    assert!(
        checkpoints.len() > 1,
        "checkpoint hook did not observe the run"
    );
    assert!(
        checkpoints.windows(2).all(|pair| pair[0] < pair[1]),
        "checkpoint counters are not strictly monotone: {checkpoints:?}"
    );
    assert_eq!(
        format!("{uninterrupted:#?}"),
        format!("{checkpointed:#?}"),
        "a no-action checkpoint changed reported scientific state"
    );
    assert_eq!(uninterrupted_ledger.spent(), checkpointed_ledger.spent());
    assert_eq!(
        uninterrupted_ledger.quench_boundaries().len(),
        checkpointed_ledger.quench_boundaries().len()
    );
    assert_eq!(uninterrupted_bias.n_basins(), checkpointed_bias.n_basins());
    assert_eq!(
        uninterrupted_rng.random::<u64>(),
        checkpointed_rng.random::<u64>(),
        "checkpoint handling consumed the local random stream"
    );
}

#[test]
fn checkpoint_boundary_proposal_is_quenched_and_chain_continues() {
    let mut cfg = Config::recommended(6);
    cfg.relax_steps = 1;
    let mut rng = StdRng::seed_from_u64(0xb0_0d_a7);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(4_000);
    let mut bias = fresh_bias(&cfg);
    let mut relax = toy_relax;
    let proposed = start.mapv(|value| 1.1 * value);
    let mut offered = false;
    let mut observed_after_transport = false;
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        if !offered {
            offered = true;
            return CheckpointAction::BoundaryProposal {
                state: proposed.clone(),
                action: "test-boundary".to_string(),
            };
        }
        observed_after_transport |= snapshot
            .accepted_transitions()
            .iter()
            .any(|transition| transition.action == "test-boundary");
        CheckpointAction::Continue
    };

    let outcome = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        None,
        &mut bias,
        &mut rng,
        211,
        &mut checkpoint,
    );

    assert!(offered, "checkpoint did not offer the boundary proposal");
    assert!(
        observed_after_transport,
        "a later checkpoint did not observe the transported edge"
    );
    assert!(
        outcome
            .accepted_transitions
            .iter()
            .any(|transition| transition.action == "test-boundary"),
        "the accepted boundary proposal is absent from the trajectory"
    );
    assert!(outcome.hops > 1, "the chain stopped at the transport");
}

#[test]
fn checkpoint_probe_is_recorded_without_becoming_the_live_chain() {
    let mut cfg = Config::recommended(6);
    cfg.relax_steps = 200;
    let mut rng = StdRng::seed_from_u64(0xfeed_600d);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(260);
    let mut bias = fresh_bias(&cfg);
    let mut relax = toy_relax;
    let proposed = start.mapv(|value| value + 4.0);
    let mut occupied_before_probe = None;
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        if occupied_before_probe.is_none() {
            occupied_before_probe = Some(snapshot.current_state().to_owned());
            return CheckpointAction::ProbeProposal {
                state: proposed.clone(),
                action: "probe".to_string(),
            };
        }
        CheckpointAction::Continue
    };

    let outcome = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        None,
        &mut bias,
        &mut rng,
        40,
        &mut checkpoint,
    );

    let probe = outcome
        .accepted_transitions
        .iter()
        .find(|transition| transition.action == "probe")
        .expect("probe transition is absent from the trajectory evidence");
    assert!(!probe.adopted, "a diagnostic probe became a live-chain hop");
    assert_eq!(
        outcome.final_state.as_ref(),
        occupied_before_probe.as_ref(),
        "a non-adopting probe replaced the occupied chain state"
    );
}

#[test]
fn unvalidated_screen_result_never_becomes_the_live_chain_state() {
    let mut cfg = Config::for_cluster(2);
    cfg.max_hops = Some(1);
    cfg.screen_margin = -100.0;
    cfg.return_screen = false;
    let start = Array1::from(vec![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0]);
    let screened = &start + 0.25;
    let mut ledger = Ledger::new(100);
    let mut rng = StdRng::seed_from_u64(0x5c_4e_e7);
    let mut relax_calls = 0usize;
    let mut relax = |ledger: &mut Ledger, _state: ArrayView1<f64>, steps: usize| {
        assert!(ledger.charge());
        relax_calls += 1;
        if relax_calls == 1 {
            assert_eq!(steps, cfg.relax_steps);
            (0.0, start.clone())
        } else {
            assert_eq!(steps, cfg.screen_steps);
            (-10.0, screened.clone())
        }
    };
    let mut gradient = |ledger: &mut Ledger, state: ArrayView1<f64>| {
        assert!(ledger.charge());
        if state == start.view() {
            Some(Array1::zeros(state.len()))
        } else {
            Some(Array1::ones(state.len()))
        }
    };

    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut gradient),
        &mut rng,
    );

    // A screened trial still faces the acceptance law and may carry the
    // chain: that is how a short screen finds basins at all. What it may
    // never be is an answer. The screen returns -10.0 against the
    // quenched 0.0, so a ledger that recorded it would say so here.
    assert_eq!(outcome.best, 0.0);
    assert_eq!(outcome.best_state.as_ref(), Some(&start));
    // The trajectory keeps the hop and tags it, which is what the
    // `validated` field is for: a history of record improvements alone
    // cannot reconstruct the region the chain occupies.
    let [transition] = outcome.accepted_transitions.as_slice() else {
        panic!("the screened hop is the one state change this run makes")
    };
    assert!(transition.adopted);
    assert!(
        !transition.validated,
        "a partial relaxation was tagged as meeting the quench-validity contract"
    );
}

#[test]
fn occupancy_retire_stops_the_chain() {
    let cfg = Config::recommended(6);
    let mut rng = StdRng::seed_from_u64(0xd0_0e);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(4_000);
    let mut bias = fresh_bias(&cfg);
    let mut relax = toy_relax;
    let mut checkpoints = 0usize;
    let mut checkpoint = |_: ChainCheckpoint<'_>| {
        checkpoints += 1;
        if checkpoints == 1 {
            CheckpointAction::Retire {
                reason: "mixing".to_owned(),
            }
        } else {
            CheckpointAction::Continue
        }
    };

    let outcome = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        None,
        &mut bias,
        &mut rng,
        211,
        &mut checkpoint,
    );

    assert_eq!(checkpoints, 1, "retire kept invoking the checkpoint hook");
    assert!(
        ledger.remaining() > 0,
        "retire drained the budget instead of stopping"
    );
    assert!(
        outcome.hops < 50,
        "retire left the chain walking the full budget: hops {}",
        outcome.hops
    );
}
