use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, random_cluster,
    run_with_bias, run_with_bias_at_checkpoints,
};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn toy_relax(
    ledger: &mut Ledger,
    state: ArrayView1<f64>,
    steps: usize,
) -> (f64, Array1<f64>) {
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
    let start = random_cluster(
        cfg.n_points,
        0.7,
        cfg.min_separation,
        &mut seeding_rng,
    );
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

    assert!(checkpoints.len() > 1, "checkpoint hook did not observe the run");
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
