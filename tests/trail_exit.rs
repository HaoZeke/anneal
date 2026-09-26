use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ClusterFingerprint, Config, Ledger, random_cluster, run_with_bias,
};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

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

fn stalling_config(trail: bool) -> Config {
    let mut cfg = Config::recommended(6);
    // The toy landscape contracts everything toward the origin, so
    // improvement dries up almost immediately and the stall trigger is
    // exercised early and often.
    // The toy landscape contracts every state toward the origin, so the
    // return screen would classify every accepted move as returning and no
    // entry would ever be recorded; the screen is not the mechanism under
    // test here.
    cfg.return_screen = false;
    cfg.escape_stall_patience = 4;
    cfg.escape_stall_factor = 0.0;
    cfg.escape_on_stall = false;
    cfg.restart_on_stall = false;
    cfg.symmetrise_on_stall = false;
    cfg.trail_on_stall = trail;
    cfg
}

#[test]
fn a_stalled_chain_leaves_through_its_recorded_entry() {
    let cfg = stalling_config(true);
    let mut rng = StdRng::seed_from_u64(0x7a11);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(3_000);
    let mut bias = fresh_bias(&cfg);
    let mut relax = toy_relax;

    let outcome = run_with_bias(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        None,
        &mut bias,
        &mut rng,
    );

    assert!(
        outcome.trail_escapes >= 1,
        "a chain that stalls with an entry on record must leave through it, \
         took {} trail exits",
        outcome.trail_escapes
    );
    assert!(outcome.best.is_finite());
    // Every trail exit is also a stall escape in the shared counter.
    assert!(outcome.stall_escapes >= outcome.trail_escapes);
}

#[test]
fn the_trail_exit_is_off_unless_asked_for() {
    let cfg = stalling_config(false);
    let mut rng = StdRng::seed_from_u64(0x7a11);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(3_000);
    let mut bias = fresh_bias(&cfg);
    let mut relax = toy_relax;

    let outcome = run_with_bias(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        None,
        &mut bias,
        &mut rng,
    );

    assert_eq!(outcome.trail_escapes, 0);
}
