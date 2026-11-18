use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ClusterFingerprint, Config, ContinuousSymmetry, Ledger, random_cluster, run_with_bias,
    run_with_gradient,
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
    // The toy relaxer charges every step, and the 3000-call ledger is sized
    // for 200-step quenches.
    cfg.relax_steps = 200;
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

fn squared_distance(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

#[test]
fn a_stalled_chain_does_not_count_a_return_to_its_shelf_as_a_trail_exit() {
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
    )
    .expect("finite start");

    assert_eq!(
        outcome.trail_escapes, 0,
        "the one-well landscape cannot produce a genuine trail exit"
    );
    assert!(
        outcome.tabu.1 >= 1,
        "trail returns to the one well were not refused as shelf hits"
    );
    assert!(outcome.best.is_finite());
    assert_eq!(outcome.stall_escapes, 0);
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
    )
    .expect("finite start");

    assert_eq!(outcome.trail_escapes, 0);
}

#[test]
fn a_stalled_chain_leaves_its_basin_via_the_trail_arm() {
    let n = 4;
    let start = Array1::from(vec![
        0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.6, 1.0, 0.0, 0.6, 0.4, 0.9,
    ]);
    let shelf = Array1::from(vec![
        0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.6, 1.0, 0.0, 0.6, 0.4, 0.9,
    ]);
    let escaped = Array1::from(vec![
        2.2, 0.1, 0.0, 3.0, 0.1, 0.0, 2.5, 1.1, 0.2, 2.5, 0.5, 1.0,
    ]);
    let mut cfg = Config::recommended(n);
    cfg.restart_patience = 1_000_000;
    cfg.escape_stall_patience = 2;
    cfg.escape_stall_factor = 0.0;
    cfg.escape_on_stall = false;
    cfg.trail_on_stall = true;
    cfg.restart_on_stall = false;
    cfg.relax_steps = 80;
    cfg.screen_steps = 1;
    cfg.max_hops = Some(10);
    cfg.continuous_symmetry = ContinuousSymmetry::Off;
    cfg.orbit_complete_on_new = false;
    cfg.energy_lattice = false;
    cfg.surfaces.clear();
    cfg.symmetrise_on_stall = false;
    cfg.return_screen = false;

    let mut call = 0usize;
    let mut rng = StdRng::seed_from_u64(2000);
    let mut ledger = Ledger::new(20_000);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        let before = led.spent();
        if steps > 1 {
            call += 1;
        }
        let (well, offset) = if call == 1 && steps > 1 {
            (&start, -0.5)
        } else if call >= 4 && steps > 1 {
            (&escaped, -0.7)
        } else {
            (&shelf, -1.0)
        };
        let mut cur = x.to_owned();
        for _ in 0..steps {
            if !led.charge() {
                break;
            }
            cur = well + &((&cur - well) * 0.85);
        }
        let d = &cur - well;
        let e = offset;
        if steps > 1 {
            led.record_quench_boundary(before, e, cur.clone(), Some(2.0 * d));
        }
        (e, cur)
    };

    let outcome = run_with_gradient(&cfg, start.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");

    assert!(
        outcome.trail_escapes >= 1,
        "the stalled chain never exited through trail: hops={} stop={:?} call={} tabu={:?} transitions={:?}",
        outcome.hops,
        outcome.stop,
        call,
        outcome.tabu,
        outcome
            .accepted_transitions
            .iter()
            .map(|step| (&step.action, step.adopted))
            .collect::<Vec<_>>()
    );
    assert!(
        outcome
            .accepted_transitions
            .iter()
            .any(|transition| transition.action == "stall-trail" && transition.adopted),
        "no adopted trail leave was recorded"
    );
    let final_state = outcome.final_state.as_ref().expect("final state");
    assert!(
        squared_distance(final_state.view(), escaped.view())
            < squared_distance(final_state.view(), shelf.view()),
        "the chain did not finish in a basin different from the stalled shelf"
    );
}
