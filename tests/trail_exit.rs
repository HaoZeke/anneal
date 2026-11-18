use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ClusterFingerprint, Config, ContinuousSymmetry, Ledger, random_cluster, run_with_bias,
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
    )
    .expect("finite start");

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

/// Every quench is the origin, so the return screen never records an
/// entry. A drawn trail then cannot move the chain. With the restart
/// also armed, every stall has to be served by the restart: the trail
/// is redrawn, not left as an unscored draw.
#[test]
fn a_trail_without_an_entry_never_serves_the_stall() {
    let mut cfg = stalling_config(true);
    cfg.return_screen = true;
    cfg.restart_on_stall = true;
    cfg.escape_stall_patience = 1;
    cfg.bayes_screen = false;
    cfg.calibrated_screen = false;
    cfg.screen_steps = 1;
    cfg.relax_steps = 1;
    cfg.tabu_on_stall = false;
    cfg.energy_lattice = false;
    cfg.prior_soap = false;
    cfg.continuous_symmetry = ContinuousSymmetry::Off;
    cfg.surfaces.clear();
    let mut rng = StdRng::seed_from_u64(0x7a11);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(400);
    let mut bias = fresh_bias(&cfg);
    let mut relax = |ledger: &mut Ledger, state: ArrayView1<f64>, steps: usize| {
        for _ in 0..steps {
            if !ledger.charge() {
                break;
            }
        }
        (0.0, Array1::zeros(state.len()))
    };
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
        "a trail with no entry served {} stalls (restarts {}, hops {})",
        outcome.trail_escapes, outcome.restarts, outcome.hops
    );
    assert!(
        outcome.restarts > 0,
        "no stall was served (hops {})",
        outcome.hops
    );
    assert_eq!(
        outcome.hops,
        outcome.restarts * 2,
        "idle trail draws still consumed stalls: hops {} restarts {}",
        outcome.hops,
        outcome.restarts
    );
}

#[test]
fn a_redrawn_stall_that_exhausts_budget_keeps_finite_final_energy() {
    let mut cfg = stalling_config(true);
    cfg.return_screen = true;
    cfg.restart_on_stall = true;
    cfg.escape_stall_patience = 1;
    cfg.bayes_screen = false;
    cfg.calibrated_screen = false;
    cfg.screen_steps = 1;
    cfg.relax_steps = 1;
    cfg.tabu_on_stall = false;
    cfg.energy_lattice = false;
    cfg.prior_soap = false;
    cfg.continuous_symmetry = ContinuousSymmetry::Off;
    cfg.surfaces.clear();
    for budget in 10..=24 {
        let mut rng = StdRng::seed_from_u64(0x7a11);
        let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(budget);
        let mut bias = fresh_bias(&cfg);
        let mut relax = |ledger: &mut Ledger, state: ArrayView1<f64>, steps: usize| {
            let mut charged = 0usize;
            for _ in 0..steps {
                if !ledger.charge() {
                    break;
                }
                charged += 1;
            }
            if charged == 0 {
                return (f64::INFINITY, state.to_owned());
            }
            (0.0, Array1::zeros(state.len()))
        };
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
        assert!(outcome.best.is_finite(), "budget {budget}");
        assert!(
            outcome.final_energy.is_finite(),
            "budget {budget} gave non-finite final_energy {}",
            outcome.final_energy
        );
        let final_state = outcome.final_state.as_ref().expect("final_state");
        assert!(final_state.iter().all(|v| v.is_finite()));
        assert!(
            outcome
                .accepted_transitions
                .iter()
                .all(|edge| edge.to_energy.is_finite()),
            "budget {budget} recorded non-finite transition"
        );
    }
}

#[test]
fn a_redrawn_climb_that_exhausts_budget_keeps_finite_final_energy() {
    anneal_core::install_rgpot_kernels!();
    use anneal_core::methods::cluster_search::search_from;
    use anneal_core::potentials::PairPotential;
    use eindir_core::Objective;

    let pair = PairPotential::lennard_jones(6);
    let mut cfg = Config::recommended(6);
    cfg.escape_stall_patience = 1;
    for seed in [2u64, 3, 4] {
        let mut rng = StdRng::seed_from_u64(seed);
        let start = anneal_core::methods::cluster_hopping::random_cluster_in_radius(
            6,
            cfg.start_radius(),
            cfg.min_separation,
            &mut rng,
        );
        let mut ledger = Ledger::new(500);
        let (out, _) =
            search_from(&pair, &cfg, &mut ledger, start.view(), seed).expect("finite start");
        assert!(out.best.is_finite(), "seed {seed}");
        assert!(
            out.final_energy.is_finite(),
            "seed {seed} gave non-finite final_energy {}",
            out.final_energy
        );
        let final_state = out.final_state.as_ref().expect("final_state");
        assert!(final_state.iter().all(|v| v.is_finite()));
        let eval_final = pair.eval(final_state.view());
        assert!(
            (eval_final - out.final_energy).abs() < 1e-6,
            "seed {seed}: final_energy {} != eval(final_state) {eval_final}",
            out.final_energy
        );
    }
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
