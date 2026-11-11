// Copyright (c) anneal developers
//! A failed stall growth used to quench one extra lattice and leave the
//! same minimum in place. During that hold every proposal is an all-points
//! displacement, and growth waits until the incumbent drops.

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, run_with_gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn line(n: usize) -> Array1<f64> {
    let mut start = Array1::zeros(3 * n);
    for i in 0..n {
        start[3 * i] = i as f64 * 1.3;
    }
    start
}

#[test]
fn a_held_stall_on_a_line_accepts_an_all_points_step() {
    let home = line(13);
    let mut cfg = Config::for_cluster(13);
    cfg.max_hops = Some(5);
    cfg.escape_on_stall = true;
    cfg.restart_on_stall = false;
    cfg.trail_on_stall = false;
    cfg.budget_window = false;
    cfg.restart_patience = 1;
    cfg.bayes_screen = false;
    cfg.calibrated_screen = false;
    cfg.finish_improving = false;
    cfg.return_screen = false;
    cfg.screen_margin = 1.0e9;
    cfg.screen_steps = 1;
    cfg.relax_steps = 1;
    cfg.temperature = 1.0e-4;
    cfg.escape_max_climb = 1;
    cfg.polish_records = 0;
    cfg.symmetrise_on_stall = false;
    cfg.jump_on_stall = false;
    cfg.point_symmetrise_on_new = false;
    cfg.orbit_complete_on_new = false;
    cfg.allocate_moves = false;
    // LeanBurst has no all-points kernel. An `all` acceptance can come
    // only from the stall hold.
    cfg.move_library = MoveLibrary::LeanBurst;
    let mut relax = |ledger: &mut Ledger, _x: ArrayView1<f64>, _steps: usize| {
        let _ = ledger.charge();
        (-1.0, home.clone())
    };
    let mut ledger = Ledger::new(20_000);
    let mut rng = StdRng::seed_from_u64(7);
    let outcome = run_with_gradient(&cfg, home.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    let all = outcome
        .accepted_transitions
        .iter()
        .filter(|step| step.action == "all")
        .count();
    assert!(
        all >= 1,
        "the hold never emitted an all-points acceptance ({all} of {})",
        outcome.accepted_transitions.len()
    );
}
