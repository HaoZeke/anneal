// Copyright (c) anneal developers
//! A flat stall keeps drawing only from the configured general move library.

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
fn a_flat_stall_uses_only_general_lean_burst_actions() {
    let home = line(13);
    let mut cfg = Config::for_cluster(13);
    cfg.max_hops = Some(5);
    cfg.escape_on_stall = false;
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
    cfg.jump_on_stall = false;
    cfg.allocate_moves = false;
    cfg.move_library = MoveLibrary::LeanBurst;
    let mut relax = |ledger: &mut Ledger, _x: ArrayView1<f64>, _steps: usize| {
        let _ = ledger.charge();
        (-1.0, home.clone())
    };
    let mut ledger = Ledger::new(20_000);
    let mut rng = StdRng::seed_from_u64(7);
    let outcome = run_with_gradient(&cfg, home.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    assert_eq!(outcome.hops, 5);
    let actions: Vec<&str> = outcome
        .accepted_transitions
        .iter()
        .map(|step| step.action.as_str())
        .collect();
    assert!(
        actions
            .iter()
            .all(|action| matches!(*action, "surface" | "single" | "burst" | "sym" | "soap")),
        "{actions:?}"
    );
}
