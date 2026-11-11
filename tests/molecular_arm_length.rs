// Copyright (c) anneal developers
//! A per-hop rebuild can drop the cycle arm when the groups are no longer
//! three-atom waters. The allocator still has the original count. The
//! rebuilt list is installed only when its length matches that count, so
//! the next draw stays inside the list.

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, run_with_gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn a_short_molecular_rebuild_does_not_index_past_the_arm_list() {
    let n = 6;
    let mut cfg = Config::for_cluster(n);
    cfg.max_hops = Some(80);
    cfg.escape_on_stall = false;
    cfg.restart_on_stall = false;
    cfg.trail_on_stall = false;
    cfg.jump_on_stall = false;
    cfg.symmetrise_on_stall = false;
    cfg.point_symmetrise_on_new = false;
    cfg.orbit_complete_on_new = false;
    cfg.bayes_screen = false;
    cfg.calibrated_screen = false;
    cfg.finish_improving = false;
    cfg.return_screen = false;
    cfg.energy_lattice = false;
    cfg.contextual_moves = false;
    cfg.depth_reward = false;
    cfg.allocate_moves = true;
    cfg.angular_moves = false;
    cfg.displacement_only = false;
    cfg.move_library = MoveLibrary::Molecular {
        groups: vec![vec![0, 1, 2], vec![3, 4, 5]],
        reactive: false,
        reverse_cycles: true,
    };
    let mut start = Array1::zeros(3 * n);
    for i in 0..n {
        start[3 * i] = i as f64 * 10.0;
    }
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        let _ = ledger.charge();
        (-1.0, x.to_owned())
    };
    let mut ledger = Ledger::new(20_000);
    let mut rng = StdRng::seed_from_u64(0);
    let outcome = run_with_gradient(&cfg, start.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("the draw stayed inside the arm list");
    assert!(outcome.best.is_finite());
    assert!(ledger.spent() > 0, "the chain spent no charge");
}
