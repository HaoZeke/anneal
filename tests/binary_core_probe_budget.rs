//! The labelled binary-core probe leaves the walk its share of the budget.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::cluster_hopping::{Config, Ledger, run};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

const BUDGET: usize = 4_000;

fn quiet(n: usize) -> Config {
    let mut cfg = Config::for_cluster(n);
    cfg.replicas = 1;
    cfg.angular_moves = false;
    cfg.soft_perturb = false;
    cfg.cov_perturb = false;
    cfg.hmc = None;
    cfg.two_phase = None;
    cfg.surfaces.clear();
    cfg.return_screen = false;
    cfg.bayes_screen = false;
    cfg.screen_margin = f64::INFINITY;
    cfg.screen_steps = 2;
    cfg.relax_steps = 3;
    cfg.polish_records = 0;
    cfg.restart_on_stall = false;
    cfg.escape_on_stall = false;
    cfg.trail_on_stall = false;
    cfg.jump_on_stall = false;
    cfg.path_on_stall = false;
    cfg.symmetrise_on_stall = false;
    cfg.point_symmetrise_on_new = false;
    cfg.orbit_complete_on_new = false;
    cfg.superbasin_escape = false;
    cfg
}

fn tetrahedron() -> Array1<f64> {
    array![
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0
    ]
}

/// The budget of the first side ledger, with the calls the chain's own
/// ledger had spent when it opened.
fn first_side_ledger() -> (usize, usize) {
    let mut cfg = quiet(4);
    cfg.species = Some(vec![29, 29, 79, 79]);
    let chain_spent = AtomicUsize::new(0);
    let side = Mutex::new(None);
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        if ledger.budget() == BUDGET {
            let _ = ledger.charge();
            chain_spent.store(ledger.spent(), Ordering::Relaxed);
        } else {
            side.lock()
                .unwrap()
                .get_or_insert((ledger.budget(), chain_spent.load(Ordering::Relaxed)));
            let _ = ledger.charge();
        }
        (0.0, x.to_owned())
    };
    let mut ledger = Ledger::new(BUDGET);
    let mut rng = StdRng::seed_from_u64(5);
    run(
        &cfg,
        tetrahedron().view(),
        &mut ledger,
        &mut relax,
        &mut rng,
    )
    .expect("finite start");
    side.into_inner()
        .unwrap()
        .expect("the labelled run opens the binary-core probe")
}

#[test]
#[cfg(feature = "rgsaddle")]
fn a_feature_build_leaves_the_walk_half_of_what_is_left() {
    let (probe, spent) = first_side_ledger();
    let left = BUDGET - spent;
    assert!(probe <= left.div_ceil(2), "probe {probe} of {left} left");
    assert!(probe > 0);
}

#[test]
#[cfg(not(feature = "rgsaddle"))]
fn a_default_build_offers_the_probe_what_is_left() {
    let (probe, spent) = first_side_ledger();
    assert_eq!(probe, BUDGET - spent);
}
