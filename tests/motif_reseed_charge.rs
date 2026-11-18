//! Every objective call in a labelled search is charged to the caller.

use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::cluster_hopping::{Config, Ledger, run};
use ndarray::{Array1, ArrayView1, array};
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

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

/// Runs a labelled chain on a flat surface and returns the charged count
/// with the number of calls any ledger accepted.
fn flat_labelled(budget: usize, seed: u64) -> (usize, usize) {
    let mut cfg = quiet(4);
    cfg.species = Some(vec![29, 29, 79, 79]);
    let accepted = AtomicUsize::new(0);
    // Nothing quenches below the opening zero, so the incumbent stands for
    // the whole run.
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        if ledger.charge() {
            accepted.fetch_add(1, Ordering::Relaxed);
        }
        (0.0, x.to_owned())
    };
    let mut ledger = Ledger::new(budget);
    let mut rng = StdRng::seed_from_u64(seed);
    let out = run(
        &cfg,
        tetrahedron().view(),
        &mut ledger,
        &mut relax,
        &mut rng,
    )
    .expect("finite start");
    (out.charged, accepted.load(Ordering::Relaxed))
}

#[test]
fn every_call_on_a_flat_labelled_surface_is_charged() {
    let (charged, accepted) = flat_labelled(400, 5);
    assert_eq!(charged, accepted, "probe calls went uncharged");
    assert!(charged <= 400);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn every_accepted_call_of_a_labelled_run_is_charged(
        budget in 8usize..600,
        seed in any::<u64>(),
    ) {
        let (charged, accepted) = flat_labelled(budget, seed);
        prop_assert_eq!(charged, accepted);
        prop_assert!(charged <= budget);
    }
}
