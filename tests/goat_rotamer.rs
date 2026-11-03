//! GOAT's energy and rotational-constant test as a basin identity on the
//! cluster hop.

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, Outcome, run};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn bare() -> Config {
    let mut cfg = Config::for_cluster(3);
    cfg.move_library = MoveLibrary::WalesDoye;
    cfg.angular_moves = false;
    cfg.return_screen = false;
    cfg.bayes_screen = false;
    cfg.calibrated_screen = false;
    cfg.screen_margin = f64::INFINITY;
    cfg.screen_steps = 1;
    cfg.relax_steps = 1;
    cfg.polish_records = 0;
    cfg.orbit_complete_on_new = false;
    cfg.surfaces.clear();
    cfg.merge_radius = 1e-6;
    cfg.max_hops = Some(12);
    cfg
}

fn triangle(scale: f64) -> Array1<f64> {
    array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] * scale
}

/// Every relaxation returns the start swollen by a further 1e-5 and raised
/// by a further 1e-3. Each is a distinct descriptor at this merge radius,
/// and each is within the energy window and the rotational-constant
/// tolerance of the start.
fn swelling(cfg: &Config) -> Outcome {
    let mut calls = 0usize;
    let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, _: usize| {
        let _ = ledger.charge();
        calls += 1;
        let k = calls as f64;
        (1.0e-3 * k, triangle(1.0 + 1.0e-5 * k))
    };
    let mut rng = StdRng::seed_from_u64(5);
    let mut ledger = Ledger::new(10_000);
    run(cfg, triangle(1.0).view(), &mut ledger, &mut relax, &mut rng).expect("finite start")
}

#[test]
fn rotamers_of_the_start_stay_in_its_basin() {
    let mut cfg = bare();
    cfg.rotamer_test = true;
    let out = swelling(&cfg);
    assert_eq!(out.basins, 1, "{} basins", out.basins);
    assert!(out.rotamers > 0);
}

/// The start relaxes to itself. Odd hops then reach a new minimum far above
/// it, which the controller registers and refuses, and even hops return
/// that minimum swollen by 1e-5 and raised by 5e-4. That is a rotamer of a
/// basin the chain never stood in, so only the controller's identity can
/// place it. The screen previews the quench of its hop.
fn refused_rotamers(cfg: &Config) -> Outcome {
    let mut quenches = 0usize;
    let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, iters: usize| {
        let _ = ledger.charge();
        let hop = if iters == cfg.relax_steps {
            quenches += 1;
            quenches - 1
        } else {
            quenches
        };
        if hop == 0 {
            return (0.0, triangle(1.0));
        }
        let j = hop.div_ceil(2) as f64;
        let scale = 2.0 + 0.1 * j;
        if hop % 2 == 1 {
            (100.0 + j, triangle(scale))
        } else {
            (100.0 + j + 5.0e-4, triangle(scale * (1.0 + 1.0e-5)))
        }
    };
    let mut rng = StdRng::seed_from_u64(5);
    let mut ledger = Ledger::new(10_000);
    run(cfg, triangle(1.0).view(), &mut ledger, &mut relax, &mut rng).expect("finite start")
}

#[test]
fn the_controller_counts_the_rotamers_it_places() {
    let mut cfg = bare();
    cfg.relax_steps = 2;
    cfg.minima_hopping = true;
    cfg.rotamer_test = true;
    let out = refused_rotamers(&cfg);
    assert!(out.rotamers > 0, "{} rotamers", out.rotamers);
    cfg.rotamer_test = false;
    assert_eq!(refused_rotamers(&cfg).rotamers, 0);
}

#[test]
fn without_the_test_each_swelling_is_its_own_basin() {
    let out = swelling(&bare());
    assert!(out.basins > 1, "{} basins", out.basins);
    assert_eq!(out.rotamers, 0);
}

#[test]
fn the_test_is_off_by_default_and_keeps_the_digest() {
    let cfg = Config::for_cluster(13);
    assert!(!cfg.rotamer_test);
    assert!(!cfg.resolved_json().unwrap().contains("rotamer_test"));
}
