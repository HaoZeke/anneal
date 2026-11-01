//! A hop cap bounds every hop the chain takes.
//!
//! Delayed acceptance rejects a trial at its first stage and moves on without
//! paying the quench. That rejection is still a hop, so a run whose cap falls
//! on one has to stop there like any other. So does a move that takes a
//! second hop inside one iteration, and each hop the climb arm takes.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, optimize_with_gradient, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};

fn lj_run(cfg: &Config, seed: u64) -> Outcome {
    let pot = PairPotential::lennard_jones(cfg.n_points);
    let mut ledger = Ledger::new(1_000_000);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(pot.value_and_gradient(v))
            },
            |_, _| true,
        );
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
        if !led.charge() {
            return None;
        }
        Some(pot.value_and_gradient(x).1)
    };
    optimize_with_gradient(cfg, &mut ledger, &mut relax, Some(&mut grad), seed)
        .expect("finite start")
}

#[test]
fn a_stage_one_reject_counts_against_the_hop_cap() {
    let mut cfg = Config::for_cluster(13);
    cfg.delayed_acceptance = true;
    // A surrogate that never abstains decides every hop after its warmup, so
    // the first stage rejects often enough for a cap to fall on a rejection.
    cfg.surrogate_tolerance = f64::INFINITY;
    let mut stage_one_rejects = 0;
    // The surrogate abstains through its first 64 quenches, so the caps start
    // past that.
    for cap in 80..=140 {
        cfg.max_hops = Some(cap);
        let out = lj_run(&cfg, 7);
        assert!(
            out.hops <= cap,
            "{} hops ran under a cap of {cap}",
            out.hops
        );
        stage_one_rejects += out.delayed.map_or(0, |d| d.1);
    }
    assert!(
        stage_one_rejects > 0,
        "the first stage never rejected, so the cap was not tested against it"
    );
}

/// The stall moves of the recommended stack take their own hop after the
/// quench, inside the same iteration, so the cap is checked before each one.
#[test]
fn a_stall_move_after_the_last_hop_is_not_taken() {
    let mut cfg = Config::recommended(13);
    for cap in 80..=140 {
        cfg.max_hops = Some(cap);
        let out = lj_run(&cfg, 7);
        assert!(
            out.hops <= cap,
            "{} hops ran under a cap of {cap}",
            out.hops
        );
    }
}

/// A chain on a line that reaches the climb arm on its second hop.
///
/// The stall patience is one hop and every other move that takes a second
/// hop is off, so the climb is the only arm that can add hops.
fn climb_run(
    n: usize,
    mut relax: impl FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + Send,
) -> Outcome {
    let mut start = Array1::zeros(3 * n);
    for i in 0..n {
        start[3 * i] = i as f64 * 1.3;
    }
    let mut cfg = Config::for_cluster(n);
    cfg.max_hops = Some(3);
    cfg.escape_on_stall = true;
    cfg.restart_on_stall = false;
    cfg.trail_on_stall = false;
    cfg.budget_window = true;
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
    cfg.escape_lanczos_steps = 4;
    cfg.polish_records = 0;
    cfg.symmetrise_on_stall = false;
    cfg.jump_on_stall = false;
    cfg.point_symmetrise_on_new = false;
    cfg.orbit_complete_on_new = false;
    cfg.allocate_moves = false;
    let mut grad = |_: &mut Ledger, x: ArrayView1<f64>| Some(Array1::zeros(x.len()));
    let mut ledger = Ledger::new(100_000);
    let mut rng = StdRng::seed_from_u64(7);
    run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start")
}

/// The climb arm grows the lattice orders and then quenches an activation,
/// and each of those is a hop. Two hops into a cap of three, a growth that
/// deepens the incumbent takes the last one and the activation waits.
#[test]
fn a_climb_with_one_hop_left_takes_one_hop() {
    // The opening quench and two ordinary hops stay on the incumbent, so
    // every later quench is a growth or an activation that deepens it.
    let mut calls = 0usize;
    let out = climb_run(6, |ledger, x, _| {
        let _ = ledger.charge();
        let energy = if calls < 5 {
            -1.0
        } else {
            -1000.0 - calls as f64
        };
        calls += 1;
        (energy, x.to_owned())
    });
    assert!(out.hops <= 3, "{} hops ran under a cap of 3", out.hops);
    assert!(out.best < -1000.0, "the climb arm did not grow the lattice");
}

/// A growth that adopts an icosahedron opens the icosahedral growth in the
/// same climb, which is a second hop before the activation.
#[test]
fn an_icosahedral_growth_with_one_hop_left_takes_one_hop() {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let radius = 1.1 / (1.0 + phi * phi).sqrt();
    let mut ico = vec![0.0; 3];
    for (a, b) in [(1.0, phi), (1.0, -phi), (-1.0, phi), (-1.0, -phi)] {
        for vertex in [[0.0, a, b], [a, b, 0.0], [b, 0.0, a]] {
            ico.extend(vertex.iter().map(|c| c * radius));
        }
    }
    let ico = Array1::from(ico);
    // Quenches 5 to 9 are the first growth, in library order. The fourth of
    // them is the icosahedral source, and it wins. Every later quench is
    // deeper still and returns the same icosahedron.
    let mut calls = 0usize;
    let out = climb_run(13, |ledger, x, _| {
        let _ = ledger.charge();
        let (energy, state) = match calls {
            0..5 => (-1.0, x.to_owned()),
            8 => (-5000.0, ico.clone()),
            5..10 => (-1000.0 - calls as f64, x.to_owned()),
            _ => (-6000.0 - calls as f64, ico.clone()),
        };
        calls += 1;
        (energy, state)
    });
    assert!(out.hops <= 3, "{} hops ran under a cap of 3", out.hops);
    assert_eq!(out.best, -5000.0, "the icosahedron did not win the growth");
}
