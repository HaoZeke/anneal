//! Every way a cluster hop run can end is reported in `Outcome::stop`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anneal_core::hmc::hop::HopConfig;
use anneal_core::hmc::metric::MetricKind;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, Config, ContinuousSymmetry, HopStop, Ledger, Outcome, run,
    run_with_energy_gradient, run_with_energy_gradient_at_checkpoints, run_with_gradient,
};
use ndarray::{Array1, ArrayView1, array, s};
use rand::SeedableRng;
use rand::rngs::StdRng;

const SCREEN: usize = 2;
const FULL: usize = 3;

/// A chain with no stall response, symmetry move, or extra proposal, so each
/// hop is one screen relaxation and one full relaxation.
fn bare(n: usize) -> Config {
    quiet(Config::for_cluster(n))
}

fn quiet(mut cfg: Config) -> Config {
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
    cfg.screen_steps = SCREEN;
    cfg.relax_steps = FULL;
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

fn triangle() -> Array1<f64> {
    array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// Returns the input at zero energy for one charge.
fn flat(ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize) -> (f64, Array1<f64>) {
    let _ = ledger.charge();
    (0.0, x.to_owned())
}

fn zero_gradient(ledger: &mut Ledger, x: ArrayView1<f64>) -> Option<Array1<f64>> {
    ledger.charge().then(|| Array1::zeros(x.len()))
}

fn zero_energy_gradient(ledger: &mut Ledger, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
    ledger.charge().then(|| (0.0, Array1::zeros(x.len())))
}

fn drain(ledger: &mut Ledger) {
    while ledger.charge() {}
}

fn flat_run(cfg: &Config, ledger: &mut Ledger) -> Outcome {
    let mut relax = flat;
    let mut rng = StdRng::seed_from_u64(5);
    run(cfg, triangle().view(), ledger, &mut relax, &mut rng).expect("finite start")
}

#[test]
fn a_stop_raised_before_the_opening_quench_is_a_caller_stop() {
    let mut ledger = Ledger::with_stop(32, Arc::new(AtomicBool::new(true)));
    let out = flat_run(&bare(3), &mut ledger);
    assert_eq!(out.stop, Some(HopStop::CallerStop));
    assert_eq!(out.charged, 0);
}

#[test]
fn a_stop_raised_inside_a_refused_opening_quench_is_a_caller_stop() {
    let stop = Arc::new(AtomicBool::new(false));
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        assert!(ledger.charge());
        stop.store(true, Ordering::Relaxed);
        (f64::NAN, x.to_owned())
    };
    let mut ledger = Ledger::with_stop(32, Arc::clone(&stop));
    let mut rng = StdRng::seed_from_u64(5);
    let out = run(
        &bare(3),
        triangle().view(),
        &mut ledger,
        &mut relax,
        &mut rng,
    )
    .expect("a caller stop is not a refused start");
    assert_eq!(out.stop, Some(HopStop::CallerStop));
    assert_eq!(out.charged, 1);
}

#[test]
fn an_opening_quench_at_the_target_ends_before_the_first_hop() {
    let mut ledger = Ledger::new(64);
    ledger.stop_below(1.0);
    let out = flat_run(&bare(3), &mut ledger);
    assert_eq!(out.stop, Some(HopStop::Target));
    assert_eq!(out.hops, 0);
}

#[test]
fn a_stop_raised_after_the_full_quench_ends_at_the_next_hop() {
    let stop = Arc::new(AtomicBool::new(false));
    let grads = AtomicUsize::new(0);
    let mut grad = |ledger: &mut Ledger, x: ArrayView1<f64>| {
        if grads.fetch_add(1, Ordering::Relaxed) == 1 {
            stop.store(true, Ordering::Relaxed);
        }
        zero_gradient(ledger, x)
    };
    let mut relax = flat;
    let mut ledger = Ledger::with_stop(256, Arc::clone(&stop));
    let mut rng = StdRng::seed_from_u64(5);
    let out = run_with_gradient(
        &bare(3),
        triangle().view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    assert_eq!(out.stop, Some(HopStop::CallerStop));
    assert_eq!(out.hops, 1);
}

#[test]
fn a_retiring_checkpoint_records_its_reason() {
    let mut relax = flat;
    let mut checkpoint = |_: ChainCheckpoint<'_>| CheckpointAction::Retire {
        reason: "saturated".to_owned(),
    };
    let mut ledger = Ledger::new(256);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run_with_energy_gradient_at_checkpoints(
        &bare(3),
        triangle().view(),
        &mut ledger,
        &mut relax,
        None,
        None,
        &mut rng,
        1,
        &mut checkpoint,
    )
    .expect("finite start");
    assert_eq!(
        out.stop,
        Some(HopStop::Retired {
            reason: "saturated".to_owned()
        })
    );
    assert!(ledger.remaining() > 0);
}

#[test]
fn a_budget_spent_by_a_hop_ends_at_the_next_hop() {
    let mut ledger = Ledger::new(9);
    let out = flat_run(&bare(3), &mut ledger);
    assert_eq!(out.stop, Some(HopStop::Budget));
    assert_eq!(out.charged, 9);
}

#[test]
fn a_budget_spent_by_the_inversion_quench_ends_the_run() {
    let mut cfg = bare(3);
    cfg.continuous_symmetry = ContinuousSymmetry::Inversion { interval: 1 };
    let calls = AtomicUsize::new(0);
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        if calls.fetch_add(1, Ordering::Relaxed) == 1 {
            drain(ledger);
        }
        flat(ledger, x, steps)
    };
    let mut ledger = Ledger::new(64);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run(&cfg, triangle().view(), &mut ledger, &mut relax, &mut rng).expect("finite");
    assert_eq!(out.stop, Some(HopStop::Budget));
    assert_eq!(out.continuous_symmetry.0, 1);
    assert_eq!((calls.load(Ordering::Relaxed), out.hops), (2, 0));
}

/// Two small atoms and one large one, the diameter ratio that turns on the
/// inner-outer exchange and the species escape.
fn wide(cfg: &mut Config) {
    cfg.species = Some(vec![1, 1, 2]);
    cfg.species_sigma = Some(vec![1.0, 1.0, 1.4]);
}

#[test]
fn a_budget_spent_by_the_species_exchange_ends_the_run() {
    let mut cfg = bare(3);
    wide(&mut cfg);
    let calls = AtomicUsize::new(0);
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        if calls.fetch_add(1, Ordering::Relaxed) == 1 {
            drain(ledger);
        }
        flat(ledger, x, steps)
    };
    let mut energy = zero_energy_gradient;
    let mut ledger = Ledger::new(64);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run_with_energy_gradient(
        &cfg,
        triangle().view(),
        &mut ledger,
        &mut relax,
        None,
        Some(&mut energy),
        &mut rng,
    )
    .expect("finite start");
    assert_eq!(out.stop, Some(HopStop::Budget));
    assert_eq!((calls.load(Ordering::Relaxed), out.hops), (2, 0));
    assert!(!out.nve_unavailable);
}

#[test]
#[cfg(not(feature = "rgsaddle"))]
fn a_requested_nve_escape_without_rgsaddle_says_so() {
    let mut cfg = bare(3);
    cfg.minima_hopping = true;
    cfg.md_escape = true;
    let mut relax = flat;
    let mut grad = zero_gradient;
    let mut ledger = Ledger::new(64);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run_with_gradient(
        &cfg,
        triangle().view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    assert_eq!(out.stop, Some(HopStop::NveUnavailable));
    assert!(out.nve_unavailable);
    assert_eq!(out.md_escape, (0, 0, 0));
}

#[test]
#[cfg(not(feature = "rgsaddle"))]
fn a_species_escape_without_rgsaddle_says_so() {
    let mut cfg = bare(3);
    wide(&mut cfg);
    let mut relax = flat;
    let mut energy = zero_energy_gradient;
    let mut ledger = Ledger::new(256);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run_with_energy_gradient(
        &cfg,
        triangle().view(),
        &mut ledger,
        &mut relax,
        None,
        Some(&mut energy),
        &mut rng,
    )
    .expect("finite start");
    assert_eq!(out.stop, Some(HopStop::NveUnavailable));
    assert!(out.nve_unavailable);
    assert_eq!(out.md_escape, (0, 0, 0));
}

#[test]
fn a_hamiltonian_proposal_without_a_trajectory_says_why() {
    let mut cfg = bare(3);
    cfg.hmc = Some(HopConfig::new(3, MetricKind::Identity));
    let mut relax = flat;
    let mut declines = |_: &mut Ledger, _: ArrayView1<f64>| None;
    let mut ledger = Ledger::new(64);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run_with_energy_gradient(
        &cfg,
        triangle().view(),
        &mut ledger,
        &mut relax,
        None,
        Some(&mut declines),
        &mut rng,
    )
    .expect("finite start");
    assert_eq!(out.stop, Some(HopStop::NoTrajectory));
    assert_eq!((out.hops, out.charged), (0, 1));

    let mut charges = zero_energy_gradient;
    let mut ledger = Ledger::new(8);
    let out = run_with_energy_gradient(
        &cfg,
        triangle().view(),
        &mut ledger,
        &mut relax,
        None,
        Some(&mut charges),
        &mut rng,
    )
    .expect("finite start");
    assert_eq!(out.stop, Some(HopStop::Budget));
    assert_eq!((out.hops, out.charged), (0, 8));
}

/// Raises the stop inside the first hop relaxation of `steps` steps.
fn stop_inside(steps_to_stop: usize) -> (Outcome, usize) {
    let stop = Arc::new(AtomicBool::new(false));
    let calls = AtomicUsize::new(0);
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        if calls.fetch_add(1, Ordering::Relaxed) > 0 && steps == steps_to_stop {
            stop.store(true, Ordering::Relaxed);
        }
        flat(ledger, x, steps)
    };
    let mut ledger = Ledger::with_stop(256, Arc::clone(&stop));
    let mut rng = StdRng::seed_from_u64(5);
    let out = run(
        &bare(3),
        triangle().view(),
        &mut ledger,
        &mut relax,
        &mut rng,
    )
    .expect("finite start");
    (out, calls.load(Ordering::Relaxed))
}

#[test]
fn a_stop_raised_inside_the_screen_ends_the_hop() {
    let (out, calls) = stop_inside(SCREEN);
    assert_eq!(out.stop, Some(HopStop::CallerStop));
    assert_eq!((calls, out.hops), (2, 0));
}

#[test]
fn a_stop_raised_inside_the_full_quench_ends_the_hop() {
    let (out, calls) = stop_inside(FULL);
    assert_eq!(out.stop, Some(HopStop::CallerStop));
    assert_eq!((calls, out.hops), (3, 0));
}

#[test]
fn a_hop_cap_reached_on_a_refused_quench_says_so() {
    let mut cfg = bare(3);
    cfg.max_hops = Some(2);
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        let (energy, state) = flat(ledger, x, steps);
        let refused = ledger.spent() > 1 && steps == FULL;
        (if refused { f64::NAN } else { energy }, state)
    };
    let mut ledger = Ledger::new(256);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run(&cfg, triangle().view(), &mut ledger, &mut relax, &mut rng).expect("finite");
    assert_eq!(out.stop, Some(HopStop::MaxHops));
    assert_eq!((out.hops, out.unconverged_records), (2, 2));
}

#[test]
fn a_hop_quench_at_the_target_ends_the_run() {
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        let (energy, state) = flat(ledger, x, steps);
        let deep = ledger.spent() > 1 && steps == FULL;
        (if deep { -5.0 } else { energy }, state)
    };
    let mut ledger = Ledger::new(256);
    ledger.stop_below(-1.0);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run(
        &bare(3),
        triangle().view(),
        &mut ledger,
        &mut relax,
        &mut rng,
    )
    .expect("finite");
    assert_eq!(out.stop, Some(HopStop::Target));
    assert_eq!((out.hops, out.best), (1, -5.0));
}

#[test]
fn a_hop_cap_reached_on_an_ordinary_hop_says_so() {
    let mut cfg = bare(3);
    cfg.max_hops = Some(3);
    let mut ledger = Ledger::new(256);
    let out = flat_run(&cfg, &mut ledger);
    assert_eq!(out.stop, Some(HopStop::MaxHops));
    assert_eq!(out.hops, 3);
    assert!(ledger.remaining() > 0);
}

/// Thirteen points on a Mackay icosahedron, the plateau the CSM reseed leaves.
fn icosahedron() -> Array1<f64> {
    let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
    let scale = 1.0 / (1.0 + phi * phi).sqrt();
    let mut x = vec![0.0; 3];
    for a in [-1.0, 1.0] {
        for b in [-phi, phi] {
            for vertex in [[0.0, a, b], [a, b, 0.0], [b, 0.0, a]] {
                x.extend(vertex.iter().map(|c| c * scale));
            }
        }
    }
    Array1::from(x)
}

#[test]
fn a_budget_spent_by_the_icosahedral_reseed_ends_the_run() {
    let mut cfg = bare(13);
    cfg.continuous_symmetry = ContinuousSymmetry::Measure {
        interval: 1000,
        reseed_hops: 1,
    };
    let ico = icosahedron();
    let calls = AtomicUsize::new(0);
    let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, _: usize| {
        if calls.fetch_add(1, Ordering::Relaxed) == 3 {
            drain(ledger);
        }
        let _ = ledger.charge();
        (0.0, ico.clone())
    };
    let mut ledger = Ledger::new(256);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run(
        &cfg,
        icosahedron().view(),
        &mut ledger,
        &mut relax,
        &mut rng,
    )
    .expect("finite");
    assert_eq!(out.stop, Some(HopStop::Budget));
    assert_eq!((calls.load(Ordering::Relaxed), out.hops), (4, 1));
}

#[test]
fn a_budget_spent_by_the_rigid_reseed_ends_the_run() {
    let cfg = quiet(Config::for_tip4p(2));
    let rest = array![-1.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let reseeds = AtomicUsize::new(0);
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _: usize| {
        // A rigid move turns or translates. Only a reseed redraws both.
        let moved = |part: ArrayView1<f64>, rest: ArrayView1<f64>| {
            part.iter().zip(rest).any(|(a, b)| (a - b).abs() > 1e-9)
        };
        if moved(x.slice(s![..6]), rest.slice(s![..6]))
            && moved(x.slice(s![6..]), rest.slice(s![6..]))
        {
            reseeds.fetch_add(1, Ordering::Relaxed);
            drain(ledger);
        }
        let _ = ledger.charge();
        (0.0, rest.clone())
    };
    let mut ledger = Ledger::new(100_000);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run(&cfg, rest.view(), &mut ledger, &mut relax, &mut rng).expect("finite");
    assert_eq!(out.stop, Some(HopStop::Budget));
    assert_eq!((reseeds.load(Ordering::Relaxed), out.hops), (1, 1001));
}

/// Runs a flat surface until an NVE escape has spent the budget.
#[cfg(feature = "rgsaddle")]
fn nve_spends_the_budget(cfg: &Config) -> Outcome {
    let mut relax = flat;
    let mut grad = zero_gradient;
    let mut energy = zero_energy_gradient;
    let mut ledger = Ledger::new(40);
    let mut rng = StdRng::seed_from_u64(5);
    run_with_energy_gradient(
        cfg,
        triangle().view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        Some(&mut energy),
        &mut rng,
    )
    .expect("finite start")
}

#[test]
#[cfg(feature = "rgsaddle")]
fn a_budget_spent_by_the_nve_escape_ends_the_run() {
    let mut cfg = bare(3);
    cfg.minima_hopping = true;
    cfg.md_escape = true;
    let out = nve_spends_the_budget(&cfg);
    assert_eq!(out.stop, Some(HopStop::Budget));
    assert!(!out.nve_unavailable);
    assert_eq!((out.hops, out.charged), (0, 40));
}

#[test]
#[cfg(feature = "rgsaddle")]
fn a_budget_spent_by_the_species_escape_ends_the_run() {
    let mut cfg = bare(3);
    wide(&mut cfg);
    let out = nve_spends_the_budget(&cfg);
    assert_eq!(out.stop, Some(HopStop::Budget));
    assert!(!out.nve_unavailable);
    assert_eq!(out.charged, 40);
}
