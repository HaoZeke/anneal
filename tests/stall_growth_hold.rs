// Copyright (c) anneal developers
//! A stall growth that misses the incumbent does not quench another lattice
//! until that incumbent drops. The charge it would have spent is one
//! all-points displacement.

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, run_with_gradient};
use anneal_core::soap::{SoapSpec, ih_dominated};
use ndarray::{Array1, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};

fn icosahedron() -> Array1<f64> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let radius = 1.1 / (1.0 + phi * phi).sqrt();
    let mut coords = vec![0.0; 3];
    for (a, b) in [(1.0, phi), (1.0, -phi), (-1.0, phi), (-1.0, -phi)] {
        for vertex in [[0.0, a, b], [a, b, 0.0], [b, 0.0, a]] {
            coords.extend(vertex.iter().map(|c| c * radius));
        }
    }
    Array1::from(coords)
}

fn line(n: usize) -> Array1<f64> {
    let mut start = Array1::zeros(3 * n);
    for i in 0..n {
        start[3 * i] = i as f64 * 1.3;
    }
    start
}

fn rms(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt()
}

fn gyration(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n == 0 {
        return 0.0;
    }
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in &mut c {
        *v /= n as f64;
    }
    let mut s = 0.0;
    for i in 0..n {
        for k in 0..3 {
            let d = x[3 * i + k] - c[k];
            s += d * d;
        }
    }
    (s / n as f64).sqrt()
}

fn stall_cfg(n: usize, budget_window: bool) -> Config {
    let mut cfg = Config::for_cluster(n);
    cfg.max_hops = Some(5);
    cfg.escape_on_stall = true;
    cfg.restart_on_stall = false;
    cfg.trail_on_stall = false;
    cfg.budget_window = budget_window;
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
    cfg.move_library = MoveLibrary::WalesDoye;
    cfg
}

/// Quenches that rebuild a lattice sit far from the occupied geometry.
/// An all-points step does not.
fn count_far(
    cfg: &Config,
    home: &Array1<f64>,
    lattice: impl Fn(f64, f64, usize) -> bool + Sync,
    energy_of: impl Fn(usize, f64) -> f64 + Sync,
) -> (usize, usize, f64) {
    let mut calls = 0usize;
    let mut far = 0usize;
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        calls += 1;
        let distance = rms(x, home.view());
        if lattice(distance, gyration(x), steps) {
            far += 1;
        }
        let _ = ledger.charge();
        (energy_of(calls, distance), home.clone())
    };
    let mut ledger = Ledger::new(10_000);
    let mut rng = StdRng::seed_from_u64(7);
    let outcome = run_with_gradient(cfg, home.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    (calls, far, outcome.best)
}

#[test]
fn a_failed_icosahedral_growth_does_not_repeat() {
    let home = icosahedron();
    assert!(
        ih_dominated(home.view(), SoapSpec::default()),
        "the fixture is not icosahedral"
    );
    let cfg = stall_cfg(13, false);
    let (calls, far, best) = count_far(
        &cfg,
        &home,
        |distance, _gyration, steps| steps > 0 && distance > 3.0,
        |_call, _distance| -1.0,
    );
    assert_eq!(
        far, 3,
        "fcc, hcp and pentagonal ran more than once ({calls} quenches)"
    );
    assert!(
        calls > far + 5,
        "the held stalls did not spend a displacement ({calls} quenches)"
    );
    assert!((best + 1.0).abs() < 1e-12);
}

#[test]
fn a_failed_lattice_growth_does_not_repeat_off_the_icosahedron() {
    let home = line(13);
    assert!(
        !ih_dominated(home.view(), SoapSpec::default()),
        "the line was classified icosahedral"
    );
    let cfg = stall_cfg(13, true);
    // A lattice rebuild is compact. The line and an all-points step on it
    // are not. The five library sources land as one consecutive run.
    let compact = std::sync::Mutex::new(Vec::new());
    let (calls, _far, best) = count_far(
        &cfg,
        &home,
        |distance, gyration, steps| {
            let hit = steps > 0 && distance > 3.0 && gyration < 3.0;
            compact.lock().unwrap().push(hit);
            hit
        },
        |_call, _distance| -1.0,
    );
    let flags = compact.lock().unwrap().clone();
    let mut runs = 0usize;
    let mut streak = 0usize;
    for hit in flags {
        if hit {
            streak += 1;
            if streak == 5 {
                runs += 1;
            }
        } else {
            streak = 0;
        }
    }
    assert_eq!(
        runs, 1,
        "the lattice library ran {runs} times in {calls} quenches"
    );
    assert!((best + 1.0).abs() < 1e-12);
}

fn dry(home: &Array1<f64>, budget_window: bool, budget: usize) {
    let mut cfg = stall_cfg(home.len() / 3, budget_window);
    cfg.max_hops = Some(3);
    let mut relax = |ledger: &mut Ledger, _x: ArrayView1<f64>, _steps: usize| {
        let _ = ledger.charge();
        (-1.0, home.clone())
    };
    let mut ledger = Ledger::new(budget);
    let mut rng = StdRng::seed_from_u64(11);
    let outcome = run_with_gradient(&cfg, home.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    assert_eq!(
        ledger.spent(),
        budget,
        "the dry ledger did not stop at the budget"
    );
    assert!(outcome.best.is_finite(), "the dry run has no finite best");
}

#[test]
fn a_dry_ledger_stops_both_growth_loops() {
    dry(&icosahedron(), false, 4);
    dry(&line(13), true, 4);
}

#[test]
fn an_icosahedral_growth_that_deepens_the_incumbent_is_kept() {
    let home = icosahedron();
    let cfg = stall_cfg(13, false);
    let (_calls, far, best) = count_far(
        &cfg,
        &home,
        |distance, _gyration, steps| steps > 0 && distance > 3.0,
        |_call, distance| if distance > 3.0 { -80.0 } else { -1.0 },
    );
    assert!(far >= 3, "the three lattice sources did not quench");
    assert!((best + 80.0).abs() < 1e-12);
}

#[test]
fn a_short_growth_state_is_skipped_and_a_dry_ledger_stops_the_rest() {
    let home = icosahedron();
    let mut cfg = stall_cfg(13, false);
    cfg.max_hops = Some(3);
    let mut short = 0usize;
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        let distance = rms(x, home.view());
        let _ = ledger.charge();
        if steps > 0 && distance > 3.0 && short < 1 {
            short += 1;
            return (-1.0, Array1::zeros(3));
        }
        (-1.0, home.clone())
    };
    let mut ledger = Ledger::new(8);
    let mut rng = StdRng::seed_from_u64(7);
    let outcome = run_with_gradient(&cfg, home.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    assert_eq!(short, 1, "the short lattice state was not returned once");
    assert!(
        (outcome.best + 1.0).abs() < 1e-12,
        "the short state was adopted, best {}",
        outcome.best
    );
}

#[test]
fn a_non_icosahedral_stall_without_the_budget_window_accepts_all_points() {
    let home = line(13);
    assert!(
        !ih_dominated(home.view(), SoapSpec::default()),
        "the line was classified icosahedral"
    );
    let mut cfg = stall_cfg(13, false);
    // Recommended uses LeanBurst, which has no all-points kernel.
    // An `all` acceptance can come only from the stall hold.
    cfg.move_library = MoveLibrary::LeanBurst;
    assert!(
        !cfg.budget_window,
        "recommended leaves the budget window off"
    );
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

#[test]
fn a_held_growth_keeps_the_move_set() {
    let home = icosahedron();
    assert!(
        ih_dominated(home.view(), SoapSpec::default()),
        "the fixture is not icosahedral"
    );
    let mut cfg = stall_cfg(13, false);
    // LeanBurst has no all-points kernel. An `all` acceptance is the hold.
    // The paper patience is thousands of force calls, so hops between leaves
    // must keep the library. Patience 1 makes every hop a leave and hides that.
    cfg.move_library = MoveLibrary::LeanBurst;
    cfg.soap_mode = anneal_core::methods::cluster_hopping::SoapProposalMode::Off;
    cfg.restart_patience = 20;
    cfg.max_hops = Some(40);
    cfg.allocate_moves = false;
    let mut actions = Vec::new();
    let mut relax = |ledger: &mut Ledger, _x: ArrayView1<f64>, _steps: usize| {
        let _ = ledger.charge();
        (-1.0, home.clone())
    };
    let mut ledger = Ledger::new(20_000);
    let mut rng = StdRng::seed_from_u64(7);
    let outcome = run_with_gradient(&cfg, home.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    for step in &outcome.accepted_transitions {
        actions.push(step.action.clone());
    }
    let held = actions.iter().position(|action| action == "all");
    let Some(held_at) = held else {
        panic!("the hold never emitted an all-points acceptance: {actions:?}");
    };
    let library = ["single", "surface", "sym", "burst"];
    let kept = actions[held_at + 1..]
        .iter()
        .any(|action| library.contains(&action.as_str()));
    assert!(
        kept,
        "after the hold at {held_at} the move set was only all-points: {actions:?}"
    );
}

#[test]
fn an_all_points_improvement_allows_growth_again() {
    let home = icosahedron();
    let cfg = stall_cfg(13, false);
    // The held displacement is the near quench after the first three
    // lattice rebuilds. A new best there clears the hold, so the next
    // stall grows the three lattices again.
    let (calls, far, best) = count_far(
        &cfg,
        &home,
        |distance, _gyration, steps| steps > 0 && distance > 3.0,
        |call, distance| {
            if distance <= 3.0 && call >= 8 {
                -40.0
            } else {
                -1.0
            }
        },
    );
    assert!(
        far >= 6,
        "growth did not run again after the incumbent dropped ({far} lattice quenches, {calls} total)"
    );
    assert!((best + 40.0).abs() < 1e-12);
}
