//! Lean preset, stopped on a count of full local relaxations.
//!
//! `Config::for_cluster` with `MoveLibrary::Lean` is the `lean` arm of
//! `lj_cluster_search`. A full local relaxation is a warm L-BFGS call made
//! with `relax_steps`. The short screen is not counted: it is a prefix, not
//! a local minimum. The run stops at the relaxation budget, or once a full
//! relaxation has been checked against the Cambridge value passed in.
//!
//! The ledger is closed on the call after that relaxation. Closing inside
//! the successful call makes the search discard the minimum before it can
//! record the coordinates.
//!
//! Usage: `lj_lean_relax_budget <n> <relaxations> <seed> <reference>`

use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, optimize_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;

const HIT_TOL: f64 = 1.0e-4;

fn hostname() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_default()
}

fn main() {
    if hostname() != "bench" {
        eprintln!("uts hostname is not bench");
        std::process::exit(2);
    }
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let budget: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let seed: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let reference: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(f64::NAN);
    if n < 2 || budget == 0 || !reference.is_finite() {
        eprintln!("usage: lj_lean_relax_budget <n> <relaxations> <seed> <reference>");
        std::process::exit(2);
    }

    let mut cfg = Config::for_cluster(n);
    cfg.move_library = MoveLibrary::Lean;
    let screen_steps = cfg.screen_steps;
    let potential = PairPotential::lennard_jones(n);
    let stop = Arc::new(AtomicBool::new(false));
    let mut ledger = Ledger::with_stop(usize::MAX / 4, Arc::clone(&stop));

    let mut opt = WarmLbfgs::default();
    opt.gtol = cfg.record_gradient;
    let mut full = 0usize;
    let mut screens = 0usize;
    let mut hit_at: Option<usize> = None;
    // 0 searching, 1 let the search record the minimum, 2 close the ledger.
    let mut phase = 0u8;
    let started = Instant::now();
    let outcome = {
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            if phase == 2 {
                stop.store(true, Ordering::Relaxed);
                return (f64::INFINITY, x.to_owned());
            }
            if phase == 1 {
                phase = 2;
                if !led.charge() {
                    return (f64::INFINITY, x.to_owned());
                }
                let (energy, _) = potential.value_and_gradient(x);
                return (energy, x.to_owned());
            }
            if iters == 0 {
                if !led.charge() {
                    return (f64::INFINITY, x.to_owned());
                }
                let (energy, _) = potential.value_and_gradient(x);
                return (energy, x.to_owned());
            }
            let counts = iters > screen_steps;
            opt.forget();
            let (energy, state, _) = opt.minimize(x, iters, |point| {
                if !led.charge() {
                    return None;
                }
                Some(potential.value_and_gradient(point))
            });
            if counts {
                full += 1;
                if hit_at.is_none() && energy < reference + HIT_TOL {
                    let (fresh, gradient) = potential.value_and_gradient(state.view());
                    let gmax = gradient
                        .iter()
                        .fold(0.0_f64, |peak, value| peak.max(value.abs()));
                    if fresh < reference + HIT_TOL && gmax < 1.0e-3 {
                        hit_at = Some(full);
                        phase = 1;
                    }
                }
                if full >= budget {
                    phase = 1;
                }
            } else {
                screens += 1;
            }
            (energy, state)
        };
        optimize_with_gradient(&cfg, &mut ledger, &mut relax, None, seed)
    };
    let wall_s = started.elapsed().as_secs_f64();
    let (best, hit) = match outcome
        .as_ref()
        .ok()
        .and_then(|out| out.best_state.as_ref())
    {
        Some(state) => {
            let (energy, gradient) = potential.value_and_gradient(state.view());
            let gmax = gradient
                .iter()
                .fold(0.0_f64, |peak, value| peak.max(value.abs()));
            let reached = energy < reference + HIT_TOL && gmax < 1.0e-3;
            (energy, reached)
        }
        None => (f64::INFINITY, false),
    };
    let to_hit = if hit {
        hit_at.map(|k| k.to_string()).unwrap_or_default()
    } else {
        String::new()
    };
    println!(
        "arm=lean n={n} seed={seed} hit={} relaxations={full} relaxations_to_hit={to_hit} screens={screens} best={best:.6} wall_s={wall_s:.6} charged={}",
        i32::from(hit),
        ledger.spent()
    );
    if outcome.is_err() && !hit && full == 0 {
        std::process::exit(1);
    }
}
