// Copyright (c) anneal developers
//! One seed of `Config::for_cluster` or `Config::recommended` on a Morse cluster.
//!
//! ```text
//! ase_nonlj run <system> <budget> <seed> [for_cluster|recommended] [wall_cap_s]
//! ase_nonlj eval <system> <seed>
//! ```
//!
//! `system` is `m38r6`, `m38r14`, `m55r6` or `m55r14`. The potential is
//! `PairPotential::morse`, which installs `rgpot_pair::cambridge_morse`. Both
//! presets scale lengths by `PairKind::length_scale()` so steps, neighbour
//! cutoffs and starting sphere match the Morse pair minimum `r0 = 1`.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const TOL: f64 = 0.001;

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("eval") => cmd_eval(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        _ => cmd_run(&args),
    }
}

fn cmd_eval(args: &[String]) {
    let name = args.first().map(String::as_str).unwrap_or("");
    let seed: u64 = args.get(1).and_then(|v| v.parse().ok()).expect("seed");
    let Some((n, rho, _)) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let length = PairKind::Morse { rho }.length_scale();
    let cfg = Config::recommended_with_scales(n, length, 1.0);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let potential = PairPotential::morse(n, rho);
    let (energy, grad) = potential.value_and_gradient(start.view());
    println!("energy={energy:.16e}");
    for atom in start.exact_chunks(3) {
        println!("coord {:.16e} {:.16e} {:.16e}", atom[0], atom[1], atom[2]);
    }
    for atom in grad.exact_chunks(3) {
        println!("grad {:.16e} {:.16e} {:.16e}", atom[0], atom[1], atom[2]);
    }
}

fn cmd_run(args: &[String]) {
    let name = args.first().map(String::as_str).unwrap_or("");
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).expect("seed");
    let preset = args.get(3).map(String::as_str).unwrap_or("for_cluster");
    let cap_s = args.get(4).and_then(|v| v.parse::<f64>().ok());
    let Some((n, rho, reference)) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let level = reference + TOL;
    let length = PairKind::Morse { rho }.length_scale();
    let cfg = match preset {
        "for_cluster" => for_cluster_cfg(n, length),
        "recommended" => Config::recommended_with_scales(n, length, 1.0),
        _ => {
            eprintln!("unknown preset {preset}");
            std::process::exit(2);
        }
    };
    let potential = PairPotential::morse(n, rho);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = if preset == "recommended" {
        random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
    } else {
        let reduced = (3.0 * n as f64 / (4.0 * std::f64::consts::PI * 0.7)).cbrt();
        random_cluster_in_radius(n, reduced * cfg.length_scale, cfg.min_separation, &mut rng)
    };
    assert_eq!(start.len(), 3 * n, "the start did not place every atom");
    let inside = AtomicU64::new(0);
    let started = Instant::now();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let energy = |x: ArrayView1<f64>| {
        let mark = Instant::now();
        let out = potential.value_and_gradient(x);
        inside.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    };
    let mut ledger = Ledger::with_stop(budget, Arc::clone(&stop_flag));
    ledger.stop_below(level);
    let mut optimiser = WarmLbfgs::default();
    let first_seen = AtomicUsize::new(usize::MAX);
    let mark = |led: &Ledger| {
        if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
            first_seen.store(led.spent(), Ordering::Relaxed);
        }
    };
    let flag_relax = Arc::clone(&stop_flag);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        if cap_s.is_some_and(|cap| started.elapsed().as_secs_f64() >= cap) {
            flag_relax.store(true, Ordering::Relaxed);
            return (f64::INFINITY, x.to_owned());
        }
        mark(led);
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if cap_s.is_some_and(|cap| started.elapsed().as_secs_f64() >= cap) {
                flag_relax.store(true, Ordering::Relaxed);
                return None;
            }
            mark(led);
            if !led.charge() {
                return None;
            }
            Some(energy(v))
        });
        (value, relaxed)
    };
    let flag_grad = stop_flag;
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if cap_s.is_some_and(|cap| started.elapsed().as_secs_f64() >= cap) {
            flag_grad.store(true, Ordering::Relaxed);
            return None;
        }
        mark(led);
        if !led.charge() {
            return None;
        }
        Some(energy(v).1)
    };
    let mut run_rng = StdRng::seed_from_u64(seed);
    let outcome: Outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut run_rng,
    )
    .expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let inside_s = inside.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_s = (wall - inside_s).max(0.0);
    let best = outcome.best;
    let hit = best <= level;
    let seen = first_seen.load(Ordering::Relaxed);
    let first_calls = match (hit, seen) {
        (false, _) => "-".to_string(),
        (true, usize::MAX) => outcome.charged.to_string(),
        (true, calls) => calls.to_string(),
    };
    let stop = if hit {
        "hit"
    } else if outcome.charged >= budget {
        "budget"
    } else {
        "wall"
    };
    println!(
        "system={name} arm={preset} seed={seed} budget={budget} hit={} best={best:.8} \
         first_calls={first_calls} charged={} wall_s={wall:.6} inside_s={inside_s:.6} \
         outside_s={outside_s:.6} reference={reference:.8} tol={TOL} stop={stop}",
        u8::from(hit),
        outcome.charged,
    );
}

fn for_cluster_cfg(n: usize, length: f64) -> Config {
    let mut cfg = Config::with_scales(n, length, 1.0);
    let proto = Config::for_cluster(n);
    cfg.bayes_screen = proto.bayes_screen;
    cfg.bayes_threshold = proto.bayes_threshold;
    cfg.bayes_exploration = proto.bayes_exploration;
    cfg.bayes_warmup = proto.bayes_warmup;
    cfg
}

fn system(name: &str) -> Option<(usize, f64, f64)> {
    Some(match name {
        "m38r6" => (38, 6.0, -157.477108),
        "m38r14" => (38, 14.0, -144.321054),
        "m55r6" => (55, 6.0, -250.286609),
        "m55r14" => (55, 14.0, -220.646208),
        _ => return None,
    })
}
