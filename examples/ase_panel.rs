// Copyright (c) anneal developers
//! One seed of Config::for_cluster for the ASE panel.
//!
//! ```text
//! ase_panel <system> <budget> <seed>
//! ```
//!
//! `system` is `lj38`, `lj55`, `lj98`, `lj104`, `lj150` or `au55`.
//! Lennard-Jones uses PairPotential::lennard_jones. Gold uses
//! Gupta::copper_gold(0, 55), the same potential asg_eval serves, and
//! for_cluster's scales follow that pair's length and well depth.
//! One `system=` line is printed. `inside_s` is the time inside
//! value_and_gradient.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Gupta, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const TOL: f64 = 0.001;

enum Kind {
    Lj,
    Gold,
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map(String::as_str).unwrap_or("");
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).expect("seed");
    let Some((n, kind, reference)) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let level = reference + TOL;
    let (cfg, pot): (Config, Box<dyn Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync>) =
        match kind {
            Kind::Lj => {
                let potential = PairPotential::lennard_jones(n);
                (Config::for_cluster(n), Box::new(move |x| potential.value_and_gradient(x)))
            }
            Kind::Gold => {
                let (length, well) = gupta_scales();
                let mut cfg = Config::with_scales(n, length, well);
                let proto = Config::for_cluster(n);
                cfg.bayes_screen = proto.bayes_screen;
                cfg.bayes_threshold = proto.bayes_threshold;
                cfg.bayes_exploration = proto.bayes_exploration;
                cfg.bayes_warmup = proto.bayes_warmup;
                let gupta = Gupta::copper_gold(0, n);
                (cfg, Box::new(move |x| gupta.value_and_gradient(x)))
            }
        };
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
    let inside = AtomicU64::new(0);
    let energy = |x: ArrayView1<f64>| {
        let mark = Instant::now();
        let out = pot(x);
        inside.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    };
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(level);
    let mut optimiser = WarmLbfgs::default();
    let mut relaxations = 0usize;
    let first_seen = AtomicUsize::new(usize::MAX);
    let mark = |led: &Ledger| {
        if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
            first_seen.store(led.spent(), Ordering::Relaxed);
        }
    };
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        mark(led);
        relaxations += 1;
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            mark(led);
            if !led.charge() {
                return None;
            }
            Some(energy(v))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        mark(led);
        if !led.charge() {
            return None;
        }
        Some(energy(v).1)
    };
    let mut run_rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
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
    let hit = best < level;
    let seen = first_seen.load(Ordering::Relaxed);
    let first_calls = match (hit, seen) {
        (false, _) => "-".to_string(),
        (true, usize::MAX) => outcome.charged.to_string(),
        (true, calls) => calls.to_string(),
    };
    println!(
        "system={name} arm=base seed={seed} budget={budget} hit={} best={best:.8} \
         first_calls={first_calls} charged={} wall_s={wall:.6} inside_s={inside_s:.6} \
         outside_s={outside_s:.6} reference={reference:.8} tol={TOL}",
        u8::from(hit),
        outcome.charged,
    );
}

fn system(name: &str) -> Option<(usize, Kind, f64)> {
    Some(match name {
        "lj38" => (38, Kind::Lj, -173.928427),
        "lj55" => (55, Kind::Lj, -279.248470),
        "lj98" => (98, Kind::Lj, -543.665361),
        "lj104" => (104, Kind::Lj, -582.086642),
        "lj150" => (150, Kind::Lj, -893.310258),
        "au55" => (55, Kind::Gold, -191.03085280),
        _ => return None,
    })
}

/// Pair minimum over 2^(1/6), and the well depth. Same probe as asg_eval.
fn gupta_scales() -> (f64, f64) {
    let potential = Gupta::copper_gold(0, 2);
    let mut best_r = 0.0;
    let mut best_e = f64::INFINITY;
    let mut r = 1.8;
    while r <= 4.5 {
        let mut x = Array1::zeros(6);
        x[3] = r;
        let energy = potential.value_and_gradient(x.view()).0;
        if energy < best_e {
            best_e = energy;
            best_r = r;
        }
        r += 0.002;
    }
    (best_r / 2.0_f64.powf(1.0 / 6.0), -best_e)
}
