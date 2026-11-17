// Copyright (c) anneal developers
//! One seed of Config::for_cluster for the ASE panel.
//!
//! ```text
//! ase_panel <system> <budget> <seed> [for_cluster|recommended] [wall_cap_s]
//! ```
//!
//! `system` is `lj38`, `lj55`, `lj98`, `lj104`, `lj150` or `au55`.
//! Lennard-Jones uses PairPotential::lennard_jones. Gold uses
//! Gupta::copper_gold(0, 55), the same potential asg_eval serves.
//! `for_cluster` is the plain basin hop. `recommended` is the headline preset.
//! Scales for gold follow that pair's length and well depth.
//! An optional wall cap stops the walk. One `system=` line is printed.
//! `inside_s` is the time inside value_and_gradient.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Gupta, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const TOL: f64 = 0.001;

#[derive(Clone, Copy)]
enum Kind {
    Lj,
    Gold,
}

type Potential = dyn Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync;

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map(String::as_str).unwrap_or("");
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).expect("seed");
    let preset = args.get(3).map(String::as_str).unwrap_or("for_cluster");
    let cap_s = args.get(4).and_then(|v| v.parse::<f64>().ok());
    let Some((n, kind, reference)) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let level = reference + TOL;
    let cfg = match (kind, preset) {
        (_, "for_cluster") => for_cluster_cfg(n, kind),
        (_, "recommended") => recommended_cfg(n, kind),
        _ => {
            eprintln!("unknown preset {preset}");
            std::process::exit(2);
        }
    };
    let pot: Box<Potential> = match kind {
        Kind::Lj => {
            let potential = PairPotential::lennard_jones(n);
            Box::new(move |x| potential.value_and_gradient(x))
        }
        Kind::Gold => {
            let gupta = Gupta::copper_gold(0, n);
            Box::new(move |x| gupta.value_and_gradient(x))
        }
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let start = if preset == "recommended" {
        random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
    } else {
        match kind {
            Kind::Lj => random_cluster(n, 0.7, cfg.min_separation, &mut rng),
            Kind::Gold => {
                let reduced = (3.0 * n as f64 / (4.0 * std::f64::consts::PI * 0.7)).cbrt();
                random_cluster_in_radius(n, reduced * cfg.length_scale, cfg.min_separation, &mut rng)
            }
        }
    };
    assert_eq!(start.len(), 3 * n, "the start did not place every atom");
    let inside = AtomicU64::new(0);
    let started = Instant::now();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let energy = |x: ArrayView1<f64>| {
        let mark = Instant::now();
        let out = pot(x);
        inside.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    };
    let mut ledger = Ledger::with_stop(budget, Arc::clone(&stop_flag));
    ledger.stop_below(level);
    let mut optimiser = WarmLbfgs::default();
    let mut relaxations = 0usize;
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
        relaxations += 1;
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
    let hit = is_hit(best, reference);
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

fn for_cluster_cfg(n: usize, kind: Kind) -> Config {
    match kind {
        Kind::Lj => Config::for_cluster(n),
        Kind::Gold => {
            let (length, well) = gupta_scales();
            Config::for_cluster_with_scales(n, length, well)
        }
    }
}

fn recommended_cfg(n: usize, kind: Kind) -> Config {
    match kind {
        Kind::Lj => Config::recommended(n),
        Kind::Gold => {
            let (length, well) = gupta_scales();
            Config::recommended_with_scales(n, length, well)
        }
    }
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

fn is_hit(best: f64, reference: f64) -> bool {
    best <= reference + TOL
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

#[cfg(test)]
mod tests {
    use super::is_hit;

    #[test]
    fn hit_window_includes_its_declared_boundary() {
        assert!(is_hit(-10.0 + 0.001, -10.0));
        assert!(!is_hit(-10.0 + 0.001_001, -10.0));
    }
}
