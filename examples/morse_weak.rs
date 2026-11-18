// Copyright (c) anneal developers
//! One seed of `Config::recommended` or `Config::recommended_adaptive` on
//! the twelve Cambridge Morse clusters, eight Lennard-Jones clusters, and
//! Au55.
//!
//! ```text
//! morse_weak run <system> <budget> <seed> [recommended|recommended_adaptive]
//! ```
//!
//! Lennard-Jones uses `PairPotential::lennard_jones` and `length_scale = 1`.
//! Morse uses `PairPotential::morse` (`rgpot_pair::cambridge_morse`) and
//! `PairKind::Morse { rho }.length_scale()`. Both arms run with no wall cap,
//! stopping at `best <= reference + 0.001` or at the paper call budget.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Gupta, PairKind, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const TOL: f64 = 0.001;

#[derive(Clone, Copy)]
enum Kind {
    LennardJones,
    Morse(f64),
    Gold,
}

type Potential = dyn Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync;

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = match args.first().map(String::as_str) {
        Some("run") => &args[1..],
        _ => &args[..],
    };
    cmd_run(rest);
}

fn cmd_run(args: &[String]) {
    let name = args.first().map(String::as_str).unwrap_or("");
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).expect("seed");
    let arm = args.get(3).map(String::as_str).unwrap_or("recommended");
    let Some((n, kind, reference)) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let level = reference + TOL;
    let (length, energy_scale) = match kind {
        Kind::LennardJones => (PairKind::LennardJones.length_scale(), 1.0),
        Kind::Morse(rho) => (PairKind::Morse { rho }.length_scale(), 1.0),
        Kind::Gold => gupta_scales(),
    };
    let cfg = match arm {
        "recommended" => Config::recommended_with_scales(n, length, energy_scale),
        "recommended_adaptive" => Config::recommended_adaptive_with_scales(n, length, energy_scale),
        _ => {
            eprintln!("unknown arm {arm}");
            std::process::exit(2);
        }
    };
    let potential: Box<Potential> = match kind {
        Kind::LennardJones => {
            let potential = PairPotential::lennard_jones(n);
            Box::new(move |x| potential.value_and_gradient(x))
        }
        Kind::Morse(rho) => {
            let potential = PairPotential::morse(n, rho);
            Box::new(move |x| potential.value_and_gradient(x))
        }
        Kind::Gold => {
            let potential = Gupta::copper_gold(0, n);
            Box::new(move |x| potential.value_and_gradient(x))
        }
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    assert_eq!(start.len(), 3 * n, "the start did not place every atom");
    let inside = AtomicU64::new(0);
    let started = Instant::now();
    let energy = |x: ArrayView1<f64>| {
        let mark = Instant::now();
        let out = potential(x);
        inside.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    };
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(level);
    let mut optimiser = WarmLbfgs::default();
    let first_seen = AtomicUsize::new(usize::MAX);
    let mark = |led: &Ledger| {
        if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
            first_seen.store(led.spent(), Ordering::Relaxed);
        }
    };
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        mark(led);
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
    let stop = if hit { "hit" } else { "budget" };
    println!(
        "system={name} arm={arm} seed={seed} budget={budget} hit={} best={best:.8} \
         first_calls={first_calls} charged={} wall_s={wall:.6} inside_s={inside_s:.6} \
         outside_s={outside_s:.6} reference={reference:.8} tol={TOL} stop={stop} \
         restarts={} orbits={} hops={}",
        u8::from(hit),
        outcome.charged,
        outcome.restarts,
        outcome.orbits.0,
        outcome.hops,
    );
}

fn is_hit(best: f64, reference: f64) -> bool {
    best <= reference + TOL
}

fn system(name: &str) -> Option<(usize, Kind, f64)> {
    Some(match name {
        "lj38" => (38, Kind::LennardJones, -173.928427),
        "lj55" => (55, Kind::LennardJones, -279.248470),
        "lj75" => (75, Kind::LennardJones, -397.492331),
        "lj98" => (98, Kind::LennardJones, -543.665361),
        "lj102" => (102, Kind::LennardJones, -569.363652),
        "lj103" => (103, Kind::LennardJones, -575.766131),
        "lj104" => (104, Kind::LennardJones, -582.086642),
        "lj150" => (150, Kind::LennardJones, -893.310258),
        "au55" => (55, Kind::Gold, -191.03085280),
        "m38r3" => (38, Kind::Morse(3.0), -249.159174),
        "m38r6" => (38, Kind::Morse(6.0), -157.477108),
        "m38r10" => (38, Kind::Morse(10.0), -145.849817),
        "m38r14" => (38, Kind::Morse(14.0), -144.321054),
        "m55r3" => (55, Kind::Morse(3.0), -417.918562),
        "m55r6" => (55, Kind::Morse(6.0), -250.286609),
        "m55r10" => (55, Kind::Morse(10.0), -225.814286),
        "m55r14" => (55, Kind::Morse(14.0), -220.646208),
        "m75r6" => (75, Kind::Morse(6.0), -351.472365),
        "m75r14" => (75, Kind::Morse(14.0), -318.407330),
        "m80r6" => (80, Kind::Morse(6.0), -378.333471),
        "m80r14" => (80, Kind::Morse(14.0), -340.811371),
        _ => return None,
    })
}

fn gupta_scales() -> (f64, f64) {
    (2.8850, 2.0390)
}

#[cfg(test)]
mod tests {
    use super::{Kind, is_hit, system};
    use anneal_core::potentials::PairPotential;

    #[test]
    fn hit_window_includes_boundary() {
        assert!(is_hit(-10.0 + 0.001, -10.0));
        assert!(!is_hit(-10.0 + 0.001_001, -10.0));
    }

    #[test]
    fn morse_minima_match_ccd() {
        for name in [
            "m38r6", "m38r10", "m38r14", "m55r3", "m55r6", "m55r10", "m55r14", "m75r6", "m75r14",
        ] {
            let (n, kind, reference) = system(name).expect("known system");
            let Kind::Morse(rho) = kind else {
                panic!("morse system had a non-Morse kind")
            };
            let ccd = PairPotential::ccd_morse_minimum(n, rho).expect("ccd");
            assert!((reference - ccd).abs() < 1e-12, "{name}");
        }
    }
}
