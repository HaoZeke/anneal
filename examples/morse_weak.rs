// Copyright (c) anneal developers
//! One seed of `Config::recommended` or `Config::recommended_adaptive` on
//! the twelve Cambridge Morse clusters and the five Lennard-Jones clusters.
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
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const TOL: f64 = 0.001;

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
    let Some((n, rho, reference)) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let level = reference + TOL;
    let length = match rho {
        None => PairKind::LennardJones.length_scale(),
        Some(value) => PairKind::Morse { rho: value }.length_scale(),
    };
    let cfg = match arm {
        "recommended" => Config::recommended_with_scales(n, length, 1.0),
        "recommended_adaptive" => Config::recommended_adaptive_with_scales(n, length, 1.0),
        _ => {
            eprintln!("unknown arm {arm}");
            std::process::exit(2);
        }
    };
    let potential = match rho {
        None => PairPotential::lennard_jones(n),
        Some(value) => PairPotential::morse(n, value),
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    assert_eq!(start.len(), 3 * n, "the start did not place every atom");
    let inside = AtomicU64::new(0);
    let started = Instant::now();
    let energy = |x: ArrayView1<f64>| {
        let mark = Instant::now();
        let out = potential.value_and_gradient(x);
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

fn system(name: &str) -> Option<(usize, Option<f64>, f64)> {
    Some(match name {
        "lj38" => (38, None, -173.928427),
        "lj55" => (55, None, -279.248470),
        "lj75" => (75, None, -397.492331),
        "lj98" => (98, None, -543.665361),
        "lj104" => (104, None, -582.086642),
        "m38r3" => (38, Some(3.0), -249.159174),
        "m38r6" => (38, Some(6.0), -157.477108),
        "m38r10" => (38, Some(10.0), -145.849817),
        "m38r14" => (38, Some(14.0), -144.321054),
        "m55r3" => (55, Some(3.0), -417.918562),
        "m55r6" => (55, Some(6.0), -250.286609),
        "m55r10" => (55, Some(10.0), -225.814286),
        "m55r14" => (55, Some(14.0), -220.646208),
        "m75r6" => (75, Some(6.0), -351.472365),
        "m75r14" => (75, Some(14.0), -318.407330),
        "m80r6" => (80, Some(6.0), -378.333471),
        "m80r14" => (80, Some(14.0), -340.811371),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{is_hit, system};
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
            let (n, rho, reference) = system(name).expect("known system");
            let ccd = PairPotential::ccd_morse_minimum(n, rho.expect("morse")).expect("ccd");
            assert!((reference - ccd).abs() < 1e-12, "{name}");
        }
    }
}
