//! One seed of `Config::for_cluster`, plain or with the trail leave.
//!
//! ```text
//! leave_gate <system> <base|leave> <budget> <seed>
//! ```
//!
//! `base` is [`Config::for_cluster`]. `leave` is that configuration with
//! [`Config::trail_on_stall`] set. The trail restarts from the screened
//! entry of the current basin. No other flag changes. The potential, the
//! length scale and the energy scale are the ones `h2h_arm3` uses.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Size, reference and hit tolerance. The numbers are the `h2h_arm3` pins.
fn system(name: &str) -> Option<(usize, f64, f64, Option<f64>)> {
    Some(match name {
        "lj38" => (38, -173.928427, 0.001, None),
        "lj55" => (55, -279.248470, 0.001, None),
        "lj75" => (75, -397.492331, 0.001, None),
        "lj98" => (98, -543.665361, 0.001, None),
        "m55r6" => (55, -250.286609, 0.001, Some(6.0)),
        "m55r14" => (55, -220.646208, 0.001, Some(14.0)),
        "m80r6" => (80, -378.333471, 0.001, Some(6.0)),
        "m80r14" => (80, -340.811371, 0.001, Some(14.0)),
        _ => return None,
    })
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map(String::as_str).unwrap_or("lj75");
    let arm = args.get(1).map(String::as_str).unwrap_or("base");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(3).and_then(|v| v.parse().ok()).expect("seed");
    let Some((n, reference, tol, morse)) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let leave = match arm {
        "base" => false,
        "leave" => true,
        _ => {
            eprintln!("unknown arm {arm}; expected base or leave");
            std::process::exit(2);
        }
    };
    let level = reference + tol;
    let (pot, pair) = match morse {
        Some(rho) => (PairPotential::morse(n, rho), PairKind::Morse { rho }),
        None => (PairPotential::lennard_jones(n), PairKind::LennardJones),
    };
    let mut cfg = Config::for_cluster(n);
    if leave {
        cfg.trail_on_stall = true;
    }
    let scale = pair.r_min() / 2.0_f64.powf(1.0 / 6.0);
    cfg.min_separation *= scale;
    cfg.container *= scale;
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);

    let inside_ns = Arc::new(AtomicU64::new(0));
    let clock = Arc::clone(&inside_ns);
    let energy = move |v: ArrayView1<f64>| {
        let started = Instant::now();
        let out = pot.value_and_gradient(v);
        clock.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    };

    let mut ledger = Ledger::new(budget);
    ledger.stop_below(level);
    let mut optimiser = WarmLbfgs::default();
    let mut relaxations = 0usize;
    let first_seen = AtomicUsize::new(usize::MAX);
    let mark = |led: &Ledger, _relaxations: usize| {
        if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
            first_seen.store(led.spent(), Ordering::Relaxed);
        }
    };
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        mark(led, relaxations);
        relaxations += 1;
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            mark(led, relaxations - 1);
            if !led.charge() {
                return None;
            }
            Some(energy(v))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        mark(led, usize::MAX);
        if !led.charge() {
            return None;
        }
        Some(energy(v).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome: Outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let inside = inside_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside = (wall - inside).max(0.0);
    let best = outcome.best;
    let hit = best < level;
    let seen = first_seen.load(Ordering::Relaxed);
    let first_calls = if hit && seen != usize::MAX {
        seen.to_string()
    } else if hit {
        outcome.charged.to_string()
    } else {
        "-".to_string()
    };
    println!(
        "system={name} arm={arm} seed={seed} budget={budget} hit={} best={best:.8} \
         first_calls={first_calls} charged={} hops={} relaxations={relaxations} \
         reference={reference:.6} tol={tol} trail={} wall_s={wall:.6} \
         inside_s={inside:.6} outside_s={outside:.6}",
        u8::from(hit),
        outcome.charged,
        outcome.hops,
        u8::from(leave),
    );
}
