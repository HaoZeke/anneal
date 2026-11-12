// Copyright (c) anneal developers

//! LJ55 default arm used beside the AGOX board.
//!
//! Same search as `h2h_arm3 lj55 default <budget> <seed>`:
//! `Config::for_cluster`, `random_cluster` at density 0.7, `WarmLbfgs`,
//! and `run_with_gradient`. The budget is the paper ceiling, 1000000.
//! The ledger stops once a recorded energy is below the board tolerance
//! on the reference -279.248470. The force check uses the same bar as
//! the board, on the best state, after the timed search returns.
//!
//! Usage: `lj55_board_arm <seed> <budget>`

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

const N: usize = 55;
const REFERENCE: f64 = -279.248470;
const HIT_TOL: f64 = 1.0e-4;
const FORCE_BAR: f64 = 0.0514221;

fn first_line(path: &str) -> String {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| text.lines().next().map(|line| line.trim().to_string()))
        .unwrap_or_else(|| "absent".to_string())
}

fn main() {
    let hostname = first_line("/proc/sys/kernel/hostname");
    let boot_id = first_line("/proc/sys/kernel/random/boot_id");
    let uptime_s = first_line("/proc/uptime").split_whitespace().next().unwrap_or("absent").to_string();
    let mut stdout = std::io::stdout();
    writeln!(stdout, "hostname={hostname}").unwrap();
    writeln!(stdout, "boot_id={boot_id}").unwrap();
    writeln!(stdout, "uptime_s={uptime_s}").unwrap();
    stdout.flush().unwrap();
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .expect("seed");
    let budget: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .expect("budget");
    let level = REFERENCE + HIT_TOL;
    let cfg = Config::for_cluster(N);
    let pot = PairPotential::lennard_jones(N);
    let mut start_rng = StdRng::seed_from_u64(seed);
    let start = random_cluster(N, 0.7, cfg.min_separation, &mut start_rng);

    let energy_calls = Arc::new(AtomicUsize::new(0));
    let grad_calls = Arc::new(AtomicUsize::new(0));
    let potential_calls = Arc::new(AtomicUsize::new(0));
    let inside_ns = Arc::new(AtomicU64::new(0));
    let pot_calls = Arc::clone(&potential_calls);
    let inside = Arc::clone(&inside_ns);
    let energy = Box::new(move |x: ArrayView1<f64>| {
        let started = Instant::now();
        let out = pot.value_and_gradient(x);
        inside.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        pot_calls.fetch_add(1, Ordering::Relaxed);
        out
    });

    let mut ledger = Ledger::new(budget);
    // `stop_below` fires on `energy <= target`. One ulp under the board
    // level keeps that the same as `energy < reference + HIT_TOL`.
    ledger.stop_below(level.next_down());
    let mut optimiser = WarmLbfgs::default();
    let mut relaxations = 0usize;
    let first_seen = AtomicUsize::new(usize::MAX);
    let mark = |led: &Ledger| {
        if led.target_reached() && first_seen.load(Ordering::Relaxed) == usize::MAX {
            first_seen.store(led.spent(), Ordering::Relaxed);
        }
    };
    let energy_hits = Arc::clone(&energy_calls);
    let grad_hits = Arc::clone(&grad_calls);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        mark(led);
        relaxations += 1;
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            mark(led);
            if !led.charge() {
                return None;
            }
            energy_hits.fetch_add(1, Ordering::Relaxed);
            grad_hits.fetch_add(1, Ordering::Relaxed);
            Some(energy(v))
        });
        (value, relaxed)
    };
    let energy_only = Arc::clone(&grad_calls);
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        mark(led);
        if !led.charge() {
            return None;
        }
        energy_only.fetch_add(1, Ordering::Relaxed);
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
    let wall_s = started.elapsed().as_secs_f64();
    let inside_s = inside_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_s = wall_s - inside_s;

    let best = outcome.best;
    let (best_force, force_ok) = match outcome.best_state.as_ref() {
        Some(state) => {
            let audit = PairPotential::lennard_jones(N);
            let (_, gradient) = audit.value_and_gradient(state.view());
            let force = gradient.iter().copied().map(f64::abs).fold(0.0, f64::max);
            (force, force <= FORCE_BAR)
        }
        None => (f64::INFINITY, false),
    };
    let energy_ok = best < level;
    let calls_ok = outcome.charged <= budget;
    let board_hit = energy_ok && force_ok && calls_ok;
    let stop = if board_hit {
        "hit"
    } else if outcome.charged >= budget {
        "ceiling"
    } else if energy_ok && !force_ok {
        "force"
    } else {
        "return"
    };
    let seen = first_seen.load(Ordering::Relaxed);
    let first_calls = if seen == usize::MAX {
        outcome.charged
    } else {
        seen
    };

    println!("hostname={hostname}");
    println!("boot_id={boot_id}");
    println!("uptime_s={uptime_s}");
    println!("code=anneal");
    println!("n={N}");
    println!("seed={seed}");
    println!("budget={budget}");
    println!("reference={REFERENCE:.6}");
    println!("hit_tol={HIT_TOL:.6e}");
    println!("force_bar={FORCE_BAR:.7}");
    println!("best={best:.8}");
    println!("best_force={best_force:.6e}");
    println!("board_hit={}", u8::from(board_hit));
    println!("energy_calls={}", energy_calls.load(Ordering::Relaxed));
    println!("grad_calls={}", grad_calls.load(Ordering::Relaxed));
    println!("potential_calls={}", potential_calls.load(Ordering::Relaxed));
    println!("charged={}", outcome.charged);
    println!("first_calls={first_calls}");
    println!("hops={}", outcome.hops);
    println!("relaxations={relaxations}");
    println!("inside_s={inside_s:.6}");
    println!("outside_s={outside_s:.6}");
    println!("wall_s={wall_s:.6}");
    println!("stop={stop}");
    println!(
        "config_sha256={}",
        cfg.resolved_sha256().unwrap_or_else(|_| "unset".into())
    );
}
