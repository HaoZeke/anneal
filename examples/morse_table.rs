//! Morse clusters from the Cambridge table, scored against its global minima.
//!
//! `cargo run --release --example morse_table -- <table.csv> <rho> <sizes> <budget> <seed0> <seeds> [reduced]`
//!
//! `examples/fixtures/morse_ccd.csv` holds the table's bold minima for
//! rho = 3, 6, 10 and 14. `<sizes>` is `all` or a comma list. The search runs
//! `Config::recommended_with_scales` at the pair's own length scale, or
//! `Config::recommended` on reduced units with `reduced`. Every potential call is counted, and a
//! run is a hit at the first call whose energy is within 1e-3 of the table's
//! minimum with every gradient component at most 1e-3; the search stops
//! there. One tab-separated line per run: rho, size, seed, hit, calls, best
//! energy, gap to the minimum, then hops, accepted hops, mean accepted step,
//! screened out, basins and per-arm draws/accepts.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const TOLERANCE: f64 = 1.0e-3;

struct Counted {
    pot: PairPotential,
    target: f64,
    calls: AtomicUsize,
    hit_at: AtomicUsize,
    best: Mutex<f64>,
    stop: Arc<AtomicBool>,
}

impl Counted {
    fn evaluate(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        let (e, g) = self.pot.value_and_gradient(x);
        let converged = g.iter().all(|v| v.abs() <= TOLERANCE);
        if converged {
            let mut best = self.best.lock().expect("best");
            *best = best.min(e);
        }
        if converged && e <= self.target + TOLERANCE {
            let _ = self
                .hit_at
                .compare_exchange(0, call, Ordering::Relaxed, Ordering::Relaxed);
            self.stop.store(true, Ordering::Relaxed);
        }
        (e, g)
    }
}

impl Objective<f64> for Counted {
    fn dim(&self) -> usize {
        Objective::dim(&self.pot)
    }
    fn bounds(&self) -> &Bounds<f64> {
        self.pot.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evaluate(x).0
    }
}

impl Gradient<f64> for Counted {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.evaluate(x).1
    }
    fn dim(&self) -> usize {
        Objective::dim(&self.pot)
    }
}

impl DifferentiableObjective<f64> for Counted {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.evaluate(x)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() == 7 || (args.len() == 8 && args[7] == "reduced"),
        "usage: <table.csv> <rho> <sizes> <budget> <seed0> <seeds> [reduced]"
    );
    let reduced = args.len() == 8;
    let rho: f64 = args[2].parse().expect("rho");
    let table: Vec<(usize, f64)> = std::fs::read_to_string(&args[1])
        .expect("table")
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let row = (f[1].parse().ok()?, f[4].parse().ok()?);
            (f[0].parse::<f64>().ok()? == rho).then_some(row)
        })
        .collect();
    let sizes: Option<Vec<usize>> = (args[3] != "all").then(|| {
        args[3]
            .split(',')
            .map(|v| v.parse().expect("size"))
            .collect()
    });
    let budget: usize = args[4].parse().expect("budget");
    let seed0: u64 = args[5].parse().expect("seed0");
    let seeds: u64 = args[6].parse().expect("seeds");
    let jobs: Vec<(usize, f64, u64)> = table
        .iter()
        .filter(|(n, _)| sizes.as_ref().is_none_or(|s| s.contains(n)))
        .flat_map(|&(n, e)| (seed0..seed0 + seeds).map(move |s| (n, e, s)))
        .collect();
    let lines: Vec<String> = jobs
        .par_iter()
        .map(|&(n, reference, seed)| {
            let stop = Arc::new(AtomicBool::new(false));
            let pot = Counted {
                pot: PairPotential::morse(n, rho),
                target: reference,
                calls: AtomicUsize::new(0),
                hit_at: AtomicUsize::new(0),
                best: Mutex::new(f64::INFINITY),
                stop: Arc::clone(&stop),
            };
            let cfg = if reduced {
                Config::recommended(n)
            } else {
                Config::recommended_with_scales(n, pot.pot.kind().length_scale(), 1.0)
            };
            let mut ledger = Ledger::with_stop(budget, stop);
            let (out, _) = search(&pot, &cfg, &mut ledger, seed).expect("search");
            let hit = pot.hit_at.load(Ordering::Relaxed);
            let best = *pot.best.lock().expect("best");
            let arms: Vec<String> = out
                .arms
                .iter()
                .map(|(name, d, a, _)| format!("{name}:{d}/{a}"))
                .collect();
            format!(
                "{rho}\t{n}\t{seed}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{:.4}\t{}\t{}\t{}",
                u8::from(hit > 0),
                if hit > 0 {
                    hit
                } else {
                    pot.calls.load(Ordering::Relaxed)
                },
                best,
                best - reference,
                out.hops,
                out.accepted,
                out.mean_step,
                out.screened_out,
                out.basins,
                arms.join(",")
            )
        })
        .collect();
    for line in lines {
        println!("{line}");
    }
}
