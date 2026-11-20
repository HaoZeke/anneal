// Copyright (c) anneal developers

//! Single-seed rigid TIP4P gate runner with exact IEEE-754 bit reporting.
//!
//! Usage: `tip4p_gate_driver <n> <budget> <seed> <reference>`
//!
//! Runs `Config::recommended_tip4p(n)` with `gmin_water_geometry()` on
//! `GminTip4p::new(n)` from the standard `h2h_seed` start, stopping strictly
//! on the evaluation budget (budget-only stop). Never reads `H2H_*` variables.

use anneal_core::methods::cluster_hopping::{Config, Ledger, Outcome};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::potentials::{GminTip4p, gmin_water_geometry, random_tip4p_cluster};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

struct Seen {
    calls: usize,
    lowest: f64,
    first_hit_call: Option<usize>,
}

struct Counted {
    model: GminTip4p,
    budget: usize,
    reference: f64,
    hit_tol: f64,
    stop: Arc<AtomicBool>,
    seen: Mutex<Seen>,
    pot_ns: AtomicU64,
}

impl Counted {
    fn counted(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mark = Instant::now();
        let (e, g) = self.model.value_and_gradient(x);
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let mut seen = self.seen.lock().unwrap();
        seen.calls += 1;
        if seen.calls <= self.budget {
            if e < seen.lowest {
                seen.lowest = e;
            }
            if seen.first_hit_call.is_none()
                && anneal_core::energy_hits(e, self.reference, self.hit_tol)
            {
                seen.first_hit_call = Some(seen.calls);
            }
        }
        if seen.calls >= self.budget {
            self.stop.store(true, Ordering::Relaxed);
        }
        (e, g)
    }
}

impl Objective<f64> for Counted {
    fn dim(&self) -> usize {
        Objective::dim(&self.model)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.model.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.counted(x).0
    }
}

impl Gradient<f64> for Counted {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.counted(x).1
    }

    fn dim(&self) -> usize {
        Objective::dim(self)
    }
}

impl DifferentiableObjective<f64> for Counted {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.counted(x)
    }
}

fn sha256_coords(coords: ArrayView1<f64>) -> String {
    let mut hasher = Sha256::new();
    for &v in coords.iter() {
        hasher.update(v.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn exact_f64_string(value: f64) -> String {
    let text = format!("{value}");
    let parsed: f64 = text.parse().expect("round-trip f64 parse");
    assert_eq!(
        parsed.to_bits(),
        value.to_bits(),
        "f64 string did not round-trip bit-exactly"
    );
    text
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        4,
        "usage: tip4p_gate_driver <n> <budget> <seed> <reference>"
    );
    let n: usize = args[0].parse().expect("n");
    let budget: usize = args[1].parse().expect("budget");
    let seed: u64 = args[2].parse().expect("seed");
    let reference: f64 = args[3].parse().expect("reference");
    let hit_tol = anneal_core::DEFAULT_HIT_TOL;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
    let start = random_tip4p_cluster(n, &mut rng);
    let start_sha256 = sha256_coords(start.view());

    let mut cfg = Config::recommended_tip4p(n);
    cfg.rigid_body_geometry = Some(gmin_water_geometry());

    let stop = Arc::new(AtomicBool::new(false));
    let objective = Counted {
        model: GminTip4p::new(n),
        budget,
        reference,
        hit_tol,
        stop: Arc::clone(&stop),
        seen: Mutex::new(Seen {
            calls: 0,
            lowest: f64::INFINITY,
            first_hit_call: None,
        }),
        pot_ns: AtomicU64::new(0),
    };
    let mut ledger = Ledger::with_stop(budget, stop);

    let started = Instant::now();
    let out: Outcome = search_from(&objective, &cfg, &mut ledger, start.view(), seed)
        .expect("finite start")
        .0;
    let wall_s = started.elapsed().as_secs_f64();
    let inside_s = objective.pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_s = (wall_s - inside_s).max(0.0);
    let seen = objective.seen.into_inner().unwrap();

    let hit = anneal_core::energy_hits(seen.lowest, reference, hit_tol);
    assert_eq!(
        hit,
        seen.first_hit_call.is_some(),
        "hit flag disagreed with first_hit_call"
    );

    let best_str = exact_f64_string(seen.lowest);
    let best_bits = format!("{:016x}", seen.lowest.to_bits());
    let outcome_best_str = exact_f64_string(out.best);
    let outcome_best_bits = format!("{:016x}", out.best.to_bits());
    let calls_to_hit = seen
        .first_hit_call
        .map(|c| c.to_string())
        .unwrap_or_else(|| "-".to_string());

    println!(
        "gate_row system=tip4p{n} seed={seed} budget={budget} hit={} calls={} calls_to_hit={calls_to_hit} charged={} wall_s={wall_s:.9} inside_s={inside_s:.9} outside_s={outside_s:.9} best={best_str} best_bits={best_bits} outcome_best={outcome_best_str} outcome_best_bits={outcome_best_bits} hops={} basins={} start_sha256={start_sha256}",
        u8::from(hit),
        seen.calls,
        ledger.spent(),
        out.hops,
        out.basins,
    );
}
