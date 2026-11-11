// Copyright (c) anneal developers

//! One anneal row of the Lennard-Jones board.
//!
//! ```text
//! lj_board_row SIZE SEED
//! ```
//!
//! `SIZE` is 38, 55, 75, 98, or 104. `SEED` is 6400 through 6405.
//! The budget and the reference minimum are the paper values shared
//! with the ASE and AGOX arms. The search is [`Config::recommended`].
//! A hit is an energy below the reference plus `1e-4`, a maximum
//! absolute gradient component at or below `0.0514221`, and a call
//! count inside the budget. The process requires `RAYON_NUM_THREADS=1`.

use std::io::{Write, stdout};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{Config, HopStop, Ledger, optimize_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};

const HIT_TOL: f64 = 1.0e-4;
const FORCE_BAR: f64 = 0.0514221;

struct Score {
    best: f64,
    hit: u8,
    hit_calls: usize,
    hit_energy: f64,
    hit_force: f64,
    hit_state: Option<Array1<f64>>,
    last_report: Instant,
    stop: Arc<AtomicBool>,
    reference: f64,
    budget: usize,
}

impl Score {
    fn observe(&mut self, calls: usize, energy: f64, state: ArrayView1<f64>, grad: &Array1<f64>) {
        let fmax = grad.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        if self.hit == 0 && energy.is_finite() && energy < self.best {
            self.best = energy;
        }
        if self.hit == 0
            && energy.is_finite()
            && fmax.is_finite()
            && energy < self.reference + HIT_TOL
            && fmax <= FORCE_BAR
            && calls <= self.budget
        {
            self.hit = 1;
            self.hit_calls = calls;
            self.hit_energy = energy;
            self.hit_force = fmax;
            self.hit_state = Some(state.to_owned());
            self.stop.store(true, Ordering::Relaxed);
            emit(&format!(
                "progress calls={calls} best={:.6} hit=1 force={fmax:.6e}",
                self.best
            ));
        }
        let due = self.last_report.elapsed().as_secs() >= 60 || calls.is_multiple_of(20_000);
        if due {
            self.last_report = Instant::now();
            emit(&format!(
                "progress calls={calls} best={:.6} hit={}",
                self.best, self.hit
            ));
        }
    }
}

fn emit(line: &str) {
    let mut out = stdout();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

fn budget_of(n: usize) -> Option<usize> {
    match n {
        38 => Some(400_000),
        55 => Some(1_000_000),
        75 | 98 | 104 => Some(4_000_000),
        _ => None,
    }
}

fn reference_of(n: usize) -> Option<f64> {
    match n {
        38 => Some(-173.928427),
        55 => Some(-279.248470),
        75 => Some(-397.492331),
        98 => Some(-543.665361),
        104 => Some(-582.086642),
        _ => None,
    }
}

fn read_trimmed(path: &str) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Hostname and boot id, read when the run starts.
fn announce_host() {
    let hostname = read_trimmed("/proc/sys/kernel/hostname");
    let boot_id = read_trimmed("/proc/sys/kernel/random/boot_id");
    if hostname.is_empty() || boot_id.is_empty() {
        eprintln!("host identity unread");
        std::process::exit(2);
    }
    emit(&format!("hostname={hostname}"));
    emit(&format!("boot_id={boot_id}"));
    emit("boot_id_path=/proc/sys/kernel/random/boot_id");
}

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_default();
    if threads != "1" {
        eprintln!("RAYON_NUM_THREADS must be 1, got {threads:?}");
        std::process::exit(2);
    }
    let mut args = std::env::args().skip(1);
    let (Some(n_arg), Some(seed_arg)) = (args.next(), args.next()) else {
        eprintln!("usage: lj_board_row SIZE SEED");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: lj_board_row SIZE SEED");
        std::process::exit(2);
    }
    let n: usize = n_arg.parse().unwrap_or(0);
    let seed: u64 = seed_arg.parse().unwrap_or(0);
    let Some(budget) = budget_of(n) else {
        eprintln!("size {n_arg} has no paper budget");
        std::process::exit(2);
    };
    let Some(reference) = reference_of(n) else {
        eprintln!("size {n_arg} has no reference minimum");
        std::process::exit(2);
    };
    if !(6400..=6405).contains(&seed) {
        eprintln!("seed {seed} is not in 6400 through 6405");
        std::process::exit(2);
    }

    announce_host();
    let started = Instant::now();
    let cfg = Config::recommended(n);
    let digest = match cfg.resolved_sha256() {
        Ok(digest) => digest,
        Err(err) => {
            eprintln!("config digest: {err}");
            std::process::exit(1);
        }
    };
    emit(&format!(
        "code=anneal n={n} seed={seed} budget={budget} reference={reference:.6} hit_tol={HIT_TOL:.6e} force_bar={FORCE_BAR:.7} config=recommended config_sha256={digest} rayon_threads={threads}"
    ));

    let stop = Arc::new(AtomicBool::new(false));
    let mut ledger = Ledger::with_stop(budget, Arc::clone(&stop));
    let _bind = ledger.bind_stop();
    let potential = PairPotential::lennard_jones(n);
    let score = Mutex::new(Score {
        best: f64::INFINITY,
        hit: 0,
        hit_calls: 0,
        hit_energy: f64::NAN,
        hit_force: f64::NAN,
        hit_state: None,
        last_report: Instant::now(),
        stop,
        reference,
        budget,
    });
    let energy_calls = Arc::new(AtomicUsize::new(0));
    let gradient_calls = Arc::new(AtomicUsize::new(0));
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let (energy, gradient) = potential.value_and_gradient(v);
            energy_calls.fetch_add(1, Ordering::Relaxed);
            gradient_calls.fetch_add(1, Ordering::Relaxed);
            score
                .lock()
                .expect("score")
                .observe(led.spent(), energy, v, &gradient);
            Some((energy, gradient))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let (energy, gradient) = potential.value_and_gradient(v);
        energy_calls.fetch_add(1, Ordering::Relaxed);
        gradient_calls.fetch_add(1, Ordering::Relaxed);
        score
            .lock()
            .expect("score")
            .observe(led.spent(), energy, v, &gradient);
        Some(gradient)
    };
    let outcome = match optimize_with_gradient(&cfg, &mut ledger, &mut relax, Some(&mut grad), seed)
    {
        Ok(outcome) => outcome,
        Err(err) => {
            eprintln!("search: {err}");
            std::process::exit(1);
        }
    };
    let wall_s = started.elapsed().as_secs_f64();
    let score = score.lock().expect("score");
    let charged = ledger.spent();
    let mut hit = score.hit;
    let mut calls = if hit == 1 { score.hit_calls } else { charged };
    let mut stop_name = if hit == 1 { "hit" } else { "ceiling" };
    let mut exit_code = 0;
    if hit == 1 {
        let state = score.hit_state.as_ref().expect("hit stores the geometry");
        let (energy, gradient) = potential.value_and_gradient(state.view());
        energy_calls.fetch_add(1, Ordering::Relaxed);
        gradient_calls.fetch_add(1, Ordering::Relaxed);
        let fmax = gradient
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let ok = energy < reference + HIT_TOL
            && fmax <= FORCE_BAR
            && calls <= budget
            && (energy - score.hit_energy).abs() <= 1.0e-6;
        emit(&format!(
            "audit_energy={energy:.8} audit_force={fmax:.6e} audit_ok={ok}"
        ));
        if !ok {
            hit = 0;
            calls = charged;
            stop_name = "audit";
            exit_code = 1;
        }
    }
    let hop_stop = outcome.stop.as_ref().map(HopStop::code).unwrap_or("-");
    let energy_n = energy_calls.load(Ordering::Relaxed);
    let gradient_n = gradient_calls.load(Ordering::Relaxed);
    emit(&format!(
        "best={:.6} hit={hit} calls={calls} charged={charged} energy_calls={energy_n} gradient_calls={gradient_n} hit_energy={:.8} hit_force={:.6e} quenched_best={:.6} hops={} hop_stop={hop_stop} wall_s={wall_s:.6} stop={stop_name} exit_code={exit_code}",
        score.best, score.hit_energy, score.hit_force, outcome.best, outcome.hops,
    ));
    std::process::exit(exit_code);
}
