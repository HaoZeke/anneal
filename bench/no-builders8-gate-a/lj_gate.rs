// Copyright (c) anneal developers
//! One recommended-preset Lennard-Jones seed for the paired gate.
//!
//! ```text
//! lj_gate coords N SEED PATH
//! lj_gate run N SEED BUDGET REFERENCE PATH
//! ```
//!
//! `coords` writes the start `Config::recommended` would quench for that
//! seed. `run` quenches the coordinates in `PATH` and does not draw a new
//! start. The search is `Config::recommended` with the rgpot Lennard-Jones
//! kernel. A hit is a finite quenched energy at or below the reference plus
//! the crate's default tolerance. A miss is reported as the full budget.
//! Time inside the potential is the sum of the charged evaluations.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

fn main() {
    refuse_overrides();
    anneal_core::install_rgpot_kernels!();
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_default();
    let status = match command.as_str() {
        "coords" => cmd_coords(&mut args),
        "run" => cmd_run(&mut args),
        _ => {
            eprintln!("usage: lj_gate coords N SEED PATH | lj_gate run N SEED BUDGET REFERENCE PATH");
            2
        }
    };
    std::process::exit(status);
}

fn cmd_coords(args: &mut impl Iterator<Item = String>) -> i32 {
    let Some(n) = parse_usize(args.next().as_deref(), "N") else {
        return 2;
    };
    let Some(seed) = parse_u64(args.next().as_deref(), "SEED") else {
        return 2;
    };
    let Some(path) = args.next() else {
        eprintln!("coords needs a path");
        return 2;
    };
    if args.next().is_some() || n < 2 {
        eprintln!("coords arguments are unusable");
        return 2;
    }
    let cfg = Config::recommended(n);
    let start = start_of(&cfg, n, seed);
    if let Err(err) = write_start(&path, n, seed, &start) {
        eprintln!("coords: {err}");
        return 1;
    }
    println!("start_sha256={} n={n} seed={seed}", start_sha256(&start));
    0
}

fn cmd_run(args: &mut impl Iterator<Item = String>) -> i32 {
    let Some(n) = parse_usize(args.next().as_deref(), "N") else {
        return 2;
    };
    let Some(seed) = parse_u64(args.next().as_deref(), "SEED") else {
        return 2;
    };
    let Some(budget) = parse_usize(args.next().as_deref(), "BUDGET") else {
        return 2;
    };
    let Some(reference_text) = args.next() else {
        eprintln!("run needs a reference");
        return 2;
    };
    let Some(path) = args.next() else {
        eprintln!("run needs a start path");
        return 2;
    };
    if args.next().is_some() || n < 2 || budget == 0 {
        eprintln!("run arguments are unusable");
        return 2;
    }
    let Ok(reference) = reference_text.parse::<f64>() else {
        eprintln!("reference is not a number");
        return 2;
    };
    if !reference.is_finite() {
        eprintln!("reference is not finite");
        return 2;
    }
    let start = match read_start(&path, n, seed) {
        Ok(start) => start,
        Err(err) => {
            eprintln!("start: {err}");
            return 1;
        }
    };
    let affinity = match cpu_affinity() {
        Ok(affinity) => affinity,
        Err(err) => {
            eprintln!("affinity: {err}");
            return 1;
        }
    };
    if affinity != "0" {
        eprintln!("affinity {affinity} is not core 0");
        return 1;
    }
    let cfg = Config::recommended(n);
    let (outcome, wall_s, inside_s) = search(&cfg, n, &start, seed, budget, reference);
    let hit = anneal_core::energy_hits(outcome.best, reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let calls = if hit {
        at_hit.unwrap_or(outcome.charged)
    } else {
        budget
    };
    let outside_s = wall_s - inside_s;
    println!(
        "gate_row n={n} seed={seed} budget={budget} hit={} calls={calls} charged={} best_bits={} reference_bits={} wall_bits={} inside_bits={} outside_bits={} start_sha256={} affinity={affinity}",
        u8::from(hit),
        outcome.charged,
        bits(outcome.best),
        bits(reference),
        bits(wall_s),
        bits(inside_s),
        bits(outside_s),
        start_sha256(&start),
    );
    0
}

fn search(
    cfg: &Config,
    n: usize,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
    reference: f64,
) -> (anneal_core::methods::cluster_hopping::Outcome, f64, f64) {
    let potential = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = AtomicU64::new(0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let evaluated = potential.value_and_gradient(point);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Some(evaluated)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = potential.value_and_gradient(point).1;
        pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(gradient)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let inside_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    (outcome, started.elapsed().as_secs_f64(), inside_s)
}

fn start_of(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn write_start(path: &str, n: usize, seed: u64, coords: &Array1<f64>) -> std::io::Result<()> {
    if coords.len() != 3 * n {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "start length does not match N",
        ));
    }
    let mut text = format!("n {n}\nseed {seed}\n");
    for value in coords.iter() {
        text.push_str(&bits(*value));
        text.push('\n');
    }
    std::fs::write(path, text)
}

fn read_start(path: &str, n: usize, seed: u64) -> Result<Array1<f64>, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    let mut lines = text.lines();
    let header_n = lines
        .next()
        .ok_or("start file is empty")?
        .strip_prefix("n ")
        .ok_or("start file has no n")?;
    let header_seed = lines
        .next()
        .ok_or("start file has no seed")?
        .strip_prefix("seed ")
        .ok_or("start file has no seed")?;
    if header_n != n.to_string() || header_seed != seed.to_string() {
        return Err("start file does not match N and SEED".to_string());
    }
    let mut values = Vec::with_capacity(3 * n);
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bits = u64::from_str_radix(line, 16).map_err(|_| "start coordinate is not hex")?;
        values.push(f64::from_bits(bits));
    }
    if values.len() != 3 * n {
        return Err("start file has the wrong number of coordinates".to_string());
    }
    Ok(Array1::from_vec(values))
}

fn start_sha256(coords: &Array1<f64>) -> String {
    let mut hasher = Sha256::new();
    for value in coords.iter() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn cpu_affinity() -> Result<String, String> {
    let text = std::fs::read_to_string("/proc/self/status").map_err(|err| err.to_string())?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
            return Ok(rest.trim().to_string());
        }
    }
    Err("Cpus_allowed_list is missing".to_string())
}

fn refuse_overrides() {
    let mut bad = Vec::new();
    for (key, value) in std::env::vars() {
        if key.starts_with("H2H_") {
            bad.push(key);
            continue;
        }
        if matches!(
            key.as_str(),
            "OMP_NUM_THREADS"
                | "MKL_NUM_THREADS"
                | "OPENBLAS_NUM_THREADS"
                | "NUMBA_NUM_THREADS"
                | "VECLIB_MAXIMUM_THREADS"
                | "BLIS_NUM_THREADS"
                | "RAYON_NUM_THREADS"
                | "TOKIO_WORKER_THREADS"
                | "POLARS_MAX_THREADS"
                | "NUMEXPR_NUM_THREADS"
                | "GOTO_NUM_THREADS"
                | "TBB_NUM_THREADS"
        ) && value != "1"
        {
            bad.push(key);
        }
    }
    if !bad.is_empty() {
        eprintln!("refusing thread or override environment: {}", bad.join(","));
        std::process::exit(2);
    }
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        if std::env::var(key).ok().as_deref() != Some("1") {
            eprintln!("missing {key}=1");
            std::process::exit(2);
        }
    }
}

fn parse_usize(text: Option<&str>, name: &str) -> Option<usize> {
    let Some(text) = text else {
        eprintln!("missing {name}");
        return None;
    };
    match text.parse::<usize>() {
        Ok(value) => Some(value),
        Err(_) => {
            eprintln!("{name} is not an integer");
            None
        }
    }
}

fn parse_u64(text: Option<&str>, name: &str) -> Option<u64> {
    let Some(text) = text else {
        eprintln!("missing {name}");
        return None;
    };
    match text.parse::<u64>() {
        Ok(value) => Some(value),
        Err(_) => {
            eprintln!("{name} is not an integer");
            None
        }
    }
}
