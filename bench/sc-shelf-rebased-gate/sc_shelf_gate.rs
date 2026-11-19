#![forbid(unsafe_code)]
// Copyright (c) anneal developers
//! One recommended-preset Sutton-Chen seed.
//!
//! ```text
//! sc_shelf_gate run SYSTEM SEED
//! sc_shelf_gate budget SYSTEM
//! sc_shelf_gate coords SYSTEM SEED PATH
//! ```
//!
//! The systems, budgets and references are the Cambridge rows named by
//! `scripts/metal_h2h/run_sc.py`. The potential is the in-tree reduced
//! Sutton-Chen potential. The search is `Config::recommended` with a
//! charged ledger. A hit is a quenched energy at or below the reference
//! plus 0.001. A miss is the full budget.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{SuttonChen, SuttonChenFamily};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
    family: SuttonChenFamily,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => ExitCode::from(code),
    }
}

fn run() -> Result<(), u8> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or_else(|| {
        eprintln!("usage: sc_shelf_gate run|budget|coords ...");
        2
    })?;
    match command.as_str() {
        "run" => cmd_run(&mut args),
        "budget" => cmd_budget(&mut args),
        "coords" => cmd_coords(&mut args),
        _ => {
            eprintln!("unknown command {command}");
            Err(2)
        }
    }
}

fn spec_named(name: &str) -> Result<Spec, u8> {
    // Printed Cambridge table. Doye and Wales, New J. Chem. 22, 733 (1998).
    // Budgets are the Lennard-Jones paper map: 400000 at 38 and 1000000 at
    // 55 and at 80.
    let (n, budget, reference, family) = match name {
        "sc126-38" => (38, 400_000, -35419.9804, SuttonChenFamily::Sc12_6),
        "sc126-55" => (55, 1_000_000, -52884.6806, SuttonChenFamily::Sc12_6),
        "sc126-80" => (80, 1_000_000, -78414.6271, SuttonChenFamily::Sc12_6),
        "sc108-38" => (38, 400_000, -10117.2454, SuttonChenFamily::Sc10_8),
        "sc108-55" => (55, 1_000_000, -14814.5225, SuttonChenFamily::Sc10_8),
        "sc108-80" => (80, 1_000_000, -21819.9952, SuttonChenFamily::Sc10_8),
        "sc96-38" => (38, 400_000, -8917.7056, SuttonChenFamily::Sc9_6),
        "sc96-55" => (55, 1_000_000, -13217.8963, SuttonChenFamily::Sc9_6),
        "sc96-80" => (80, 1_000_000, -19560.8966, SuttonChenFamily::Sc9_6),
        _ => {
            eprintln!("unknown system {name}");
            return Err(2);
        }
    };
    Ok(Spec {
        n,
        budget,
        reference,
        family,
    })
}

fn cmd_budget(args: &mut impl Iterator<Item = String>) -> Result<(), u8> {
    let name = args.next().ok_or_else(|| {
        eprintln!("budget needs a system");
        2
    })?;
    if args.next().is_some() {
        eprintln!("budget takes one system");
        return Err(2);
    }
    let spec = spec_named(&name)?;
    println!("{}", spec.budget);
    println!("{:.10}", spec.reference);
    Ok(())
}

fn start_of(n: usize, seed: u64) -> Array1<f64> {
    let cfg = Config::recommended(n);
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn cmd_coords(args: &mut impl Iterator<Item = String>) -> Result<(), u8> {
    let name = args.next().ok_or_else(|| {
        eprintln!("coords needs a system");
        2
    })?;
    let seed = required_seed(args)?;
    let path = args.next().ok_or_else(|| {
        eprintln!("coords needs a path");
        2
    })?;
    if args.next().is_some() {
        eprintln!("coords takes a system, a seed and a path");
        return Err(2);
    }
    let spec = spec_named(&name)?;
    let start = start_of(spec.n, seed);
    write_coords(&path, &start).map_err(|err| {
        eprintln!("coords: {err}");
        1
    })
}

fn cmd_run(args: &mut impl Iterator<Item = String>) -> Result<(), u8> {
    let name = args.next().ok_or_else(|| {
        eprintln!("run needs a system");
        2
    })?;
    let seed = required_seed(args)?;
    if args.next().is_some() {
        eprintln!("run takes a system and a seed");
        return Err(2);
    }
    let spec = spec_named(&name)?;
    let cfg = Config::recommended(spec.n);
    let start = start_of(spec.n, seed);
    let (outcome, wall_s, pot_s) = search(&cfg, &spec, &start, seed)?;
    let level = spec.reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(outcome.charged)
    } else {
        spec.budget
    };
    println!("n,seed,budget,hit,evals_to_hit,charged,best,wall_s,pot_s");
    println!(
        "{n},{seed},{budget},{},{evals},{},{:.12},{wall_s:.6},{pot_s:.6}",
        u8::from(hit),
        outcome.charged,
        outcome.best,
        n = spec.n,
        budget = spec.budget,
    );
    Ok(())
}

fn search(
    cfg: &Config,
    spec: &Spec,
    start: &Array1<f64>,
    seed: u64,
) -> Result<(anneal_core::methods::cluster_hopping::Outcome, f64, f64), u8> {
    let potential = SuttonChen::new(spec.n, spec.family);
    let mut ledger = Ledger::new(spec.budget);
    ledger.stop_below(spec.reference + HIT_TOL);
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
    .map_err(|err| {
        eprintln!("search failed: {err}");
        1
    })?;
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    Ok((outcome, started.elapsed().as_secs_f64(), pot_s))
}

fn required_seed(args: &mut impl Iterator<Item = String>) -> Result<u64, u8> {
    let Some(text) = args.next() else {
        eprintln!("missing seed");
        return Err(2);
    };
    text.parse::<u64>().map_err(|_| {
        eprintln!("seed is not an integer");
        2
    })
}

fn write_coords(path: &str, coords: &Array1<f64>) -> std::io::Result<()> {
    let mut text = String::new();
    for atom in coords.exact_chunks(3) {
        text.push_str(&format!(
            "{:.10} {:.10} {:.10}\n",
            atom[0], atom[1], atom[2]
        ));
    }
    std::fs::write(path, text)
}
