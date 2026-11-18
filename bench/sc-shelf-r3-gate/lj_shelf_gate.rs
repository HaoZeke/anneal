// Copyright (c) anneal developers
//! One recommended-preset Lennard-Jones seed.
//!
//! ```text
//! lj_shelf_gate N SEED BUDGET REFERENCE
//! ```
//!
//! All four arguments are required. The search is `Config::recommended`
//! with the rgpot Lennard-Jones kernel, a charged ledger, and the same
//! stop as the paper head-to-head: a quenched energy at or below the
//! reference plus 0.001, or the budget. A miss is reported as the full
//! budget. Evaluations inside the potential are timed separately from
//! the wall clock.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

fn main() -> ExitCode {
    anneal_core::install_rgpot_kernels!();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => ExitCode::from(code),
    }
}

fn run() -> Result<(), u8> {
    let mut args = std::env::args().skip(1);
    let n = required_usize(&mut args, "N")?;
    let seed = required_u64(&mut args, "SEED")?;
    let budget = required_usize(&mut args, "BUDGET")?;
    let reference = required_f64(&mut args, "REFERENCE")?;
    if args.next().is_some() {
        eprintln!("usage: lj_shelf_gate N SEED BUDGET REFERENCE");
        return Err(2);
    }
    if n < 2 || budget == 0 || !reference.is_finite() {
        eprintln!("N, BUDGET and REFERENCE are unusable");
        return Err(2);
    }

    let cfg = Config::recommended(n);
    let start = {
        let mut rng = StdRng::seed_from_u64(seed);
        random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
    };
    let (outcome, wall_s, pot_s) = search(&cfg, n, &start, seed, budget, reference)?;
    let level = reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(outcome.charged)
    } else {
        budget
    };
    println!("n,seed,budget,hit,evals_to_hit,charged,best,wall_s,pot_s");
    println!(
        "{n},{seed},{budget},{},{evals},{},{:.12},{wall_s:.6},{pot_s:.6}",
        u8::from(hit),
        outcome.charged,
        outcome.best,
    );
    Ok(())
}

fn search(
    cfg: &Config,
    n: usize,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
    reference: f64,
) -> Result<(anneal_core::methods::cluster_hopping::Outcome, f64, f64), u8> {
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
    .map_err(|err| {
        eprintln!("search failed: {err}");
        1
    })?;
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    Ok((outcome, started.elapsed().as_secs_f64(), pot_s))
}

fn required_usize(args: &mut impl Iterator<Item = String>, name: &str) -> Result<usize, u8> {
    let Some(text) = args.next() else {
        eprintln!("missing {name}");
        return Err(2);
    };
    text.parse::<usize>().map_err(|_| {
        eprintln!("{name} is not an integer");
        2
    })
}

fn required_u64(args: &mut impl Iterator<Item = String>, name: &str) -> Result<u64, u8> {
    let Some(text) = args.next() else {
        eprintln!("missing {name}");
        return Err(2);
    };
    text.parse::<u64>().map_err(|_| {
        eprintln!("{name} is not an integer");
        2
    })
}

fn required_f64(args: &mut impl Iterator<Item = String>, name: &str) -> Result<f64, u8> {
    let Some(text) = args.next() else {
        eprintln!("missing {name}");
        return Err(2);
    };
    text.parse::<f64>().map_err(|_| {
        eprintln!("{name} is not a number");
        2
    })
}
