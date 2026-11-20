// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Lennard-Jones size, with switches
//! for the symmetry completion.
//!
//! ```text
//! anneal_lj_run run <system> <seed>
//! ```
//!
//! The run charges the ledger one call for each energy-and-gradient pair of
//! a quench and one for each gradient the climb reads, as `h2h_gmin_proof`
//! does, and prints the same row. It stops at the first quenched energy at
//! or below the reference plus 0.001 unless `H2H_NOSTOP` is set, which runs
//! the whole budget so two builds can be compared call for call after a hit.
//! `H2H_BUDGET` replaces the budget. `H2H_COMPLETION=off` runs without
//! the symmetry completion of a new basin, `once` completes each basin
//! once per chain. `BITS` on stderr gives the bits of the best energy
//! and the charged count, so two builds can be compared exactly.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    let (n, reference, budget) = match name {
        "lj13" => (13, -44.326801, 20_000),
        "lj38" => (38, -173.928427, 400_000),
        "lj55" => (55, -279.248470, 1_000_000),
        "lj75" => (75, -397.492331, 4_000_000),
        "lj98" => (98, -543.665361, 4_000_000),
        "lj104" => (104, -582.086642, 4_000_000),
        _ => return None,
    };
    Some(Spec {
        n,
        budget,
        reference,
    })
}

fn switch(cfg: &mut Config) {
    match std::env::var("H2H_COMPLETION").as_deref() {
        Ok("off") => cfg.symmetry_completion_on_new = false,
        Ok("once") => cfg.symmetry_completion_once = true,
        _ => {}
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("run") {
        eprintln!("usage: anneal_lj_run run <system> <seed>");
        std::process::exit(2);
    }
    let Some(spec) = args.get(1).and_then(|name| system_spec(name)) else {
        eprintln!("unknown system");
        std::process::exit(2);
    };
    let Some(seed) = args.get(2).and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        std::process::exit(2);
    };
    let budget = std::env::var("H2H_BUDGET")
        .ok()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|&budget| budget > 0)
        .unwrap_or(spec.budget);
    let mut cfg = Config::recommended(spec.n);
    switch(&mut cfg);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(spec.n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let (outcome, wall, pot_s) = search(&cfg, spec.n, &start, seed, budget, spec.reference);
    let hit = anneal_core::energy_hits(outcome.best, spec.reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, spec.reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(outcome.charged)
    } else {
        budget
    };
    eprintln!(
        "BITS best={:016x} charged={}",
        outcome.best.to_bits(),
        outcome.charged
    );
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s"
    );
    println!(
        "{},anneal,{seed},{budget},{hit},{evals},{},{wall:.6},{:.8},{},{},{},{},{pot_s:.6}",
        args[1],
        outcome.charged,
        outcome.best,
        git(&["rev-parse", "HEAD"]),
        git(&["rev-parse", "HEAD^{tree}"]),
        cpu_model(),
        cpu_pins(),
    );
}

fn search(
    cfg: &Config,
    n: usize,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
    reference: f64,
) -> (Outcome, f64, f64) {
    let potential = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    if std::env::var("H2H_NOSTOP").is_err() {
        ledger.stop_below(reference + HIT_TOL);
    }
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = std::sync::atomic::AtomicU64::new(0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let evaluated = potential.value_and_gradient(v);
            pot_ns.fetch_add(
                mark.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            Some(evaluated)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = potential.value_and_gradient(v).1;
        pot_ns.fetch_add(
            mark.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
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
    let pot_s = pot_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e9;
    (outcome, started.elapsed().as_secs_f64(), pot_s)
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|name| name.trim().replace(',', " "))
        })
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_pins() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("Cpus_allowed_list"))
                .and_then(|line| line.split(':').nth(1))
                .map(|list| list.trim().replace(',', ";"))
        })
        .unwrap_or_else(|| "unknown".into())
}
