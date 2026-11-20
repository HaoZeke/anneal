// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Doye-Meyer binary Lennard-Jones
//! cluster, scored against the Cambridge Cluster Database header.
//!
//! ```text
//! h2h_blj run <system> <seed>
//! ```
//!
//! The configuration is `Config::recommended` with two species labels and no
//! diameters. The potential is `BinaryLennardJones` with every epsilon 1,
//! sigma_AA 1 and sigma_AB the mean of the two diameters, through the rgpot
//! kernels. A hit is a quenched energy at or below the header plus 0.001. A
//! miss is the full budget. The budgets are the sizes of the Lennard-Jones
//! paper map: 4e5 at 38, 1e6 at 55 and 4e6 at 98.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::BinaryLennardJones;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = 0.001;
const LABEL_A: u32 = 200;
const LABEL_B: u32 = 201;

struct Spec {
    n: usize,
    n_a: usize,
    sigma_bb: f64,
    budget: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    let (n, n_a, sigma_bb, reference, budget) = match name {
        "blj105n38" => (38, 8, 1.05, -177.2090846748, 400_000),
        "blj105n55" => (55, 13, 1.05, -283.1545313810, 1_000_000),
        "blj115n38" => (38, 9, 1.15, -187.3676867453, 400_000),
        "blj115n55" => (55, 16, 1.15, -289.2931853058, 1_000_000),
        "blj130n38" => (38, 13, 1.30, -191.1840192602, 400_000),
        "blj130n55" => (55, 19, 1.30, -306.7337643721, 1_000_000),
        "blj130n98" => (98, 41, 1.30, -590.7874132760, 4_000_000),
        _ => return None,
    };
    Some(Spec {
        n,
        n_a,
        sigma_bb,
        budget,
        reference,
    })
}

fn labelled(spec: &Spec) -> Config {
    let mut labels = vec![LABEL_A; spec.n];
    for label in labels.iter_mut().skip(spec.n_a) {
        *label = LABEL_B;
    }
    Config::recommended(spec.n).with_species(labels, None)
}

fn potential(spec: &Spec) -> BinaryLennardJones {
    BinaryLennardJones::new(
        spec.n_a,
        spec.n - spec.n_a,
        1.0,
        1.0,
        1.0,
        1.0,
        0.5 * (1.0 + spec.sigma_bb),
        spec.sigma_bb,
    )
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 || args[0] != "run" {
        eprintln!("usage: h2h_blj run <system> <seed>");
        std::process::exit(2);
    }
    let Some(spec) = system_spec(&args[1]) else {
        eprintln!("unknown system {}", args[1]);
        std::process::exit(2);
    };
    let Ok(seed) = args[2].parse::<u64>() else {
        eprintln!("bad seed {}", args[2]);
        std::process::exit(2);
    };
    let cfg = labelled(&spec);
    let mut start_rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        spec.n,
        cfg.start_radius(),
        cfg.min_separation,
        &mut start_rng,
    );
    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed);
    let level = spec.reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(outcome.charged).min(spec.budget)
    } else {
        spec.budget
    };
    println!("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s");
    println!(
        "{},anneal,{seed},{},{hit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{:.6}",
        args[1],
        spec.budget,
        outcome.charged,
        outcome.best,
        (wall - pot_s).max(0.0),
    );
}

fn search(cfg: &Config, spec: &Spec, start: &Array1<f64>, seed: u64) -> (Outcome, f64, f64) {
    let evaluated = potential(spec);
    let mut ledger = Ledger::new(spec.budget);
    ledger.stop_below(spec.reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = std::sync::atomic::AtomicU64::new(0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let pair = evaluated.value_and_gradient(v);
            pot_ns.fetch_add(
                mark.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            Some(pair)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = evaluated.value_and_gradient(v).1;
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
