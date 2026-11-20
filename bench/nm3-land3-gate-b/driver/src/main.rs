// Copyright (c) anneal developers
//! One seed of the seven Doye-Meyer binary Lennard-Jones clusters.
//!
//! ```text
//! nm3_gate_driver run <system> <seed> [start]
//! nm3_gate_driver coords <system> <seed> <path>
//! nm3_gate_driver energy <system> <path>
//! nm3_gate_driver budget <system>
//! ```
//!
//! The preset is `Config::recommended` plus species labels 200 and 201 and
//! no diameters. Budgets are the Lennard-Jones paper map for 38, 55 and 98
//! atoms. `H2H_BUDGET` is refused. A hit is a finite ledger best at or
//! below the Cambridge header plus `DEFAULT_HIT_TOL`.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::BinaryLennardJones;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;
const LABEL_A: u32 = 200;
const LABEL_B: u32 = 201;

#[derive(Clone, Copy)]
struct Spec {
    n: usize,
    n_a: usize,
    sigma_bb: f64,
    budget: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    let (n, n_a, sigma_bb, budget, reference) = match name {
        "blj105n38" => (38, 8, 1.05, 400_000, -177.2090846748),
        "blj105n55" => (55, 13, 1.05, 1_000_000, -283.1545313810),
        "blj115n38" => (38, 9, 1.15, 400_000, -187.3676867453),
        "blj115n55" => (55, 16, 1.15, 1_000_000, -289.2931853058),
        "blj130n38" => (38, 13, 1.30, 400_000, -191.1840192602),
        "blj130n55" => (55, 19, 1.30, 1_000_000, -306.7337643721),
        "blj130n98" => (98, 41, 1.30, 4_000_000, -590.7874132760),
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

fn potential(spec: Spec) -> BinaryLennardJones {
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

fn build_config(spec: Spec) -> Config {
    let mut labels = vec![LABEL_A; spec.n];
    for label in labels.iter_mut().skip(spec.n_a) {
        *label = LABEL_B;
    }
    Config::recommended(spec.n).with_species(labels, None)
}

fn full(value: f64) -> String {
    format!("{value:?}")
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "coords" => cmd_coords(&args),
        "energy" => cmd_energy(&args),
        "budget" => cmd_budget(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

fn cmd_budget(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "budget") else {
        return 2;
    };
    println!("{}", spec.budget);
    println!("{}", full(spec.reference));
    0
}

fn cmd_coords(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "coords") else {
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|value| value.parse::<u64>().ok()) else {
        eprintln!("coords needs a seed");
        return 2;
    };
    let Some(path) = args.get(3) else {
        eprintln!("coords needs a path");
        return 2;
    };
    let cfg = configured(spec);
    let start = start_of(&cfg, spec.n, seed);
    match write_coords(path, &start) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("coords: {err}");
            1
        }
    }
}

fn cmd_energy(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "energy") else {
        return 2;
    };
    let Some(path) = args.get(2) else {
        eprintln!("energy needs a coordinate file");
        return 2;
    };
    let coords = match read_coords(path) {
        Ok(coords) => coords,
        Err(err) => {
            eprintln!("energy: {err}");
            return 1;
        }
    };
    if coords.len() != 3 * spec.n {
        eprintln!("energy: expected {} numbers", 3 * spec.n);
        return 2;
    }
    let (energy, _) = potential(spec).value_and_gradient(coords.view());
    println!("{}", full(energy));
    println!("{:016x}", energy.to_bits());
    0
}

fn cmd_run(args: &[String]) -> i32 {
    if std::env::var_os("H2H_BUDGET").is_some() {
        eprintln!("H2H_BUDGET must not be set");
        return 2;
    }
    if let Err(err) = Config::check_removed_env() {
        eprintln!("{err}");
        return 2;
    }
    let Some(spec) = spec_of(args, "run") else {
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|value| value.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let cfg = configured(spec);
    let start = if let Some(path) = args.get(3) {
        match read_coords(path) {
            Ok(coords) => coords,
            Err(err) => {
                eprintln!("start: {err}");
                return 1;
            }
        }
    } else {
        start_of(&cfg, spec.n, seed)
    };
    if start.len() != 3 * spec.n {
        eprintln!("start: expected {} numbers", 3 * spec.n);
        return 2;
    }
    let (outcome, wall, pot_s) = search(&cfg, spec, &start, seed, spec.budget);
    report(&args[1], seed, spec, &outcome, wall, pot_s);
    0
}

fn spec_of(args: &[String], command: &str) -> Option<Spec> {
    let Some(name) = args.get(1).map(String::as_str) else {
        eprintln!("{command} needs a system");
        return None;
    };
    let Some(spec) = system_spec(name) else {
        eprintln!("unknown system {name}");
        return None;
    };
    Some(spec)
}

fn configured(spec: Spec) -> Config {
    build_config(spec)
}

fn start_of(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn search(
    cfg: &Config,
    spec: Spec,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) -> (Outcome, f64, f64) {
    let evaluated = potential(spec);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(spec.reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = AtomicU64::new(0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let pair = evaluated.value_and_gradient(v);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    (outcome, started.elapsed().as_secs_f64(), pot_s)
}

fn report(system: &str, seed: u64, spec: Spec, outcome: &Outcome, wall: f64, pot_s: f64) {
    let hit = anneal_core::energy_hits(outcome.best, spec.reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, spec.reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        match at_hit {
            Some(spent) if spent < spec.budget => spent,
            _ => outcome.charged.min(spec.budget),
        }
    } else {
        spec.budget
    };
    let outside = (wall - pot_s).max(0.0);
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,best_bits,pot_s,outside_s"
    );
    println!(
        "{system},anneal,{seed},{},{hit},{evals},{},{wall:.6},{},{:016x},{pot_s:.6},{outside:.6}",
        spec.budget,
        outcome.charged,
        full(outcome.best),
        outcome.best.to_bits(),
    );
}

fn write_coords(path: &str, coords: &Array1<f64>) -> std::io::Result<()> {
    let mut text = String::new();
    for atom in coords.exact_chunks(3) {
        text.push_str(&format!(
            "{:.16} {:.16} {:.16}\n",
            atom[0], atom[1], atom[2]
        ));
    }
    std::fs::write(path, text)
}

fn read_coords(path: &str) -> std::io::Result<Array1<f64>> {
    let text = std::fs::read_to_string(path)?;
    let mut values = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let numbers = if fields.len() == 4 && matches!(fields[0], "A" | "B" | "LA" | "LB" | "X") {
            &fields[1..]
        } else {
            fields.as_slice()
        };
        for word in numbers {
            if let Ok(value) = word.parse::<f64>() {
                values.push(value);
            }
        }
    }
    Ok(Array1::from_vec(values))
}
