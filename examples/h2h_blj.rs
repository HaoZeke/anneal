// Copyright (c) anneal developers
//! One seed of `Config::recommended` on selected Doye-Meyer BLJ systems.
//!
//! Commands:
//! - `run <system> <seed>`
//! - `coords <system> <seed> <path>`
//! - `budget <system>`

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::BinaryLennardJones;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;
const LABEL_A: u32 = 200;
const LABEL_B: u32 = 201;

struct Spec {
    n: usize,
    n_a: usize,
    sigma_bb: f64,
    budget: usize,
    reference: f64,
}

fn paper_budget(n: usize) -> Option<usize> {
    match n {
        38 => Some(400_000),
        55 => Some(1_000_000),
        98 => Some(4_000_000),
        _ => None,
    }
}

fn system_spec(name: &str) -> Option<Spec> {
    let canonical = match name {
        "blj38s105" | "blj105n38" => "blj105n38",
        "blj55s130" | "blj130n55" => "blj130n55",
        "blj115n38" => "blj115n38",
        "blj115n55" => "blj115n55",
        "blj130n38" => "blj130n38",
        "blj130n98" => "blj130n98",
        _ => return None,
    };
    let (n, n_a, sigma_bb, reference) = match canonical {
        "blj105n38" => (38, 8, 1.05, -177.2090846748),
        "blj115n38" => (38, 9, 1.15, -187.3676867453),
        "blj130n38" => (38, 13, 1.30, -191.1840192602),
        "blj115n55" => (55, 16, 1.15, -289.2931853058),
        "blj130n55" => (55, 19, 1.30, -306.7337643721),
        "blj130n98" => (98, 41, 1.30, -590.7874132760),
        _ => unreachable!(),
    };
    Some(Spec {
        n,
        n_a,
        sigma_bb,
        budget: paper_budget(n)?,
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

fn env_budget() -> Option<usize> {
    let text = std::env::var("H2H_BUDGET").ok()?;
    let budget = text.parse::<usize>().ok()?;
    (budget > 0).then_some(budget)
}

fn start_of(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn write_coords(path: &str, start: &Array1<f64>) -> Result<(), std::io::Error> {
    let mut text = String::new();
    for value in start {
        text.push_str(&format!("{value:.16}\n"));
    }
    std::fs::write(path, text)
}

fn cpu_model() -> String {
    for line in std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
    {
        if let Some(rest) = line.strip_prefix("model name\t: ") {
            return rest.to_string();
        }
    }
    String::new()
}

fn cpu_pins() -> String {
    let mut ids: Vec<_> = std::env::var("CPUS_ALLOWED_LIST")
        .ok()
        .filter(|v| !v.is_empty())
        .into_iter()
        .collect();
    if ids.is_empty() {
        let affinity = std::fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .find(|line| line.starts_with("Cpus_allowed_list:"))
            .and_then(|line| line.split_once('\t'))
            .map(|(_, value)| value.trim().to_string())
            .unwrap_or_default();
        if !affinity.is_empty() {
            ids.push(affinity);
        }
    }
    ids.join(" ")
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_default()
}

fn search(
    cfg: &Config,
    spec: &Spec,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) -> (Outcome, f64, f64) {
    let sigma_ab = 0.5 * (1.0 + spec.sigma_bb);
    let potential = BinaryLennardJones::new(
        spec.n_a,
        spec.n - spec.n_a,
        1.0,
        1.0,
        1.0,
        1.0,
        sigma_ab,
        spec.sigma_bb,
    );
    let mut ledger = Ledger::new(budget);
    if std::env::var_os("H2H_FORCE_FULL").is_none() {
        ledger.stop_below(spec.reference + HIT_TOL);
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

fn evals_to_hit(hit: bool, calls_at_hit: Option<usize>, budget: usize) -> usize {
    if hit {
        calls_at_hit.unwrap_or(budget)
    } else {
        budget
    }
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("run needs a system");
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let budget = env_budget().unwrap_or(spec.budget);
    let start = start_of(&labelled(&spec), spec.n, seed);
    if let Some(path) = args.get(3)
        && let Err(err) = write_coords(path, &start)
    {
        eprintln!("coords: {err}");
        return 1;
    }
    let (outcome, wall, pot_s) = search(&labelled(&spec), &spec, &start, seed, budget);
    let hit = anneal_core::energy_hits(outcome.best, spec.reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, spec.reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let evals = evals_to_hit(hit, at_hit.or(hit.then_some(outcome.charged)), budget);
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s"
    );
    println!(
        "{system},anneal,{seed},{},{hit},{evals},{},{wall:.6},{:.8},{},{},{},{},{pot_s:.6}",
        budget,
        outcome.charged,
        outcome.best,
        git(&["rev-parse", "HEAD"]),
        git(&["rev-parse", "HEAD^{tree}"]),
        cpu_model(),
        cpu_pins(),
    );
    0
}

fn cmd_budget(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("budget needs a system");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    println!("{}", spec.budget);
    println!("{}", spec.reference);
    0
}

fn cmd_coords(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("coords needs a system");
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("coords needs a seed");
        return 2;
    };
    let Some(path) = args.get(3) else {
        eprintln!("coords needs a path");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    match write_coords(path, &start_of(&labelled(&spec), spec.n, seed)) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("coords: {err}");
            1
        }
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        Some("budget") => cmd_budget(&args[1..]),
        Some("coords") => cmd_coords(&args[1..]),
        _ => {
            eprintln!("usage: h2h_blj <run|budget|coords> ...");
            2
        }
    };
    std::process::exit(code);
}
