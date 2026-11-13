// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Cambridge Morse cluster.
//!
//! ```text
//! h2h_morse run <system> <seed>
//! h2h_morse coords <system> <seed> <path>
//! h2h_morse budget <system>
//! ```
//!
//! The potential is the in-tree Morse pair, Doye and Wales' form with
//! `epsilon = 1` and `r_0 = 1`. `rho` is the range. When a second neighbour
//! is still bound, the search quenches one Mackay cap before the walk.
//! Otherwise it is the recommended preset. A hit is a quenched energy at
//! or below the Cambridge table plus 0.001. `pot_s` is the time inside
//! the potential.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

struct Spec {
    n: usize,
    rho: f64,
    budget: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    // Cambridge Cluster Database, Doye and Wales, Morse table.
    // Columns are rho = 3, 6, 10, 14. A bold entry is the global minimum
    // at that range. Budgets follow the Lennard-Jones paper list: 4e5 at
    // 38, 1e6 at 55, and 4e6 at 75 and above.
    let (n, rho, reference) = match name {
        "m38r6" => (38, 6.0, -157.477108),
        "m38r14" => (38, 14.0, -144.321054),
        "m55r6" => (55, 6.0, -250.286609),
        "m55r14" => (55, 14.0, -220.646208),
        "m80r14" => (80, 14.0, -340.811371),
        _ => return None,
    };
    let budget = match n {
        38 => 400_000,
        55 => 1_000_000,
        80 => 4_000_000,
        _ => return None,
    };
    Some(Spec {
        n,
        rho,
        budget,
        reference,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "coords" => cmd_coords(&args),
        "budget" => cmd_budget(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
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
    println!("{:.10}", spec.reference);
    println!("{}", spec.rho);
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
    let cfg = Config::recommended(spec.n);
    let start = start_of(&cfg, spec.n, seed);
    match write_coords(path, &start) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("coords: {err}");
            1
        }
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
    let mut cfg = Config::recommended(spec.n);
    // A second neighbour in an fcc shell. Bound by more than a twentieth of
    // the well, one Mackay cap is quenched before the walk. Rho = 14 is
    // not, and the search is then the recommended one.
    let kind = PairKind::Morse { rho: spec.rho };
    cfg.long_ranged = kind.second_neighbour_energy() < -0.05;
    let start = if let Some(path) = args.get(3) {
        match read_coords(path) {
            Ok(coords) if coords.len() == 3 * spec.n => coords,
            Ok(coords) => {
                eprintln!("coords length {} is not {}", coords.len(), 3 * spec.n);
                return 1;
            }
            Err(err) => {
                eprintln!("coords: {err}");
                return 1;
            }
        }
    } else {
        start_of(&cfg, spec.n, seed)
    };
    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed, budget);
    let level = spec.reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(budget)
    } else {
        budget
    };
    println!("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s");
    let outside = (wall - pot_s).max(0.0);
    println!(
        "{system},anneal,{seed},{budget},{hit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6}",
        outcome.charged, outcome.best,
    );
    for (name, draws, accepts, best) in &outcome.arms {
        println!("arm {name} draws {draws} accepts {accepts} best {best:.6}");
    }
    0
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

fn search(
    cfg: &Config,
    spec: &Spec,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) -> (anneal_core::methods::cluster_hopping::Outcome, f64, f64) {
    let potential = PairPotential::morse(spec.n, spec.rho);
    let mut ledger = Ledger::new(budget);
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

fn read_coords(path: &str) -> std::io::Result<Array1<f64>> {
    let text = std::fs::read_to_string(path)?;
    let mut values = Vec::new();
    for line in text.lines() {
        for word in line.split_whitespace() {
            if let Ok(value) = word.parse::<f64>() {
                values.push(value);
            }
        }
    }
    Ok(Array1::from_vec(values))
}

#[cfg(test)]
mod tests {
    use super::system_spec;

    #[test]
    fn the_cambridge_morse_table_is_the_hit_target() {
        let m38_6 = system_spec("m38r6").unwrap();
        assert_eq!((m38_6.n, m38_6.budget), (38, 400_000));
        assert!((m38_6.rho - 6.0).abs() < 1e-12);
        assert!((m38_6.reference + 157.477108).abs() < 1e-12);
        let m38_14 = system_spec("m38r14").unwrap();
        assert!((m38_14.reference + 144.321054).abs() < 1e-12);
        let m55_6 = system_spec("m55r6").unwrap();
        assert_eq!(m55_6.budget, 1_000_000);
        assert!((m55_6.reference + 250.286609).abs() < 1e-12);
        let m55_14 = system_spec("m55r14").unwrap();
        assert!((m55_14.reference + 220.646208).abs() < 1e-12);
        let m80 = system_spec("m80r14").unwrap();
        assert_eq!((m80.n, m80.budget), (80, 4_000_000));
        assert!((m80.reference + 340.811371).abs() < 1e-12);
        assert!(m80.n >= 6);
    }
}
