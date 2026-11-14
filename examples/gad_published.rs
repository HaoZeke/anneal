// Copyright (c) anneal developers
//! One seed of `Config::recommended` for the deposited-GAD comparison.
//!
//! ```text
//! gad_published run <lj31|lj38> <seed>
//! gad_published coords <lj31|lj38> <seed> <path>
//! ```
//!
//! The ledger matches `h2h_gmin_proof`: one charge for each
//! energy-and-gradient pair inside a quench, and one charge for each
//! gradient the climb reads. The search stops when the quenched energy
//! is at or below the Cambridge minimum plus 0.001, or when 400000
//! calls are spent. That ceiling is the LJ38 entry of the elja map.
//! LJ31 is not on the map; this binary uses the same ceiling. A miss
//! is the full ceiling. `pot_s` is time inside the potential.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = 0.001;
/// LJ38 line of the elja budget map. Applied to LJ31 as well.
const BUDGET: usize = 400_000;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "coords" => cmd_coords(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

struct Spec {
    n: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    match name {
        "lj31" => Some(Spec {
            n: 31,
            reference: -133.586422,
        }),
        "lj38" => Some(Spec {
            n: 38,
            reference: -173.928427,
        }),
        _ => None,
    }
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("run needs a system");
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let budget = env_budget().unwrap_or(BUDGET);
    let cfg = Config::recommended(spec.n);
    let start = start_of(&cfg, spec.n, seed);
    let (outcome, wall, pot_s) = search(&cfg, spec.n, &start, seed, budget, spec.reference);
    let level = spec.reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = evals_to_hit(hit, at_hit.or(hit.then_some(outcome.charged)), budget);
    let outside = (wall - pot_s).max(0.0);
    let hit_bit = u8::from(hit);
    println!("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s");
    println!(
        "{system},anneal,{seed},{budget},{hit_bit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6}",
        outcome.charged, outcome.best,
    );
    0
}

fn cmd_coords(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("coords needs a system");
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|text| text.parse::<u64>().ok()) else {
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
    n: usize,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
    reference: f64,
) -> (Outcome, f64, f64) {
    let potential = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
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

#[cfg(test)]
mod tests {
    use super::{BUDGET, HIT_TOL, evals_to_hit, system_spec};
    use anneal_core::methods::cluster_hopping::{Config, MoveLibrary};

    #[test]
    fn both_sizes_use_the_lj38_ceiling_and_the_cambridge_energies() {
        let lj31 = system_spec("lj31").unwrap();
        let lj38 = system_spec("lj38").unwrap();
        assert_eq!((lj31.n, BUDGET), (31, 400_000));
        assert_eq!((lj38.n, BUDGET), (38, 400_000));
        assert!((lj31.reference + 133.586422).abs() < 1e-12);
        assert!((lj38.reference + 173.928427).abs() < 1e-12);
        assert!((HIT_TOL - 0.001).abs() < 1e-15);
        assert!(system_spec("lj55").is_none());
    }

    #[test]
    fn the_preset_is_recommended() {
        for n in [31, 38] {
            let cfg = Config::recommended(n);
            assert_eq!(cfg.move_library, MoveLibrary::LeanBurst);
            assert!(cfg.escape_on_stall);
            assert!(!cfg.surfaces.is_empty());
        }
    }

    #[test]
    fn a_miss_is_the_ceiling() {
        assert_eq!(evals_to_hit(false, None, BUDGET), BUDGET);
        assert_eq!(evals_to_hit(true, Some(1200), BUDGET), 1200);
    }
}
