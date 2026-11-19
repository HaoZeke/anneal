// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Doye-Meyer binary Lennard-Jones cluster.
//!
//! ```text
//! h2h_blj run <system> <seed>
//! h2h_blj budget <system>
//! ```
//!
//! The configuration is `Config::recommended` with two species labels and
//! no diameters. The labels are outside the covalent-radius table, so the
//! search keeps the one-component preset and adds the generic unlike-pair
//! exchange. Every well depth is 1, `sigma_AA` is 1, and `sigma_AB` is the
//! arithmetic mean of the like diameters. `pot_s` is the time spent inside
//! the potential. A hit is a quenched energy at or below the Cambridge
//! Cluster Database header plus 0.001. A miss is the full budget.

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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "budget" => cmd_budget(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

struct Spec {
    n: usize,
    n_a: usize,
    sigma_bb: f64,
    budget: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    // Doye and Meyer, Cambridge Cluster Database binary Lennard-Jones.
    // Budgets are the matching sizes on the GMIN panel: 4e5 at 38, 1e6 at
    // 55, and 4e6 at 98.
    let (n, n_a, sigma_bb, reference) = match name {
        "blj105n38" => (38, 8, 1.05, -177.2090846748),
        "blj105n55" => (55, 13, 1.05, -283.1545313810),
        "blj115n38" => (38, 9, 1.15, -187.3676867453),
        "blj115n55" => (55, 16, 1.15, -289.2931853058),
        "blj130n38" => (38, 13, 1.30, -191.1840192602),
        "blj130n55" => (55, 19, 1.30, -306.7337643721),
        "blj130n98" => (98, 41, 1.30, -590.7874132760),
        _ => return None,
    };
    Some(Spec {
        n,
        n_a,
        sigma_bb,
        budget: paper_budget(n)?,
        reference,
    })
}

fn paper_budget(n: usize) -> Option<usize> {
    match n {
        38 => Some(400_000),
        55 => Some(1_000_000),
        98 => Some(4_000_000),
        _ => None,
    }
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

fn env_budget() -> Option<usize> {
    let text = std::env::var("H2H_BUDGET").ok()?;
    let budget = text.parse::<usize>().ok()?;
    (budget > 0).then_some(budget)
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
    let cfg = labelled(&spec);
    let start = start_of(&cfg, spec.n, seed);
    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed, budget);
    let level = spec.reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = evals_to_hit(hit, at_hit.or(hit.then_some(outcome.charged)), budget);
    let outside = (wall - pot_s).max(0.0);
    println!("best_bits {}", outcome.best.to_bits());
    println!("outcome_bits {}", outcome.best.to_bits());
    println!("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s");
    println!(
        "{system},anneal,{seed},{budget},{hit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6}",
        outcome.charged, outcome.best,
    );
    0
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
) -> (Outcome, f64, f64) {
    let evaluated = potential(spec);
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

fn evals_to_hit(hit: bool, calls_at_hit: Option<usize>, budget: usize) -> usize {
    if hit {
        calls_at_hit.unwrap_or(budget)
    } else {
        budget
    }
}

#[cfg(test)]
mod tests {
    use super::{LABEL_A, LABEL_B, paper_budget, system_spec};
    use anneal_core::methods::cluster_hopping::Config;

    #[test]
    fn budgets_match_the_gmin_panel_sizes() {
        assert_eq!(paper_budget(38), Some(400_000));
        assert_eq!(paper_budget(55), Some(1_000_000));
        assert_eq!(paper_budget(98), Some(4_000_000));
        assert_eq!(paper_budget(75), None);
        let blj = system_spec("blj105n38").unwrap();
        assert_eq!((blj.n, blj.n_a, blj.budget), (38, 8, 400_000));
        assert!((blj.reference + 177.2090846748).abs() < 1e-12);
        assert!((system_spec("blj130n98").unwrap().reference + 590.7874132760).abs() < 1e-12);
        assert_eq!(system_spec("blj130n98").unwrap().n_a, 41);
        assert!(system_spec("lj38").is_none());
    }

    #[test]
    fn two_species_keeps_the_preset_and_adds_no_diameters() {
        let spec = system_spec("blj105n38").unwrap();
        let stock = Config::recommended(spec.n);
        let mut labels = vec![LABEL_A; spec.n];
        for label in labels.iter_mut().skip(spec.n_a) {
            *label = LABEL_B;
        }
        let cfg = Config::recommended(spec.n).with_species(labels, None);
        assert!(cfg.species_sigma.is_none());
        assert_eq!(cfg.container, stock.container);
        assert_eq!(cfg.min_separation, stock.min_separation);
        assert_eq!(cfg.neighbour_cutoff, stock.neighbour_cutoff);
        assert_eq!(cfg.move_library, stock.move_library);
        assert_eq!(cfg.homotop_descent, stock.homotop_descent);
        assert_eq!(cfg.homotop_reorder, stock.homotop_reorder);
        assert!(!cfg.homotop_reorder);
        let species = cfg.species.expect("labels");
        assert_eq!(species.iter().filter(|z| **z == LABEL_A).count(), spec.n_a);
        assert_eq!(
            species.iter().filter(|z| **z == LABEL_B).count(),
            spec.n - spec.n_a
        );
    }
}
