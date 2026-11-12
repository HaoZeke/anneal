// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Cambridge Gupta cluster.
//!
//! ```text
//! h2h_metal run <system> <seed> [coords]
//! h2h_metal coords <system> <seed> <path>
//! h2h_metal energy <system> <path>
//! h2h_metal grad <system> <path>
//! h2h_metal budget <system>
//! ```
//!
//! `run` charges one call for each energy-and-gradient pair inside a quench
//! and one call for each gradient the climb reads. It stops when the quenched
//! energy is within 0.001 of the Cambridge minimum, or when the budget is
//! spent. A miss is the full budget. `pot_s` is the time inside the potential.
//! Zinc is the one-species preset. Copper–gold adds the homotop that
//! `Config::recommended` installs for two species.
//!
//! The Cambridge energies are the Gupta zinc table of Doye, Phys. Rev. B 68,
//! 195418 (2003), and the `CuAu_38` archive (Wu, Cai and Shao, J. Comput.
//! Chem. 30, 1992, 2009).

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::Gupta;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "coords" => cmd_coords(&args),
        "energy" => cmd_energy(&args),
        "grad" => cmd_grad(&args),
        "budget" => cmd_budget(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

#[derive(Clone, Copy)]
enum Kind {
    Zinc,
    CopperGold { n_copper: usize },
}

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
    kind: Kind,
}

fn system_spec(name: &str) -> Option<Spec> {
    let (n, reference, budget, kind) = match name {
        "zn38" => (38, -50.499983, 50_000, Kind::Zinc),
        "zn55" => (55, -73.310032, 80_000, Kind::Zinc),
        "zn75" => (75, -100.203580, 100_000, Kind::Zinc),
        "zn80" => (80, -106.927940, 100_000, Kind::Zinc),
        "cu19au19" => (38, -122.99148391, 80_000, Kind::CopperGold { n_copper: 19 }),
        _ => return None,
    };
    Some(Spec {
        n,
        budget,
        reference,
        kind,
    })
}

fn potential(spec: &Spec) -> Gupta {
    match spec.kind {
        Kind::Zinc => Gupta::zinc(spec.n),
        Kind::CopperGold { n_copper } => Gupta::copper_gold(n_copper, spec.n - n_copper),
    }
}

fn config(spec: &Spec) -> Config {
    match spec.kind {
        Kind::Zinc => Config::recommended(spec.n),
        Kind::CopperGold { n_copper } => {
            let species = Gupta::copper_gold_species(n_copper, spec.n - n_copper);
            Config::recommended(spec.n).with_species(species, None)
        }
    }
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "run") else {
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|value| value.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let budget = env_budget().unwrap_or(spec.budget);
    let cfg = config(&spec);
    let start = start_of(&cfg, spec.n, seed);
    if let Some(path) = args.get(3)
        && let Err(err) = write_coords(path, &start)
    {
        eprintln!("coords: {err}");
        return 1;
    }
    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed, budget);
    report(
        &args[1],
        seed,
        budget,
        &outcome,
        wall,
        pot_s,
        spec.reference,
    );
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
    let cfg = config(&spec);
    let start = start_of(&cfg, spec.n, seed);
    match write_coords(path, &start) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("coords: {err}");
            1
        }
    }
}

fn cmd_budget(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "budget") else {
        return 2;
    };
    println!("{}", spec.budget);
    println!("{:.10}", spec.reference);
    0
}

fn cmd_energy(args: &[String]) -> i32 {
    let Some((spec, coords)) = geometry_of(args, "energy") else {
        return 2;
    };
    let (energy, _) = potential(&spec).value_and_gradient(coords.view());
    println!("{energy:.12}");
    0
}

fn cmd_grad(args: &[String]) -> i32 {
    let Some((spec, coords)) = geometry_of(args, "grad") else {
        return 2;
    };
    let (energy, gradient) = potential(&spec).value_and_gradient(coords.view());
    println!("{energy:.12}");
    for atom in gradient.exact_chunks(3) {
        println!("{:.12} {:.12} {:.12}", atom[0], atom[1], atom[2]);
    }
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

fn geometry_of(args: &[String], command: &str) -> Option<(Spec, Array1<f64>)> {
    let spec = spec_of(args, command)?;
    let Some(path) = args.get(2) else {
        eprintln!("{command} needs a coordinate file");
        return None;
    };
    let coords = match read_coords(path) {
        Ok(coords) => coords,
        Err(err) => {
            eprintln!("{command}: {err}");
            return None;
        }
    };
    if coords.len() != 3 * spec.n {
        eprintln!("{command}: expected {} numbers", 3 * spec.n);
        return None;
    }
    Some((spec, coords))
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

fn report(
    system: &str,
    seed: u64,
    budget: usize,
    outcome: &Outcome,
    wall: f64,
    pot_s: f64,
    reference: f64,
) {
    let level = reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = evals_to_hit(hit, at_hit.or(hit.then_some(outcome.charged)), budget);
    let outside = (wall - pot_s).max(0.0);
    let calls = outcome.charged.max(1) as f64;
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s"
    );
    println!(
        "{system},anneal,{seed},{budget},{hit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6},{:.9},{:.9}",
        outcome.charged,
        outcome.best,
        pot_s / calls,
        outside / calls,
    );
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

fn read_coords(path: &str) -> std::io::Result<Array1<f64>> {
    let text = std::fs::read_to_string(path)?;
    let mut values = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let numbers = if fields.len() == 4 && fields[0].parse::<u32>().is_ok() {
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

#[cfg(test)]
mod tests {
    use super::{evals_to_hit, system_spec};
    use anneal_core::methods::cluster_hopping::Config;
    use anneal_core::potentials::Gupta;

    #[test]
    fn the_campaign_uses_the_cambridge_energies() {
        let zn38 = system_spec("zn38").unwrap();
        assert_eq!((zn38.n, zn38.budget), (38, 50_000));
        assert!((zn38.reference + 50.499983).abs() < 1e-12);
        assert_eq!(system_spec("zn55").unwrap().budget, 80_000);
        assert_eq!(system_spec("zn75").unwrap().budget, 100_000);
        assert_eq!(system_spec("zn80").unwrap().budget, 100_000);
        let mixed = system_spec("cu19au19").unwrap();
        assert_eq!(mixed.n, 38);
        assert!((mixed.reference + 122.99148391).abs() < 1e-12);
        assert!(system_spec("zn31").is_none());
    }

    #[test]
    fn copper_gold_uses_the_recommended_homotop() {
        let spec = system_spec("cu19au19").unwrap();
        let cfg = super::config(&spec);
        assert!(
            cfg.move_library
                .kernels(&cfg)
                .iter()
                .any(|kernel| kernel.name() == "swap")
        );
        let zinc = super::config(&system_spec("zn38").unwrap());
        assert!(
            zinc.move_library
                .kernels(&zinc)
                .iter()
                .all(|kernel| kernel.name() != "swap")
        );
        assert_eq!(
            Gupta::copper_gold_species(19, 19),
            cfg.species.clone().unwrap()
        );
    }

    #[test]
    fn a_miss_is_the_full_budget() {
        assert_eq!(evals_to_hit(false, None, 50_000), 50_000);
        assert_eq!(evals_to_hit(true, Some(1200), 50_000), 1200);
    }

    #[test]
    fn zinc_stays_on_the_stock_preset() {
        let cfg = Config::recommended(38);
        let used = super::config(&system_spec("zn38").unwrap());
        assert_eq!(used.move_library, cfg.move_library);
        assert_eq!(used.temperature, cfg.temperature);
        assert!(used.species.is_none());
    }
}
