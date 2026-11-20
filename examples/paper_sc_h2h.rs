// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Sutton-Chen cluster.
//!
//! The potential is the in-tree Sutton-Chen row, the same reduced units and
//! the same analytic derivative as GMIN `sc.f90`. The search, the start, the
//! budgets and the hit rule are those of `h2h_sc`.
//!
//! ```text
//! paper_sc_h2h run <system> <seed> [coords]
//! paper_sc_h2h coords <system> <seed> <path>
//! paper_sc_h2h budget <system>
//! ```

#![forbid(unsafe_code)]

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{SuttonChen, SuttonChenFamily};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
    family: SuttonChenFamily,
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

fn family(name: &str) -> Option<SuttonChenFamily> {
    match name {
        "126" => Some(SuttonChenFamily::Sc12_6),
        "108" => Some(SuttonChenFamily::Sc10_8),
        "96" => Some(SuttonChenFamily::Sc9_6),
        _ => None,
    }
}

fn system_spec(name: &str) -> Option<Spec> {
    let (stem, n) = name.rsplit_once('-')?;
    let stem = stem.strip_prefix("sc")?;
    let n: usize = n.parse().ok()?;
    let family = family(stem)?;
    // Printed Cambridge table, four digits. Doye and Wales, New J. Chem. 22,
    // 733 (1998). 38 and 55 are the LJ paper-map sizes. 80 is not in that map.
    let (reference, budget) = match (stem, n) {
        ("126", 38) => (-35419.9804, 400_000),
        ("126", 55) => (-52884.6806, 1_000_000),
        ("126", 80) => (-78414.6271, 1_000_000),
        ("108", 38) => (-10117.2454, 400_000),
        ("108", 55) => (-14814.5225, 1_000_000),
        ("108", 80) => (-21819.9952, 1_000_000),
        ("96", 38) => (-8917.7056, 400_000),
        ("96", 55) => (-13217.8963, 1_000_000),
        ("96", 80) => (-19560.8966, 1_000_000),
        _ => return None,
    };
    Some(Spec {
        n,
        budget,
        reference,
        family,
    })
}

fn config(spec: &Spec) -> Config {
    Config::recommended(spec.n)
}

fn potential(spec: &Spec) -> SuttonChen {
    SuttonChen::new(spec.n, spec.family)
}

fn evaluate(pot: &SuttonChen, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    pot.value_and_gradient(x)
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
    let pot = potential(&spec);
    let (outcome, wall, pot_s) = search(&pot, &cfg, &spec, &start, seed, budget);
    report(&args[1], seed, budget, &outcome, wall, pot_s, spec.reference);
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
    pot: &SuttonChen,
    cfg: &Config,
    spec: &Spec,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) -> (Outcome, f64, f64) {
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
            let pair = evaluate(pot, v);
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
        let gradient = evaluate(pot, v).1;
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
    let hit = anneal_core::energy_hits(outcome.best, reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(budget)
    } else {
        budget
    };
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
    if let Ok(path) = std::env::var("H2H_BEST")
        && let Some(state) = outcome.best_state.as_ref()
        && let Err(err) = write_coords(&path, state)
    {
        eprintln!("best: {err}");
    }
    if let Ok(path) = std::env::var("H2H_FINAL")
        && let Some(state) = outcome.final_state.as_ref()
        && let Err(err) = write_coords(&path, state)
    {
        eprintln!("final: {err}");
    }
    eprintln!(
        "diag final={:.8} hops={} accepted={} screened={} returned={} restarts={} stall_escapes={} trail_escapes={} paths={} path_improvements={} stall_gain={:.6}",
        outcome.final_energy,
        outcome.hops,
        outcome.accepted,
        outcome.screened_out,
        outcome.returned,
        outcome.restarts,
        outcome.stall_escapes,
        outcome.trail_escapes,
        outcome.paths,
        outcome.path_improvements,
        outcome.stall_escape_gain,
    );
    for (name, draws, accepts, best) in &outcome.arms {
        eprintln!("arm {name} draws {draws} accepts {accepts} best {best:.8}");
    }
    for count in &outcome.proposal_counts {
        eprintln!(
            "move {} proposed {} accepted {} improved {}",
            count.move_type, count.proposed, count.accepted, count.improved
        );
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
    use super::system_spec;
    use anneal_core::methods::cluster_hopping::Config;

    #[test]
    fn the_campaign_uses_the_cambridge_table_and_the_lj_preset() {
        let sc38 = system_spec("sc108-38").unwrap();
        assert_eq!((sc38.n, sc38.budget), (38, 400_000));
        assert!((sc38.reference + 10117.2454).abs() < 1e-10);
        assert_eq!(system_spec("sc108-55").unwrap().budget, 1_000_000);
        assert_eq!(system_spec("sc126-80").unwrap().budget, 1_000_000);
        assert!(system_spec("sc108-13").is_none());
        let cfg = super::config(&sc38);
        let stock = Config::recommended(38);
        assert_eq!(cfg.temperature, stock.temperature);
        assert_eq!(cfg.basin_stride, 1);
        assert!(cfg.species.is_none());
        assert_eq!(cfg.move_library, stock.move_library);
    }
}
