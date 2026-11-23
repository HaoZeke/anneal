// Copyright (c) anneal developers
//! One seed of anneal on the ABCluster head-to-head panel.
//!
//! ```text
//! h2h_abcluster run <system> <seed> <preset>
//! h2h_abcluster coords <system> <seed> <path>
//! h2h_abcluster energy <system> <path>
//! h2h_abcluster budget <system>
//! ```
//!
//! Systems are the Cambridge Lennard-Jones minima LJ38, LJ55, LJ75, LJ98 and
//! LJ104, and Gupta Au55 with the Cleri-Rosato gold parameters. `preset` is
//! `recommended` for [`Config::recommended`] or `cluster` for
//! [`Config::for_cluster`]. Gupta Au55 adds the species table of
//! `Gupta::copper_gold_species` and one exact basin step in eight, as
//! `h2h_metal` does.
//!
//! `run` charges the ledger the way `h2h_gmin_proof` does: one charge for
//! each energy-and-gradient pair inside a quench and one for each gradient
//! the climb reads. The search stops when a quenched energy is at or below
//! the Cambridge minimum plus 0.001, or when the budget is spent. A miss is
//! reported as the full budget. `pot_s` is the time inside the potential,
//! read with `Instant::now()` on either side of each call.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Gupta, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Preset {
    Recommended,
    Cluster,
}

impl Preset {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "recommended" => Some(Self::Recommended),
            "cluster" => Some(Self::Cluster),
            _ => None,
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::Recommended => "anneal-recommended",
            Self::Cluster => "anneal-cluster",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    LennardJones,
    Gold,
}

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
    kind: Kind,
}

/// Cambridge Cluster Database energies. The Lennard-Jones budgets are the
/// paper map of `h2h_gmin_proof`; Au55 is the `CuAu_55` archive energy
/// and the 55-atom budget.
fn system_spec(name: &str) -> Option<Spec> {
    let (n, reference, budget, kind) = match name {
        "lj38" => (38, -173.928427, 400_000, Kind::LennardJones),
        "lj55" => (55, -279.248470, 1_000_000, Kind::LennardJones),
        "lj75" => (75, -397.492331, 4_000_000, Kind::LennardJones),
        "lj98" => (98, -543.665361, 4_000_000, Kind::LennardJones),
        "lj104" => (104, -582.086642, 4_000_000, Kind::LennardJones),
        "au55" => (55, -191.03085280, 1_000_000, Kind::Gold),
        _ => return None,
    };
    Some(Spec {
        n,
        budget,
        reference,
        kind,
    })
}

enum Surface {
    LennardJones(PairPotential),
    Gold(Gupta),
}

impl Surface {
    fn of(spec: &Spec) -> Self {
        match spec.kind {
            Kind::LennardJones => Self::LennardJones(PairPotential::lennard_jones(spec.n)),
            Kind::Gold => Self::Gold(Gupta::copper_gold(0, spec.n)),
        }
    }

    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Self::LennardJones(pot) => pot.value_and_gradient(x),
            Self::Gold(pot) => pot.value_and_gradient(x),
        }
    }
}

fn config(spec: &Spec, preset: Preset) -> Config {
    let base = match preset {
        Preset::Recommended => Config::recommended(spec.n),
        Preset::Cluster => Config::for_cluster(spec.n),
    };
    match spec.kind {
        Kind::LennardJones => base,
        Kind::Gold => {
            let species = Gupta::copper_gold_species(0, spec.n);
            let mut cfg = base.with_species(species, None);
            cfg.basin_stride = 8;
            cfg
        }
    }
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "run") else {
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let Some(preset) = args.get(3).and_then(|v| Preset::parse(v)) else {
        eprintln!("run needs a preset, recommended or cluster");
        return 2;
    };
    let budget = env_budget().unwrap_or(spec.budget);
    let cfg = config(&spec, preset);
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
    println!("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s");
    println!(
        "{},{},{seed},{budget},{},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6}",
        args[1],
        preset.code(),
        u8::from(hit),
        outcome.charged,
        outcome.best,
    );
    0
}

fn cmd_coords(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "coords") else {
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
    let cfg = config(&spec, Preset::Recommended);
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
    let (energy, _) = Surface::of(&spec).value_and_gradient(coords.view());
    println!("{energy:.12}");
    0
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
    let spec = system_spec(name);
    if spec.is_none() {
        eprintln!("unknown system {name}");
    }
    spec
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
    let surface = Surface::of(spec);
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
            let evaluated = surface.value_and_gradient(v);
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
        let gradient = surface.value_and_gradient(v).1;
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

/// A hit keeps the charged count at the first quenched record at or below
/// the level. A miss is the whole budget.
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
    use super::{Kind, Preset, Surface, config, evals_to_hit, system_spec};
    use anneal_core::methods::cluster_hopping::MoveLibrary;
    use ndarray::Array1;

    #[test]
    fn the_panel_uses_the_cambridge_energies_and_the_paper_budgets() {
        let table = [
            ("lj38", 38, 400_000, -173.928427),
            ("lj55", 55, 1_000_000, -279.248470),
            ("lj75", 75, 4_000_000, -397.492331),
            ("lj98", 98, 4_000_000, -543.665361),
            ("lj104", 104, 4_000_000, -582.086642),
            ("au55", 55, 1_000_000, -191.03085280),
        ];
        for (name, n, budget, reference) in table {
            let spec = system_spec(name).unwrap();
            assert_eq!((spec.n, spec.budget), (n, budget), "{name}");
            assert!((spec.reference - reference).abs() < 1e-12, "{name}");
        }
        assert_eq!(system_spec("au55").unwrap().kind, Kind::Gold);
        assert!(system_spec("lj102").is_none());
    }

    #[test]
    fn the_presets_are_the_two_stock_constructors() {
        let spec = system_spec("lj38").unwrap();
        let recommended = config(&spec, Preset::Recommended);
        let cluster = config(&spec, Preset::Cluster);
        assert_eq!(recommended.move_library, MoveLibrary::LeanBurst);
        assert_ne!(cluster.move_library, MoveLibrary::LeanBurst);
        assert_eq!(recommended.relax_steps, 1000);
        assert_eq!(cluster.relax_steps, 200);
    }

    #[test]
    fn a_miss_is_charged_the_full_budget() {
        assert_eq!(evals_to_hit(false, None, 400_000), 400_000);
        assert_eq!(evals_to_hit(true, Some(1200), 400_000), 1200);
        assert_eq!(evals_to_hit(true, None, 4_000_000), 4_000_000);
    }

    #[test]
    fn two_gold_atoms_sit_at_the_gupta_dimer_energy() {
        anneal_core::install_rgpot_kernels!();
        let spec = super::Spec {
            n: 2,
            budget: 1,
            reference: 0.0,
            kind: Kind::Gold,
        };
        let r0 = 2.884_f64;
        let x = Array1::from_vec(vec![0.0, 0.0, 0.0, r0, 0.0, 0.0]);
        let (energy, _) = Surface::of(&spec).value_and_gradient(x.view());
        let expected = 2.0 * 0.2061 - 2.0 * 1.790;
        assert!((energy - expected).abs() < 1e-12, "{energy} vs {expected}");
    }
}
