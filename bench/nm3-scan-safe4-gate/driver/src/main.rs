// Copyright (c) anneal developers
//! Serial gate driver for Morse M75 rho14 and the seven Doye-Meyer BLJ clusters.
//!
//! Uses `Config::recommended` (plus species labels `200`/`201` with no diameters
//! for BLJ) at the repository paper budgets (`PAPER_BUDGETS`), never reading
//! `H2H_BUDGET` or any `H2H_*` environment variable. Hit tolerance is
//! `anneal_core::DEFAULT_HIT_TOL` (0.001).

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{BinaryLennardJones, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;
const LABEL_A: u32 = 200;
const LABEL_B: u32 = 201;

#[derive(Clone, Copy)]
enum Kind {
    Morse { rho: f64 },
    Blj { n_a: usize, sigma_bb: f64 },
}

#[derive(Clone, Copy)]
struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
    kind: Kind,
}

fn system_spec(name: &str) -> Option<Spec> {
    match name {
        "m75r14" => Some(Spec {
            n: 75,
            budget: 4_000_000,
            reference: -318.407330,
            kind: Kind::Morse { rho: 14.0 },
        }),
        "blj105n38" => Some(Spec {
            n: 38,
            budget: 400_000,
            reference: -177.2090846748,
            kind: Kind::Blj {
                n_a: 8,
                sigma_bb: 1.05,
            },
        }),
        "blj105n55" => Some(Spec {
            n: 55,
            budget: 1_000_000,
            reference: -283.1545313810,
            kind: Kind::Blj {
                n_a: 13,
                sigma_bb: 1.05,
            },
        }),
        "blj115n38" => Some(Spec {
            n: 38,
            budget: 400_000,
            reference: -187.3676867453,
            kind: Kind::Blj {
                n_a: 9,
                sigma_bb: 1.15,
            },
        }),
        "blj115n55" => Some(Spec {
            n: 55,
            budget: 1_000_000,
            reference: -289.2931853058,
            kind: Kind::Blj {
                n_a: 16,
                sigma_bb: 1.15,
            },
        }),
        "blj130n38" => Some(Spec {
            n: 38,
            budget: 400_000,
            reference: -191.1840192602,
            kind: Kind::Blj {
                n_a: 13,
                sigma_bb: 1.30,
            },
        }),
        "blj130n55" => Some(Spec {
            n: 55,
            budget: 1_000_000,
            reference: -306.7337643721,
            kind: Kind::Blj {
                n_a: 19,
                sigma_bb: 1.30,
            },
        }),
        "blj130n98" => Some(Spec {
            n: 98,
            budget: 4_000_000,
            reference: -590.7874132760,
            kind: Kind::Blj {
                n_a: 41,
                sigma_bb: 1.30,
            },
        }),
        _ => None,
    }
}

enum Potential {
    Morse(PairPotential),
    Blj(BinaryLennardJones),
}

impl Potential {
    fn new(spec: Spec) -> Self {
        match spec.kind {
            Kind::Morse { rho } => Self::Morse(PairPotential::morse(spec.n, rho)),
            Kind::Blj { n_a, sigma_bb } => Self::Blj(BinaryLennardJones::new(
                n_a,
                spec.n - n_a,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5 * (1.0 + sigma_bb),
                sigma_bb,
            )),
        }
    }

    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Self::Morse(pot) => pot.value_and_gradient(x),
            Self::Blj(pot) => pot.value_and_gradient(x),
        }
    }
}

fn build_config(
    spec: Spec,
) -> Result<Config, anneal_core::methods::cluster_hopping::RemovedConfigKeyError> {
    Config::check_removed_env()?;
    let cfg = match spec.kind {
        Kind::Morse { .. } => Config::recommended(spec.n),
        Kind::Blj { n_a, .. } => {
            let mut labels = vec![LABEL_A; spec.n];
            for label in labels.iter_mut().skip(n_a) {
                *label = LABEL_B;
            }
            Config::recommended(spec.n).with_species(labels, None)
        }
    };
    Ok(cfg)
}

fn check_no_h2h_env() {
    for (k, _) in std::env::vars() {
        if k.starts_with("H2H_") {
            eprintln!("{k} must not be set");
            std::process::exit(2);
        }
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    check_no_h2h_env();
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
    println!("{:?}", spec.reference);
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
    let cfg = match build_config(spec) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{err}");
            return 2;
        }
    };
    let start = start_of(&cfg, spec.n, seed);
    let mut text = String::new();
    for chunk in start.as_slice().expect("contiguous").chunks(3) {
        use std::fmt::Write as _;
        writeln!(text, "{:.17e} {:.17e} {:.17e}", chunk[0], chunk[1], chunk[2]).unwrap();
    }
    if let Err(err) = std::fs::write(path, text) {
        eprintln!("coords: {err}");
        return 1;
    }
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
    let coords_out = args.get(3);
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let budget = spec.budget;
    let cfg = match build_config(spec) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{err}");
            return 2;
        }
    };
    let start = start_of(&cfg, spec.n, seed);
    let (outcome, wall, pot_s) = search(&cfg, spec, &start, seed, budget);
    if let Some(path) = coords_out {
        let mut text = String::new();
        let best_state = outcome.best_state.as_ref().expect("best_state");
        for chunk in best_state.as_slice().expect("contiguous").chunks(3) {
            use std::fmt::Write as _;
            writeln!(text, "{:.17e} {:.17e} {:.17e}", chunk[0], chunk[1], chunk[2]).unwrap();
        }
        if let Err(err) = std::fs::write(path, text) {
            eprintln!("coords_out: {err}");
            return 1;
        }
    }
    let hit = anneal_core::energy_hits(outcome.best, spec.reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, spec.reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        match at_hit {
            Some(spent) if spent < budget => spent,
            _ => outcome.charged.min(budget),
        }
    } else {
        budget
    };
    let outside = (wall - pot_s).max(0.0);
    let best_bits = outcome.best.to_bits();
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,best_bits,pot_s,outside_s"
    );
    println!(
        "{system},anneal,{seed},{budget},{hit},{evals},{},{wall:.6},{:?},{best_bits:016x},{pot_s:.6},{outside:.6}",
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
    spec: Spec,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) -> (Outcome, f64, f64) {
    let potential = Potential::new(spec);
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
            let evaluated = potential.value_and_gradient(v);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
    .expect("cluster hopping search");
    let wall = started.elapsed().as_secs_f64();
    let pot_s = (pot_ns.load(Ordering::Relaxed) as f64) * 1e-9;
    (outcome, wall, pot_s)
}
