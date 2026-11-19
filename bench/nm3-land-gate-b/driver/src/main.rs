// Copyright (c) anneal developers
//! Serial gate driver for Morse M75 rho14 and the seven Doye-Meyer BLJ clusters.
//!
//! ```text
//! nm3-gate-driver run <system> <seed>
//! nm3-gate-driver coords <system> <seed> <path>
//! nm3-gate-driver energy <system> <path>
//! nm3-gate-driver budget <system>
//! ```
//!
//! Uses `Config::recommended` (plus species labels `200`/`201` with no diameters
//! for BLJ) at the repository paper budgets (`PAPER_BUDGETS`), never reading
//! `H2H_BUDGET`. Hit tolerance is `anneal_core::DEFAULT_HIT_TOL` (0.001).

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

fn build_config(spec: Spec) -> Result<Config, anneal_core::methods::cluster_hopping::RemovedConfigKeyError> {
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
    match write_coords(path, &start) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("coords: {err}");
            1
        }
    }
}

fn cmd_energy(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("energy needs a system");
        return 2;
    };
    let Some(path) = args.get(2) else {
        eprintln!("energy needs a coordinate file");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
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
    let (energy, _) = Potential::new(spec).value_and_gradient(coords.view());
    println!("{energy:.12}");
    0
}

fn cmd_run(args: &[String]) -> i32 {
    if std::env::var_os("H2H_BUDGET").is_some() {
        eprintln!("H2H_BUDGET must not be set");
        return 2;
    }
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
    println!("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s");
    println!(
        "{system},anneal,{seed},{budget},{hit},{evals},{},{wall:.6},{:.10},{pot_s:.6},{outside:.6}",
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
    .expect("finite start");
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
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
        let fields: Vec<&str> = line.split_whitespace().collect();
        let numbers = if fields.len() == 4
            && matches!(fields[0], "A" | "B" | "LA" | "LB" | "X")
        {
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
    use super::{HIT_TOL, Kind, LABEL_A, LABEL_B, Potential, build_config, system_spec};
    use anneal_core::methods::cluster_hopping::Config;
    use anneal_core::potentials::PairPotential;

    #[test]
    fn morse_m75_rho14_matches_ccd_table_and_paper_budget() {
        let spec = system_spec("m75r14").expect("m75r14");
        assert_eq!(spec.n, 75);
        assert_eq!(spec.budget, 4_000_000);
        assert!((HIT_TOL - 0.001).abs() < 1e-15);
        assert_eq!(
            PairPotential::ccd_morse_minimum(75, 14.0),
            Some(spec.reference)
        );
        let cfg = build_config(spec).expect("config");
        assert_eq!(
            cfg.resolved_json().expect("resolved"),
            Config::recommended(75).resolved_json().expect("resolved")
        );
    }

    #[test]
    fn blj_panel_matches_ccd_headers_and_paper_budgets() {
        anneal_core::install_rgpot_kernels!();
        let root = std::env::var("ANNEAL_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.."));
        let ccd_dir = root.join("tests/fixtures/blj-ccd");
        let systems = [
            "blj105n38",
            "blj105n55",
            "blj115n38",
            "blj115n55",
            "blj130n38",
            "blj130n55",
            "blj130n98",
        ];
        for name in systems {
            let spec = system_spec(name).expect(name);
            let expected_budget = match spec.n {
                38 => 400_000,
                55 => 1_000_000,
                98 => 4_000_000,
                other => panic!("unexpected size {other}"),
            };
            assert_eq!(spec.budget, expected_budget, "{name}");

            let text = std::fs::read_to_string(ccd_dir.join(name)).expect("ccd fixture");
            let mut lines = text.lines();
            let n: usize = lines.next().unwrap().trim().parse().unwrap();
            assert_eq!(n, spec.n, "{name}");
            let header = lines.next().unwrap();
            let tokens: Vec<&str> = header.split_whitespace().collect();
            let energy: f64 = tokens[tokens.iter().position(|t| *t == "1=").unwrap() + 1]
                .parse()
                .unwrap();
            let idx = tokens.iter().position(|t| *t == "(NA,SIG_BB)").unwrap();
            let n_a: usize = tokens[idx + 1].parse().unwrap();
            let sigma_bb: f64 = tokens[idx + 2].parse().unwrap();
            let Kind::Blj {
                n_a: got_a,
                sigma_bb: got_sigma,
            } = spec.kind
            else {
                panic!("expected BLJ kind");
            };
            assert_eq!(got_a, n_a, "{name}");
            assert!((got_sigma - sigma_bb).abs() < 1e-12, "{name}");
            assert!((spec.reference - energy).abs() < 1e-12, "{name}");

            let mut a = Vec::new();
            let mut b = Vec::new();
            for line in lines {
                let mut parts = line.split_whitespace();
                let Some(label) = parts.next() else {
                    continue;
                };
                let coords: Vec<f64> = parts.map(|w| w.parse().unwrap()).collect();
                match label {
                    "LA" | "A" => a.extend(coords),
                    "LB" | "B" => b.extend(coords),
                    other => panic!("unexpected label {other}"),
                }
            }
            a.extend(b);
            let x = ndarray::Array1::from_vec(a);
            let (got_e, _) = Potential::new(spec).value_and_gradient(x.view());
            assert!(
                (got_e - spec.reference).abs() < 1e-9,
                "{name}: energy {got_e} != {}",
                spec.reference
            );

            let cfg = build_config(spec).expect("config");
            assert!(cfg.species_sigma.is_none(), "{name}");
            let species = cfg.species.expect("species");
            assert_eq!(species.iter().filter(|z| **z == LABEL_A).count(), n_a);
            assert_eq!(
                species.iter().filter(|z| **z == LABEL_B).count(),
                spec.n - n_a
            );
        }
    }
}
