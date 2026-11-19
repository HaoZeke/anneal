// Copyright (c) anneal developers
//! One run of the weak7 source-tree comparison on one cell.
//!
//! ```text
//! morse_weak7_gate run <cell> <seed>
//! ```
//!
//! A base build uses `Config::recommended(n)`. A build with the runner's
//! `weak7` feature uses `Config::recommended_adaptive(n)`. The adaptive preset
//! is the arm under test; the recommended preset stays the default single
//! chain. Both draw the start
//! from the stock preset's sphere and separation, so a seed names one geometry
//! for every arm, and every search uses a `StdRng` seeded with that seed.
//!
//! A hit is a quenched energy at or below the reference plus
//! `anneal_core::DEFAULT_HIT_TOL`. The ledger charges each energy-and-gradient
//! pair inside a quench and each gradient the climb reads, as `h2h_gmin_proof`
//! and `h2h_morse` do. A miss is reported as the full budget. `pot_s` is the
//! time inside the potential.
//!
//! The process prints one `gate_row` line of `key=value` pairs. Besides the
//! outcome it reports what the process saw of itself: the core it ran on, its
//! affinity mask, the thread count and the scheduler's own account of run
//! time, wait time and context switches over the search. Those let a reader
//! tell a pinned, undisturbed run from one that was not.

#![forbid(unsafe_code)]

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

#[derive(Clone, Copy)]
enum Pair {
    LennardJones,
    Morse(f64),
}

impl Pair {
    fn potential(self, n: usize) -> PairPotential {
        match self {
            Self::LennardJones => PairPotential::lennard_jones(n),
            Self::Morse(rho) => PairPotential::morse(n, rho),
        }
    }

    fn kind(self) -> &'static str {
        match self {
            Self::LennardJones => "lj",
            Self::Morse(_) => "morse",
        }
    }

    fn rho(self) -> String {
        match self {
            Self::LennardJones => "na".to_string(),
            Self::Morse(rho) => format!("{rho:.1}"),
        }
    }
}

struct Cell {
    name: &'static str,
    pair: Pair,
    n: usize,
    reference: f64,
    budget: usize,
}

/// Cambridge Cluster Database minima. Budgets follow the Lennard-Jones paper
/// list: 4e5 up to 38 atoms, 1e6 below 75, 4e6 from 75.
const CELLS: [Cell; 8] = [
    Cell {
        name: "lj38",
        pair: Pair::LennardJones,
        n: 38,
        reference: -173.928427,
        budget: 400_000,
    },
    Cell {
        name: "m38r10",
        pair: Pair::Morse(10.0),
        n: 38,
        reference: -145.849817,
        budget: 400_000,
    },
    Cell {
        name: "lj75",
        pair: Pair::LennardJones,
        n: 75,
        reference: -397.492331,
        budget: 4_000_000,
    },
    Cell {
        name: "lj98",
        pair: Pair::LennardJones,
        n: 98,
        reference: -543.665361,
        budget: 4_000_000,
    },
    Cell {
        name: "m75r6",
        pair: Pair::Morse(6.0),
        n: 75,
        reference: -351.472365,
        budget: 4_000_000,
    },
    Cell {
        name: "m75r14",
        pair: Pair::Morse(14.0),
        n: 75,
        reference: -318.407330,
        budget: 4_000_000,
    },
    Cell {
        name: "m80r6",
        pair: Pair::Morse(6.0),
        n: 80,
        reference: -378.333471,
        budget: 4_000_000,
    },
    Cell {
        name: "m80r14",
        pair: Pair::Morse(14.0),
        n: 80,
        reference: -340.811371,
        budget: 4_000_000,
    },
];

#[cfg(not(feature = "weak7"))]
fn arm_config(n: usize) -> Config {
    Config::recommended(n)
}

#[cfg(feature = "weak7")]
fn arm_config(n: usize) -> Config {
    Config::recommended_adaptive(n)
}

#[cfg(not(feature = "weak7"))]
fn arm_mode() -> &'static str {
    "base"
}

#[cfg(feature = "weak7")]
fn arm_mode() -> &'static str {
    "weak7"
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = match args.first().map(String::as_str) {
        Some("run") => match cmd_run(&args) {
            Ok(()) => 0,
            Err(message) => {
                eprintln!("morse_weak7_gate: {message}");
                1
            }
        },
        _ => {
            eprintln!("usage: morse_weak7_gate run <cell> <seed>");
            2
        }
    };
    std::process::exit(status);
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let [_, cell_name, seed_text] = args else {
        return Err("run needs a cell and a seed".to_string());
    };
    let cell = CELLS
        .iter()
        .find(|cell| cell.name == cell_name)
        .ok_or_else(|| format!("unknown cell {cell_name}"))?;
    let seed: u64 = seed_text
        .parse()
        .map_err(|err| format!("seed {seed_text:?}: {err}"))?;
    Config::check_removed_env().map_err(|err| err.to_string())?;
    let stock = Config::recommended(cell.n);
    let cfg = arm_config(cell.n);
    if cfg.start_radius().to_bits() != stock.start_radius().to_bits()
        || cfg.min_separation.to_bits() != stock.min_separation.to_bits()
    {
        return Err(format!("arm {} does not share the stock start", arm_mode()));
    }
    let start = start_of(&stock, cell.n, seed);
    let start_sha256 = coordinates_sha256(&start);
    let config_sha256 = cfg.resolved_sha256().map_err(|err| err.to_string())?;
    let before = Snapshot::take()?;
    let run = search(&cfg, cell, &start, seed, cell.budget)?;
    let after = Snapshot::take()?;
    let outcome = &run.outcome;
    let hit = anneal_core::energy_hits(outcome.best, cell.reference, HIT_TOL);
    let evals = evals_to_hit(outcome, cell.reference, cell.budget);
    let outside = (run.wall_s - run.pot_s).max(0.0);
    let row = [
        ("cell", cell.name.to_string()),
        ("mode", arm_mode().to_string()),
        ("seed", seed.to_string()),
        ("n", cell.n.to_string()),
        ("kind", cell.pair.kind().to_string()),
        ("rho", cell.pair.rho()),
        ("budget", cell.budget.to_string()),
        ("reference", format!("{:.6}", cell.reference)),
        ("replicas", cfg.replicas.to_string()),
        ("hit", u8::from(hit).to_string()),
        ("evals_to_hit", evals.to_string()),
        ("charged", outcome.charged.to_string()),
        ("best", outcome.best.to_string()),
        ("best_bits", format!("{:016x}", outcome.best.to_bits())),
        ("wall_s", format!("{:.6}", run.wall_s)),
        ("pot_s", format!("{:.6}", run.pot_s)),
        ("outside_s", format!("{outside:.6}")),
        (
            "cpu_ns",
            after.cpu_ns.saturating_sub(before.cpu_ns).to_string(),
        ),
        (
            "wait_ns",
            after.wait_ns.saturating_sub(before.wait_ns).to_string(),
        ),
        (
            "vol_ctxt",
            after.voluntary.saturating_sub(before.voluntary).to_string(),
        ),
        (
            "invol_ctxt",
            after
                .involuntary
                .saturating_sub(before.involuntary)
                .to_string(),
        ),
        ("threads", after.threads.to_string()),
        ("core_start", before.core.to_string()),
        ("core_end", after.core.to_string()),
        ("cpus_allowed", after.allowed.clone()),
        ("cpus_allowed_list", after.allowed_list.clone()),
        ("start_sha256", start_sha256),
        ("config_sha256", config_sha256),
    ];
    let fields: Vec<String> = row
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect();
    println!("gate_row {}", fields.join(" "));
    Ok(())
}

fn start_of(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn coordinates_sha256(coords: &Array1<f64>) -> String {
    let mut bytes = Vec::with_capacity(8 * coords.len());
    for value in coords.iter() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    format!("{:x}", Sha256::digest(&bytes))
}

struct Run {
    outcome: Outcome,
    wall_s: f64,
    pot_s: f64,
}

fn search(
    cfg: &Config,
    cell: &Cell,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) -> Result<Run, String> {
    let potential = cell.pair.potential(cell.n);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(cell.reference + HIT_TOL);
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
    .map_err(|err| format!("search: {err}"))?;
    let wall_s = started.elapsed().as_secs_f64();
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    Ok(Run {
        outcome,
        wall_s,
        pot_s,
    })
}

/// A hit keeps the charged count at the first record at or below the level,
/// or the charged count when the hitting quench was polished into the ledger
/// without a row of its own. A miss is the whole budget.
fn evals_to_hit(outcome: &Outcome, reference: f64, budget: usize) -> usize {
    if !anneal_core::energy_hits(outcome.best, reference, HIT_TOL) {
        return budget;
    }
    outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, reference, HIT_TOL))
        .map_or(outcome.charged, |(_, spent, _, _)| *spent)
        .min(budget)
}

/// What the process reads of itself from `/proc/self`.
struct Snapshot {
    cpu_ns: u64,
    wait_ns: u64,
    voluntary: u64,
    involuntary: u64,
    threads: u64,
    core: u64,
    allowed: String,
    allowed_list: String,
}

impl Snapshot {
    fn take() -> Result<Self, String> {
        let schedstat = read("/proc/self/schedstat")?;
        let times: Vec<u64> = schedstat
            .split_whitespace()
            .map(|word| {
                word.parse::<u64>()
                    .map_err(|err| format!("schedstat {word:?}: {err}"))
            })
            .collect::<Result<_, _>>()?;
        let [cpu_ns, wait_ns, ..] = times[..] else {
            return Err("schedstat has fewer than two fields".to_string());
        };
        let status = read("/proc/self/status")?;
        let stat = read("/proc/self/stat")?;
        Ok(Self {
            cpu_ns,
            wait_ns,
            voluntary: status_number(&status, "voluntary_ctxt_switches")?,
            involuntary: status_number(&status, "nonvoluntary_ctxt_switches")?,
            threads: status_number(&status, "Threads")?,
            core: processor_of(&stat).ok_or("stat has no processor field")?,
            allowed: status_field(&status, "Cpus_allowed")?.replace(',', ""),
            allowed_list: status_field(&status, "Cpus_allowed_list")?.to_string(),
        })
    }
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|err| format!("{path}: {err}"))
}

fn status_field<'a>(status: &'a str, key: &str) -> Result<&'a str, String> {
    status
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name == key).then_some(value.trim())
        })
        .ok_or_else(|| format!("/proc/self/status has no {key}"))
}

fn status_number(status: &str, key: &str) -> Result<u64, String> {
    let text = status_field(status, key)?;
    text.parse().map_err(|err| format!("{key} {text:?}: {err}"))
}

/// Field 39 of `/proc/self/stat`, the core the thread last ran on. The command
/// name is field 2 and may hold spaces and parentheses, so the count starts
/// after the last closing parenthesis, at field 3.
fn processor_of(stat: &str) -> Option<u64> {
    let after = &stat[stat.rfind(')')? + 1..];
    after.split_whitespace().nth(36)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_follow_the_paper_list() {
        for cell in &CELLS {
            let want = if cell.n <= 38 {
                400_000
            } else if cell.n < 75 {
                1_000_000
            } else {
                4_000_000
            };
            assert_eq!(cell.budget, want, "{}", cell.name);
        }
    }

    #[test]
    fn morse_references_match_the_library_table() {
        for cell in &CELLS {
            if let Pair::Morse(rho) = cell.pair
                && let Some(table) = PairPotential::ccd_morse_minimum(cell.n, rho)
            {
                assert!((table - cell.reference).abs() < 1e-12, "{}", cell.name);
            }
        }
    }

    #[test]
    fn references_are_the_published_minima() {
        let want = [
            ("lj38", -173.928427),
            ("m38r10", -145.849817),
            ("lj75", -397.492331),
            ("lj98", -543.665361),
            ("m75r6", -351.472365),
            ("m75r14", -318.407330),
            ("m80r6", -378.333471),
            ("m80r14", -340.811371),
        ];
        for (name, reference) in want {
            let cell = CELLS.iter().find(|cell| cell.name == name).unwrap();
            assert!((cell.reference - reference).abs() < 1e-12, "{name}");
        }
    }

    #[test]
    fn the_processor_is_the_thirty_ninth_field() {
        let fields: Vec<String> = (3..=52)
            .map(|index| match index {
                3 => "S".to_string(),
                39 => "6".to_string(),
                other => other.to_string(),
            })
            .collect();
        let stat = format!("77 (odd ) name) {}", fields.join(" "));
        assert_eq!(processor_of(&stat), Some(6));
    }

    #[test]
    fn status_fields_are_read_by_name() {
        let status = "Name:\tx\nThreads:\t1\nCpus_allowed:\t00,0000000f\nCpus_allowed_list:\t0-3\n";
        assert_eq!(status_number(status, "Threads"), Ok(1));
        assert_eq!(status_field(status, "Cpus_allowed_list"), Ok("0-3"));
        assert!(status_field(status, "Missing").is_err());
    }

    #[test]
    #[cfg(feature = "weak7")]
    fn the_weak7_arm_is_the_single_chain_inside_the_allowance() {
        let cell = CELLS.iter().find(|cell| cell.name == "lj38").unwrap();
        let stock = Config::recommended(cell.n);
        let adaptive = Config::recommended_adaptive(cell.n);
        assert_eq!(
            adaptive.start_radius().to_bits(),
            stock.start_radius().to_bits()
        );
        assert_eq!(
            adaptive.min_separation.to_bits(),
            stock.min_separation.to_bits()
        );
        let start = start_of(&stock, cell.n, 2000);
        let single = search(&stock, cell, &start, 2000, 10_000).unwrap();
        let held = search(&adaptive, cell, &start, 2000, 10_000).unwrap();
        assert_eq!(held.outcome.charged, single.outcome.charged);
        assert_eq!(held.outcome.best.to_bits(), single.outcome.best.to_bits());
        assert_eq!(held.outcome.improvements, single.outcome.improvements);
    }
}
