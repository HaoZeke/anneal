// Copyright (c) anneal developers
//! Sutton–Chen 10-8 gold for the continuous-MCTS head-to-head.
//!
//! ```text
//! cmcts_au search <recommended|for_cluster> <n> <seed> [budget]
//! cmcts_au energy <n>
//! cmcts_au quench <n> <max_iter> <budget>
//! cmcts_au serve
//! cmcts_au calibrate <n> <max_iter> <seed>...
//! ```
//!
//! The potential is the in-tree reduced row: `n = 10`, `m = 8`,
//! `c = 34.408`, `ε = 1`, `a = 1`. A search stops once the best energy
//! is within `1e-3` of the Cambridge global minimum, or when the
//! energy-and-gradient budget is spent. `pot_s` is time inside that
//! evaluation. The budget argument is overridden by `H2H_BUDGET` when
//! that variable is set.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{SuttonChen, SuttonChenFamily};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = 1e-3;
const FAMILY: SuttonChenFamily = SuttonChenFamily::Sc10_8;
/// Lennard-Jones temperature that `Config::with_scales` multiplies by the energy scale.
const LJ_TEMPERATURE: f64 = 0.8;
/// Largest difference allowed between the configured and the published temperature.
const TEMPERATURE_MATCH: f64 = 1e-9;

struct Quenched {
    energy: f64,
    point: Array1<f64>,
    calls: usize,
    pot_ns: u64,
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("search");
    let status = match cmd {
        "search" => cmd_search(&args),
        "energy" => cmd_energy(&args),
        "quench" => cmd_quench(&args),
        "serve" => cmd_serve(),
        "calibrate" => cmd_calibrate(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

fn reference(n: usize) -> Option<f64> {
    // Cambridge Cluster Database, Sutton-Chen 10-8 table, four printed
    // digits. Doye and Wales, New J. Chem. 22, 733 (1998).
    match n {
        35 => Some(-9276.2927),
        38 => Some(-10117.2454),
        55 => Some(-14814.5225),
        _ => None,
    }
}

fn without_template_priors(mut cfg: Config) -> Config {
    // No motif/template priors in this panel: keep both anneal arms on
    // pure stochastic starts and proposal dynamics.
    cfg.prior_soap = false;
    cfg.prior_reseed = false;
    cfg.prior_growth = false;
    cfg.prior_core = false;
    cfg.prior_handover = false;
    cfg.prior_motif = false;
    cfg
}

fn preset(name: &str, n: usize) -> Option<Config> {
    let nn = std::f64::consts::FRAC_1_SQRT_2;
    let length_scale = nn / 2.0_f64.powf(1.0 / 6.0);
    let temperature = FAMILY.basin_hopping_temperature();
    // `*_with_scales` multiplies the Lennard-Jones temperature 0.8 by the
    // energy scale, so this scale makes the Metropolis temperature the
    // published T*.
    let energy_scale = temperature / LJ_TEMPERATURE;
    let cfg = match name {
        "recommended" => Config::recommended_with_scales(n, length_scale, energy_scale),
        "for_cluster" => Config::for_cluster_with_scales(n, length_scale, energy_scale),
        _ => return None,
    };
    assert!((cfg.temperature - temperature).abs() < TEMPERATURE_MATCH);
    Some(without_template_priors(cfg))
}

fn budget_from(args: Option<&String>) -> Option<usize> {
    if let Ok(text) = std::env::var("H2H_BUDGET")
        && let Ok(budget) = text.parse::<usize>()
        && budget > 0
    {
        return Some(budget);
    }
    args.and_then(|text| text.parse::<usize>().ok())
        .filter(|budget| *budget > 0)
}

fn read_coords(text: &str, n: usize) -> Result<Array1<f64>, String> {
    let values: Vec<f64> = text
        .split_whitespace()
        .map(|word| {
            word.parse::<f64>()
                .map_err(|_| format!("not a coordinate: {word}"))
        })
        .collect::<Result<_, _>>()?;
    if values.len() != 3 * n {
        return Err(format!("expected {} numbers, got {}", 3 * n, values.len()));
    }
    Ok(Array1::from_vec(values))
}

fn coords_line(point: &Array1<f64>) -> String {
    point
        .iter()
        .map(|value| format!("{value:.16}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn timed_pair(pot: &SuttonChen, x: ArrayView1<f64>, pot_ns: &AtomicU64) -> (f64, Array1<f64>) {
    let mark = Instant::now();
    let pair = pot.value_and_gradient(x);
    pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
    pair
}

fn quench(pot: &SuttonChen, x0: &Array1<f64>, max_iter: usize, budget: usize) -> Quenched {
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = AtomicU64::new(0);
    let mut calls = 0usize;
    let (energy, point, _) = optimiser.minimize(x0.view(), max_iter, |x| {
        if calls >= budget {
            return None;
        }
        calls += 1;
        Some(timed_pair(pot, x, &pot_ns))
    });
    Quenched {
        energy,
        point,
        calls,
        pot_ns: pot_ns.load(Ordering::Relaxed),
    }
}

fn start_of(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn search(cfg: &Config, n: usize, seed: u64, budget: usize, level: f64) -> (Outcome, f64, f64) {
    let pot = SuttonChen::new(n, FAMILY);
    let start = start_of(cfg, n, seed);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(level);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = AtomicU64::new(0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            Some(timed_pair(&pot, v, &pot_ns))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(timed_pair(&pot, v, &pot_ns).1)
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

#[derive(Copy, Clone)]
struct SearchRow<'a> {
    preset: &'a str,
    n: usize,
    seed: u64,
    budget: usize,
    reference: f64,
    outcome: &'a Outcome,
    wall: f64,
    pot_s: f64,
}

fn write_search_row(row: &SearchRow<'_>) {
    let SearchRow {
        preset: preset_name,
        n,
        seed,
        budget,
        reference: reference_energy,
        outcome,
        wall,
        pot_s,
    } = *row;
    let hit = (outcome.best - reference_energy).abs() <= HIT_TOL;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| (*energy - reference_energy).abs() <= HIT_TOL)
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(budget)
    } else {
        budget
    };
    let outside = (wall - pot_s).max(0.0);
    println!(
        "{preset_name},au{n},{seed},{budget},{hit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6}",
        outcome.charged, outcome.best,
    );
}

fn cmd_search(args: &[String]) -> i32 {
    let Some(name) = args.get(1).map(String::as_str) else {
        eprintln!("search needs a preset");
        return 2;
    };
    let Some(n) = args.get(2).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("search needs a size");
        return 2;
    };
    let Some(seed) = args.get(3).and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("search needs a seed");
        return 2;
    };
    let Some(cfg) = preset(name, n) else {
        eprintln!("unknown preset {name}");
        return 2;
    };
    let Some(reference_energy) = reference(n) else {
        eprintln!("no Cambridge energy for N={n}");
        return 2;
    };
    let Some(budget) = budget_from(args.get(4)) else {
        eprintln!("search needs a positive budget");
        return 2;
    };
    let (outcome, wall, pot_s) = search(&cfg, n, seed, budget, reference_energy + HIT_TOL);
    write_search_row(&SearchRow {
        preset: name,
        n,
        seed,
        budget,
        reference: reference_energy,
        outcome: &outcome,
        wall,
        pot_s,
    });
    0
}

fn stdin_coords(n: usize) -> Result<Array1<f64>, String> {
    let mut text = String::new();
    io::stdin()
        .read_line(&mut text)
        .map_err(|err| err.to_string())?;
    read_coords(&text, n)
}

fn cmd_energy(args: &[String]) -> i32 {
    let Some(n) = args.get(1).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("energy needs a size");
        return 2;
    };
    let coords = match stdin_coords(n) {
        Ok(coords) => coords,
        Err(err) => {
            eprintln!("energy: {err}");
            return 1;
        }
    };
    let pot = SuttonChen::new(n, FAMILY);
    let (energy, _) = pot.value_and_gradient(coords.view());
    println!("{energy:.16}");
    0
}

fn cmd_quench(args: &[String]) -> i32 {
    let Some(n) = args.get(1).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("quench needs a size");
        return 2;
    };
    let Some(max_iter) = args.get(2).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("quench needs a step cap");
        return 2;
    };
    let Some(budget) = args.get(3).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("quench needs a budget");
        return 2;
    };
    let coords = match stdin_coords(n) {
        Ok(coords) => coords,
        Err(err) => {
            eprintln!("quench: {err}");
            return 1;
        }
    };
    let pot = SuttonChen::new(n, FAMILY);
    let quenched = quench(&pot, &coords, max_iter, budget);
    println!(
        "{:.16} {} {}",
        quenched.energy, quenched.calls, quenched.pot_ns
    );
    println!("{}", coords_line(&quenched.point));
    0
}

fn cmd_serve() -> i32 {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut out = io::stdout();
    while let Some(command) = lines.next() {
        let command = match command {
            Ok(command) => command,
            Err(err) => {
                eprintln!("serve: {err}");
                return 1;
            }
        };
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "quit" => break,
            "energy" => {
                let Some(n) = parts.get(1).and_then(|text| text.parse::<usize>().ok()) else {
                    eprintln!("energy needs a size");
                    return 2;
                };
                let Some(coords) = lines.next() else {
                    eprintln!("energy needs coordinates");
                    return 2;
                };
                let coords = match coords.and_then(|text| {
                    read_coords(&text, n)
                        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
                }) {
                    Ok(coords) => coords,
                    Err(err) => {
                        eprintln!("energy: {err}");
                        return 1;
                    }
                };
                let pot = SuttonChen::new(n, FAMILY);
                let (energy, _) = pot.value_and_gradient(coords.view());
                if writeln!(out, "{energy:.16}").is_err() || out.flush().is_err() {
                    return 1;
                }
            }
            "quench" => {
                let Some(n) = parts.get(1).and_then(|text| text.parse::<usize>().ok()) else {
                    eprintln!("quench needs a size");
                    return 2;
                };
                let Some(max_iter) = parts.get(2).and_then(|text| text.parse::<usize>().ok())
                else {
                    eprintln!("quench needs a step cap");
                    return 2;
                };
                let Some(budget) = parts.get(3).and_then(|text| text.parse::<usize>().ok()) else {
                    eprintln!("quench needs a budget");
                    return 2;
                };
                let Some(coords) = lines.next() else {
                    eprintln!("quench needs coordinates");
                    return 2;
                };
                let coords = match coords.and_then(|text| {
                    read_coords(&text, n)
                        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
                }) {
                    Ok(coords) => coords,
                    Err(err) => {
                        eprintln!("quench: {err}");
                        return 1;
                    }
                };
                let pot = SuttonChen::new(n, FAMILY);
                let quenched = quench(&pot, &coords, max_iter, budget);
                if writeln!(
                    out,
                    "{:.16} {} {}",
                    quenched.energy, quenched.calls, quenched.pot_ns
                )
                .is_err()
                    || writeln!(out, "{}", coords_line(&quenched.point)).is_err()
                    || out.flush().is_err()
                {
                    return 1;
                }
            }
            other => {
                eprintln!("unknown serve command {other}");
                return 2;
            }
        }
    }
    0
}

fn cmd_calibrate(args: &[String]) -> i32 {
    let Some(n) = args.get(1).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("calibrate needs a size");
        return 2;
    };
    let Some(max_iter) = args.get(2).and_then(|text| text.parse::<usize>().ok()) else {
        eprintln!("calibrate needs a step cap");
        return 2;
    };
    if args.len() < 4 {
        eprintln!("calibrate needs seeds");
        return 2;
    }
    let Some(cfg) = preset("recommended", n) else {
        eprintln!("no preset for N={n}");
        return 2;
    };
    let pot = SuttonChen::new(n, FAMILY);
    for text in &args[3..] {
        let Some(seed) = text.parse::<u64>().ok() else {
            eprintln!("bad seed {text}");
            return 2;
        };
        let start = start_of(&cfg, n, seed);
        let quenched = quench(&pot, &start, max_iter, usize::MAX / 4);
        println!("{seed} {} {:.8}", quenched.calls, quenched.energy);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{FAMILY, LJ_TEMPERATURE, preset};
    use anneal_core::methods::cluster_hopping::{Config, OrbitCore};

    #[test]
    fn presets_use_sutton_chen_gold_scales_without_template_priors() {
        let length_scale = std::f64::consts::FRAC_1_SQRT_2 / 2.0_f64.powf(1.0 / 6.0);
        let temperature = FAMILY.basin_hopping_temperature();
        let energy_scale = temperature / LJ_TEMPERATURE;

        let recommended = preset("recommended", 38).expect("recommended preset");
        let expected_recommended = Config::recommended_with_scales(38, length_scale, energy_scale);
        assert!((recommended.length_scale - length_scale).abs() < 1e-12);
        assert!((recommended.temperature - temperature).abs() < 1e-9);
        assert_eq!(recommended.energy_scale, expected_recommended.energy_scale);
        assert!(!recommended.prior_soap);
        assert!(!recommended.prior_reseed);
        assert!(!recommended.prior_growth);
        assert!(!recommended.prior_core);
        assert!(!recommended.prior_handover);
        assert!(!recommended.prior_motif);
        assert!(matches!(recommended.orbit_core, OrbitCore::Fraction));

        let for_cluster = preset("for_cluster", 38).expect("for_cluster preset");
        assert!((for_cluster.length_scale - length_scale).abs() < 1e-12);
        assert!((for_cluster.temperature - temperature).abs() < 1e-9);
        assert!(!for_cluster.prior_soap);
        assert!(!for_cluster.prior_motif);
        assert!(matches!(for_cluster.orbit_core, OrbitCore::Both(_)));

        assert!(preset("unknown", 38).is_none());
    }
}
