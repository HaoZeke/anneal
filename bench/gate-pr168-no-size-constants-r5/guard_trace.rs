// Copyright (c) anneal developers
//! One default-preset guard seed, with a bit-level trajectory hash.
//!
//! Lennard-Jones and Morse use `PairPotential` and `Config::recommended`.
//! Au55 uses `Gupta::copper_gold` and the species the metal example installs.
//! The hexamer uses `GminTip4p` and `Config::recommended_tip4p`.
//! Each run stops at the published minimum plus `DEFAULT_HIT_TOL`, or at
//! the paper budget. Nothing else is changed.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{GminTip4p, Gupta, PairPotential, gmin_water_geometry, random_tip4p_cluster};
use eindir_core::gradient::DifferentiableObjective;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

struct Done {
    budget: usize,
    hit: bool,
    calls: usize,
    charged: usize,
    wall_s: f64,
    best_bits: String,
    trajectory: String,
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let core = affinity();
    if core != "0" {
        eprintln!("affinity {core}");
        std::process::exit(3);
    }
    let mut args = std::env::args().skip(1);
    let Some(system) = args.next() else {
        eprintln!("need a system");
        std::process::exit(2);
    };
    let Some(seed) = args.next().and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("need a seed");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("too many arguments");
        std::process::exit(2);
    }
    if let Err(err) = Config::check_removed_env() {
        eprintln!("removed config: {err}");
        std::process::exit(2);
    }
    let done = match system.as_str() {
        "lj38" => run_lj(38, 400_000, -173.928427, seed),
        "lj75" => run_lj(75, 4_000_000, -397.492331, seed),
        "m80r6" => run_morse(80, 6.0, 4_000_000, -378.333471, seed),
        "au55" => run_gold(55, 1_000_000, -191.03085280, seed),
        "hexamer" => run_hexamer(seed),
        other => {
            eprintln!("unknown system {other}");
            std::process::exit(2);
        }
    };
    println!(
        "GATE system={system} seed={seed} budget={} hit={} calls={} charged={} wall_s={:.9} best_bits={} trajectory={} core={core}",
        done.budget,
        u8::from(done.hit),
        done.calls,
        done.charged,
        done.wall_s,
        done.best_bits,
        done.trajectory,
    );
}

fn affinity() -> String {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    text.lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .map(|rest| rest.trim().to_string())
        .unwrap_or_default()
}

fn run_lj(n: usize, budget: usize, reference: f64, seed: u64) -> Done {
    let cfg = Config::recommended(n);
    let potential = PairPotential::lennard_jones(n);
    run_lbfgs("lj", n, budget, reference, seed, cfg, &potential)
}

fn run_morse(n: usize, rho: f64, budget: usize, reference: f64, seed: u64) -> Done {
    let cfg = Config::recommended(n);
    let potential = PairPotential::morse(n, rho);
    run_lbfgs("morse", n, budget, reference, seed, cfg, &potential)
}

fn run_gold(n: usize, budget: usize, reference: f64, seed: u64) -> Done {
    let cfg = Config::recommended(n).with_species(Gupta::copper_gold_species(0, n), None);
    let potential = Gupta::copper_gold(0, n);
    run_lbfgs("au", n, budget, reference, seed, cfg, &potential)
}

fn run_lbfgs<P>(
    _label: &str,
    n: usize,
    budget: usize,
    reference: f64,
    seed: u64,
    cfg: Config,
    potential: &P,
) -> Done
where
    P: DifferentiableObjective<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    finish(budget, reference, started, outcome)
}

fn run_hexamer(seed: u64) -> Done {
    let n = 6usize;
    let budget = 500_000usize;
    let reference = -197.7805260034;
    let mut cfg = Config::recommended_tip4p(n);
    cfg.rigid_body_geometry = Some(gmin_water_geometry());
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
    let start = random_tip4p_cluster(n, &mut rng);
    let potential = GminTip4p::new(n);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
    let started = Instant::now();
    let (outcome, _) = search_from(&potential, &cfg, &mut ledger, start.view(), seed).expect("finite start");
    finish(budget, reference, started, outcome)
}

fn finish(budget: usize, reference: f64, started: Instant, outcome: Outcome) -> Done {
    let wall_s = started.elapsed().as_secs_f64();
    let hit = anneal_core::energy_hits(outcome.best, reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let calls = if hit {
        at_hit.unwrap_or(outcome.charged).min(budget)
    } else {
        budget
    };
    Done {
        budget,
        hit,
        calls,
        charged: outcome.charged,
        wall_s,
        best_bits: format!("{:016x}", outcome.best.to_bits()),
        trajectory: trajectory_hash(&outcome),
    }
}

fn trajectory_hash(outcome: &Outcome) -> String {
    let mut hasher = Sha256::new();
    push_f64(&mut hasher, outcome.best);
    push_f64(&mut hasher, outcome.final_energy);
    push_usize(&mut hasher, outcome.charged);
    push_usize(&mut hasher, outcome.hops);
    push_usize(&mut hasher, outcome.basins);
    push_usize(&mut hasher, outcome.accepted_transitions.len());
    for edge in &outcome.accepted_transitions {
        push_usize(&mut hasher, edge.hop);
        hasher.update(edge.action.as_bytes());
        hasher.update([0]);
        push_f64(&mut hasher, edge.from_energy);
        push_f64(&mut hasher, edge.to_energy);
        hasher.update([u8::from(edge.adopted), u8::from(edge.validated)]);
        for value in edge.from_state.iter() {
            push_f64(&mut hasher, *value);
        }
        for value in edge.to_state.iter() {
            push_f64(&mut hasher, *value);
        }
    }
    push_usize(&mut hasher, outcome.improvements.len());
    for (hop, spent, basins, energy) in &outcome.improvements {
        push_usize(&mut hasher, *hop);
        push_usize(&mut hasher, *spent);
        push_usize(&mut hasher, *basins);
        push_f64(&mut hasher, *energy);
    }
    let digest = hasher.finalize();
    let mut text = String::with_capacity(digest.len() * 2);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

fn push_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_le_bytes());
}

fn push_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}
