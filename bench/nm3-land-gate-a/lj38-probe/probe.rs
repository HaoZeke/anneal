// Copyright (c) anneal developers

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::search_lennard_jones;
use anneal_core::potentials::{ClusterKernels, PairPotential, install_cluster_kernels};

const REFERENCE: f64 = -173.928427;

static POTENTIAL_NS: AtomicU64 = AtomicU64::new(0);
static KERNEL_CALLS: AtomicU64 = AtomicU64::new(0);

fn timed_cluster_lj(coordinates: &[f64], energy: &mut f64, gradient: &mut [f64]) {
    let started = Instant::now();
    rgpot_pair::cluster_lj(coordinates, energy, gradient);
    POTENTIAL_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    KERNEL_CALLS.fetch_add(1, Ordering::Relaxed);
}

fn affinity_list() -> String {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    text.lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .map(str::trim)
        .unwrap_or("")
        .to_string()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(seed) = args.next().and_then(|value| value.parse::<u64>().ok()) else {
        eprintln!("a seed and a budget are required");
        std::process::exit(2);
    };
    let Some(budget) = args.next().and_then(|value| value.parse::<usize>().ok()) else {
        eprintln!("a seed and a budget are required");
        std::process::exit(2);
    };
    if budget == 0 {
        eprintln!("budget must be positive");
        std::process::exit(2);
    }
    install_cluster_kernels(ClusterKernels::new(
        timed_cluster_lj,
        rgpot_pair::cambridge_morse,
        rgpot_pair::gupta,
        rgpot_pair::binary_lj,
    ));
    POTENTIAL_NS.store(0, Ordering::Relaxed);
    KERNEL_CALLS.store(0, Ordering::Relaxed);

    let mut cfg = Config::recommended(38);
    cfg.relax_steps = 200;
    cfg.move_library = MoveLibrary::LeanBurst;
    cfg.trail_on_stall = false;
    cfg.escape_on_stall = false;
    cfg.restart_on_stall = false;
    cfg.bayes_screen = false;
    cfg.surfaces.clear();

    let potential = PairPotential::lennard_jones(cfg.n_points);
    let mut ledger = Ledger::new(budget);
    let started = Instant::now();
    let (outcome, _) =
        search_lennard_jones(&potential, &cfg, &mut ledger, seed).expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let potential_s = POTENTIAL_NS.load(Ordering::Relaxed) as f64 / 1.0e9;
    let kernel_calls = KERNEL_CALLS.load(Ordering::Relaxed);
    let outside = (wall - potential_s).max(0.0);
    let hit = anneal_core::energy_hits(outcome.best, REFERENCE, anneal_core::SEARCH_HIT_TOL);
    let calls = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| {
            anneal_core::energy_hits(*energy, REFERENCE, anneal_core::SEARCH_HIT_TOL)
        })
        .map(|(_, spent, _, _)| *spent)
        .unwrap_or(budget);
    println!(
        "system,seed,budget,hit,calls_to_hit,charged,wall_s,potential_s,outside_s,best,affinity,kernel_calls"
    );
    println!(
        "lj38-plain,{seed},{budget},{},{calls},{},{wall:.9},{potential_s:.9},{outside:.9},{:.12},{},{kernel_calls}",
        i32::from(hit),
        ledger.spent(),
        outcome.best,
        affinity_list(),
    );
}
