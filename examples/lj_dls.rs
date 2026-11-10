// Copyright (c) anneal developers
//! One published dynamic-lattice search at a charged budget.
//!
//! Usage:
//! `lj_dls <n> <budget> <seed> <lowest-index|uniform> <stop> <prefix>`
//!
//! `stop` is an energy that ends the run, or `-` to spend the budget.
//! The start is the same sphere draw the head-to-head driver uses.
//! Each new lowest energy is written to `<prefix>.trace.csv`.

use anneal_core::methods::cluster_hopping::{Config, Ledger, random_cluster_in_radius};
use anneal_core::methods::dls_published::{self, TieCut};
use anneal_core::methods::lattice_search::Quench;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fmt::Write as _;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 6, "see the usage line in the source");
    let n: usize = args[0].parse().expect("n");
    let budget: usize = args[1].parse().expect("budget");
    let seed: u64 = args[2].parse().expect("seed");
    let tie = match args[3].as_str() {
        "lowest-index" => TieCut::LowestIndex,
        "uniform" => TieCut::Uniform,
        other => panic!("unknown tie cut {other}"),
    };
    let stop = if args[4] == "-" {
        None
    } else {
        Some(args[4].parse::<f64>().expect("stop"))
    };
    let prefix = &args[5];
    let cfg = Config::for_cluster(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let quench = Quench {
        max_iterations: 400,
        rms_tolerance: 1.0e-4,
        ..Quench::default()
    };
    let mut ledger = Ledger::new(budget);
    let started = std::time::Instant::now();
    let outcome = dls_published::run(
        n,
        &mut ledger,
        &quench,
        start.as_slice().expect("contiguous start"),
        cfg.start_radius(),
        cfg.min_separation,
        &mut rng,
        tie,
        stop,
    );
    let wall = started.elapsed().as_secs_f64();
    let mut trace = String::from("charged,energy\n");
    for (_, charged, _, energy) in &outcome.improvements {
        writeln!(trace, "{charged},{energy:.10}").unwrap();
    }
    std::fs::write(format!("{prefix}.trace.csv"), trace).expect("trace");
    if let Some(state) = outcome.best_state.as_ref() {
        let mut xyz = format!("{n}\nenergy {:.10}\n", outcome.best);
        for p in state.exact_chunks(3) {
            writeln!(xyz, "X {:.10} {:.10} {:.10}", p[0], p[1], p[2]).unwrap();
        }
        std::fs::write(format!("{prefix}.xyz"), xyz).expect("geometry");
    }
    println!("n: {n}");
    println!("seed: {seed}");
    println!("tie: {}", args[3]);
    println!("budget: {budget}");
    println!("charged: {}", ledger.spent());
    println!("best: {:.10}", outcome.best);
    println!("generations: {}", outcome.hops);
    println!("restarts: {}", outcome.basins);
    println!("wall_seconds: {wall:.3}");
}
