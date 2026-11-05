//! One weak-scaling cell.
//!
//! Usage: `weak_scale <pt|population|dmc> <workers> <seed>`

use anneal_core::weak_scale::run_cell;

fn main() {
    let mut args = std::env::args().skip(1);
    let family = args.next().unwrap_or_default();
    let workers = args
        .next()
        .and_then(|text| text.parse::<usize>().ok())
        .unwrap_or(0);
    let seed = args
        .next()
        .and_then(|text| text.parse::<u64>().ok())
        .unwrap_or(0);
    if !matches!(family.as_str(), "pt" | "population" | "dmc") || workers < 1 {
        eprintln!("usage: weak_scale <pt|population|dmc> <workers> <seed>");
        std::process::exit(2);
    }
    let cell = run_cell(&family, workers, seed);
    println!(
        "family={family} workers={} pool_threads={} atoms={} replicas={} charged={} wall_s={:.9} best={:.16e} seed={seed}",
        cell.workers,
        cell.pool_threads,
        cell.atoms,
        cell.replicas,
        cell.charged,
        cell.wall_s,
        cell.best,
    );
}
