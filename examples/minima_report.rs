//! Report the minima a corpus holds for one system at one temperature.
//!
//! Usage: `minima_report <corpus> <system> <temperature> [tolerance]`.
//! Prints the distinct energies lowest first with how many seeds reached
//! each, then the seed count and the total frames behind the report.

use anneal_core::minima_db::MinimaCorpus;
use std::collections::BTreeSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: minima_report <corpus> <system> <temperature> [tolerance]");
        std::process::exit(2);
    }
    let path = &args[1];
    let system = &args[2];
    let temperature: f64 = args[3].parse().expect("temperature is a number");
    let tolerance: f64 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1e-6);
    let corpus = MinimaCorpus::open(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let minima = corpus.minima(system, temperature).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let seeds: BTreeSet<u64> = minima.iter().map(|m| m.set.seed).collect();
    let mut distinct: Vec<(f64, BTreeSet<u64>)> = Vec::new();
    for minimum in &minima {
        match distinct
            .iter_mut()
            .find(|(energy, _)| (energy - minimum.energy).abs() <= tolerance)
        {
            Some((_, reached)) => {
                reached.insert(minimum.set.seed);
            }
            None => distinct.push((minimum.energy, BTreeSet::from([minimum.set.seed]))),
        }
    }
    println!(
        "{system} at T={temperature}: {} distinct minima from {} seeds ({} frames)",
        distinct.len(),
        seeds.len(),
        minima.len()
    );
    for (energy, reached) in distinct.iter().take(40) {
        println!("  {energy:.6}  seeds {}/{}", reached.len(), seeds.len());
    }
}
