//! Report the minima a corpus holds for one system at one temperature.
//!
//! Usage: `minima_report <root> <signature-hex> <system> <temperature> [tolerance]`.
//! Prints the distinct energies lowest first with how many seeds reached
//! each, then the seed count and the total frames behind the report.

use anneal_core::minima_db::MinimaCorpus;
use std::collections::BTreeSet;

fn signature_digest(text: &str) -> Result<[u8; 32], &'static str> {
    if text.len() != 64 || !text.is_ascii() {
        return Err("system signature must be 64 hexadecimal characters");
    }
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * index..2 * index + 2], 16)
            .map_err(|_| "system signature must be hexadecimal")?;
    }
    Ok(digest)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: minima_report <root> <signature-hex> <system> <temperature> [tolerance]");
        std::process::exit(2);
    }
    let root = &args[1];
    let digest = signature_digest(&args[2]).unwrap_or_else(|error| {
        eprintln!("{}: {error}", args[2]);
        std::process::exit(2);
    });
    let system = &args[3];
    let temperature: f64 = args[4].parse().expect("temperature is a number");
    let tolerance: f64 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(1e-6);
    let corpus = MinimaCorpus::open(root, digest).unwrap_or_else(|e| {
        eprintln!("{root}: {e}");
        std::process::exit(1);
    });
    let minima = corpus.minima(system, temperature).unwrap_or_else(|e| {
        eprintln!("{}: {e}", corpus.path().display());
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
