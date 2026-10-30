//! Bitwise digest of `Config::recommended` Lennard-Jones searches.
//!
//! Usage: `lj_digest [lj38|lj55|lj75|all]`. Each line hashes one seed's
//! outcome: the best value and state, the final state, the charged count,
//! hops, basins, screened and returned counts, and every improvement record,
//! all as IEEE-754 bit patterns. The resolved configuration digest is
//! printed per system. Two builds agree on Lennard-Jones only if every line
//! agrees.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;
use sha2::{Digest, Sha256};

const SYSTEMS: [(&str, usize, usize); 3] = [
    ("lj38", 38, 60_000),
    ("lj55", 55, 120_000),
    ("lj75", 75, 240_000),
];
const SEED_START: u64 = 2000;
const SEED_COUNT: u64 = 8;

fn bits(hasher: &mut Sha256, values: impl IntoIterator<Item = f64>) {
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

fn count(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() {
    let pick = std::env::args().nth(1).unwrap_or_else(|| "all".to_owned());
    let mut total = Sha256::new();
    for (name, n, budget) in SYSTEMS {
        if pick != "all" && pick != name {
            continue;
        }
        let cfg = Config::recommended(n);
        let config_digest = cfg.resolved_sha256().expect("config serializes");
        println!("{name} config {config_digest} budget {budget}");
        total.update(config_digest.as_bytes());
        let potential = PairPotential::lennard_jones(n);
        for seed in SEED_START..SEED_START + SEED_COUNT {
            let mut ledger = Ledger::new(budget);
            let (outcome, _) = search(&potential, &cfg, &mut ledger, seed).expect("finite start");
            let mut hasher = Sha256::new();
            bits(&mut hasher, [outcome.best, outcome.final_energy]);
            for state in [&outcome.best_state, &outcome.final_state] {
                match state {
                    Some(x) => bits(&mut hasher, x.iter().copied()),
                    None => count(&mut hasher, usize::MAX),
                }
            }
            for value in [
                outcome.charged,
                outcome.hops,
                outcome.basins,
                outcome.screened_out,
                outcome.returned,
            ] {
                count(&mut hasher, value);
            }
            for (hop, charged, basins, energy) in &outcome.improvements {
                count(&mut hasher, *hop);
                count(&mut hasher, *charged);
                count(&mut hasher, *basins);
                bits(&mut hasher, [*energy]);
            }
            let digest = hasher.finalize();
            total.update(digest);
            println!(
                "{name} seed {seed} best {:.6} charged {} hops {} digest {}",
                outcome.best,
                outcome.charged,
                outcome.hops,
                hex(&digest)
            );
        }
    }
    println!("total {}", hex(&total.finalize()));
}
