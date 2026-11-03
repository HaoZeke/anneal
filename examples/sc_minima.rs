//! Read the Sutton-Chen minima `sc_h2h` wrote to readcon-db and say what they are.
//!
//! Prints the distinct energies lowest first, how many runs reached each,
//! and the common-neighbour signatures of one structure at that energy.
//! With an output directory, the lowest `top` distinct minima are also
//! written there as Cartesian coordinate (XYZ) files.
//!
//! ```text
//! sc_minima <root> <signature-hex> <system> <temperature> <length> [top] [xyz-dir]
//! ```
//!
//! `length` is the preset length of the run, `nn / 2^(1/6)`. The bond
//! cutoff is `CONTACT_CUTOFF`, 1.35, times it, between the first and second shells.

use anneal_core::minima_db::MinimaCorpus;
use anneal_core::soap::CONTACT_CUTOFF;
use anneal_core::structure::cna;
use ndarray::ArrayView1;
use std::collections::BTreeSet;
use std::io::Write as _;

/// Bytes of a corpus signature, a SHA-256 digest.
const SIGNATURE_BYTES: usize = 32;
/// Last bin of the coordination histogram, which also counts every higher coordination.
const MAX_COORDINATION: usize = 13;
/// Distinct minima printed when the command line gives no count.
const DEFAULT_TOP: usize = 10;
/// Energy within which neighbouring minima, lowest first, count as one.
const ENERGY_GROUPING: f64 = 1e-4;

fn digest(text: &str) -> [u8; SIGNATURE_BYTES] {
    assert!(
        text.len() == 2 * SIGNATURE_BYTES,
        "signature must be 64 hexadecimal characters"
    );
    let mut d = [0u8; SIGNATURE_BYTES];
    for (i, b) in d.iter_mut().enumerate() {
        *b = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hexadecimal signature");
    }
    d
}

/// Neighbour counts within `cutoff`, as a histogram from 0 to [`MAX_COORDINATION`].
fn coordination(x: ArrayView1<f64>, cutoff: f64) -> [usize; MAX_COORDINATION + 1] {
    let n = x.len() / 3;
    let mut z = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d2: f64 = (0..3).map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2)).sum();
            if d2 < cutoff * cutoff {
                z[i] += 1;
                z[j] += 1;
            }
        }
    }
    let mut h = [0usize; MAX_COORDINATION + 1];
    for c in z {
        h[c.min(MAX_COORDINATION)] += 1;
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: sc_minima <root> <signature-hex> <system> <temperature> <length> [top] [xyz-dir]"
        );
        std::process::exit(2);
    }
    let corpus = MinimaCorpus::open(&args[1], digest(&args[2])).expect("corpus");
    let system = &args[3];
    let temperature: f64 = args[4].parse().expect("temperature");
    let length: f64 = args[5].parse().expect("length");
    let top: usize = args.get(6).map_or(DEFAULT_TOP, |v| v.parse().expect("top"));
    let out = args.get(7);
    let cutoff = CONTACT_CUTOFF * length;
    let minima = corpus.minima(system, temperature).expect("minima");
    let mut groups: Vec<(f64, BTreeSet<u64>, Vec<f64>)> = Vec::new();
    for m in &minima {
        match groups.last_mut() {
            Some((e, seeds, _)) if (m.energy - *e).abs() <= ENERGY_GROUPING => {
                seeds.insert(m.set.seed);
            }
            _ => groups.push((
                m.energy,
                BTreeSet::from([m.set.seed]),
                m.coordinates.clone(),
            )),
        }
    }
    let runs: BTreeSet<u64> = minima.iter().map(|m| m.set.seed).collect();
    println!("system: {system}");
    println!("frames: {}", minima.len());
    println!("chains: {}", runs.len());
    println!("distinct: {}", groups.len());
    println!("energy chains bonds 555 421 422 other coordination(z:count)");
    for (k, (e, seeds, x)) in groups.iter().take(top).enumerate() {
        let n = x.len() / 3;
        let view = ArrayView1::from(x.as_slice());
        let c = cna(view, n, cutoff);
        let named = c.get((5, 5, 5)) + c.get((4, 2, 1)) + c.get((4, 2, 2));
        let z = coordination(view, cutoff);
        let zs: Vec<String> = z
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(i, c)| format!("{i}:{c}"))
            .collect();
        println!(
            "{e:.6} {} {} {} {} {} {} {}",
            seeds.len(),
            c.bonds,
            c.get((5, 5, 5)),
            c.get((4, 2, 1)),
            c.get((4, 2, 2)),
            c.bonds.saturating_sub(named),
            zs.join(",")
        );
        if let Some(dir) = out {
            std::fs::create_dir_all(dir).expect("xyz directory");
            let mut f = std::fs::File::create(format!("{dir}/{system}-{k:02}.xyz")).expect("xyz");
            writeln!(f, "{n}\nenergy={e:.10}").expect("xyz");
            for i in 0..n {
                writeln!(
                    f,
                    "X {:.8} {:.8} {:.8}",
                    x[3 * i],
                    x[3 * i + 1],
                    x[3 * i + 2]
                )
                .expect("xyz");
            }
        }
    }
}
