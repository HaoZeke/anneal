//! Write a Lennard-Jones minima book for `anneal landfold`.
//!
//! ```text
//! landfold_lj_book N COUNT SEED OUT.min [FIXTURE.xyz ...]
//! ```
//!
//! Eight basin-hopping chains run in parallel. With `k` fixtures, even
//! chain `c` starts from fixture `(c / 2) mod k` and odd chains from a
//! random cluster. With none, every chain starts from a random cluster.
//! Each step displaces every atom uniformly within 0.4 reduced units,
//! quenches with L-BFGS, and accepts by Metropolis at kT = 0.8.
//!
//! Every converged quench is a candidate. Minima whose energies agree to
//! 1e-6 are one minimum. The book reads the chains step by step, chain 0
//! first at each step, and keeps the first COUNT distinct minima. Each
//! minimum is one `.min` line: energy, then 3N coordinates.

use std::env;
use std::fs;

use anneal_core::methods::cluster_hopping::random_cluster;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHAINS: usize = 8;
const KICK: f64 = 0.4;
const KT: f64 = 0.8;
const GTOL: f64 = 1e-4;

fn load_xyz(path: &str) -> Vec<f64> {
    let text = fs::read_to_string(path).unwrap_or_else(|err| panic!("{path}: {err}"));
    text.lines()
        .skip(2)
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .take(3)
                .map(|t| t.parse::<f64>().expect("xyz coordinate"))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn quench(pot: &PairPotential, x: &Array1<f64>) -> Option<(f64, Array1<f64>)> {
    let mut opt = WarmLbfgs::default();
    let (energy, minimum, _) = opt.minimize(x.view(), 4000, |v| Some(pot.value_and_gradient(v)));
    let (_, g) = pot.value_and_gradient(minimum.view());
    let ginf = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    (energy.is_finite() && ginf < GTOL).then_some((energy, minimum))
}

fn chain(n: usize, steps: usize, seed: u64, start: Option<Vec<f64>>) -> Vec<(f64, Vec<f64>)> {
    let pot = PairPotential::lennard_jones(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let x0 = start.map_or_else(|| random_cluster(n, 0.7, 0.85, &mut rng), Array1::from_vec);
    let (mut energy, mut here) = quench(&pot, &x0).expect("start quenches");
    let mut found = vec![(energy, here.to_vec())];
    for _ in 0..steps {
        let mut trial = here.clone();
        for v in trial.iter_mut() {
            *v += rng.random_range(-KICK..KICK);
        }
        let Some((e, x)) = quench(&pot, &trial) else {
            continue;
        };
        found.push((e, x.to_vec()));
        if e <= energy || rng.random::<f64>() < (-(e - energy) / KT).exp() {
            energy = e;
            here = x;
        }
    }
    found
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    assert!(
        args.len() >= 4,
        "usage: landfold_lj_book N COUNT SEED OUT.min [FIXTURE.xyz ...]"
    );
    let n: usize = args[0].parse().expect("N");
    let count: usize = args[1].parse().expect("COUNT");
    let seed: u64 = args[2].parse().expect("SEED");
    let out = &args[3];
    let fixtures: Vec<Vec<f64>> = args[4..].iter().map(|p| load_xyz(p)).collect();
    for f in &fixtures {
        assert_eq!(f.len(), 3 * n, "fixture has {} values, not 3N", f.len());
    }
    let steps = count.div_ceil(CHAINS) * 4;
    let chains: Vec<Vec<(f64, Vec<f64>)>> = (0..CHAINS)
        .into_par_iter()
        .map(|c| {
            let start = (!fixtures.is_empty() && c % 2 == 0)
                .then(|| fixtures[(c / 2) % fixtures.len()].clone());
            chain(
                n,
                steps,
                seed.wrapping_mul(1000).wrapping_add(c as u64),
                start,
            )
        })
        .collect();
    let mut keys: Vec<f64> = Vec::new();
    let mut book = String::new();
    let mut kept = 0;
    'rounds: for step in 0..=steps {
        for found in &chains {
            let Some((e, x)) = found.get(step) else {
                continue;
            };
            if keys.iter().any(|k| (k - e).abs() < 1e-6) {
                continue;
            }
            keys.push(*e);
            book.push_str(&format!("{e:.10}"));
            for v in x {
                book.push_str(&format!(" {v:.8}"));
            }
            book.push('\n');
            kept += 1;
            if kept == count {
                break 'rounds;
            }
        }
    }
    fs::write(out, book).unwrap_or_else(|err| panic!("{out}: {err}"));
    let lowest = keys.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!("LJ{n}: {kept} distinct minima, lowest {lowest:.6}, written to {out}");
}
