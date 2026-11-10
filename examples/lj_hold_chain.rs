//! One hundred all-points chains from one geometry.
//!
//! Each chain takes twenty-five steps at temperature 0.8 and step 0.38,
//! then quenches for two hundred warm steps. The flags use the 0.001
//! global-minimum stop and the published Cs energy. The printed energy
//! is what a later 1e-6 count reads.

use std::env;
use std::fs;
use std::process::ExitCode;

use anneal_core::methods::cluster_hopping::ClusterMove;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const CHAINS: usize = 100;
const LENGTH: usize = 25;
const STEP: f64 = 0.38;
const TEMPERATURE: f64 = 0.8;
const RELAX_STEPS: usize = 200;
const SEED: u64 = 1_042_004;
const GM_STOP: f64 = -582.085642;
const CS_ENERGY: f64 = -582.038429;

fn read_xyz(path: &str) -> Array1<f64> {
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("coordinates {err}");
        std::process::exit(2);
    });
    let mut coords = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(x) = parts.next() else {
            continue;
        };
        let (Ok(x), Ok(y), Ok(z)) = (
            x.parse::<f64>(),
            parts.next().unwrap_or("").parse::<f64>(),
            parts.next().unwrap_or("").parse::<f64>(),
        ) else {
            continue;
        };
        coords.extend([x, y, z]);
    }
    Array1::from(coords)
}

fn quench(optim: &mut WarmLbfgs, potential: &PairPotential, x: &Array1<f64>) -> (f64, Array1<f64>) {
    optim.forget();
    let (energy, point, _) = optim.minimize(x.view(), RELAX_STEPS, |v| {
        Some(potential.value_and_gradient(v))
    });
    (energy, point)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: lj-hold-chain <geometry>");
        return ExitCode::from(2);
    };
    let start = read_xyz(&path);
    let n = start.len() / 3;
    if n == 0 || start.len() != 3 * n {
        eprintln!("coordinates");
        return ExitCode::from(2);
    }
    let potential = PairPotential::lennard_jones(n);
    let mut optim = WarmLbfgs::default();
    let (energy0, here0) = quench(&mut optim, &potential, &start);
    let mut rng = StdRng::seed_from_u64(SEED);
    let kernel = ClusterMove::AllPoints { step: STEP };
    let mut gm_hits = 0usize;
    let mut cs_hits = 0usize;
    let mut best = energy0;
    for chain in 1..=CHAINS {
        let mut energy = energy0;
        let mut here = here0.clone();
        for _ in 0..LENGTH {
            let trial = kernel.propose(here.view(), TEMPERATURE, &mut rng);
            let (next, point) = quench(&mut optim, &potential, &trial);
            let accept =
                next <= energy || rng.random::<f64>() < ((energy - next) / TEMPERATURE).exp();
            if accept {
                energy = next;
                here = point;
            }
        }
        if energy <= GM_STOP {
            gm_hits += 1;
        }
        if energy <= CS_ENERGY {
            cs_hits += 1;
        }
        if energy < best {
            best = energy;
        }
        println!(
            "chain\t{chain}\t{energy:.8}\tgm={}\tico={}",
            energy <= GM_STOP,
            energy <= CS_ENERGY
        );
    }
    println!("chains\t{CHAINS}");
    println!("length\t{LENGTH}");
    println!("gm_hits\t{gm_hits}");
    println!("ico_hits\t{cs_hits}");
    println!("best\t{best:.8}");
    ExitCode::SUCCESS
}
