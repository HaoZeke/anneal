// Copyright (c) anneal developers
//! Evaluator for the native-unit ASE head-to-head.
//!
//! ```text
//! ase_native_eval scales SYSTEM
//! ase_native_eval serve SYSTEM
//! ```
//!
//! `scales` prints the potential parameters and the length and energy
//! scales the anneal preset uses. `serve` reads little-endian f64
//! coordinates (3n of them) and writes the energy and the 3n gradient
//! components, little-endian, until stdin closes. One record is one
//! value-and-gradient. Lennard-Jones and Morse are the untruncated pair
//! potentials. Gold is Gupta::copper_gold(0, 55).

use anneal_core::potentials::{Gupta, PairPotential};
use ndarray::{Array1, ArrayView1};
use std::io::{self, Read, Write};

const LJ_REFERENCES: &[(&str, usize, i64, f64)] = &[
    ("lj38", 38, 400_000, -173.928427),
    ("lj55", 55, 1_000_000, -279.248470),
    ("lj98", 98, 4_000_000, -543.665361),
    ("lj104", 104, 4_000_000, -582.086642),
    ("lj150", 150, 4_000_000, -893.310258),
];

const MORSE_REFERENCES: &[(&str, usize, f64, i64, f64)] = &[
    ("m38r6", 38, 6.0, 400_000, -157.477108),
    ("m38r14", 38, 14.0, 400_000, -144.321054),
    ("m55r6", 55, 6.0, 1_000_000, -250.286609),
    ("m55r14", 55, 14.0, 1_000_000, -220.646208),
];

enum Kind {
    Lj { n: usize, budget: i64, reference: f64 },
    Morse { n: usize, rho: f64, budget: i64, reference: f64 },
    Gold { n: usize, budget: i64, reference: f64 },
}

fn system(name: &str) -> Option<Kind> {
    for (label, n, budget, reference) in LJ_REFERENCES {
        if *label == name {
            return Some(Kind::Lj {
                n: *n,
                budget: *budget,
                reference: *reference,
            });
        }
    }
    for (label, n, rho, budget, reference) in MORSE_REFERENCES {
        if *label == name {
            return Some(Kind::Morse {
                n: *n,
                rho: *rho,
                budget: *budget,
                reference: *reference,
            });
        }
    }
    if name == "au55" {
        return Some(Kind::Gold {
            n: 55,
            budget: 1_000_000,
            reference: -191.03085280,
        });
    }
    None
}

/// Pair-minimum length and well, the probe `examples/ase_panel.rs` uses.
fn gupta_scales() -> (f64, f64) {
    let potential = Gupta::copper_gold(0, 2);
    let mut best_r = 0.0;
    let mut best_e = f64::INFINITY;
    let mut r = 1.8;
    while r <= 4.5 {
        let mut x = Array1::zeros(6);
        x[3] = r;
        let energy = potential.value_and_gradient(x.view()).0;
        if energy < best_e {
            best_e = energy;
            best_r = r;
        }
        r += 0.002;
    }
    (best_r / 2.0_f64.powf(1.0 / 6.0), -best_e)
}

fn morse_length() -> f64 {
    2.0_f64.powf(-1.0 / 6.0)
}

struct Scales {
    n: usize,
    kind: &'static str,
    epsilon: f64,
    sigma: f64,
    rho: f64,
    length_scale: f64,
    energy_scale: f64,
    reference: f64,
    budget: i64,
}

fn scales_of(kind: &Kind) -> Scales {
    match *kind {
        Kind::Lj { n, budget, reference } => Scales {
            n,
            kind: "lj",
            epsilon: 1.0,
            sigma: 1.0,
            rho: 0.0,
            length_scale: 1.0,
            energy_scale: 1.0,
            reference,
            budget,
        },
        Kind::Morse { n, rho, budget, reference } => Scales {
            n,
            kind: "morse",
            epsilon: 1.0,
            sigma: 1.0,
            rho,
            length_scale: morse_length(),
            energy_scale: 1.0,
            reference,
            budget,
        },
        Kind::Gold { n, budget, reference } => {
            let (length_scale, energy_scale) = gupta_scales();
            Scales {
                n,
                kind: "gupta",
                epsilon: 1.0,
                sigma: 1.0,
                rho: 0.0,
                length_scale,
                energy_scale,
                reference,
                budget,
            }
        }
    }
}

fn print_scales(name: &str, scales: &Scales) {
    println!(
        "system={name} n={} kind={} epsilon={:.16e} sigma={:.16e} rho={:.16e} \
         length_scale={:.16e} energy_scale={:.16e} reference={:.16e} \
         reference_bits={} budget={}",
        scales.n,
        scales.kind,
        scales.epsilon,
        scales.sigma,
        scales.rho,
        scales.length_scale,
        scales.energy_scale,
        scales.reference,
        scales.reference.to_bits(),
        scales.budget,
    );
}

enum Pot {
    Pair(PairPotential),
    Gold(Gupta),
}

impl Pot {
    fn new(kind: &Kind) -> Self {
        match *kind {
            Kind::Lj { n, .. } => Pot::Pair(PairPotential::lennard_jones(n)),
            Kind::Morse { n, rho, .. } => Pot::Pair(PairPotential::morse(n, rho)),
            Kind::Gold { n, .. } => Pot::Gold(Gupta::copper_gold(0, n)),
        }
    }

    fn eval(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let view = ArrayView1::from(x);
        let (energy, gradient) = match self {
            Pot::Pair(potential) => potential.value_and_gradient(view),
            Pot::Gold(potential) => potential.value_and_gradient(view),
        };
        (energy, gradient.to_vec())
    }
}

fn read_exact(input: &mut impl Read, bytes: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0;
    while filled < bytes.len() {
        match input.read(&mut bytes[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short coordinate record",
                ))
            }
            Ok(n) => filled += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(true)
}

fn serve(pot: &Pot, n: usize) -> io::Result<()> {
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut request = vec![0u8; 8 * 3 * n];
    let mut coords = vec![0.0f64; 3 * n];
    let mut reply = vec![0u8; 8 * (1 + 3 * n)];
    loop {
        if !read_exact(&mut input, &mut request)? {
            return Ok(());
        }
        for (index, slot) in coords.iter_mut().enumerate() {
            let start = 8 * index;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&request[start..start + 8]);
            *slot = f64::from_le_bytes(bytes);
        }
        let (energy, gradient) = pot.eval(&coords);
        reply[..8].copy_from_slice(&energy.to_le_bytes());
        for (index, component) in gradient.iter().enumerate() {
            let start = 8 * (1 + index);
            reply[start..start + 8].copy_from_slice(&component.to_le_bytes());
        }
        output.write_all(&reply)?;
        output.flush()?;
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("");
    let name = args.get(1).map(String::as_str).unwrap_or("");
    let Some(kind) = system(name) else {
        eprintln!("unknown system {name}");
        std::process::exit(2);
    };
    let scales = scales_of(&kind);
    match command {
        "scales" => print_scales(name, &scales),
        "serve" => {
            let pot = Pot::new(&kind);
            if let Err(err) = serve(&pot, scales.n) {
                eprintln!("serve: {err}");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("usage: ase_native_eval scales|serve SYSTEM");
            std::process::exit(2);
        }
    }
}
