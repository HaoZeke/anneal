// Copyright (c) anneal developers
//! Energy and gradient server for the standalone minima-hopping harness.
//!
//! ```text
//! mhpkg_eval spec <cell>
//! mhpkg_eval energy <cell> <path>
//! mhpkg_eval serve <cell>
//! ```
//!
//! `serve` reads one line of Cartesian coordinates and writes
//! `energy seconds gradient...`. One line is one `value_and_gradient`.
//! An empty line or `quit` closes the server. The potentials are the
//! same constructors the head-to-head examples quench:
//! `PairPotential::lennard_jones`, `PairPotential::morse`, and
//! `Gupta::copper_gold`.

use anneal_core::potentials::{Gupta, PairPotential};
use ndarray::{Array1, ArrayView1};
use std::io::{BufRead, Write};
use std::time::Instant;

enum Pot {
    Pair(PairPotential),
    Gupta(Gupta),
}

impl Pot {
    fn eval(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Pot::Pair(pot) => pot.value_and_gradient(x),
            Pot::Gupta(pot) => pot.value_and_gradient(x),
        }
    }
}

struct Cell {
    n: usize,
    reference: f64,
    budget: usize,
    kind: &'static str,
}

fn cell(name: &str) -> Option<(Cell, Pot)> {
    let spec = match name {
        "lj38" => ("lj", 38, -173.928427, 400_000),
        "lj75" => ("lj", 75, -397.492331, 4_000_000),
        "lj98" => ("lj", 98, -543.665361, 4_000_000),
        "lj104" => ("lj", 104, -582.086642, 4_000_000),
        "m80r6" => ("morse", 80, -378.333471, 4_000_000),
        "m80r14" => ("morse", 80, -340.811371, 4_000_000),
        "au38" => ("gupta", 38, -130.54001708, 400_000),
        _ => return None,
    };
    let (kind, n, reference, budget) = spec;
    let pot = match name {
        "lj38" | "lj75" | "lj98" | "lj104" => Pot::Pair(PairPotential::lennard_jones(n)),
        "m80r6" => Pot::Pair(PairPotential::morse(n, 6.0)),
        "m80r14" => Pot::Pair(PairPotential::morse(n, 14.0)),
        "au38" => Pot::Gupta(Gupta::copper_gold(0, n)),
        _ => return None,
    };
    Some((
        Cell {
            n,
            reference,
            budget,
            kind,
        },
        pot,
    ))
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("spec");
    let status = match cmd {
        "spec" => cmd_spec(&args),
        "energy" => cmd_energy(&args),
        "serve" => cmd_serve(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

fn cmd_spec(args: &[String]) -> i32 {
    let Some(name) = args.get(1).map(String::as_str) else {
        eprintln!("spec needs a cell");
        return 2;
    };
    let Some((spec, _)) = cell(name) else {
        eprintln!("unknown cell {name}");
        return 2;
    };
    println!(
        "cell={name} n={} dim={} kind={} reference={:.8} budget={}",
        spec.n,
        3 * spec.n,
        spec.kind,
        spec.reference,
        spec.budget
    );
    0
}

fn load_coords(name: &str, path: &str) -> Result<(Cell, Pot, Array1<f64>), i32> {
    let Some((spec, pot)) = cell(name) else {
        eprintln!("unknown cell {name}");
        return Err(2);
    };
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("coords: {err}");
            return Err(1);
        }
    };
    let mut values = Vec::new();
    for line in text.lines() {
        for word in line.split_whitespace() {
            match word.parse::<f64>() {
                Ok(value) => values.push(value),
                Err(_) => {
                    eprintln!("coords: {word} is not a number");
                    return Err(1);
                }
            }
        }
    }
    if values.len() != 3 * spec.n {
        eprintln!("coords: expected {} numbers", 3 * spec.n);
        return Err(1);
    }
    Ok((spec, pot, Array1::from_vec(values)))
}

fn cmd_energy(args: &[String]) -> i32 {
    let Some(name) = args.get(1).map(String::as_str) else {
        eprintln!("energy needs a cell");
        return 2;
    };
    let Some(path) = args.get(2).map(String::as_str) else {
        eprintln!("energy needs a coordinate file");
        return 2;
    };
    let (_spec, pot, coords) = match load_coords(name, path) {
        Ok(loaded) => loaded,
        Err(status) => return status,
    };
    let (energy, _) = pot.eval(coords.view());
    println!("{energy:.10}");
    0
}

fn cmd_serve(args: &[String]) -> i32 {
    let Some(name) = args.get(1).map(String::as_str) else {
        eprintln!("serve needs a cell");
        return 2;
    };
    let Some((spec, pot)) = cell(name) else {
        eprintln!("unknown cell {name}");
        return 2;
    };
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                eprintln!("serve: {err}");
                return 1;
            }
        };
        let line = line.trim();
        if line.is_empty() || line == "quit" {
            break;
        }
        let mut values = Vec::with_capacity(3 * spec.n);
        let mut bad = false;
        for word in line.split_whitespace() {
            match word.parse::<f64>() {
                Ok(value) => values.push(value),
                Err(_) => {
                    eprintln!("serve: {word} is not a number");
                    bad = true;
                    break;
                }
            }
        }
        if bad {
            return 1;
        }
        if values.len() != 3 * spec.n {
            eprintln!("serve: expected {} numbers", 3 * spec.n);
            return 1;
        }
        let coords = Array1::from_vec(values);
        let mark = Instant::now();
        let (energy, gradient) = pot.eval(coords.view());
        let seconds = mark.elapsed().as_secs_f64();
        if write!(stdout, "{energy:.16} {seconds:.9}").is_err() {
            return 1;
        }
        for component in gradient.iter() {
            if write!(stdout, " {component:.16}").is_err() {
                return 1;
            }
        }
        if writeln!(stdout).is_err() || stdout.flush().is_err() {
            return 1;
        }
    }
    0
}
