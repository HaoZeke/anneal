// Copyright (c) anneal developers
//! One (H2O)6 energy and gradient through the anneal arm's GFN2 callback.
//!
//! The returned pair is the body of `ForceObjective::value_and_gradient`
//! in `examples/gfn2_water_h2h.rs`: the rgpot handle, then the O-H wall
//! when that wall is nonzero. The example's call counter and optional
//! target stop do not change the pair.

#[path = "../../../../examples/common/rgpot_eindir.rs"]
mod rgpot_eindir;
#[path = "../../../../examples/common/water_oh_wall.rs"]
mod water_oh_wall;

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Mutex;

use anneal_core::catalog::molecular::{bohr_angstrom, hartree_ev};
use eindir_core::gradient::DifferentiableObjective;
use ndarray::ArrayView1;
use rgpot_eindir::RgpotObjective;
use water_oh_wall::oh_wall;

const MOLECULES: usize = 6;

struct AnnealCallback {
    dim: usize,
    n_molecules: usize,
    eval: Mutex<Box<dyn FnMut(&[f64]) -> (f64, Vec<f64>) + Send>>,
}

impl AnnealCallback {
    fn value_and_gradient(&self, coordinates: &[f64]) -> (f64, Vec<f64>) {
        let (energy, mut gradient) = (self.eval.lock().expect("engine lock").as_mut())(coordinates);
        if gradient.len() != self.dim || !energy.is_finite() {
            return (f64::NAN, vec![f64::NAN; self.dim]);
        }
        let (wall, wall_gradient) = oh_wall(coordinates, self.n_molecules);
        if wall != 0.0 {
            for (component, extra) in gradient.iter_mut().zip(wall_gradient) {
                *component += extra;
            }
        }
        (energy + wall, gradient)
    }
}

fn read_xyz(path: &Path) -> Vec<f64> {
    let text = fs::read_to_string(path).unwrap_or_else(|error| panic!("read xyz: {error}"));
    let mut coords = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index < 2 || line.split_whitespace().count() < 4 {
            continue;
        }
        let mut parts = line.split_whitespace();
        let _symbol = parts.next();
        for _axis in 0..3 {
            let value: f64 = parts
                .next()
                .unwrap_or_else(|| panic!("short xyz line"))
                .parse()
                .unwrap_or_else(|error| panic!("xyz coordinate: {error}"));
            coords.push(value);
        }
    }
    coords
}

fn open_callback(restart: bool) -> AnnealCallback {
    let atmnrs: Vec<i32> = (0..MOLECULES).flat_map(|_| [8, 1, 1]).collect();
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let mut pot = RgpotObjective::xtb(&atmnrs, box_);
    if restart {
        pot.restart();
    }
    let dim = 3 * atmnrs.len();
    AnnealCallback {
        dim,
        n_molecules: MOLECULES,
        eval: Mutex::new(Box::new(move |coordinates: &[f64]| {
            let (energy, gradient) = pot
                .wrapper()
                .value_and_gradient(ArrayView1::from(coordinates));
            let values = if gradient.is_standard_layout() {
                gradient.to_vec()
            } else {
                gradient.iter().copied().collect()
            };
            (energy, values)
        })),
    }
}

fn write_vec(out: &mut String, label: &str, values: &[f64]) {
    out.push_str(label);
    out.push('\n');
    for value in values {
        out.push_str(&format!("{value:.16e}\n"));
    }
}

fn fd(coords: &[f64], step: f64, restart: bool, report: &Path) -> ExitCode {
    let callback = open_callback(restart);
    let (energy, gradient) = callback.value_and_gradient(coords);
    if !energy.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
        eprintln!("anneal callback returned a non-finite value");
        return ExitCode::from(1);
    }
    let mut finite_difference = vec![0.0; coords.len()];
    for index in 0..coords.len() {
        let mut plus = coords.to_vec();
        let mut minus = coords.to_vec();
        plus[index] += step;
        minus[index] -= step;
        let (above, _) = callback.value_and_gradient(&plus);
        let (below, _) = callback.value_and_gradient(&minus);
        if !above.is_finite() || !below.is_finite() {
            eprintln!("finite-difference energy was non-finite");
            return ExitCode::from(1);
        }
        finite_difference[index] = (above - below) / (2.0 * step);
    }
    let norm = gradient.iter().map(|value| value * value).sum::<f64>().sqrt();
    let delta = if norm > 0.0 {
        let step_len = 0.01 / norm;
        let moved: Vec<f64> = coords
            .iter()
            .zip(&gradient)
            .map(|(coordinate, component)| coordinate - step_len * component)
            .collect();
        let (again, _) = callback.value_and_gradient(coords);
        let (lowered, _) = callback.value_and_gradient(&moved);
        lowered - again
    } else {
        f64::NAN
    };
    let mut text = String::new();
    text.push_str(&format!("hartree_eV {:.16e}\n", hartree_ev()));
    text.push_str(&format!("bohr_A {:.16e}\n", bohr_angstrom()));
    text.push_str(&format!("restart {restart}\n"));
    text.push_str(&format!("energy_eV {energy:.16e}\n"));
    text.push_str(&format!("downhill_delta_eV {delta:.16e}\n"));
    text.push_str(&format!("n {}\n", gradient.len()));
    write_vec(&mut text, "g", &gradient);
    write_vec(&mut text, "fd", &finite_difference);
    fs::write(report, text).unwrap_or_else(|error| panic!("write report: {error}"));
    ExitCode::SUCCESS
}

fn once(coords: &[f64], restart: bool, report: &Path) -> ExitCode {
    let callback = open_callback(restart);
    let (energy, gradient) = callback.value_and_gradient(coords);
    if !energy.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
        eprintln!("anneal callback returned a non-finite value");
        return ExitCode::from(1);
    }
    let mut text = String::new();
    text.push_str(&format!("energy_eV {energy:.16e}\n"));
    text.push_str(&format!("n {}\n", gradient.len()));
    fs::write(report, text).unwrap_or_else(|error| panic!("write report: {error}"));
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("");
    match mode {
        "fd" => {
            if args.len() != 5 {
                eprintln!("usage: gfn2-fd-gate fd <xyz> <step_A> <restart> <report>");
                return ExitCode::from(2);
            }
            let coords = read_xyz(Path::new(&args[1]));
            if coords.len() != 9 * MOLECULES {
                eprintln!("expected {} coordinates", 9 * MOLECULES);
                return ExitCode::from(2);
            }
            let step: f64 = args[2].parse().expect("step");
            let restart = args[3] == "1";
            fd(&coords, step, restart, Path::new(&args[4]))
        }
        "once" => {
            if args.len() != 4 {
                eprintln!("usage: gfn2-fd-gate once <xyz> <restart> <report>");
                return ExitCode::from(2);
            }
            let coords = read_xyz(Path::new(&args[1]));
            if coords.len() != 9 * MOLECULES {
                eprintln!("expected {} coordinates", 9 * MOLECULES);
                return ExitCode::from(2);
            }
            let restart = args[2] == "1";
            once(&coords, restart, Path::new(&args[3]))
        }
        _ => {
            eprintln!("usage: gfn2-fd-gate fd|once ...");
            ExitCode::from(2)
        }
    }
}
