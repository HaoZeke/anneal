// Copyright (c) anneal developers
//! One warm GFN2 session against a fresh guess on every frame in a directory.
//!
//! ```text
//! gfn2_warm_gate <frames directory>
//! ```
//!
//! `*.xyz` files are walked in name order on one engine, then again with
//! `restart` before each frame. The electronic temperature is
//! `RGPOT_XTB_ETEMP` (300 K when unset). `RGPOT_XTB_WARM_MIN_GAP`, when
//! set, is the gap in eV below which a stored wavefunction is not reused.

#[path = "common/rgpot_eindir.rs"]
mod rgpot_eindir;

use std::fs;
use std::path::Path;

use eindir_core::gradient::DifferentiableObjective;
use ndarray::ArrayView1;
use rgpot_eindir::RgpotObjective;

fn atomic_number(symbol: &str) -> i32 {
    match symbol {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "Si" => 14,
        other => panic!("unsupported element {other}"),
    }
}

fn read_xyz(path: &Path) -> (Vec<i32>, Vec<f64>) {
    let text =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let mut numbers = Vec::new();
    let mut coords = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index < 2 || line.split_whitespace().count() < 4 {
            continue;
        }
        let mut parts = line.split_whitespace();
        let symbol = parts.next().unwrap();
        numbers.push(atomic_number(symbol));
        for _axis in 0..3 {
            coords.push(
                parts
                    .next()
                    .unwrap_or_else(|| panic!("short xyz line in {}", path.display()))
                    .parse::<f64>()
                    .unwrap_or_else(|error| panic!("xyz coordinate: {error}")),
            );
        }
    }
    (numbers, coords)
}

fn energy(pot: &RgpotObjective, coords: &[f64]) -> f64 {
    pot.wrapper().value_and_gradient(ArrayView1::from(coords)).0
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let dir = std::env::args().nth(1).expect("frames directory");
    let mut frames: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("read {dir}: {error}"))
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("xyz"))
        .collect();
    frames.sort();
    assert!(!frames.is_empty(), "no xyz frames in {dir}");
    let (numbers, _) = read_xyz(&frames[0]);
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let warm = RgpotObjective::xtb(&numbers, box_);
    let mut fresh = RgpotObjective::xtb(&numbers, box_);
    let mut max_abs = 0.0_f64;
    let mut compared = 0usize;
    println!("frame warm_eV fresh_eV abs_eV");
    for (index, path) in frames.iter().enumerate() {
        let (_, coords) = read_xyz(path);
        let warm_e = energy(&warm, &coords);
        fresh.restart();
        let fresh_e = energy(&fresh, &coords);
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("frame");
        if warm_e.is_finite() && fresh_e.is_finite() {
            let diff = (warm_e - fresh_e).abs();
            max_abs = max_abs.max(diff);
            compared += 1;
            println!("{index} {name} {warm_e:.10} {fresh_e:.10} {diff:.3e}");
        } else if warm_e.is_finite() == fresh_e.is_finite() {
            println!("{index} {name} nan nan unconverged");
        } else {
            println!(
                "{index} {name} {} {} disagree",
                if warm_e.is_finite() {
                    format!("{warm_e:.10}")
                } else {
                    "nan".to_string()
                },
                if fresh_e.is_finite() {
                    format!("{fresh_e:.10}")
                } else {
                    "nan".to_string()
                }
            );
            std::process::exit(2);
        }
    }
    println!("max_abs_eV {max_abs:.3e} compared {compared}");
}
