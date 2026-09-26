//! One MD segment through an external engine, as a liveness check.
//!
//!     md_segment_check lammps /path/to/lmp cluster.xyz [steps] [temp]
//!
//! Reads an xyz file, propagates one segment, and prints the per-atom
//! RMS displacement. A finite, nonzero displacement with every
//! coordinate finite is the pass condition; anything else exits
//! nonzero. This checks the engine wiring, not any search outcome.

use anneal_core::md_engine::engine_by_name;
use ndarray::Array1;

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, engine_name, binary, xyz, rest @ ..] = arguments.as_slice() else {
        eprintln!("usage: md_segment_check ENGINE BINARY XYZ [STEPS] [TEMP]");
        std::process::exit(2);
    };
    let steps = rest
        .first()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200);
    let temperature = rest
        .get(1)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.0);
    let text = std::fs::read_to_string(xyz).expect("xyz file must be readable");
    let values: Vec<f64> = text
        .lines()
        .skip(2)
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .filter_map(|token| token.parse::<f64>().ok())
        })
        .collect();
    assert!(!values.is_empty(), "xyz body held no coordinates");
    let x = Array1::from(values);
    let workdir = std::env::temp_dir().join(format!("md-segment-check-{engine_name}"));
    let engine = engine_by_name(engine_name, std::path::Path::new(binary), &workdir)
        .expect("engine must be lammps or gromacs");
    let y = match engine.propagate(x.view(), steps, temperature, 42) {
        Ok(y) => y,
        Err(error) => {
            eprintln!("segment failed: {error}");
            std::process::exit(1);
        }
    };
    assert_eq!(y.len(), x.len(), "engine changed the atom count");
    let displacement = (x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / (x.len() as f64 / 3.0))
        .sqrt();
    assert!(y.iter().all(|value| value.is_finite()), "non-finite output");
    assert!(
        displacement.is_finite() && displacement > 0.0,
        "no displacement"
    );
    println!(
        "{engine_name}: {steps} steps at T*={temperature}, per-atom rms displacement {displacement:.4}"
    );
}
