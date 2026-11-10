//! Re-quench one Lennard-Jones geometry and print the tightened energy.
//!
//! Usage: `lj_requench <xyz>`
//!
//! The minimiser stops at a gradient infinity norm of 1e-8. The geometry
//! is whatever the file holds. Nothing here knows a published structure.

use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::Array1;

fn load(path: &str) -> Array1<f64> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let mut coords = Vec::new();
    for line in text.lines() {
        let mut values = Vec::new();
        for token in line.split_whitespace() {
            if let Ok(value) = token.parse::<f64>() {
                values.push(value);
            }
        }
        if values.len() >= 3 {
            coords.extend_from_slice(&values[values.len() - 3..]);
        }
    }
    Array1::from(coords)
}

fn main() {
    let path = std::env::args().nth(1).expect("xyz path");
    let x = load(&path);
    let n = x.len() / 3;
    assert!(
        n >= 2 && x.len() == 3 * n,
        "need 3N coordinates, got {}",
        x.len()
    );
    let potential = PairPotential::lennard_jones(n);
    let mut opt = WarmLbfgs::default();
    opt.gtol = 1e-8;
    let (energy, state, calls) =
        opt.minimize(x.view(), 4000, |v| Some(potential.value_and_gradient(v)));
    let (_, gradient) = potential.value_and_gradient(state.view());
    let mut inf = 0.0_f64;
    for value in gradient.iter() {
        inf = inf.max(value.abs());
    }
    println!("n: {n}");
    println!("calls: {calls}");
    println!("energy: {energy:.12}");
    println!("gradient_inf: {inf:.3e}");
}
