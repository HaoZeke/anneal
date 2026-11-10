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
    let mut state = x;
    let mut energy = 0.0;
    let mut calls = 0usize;
    let mut inf = f64::INFINITY;
    for _ in 0..8 {
        let (next_energy, next_state, next_calls) = opt.minimize(state.view(), 2000, |v| {
            Some(potential.value_and_gradient(v))
        });
        calls += next_calls;
        state = next_state;
        energy = next_energy;
        let (_, gradient) = potential.value_and_gradient(state.view());
        calls += 1;
        inf = 0.0;
        for value in gradient.iter() {
            inf = inf.max(value.abs());
        }
        if inf <= 1e-8 {
            break;
        }
        opt.forget();
    }
    for _ in 0..40 {
        if inf <= 1e-8 {
            break;
        }
        let (_, gradient) = potential.value_and_gradient(state.view());
        calls += 1;
        let mut direction = gradient.clone();
        for value in direction.iter_mut() {
            *value = -*value;
        }
        let slope: f64 = gradient
            .iter()
            .zip(direction.iter())
            .map(|(g, d)| g * d)
            .sum();
        let mut step = 1.0e-2;
        let mut polished = false;
        for _ in 0..12 {
            let mut trial = state.clone();
            for (coord, slope_component) in trial.iter_mut().zip(direction.iter()) {
                *coord += step * slope_component;
            }
            let (trial_energy, trial_gradient) = potential.value_and_gradient(trial.view());
            calls += 1;
            if trial_energy <= energy + 1e-4 * step * slope {
                state = trial;
                energy = trial_energy;
                inf = trial_gradient.iter().fold(0.0, |held, v| held.max(v.abs()));
                polished = true;
                break;
            }
            step *= 0.3;
        }
        if !polished {
            break;
        }
    }
    println!("n: {n}");
    println!("calls: {calls}");
    println!("energy: {energy:.12}");
    println!("gradient_inf: {inf:.3e}");
    println!("reached_1e-8: {}", inf <= 1e-8);
}
