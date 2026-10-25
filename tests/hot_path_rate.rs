//! Release-mode rates for the triage. Not a search result.
use std::time::Instant;

use anneal_core::catalog::{packing_distance, packing_fingerprint};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::Array1;

fn lattice(n: usize, spacing: f64) -> Array1<f64> {
    let mut x = Array1::zeros(3 * n);
    let mut placed = 0usize;
    let side = (n as f64).cbrt().ceil() as usize + 1;
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                if placed >= n {
                    return x;
                }
                x[3 * placed] = i as f64 * spacing;
                x[3 * placed + 1] = j as f64 * spacing;
                x[3 * placed + 2] = k as f64 * spacing;
                placed += 1;
            }
        }
    }
    x
}

fn quench(steps: usize, start: &Array1<f64>, pot: &PairPotential) -> (f64, Array1<f64>, usize) {
    let mut opt = WarmLbfgs::default();
    let mut calls = 0usize;
    let (energy, point, _) = opt.minimize(start.view(), steps, |x| {
        calls += 1;
        Some(pot.value_and_gradient(x))
    });
    (energy, point, calls)
}

#[test]
fn release_rates_for_quench_screen_fingerprint_and_bank() {
    let pot = PairPotential::lennard_jones(38);
    let start = lattice(38, 1.2);
    let mut quench_calls = 0usize;
    let mut screen_calls = 0usize;
    let quench_start = Instant::now();
    let mut quenched = start.clone();
    for _ in 0..3 {
        let (energy, point, calls) = quench(200, &start, &pot);
        assert!(energy.is_finite());
        quench_calls += calls;
        quenched = point;
    }
    let quench_secs = quench_start.elapsed().as_secs_f64();

    let screen_start = Instant::now();
    for _ in 0..3 {
        let (energy, _, calls) = quench(25, &start, &pot);
        assert!(energy.is_finite());
        screen_calls += calls;
    }
    let screen_secs = screen_start.elapsed().as_secs_f64();

    let finger_start = Instant::now();
    let mut prints = Vec::new();
    for shift in 0..16 {
        let mut moved = quenched.clone();
        moved[0] += 0.01 * shift as f64;
        let print = packing_fingerprint(moved.as_slice().unwrap()).expect("fingerprint");
        prints.push(print);
    }
    let finger_secs = finger_start.elapsed().as_secs_f64();

    let bank_start = Instant::now();
    let mut lookups = 0usize;
    for query in &prints {
        let mut best = f64::INFINITY;
        for held in &prints {
            best = best.min(packing_distance(query, held));
            lookups += 1;
        }
        assert!(best.is_finite());
    }
    let bank_secs = bank_start.elapsed().as_secs_f64();

    eprintln!(
        "{{\"quench_calls\":{quench_calls},\"quench_s\":{quench_secs:.6},\"quench_per_s\":{:.3},\"screen_calls\":{screen_calls},\"screen_s\":{screen_secs:.6},\"screen_per_s\":{:.3},\"fingerprints\":{},\"finger_s\":{finger_secs:.6},\"finger_per_s\":{:.3},\"bank_lookups\":{lookups},\"bank_s\":{bank_secs:.6},\"bank_per_s\":{:.3}}}",
        quench_calls as f64 / quench_secs,
        screen_calls as f64 / screen_secs,
        prints.len(),
        prints.len() as f64 / finger_secs,
        lookups as f64 / bank_secs,
    );
}
