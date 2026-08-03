//! Evaluations per relaxation on a Lennard-Jones cluster, with and without
//! curvature carried between relaxations.
//!
//! The work ledger charges evaluations, so this ratio is the multiplier on how
//! many hops a budget buys.

use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// Lennard-Jones value and gradient in reduced units, no cutoff.
fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d: [f64; 3] = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            let inv12 = inv6 * inv6;
            e += 4.0 * (inv12 - inv6);
            let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= coef * d[k];
                g[3 * j + k] += coef * d[k];
            }
        }
    }
    (e, g)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(75);
    let hops: usize = 400;
    let step = 0.38;
    let mut rng = Pcg64::seed_from_u64(20260802);

    // A relaxed starting structure, shared by both arms.
    let radius = 0.9 * (n as f64).cbrt();
    let mut seed: Array1<f64> = Array1::zeros(3 * n);
    for v in seed.iter_mut() {
        *v = rng.random_range(-radius..radius);
    }
    let mut warm0 = WarmLbfgs::default();
    let (_, base, _) = warm0.minimize(seed.view(), 3000, |v| Some(lj(v)));

    // Same perturbation sequence for both arms, so the only difference is
    // whether curvature survives between relaxations.
    let perturbations: Vec<Array1<f64>> = (0..hops)
        .map(|_| {
            let mut p = base.clone();
            for v in p.iter_mut() {
                *v += rng.random_range(-step..step);
            }
            p
        })
        .collect();

    let mut warm = WarmLbfgs::default();
    warm.minimize(base.view(), 3000, |v| Some(lj(v)));
    let mut warm_evals = 0usize;
    let mut warm_best = f64::INFINITY;
    for p in &perturbations {
        let (f, _, c) = warm.minimize(p.view(), 3000, |v| Some(lj(v)));
        warm_evals += c;
        warm_best = warm_best.min(f);
    }

    let mut cold_evals = 0usize;
    let mut cold_best = f64::INFINITY;
    for p in &perturbations {
        let mut cold = WarmLbfgs::default();
        let (f, _, c) = cold.minimize(p.view(), 3000, |v| Some(lj(v)));
        cold_evals += c;
        cold_best = cold_best.min(f);
    }

    println!("LJ{n}, {hops} relaxations from perturbed minima, step {step}");
    println!(
        "  cold  {:>8} evals  {:>7.1} per relaxation  best {:.6}",
        cold_evals,
        cold_evals as f64 / hops as f64,
        cold_best
    );
    println!(
        "  warm  {:>8} evals  {:>7.1} per relaxation  best {:.6}",
        warm_evals,
        warm_evals as f64 / hops as f64,
        warm_best
    );
    println!(
        "  ratio {:.2}x  (hops per unit budget)",
        cold_evals as f64 / warm_evals as f64
    );
}
