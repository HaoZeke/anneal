//! Evaluations per relaxation on a Lennard-Jones cluster, with and without
//! curvature carried between relaxations.
//!
//! The work ledger charges evaluations, so this ratio is the multiplier on how
//! many hops a budget buys.

use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};

/// Deterministic uniform stream, so the example pulls in no extra dependency
/// and both arms see the identical perturbation sequence.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
    }
}

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

/// Pushes apart any pair closer than `min_sep`.
///
/// A uniform perturbation on a cluster occasionally lands two points on top of
/// each other, and the r^-12 term then produces a gradient near 1e13 that no
/// line search recovers from. Without this the relaxations being timed are
/// failures rather than relaxations, and the cost comparison measures nothing.
fn repair(x: &mut Array1<f64>, min_sep: f64) {
    let n = x.len() / 3;
    for _ in 0..50 {
        let mut worst = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let d: [f64; 3] = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r < min_sep && r > 1e-12 {
                    let push = 0.5 * (min_sep - r) / r;
                    for k in 0..3 {
                        x[3 * i + k] += push * d[k];
                        x[3 * j + k] -= push * d[k];
                    }
                    worst = worst.max(min_sep - r);
                } else if r <= 1e-12 {
                    x[3 * i] += 0.5 * min_sep;
                    worst = min_sep;
                }
            }
        }
        if worst == 0.0 {
            return;
        }
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(75);
    let hops: usize = 400;
    let step = 0.38;
    let mut rng = Rng(20260802);

    // A relaxed starting structure, shared by both arms.
    let radius = 0.9 * (n as f64).cbrt();
    let mut seed: Array1<f64> = Array1::zeros(3 * n);
    for v in seed.iter_mut() {
        *v = rng.uniform(-radius, radius);
    }
    repair(&mut seed, 0.85);
    let mut warm0 = WarmLbfgs::default();
    let (_, base, _) = warm0.minimize(seed.view(), 3000, |v| Some(lj(v)));

    // Same perturbation sequence for both arms, so the only difference is
    // whether curvature survives between relaxations.
    let perturbations: Vec<Array1<f64>> = (0..hops)
        .map(|_| {
            let mut p = base.clone();
            for v in p.iter_mut() {
                *v += rng.uniform(-step, step);
            }
            repair(&mut p, 0.85);
            p
        })
        .collect();

    let mut warm = WarmLbfgs::default();
    warm.minimize(base.view(), 3000, |v| Some(lj(v)));
    let mut warm_evals = 0usize;
    let mut warm_best = f64::INFINITY;
    let mut warm_gmax = 0.0_f64;
    for p in &perturbations {
        let (f, x, c) = warm.minimize(p.view(), 3000, |v| Some(lj(v)));
        warm_evals += c;
        warm_best = warm_best.min(f);
        // Convergence has to be checked, not assumed: a minimiser that stops
        // early looks cheap per relaxation while returning points that are not
        // minima, which would make the cost comparison meaningless.
        let (_, g) = lj(x.view());
        warm_gmax = warm_gmax.max(g.iter().fold(0.0_f64, |a, v| a.max(v.abs())));
    }

    let mut cold_evals = 0usize;
    let mut cold_best = f64::INFINITY;
    let mut cold_gmax = 0.0_f64;
    for p in &perturbations {
        let mut cold = WarmLbfgs::default();
        let (f, x, c) = cold.minimize(p.view(), 3000, |v| Some(lj(v)));
        cold_evals += c;
        cold_best = cold_best.min(f);
        let (_, g) = lj(x.view());
        cold_gmax = cold_gmax.max(g.iter().fold(0.0_f64, |a, v| a.max(v.abs())));
    }

    println!("LJ{n}, {hops} relaxations from perturbed minima, step {step}");
    println!(
        "  cold  {:>8} evals  {:>7.1} per relaxation  best {:.6}  worst |g| {:.2e}",
        cold_evals,
        cold_evals as f64 / hops as f64,
        cold_best,
        cold_gmax
    );
    println!(
        "  warm  {:>8} evals  {:>7.1} per relaxation  best {:.6}  worst |g| {:.2e}",
        warm_evals,
        warm_evals as f64 / hops as f64,
        warm_best,
        warm_gmax
    );
    println!(
        "  ratio {:.2}x  (hops per unit budget)",
        cold_evals as f64 / warm_evals as f64
    );
}
