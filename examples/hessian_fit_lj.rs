//! How well the model Hessian predicts real quench depths, calibrated and not.
//!
//! The depth `1/2 g^T H^-1 g` is only useful to a first-stage acceptance if the
//! number tracks what the relaxation actually recovers. This measures that on a
//! Lennard-Jones cluster: perturb a minimum, record the gradient, quench with
//! the crate's warm L-BFGS, and compare the predicted depth against the energy
//! the descent gave back.
//!
//! Two arms, on the same held-out descents. The defaults are the covalent
//! constants; the fit is [`anneal_core::hessian_fit`] run on the earlier half of
//! the same descents. Both arms are given their own force-constant scale,
//! matched on the training half, because `k0` multiplies every prediction
//! equally and any consumer of the depth absorbs it: scoring the defaults at
//! `k0 = 1` would measure a scale mismatch and report it as a shape improvement.
//!
//! ```text
//! cargo run --release --example hessian_fit_lj -- 38 0.25
//! ```
//!
//! Setting `SCAN=1` prints the objective over the whole search box instead of
//! only its minimum, which is where the shape of the box in
//! [`anneal_core::hessian_fit`] comes from: `AHI`, `FLO` and `FHI` move the
//! scanned range so a ceiling can be checked for being a ceiling.

use anneal_core::hessian_fit::{Descent, HessianFit, calibrate, median_relative_error, rescaled};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::model_hessian::ModelParams;
use ndarray::{Array1, ArrayView1};

/// Deterministic uniform stream, so the example pulls in no extra dependency
/// and a reported number is reproducible.
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
/// A uniform perturbation occasionally lands two points on top of each other,
/// and the r^-12 term then produces a gradient near 1e13 that no line search
/// recovers from. Such a descent is a failure rather than a relaxation, and its
/// depth would calibrate the operator against a repair rather than a fall.
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

/// Largest absolute gradient component, the convergence test the descents face.
fn gmax(g: ArrayView1<f64>) -> f64 {
    g.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

/// Quantile of a sample, by sorting a copy.
fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

/// Relative errors of one parameter set over a set of descents.
fn relative_errors(samples: &[Descent], params: ModelParams, iters: usize) -> Vec<f64> {
    samples
        .iter()
        .map(|s| {
            let p = anneal_core::model_hessian::depth_with(
                s.x.view(),
                s.n(),
                s.gradient.view(),
                iters,
                params,
            );
            (p / s.observed - 1.0).abs()
        })
        .collect()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(38);
    let step: f64 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.25);
    let hops: usize = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(240);
    // The truncation the consumer would use. The fit runs at its own, shallower
    // one; the comparison here is at the consumer's, so a shape that only looks
    // good at the fit's truncation would show up as a loss.
    let iters = 40usize;
    let mut rng = Rng(20260805);

    // A relaxed structure to perturb from, so the descents start where a search
    // starts them: a step away from a minimum, not from random coordinates.
    let radius = 0.9 * (n as f64).cbrt();
    let mut seed: Array1<f64> = Array1::zeros(3 * n);
    for v in seed.iter_mut() {
        *v = rng.uniform(-radius, radius);
    }
    repair(&mut seed, 0.85);
    let mut warm = WarmLbfgs::default();
    let (base_e, base, _) = warm.minimize(seed.view(), 4000, |v| Some(lj(v)));

    let mut samples: Vec<Descent> = Vec::with_capacity(hops);
    let mut refused = 0usize;
    let mut evals = 0usize;
    for _ in 0..hops {
        let mut p = base.clone();
        for v in p.iter_mut() {
            *v += rng.uniform(-step, step);
        }
        repair(&mut p, 0.85);
        let (e_start, g_start) = lj(p.view());
        // Cold each time, so the depth observed is the descent from this point
        // and not a descent helped by curvature carried from the last one.
        let mut relax = WarmLbfgs::default();
        let (e_end, x_end, c) = relax.minimize(p.view(), 4000, |v| Some(lj(v)));
        evals += c + 1;
        let (_, g_end) = lj(x_end.view());
        // An unconverged descent has not finished recovering, so its depth is a
        // lower bound rather than an observation.
        if gmax(g_end.view()) > 1e-4 || !(e_start - e_end > 0.0) {
            refused += 1;
            continue;
        }
        samples.push(Descent {
            x: p,
            gradient: g_start,
            observed: e_start - e_end,
        });
    }

    // Split in run order rather than at random: a calibration that only works
    // when it has seen the future is not a calibration.
    let cut = samples.len() / 2;
    let (train, test) = samples.split_at(cut);

    let t0 = std::time::Instant::now();
    let fit = calibrate(train, anneal_core::hessian_fit::FIT_ITERS);
    let refit_time = t0.elapsed();

    let defaults = ModelParams::default();
    let default_scaled = rescaled(train, defaults, iters);

    println!("LJ{n}, perturbation {step}, {hops} descents from a minimum at {base_e:.6}");
    println!(
        "  {} usable, {} refused unconverged, {} evaluations charged to generating them",
        samples.len(),
        refused,
        evals
    );
    println!("  {} train / {} held out", train.len(), test.len());
    println!(
        "  defaults        alpha {:.3}  floor {:.5}  k0 {:.4} (scale matched on train)",
        default_scaled.alpha, default_scaled.floor, default_scaled.k0
    );
    match fit {
        None => {
            println!("  no fit: nothing in the search box beat the defaults on the training half")
        }
        Some(c) => {
            println!(
                "  fitted          alpha {:.3}  floor {:.5}  k0 {:.4}  train log spread {:.4}",
                c.params.alpha, c.params.floor, c.params.k0, c.spread
            );
            println!(
                "  refit cost      {refit_time:?} on {} samples",
                train.len()
            );

            let before = relative_errors(test, default_scaled, iters);
            let after = relative_errors(test, c.params, iters);
            let med_before = median_relative_error(test, default_scaled, iters);
            let med_after = median_relative_error(test, c.params, iters);
            println!("  held-out median relative depth error");
            println!("    defaults      {med_before:.4}");
            println!("    calibrated    {med_after:.4}");
            println!(
                "    ratio         {:.2}x",
                if med_after > 0.0 {
                    med_before / med_after
                } else {
                    f64::INFINITY
                }
            );
            println!("  held-out relative error at the 25th, 50th, 75th, 95th percentile");
            println!(
                "    defaults      {:.3} {:.3} {:.3} {:.3}",
                quantile(&before, 0.25),
                quantile(&before, 0.5),
                quantile(&before, 0.75),
                quantile(&before, 0.95)
            );
            println!(
                "    calibrated    {:.3} {:.3} {:.3} {:.3}",
                quantile(&after, 0.25),
                quantile(&after, 0.5),
                quantile(&after, 0.75),
                quantile(&after, 0.95)
            );
            let raw = median_relative_error(test, defaults, iters);
            println!("  for reference, the defaults with k0 = 1 and no scale matching: {raw:.4}");
        }
    }

    if std::env::var("SCAN").is_ok() {
        let ahi: f64 = std::env::var("AHI")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24.0);
        let flo: f64 = std::env::var("FLO")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-4);
        let fhi: f64 = std::env::var("FHI")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300.0);
        for i in 0..13 {
            let alpha = (0.2f64.ln() + (ahi.ln() - 0.2f64.ln()) * i as f64 / 12.0).exp();
            let mut row = String::new();
            for j in 0..13 {
                let floor = (flo.ln() + (fhi.ln() - flo.ln()) * j as f64 / 12.0).exp();
                let s = anneal_core::hessian_fit::spread_of(train, alpha, floor, iters)
                    .map_or(f64::NAN, |c| c.spread);
                row.push_str(&format!(" {s:.4}"));
            }
            println!("SCAN alpha={alpha:7.3}{row}");
        }
        let mut hdr = String::new();
        for j in 0..13 {
            let floor = (flo.ln() + (fhi.ln() - flo.ln()) * j as f64 / 12.0).exp();
            hdr.push_str(&format!(" {floor:.2e}"));
        }
        println!("SCAN floors{hdr}");
    }

    // And the same data through the accumulator, which is how a run would use
    // it: sample by sample, on a schedule, with no split to hand and only the
    // last `capacity` descents in view. Fed the training half only, so the
    // held-out score means the same thing as the batch fit's.
    let mut acc = HessianFit::new();
    let mut fits = 0usize;
    for s in train.iter() {
        if acc.observe(s.x.view(), s.gradient.view(), s.observed) {
            fits += 1;
        }
    }
    println!(
        "  online: {fits} refits over {} descents, ending at alpha {:.3} floor {:.5} k0 {:.4}",
        train.len(),
        acc.params().alpha,
        acc.params().floor,
        acc.params().k0
    );
    println!(
        "  online held-out median relative depth error {:.4}, against {:.4} for the defaults",
        median_relative_error(test, acc.params(), iters),
        median_relative_error(test, default_scaled, iters)
    );
}
