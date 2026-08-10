//! Project LJ75 Marks and the Mackay ico into the SOAP cloud.
//!
//! Usage: soap_funnels <marks.xyz> <ico.xyz>
//! Each file is 75 lines of `x y z` (Cambridge points format).
//!
//! Prints whether each structure is a stationary point of the mean
//! residual R = ½ Σ ||p_i − μ||², the distance between the two clouds,
//! and the size of J⁺(μ_Marks − μ_ico) on the ico (the leftover that
//! would have to exist for a Jacobian hop to leave Ih toward Marks).

use anneal_core::soap::{
    SoapSpec, class_masses, ih_dominated, jacobian_z, local_spectra_z, mean_residual_rms,
    power_spectrum,
};
use ndarray::{Array1, ArrayView1};
use std::env;
use std::fs;

fn lj(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut e = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d0 = x[3 * i] - x[3 * j];
            let d1 = x[3 * i + 1] - x[3 * j + 1];
            let d2 = x[3 * i + 2] - x[3 * j + 2];
            let r2 = d0 * d0 + d1 * d1 + d2 * d2;
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            e += 4.0 * (inv6 * inv6 - inv6);
        }
    }
    e
}

fn load_points(path: &str, n: usize) -> Array1<f64> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut vals = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = t.split_whitespace().collect();
        let start = if parts.len() >= 4 { 1 } else { 0 };
        if parts.len() < start + 3 {
            continue;
        }
        for k in 0..3 {
            vals.push(parts[start + k].parse::<f64>().expect(path));
        }
    }
    assert_eq!(
        vals.len(),
        3 * n,
        "{path} has {} coords, want {}",
        vals.len(),
        3 * n
    );
    Array1::from(vals)
}

fn mu_of(loc: &ndarray::Array2<f64>) -> Vec<f64> {
    let n = loc.nrows();
    let d = loc.ncols();
    let mut mu = vec![0.0; d];
    if n == 0 {
        return mu;
    }
    for i in 0..n {
        for t in 0..d {
            mu[t] += loc[[i, t]] / n as f64;
        }
    }
    mu
}

fn nn_cloud(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>) -> f64 {
    let mut s = 0.0;
    for i in 0..a.nrows() {
        let mut best = f64::INFINITY;
        for j in 0..b.nrows() {
            let mut d2 = 0.0;
            for t in 0..a.ncols() {
                let d = a[[i, t]] - b[[j, t]];
                d2 += d * d;
            }
            best = best.min(d2);
        }
        s += best.sqrt();
    }
    s / a.nrows().max(1) as f64
}

fn residual_grad_norm(x: ArrayView1<f64>, spec: SoapSpec) -> f64 {
    let loc = local_spectra_z(x, spec, None);
    let n = loc.nrows();
    let dim = loc.ncols();
    let mu = mu_of(&loc);
    let mut dp = Array1::zeros(n * dim);
    for i in 0..n {
        for t in 0..dim {
            dp[i * dim + t] = loc[[i, t]] - mu[t];
        }
    }
    let j = jacobian_z(x, spec, None);
    // g = J^T (p − μ). Vanishes at a SOAP-stationary geometry.
    let mut g2 = 0.0;
    for k in 0..x.len() {
        let mut s = 0.0;
        for r in 0..n * dim {
            s += j[[r, k]] * dp[r];
        }
        g2 += s * s;
    }
    g2.sqrt()
}

fn report(label: &str, x: ArrayView1<f64>, spec: SoapSpec) {
    let e = lj(x);
    let loc = local_spectra_z(x, spec, None);
    let mu = mu_of(&loc);
    let mut cloud = 0.0;
    for i in 0..loc.nrows() {
        for t in 0..loc.ncols() {
            let d = loc[[i, t]] - mu[t];
            cloud += d * d;
        }
    }
    let cloud = (cloud / loc.nrows().max(1) as f64).sqrt();
    let mean = mean_residual_rms(x, spec);
    let g = residual_grad_norm(x, spec);
    let p = power_spectrum(x, spec);
    let masses = class_masses(x, spec);
    println!(
        "{label}  E {e:.6}  class555_rms {mean:.6}  allatom_rms {cloud:.6}  \
         ||J^T(p-μ)|| {g:.6}  ||μ|| {:.6}  ih {}  mass555/421/422 {:.2}/{:.2}/{:.2}",
        p.iter().map(|v| v * v).sum::<f64>().sqrt(),
        ih_dominated(x, spec),
        masses[0],
        masses[1],
        masses[2]
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let marks_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("points/75");
    let ico_path = args.get(2).map(String::as_str).unwrap_or("points/75i");
    let spec = SoapSpec::default();
    let marks = load_points(marks_path, 75);
    let ico = load_points(ico_path, 75);
    report("Marks", marks.view(), spec);
    report("Ih   ", ico.view(), spec);

    let lm = local_spectra_z(marks.view(), spec, None);
    let li = local_spectra_z(ico.view(), spec, None);
    let mm = mu_of(&lm);
    let mi = mu_of(&li);
    let mut dmu = 0.0;
    for t in 0..mm.len() {
        let d = mm[t] - mi[t];
        dmu += d * d;
    }
    let dmu = dmu.sqrt();
    let ico_to_marks = nn_cloud(&li, &lm);
    let marks_to_ico = nn_cloud(&lm, &li);
    println!(
        "||μ_Marks − μ_Ih|| {dmu:.6}  cloud nn Ih→Marks {ico_to_marks:.6}  Marks→Ih {marks_to_ico:.6}"
    );

    // Leftover on Ih toward the Marks mean: stacked target μ_Marks.
    let dim = spec.dim();
    let n = 75;
    let mut target = Array1::zeros(n * dim);
    for i in 0..n {
        for t in 0..dim {
            target[i * dim + t] = mm[t];
        }
    }
    let loc = local_spectra_z(ico.view(), spec, None);
    let mut dp2 = 0.0;
    for i in 0..n {
        for t in 0..dim {
            let d = target[i * dim + t] - loc[[i, t]];
            dp2 += d * d;
        }
    }
    println!(
        "Ih leftover toward μ_Marks  ||Δp|| {:.6}  (zero ⇒ J^+ has nothing to pull)",
        dp2.sqrt()
    );
}
