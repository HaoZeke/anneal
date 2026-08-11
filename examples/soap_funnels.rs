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
    SoapSpec, class_masses, ih_dominated, jacobian_z, local_nu3, local_spectra_z, mean_residual_rms,
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
        let start = if parts.first().is_some_and(|p| p.parse::<f64>().is_ok()) {
            0
        } else if parts.len() >= 4 && parts[1].parse::<f64>().is_ok() {
            1
        } else {
            continue;
        };
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

    // Literature structure fingerprints from the same local SOAP.
    // Caro (PRB 2019): q = p/||p||, kernel (q·q')^ζ.
    // De/Bartók/Csányi/Ceriotti (2016): average kernel = μ_q·μ_q' for
    // linear ζ=1; they note it is blind when two clouds share a mean.
    // REMatch is the regularized best-match (Sinkhorn) on the kernel
    // matrix. featomic's SOAP power spectrum is this same 3-body p;
    // λ-SOAP / CG products are the next body order.
    let qm = caro_q(&lm);
    let qi = caro_q(&li);
    let k_avg = average_kernel(&qi, &qm, 1.0);
    let k_avg4 = average_kernel(&qi, &qm, 4.0);
    let d_avg = (2.0 - 2.0 * k_avg).max(0.0).sqrt();
    let d_avg4 = (2.0 - 2.0 * k_avg4).max(0.0).sqrt();
    let k_rem = rematch(&qi, &qm, 4.0, 0.05, 80);
    let d_rem = (2.0 - 2.0 * k_rem).max(0.0).sqrt();
    let k_best = best_match(&qi, &qm, 4.0);
    let d_best = (2.0 - 2.0 * k_best).max(0.0).sqrt();
    println!(
        "Caro-q AVG ζ=1  K {k_avg:.6}  D {d_avg:.6}   AVG ζ=4  K {k_avg4:.6}  D {d_avg4:.6}"
    );
    println!("REMatch ζ=4 γ=0.05  K {k_rem:.6}  D {d_rem:.6}   best-match ζ=4  K {k_best:.6}  D {d_best:.6}");

    let fr_m = anneal_core::structure::atom_triplet_fracs(marks.view(), 75, 1.35);
    let fr_i = anneal_core::structure::atom_triplet_fracs(ico.view(), 75, 1.35);
    let mut k_555_421 = 0.0;
    let mut n_pair = 0usize;
    for i in 0..75 {
        if fr_i[i][0] < 0.5 {
            continue;
        }
        for j in 0..75 {
            if fr_m[j][1] < 0.3 {
                continue;
            }
            k_555_421 += kernel_zeta(&qi, i, &qm, j, 4.0);
            n_pair += 1;
        }
    }
    if n_pair > 0 {
        println!(
            "mean k^4(ico 555, Marks 421) {:.6}  over {n_pair} pairs  (n_max={} l_max={})",
            k_555_421 / n_pair as f64,
            spec.n_max,
            spec.l_max
        );
    }

    for (n_max, l_max) in [(3usize, 3usize), (4, 6), (6, 6)] {
        let hi = SoapSpec {
            n_max,
            l_max,
            rcut_nn: 3.5,
        };
        let qm = caro_q(&local_spectra_z(marks.view(), hi, None));
        let qi = caro_q(&local_spectra_z(ico.view(), hi, None));
        let mut s = 0.0;
        let mut np = 0usize;
        for i in 0..75 {
            if fr_i[i][0] < 0.5 {
                continue;
            }
            for j in 0..75 {
                if fr_m[j][1] < 0.3 {
                    continue;
                }
                s += kernel_zeta(&qi, i, &qm, j, 4.0);
                np += 1;
            }
        }
        let k_avg4 = average_kernel(&qi, &qm, 4.0);
        let d_avg4 = (2.0 - 2.0 * k_avg4).max(0.0).sqrt();
        println!(
            "n_max={n_max} l_max={l_max}  AVG ζ=4 D {d_avg4:.6}  k^4(fivefold,421) {:.6}  pairs {np}",
            if np > 0 { s / np as f64 } else { 0.0 }
        );
    }

    let nm = caro_q(&local_nu3(marks.view(), spec));
    let ni = caro_q(&local_nu3(ico.view(), spec));
    let k_avg4 = average_kernel(&ni, &nm, 4.0);
    let d_avg4 = (2.0 - 2.0 * k_avg4).max(0.0).sqrt();
    let k_rem = rematch(&ni, &nm, 4.0, 0.05, 80);
    let d_rem = (2.0 - 2.0 * k_rem).max(0.0).sqrt();
    let mut s = 0.0;
    let mut np = 0usize;
    for i in 0..75 {
        if fr_i[i][0] < 0.5 {
            continue;
        }
        for j in 0..75 {
            if fr_m[j][1] < 0.3 {
                continue;
            }
            s += kernel_zeta(&ni, i, &nm, j, 4.0);
            np += 1;
        }
    }
    println!(
        "nu3=PS+triple  AVG ζ=4 D {d_avg4:.6}  REMatch D {d_rem:.6}  k^4(fivefold,421) {:.6}  pairs {np}",
        if np > 0 { s / np as f64 } else { 0.0 }
    );

    // Raw 4-body channels, not Caro-renormalized against the power spectrum.
    let nm_raw = local_nu3(marks.view(), spec);
    let ni_raw = local_nu3(ico.view(), spec);
    let d0 = local_spectra_z(marks.view(), spec, None).ncols();
    let mut ico5 = vec![0.0; spec.nu3_dim()];
    let mut n5 = 0.0;
    let mut mk4 = vec![0.0; spec.nu3_dim()];
    let mut n4 = 0.0;
    for i in 0..75 {
        if fr_i[i][0] > 0.5 {
            n5 += 1.0;
            for t in 0..spec.nu3_dim() {
                ico5[t] += ni_raw[[i, d0 + t]];
            }
        }
        if fr_m[i][1] > 0.3 {
            n4 += 1.0;
            for t in 0..spec.nu3_dim() {
                mk4[t] += nm_raw[[i, d0 + t]];
            }
        }
    }
    if n5 > 0.0 && n4 > 0.0 {
        print!("raw triple ico-fivefold");
        for t in 0..spec.nu3_dim() {
            print!(" {:.6}", ico5[t] / n5);
        }
        print!("   Marks close-packed");
        for t in 0..spec.nu3_dim() {
            print!(" {:.6}", mk4[t] / n4);
        }
        println!("   n {n5:.0}/{n4:.0}");
    }
}

fn caro_q(loc: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let mut q = loc.clone();
    for i in 0..q.nrows() {
        let mut n2 = 0.0;
        for t in 0..q.ncols() {
            n2 += q[[i, t]] * q[[i, t]];
        }
        let n = n2.sqrt().max(1e-15);
        for t in 0..q.ncols() {
            q[[i, t]] /= n;
        }
    }
    q
}

fn kernel_zeta(a: &ndarray::Array2<f64>, i: usize, b: &ndarray::Array2<f64>, j: usize, zeta: f64) -> f64 {
    let mut s = 0.0;
    for t in 0..a.ncols() {
        s += a[[i, t]] * b[[j, t]];
    }
    s.max(0.0).powf(zeta)
}

fn average_kernel(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>, zeta: f64) -> f64 {
    let na = a.nrows();
    let nb = b.nrows();
    let mut s = 0.0;
    for i in 0..na {
        for j in 0..nb {
            s += kernel_zeta(a, i, b, j, zeta);
        }
    }
    s / (na.max(1) * nb.max(1)) as f64
}

fn best_match(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>, zeta: f64) -> f64 {
    let n = a.nrows().min(b.nrows());
    let mut used = vec![false; b.nrows()];
    let mut s = 0.0;
    for i in 0..n {
        let mut best = -1.0;
        let mut bj = 0usize;
        for j in 0..b.nrows() {
            if used[j] {
                continue;
            }
            let k = kernel_zeta(a, i, b, j, zeta);
            if k > best {
                best = k;
                bj = j;
            }
        }
        used[bj] = true;
        s += best;
    }
    s / n.max(1) as f64
}

/// De/Ceriotti REMatch: Sinkhorn on C_ij = 1 − k^ζ, entropy γ.
fn rematch(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>, zeta: f64, gamma: f64, iters: usize) -> f64 {
    let n = a.nrows();
    let m = b.nrows();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut k = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            k[i][j] = kernel_zeta(a, i, b, j, zeta);
        }
    }
    let mut u = vec![1.0 / n as f64; n];
    let mut v = vec![1.0 / m as f64; m];
    let g = gamma.max(1e-6);
    for _ in 0..iters {
        for i in 0..n {
            let mut z = 0.0;
            for j in 0..m {
                z += v[j] * (k[i][j] / g).exp();
            }
            u[i] = 1.0 / (n as f64 * z.max(1e-300));
        }
        for j in 0..m {
            let mut z = 0.0;
            for i in 0..n {
                z += u[i] * (k[i][j] / g).exp();
            }
            v[j] = 1.0 / (m as f64 * z.max(1e-300));
        }
    }
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..m {
            s += u[i] * v[j] * (k[i][j] / g).exp() * k[i][j];
        }
    }
    s
}
