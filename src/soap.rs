//! SOAP power spectrum and the Cartesian pullback through its Jacobian.
//!
//! Local fingerprints are per-atom power spectra
//! `p_{nn'l}(i) = Σ_m c_{nlm}(i) c_{n'lm}(i)` with
//! `c_{nlm}(i) = Σ_{j≠i} w_n(r_{ij}) Y_{lm}(hat r_{ij})`.
//! The map is `R^{3N} → R^{N n_feat}`. Its Jacobian is analytic: each pair
//! contributes `∂w/∂r` and `∂Y/∂r̂` through the projector
//! `(I − r̂ r̂ᵀ)/r`. Finite differences are not that map — they cost `O(N)`
//! SOAP evals, they jump at the cutoff, and a 24-D *global* average has
//! rank at most 24 in `R^{3N}`, so it cannot see which atoms carry
//! icosahedral versus fivefold-join environments.
//!
//! A residual step takes `Δp_i = p_i − μ` (local environments away from
//! the mean) and `ΔR = argmin ||J ΔR − Δp||² + λ||ΔR||²`, then strips
//! the rigid kernel of `J`. Not a one-atom hop and not a Marks/CSA oracle.

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use std::f64::consts::PI;

/// Radial and angular resolution, and the cutoff in nearest-neighbour units.
#[derive(Debug, Clone, Copy)]
pub struct SoapSpec {
    /// Radial functions `n = 0..n_max`.
    pub n_max: usize,
    /// Angular momentum `l = 0..l_max`.
    pub l_max: usize,
    /// Cutoff in coordinate units. Fixed: a moving median-NN cutoff is
    /// not a map `R^{3N} → p` and has no Jacobian.
    pub rcut_nn: f64,
}

impl Default for SoapSpec {
    fn default() -> Self {
        Self {
            n_max: 3,
            l_max: 3,
            rcut_nn: 3.5,
        }
    }
}

impl SoapSpec {
    /// Length of the packed power spectrum `n ≤ n', l ≤ l_max`.
    pub fn dim(self) -> usize {
        let n = self.n_max;
        n * (n + 1) / 2 * (self.l_max + 1)
    }
}

/// Packed average SOAP of `x` (flattened 3N).
pub fn power_spectrum(x: ArrayView1<f64>, spec: SoapSpec) -> Array1<f64> {
    let loc = local_spectra(x, spec);
    let n = loc.nrows();
    let mut acc = Array1::<f64>::zeros(spec.dim());
    if n == 0 {
        return acc;
    }
    for i in 0..n {
        for t in 0..spec.dim() {
            acc[t] += loc[[i, t]];
        }
    }
    acc / n as f64
}

/// Per-atom power spectra, shape `(N, dim)`.
pub fn local_spectra(x: ArrayView1<f64>, spec: SoapSpec) -> Array2<f64> {
    let n_at = x.len() / 3;
    let dim = spec.dim();
    let mut out = Array2::<f64>::zeros((n_at, dim));
    if n_at < 2 {
        return out;
    }
    let rcut = spec.rcut_nn;
    if !(rcut > 0.0) {
        return out;
    }
    for i in 0..n_at {
        let (p, _) = atom_expand(x, i, n_at, rcut, spec);
        for t in 0..dim {
            out[[i, t]] = p[t];
        }
    }
    out
}

/// Analytic Jacobian of the *stacked* local spectra, shape `(N dim, 3N)`.
pub fn jacobian(x: ArrayView1<f64>, spec: SoapSpec) -> Array2<f64> {
    let n_at = x.len() / 3;
    let dim = spec.dim();
    let mut j = Array2::<f64>::zeros((n_at * dim, n_at * 3));
    if n_at < 2 {
        return j;
    }
    let rcut = spec.rcut_nn;
    if !(rcut > 0.0) {
        return j;
    }
    let n_lm = (spec.l_max + 1) * (spec.l_max + 1);
    let mut c = vec![vec![0.0; spec.n_max * n_lm]; n_at];
    for i in 0..n_at {
        let (_, ci) = atom_expand(x, i, n_at, rcut, spec);
        c[i] = ci;
    }
    for i in 0..n_at {
        let xi = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
        for jj in 0..n_at {
            if jj == i {
                continue;
            }
            let d = [
                x[3 * jj] - xi[0],
                x[3 * jj + 1] - xi[1],
                x[3 * jj + 2] - xi[2],
            ];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r >= rcut || r < 1e-12 {
                continue;
            }
            let u = [d[0] / r, d[1] / r, d[2] / r];
            let (ylm, dylm) = tesseral(u, spec.l_max);
            let fc = fcut(r, rcut);
            let dfc = dfcut(r, rcut);
            for n in 0..spec.n_max {
                let g = radial(n, r, rcut);
                let dg = dradial(n, r, rcut);
                let w = g * fc;
                let dw = dg * fc + g * dfc;
                for lm in 0..n_lm {
                    let y = ylm[lm];
                    for a in 0..3 {
                        let dy = {
                            let mut s = 0.0;
                            for b in 0..3 {
                                let proj = if a == b { 1.0 } else { 0.0 } - u[b] * u[a];
                                s += dylm[lm][b] * proj;
                            }
                            s / r
                        };
                        let dc_j = dw * u[a] * y + w * dy;
                        accumulate_dp(&mut j, i, dim, spec, n_lm, &c[i], n, lm, dc_j, 3 * jj + a);
                        accumulate_dp(&mut j, i, dim, spec, n_lm, &c[i], n, lm, -dc_j, 3 * i + a);
                    }
                }
            }
        }
    }
    j
}

fn accumulate_dp(
    j: &mut Array2<f64>,
    i: usize,
    dim: usize,
    spec: SoapSpec,
    n_lm: usize,
    c: &[f64],
    n: usize,
    lm: usize,
    dc: f64,
    col: usize,
) {
    // p_{n n' l} = Σ_m c_{n lm} c_{n' lm}. lm here is the packed (l,m) index.
    let l = lm_to_l(lm);
    let mut t = 0usize;
    for na in 0..spec.n_max {
        for np in na..spec.n_max {
            for ll in 0..=spec.l_max {
                if ll == l {
                    let c_n = c[na * n_lm + lm];
                    let c_np = c[np * n_lm + lm];
                    let d = if na == n && np == n {
                        2.0 * c_n * dc
                    } else if na == n {
                        dc * c_np
                    } else if np == n {
                        c_n * dc
                    } else {
                        0.0
                    };
                    j[[i * dim + t, col]] += d;
                }
                t += 1;
            }
        }
    }
}

fn lm_to_l(lm: usize) -> usize {
    // lm = l^2 + (m+l), so l = floor(sqrt(lm))
    let mut l = 0usize;
    while (l + 1) * (l + 1) <= lm {
        l += 1;
    }
    l
}

/// Cartesian displacement that realises a SOAP residual through analytic `J`.
pub fn pullback(x: ArrayView1<f64>, target: ArrayView1<f64>, spec: SoapSpec) -> Array1<f64> {
    let loc = local_spectra(x, spec);
    let n_at = loc.nrows();
    let dim = spec.dim();
    let mut dp = Array1::zeros(n_at * dim);
    if target.len() == dim {
        let p = power_spectrum(x, spec);
        for i in 0..n_at {
            for t in 0..dim {
                dp[i * dim + t] = (target[t] - p[t]) / n_at.max(1) as f64;
            }
        }
    } else if target.len() == n_at * dim {
        for i in 0..n_at {
            for t in 0..dim {
                dp[i * dim + t] = target[i * dim + t] - loc[[i, t]];
            }
        }
    } else {
        return Array1::zeros(x.len());
    }
    let j = jacobian(x, spec);
    let mut dr = tikhonov_jtj(&j, dp.view(), 1e-3);
    strip_rigid(x, &mut dr);
    dr
}

/// Residual step: local environments move away from their mean, pulled
/// back by analytic `J`. `rmsd` is a cap, not a rescaling of a null FD.
pub fn step_away<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    _observed: &[Array1<f64>],
    spec: SoapSpec,
    rmsd: f64,
    _rng: &mut R,
) -> Array1<f64> {
    let loc = local_spectra(x, spec);
    let n_at = loc.nrows();
    let dim = spec.dim();
    if n_at == 0 || dim == 0 {
        return x.to_owned();
    }
    let mut mu = vec![0.0; dim];
    for i in 0..n_at {
        for t in 0..dim {
            mu[t] += loc[[i, t]] / n_at as f64;
        }
    }
    let mut target = Array1::zeros(n_at * dim);
    for i in 0..n_at {
        for t in 0..dim {
            // p_i + (p_i − μ) = 2 p_i − μ: walk away from the mean environment.
            target[i * dim + t] = 2.0 * loc[[i, t]] - mu[t];
        }
    }
    let mut dr = pullback(x, target.view(), spec);
    let cap = rmsd.max(1e-6);
    let n = n_at.max(1) as f64;
    let cur = (dr.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    if cur > cap {
        dr *= cap / cur;
    }
    &x.to_owned() + &dr
}

/// Finite-difference Jacobian of the *global* average, test-only.
#[cfg(test)]
pub fn jacobian_fd(x: ArrayView1<f64>, spec: SoapSpec, eps: f64) -> Array2<f64> {
    let dim = spec.dim();
    let n = x.len();
    let mut j = Array2::<f64>::zeros((dim, n));
    let eps = eps.max(1e-6);
    let mut xp = x.to_owned();
    for k in 0..n {
        let old = xp[k];
        xp[k] = old + eps;
        let plus = power_spectrum(xp.view(), spec);
        xp[k] = old - eps;
        let minus = power_spectrum(xp.view(), spec);
        xp[k] = old;
        let col = (&plus - &minus) / (2.0 * eps);
        for a in 0..dim {
            j[[a, k]] = col[a];
        }
    }
    j
}

fn atom_expand(
    x: ArrayView1<f64>,
    i: usize,
    n_at: usize,
    rcut: f64,
    spec: SoapSpec,
) -> (Array1<f64>, Vec<f64>) {
    let n_max = spec.n_max;
    let l_max = spec.l_max;
    let n_lm = (l_max + 1) * (l_max + 1);
    let mut c = vec![0.0; n_max * n_lm];
    let xi = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
    for j in 0..n_at {
        if j == i {
            continue;
        }
        let d = [x[3 * j] - xi[0], x[3 * j + 1] - xi[1], x[3 * j + 2] - xi[2]];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if r >= rcut || r < 1e-12 {
            continue;
        }
        let u = [d[0] / r, d[1] / r, d[2] / r];
        let (ylm, _) = tesseral(u, l_max);
        let fc = fcut(r, rcut);
        for n in 0..n_max {
            let w = radial(n, r, rcut) * fc;
            let base = n * n_lm;
            for (lm, &y) in ylm.iter().enumerate() {
                c[base + lm] += w * y;
            }
        }
    }
    let mut p = Array1::<f64>::zeros(spec.dim());
    let mut t = 0usize;
    for n in 0..n_max {
        for np in n..n_max {
            for l in 0..=l_max {
                let mut s = 0.0;
                for m in -(l as i32)..=(l as i32) {
                    let lm = lm_index(l, m);
                    s += c[n * n_lm + lm] * c[np * n_lm + lm];
                }
                p[t] = s;
                t += 1;
            }
        }
    }
    (p, c)
}

fn radial(n: usize, r: f64, rcut: f64) -> f64 {
    if r <= 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    let u = (r / rcut).clamp(0.0, 1.0);
    u.powi(n as i32) * (-0.5 * (r / (rcut / 3.0)).powi(2)).exp()
}

fn dradial(n: usize, r: f64, rcut: f64) -> f64 {
    if r <= 1e-15 {
        return 0.0;
    }
    let g = radial(n, r, rcut);
    let sigma = rcut / 3.0;
    g * (n as f64 / r - r / (sigma * sigma))
}

fn fcut(r: f64, rcut: f64) -> f64 {
    if r >= rcut {
        0.0
    } else {
        0.5 * (1.0 + (PI * r / rcut).cos())
    }
}

fn dfcut(r: f64, rcut: f64) -> f64 {
    if r >= rcut || r <= 0.0 {
        0.0
    } else {
        -0.5 * (PI / rcut) * (PI * r / rcut).sin()
    }
}

fn lm_index(l: usize, m: i32) -> usize {
    l * l + (m + l as i32) as usize
}

/// Real tesseral Y_lm(u) and ∂Y/∂u of the harmonic polynomial, |u|=1.
fn tesseral(u: [f64; 3], l_max: usize) -> (Vec<f64>, Vec<[f64; 3]>) {
    let n_lm = (l_max + 1) * (l_max + 1);
    let mut y = vec![0.0; n_lm];
    let mut dy = vec![[0.0; 3]; n_lm];
    let (x, yy, z) = (u[0], u[1], u[2]);
    let s = (4.0 * PI).sqrt();
    // l = 0
    y[0] = 1.0 / s;
    if l_max == 0 {
        return (y, dy);
    }
    let n1 = (3.0 / (4.0 * PI)).sqrt();
    y[lm_index(1, -1)] = n1 * yy;
    dy[lm_index(1, -1)] = [0.0, n1, 0.0];
    y[lm_index(1, 0)] = n1 * z;
    dy[lm_index(1, 0)] = [0.0, 0.0, n1];
    y[lm_index(1, 1)] = n1 * x;
    dy[lm_index(1, 1)] = [n1, 0.0, 0.0];
    if l_max == 1 {
        return (y, dy);
    }
    let n2m = (15.0 / (4.0 * PI)).sqrt();
    let n20 = (5.0 / (16.0 * PI)).sqrt();
    let n22 = (15.0 / (16.0 * PI)).sqrt();
    y[lm_index(2, -2)] = n2m * x * yy;
    dy[lm_index(2, -2)] = [n2m * yy, n2m * x, 0.0];
    y[lm_index(2, -1)] = n2m * yy * z;
    dy[lm_index(2, -1)] = [0.0, n2m * z, n2m * yy];
    // 3z^2-1 = 2z^2-x^2-y^2 on the sphere
    y[lm_index(2, 0)] = n20 * (2.0 * z * z - x * x - yy * yy);
    dy[lm_index(2, 0)] = [n20 * (-2.0 * x), n20 * (-2.0 * yy), n20 * 4.0 * z];
    y[lm_index(2, 1)] = n2m * x * z;
    dy[lm_index(2, 1)] = [n2m * z, 0.0, n2m * x];
    y[lm_index(2, 2)] = n22 * (x * x - yy * yy);
    dy[lm_index(2, 2)] = [n22 * 2.0 * x, n22 * (-2.0 * yy), 0.0];
    if l_max == 2 {
        return (y, dy);
    }
    let n33 = (35.0 / (32.0 * PI)).sqrt();
    let n32 = (105.0 / (4.0 * PI)).sqrt();
    let n31 = (21.0 / (32.0 * PI)).sqrt();
    let n30 = (7.0 / (16.0 * PI)).sqrt();
    let n32z = (105.0 / (16.0 * PI)).sqrt();
    y[lm_index(3, -3)] = n33 * yy * (3.0 * x * x - yy * yy);
    dy[lm_index(3, -3)] = [n33 * yy * 6.0 * x, n33 * (3.0 * x * x - 3.0 * yy * yy), 0.0];
    y[lm_index(3, -2)] = n32 * x * yy * z;
    dy[lm_index(3, -2)] = [n32 * yy * z, n32 * x * z, n32 * x * yy];
    // y(5z^2-1) = y(4z^2-x^2-y^2)
    y[lm_index(3, -1)] = n31 * yy * (4.0 * z * z - x * x - yy * yy);
    dy[lm_index(3, -1)] = [
        n31 * yy * (-2.0 * x),
        n31 * (4.0 * z * z - x * x - 3.0 * yy * yy),
        n31 * yy * 8.0 * z,
    ];
    // z(5z^2-3) = z(2z^2-3x^2-3y^2)
    y[lm_index(3, 0)] = n30 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * yy * yy);
    dy[lm_index(3, 0)] = [
        n30 * (-6.0 * x * z),
        n30 * (-6.0 * yy * z),
        n30 * (6.0 * z * z - 3.0 * x * x - 3.0 * yy * yy),
    ];
    y[lm_index(3, 1)] = n31 * x * (4.0 * z * z - x * x - yy * yy);
    dy[lm_index(3, 1)] = [
        n31 * (4.0 * z * z - 3.0 * x * x - yy * yy),
        n31 * x * (-2.0 * yy),
        n31 * x * 8.0 * z,
    ];
    y[lm_index(3, 2)] = n32z * z * (x * x - yy * yy);
    dy[lm_index(3, 2)] = [
        n32z * z * 2.0 * x,
        n32z * z * (-2.0 * yy),
        n32z * (x * x - yy * yy),
    ];
    y[lm_index(3, 3)] = n33 * x * (x * x - 3.0 * yy * yy);
    dy[lm_index(3, 3)] = [
        n33 * (3.0 * x * x - 3.0 * yy * yy),
        n33 * x * (-6.0 * yy),
        0.0,
    ];
    let _ = l_max;
    (y, dy)
}

fn strip_rigid(x: ArrayView1<f64>, dr: &mut Array1<f64>) {
    let n = x.len() / 3;
    if n == 0 {
        return;
    }
    let mut com = [0.0; 3];
    let mut mean_dr = [0.0; 3];
    for i in 0..n {
        for a in 0..3 {
            com[a] += x[3 * i + a];
            mean_dr[a] += dr[3 * i + a];
        }
    }
    let inv = 1.0 / n as f64;
    for a in 0..3 {
        com[a] *= inv;
        mean_dr[a] *= inv;
    }
    for i in 0..n {
        for a in 0..3 {
            dr[3 * i + a] -= mean_dr[a];
        }
    }
    // Least-squares ω: I ω = Σ r × dr
    let mut inertia = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for i in 0..n {
        let r = [
            x[3 * i] - com[0],
            x[3 * i + 1] - com[1],
            x[3 * i + 2] - com[2],
        ];
        let v = [dr[3 * i], dr[3 * i + 1], dr[3 * i + 2]];
        rhs[0] += r[1] * v[2] - r[2] * v[1];
        rhs[1] += r[2] * v[0] - r[0] * v[2];
        rhs[2] += r[0] * v[1] - r[1] * v[0];
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        for a in 0..3 {
            inertia[a][a] += r2;
            for b in 0..3 {
                inertia[a][b] -= r[a] * r[b];
            }
        }
    }
    for a in 0..3 {
        inertia[a][a] += 1e-9;
    }
    if let Some(w) = solve3(inertia, rhs) {
        for i in 0..n {
            let r = [
                x[3 * i] - com[0],
                x[3 * i + 1] - com[1],
                x[3 * i + 2] - com[2],
            ];
            dr[3 * i] -= w[1] * r[2] - w[2] * r[1];
            dr[3 * i + 1] -= w[2] * r[0] - w[0] * r[2];
            dr[3 * i + 2] -= w[0] * r[1] - w[1] * r[0];
        }
    }
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv = 1.0 / det;
    let mut c = [[0.0; 3]; 3];
    c[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv;
    c[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv;
    c[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv;
    c[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv;
    c[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv;
    c[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv;
    c[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv;
    c[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv;
    c[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv;
    Some([
        c[0][0] * b[0] + c[0][1] * b[1] + c[0][2] * b[2],
        c[1][0] * b[0] + c[1][1] * b[1] + c[1][2] * b[2],
        c[2][0] * b[0] + c[2][1] * b[1] + c[2][2] * b[2],
    ])
}

/// Solve `(J^T J + λ I) dr = J^T dp`.
fn tikhonov_jtj(j: &Array2<f64>, dp: ArrayView1<f64>, lambda: f64) -> Array1<f64> {
    let nfeat = j.nrows();
    let ncoord = j.ncols();
    let mut a = Array2::<f64>::zeros((ncoord, ncoord));
    let mut rhs = Array1::<f64>::zeros(ncoord);
    for c in 0..ncoord {
        for i in 0..nfeat {
            rhs[c] += j[[i, c]] * dp[i];
        }
        for d in 0..=c {
            let mut s = 0.0;
            for i in 0..nfeat {
                s += j[[i, c]] * j[[i, d]];
            }
            a[[c, d]] = s;
            a[[d, c]] = s;
        }
        a[[c, c]] += lambda.max(1e-12);
    }
    chol_solve(&a, &rhs).unwrap_or_else(|| Array1::zeros(ncoord))
}

fn chol_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = b.len();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = s / l[[i, i]];
    }
    let mut z = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }
    Some(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn tetra() -> Array1<f64> {
        // Regular tetrahedron, edge ~√8.
        Array1::from_vec(vec![
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        ])
    }

    fn squashed() -> Array1<f64> {
        Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.15, 0.08, 0.02, 0.18, 1.22, 0.11, 0.95, 0.85, 1.28,
        ])
    }

    fn rotate_z(x: ArrayView1<f64>, ang: f64) -> Array1<f64> {
        let c = ang.cos();
        let s = ang.sin();
        let n = x.len() / 3;
        let mut y = Array1::zeros(x.len());
        for i in 0..n {
            let xx = x[3 * i];
            let yy = x[3 * i + 1];
            y[3 * i] = c * xx - s * yy;
            y[3 * i + 1] = s * xx + c * yy;
            y[3 * i + 2] = x[3 * i + 2];
        }
        y
    }

    #[test]
    fn packed_dim_is_n_times_nplus1_over_2_times_lplus1() {
        let s = SoapSpec {
            n_max: 3,
            l_max: 3,
            rcut_nn: 2.5,
        };
        assert_eq!(s.dim(), 24);
    }

    #[test]
    fn distinct_shapes_have_distinct_soap() {
        let spec = SoapSpec::default();
        let a = tetra();
        let b = squashed();
        let pa = power_spectrum(a.view(), spec);
        let pb = power_spectrum(b.view(), spec);
        let d: f64 = pa
            .iter()
            .zip(pb.iter())
            .map(|(u, v)| (u - v) * (u - v))
            .sum();
        assert!(d.sqrt() > 1e-3, "soap distance {d}");
    }

    #[test]
    fn soap_is_invariant_to_rotation_and_translation() {
        let spec = SoapSpec::default();
        let a = tetra();
        let mut t = rotate_z(a.view(), 0.7);
        for i in 0..t.len() / 3 {
            t[3 * i] += 3.0;
            t[3 * i + 1] -= 1.5;
        }
        let pa = power_spectrum(a.view(), spec);
        let pt = power_spectrum(t.view(), spec);
        let d: f64 = pa
            .iter()
            .zip(pt.iter())
            .map(|(u, v)| (u - v) * (u - v))
            .sum();
        assert!(
            d.sqrt() < 1e-6,
            "SOAP moved under rigid motion: {}",
            d.sqrt()
        );
    }

    #[test]
    fn pullback_reduces_soap_distance_to_the_target() {
        let spec = SoapSpec::default();
        let x = squashed();
        let p0 = power_spectrum(x.view(), spec);
        let mut x_tgt = x.clone();
        x_tgt[1] += 0.12;
        x_tgt[6] -= 0.10;
        let target = power_spectrum(x_tgt.view(), spec);
        let y = &x + &pullback(x.view(), target.view(), spec);
        let p1 = power_spectrum(y.view(), spec);
        let d0: f64 = p0
            .iter()
            .zip(target.iter())
            .map(|(u, v)| (u - v) * (u - v))
            .sum();
        let d1: f64 = p1
            .iter()
            .zip(target.iter())
            .map(|(u, v)| (u - v) * (u - v))
            .sum();
        assert!(
            d1 < d0 * 0.8,
            "pullback did not approach the SOAP target: {d0} -> {d1}"
        );
    }

    #[test]
    fn step_away_moves_more_than_one_atom() {
        let spec = SoapSpec::default();
        let x = squashed();
        let p = power_spectrum(x.view(), spec);
        let mut rng = StdRng::seed_from_u64(2);
        let y = step_away(x.view(), &[p], spec, 0.5, &mut rng);
        let mut moved = 0usize;
        let n = x.len() / 3;
        for i in 0..n {
            let mut d2 = 0.0;
            for k in 0..3 {
                let d = y[3 * i + k] - x[3 * i + k];
                d2 += d * d;
            }
            if d2.sqrt() > 0.05 {
                moved += 1;
            }
        }
        assert!(
            moved >= 2,
            "SOAP pullback moved {moved} atoms; expected a concerted step"
        );
    }

    #[test]
    fn analytic_j_matches_fd_inside_the_cutoff() {
        let spec = SoapSpec {
            n_max: 3,
            l_max: 3,
            rcut_nn: 4.0,
        };
        let mut x = tetra();
        x[0] += 0.35;
        x[4] -= 0.22;
        x[8] += 0.18;
        let ja = global_from_local(x.view(), spec);
        let jf = jacobian_fd(x.view(), spec, 1e-5);
        let mut max_a = 0.0_f64;
        let mut max_d = 0.0_f64;
        for t in 0..ja.nrows() {
            for k in 0..ja.ncols() {
                max_a = max_a.max(jf[[t, k]].abs());
                max_d = max_d.max((ja[[t, k]] - jf[[t, k]]).abs());
            }
        }
        assert!(
            max_d < 1e-4 * max_a.max(1e-6_f64) + 1e-6,
            "analytic J disagrees with FD: max|Δ|={max_d} max|FD|={max_a}"
        );
    }

    #[test]
    fn global_soap_j_annihilates_translations() {
        let spec = SoapSpec::default();
        let mut x = tetra();
        x[1] += 0.2;
        let j = global_from_local(x.view(), spec);
        for a in 0..3 {
            let mut nrm = 0.0;
            for t in 0..j.nrows() {
                let mut s = 0.0;
                for i in 0..x.len() / 3 {
                    s += j[[t, 3 * i + a]];
                }
                nrm += s * s;
            }
            assert!(
                nrm.sqrt() < 1e-7,
                "translation {a} is not in ker J: {}",
                nrm.sqrt()
            );
        }
    }

    fn global_from_local(x: ArrayView1<f64>, spec: SoapSpec) -> Array2<f64> {
        let jl = jacobian(x, spec);
        let n = x.len() / 3;
        let dim = spec.dim();
        let mut g = Array2::<f64>::zeros((dim, 3 * n));
        if n == 0 {
            return g;
        }
        for i in 0..n {
            for t in 0..dim {
                for k in 0..3 * n {
                    g[[t, k]] += jl[[i * dim + t, k]] / n as f64;
                }
            }
        }
        g
    }
}
