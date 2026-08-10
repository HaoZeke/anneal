//! SOAP power spectrum and the Cartesian pullback through its Jacobian.
//!
//! The density expansion coefficients are
//! `c_{nlm}(i) = Σ_{j≠i} g_n(r_{ij}) Y_{lm}(hat r_{ij}) f_cut(r_{ij})`.
//! The rotationally invariant power spectrum is
//! `p_{nn'l}(i) = Σ_m c_{nlm}(i) c_{n'lm}(i)`, averaged over atoms.
//!
//! A step `Δp` in that space is pulled back by Tikhonov
//! `ΔR = argmin ||J ΔR − Δp||² + λ||ΔR||²` with `J = ∂p/∂R` from
//! central differences. That is a concerted multi-atom move. It is not a
//! one-atom hop and it is not a Marks/CSA oracle: `Δp` is a residual
//! direction in the observed SOAP span.

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
    /// Cutoff as a multiple of the structure's median nearest-neighbour distance.
    pub rcut_nn: f64,
}

impl Default for SoapSpec {
    fn default() -> Self {
        Self {
            n_max: 3,
            l_max: 3,
            rcut_nn: 2.5,
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
    let n_at = x.len() / 3;
    let mut acc = Array1::<f64>::zeros(spec.dim());
    if n_at < 2 {
        return acc;
    }
    let rcut = spec.rcut_nn * median_nn(x);
    if !(rcut > 0.0) {
        return acc;
    }
    for i in 0..n_at {
        let p = atom_soap(x, i, n_at, rcut, spec);
        acc += &p;
    }
    acc / n_at as f64
}

/// Central-difference Jacobian `∂p/∂R`, shape `(dim, 3N)`.
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

/// Cartesian displacement that realises `target − p(x)` in SOAP space.
pub fn pullback(x: ArrayView1<f64>, target: ArrayView1<f64>, spec: SoapSpec) -> Array1<f64> {
    let p = power_spectrum(x, spec);
    let mut dp = Array1::zeros(p.len());
    for i in 0..p.len() {
        dp[i] = target.get(i).copied().unwrap_or(0.0) - p[i];
    }
    let j = jacobian_fd(x, spec, 1e-4);
    tikhonov(&j, dp.view(), 1e-4)
}

/// A residual SOAP step from `x`: direction orthogonal to the span of
/// `observed` power spectra (or a random direction if that span is full),
/// pulled back and scaled to Cartesian RMSD `rmsd`.
pub fn step_away<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    observed: &[Array1<f64>],
    spec: SoapSpec,
    rmsd: f64,
    rng: &mut R,
) -> Array1<f64> {
    let p = power_spectrum(x, spec);
    let dim = p.len();
    if dim == 0 || x.len() < 3 {
        return x.to_owned();
    }
    let u = residual_direction(&p, observed, rng);
    let step = 0.2 * p.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-3);
    let target = &p + &(u * step);
    let mut dr = pullback(x, target.view(), spec);
    let n_at = (x.len() / 3).max(1) as f64;
    let cur = (dr.iter().map(|v| v * v).sum::<f64>() / n_at).sqrt();
    let want = rmsd.max(1e-6);
    if cur > 1e-12 {
        dr *= want / cur;
    }
    &x.to_owned() + &dr
}

fn residual_direction<R: Rng + ?Sized>(
    p: &Array1<f64>,
    observed: &[Array1<f64>],
    rng: &mut R,
) -> Array1<f64> {
    let dim = p.len();
    let mut u = Array1::from_vec((0..dim).map(|_| rng.random::<f64>() - 0.5).collect());
    // Deflate directions already present in the archive mean and the
    // incumbent SOAP. What remains is the residual cell in p-space.
    if !observed.is_empty() {
        let mut mu = Array1::<f64>::zeros(dim);
        let mut n = 0.0;
        for q in observed {
            if q.len() != dim {
                continue;
            }
            mu = mu + q;
            n += 1.0;
        }
        if n > 0.0 {
            mu /= n;
            let d = p - &mu;
            let dn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            if dn > 1e-12 {
                // Walk *away* from the observed mean, then remove that
                // component from the random draw so U is not the ico axis.
                let proj = u.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f64>() / (dn * dn);
                for i in 0..dim {
                    u[i] -= proj * d[i];
                }
                u = u + &(&d / dn);
            }
        }
    }
    let nn = u.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    u / nn
}

fn atom_soap(x: ArrayView1<f64>, i: usize, n_at: usize, rcut: f64, spec: SoapSpec) -> Array1<f64> {
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
        let fc = fcut(r, rcut);
        let ylm = real_ylm_all(d, l_max);
        for n in 0..n_max {
            let gn = radial(n, r, rcut) * fc;
            let base = n * n_lm;
            for (lm, &y) in ylm.iter().enumerate() {
                c[base + lm] += gn * y;
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
    p
}

fn radial(n: usize, r: f64, rcut: f64) -> f64 {
    let u = (r / rcut).clamp(0.0, 1.0);
    u.powi(n as i32) * (-0.5 * (r / (rcut / 3.0)).powi(2)).exp()
}

fn fcut(r: f64, rcut: f64) -> f64 {
    if r >= rcut {
        0.0
    } else {
        0.5 * (1.0 + (PI * r / rcut).cos())
    }
}

fn lm_index(l: usize, m: i32) -> usize {
    l * l + (m + l as i32) as usize
}

fn real_ylm_all(d: [f64; 3], l_max: usize) -> Vec<f64> {
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-15);
    let x = d[0] / r;
    let y = d[1] / r;
    let z = d[2] / r;
    let ct = z.clamp(-1.0, 1.0);
    let st = (1.0 - ct * ct).sqrt();
    let phi = y.atan2(x);
    let n_lm = (l_max + 1) * (l_max + 1);
    let mut out = vec![0.0; n_lm];
    for l in 0..=l_max {
        for m in -(l as i32)..=(l as i32) {
            out[lm_index(l, m)] = real_ylm(l, m, ct, st, phi);
        }
    }
    out
}

fn real_ylm(l: usize, m: i32, ct: f64, st: f64, phi: f64) -> f64 {
    let ma = m.unsigned_abs() as usize;
    let p = assoc_legendre(l, ma, ct, st);
    let nrm = sph_norm(l, ma);
    if m == 0 {
        nrm * p
    } else if m > 0 {
        std::f64::consts::SQRT_2 * nrm * p * (ma as f64 * phi).cos()
    } else {
        std::f64::consts::SQRT_2 * nrm * p * (ma as f64 * phi).sin()
    }
}

fn sph_norm(l: usize, m: usize) -> f64 {
    // sqrt((2l+1)/4π · (l-m)!/(l+m)!)
    let mut f = (2 * l + 1) as f64 / (4.0 * PI);
    for k in 0..(2 * m) {
        f /= (l - m + 1 + k) as f64;
    }
    f.sqrt()
}

/// Ferrers `P_l^m(cos θ)` with `(sin θ)^m` factored as `st^m`.
fn assoc_legendre(l: usize, m: usize, ct: f64, st: f64) -> f64 {
    if m > l {
        return 0.0;
    }
    // P_m^m = (-1)^m (2m-1)!! st^m
    let mut pmm = 1.0;
    if m > 0 {
        let mut odd = 1.0;
        for k in 1..=m {
            pmm *= -odd * st;
            odd += 2.0;
        }
    }
    if l == m {
        return pmm;
    }
    let mut pmmp1 = ct * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return pmmp1;
    }
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        pll = (ct * (2 * ll - 1) as f64 * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

fn median_nn(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return 1.0;
    }
    let mut nn = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d2: f64 = (0..3)
                .map(|k| {
                    let d = x[3 * i + k] - x[3 * j + k];
                    d * d
                })
                .sum();
            if d2 < best {
                best = d2;
            }
        }
        nn.push(best.sqrt());
    }
    nn.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    nn[n / 2]
}

/// Solve `(J J^T + λ I) μ = dp`, return `J^T μ`.
fn tikhonov(j: &Array2<f64>, dp: ArrayView1<f64>, lambda: f64) -> Array1<f64> {
    let nfeat = j.nrows();
    let ncoord = j.ncols();
    let mut a = Array2::<f64>::zeros((nfeat, nfeat));
    for i in 0..nfeat {
        for k in 0..nfeat {
            let mut s = 0.0;
            for c in 0..ncoord {
                s += j[[i, c]] * j[[k, c]];
            }
            a[[i, k]] = s;
        }
        a[[i, i]] += lambda.max(1e-12);
    }
    let mu = match chol_solve(&a, &dp.to_owned()) {
        Some(v) => v,
        None => return Array1::zeros(ncoord),
    };
    let mut dr = Array1::<f64>::zeros(ncoord);
    for c in 0..ncoord {
        let mut s = 0.0;
        for i in 0..nfeat {
            s += j[[i, c]] * mu[i];
        }
        dr[c] = s;
    }
    dr
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
        // Regular tetrahedron, edge ~√2.
        Array1::from_vec(vec![
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
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
        let mut b = tetra();
        b[0] += 1.4;
        b[4] -= 0.8;
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
        let mut x = tetra();
        x[0] += 0.45;
        x[5] -= 0.30;
        let p0 = power_spectrum(x.view(), spec);
        // Target is the SOAP of a nearby Cartesian point, so it lies in
        // the range of J rather than in a symmetry-null direction.
        let mut x_tgt = x.clone();
        x_tgt[1] += 0.08;
        x_tgt[6] -= 0.06;
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
        let x = tetra();
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
}
