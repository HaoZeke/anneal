//! ACE ν=3 / λ-SOAP invariants: CG contraction of one spherical expansion.
//!
//! SOAP is the ν=2 power spectrum `Σ_m c_{nlm} c_{n'lm}`. The next
//! rotationally invariant is the bispectrum
//! `B^{l1 l2 l3}_n = Σ CG(l1 m1, l2 m2 | l3 m3) C_{n l1 m1} C_{n l2 m2} C*_{n l3 m3}`.
//! That is the scalar ACE ν=3 / featomic λ=0 product of three expansions.
//! Surface coordination is not this map: SOFI/IRA already return a length.

use ndarray::Array2;
use std::f64::consts::SQRT_2;

/// Allowed `(l1, l2, l3)` with `l1 ≤ l2 ≤ l3`, triangle, and even parity.
pub fn triples(l_max: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for l1 in 0..=l_max {
        for l2 in l1..=l_max {
            for l3 in l2..=l_max {
                if l1 + l2 < l3 {
                    continue;
                }
                if (l1 + l2 + l3) % 2 == 1 {
                    continue;
                }
                v.push((l1, l2, l3));
            }
        }
    }
    v
}

/// `n_max` times the number of CG triples at `l_max`.
pub fn dim(n_max: usize, l_max: usize) -> usize {
    n_max * triples(l_max).len()
}

fn fact(n: i32) -> f64 {
    const F: [f64; 21] = [
        1.0,
        1.0,
        2.0,
        6.0,
        24.0,
        120.0,
        720.0,
        5040.0,
        40320.0,
        362880.0,
        3628800.0,
        39916800.0,
        479001600.0,
        6_227_020_800.0,
        87_178_291_200.0,
        1_307_674_368_000.0,
        20_922_789_888_000.0,
        355_687_428_096_000.0,
        6_402_373_705_728_000.0,
        121_645_100_408_832_000.0,
        2_432_902_008_176_640_000.0,
    ];
    if n < 0 || n as usize >= F.len() {
        0.0
    } else {
        F[n as usize]
    }
}

fn wigner_3j(l1: i32, l2: i32, l3: i32, m1: i32, m2: i32, m3: i32) -> f64 {
    if m1 + m2 + m3 != 0 {
        return 0.0;
    }
    if l3 < (l1 - l2).abs() || l3 > l1 + l2 {
        return 0.0;
    }
    if m1.abs() > l1 || m2.abs() > l2 || m3.abs() > l3 {
        return 0.0;
    }
    if (l1 + l2 + l3) % 2 != 0 && m1 == 0 && m2 == 0 && m3 == 0 {
        return 0.0;
    }
    let tmin = 0.max(l2 - l3 - m1).max(l1 - l3 + m2);
    let tmax = (l1 + l2 - l3).min(l1 - m1).min(l2 + m2);
    if tmin > tmax {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut t = tmin;
    while t <= tmax {
        let denom = fact(t)
            * fact(l1 + l2 - l3 - t)
            * fact(l1 - m1 - t)
            * fact(l2 + m2 - t)
            * fact(l3 - l2 + m1 + t)
            * fact(l3 - l1 - m2 + t);
        if denom != 0.0 {
            let sgn = if t % 2 == 0 { 1.0 } else { -1.0 };
            sum += sgn / denom;
        }
        t += 1;
    }
    let pref =
        fact(l1 + l2 - l3) * fact(l1 - l2 + l3) * fact(-l1 + l2 + l3) / fact(l1 + l2 + l3 + 1);
    let pref = (pref
        * fact(l1 - m1)
        * fact(l1 + m1)
        * fact(l2 - m2)
        * fact(l2 + m2)
        * fact(l3 - m3)
        * fact(l3 + m3))
    .sqrt();
    let phase = if (l1 - l2 - m3) % 2 == 0 { 1.0 } else { -1.0 };
    phase * pref * sum
}

fn cg(l1: i32, m1: i32, l2: i32, m2: i32, l3: i32, m3: i32) -> f64 {
    if m1 + m2 != m3 {
        return 0.0;
    }
    let phase = if (l1 - l2 + m3) % 2 == 0 { 1.0 } else { -1.0 };
    phase * ((2 * l3 + 1) as f64).sqrt() * wigner_3j(l1, l2, l3, m1, m2, -m3)
}

fn lm_index(l: usize, m: i32) -> usize {
    l * l + (m + l as i32) as usize
}

/// Tesseral packed `c[n * n_lm + lm]` → complex `C_m` for one `(n, l)`.
fn complex_c(c: &[f64], n: usize, l: usize, n_lm: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); 2 * l + 1];
    let base = n * n_lm;
    out[l] = (c[base + lm_index(l, 0)], 0.0);
    for m in 1..=l as i32 {
        let rp = c[base + lm_index(l, m)];
        let rm = c[base + lm_index(l, -m)];
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        let re = sign * rp / SQRT_2;
        let im = sign * rm / SQRT_2;
        out[(l as i32 + m) as usize] = (re, im);
        // C_{-m} = (-1)^m conj(C_m)
        out[(l as i32 - m) as usize] = (sign * re, -sign * im);
    }
    out
}

fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// ACE ν=3 scalars, length [`dim`].
pub fn from_c(c: &[f64], n_max: usize, l_max: usize) -> Vec<f64> {
    if l_max <= CG_CACHE_MAX {
        from_c_cached(c, n_max, l_max)
    } else {
        from_c_general(c, n_max, l_max)
    }
}

const CG_CACHE_MAX: usize = 8;

fn from_c_cached(c: &[f64], n_max: usize, l_max: usize) -> Vec<f64> {
    let n_lm = (l_max + 1) * (l_max + 1);
    let trips = triples_cached(l_max);
    let table = cg_table(l_max);
    let mut out = vec![0.0; n_max * trips.len()];
    for n in 0..n_max {
        for (k, &(l1, l2, l3)) in trips.iter().enumerate() {
            let mut acc = 0.0;
            for m1 in -(l1 as i32)..=(l1 as i32) {
                for m2 in -(l2 as i32)..=(l2 as i32) {
                    let m3 = m1 + m2;
                    if m3.abs() > l3 as i32 {
                        continue;
                    }
                    let g = table.get(l1, m1, l2, m2, l3, m3);
                    if g == 0.0 {
                        continue;
                    }
                    let a = coeff(c, n, l1, m1, n_lm);
                    let b = coeff(c, n, l2, m2, n_lm);
                    let d = coeff(c, n, l3, m3, n_lm);
                    let p = cmul(cmul(a, b), (d.0, -d.1));
                    acc += g * p.0;
                }
            }
            out[n * trips.len() + k] = acc;
        }
    }
    out
}

fn from_c_general(c: &[f64], n_max: usize, l_max: usize) -> Vec<f64> {
    let n_lm = (l_max + 1) * (l_max + 1);
    let trips = triples(l_max);
    let mut out = vec![0.0; n_max * trips.len()];
    for n in 0..n_max {
        for (k, &(l1, l2, l3)) in trips.iter().enumerate() {
            let c1 = complex_c(c, n, l1, n_lm);
            let c2 = complex_c(c, n, l2, n_lm);
            let c3 = complex_c(c, n, l3, n_lm);
            let mut acc = 0.0;
            for m1 in -(l1 as i32)..=(l1 as i32) {
                for m2 in -(l2 as i32)..=(l2 as i32) {
                    let m3 = m1 + m2;
                    if m3.abs() > l3 as i32 {
                        continue;
                    }
                    let g = cg(l1 as i32, m1, l2 as i32, m2, l3 as i32, m3);
                    if g == 0.0 {
                        continue;
                    }
                    let a = c1[(l1 as i32 + m1) as usize];
                    let b = c2[(l2 as i32 + m2) as usize];
                    let d = c3[(l3 as i32 + m3) as usize];
                    let p = cmul(cmul(a, b), (d.0, -d.1));
                    acc += g * p.0;
                }
            }
            out[n * trips.len() + k] = acc;
        }
    }
    out
}

/// One complex coefficient, matching [`complex_c`] without the temporary.
fn coeff(c: &[f64], n: usize, l: usize, m: i32, n_lm: usize) -> (f64, f64) {
    let base = n * n_lm;
    let at = |mm: i32| c[base + lm_index(l, mm)];
    if m == 0 {
        return (at(0), 0.0);
    }
    let am = m.abs();
    let sign = if am % 2 == 0 { 1.0 } else { -1.0 };
    let re = sign * at(am) / SQRT_2;
    let im = sign * at(-am) / SQRT_2;
    if m > 0 {
        (re, im)
    } else {
        (sign * re, -sign * im)
    }
}

fn triples_cached(l_max: usize) -> &'static [(usize, usize, usize)] {
    use std::sync::OnceLock;
    static TABLES: [OnceLock<Vec<(usize, usize, usize)>>; CG_CACHE_MAX + 1] =
        [const { OnceLock::new() }; CG_CACHE_MAX + 1];
    TABLES[l_max].get_or_init(|| triples(l_max)).as_slice()
}

struct CgTable {
    l_max: usize,
    span: usize,
    slots: usize,
    values: Vec<f64>,
}

impl CgTable {
    fn build(l_max: usize) -> Self {
        let span = 2 * l_max + 1;
        let slots = (l_max + 1) * span;
        let mut values = vec![0.0; slots * slots * slots];
        let index = |l: usize, m: i32| l * span + (m + l_max as i32) as usize;
        for l1 in 0..=l_max {
            for m1 in -(l1 as i32)..=l1 as i32 {
                for l2 in 0..=l_max {
                    for m2 in -(l2 as i32)..=l2 as i32 {
                        for l3 in 0..=l_max {
                            for m3 in -(l3 as i32)..=l3 as i32 {
                                let g = cg(l1 as i32, m1, l2 as i32, m2, l3 as i32, m3);
                                let i =
                                    (index(l1, m1) * slots + index(l2, m2)) * slots + index(l3, m3);
                                values[i] = g;
                            }
                        }
                    }
                }
            }
        }
        Self {
            l_max,
            span,
            slots,
            values,
        }
    }

    fn get(&self, l1: usize, m1: i32, l2: usize, m2: i32, l3: usize, m3: i32) -> f64 {
        let index = |l: usize, m: i32| l * self.span + (m + self.l_max as i32) as usize;
        let i = (index(l1, m1) * self.slots + index(l2, m2)) * self.slots + index(l3, m3);
        self.values[i]
    }
}

fn cg_table(l_max: usize) -> &'static CgTable {
    use std::sync::OnceLock;
    static TABLES: [OnceLock<CgTable>; CG_CACHE_MAX + 1] =
        [const { OnceLock::new() }; CG_CACHE_MAX + 1];
    TABLES[l_max].get_or_init(|| CgTable::build(l_max))
}

/// `∂B/∂c` for one centre, shape `(ace_dim, n_max n_lm)`.
pub fn d_from_c(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
    let n_lm = (l_max + 1) * (l_max + 1);
    let n_c = n_max * n_lm;
    let n_b = dim(n_max, l_max);
    let mut j = Array2::<f64>::zeros((n_b, n_c));
    if n_c == 0 {
        return j;
    }
    let eps = 1e-7;
    let mut cp = c.to_vec();
    if cp.len() < n_c {
        cp.resize(n_c, 0.0);
    }
    let b0 = from_c(&cp, n_max, l_max);
    let _ = b0;
    for k in 0..n_c {
        let old = cp[k];
        cp[k] = old + eps;
        let bp = from_c(&cp, n_max, l_max);
        cp[k] = old - eps;
        let bm = from_c(&cp, n_max, l_max);
        cp[k] = old;
        for r in 0..n_b {
            j[[r, k]] = (bp[r] - bm[r]) / (2.0 * eps);
        }
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_112_is_normalised() {
        // ⟨1 1, 1 1 | 2 2⟩ = 1
        assert!((cg(1, 1, 1, 1, 2, 2) - 1.0).abs() < 1e-12);
        // ⟨1 0, 1 0 | 2 0⟩ = √(2/3)
        assert!((cg(1, 0, 1, 0, 2, 0) - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn the_cached_contraction_matches_the_direct_one() {
        let c: Vec<f64> = (0..3 * 7 * 7)
            .map(|i| ((i * 17) % 11) as f64 * 0.01)
            .collect();
        for l_max in [3usize, 6] {
            let cached = from_c(&c, 3, l_max);
            let direct = from_c_general(&c, 3, l_max);
            assert_eq!(cached.len(), direct.len());
            for (a, b) in cached.iter().zip(direct.iter()) {
                assert!((a - b).abs() <= 1e-12, "{a} vs {b} at l_max {l_max}");
            }
        }
    }

    #[test]
    fn triples_at_lmax3_include_112() {
        let t = triples(3);
        assert!(t.contains(&(1, 1, 2)));
        assert!(t.contains(&(0, 0, 0)));
        assert!(!t.contains(&(1, 1, 1)));
    }

    /// The water SOAP pullback asks for this derivative once per atom per
    /// channel. A finite-difference pass at l_max 6 is a few seconds each,
    /// and sixty waters times two channels never returns to the quench.
    #[test]
    fn the_lmax6_derivative_returns_before_a_quench_could() {
        let n_max = 3;
        let l_max = 6;
        let n_c = n_max * (l_max + 1) * (l_max + 1);
        let c: Vec<f64> = (0..n_c).map(|i| ((i * 17) % 11) as f64 * 0.01).collect();
        let started = std::time::Instant::now();
        let j = d_from_c(&c, n_max, l_max);
        let elapsed = started.elapsed();
        assert_eq!(j.nrows(), dim(n_max, l_max));
        assert_eq!(j.ncols(), n_c);
        assert!(
            elapsed.as_millis() < 200,
            "ACE derivative took {elapsed:?}; a (H2O)20 pullback would not return"
        );
    }
}
