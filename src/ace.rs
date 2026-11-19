//! ACE ν=3 / λ-SOAP invariants: CG contraction of one spherical expansion.
//!
//! SOAP is the ν=2 power spectrum `Σ_m c_{nlm} c_{n'lm}`. The next
//! rotationally invariant is the bispectrum
//! `B^{l1 l2 l3}_n = Σ CG(l1 m1, l2 m2 | l3 m3) C_{n l1 m1} C_{n l2 m2} C*_{n l3 m3}`.
//! That is the scalar ACE ν=3 / featomic λ=0 product of three expansions.
//! Surface coordination is not this map: SOFI/IRA already return a length.

use ndarray::Array2;
use std::cell::RefCell;
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
    if n < 0 {
        return 0.0;
    }
    let n = n as usize;
    if n < F.len() {
        return F[n];
    }
    // The cached tables reach l = 8, where the 3j prefactor divides by
    // (l1 + l2 + l3 + 1)! = 25!.
    (F.len()..=n).fold(F[F.len() - 1], |product, k| product * k as f64)
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
    let n = if l_max <= CG_CACHE_MAX {
        n_max * triples_cached(l_max).len()
    } else {
        dim(n_max, l_max)
    };
    let mut out = vec![0.0; n];
    from_c_into(c, n_max, l_max, &mut out);
    out
}

/// [`from_c`] writing into `out`, which must hold [`dim`] entries.
pub fn from_c_into(c: &[f64], n_max: usize, l_max: usize, out: &mut [f64]) {
    if l_max <= CG_CACHE_MAX {
        let n = n_max * triples_cached(l_max).len();
        if out.len() < n {
            return;
        }
        from_c_cached(c, n_max, l_max, out);
    } else {
        let owned = from_c_general(c, n_max, l_max);
        if out.len() < owned.len() {
            return;
        }
        out[..owned.len()].copy_from_slice(&owned);
    }
}

const CG_CACHE_MAX: usize = 8;

/// One nonzero CG term. `i1`, `i2`, `i3` are `lm_index` offsets, so the
/// contraction adds the same products in the same order as the `m` loops.
#[derive(Clone, Copy)]
struct CgTerm {
    i1: usize,
    i2: usize,
    i3: usize,
    g: f64,
}

fn nonzero_terms(l_max: usize) -> &'static [Vec<CgTerm>] {
    use std::sync::OnceLock;
    static TABLES: [OnceLock<Vec<Vec<CgTerm>>>; CG_CACHE_MAX + 1] =
        [const { OnceLock::new() }; CG_CACHE_MAX + 1];
    TABLES[l_max].get_or_init(|| {
        triples(l_max)
            .into_iter()
            .map(|(l1, l2, l3)| {
                let mut terms = Vec::new();
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
                        terms.push(CgTerm {
                            i1: lm_index(l1, m1),
                            i2: lm_index(l2, m2),
                            i3: lm_index(l3, m3),
                            g,
                        });
                    }
                }
                terms
            })
            .collect()
    })
}

fn from_c_cached(c: &[f64], n_max: usize, l_max: usize, out: &mut [f64]) {
    let n_lm = (l_max + 1) * (l_max + 1);
    let n_c = n_max * n_lm;
    let trips = triples_cached(l_max);
    let terms = nonzero_terms(l_max);
    // Complex coefficients once per radial channel. The `m` loops used to
    // rebuild each one for every triple that touches it. The buffers are
    // reused across centres; each call overwrites the prefix it reads.
    ACE_SCRATCH.with(|slot| {
        let mut scratch = slot.borrow_mut();
        scratch.ensure(n_c);
        for n in 0..n_max {
            for l in 0..=l_max {
                for m in -(l as i32)..=(l as i32) {
                    let (cr, ci) = coeff(c, n, l, m, n_lm);
                    let at = n * n_lm + lm_index(l, m);
                    scratch.re[at] = cr;
                    scratch.im[at] = ci;
                }
            }
        }
        for n in 0..n_max {
            let base = n * n_lm;
            for (k, triple_terms) in terms.iter().enumerate() {
                let mut acc = 0.0;
                // SAFETY: same index bound as the adjoint. Each radial
                // channel owns `n_lm` coefficients and the scratch holds
                // `n_c` of them.
                debug_assert!(base + n_lm <= scratch.re.len());
                unsafe {
                    let re = scratch.re.as_ptr();
                    let im = scratch.im.as_ptr();
                    for term in triple_terms {
                        let a = (*re.add(base + term.i1), *im.add(base + term.i1));
                        let b = (*re.add(base + term.i2), *im.add(base + term.i2));
                        let d = (*re.add(base + term.i3), *im.add(base + term.i3));
                        let p = cmul(cmul(a, b), (d.0, -d.1));
                        acc += term.g * p.0;
                    }
                }
                out[n * trips.len() + k] = acc;
            }
        }
    });
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

/// `∂B/∂c` for one centre, shape `(ace_dim, n_max n_lm)`.
///
/// The contraction is multilinear in the three complex channels, so each
/// CG term contributes to the three tesseral coefficients it reads. A
/// central difference repeats the whole contraction twice per coefficient:
/// at `l_max = 6` that is a few hundred contractions per atom, and a
/// second species doubles the atoms' channels. The products below are the
/// derivative of `Re(a b conj(d))`.
pub fn d_from_c(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
    let n_lm = (l_max + 1) * (l_max + 1);
    let n_c = n_max * n_lm;
    let trips = triples(l_max);
    let n_b = n_max * trips.len();
    let mut j = Array2::<f64>::zeros((n_b, n_c));
    if n_c == 0 || n_max == 0 || trips.is_empty() {
        return j;
    }
    let mut cp = c.to_vec();
    if cp.len() < n_c {
        cp.resize(n_c, 0.0);
    }
    let terms: Vec<Vec<CgTerm>> = if l_max <= CG_CACHE_MAX {
        nonzero_terms(l_max).to_vec()
    } else {
        trips
            .iter()
            .map(|&(l1, l2, l3)| finite_terms(l1, l2, l3))
            .collect()
    };
    let mut re = vec![0.0; n_c];
    let mut im = vec![0.0; n_c];
    for n in 0..n_max {
        for l in 0..=l_max {
            for m in -(l as i32)..=(l as i32) {
                let (cr, ci) = coeff(&cp, n, l, m, n_lm);
                let at = n * n_lm + lm_index(l, m);
                re[at] = cr;
                im[at] = ci;
            }
        }
    }
    for n in 0..n_max {
        let base = n * n_lm;
        for (k, triple_terms) in terms.iter().enumerate() {
            let row = n * trips.len() + k;
            let (l1, l2, l3) = trips[k];
            for term in triple_terms {
                let ar = re[base + term.i1];
                let ai = im[base + term.i1];
                let br = re[base + term.i2];
                let bi = im[base + term.i2];
                let dr = re[base + term.i3];
                let di = im[base + term.i3];
                let g = term.g;
                // ∂Re(a b conj(d)) / ∂(re, im) of each factor.
                scatter(
                    &mut j,
                    row,
                    base,
                    l1,
                    m_of(l1, term.i1),
                    g * (br * dr + bi * di),
                    g * (-bi * dr + br * di),
                );
                scatter(
                    &mut j,
                    row,
                    base,
                    l2,
                    m_of(l2, term.i2),
                    g * (ar * dr + ai * di),
                    g * (-ai * dr + ar * di),
                );
                scatter(
                    &mut j,
                    row,
                    base,
                    l3,
                    m_of(l3, term.i3),
                    g * (ar * br - ai * bi),
                    g * (ar * bi + ai * br),
                );
            }
        }
    }
    j
}

/// `(∂B/∂c)ᵀ w` for one centre, added into `out`.
///
/// Same products as [`d_from_c`], contracted with `w` while the term is in
/// hand, so the Jacobian matrix is never stored. `w` is one weight per
/// invariant, in the order [`from_c`] returns, and `out` is one entry per
/// tesseral coefficient.
pub fn adjoint_from_c(c: &[f64], n_max: usize, l_max: usize, weights: &[f64], out: &mut [f64]) {
    let n_lm = (l_max + 1) * (l_max + 1);
    let n_c = n_max * n_lm;
    if n_c == 0 || n_max == 0 || out.len() < n_c {
        return;
    }
    let trips_owned = (l_max > CG_CACHE_MAX).then(|| triples(l_max));
    let trips: &[(usize, usize, usize)] = match trips_owned.as_deref() {
        Some(owned) => owned,
        None => triples_cached(l_max),
    };
    if trips.is_empty() || weights.len() != n_max * trips.len() {
        return;
    }
    let terms_owned = (l_max > CG_CACHE_MAX).then(|| {
        trips
            .iter()
            .map(|&(l1, l2, l3)| finite_terms(l1, l2, l3))
            .collect::<Vec<_>>()
    });
    let terms: &[Vec<CgTerm>] = match terms_owned.as_deref() {
        Some(owned) => owned,
        None => nonzero_terms(l_max),
    };
    // Pad a short coefficient vector the way the matrix derivative does.
    let padded;
    let cp: &[f64] = if c.len() >= n_c {
        c
    } else {
        let mut grown = c.to_vec();
        grown.resize(n_c, 0.0);
        padded = grown;
        &padded
    };
    // Sum ∂B/∂C in the complex basis, then map each `(l, m)` once.
    // Scatter is linear, so this is `(∂B/∂c)ᵀ w` without a stored row
    // and without a tesseral branch on every CG term.
    ACE_SCRATCH.with(|slot| {
        let mut scratch = slot.borrow_mut();
        scratch.ensure(n_c.max(n_lm));
        for n in 0..n_max {
            for l in 0..=l_max {
                for m in -(l as i32)..=(l as i32) {
                    let (cr, ci) = coeff(cp, n, l, m, n_lm);
                    let at = n * n_lm + lm_index(l, m);
                    scratch.re[at] = cr;
                    scratch.im[at] = ci;
                }
            }
        }
        for n in 0..n_max {
            let base = n * n_lm;
            for (k, triple_terms) in terms.iter().enumerate() {
                let weight = weights[n * trips.len() + k];
                if weight == 0.0 {
                    continue;
                }
                let (l1, l2, l3) = trips[k];
                scratch.dre[..n_lm].fill(0.0);
                scratch.dim[..n_lm].fill(0.0);
                // SAFETY: CgTerm indices are `lm_index` values, and each
                // radial channel owns `n_lm` coefficients. The checks above
                // sized the buffers, so the inner products skip per-term checks.
                debug_assert!(base + n_lm <= scratch.re.len());
                debug_assert!(scratch.dre.len() >= n_lm);
                unsafe {
                    let re = scratch.re.as_ptr();
                    let im = scratch.im.as_ptr();
                    let dre = scratch.dre.as_mut_ptr();
                    let dim_p = scratch.dim.as_mut_ptr();
                    for term in triple_terms {
                        let i1 = term.i1;
                        let i2 = term.i2;
                        let i3 = term.i3;
                        let ar = *re.add(base + i1);
                        let ai = *im.add(base + i1);
                        let br = *re.add(base + i2);
                        let bi = *im.add(base + i2);
                        let dr = *re.add(base + i3);
                        let di = *im.add(base + i3);
                        let g = term.g;
                        *dre.add(i1) += g * (br * dr + bi * di);
                        *dim_p.add(i1) += g * (-bi * dr + br * di);
                        *dre.add(i2) += g * (ar * dr + ai * di);
                        *dim_p.add(i2) += g * (-ai * dr + ar * di);
                        *dre.add(i3) += g * (ar * br - ai * bi);
                        *dim_p.add(i3) += g * (ar * bi + ai * br);
                    }
                }
                let ls = [l1, l2, l3];
                for (slot_l, &l) in ls.iter().enumerate() {
                    if ls[..slot_l].contains(&l) {
                        continue;
                    }
                    for m in -(l as i32)..=(l as i32) {
                        let at = lm_index(l, m);
                        let d_re = scratch.dre[at];
                        let d_im = scratch.dim[at];
                        if d_re == 0.0 && d_im == 0.0 {
                            continue;
                        }
                        scatter_adjoint(out, base, l, m, weight, d_re, d_im);
                    }
                }
            }
        }
    });
}

struct AceScratch {
    re: Vec<f64>,
    im: Vec<f64>,
    dre: Vec<f64>,
    dim: Vec<f64>,
}

impl AceScratch {
    fn ensure(&mut self, n: usize) {
        if self.re.len() < n {
            self.re.resize(n, 0.0);
            self.im.resize(n, 0.0);
        }
        if self.dre.len() < n {
            self.dre.resize(n, 0.0);
            self.dim.resize(n, 0.0);
        }
    }
}

thread_local! {
    static ACE_SCRATCH: RefCell<AceScratch> = const {
        RefCell::new(AceScratch {
            re: Vec::new(),
            im: Vec::new(),
            dre: Vec::new(),
            dim: Vec::new(),
        })
    };
}

fn m_of(l: usize, index: usize) -> i32 {
    index as i32 - (l * l + l) as i32
}

fn finite_terms(l1: usize, l2: usize, l3: usize) -> Vec<CgTerm> {
    let mut terms = Vec::new();
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
            terms.push(CgTerm {
                i1: lm_index(l1, m1),
                i2: lm_index(l2, m2),
                i3: lm_index(l3, m3),
                g,
            });
        }
    }
    terms
}

fn scatter_adjoint(
    out: &mut [f64],
    base: usize,
    l: usize,
    m: i32,
    weight: f64,
    d_re: f64,
    d_im: f64,
) {
    if m == 0 {
        out[base + lm_index(l, 0)] += weight * d_re;
        return;
    }
    let am = m.abs();
    let sign = if am % 2 == 0 { 1.0 } else { -1.0 };
    let pos = base + lm_index(l, am);
    let neg = base + lm_index(l, -am);
    if m > 0 {
        out[pos] += weight * (d_re * sign / SQRT_2);
        out[neg] += weight * (d_im * sign / SQRT_2);
    } else {
        out[pos] += weight * (d_re / SQRT_2);
        out[neg] += weight * (d_im * (-1.0 / SQRT_2));
    }
}

/// Add `∂p/∂re * dre_scale` and `∂p/∂im * dim_scale` into the tesseral row.
fn scatter(j: &mut Array2<f64>, row: usize, base: usize, l: usize, m: i32, d_re: f64, d_im: f64) {
    if m == 0 {
        j[[row, base + lm_index(l, 0)]] += d_re;
        return;
    }
    let am = m.abs();
    let sign = if am % 2 == 0 { 1.0 } else { -1.0 };
    let pos = base + lm_index(l, am);
    let neg = base + lm_index(l, -am);
    if m > 0 {
        j[[row, pos]] += d_re * sign / SQRT_2;
        j[[row, neg]] += d_im * sign / SQRT_2;
    } else {
        j[[row, pos]] += d_re / SQRT_2;
        j[[row, neg]] += d_im * (-1.0 / SQRT_2);
    }
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

    /// The cached tables go to l = 8, where l1 + l2 + l3 + 1 reaches 25.
    #[test]
    fn cg_stays_normalised_past_twenty_factorial() {
        for (l1, l2, l3) in [(7, 7, 6), (7, 7, 7), (8, 8, 8)] {
            let sum: f64 = (-l1..=l1)
                .map(|m1| cg(l1, m1, l2, -m1, l3, 0).powi(2))
                .sum();
            assert!((sum - 1.0).abs() < 1e-10, "({l1} {l2} {l3}): {sum}");
        }
        let c: Vec<f64> = (0..3 * 9 * 9)
            .map(|i| ((i * 17) % 11) as f64 * 0.01)
            .collect();
        for l_max in [7usize, 8] {
            assert!(
                from_c(&c, 3, l_max).iter().all(|v| v.is_finite()),
                "l_max {l_max}"
            );
        }
    }

    /// `l = 7` and `l = 8` divide by `21!` through `25!`. The reference is the
    /// same Racah sum the table implements, with every factorial an integer,
    /// plus the closed form `⟨l l, l l | 2l 2l⟩ = 1`.
    #[test]
    fn l7_and_l8_couplings_match_the_exact_racah_sum() {
        let facts = factorials_through(33);
        let near = |got: f64, exact: f64, label: &str| {
            let tol = 1e-12 * (1.0 + exact.abs());
            assert!((got - exact).abs() <= tol, "{label}: {got} vs {exact}");
        };
        // One stretched state, Condon-Shortley phase positive.
        near(cg(7, 7, 7, 7, 14, 14), 1.0, "stretched l = 7");
        near(cg(8, 8, 8, 8, 16, 16), 1.0, "stretched l = 8");
        let small = exact_cg(&facts, 1, 0, 1, 0, 2, 0);
        near(small, (2.0_f64 / 3.0).sqrt(), "exact ⟨1 0, 1 0 | 2 0⟩");
        near(
            exact_cg(&facts, 1, 1, 1, 1, 2, 2),
            1.0,
            "exact ⟨1 1, 1 1 | 2 2⟩",
        );
        for l_max in [7usize, 8] {
            for (l1, l2, l3) in triples(l_max) {
                if l1 + l2 + l3 < 20 {
                    continue;
                }
                for m1 in -(l1 as i32)..=l1 as i32 {
                    for m2 in -(l2 as i32)..=l2 as i32 {
                        let m3 = m1 + m2;
                        if m3.abs() > l3 as i32 {
                            continue;
                        }
                        let exact = exact_cg(&facts, l1 as i32, m1, l2 as i32, m2, l3 as i32, m3);
                        near(
                            cg(l1 as i32, m1, l2 as i32, m2, l3 as i32, m3),
                            exact,
                            &format!("({l1} {m1}, {l2} {m2} | {l3} {m3})"),
                        );
                    }
                }
            }
        }
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

    #[test]
    fn the_analytic_derivative_matches_a_central_difference() {
        let n_max = 2usize;
        let l_max = 3usize;
        let n_c = n_max * (l_max + 1) * (l_max + 1);
        let c: Vec<f64> = (0..n_c)
            .map(|i| ((i * 17) % 11) as f64 * 0.01 - 0.05)
            .collect();
        let got = d_from_c(&c, n_max, l_max);
        let eps = 1e-6;
        for k in 0..n_c {
            let mut cp = c.clone();
            cp[k] += eps;
            let bp = from_c(&cp, n_max, l_max);
            cp[k] -= 2.0 * eps;
            let bm = from_c(&cp, n_max, l_max);
            for r in 0..bp.len() {
                let expect = (bp[r] - bm[r]) / (2.0 * eps);
                let delta = (got[[r, k]] - expect).abs();
                assert!(
                    delta < 1e-6,
                    "row {r} col {k}: {} vs {expect} (delta {delta})",
                    got[[r, k]]
                );
            }
        }
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

    #[test]
    fn the_adjoint_matches_the_transposed_derivative() {
        for (n_max, l_max) in [(1usize, 1usize), (3, 6), (2, 8), (1, 9)] {
            let n_c = n_max * (l_max + 1) * (l_max + 1);
            let n_b = dim(n_max, l_max);
            let c: Vec<f64> = (0..n_c)
                .map(|i| ((i * 17 + 3) % 13) as f64 * 0.07 - 0.2)
                .collect();
            let weights: Vec<f64> = (0..n_b)
                .map(|i| {
                    if i % 5 == 0 {
                        0.0
                    } else {
                        ((i % 9) as f64) - 4.0
                    }
                })
                .collect();
            let j = d_from_c(&c, n_max, l_max);
            let mut expect = vec![0.0; n_c];
            for row in 0..n_b {
                for col in 0..n_c {
                    expect[col] += j[[row, col]] * weights[row];
                }
            }
            let mut got = vec![0.0; n_c];
            adjoint_from_c(&c, n_max, l_max, &weights, &mut got);
            let scale = expect.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let worst = got
                .iter()
                .zip(&expect)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst <= 1e-12 * scale.max(1.0),
                "adjoint off by {worst:e} at n_max {n_max} l_max {l_max}"
            );
        }
    }

    /// Little-endian base 2^64.
    #[derive(Clone)]
    struct Nat {
        limbs: Vec<u64>,
    }

    impl Nat {
        fn zero() -> Self {
            Self { limbs: vec![0] }
        }

        fn from_u64(value: u64) -> Self {
            Self { limbs: vec![value] }
        }

        fn normalized(mut limbs: Vec<u64>) -> Self {
            while limbs.len() > 1 && limbs.last() == Some(&0) {
                limbs.pop();
            }
            if limbs.is_empty() {
                limbs.push(0);
            }
            Self { limbs }
        }

        fn is_zero(&self) -> bool {
            self.limbs.iter().all(|limb| *limb == 0)
        }

        fn bits(&self) -> u32 {
            if self.is_zero() {
                return 0;
            }
            let top = *self.limbs.last().expect("a non-zero integer has a limb");
            (self.limbs.len() as u32 - 1) * 64 + (64 - top.leading_zeros())
        }

        fn mul(&self, other: &Self) -> Self {
            if self.is_zero() || other.is_zero() {
                return Self::zero();
            }
            let mut out = vec![0u64; self.limbs.len() + other.limbs.len()];
            for (i, &left) in self.limbs.iter().enumerate() {
                let mut carry = 0u128;
                for (j, &right) in other.limbs.iter().enumerate() {
                    carry += u128::from(out[i + j]) + u128::from(left) * u128::from(right);
                    out[i + j] = carry as u64;
                    carry >>= 64;
                }
                let mut k = i + other.limbs.len();
                while carry > 0 {
                    carry += u128::from(out[k]);
                    out[k] = carry as u64;
                    carry >>= 64;
                    k += 1;
                }
            }
            Self::normalized(out)
        }

        fn mul_u64(&self, factor: u64) -> Self {
            self.mul(&Self::from_u64(factor))
        }

        fn add(&self, other: &Self) -> Self {
            let width = self.limbs.len().max(other.limbs.len());
            let mut out = Vec::with_capacity(width + 1);
            let mut carry = 0u128;
            for i in 0..width {
                carry += u128::from(self.limbs.get(i).copied().unwrap_or(0))
                    + u128::from(other.limbs.get(i).copied().unwrap_or(0));
                out.push(carry as u64);
                carry >>= 64;
            }
            if carry > 0 {
                out.push(carry as u64);
            }
            Self::normalized(out)
        }

        fn cmp_nat(&self, other: &Self) -> std::cmp::Ordering {
            // Leading zeros are already stripped, except the zero value's one limb.
            let a_len = if self.is_zero() { 0 } else { self.limbs.len() };
            let b_len = if other.is_zero() {
                0
            } else {
                other.limbs.len()
            };
            match a_len.cmp(&b_len) {
                std::cmp::Ordering::Equal => self.limbs[..a_len]
                    .iter()
                    .rev()
                    .cmp(other.limbs[..b_len].iter().rev()),
                order => order,
            }
        }

        fn sub(&self, other: &Self) -> Self {
            let mut out = Vec::with_capacity(self.limbs.len());
            let mut borrow = 0i128;
            for i in 0..self.limbs.len() {
                let mut difference = self.limbs[i] as i128
                    - i128::from(other.limbs.get(i).copied().unwrap_or(0))
                    - borrow;
                if difference < 0 {
                    difference += 1i128 << 64;
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                out.push(difference as u64);
            }
            assert_eq!(borrow, 0, "subtracting a larger integer");
            Self::normalized(out)
        }

        /// The integer formed by the bits at and above `shift`, low 64 of them.
        fn shr_u64(&self, shift: u32) -> u64 {
            let limb = (shift / 64) as usize;
            let bit = shift % 64;
            let low = self.limbs.get(limb).copied().unwrap_or(0);
            if bit == 0 {
                return low;
            }
            let high = self.limbs.get(limb + 1).copied().unwrap_or(0);
            (low >> bit) | (high << (64 - bit))
        }
    }

    struct Rat {
        sign: i8,
        num: Nat,
        den: Nat,
    }

    impl Rat {
        fn zero() -> Self {
            Self {
                sign: 0,
                num: Nat::zero(),
                den: Nat::from_u64(1),
            }
        }

        fn add_recip(&mut self, negative: bool, den_term: Nat) {
            let term = if negative { -1 } else { 1 };
            if self.sign == 0 {
                *self = Self {
                    sign: term,
                    num: Nat::from_u64(1),
                    den: den_term,
                };
                return;
            }
            let left = self.num.mul(&den_term);
            let (sign, num) = add_signed(self.sign, &left, term, &self.den);
            let den = self.den.mul(&den_term);
            *self = if sign == 0 {
                Self::zero()
            } else {
                Self { sign, num, den }
            };
        }
    }

    fn add_signed(sign_a: i8, a: &Nat, sign_b: i8, b: &Nat) -> (i8, Nat) {
        if sign_a == sign_b {
            return (sign_a, a.add(b));
        }
        match a.cmp_nat(b) {
            std::cmp::Ordering::Equal => (0, Nat::zero()),
            std::cmp::Ordering::Greater => (sign_a, a.sub(b)),
            std::cmp::Ordering::Less => (sign_b, b.sub(a)),
        }
    }

    fn factorials_through(max: u32) -> Vec<Nat> {
        let mut facts = Vec::with_capacity(max as usize + 1);
        let mut running = Nat::from_u64(1);
        facts.push(running.clone());
        for k in 1..=max {
            running = running.mul_u64(u64::from(k));
            facts.push(running.clone());
        }
        facts
    }

    fn product(facts: &[Nat], args: &[i32]) -> Option<Nat> {
        let mut acc = Nat::from_u64(1);
        for &arg in args {
            if arg < 0 {
                return None;
            }
            acc = acc.mul(facts.get(arg as usize)?);
        }
        Some(acc)
    }

    /// Top 53 bits and the power of two they were shifted by.
    fn mant_exp(value: &Nat) -> (u64, i32) {
        const KEEP: u32 = 53;
        let bits = value.bits();
        if bits <= KEEP {
            return (value.limbs.first().copied().unwrap_or(0), 0);
        }
        (value.shr_u64(bits - KEEP), (bits - KEEP) as i32)
    }

    fn ratio_f64(num: &Nat, den: &Nat) -> f64 {
        let (numerator, num_exp) = mant_exp(num);
        let (denominator, den_exp) = mant_exp(den);
        (numerator as f64 / denominator as f64) * 2.0f64.powi(num_exp - den_exp)
    }

    fn exact_wigner(facts: &[Nat], l1: i32, l2: i32, l3: i32, m1: i32, m2: i32, m3: i32) -> f64 {
        if m1 + m2 + m3 != 0 || l3 < (l1 - l2).abs() || l3 > l1 + l2 {
            return 0.0;
        }
        let tmin = 0.max(l2 - l3 - m1).max(l1 - l3 + m2);
        let tmax = (l1 + l2 - l3).min(l1 - m1).min(l2 + m2);
        if tmin > tmax {
            return 0.0;
        }
        let mut sum = Rat::zero();
        for t in tmin..=tmax {
            let Some(den) = product(
                facts,
                &[
                    t,
                    l1 + l2 - l3 - t,
                    l1 - m1 - t,
                    l2 + m2 - t,
                    l3 - l2 + m1 + t,
                    l3 - l1 - m2 + t,
                ],
            ) else {
                continue;
            };
            sum.add_recip(t % 2 != 0, den);
        }
        if sum.sign == 0 {
            return 0.0;
        }
        let Some(numerator) = product(
            facts,
            &[
                l1 + l2 - l3,
                l1 - l2 + l3,
                -l1 + l2 + l3,
                l1 - m1,
                l1 + m1,
                l2 - m2,
                l2 + m2,
                l3 - m3,
                l3 + m3,
            ],
        ) else {
            return 0.0;
        };
        let denominator = facts[(l1 + l2 + l3 + 1) as usize].clone();
        let square_num = sum.num.mul(&sum.num).mul(&numerator);
        let square_den = sum.den.mul(&sum.den).mul(&denominator);
        let phase = if (l1 - l2 - m3) % 2 == 0 { 1.0 } else { -1.0 };
        phase * f64::from(sum.sign) * ratio_f64(&square_num, &square_den).sqrt()
    }

    fn exact_cg(facts: &[Nat], l1: i32, m1: i32, l2: i32, m2: i32, l3: i32, m3: i32) -> f64 {
        if m1 + m2 != m3 {
            return 0.0;
        }
        let phase = if (l1 - l2 + m3) % 2 == 0 { 1.0 } else { -1.0 };
        phase * ((2 * l3 + 1) as f64).sqrt() * exact_wigner(facts, l1, l2, l3, m1, m2, -m3)
    }
}
