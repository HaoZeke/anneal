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

/// One nonzero CG term. `i1`, `i2`, `i3` are `lm_index` offsets, so the
/// contraction adds the same products in the same order as the `m` loops.
struct CgTerm {
    i1: usize,
    i2: usize,
    i3: usize,
    g: f64,
    /// Bits set at `i1`, `i2` and `i3`, so a perturbed coefficient can tell
    /// which products it reaches.
    mask: u128,
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
                        let (i1, i2, i3) = (lm_index(l1, m1), lm_index(l2, m2), lm_index(l3, m3));
                        terms.push(CgTerm {
                            i1,
                            i2,
                            i3,
                            g,
                            mask: (1u128 << i1) | (1u128 << i2) | (1u128 << i3),
                        });
                    }
                }
                terms
            })
            .collect()
    })
}

fn from_c_cached(c: &[f64], n_max: usize, l_max: usize) -> Vec<f64> {
    let n_lm = (l_max + 1) * (l_max + 1);
    let trips = triples_cached(l_max);
    let terms = nonzero_terms(l_max);
    // Complex coefficients once per radial channel. The `m` loops used to
    // rebuild each one for every triple that touches it.
    let mut re = vec![0.0; n_max * n_lm];
    let mut im = vec![0.0; n_max * n_lm];
    for n in 0..n_max {
        for l in 0..=l_max {
            for m in -(l as i32)..=(l as i32) {
                let (cr, ci) = coeff(c, n, l, m, n_lm);
                let at = n * n_lm + lm_index(l, m);
                re[at] = cr;
                im[at] = ci;
            }
        }
    }
    let mut out = vec![0.0; n_max * trips.len()];
    for n in 0..n_max {
        let base = n * n_lm;
        for (k, triple_terms) in terms.iter().enumerate() {
            let mut acc = 0.0;
            for term in triple_terms {
                let a = (re[base + term.i1], im[base + term.i1]);
                let b = (re[base + term.i2], im[base + term.i2]);
                let d = (re[base + term.i3], im[base + term.i3]);
                let p = cmul(cmul(a, b), (d.0, -d.1));
                acc += term.g * p.0;
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

/// `∂B/∂c` for one centre, shape `(ace_dim, n_max n_lm)`.
pub fn d_from_c(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
    if l_max <= CG_CACHE_MAX {
        d_from_c_cached(c, n_max, l_max)
    } else {
        d_from_c_general(c, n_max, l_max)
    }
}

fn fill_radial_coefficients(
    c: &[f64],
    n: usize,
    l_max: usize,
    n_lm: usize,
    re: &mut [f64],
    im: &mut [f64],
) {
    for l in 0..=l_max {
        for m in -(l as i32)..=(l as i32) {
            let (cr, ci) = coeff(c, n, l, m, n_lm);
            let at = lm_index(l, m);
            re[at] = cr;
            im[at] = ci;
        }
    }
}

#[cfg(test)]
fn contract_terms(re: &[f64], im: &[f64], terms: &[CgTerm]) -> f64 {
    let mut acc = 0.0;
    for term in terms {
        let a = (re[term.i1], im[term.i1]);
        let b = (re[term.i2], im[term.i2]);
        let d = (re[term.i3], im[term.i3]);
        let p = cmul(cmul(a, b), (d.0, -d.1));
        acc += term.g * p.0;
    }
    acc
}

fn update_perturbed_coefficient(
    c: &[f64],
    n: usize,
    lm: usize,
    n_lm: usize,
    re: &mut [f64],
    im: &mut [f64],
) {
    let mut l = 0;
    while (l + 1) * (l + 1) <= lm {
        l += 1;
    }
    let m = lm as i32 - (l * l + l) as i32;
    let abs_m = m.abs();
    for changed_m in [-abs_m, abs_m] {
        let (cr, ci) = coeff(c, n, l, changed_m, n_lm);
        let at = lm_index(l, changed_m);
        re[at] = cr;
        im[at] = ci;
        if abs_m == 0 {
            break;
        }
    }
}

fn triple_uses_l((l1, l2, l3): (usize, usize, usize), l: usize) -> bool {
    l == l1 || l == l2 || l == l3
}

/// The per-coefficient finite difference with every product recomputed.
#[cfg(test)]
fn d_from_c_recomputed(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
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
    let trips = triples_cached(l_max);
    let terms = nonzero_terms(l_max);
    let mut base_re = vec![0.0; n_max * n_lm];
    let mut base_im = vec![0.0; n_max * n_lm];
    for n in 0..n_max {
        let start = n * n_lm;
        fill_radial_coefficients(
            &cp,
            n,
            l_max,
            n_lm,
            &mut base_re[start..start + n_lm],
            &mut base_im[start..start + n_lm],
        );
    }
    let mut plus_re = vec![0.0; n_lm];
    let mut plus_im = vec![0.0; n_lm];
    let mut minus_re = vec![0.0; n_lm];
    let mut minus_im = vec![0.0; n_lm];
    for k in 0..n_c {
        let n = k / n_lm;
        let lm = k % n_lm;
        let mut l = 0;
        while (l + 1) * (l + 1) <= lm {
            l += 1;
        }
        let old = cp[k];
        let start = n * n_lm;
        plus_re.copy_from_slice(&base_re[start..start + n_lm]);
        plus_im.copy_from_slice(&base_im[start..start + n_lm]);
        minus_re.copy_from_slice(&base_re[start..start + n_lm]);
        minus_im.copy_from_slice(&base_im[start..start + n_lm]);
        cp[k] = old + eps;
        update_perturbed_coefficient(&cp, n, lm, n_lm, &mut plus_re, &mut plus_im);
        cp[k] = old - eps;
        update_perturbed_coefficient(&cp, n, lm, n_lm, &mut minus_re, &mut minus_im);
        cp[k] = old;
        for (triple, (&(l1, l2, l3), triple_terms)) in trips.iter().zip(terms).enumerate() {
            if !triple_uses_l((l1, l2, l3), l) {
                continue;
            }
            let bp = contract_terms(&plus_re, &plus_im, triple_terms);
            let bm = contract_terms(&minus_re, &minus_im, triple_terms);
            j[[n * trips.len() + triple, k]] = (bp - bm) / (2.0 * eps);
        }
    }
    j
}

/// Selective finite difference of the contraction in `c`.
///
/// A perturbed coefficient changes the products that name its `(l, ±|m|)`
/// entries and no others. Every other product is the unperturbed one, so
/// its contribution `g · Re(p)` is computed once per radial channel and the
/// sum over terms adds that stored value in the original term order. The
/// result is bit-identical to recomputing each product.
fn d_from_c_cached(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
    d_from_c_stored(c, n_max, l_max)
}

fn d_from_c_stored(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
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
    let trips = triples_cached(l_max);
    let terms = nonzero_terms(l_max);
    let mut offsets = Vec::with_capacity(terms.len() + 1);
    offsets.push(0usize);
    for triple_terms in terms {
        offsets.push(offsets[offsets.len() - 1] + triple_terms.len());
    }
    let mut base_re = vec![0.0; n_lm];
    let mut base_im = vec![0.0; n_lm];
    let mut base_products = vec![0.0; offsets[terms.len()]];
    // Plus and minus displacements of the `+|m|` and `-|m|` coefficients.
    let mut lane_re = vec![vec![0.0; n_lm]; 4];
    let mut lane_im = vec![vec![0.0; n_lm]; 4];
    for n in 0..n_max {
        fill_radial_coefficients(&cp, n, l_max, n_lm, &mut base_re, &mut base_im);
        for (triple, triple_terms) in terms.iter().enumerate() {
            for (at, term) in triple_terms.iter().enumerate() {
                base_products[offsets[triple] + at] = term_product(&base_re, &base_im, term);
            }
        }
        for l in 0..=l_max {
            for abs_m in 0..=l {
                let signed = [abs_m as i32, -(abs_m as i32)];
                let touched = (1u128 << lm_index(l, signed[0])) | (1u128 << lm_index(l, signed[1]));
                let sides = if abs_m == 0 { 1 } else { 2 };
                for side in 0..2 {
                    // A zero order has one coefficient; the unused lanes repeat it.
                    let lm = lm_index(l, signed[side.min(sides - 1)]);
                    let k = n * n_lm + lm;
                    let old = cp[k];
                    for sign in 0..2 {
                        let lane = 2 * side + sign;
                        lane_re[lane].copy_from_slice(&base_re);
                        lane_im[lane].copy_from_slice(&base_im);
                        cp[k] = if sign == 0 { old + eps } else { old - eps };
                        update_perturbed_coefficient(
                            &cp,
                            n,
                            lm,
                            n_lm,
                            &mut lane_re[lane],
                            &mut lane_im[lane],
                        );
                    }
                    cp[k] = old;
                }
                for (triple, (&(l1, l2, l3), triple_terms)) in trips.iter().zip(terms).enumerate() {
                    if !triple_uses_l((l1, l2, l3), l) {
                        continue;
                    }
                    let stored = &base_products[offsets[triple]..offsets[triple + 1]];
                    let mut acc = [0.0f64; 4];
                    for (term, &unperturbed) in triple_terms.iter().zip(stored) {
                        if term.mask & touched == 0 {
                            for lane in &mut acc {
                                *lane += unperturbed;
                            }
                        } else {
                            for (lane, value) in acc.iter_mut().enumerate() {
                                *value += term_product(&lane_re[lane], &lane_im[lane], term);
                            }
                        }
                    }
                    for side in 0..sides {
                        let k = n * n_lm + lm_index(l, signed[side]);
                        j[[n * trips.len() + triple, k]] =
                            (acc[2 * side] - acc[2 * side + 1]) / (2.0 * eps);
                    }
                }
            }
        }
    }
    j
}

#[inline(always)]
fn term_product(re: &[f64], im: &[f64], term: &CgTerm) -> f64 {
    let a = (re[term.i1], im[term.i1]);
    let b = (re[term.i2], im[term.i2]);
    let d = (re[term.i3], im[term.i3]);
    let p = cmul(cmul(a, b), (d.0, -d.1));
    term.g * p.0
}

#[cfg(test)]
fn d_from_c_cached_full_finite_difference(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
    let n_lm = (l_max + 1) * (l_max + 1);
    let n_c = n_max * n_lm;
    let n_b = dim(n_max, l_max);
    let mut j = Array2::<f64>::zeros((n_b, n_c));
    let eps = 1e-7;
    let mut cp = c.to_vec();
    if cp.len() < n_c {
        cp.resize(n_c, 0.0);
    }
    for k in 0..n_c {
        let old = cp[k];
        cp[k] = old + eps;
        let bp = from_c_cached(&cp, n_max, l_max);
        cp[k] = old - eps;
        let bm = from_c_cached(&cp, n_max, l_max);
        cp[k] = old;
        for r in 0..n_b {
            j[[r, k]] = (bp[r] - bm[r]) / (2.0 * eps);
        }
    }
    j
}

fn d_from_c_general(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
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
    for k in 0..n_c {
        let old = cp[k];
        cp[k] = old + eps;
        let bp = from_c_general(&cp, n_max, l_max);
        cp[k] = old - eps;
        let bm = from_c_general(&cp, n_max, l_max);
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
    fn selective_derivative_is_bit_identical_to_full_finite_difference() {
        let n_max = 3;
        for l_max in [3usize, 6] {
            let n_c = n_max * (l_max + 1) * (l_max + 1);
            let c: Vec<f64> = (0..n_c)
                .map(|i| ((i * 17 + 3) % 23) as f64 * 0.013 - 0.11)
                .collect();
            let selective = d_from_c_cached(&c, n_max, l_max);
            let full = d_from_c_cached_full_finite_difference(&c, n_max, l_max);
            for (index, (&actual, &expected)) in selective.iter().zip(full.iter()).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "derivative differs at flat index {index}, l_max {l_max}"
                );
            }
        }
    }

    #[test]
    fn stored_products_reproduce_the_recomputed_derivative_bit_for_bit() {
        for (n_max, l_max) in [(2usize, 3usize), (3, 6), (4, 5)] {
            let n_c = n_max * (l_max + 1) * (l_max + 1);
            for seed in 0..4usize {
                let c: Vec<f64> = (0..n_c)
                    .map(|i| (((i + 5 * seed) * 29 + 7) % 37) as f64 * 0.021 - 0.37)
                    .collect();
                let stored = d_from_c_cached(&c, n_max, l_max);
                let recomputed = d_from_c_recomputed(&c, n_max, l_max);
                assert_eq!(stored.dim(), recomputed.dim());
                for (index, (&a, &b)) in stored.iter().zip(recomputed.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "derivative differs at flat index {index}, n_max {n_max}, l_max {l_max}"
                    );
                }
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
