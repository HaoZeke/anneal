//! ACE ν=3 / λ-SOAP invariants: CG contraction of one spherical expansion.
//!
//! SOAP is the ν=2 power spectrum `Σ_m c_{nlm} c_{n'lm}`. The next
//! rotationally invariant is the bispectrum
//! `B^{l1 l2 l3}_n = Σ CG(l1 m1, l2 m2 | l3 m3) C_{n l1 m1} C_{n l2 m2} C*_{n l3 m3}`.
//! That is the scalar ACE ν=3 / featomic λ=0 product of three expansions.
//! Surface coordination is not this map: SOFI/IRA already return a length.

use ndarray::Array2;
use ndarray::ShapeBuilder;
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
    let n_trips = if l_max <= CG_CACHE_MAX {
        triples_cached(l_max).len()
    } else {
        triples(l_max).len()
    };
    let mut out = vec![0.0; n_max * n_trips];
    from_c_into(c, n_max, l_max, &mut out);
    out
}

fn from_c_into(c: &[f64], n_max: usize, l_max: usize, out: &mut [f64]) {
    if l_max <= CG_CACHE_MAX {
        contract_cached(c, n_max, l_max, out);
    } else {
        let direct = from_c_general(c, n_max, l_max);
        out.copy_from_slice(&direct);
    }
}

const CG_CACHE_MAX: usize = 8;

/// One nonzero coupling, in the dense `(m1, m2)` scan order.
#[derive(Clone, Copy)]
struct Term {
    g: f64,
    a: u16,
    b: u16,
    d: u16,
}

/// Nonzero Clebsch–Gordan products for one `l_max`.
///
/// A cpu-clock profile of the Pt(111) heptamer proposal put 65% of the
/// samples in `from_c`. The instructions were the index arithmetic into
/// the dense table, the tesseral-to-complex conversion repeated for every
/// magnetic pair, and `g * Re(C C conj C)`. Most table entries are zero.
/// This list keeps the nonzero couplings, and the complex coefficients are
/// converted once per call. Each scalar is still the sum of those products
/// in the dense loop's order.
struct ContractionPlan {
    n_lm: usize,
    spans: Vec<(u32, u32)>,
    terms: Vec<Term>,
}

fn contraction_plan(l_max: usize) -> &'static ContractionPlan {
    use std::sync::OnceLock;
    static PLANS: [OnceLock<ContractionPlan>; CG_CACHE_MAX + 1] =
        [const { OnceLock::new() }; CG_CACHE_MAX + 1];
    PLANS[l_max].get_or_init(|| {
        let n_lm = (l_max + 1) * (l_max + 1);
        let trips = triples_cached(l_max);
        let table = cg_table(l_max);
        let mut spans = Vec::with_capacity(trips.len());
        let mut terms = Vec::new();
        for &(l1, l2, l3) in trips {
            let start = terms.len();
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
                    let a = lm_index(l1, m1);
                    let b = lm_index(l2, m2);
                    let d = lm_index(l3, m3);
                    debug_assert!(a < n_lm && b < n_lm && d < n_lm);
                    terms.push(Term {
                        g,
                        a: a as u16,
                        b: b as u16,
                        d: d as u16,
                    });
                }
            }
            let len = terms.len() - start;
            spans.push((start as u32, len as u32));
        }
        ContractionPlan { n_lm, spans, terms }
    })
}

/// `Re(a b conj(d))`, the same operations as `cmul(cmul(a, b), (d.0, -d.1)).0`.
#[inline(always)]
fn real_conj_product(a: (f64, f64), b: (f64, f64), d: (f64, f64)) -> f64 {
    let ab0 = a.0 * b.0 - a.1 * b.1;
    let ab1 = a.0 * b.1 + a.1 * b.0;
    ab0 * d.0 - ab1 * (-d.1)
}

fn contract_cached(c: &[f64], n_max: usize, l_max: usize, out: &mut [f64]) {
    let plan = contraction_plan(l_max);
    let n_lm = plan.n_lm;
    let n_trips = plan.spans.len();
    debug_assert_eq!(out.len(), n_max * n_trips);
    let slots = n_max * n_lm;
    let mut heap = Vec::new();
    let mut stack = [(0.0, 0.0); 256];
    let cx = if slots <= stack.len() {
        &mut stack[..slots]
    } else {
        heap.resize(slots, (0.0, 0.0));
        &mut heap[..]
    };
    for n in 0..n_max {
        for l in 0..=l_max {
            for m in -(l as i32)..=(l as i32) {
                let lm = lm_index(l, m);
                cx[n * n_lm + lm] = coeff(c, n, l, m, n_lm);
            }
        }
    }
    if n_max == 3 {
        contract_three(cx, n_lm, n_trips, plan, out);
        return;
    }
    let mut acc = vec![0.0; n_max];
    for (k, &(start, len)) in plan.spans.iter().enumerate() {
        acc.fill(0.0);
        let terms = &plan.terms[start as usize..(start + len) as usize];
        for term in terms {
            let g = term.g;
            for n in 0..n_max {
                let base = n * n_lm;
                let a = cx[base + term.a as usize];
                let b = cx[base + term.b as usize];
                let d = cx[base + term.d as usize];
                acc[n] += g * real_conj_product(a, b, d);
            }
        }
        for n in 0..n_max {
            out[n * n_trips + k] = acc[n];
        }
    }
}

fn contract_three(
    cx: &[(f64, f64)],
    n_lm: usize,
    n_trips: usize,
    plan: &ContractionPlan,
    out: &mut [f64],
) {
    for (k, &(start, len)) in plan.spans.iter().enumerate() {
        let mut a0 = 0.0;
        let mut a1 = 0.0;
        let mut a2 = 0.0;
        let terms = &plan.terms[start as usize..(start + len) as usize];
        for term in terms {
            let g = term.g;
            let p0 = product_at(cx, 0, n_lm, term);
            let p1 = product_at(cx, 1, n_lm, term);
            let p2 = product_at(cx, 2, n_lm, term);
            a0 += g * p0;
            a1 += g * p1;
            a2 += g * p2;
        }
        out[k] = a0;
        out[n_trips + k] = a1;
        out[2 * n_trips + k] = a2;
    }
}

#[inline(always)]
fn product_at(cx: &[(f64, f64)], n: usize, n_lm: usize, term: &Term) -> f64 {
    let base = n * n_lm;
    let ia = base + term.a as usize;
    let ib = base + term.b as usize;
    let id = base + term.d as usize;
    debug_assert!(ia < cx.len() && ib < cx.len() && id < cx.len());
    // SAFETY: `term` stores `lm_index` values, each `< n_lm`, and `base` is
    // `n * n_lm` for `n < n_max`. `cx` has `n_max * n_lm` entries, so the
    // three indexes are inside it. The assert above checks that in debug.
    let (a, b, d) = unsafe {
        (
            *cx.get_unchecked(ia),
            *cx.get_unchecked(ib),
            *cx.get_unchecked(id),
        )
    };
    real_conj_product(a, b, d)
}

/// The dense cached contraction, kept so a test can require bit identity.
#[cfg(test)]
pub(crate) fn from_c_dense(c: &[f64], n_max: usize, l_max: usize) -> Vec<f64> {
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
    if n_max == 3 && l_max <= CG_CACHE_MAX {
        // The Pt(111) proposal spends this derivative inside every ACE
        // Jacobian. n_max is 3 there, and each of the 2 n_c contractions was
        // repeating products that do not read the perturbed coefficient.
        d_from_c_reused(c, l_max)
    } else {
        d_from_c_rescan(c, n_max, l_max)
    }
}

/// Finite-difference Jacobian. Two full contractions per coefficient.
///
/// Kept so the reused derivative can be checked against this scan, including
/// the `n_max == 3` path the proposal runs.
pub(crate) fn d_from_c_rescan(c: &[f64], n_max: usize, l_max: usize) -> Array2<f64> {
    let n_lm = (l_max + 1) * (l_max + 1);
    let n_c = n_max * n_lm;
    let n_b = dim(n_max, l_max);
    // Column-major so the contraction reads one coefficient's column
    // contiguously. The entries match the row-major finite difference.
    let mut j = Array2::<f64>::zeros((n_b, n_c).f());
    if n_c == 0 {
        return j;
    }
    let eps = 1e-7;
    let mut cp = c.to_vec();
    if cp.len() < n_c {
        cp.resize(n_c, 0.0);
    }
    let mut bp = vec![0.0; n_b];
    let mut bm = vec![0.0; n_b];
    for k in 0..n_c {
        let old = cp[k];
        cp[k] = old + eps;
        from_c_into(&cp, n_max, l_max, &mut bp);
        cp[k] = old - eps;
        from_c_into(&cp, n_max, l_max, &mut bm);
        cp[k] = old;
        for r in 0..n_b {
            j[[r, k]] = (bp[r] - bm[r]) / (2.0 * eps);
        }
    }
    j
}

fn lm_parts(lm: usize) -> (usize, i32) {
    let mut l = 0usize;
    while (l + 1) * (l + 1) <= lm {
        l += 1;
    }
    let m = lm as i32 - (l * l) as i32 - l as i32;
    (l, m)
}

/// Complex `(l, m)` slots whose value changes when the real coefficient
/// `slot` is perturbed. `m = 0` reads one real entry. Every other real
/// entry is shared by `+m` and `-m`.
fn complex_dependents(slot: usize) -> (usize, [usize; 2]) {
    let (l, m) = lm_parts(slot);
    if m == 0 {
        (1, [slot, 0])
    } else {
        let am = if m < 0 { -m } else { m };
        (2, [lm_index(l, am), lm_index(l, -am)])
    }
}

fn term_depends(term: &Term, n_dep: usize, dep: [usize; 2]) -> bool {
    let hit = |index: u16| {
        let index = index as usize;
        index == dep[0] || (n_dep == 2 && index == dep[1])
    };
    hit(term.a) || hit(term.b) || hit(term.d)
}

fn fill_complex(c: &[f64], n_max: usize, l_max: usize, n_lm: usize, cx: &mut [(f64, f64)]) {
    for n in 0..n_max {
        for l in 0..=l_max {
            for m in -(l as i32)..=(l as i32) {
                let lm = lm_index(l, m);
                cx[n * n_lm + lm] = coeff(c, n, l, m, n_lm);
            }
        }
    }
}

fn refresh_complex(
    c: &[f64],
    n: usize,
    n_lm: usize,
    n_dep: usize,
    dep: [usize; 2],
    cx: &mut [(f64, f64)],
) {
    for i in 0..n_dep {
        let lm = dep[i];
        let (l, m) = lm_parts(lm);
        cx[n * n_lm + lm] = coeff(c, n, l, m, n_lm);
    }
}

/// `(B(c+εe_k) - B(c-εe_k)) / 2ε` for `n_max = 3`, in the contraction's order.
///
/// A term whose three complex factors ignore `e_k` has the same product at
/// `c`, `c+εe_k` and `c-εe_k`. That product is computed once and added with
/// `acc += g * p`, which is the operation [`contract_three`] performs. Terms
/// that read `e_k` are evaluated at the shifted coefficients. Spans with no
/// such term differ by zero, and `(b - b) / 2ε` is `+0.0` for a finite `b`,
/// so those entries are stored as `0.0`.
fn d_from_c_reused(c: &[f64], l_max: usize) -> Array2<f64> {
    let n_max = 3usize;
    let plan = contraction_plan(l_max);
    let n_lm = plan.n_lm;
    let n_trips = plan.spans.len();
    let n_c = n_max * n_lm;
    let n_b = n_max * n_trips;
    debug_assert_eq!(n_b, dim(n_max, l_max));
    let mut j = Array2::<f64>::zeros((n_b, n_c).f());
    let eps = 1e-7;
    let mut cp = c.to_vec();
    if cp.len() < n_c {
        cp.resize(n_c, 0.0);
    }
    let mut cx = vec![(0.0, 0.0); n_c];
    fill_complex(&cp, n_max, l_max, n_lm, &mut cx);
    let mut cached = vec![0.0; plan.terms.len() * n_max];
    for (ti, term) in plan.terms.iter().enumerate() {
        for n in 0..n_max {
            cached[ti * n_max + n] = product_at(&cx, n, n_lm, term);
        }
    }
    // Unperturbed contraction, same `acc += g * p` order as `contract_three`.
    // An output that ignores the perturbed coefficient has this value on both
    // sides of the difference. Finite values cancel to +0; a non-finite value
    // cancels to NaN, which is what subtracting the two contractions produces.
    let mut base = vec![0.0; n_b];
    for n in 0..n_max {
        for (k, &(start, len)) in plan.spans.iter().enumerate() {
            let terms = &plan.terms[start as usize..(start + len) as usize];
            let mut acc = 0.0;
            for (offset, term) in terms.iter().enumerate() {
                let ti = start as usize + offset;
                acc += term.g * cached[ti * n_max + n];
            }
            base[n * n_trips + k] = acc;
        }
    }
    let mut spans_for_slot = Vec::with_capacity(n_lm);
    for slot in 0..n_lm {
        let (n_dep, dep) = complex_dependents(slot);
        let mut spans = Vec::new();
        for (k, &(start, len)) in plan.spans.iter().enumerate() {
            let terms = &plan.terms[start as usize..(start + len) as usize];
            if terms.iter().any(|term| term_depends(term, n_dep, dep)) {
                spans.push(k);
            }
        }
        spans_for_slot.push((n_dep, dep, spans));
    }
    let mut bp = vec![0.0; n_b];
    let mut bm = vec![0.0; n_b];
    for k in 0..n_c {
        let n_star = k / n_lm;
        let slot = k - n_star * n_lm;
        let (n_dep, dep, spans) = &spans_for_slot[slot];
        let old = cp[k];
        cp[k] = old + eps;
        refresh_complex(&cp, n_star, n_lm, *n_dep, *dep, &mut cx);
        bp.fill(0.0);
        accumulate_shifted(
            &cx, &cached, plan, n_star, *n_dep, *dep, spans, n_lm, &mut bp,
        );
        cp[k] = old - eps;
        refresh_complex(&cp, n_star, n_lm, *n_dep, *dep, &mut cx);
        bm.fill(0.0);
        accumulate_shifted(
            &cx, &cached, plan, n_star, *n_dep, *dep, spans, n_lm, &mut bm,
        );
        cp[k] = old;
        refresh_complex(&cp, n_star, n_lm, *n_dep, *dep, &mut cx);
        for r in 0..n_b {
            j[[r, k]] = (base[r] - base[r]) / (2.0 * eps);
        }
        for &span in spans.iter() {
            let r = n_star * n_trips + span;
            j[[r, k]] = (bp[r] - bm[r]) / (2.0 * eps);
        }
    }
    j
}

fn accumulate_shifted(
    cx: &[(f64, f64)],
    cached: &[f64],
    plan: &ContractionPlan,
    n_star: usize,
    n_dep: usize,
    dep: [usize; 2],
    spans: &[usize],
    n_lm: usize,
    out: &mut [f64],
) {
    let n_trips = plan.spans.len();
    let n_max = 3usize;
    for &k in spans {
        let (start, len) = plan.spans[k];
        let terms = &plan.terms[start as usize..(start + len) as usize];
        let mut acc = 0.0;
        for (offset, term) in terms.iter().enumerate() {
            let ti = start as usize + offset;
            let p = if term_depends(term, n_dep, dep) {
                product_at(cx, n_star, n_lm, term)
            } else {
                cached[ti * n_max + n_star]
            };
            acc += term.g * p;
        }
        out[n_star * n_trips + k] = acc;
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

    #[test]
    fn the_sparse_contraction_matches_the_dense_one_bit_for_bit() {
        for l_max in 0..=CG_CACHE_MAX {
            let n_lm = (l_max + 1) * (l_max + 1);
            for n_max in [1usize, 3, 4] {
                let c: Vec<f64> = (0..n_max * n_lm)
                    .map(|i| ((i * 17 + l_max * 3) % 23) as f64 * 0.07 - 0.4)
                    .collect();
                assert_same_bits(&c, n_max, l_max);
                let mut shifted = c.clone();
                let eps = 1e-7;
                for k in [0usize, c.len() / 2, c.len() - 1] {
                    let old = shifted[k];
                    shifted[k] = old + eps;
                    assert_same_bits(&shifted, n_max, l_max);
                    shifted[k] = old - eps;
                    assert_same_bits(&shifted, n_max, l_max);
                    shifted[k] = old;
                }
            }
        }
    }

    fn assert_same_bits(c: &[f64], n_max: usize, l_max: usize) {
        let sparse = from_c(c, n_max, l_max);
        let dense = from_c_dense(c, n_max, l_max);
        assert_eq!(sparse.len(), dense.len());
        for (i, (a, b)) in sparse.iter().zip(&dense).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "l_max {l_max} n_max {n_max} slot {i}: {a} vs {b}"
            );
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

    /// The water SOAP pullback asks for this derivative once per atom per
    /// channel. A finite-difference pass at l_max 6 is a few seconds each,
    /// and sixty waters times two channels never returns to the quench.
    #[test]
    fn the_reused_derivative_matches_the_rescan_bit_for_bit() {
        for l_max in 0..=CG_CACHE_MAX {
            let n_max = 3;
            let n_lm = (l_max + 1) * (l_max + 1);
            let c: Vec<f64> = (0..n_max * n_lm)
                .map(|i| ((i * 17 + l_max * 5) % 23) as f64 * 0.07 - 0.4)
                .collect();
            assert_derivative_bits(&c, l_max);
            let mut shifted = c.clone();
            let eps = 1e-7;
            for k in [0usize, c.len() / 2, c.len() - 1] {
                let old = shifted[k];
                shifted[k] = old + eps;
                assert_derivative_bits(&shifted, l_max);
                shifted[k] = old - eps;
                assert_derivative_bits(&shifted, l_max);
                shifted[k] = old;
            }
        }
    }

    fn assert_derivative_bits(c: &[f64], l_max: usize) {
        let reused = d_from_c(c, 3, l_max);
        let rescan = d_from_c_rescan(c, 3, l_max);
        assert_eq!(reused.dim(), rescan.dim());
        for (i, (a, b)) in reused.iter().zip(rescan.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "l_max {l_max} entry {i}: {a} vs {b}"
            );
        }
    }

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
