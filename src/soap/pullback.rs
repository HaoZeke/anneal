//! Normal equations of the stacked SOAP and ACE pullback, built one centre
//! at a time.
//!
//! The Gauss-Newton step solves `(JᵀJ + λI) dr = Jᵀ dp` with `J` the
//! stacked analytic Jacobian. A centre's rows depend only on that centre and
//! its neighbours, so the block of `J` that a centre owns has columns for
//! those atoms alone. This module never forms the full `J`. It builds each
//! centre's block, adds the block's contribution to `JᵀJ` and `Jᵀ dp`, and
//! moves on.
//!
//! Every number is the one the dense route produces, bit for bit:
//!
//! * a Jacobian entry is summed from the same products in the same order;
//! * an entry of `JᵀJ` accumulates `J[r,c] J[r,d]` over rows `r` in
//!   ascending order, one multiply and one add per row, with no fused
//!   multiply-add and no reassociation;
//! * a centre's rows are contiguous, and the accumulator for a pair of
//!   columns is carried from centre to centre, so the running sum is the
//!   dense route's running sum. Rows outside a centre's neighbourhood hold
//!   zeros, and adding a zero product does not change a sum that is never
//!   negative zero.
//!
//! Vector instructions only run independent entries side by side.

use super::*;

/// `JᵀJ` and `Jᵀ dp`, with `JᵀJ` filled on both triangles.
pub(super) struct NormalEquations {
    pub(super) gram: Vec<f64>,
    pub(super) rhs: Vec<f64>,
}

/// One term of a SOAP power-spectrum row: which row, which partner
/// coefficient, and whether the diagonal factor of two applies.
#[derive(Clone, Copy)]
struct SoapTerm {
    row: usize,
    partner: usize,
    doubled: bool,
}

struct Tables {
    lm_l: Vec<usize>,
    soap_terms: Vec<Vec<Vec<SoapTerm>>>,
    lms_of_triple: Vec<Vec<usize>>,
}

fn tables(spec: SoapSpec) -> Tables {
    let n_max = spec.n_max;
    let n_lm = (spec.l_max + 1) * (spec.l_max + 1);
    let angular = spec.l_max + 1;
    let pair_index = |na: usize, np: usize| na * n_max - na * na.saturating_sub(1) / 2 + np - na;
    let lm_l: Vec<usize> = (0..n_lm).map(lm_to_l).collect();
    let mut soap_terms = vec![vec![Vec::new(); angular]; n_max];
    for n in 0..n_max {
        for l in 0..angular {
            let terms = &mut soap_terms[n][l];
            for na in 0..n {
                terms.push(SoapTerm {
                    row: pair_index(na, n) * angular + l,
                    partner: na,
                    doubled: false,
                });
            }
            for np in n..n_max {
                terms.push(SoapTerm {
                    row: pair_index(n, np) * angular + l,
                    partner: np,
                    doubled: np == n,
                });
            }
        }
    }
    let lms_of_triple = crate::ace::triples(spec.l_max)
        .into_iter()
        .map(|(l1, l2, l3)| {
            (0..n_lm)
                .filter(|&lm| {
                    let l = lm_l[lm];
                    l == l1 || l == l2 || l == l3
                })
                .collect()
        })
        .collect();
    Tables {
        lm_l,
        soap_terms,
        lms_of_triple,
    }
}

/// `Jᵀ dp`, the steepest-descent direction of the descriptor leftover:
/// `-½ ∇ ‖dp - J dr‖²` at `dr = 0`.
///
/// A centre's rows reach only its own neighbourhood, and `dp` enters through
/// the chain rule alone. The adjoint of a centre's coefficients,
/// `aᵢ = (∂ rows / ∂ cᵢ)ᵀ dpᵢ`, is formed once, and each pair then needs one
/// contraction with the derivative of its coefficients. No row of `J` and no
/// entry of `JᵀJ` is built, so the cost is a pass over the pairs and one
/// coefficient Jacobian per centre. A centre whose `dp` row is zero costs
/// nothing, which is how fixed atoms drop out.
///
/// Returns `None` for a shape mismatch or a non-finite result, so the caller
/// can fall back to the full solve.
pub(super) fn descent_direction(
    x: ArrayView1<f64>,
    spec: SoapSpec,
    species: Option<&[u32]>,
    coefficients: &[Vec<f64>],
    dp: &[f64],
) -> Option<Vec<f64>> {
    let n_at = x.len() / 3;
    let ace1 = spec.nu3_dim();
    if n_at < 2 || !(spec.rcut_nn > 0.0) || ace1 == 0 || coefficients.len() != n_at {
        return None;
    }
    let n_max = spec.n_max;
    let n_lm = (spec.l_max + 1) * (spec.l_max + 1);
    let n_c = n_max * n_lm;
    let channels = neighbor_channels(species);
    let n_chan = channels.len();
    let dim1 = spec.dim();
    let d0 = dim1 * n_chan;
    let dim = d0 + ace1 * n_chan;
    if dp.len() != n_at * dim || !dp.iter().all(|value| value.is_finite()) {
        return None;
    }
    let triples = ace1 / n_max;
    let tab = tables(spec);
    let mut grad = vec![0.0; 3 * n_at];
    let mut adjoint = vec![0.0; n_chan * n_c];
    let mut scaled = vec![0.0; n_lm];
    for i in 0..n_at {
        let row = &dp[i * dim..(i + 1) * dim];
        if row.iter().all(|&value| value == 0.0) {
            continue;
        }
        adjoint.fill(0.0);
        for ch in 0..n_chan {
            let c_ch = &coefficients[i][ch * n_c..(ch + 1) * n_c];
            let a = &mut adjoint[ch * n_c..(ch + 1) * n_c];
            let soap = &row[ch * dim1..(ch + 1) * dim1];
            if soap.iter().any(|&value| value != 0.0) {
                for n in 0..n_max {
                    for lm in 0..n_lm {
                        let mut sum = 0.0;
                        for term in &tab.soap_terms[n][tab.lm_l[lm]] {
                            let s = if term.doubled {
                                2.0 * c_ch[n * n_lm + lm]
                            } else {
                                c_ch[term.partner * n_lm + lm]
                            };
                            sum += soap[term.row] * s;
                        }
                        a[n * n_lm + lm] += sum;
                    }
                }
            }
            let ace = &row[d0 + ch * ace1..d0 + (ch + 1) * ace1];
            if ace.iter().any(|&value| value != 0.0) {
                let db = crate::ace::d_from_c(c_ch, n_max, spec.l_max);
                let db = db
                    .as_slice()
                    .expect("ACE coefficient Jacobian is contiguous");
                for (t, &weight) in ace.iter().enumerate() {
                    if weight == 0.0 {
                        continue;
                    }
                    let n = t / triples;
                    let base = t * n_c + n * n_lm;
                    for &lm in &tab.lms_of_triple[t % triples] {
                        a[n * n_lm + lm] += weight * db[base + lm];
                    }
                }
            }
        }
        for pair in pair_derivatives_of(x, spec, i) {
            let ch = neighbor_channel(species, pair.neighbor, &channels);
            let a = &adjoint[ch * n_c..(ch + 1) * n_c];
            // push = Σ a[n,lm] (g'ₙ u Yₗₘ + gₙ ∇Yₗₘ), grouped so each
            // harmonic is visited once per radial channel.
            let mut along = 0.0;
            scaled.fill(0.0);
            for n in 0..n_max {
                let mut projected = 0.0;
                for lm in 0..n_lm {
                    projected += a[n * n_lm + lm] * pair.ylm[lm];
                    scaled[lm] += pair.weight[n] * a[n * n_lm + lm];
                }
                along += pair.dweight[n] * projected;
            }
            let mut push = [along * pair.u[0], along * pair.u[1], along * pair.u[2]];
            for lm in 0..n_lm {
                let dy = pair.dy_cartesian[lm];
                for k in 0..3 {
                    push[k] += scaled[lm] * dy[k];
                }
            }
            for k in 0..3 {
                grad[3 * pair.neighbor + k] += push[k];
                grad[3 * i + k] -= push[k];
            }
        }
    }
    grad.iter().all(|value| value.is_finite()).then_some(grad)
}

/// Builds `JᵀJ` and `Jᵀ dp` for all-mobile atoms, or `None` when the dense
/// route has to decide (too few atoms, a non-finite entry, a shape mismatch).
pub(super) fn normal_equations(
    x: ArrayView1<f64>,
    spec: SoapSpec,
    species: Option<&[u32]>,
    coefficients: &[Vec<f64>],
    dp: &[f64],
) -> Option<NormalEquations> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected above.
        return unsafe { normal_equations_avx2(x, spec, species, coefficients, dp) };
    }
    normal_equations_impl(x, spec, species, coefficients, dp)
}

/// The same sums compiled without vector extensions.
#[cfg(test)]
pub(super) fn normal_equations_portable(
    x: ArrayView1<f64>,
    spec: SoapSpec,
    species: Option<&[u32]>,
    coefficients: &[Vec<f64>],
    dp: &[f64],
) -> Option<NormalEquations> {
    normal_equations_impl(x, spec, species, coefficients, dp)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normal_equations_avx2(
    x: ArrayView1<f64>,
    spec: SoapSpec,
    species: Option<&[u32]>,
    coefficients: &[Vec<f64>],
    dp: &[f64],
) -> Option<NormalEquations> {
    normal_equations_impl(x, spec, species, coefficients, dp)
}

#[inline(always)]
fn normal_equations_impl(
    x: ArrayView1<f64>,
    spec: SoapSpec,
    species: Option<&[u32]>,
    coefficients: &[Vec<f64>],
    dp: &[f64],
) -> Option<NormalEquations> {
    let n_at = x.len() / 3;
    let rcut = spec.rcut_nn;
    let ace1 = spec.nu3_dim();
    if n_at < 2 || !(rcut > 0.0) || ace1 == 0 || coefficients.len() != n_at {
        return None;
    }
    let n_max = spec.n_max;
    let n_lm = (spec.l_max + 1) * (spec.l_max + 1);
    let n_c = n_max * n_lm;
    let channels = neighbor_channels(species);
    let n_chan = channels.len();
    let dim1 = spec.dim();
    let d0 = dim1 * n_chan;
    let d1 = ace1 * n_chan;
    let dim = d0 + d1;
    if dp.len() != n_at * dim || !dp.iter().all(|value| value.is_finite()) {
        return None;
    }
    let triples = ace1 / n_max;
    let tab = tables(spec);
    let n3 = 3 * n_at;

    let mut dbdc = Vec::with_capacity(n_at * n_chan);
    for centre in coefficients {
        for ch in 0..n_chan {
            let c0 = ch * n_c;
            dbdc.push(crate::ace::d_from_c(
                &centre[c0..c0 + n_c],
                n_max,
                spec.l_max,
            ));
        }
    }
    let pairs = pair_derivative_cache(x, spec);

    let mut gram = vec![0.0; n3 * n3];
    let mut rhs = vec![0.0; n3];
    let mut block = Vec::<[f64; 4]>::new();
    let mut panels = Vec::<f64>::new();
    let mut accumulators = Vec::<f64>::new();
    let mut dc = vec![[0.0f64; 4]; n_c];
    let mut atoms = Vec::<usize>::new();

    for i in 0..n_at {
        let local = &pairs[i];
        if local.is_empty() {
            continue;
        }
        atoms.clear();
        let mut centre_position = local.len();
        for (q, pair) in local.iter().enumerate() {
            if centre_position == local.len() && pair.neighbor > i {
                centre_position = q;
            }
        }
        for (q, pair) in local.iter().enumerate() {
            if q == centre_position {
                atoms.push(i);
            }
            atoms.push(pair.neighbor);
        }
        if centre_position == local.len() {
            atoms.push(i);
        }
        let size = atoms.len();
        block.clear();
        block.resize(dim * size, [0.0; 4]);
        let own = centre_position;

        for (q, pair) in local.iter().enumerate() {
            let at = if q < centre_position { q } else { q + 1 };
            let jj = pair.neighbor;
            let ch = neighbor_channel(species, jj, &channels);
            let c_ch = &coefficients[i][ch * n_c..(ch + 1) * n_c];
            let u = pair.u;
            for n in 0..n_max {
                let w = pair.weight[n];
                let dw = pair.dweight[n];
                for lm in 0..n_lm {
                    let yv = pair.ylm[lm];
                    let dy = pair.dy_cartesian[lm];
                    dc[n * n_lm + lm] = [
                        dw * u[0] * yv + w * dy[0],
                        dw * u[1] * yv + w * dy[1],
                        dw * u[2] * yv + w * dy[2],
                        0.0,
                    ];
                }
            }
            for n in 0..n_max {
                let dc_n = &dc[n * n_lm..(n + 1) * n_lm];
                let c_n = &c_ch[n * n_lm..(n + 1) * n_lm];
                for lm in 0..n_lm {
                    let l = tab.lm_l[lm];
                    let dcv = dc_n[lm];
                    for term in &tab.soap_terms[n][l] {
                        let s = if term.doubled {
                            2.0 * c_n[lm]
                        } else {
                            c_ch[term.partner * n_lm + lm]
                        };
                        let row = (ch * dim1 + term.row) * size;
                        let mut value = [0.0f64; 4];
                        for a in 0..4 {
                            value[a] = s * dcv[a];
                        }
                        let to = &mut block[row + at];
                        for a in 0..4 {
                            to[a] += value[a];
                        }
                        let from = &mut block[row + own];
                        for a in 0..4 {
                            from[a] -= value[a];
                        }
                    }
                }
            }
            let db = dbdc[i * n_chan + ch]
                .as_slice()
                .expect("ACE coefficient Jacobian is contiguous");
            for t in 0..ace1 {
                let n = t / triples;
                let lms = &tab.lms_of_triple[t % triples];
                let db_n = &db[t * n_c + n * n_lm..t * n_c + (n + 1) * n_lm];
                let dc_n = &dc[n * n_lm..(n + 1) * n_lm];
                let row = (d0 + ch * ace1 + t) * size;
                let mut sum = [0.0f64; 4];
                let mut centre = block[row + own];
                for &lm in lms {
                    let g = db_n[lm];
                    let dcv = dc_n[lm];
                    let mut value = [0.0f64; 4];
                    for a in 0..4 {
                        value[a] = g * dcv[a];
                    }
                    for a in 0..4 {
                        sum[a] += value[a];
                    }
                    for a in 0..4 {
                        centre[a] -= value[a];
                    }
                }
                block[row + at] = sum;
                block[row + own] = centre;
            }
        }

        for row in 0..dim {
            let weight = dp[i * dim + row];
            if weight == 0.0 {
                continue;
            }
            for (position, &atom) in atoms.iter().enumerate() {
                let entry = block[row * size + position];
                for a in 0..3 {
                    rhs[3 * atom + a] += entry[a] * weight;
                }
            }
        }

        let m = 3 * size;
        let n_panels = m.div_ceil(8);
        let width = 8 * n_panels;
        panels.clear();
        panels.resize(n_panels * dim * 8, 0.0);
        let mut finite = true;
        let mut flat = vec![0.0f64; width];
        for row in 0..dim {
            for (position, entry) in block[row * size..(row + 1) * size].iter().enumerate() {
                flat[3 * position] = entry[0];
                flat[3 * position + 1] = entry[1];
                flat[3 * position + 2] = entry[2];
            }
            finite &= flat[..m].iter().all(|value| value.is_finite());
            for (panel, chunk) in flat.chunks_exact(8).enumerate() {
                panels[panel * dim * 8 + row * 8..panel * dim * 8 + row * 8 + 8]
                    .copy_from_slice(chunk);
            }
        }
        if !finite {
            return None;
        }
        accumulators.clear();
        accumulators.resize(width * width, 0.0);
        for (position, &atom) in atoms.iter().enumerate() {
            for a in 0..3 {
                let c = 3 * position + a;
                let row = (3 * atom + a) * n3;
                for (other, &partner) in atoms.iter().enumerate().skip(position) {
                    let from = if other == position { a } else { 0 };
                    let at = c * width + 3 * other + from;
                    let source = row + 3 * partner + from;
                    for q in from..3 {
                        accumulators[at + q - from] = gram[source + q - from];
                    }
                }
            }
        }
        gram_panels(&panels, dim, n_panels, &mut accumulators, width);
        for (position, &atom) in atoms.iter().enumerate() {
            for a in 0..3 {
                let c = 3 * position + a;
                let row = (3 * atom + a) * n3;
                for (other, &partner) in atoms.iter().enumerate().skip(position) {
                    let from = if other == position { a } else { 0 };
                    let at = c * width + 3 * other + from;
                    let target = row + 3 * partner + from;
                    for q in from..3 {
                        gram[target + q - from] = accumulators[at + q - from];
                    }
                }
            }
        }
    }

    for c in 0..n3 {
        for d in 0..c {
            gram[c * n3 + d] = gram[d * n3 + c];
        }
    }
    Some(NormalEquations { gram, rhs })
}

/// Adds `Σ_r B[r,c] B[r,d]` to `acc[c, d]` for every `d` in the panel that
/// holds `c` or beyond it.
///
/// `panels` stores `B` as column panels of eight, each `rows × 8` and
/// contiguous. A `4 × 8` tile of sums stays in registers while the rows
/// stream past, one multiply and one add per row and entry.
#[inline(always)]
fn gram_panels_impl(panels: &[f64], rows: usize, n_panels: usize, acc: &mut [f64], width: usize) {
    for block_c in 0..2 * n_panels {
        let c0 = 4 * block_c;
        let offset = c0 % 8;
        let own = &panels[(c0 / 8) * rows * 8..(c0 / 8 + 1) * rows * 8];
        for panel_d in (c0 / 8)..n_panels {
            let d0 = 8 * panel_d;
            let other = &panels[panel_d * rows * 8..(panel_d + 1) * rows * 8];
            let mut tile = [[0.0f64; 8]; 4];
            for (ci, line) in tile.iter_mut().enumerate() {
                line.copy_from_slice(&acc[(c0 + ci) * width + d0..(c0 + ci) * width + d0 + 8]);
            }
            for r in 0..rows {
                let b: [f64; 8] = other[r * 8..r * 8 + 8].try_into().unwrap();
                let a: [f64; 4] = own[r * 8 + offset..r * 8 + offset + 4].try_into().unwrap();
                for ci in 0..4 {
                    for q in 0..8 {
                        tile[ci][q] += a[ci] * b[q];
                    }
                }
            }
            for (ci, line) in tile.iter().enumerate() {
                acc[(c0 + ci) * width + d0..(c0 + ci) * width + d0 + 8].copy_from_slice(line);
            }
        }
    }
}

#[inline(always)]
fn gram_panels(panels: &[f64], rows: usize, n_panels: usize, acc: &mut [f64], width: usize) {
    gram_panels_impl(panels, rows, n_panels, acc, width);
}

/// Solves `A z = b` by the Cholesky factorisation the dense route uses,
/// reordered so the trailing update runs along contiguous rows.
///
/// Entry `A[i,j]` loses `L[i,k] L[j,k]` for `k = 0, 1, …` in that order,
/// exactly as the left-looking loop subtracts it, and every pivot and
/// quotient is the same expression.
pub(super) fn cholesky_solve(mut a: Vec<f64>, n: usize, b: &[f64]) -> Option<Vec<f64>> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected above.
        return unsafe { cholesky_solve_avx2(&mut a, n, b) };
    }
    cholesky_solve_impl(&mut a, n, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cholesky_solve_avx2(a: &mut [f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    cholesky_solve_impl(a, n, b)
}

#[inline(always)]
fn cholesky_solve_impl(a: &mut [f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut column = vec![0.0; n];
    for k in 0..n {
        let s = a[k * n + k];
        if s <= 0.0 {
            return None;
        }
        let pivot = s.sqrt();
        a[k * n + k] = pivot;
        column[k] = pivot;
        for i in (k + 1)..n {
            let value = a[i * n + k] / pivot;
            a[i * n + k] = value;
            column[i] = value;
        }
        for i in (k + 1)..n {
            let lik = column[i];
            let row = &mut a[i * n + k + 1..i * n + i + 1];
            for (entry, &ljk) in row.iter_mut().zip(&column[k + 1..=i]) {
                *entry -= lik * ljk;
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * y[k];
        }
        y[i] = s / a[i * n + i];
    }
    let mut z = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * z[k];
        }
        z[i] = s / a[i * n + i];
    }
    Some(z)
}
