// SPDX-License-Identifier: MIT
// Copyright (c) anneal developers
// Portions Copyright (c) 2022 Theochem @ Háskóli Íslands

//! Dense minimum-cost bipartite assignment.
//!
//! Every dimension uses the column-first Kuhn-Munkres permutation. A
//! successive-shortest-path certificate returns that permutation directly
//! when the optimum forces it; only an ambiguous optimum runs the walk. This
//! keeps one deterministic tie break without a matrix-size crossover.

/// Returns the column assigned to each row of a finite `n × n` cost matrix.
///
/// `costs` is row-major. The result is a permutation: every row and every
/// column occurs exactly once. Invalid shapes, non-finite entries, or failure
/// to construct a perfect matching return `None`.
pub fn minimum_cost_assignment(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    certified_assignment(costs, n).map(|(assignment, _)| assignment)
}

/// [`minimum_cost_assignment`] and the [`Certificate`] that forced it, if one did.
pub(crate) fn certified_assignment(
    costs: &[f64],
    n: usize,
) -> Option<(Vec<usize>, Option<Certificate>)> {
    ASSIGNMENT_IS_UNIQUE.with(|cell| cell.set(false));
    if n == 0 {
        return costs.is_empty().then(|| (Vec::new(), None));
    }
    if !well_formed(costs, n) {
        return None;
    }
    forced_kuhn_munkres(costs, n)
}

fn well_formed(costs: &[f64], n: usize) -> bool {
    n.checked_mul(n) == Some(costs.len()) && costs.iter().all(|value| value.is_finite())
}

/// Duals under which one assignment is the only one a Kuhn-Munkres walk
/// can return.
///
/// They are checked against the costs they are used with, so duals from a
/// nearby problem, such as the transpose or a rotated copy, prove an
/// assignment of that problem without solving it.
#[derive(Clone, Debug)]
pub(crate) struct Certificate {
    row_potential: Vec<f64>,
    column_potential: Vec<f64>,
}

impl Certificate {
    /// The certificate of the transposed costs, whose assignment is the inverse.
    pub(crate) fn transposed(&self) -> Self {
        Self {
            row_potential: self.column_potential.clone(),
            column_potential: self.row_potential.clone(),
        }
    }

    /// One extra column dual, so a test can hand the checker a vector of the
    /// wrong length. The stored optimum is unchanged.
    #[cfg(test)]
    fn with_an_extra_column_dual(mut self) -> Self {
        self.column_potential.push(0.0);
        self
    }
}

/// Whether `certificate` forces `assignment` on `costs`, in which case
/// [`minimum_cost_assignment`] returns `assignment`.
pub(crate) fn certifies(
    costs: &[f64],
    n: usize,
    assignment: &[usize],
    certificate: &Certificate,
) -> bool {
    if n == 0
        || assignment.len() != n
        || certificate.row_potential.len() != n + 1
        || certificate.column_potential.len() != n + 1
        || !well_formed(costs, n)
    {
        return false;
    }
    let mut used = vec![false; n];
    for &column in assignment {
        if column >= n || std::mem::replace(&mut used[column], true) {
            return false;
        }
    }
    optimum_is_unique(
        costs,
        n,
        assignment,
        &certificate.row_potential,
        &certificate.column_potential,
        Separation::KuhnMunkres,
    )
}

/// # Safety
///
/// `row` has one entry per column `1..=row.len()`, and the other slices
/// are addressable at `0..=row.len()`. `column` is one of those indices.
#[allow(clippy::too_many_arguments)]
unsafe fn scan_columns(
    row: &[f64],
    row_dual: f64,
    column_potential: &[f64],
    min_reduced: &mut [f64],
    predecessor: &mut [usize],
    used: &[u8],
    column: usize,
) -> (f64, usize) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe {
            scan_avx2(
                row,
                row_dual,
                column_potential,
                min_reduced,
                predecessor,
                used,
                column,
            )
        };
    }
    unsafe {
        scan_scalar(
            row,
            row_dual,
            column_potential,
            min_reduced,
            predecessor,
            used,
            column,
        )
    }
}

/// # Safety
///
/// Same slice contract as [`scan_columns`].
#[allow(clippy::too_many_arguments)]
unsafe fn scan_scalar(
    row: &[f64],
    row_dual: f64,
    column_potential: &[f64],
    min_reduced: &mut [f64],
    predecessor: &mut [usize],
    used: &[u8],
    column: usize,
) -> (f64, usize) {
    let n = row.len();
    let mut delta = f64::INFINITY;
    let mut next_column = 0usize;
    let mut candidate = 1usize;
    while candidate <= n {
        unsafe {
            if *used.as_ptr().add(candidate) == 0 {
                let reduced = *row.as_ptr().add(candidate - 1)
                    - row_dual
                    - *column_potential.as_ptr().add(candidate);
                let slot = min_reduced.as_mut_ptr().add(candidate);
                if reduced < *slot {
                    *slot = reduced;
                    *predecessor.as_mut_ptr().add(candidate) = column;
                }
                if *slot < delta {
                    delta = *slot;
                    next_column = candidate;
                }
            }
        }
        candidate += 1;
    }
    (delta, next_column)
}

/// Four columns at a time. A used column keeps its reduced cost and is
/// not eligible for the running minimum. The minimum is still taken from
/// the lowest column index, which is the scalar tie break.
///
/// # Safety
///
/// AVX2 must be enabled. Slice lengths are the same contract as [`scan_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn scan_avx2(
    row: &[f64],
    row_dual: f64,
    column_potential: &[f64],
    min_reduced: &mut [f64],
    predecessor: &mut [usize],
    used: &[u8],
    column: usize,
) -> (f64, usize) {
    use std::arch::x86_64::{
        _CMP_LT_OQ, _mm_cmpeq_epi8, _mm_cvtsi32_si128, _mm_setzero_si128, _mm256_and_pd,
        _mm256_blendv_pd, _mm256_castsi256_pd, _mm256_cmp_pd, _mm256_cvtepi8_epi64,
        _mm256_loadu_pd, _mm256_movemask_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };

    let n = row.len();
    let mut delta = f64::INFINITY;
    let mut next_column = 0usize;
    let mut candidate = 1usize;
    let dual = _mm256_set1_pd(row_dual);
    let zero = _mm_setzero_si128();
    // The column index is in range by the scan contract. Slice indexing
    // here was a bounds check on every lane, and that check was most of
    // the scan. The arithmetic is the scalar loop: subtract the dual,
    // then the column potential, keep a strict `<` improvement, and take
    // the lowest unused column whose reduced cost is strictly smaller
    // than the running minimum.
    let row_ptr = row.as_ptr();
    let potential_ptr = column_potential.as_ptr();
    let min_ptr = min_reduced.as_mut_ptr();
    let pred_ptr = predecessor.as_mut_ptr();
    let used_ptr = used.as_ptr();
    while candidate + 3 <= n {
        let packed = unsafe { std::ptr::read_unaligned(used_ptr.add(candidate) as *const i32) };
        let bytes = _mm_cvtsi32_si128(packed);
        let eq = _mm_cmpeq_epi8(bytes, zero);
        let open = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(eq));
        let mut lanes = [0.0_f64; 4];
        unsafe {
            let cost = _mm256_loadu_pd(row_ptr.add(candidate - 1));
            let potential = _mm256_loadu_pd(potential_ptr.add(candidate));
            let reduced = _mm256_sub_pd(_mm256_sub_pd(cost, dual), potential);
            let previous = _mm256_loadu_pd(min_ptr.add(candidate));
            let lower = _mm256_and_pd(_mm256_cmp_pd(reduced, previous, _CMP_LT_OQ), open);
            let updated = _mm256_blendv_pd(previous, reduced, lower);
            _mm256_storeu_pd(min_ptr.add(candidate), updated);
            _mm256_storeu_pd(lanes.as_mut_ptr(), updated);
            let improved = _mm256_movemask_pd(lower);
            if improved & 1 != 0 {
                *pred_ptr.add(candidate) = column;
            }
            if improved & 2 != 0 {
                *pred_ptr.add(candidate + 1) = column;
            }
            if improved & 4 != 0 {
                *pred_ptr.add(candidate + 2) = column;
            }
            if improved & 8 != 0 {
                *pred_ptr.add(candidate + 3) = column;
            }
            for offset in 0..4 {
                if *used_ptr.add(candidate + offset) == 0 && lanes[offset] < delta {
                    delta = lanes[offset];
                    next_column = candidate + offset;
                }
            }
        }
        candidate += 4;
    }
    while candidate <= n {
        unsafe {
            if *used_ptr.add(candidate) == 0 {
                let reduced =
                    *row_ptr.add(candidate - 1) - row_dual - *potential_ptr.add(candidate);
                let slot = &mut *min_ptr.add(candidate);
                if reduced < *slot {
                    *slot = reduced;
                    *pred_ptr.add(candidate) = column;
                }
                if *slot < delta {
                    delta = *slot;
                    next_column = candidate;
                }
            }
        }
        candidate += 1;
    }
    (delta, next_column)
}

/// Subtract `delta` from the unused reduced costs, and from the column
/// dual of every used column. One subtraction per updated lane, matching
/// the scalar loop.
fn adjust_reduced(
    delta: f64,
    n: usize,
    used: &[u8],
    min_reduced: &mut [f64],
    column_potential: &mut [f64],
    column_match: &[usize],
    row_potential: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            adjust_reduced_avx2(
                delta,
                n,
                used,
                min_reduced,
                column_potential,
                column_match,
                row_potential,
            );
        }
        return;
    }
    adjust_reduced_scalar(
        delta,
        n,
        used,
        min_reduced,
        column_potential,
        column_match,
        row_potential,
    );
}

fn adjust_reduced_scalar(
    delta: f64,
    n: usize,
    used: &[u8],
    min_reduced: &mut [f64],
    column_potential: &mut [f64],
    column_match: &[usize],
    row_potential: &mut [f64],
) {
    for index in 0..=n {
        if used[index] == 1 {
            row_potential[column_match[index]] += delta;
            column_potential[index] -= delta;
        } else {
            min_reduced[index] -= delta;
        }
    }
}

/// # Safety
///
/// AVX2 must be enabled. `used`, `min_reduced`, `column_potential`, and
/// `column_match` are addressable at `0..=n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn adjust_reduced_avx2(
    delta: f64,
    n: usize,
    used: &[u8],
    min_reduced: &mut [f64],
    column_potential: &mut [f64],
    column_match: &[usize],
    row_potential: &mut [f64],
) {
    use std::arch::x86_64::{
        _mm_cmpeq_epi8, _mm_cvtsi32_si128, _mm_setzero_si128, _mm256_blendv_pd,
        _mm256_castsi256_pd, _mm256_cvtepi8_epi64, _mm256_loadu_pd, _mm256_set1_pd,
        _mm256_storeu_pd, _mm256_sub_pd,
    };
    adjust_one(
        0,
        delta,
        used,
        min_reduced,
        column_potential,
        column_match,
        row_potential,
    );
    let step = _mm256_set1_pd(delta);
    let zero = _mm_setzero_si128();
    let mut index = 1usize;
    while index + 3 <= n {
        let packed = unsafe { std::ptr::read_unaligned(used.as_ptr().add(index) as *const i32) };
        let unused = _mm_cmpeq_epi8(_mm_cvtsi32_si128(packed), zero);
        let open = _mm256_castsi256_pd(_mm256_cvtepi8_epi64(unused));
        unsafe {
            let previous = _mm256_loadu_pd(min_reduced.as_ptr().add(index));
            let subtracted = _mm256_sub_pd(previous, step);
            let updated = _mm256_blendv_pd(previous, subtracted, open);
            _mm256_storeu_pd(min_reduced.as_mut_ptr().add(index), updated);
        }
        unsafe {
            let used_ptr = used.as_ptr();
            let match_ptr = column_match.as_ptr();
            let row_ptr = row_potential.as_mut_ptr();
            let column_ptr = column_potential.as_mut_ptr();
            for offset in 0..4 {
                let at = index + offset;
                if *used_ptr.add(at) == 1 {
                    let matched = *match_ptr.add(at);
                    *row_ptr.add(matched) += delta;
                    *column_ptr.add(at) -= delta;
                }
            }
        }
        index += 4;
    }
    while index <= n {
        adjust_one(
            index,
            delta,
            used,
            min_reduced,
            column_potential,
            column_match,
            row_potential,
        );
        index += 1;
    }
}

fn adjust_one(
    index: usize,
    delta: f64,
    used: &[u8],
    min_reduced: &mut [f64],
    column_potential: &mut [f64],
    column_match: &[usize],
    row_potential: &mut [f64],
) {
    if used[index] == 1 {
        row_potential[column_match[index]] += delta;
        column_potential[index] -= delta;
    } else {
        min_reduced[index] -= delta;
    }
}

/// A permutation that gives every row its own strict nearest column.
///
/// Each row minimum is strictly smaller than the other entries of that
/// row, and the columns are all different. That matching meets the
/// row-minimum lower bound, so it is the only optimal assignment and
/// the same permutation successive shortest paths returns. A tie or a
/// shared column returns `None` and the solver runs.
pub(crate) fn strict_unique_row_minima(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    if n == 0 || n.checked_mul(n) != Some(costs.len()) {
        return None;
    }
    let mut assignment = vec![0usize; n];
    let mut taken = vec![false; n];
    for row in 0..n {
        let base = row * n;
        let mut best_column = 0usize;
        let mut best = costs[base];
        if !best.is_finite() {
            return None;
        }
        for column in 1..n {
            let cost = costs[base + column];
            if !cost.is_finite() {
                return None;
            }
            if cost < best {
                best = cost;
                best_column = column;
            }
        }
        let winners = (0..n)
            .filter(|&column| !(costs[base + column] > best))
            .count();
        if winners != 1 || taken[best_column] {
            return None;
        }
        taken[best_column] = true;
        assignment[row] = best_column;
    }
    Some(assignment)
}

#[cfg(test)]
pub(crate) fn successive_shortest_paths(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    successive_shortest_paths_scanned(costs, n).map(|(assignment, unique)| {
        ASSIGNMENT_IS_UNIQUE.with(|cell| cell.set(unique));
        assignment
    })
}

/// Same permutation as [`successive_shortest_paths`] without the uniqueness scan.
///
/// The symmetry residual only uses the permutation. The alternating-cycle
/// search that feeds the inverse-operation cache does not change it, and
/// that cache reads the flag only after its own call to
/// [`minimum_cost_assignment`].
pub(crate) fn successive_shortest_paths_matching(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    solve_shortest_paths(costs, n, None).map(|(assignment, _)| assignment)
}

/// The Kuhn-Munkres permutation, from the shortest-path scan when the
/// optimum forces it.
///
/// Every edge the walk matches is within [`kuhn_munkres_tolerance`] of
/// tight, and its duals stay feasible, so its assignment costs at most `n`
/// tolerances more than the optimum. When every other assignment costs more
/// than that above the shortest-path optimum, the walk can only return that
/// optimum. The scan is O(n³); the walk restarts its search after every dual
/// step. A near tie runs the walk, which keeps its own tie break.
fn forced_kuhn_munkres(costs: &[f64], n: usize) -> Option<(Vec<usize>, Option<Certificate>)> {
    if let Some((assignment, Some(certificate))) =
        solve_shortest_paths(costs, n, Some(Separation::KuhnMunkres))
    {
        ASSIGNMENT_IS_UNIQUE.with(|cell| cell.set(true));
        return Some((assignment, Some(certificate)));
    }
    kuhn_munkres(costs, n).map(|assignment| (assignment, None))
}

/// How far every other assignment must cost above the optimum.
#[derive(Clone, Copy)]
enum Separation {
    /// More than dual noise: no other optimal edge.
    #[cfg(test)]
    Noise,
    /// More than a Kuhn-Munkres walk can fall short of the optimum.
    KuhnMunkres,
}

fn kuhn_munkres_tolerance(costs: &[f64]) -> f64 {
    let scale = costs
        .iter()
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    scale * 1e-9
}

/// Whether the last successful dense scan had no other optimal edge.
///
/// A Kuhn-Munkres fallback reports false: its tied permutation need not be the
/// inverse selected by the transposed problem.
pub(crate) fn last_dense_optimum_was_unique() -> bool {
    ASSIGNMENT_IS_UNIQUE.with(|cell| cell.get())
}

std::thread_local! {
    static ASSIGNMENT_IS_UNIQUE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn successive_shortest_paths_scanned(costs: &[f64], n: usize) -> Option<(Vec<usize>, bool)> {
    solve_shortest_paths(costs, n, Some(Separation::Noise))
        .map(|(assignment, certificate)| (assignment, certificate.is_some()))
}

struct PathScratch {
    row_potential: Vec<f64>,
    column_potential: Vec<f64>,
    column_match: Vec<usize>,
    predecessor: Vec<usize>,
    min_reduced: Vec<f64>,
    used: Vec<u8>,
}

std::thread_local! {
    static PATH_SCRATCH: std::cell::RefCell<PathScratch> = const {
        std::cell::RefCell::new(PathScratch {
            row_potential: Vec::new(),
            column_potential: Vec::new(),
            column_match: Vec::new(),
            predecessor: Vec::new(),
            min_reduced: Vec::new(),
            used: Vec::new(),
        })
    };
}

fn solve_shortest_paths(
    costs: &[f64],
    n: usize,
    separation: Option<Separation>,
) -> Option<(Vec<usize>, Option<Certificate>)> {
    // Column 0 is the dummy used to grow the matching. Rows and columns of
    // `costs` are 1-indexed in the dual update so the dummy sits at index 0.
    // `used` is a byte flag, not `Vec<bool>`: the bit-packed form turns this
    // scan into a shift per column. The scratch is reused across calls on
    // this thread. One solve does not call another.
    PATH_SCRATCH.with(|cell| {
        let scratch = &mut *cell.borrow_mut();
        let width = n + 1;
        scratch.row_potential.resize(width, 0.0);
        scratch.column_potential.resize(width, 0.0);
        scratch.column_match.resize(width, 0);
        scratch.predecessor.resize(width, 0);
        scratch.min_reduced.resize(width, 0.0);
        scratch.used.resize(width, 0);
        let PathScratch {
            row_potential,
            column_potential,
            column_match,
            predecessor,
            min_reduced,
            used,
        } = scratch;
        row_potential[..width].fill(0.0);
        column_potential[..width].fill(0.0);
        column_match[..width].fill(0);
        solve_shortest_paths_in(
            costs,
            n,
            separation,
            &mut row_potential[..width],
            &mut column_potential[..width],
            &mut column_match[..width],
            &mut predecessor[..width],
            &mut min_reduced[..width],
            &mut used[..width],
        )
    })
}

fn solve_shortest_paths_in(
    costs: &[f64],
    n: usize,
    separation: Option<Separation>,
    row_potential: &mut [f64],
    column_potential: &mut [f64],
    column_match: &mut [usize],
    predecessor: &mut [usize],
    min_reduced: &mut [f64],
    used: &mut [u8],
) -> Option<(Vec<usize>, Option<Certificate>)> {
    for row in 1..=n {
        column_match[0] = row;
        let mut column = 0usize;
        min_reduced.fill(f64::INFINITY);
        used.fill(0);
        loop {
            used[column] = 1;
            let matched_row = column_match[column];
            let row_base = (matched_row - 1) * n;
            let row_dual = row_potential[matched_row];
            let (delta, next_column) = unsafe {
                scan_columns(
                    &costs[row_base..row_base + n],
                    row_dual,
                    column_potential,
                    min_reduced,
                    predecessor,
                    used,
                    column,
                )
            };
            if !delta.is_finite() {
                return None;
            }
            adjust_reduced(
                delta,
                n,
                used,
                min_reduced,
                column_potential,
                column_match,
                row_potential,
            );
            column = next_column;
            if column_match[column] == 0 {
                break;
            }
        }
        while column != 0 {
            let previous = predecessor[column];
            column_match[column] = column_match[previous];
            column = previous;
        }
    }

    let mut assignment = vec![0usize; n];
    let mut seen = vec![false; n];
    for column in 1..=n {
        let row = column_match[column];
        if row == 0 || row > n {
            return None;
        }
        let slot = row - 1;
        if seen[slot] {
            return None;
        }
        seen[slot] = true;
        assignment[slot] = column - 1;
    }
    let unique = separation.is_some_and(|separation| {
        optimum_is_unique(
            costs,
            n,
            &assignment,
            row_potential,
            column_potential,
            separation,
        )
    });
    let certificate = unique.then(|| Certificate {
        row_potential: row_potential.to_vec(),
        column_potential: column_potential.to_vec(),
    });
    Some((assignment, certificate))
}

/// Whether no alternating cycle of edges within the separation's slack
/// of tight leaves the optimum.
///
/// Any other assignment differs by such cycles, and its extra cost is the
/// sum of the reduced costs along them. A cycle with one edge above the
/// slack and every other edge within `noise` of feasible costs more than
/// the slack less `2 n noise`.
fn optimum_is_unique(
    costs: &[f64],
    n: usize,
    assignment: &[usize],
    row_potential: &[f64],
    column_potential: &[f64],
    separation: Separation,
) -> bool {
    let reduced = |row: usize, column: usize| {
        costs[row * n + column] - row_potential[row + 1] - column_potential[column + 1]
    };
    let slack = match separation {
        #[cfg(test)]
        Separation::Noise => {
            let mut scale = 1.0_f64;
            for value in costs {
                scale = scale.max(value.abs());
            }
            // A negative reduced cost of one ulp is dual noise. An edge within this
            // slack of zero can belong to another optimum; anything larger cannot.
            1.0e-8 * (1.0 + scale)
        }
        Separation::KuhnMunkres => {
            let mut noise = 0.0_f64;
            for row in 0..n {
                for column in 0..n {
                    let value = reduced(row, column);
                    noise = noise.max(if column == assignment[row] {
                        value.abs()
                    } else {
                        -value
                    });
                }
            }
            if !noise.is_finite() {
                return false;
            }
            let width = n as f64;
            2.0 * (width * kuhn_munkres_tolerance(costs) + 2.0 * width * noise)
        }
    };
    let mut matched_row = vec![0usize; n];
    for (row, &column) in assignment.iter().enumerate() {
        matched_row[column] = row;
    }
    let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); n];
    for row in 0..n {
        let matched = assignment[row];
        for column in 0..n {
            if column == matched {
                continue;
            }
            if reduced(row, column) <= slack {
                outgoing[row].push(matched_row[column]);
            }
        }
    }
    let mut color = vec![0u8; n];
    let mut stack = Vec::new();
    let mut unique = true;
    'search: for start in 0..n {
        if color[start] != 0 {
            continue;
        }
        color[start] = 1;
        stack.push((start, 0usize));
        while let Some((node, edge)) = stack.pop() {
            if edge == outgoing[node].len() {
                color[node] = 2;
                continue;
            }
            stack.push((node, edge + 1));
            let next = outgoing[node][edge];
            if color[next] == 1 {
                unique = false;
                break 'search;
            }
            if color[next] == 0 {
                color[next] = 1;
                stack.push((next, 0));
            }
        }
    }
    unique
}

/// Left-to-right minimum slack of the unseen columns of every seen row.
///
/// `delta.min(slack)` keeps the bits of a scalar scan: each slack is
/// `(row potential + column potential) - weight`, and a later equal
/// value does not replace an earlier one.
fn phase_delta(
    n: usize,
    weights: &[f64],
    row_potential: &[f64],
    column_potential: &[f64],
    rows_seen: &[bool],
    columns_seen: &[bool],
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe {
            phase_delta_avx2(
                n,
                weights,
                row_potential,
                column_potential,
                rows_seen,
                columns_seen,
            )
        };
    }
    phase_delta_scalar(
        n,
        weights,
        row_potential,
        column_potential,
        rows_seen,
        columns_seen,
    )
}

fn phase_delta_scalar(
    n: usize,
    weights: &[f64],
    row_potential: &[f64],
    column_potential: &[f64],
    rows_seen: &[bool],
    columns_seen: &[bool],
) -> f64 {
    let mut delta = f64::INFINITY;
    for row in 0..n {
        if !rows_seen[row] {
            continue;
        }
        let base = row * n;
        for column in 0..n {
            if columns_seen[column] {
                continue;
            }
            let slack = row_potential[row] + column_potential[column] - weights[base + column];
            delta = delta.min(slack);
        }
    }
    delta
}

/// # Safety
///
/// AVX2 must be enabled. `weights` holds `n * n` entries and both
/// potential slices hold `n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn phase_delta_avx2(
    n: usize,
    weights: &[f64],
    row_potential: &[f64],
    column_potential: &[f64],
    rows_seen: &[bool],
    columns_seen: &[bool],
) -> f64 {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    let mut delta = f64::INFINITY;
    let weight_ptr = weights.as_ptr();
    let column_ptr = column_potential.as_ptr();
    for row in 0..n {
        if !rows_seen[row] {
            continue;
        }
        let base = row * n;
        let dual = _mm256_set1_pd(row_potential[row]);
        let mut column = 0usize;
        while column + 4 <= n {
            let mut lanes = [0.0_f64; 4];
            unsafe {
                let weight = _mm256_loadu_pd(weight_ptr.add(base + column));
                let potential = _mm256_loadu_pd(column_ptr.add(column));
                let slack = _mm256_sub_pd(_mm256_add_pd(dual, potential), weight);
                _mm256_storeu_pd(lanes.as_mut_ptr(), slack);
            }
            for offset in 0..4 {
                if !columns_seen[column + offset] {
                    delta = delta.min(lanes[offset]);
                }
            }
            column += 4;
        }
        while column < n {
            if !columns_seen[column] {
                let slack = row_potential[row] + column_potential[column] - weights[base + column];
                delta = delta.min(slack);
            }
            column += 1;
        }
    }
    delta
}

/// Column-first Kuhn-Munkres.
/// The tight edge is the lowest column index within the scale tolerance,
/// which is the permutation the smaller Lennard-Jones walks recorded.
///
/// The numerical scheme follows the Kuhn-Munkres implementation
/// used by gpr_optim's MIT-licensed `Distance.cpp`.
#[cfg(test)]
pub(crate) fn kuhn_munkres_walk(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    kuhn_munkres(costs, n)
}

fn kuhn_munkres(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    let tight_tolerance = kuhn_munkres_tolerance(costs);
    let weights: Vec<f64> = costs.iter().map(|cost| -cost).collect();
    let mut row_potential = vec![f64::NEG_INFINITY; n];
    for row in 0..n {
        for column in 0..n {
            row_potential[row] = row_potential[row].max(weights[row * n + column]);
        }
    }
    let mut column_potential = vec![0.0_f64; n];
    let mut row_match = vec![usize::MAX; n];
    let mut column_match = vec![usize::MAX; n];
    let mut rows_seen = vec![false; n];
    let mut columns_seen = vec![false; n];

    for root in 0..n {
        let mut augmented = false;
        for _ in 0..=n {
            rows_seen.fill(false);
            columns_seen.fill(false);
            if augment_tight(
                root,
                n,
                &weights,
                &row_potential,
                &column_potential,
                tight_tolerance,
                &mut row_match,
                &mut column_match,
                &mut rows_seen,
                &mut columns_seen,
            ) {
                augmented = true;
                break;
            }

            let delta = phase_delta(
                n,
                &weights,
                &row_potential,
                &column_potential,
                &rows_seen,
                &columns_seen,
            );
            if !(delta > 0.0) || !delta.is_finite() {
                return None;
            }
            for index in 0..n {
                if rows_seen[index] {
                    row_potential[index] -= delta;
                }
                if columns_seen[index] {
                    column_potential[index] += delta;
                }
            }
        }
        if !augmented {
            return None;
        }
    }

    let mut used = vec![false; n];
    for &column in &row_match {
        if column == usize::MAX || used[column] {
            return None;
        }
        used[column] = true;
    }
    Some(row_match)
}

#[allow(clippy::too_many_arguments)]
fn augment_tight(
    row: usize,
    n: usize,
    weights: &[f64],
    row_potential: &[f64],
    column_potential: &[f64],
    tight_tolerance: f64,
    row_match: &mut [usize],
    column_match: &mut [usize],
    rows_seen: &mut [bool],
    columns_seen: &mut [bool],
) -> bool {
    rows_seen[row] = true;
    let mut column = 0usize;
    while column < n {
        // A group with no tight unseen column cannot change the first
        // column the scalar walk would enter. The slack bits match that walk.
        #[cfg(target_arch = "x86_64")]
        if column + 4 <= n
            && std::is_x86_feature_detected!("avx2")
            && unsafe {
                group_ineligible(
                    row,
                    column,
                    n,
                    weights,
                    row_potential,
                    column_potential,
                    tight_tolerance,
                    columns_seen,
                )
            }
        {
            column += 4;
            continue;
        }
        if columns_seen[column] {
            column += 1;
            continue;
        }
        let slack = row_potential[row] + column_potential[column] - weights[row * n + column];
        if slack.abs() > tight_tolerance {
            column += 1;
            continue;
        }
        columns_seen[column] = true;
        let displaced = column_match[column];
        if displaced == usize::MAX
            || augment_tight(
                displaced,
                n,
                weights,
                row_potential,
                column_potential,
                tight_tolerance,
                row_match,
                column_match,
                rows_seen,
                columns_seen,
            )
        {
            row_match[row] = column;
            column_match[column] = row;
            return true;
        }
        column += 1;
    }
    false
}

/// Whether columns `start..start+4` are all seen or strictly outside the tolerance.
///
/// # Safety
///
/// AVX2 must be enabled. `start + 4 <= n`, and the slices cover the row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn group_ineligible(
    row: usize,
    start: usize,
    n: usize,
    weights: &[f64],
    row_potential: &[f64],
    column_potential: &[f64],
    tight_tolerance: f64,
    columns_seen: &[bool],
) -> bool {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    let mut lanes = [0.0_f64; 4];
    unsafe {
        let dual = _mm256_set1_pd(row_potential[row]);
        let weight = _mm256_loadu_pd(weights.as_ptr().add(row * n + start));
        let potential = _mm256_loadu_pd(column_potential.as_ptr().add(start));
        let slack = _mm256_sub_pd(_mm256_add_pd(dual, potential), weight);
        _mm256_storeu_pd(lanes.as_mut_ptr(), slack);
    }
    lanes
        .iter()
        .enumerate()
        .all(|(offset, slack)| columns_seen[start + offset] || slack.abs() > tight_tolerance)
}

#[cfg(test)]
mod tests {
    use super::{
        certified_assignment, certifies, kuhn_munkres, last_dense_optimum_was_unique,
        minimum_cost_assignment, phase_delta_scalar, strict_unique_row_minima,
        successive_shortest_paths, successive_shortest_paths_matching,
        successive_shortest_paths_scanned,
    };
    #[cfg(target_arch = "x86_64")]
    use super::{group_ineligible, phase_delta_avx2};

    fn geometry_costs(n: usize, seed: u64, shift: usize) -> Vec<f64> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let points: Vec<[f64; 3]> = (0..n).map(|_| [next(), next(), next()]).collect();
        let mut costs = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..n {
                let image = points[(column + shift) % n];
                costs[row * n + column] = (0..3)
                    .map(|axis| (points[row][axis] - image[axis] * 0.9).powi(2))
                    .sum();
            }
        }
        costs
    }

    fn assert_walk_bits(costs: &[f64], n: usize, label: &str) {
        let walked = kuhn_munkres(costs, n).expect(label);
        let got = minimum_cost_assignment(costs, n).expect(label);
        assert_eq!(got, walked, "{label} assignment");
        assert_eq!(
            cost_of(costs, n, &got).to_bits(),
            cost_of(costs, n, &walked).to_bits(),
            "{label} cost"
        );
    }

    /// The dispatched permutation is the walk's, forced or walked, on random
    /// costs, exact ties, a flat matrix, and matrices whose optimum is forced
    /// to contain named pairs.
    #[test]
    fn a_forced_permutation_is_the_walks() {
        let mut forced = 0;
        for n in [2usize, 3, 17, 40, 80, 128] {
            for seed in 0..4u64 {
                let costs = geometry_costs(n, 0x5eed + seed, seed as usize);
                let (assignment, certificate) = certified_assignment(&costs, n).expect("square");
                let walked = kuhn_munkres(&costs, n).expect("walk");
                assert_eq!(assignment, walked, "n={n} seed={seed}");
                assert_eq!(
                    cost_of(&costs, n, &assignment).to_bits(),
                    cost_of(&costs, n, &walked).to_bits(),
                    "n={n} seed={seed} cost"
                );
                forced += usize::from(certificate.is_some());
                let ties: Vec<f64> = costs.iter().map(|cost| (cost * 4.0).floor()).collect();
                assert_walk_bits(&ties, n, &format!("ties n={n} seed={seed}"));
            }
        }
        assert!(forced > 16, "only {forced} separated optima were forced");
        let flat = vec![1.0; 25];
        let (assignment, certificate) = certified_assignment(&flat, 5).expect("flat");
        assert!(
            certificate.is_none(),
            "every permutation of a flat matrix is optimal"
        );
        assert_walk_bits(&flat, 5, "flat");
        assert_eq!(assignment, minimum_cost_assignment(&flat, 5).expect("flat"));

        // Two planted pairs sit strictly below every other entry in their
        // row and column. The free block is then either random, tied, or flat.
        for (n, pairs, fill) in [
            (6usize, vec![(0usize, 2usize), (3, 5)], "random"),
            (6, vec![(1, 1), (4, 0)], "ties"),
            (5, vec![(0, 4), (2, 1)], "flat"),
            (4, vec![(0, 1)], "duplicate"),
        ] {
            let mut state = 0x000f_0ced_u64.wrapping_add(n as u64);
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let mut costs = vec![0.0; n * n];
            for cost in &mut costs {
                *cost = match fill {
                    "ties" => (next() * 4.0).floor(),
                    "flat" => 1.0,
                    "duplicate" => next().floor(),
                    _ => next(),
                };
            }
            if fill == "duplicate" {
                for column in 0..n {
                    costs[n + column] = costs[column];
                }
            }
            for &(row, column) in &pairs {
                for other in 0..n {
                    costs[row * n + other] = 10.0;
                    costs[other * n + column] = 10.0;
                }
            }
            for &(row, column) in &pairs {
                costs[row * n + column] = 0.0;
            }
            let label = format!("forced {fill} n={n}");
            assert_walk_bits(&costs, n, &label);
            let got = minimum_cost_assignment(&costs, n).expect("forced");
            for &(row, column) in &pairs {
                assert_eq!(got[row], column, "{label} pair ({row}, {column})");
            }
        }
    }

    #[test]
    fn a_certificate_proves_the_transpose_and_not_other_costs() {
        let n = 40;
        let costs = geometry_costs(n, 0xc0de, 3);
        let (assignment, certificate) = certified_assignment(&costs, n).expect("square");
        let certificate = certificate.expect("a separated optimum");
        assert!(certifies(&costs, n, &assignment, &certificate));
        let mut transposed = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..n {
                transposed[column * n + row] = costs[row * n + column];
            }
        }
        let mut inverse = vec![0usize; n];
        for (row, &column) in assignment.iter().enumerate() {
            inverse[column] = row;
        }
        assert!(certifies(
            &transposed,
            n,
            &inverse,
            &certificate.transposed()
        ));
        assert_eq!(kuhn_munkres(&transposed, n), Some(inverse));
        let mut moved = costs.clone();
        let (first, second) = (assignment[0], assignment[1]);
        moved[first] += 1.0;
        moved[n + second] += 1.0;
        moved[second] -= 1.0;
        moved[n + first] -= 1.0;
        assert!(!certifies(&moved, n, &assignment, &certificate));
        let mut swapped = assignment.clone();
        swapped.swap(0, 1);
        assert!(!certifies(&costs, n, &swapped, &certificate));
        assert!(!certifies(&costs, n, &assignment[1..], &certificate));
    }

    #[test]
    fn a_strict_row_minimum_permutation_matches_the_solver() {
        let n = 4;
        let mut costs = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..n {
                let matched = (row + 1) % n;
                costs[row * n + column] = if column == matched {
                    0.25 * row as f64
                } else {
                    5.0 + column as f64
                };
            }
        }
        let perm = strict_unique_row_minima(&costs, n).expect("unique row minima");
        let solved = successive_shortest_paths(&costs, n).expect("solver");
        assert_eq!(perm, solved);
        assert_eq!(cost_of(&costs, n, &perm), brute(&costs, n));
    }

    #[test]
    fn a_tied_row_minimum_is_left_to_the_solver() {
        let costs = [0.0, 0.0, 1.0, 2.0];
        assert!(strict_unique_row_minima(&costs, 2).is_none());
        assert!(successive_shortest_paths(&costs, 2).is_some());
    }

    #[test]
    fn colliding_row_minima_are_left_to_the_solver() {
        let costs = [0.0, 5.0, 1.0, 4.0];
        assert!(strict_unique_row_minima(&costs, 2).is_none());
    }

    fn brute(costs: &[f64], n: usize) -> f64 {
        fn walk(costs: &[f64], n: usize, row: usize, used: &mut [bool], acc: f64, best: &mut f64) {
            if row == n {
                *best = best.min(acc);
                return;
            }
            for column in 0..n {
                if used[column] {
                    continue;
                }
                used[column] = true;
                walk(costs, n, row + 1, used, acc + costs[row * n + column], best);
                used[column] = false;
            }
        }
        let mut best = f64::INFINITY;
        walk(costs, n, 0, &mut vec![false; n], 0.0, &mut best);
        best
    }

    fn cost_of(costs: &[f64], n: usize, assignment: &[usize]) -> f64 {
        assignment
            .iter()
            .enumerate()
            .map(|(row, column)| costs[row * n + column])
            .sum()
    }

    #[test]
    fn matches_brute_force_on_small_matrices() {
        let mut state = 0x1234_5678_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as u32 % 50) as f64
        };
        for n in 1..=6 {
            for _ in 0..20 {
                let costs: Vec<f64> = (0..n * n).map(|_| next()).collect();
                let assignment = minimum_cost_assignment(&costs, n).expect("square");
                let got = cost_of(&costs, n, &assignment);
                let want = brute(&costs, n);
                assert!((got - want).abs() < 1e-8, "n={n} {got} vs {want}");
                let fast = successive_shortest_paths(&costs, n).expect("square");
                let fast_cost = cost_of(&costs, n, &fast);
                assert!(
                    (fast_cost - want).abs() < 1e-8,
                    "n={n} shortest {fast_cost}"
                );
                let unique = last_dense_optimum_was_unique();
                let slow = kuhn_munkres(&costs, n).expect("square");
                let slow_cost = cost_of(&costs, n, &slow);
                // Integer costs mod 50 produce tied optimal matchings where
                // successive shortest paths and Kuhn-Munkres break ties in a
                // different column order; their optimal costs must agree to
                // 1e-8, and when the optimum is unique the permutations must
                // also be identical.
                assert!((got - slow_cost).abs() < 1e-8, "n={n} {got} vs {slow_cost}");
                assert_eq!(assignment, slow, "n={n} dispatched tie break differed");
                if unique {
                    assert_eq!(assignment, fast, "n={n} unique optimum differed");
                }
            }
        }
    }

    /// Dual noise of one ulp is another optimum; a gap ten times larger is not.
    #[test]
    fn a_gap_inside_dual_noise_is_not_a_unique_optimum() {
        let within = [0.0, 1.0e-8, 1.0e-8, 0.0];
        let (perm, unique) = successive_shortest_paths_scanned(&within, 2).expect("within");
        assert_eq!(perm, vec![0, 1]);
        assert!(!unique, "an edge of 1e-8 sits inside the dual-noise slack");
        let still = [0.0, 2.0e-8, 2.0e-8, 0.0];
        let (perm, unique) = successive_shortest_paths_scanned(&still, 2).expect("still");
        assert_eq!(perm, vec![0, 1]);
        assert!(
            !unique,
            "an edge of 2e-8 is still inside 1e-8 * (1 + scale)"
        );
        let outside = [0.0, 1.0e-7, 1.0e-7, 0.0];
        let (perm, unique) = successive_shortest_paths_scanned(&outside, 2).expect("outside");
        assert_eq!(perm, vec![0, 1]);
        assert!(unique, "an edge of 1e-7 is a different optimum");
        let separated = [0.0, 1.0e-4, 1.0e-4, 0.0];
        assert!(
            super::forced_kuhn_munkres(&separated, 2)
                .expect("separated")
                .1
                .is_some(),
            "a gap of 1e-4 is wider than a Kuhn-Munkres tolerance"
        );
    }

    /// A certificate that is long, repeated, or out of range does not prove
    /// the assignment.
    #[test]
    fn a_certificate_rejects_a_repeated_column_and_the_wrong_length() {
        let n = 4;
        let mut costs = vec![10.0; n * n];
        for row in 0..n {
            costs[row * n + row] = 0.0;
        }
        let (assignment, certificate) = certified_assignment(&costs, n).expect("diagonal");
        let certificate = certificate.expect("the diagonal is separated");
        assert_eq!(assignment, vec![0, 1, 2, 3]);
        assert!(certifies(&costs, n, &assignment, &certificate));
        assert!(!certifies(
            &costs,
            n,
            &assignment,
            &certificate.clone().with_an_extra_column_dual()
        ));
        let mut repeated = assignment.clone();
        repeated[1] = repeated[0];
        assert!(!certifies(&costs, n, &repeated, &certificate));
        let mut out_of_range = assignment.clone();
        out_of_range[0] = n;
        assert!(
            !certifies(&costs, n, &out_of_range, &certificate),
            "a column past the matrix is not a permutation"
        );

    }

    /// Large tied matrices retain the same deterministic walk permutation.
    #[test]
    fn large_matrices_keep_the_kuhn_munkres_tie_break() {
        let mut state = 0x9e37_79b9_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as u32 % 7) as f64
        };
        for exponent in 5..=7 {
            let n = 1usize << exponent;
            for _ in 0..3 {
                let costs: Vec<f64> = (0..n * n).map(|_| next()).collect();
                let assignment = minimum_cost_assignment(&costs, n).expect("square");
                let mut seen = vec![false; n];
                for &column in &assignment {
                    assert!(!seen[column], "n={n} repeats column {column}");
                    seen[column] = true;
                }
                let slow = kuhn_munkres(&costs, n).expect("square");
                assert_eq!(assignment, slow, "n={n} tie break");
                let got = cost_of(&costs, n, &assignment);
                let want = cost_of(&costs, n, &slow);
                assert!((got - want).abs() < 1e-8, "n={n} {got} vs {want}");
            }
        }
    }

    #[test]
    fn an_empty_problem_is_an_empty_permutation() {
        assert_eq!(minimum_cost_assignment(&[], 0), Some(Vec::new()));
        assert_eq!(minimum_cost_assignment(&[0.0], 0), None);
    }

    #[test]
    fn a_bad_shape_or_a_non_finite_cost_is_refused() {
        assert_eq!(minimum_cost_assignment(&[0.0, 1.0], 2), None);
        assert_eq!(minimum_cost_assignment(&[f64::NAN], 1), None);
        assert_eq!(minimum_cost_assignment(&[f64::INFINITY], 1), None);
    }

    #[test]
    fn a_non_finite_reduced_cost_has_no_shortest_path() {
        assert!(successive_shortest_paths(&[f64::NAN], 1).is_none());
    }

    /// The byte-flag scan and the AVX2 scan have to pick the same column
    /// the original bit-vector scan picked, including on ties.
    #[test]
    fn the_faster_scan_keeps_the_original_permutation() {
        let mut state = 0x0b0b_5eed_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };
        for n in [1usize, 2, 7, 17, 32, 98, 104] {
            for trial in 0..4 {
                let costs: Vec<f64> = (0..n * n)
                    .map(|_| (next() % 5) as f64 + ((next() % 3) as f64) * 0.125)
                    .collect();
                let legacy = legacy_scan(&costs, n).expect("legacy");
                let scalar = successive_shortest_paths_scanned(&costs, n)
                    .expect("scalar")
                    .0;
                assert_eq!(scalar, legacy, "n={n} trial={trial} scalar");
                let chosen = successive_shortest_paths(&costs, n).expect("chosen");
                assert_eq!(chosen, legacy, "n={n} trial={trial} chosen");
            }
            let mut points = vec![[0.0; 3]; n];
            for point in &mut points {
                for axis in point.iter_mut() {
                    *axis = (next() % 1000) as f64 / 50.0;
                }
            }
            if n > 2 {
                points[1] = points[0];
            }
            let mut costs = vec![0.0; n * n];
            for row in 0..n {
                for column in 0..n {
                    let mut squared = 0.0;
                    for axis in 0..3 {
                        let delta = points[row][axis] - points[(column + 1) % n][axis];
                        squared += delta * delta;
                    }
                    costs[row * n + column] = squared;
                }
            }
            let legacy = legacy_scan(&costs, n).expect("legacy geometry");
            let scalar = successive_shortest_paths_scanned(&costs, n)
                .expect("scalar geometry")
                .0;
            assert_eq!(scalar, legacy, "geometry n={n}");
            let chosen = successive_shortest_paths(&costs, n).expect("chosen geometry");
            assert_eq!(chosen, legacy, "geometry chosen n={n}");
            let matching =
                successive_shortest_paths_matching(&costs, n).expect("matching geometry");
            assert_eq!(matching, legacy, "matching geometry n={n}");
        }
    }

    #[test]
    fn the_vector_slack_has_the_scalar_bits() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                return;
            }
            let mut state = 0x51ed_u64;
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                state
            };
            for n in [1usize, 3, 4, 7, 8, 55, 80] {
                let weights: Vec<f64> = (0..n * n)
                    .map(|_| ((next() % 1000) as f64 - 500.0) / 16.0)
                    .collect();
                let row_potential: Vec<f64> = (0..n)
                    .map(|_| ((next() % 200) as f64 - 100.0) / 8.0)
                    .collect();
                let column_potential: Vec<f64> = (0..n)
                    .map(|_| ((next() % 200) as f64 - 100.0) / 8.0)
                    .collect();
                let rows_seen: Vec<bool> = (0..n).map(|_| next() % 3 != 0).collect();
                let columns_seen: Vec<bool> = (0..n).map(|_| next() % 4 == 0).collect();
                let scalar = phase_delta_scalar(
                    n,
                    &weights,
                    &row_potential,
                    &column_potential,
                    &rows_seen,
                    &columns_seen,
                );
                let vector = unsafe {
                    phase_delta_avx2(
                        n,
                        &weights,
                        &row_potential,
                        &column_potential,
                        &rows_seen,
                        &columns_seen,
                    )
                };
                assert_eq!(
                    scalar.to_bits(),
                    vector.to_bits(),
                    "n={n} slack bits diverged"
                );
                let tolerance = 1.0e-9
                    * (1.0
                        + weights
                            .iter()
                            .fold(0.0_f64, |acc, value| acc.max(value.abs())));
                if n >= 4 {
                    for row in 0..n {
                        for start in (0..n - 3).step_by(4) {
                            let skipped = unsafe {
                                group_ineligible(
                                    row,
                                    start,
                                    n,
                                    &weights,
                                    &row_potential,
                                    &column_potential,
                                    tolerance,
                                    &columns_seen,
                                )
                            };
                            let scalar_skip = (0..4).all(|offset| {
                                let column = start + offset;
                                if columns_seen[column] {
                                    return true;
                                }
                                let slack = row_potential[row] + column_potential[column]
                                    - weights[row * n + column];
                                slack.abs() > tolerance
                            });
                            assert_eq!(skipped, scalar_skip, "n={n} row={row} start={start}");
                        }
                    }
                }
            }
        }
    }

    fn legacy_scan(costs: &[f64], n: usize) -> Option<Vec<usize>> {
        let mut row_potential = vec![0.0; n + 1];
        let mut column_potential = vec![0.0; n + 1];
        let mut column_match = vec![0usize; n + 1];
        let mut predecessor = vec![0usize; n + 1];
        let mut min_reduced = vec![0.0; n + 1];
        let mut used = vec![false; n + 1];
        for row in 1..=n {
            column_match[0] = row;
            let mut column = 0usize;
            min_reduced.fill(f64::INFINITY);
            used.fill(false);
            loop {
                used[column] = true;
                let matched_row = column_match[column];
                let row_base = (matched_row - 1) * n;
                let mut delta = f64::INFINITY;
                let mut next_column = 0usize;
                let row_dual = row_potential[matched_row];
                for candidate in 1..=n {
                    if used[candidate] {
                        continue;
                    }
                    let reduced =
                        costs[row_base + candidate - 1] - row_dual - column_potential[candidate];
                    if reduced < min_reduced[candidate] {
                        min_reduced[candidate] = reduced;
                        predecessor[candidate] = column;
                    }
                    if min_reduced[candidate] < delta {
                        delta = min_reduced[candidate];
                        next_column = candidate;
                    }
                }
                if !delta.is_finite() {
                    return None;
                }
                for index in 0..=n {
                    if used[index] {
                        row_potential[column_match[index]] += delta;
                        column_potential[index] -= delta;
                    } else {
                        min_reduced[index] -= delta;
                    }
                }
                column = next_column;
                if column_match[column] == 0 {
                    break;
                }
            }
            while column != 0 {
                let previous = predecessor[column];
                column_match[column] = column_match[previous];
                column = previous;
            }
        }
        let mut assignment = vec![0usize; n];
        let mut seen = vec![false; n];
        for column in 1..=n {
            let row = column_match[column];
            if row == 0 || row > n {
                return None;
            }
            let slot = row - 1;
            if seen[slot] {
                return None;
            }
            seen[slot] = true;
            assignment[slot] = column - 1;
        }
        Some(assignment)
    }
}
