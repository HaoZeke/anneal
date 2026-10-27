// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2022 Theochem @ Háskóli Íslands

//! Dense minimum-cost bipartite assignment.
//!
//! Below 98 atoms the permutation is the column-first Kuhn-Munkres matching
//! the measured LJ38, LJ55, and LJ75 walks were generated with. Several
//! optimal permutations can share a cost, and a different tie break moves
//! those walks. At 98 atoms and above the same optimum is computed by
//! successive shortest paths. A continuous-symmetry projection solves one
//! of these problems for every group operation and every trial orientation,
//! and the recursive Kuhn-Munkres walk was the dominant cost of an energy
//! evaluation at that size.

/// Matrix size at which the flat shortest-path scan replaces Kuhn-Munkres.
///
/// 98 is the atom count where one projection of the full point set stopped
/// being a small fraction of a quench. Callers that project a whole cluster
/// of this size use [`greedy_assignment`] instead: one dense solve is already
/// near the cubic floor, and a projection repeats it for every operation.
pub(crate) const SHORTEST_PATH_FROM: usize = 98;

/// Returns the column assigned to each row of a finite `n × n` cost matrix.
///
/// `costs` is row-major. The result is a permutation: every row and every
/// column occurs exactly once. Invalid shapes, non-finite entries, or failure
/// to construct a perfect matching return `None`.
pub fn minimum_cost_assignment(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    if n == 0 {
        return costs.is_empty().then(Vec::new);
    }
    if n.checked_mul(n) != Some(costs.len()) || costs.iter().any(|value| !value.is_finite()) {
        return None;
    }
    if n < SHORTEST_PATH_FROM {
        kuhn_munkres(costs, n)
    } else {
        successive_shortest_paths(costs, n)
    }
}

fn successive_shortest_paths(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    // Column 0 is the dummy used to grow the matching. Rows and columns of
    // `costs` are 1-indexed in the dual update so the dummy sits at index 0.
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

/// Row-greedy matching: each row takes the cheapest column still free.
///
/// A tie keeps the lowest column index. The result is always a permutation of
/// a finite square matrix, and it is not always a minimum-cost assignment.
/// Continuous-symmetry and fivefold matching use it from [`SHORTEST_PATH_FROM`]
/// atoms up, where the dense solver was the search. Smaller clusters stay on
/// [`minimum_cost_assignment`], whose tie break those walks were recorded with.
pub(crate) fn greedy_assignment(costs: &[f64], n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut used = vec![false; n];
    let mut assignment = vec![0usize; n];
    for row in 0..n {
        let base = row * n;
        let mut best_column = 0usize;
        let mut best_cost = f64::INFINITY;
        let mut found = false;
        for column in 0..n {
            if used[column] {
                continue;
            }
            let cost = costs[base + column];
            if !found || cost < best_cost {
                best_cost = cost;
                best_column = column;
                found = true;
            }
        }
        used[best_column] = true;
        assignment[row] = best_column;
    }
    assignment
}

/// Column-first Kuhn-Munkres. The tight edge is the lowest column index
/// within the scale tolerance, which is the permutation the smaller
/// Lennard-Jones walks recorded.
fn kuhn_munkres(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    let scale = costs
        .iter()
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    let tight_tolerance = scale * 1e-9;
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

    for root in 0..n {
        let mut augmented = false;
        for _ in 0..=n {
            let mut rows_seen = vec![false; n];
            let mut columns_seen = vec![false; n];
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

            let mut delta = f64::INFINITY;
            for row in 0..n {
                if !rows_seen[row] {
                    continue;
                }
                for column in 0..n {
                    if columns_seen[column] {
                        continue;
                    }
                    let slack =
                        row_potential[row] + column_potential[column] - weights[row * n + column];
                    delta = delta.min(slack);
                }
            }
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
    for column in 0..n {
        if columns_seen[column] {
            continue;
        }
        let slack = row_potential[row] + column_potential[column] - weights[row * n + column];
        if slack.abs() > tight_tolerance {
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
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{
        greedy_assignment, kuhn_munkres, minimum_cost_assignment, successive_shortest_paths,
    };

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
                let slow = kuhn_munkres(&costs, n).expect("square");
                assert_eq!(assignment, slow);
            }
        }
    }

    #[test]
    fn greedy_matching_is_a_permutation_and_not_always_optimal() {
        let costs = [1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 100.0];
        let greedy = greedy_assignment(&costs, 3);
        let mut seen = [false; 3];
        for column in &greedy {
            assert!(
                *column < 3 && !seen[*column],
                "not a permutation {greedy:?}"
            );
            seen[*column] = true;
        }
        let optimal = minimum_cost_assignment(&costs, 3).expect("square");
        assert!(
            cost_of(&costs, 3, &greedy) > cost_of(&costs, 3, &optimal) + 1.0,
            "this matrix is the witness that the row scan is not the dense solver"
        );
        let diagonal: Vec<f64> = (0..98 * 98)
            .map(|index| {
                let row = index / 98;
                let column = index % 98;
                if row == column { 0.0 } else { 1.0 }
            })
            .collect();
        assert_eq!(
            greedy_assignment(&diagonal, 98),
            (0..98).collect::<Vec<_>>()
        );
    }
}
