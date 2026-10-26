// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2022 Theochem @ Háskóli Íslands

//! Dense minimum-cost bipartite assignment.
//!
//! The search is the successive-shortest-path method with dual potentials
//! (Jonker and Volgenant). A continuous-symmetry projection solves one of
//! these problems for every group operation and every trial orientation, so
//! the inner scan has to be a flat loop: the recursive Kuhn-Munkres walk it
//! replaced spent its time on call overhead and was the dominant cost of an
//! energy evaluation at 98 and 150 atoms.

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

#[cfg(test)]
mod tests {
    use super::minimum_cost_assignment;

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
            }
        }
    }
}
