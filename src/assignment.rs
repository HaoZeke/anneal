//! Dense minimum-cost bipartite assignment.
//!
//! The numerical scheme follows the Kuhn--Munkres implementation used by
//! gpr_optim's MIT-licensed `Distance.cpp`. Keeping the kernel here avoids a
//! C++ boundary in the inner loop while preserving its finite-square input
//! contract and scale-aware tight-edge test.

/// Returns the column assigned to each row of a finite `n × n` cost matrix.
///
/// `costs` is row-major. The result is a permutation: every row and every
/// column occurs exactly once. Invalid shapes, non-finite entries, or failure
/// to construct a perfect matching return `None`.
pub fn minimum_cost_assignment(costs: &[f64], n: usize) -> Option<Vec<usize>> {
    if n == 0 {
        return costs.is_empty().then(Vec::new);
    }
    if n.checked_mul(n) != Some(costs.len()) || costs.iter().any(|v| !v.is_finite()) {
        return None;
    }

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
