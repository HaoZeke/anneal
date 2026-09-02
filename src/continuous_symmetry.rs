//! Continuous-symmetry projections for atomic-cluster exploration.
//!
//! A continuous-symmetry move applies every operation of a point group, finds
//! the minimum-distance atom permutation for each image, and averages the
//! aligned images. This module implements the inversion group `C_i`, the
//! point group used by the published LJ38 continuous-symmetry benchmark.
//! Inversion commutes with every rotation, so its orientation subproblem is
//! constant and requires no stochastic orientation search.
//!
//! The projection is defined for non-periodic Cartesian point sets. Assignment
//! is block diagonal in the supplied equivalence classes, preventing an
//! operation from exchanging different species or other inequivalent sites.

use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1};

use crate::assignment::minimum_cost_assignment;

/// Result of projecting a point set onto the inversion-group average.
#[derive(Debug, Clone, PartialEq)]
pub struct InversionProjection {
    /// Projected Cartesian coordinates in the input atom order.
    pub coordinates: Array1<f64>,
    /// Atom whose inverted image is matched to each input atom.
    pub assignment: Vec<usize>,
    /// RMS distance between the centred structure and its optimally permuted
    /// inverted image, before averaging.
    pub residual_rms: f64,
}

/// Computes the minimum-permutation inversion residual without changing `x`.
pub fn inversion_rms(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<f64> {
    match_inversion(x, classes).map(|matched| matched.residual_rms)
}

/// Projects a non-periodic Cartesian point set onto the `C_i` group average.
///
/// The point set is centred, the minimum-cost class-preserving bijection
/// `p` is found for the cost `||x_i + x_p(i)||²`, and the two group images
/// are averaged as `x̂_i = (x_i - x_p(i)) / 2`. The centroid is restored.
pub fn project_inversion(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<InversionProjection> {
    let matched = match_inversion(x, classes)?;
    let n = classes.len();
    if n == 0 {
        return Some(InversionProjection {
            coordinates: Array1::zeros(0),
            assignment: Vec::new(),
            residual_rms: 0.0,
        });
    }

    let centre = centroid(x, n);
    let mut coordinates = Array1::zeros(x.len());
    for atom in 0..n {
        let partner = matched.assignment[atom];
        for axis in 0..3 {
            let xi = x[3 * atom + axis] - centre[axis];
            let xp = x[3 * partner + axis] - centre[axis];
            coordinates[3 * atom + axis] = centre[axis] + 0.5 * (xi - xp);
        }
    }
    Some(InversionProjection {
        coordinates,
        assignment: matched.assignment,
        residual_rms: matched.residual_rms,
    })
}

struct InversionMatch {
    assignment: Vec<usize>,
    residual_rms: f64,
}

fn match_inversion(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<InversionMatch> {
    let n = classes.len();
    if x.len() != n.checked_mul(3)? || x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if n == 0 {
        return Some(InversionMatch {
            assignment: Vec::new(),
            residual_rms: 0.0,
        });
    }

    let centre = centroid(x, n);
    let mut blocks: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (atom, &class) in classes.iter().enumerate() {
        blocks.entry(class).or_default().push(atom);
    }

    let mut assignment = vec![usize::MAX; n];
    let mut squared_residual = 0.0_f64;
    for atoms in blocks.values() {
        let m = atoms.len();
        let mut costs = vec![0.0_f64; m * m];
        for (local_row, &row) in atoms.iter().enumerate() {
            for (local_column, &column) in atoms.iter().enumerate() {
                let mut squared = 0.0;
                for axis in 0..3 {
                    let sum =
                        (x[3 * row + axis] - centre[axis]) + (x[3 * column + axis] - centre[axis]);
                    squared += sum * sum;
                }
                costs[local_row * m + local_column] = squared;
            }
        }
        let local_assignment = minimum_cost_assignment(&costs, m)?;
        for (local_row, &local_column) in local_assignment.iter().enumerate() {
            let row = atoms[local_row];
            let column = atoms[local_column];
            assignment[row] = column;
            squared_residual += costs[local_row * m + local_column];
        }
    }
    if assignment.contains(&usize::MAX) {
        return None;
    }
    Some(InversionMatch {
        assignment,
        residual_rms: (squared_residual / n as f64).sqrt(),
    })
}

fn centroid(x: ArrayView1<'_, f64>, n: usize) -> [f64; 3] {
    let mut centre = [0.0_f64; 3];
    for atom in 0..n {
        for axis in 0..3 {
            centre[axis] += x[3 * atom + axis];
        }
    }
    for component in &mut centre {
        *component /= n as f64;
    }
    centre
}
