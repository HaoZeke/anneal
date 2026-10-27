//! Point-group detection after GMIN `ptgrp.f`, in pieces of a few hundred
//! lines. This piece is the centre of mass, the inertia tensor, Jacobi
//! eigenvectors, and `TESTSYMOP`. Later pieces name the axial group, then the
//! cubic group, then grow orbits, then make the symmetry move. Nothing here
//! is called by a search.

/// Geometric centre, then coordinates relative to it.
pub fn centre(points: &[[f64; 3]]) -> (Vec<[f64; 3]>, [f64; 3]) {
    let n = points.len() as f64;
    let mut origin = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            origin[axis] += point[axis];
        }
    }
    for axis in origin.iter_mut() {
        *axis /= n;
    }
    let shifted = points
        .iter()
        .map(|point| {
            [
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            ]
        })
        .collect();
    (shifted, origin)
}

/// Inertia tensor about the origin. `mass` is one entry per point.
///
/// This is `INERTIA` with equal masses: the diagonal is `m (r² − xᵢ²)` and
/// the off-diagonal is `−m xᵢ xⱼ`.
pub fn inertia(points: &[[f64; 3]], mass: &[f64]) -> [[f64; 3]; 3] {
    let mut tensor = [[0.0; 3]; 3];
    for (point, &weight) in points.iter().zip(mass.iter()) {
        let radius_sq = point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
        for row in 0..3 {
            tensor[row][row] += weight * (radius_sq - point[row] * point[row]);
            for col in 0..3 {
                if row != col {
                    tensor[row][col] -= weight * point[row] * point[col];
                }
            }
        }
    }
    tensor
}

/// Jacobi diagonalisation of a symmetric 3×3 matrix, matching `EIG` with
/// `N1 = 0`.
///
/// Eigenvalues come back ascending. Columns of the second matrix are the
/// eigenvectors, including the swaps `EIG` makes when two values compare equal.
#[allow(clippy::float_cmp, clippy::manual_swap)]
pub fn jacobi(matrix: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = [[0.0; 4]; 4];
    let mut b = [[0.0; 4]; 4];
    for row in 1..=3 {
        for col in 1..=3 {
            a[row][col] = matrix[row - 1][col - 1];
        }
    }
    b[1][1] = 1.0;
    for i in 2..=3 {
        for j in 1..i {
            b[i][j] = 0.0;
            b[j][i] = 0.0;
        }
        b[i][i] = 1.0;
    }
    let mut off = 0;
    for ii in 1..=3 {
        for ji in (ii + 1)..=3 {
            if a[ii][ji].abs() > 1e-10 {
                off += 1;
            }
            if a[ji][ii].abs() > 1e-10 {
                off += 1;
            }
        }
    }
    if off == 0 {
        for i in 1..=3 {
            for j in 1..=3 {
                b[i][j] = 0.0;
            }
            b[i][i] = 1.0;
        }
    } else {
        loop {
            let mut pivot = 0.0;
            let mut ii = 1;
            let mut jj = 1;
            for i in 2..=3 {
                for j in 1..i {
                    let entry = a[i][j];
                    if pivot >= entry.abs() {
                        continue;
                    }
                    pivot = entry.abs();
                    ii = i;
                    jj = j;
                }
            }
            if pivot == 0.0 {
                break;
            }
            let diag_i = a[ii][ii];
            let couple = a[ii][jj];
            let diag_j = a[jj][jj];
            let diff = 0.5 * (diag_i - diag_j);
            if diff.abs() >= couple.abs() && (couple / diff).abs() <= 1e-14 {
                a[ii][jj] = 0.0;
                a[jj][ii] = 0.0;
                continue;
            }
            let root = (0.25 * (diag_i - diag_j).powi(2) + couple * couple).sqrt();
            let sum = 0.5 * (diag_i + diag_j);
            let det = diag_j * diag_i - couple * couple;
            let (low, high) = if sum > 0.0 {
                let high = sum + root;
                (det / high, high)
            } else {
                let low = sum - root;
                (low, det / low)
            };
            let turn = if diff > 0.0 {
                a[ii][ii] = high;
                a[jj][jj] = low;
                couple / (diff + root)
            } else {
                a[ii][ii] = low;
                a[jj][jj] = high;
                couple / (diff - root)
            };
            let cosine = 1.0 / (1.0 + turn * turn).sqrt();
            let sine = turn * cosine;
            a[ii][jj] = 0.0;
            a[jj][ii] = 0.0;
            for i in 1..=3 {
                let left = b[i][ii];
                let right = b[i][jj];
                b[i][ii] = cosine * left + sine * right;
                b[i][jj] = cosine * right - sine * left;
                if i == ii || i == jj {
                    continue;
                }
                let row_i = a[i][ii];
                let row_j = a[i][jj];
                let mixed_i = cosine * row_i + sine * row_j;
                a[i][ii] = mixed_i;
                a[ii][i] = mixed_i;
                let mixed_j = row_j * cosine - row_i * sine;
                a[i][jj] = mixed_j;
                a[jj][i] = mixed_j;
            }
        }
    }
    for i in 1..=2 {
        for j in (i + 1)..=3 {
            if a[i][i] - a[j][j] < 0.0 {
                continue;
            }
            let value = a[i][i];
            a[i][i] = a[j][j];
            a[j][j] = value;
            for row in 1..=3 {
                let component = b[row][i];
                b[row][i] = b[row][j];
                b[row][j] = component;
            }
        }
    }
    let values = [a[1][1], a[2][2], a[3][3]];
    let mut vectors = [[0.0; 3]; 3];
    for col in 0..3 {
        for row in 0..3 {
            vectors[row][col] = b[row + 1][col + 1];
        }
    }
    (values, vectors)
}

/// Whether `image` is the same labelled cloud as `points` within `tolerance`.
///
/// This is `TESTSYMOP`. The partner of each atom is the first later index
/// inside the tolerance, not the nearest, and atoms of unequal mass are not
/// partners. On a miss the distance is `1000`, which is the sentinel the
/// Fortran routine writes.
#[allow(clippy::float_cmp)]
pub fn operation_holds(
    points: &[[f64; 3]],
    image: &[[f64; 3]],
    mass: &[f64],
    tolerance: f64,
) -> (bool, f64, f64) {
    let n = points.len();
    let limit = tolerance * tolerance;
    let mut taken = vec![false; n];
    let mut worst_sq = -1.0;
    let mut worst_atom = 0;
    for atom in 0..n {
        let mut found = false;
        for image_atom in 0..n {
            if taken[image_atom] || mass[image_atom] != mass[atom] {
                continue;
            }
            let dx = points[atom][0] - image[image_atom][0];
            let dy = points[atom][1] - image[image_atom][1];
            let dz = points[atom][2] - image[image_atom][2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < limit {
                found = true;
                taken[image_atom] = true;
                if dist_sq > worst_sq {
                    worst_sq = dist_sq;
                    worst_atom = atom;
                }
                break;
            }
        }
        if !found {
            return (false, 1000.0, 0.0);
        }
    }
    let point = points[worst_atom];
    let radius = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
    (true, worst_sq.sqrt(), radius)
}
