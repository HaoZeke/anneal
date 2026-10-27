//! Point-group detection after GMIN `ptgrp.f`, in pieces of a few hundred
//! lines. The pieces so far centre a cluster, diagonalise its inertia tensor,
//! test a symmetry operation, and place the cloud on the principal axes the
//! way `PTGRP` does before it names the group. Later pieces name the group,
//! grow orbits, and make the symmetry move. Nothing here is called by a search.

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

/// Rotation matrix from `ROTM` with the angle in degrees.
///
/// `axis` is 1, 2, or 3. The integer bit tests are the Fortran `IAND` checks
/// that put a 1 on the rotation axis and a sine on the other two.
fn rotation_about(axis: usize, degrees: f64) -> [[f64; 3]; 3] {
    let turn = degrees.to_radians();
    let (sin, cos) = turn.sin_cos();
    let mut matrix = [[0.0; 3]; 3];
    for index in 1..=3 {
        matrix[index - 1][index - 1] = cos;
        matrix[axis - 1][index - 1] = ((axis & index) / axis.max(index)) as f64;
    }
    for row in 1..=3 {
        for col in 1..=row {
            if row != col && row != axis && col != axis {
                let sign = 2 * ((row * col) & 1) - 1;
                matrix[row - 1][col - 1] = sign as f64 * sin;
                matrix[col - 1][row - 1] = -matrix[row - 1][col - 1];
            }
        }
    }
    matrix
}

fn multiply(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = (0..3).map(|k| left[row][k] * right[k][col]).sum();
        }
    }
    out
}

fn cross_normalized(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    let raw = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ];
    let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
    if norm < 1e-14 {
        return raw;
    }
    [raw[0] / norm, raw[1] / norm, raw[2] / norm]
}

/// Principal-axis frame used before `PTGRP` names a group.
///
/// `tolerance` is `SYMTOL3`. When the two larger moments agree within that
/// relative tolerance the unique axis is turned onto z. A left-handed frame
/// flips the y coordinate. The returned count is `IDEGEN`.
pub fn principal_frame(
    points: &[[f64; 3]],
    mass: &[f64],
    tolerance: f64,
) -> (Vec<[f64; 3]>, [f64; 3], usize) {
    let (shifted, _) = centre(points);
    let (values, mut vectors) = jacobi(inertia(&shifted, mass));
    let mut moments = values;
    let sum = moments[0] + moments[1] + moments[2];
    if ((moments[1] - moments[2]) / sum).abs() < tolerance {
        vectors = multiply(vectors, rotation_about(2, 90.0));
        moments.swap(0, 2);
    }
    let mut framed: Vec<[f64; 3]> = shifted
        .iter()
        .map(|point| {
            [
                point[0] * vectors[0][0] + point[1] * vectors[1][0] + point[2] * vectors[2][0],
                point[0] * vectors[0][1] + point[1] * vectors[1][1] + point[2] * vectors[2][1],
                point[0] * vectors[0][2] + point[1] * vectors[1][2] + point[2] * vectors[2][2],
            ]
        })
        .collect();
    let column = |which: usize| [vectors[0][which], vectors[1][which], vectors[2][which]];
    let handed = cross_normalized(column(0), column(1))
        .iter()
        .zip(column(2))
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if handed < 0.0 {
        for point in &mut framed {
            point[1] = -point[1];
        }
    }
    let mut previous = -1.0;
    let mut degenerate = 0;
    for moment in moments {
        if ((moment - previous) / sum).abs() < tolerance {
            degenerate += 1;
        }
        previous = moment;
    }
    (framed, moments, degenerate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(text: &str) -> Vec<[f64; 3]> {
        let mut lines = text.lines();
        let n: usize = lines.next().unwrap().trim().parse().unwrap();
        let _ = lines.next();
        lines
            .take(n)
            .map(|line| {
                let mut parts = line.split_whitespace().skip(1);
                [(); 3].map(|_| parts.next().unwrap().parse().unwrap())
            })
            .collect()
    }

    fn frame(text: &str) -> (Vec<[f64; 3]>, [f64; 3], [[f64; 3]; 3]) {
        let (shifted, _) = centre(&load(text));
        let mass = vec![1.0; shifted.len()];
        let (values, vectors) = jacobi(inertia(&shifted, &mass));
        (shifted, values, vectors)
    }

    fn close(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-8, "{got} against {want}");
    }

    /// GMIN `ptgrp.o` on the published minima, equal masses, `SYMTOL2 = 0.05`.
    /// LJ38 is `O h`, LJ75 Marks is `D5h`, LJ98 is `T d`. Columns of `vectors`
    /// are the eigenvectors.
    #[test]
    fn inertia_and_the_inversion_test_match_gmin() {
        #[rustfmt::skip]
        let cases: [(&str, [f64; 3], [[f64; 3]; 3], bool, f64, f64); 3] = [
            (include_str!("../tests/fixtures/lj38_fcc.xyz"),
             [61.0911772406, 61.0912485883, 61.0913093029],
             [[-0.5509065733, 0.8344584834, -0.0134531356],
              [0.8216634235, 0.5394946820, -0.1838877554],
              [0.1461888024, 0.1123589226, 0.9828551809]],
             true, 0.0000282100, 1.7495300257),
            (include_str!("../tests/fixtures/lj75_marks.xyz"),
             [183.2544230887, 183.2545781902, 219.1336141258],
             [[-0.1204782750, -0.2447361176, 0.9620754742],
              [0.9840120886, 0.0986114385, 0.1483104636],
              [-0.1311685735, 0.9645620856, 0.2289427623]],
             false, 1000.0, 2.2413150452),
            (include_str!("../tests/fixtures/lj98_gm.xyz"),
             [302.6729069658, 302.6729069662, 302.6729069670],
             [[0.8763100952, 0.3711490425, -0.3071302743],
              [-0.1039943591, 0.7682416361, 0.6316565221],
              [-0.4703889777, 0.5215871709, -0.7118153080]],
             false, 1000.0, 0.6654380894),
        ];
        for (text, values, vectors, inverts, invert_distance, identity_radius) in cases {
            let (shifted, got_values, got_vectors) = frame(text);
            for axis in 0..3 {
                close(got_values[axis], values[axis]);
                for row in 0..3 {
                    close(got_vectors[row][axis], vectors[row][axis]);
                }
            }
            let mass = vec![1.0; shifted.len()];
            let (holds, distance, radius) = operation_holds(&shifted, &shifted, &mass, 0.05);
            assert!(holds);
            close(distance, 0.0);
            close(radius, identity_radius);
            let inverted: Vec<[f64; 3]> = shifted.iter().map(|p| [-p[0], -p[1], -p[2]]).collect();
            let (holds, distance, _) = operation_holds(&shifted, &inverted, &mass, 0.05);
            assert_eq!(holds, inverts);
            close(distance, invert_distance);
        }
    }

    /// GMIN's frame before it names the group. All three minima are triply
    /// degenerate at `SYMTOL3 = 0.3`, which is why a decahedron still enters
    /// the cubic branch.
    #[test]
    fn the_principal_frame_matches_gmin() {
        #[rustfmt::skip]
        let cases: [(&str, [f64; 3], [f64; 3], usize); 3] = [
            (include_str!("../tests/fixtures/lj38_fcc.xyz"),
             [1.61424173, -0.53269125, -0.41390697],
             [61.09130930, 61.09124859, 61.09117724], 2),
            (include_str!("../tests/fixtures/lj75_marks.xyz"),
             [2.24131505, -0.00000620, -0.00000625],
             [219.13361413, 183.25457819, 183.25442309], 2),
            (include_str!("../tests/fixtures/lj98_gm.xyz"),
             [0.06972369, -0.61779616, 0.23722219],
             [302.67290697, 302.67290697, 302.67290697], 2),
        ];
        for (text, first, moments, degenerate) in cases {
            let points = load(text);
            let mass = vec![1.0; points.len()];
            let (framed, got_moments, got_degenerate) = principal_frame(&points, &mass, 0.3);
            for axis in 0..3 {
                close(framed[0][axis], first[axis]);
                close(got_moments[axis], moments[axis]);
            }
            assert_eq!(got_degenerate, degenerate);
        }
    }
}
