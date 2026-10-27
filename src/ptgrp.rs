//! Point-group detection after GMIN `ptgrp.f`, in pieces of a few hundred
//! lines. Nothing here is called by a search.
//!
//! 1. Centre, inertia, Jacobi, and `TESTSYMOP`.
//! 2. The principal frame (`IDEGEN` at `SYMTOL3`).
//! 3. The axial names (`Cn`, `Cnv`, `Cnh`, `Dn`, `Dnh`, `Sn`, `D2d`) from the
//!    `IDEGEN = 1` block. A decahedron only reaches this block after the
//!    inertia tolerance shrinks; at `0.3` it is still classed with the cubic
//!    groups.
//! 4. The cubic search (`Oh`, `Td`, `Ih`) and its generator matrices.
//! 5. Orbits of a growing core, stopping at the first non-trivial group.
//! 6. The `symmetry.f90` move, behind `gmin_symmetry`. That is the piece that
//!    changes coordinates.

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
                // The low bit is an integer, and a clear bit has to become -1.
                // Subtracting from a `usize` zero overflows.
                let sign = 2 * (((row * col) & 1) as i32) - 1;
                matrix[row - 1][col - 1] = f64::from(sign) * sin;
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

fn apply(points: &[[f64; 3]], matrix: [[f64; 3]; 3]) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|point| {
            [0, 1, 2].map(|col| {
                point[0] * matrix[0][col] + point[1] * matrix[1][col] + point[2] * matrix[2][col]
            })
        })
        .collect()
}

fn mirror(points: &[[f64; 3]], axis: usize) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|point| {
            let mut image = *point;
            image[axis - 1] = -image[axis - 1];
            image
        })
        .collect()
}

fn holds(reference: &[[f64; 3]], image: &[[f64; 3]], mass: &[f64], tolerance: f64) -> bool {
    operation_holds(reference, image, mass, tolerance).0
}

/// Atom whose orbit has size `order`, `2 order`, or `4 order`.
///
/// Radii are ordered by `PIKSR2`. The grouping tolerance is `SYMTOL1`. A
/// trailing zero closes the last orbit, which is what the cleared scratch
/// array does in `PTGRP`.
fn reference_atom(
    points: &[[f64; 3]],
    mass: &[f64],
    order: usize,
    tolerance: f64,
) -> Option<usize> {
    let n = points.len();
    let mut radius = vec![0.0; n + 1];
    let mut sorted: Vec<usize> = (0..n).collect();
    for atom in 0..n {
        let point = points[atom];
        radius[atom] =
            mass[atom] * (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
    }
    for slot in 1..n {
        let value = radius[slot];
        let atom = sorted[slot];
        let mut placed = slot;
        while placed > 0 && radius[placed - 1] > value {
            radius[placed] = radius[placed - 1];
            sorted[placed] = sorted[placed - 1];
            placed -= 1;
        }
        radius[placed] = value;
        sorted[placed] = atom;
    }
    let mut sizes = Vec::new();
    let mut count = 1;
    for index in 1..=n {
        let gap = (radius[index] - radius[index - 1]).abs();
        if gap < tolerance {
            count += 1;
        }
        if gap > tolerance {
            sizes.push(count);
            count = 1;
        }
    }
    for multiple in [1, 2, 4] {
        let mut cursor = 0;
        for &size in &sizes {
            if size == multiple * order {
                return Some(sorted[cursor]);
            }
            cursor += size;
        }
    }
    None
}

/// Align one atom of the principal orbit with a Cartesian axis, as `PTGRP`
/// does before it looks for vertical mirrors. `axis` is 1-based.
#[allow(clippy::float_cmp)]
fn align_orbit(
    points: &[[f64; 3]],
    atom: usize,
    axis: usize,
    orbit_tolerance: f64,
) -> Vec<[f64; 3]> {
    let point = points[atom];
    let mut perpendicular = 0.0;
    for coord in 0..3 {
        if coord + 1 != axis {
            perpendicular += point[coord] * point[coord];
        }
    }
    perpendicular = perpendicular.sqrt();
    if perpendicular == 0.0 {
        perpendicular = 1.0e-3;
    }
    let toward = 2 - axis / 2;
    let mut cosine = point[toward - 1] / perpendicular;
    if cosine.abs() > 1.0 {
        cosine = cosine.signum();
    }
    let mut angle = cosine.acos().to_degrees();
    let already = (0..3).any(|turn| (angle - (turn as f64) * 90.0).abs() < orbit_tolerance);
    if already {
        return points.to_vec();
    }
    let mut aligned = apply(points, rotation_about(axis, angle));
    let other = (1..=3).find(|candidate| *candidate != axis && *candidate != toward);
    if let Some(other) = other
        && aligned[atom][other - 1].abs() > 1.0e-6
    {
        angle *= -2.0;
        aligned = apply(&aligned, rotation_about(axis, angle));
    }
    aligned
}

/// Point-group name from the `IDEGEN = 1` block.
///
/// `None` means that block found no axis, which is the accidental-degeneracy
/// path. Cubic groups never enter here. The distance tolerance is `SYMTOL2`
/// and the orbit tolerance is `SYMTOL1`.
pub fn axial_name(
    framed: &[[f64; 3]],
    mass: &[f64],
    orbit_tolerance: f64,
    distance_tolerance: f64,
) -> Option<String> {
    let mut order = 0;
    let mut axis = 0;
    for fold in 2..=6 {
        for candidate in 1..=3 {
            let turned = apply(framed, rotation_about(candidate, 360.0 / fold as f64));
            if holds(framed, &turned, mass, distance_tolerance) {
                order = fold;
                axis = candidate;
            }
        }
    }
    if order == 0 {
        return None;
    }
    let mut points = framed.to_vec();
    if order == 2 {
        axis = 0;
        for candidate in 1..=3 {
            let quarter = apply(&points, rotation_about(candidate, 90.0));
            let improper = mirror(&quarter, candidate);
            if holds(&points, &improper, mass, distance_tolerance) {
                axis = candidate;
            }
        }
        if axis == 0 {
            return None;
        }
    }
    if order % 2 == 1
        && let Some(atom) = reference_atom(&points, mass, order, orbit_tolerance)
    {
        points = align_orbit(&points, atom, axis, orbit_tolerance);
    }
    let half_turn = 180.0 / (2.0 * order as f64);
    let stepped = apply(&points, rotation_about(axis, half_turn));
    let mut on_axis = false;
    let mut off_axis = false;
    for candidate in 1..=3 {
        if candidate == axis {
            continue;
        }
        let spun = apply(&points, rotation_about(candidate, 180.0));
        on_axis |= holds(&points, &spun, mass, distance_tolerance);
        let spun = apply(&stepped, rotation_about(candidate, 180.0));
        off_axis |= holds(&stepped, &spun, mass, distance_tolerance);
    }
    let dihedral = on_axis || off_axis;
    let horizontal = mirror(&points, axis);
    let horizontal = holds(&points, &horizontal, mass, distance_tolerance);
    let mut vertical = false;
    for candidate in 1..=3 {
        if candidate == axis {
            continue;
        }
        let plane = mirror(&points, candidate);
        vertical |= holds(&points, &plane, mass, distance_tolerance);
    }
    let improper = {
        let turned = apply(&points, rotation_about(axis, 180.0 / order as f64));
        let reflected = mirror(&turned, axis);
        holds(&points, &reflected, mass, distance_tolerance)
    };
    let stem = if dihedral && horizontal {
        format!("D{order}h")
    } else if order == 2 && dihedral {
        "D2d".to_string()
    } else if horizontal {
        format!("C{order}h")
    } else if vertical {
        format!("C{order}v")
    } else if improper {
        format!("S{}", 2 * order)
    } else if dihedral {
        format!("D{order}")
    } else {
        format!("C{order}")
    };
    Some(stem)
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

    /// At `SYMTOL3 = 0.3` every one of these minima is triply degenerate, so
    /// the axial block is not entered. One shrink separates the Marks
    /// decahedron, and that block then reports the same `D5h` as GMIN.
    /// LJ38 and LJ98 stay in the cubic class.
    #[test]
    fn the_axial_block_names_the_marks_decahedron() {
        let cases = [
            (include_str!("../tests/fixtures/lj38_fcc.xyz"), 2, None),
            (
                include_str!("../tests/fixtures/lj75_marks.xyz"),
                1,
                Some("D5h"),
            ),
            (include_str!("../tests/fixtures/lj98_gm.xyz"), 2, None),
        ];
        for (text, class, name) in cases {
            let points = load(text);
            let mass = vec![1.0; points.len()];
            let (_, _, loose) = principal_frame(&points, &mass, 0.3);
            assert_eq!(loose, 2);
            let (framed, _, tight) = principal_frame(&points, &mass, 0.03);
            assert_eq!(tight, class);
            let got = (tight == 1).then(|| axial_name(&framed, &mass, 0.01, 0.05).unwrap());
            assert_eq!(got.as_deref(), name);
        }
    }
}
