//! Continuous symmetry measure for a free Cartesian point set.
//!
//! Oakley, Johnston and Wales, Phys. Chem. Chem. Phys. 15, 3965 (2013),
//! Section 3.3, and GMIN `SYMMETRYCSM`. The structure is centred and scaled
//! so its largest radius is 1. For each operation, a class-blocked assignment
//! matches the operated image to the original, and the proposed geometry is
//! the mean of those operated, permuted images. [`propose`] takes its
//! orientations from the inertia tensor plus one fixed one-fifth turn, and
//! draws no random number.

use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1};

use crate::assignment::minimum_cost_assignment;

/// Projects `x` onto `ops` at the best of `orientations`.
///
/// `ops` and `orientations` are row-major orthogonal 3 by 3 matrices.
/// `classes` blocks the assignment, one label per atom. The measure is the
/// normalised squared distance from the oriented structure to its average.
/// Zero means the orientation already has the group.
pub fn project(
    x: ArrayView1<'_, f64>,
    classes: &[u32],
    ops: &[[[f64; 3]; 3]],
    orientations: &[[[f64; 3]; 3]],
) -> Option<(Array1<f64>, f64)> {
    let n = classes.len();
    if ops.is_empty() || orientations.is_empty() || x.len() != n.checked_mul(3)? {
        return None;
    }
    if x.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (centred, centroid, scale) = centre_and_scale(x, n)?;
    let mut best: Option<(f64, [[f64; 3]; 3], Vec<[f64; 3]>)> = None;
    for rotation in orientations {
        let y = apply_all(rotation, &centred);
        let mut hat = vec![[0.0; 3]; n];
        for op in ops {
            let assignment = match_operated(&y, classes, op)?;
            for atom in 0..n {
                let image = mul_vec(op, y[assignment[atom]]);
                for axis in 0..3 {
                    hat[atom][axis] += image[axis];
                }
            }
        }
        let order = ops.len() as f64;
        let mut residual = 0.0;
        let mut norm = 0.0;
        for atom in 0..n {
            for axis in 0..3 {
                hat[atom][axis] /= order;
                let delta = y[atom][axis] - hat[atom][axis];
                residual += delta * delta;
                norm += y[atom][axis] * y[atom][axis];
            }
        }
        if norm < 1e-18 {
            continue;
        }
        let measure = residual / norm;
        if best
            .as_ref()
            .is_none_or(|(incumbent, _, _)| measure < *incumbent)
        {
            best = Some((measure, *rotation, hat));
        }
    }
    let (measure, rotation, hat) = best?;
    let back = transpose(rotation);
    let mut coordinates = Array1::zeros(3 * n);
    for atom in 0..n {
        let restored = mul_vec(&back, hat[atom]);
        for axis in 0..3 {
            coordinates[3 * atom + axis] = centroid[axis] + restored[axis] * scale;
        }
    }
    Some((coordinates, measure))
}

/// Pair distance below which an average is a coordinate singularity.
pub const SINGULAR_PAIR: f64 = 1.0e-2;

/// Lowest-measure average among `C_i`, `C_5`, and `D_5h`.
///
/// `C_i` is the group of the published LJ38 continuous-symmetry benchmark.
/// `C_5` and `D_5h` are the fivefold groups of the decahedral minima. The
/// measure chooses, not the atom count. An average with a pair closer than
/// [`SINGULAR_PAIR`] is dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupProjection {
    /// Cartesian coordinates in the input atom order.
    pub coordinates: Array1<f64>,
    /// Normalised squared distance to the average. Zero is exact symmetry.
    pub measure: f64,
    /// `C_i`, `C_5`, or `D_5h`.
    pub group: &'static str,
}

/// Projects onto the library group with the lowest measure.
pub fn propose(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<GroupProjection> {
    if let Some(cached) = cached_propose(x, classes) {
        return cached;
    }
    let orientations = inertia_orientations(x, classes.len())?;
    // D_5h contains every C_5 turn, so it is solved first and those
    // permutations are reused. Equal measures keep the earlier group
    // in the published order C_i, C_5, D_5h.
    clear_op_cache();
    let mut best: Option<(usize, GroupProjection)> = None;
    for (rank, name, ops) in [
        (2, "D_5h", d5h_ops()),
        (1, "C_5", cyclic_about_z(5)),
        (0, "C_i", inversion_ops()),
    ] {
        // Inversion is not a fivefold operation, so its permutations are
        // not in the D_5h cache. The nearest-image residual is a lower
        // bound on the assigned measure: the average of a point and its
        // matched inverse sits halfway between them. A margin keeps a
        // one-ulp overestimate from skipping a tie, which would prefer
        // inversion. Above the margin, inversion cannot win the group.
        if name == "C_i"
            && let Some((_, incumbent)) = &best
            && let Some((centred, _, _)) = centre_and_scale(x, classes.len())
        {
            let bound = inversion_lower_measure(&centred, &orientations);
            let margin = 1.0e-6 * (1.0 + incumbent.measure);
            if bound > incumbent.measure + margin {
                continue;
            }
        }
        let Some((coordinates, measure)) = project(x, classes, &ops, &orientations) else {
            continue;
        };
        if closest_pair(&coordinates) < SINGULAR_PAIR {
            continue;
        }
        let replace = match &best {
            None => true,
            Some((incumbent_rank, incumbent)) => {
                measure < incumbent.measure
                    || (measure == incumbent.measure && rank < *incumbent_rank)
            }
        };
        if replace {
            best = Some((
                rank,
                GroupProjection {
                    coordinates,
                    measure,
                    group: name,
                },
            ));
        }
    }
    let best = best.map(|(_, projection)| projection);
    store_propose(x, classes, best.as_ref());
    best
}

struct ProposeCache {
    coordinates: Vec<u64>,
    classes: Vec<u32>,
    value: Option<GroupProjection>,
}

std::thread_local! {
    static PROPOSE_CACHE: std::cell::RefCell<Option<ProposeCache>> = const { std::cell::RefCell::new(None) };
}

fn same_bits(cached: &[u64], x: ArrayView1<'_, f64>) -> bool {
    cached.len() == x.len()
        && cached
            .iter()
            .zip(x.iter())
            .all(|(bits, value)| *bits == value.to_bits())
}

fn cached_propose(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<Option<GroupProjection>> {
    PROPOSE_CACHE.with(|slot| {
        let slot = slot.borrow();
        let cached = slot.as_ref()?;
        (same_bits(&cached.coordinates, x) && cached.classes == classes)
            .then(|| cached.value.clone())
    })
}

fn store_propose(x: ArrayView1<'_, f64>, classes: &[u32], value: Option<&GroupProjection>) {
    PROPOSE_CACHE.with(|slot| {
        *slot.borrow_mut() = Some(ProposeCache {
            coordinates: x.iter().map(|value| value.to_bits()).collect(),
            classes: classes.to_vec(),
            value: value.cloned(),
        });
    });
}

pub(crate) fn inversion_ops() -> Vec<[[f64; 3]; 3]> {
    vec![
        identity(),
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    ]
}

fn centre_and_scale(x: ArrayView1<'_, f64>, n: usize) -> Option<(Vec<[f64; 3]>, [f64; 3], f64)> {
    if n == 0 || x.len() != 3 * n {
        return None;
    }
    let mut centroid = [0.0; 3];
    for atom in 0..n {
        for axis in 0..3 {
            centroid[axis] += x[3 * atom + axis];
        }
    }
    for component in &mut centroid {
        *component /= n as f64;
    }
    let mut centred = vec![[0.0; 3]; n];
    let mut max_r = 0.0_f64;
    for atom in 0..n {
        let mut radius_sq = 0.0;
        for axis in 0..3 {
            centred[atom][axis] = x[3 * atom + axis] - centroid[axis];
            radius_sq += centred[atom][axis] * centred[atom][axis];
        }
        max_r = max_r.max(radius_sq.sqrt());
    }
    if max_r < 1e-12 {
        return None;
    }
    for point in &mut centred {
        for component in point.iter_mut() {
            *component /= max_r;
        }
    }
    Some((centred, centroid, max_r))
}

fn is_identity(op: &[[f64; 3]; 3]) -> bool {
    for row in 0..3 {
        for col in 0..3 {
            let want = if row == col { 1.0 } else { 0.0 };
            if (op[row][col] - want).abs() > 1e-12 {
                return false;
            }
        }
    }
    true
}

struct SolvedOp {
    y: Vec<u64>,
    classes: Vec<u32>,
    op: [[f64; 3]; 3],
    perm: Vec<usize>,
    unique: bool,
}

std::thread_local! {
    static OP_CACHE: std::cell::RefCell<Vec<SolvedOp>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn clear_op_cache() {
    OP_CACHE.with(|slot| slot.borrow_mut().clear());
}

fn coordinate_bits(y: &[[f64; 3]]) -> Vec<u64> {
    y.iter().flatten().map(|value| value.to_bits()).collect()
}

fn same_matrix(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> bool {
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn matrix_close(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> bool {
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .all(|(left, right)| (*left - *right).abs() <= 1.0e-8)
}

fn invert_perm(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (row, &column) in perm.iter().enumerate() {
        inverse[column] = row;
    }
    inverse
}

fn recall_op(y: &[[f64; 3]], classes: &[u32], op: &[[f64; 3]; 3]) -> Option<Vec<usize>> {
    let bits = coordinate_bits(y);
    let transposed = transpose(*op);
    OP_CACHE.with(|slot| {
        let slot = slot.borrow();
        for saved in slot.iter().rev() {
            if saved.y != bits || saved.classes != classes {
                continue;
            }
            if same_matrix(&saved.op, op) {
                return Some(saved.perm.clone());
            }
            if saved.unique && matrix_close(&saved.op, &transposed) {
                return Some(invert_perm(&saved.perm));
            }
        }
        None
    })
}

fn remember_op(y: &[[f64; 3]], classes: &[u32], op: &[[f64; 3]; 3], perm: &[usize], unique: bool) {
    OP_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.len() > 800 {
            slot.clear();
        }
        slot.push(SolvedOp {
            y: coordinate_bits(y),
            classes: classes.to_vec(),
            op: *op,
            perm: perm.to_vec(),
            unique,
        });
    });
}

fn match_operated(y: &[[f64; 3]], classes: &[u32], op: &[[f64; 3]; 3]) -> Option<Vec<usize>> {
    let n = classes.len();
    // The identity sends every atom to itself, so the diagonal is a minimum
    // of cost zero and the assignment problem is unnecessary.
    if is_identity(op) {
        return Some((0..n).collect());
    }
    if let Some(saved) = recall_op(y, classes, op) {
        return Some(saved);
    }
    let images: Vec<[f64; 3]> = y.iter().map(|point| mul_vec(op, *point)).collect();
    let mut blocks: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (atom, &class) in classes.iter().enumerate() {
        blocks.entry(class).or_default().push(atom);
    }
    let mut assignment = vec![usize::MAX; n];
    let mut unique = true;
    for atoms in blocks.values() {
        let width = atoms.len();
        let mut costs = vec![0.0; width * width];
        for (row_i, &row) in atoms.iter().enumerate() {
            for (col_i, &col) in atoms.iter().enumerate() {
                let mut squared = 0.0;
                for axis in 0..3 {
                    let delta = y[row][axis] - images[col][axis];
                    squared += delta * delta;
                }
                costs[row_i * width + col_i] = squared;
            }
        }
        let local = minimum_cost_assignment(&costs, width)?;
        // Below 98 atoms the recorded permutation is Kuhn-Munkres. Its
        // tie break is not the inverse of the transposed problem, so an
        // inverse is reused only for a unique dense optimum.
        unique &= width >= 98 && crate::assignment::last_dense_optimum_was_unique();
        for (row_i, &col_i) in local.iter().enumerate() {
            assignment[atoms[row_i]] = atoms[col_i];
        }
    }
    if assignment.contains(&usize::MAX) {
        return None;
    }
    remember_op(y, classes, op, &assignment, unique);
    Some(assignment)
}

pub(crate) fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn mul_vec(matrix: &[[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn apply_all(matrix: &[[f64; 3]; 3], points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    points.iter().map(|point| mul_vec(matrix, *point)).collect()
}

/// Lower bound on the inversion measure in every orientation.
///
/// The group is the identity and the central inversion. The residual is
/// half the matched distance, so the measure is the assignment cost
/// divided by four times the squared norm. Each atom's nearest inverse
/// image is at most that assignment cost, whether or not the nearest
/// images form a bijection.
fn inversion_lower_measure(centred: &[[f64; 3]], orientations: &[[[f64; 3]; 3]]) -> f64 {
    let mut best = f64::INFINITY;
    for rotation in orientations {
        let y = apply_all(rotation, centred);
        let n = y.len();
        let mut nearest = 0.0;
        let mut norm = 0.0;
        for atom in 0..n {
            let mut closest = f64::INFINITY;
            for image in 0..n {
                let mut squared = 0.0;
                for axis in 0..3 {
                    let delta = y[atom][axis] + y[image][axis];
                    squared += delta * delta;
                }
                closest = closest.min(squared);
            }
            nearest += closest;
            for axis in 0..3 {
                norm += y[atom][axis] * y[atom][axis];
            }
        }
        if norm < 1e-18 {
            continue;
        }
        best = best.min(nearest / (4.0 * norm));
    }
    best
}

fn cyclic_about_z(order: usize) -> Vec<[[f64; 3]; 3]> {
    (0..order)
        .map(|step| rot_z(std::f64::consts::TAU * step as f64 / order as f64))
        .collect()
}

fn d5h_ops() -> Vec<[[f64; 3]; 3]> {
    let turns = cyclic_about_z(5);
    let twofold = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    let mirror = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
    let mut ops = Vec::with_capacity(20);
    for turn in &turns {
        ops.push(*turn);
        ops.push(matmul(twofold, *turn));
    }
    let rotations = ops.clone();
    for rotation in rotations {
        ops.push(matmul(mirror, rotation));
    }
    ops
}

fn inertia_orientations(x: ArrayView1<'_, f64>, n: usize) -> Option<Vec<[[f64; 3]; 3]>> {
    let (centred, _, _) = centre_and_scale(x, n)?;
    let axes = jacobi(inertia(&centred));
    let mut orientations = Vec::with_capacity(6);
    for axis in 0..3 {
        let z_axis = normalize(column(axes, axis))?;
        let mut x_axis = reject(column(axes, (axis + 1) % 3), z_axis);
        if norm(x_axis) < 1e-8 {
            x_axis = perpendicular(z_axis);
        }
        let x_axis = normalize(x_axis)?;
        let y_axis = normalize(cross(z_axis, x_axis))?;
        let aligned = [x_axis, y_axis, z_axis];
        orientations.push(aligned);
        orientations.push(matmul(rot_z(std::f64::consts::TAU / 5.0), aligned));
    }
    Some(orientations)
}

fn closest_pair(x: &Array1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut best = f64::INFINITY;
    for left in 0..n {
        for right in (left + 1)..n {
            let mut squared = 0.0;
            for axis in 0..3 {
                let delta = x[3 * left + axis] - x[3 * right + axis];
                squared += delta * delta;
            }
            best = best.min(squared);
        }
    }
    best.sqrt()
}

fn inertia(centred: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mut tensor = [[0.0; 3]; 3];
    for point in centred {
        let radius_sq = point.iter().map(|value| value * value).sum::<f64>();
        for row in 0..3 {
            for col in 0..3 {
                let diagonal = if row == col { radius_sq } else { 0.0 };
                tensor[row][col] += diagonal - point[row] * point[col];
            }
        }
    }
    tensor
}

fn jacobi(mut tensor: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut vectors = identity();
    for _ in 0..16 {
        let mut row = 0;
        let mut col = 1;
        let mut largest = tensor[0][1].abs();
        for i in 0..3 {
            for j in (i + 1)..3 {
                if tensor[i][j].abs() > largest {
                    largest = tensor[i][j].abs();
                    row = i;
                    col = j;
                }
            }
        }
        if largest < 1e-15 {
            break;
        }
        let tau = (tensor[col][col] - tensor[row][row]) / (2.0 * tensor[row][col]);
        let step = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + step * step).sqrt();
        let sine = step * cosine;
        let diagonal_row = tensor[row][row];
        let diagonal_col = tensor[col][col];
        let off = tensor[row][col];
        tensor[row][row] = diagonal_row - step * off;
        tensor[col][col] = diagonal_col + step * off;
        tensor[row][col] = 0.0;
        tensor[col][row] = 0.0;
        for other in 0..3 {
            if other == row || other == col {
                continue;
            }
            let left = tensor[other][row];
            let right = tensor[other][col];
            tensor[other][row] = cosine * left - sine * right;
            tensor[row][other] = tensor[other][row];
            tensor[other][col] = sine * left + cosine * right;
            tensor[col][other] = tensor[other][col];
        }
        for entry in 0..3 {
            let left = vectors[entry][row];
            let right = vectors[entry][col];
            vectors[entry][row] = cosine * left - sine * right;
            vectors[entry][col] = sine * left + cosine * right;
        }
    }
    vectors
}

fn rot_z(angle: f64) -> [[f64; 3]; 3] {
    // The extra orientation is a turn by `τ/5`. Host `sin` and `cos` of
    // that angle are not the same function on glibc and libSystem, and
    // the continuous-symmetry proposal is built from this matrix.
    let sine = libm::sin(angle);
    let cosine = libm::cos(angle);
    [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
}

fn matmul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = left[row][0] * right[0][col]
                + left[row][1] * right[1][col]
                + left[row][2] * right[2][col];
        }
    }
    out
}

fn column(matrix: [[f64; 3]; 3], index: usize) -> [f64; 3] {
    [matrix[0][index], matrix[1][index], matrix[2][index]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn reject(vector: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    let scale = vector[0] * axis[0] + vector[1] * axis[1] + vector[2] * axis[2];
    [
        vector[0] - scale * axis[0],
        vector[1] - scale * axis[1],
        vector[2] - scale * axis[2],
    ]
}

fn perpendicular(vector: [f64; 3]) -> [f64; 3] {
    if vector[0].abs() < 0.9 {
        cross(vector, [1.0, 0.0, 0.0])
    } else {
        cross(vector, [0.0, 1.0, 0.0])
    }
}

fn norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(vector);
    (length > 1e-12).then_some([vector[0] / length, vector[1] / length, vector[2] / length])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inversion_average_matches_the_existing_projection() {
        let x = Array1::from_vec(vec![
            0.4, -0.2, 0.7, -0.1, 0.5, -0.3, 0.8, 0.1, 0.2, -0.6, -0.4, 0.9,
        ]);
        let classes = vec![0; 4];
        let (ours, _) = project(x.view(), &classes, &inversion_ops(), &[identity()]).expect("csm");
        let reference =
            crate::continuous_symmetry::project_inversion(x.view(), &classes).expect("inversion");
        for (left, right) in ours.iter().zip(reference.coordinates.iter()) {
            assert!((left - right).abs() < 1e-8, "{left} vs {right}");
        }
    }

    #[test]
    fn octahedron_measure_is_zero_and_a_shift_is_not() {
        let octahedron = Array1::from_vec(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ]);
        let classes = vec![0; 6];
        let (_, exact) =
            project(octahedron.view(), &classes, &inversion_ops(), &[identity()]).unwrap();
        assert!(exact < 1e-10, "{exact}");
        let mut shifted = octahedron.clone();
        shifted[0] += 0.3;
        let (_, noisy) =
            project(shifted.view(), &classes, &inversion_ops(), &[identity()]).unwrap();
        assert!(noisy > 1e-3, "{noisy}");
        let chosen = propose(octahedron.view(), &classes).expect("library");
        assert_eq!(chosen.group, "C_i");
        assert!(chosen.measure < 1e-8, "{}", chosen.measure);
    }

    #[test]
    fn a_coincident_set_is_not_a_projection() {
        assert!(
            project(
                Array1::zeros(9).view(),
                &[0, 0, 0],
                &inversion_ops(),
                &[identity()]
            )
            .is_none()
        );
    }

    fn ungated_projection(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<GroupProjection> {
        let orientations = inertia_orientations(x, classes.len())?;
        clear_op_cache();
        let mut best: Option<(usize, GroupProjection)> = None;
        for (rank, name, ops) in [
            (2, "D_5h", d5h_ops()),
            (1, "C_5", cyclic_about_z(5)),
            (0, "C_i", inversion_ops()),
        ] {
            let Some((coordinates, measure)) = project(x, classes, &ops, &orientations) else {
                continue;
            };
            if closest_pair(&coordinates) < SINGULAR_PAIR {
                continue;
            }
            let replace = match &best {
                None => true,
                Some((incumbent_rank, incumbent)) => {
                    measure < incumbent.measure
                        || (measure == incumbent.measure && rank < *incumbent_rank)
                }
            };
            if replace {
                best = Some((
                    rank,
                    GroupProjection {
                        coordinates,
                        measure,
                        group: name,
                    },
                ));
            }
        }
        best.map(|(_, projection)| projection)
    }

    fn assert_same_projection(gated: &GroupProjection, full: &GroupProjection) {
        assert_eq!(gated.group, full.group);
        assert_eq!(gated.measure.to_bits(), full.measure.to_bits());
        assert_eq!(gated.coordinates.len(), full.coordinates.len());
        for (left, right) in gated.coordinates.iter().zip(full.coordinates.iter()) {
            assert_eq!(left.to_bits(), right.to_bits(), "{left} vs {right}");
        }
    }

    #[test]
    fn inversion_is_skipped_only_when_it_cannot_change_the_group() {
        let octahedron = Array1::from_vec(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ]);
        let mut noisy = bipyramid();
        noisy[2] += 0.25;
        let mut chiral = Array1::zeros(36);
        for atom in 0..12 {
            let angle = 0.47 * atom as f64;
            chiral[3 * atom] = (0.4 + 0.05 * atom as f64) * angle.cos();
            chiral[3 * atom + 1] = (0.35 + 0.04 * atom as f64) * angle.sin();
            chiral[3 * atom + 2] = 0.15 * atom as f64 - 0.8;
        }
        for x in [octahedron, noisy, chiral] {
            let classes = vec![0; x.len() / 3];
            let full = ungated_projection(x.view(), &classes).expect("full");
            clear_op_cache();
            let gated = propose(x.view(), &classes).expect("gated");
            assert_same_projection(&gated, &full);
            let orientations = inertia_orientations(x.view(), classes.len()).expect("axes");
            let (centred, _, _) = centre_and_scale(x.view(), classes.len()).expect("centre");
            let bound = inversion_lower_measure(&centred, &orientations);
            let (_, solved) =
                project(x.view(), &classes, &inversion_ops(), &orientations).expect("inversion");
            assert!(
                bound <= solved + 1.0e-9 * (1.0 + solved),
                "bound {bound} exceeds the solved inversion measure {solved}"
            );
        }
        let mut noisy = bipyramid();
        noisy[2] += 0.25;
        let classes = vec![0; 7];
        let orientations = inertia_orientations(noisy.view(), 7).expect("axes");
        let (centred, _, _) = centre_and_scale(noisy.view(), 7).expect("centre");
        let bound = inversion_lower_measure(&centred, &orientations);
        let (_, fivefold) =
            project(noisy.view(), &classes, &d5h_ops(), &orientations).expect("d5h");
        let margin = 1.0e-6 * (1.0 + fivefold);
        assert!(
            bound > fivefold + margin,
            "the noisy bipyramid should settle inversion without the assignment, bound {bound} fivefold {fivefold}"
        );
    }

    fn bipyramid() -> Array1<f64> {
        let mut coords = Vec::with_capacity(21);
        for step in 0..5 {
            let angle = std::f64::consts::TAU * step as f64 / 5.0;
            coords.extend([angle.cos(), angle.sin(), 0.0]);
        }
        coords.extend([0.0, 0.0, 1.2, 0.0, 0.0, -1.2]);
        Array1::from_vec(coords)
    }

    #[test]
    fn pentagonal_bipyramid_selects_d5h_and_ignores_labels() {
        let x = bipyramid();
        let classes = vec![0; 7];
        let projection = propose(x.view(), &classes).expect("projection");
        assert!(projection.measure < 1e-8, "{}", projection.measure);
        let (_, d5h) = project(
            x.view(),
            &classes,
            &d5h_ops(),
            &inertia_orientations(x.view(), 7).expect("axes"),
        )
        .expect("d5h");
        assert!(d5h < 1e-8, "{d5h}");
        let mut reversed = Array1::zeros(x.len());
        for atom in 0..7 {
            for axis in 0..3 {
                reversed[3 * atom + axis] = x[3 * (6 - atom) + axis];
            }
        }
        let flipped = propose(reversed.view(), &classes).expect("relabelled");
        assert!((flipped.measure - projection.measure).abs() < 1e-8);
        let mut noisy = x.clone();
        noisy[2] += 0.25;
        let shifted = propose(noisy.view(), &classes).expect("noisy");
        assert!(shifted.measure > 1e-3, "{}", shifted.measure);
    }

    #[test]
    fn project_refuses_an_empty_group_a_bad_length_and_a_non_finite_point() {
        let x = Array1::from_vec(vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0]);
        let classes = [0, 0];
        assert!(project(x.view(), &classes, &[], &[identity()]).is_none());
        assert!(project(x.view(), &classes, &inversion_ops(), &[]).is_none());
        assert!(project(x.view(), &[0], &inversion_ops(), &[identity()]).is_none());
        let mut bad = x.clone();
        bad[0] = f64::NAN;
        assert!(project(bad.view(), &classes, &inversion_ops(), &[identity()]).is_none());
        assert!(centre_and_scale(Array1::zeros(0).view(), 0).is_none());
    }

    #[test]
    fn a_zero_orientation_is_not_a_measure() {
        let x = Array1::from_vec(vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0]);
        let zero = [[0.0; 3]; 3];
        assert!(project(x.view(), &[0, 0], &inversion_ops(), &[zero]).is_none());
    }

    #[test]
    fn a_pair_inside_the_singular_distance_is_not_proposed() {
        let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.005, 0.0, 0.0]);
        assert!(propose(x.view(), &[0, 0]).is_none());
    }

    #[test]
    fn a_nearly_axial_vector_uses_the_other_reference() {
        let along_x = perpendicular([1.0, 0.0, 0.0]);
        assert!(along_x[0].abs() < 1e-12);
        let along_z = perpendicular([0.0, 0.0, 1.0]);
        assert!(along_z[2].abs() < 1e-12);
    }

    #[test]
    fn d5h_has_twenty_orthogonal_operations() {
        let ops = d5h_ops();
        assert_eq!(ops.len(), 20);
        for op in ops {
            let gram = matmul(op, transpose(op));
            for row in 0..3 {
                for col in 0..3 {
                    let want = if row == col { 1.0 } else { 0.0 };
                    assert!((gram[row][col] - want).abs() < 1e-10);
                }
            }
        }
    }
}
