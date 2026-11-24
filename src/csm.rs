//! Continuous symmetry measure for a free Cartesian point set.
//!
//! Oakley, Johnston and Wales, Phys. Chem. Chem. Phys. 15, 3965 (2013),
//! Section 3.3, and GMIN `SYMMETRYCSM`. The structure is centred and scaled
//! so its largest radius is 1. For each operation, a class-blocked assignment
//! matches the operated image to the original, and the proposed geometry is
//! the mean of those operated, permuted images. [`crate::csm::propose`] takes its
//! orientations from the inertia tensor plus one fixed half turn, and
//! draws no random number.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use ndarray::{Array1, ArrayView1};

use crate::assignment::{Certificate, certified_assignment, certifies_duals};

struct ClassBlocks {
    blocks: Vec<Vec<usize>>,
    local_index: Vec<usize>,
}

impl ClassBlocks {
    fn from_classes(classes: &[u32]) -> Self {
        let n = classes.len();
        if let Some(&first) = classes.first()
            && classes.iter().all(|&class| class == first)
        {
            return Self {
                blocks: vec![(0..n).collect()],
                local_index: (0..n).collect(),
            };
        }
        let mut by_class: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (atom, &class) in classes.iter().enumerate() {
            by_class.entry(class).or_default().push(atom);
        }
        let mut local_index = vec![0usize; n];
        let mut blocks = Vec::with_capacity(by_class.len());
        for atoms in by_class.into_values() {
            for (local, &atom) in atoms.iter().enumerate() {
                local_index[atom] = local;
            }
            blocks.push(atoms);
        }
        Self {
            blocks,
            local_index,
        }
    }
}

struct MatchScratch {
    images: Vec<[f64; 3]>,
    costs: Vec<f64>,
    local: Vec<usize>,
    inv_assignment: Vec<usize>,
}

impl MatchScratch {
    fn new() -> Self {
        Self {
            images: Vec::new(),
            costs: Vec::new(),
            local: Vec::new(),
            inv_assignment: Vec::new(),
        }
    }
}

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
    project_centred(&centred, centroid, scale, classes, ops, orientations)
}

fn project_centred(
    centred: &[[f64; 3]],
    centroid: [f64; 3],
    scale: f64,
    classes: &[u32],
    ops: &[[[f64; 3]; 3]],
    orientations: &[[[f64; 3]; 3]],
) -> Option<(Array1<f64>, f64)> {
    let n = classes.len();
    let blocks = ClassBlocks::from_classes(classes);
    let mut scratch = MatchScratch::new();
    let mut y = vec![[0.0; 3]; n];
    let mut y_bits = vec![0u64; 3 * n];
    let mut hat = vec![[0.0; 3]; n];
    let mut best: Option<(f64, [[f64; 3]; 3], Vec<[f64; 3]>)> = None;
    let turns = orientation_turns(orientations, ops);
    let last_use = last_orientation_uses(&turns);
    // Keep permutations only for orientations that a later orientation
    // actually uses as a conjugacy hint. The previous rectangular cache held
    // every assignment for every orientation until projection returned.
    let mut solved: Vec<Option<Vec<Matched>>> = (0..orientations.len()).map(|_| None).collect();
    for (index, rotation) in orientations.iter().enumerate() {
        apply_all_into(rotation, centred, &mut y);
        fill_coordinate_bits(&y, &mut y_bits);
        let y_hash = hash_u64s(&y_bits);
        // An orientation that is an earlier one turned by an operation of
        // the group poses each operation's problem as the earlier one posed
        // the operation conjugated by that turn. A transposed operation
        // poses the transposed problem.
        let turn = turns[index];
        let mut here: Vec<Matched> = Vec::with_capacity(ops.len());
        hat.fill([0.0; 3]);
        for (op_index, op) in ops.iter().enumerate() {
            let conjugate = turn.and_then(|(earlier, turn)| {
                let conjugate = matmul(matmul(transpose(turn), *op), turn);
                ops.iter()
                    .position(|other| matrix_close(other, &conjugate))
                    .map(|other| Hint {
                        matched: &solved[earlier].as_ref().expect("referenced orientation")[other],
                        transposed: false,
                    })
            });
            let inverse = transpose(*op);
            let hint = conjugate.or_else(|| {
                ops[..op_index]
                    .iter()
                    .position(|other| matrix_close(other, &inverse))
                    .map(|other| Hint {
                        matched: &here[other],
                        transposed: true,
                    })
            });
            let matched = match_operated(
                &y,
                &y_bits,
                y_hash,
                classes,
                &blocks,
                op,
                hint,
                &mut scratch,
            )?;
            for atom in 0..n {
                let image = mul_vec(op, y[matched.assignment[atom]]);
                for axis in 0..3 {
                    hat[atom][axis] += image[axis];
                }
            }
            here.push(matched);
        }
        if last_use[index].is_some() {
            solved[index] = Some(here);
        }
        if let Some((earlier, _)) = turn
            && last_use[earlier] == Some(index)
        {
            solved[earlier] = None;
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
            best = Some((measure, *rotation, hat.clone()));
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

fn orientation_turns(
    orientations: &[[[f64; 3]; 3]],
    ops: &[[[f64; 3]; 3]],
) -> Vec<Option<(usize, [[f64; 3]; 3])>> {
    orientations
        .iter()
        .enumerate()
        .map(|(index, rotation)| {
            (0..index).find_map(|earlier| {
                let relative = matmul(*rotation, transpose(orientations[earlier]));
                ops.iter()
                    .find(|op| matrix_close(op, &relative))
                    .map(|turn| (earlier, *turn))
            })
        })
        .collect()
}

fn last_orientation_uses(turns: &[Option<(usize, [[f64; 3]; 3])>]) -> Vec<Option<usize>> {
    let mut last_use = vec![None; turns.len()];
    for (index, turn) in turns.iter().enumerate() {
        if let Some((earlier, _)) = turn {
            last_use[*earlier] = Some(index);
        }
    }
    last_use
}

/// Pair distance below which an average is a coordinate singularity.
pub const SINGULAR_PAIR: f64 = 1.0e-2;

/// Lowest-measure average from symmetry operations detected on the incumbent.
///
/// The detector proposes approximate operations from the current coordinates.
/// Those generators are closed into a finite operation set and projected by
/// [`project`]. An average with a pair closer than [`SINGULAR_PAIR`] is dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupProjection {
    /// Cartesian coordinates in the input atom order.
    pub coordinates: Array1<f64>,
    /// Normalised squared distance to the average. Zero is exact symmetry.
    pub measure: f64,
    /// Projection family label.
    pub group: &'static str,
}

fn inversion_ops_ref() -> &'static [[[f64; 3]; 3]] {
    static OPS: OnceLock<Vec<[[f64; 3]; 3]>> = OnceLock::new();
    OPS.get_or_init(inversion_ops)
}

/// Projects onto a detected operation set with the lowest measure.
pub fn propose(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<GroupProjection> {
    if let Some(cached) = cached_propose(x, classes) {
        return cached;
    }
    let n = classes.len();
    if n == 0 || x.len() != n.checked_mul(3)? || x.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (centred, centroid, scale) = centre_and_scale(x, n)?;
    let _orientations = inertia_orientations_centred(&centred)?;
    clear_op_cache();
    let ops = detected_ops(x);
    let mut best: Option<GroupProjection> = if ops.len() > 1 {
        project_centred(&centred, centroid, scale, classes, &ops, &[identity()]).and_then(
            |(coordinates, measure)| {
                (closest_pair(&coordinates) >= SINGULAR_PAIR).then_some(GroupProjection {
                    coordinates,
                    measure,
                    group: "detected",
                })
            },
        )
    } else {
        None
    };
    let try_inv = match &best {
        None => true,
        Some(incumbent) => {
            let bound = inversion_lower_measure_id(&centred);
            bound + 1.0e-12 < incumbent.measure
        }
    };
    if try_inv
        && let Some((coordinates, measure)) = project_centred(
            &centred,
            centroid,
            scale,
            classes,
            inversion_ops_ref(),
            &[identity()],
        )
        && closest_pair(&coordinates) >= SINGULAR_PAIR
        && best
            .as_ref()
            .is_none_or(|incumbent| measure + 1.0e-12 < incumbent.measure)
    {
        best = Some(GroupProjection {
            coordinates,
            measure,
            group: "C_i",
        });
    }
    store_propose(x, classes, best.as_ref());
    best
}
fn inversion_lower_measure_id(centred: &[[f64; 3]]) -> f64 {
    let mut residual = 0.0;
    let mut norm = 0.0;
    for p in centred {
        let mut best_d2 = f64::INFINITY;
        for q in centred {
            let dx = -p[0] - q[0];
            let dy = -p[1] - q[1];
            let dz = -p[2] - q[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
            }
        }
        residual += 0.25 * best_d2;
        norm += p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    }
    if norm < 1e-18 {
        f64::INFINITY
    } else {
        residual / norm
    }
}

fn detected_ops(x: ArrayView1<'_, f64>) -> Vec<[[f64; 3]; 3]> {
    let n = x.len() / 3;
    if n < 2 {
        return vec![identity()];
    }
    let orders: Vec<usize> = (2..=n).collect();
    // Operations outside the tolerance are not a group of this structure.
    // Filling that gap with the least-bad rotation kept a high order whose
    // turn maps every atom to itself, then closed it into a long cyclic
    // group and assigned every element.
    let candidates = crate::symmetrise::detect_generators(x, n, &orders, 0.25);
    let group = crate::symmetrise::close_symmetries(&candidates, x, n, 0.25, 64);
    if group.is_empty() {
        vec![identity()]
    } else {
        group
    }
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

fn inversion_ops() -> Vec<[[f64; 3]; 3]> {
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
    y_hash: u64,
    y: Vec<u64>,
    classes: Vec<u32>,
    op: [[f64; 3]; 3],
    matched: Matched,
    unique: bool,
}

/// One operation's permutation and, per class block in label order, the
/// certificate that forced it.
#[derive(Clone)]
struct Matched {
    assignment: Vec<usize>,
    blocks: Vec<Option<Certificate>>,
}

impl Matched {
    #[allow(dead_code)]
    fn inverse(&self) -> Self {
        Self {
            assignment: invert_perm(&self.assignment),
            blocks: self
                .blocks
                .iter()
                .map(|block| block.as_ref().map(Certificate::transposed))
                .collect(),
        }
    }
}

/// A solved problem whose permutation may also solve this one.
#[derive(Clone, Copy)]
struct Hint<'a> {
    matched: &'a Matched,
    /// The hint solved the transposed costs.
    transposed: bool,
}

std::thread_local! {
    static OP_CACHE: std::cell::RefCell<Vec<SolvedOp>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn clear_op_cache() {
    OP_CACHE.with(|slot| slot.borrow_mut().clear());
}

fn fill_coordinate_bits(y: &[[f64; 3]], out: &mut [u64]) {
    for (slot, value) in out.iter_mut().zip(y.iter().flatten()) {
        *slot = value.to_bits();
    }
}

fn hash_u64s(values: &[u64]) -> u64 {
    let mut hash = 0x9e37_79b9_7f4a_7c15_u64;
    for &value in values {
        hash = (hash.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    hash
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

fn recall_op(y_bits: &[u64], y_hash: u64, classes: &[u32], op: &[[f64; 3]; 3]) -> Option<Matched> {
    let transposed = transpose(*op);
    OP_CACHE.with(|slot| {
        let slot = slot.borrow();
        for saved in slot.iter().rev() {
            if saved.y_hash != y_hash || saved.classes != classes || saved.y != y_bits {
                continue;
            }
            if same_matrix(&saved.op, op) {
                return Some(saved.matched.clone());
            }
            if saved.unique && matrix_close(&saved.op, &transposed) {
                return Some(saved.matched.inverse());
            }
        }
        None
    })
}

fn remember_op(
    y_bits: &[u64],
    y_hash: u64,
    classes: &[u32],
    op: &[[f64; 3]; 3],
    matched: &Matched,
    unique: bool,
) {
    OP_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.len() > 800 {
            slot.clear();
        }
        slot.push(SolvedOp {
            y_hash,
            y: y_bits.to_vec(),
            classes: classes.to_vec(),
            op: *op,
            matched: matched.clone(),
            unique,
        });
    });
}

#[allow(clippy::too_many_arguments)]
fn match_operated(
    y: &[[f64; 3]],
    y_bits: &[u64],
    y_hash: u64,
    classes: &[u32],
    class_blocks: &ClassBlocks,
    op: &[[f64; 3]; 3],
    hint: Option<Hint<'_>>,
    scratch: &mut MatchScratch,
) -> Option<Matched> {
    let n = classes.len();
    // The identity sends every atom to itself, so the diagonal is a minimum
    // of cost zero and the assignment problem is unnecessary.
    if is_identity(op) {
        return Some(Matched {
            assignment: (0..n).collect(),
            blocks: vec![None; class_blocks.blocks.len()],
        });
    }
    if let Some(saved) = recall_op(y_bits, y_hash, classes, op) {
        return Some(saved);
    }
    scratch.images.resize(n, [0.0; 3]);
    for (slot, point) in scratch.images.iter_mut().zip(y.iter()) {
        *slot = mul_vec(op, *point);
    }
    if let Some(hint) = hint
        && hint.transposed
    {
        scratch.inv_assignment.resize(n, 0);
        for (row, &column) in hint.matched.assignment.iter().enumerate() {
            scratch.inv_assignment[column] = row;
        }
    }
    let mut assignment = vec![usize::MAX; n];
    let mut certificates = Vec::with_capacity(class_blocks.blocks.len());
    let mut unique = true;
    for (block, atoms) in class_blocks.blocks.iter().enumerate() {
        let width = atoms.len();
        scratch.costs.resize(width * width, 0.0);
        let costs = &mut scratch.costs[..width * width];
        for (row_i, &row) in atoms.iter().enumerate() {
            let p = y[row];
            let row_base = row_i * width;
            for (col_i, &col) in atoms.iter().enumerate() {
                let q = scratch.images[col];
                let dx = p[0] - q[0];
                let dy = p[1] - q[1];
                let dz = p[2] - q[2];
                costs[row_base + col_i] = dx * dx + dy * dy + dz * dz;
            }
        }
        let proved = hint.and_then(|hint| {
            let certificate = hint.matched.blocks.get(block)?.as_ref()?;
            scratch.local.clear();
            scratch.local.reserve(width);
            let source_assignment = if hint.transposed {
                &scratch.inv_assignment[..n]
            } else {
                &hint.matched.assignment
            };
            for &atom in atoms {
                scratch
                    .local
                    .push(class_blocks.local_index[source_assignment[atom]]);
            }
            let (row_pot, col_pot) = certificate.duals(hint.transposed);
            certifies_duals(costs, width, &scratch.local, row_pot, col_pot).then(|| {
                let cert = if hint.transposed {
                    certificate.transposed()
                } else {
                    certificate.clone()
                };
                (scratch.local.clone(), cert)
            })
        });
        let (local, certificate) = match proved {
            Some((local, certificate)) => (local, Some(certificate)),
            None => certified_assignment(costs, width)?,
        };
        unique &= width >= 98 && crate::assignment::last_dense_optimum_was_unique();
        for (row_i, &col_i) in local.iter().enumerate() {
            assignment[atoms[row_i]] = atoms[col_i];
        }
        certificates.push(certificate);
    }
    if assignment.contains(&usize::MAX) {
        return None;
    }
    let matched = Matched {
        assignment,
        blocks: certificates,
    };
    remember_op(y_bits, y_hash, classes, op, &matched, unique);
    Some(matched)
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

fn apply_all_into(matrix: &[[f64; 3]; 3], points: &[[f64; 3]], out: &mut [[f64; 3]]) {
    for (slot, point) in out.iter_mut().zip(points.iter()) {
        *slot = mul_vec(matrix, *point);
    }
}

#[cfg(test)]
fn apply_all(matrix: &[[f64; 3]; 3], points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    points.iter().map(|point| mul_vec(matrix, *point)).collect()
}

#[cfg(test)]
fn cyclic_about_z(order: usize) -> Vec<[[f64; 3]; 3]> {
    (0..order)
        .map(|step| rot_z(std::f64::consts::TAU * step as f64 / order as f64))
        .collect()
}

#[cfg(test)]
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

fn inertia_orientations_centred(centred: &[[f64; 3]]) -> Option<Vec<[[f64; 3]; 3]>> {
    let axes = jacobi(inertia(centred));
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
        orientations.push(matmul(rot_z(std::f64::consts::PI), aligned));
    }
    Some(orientations)
}

#[cfg(test)]
fn inertia_orientations(x: ArrayView1<'_, f64>, n: usize) -> Option<Vec<[[f64; 3]; 3]>> {
    let (centred, _, _) = centre_and_scale(x, n)?;
    inertia_orientations_centred(&centred)
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
    // One orientation is a half turn. Host `sin` and `cos` of that angle
    // are not the same function on glibc and libSystem, and the
    // continuous-symmetry proposal is built from this matrix.
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
    fn a_detected_projection_is_reported_with_its_generic_label() {
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
        let chosen = propose(octahedron.view(), &classes).expect("projection");
        assert_eq!(chosen.group, "detected");
        assert!(chosen.measure < 1e-8, "{}", chosen.measure);
    }

    #[test]
    fn propose_ignores_label_order_and_rises_when_a_point_moves() {
        let mut coords = Vec::with_capacity(21);
        for step in 0..5 {
            let angle = std::f64::consts::TAU * step as f64 / 5.0;
            coords.extend([angle.cos(), angle.sin(), 0.0]);
        }
        coords.extend([0.0, 0.0, 1.2, 0.0, 0.0, -1.2]);
        let x = Array1::from_vec(coords);
        let classes = vec![0; 7];
        let projection = propose(x.view(), &classes).expect("projection");
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

    /// `project` with every operation solved by the Kuhn-Munkres walk.
    fn walked_projection(
        x: ArrayView1<'_, f64>,
        classes: &[u32],
        ops: &[[[f64; 3]; 3]],
        orientations: &[[[f64; 3]; 3]],
    ) -> (Array1<f64>, f64) {
        let n = classes.len();
        let (centred, centroid, scale) = centre_and_scale(x, n).expect("centre");
        let mut best: Option<(f64, [[f64; 3]; 3], Vec<[f64; 3]>)> = None;
        for rotation in orientations {
            let y = apply_all(rotation, &centred);
            let mut hat = vec![[0.0; 3]; n];
            for op in ops {
                let mut assignment: Vec<usize> = (0..n).collect();
                if !is_identity(op) {
                    let images: Vec<[f64; 3]> = y.iter().map(|point| mul_vec(op, *point)).collect();
                    let mut blocks: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
                    for (atom, &class) in classes.iter().enumerate() {
                        blocks.entry(class).or_default().push(atom);
                    }
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
                        let local =
                            crate::assignment::kuhn_munkres_walk(&costs, width).expect("walk");
                        for (row_i, &col_i) in local.iter().enumerate() {
                            assignment[atoms[row_i]] = atoms[col_i];
                        }
                    }
                }
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
            let measure = residual / norm;
            if best
                .as_ref()
                .is_none_or(|(incumbent, _, _)| measure < *incumbent)
            {
                best = Some((measure, *rotation, hat));
            }
        }
        let (measure, rotation, hat) = best.expect("an orientation");
        let back = transpose(rotation);
        let mut coordinates = Array1::zeros(3 * n);
        for atom in 0..n {
            let restored = mul_vec(&back, hat[atom]);
            for axis in 0..3 {
                coordinates[3 * atom + axis] = centroid[axis] + restored[axis] * scale;
            }
        }
        (coordinates, measure)
    }

    /// Turned orientations and transposed operations reuse a certified
    /// permutation; the projection keeps the walk's bits.
    #[test]
    fn a_reused_permutation_keeps_the_walked_projection() {
        let mut state = 0x00c5_d5a1_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let random: Array1<f64> = (0..240).map(|_| 2.0 * next()).collect();
        let mut near = bipyramid();
        near[0] += 0.01;
        for (x, classes) in [
            (random.clone(), vec![0u32; 80]),
            (
                random,
                (0..80).map(|atom| u32::from(atom % 3 == 0)).collect(),
            ),
            (bipyramid(), vec![0u32; 7]),
            (near, vec![0u32; 7]),
        ] {
            let orientations = inertia_orientations(x.view(), classes.len()).expect("axes");
            for ops in [turns_about_z(4), turns_about_z(3), inversion_ops()] {
                let (coordinates, measure) =
                    project(x.view(), &classes, &ops, &orientations).expect("projection");
                let (walked, walked_measure) =
                    walked_projection(x.view(), &classes, &ops, &orientations);
                assert_eq!(measure.to_bits(), walked_measure.to_bits());
                for (left, right) in coordinates.iter().zip(walked.iter()) {
                    assert_eq!(left.to_bits(), right.to_bits(), "{left} vs {right}");
                }
            }
        }
    }

    #[test]
    fn projection_retains_one_match_row_for_related_turns() {
        for order in [12, 24, 48, 96] {
            let ops = turns_about_z(order);
            let turns = orientation_turns(&ops, &ops);
            let last_use = last_orientation_uses(&turns);
            assert_eq!(last_use.iter().flatten().count(), 1);
            assert_eq!(last_use[0], Some(order - 1));
            assert!(last_use[1..].iter().all(Option::is_none));
        }
    }

    fn turns_about_z(order: usize) -> Vec<[[f64; 3]; 3]> {
        (0..order)
            .map(|step| rot_z(std::f64::consts::TAU * step as f64 / order as f64))
            .collect()
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
    fn propose_considers_inversion_when_no_rotational_symmetry_is_detected() {
        // Centrosymmetric chiral-like set: ±a, ±b, ±c with unequal radii and no C2/mirror.
        let x = Array1::from_vec(vec![
            1.2, 0.1, 0.3, -1.2, -0.1, -0.3, 0.2, 1.7, -0.4, -0.2, -1.7, 0.4, -0.5, 0.3, 2.1, 0.5,
            -0.3, -2.1,
        ]);
        let mut perturbed = x.clone();
        perturbed[0] += 0.04;
        perturbed[4] -= 0.03;
        let classes = [0; 6];
        let prop = propose(perturbed.view(), &classes).expect("inversion proposal");
        assert_eq!(prop.group, "C_i");
        assert!(prop.measure < 0.01, "measure={}", prop.measure);
    }

    #[test]
    fn propose_and_project_match_walked_reference_across_classes() {
        let mut x = bipyramid();
        x[2] += 0.17;
        x[6] -= 0.09;
        let orientations = inertia_orientations(x.view(), 7).expect("axes");
        for classes in [vec![0u32; 7], vec![0, 1, 0, 1, 0, 2, 2]] {
            clear_op_cache();
            for ops in [d5h_ops(), cyclic_about_z(5), inversion_ops()] {
                let (actual_coords, actual_measure) =
                    project(x.view(), &classes, &ops, &orientations).expect("project");
                let (walked_coords, walked_measure) =
                    walked_projection(x.view(), &classes, &ops, &orientations);
                assert_eq!(actual_measure.to_bits(), walked_measure.to_bits());
                assert_eq!(
                    actual_coords
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    walked_coords
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            let proposed = propose(x.view(), &classes).expect("propose");
            assert!(proposed.measure.is_finite());
        }
    }
}
