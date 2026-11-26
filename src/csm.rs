//! Continuous symmetry measure for a free Cartesian point set.
//!
//! Oakley, Johnston and Wales, Phys. Chem. Chem. Phys. 15, 3965 (2013),
//! Section 3.3, and GMIN `SYMMETRYCSM`. The structure is centred and scaled
//! so its largest radius is 1. For each operation, a class-blocked assignment
//! matches the operated image to the original, and the proposed geometry is
//! the mean of those operated, permuted images. [`crate::csm::propose`]
//! projects onto the `D_5h`, `C_5` and `C_i` templates at orientations from
//! the inertia tensor plus one fixed one-fifth turn.
//! [`crate::csm::propose_for`] with detected symmetry on instead projects,
//! in the input frame, onto the operations detected on the structure. An
//! operation is kept when each image lands within 0.25 of a distinct atom
//! of its own class. A short swap of two classes does not. Neither draws
//! a random number.

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
    if !is_closed_ops(ops) {
        return None;
    }
    project_verified(x, classes, ops, orientations)
}

fn project_verified(
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
    let mut solved: Vec<Vec<Matched>> = Vec::with_capacity(orientations.len());
    for (index, rotation) in orientations.iter().enumerate() {
        apply_all_into(rotation, centred, &mut y);
        fill_coordinate_bits(&y, &mut y_bits);
        let y_hash = hash_u64s(&y_bits);
        // An orientation that is an earlier one turned by an operation of
        // the group poses each operation's problem as the earlier one posed
        // the operation conjugated by that turn. A transposed operation
        // poses the transposed problem.
        let turn = (0..index).find_map(|earlier| {
            let relative = matmul(*rotation, transpose(orientations[earlier]));
            ops.iter()
                .find(|op| matrix_close(op, &relative))
                .map(|turn| (earlier, *turn))
        });
        let mut here: Vec<Matched> = Vec::with_capacity(ops.len());
        hat.fill([0.0; 3]);
        for (op_index, op) in ops.iter().enumerate() {
            let conjugate = turn.and_then(|(earlier, turn)| {
                let conjugate = matmul(matmul(transpose(turn), *op), turn);
                ops.iter()
                    .position(|other| matrix_close(other, &conjugate))
                    .map(|other| Hint {
                        matched: &solved[earlier][other],
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
        solved.push(here);
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

fn d5h_ops_ref() -> &'static [[[f64; 3]; 3]] {
    static OPS: OnceLock<Vec<[[f64; 3]; 3]>> = OnceLock::new();
    OPS.get_or_init(d5h_ops)
}

fn c5_ops_ref() -> &'static [[[f64; 3]; 3]] {
    static OPS: OnceLock<Vec<[[f64; 3]; 3]>> = OnceLock::new();
    OPS.get_or_init(|| cyclic_about_z(5))
}

fn inversion_ops_ref() -> &'static [[[f64; 3]; 3]] {
    static OPS: OnceLock<Vec<[[f64; 3]; 3]>> = OnceLock::new();
    OPS.get_or_init(inversion_ops)
}

/// Projects onto `D_5h`, `C_5`, or `C_i`, returning the group with the lowest
/// measure. This is [`propose_for`] with detected symmetry off.
pub fn propose(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<GroupProjection> {
    propose_for(false, x, classes)
}

/// The lowest-measure projection, onto operations detected on `x` when
/// `detected_symmetry` is on and onto the `D_5h`, `C_5` and `C_i` templates
/// when it is off.
///
/// A detected projection keeps only operations that carry every atom onto
/// a distinct atom of its own class, and its [`GroupProjection::group`] is
/// `"detected"`. Each image must land within [`DETECTION_TOLERANCE`] of
/// that atom on unit-radius coordinates. A swap onto a different class
/// fails even when the two atoms lie closer than that bound.
pub fn propose_for(
    detected_symmetry: bool,
    x: ArrayView1<'_, f64>,
    classes: &[u32],
) -> Option<GroupProjection> {
    if detected_symmetry {
        propose_detected(x, classes)
    } else {
        propose_templates(x, classes)
    }
}

fn propose_detected(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<GroupProjection> {
    if let Some(cached) = cached_propose(x, classes) {
        return cached;
    }
    let n = classes.len();
    if x.len() != n.checked_mul(3)? || x.iter().any(|value| !value.is_finite()) {
        return None;
    }
    clear_op_cache();
    // Detected operations are already in the input frame, so projecting
    // once at the identity avoids repeating the same assignment across
    // inertia orientations and lets the op cache reuse solved inverses.
    let best = detected_ops(x, classes)
        .and_then(|ops| project_verified(x, classes, &ops, &[identity()]))
        .and_then(|(coordinates, measure)| {
            (closest_pair(&coordinates) >= SINGULAR_PAIR).then_some(GroupProjection {
                coordinates,
                measure,
                group: "detected",
            })
        });
    store_propose(x, classes, best.as_ref());
    best
}

/// Operations of the input frame, written in the frame of `rotation`.
#[cfg(test)]
fn conjugate_ops(ops: &[[[f64; 3]; 3]], rotation: &[[f64; 3]; 3]) -> Vec<[[f64; 3]; 3]> {
    let back = transpose(*rotation);
    ops.iter()
        .map(|op| matmul(matmul(*rotation, *op), back))
        .collect()
}

/// Largest per-atom distance, on unit-radius coordinates, that
/// [`preserves_classes`] accepts. Detection uses the same cap. The image
/// must land on a distinct atom of the same class. A nearer atom of
/// another class is a swap and is rejected.
const DETECTION_TOLERANCE: f64 = 0.25;

/// The closed operation set detected on `x` whose every element carries each
/// atom onto the atoms of its own class.
fn detected_ops(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<Vec<[[f64; 3]; 3]>> {
    let n = x.len() / 3;
    if n < 2 || classes.len() != n {
        return None;
    }
    let (centred, _, _) = centre_and_scale(x, n)?;
    let mut scaled = Array1::zeros(3 * n);
    for (atom, point) in centred.iter().enumerate() {
        for axis in 0..3 {
            scaled[3 * atom + axis] = point[axis];
        }
    }
    let orders: Vec<usize> = (2..=n).collect();
    // Detection runs on unit-radius centred coordinates so it matches the
    // scale normalisation inside `project`.
    let mut candidates =
        crate::symmetrise::detect_all(scaled.view(), n, &orders, DETECTION_TOLERANCE);
    let one_class = classes.iter().all(|&class| class == classes[0]);
    if !one_class {
        // The scan matches each image to the nearest atom of any class, so
        // an operation that swaps species passes it. Projection matches
        // within each class and cannot use one.
        candidates.retain(|cand| {
            preserves_classes(&crate::symmetrise::candidate_op(cand), &centred, classes)
        });
    }
    let generators = select_generators(&candidates);
    if generators.is_empty() {
        return None;
    }
    let ops: Vec<[[f64; 3]; 3]> = generators
        .iter()
        .map(crate::symmetrise::candidate_op)
        .collect();
    let group = close_detected_ops(&ops)?;
    (one_class
        || group
            .iter()
            .all(|op| preserves_classes(op, &centred, classes)))
    .then_some(group)
}

/// Whether `op` carries `centred` onto itself class by class.
///
/// Each image must land within [`DETECTION_TOLERANCE`] of a distinct atom
/// of the same class. The bound is the detector's deviation cap, applied
/// per atom rather than as a mean. A swap onto another class fails however
/// short the gap. An octahedron axis against its equator fails with it.
fn preserves_classes(op: &[[f64; 3]; 3], centred: &[[f64; 3]], classes: &[u32]) -> bool {
    let n = centred.len();
    if n == 0 || classes.len() != n {
        return false;
    }
    let limit = DETECTION_TOLERANCE * DETECTION_TOLERANCE;
    let mut claimed = vec![false; n];
    for (atom, &point) in centred.iter().enumerate() {
        let image = mul_vec(op, point);
        let mut partner = 0;
        let mut best = f64::INFINITY;
        for (other, other_point) in centred.iter().enumerate() {
            let distance = (0..3)
                .map(|axis| (image[axis] - other_point[axis]).powi(2))
                .sum::<f64>();
            if distance < best {
                best = distance;
                partner = other;
            }
        }
        if best >= limit || classes[partner] != classes[atom] || claimed[partner] {
            return false;
        }
        claimed[partner] = true;
    }
    true
}

const CSM_CLOSURE: f64 = 0.27;
const CSM_GROUP_CAP: usize = 120;

fn close_detected_ops(gens: &[[[f64; 3]; 3]]) -> Option<Vec<[[f64; 3]; 3]>> {
    let group = crate::symmetrise::close_group_checked(gens, CSM_CLOSURE, CSM_GROUP_CAP)?;
    // The closure always contains the identity; one element means no
    // non-trivial symmetry was detected.
    (group.len() >= 2).then_some(group)
}

fn select_generators(
    candidates: &[crate::symmetrise::Candidate],
) -> Vec<crate::symmetrise::Candidate> {
    const PARALLEL_COSINE: f64 = 0.995;
    const ORTHOGONAL_SLACK: f64 = 0.05;
    let mut picked: Vec<crate::symmetrise::Candidate> = Vec::new();
    for &cand in candidates {
        if let Some(existing) = picked.iter_mut().find(|prev| {
            let dot = prev.axis[0] * cand.axis[0]
                + prev.axis[1] * cand.axis[1]
                + prev.axis[2] * cand.axis[2];
            dot.abs() > PARALLEL_COSINE && prev.improper == cand.improper
        }) {
            if !cand.improper
                && cand.order > existing.order
                && cand.order.is_multiple_of(existing.order)
                && cand.deviation <= existing.deviation + 1.0e-9
            {
                *existing = cand;
            }
            continue;
        }
        picked.push(cand);
    }
    picked.sort_by(|left, right| {
        let left_primary = !left.improper && left.order >= 3;
        let right_primary = !right.improper && right.order >= 3;
        right_primary
            .cmp(&left_primary)
            .then_with(|| {
                if left_primary {
                    right.order.cmp(&left.order)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then_with(|| {
                left.deviation
                    .partial_cmp(&right.deviation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right.order.cmp(&left.order))
    });
    let mut axes: Vec<[f64; 3]> = Vec::with_capacity(picked.len());
    for cand in &mut picked {
        let mut axis = cand.axis;
        for &anchor in &axes {
            let dot = axis[0] * anchor[0] + axis[1] * anchor[1] + axis[2] * anchor[2];
            if dot.abs() > 0.0 && dot.abs() < ORTHOGONAL_SLACK {
                axis = reject(axis, anchor);
                if let Some(unit) = normalize(axis) {
                    axis = unit;
                }
            }
        }
        cand.axis = axis;
        axes.push(axis);
    }
    picked
}

fn matrix_within(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3], tolerance: f64) -> bool {
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .all(|(left, right)| (*left - *right).abs() <= tolerance)
}

fn is_closed_ops(ops: &[[[f64; 3]; 3]]) -> bool {
    is_closed_within(ops, 1.0e-5)
}

fn is_closed_within(ops: &[[[f64; 3]; 3]], tolerance: f64) -> bool {
    if ops.is_empty() {
        return false;
    }
    let id = identity();
    if !ops
        .iter()
        .any(|op| matrix_within(op, &id, tolerance.clamp(1.0e-12, 1.0e-6)))
        && !ops.iter().any(|op| matrix_within(op, &id, tolerance))
    {
        return false;
    }
    for op in ops {
        if !matrix_within(&matmul(*op, transpose(*op)), &id, 1.0e-5) {
            return false;
        }
    }
    for left in ops {
        for right in ops {
            let product = matmul(*left, *right);
            if !ops
                .iter()
                .any(|other| matrix_within(other, &product, tolerance))
            {
                return false;
            }
        }
    }
    true
}

/// Projects onto `D_5h`, `C_5`, or `C_i`, returning the group with the lowest measure.
pub(crate) fn propose_templates(
    x: ArrayView1<'_, f64>,
    classes: &[u32],
) -> Option<GroupProjection> {
    if let Some(cached) = cached_template_propose(x, classes) {
        return cached;
    }
    let n = classes.len();
    if n == 0 || x.len() != n.checked_mul(3)? || x.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (centred, centroid, scale) = centre_and_scale(x, n)?;
    let orientations = inertia_orientations_centred(&centred, std::f64::consts::TAU / 5.0)?;
    // D_5h contains every C_5 turn, so it is solved first and those
    // permutations are reused. Equal measures keep the earlier group
    // in the published order C_i, C_5, D_5h.
    clear_op_cache();
    let mut best: Option<(usize, GroupProjection)> = None;
    for (rank, name, ops) in [
        (2, "D_5h", d5h_ops_ref()),
        (1, "C_5", c5_ops_ref()),
        (0, "C_i", inversion_ops_ref()),
    ] {
        if name == "C_i"
            && let Some((_, incumbent)) = &best
        {
            let bound = inversion_lower_measure(&centred, &orientations);
            let margin = 1.0e-6 * (1.0 + incumbent.measure);
            if bound > incumbent.measure + margin {
                continue;
            }
        }
        let Some((coordinates, measure)) =
            project_centred(&centred, centroid, scale, classes, ops, &orientations)
        else {
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
    store_template_propose(x, classes, best.as_ref());
    best
}

std::thread_local! {
    static TEMPLATE_PROPOSE_CACHE: std::cell::RefCell<Option<ProposeCache>> = const { std::cell::RefCell::new(None) };
}

fn cached_template_propose(
    x: ArrayView1<'_, f64>,
    classes: &[u32],
) -> Option<Option<GroupProjection>> {
    TEMPLATE_PROPOSE_CACHE.with(|slot| {
        let slot = slot.borrow();
        let cached = slot.as_ref()?;
        (same_bits(&cached.coordinates, x) && cached.classes == classes)
            .then(|| cached.value.clone())
    })
}

fn store_template_propose(
    x: ArrayView1<'_, f64>,
    classes: &[u32],
    value: Option<&GroupProjection>,
) {
    TEMPLATE_PROPOSE_CACHE.with(|slot| {
        *slot.borrow_mut() = Some(ProposeCache {
            coordinates: x.iter().map(|value| value.to_bits()).collect(),
            classes: classes.to_vec(),
            value: value.cloned(),
        });
    });
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

/// Lower bound on the inversion measure in every orientation.
///
/// The group is the identity and the central inversion. The residual is
/// half the matched distance, so the measure is the assignment cost
/// divided by four times the squared norm. Each atom's nearest inverse
/// image is at most that assignment cost, whether or not the nearest
/// images form a bijection.
fn inversion_lower_measure(centred: &[[f64; 3]], orientations: &[[[f64; 3]; 3]]) -> f64 {
    let n = centred.len();
    let mut y = vec![[0.0; 3]; n];
    let mut best = f64::INFINITY;
    for rotation in orientations {
        apply_all_into(rotation, centred, &mut y);
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
        // Below 98 atoms the recorded permutation is Kuhn-Munkres. Its
        // tie break is not the inverse of the transposed problem, so the
        // cache reuses an inverse only for a unique dense optimum. A
        // certificate is checked against the costs it is reused with.
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

fn inertia_orientations_centred(
    centred: &[[f64; 3]],
    extra_turn: f64,
) -> Option<Vec<[[f64; 3]; 3]>> {
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
        orientations.push(matmul(rot_z(extra_turn), aligned));
    }
    Some(orientations)
}

#[cfg(test)]
fn inertia_orientations(x: ArrayView1<'_, f64>, n: usize) -> Option<Vec<[[f64; 3]; 3]>> {
    inertia_orientations_with(x, n, std::f64::consts::PI)
}

#[cfg(test)]
fn inertia_orientations_with(
    x: ArrayView1<'_, f64>,
    n: usize,
    extra_turn: f64,
) -> Option<Vec<[[f64; 3]; 3]>> {
    let (centred, _, _) = centre_and_scale(x, n)?;
    inertia_orientations_centred(&centred, extra_turn)
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
        let chosen = propose_detected(octahedron.view(), &classes).expect("projection");
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
        let projection = propose_detected(x.view(), &classes).expect("projection");
        let mut reversed = Array1::zeros(x.len());
        for atom in 0..7 {
            for axis in 0..3 {
                reversed[3 * atom + axis] = x[3 * (6 - atom) + axis];
            }
        }
        let flipped = propose_detected(reversed.view(), &classes).expect("relabelled");
        assert!((flipped.measure - projection.measure).abs() < 1e-8);
        let mut noisy = x.clone();
        noisy[2] += 0.25;
        let shifted = propose_detected(noisy.view(), &classes).expect("noisy");
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
            for ops in [
                d5h_ops(),
                cyclic_about_z(5),
                turns_about_z(4),
                turns_about_z(3),
                inversion_ops(),
            ] {
                clear_op_cache();
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
        assert!(propose_detected(x.view(), &[0, 0]).is_none());
    }

    fn rotation_from(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
        let axis = normalize(axis).expect("axis");
        let sine = libm::sin(angle);
        let cosine = libm::cos(angle);
        let turn = 1.0 - cosine;
        let (x, y, z) = (axis[0], axis[1], axis[2]);
        [
            [
                turn * x * x + cosine,
                turn * x * y - sine * z,
                turn * x * z + sine * y,
            ],
            [
                turn * x * y + sine * z,
                turn * y * y + cosine,
                turn * y * z - sine * x,
            ],
            [
                turn * x * z - sine * y,
                turn * y * z + sine * x,
                turn * z * z + cosine,
            ],
        ]
    }

    fn apply_rotation(x: &Array1<f64>, rotation: [[f64; 3]; 3]) -> Array1<f64> {
        let mut out = Array1::zeros(x.len());
        let n = x.len() / 3;
        for atom in 0..n {
            let point = [x[3 * atom], x[3 * atom + 1], x[3 * atom + 2]];
            let turned = mul_vec(&rotation, point);
            for axis in 0..3 {
                out[3 * atom + axis] = turned[axis];
            }
        }
        out
    }

    /// Operations written in the input frame, applied after an orientation,
    /// give the same average as applying them before that orientation.
    #[test]
    fn conjugating_operations_matches_the_input_frame() {
        let x = bipyramid();
        let classes = vec![0u32; 7];
        let ops = inversion_ops();
        let (plain, plain_measure) =
            project(x.view(), &classes, &ops, &[identity()]).expect("input frame");
        let orientations = inertia_orientations(x.view(), 7).expect("axes");
        for orientation in orientations {
            let turned = conjugate_ops(&ops, &orientation);
            let (spun, spun_measure) =
                project(x.view(), &classes, &turned, &[orientation]).expect("oriented");
            assert!(
                (plain_measure - spun_measure).abs() < 1e-8,
                "{plain_measure} vs {spun_measure}"
            );
            for (left, right) in plain.iter().zip(spun.iter()) {
                assert!((left - right).abs() < 1e-6, "{left} vs {right}");
            }
        }
    }

    /// A rigid rotation does not change the detected measure.
    #[test]
    fn a_rotation_leaves_the_detected_measure_unchanged() {
        let symmetric = Array1::from_vec(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ]);
        let mut distorted = symmetric.clone();
        distorted[0] += 0.05;
        let rotation = rotation_from([1.0, 2.0, -0.5], 0.7);
        let mut measures = Vec::new();
        for structure in [symmetric, distorted] {
            let classes = vec![0u32; structure.len() / 3];
            let plain = propose_detected(structure.view(), &classes).expect("projection");
            let turned = apply_rotation(&structure, rotation);
            let spun = propose_detected(turned.view(), &classes).expect("rotated");
            let gap = (plain.measure - spun.measure).abs();
            assert!(
                gap < 1e-4,
                "plain {} spun {} gap {gap}",
                plain.measure,
                spun.measure
            );
            measures.push(plain.measure);
        }
        assert!(measures[0] < 1e-8, "exact octahedron {}", measures[0]);
        assert!(
            measures[1] > measures[0] + 1e-6,
            "distortion did not raise the measure: {}",
            measures[1]
        );
    }

    #[test]
    fn an_asymmetric_cloud_is_not_projected() {
        let x = Array1::from_vec(vec![
            0.2, -0.5, 0.1, 1.1, 0.3, -0.4, -0.6, 0.8, 0.7, 0.4, -0.2, 1.3,
        ]);
        assert!(propose_detected(x.view(), &[0, 0, 0, 0]).is_none());
    }

    #[test]
    fn a_nearly_axial_vector_uses_the_other_reference() {
        let along_x = perpendicular([1.0, 0.0, 0.0]);
        assert!(along_x[0].abs() < 1e-12);
        let along_z = perpendicular([0.0, 0.0, 1.0]);
        assert!(along_z[2].abs() < 1e-12);
    }

    fn icosahedron() -> Array1<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut coords = Vec::with_capacity(36);
        for &s1 in &[-1.0_f64, 1.0] {
            for &s2 in &[-1.0_f64, 1.0] {
                coords.extend([0.0, s1, s2 * phi]);
                coords.extend([s1, s2 * phi, 0.0]);
                coords.extend([s2 * phi, 0.0, s1]);
            }
        }
        Array1::from_vec(coords)
    }

    #[test]
    fn propose_is_invariant_under_uniform_scaling() {
        let mut x = bipyramid();
        // Perturb one point: at scale 1.0 the distortion is within 0.25 on
        // unit-radius coordinates, whereas on raw coordinates scaled by 10x or
        // 0.05x a fixed 0.25 tolerance would flip detection or admit spurious
        // operations.
        x[2] += 0.25;
        let classes = vec![0u32; 7];
        let base = propose_detected(x.view(), &classes).expect("base scale projection");
        for scale in [0.1_f64, 0.25, 4.0, 25.0] {
            let scaled = x.mapv(|v| v * scale);
            clear_op_cache();
            let got = propose_detected(scaled.view(), &classes).expect("scaled projection");
            assert!(
                (got.measure - base.measure).abs() < 1e-10,
                "scale={scale} measure {} vs {}",
                got.measure,
                base.measure
            );
            for (g, b) in got.coordinates.iter().zip(base.coordinates.iter()) {
                assert!(
                    (g - b * scale).abs() < 1e-8 * scale.max(1.0),
                    "scale={scale} coord {g} vs {}",
                    b * scale
                );
            }
        }
    }

    #[test]
    fn propose_admits_icosahedral_group_and_closes_approximate_generators() {
        let ico = icosahedron();
        let classes = vec![0u32; 12];
        let ih = detected_ops(ico.view(), &classes).expect("icosahedral group");
        assert_eq!(ih.len(), 120, "Ih must have 120 operations");
        let ico_proj = propose_detected(ico.view(), &classes).expect("icosahedral projection");
        assert!(ico_proj.measure < 1e-8, "{}", ico_proj.measure);

        // Perturb the icosahedron slightly: exact 1e-6 generate_group runs to
        // its cap and fails, while tolerant closure closes it into a valid
        // finite group (I, 60 elements).
        let mut approx = ico.clone();
        approx[0] += 0.03;
        approx[4] -= 0.02;
        let (centred, _, _) = centre_and_scale(approx.view(), 12).expect("scale");
        let mut scaled = Array1::zeros(36);
        for (atom, p) in centred.iter().enumerate() {
            for axis in 0..3 {
                scaled[3 * atom + axis] = p[axis];
            }
        }
        let orders: Vec<usize> = (2..=12).collect();
        let cands = crate::symmetrise::detect_all(scaled.view(), 12, &orders, 0.25);
        let gens = select_generators(&cands);
        assert!(
            crate::symmetrise::generate_group(&gens, 120).is_err(),
            "exact 1e-6 closure should fail on approximate generators"
        );
        let approx_ops = detected_ops(approx.view(), &classes).expect("approximate closure");
        assert_eq!(
            approx_ops.len(),
            60,
            "approximate icosahedron closes to I (60)"
        );
        assert!(is_closed_within(&approx_ops, CSM_CLOSURE));
        let approx_proj =
            propose_detected(approx.view(), &classes).expect("approximate projection");
        assert!(approx_proj.measure > 1e-5);
    }

    #[test]
    fn capped_non_closed_group_is_refused_by_propose_and_project() {
        // Non-commensurate C5 (about z) and tilted C4 (about [1, 0.2, 0.3])
        // generate a dense subgroup of SO(3) that never closes within cap.
        let c5 = rotation_from([0.0, 0.0, 1.0], std::f64::consts::TAU / 5.0);
        let c4 = rotation_from([1.0, 0.2, 0.3], std::f64::consts::FRAC_PI_2);
        assert!(
            close_detected_ops(&[c5, c4]).is_none(),
            "non-closing generator pair must be refused rather than silently truncated"
        );
        let classes = vec![0u32; 4];
        let unclosed = vec![identity(), rot_z(std::f64::consts::TAU / 5.0)];
        assert!(
            project(
                Array1::from_vec(vec![
                    1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0
                ])
                .view(),
                &classes,
                &unclosed,
                &[identity()]
            )
            .is_none(),
            "project must refuse a non-closed operation set"
        );
    }

    /// An octahedron whose axial pair is a second species has `O_h` as a
    /// point set and `D_4h` as a labelled structure. The eight-fold larger
    /// set carries axial atoms onto equatorial sites, and projecting onto it
    /// pulls the structure off itself.
    #[test]
    fn detected_ops_exclude_class_swapping_operations() {
        let octahedron = Array1::from_vec(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ]);
        let blind = detected_ops(octahedron.view(), &[0; 6]).expect("O_h");
        assert_eq!(blind.len(), 48);
        let classes = [0, 0, 0, 0, 1, 1];
        let (centred, _, _) = centre_and_scale(octahedron.view(), 6).expect("scale");
        let nearest = |image: [f64; 3]| {
            (0..6)
                .min_by(|&left, &right| {
                    let gap = |atom: usize| {
                        (0..3)
                            .map(|axis| (image[axis] - centred[atom][axis]).powi(2))
                            .sum::<f64>()
                    };
                    gap(left).total_cmp(&gap(right))
                })
                .expect("atoms")
        };
        let swaps = |op: &[[f64; 3]; 3]| {
            (0..6).any(|atom| classes[nearest(mul_vec(op, centred[atom]))] != classes[atom])
        };
        assert_eq!(blind.iter().filter(|op| swaps(op)).count(), 32);

        let aware = detected_ops(octahedron.view(), &classes).expect("D_4h");
        assert_eq!(aware.len(), 16);
        assert!(aware.iter().all(|op| !swaps(op)));
        assert!(is_closed_ops(&aware));

        let (_, blind_measure) =
            project(octahedron.view(), &classes, &blind, &[identity()]).expect("blind");
        assert!(blind_measure > 1e-2, "{blind_measure}");
        let projection = propose_detected(octahedron.view(), &classes).expect("D_4h projection");
        assert_eq!(projection.group, "detected");
        assert!(projection.measure < 1e-12, "{}", projection.measure);
        for (got, want) in projection.coordinates.iter().zip(octahedron.iter()) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    /// Two atoms 0.08 apart. The half turn about z exchanges each with the
    /// neighbour of the other class. The mean same-class RMS is that gap,
    /// under 0.25. The labelled projection must not follow the turn.
    #[test]
    fn a_short_class_swap_is_rejected() {
        let coords = Array1::from_vec(vec![
            1.0, 0.04, 0.0, 1.0, -0.04, 0.0, -1.0, -0.04, 0.0, -1.0, 0.04, 0.0,
        ]);
        let swapped = [0u32, 1, 1, 0];
        let (centred, _, _) = centre_and_scale(coords.view(), 4).expect("scale");
        let half = rot_z(std::f64::consts::PI);
        let mut total = 0.0;
        for (atom, &point) in centred.iter().enumerate() {
            let image = mul_vec(&half, point);
            let nearest = centred
                .iter()
                .zip(&swapped)
                .filter(|&(_, &class)| class == swapped[atom])
                .map(|(other, _)| {
                    (0..3)
                        .map(|axis| (image[axis] - other[axis]).powi(2))
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min);
            total += nearest;
        }
        let rms = (total / 4.0).sqrt();
        assert!(rms < DETECTION_TOLERANCE, "{rms}");
        let moved = propose_detected(coords.view(), &swapped);
        let shift = moved.as_ref().map_or(f64::NAN, |projection| {
            projection
                .coordinates
                .iter()
                .zip(coords.iter())
                .map(|(got, want)| (got - want).abs())
                .fold(0.0, f64::max)
        });
        assert!(
            !preserves_classes(&half, &centred, &swapped),
            "a class swap closer than 0.25 was kept, rms {rms}, shift {shift}"
        );
        let projection = moved.expect("labelled mirror");
        assert!(shift < 1e-8, "{shift}");
        assert!(projection.measure < 1e-10, "{}", projection.measure);
    }

    /// Exchanging two adjacent atoms of one class is a mirror of this
    /// dimer. The projection stays on the input.
    #[test]
    fn an_adjacent_same_class_exchange_stays_put() {
        let coords = Array1::from_vec(vec![
            1.0, 0.04, 0.0, 1.0, -0.04, 0.0, -1.0, -0.04, 0.0, -1.0, 0.04, 0.0,
        ]);
        let classes = [0u32; 4];
        let projection = propose_detected(coords.view(), &classes).expect("same class");
        let shift = projection
            .coordinates
            .iter()
            .zip(coords.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0_f64, f64::max);
        assert!(shift < 1e-8, "{shift}");
        assert!(projection.measure < 1e-10, "{}", projection.measure);
    }

    #[test]
    fn propose_projects_once_and_reuses_op_cache() {
        let ico = icosahedron();
        let classes = vec![0u32; 12];
        let ih = detected_ops(ico.view(), &classes).expect("Ih");
        assert_eq!(ih.len(), 120);
        let first = propose_detected(ico.view(), &classes).expect("first projection");
        let cached_ops = OP_CACHE.with(|slot| slot.borrow().len());
        assert!(
            cached_ops > 0 && cached_ops < ih.len(),
            "expected single-orientation op cache reuse, got {cached_ops} entries for {} ops",
            ih.len()
        );
        let second = propose_detected(ico.view(), &classes).expect("cached projection");
        assert_eq!(first, second);
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
        let chosen = propose_templates(octahedron.view(), &classes).expect("library");
        assert_eq!(chosen.group, "C_i");
        assert!(chosen.measure < 1e-8, "{}", chosen.measure);
    }

    fn ungated_projection(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<GroupProjection> {
        let orientations =
            inertia_orientations_with(x, classes.len(), std::f64::consts::TAU / 5.0)?;
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
            let gated = propose_templates(x.view(), &classes).expect("gated");
            assert_same_projection(&gated, &full);
            let orientations =
                inertia_orientations_with(x.view(), classes.len(), std::f64::consts::TAU / 5.0)
                    .expect("axes");
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
        let orientations =
            inertia_orientations_with(noisy.view(), 7, std::f64::consts::TAU / 5.0).expect("axes");
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

    #[test]
    fn pentagonal_bipyramid_selects_d5h_and_ignores_labels() {
        let x = bipyramid();
        let classes = vec![0; 7];
        let projection = propose_templates(x.view(), &classes).expect("projection");
        assert!(projection.measure < 1e-8, "{}", projection.measure);
        let (_, d5h) = project(
            x.view(),
            &classes,
            &d5h_ops(),
            &inertia_orientations_with(x.view(), 7, std::f64::consts::TAU / 5.0).expect("axes"),
        )
        .expect("d5h");
        assert!(d5h < 1e-8, "{d5h}");
        let mut reversed = Array1::zeros(x.len());
        for atom in 0..7 {
            for axis in 0..3 {
                reversed[3 * atom + axis] = x[3 * (6 - atom) + axis];
            }
        }
        let flipped = propose(reversed.view(), &classes).expect("relabeled");
        assert!((flipped.measure - projection.measure).abs() < 1e-8);
        let mut noisy = x.clone();
        noisy[2] += 0.25;
        let shifted = propose_templates(noisy.view(), &classes).expect("noisy");
        assert!(shifted.measure > 1e-3, "{}", shifted.measure);
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
            let proposed = propose_templates(x.view(), &classes).expect("propose");
            let ungated = ungated_projection(x.view(), &classes).expect("ungated");
            assert_eq!(proposed.group, ungated.group);
            assert_eq!(proposed.measure.to_bits(), ungated.measure.to_bits());
            assert_eq!(
                proposed
                    .coordinates
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                ungated
                    .coordinates
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}
