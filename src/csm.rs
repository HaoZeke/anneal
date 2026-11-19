//! Continuous symmetry measure for a free Cartesian point set.
//!
//! Oakley, Johnston and Wales, Phys. Chem. Chem. Phys. 15, 3965 (2013),
//! Section 3.3, and GMIN `SYMMETRYCSM`. The structure is centred and scaled
//! so its largest radius is 1. For each operation, a class-blocked assignment
//! matches the operated image to the original, and the proposed geometry is
//! the mean of those operated, permuted images. [`crate::csm::propose`] takes its
//! orientations from the inertia tensor plus one fixed one-fifth turn, and
//! draws no random number.

use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1};

use crate::assignment::{Certificate, certified_assignment, certifies};

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
    project_with_op_measure(x, classes, ops, orientations)
        .map(|(coordinates, measure, _)| (coordinates, measure))
}

fn project_with_op_measure(
    x: ArrayView1<'_, f64>,
    classes: &[u32],
    ops: &[[[f64; 3]; 3]],
    orientations: &[[[f64; 3]; 3]],
) -> Option<(Array1<f64>, f64, f64)> {
    let n = classes.len();
    if ops.is_empty() || orientations.is_empty() || x.len() != n.checked_mul(3)? {
        return None;
    }
    if x.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (centred, centroid, scale) = centre_and_scale(x, n)?;
    let mut best: Option<(f64, f64, f64, f64, f64, usize, [[f64; 3]; 3], Vec<[f64; 3]>)> = None;
    let mut solved: Vec<Vec<Matched>> = Vec::with_capacity(orientations.len());
    for (index, rotation) in orientations.iter().enumerate() {
        let y = apply_all(rotation, &centred);
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
        let mut hat = vec![[0.0; 3]; n];
        let mut op_residual = 0.0;
        let mut worst_residual = 0.0_f64;
        let mut wide_residual = 0.0_f64;
        let mut wide_square = 0.0_f64;
        let mut wide_count = 0.0_f64;
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
            let matched = match_operated(&y, classes, op, hint)?;
            let mut this_residual = 0.0;
            for atom in 0..n {
                let image = mul_vec(op, y[matched.assignment[atom]]);
                for axis in 0..3 {
                    hat[atom][axis] += image[axis];
                    let delta = y[atom][axis] - image[axis];
                    let squared = delta * delta;
                    op_residual += squared;
                    this_residual += squared;
                }
            }
            if !is_identity(op) {
                worst_residual = worst_residual.max(this_residual);
                if improper_or_wide(op) {
                    wide_residual += this_residual;
                    wide_square += this_residual * this_residual;
                    wide_count += 1.0;
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
        let non_identity = ops.len().saturating_sub(1).max(1) as f64;
        let op_measure = op_residual / (2.0 * non_identity * norm);
        let worst = worst_residual / (2.0 * norm);
        let wide = if wide_count > 0.0 {
            wide_residual / (2.0 * wide_count * norm)
        } else {
            op_measure
        };
        let wide_se = if wide_count > 1.0 {
            let mean = wide_residual / wide_count;
            let variance = (wide_square / wide_count - mean * mean).max(0.0);
            variance.sqrt() / (wide_count.sqrt() * 2.0 * norm)
        } else {
            0.0
        };
        if best
            .as_ref()
            .is_none_or(|(incumbent, _, _, _, _, _, _, _)| measure < *incumbent)
        {
            best = Some((
                measure,
                op_measure,
                worst,
                wide,
                wide_se,
                wide_count.round() as usize,
                *rotation,
                hat,
            ));
        }
    }
    let (measure, op_measure, worst, wide, wide_se, wide_ops, rotation, hat) = best?;
    LAST_OP_SPLIT.with(|slot| slot.set((worst, wide)));
    LAST_OP_SE.with(|slot| slot.set(wide_se));
    LAST_WIDE_OPS.with(|slot| slot.set(wide_ops));
    let back = transpose(rotation);
    let mut coordinates = Array1::zeros(3 * n);
    for atom in 0..n {
        let restored = mul_vec(&back, hat[atom]);
        for axis in 0..3 {
            coordinates[3 * atom + axis] = centroid[axis] + restored[axis] * scale;
        }
    }
    Some((coordinates, measure, op_measure))
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
    /// `C_i`, `C_5`, `D_5h`, `C_n`, or `D_nh`.
    pub group: &'static str,
    /// Principal rotational order (`2` for inversion).
    pub order: usize,
}

/// Projects onto the library group with the lowest measure.
pub fn propose(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<GroupProjection> {
    if let Some(cached) = cached_propose(x, classes, false) {
        return cached;
    }
    let orientations = inertia_orientations(x, classes.len())?;
    // D_5h contains every C_5 turn, so it is solved first and those
    // permutations are reused. Equal measures keep the earlier group
    // in the published order C_i, C_5, D_5h.
    clear_op_cache();
    let mut best: Option<(usize, GroupProjection)> = None;
    for (rank, order, name, ops) in [
        (2, 5, "D_5h", d5h_ops()),
        (1, 5, "C_5", cyclic_about_z(5)),
        (0, 2, "C_i", inversion_ops()),
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
                    order,
                },
            ));
        }
    }
    let best = best.map(|(_, projection)| projection);
    store_propose(x, classes, false, best.as_ref());
    best
}

/// Projects onto `C_i`, `C_n`, or `D_nh`.
///
/// The fixed channel (`C_i`, `C_5`, `D_5h`) is always scored. A `C_5` winner
/// is replaced by `D_5h` when the dihedral group's per-operation residual is
/// lower: the raw centroid measure lets the mirrors cancel. Any other order
/// is the dominant angular period of a principal shell. It replaces the fixed
/// winner only when the mean residual of its wide turns (a right angle or
/// more, and every improper isometry) is lower by more than the combined
/// standard error of those two means. Acute turns match a compact body to
/// itself and are not part of that mean. Inversion does not displace a
/// rotational winner.
pub fn propose_general(x: ArrayView1<'_, f64>, classes: &[u32]) -> Option<GroupProjection> {
    if let Some(cached) = cached_propose(x, classes, true) {
        return cached;
    }
    let n = classes.len();
    let (centred, _, _) = centre_and_scale(x, n)?;
    let base = principal_orientations(&centred)?;
    let mut cyclic5: Option<Scored> = None;
    let mut dihedral5: Option<Scored> = None;
    let mut inversion: Option<Scored> = None;
    clear_op_cache();
    let fivefold = with_phase_orientations(&base, 5);
    if let Some(scored) = score_group(x, classes, 5, "D_nh", &d5h_ops(), &fivefold) {
        dihedral5 = Some(scored);
    }
    if let Some(scored) = score_group(x, classes, 5, "C_n", &cyclic_about_z(5), &fivefold) {
        cyclic5 = Some(scored);
    }
    clear_op_cache();
    if let Some(scored) = score_group(x, classes, 2, "C_i", &inversion_ops(), &base) {
        inversion = Some(scored);
    }
    // The published fixed-channel order: raw centroid measure, with D_5h
    // ahead of C_5 ahead of C_i on a tie.
    let mut incumbent: Option<Scored> = None;
    for (rank, candidate) in [
        (2, dihedral5.clone()),
        (1, cyclic5.clone()),
        (0, inversion.clone()),
    ] {
        let Some(candidate) = candidate else {
            continue;
        };
        let replace = match &incumbent {
            None => true,
            Some(held) => {
                candidate.measure < held.measure
                    || (candidate.measure == held.measure && rank < held.rank)
            }
        };
        if replace {
            incumbent = Some(Scored { rank, ..candidate });
        }
    }
    if let (Some(held), Some(dihedral), Some(cyclic)) = (&incumbent, &dihedral5, &cyclic5)
        && held.group == "C_n"
        && held.order == 5
        && dihedral.op < cyclic.op
    {
        incumbent = Some(dihedral.clone());
    }
    if let Some(held) = incumbent.as_mut() {
        for order in shell_orders(&centred) {
            if order == 5 {
                continue;
            }
            clear_op_cache();
            let orientations = with_phase_orientations(&base, order);
            let dihedral = score_group(
                x,
                classes,
                order,
                "D_nh",
                &dihedral_horizontal_ops(order),
                &orientations,
            );
            let cyclic = score_group(
                x,
                classes,
                order,
                "C_n",
                &cyclic_about_z(order),
                &orientations,
            );
            let Some(alternative) = better_by_operation(dihedral, cyclic) else {
                continue;
            };
            if wide_improvement(held, &alternative) {
                *held = alternative;
            }
        }
    }
    let best = incumbent.map(|scored| scored.projection());
    store_propose(x, classes, true, best.as_ref());
    best
}

/// Per-operation residual of the wide turns, and the fixed-channel bookkeeping.
#[derive(Clone)]
struct Scored {
    coordinates: Array1<f64>,
    measure: f64,
    group: &'static str,
    order: usize,
    rank: usize,
    op: f64,
    wide: f64,
    se: f64,
    wide_ops: usize,
}

impl Scored {
    fn projection(self) -> GroupProjection {
        GroupProjection {
            coordinates: self.coordinates,
            measure: self.measure,
            group: self.group,
            order: self.order,
        }
    }
}

fn score_group(
    x: ArrayView1<'_, f64>,
    classes: &[u32],
    order: usize,
    name: &'static str,
    ops: &[[[f64; 3]; 3]],
    orientations: &[[[f64; 3]; 3]],
) -> Option<Scored> {
    #[cfg(test)]
    SCORED_ORDERS.with(|slot| slot.borrow_mut().push(order));
    let (coordinates, measure, op) = project_with_op_measure(x, classes, ops, orientations)?;
    if closest_pair(&coordinates) < SINGULAR_PAIR {
        return None;
    }
    let (wide, se, wide_ops) = LAST_OP_SPLIT.with(|slot| {
        let wide = slot.get().1;
        let se = LAST_OP_SE.with(|se| se.get());
        let wide_ops = LAST_WIDE_OPS.with(|count| count.get());
        (wide, se, wide_ops)
    });
    Some(Scored {
        coordinates,
        measure,
        group: name,
        order,
        rank: 0,
        op,
        wide,
        se,
        wide_ops,
    })
}

/// `D_nh` wins a tie of per-operation residuals. A strict cyclic improvement
/// is a missing mirror, not a smaller group to be preferred by default.
fn better_by_operation(dihedral: Option<Scored>, cyclic: Option<Scored>) -> Option<Scored> {
    match (dihedral, cyclic) {
        (Some(dihedral), Some(cyclic)) if cyclic.op < dihedral.op => Some(cyclic),
        (Some(dihedral), _) => Some(dihedral),
        (None, cyclic) => cyclic,
    }
}

/// The wide-turn means differ by more than the combined standard error.
///
/// One standard error is the sampling error of each mean, not a tuned
/// acceptance window. A one-operation inversion has no such error and does
/// not displace a rotational incumbent.
fn wide_improvement(incumbent: &Scored, alternative: &Scored) -> bool {
    if alternative.order == 2 && incumbent.order != 2 {
        return false;
    }
    if alternative.wide_ops < 2 {
        return false;
    }
    // A lower wide-turn mean that moves the centroid further from the
    // structure is the small-turn dilution: the average stays put while a
    // few wide turns look slightly better. The centroid has to improve too.
    if alternative.measure > incumbent.measure {
        return false;
    }
    let gap = incumbent.wide - alternative.wide;
    // A one-operation incumbent has no sampling error of its own. The
    // challenger's error is then the only one that is defined.
    let noise = if incumbent.wide_ops < 2 {
        alternative.se
    } else {
        incumbent.se.hypot(alternative.se)
    };
    gap > noise
}

/// Order of one principal shell, when that shell is a single ring.
///
/// A half turn is already carried by inversion, so a cyclic shell starts at
/// order three. Fewer off-axis atoms than two such shells cannot be split,
/// and the set is that order. A larger set contributes its outermost ring
/// only when the ring's azimuthal gaps equal one full turn over that
/// population: a high order on a compact body is a small turn, and the
/// identity assignment then looks better than a real symmetry.
fn shell_orders(centred: &[[f64; 3]]) -> Vec<usize> {
    let Some(axes) = principal_orientations(centred) else {
        return Vec::new();
    };
    let mut orders = Vec::new();
    for aligned in &axes {
        if let Some(order) = dominant_order(centred, aligned[2])
            && !orders.contains(&order)
        {
            orders.push(order);
        }
    }
    orders
}

fn dominant_order(centred: &[[f64; 3]], axis: [f64; 3]) -> Option<usize> {
    let tangent = normalize(perpendicular(axis))?;
    let bitangent = normalize(cross(axis, tangent))?;
    let mut samples = Vec::new();
    for point in centred {
        let height = point[0] * axis[0] + point[1] * axis[1] + point[2] * axis[2];
        let radial = [
            point[0] - height * axis[0],
            point[1] - height * axis[1],
            point[2] - height * axis[2],
        ];
        let rho = norm(radial);
        if rho > 1e-8 {
            let x = radial[0] * tangent[0] + radial[1] * tangent[1] + radial[2] * tangent[2];
            let y = radial[0] * bitangent[0] + radial[1] * bitangent[1] + radial[2] * bitangent[2];
            samples.push((rho, y.atan2(x)));
        }
    }
    let mut radii: Vec<f64> = samples.iter().map(|sample| sample.0).collect();
    let order = ring_order(&mut radii)?;
    // Two minimum shells cannot fit, so there is no outer ring to test.
    if samples.len() < minimum_shell_order() * 2 {
        return Some(order);
    }
    outer_ring_is_regular(&samples, order).then_some(order)
}

/// Smallest cyclic order that is not the half turn already in `C_i`.
fn minimum_shell_order() -> usize {
    3
}

fn ring_order(radii: &mut [f64]) -> Option<usize> {
    let minimum = minimum_shell_order();
    if radii.len() < minimum {
        return None;
    }
    radii.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    if radii.len() < minimum * 2 {
        return Some(radii.len());
    }
    // The gap that peels the axis is not a shell boundary. The outer half,
    // including the step that separates it from the body, is the shell.
    let start = radii.len() / 2;
    let mut gap = 0.0;
    let mut cut = radii[start];
    for pair in radii[start - 1..].windows(2) {
        let step = pair[1] - pair[0];
        if step > gap {
            gap = step;
            cut = pair[1];
        }
    }
    if gap == 0.0 {
        if radii[0] == radii[radii.len() - 1] {
            return Some(radii.len());
        }
        let shell = radii.len() - (start - 1);
        return (shell >= minimum).then_some(shell);
    }
    let outer = radii.iter().filter(|rho| **rho >= cut).count();
    (outer >= minimum).then_some(outer)
}

/// Azimuthal gaps of a regular ring equal one turn divided by the population.
///
/// The excess of the largest gap over that step is rounding of the axis,
/// not an acceptance window: the gaps sum to one turn, so any missing
/// vertex makes the largest gap strictly larger than the step.
fn outer_ring_is_regular(samples: &[(f64, f64)], order: usize) -> bool {
    if order < minimum_shell_order() || order > samples.len() {
        return false;
    }
    let mut ranked: Vec<usize> = (0..samples.len()).collect();
    ranked.sort_by(|&left, &right| {
        samples[right]
            .0
            .partial_cmp(&samples[left].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut angles: Vec<f64> = ranked
        .into_iter()
        .take(order)
        .map(|index| samples[index].1)
        .collect();
    angles.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let step = std::f64::consts::TAU / order as f64;
    let mut max_gap = angles[0] + std::f64::consts::TAU - angles[order - 1];
    for pair in angles.windows(2) {
        max_gap = max_gap.max(pair[1] - pair[0]);
    }
    max_gap <= step + step * f64::EPSILON.sqrt()
}

struct ProposeCache {
    general: bool,
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

fn cached_propose(
    x: ArrayView1<'_, f64>,
    classes: &[u32],
    general: bool,
) -> Option<Option<GroupProjection>> {
    PROPOSE_CACHE.with(|slot| {
        let slot = slot.borrow();
        let cached = slot.as_ref()?;
        (cached.general == general
            && same_bits(&cached.coordinates, x)
            && cached.classes == classes)
            .then(|| cached.value.clone())
    })
}

fn store_propose(
    x: ArrayView1<'_, f64>,
    classes: &[u32],
    general: bool,
    value: Option<&GroupProjection>,
) {
    PROPOSE_CACHE.with(|slot| {
        *slot.borrow_mut() = Some(ProposeCache {
            general,
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

std::thread_local! {
    static LAST_OP_SPLIT: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((0.0, 0.0)) };
    static LAST_OP_SE: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    static LAST_WIDE_OPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// Orders passed to the assignment while a test calls `propose_general`.
#[cfg(test)]
std::thread_local! {
    static SCORED_ORDERS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Proper rotations near or past a right angle, and every improper isometry.
///
/// A high-order turn by a few degrees matches atoms to themselves on any
/// compact body, so it is not a symmetry constraint. Reflections, inversion
/// and wide turns are.
fn improper_or_wide(op: &[[f64; 3]; 3]) -> bool {
    let det = op[0][0] * (op[1][1] * op[2][2] - op[1][2] * op[2][1])
        - op[0][1] * (op[1][0] * op[2][2] - op[1][2] * op[2][0])
        + op[0][2] * (op[1][0] * op[2][1] - op[1][1] * op[2][0]);
    if det < 0.0 {
        return true;
    }
    let trace = op[0][0] + op[1][1] + op[2][2];
    let cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    cosine.acos() >= std::f64::consts::FRAC_PI_2
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

fn recall_op(y: &[[f64; 3]], classes: &[u32], op: &[[f64; 3]; 3]) -> Option<Matched> {
    let bits = coordinate_bits(y);
    let transposed = transpose(*op);
    OP_CACHE.with(|slot| {
        let slot = slot.borrow();
        for saved in slot.iter().rev() {
            if saved.y != bits || saved.classes != classes {
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
    y: &[[f64; 3]],
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
            y: coordinate_bits(y),
            classes: classes.to_vec(),
            op: *op,
            matched: matched.clone(),
            unique,
        });
    });
}

fn match_operated(
    y: &[[f64; 3]],
    classes: &[u32],
    op: &[[f64; 3]; 3],
    hint: Option<Hint<'_>>,
) -> Option<Matched> {
    let n = classes.len();
    let mut blocks: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (atom, &class) in classes.iter().enumerate() {
        blocks.entry(class).or_default().push(atom);
    }
    // The identity sends every atom to itself, so the diagonal is a minimum
    // of cost zero and the assignment problem is unnecessary.
    if is_identity(op) {
        return Some(Matched {
            assignment: (0..n).collect(),
            blocks: vec![None; blocks.len()],
        });
    }
    if let Some(saved) = recall_op(y, classes, op) {
        return Some(saved);
    }
    let images: Vec<[f64; 3]> = y.iter().map(|point| mul_vec(op, *point)).collect();
    let hinted = hint.map(|hint| {
        if hint.transposed {
            hint.matched.inverse()
        } else {
            hint.matched.clone()
        }
    });
    let mut local_index = vec![0usize; n];
    for atoms in blocks.values() {
        for (local, &atom) in atoms.iter().enumerate() {
            local_index[atom] = local;
        }
    }
    let mut assignment = vec![usize::MAX; n];
    let mut certificates = Vec::with_capacity(blocks.len());
    let mut unique = true;
    for (block, atoms) in blocks.values().enumerate() {
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
        let proved = hinted.as_ref().and_then(|hinted| {
            let certificate = hinted.blocks.get(block)?.as_ref()?;
            let local: Vec<usize> = atoms
                .iter()
                .map(|&atom| local_index[hinted.assignment[atom]])
                .collect();
            certifies(&costs, width, &local, certificate).then(|| (local, certificate.clone()))
        });
        let (local, certificate) = match proved {
            Some((local, certificate)) => (local, Some(certificate)),
            None => certified_assignment(&costs, width)?,
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
    remember_op(y, classes, op, &matched, unique);
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
    dihedral_horizontal_ops(5)
}

fn dihedral_horizontal_ops(order: usize) -> Vec<[[f64; 3]; 3]> {
    let turns = cyclic_about_z(order);
    let twofold = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    let mirror = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
    let mut ops = Vec::with_capacity(4 * turns.len());
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

fn principal_orientations(centred: &[[f64; 3]]) -> Option<Vec<[[f64; 3]; 3]>> {
    let axes = jacobi(inertia(centred));
    let mut orientations = Vec::with_capacity(3);
    for axis in 0..3 {
        let z_axis = normalize(column(axes, axis))?;
        let mut x_axis = reject(column(axes, (axis + 1) % 3), z_axis);
        if norm(x_axis) < 1e-8 {
            x_axis = perpendicular(z_axis);
        }
        let x_axis = normalize(x_axis)?;
        let y_axis = normalize(cross(z_axis, x_axis))?;
        orientations.push([x_axis, y_axis, z_axis]);
    }
    Some(orientations)
}

fn with_phase_orientations(base: &[[[f64; 3]; 3]], order: usize) -> Vec<[[f64; 3]; 3]> {
    let step = rot_z(std::f64::consts::TAU / order as f64);
    let mut orientations = Vec::with_capacity(2 * base.len());
    for aligned in base {
        orientations.push(*aligned);
        orientations.push(matmul(step, *aligned));
    }
    orientations
}

fn inertia_orientations(x: ArrayView1<'_, f64>, n: usize) -> Option<Vec<[[f64; 3]; 3]>> {
    let (centred, _, _) = centre_and_scale(x, n)?;
    let base = principal_orientations(&centred)?;
    Some(with_phase_orientations(&base, 5))
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
        for (rank, order, name, ops) in [
            (2, 5, "D_5h", d5h_ops()),
            (1, 5, "C_5", cyclic_about_z(5)),
            (0, 2, "C_i", inversion_ops()),
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
                        order,
                    },
                ));
            }
        }
        best.map(|(_, projection)| projection)
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
            for ops in [d5h_ops(), cyclic_about_z(5), inversion_ops()] {
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

    #[test]
    fn general_propose_detects_orders_and_normalises_per_operation() {
        // 1. Order 3 at the lower and upper loop bound (n = 3): an isosceles
        // planar triangle selects D_3h while the fixed channel proposes D_5h.
        let triangle = Array1::from_vec(vec![
            1.15,
            0.0,
            0.0,
            -0.5,
            0.866_025_403_784_438_6,
            0.0,
            -0.5,
            -0.866_025_403_784_438_6,
            0.0,
        ]);
        let tri_classes = [0u32; 3];
        let fixed_tri = propose(triangle.view(), &tri_classes).expect("fixed triangle");
        assert_eq!((fixed_tri.group, fixed_tri.order), ("D_5h", 5));
        let gen_tri = propose_general(triangle.view(), &tri_classes).expect("general triangle");
        assert_eq!((gen_tri.group, gen_tri.order), ("D_nh", 3));
        // Cache separation between propose and propose_general.
        let cached_gen = propose_general(triangle.view(), &tri_classes).expect("cached general");
        assert_eq!(cached_gen, gen_tri);
        let cached_fixed = propose(triangle.view(), &tri_classes).expect("cached fixed");
        assert_eq!((cached_fixed.group, cached_fixed.order), ("D_5h", 5));

        // 2. Order 4 at the upper loop bound (n = 4): a D_2d-puckered 4-ring
        // breaks inversion on every vertex while keeping seven of the fifteen
        // non-identity D_4h operations exact, so it selects D_4h.
        let square = Array1::from_vec(vec![
            1.0, 0.0, 0.15, 0.0, 1.0, -0.15, -1.0, 0.0, 0.15, 0.0, -1.0, -0.15,
        ]);
        let sq_proj = propose_general(square.view(), &[0u32; 4]).expect("square");
        assert_eq!((sq_proj.group, sq_proj.order), ("D_nh", 4));

        // 3. Chiral cyclic C_3 (n = 6): two unequally sized, twisted triangles
        // break horizontal mirror, C_2', and inversion while keeping C_3.
        let mut chiral_c3 = Vec::with_capacity(18);
        for k in 0..3 {
            let a = std::f64::consts::TAU * k as f64 / 3.0;
            chiral_c3.extend([a.cos(), a.sin(), 0.6]);
        }
        for k in 0..3 {
            let a = std::f64::consts::TAU * k as f64 / 3.0 + 0.35;
            chiral_c3.extend([0.7 * a.cos(), 0.7 * a.sin(), -0.4]);
        }
        let c3_arr = Array1::from_vec(chiral_c3);
        let c3_proj = propose_general(c3_arr.view(), &[0u32; 6]).expect("chiral c3");
        assert_eq!((c3_proj.group, c3_proj.order), ("C_n", 3));
        assert!(c3_proj.measure < 1e-12, "{}", c3_proj.measure);

        // 4. Centrosymmetric C_i (n = 4) with no rotational axis of order >= 3.
        let ci_cluster = Array1::from_vec(vec![
            1.3, 0.2, 0.5, -1.3, -0.2, -0.5, 0.1, 1.7, -0.4, -0.1, -1.7, 0.4,
        ]);
        let ci_proj = propose_general(ci_cluster.view(), &[0u32; 4]).expect("ci cluster");
        assert_eq!((ci_proj.group, ci_proj.order), ("C_i", 2));
        assert!(ci_proj.measure < 1e-12, "{}", ci_proj.measure);

        // Verify the exact C_i relation op_measure == 2 * measure on a noisy
        // 4-atom point set.
        let mut noisy_ci = ci_cluster;
        noisy_ci[0] += 0.15;
        let (centred_ci, _, _) = centre_and_scale(noisy_ci.view(), 4).unwrap();
        let base_ci = principal_orientations(&centred_ci).unwrap();
        let (_, m_ci, op_ci) =
            project_with_op_measure(noisy_ci.view(), &[0u32; 4], &inversion_ops(), &base_ci)
                .unwrap();
        assert!(m_ci > 1e-4 && (op_ci - 2.0 * m_ci).abs() < 1e-12);

        // 5. Singular-pair rejection in both the axial loop (n = 3) and C_i (n = 2).
        let singular_pair = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.005, 0.0, 0.0]);
        assert!(propose_general(singular_pair.view(), &[0, 0]).is_none());
        let singular_axial =
            Array1::from_vec(vec![0.0, 0.0, 0.0, 0.005, 0.0, 0.0, -0.004, 0.003, 0.0]);
        assert!(propose_general(singular_axial.view(), &[0, 0, 0]).is_none());

        // 6. Competitive 7-atom perturbed pentagonal bipyramid where raw
        // centroid CSM prefers D_4h (m4_d < m5_d) by vector cancellation of
        // inconsistent operations and unnormalised |G| division prefers C_5,
        // while per-non-identity-operation CSM selects D_5h (order = 5).
        let mut s = 0x5ce9_bd63_87b8_32f7u64;
        let mut x = bipyramid();
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            *v += 0.38 * u;
        }
        let classes = vec![0u32; 7];
        let (centred, _, _) = centre_and_scale(x.view(), 7).unwrap();
        let base = principal_orientations(&centred).unwrap();
        clear_op_cache();
        let ori4 = with_phase_orientations(&base, 4);
        let (_, m4_d, op4_d) =
            project_with_op_measure(x.view(), &classes, &dihedral_horizontal_ops(4), &ori4)
                .unwrap();
        clear_op_cache();
        let ori5 = with_phase_orientations(&base, 5);
        let (_, m5_d, op5_d) =
            project_with_op_measure(x.view(), &classes, &dihedral_horizontal_ops(5), &ori5)
                .unwrap();
        let (_, _, op5_c) =
            project_with_op_measure(x.view(), &classes, &cyclic_about_z(5), &ori5).unwrap();
        assert!(
            m4_d < m5_d,
            "raw centroid CSM must prefer D_4h ({m4_d}) over D_5h ({m5_d})"
        );
        assert!(
            op5_d < op4_d && op5_d < op5_c,
            "per-operation CSM must prefer D_5h ({op5_d}) over D_4h ({op4_d}) and C_5 ({op5_c})"
        );
        assert!(
            op5_c * 4.0 / 5.0 < op5_d * 19.0 / 20.0,
            "dividing by |G| instead of |G| - 1 would prefer C_5 over D_5h"
        );
        let chosen = propose_general(x.view(), &classes).expect("competitive bipyramid");
        assert_eq!((chosen.group, chosen.order), ("D_nh", 5));
    }

    /// A compact cloud has no regular outer ring. Scanning every order anyway
    /// solves a small turn, whose per-operation residual undercuts the
    /// fivefold group and is the projection the search must not take.
    #[test]
    fn general_channel_leaves_a_compact_cloud_on_the_fixed_groups() {
        let n = 24;
        let mut state = 0x5ce9_bd63_87b8_32f7u64;
        let mut coords = Vec::with_capacity(3 * n);
        while coords.len() < 3 * n {
            let mut point = [0.0; 3];
            for component in &mut point {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *component = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            }
            if point.iter().map(|value| value * value).sum::<f64>() <= 1.0 {
                coords.extend(point);
            }
        }
        let cloud = Array1::from_vec(coords);
        let classes = vec![0u32; n];
        SCORED_ORDERS.with(|slot| slot.borrow_mut().clear());
        let chosen = propose_general(cloud.view(), &classes).expect("compact cloud");
        let scored = SCORED_ORDERS.with(|slot| slot.borrow().clone());
        assert!(
            scored.iter().all(|order| *order == 2 || *order == 5),
            "orders outside the fixed channel were solved: {scored:?}"
        );
        assert_eq!(scored.iter().filter(|order| **order == 5).count(), 2);
        assert_eq!(scored.iter().filter(|order| **order == 2).count(), 1);

        let fixed = propose(cloud.view(), &classes).expect("fixed channel");
        let upgraded = fixed.group == "C_5" && chosen.group == "D_nh" && chosen.order == 5;
        let same = match fixed.group {
            "D_5h" => chosen.group == "D_nh" && chosen.order == 5,
            "C_5" => chosen.group == "C_n" && chosen.order == 5,
            "C_i" => chosen.group == "C_i" && chosen.order == 2,
            _ => false,
        };
        assert!(
            same || upgraded,
            "fixed {} {} versus general {} {}",
            fixed.group,
            fixed.order,
            chosen.group,
            chosen.order
        );

        let (centred, _, _) = centre_and_scale(cloud.view(), n).unwrap();
        let base = principal_orientations(&centred).unwrap();
        clear_op_cache();
        let fivefold = with_phase_orientations(&base, 5);
        let (_, _, op_fivefold) =
            project_with_op_measure(cloud.view(), &classes, &d5h_ops(), &fivefold)
                .expect("fivefold");
        clear_op_cache();
        let turned = with_phase_orientations(&base, n);
        let (_, _, op_turn) =
            project_with_op_measure(cloud.view(), &classes, &cyclic_about_z(n), &turned)
                .expect("full turn");
        assert!(
            op_turn < op_fivefold,
            "a full-order turn must undercut the fivefold residual ({op_turn} vs {op_fivefold}) or this cloud no longer catches the scan"
        );
        assert_ne!(chosen.order, n);
    }
}
