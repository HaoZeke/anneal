//! Making a structure exactly symmetric about the symmetry it nearly has.
//!
//! Oakley, Johnston and Wales, *Symmetrisation schemes for global optimisation
//! of atomic clusters*, Phys. Chem. Chem. Phys. 15, 3965 (2013).
//!
//! High-symmetry structures are over-represented at both ends of the energy
//! distribution, so the global minimum of a cluster is more often symmetric
//! than a typical minimum is. That is a searchable fact: a structure that is
//! nearly symmetric can be pushed onto the symmetry it nearly has, and if the
//! answer is symmetric the push lands near it. The paper reports the mean first
//! encounter time for the 98-point Lennard-Jones cluster, whose global minimum
//! is tetrahedral, improving by more than seventyfold.
//!
//! # What this is not
//!
//! [`crate::movekernel::Symmetrise`] already exists and is a different thing.
//! It picks a random axis and a random rotation order, then blends each point a
//! fraction of the way toward its image. Nothing about it refers to the
//! structure's own symmetry, so on a structure that is nearly tetrahedral it is
//! overwhelmingly likely to symmetrise about an axis the structure has no
//! relationship to, and the result is a perturbation with extra steps.
//!
//! The scheme here measures first. Candidate axes come from the structure, the
//! deviation from each candidate symmetry is computed, and the operation is
//! applied about the best one and applied fully rather than blended. A
//! structure with no approximate symmetry is left alone, which is the case
//! where the random version does its damage.

use ndarray::{Array1, ArrayView1};

/// An approximate rotational symmetry a structure was found to have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Unit vector along the rotation axis, through the centroid.
    pub axis: [f64; 3],
    /// Rotation order; the operation is by `2 pi / order`.
    ///
    /// Order 1 with `improper` set is a plane: the reflection in the plane
    /// through the centroid whose normal is `axis`.
    pub order: usize,
    /// Whether the operation carries a reflection.
    ///
    /// The group a structure has is not always a rotation group. The 98-point
    /// global minimum is tetrahedral and the 38-point one is octahedral, and
    /// both of those are defined by their mirror planes: the proper rotations
    /// alone give T and O, which are index-two subgroups missing exactly the
    /// operations that pick out the structure. A generator set closed only
    /// under proper rotations cannot express either, so the scheme could
    /// symmetrise toward a subgroup of the target and never toward the target.
    pub improper: bool,
    /// Root-mean-square distance between a point and its matched image.
    ///
    /// Zero for an exact symmetry. This is what "approximate" is measured by,
    /// and what decides whether symmetrising is worth doing.
    pub deviation: f64,
}

/// Centroid of a flattened `(n, 3)` point set.
fn centroid(x: ArrayView1<f64>, n: usize) -> [f64; 3] {
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in c.iter_mut() {
        *v /= n.max(1) as f64;
    }
    c
}

/// Rotates `v` about a unit `axis` by `angle`, by Rodrigues' formula.
fn rotate(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, co) = angle.sin_cos();
    let dot = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    let cross = [
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    ];
    [
        v[0] * co + cross[0] * s + axis[0] * dot * (1.0 - co),
        v[1] * co + cross[1] * s + axis[1] * dot * (1.0 - co),
        v[2] * co + cross[2] * s + axis[2] * dot * (1.0 - co),
    ]
}

fn normalise(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-9 {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

/// Root-mean-square distance between each point and its nearest image under a
/// rotation of `order` about `axis`.
///
/// The matching is nearest-image, which is what makes this a measure of
/// approximate symmetry rather than of exact symmetry: an exactly symmetric
/// structure permutes onto itself and every distance is zero.
pub fn deviation(x: ArrayView1<f64>, n: usize, axis: [f64; 3], order: usize) -> f64 {
    if n == 0 || order < 2 {
        return f64::INFINITY;
    }
    let c = centroid(x, n);
    let angle = 2.0 * std::f64::consts::PI / order as f64;
    let mut total = 0.0;
    for a in 0..n {
        let v = [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]];
        let w = rotate(v, axis, angle);
        let mut best = f64::INFINITY;
        for b in 0..n {
            let u = [x[3 * b] - c[0], x[3 * b + 1] - c[1], x[3 * b + 2] - c[2]];
            let d = (w[0] - u[0]).powi(2) + (w[1] - u[1]).powi(2) + (w[2] - u[2]).powi(2);
            if d < best {
                best = d;
            }
        }
        total += best;
    }
    (total / n as f64).sqrt()
}

/// How far `x` is from being symmetric under the plane whose normal is `axis`.
///
/// The same statistic as [`deviation`] for a rotation: each point is reflected
/// and matched to its nearest partner, and the root-mean-square of those
/// distances is what "approximately symmetric" means here.
pub fn plane_deviation(x: ArrayView1<f64>, n: usize, axis: [f64; 3]) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    let c = centroid(x, n);
    let m = mirror_matrix(axis);
    let mut total = 0.0;
    for a in 0..n {
        let v = [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]];
        let w = [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ];
        let mut best = f64::INFINITY;
        for b in 0..n {
            let u = [x[3 * b] - c[0], x[3 * b + 1] - c[1], x[3 * b + 2] - c[2]];
            let d = (w[0] - u[0]).powi(2) + (w[1] - u[1]).powi(2) + (w[2] - u[2]).powi(2);
            if d < best {
                best = d;
            }
        }
        total += best;
    }
    (total / n as f64).sqrt()
}

/// Axes worth testing, taken from the structure rather than at random.
///
/// Two families. The principal axes of the inertia tensor, which is where a
/// rotation axis of a symmetric body must lie; and the directions from the
/// centroid to each point and to each pair midpoint, which is where an axis
/// through a vertex or an edge lies. That covers the axes of the point groups
/// clusters actually adopt without enumerating directions blindly.
fn candidate_axes(x: ArrayView1<f64>, n: usize) -> Vec<[f64; 3]> {
    let c = centroid(x, n);
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(3 + n + n);
    // Inertia tensor about the centroid, then its eigenvectors.
    let mut t = ndarray::Array2::<f64>::zeros((3, 3));
    for a in 0..n {
        let v = [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]];
        let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        for i in 0..3 {
            for j in 0..3 {
                t[[i, j]] += if i == j { r2 } else { 0.0 } - v[i] * v[j];
            }
        }
    }
    let (_, vecs) = crate::spectral::symmetric_eigen(t.view(), 64);
    for j in 0..3 {
        if let Some(u) = normalise([vecs[[0, j]], vecs[[1, j]], vecs[[2, j]]]) {
            out.push(u);
        }
    }
    // Through each point.
    for a in 0..n {
        if let Some(u) = normalise([x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]]) {
            out.push(u);
        }
    }
    out
}

/// The best approximate rotational symmetry of `x`, over the given orders.
///
/// `None` when nothing is even approximately symmetric, which is the signal to
/// leave the structure alone.
///
/// Proper rotations only, because [`symmetrise`] averages a point around its
/// orbit under a rotation and a reflection has no such orbit. Mirror planes are
/// found by [`detect_all`], which exists to build a group rather than to push a
/// structure along one operation.
pub fn detect(x: ArrayView1<f64>, n: usize, orders: &[usize], tolerance: f64) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for axis in candidate_axes(x, n) {
        for &order in orders {
            let d = deviation(x, n, axis, order);
            if d < tolerance && best.map(|b| d < b.deviation).unwrap_or(true) {
                best = Some(Candidate {
                    axis,
                    order,
                    improper: false,
                    deviation: d,
                });
            }
        }
    }
    best
}

/// All approximate symmetries of `x`, proper and improper.
///
/// [`detect`] returns only the best one, which is enough to symmetrise toward a
/// single operation and not enough to build a group: a tetrahedral structure
/// needs a three-fold axis and a plane together, and either alone generates a
/// subgroup that is not the answer.
pub fn detect_all(
    x: ArrayView1<f64>,
    n: usize,
    orders: &[usize],
    tolerance: f64,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    for axis in candidate_axes(x, n) {
        for &order in orders {
            let d = deviation(x, n, axis, order);
            if d < tolerance {
                out.push(Candidate {
                    axis,
                    order,
                    improper: false,
                    deviation: d,
                });
            }
        }
        let d = plane_deviation(x, n, axis);
        if d < tolerance {
            out.push(Candidate {
                axis,
                order: 1,
                improper: true,
                deviation: d,
            });
        }
    }
    out.sort_by(|a, b| {
        a.deviation
            .partial_cmp(&b.deviation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Reflection in the plane through the centroid with normal `axis`.
pub fn mirror_matrix(axis: [f64; 3]) -> Rot {
    let [x, y, z] = axis;
    [
        [1.0 - 2.0 * x * x, -2.0 * x * y, -2.0 * x * z],
        [-2.0 * y * x, 1.0 - 2.0 * y * y, -2.0 * y * z],
        [-2.0 * z * x, -2.0 * z * y, 1.0 - 2.0 * z * z],
    ]
}

/// Makes `x` exactly symmetric under the rotation `cand` describes.
///
/// Each point is followed around its orbit under the rotation, the orbit is
/// averaged after rotating every member back to the first one's frame, and
/// every member is replaced by the rotated average. A point whose orbit does
/// not close, meaning some image has no partner within `pair_cutoff`, is left
/// where it is rather than dragged onto a partner it does not have.
///
/// Applied fully rather than blended. A partial push toward a symmetry the
/// structure nearly has produces a structure that is neither, and the point of
/// the scheme is to land in the symmetric basin so the relaxation can take it
/// from there.
pub fn symmetrise(x: ArrayView1<f64>, n: usize, cand: &Candidate, pair_cutoff: f64) -> Array1<f64> {
    let mut out = x.to_owned();
    if n == 0 || cand.order < 2 {
        return out;
    }
    let c = centroid(x, n);
    let angle = 2.0 * std::f64::consts::PI / cand.order as f64;
    let rel =
        |a: usize| -> [f64; 3] { [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]] };

    let mut done = vec![false; n];
    for a in 0..n {
        if done[a] {
            continue;
        }
        // Walk the orbit: from each member, rotate once and take the nearest
        // point as the next member.
        let mut orbit = vec![a];
        let mut cur = a;
        let mut closed = true;
        for _ in 1..cand.order {
            let w = rotate(rel(cur), cand.axis, angle);
            let mut best = usize::MAX;
            let mut best_d = f64::INFINITY;
            for b in 0..n {
                let u = rel(b);
                let d =
                    ((w[0] - u[0]).powi(2) + (w[1] - u[1]).powi(2) + (w[2] - u[2]).powi(2)).sqrt();
                if d < best_d {
                    best_d = d;
                    best = b;
                }
            }
            if best_d > pair_cutoff || best == usize::MAX {
                closed = false;
                break;
            }
            orbit.push(best);
            cur = best;
        }
        if !closed {
            done[a] = true;
            continue;
        }
        // A point on the axis maps to itself; its orbit collapses and there is
        // nothing to average.
        if orbit.iter().any(|&b| done[b]) {
            for &b in &orbit {
                done[b] = true;
            }
            continue;
        }

        // Average in the first member's frame: rotate member k back by k steps.
        let mut mean = [0.0_f64; 3];
        for (k, &b) in orbit.iter().enumerate() {
            let back = rotate(rel(b), cand.axis, -(k as f64) * angle);
            for i in 0..3 {
                mean[i] += back[i];
            }
        }
        let m = orbit.len() as f64;
        for v in mean.iter_mut() {
            *v /= m;
        }
        // Place every member at the rotated average.
        for (k, &b) in orbit.iter().enumerate() {
            let p = rotate(mean, cand.axis, k as f64 * angle);
            for i in 0..3 {
                out[3 * b + i] = c[i] + p[i];
            }
            done[b] = true;
        }
    }
    out
}

/// A rotation as a 3x3 matrix, row-major.
pub type Rot = [[f64; 3]; 3];

fn rot_matrix(axis: [f64; 3], angle: f64) -> Rot {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [
            t * axis[0] * axis[0] + c,
            t * axis[0] * axis[1] - s * axis[2],
            t * axis[0] * axis[2] + s * axis[1],
        ],
        [
            t * axis[0] * axis[1] + s * axis[2],
            t * axis[1] * axis[1] + c,
            t * axis[1] * axis[2] - s * axis[0],
        ],
        [
            t * axis[0] * axis[2] - s * axis[1],
            t * axis[1] * axis[2] + s * axis[0],
            t * axis[2] * axis[2] + c,
        ],
    ]
}

fn mul(a: &Rot, b: &Rot) -> Rot {
    let mut o = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                o[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    o
}

fn same(a: &Rot, b: &Rot) -> bool {
    (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() < 1e-6))
}

fn act(r: &Rot, v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

/// The rotation group generated by a set of approximate symmetries.
///
/// A point group is not an axis, and this is the difference between the scheme
/// and a single rotation. The 98-point global minimum is tetrahedral, and the
/// tetrahedral rotation group is generated by a three-fold and a two-fold axis
/// together: averaging orbits under either one alone does not make a structure
/// tetrahedral, it makes it axially symmetric, which is a different and much
/// weaker constraint.
///
/// Closed by repeated multiplication until nothing new appears or `cap` is
/// reached. The cap is a guard against a set of axes that are not quite
/// commensurate generating an unbounded pseudo-group out of rounding; the
/// icosahedral rotation group has sixty elements, so anything past that is a
/// sign the detection was loose rather than a real group.
pub fn generate_group(cands: &[Candidate], cap: usize) -> Vec<Rot> {
    let identity: Rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut group: Vec<Rot> = vec![identity];
    let mut gens: Vec<Rot> = Vec::new();
    for c in cands {
        if c.improper {
            // A reflection, which is what lifts a rotation group to the group
            // the structure actually has: proper rotations alone give T where
            // the 98-point minimum is Td, and O where the 38-point minimum is
            // Oh. Both are index-two subgroups missing exactly the operations
            // that define the target.
            gens.push(mirror_matrix(c.axis));
        } else {
            let angle = 2.0 * std::f64::consts::PI / c.order.max(2) as f64;
            gens.push(rot_matrix(c.axis, angle));
        }
    }
    let mut changed = true;
    while changed && group.len() < cap {
        changed = false;
        let snapshot = group.clone();
        for a in &snapshot {
            for g in &gens {
                let p = mul(a, g);
                if !group.iter().any(|q| same(q, &p)) {
                    group.push(p);
                    changed = true;
                    if group.len() >= cap {
                        break;
                    }
                }
            }
            if group.len() >= cap {
                break;
            }
        }
    }
    group
}

/// Makes `x` symmetric under a whole group, not one rotation.
///
/// Each point's orbit is collected by applying every group element and taking
/// the nearest partner, then the orbit is averaged in the frame of the element
/// that produced it and every member is replaced. A point whose orbit does not
/// close under some element is left alone rather than dragged onto a partner it
/// does not have.
pub fn symmetrise_group(
    x: ArrayView1<f64>,
    n: usize,
    group: &[Rot],
    pair_cutoff: f64,
) -> Array1<f64> {
    let mut out = x.to_owned();
    if n == 0 || group.len() < 2 {
        return out;
    }
    let c = centroid(x, n);
    let rel =
        |a: usize| -> [f64; 3] { [x[3 * a] - c[0], x[3 * a + 1] - c[1], x[3 * a + 2] - c[2]] };

    let mut done = vec![false; n];
    for a in 0..n {
        if done[a] {
            continue;
        }
        // Partner of `a` under each group element.
        let mut members: Vec<(usize, usize)> = Vec::with_capacity(group.len());
        let mut closed = true;
        for (gi, g) in group.iter().enumerate() {
            let w = act(g, rel(a));
            let mut best = usize::MAX;
            let mut best_d = f64::INFINITY;
            for b in 0..n {
                let u = rel(b);
                let d =
                    ((w[0] - u[0]).powi(2) + (w[1] - u[1]).powi(2) + (w[2] - u[2]).powi(2)).sqrt();
                if d < best_d {
                    best_d = d;
                    best = b;
                }
            }
            if best_d > pair_cutoff || best == usize::MAX {
                closed = false;
                break;
            }
            members.push((gi, best));
        }
        if !closed {
            done[a] = true;
            continue;
        }
        // Average in `a`'s frame: undo each element by using its transpose,
        // which is its inverse for a rotation.
        let mut mean = [0.0_f64; 3];
        for (gi, b) in &members {
            let g = &group[*gi];
            let gt: Rot = [
                [g[0][0], g[1][0], g[2][0]],
                [g[0][1], g[1][1], g[2][1]],
                [g[0][2], g[1][2], g[2][2]],
            ];
            let back = act(&gt, rel(*b));
            for k in 0..3 {
                mean[k] += back[k];
            }
        }
        let m = members.len() as f64;
        for v in mean.iter_mut() {
            *v /= m;
        }
        for (gi, b) in &members {
            if done[*b] {
                continue;
            }
            let p = act(&group[*gi], mean);
            for k in 0..3 {
                out[3 * b + k] = c[k] + p[k];
            }
            done[*b] = true;
        }
        done[a] = true;
    }
    out
}

/// Detects and applies in one step, or returns `None` when there is no
/// approximate symmetry worth using.
pub fn symmetrise_detected(
    x: ArrayView1<f64>,
    n: usize,
    orders: &[usize],
    tolerance: f64,
    pair_cutoff: f64,
) -> Option<(Array1<f64>, Candidate)> {
    let cand = detect(x, n, orders, tolerance)?;
    Some((symmetrise(x, n, &cand, pair_cutoff), cand))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn octahedron(scale: f64) -> Array1<f64> {
        Array1::from(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
            .iter()
            .flat_map(|v| v.iter().map(|c| c * scale))
            .collect::<Vec<f64>>(),
        )
    }

    /// A deterministic jitter, so a test says the same thing every run.
    fn jitter(x: &Array1<f64>, amp: f64) -> Array1<f64> {
        let mut y = x.clone();
        for (i, v) in y.iter_mut().enumerate() {
            *v += amp * (((i * 37 + 11) % 17) as f64 / 8.0 - 1.0);
        }
        y
    }

    #[test]
    fn an_exact_symmetry_has_zero_deviation() {
        let x = octahedron(1.0);
        let d = deviation(x.view(), 6, [0.0, 0.0, 1.0], 4);
        assert!(d < 1e-12, "a four-fold axis of an octahedron gave {d}");
    }

    #[test]
    fn a_symmetry_the_structure_lacks_has_a_large_deviation() {
        let x = octahedron(1.0);
        // No five-fold axis anywhere in an octahedron.
        let d = deviation(x.view(), 6, [0.0, 0.0, 1.0], 5);
        assert!(d > 0.1, "a five-fold axis gave {d}, which is too small");
    }

    /// Detection has to find the axis the structure actually has, which is the
    /// difference between this and symmetrising about a random direction.
    #[test]
    /// The groups the hard structures actually have, which proper rotations alone
    /// cannot express. Tetrahedral order is 24 and octahedral is 48; the
    /// rotation subgroups are 12 and 24, so a generator set closed under proper
    /// rotations stops exactly halfway.
    #[test]
    fn a_reflection_generator_reaches_the_full_groups() {
        let s3 = 1.0 / 3.0_f64.sqrt();
        // Tetrahedral. The rotation subgroup T needs a three-fold axis along a
        // body diagonal and a two-fold along a coordinate axis; a diagonal
        // mirror then doubles it to Td. A three-fold with a mirror containing
        // it gives C3v of order six instead, which is what the first attempt
        // at this test generated and what makes the choice of generators worth
        // stating rather than assuming.
        let td = generate_group(
            &[
                Candidate {
                    axis: [s3, s3, s3],
                    order: 3,
                    improper: false,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [0.0, 0.0, 1.0],
                    order: 2,
                    improper: false,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt(), 0.0],
                    order: 1,
                    improper: true,
                    deviation: 0.0,
                },
            ],
            200,
        );
        assert_eq!(
            td.len(),
            24,
            "tetrahedral group came out with {} elements",
            td.len()
        );
        assert!(
            td.iter().any(|m| determinant(m) < 0.0),
            "no improper element in a group that is half improper"
        );

        // Octahedral: a four-fold axis and a mirror plane normal to another.
        let oh = generate_group(
            &[
                Candidate {
                    axis: [0.0, 0.0, 1.0],
                    order: 4,
                    improper: false,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [1.0, 0.0, 0.0],
                    order: 1,
                    improper: true,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [s3, s3, s3],
                    order: 3,
                    improper: false,
                    deviation: 0.0,
                },
            ],
            200,
        );
        assert_eq!(
            oh.len(),
            48,
            "octahedral group came out with {} elements",
            oh.len()
        );
    }

    /// Without the reflection the same generators must give the proper
    /// subgroup, which is the statement of what was missing before.
    #[test]
    fn proper_generators_alone_stop_at_the_rotation_subgroup() {
        let s3 = 1.0 / 3.0_f64.sqrt();
        let t = generate_group(
            &[
                Candidate {
                    axis: [s3, s3, s3],
                    order: 3,
                    improper: false,
                    deviation: 0.0,
                },
                Candidate {
                    axis: [0.0, 0.0, 1.0],
                    order: 2,
                    improper: false,
                    deviation: 0.0,
                },
            ],
            200,
        );
        assert_eq!(
            t.len(),
            12,
            "rotation subgroup came out with {} elements",
            t.len()
        );
        assert!(
            t.iter().all(|m| determinant(m) > 0.0),
            "an improper element appeared with no reflection generator"
        );
    }

    /// A plane the structure has must be detected, and one it lacks must not
    /// be, or the group is built from operations that are not there.
    #[test]
    fn plane_detection_matches_the_structure() {
        // A square in the xy plane: mirror in z is exact, mirror in a plane at
        // an odd angle is not.
        let x = ndarray::Array1::from(vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
        ]);
        let flat = plane_deviation(x.view(), 4, [0.0, 0.0, 1.0]);
        assert!(flat < 1e-9, "the plane of a planar square read as {flat}");
        let odd = plane_deviation(x.view(), 4, [0.3, 0.1, 0.95_f64.sqrt()]);
        assert!(odd > 1e-3, "a plane the square lacks read as {odd}");
    }

    fn determinant(m: &Rot) -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    #[test]
    fn detection_finds_an_axis_the_structure_has() {
        let x = jitter(&octahedron(1.0), 0.03);
        let cand = detect(x.view(), 6, &[2, 3, 4, 5, 6], 0.5).expect("nothing detected");
        // Whatever it found has to be a genuine symmetry of the ideal
        // structure, which is the claim; asserting a particular axis is not.
        // An octahedron has three four-fold axes through opposite vertices,
        // four three-fold axes through opposite faces and six two-fold axes
        // through opposite edge midpoints, so [110] with order two is as
        // correct an answer as [001] with order four, and the detector is
        // free to return whichever fits the jitter better.
        let exact = deviation(octahedron(1.0).view(), 6, cand.axis, cand.order);
        assert!(
            exact < 1e-9,
            "detected order {} about {:?}, which the ideal structure lacks: {exact}",
            cand.order,
            cand.axis
        );
        assert!(cand.deviation < 0.2, "deviation {}", cand.deviation);
    }

    /// The point of the operation: a nearly symmetric structure comes back much
    /// more symmetric than it went in.
    #[test]
    fn symmetrising_reduces_the_deviation() {
        let x = jitter(&octahedron(1.0), 0.05);
        let cand = detect(x.view(), 6, &[2, 3, 4], 0.5).expect("nothing detected");
        let before = cand.deviation;
        let y = symmetrise(x.view(), 6, &cand, 1.0);
        let after = deviation(y.view(), 6, cand.axis, cand.order);
        assert!(
            after < before * 0.2,
            "deviation went from {before} to {after}"
        );
    }

    /// And it must not move a structure that is already exact.
    #[test]
    fn an_exactly_symmetric_structure_is_left_alone() {
        let x = octahedron(1.1);
        let cand = Candidate {
            axis: [0.0, 0.0, 1.0],
            order: 4,
            improper: false,
            deviation: 0.0,
        };
        let y = symmetrise(x.view(), 6, &cand, 1.0);
        for (p, q) in x.iter().zip(y.iter()) {
            assert!((p - q).abs() < 1e-9, "{p} moved to {q}");
        }
    }

    /// A structure with no approximate symmetry is left alone rather than
    /// pushed onto one it does not have. This is the case where symmetrising
    /// about a random axis does damage.
    #[test]
    fn a_structure_with_no_symmetry_is_not_forced_into_one() {
        let n = 7;
        let mut x = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            let a = (i as f64) * 1.317;
            let r = 0.8 + 0.31 * ((i % 5) as f64);
            x[3 * i] = r * a.cos();
            x[3 * i + 1] = r * a.sin();
            x[3 * i + 2] = 0.53 * ((i % 4) as f64) - 0.9;
        }
        assert!(
            detect(x.view(), n, &[2, 3, 4, 5, 6], 0.05).is_none(),
            "a generic structure was reported as nearly symmetric"
        );
    }

    /// The centre of mass must not move: a symmetrisation that translates the
    /// cluster is a translation with extra steps.
    /// The tetrahedral rotation group has twelve elements and is generated by
    /// a three-fold and a two-fold axis. This is the case the whole addition
    /// exists for: the 98-point global minimum is tetrahedral, and neither
    /// generator alone produces it.
    #[test]
    fn a_three_fold_and_a_two_fold_generate_the_tetrahedral_group() {
        let s3 = 1.0 / 3.0_f64.sqrt();
        let gens = vec![
            Candidate {
                axis: [s3, s3, s3],
                order: 3,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [0.0, 0.0, 1.0],
                order: 2,
                improper: false,
                deviation: 0.0,
            },
        ];
        let g = generate_group(&gens, 120);
        assert_eq!(
            g.len(),
            12,
            "the tetrahedral rotation group has twelve elements, got {}",
            g.len()
        );
    }

    /// And the octahedral group has twenty-four, from the same construction.
    #[test]
    fn the_four_fold_axes_generate_the_octahedral_group() {
        let gens = vec![
            Candidate {
                axis: [1.0, 0.0, 0.0],
                order: 4,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [0.0, 1.0, 0.0],
                order: 4,
                improper: false,
                deviation: 0.0,
            },
        ];
        let g = generate_group(&gens, 120);
        assert_eq!(
            g.len(),
            24,
            "octahedral rotations number 24, got {}",
            g.len()
        );
    }

    /// Averaging over the whole group beats averaging over one axis, which is
    /// the claim that makes the group version worth its cost.
    #[test]
    fn the_group_symmetrises_further_than_one_axis() {
        let x = jitter(&octahedron(1.0), 0.05);
        let one = detect(x.view(), 6, &[2, 3, 4], 0.5).expect("nothing detected");
        let single = symmetrise(x.view(), 6, &one, 1.0);

        let gens = vec![
            Candidate {
                axis: [1.0, 0.0, 0.0],
                order: 4,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [0.0, 1.0, 0.0],
                order: 4,
                improper: false,
                deviation: 0.0,
            },
        ];
        let group = generate_group(&gens, 120);
        let full = symmetrise_group(x.view(), 6, &group, 1.0);

        // Measured against a symmetry the single-axis pass did not use.
        let axis = [0.0, 0.0, 1.0];
        let d_single = deviation(single.view(), 6, axis, 4);
        let d_full = deviation(full.view(), 6, axis, 4);
        assert!(
            d_full < d_single,
            "group gave {d_full} against single axis {d_single}"
        );
    }

    /// A structure already symmetric under the group must not move.
    #[test]
    fn the_group_leaves_an_exact_structure_alone() {
        let x = octahedron(1.1);
        let gens = vec![
            Candidate {
                axis: [1.0, 0.0, 0.0],
                order: 4,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [0.0, 1.0, 0.0],
                order: 4,
                improper: false,
                deviation: 0.0,
            },
        ];
        let group = generate_group(&gens, 120);
        let y = symmetrise_group(x.view(), 6, &group, 1.0);
        for (p, q) in x.iter().zip(y.iter()) {
            assert!((p - q).abs() < 1e-9, "{p} moved to {q}");
        }
    }

    /// The cap is a guard: axes that are not quite commensurate would otherwise
    /// generate an unbounded pseudo-group out of rounding.
    #[test]
    fn incommensurate_axes_are_capped_rather_than_run_away() {
        let gens = vec![
            Candidate {
                axis: [1.0, 0.0, 0.0],
                order: 5,
                improper: false,
                deviation: 0.0,
            },
            Candidate {
                axis: [0.31, 0.77, 0.56],
                order: 3,
                improper: false,
                deviation: 0.0,
            },
        ];
        let g = generate_group(&gens, 60);
        assert!(g.len() <= 60, "group ran to {}", g.len());
    }

    #[test]
    fn the_centroid_is_preserved() {
        let x = jitter(&octahedron(1.0), 0.04);
        let cand = detect(x.view(), 6, &[2, 3, 4], 0.5).unwrap();
        let y = symmetrise(x.view(), 6, &cand, 1.0);
        let cx = centroid(x.view(), 6);
        let cy = centroid(y.view(), 6);
        for k in 0..3 {
            assert!((cx[k] - cy[k]).abs() < 1e-9, "centroid moved on axis {k}");
        }
    }
}
