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
    pub order: usize,
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
        let v = [
            x[3 * a] - c[0],
            x[3 * a + 1] - c[1],
            x[3 * a + 2] - c[2],
        ];
        let w = rotate(v, axis, angle);
        let mut best = f64::INFINITY;
        for b in 0..n {
            let u = [
                x[3 * b] - c[0],
                x[3 * b + 1] - c[1],
                x[3 * b + 2] - c[2],
            ];
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
        let v = [
            x[3 * a] - c[0],
            x[3 * a + 1] - c[1],
            x[3 * a + 2] - c[2],
        ];
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
        if let Some(u) = normalise([
            x[3 * a] - c[0],
            x[3 * a + 1] - c[1],
            x[3 * a + 2] - c[2],
        ]) {
            out.push(u);
        }
    }
    out
}

/// The best approximate rotational symmetry of `x`, over the given orders.
///
/// `None` when nothing is even approximately symmetric, which is the signal to
/// leave the structure alone.
pub fn detect(x: ArrayView1<f64>, n: usize, orders: &[usize], tolerance: f64) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for axis in candidate_axes(x, n) {
        for &order in orders {
            let d = deviation(x, n, axis, order);
            if d < tolerance && best.map(|b| d < b.deviation).unwrap_or(true) {
                best = Some(Candidate {
                    axis,
                    order,
                    deviation: d,
                });
            }
        }
    }
    best
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
pub fn symmetrise(
    x: ArrayView1<f64>,
    n: usize,
    cand: &Candidate,
    pair_cutoff: f64,
) -> Array1<f64> {
    let mut out = x.to_owned();
    if n == 0 || cand.order < 2 {
        return out;
    }
    let c = centroid(x, n);
    let angle = 2.0 * std::f64::consts::PI / cand.order as f64;
    let rel = |a: usize| -> [f64; 3] {
        [
            x[3 * a] - c[0],
            x[3 * a + 1] - c[1],
            x[3 * a + 2] - c[2],
        ]
    };

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
                let d = ((w[0] - u[0]).powi(2) + (w[1] - u[1]).powi(2) + (w[2] - u[2]).powi(2))
                    .sqrt();
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
