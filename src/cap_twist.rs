// Copyright (c) anneal developers
//! Turn one cap of a cluster about a local symmetry axis by half a turn of
//! that axis.
//!
//! The close-packed morphologies a cluster minimum adopts differ by how
//! layers sit on one another. Two stackings about a k-fold axis are related
//! by a turn of pi / k: an fcc and an hcp stacking about a threefold axis,
//! an icosahedral and a decahedral cap about a fivefold one. A turn by a
//! whole 2 pi / k maps the order onto itself and changes nothing.
//!
//! Every quantity is read off the configuration: the length scale from the
//! nearest-neighbour distance, the pivot from the better-coordinated points,
//! the axis and its order from the pivot's own neighbour shell, and the cap
//! as the points beyond half a layer from the pivot along that axis. No
//! structure, motif or size enters.

use ndarray::{Array1, ArrayView1};
use rand::Rng;

/// Neighbour cutoff in units of the nearest-neighbour distance, the first
/// minimum of a close-packed radial distribution.
const SHELL: f64 = 1.35;

/// Orders of the local axes looked for.
const ORDERS: [usize; 4] = [3, 4, 5, 6];

/// Scores closer than this, in nearest-neighbour units, are a tie.
const TIE: f64 = 1e-9;

/// A proposal: `x` with one cap turned, or `x` itself when no cap is found.
pub fn propose<R: Rng + ?Sized>(x: ArrayView1<f64>, n: usize, rng: &mut R) -> Array1<f64> {
    if n < 4 || x.len() != 3 * n {
        return x.to_owned();
    }
    let scale = crate::lattice::nearest_neighbour_scale(x);
    if !(scale.is_finite() && scale > 0.0) {
        return x.to_owned();
    }
    let shells = shells(x, n, scale);
    let mut counts: Vec<usize> = shells.iter().map(Vec::len).collect();
    counts.sort_unstable();
    let median = counts[n / 2];
    let pivots: Vec<usize> = (0..n)
        .filter(|&i| shells[i].len() >= median.max(3))
        .collect();
    if pivots.is_empty() {
        return x.to_owned();
    }
    let pivot = pivots[rng.random_range(0..pivots.len())];
    let upward = rng.random::<bool>();
    let forward = rng.random::<bool>();
    turn(x, n, scale, &shells[pivot], pivot, upward, forward).unwrap_or_else(|| x.to_owned())
}

/// Neighbours of each point within [`SHELL`] nearest-neighbour distances.
fn shells(x: ArrayView1<f64>, n: usize, scale: f64) -> Vec<Vec<usize>> {
    let reach2 = (SHELL * scale) * (SHELL * scale);
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i && dist2(point(x, i), point(x, j)) <= reach2)
                .collect()
        })
        .collect()
}

/// Turns the cap beyond half a layer from `pivot` along its local axis.
///
/// `upward` picks the side of the pivot the cap is on and `forward` the
/// sense of the turn. `None` when the pivot's shell has no axis, no layer
/// above the pivot, or a cut that moves no point or every point.
fn turn(
    x: ArrayView1<f64>,
    n: usize,
    scale: f64,
    shell: &[usize],
    pivot: usize,
    upward: bool,
    forward: bool,
) -> Option<Array1<f64>> {
    let centre = point(x, pivot);
    let offsets: Vec<[f64; 3]> = shell
        .iter()
        .map(|&j| scaled(sub(point(x, j), centre), 1.0 / scale))
        .collect();
    let (axis, order) = local_axis(&offsets)?;
    let axis = if upward { axis } else { scaled(axis, -1.0) };
    let layer = offsets
        .iter()
        .map(|o| dot(*o, axis))
        .filter(|&h| h > 0.25)
        .min_by(f64::total_cmp)?;
    let cut = 0.5 * layer * scale;
    let sign = if forward { 1.0 } else { -1.0 };
    let (s, c) = (sign * std::f64::consts::PI / order as f64).sin_cos();
    let mut out = x.to_owned();
    let mut moved = 0usize;
    for j in 0..n {
        let v = sub(point(x, j), centre);
        if dot(v, axis) <= cut {
            continue;
        }
        let r = rotate(v, axis, s, c);
        for k in 0..3 {
            out[3 * j + k] = centre[k] + r[k];
        }
        moved += 1;
    }
    (moved > 0 && moved < n).then_some(out)
}

fn point(x: ArrayView1<f64>, i: usize) -> [f64; 3] {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

/// The axis through the origin, and its order, under which `shell` is
/// closest to mapping onto itself.
///
/// Candidates are the directions to each shell point, to each pair
/// midpoint and to each triangle centroid of mutually bonded points, which
/// is where a vertex, edge or face axis of a shell lies. The score is the
/// root-mean-square distance from each turned point to its nearest shell
/// point.
fn local_axis(shell: &[[f64; 3]]) -> Option<([f64; 3], usize)> {
    let m = shell.len();
    if m < 3 {
        return None;
    }
    let bonded = |a: usize, b: usize| dist2(shell[a], shell[b]) <= SHELL * SHELL;
    let mut axes: Vec<[f64; 3]> = shell.to_vec();
    for a in 0..m {
        for b in a + 1..m {
            if !bonded(a, b) {
                continue;
            }
            axes.push(add(shell[a], shell[b]));
            for c in b + 1..m {
                if bonded(a, c) && bonded(b, c) {
                    axes.push(add(add(shell[a], shell[b]), shell[c]));
                }
            }
        }
    }
    let mut best: Option<([f64; 3], usize, f64)> = None;
    for raw in axes {
        let norm = dot(raw, raw).sqrt();
        if norm < 1e-9 {
            continue;
        }
        let axis = scaled(raw, 1.0 / norm);
        for order in ORDERS {
            let angle = 2.0 * std::f64::consts::PI / order as f64;
            let (s, c) = angle.sin_cos();
            let mut total = 0.0;
            for q in shell {
                let r = rotate(*q, axis, s, c);
                let near = shell
                    .iter()
                    .map(|t| dist2(r, *t))
                    .fold(f64::INFINITY, f64::min);
                total += near;
            }
            let score = (total / m as f64).sqrt();
            // Within rounding of a tie the higher order wins: an icosahedral
            // shell maps onto itself under its threefold axes as well as its
            // fivefold ones, and the fivefold is the axis its stacking turns
            // about.
            let better = best
                .is_none_or(|(_, held, b)| score < b - TIE || (score <= b + TIE && order > held));
            if better {
                best = Some((axis, order, score));
            }
        }
    }
    best.map(|(axis, order, _)| (axis, order))
}

fn rotate(v: [f64; 3], k: [f64; 3], s: f64, c: f64) -> [f64; 3] {
    let kv = dot(k, v);
    let cross = [
        k[1] * v[2] - k[2] * v[1],
        k[2] * v[0] - k[0] * v[2],
        k[0] * v[1] - k[1] * v[0],
    ];
    [
        v[0] * c + cross[0] * s + k[0] * kv * (1.0 - c),
        v[1] * c + cross[1] * s + k[1] * kv * (1.0 - c),
        v[2] * c + cross[2] * s + k[2] * kv * (1.0 - c),
    ]
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scaled(a: [f64; 3], f: f64) -> [f64; 3] {
    [a[0] * f, a[1] * f, a[2] * f]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const GOLDEN: f64 = 1.618_033_988_749_895;

    /// A centred 13-point icosahedron of unit circumradius with a vertex on
    /// the z axis.
    fn icosahedron() -> Array1<f64> {
        let mut v: Vec<[f64; 3]> = Vec::new();
        for a in [-1.0, 1.0] {
            for b in [-GOLDEN, GOLDEN] {
                v.push([0.0, a, b]);
                v.push([a, b, 0.0]);
                v.push([b, 0.0, a]);
            }
        }
        // Rotate a vertex, (0, 1, golden), onto +z.
        let t = [0.0, 1.0, GOLDEN];
        let norm = dot(t, t).sqrt();
        let z = [0.0, 0.0, 1.0];
        let k = [
            t[1] * z[2] - t[2] * z[1],
            t[2] * z[0] - t[0] * z[2],
            t[0] * z[1] - t[1] * z[0],
        ];
        let kn = dot(k, k).sqrt();
        let k = scaled(k, 1.0 / kn);
        let angle = (dot(t, z) / norm).acos();
        let (s, c) = angle.sin_cos();
        let mut x = vec![0.0, 0.0, 0.0];
        for p in v {
            let r = scaled(rotate(p, k, s, c), 1.0 / norm);
            x.extend(r);
        }
        Array1::from(x)
    }

    /// A centred 13-point cuboctahedron, the fcc first shell.
    fn cuboctahedron() -> Array1<f64> {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let mut x = vec![0.0, 0.0, 0.0];
        for a in [-h, h] {
            for b in [-h, h] {
                x.extend([a, b, 0.0]);
                x.extend([a, 0.0, b]);
                x.extend([0.0, a, b]);
            }
        }
        Array1::from(x)
    }

    fn offsets(x: &Array1<f64>, i: usize) -> Vec<[f64; 3]> {
        let n = x.len() / 3;
        let scale = crate::lattice::nearest_neighbour_scale(x.view());
        shells(x.view(), n, scale)[i]
            .iter()
            .map(|&j| scaled(sub(point(x.view(), j), point(x.view(), i)), 1.0 / scale))
            .collect()
    }

    #[test]
    fn the_icosahedral_shell_has_a_fivefold_axis_and_the_fcc_shell_a_threefold_one() {
        let ico = icosahedron();
        let (axis, order) = local_axis(&offsets(&ico, 0)).expect("axis");
        assert_eq!(order, 5);
        let along = offsets(&ico, 0)
            .iter()
            .map(|o| (dot(*o, axis) / dot(*o, *o).sqrt()).abs())
            .fold(0.0, f64::max);
        assert!(
            (along - 1.0).abs() < 1e-9,
            "a fivefold axis runs through a vertex"
        );
        let (_, order) = local_axis(&offsets(&cuboctahedron(), 0)).expect("axis");
        assert_eq!(order, 3);
    }

    /// Turning the upper half of an icosahedron by a tenth of a turn about
    /// its fivefold axis eclipses the two pentagons: the decahedral order.
    #[test]
    fn a_half_turn_of_the_fivefold_axis_eclipses_the_icosahedral_rings() {
        let x = icosahedron();
        let scale = crate::lattice::nearest_neighbour_scale(x.view());
        let shell = &shells(x.view(), 13, scale)[0];
        for upward in [true, false] {
            for forward in [true, false] {
                let y = turn(x.view(), 13, scale, shell, 0, upward, forward).expect("turn");
                let azimuth = |i: usize| y[3 * i + 1].atan2(y[3 * i]);
                let ring = |sign: f64| -> Vec<f64> {
                    (1..13)
                        .filter(|&i| y[3 * i + 2] * sign > 0.1 && y[3 * i + 2] * sign < 0.9)
                        .map(azimuth)
                        .collect()
                };
                let (upper, lower) = (ring(1.0), ring(-1.0));
                assert_eq!((upper.len(), lower.len()), (5, 5));
                for a in &upper {
                    let near = lower
                        .iter()
                        .map(|b| {
                            let d = (a - b).rem_euclid(std::f64::consts::TAU);
                            d.min(std::f64::consts::TAU - d)
                        })
                        .fold(f64::INFINITY, f64::min);
                    assert!(near < 1e-9, "upper ring point {a} is not over a lower one");
                }
            }
        }
    }

    #[test]
    fn the_cap_moves_rigidly_and_the_rest_stays() {
        let mut r = StdRng::seed_from_u64(3);
        let x = crate::methods::cluster_hopping::random_cluster_in_radius(38, 2.2, 0.7, &mut r);
        let mut changed = 0;
        for _ in 0..50 {
            let y = propose(x.view(), 38, &mut r);
            assert_eq!(y.len(), x.len());
            let moved: Vec<bool> = (0..38)
                .map(|i| (0..3).any(|k| y[3 * i + k] != x[3 * i + k]))
                .collect();
            if moved.iter().any(|&m| m) {
                changed += 1;
                assert!(!moved.iter().all(|&m| m));
            }
            for i in 0..38 {
                for j in 0..38 {
                    if moved[i] == moved[j] {
                        let a = dist2(point(x.view(), i), point(x.view(), j)).sqrt();
                        let b = dist2(point(y.view(), i), point(y.view(), j)).sqrt();
                        assert!((a - b).abs() < 1e-9, "distance {i}-{j} changed");
                    }
                }
            }
        }
        assert!(changed > 25, "only {changed} of 50 proposals turned a cap");
    }

    #[test]
    fn a_tiny_or_malformed_input_comes_back_unchanged() {
        let mut r = StdRng::seed_from_u64(1);
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(propose(x.view(), 3, &mut r), x);
        let y = Array1::from(vec![0.0; 12]);
        assert_eq!(propose(y.view(), 4, &mut r), y);
        assert_eq!(propose(x.view(), 4, &mut r), x);
    }
}
