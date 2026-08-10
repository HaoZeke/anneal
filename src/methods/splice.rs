//! Cut-and-splice mixing of two quenched clusters.
//!
//! Lee, Lee and Scheraga, arXiv cond-mat/0307690. The published conformational
//! space annealing hit rates rest on this operator: a random plane through the
//! centroid region takes one side from parent A and the complementary side from
//! parent B, then repairs the atom count so N is preserved. The bank in
//! [`crate::methods::bank`] decides what to keep; this module is the mix.
//!
//! The result is a 3N trial, not a minimum. The caller quenches.

use ndarray::{Array1, ArrayView1};
use rand::Rng;

/// Mixes two quenched clusters by a random plane cut.
///
/// `a` and `b` are flattened 3N coordinates of the same N. `species`, when
/// given, is one label per point and is shared by both parents; the child keeps
/// that composition. `min_sep` is the closest approach enforced after the
/// splice, in the same length units as the coordinates.
///
/// The plane has a random orientation and an offset through the centroid
/// region. Atoms on one side come from `a`, the complementary side from `b`.
/// If that selection is not N atoms (or not the parent composition), unused
/// atoms nearest the complementary side of the donor fill the deficit.
pub fn cut_and_splice<R: Rng + ?Sized>(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    species: Option<&[u32]>,
    min_sep: f64,
    rng: &mut R,
) -> Array1<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "cut-and-splice parents must have the same length, got {} and {}",
        a.len(),
        b.len()
    );
    assert_eq!(
        a.len() % 3,
        0,
        "cluster coordinates are 3N, got length {}",
        a.len()
    );
    let n = a.len() / 3;
    if let Some(sp) = species {
        assert_eq!(
            sp.len(),
            n,
            "species must have one entry per point, got {} for N={n}",
            sp.len()
        );
    }
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1 {
        return if rng.random::<bool>() {
            a.to_owned()
        } else {
            b.to_owned()
        };
    }

    let (left, right) = if rng.random::<bool>() { (a, b) } else { (b, a) };

    let span = 0.5 * (extent(left, n) + extent(right, n));
    let mut last_normal = [0.0, 0.0, 1.0];
    let mut last_offset = 0.0;
    for _ in 0..8 {
        let normal = random_unit(rng);
        let offset = if span > 1e-12 {
            rng.random_range(-0.4 * span..0.4 * span)
        } else {
            0.0
        };
        last_normal = normal;
        last_offset = offset;
        let (child, from_left, from_right) =
            assemble(left, right, species, n, normal, offset, false);
        if from_left > 0 && from_right > 0 {
            return finish(child, n, min_sep);
        }
    }
    let (child, _, _) = assemble(left, right, species, n, last_normal, last_offset, true);
    finish(child, n, min_sep)
}

/// Mixes two quenched clusters with a prescribed cutting plane.
///
/// `normal` is a direction, not necessarily unit. `offset` is the signed
/// distance from the origin in the recentred frame: an atom at `r` sits on the
/// `a` side when `normal · r > offset`.
pub fn splice_at_plane(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    species: Option<&[u32]>,
    min_sep: f64,
    normal: [f64; 3],
    offset: f64,
) -> Array1<f64> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 3, 0);
    let n = a.len() / 3;
    if let Some(sp) = species {
        assert_eq!(sp.len(), n);
    }
    if n == 0 {
        return Array1::zeros(0);
    }
    let (child, _, _) = assemble(a, b, species, n, normal, offset, false);
    finish(child, n, min_sep)
}

fn assemble(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    species: Option<&[u32]>,
    n: usize,
    normal: [f64; 3],
    offset: f64,
    force_split: bool,
) -> (Vec<[f64; 3]>, usize, usize) {
    let ca = centroid(a, n);
    let cb = centroid(b, n);
    let nrm = match normalise(normal) {
        Some(v) => v,
        None => [0.0, 0.0, 1.0],
    };

    let label = |i: usize| species.map(|s| s[i]).unwrap_or(0);
    let mut types: Vec<u32> = (0..n).map(label).collect();
    types.sort_unstable();
    types.dedup();

    let mut child: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut from_a = 0usize;
    let mut from_b = 0usize;

    for &sp in &types {
        let mut a_idx: Vec<usize> = (0..n).filter(|&i| label(i) == sp).collect();
        let mut b_idx: Vec<usize> = (0..n).filter(|&i| label(i) == sp).collect();
        let target = a_idx.len();
        a_idx.sort_by(|&i, &j| {
            signed(a, i, ca, nrm, offset)
                .partial_cmp(&signed(a, j, ca, nrm, offset))
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });
        b_idx.sort_by(|&i, &j| {
            signed(b, i, cb, nrm, offset)
                .partial_cmp(&signed(b, j, cb, nrm, offset))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let plus = a_idx
            .iter()
            .copied()
            .filter(|&i| signed(a, i, ca, nrm, offset) > 0.0)
            .count();
        // A forced split leaves at least one atom of a multi-atom species for
        // the donor. A singleton species can come from either parent.
        let take_a = if force_split && target >= 2 {
            (target / 2).clamp(1, target - 1)
        } else {
            plus.min(target)
        };
        let take_b = target - take_a;

        for &i in a_idx.iter().take(take_a) {
            child.push(recentred_atom(a, i, ca));
            from_a += 1;
        }
        for &i in b_idx.iter().take(take_b) {
            child.push(recentred_atom(b, i, cb));
            from_b += 1;
        }
    }

    (child, from_a, from_b)
}

fn finish(mut child: Vec<[f64; 3]>, n: usize, min_sep: f64) -> Array1<f64> {
    // The assembler preserves N; a mismatch here is a programming error.
    debug_assert_eq!(child.len(), n);
    if child.len() != n {
        child.resize(n, [0.0; 3]);
    }
    recentre_points(&mut child);
    if min_sep > 0.0 {
        push_apart(&mut child, min_sep);
        recentre_points(&mut child);
    }
    let mut out = Array1::zeros(3 * n);
    for (i, p) in child.iter().enumerate() {
        for k in 0..3 {
            out[3 * i + k] = p[k];
        }
    }
    out
}

fn centroid(x: ArrayView1<f64>, n: usize) -> [f64; 3] {
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    let inv = 1.0 / n.max(1) as f64;
    [c[0] * inv, c[1] * inv, c[2] * inv]
}

fn recentred_atom(x: ArrayView1<f64>, i: usize, c: [f64; 3]) -> [f64; 3] {
    [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]]
}

fn signed(x: ArrayView1<f64>, i: usize, c: [f64; 3], n: [f64; 3], offset: f64) -> f64 {
    let p = recentred_atom(x, i, c);
    p[0] * n[0] + p[1] * n[1] + p[2] * n[2] - offset
}

fn extent(x: ArrayView1<f64>, n: usize) -> f64 {
    let c = centroid(x, n);
    let mut best = 0.0;
    for i in 0..n {
        let p = recentred_atom(x, i, c);
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if r > best {
            best = r;
        }
    }
    best
}

fn recentre_points(pts: &mut [[f64; 3]]) {
    if pts.is_empty() {
        return;
    }
    let n = pts.len() as f64;
    let mut c = [0.0; 3];
    for p in pts.iter() {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    for k in 0..3 {
        c[k] /= n;
    }
    for p in pts.iter_mut() {
        for k in 0..3 {
            p[k] -= c[k];
        }
    }
}

/// Pushes overlapping points apart to `min_sep`.
///
/// A trial with two points on top of each other has an enormous value under
/// any repulsive potential, and a quasi-Newton relaxation started there fails
/// on its first line search.
fn push_apart(pts: &mut [[f64; 3]], min_sep: f64) {
    let n = pts.len();
    for _ in 0..40 {
        let mut moved = false;
        for a in 0..n {
            for b in (a + 1)..n {
                let mut d = [0.0; 3];
                let mut r2 = 0.0;
                for k in 0..3 {
                    d[k] = pts[a][k] - pts[b][k];
                    r2 += d[k] * d[k];
                }
                let r = r2.sqrt();
                if r < min_sep && r > 1e-9 {
                    let push = 0.5 * (min_sep - r) / r;
                    for k in 0..3 {
                        pts[a][k] += push * d[k];
                        pts[b][k] -= push * d[k];
                    }
                    moved = true;
                } else if r <= 1e-9 {
                    pts[a][0] += 0.5 * min_sep;
                    pts[b][0] -= 0.5 * min_sep;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
}

fn normalise(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

fn random_unit<R: Rng + ?Sized>(rng: &mut R) -> [f64; 3] {
    // Gaussian 3-vector, normalised: uniform on the sphere, matching the
    // Box-Muller draws used to seed clusters.
    loop {
        let mut v = [0.0; 3];
        let mut n2 = 0.0;
        for k in 0..3 {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            v[k] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            n2 += v[k] * v[k];
        }
        let n = n2.sqrt();
        if n > 1e-12 {
            return [v[0] / n, v[1] / n, v[2] / n];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// A thirteen-point icosahedron: a centre and twelve vertices.
    fn icosahedron13() -> Array1<f64> {
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, p],
            [0.0, 1.0, -p],
            [0.0, -1.0, p],
            [0.0, -1.0, -p],
            [1.0, p, 0.0],
            [1.0, -p, 0.0],
            [-1.0, p, 0.0],
            [-1.0, -p, 0.0],
            [p, 0.0, 1.0],
            [-p, 0.0, 1.0],
            [p, 0.0, -1.0],
            [-p, 0.0, -1.0],
        ];
        let s = 1.0 / (1.0 + p * p).sqrt();
        let mut x = Array1::<f64>::zeros(3 * 13);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * (i + 1) + k] = s * v[k];
            }
        }
        x
    }

    /// Cuboctahedral 13-point fragment: a centre and its twelve fcc neighbours.
    fn cuboctahedron13() -> Array1<f64> {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, -1.0],
        ];
        let mut x = Array1::zeros(39);
        for (i, p) in pts.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = p[k] * 0.75;
            }
        }
        x
    }

    /// Pentagonal bipyramid, the seven-point Lennard-Jones global minimum.
    fn pentagonal_bipyramid7() -> Array1<f64> {
        let mut x = Array1::zeros(21);
        for i in 0..5 {
            let t = i as f64 * std::f64::consts::TAU / 5.0;
            x[3 * i] = t.cos();
            x[3 * i + 1] = t.sin();
            x[3 * i + 2] = 0.0;
        }
        x[15] = 0.0;
        x[16] = 0.0;
        x[17] = 1.0;
        x[18] = 0.0;
        x[19] = 0.0;
        x[20] = -1.0;
        x
    }

    /// Planar hexagon plus its centre: a compact 7-point shape that is not the
    /// pentagonal bipyramid.
    fn hexagon_plus_centre7() -> Array1<f64> {
        let mut x = Array1::zeros(21);
        for i in 0..6 {
            let t = i as f64 * std::f64::consts::TAU / 6.0;
            x[3 * (i + 1)] = t.cos();
            x[3 * (i + 1) + 1] = t.sin();
            x[3 * (i + 1) + 2] = 0.0;
        }
        x
    }

    fn pair_spectrum(x: ArrayView1<f64>) -> Vec<f64> {
        let n = x.len() / 3;
        let mut d = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let mut r2 = 0.0;
                for k in 0..3 {
                    let v = x[3 * i + k] - x[3 * j + k];
                    r2 += v * v;
                }
                d.push(r2.sqrt());
            }
        }
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        d
    }

    fn spectrum_l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    fn all_finite(x: ArrayView1<f64>) -> bool {
        x.iter().all(|v| v.is_finite())
    }

    #[test]
    fn n_is_conserved() {
        let a = icosahedron13();
        let b = cuboctahedron13();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..20 {
            let c = cut_and_splice(a.view(), b.view(), None, 0.85, &mut rng);
            assert_eq!(c.len(), 39, "child length {}", c.len());
        }
        let p = pentagonal_bipyramid7();
        let q = hexagon_plus_centre7();
        for _ in 0..20 {
            let c = cut_and_splice(p.view(), q.view(), None, 0.85, &mut rng);
            assert_eq!(c.len(), 21, "child length {}", c.len());
        }
    }

    #[test]
    fn coordinates_are_finite() {
        let a = icosahedron13();
        let b = cuboctahedron13();
        let mut rng = StdRng::seed_from_u64(3);
        let c = cut_and_splice(a.view(), b.view(), None, 0.85, &mut rng);
        assert!(all_finite(c.view()), "child had a non-finite coordinate");
        let d = splice_at_plane(a.view(), b.view(), None, 0.85, [0.0, 0.0, 1.0], 0.0);
        assert!(all_finite(d.view()));
    }

    /// A known pair must not come back as either parent: the mix is the point.
    #[test]
    fn a_known_pair_is_not_copied() {
        let a = icosahedron13();
        let b = cuboctahedron13();
        let c = splice_at_plane(a.view(), b.view(), None, 0.85, [0.0, 0.0, 1.0], 0.0);
        assert_eq!(c.len(), a.len());
        assert!(all_finite(c.view()));
        let sa = pair_spectrum(a.view());
        let sb = pair_spectrum(b.view());
        let sc = pair_spectrum(c.view());
        let da = spectrum_l2(&sc, &sa);
        let db = spectrum_l2(&sc, &sb);
        assert!(
            da > 1e-3,
            "child matched parent A, pair-spectrum distance {da}"
        );
        assert!(
            db > 1e-3,
            "child matched parent B, pair-spectrum distance {db}"
        );
    }

    /// LJ13 icosahedron spliced with the cuboctahedron is a different shape.
    #[test]
    fn lj13_mix_is_a_different_shape() {
        let a = icosahedron13();
        let b = cuboctahedron13();
        let mut rng = StdRng::seed_from_u64(11);
        let c = cut_and_splice(a.view(), b.view(), None, 0.85, &mut rng);
        assert_eq!(c.len(), 39);
        assert!(all_finite(c.view()));
        let sa = pair_spectrum(a.view());
        let sb = pair_spectrum(b.view());
        let sc = pair_spectrum(c.view());
        let parents = spectrum_l2(&sa, &sb);
        assert!(
            parents > 0.1,
            "the two LJ13 parents are the same shape: {parents}"
        );
        assert!(
            spectrum_l2(&sc, &sa) > 1e-3 && spectrum_l2(&sc, &sb) > 1e-3,
            "LJ13 child matched a parent (dA={}, dB={})",
            spectrum_l2(&sc, &sa),
            spectrum_l2(&sc, &sb)
        );
    }

    /// LJ7 pentagonal bipyramid spliced with a planar hexagon-plus-centre.
    #[test]
    fn lj7_mix_is_a_different_shape() {
        let a = pentagonal_bipyramid7();
        let b = hexagon_plus_centre7();
        let c = splice_at_plane(a.view(), b.view(), None, 0.85, [0.0, 0.0, 1.0], 0.0);
        assert_eq!(c.len(), 21);
        assert!(all_finite(c.view()));
        let sa = pair_spectrum(a.view());
        let sb = pair_spectrum(b.view());
        let sc = pair_spectrum(c.view());
        assert!(
            spectrum_l2(&sa, &sb) > 0.1,
            "the two LJ7 parents are the same shape"
        );
        assert!(
            spectrum_l2(&sc, &sa) > 1e-3 && spectrum_l2(&sc, &sb) > 1e-3,
            "LJ7 child matched a parent (dA={}, dB={})",
            spectrum_l2(&sc, &sa),
            spectrum_l2(&sc, &sb)
        );
    }

    #[test]
    fn species_counts_are_preserved() {
        let a = pentagonal_bipyramid7();
        let b = hexagon_plus_centre7();
        // Four of type 1, three of type 2: a binary 7-point composition.
        let species = [1u32, 1, 1, 1, 2, 2, 2];
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..16 {
            let c = cut_and_splice(a.view(), b.view(), Some(&species), 0.85, &mut rng);
            assert_eq!(c.len(), 21);
            assert!(all_finite(c.view()));
        }
        let c = splice_at_plane(
            a.view(),
            b.view(),
            Some(&species),
            0.85,
            [1.0, 0.0, 0.0],
            0.0,
        );
        assert_eq!(c.len(), 21);
    }

    #[test]
    fn a_shifted_parent_is_recentred_before_the_cut() {
        let a = icosahedron13();
        let mut b = cuboctahedron13();
        for v in b.iter_mut() {
            *v += 10.0;
        }
        let c = splice_at_plane(a.view(), b.view(), None, 0.85, [0.0, 0.0, 1.0], 0.0);
        assert!(all_finite(c.view()));
        let n = 13;
        let mut cx = [0.0; 3];
        for i in 0..n {
            for k in 0..3 {
                cx[k] += c[3 * i + k];
            }
        }
        for k in 0..3 {
            cx[k] /= n as f64;
            assert!(cx[k].abs() < 1e-9, "child centroid {cx:?}");
        }
    }
}
