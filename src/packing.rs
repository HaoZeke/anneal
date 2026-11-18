//! Local surface rearrangements of the current structure.
//!
//! Twin a dense plane as a proposal, or hop the least-bound atoms onto vacant
//! hollows of the structure they stand on. Neither reads a template, and
//! neither completes a structure from a symmetry. Cut-and-splice is not this
//! surface. Catalog Leave is not this surface.

use ndarray::{Array1, ArrayView1};
use rand::Rng;

/// Twin across one of the densest planes. Always a coordinate vector;
/// identity if no plane is found.
pub fn propose_twin<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    n_points: usize,
    rng: &mut R,
) -> Array1<f64> {
    crate::twin::propose(x, n_points, rng)
}

/// Fewest points with a surface hop: a bonded triangle and an atom to move onto its hollow.
const SURFACE_HOP_MIN_POINTS: usize = 4;
/// Fraction of the mean bond length within which an atom makes a hollow site occupied.
const VACANT_BOND_FRACTION: f64 = 0.8;
/// Fraction of the mean bond length within which two hollow sites are merged into one.
const MERGE_BOND_FRACTION: f64 = 0.3;

/// Surface hops of the least-bound atoms: each atom with the lowest
/// coordination, placed on each vacant hollow site where it keeps at least
/// as many neighbours as it has.
///
/// Neighbours are pairs closer than `cutoff`, which must sit between the
/// first and second shells. A hollow site is above a triangle of mutually
/// bonded atoms, on the side away from the centroid, at the mean bond length
/// from all three. It is vacant when no atom is within
/// `VACANT_BOND_FRACTION` bond lengths or `min_separation`, the moving atom
/// included, so a hop is at least that far. Sites closer than
/// `MERGE_BOND_FRACTION` bond lengths to an earlier one are merged.
///
/// The order is the atoms' order, then the triangles' order, so equal
/// inputs give equal candidates.
pub fn surface_hops(
    x: ArrayView1<f64>,
    n_points: usize,
    cutoff: f64,
    min_separation: f64,
) -> Vec<Array1<f64>> {
    let point = |i: usize| [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    if n_points < SURFACE_HOP_MIN_POINTS || x.len() != 3 * n_points {
        return Vec::new();
    }
    let mut bonded = vec![false; n_points * n_points];
    let mut z = vec![0usize; n_points];
    let (mut bond_sum, mut bonds) = (0.0, 0usize);
    for i in 0..n_points {
        for j in (i + 1)..n_points {
            let d = sub(point(i), point(j));
            let r = dot(d, d).sqrt();
            if r < cutoff {
                bonded[i * n_points + j] = true;
                bonded[j * n_points + i] = true;
                z[i] += 1;
                z[j] += 1;
                bond_sum += r;
                bonds += 1;
            }
        }
    }
    if bonds == 0 {
        return Vec::new();
    }
    let bond = bond_sum / bonds as f64;
    let mut centroid = [0.0; 3];
    for i in 0..n_points {
        for (k, c) in centroid.iter_mut().enumerate() {
            *c += x[3 * i + k] / n_points as f64;
        }
    }
    let vacant = |s: [f64; 3]| {
        let clear = (VACANT_BOND_FRACTION * bond).max(min_separation);
        (0..n_points).all(|i| {
            let d = sub(point(i), s);
            dot(d, d) >= clear * clear
        })
    };
    let mut sites: Vec<[f64; 3]> = Vec::new();
    for i in 0..n_points {
        for j in (i + 1)..n_points {
            if !bonded[i * n_points + j] {
                continue;
            }
            for k in (j + 1)..n_points {
                if !(bonded[i * n_points + k] && bonded[j * n_points + k]) {
                    continue;
                }
                let (a, b, c) = (point(i), point(j), point(k));
                let mid = [
                    (a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0,
                    (a[2] + b[2] + c[2]) / 3.0,
                ];
                let (u, v) = (sub(b, a), sub(c, a));
                let mut normal = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let length = dot(normal, normal).sqrt();
                if length == 0.0 {
                    continue;
                }
                let outward = if dot(normal, sub(mid, centroid)) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                for w in normal.iter_mut() {
                    *w *= outward / length;
                }
                let radius = [a, b, c]
                    .iter()
                    .map(|p| {
                        let d = sub(*p, mid);
                        dot(d, d).sqrt()
                    })
                    .sum::<f64>()
                    / 3.0;
                if radius >= bond {
                    continue;
                }
                let h = (bond * bond - radius * radius).sqrt();
                let s = [
                    mid[0] + h * normal[0],
                    mid[1] + h * normal[1],
                    mid[2] + h * normal[2],
                ];
                let merged = sites.iter().any(|t| {
                    let d = sub(*t, s);
                    dot(d, d) < (MERGE_BOND_FRACTION * bond).powi(2)
                });
                if !merged && vacant(s) {
                    sites.push(s);
                }
            }
        }
    }
    let lowest = z.iter().copied().min().unwrap_or(0);
    let mut out = Vec::new();
    for mover in (0..n_points).filter(|&i| z[i] == lowest) {
        for s in &sites {
            let neighbours = (0..n_points)
                .filter(|&i| i != mover)
                .filter(|&i| {
                    let d = sub(point(i), *s);
                    dot(d, d) < cutoff * cutoff
                })
                .count();
            if neighbours >= z[mover] {
                let mut y = x.to_owned();
                for k in 0..3 {
                    y[3 * mover + k] = s[k];
                }
                out.push(y);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn twin_propose_returns_a_state_of_the_same_length() {
        let x = array![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.87, 0.0, 0.5, 0.29, 0.82
        ];
        let mut rng = StdRng::seed_from_u64(1);
        let y = propose_twin(x.view(), 4, &mut rng);
        assert_eq!(y.len(), x.len());
    }

    /// Points of the centred icosahedron, the centre and its twelve vertices.
    const ICOSAHEDRON_POINTS: usize = 13;
    /// Minimum separation of the completions and the hops.
    const MIN_SEPARATION: f64 = 0.5;
    /// Bond cutoff of the hop tests, above the icosahedron's edge length.
    const BOND_CUTOFF: f64 = 1.2;
    /// Bonded atoms around a hollow site, the triangle under it.
    const HOLLOW_NEIGHBOURS: usize = 3;
    /// Closest any atom may be to a hop's site, so the hop makes no overlap.
    const NO_OVERLAP: f64 = 0.75;

    /// A centred icosahedron with unit nearest-neighbour distance from the
    /// centre.
    fn icosahedron() -> Vec<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let s = 1.0 / (1.0 + phi * phi).sqrt();
        let mut x = vec![0.0; 3];
        for a in [-1.0, 1.0] {
            for b in [-1.0, 1.0] {
                x.extend([0.0, a * s, b * phi * s]);
                x.extend([a * s, b * phi * s, 0.0]);
                x.extend([b * phi * s, 0.0, a * s]);
            }
        }
        x
    }

    /// Every vertex of a closed icosahedron has six neighbours and a hollow
    /// site has three, so no hop keeps its bonds.
    #[test]
    fn a_closed_icosahedron_has_no_surface_hop() {
        let x = Array1::from(icosahedron());
        assert!(surface_hops(x.view(), ICOSAHEDRON_POINTS, BOND_CUTOFF, MIN_SEPARATION).is_empty());
    }

    /// An atom on top of a vertex has one neighbour. Each hop moves that
    /// atom only, onto a site with at least three neighbours and no overlap.
    #[test]
    fn an_atop_atom_hops_to_hollow_sites_and_nothing_else_moves() {
        let mut x = icosahedron();
        let top = [x[3] * 2.0, x[4] * 2.0, x[5] * 2.0];
        x.extend(top);
        let x = Array1::from(x);
        let hops = surface_hops(
            x.view(),
            ICOSAHEDRON_POINTS + 1,
            BOND_CUTOFF,
            MIN_SEPARATION,
        );
        assert!(!hops.is_empty());
        for y in &hops {
            for i in 0..ICOSAHEDRON_POINTS {
                for k in 0..3 {
                    assert_eq!(y[3 * i + k], x[3 * i + k]);
                }
            }
            let site = [y[39], y[40], y[41]];
            let distances: Vec<f64> = (0..ICOSAHEDRON_POINTS)
                .map(|i| {
                    (0..3)
                        .map(|k| (y[3 * i + k] - site[k]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            assert!(distances.iter().filter(|d| **d < BOND_CUTOFF).count() >= HOLLOW_NEIGHBOURS);
            assert!(distances.iter().all(|d| *d > NO_OVERLAP));
        }
        assert_eq!(
            hops,
            surface_hops(
                x.view(),
                ICOSAHEDRON_POINTS + 1,
                BOND_CUTOFF,
                MIN_SEPARATION
            )
        );
    }
}
