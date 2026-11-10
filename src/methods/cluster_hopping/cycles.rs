//! Reversal of directed hydrogen-bond cycles in rigid water.
//!
//! On a fixed oxygen network the energy still depends on which way each
//! hydrogen bond points. Turning one molecule alone breaks a bond, and the
//! quench can turn the molecule straight back to the minimum it left. This move
//! reverses a directed cycle of bonds instead, so every molecule on it still
//! donates and accepts one bond along the cycle. The idea is the loop move
//! for proton arrangements in ice (Rahman and Stillinger 1972). Here it is a
//! minimiser's move: it has no proposal ratio, so a chain that uses it does
//! not sample proton disorder at equilibrium.

use ndarray::Array1;
use rand::Rng;

use crate::rigid_body::RigidBodyGeometry;

/// An O to H distance under this, in angstroms, between different molecules
/// is a hydrogen bond.
const BOND: f64 = 2.5;

/// Oxygen and hydrogen site indices of a water geometry: exactly one site
/// of atomic number 8 and two of atomic number 1.
pub(crate) fn water_sites(geometry: &RigidBodyGeometry) -> Option<(usize, [usize; 2])> {
    let of = |z: u32| -> Vec<usize> {
        let species = geometry.species();
        (0..species.len()).filter(|&s| species[s] == z).collect()
    };
    match (of(8).as_slice(), of(1).as_slice()) {
        (&[o], &[h1, h2]) => Some((o, [h1, h2])),
        _ => None,
    }
}

/// Lab-frame sites of every molecule, indexed by molecule and then site.
fn lab_sites(geometry: &RigidBodyGeometry, n: usize, x: &Array1<f64>) -> Vec<Vec<[f64; 3]>> {
    let per = geometry.sites().len();
    let flat = geometry.expand(n, x.view());
    (0..n)
        .map(|i| {
            (0..per)
                .map(|s| std::array::from_fn(|k| flat[3 * (per * i + s) + k]))
                .collect()
        })
        .collect()
}

/// Donated hydrogen bonds of every molecule, as the acceptor and which of
/// the two hydrogens donates (0 or 1).
fn donations(sites: &[Vec<[f64; 3]>], o: usize, h: [usize; 2]) -> Vec<Vec<(usize, usize)>> {
    let n = sites.len();
    (0..n)
        .map(|i| {
            let mut out = Vec::new();
            for j in (0..n).filter(|&j| j != i) {
                for (which, &s) in h.iter().enumerate() {
                    if norm(sub(sites[i][s], sites[j][o])) < BOND {
                        out.push((j, which));
                    }
                }
            }
            out
        })
        .collect()
}

/// Reverses the sense of a directed hydrogen-bond cycle.
///
/// A walk from a random donor follows a random donated bond at each step.
/// It stops at a molecule it has visited, which closes a cycle. It never
/// steps straight back to the molecule it came from. So every cycle has at
/// least three molecules: two waters that donate to each other would keep
/// the same two bonds when both turn.
///
/// Each molecule on the cycle then turns about its oxygen. The turn maps
/// the plane of its two hydrogens onto the plane of its other hydrogen and
/// the direction to the previous oxygen, and it maps bisector onto
/// bisector. That new pair of directions is not at the H-O-H angle, so the
/// donating hydrogen ends close to that oxygen while the other hydrogen
/// moves a little too.
/// The quench settles both.
///
/// The walk is not uniform over cycles, so the move has no proposal ratio.
/// Returns `false`, with `x` unchanged, when the geometry is not a water or
/// the walk finds no cycle.
pub(crate) fn reverse_cycle<R: Rng + ?Sized>(
    geometry: &RigidBodyGeometry,
    n: usize,
    x: &mut Array1<f64>,
    rng: &mut R,
) -> bool {
    let Some((o, h)) = water_sites(geometry) else {
        return false;
    };
    if n < 2 || x.len() != 6 * n {
        return false;
    }
    let sites = lab_sites(geometry, n, x);
    let donated = donations(&sites, o, h);
    let donors: Vec<usize> = (0..n).filter(|&i| !donated[i].is_empty()).collect();
    if donors.is_empty() {
        return false;
    }
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut at = donors[rng.random_range(0..donors.len())];
    let cycle = loop {
        let back = path.last().map(|&(m, _)| m);
        let out: Vec<(usize, usize)> = donated[at]
            .iter()
            .copied()
            .filter(|&(j, _)| Some(j) != back)
            .collect();
        if out.is_empty() || path.len() > n {
            return false;
        }
        let (next, which) = out[rng.random_range(0..out.len())];
        path.push((at, which));
        if let Some(start) = path.iter().position(|&(m, _)| m == next) {
            break path.split_off(start);
        }
        at = next;
    };
    let mut turns = Vec::with_capacity(cycle.len());
    for (s, &(k, which)) in cycle.iter().enumerate() {
        let prev = cycle[(s + cycle.len() - 1) % cycle.len()].0;
        let oxygen = sites[k][o];
        let bonded = unit(sub(sites[k][h[which]], oxygen));
        let other = unit(sub(sites[k][h[1 - which]], oxygen));
        let target = unit(sub(sites[prev][o], oxygen));
        if norm(cross(target, other)) < 1e-6 {
            return false;
        }
        turns.push((k, oxygen, frame_rotation((bonded, other), (target, other))));
    }
    for (k, oxygen, turn) in turns {
        let centre = [x[3 * k], x[3 * k + 1], x[3 * k + 2]];
        let moved = add(oxygen, mat_vec(turn, sub(centre, oxygen)));
        let base = 3 * n + 3 * k;
        let w = crate::potentials::compose_rotation(turn, [x[base], x[base + 1], x[base + 2]]);
        for a in 0..3 {
            x[3 * k + a] = moved[a];
            x[base + a] = w[a];
        }
    }
    true
}

/// The rotation that takes the frame of `from` onto the frame of `to`.
///
/// Each pair spans a frame through its bisector and its normal. The two
/// pairs need not open at the same angle; the bisectors and the planes match.
fn frame_rotation(from: ([f64; 3], [f64; 3]), to: ([f64; 3], [f64; 3])) -> [[f64; 3]; 3] {
    let frame = |(a, b): ([f64; 3], [f64; 3])| {
        let bisector = unit(add(a, b));
        let normal = unit(cross(a, b));
        [bisector, normal, cross(bisector, normal)]
    };
    let f = frame(from);
    let t = frame(to);
    std::array::from_fn(|row| {
        std::array::from_fn(|col| (0..3).map(|k| t[k][row] * f[k][col]).sum())
    })
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k] + b[k])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k] - b[k])
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|v| v * s)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| a[k] * b[k]).sum()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    scale(a, 1.0 / norm(a))
}

fn mat_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    m.map(|row| dot(row, v))
}

/// Whether groups are three-atom water: one oxygen and two hydrogens each.
pub(crate) fn is_water_groups(groups: &[Vec<usize>]) -> bool {
    !groups.is_empty() && groups.iter().all(|g| g.len() == 3)
}

/// Lab-frame atom positions of each molecule from Cartesian coordinates.
fn cartesian_sites(groups: &[Vec<usize>], x: &ndarray::Array1<f64>) -> Vec<[[f64; 3]; 3]> {
    groups
        .iter()
        .map(|atoms| {
            std::array::from_fn(|s| {
                let a = atoms.get(s).copied().unwrap_or(0);
                std::array::from_fn(|k| x.get(3 * a + k).copied().unwrap_or(0.0))
            })
        })
        .collect()
}

/// H-bond donations in Cartesian water: for each molecule, the acceptor index
/// and which hydrogen (0 or 1) donates.
fn cartesian_donations(sites: &[[[f64; 3]; 3]]) -> Vec<Vec<(usize, usize)>> {
    let n = sites.len();
    (0..n)
        .map(|i| {
            let _oxygen_i = sites[i][0];
            (0..n)
                .filter(|&j| j != i)
                .flat_map(|j| {
                    let oxygen_j = sites[j][0];
                    [1, 2].iter().enumerate().filter_map(move |(which, &h)| {
                        let hydrogen = sites[i][h];
                        if norm(sub(hydrogen, oxygen_j)) < BOND {
                            Some((j, which))
                        } else {
                            None
                        }
                    })
                })
                .collect()
        })
        .collect()
}

/// Reverses a directed H-bond cycle in Cartesian water coordinates.
///
/// Each group must be `[O, H, H]` in that order. The function finds a directed
/// cycle of H-bonds and rotates each molecule on the cycle about its oxygen
/// so the donations point the other way. The quench settles each molecule.
///
/// Returns `true` when a cycle was reversed, `false` otherwise.
pub(crate) fn reverse_cycle_cartesian<R: rand::Rng + ?Sized>(
    groups: &[Vec<usize>],
    x: &mut ndarray::Array1<f64>,
    rng: &mut R,
) -> bool {
    if !is_water_groups(groups) || groups.len() < 2 {
        return false;
    }
    let sites = cartesian_sites(groups, x);
    let donated = cartesian_donations(&sites);
    let donors: Vec<usize> = (0..groups.len())
        .filter(|&i| !donated[i].is_empty())
        .collect();
    if donors.is_empty() {
        return false;
    }
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut at = donors[rng.random_range(0..donors.len())];
    let cycle = loop {
        let back = path.last().map(|&(m, _)| m);
        let out: Vec<(usize, usize)> = donated[at]
            .iter()
            .copied()
            .filter(|&(j, _)| Some(j) != back)
            .collect();
        if out.is_empty() || path.len() > groups.len() {
            return false;
        }
        let (next, which) = out[rng.random_range(0..out.len())];
        path.push((at, which));
        if let Some(start) = path.iter().position(|&(m, _)| m == next) {
            break path.split_off(start);
        }
        at = next;
    };
    if cycle.len() < 3 {
        return false;
    }
    let mut turns: Vec<(usize, [f64; 3], [[f64; 3]; 3])> = Vec::with_capacity(cycle.len());
    for (s, &(k, which)) in cycle.iter().enumerate() {
        let prev = cycle[(s + cycle.len() - 1) % cycle.len()].0;
        let oxygen = sites[k][0];
        let h_donating = sites[k][1 + which];
        let h_other = sites[k][1 + (1 - which)];
        let bonded = unit(sub(h_donating, oxygen));
        let other = unit(sub(h_other, oxygen));
        let target = unit(sub(sites[prev][0], oxygen));
        if norm(cross(target, other)) < 1e-6 {
            return false;
        }
        turns.push((k, oxygen, frame_rotation((bonded, other), (target, other))));
    }
    for (k, oxygen, turn) in turns {
        let atoms = &groups[k];
        for (s, &a) in atoms.iter().enumerate() {
            let old = std::array::from_fn(|axis| x[3 * a + axis]);
            let new = if s == 0 {
                old
            } else {
                add(oxygen, mat_vec(turn, sub(old, oxygen)))
            };
            for axis in 0..3 {
                x[3 * a + axis] = new[axis];
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::BTreeSet;

    fn geometry() -> RigidBodyGeometry {
        crate::potentials::Tip4pCluster::new(2).rigid_geometry()
    }

    fn bond_sets(
        n: usize,
        x: &Array1<f64>,
    ) -> (BTreeSet<(usize, usize)>, BTreeSet<(usize, usize)>) {
        let g = geometry();
        let (o, h) = water_sites(&g).expect("water");
        let directed: BTreeSet<(usize, usize)> = donations(&lab_sites(&g, n, x), o, h)
            .iter()
            .enumerate()
            .flat_map(|(i, out)| out.iter().map(move |&(j, _)| (i, j)))
            .collect();
        let undirected = directed
            .iter()
            .map(|&(i, j)| (i.min(j), i.max(j)))
            .collect();
        (directed, undirected)
    }

    /// A ring of `n` waters with its oxygens in one plane, 2.8 angstroms apart.
    /// Each donates one hydrogen bond to the next and points its other
    /// hydrogen out of the plane.
    fn ring(n: usize) -> Array1<f64> {
        let g = geometry();
        let (o, h) = water_sites(&g).expect("water");
        let body = g.sites();
        let radius = 2.8 / (2.0 * (std::f64::consts::PI / n as f64).sin());
        let oxygen = |i: usize| {
            let a = std::f64::consts::TAU * (i % n) as f64 / n as f64;
            [radius * a.cos(), radius * a.sin(), 0.0]
        };
        let h1 = unit(sub(body[h[0]], body[o]));
        let h2 = unit(sub(body[h[1]], body[o]));
        let hoh = dot(h1, h2).acos();
        let mut x = Array1::zeros(6 * n);
        for i in 0..n {
            let at = oxygen(i);
            let first = unit(sub(oxygen(i + 1), at));
            let second = add(scale(first, hoh.cos()), scale([0.0, 0.0, 1.0], hoh.sin()));
            let r = frame_rotation((h1, h2), (first, second));
            let centre = sub(at, mat_vec(r, body[o]));
            let w = crate::potentials::compose_rotation(r, [0.0; 3]);
            for k in 0..3 {
                x[3 * i + k] = centre[k];
                x[3 * n + 3 * i + k] = w[k];
            }
        }
        x
    }

    #[test]
    fn reversing_a_ring_keeps_the_network_and_flips_every_bond() {
        let n = 5;
        let mut x = ring(n);
        let (directed, undirected) = bond_sets(n, &x);
        let forward: BTreeSet<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        assert_eq!(directed, forward);
        let mut rng = StdRng::seed_from_u64(4);
        assert!(reverse_cycle(&geometry(), n, &mut x, &mut rng));
        let (directed_after, undirected_after) = bond_sets(n, &x);
        assert_eq!(undirected_after, undirected);
        let backward: BTreeSet<(usize, usize)> = forward.iter().map(|&(i, j)| (j, i)).collect();
        assert_eq!(directed_after, backward);
    }

    #[test]
    fn a_cluster_without_a_cycle_is_left_alone() {
        // Two waters far apart donate nothing.
        let n = 2;
        let mut x = Array1::zeros(6 * n);
        x[3] = 10.0;
        let before = x.clone();
        let mut rng = StdRng::seed_from_u64(1);
        assert!(!reverse_cycle(&geometry(), n, &mut x, &mut rng));
        assert_eq!(x, before);
    }

    #[test]
    fn a_pair_that_donates_both_ways_is_left_alone() {
        let n = 2;
        let mut x = Array1::zeros(6 * n);
        x[0] = -1.5;
        x[3] = 1.5;
        let (directed, _) = bond_sets(n, &x);
        assert_eq!(directed, BTreeSet::from([(0, 1), (1, 0)]));
        let before = x.clone();
        let mut rng = StdRng::seed_from_u64(1);
        assert!(!reverse_cycle(&geometry(), n, &mut x, &mut rng));
        assert_eq!(x, before);
    }

    #[test]
    fn a_geometry_without_one_oxygen_and_two_hydrogens_is_not_a_water() {
        let g =
            RigidBodyGeometry::new(vec![[0.0; 3], [1.0, 0.0, 0.0]], vec![6, 8]).expect("finite");
        assert!(water_sites(&g).is_none());
        assert_eq!(water_sites(&geometry()), Some((0, [1, 2])));
    }
}
