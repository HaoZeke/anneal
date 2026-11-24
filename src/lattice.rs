//! The lattice body a stalled walk tries before it moves on.
//!
//! Hopping moves a structure by a displacement and a quench, and what a hop
//! can reach is set by that move. A walk that has stalled in a basin has
//! nothing in the move to change what it reaches, and a set of positions is a
//! complete proposal: points are indistinguishable, so growing one and
//! quenching into it crosses a funnel boundary in one step.
//!
//! Everything here is read off the structure the walk is standing on. The
//! contact graph is the Gabriel graph of its pair distances, which has no
//! scale or cutoff of its own. The tightest tetrahedron of mutual contacts
//! at its best-coordinated point gives three edges, a primitive basis for the
//! periodic packing that tetrahedron evidences, and the body is the compact
//! cut of that lattice about the incumbent's centre of geometry. No packing is
//! named, nothing is stored, and no size is assumed: the same code reads a
//! close-packed Lennard-Jones fragment, whatever a Morse potential prefers,
//! and a molecular geometry.

use ndarray::{Array1, ArrayView1};
#[cfg(test)]
use rand::Rng;
#[cfg(test)]
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};
#[cfg(test)]
use std::f64::consts::PI;

/// Mean distance to the nearest other point.
///
/// A length scale taken from the structure, so nothing that uses it carries
/// knowledge of the potential.
pub fn nearest_neighbour_scale(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..3)
                .map(|k| {
                    let v = x[3 * i + k] - x[3 * j + k];
                    v * v
                })
                .sum();
            best = best.min(d);
        }
        total += best.sqrt();
    }
    total / n as f64
}

#[cfg(test)]
fn random_rotation<R: Rng + ?Sized>(rng: &mut R) -> [[f64; 3]; 3] {
    let (u1, u2, u3) = (
        rng.random::<f64>(),
        rng.random::<f64>(),
        rng.random::<f64>(),
    );
    let (x, y, z, w) = (
        (1.0 - u1).sqrt() * (2.0 * PI * u2).sin(),
        (1.0 - u1).sqrt() * (2.0 * PI * u2).cos(),
        u1.sqrt() * (2.0 * PI * u3).sin(),
        u1.sqrt() * (2.0 * PI * u3).cos(),
    );
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn distance2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
}

/// Parameter-free contact graph: two points are neighbours when their
/// diameter sphere contains no third point. This Gabriel rule obtains its
/// complete scale from the incumbent's pair distances. `distances` stores
/// squared distances: `d2(i, k) + d2(j, k) < d2(i, j)` is exactly the test
/// that `k` lies inside the sphere with `ij` as its diameter.
fn observed_contact_graph(x: ArrayView1<f64>) -> Vec<Vec<usize>> {
    let n = x.len() / 3;
    let mut distances = vec![0.0; n * n];
    for i in 0..n {
        let a = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
        for j in (i + 1)..n {
            let b = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
            let d2 = distance2(&a, &b);
            distances[i * n + j] = d2;
            distances[j * n + i] = d2;
        }
    }
    let mut graph = vec![Vec::new(); n];
    for i in 0..n {
        let row_i = &distances[i * n..(i + 1) * n];
        for j in (i + 1)..n {
            let edge = row_i[j];
            if edge > 0.0 {
                let row_j = &distances[j * n..(j + 1) * n];
                let blocked = (0..n).any(|k| k != i && k != j && row_i[k] + row_j[k] < edge);
                if !blocked {
                    graph[i].push(j);
                    graph[j].push(i);
                }
            }
        }
    }
    graph
}

fn basis_is_well_conditioned(basis: &[[f64; 3]; 3]) -> bool {
    let norms = basis.map(|vector| vector[0].hypot(vector[1]).hypot(vector[2]));
    if norms.iter().any(|norm| !norm.is_finite() || *norm == 0.0) {
        return false;
    }
    let unit: [[f64; 3]; 3] = std::array::from_fn(|vector| {
        std::array::from_fn(|axis| basis[vector][axis] / norms[vector])
    });
    let cross = [
        unit[1][1] * unit[2][2] - unit[1][2] * unit[2][1],
        unit[1][2] * unit[2][0] - unit[1][0] * unit[2][2],
        unit[1][0] * unit[2][1] - unit[1][1] * unit[2][0],
    ];
    let normalized_determinant = unit[0]
        .iter()
        .zip(cross)
        .map(|(component, cofactor)| component * cofactor)
        .sum::<f64>()
        .abs();
    // Inverting a basis amplifies roundoff by the reciprocal normalized
    // determinant. Requiring it above sqrt(epsilon) keeps that amplification
    // below 1/sqrt(epsilon), leaving at least half the significand reliable.
    normalized_determinant > f64::EPSILON.sqrt()
}

fn lattice_point(site: [i32; 3], basis: &[[f64; 3]; 3]) -> [f64; 3] {
    [0, 1, 2].map(|axis| {
        (0..3)
            .map(|vector| f64::from(site[vector]) * basis[vector][axis])
            .sum()
    })
}

#[cfg(test)]
fn lattice_neighbours(site: [i32; 3]) -> impl Iterator<Item = [i32; 3]> {
    (0..3).flat_map(move |axis| {
        [-1, 1].map(move |step| {
            let mut neighbour = site;
            neighbour[axis] += step;
            neighbour
        })
    })
}

const NEIGHBOUR_OFFSETS: [[i32; 3]; 6] = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
];

const IN_FRONTIER: u32 = u32::MAX;

#[derive(Default)]
struct SiteHasher(u64);

impl std::hash::Hasher for SiteHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0.rotate_left(5) ^ u64::from(byte)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.write_u64(i as u32 as u64);
    }
}

type SiteMap = HashMap<[i32; 3], u32, std::hash::BuildHasherDefault<SiteHasher>>;

#[derive(Clone, Copy)]
struct FrontierSite {
    radius: f64,
    site: [i32; 3],
}

impl PartialEq for FrontierSite {
    fn eq(&self, other: &Self) -> bool {
        self.radius.to_bits() == other.radius.to_bits() && self.site == other.site
    }
}

impl Eq for FrontierSite {}

impl PartialOrd for FrontierSite {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrontierSite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .radius
            .total_cmp(&self.radius)
            .then_with(|| other.site.cmp(&self.site))
    }
}

#[derive(Clone, Copy)]
struct PeelSite {
    coordination: u8,
    radius: f64,
    site: [i32; 3],
    id: u32,
}

impl PartialEq for PeelSite {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.coordination == other.coordination
            && self.radius.to_bits() == other.radius.to_bits()
            && self.site == other.site
    }
}

impl Eq for PeelSite {}

impl PartialOrd for PeelSite {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PeelSite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .coordination
            .cmp(&self.coordination)
            .then_with(|| self.radius.total_cmp(&other.radius))
            .then_with(|| other.site.cmp(&self.site))
    }
}

/// Distinct 4-point frames on `n` points, `C(n, 4)`.
///
/// The product of four consecutive integers is divisible by 24, so each
/// division below is exact while the product fits. A product that does not
/// fit saturates, and the cloud is then bounded by the largest count this
/// process can store.
fn four_point_frames(n: usize) -> usize {
    if n < 4 {
        return 0;
    }
    let mut frames = n as u128;
    frames = frames.saturating_mul((n - 1) as u128) / 2;
    frames = frames.saturating_mul((n - 2) as u128) / 3;
    frames = frames.saturating_mul((n - 3) as u128) / 4;
    usize::try_from(frames).unwrap_or(usize::MAX)
}

fn candidate_from_basis(basis: &[[f64; 3]; 3], centre: [f64; 3], n: usize) -> Option<Array1<f64>> {
    if n == 0 {
        return Some(Array1::zeros(0));
    }
    // The finished body holds `C(n, 4)` distinct 4-point frames. One frame
    // cannot hold a second site, and a flat basis would otherwise keep the
    // frontier moving without ever enclosing one.
    let frames = four_point_frames(n);
    if frames <= 1 {
        return (n <= 1).then(|| Array1::zeros(3 * n));
    }
    if !basis_is_well_conditioned(basis) || centre.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a.into_iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
    let reciprocal = [
        cross(basis[1], basis[2]),
        cross(basis[2], basis[0]),
        cross(basis[0], basis[1]),
    ];
    let determinant = dot(basis[0], reciprocal[0]);
    if !determinant.is_finite() || determinant == 0.0 {
        return None;
    }
    let fractional = [0, 1, 2].map(|component| dot(centre, reciprocal[component]) / determinant);
    if fractional.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let seed = fractional.map(f64::round);
    if seed.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let shift: [f64; 3] = [0, 1, 2].map(|component| {
        (0..3)
            .map(|vector| seed[vector] * basis[vector][component])
            .sum()
    });
    if shift.iter().any(|value| !value.is_finite()) {
        return None;
    }
    for vector in basis {
        let step2 = distance2(vector, &[0.0; 3]);
        let roundtrip =
            [0, 1, 2].map(|component| (shift[component] + vector[component]) - shift[component]);
        if !step2.is_finite()
            || step2 == 0.0
            || distance2(&roundtrip, vector) > f64::EPSILON * step2
        {
            return None;
        }
    }
    // A zero seed keeps the caller's centre bits, so a near-origin centre
    // still matches the unshifted reference cloud.
    let local_centre = if seed == [0.0; 3] {
        centre
    } else {
        [0, 1, 2].map(|component| {
            (0..3)
                .map(|vector| (fractional[vector] - seed[vector]) * basis[vector][component])
                .sum()
        })
    };
    if local_centre.iter().any(|value| !value.is_finite()) {
        return None;
    }
    // C(n, 4) names every 4-point frame. The proportional site cap stops
    // growth when that count is large. Either bound, once hit before n
    // interior sites exist, returns no body.
    let max_sites = 64usize.saturating_mul(n.saturating_add(4));
    let cap = frames.min(max_sites);

    let mut sites = vec![[0; 3]];
    let mut radii = vec![distance2(&lattice_point([0; 3], basis), &local_centre)];
    let mut coordination = vec![0u8];
    let mut neighbors = vec![[IN_FRONTIER; 6]];
    let mut site_state = SiteMap::default();
    site_state.insert([0; 3], 0);
    let mut frontier = BinaryHeap::new();
    for offset in NEIGHBOUR_OFFSETS {
        site_state.insert(offset, IN_FRONTIER);
        frontier.push(FrontierSite {
            radius: distance2(&lattice_point(offset, basis), &local_centre),
            site: offset,
        });
    }
    let mut interior = 0usize;

    while sites.len() < cap && interior < n {
        let Some(FrontierSite { radius, site: next }) = frontier.pop() else {
            break;
        };
        let Ok(next_id) = u32::try_from(sites.len()) else {
            return None;
        };
        if next_id == IN_FRONTIER {
            return None;
        }
        let next_idx = next_id as usize;
        sites.push(next);
        radii.push(radius);
        coordination.push(0);
        neighbors.push([IN_FRONTIER; 6]);
        site_state.insert(next, next_id);

        for (dir, offset) in NEIGHBOUR_OFFSETS.iter().enumerate() {
            let neighbour = [
                next[0] + offset[0],
                next[1] + offset[1],
                next[2] + offset[2],
            ];
            match site_state.entry(neighbour) {
                Entry::Occupied(entry) => {
                    let state = *entry.get();
                    if state != IN_FRONTIER {
                        let nb_idx = state as usize;
                        neighbors[next_idx][dir] = state;
                        neighbors[nb_idx][dir ^ 1] = next_id;
                        coordination[next_idx] += 1;
                        if coordination[next_idx] == 6 {
                            interior += 1;
                        }
                        coordination[nb_idx] += 1;
                        if coordination[nb_idx] == 6 {
                            interior += 1;
                        }
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(IN_FRONTIER);
                    frontier.push(FrontierSite {
                        radius: distance2(&lattice_point(neighbour, basis), &local_centre),
                        site: neighbour,
                    });
                }
            }
        }
    }
    // The cap was reached before n interior sites existed, so there is no
    // body to peel.
    if interior < n {
        return None;
    }

    let mut alive = vec![true; sites.len()];
    let mut alive_count = sites.len();
    if alive_count > n {
        let mut peel = BinaryHeap::from(
            (0..sites.len())
                .map(|id| PeelSite {
                    coordination: coordination[id],
                    radius: radii[id],
                    site: sites[id],
                    id: id as u32,
                })
                .collect::<Vec<_>>(),
        );
        while alive_count > n {
            let entry = peel.pop().expect("a live site remains while peeling");
            let idx = entry.id as usize;
            if !alive[idx] || coordination[idx] != entry.coordination {
                continue;
            }
            alive[idx] = false;
            alive_count -= 1;
            for nb in neighbors[idx] {
                if nb != IN_FRONTIER {
                    let nb_idx = nb as usize;
                    if alive[nb_idx] {
                        coordination[nb_idx] -= 1;
                        peel.push(PeelSite {
                            coordination: coordination[nb_idx],
                            radius: radii[nb_idx],
                            site: sites[nb_idx],
                            id: nb,
                        });
                    }
                }
            }
        }
    }

    let mut out = Array1::zeros(3 * n);
    for (i, &site) in sites
        .iter()
        .enumerate()
        .filter(|(id, _)| alive[*id])
        .map(|(_, site)| site)
        .enumerate()
    {
        let point = lattice_point(site, basis);
        for component in 0..3 {
            let placed = if seed == [0.0; 3] {
                point[component]
            } else {
                shift[component] + point[component]
            };
            if !placed.is_finite() {
                return None;
            }
            out[3 * i + component] = placed;
        }
    }
    Some(out)
}

/// One compact lattice body generated by a primitive basis observed in `x`.
///
/// A triple of mutually neighbouring offsets is a local tetrahedron. Its
/// three edges are a primitive basis for the periodic packing already
/// evidenced by that tetrahedron, without naming the packing or consulting a
/// reference structure. The tightest observed tetrahedron supplies the basis;
/// the incumbent's centre of geometry breaks equal-radius peel ties.
pub fn observed_basis_candidate(
    x: ArrayView1<f64>,
    require_quantized_residuals: bool,
) -> Option<Array1<f64>> {
    let n = x.len() / 3;
    if n < 4 || x.len() != 3 * n || x.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let graph = observed_contact_graph(x);
    let centre = (0..n).max_by_key(|&i| (graph[i].len(), std::cmp::Reverse(i)))?;
    let shell = &graph[centre];
    let origin = [x[3 * centre], x[3 * centre + 1], x[3 * centre + 2]];
    let body_centre = [0, 1, 2].map(|axis| {
        let sum: f64 = x.iter().skip(axis).step_by(3).sum();
        let mean = if sum.is_finite() {
            sum / n as f64
        } else {
            x.iter()
                .skip(axis)
                .step_by(3)
                .map(|value| *value / n as f64)
                .sum()
        };
        mean - origin[axis]
    });
    if body_centre.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut best = None;
    for i in 0..shell.len() {
        for j in (i + 1)..shell.len() {
            for k in (j + 1)..shell.len() {
                let [a, b, c] = [shell[i], shell[j], shell[k]];
                if !graph[a].contains(&b) || !graph[a].contains(&c) || !graph[b].contains(&c) {
                    continue;
                }
                let basis =
                    [a, b, c].map(|point| [0, 1, 2].map(|axis| x[3 * point + axis] - origin[axis]));
                if !basis_is_well_conditioned(&basis) {
                    continue;
                }
                let cross = [
                    basis[1][1] * basis[2][2] - basis[1][2] * basis[2][1],
                    basis[1][2] * basis[2][0] - basis[1][0] * basis[2][2],
                    basis[1][0] * basis[2][1] - basis[1][1] * basis[2][0],
                ];
                let volume = basis[0]
                    .iter()
                    .zip(cross)
                    .map(|(component, cofactor)| component * cofactor)
                    .sum::<f64>()
                    .abs();
                let points =
                    [a, b, c].map(|point| [x[3 * point], x[3 * point + 1], x[3 * point + 2]]);
                let edges = [
                    distance2(&origin, &points[0]),
                    distance2(&origin, &points[1]),
                    distance2(&origin, &points[2]),
                    distance2(&points[0], &points[1]),
                    distance2(&points[0], &points[2]),
                    distance2(&points[1], &points[2]),
                ];
                let total = edges.iter().sum::<f64>();
                if !volume.is_finite()
                    || !(volume > 0.0)
                    || !total.is_finite()
                    || edges.iter().any(|edge| !edge.is_finite())
                {
                    continue;
                }
                let longest = edges.iter().copied().max_by(f64::total_cmp)?;
                if best
                    .as_ref()
                    .is_none_or(|(best_longest, best_total, best_volume, _)| {
                        longest < *best_longest
                            || (longest == *best_longest && total < *best_total)
                            || (longest == *best_longest
                                && total == *best_total
                                && volume > *best_volume)
                    })
                {
                    best = Some((longest, total, volume, basis));
                }
            }
        }
    }
    let (_, _, _, basis) = best?;
    let contact_lengths = graph.iter().enumerate().flat_map(|(i, neighbours)| {
        neighbours
            .iter()
            .copied()
            .filter(move |&j| i < j)
            .map(move |j| {
                let a = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
                let b = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
                distance2(&a, &b).sqrt()
            })
    });
    let (shortest, longest) = contact_lengths.fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(shortest, longest), length| (shortest.min(length), longest.max(length)),
    );
    if !shortest.is_finite() || !longest.is_finite() || !(longest > 0.0) {
        return None;
    }
    let contact_spread = longest - shortest;
    if !contact_spread.is_finite() {
        return None;
    }
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a.into_iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
    let reciprocal = [
        cross(basis[1], basis[2]),
        cross(basis[2], basis[0]),
        cross(basis[0], basis[1]),
    ];
    let determinant = dot(basis[0], reciprocal[0]);
    if !determinant.is_finite() || determinant == 0.0 {
        return None;
    }
    let mut residuals = Vec::with_capacity(n);
    for point in 0..n {
        let offset = [0, 1, 2].map(|axis| x[3 * point + axis] - origin[axis]);
        if offset.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let nearest = [0, 1, 2].map(|axis| (dot(offset, reciprocal[axis]) / determinant).round());
        if nearest.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let fitted: [f64; 3] = [0, 1, 2].map(|axis| {
            (0..3)
                .map(|vector| nearest[vector] * basis[vector][axis])
                .sum()
        });
        if fitted.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let residual = distance2(&offset, &fitted).sqrt();
        if !residual.is_finite() {
            return None;
        }
        residuals.push(residual);
    }
    let max_degree = graph.iter().map(Vec::len).max()?;
    let residual_floor = f64::EPSILON.sqrt() * longest;
    let core_tolerance = contact_spread.max(residual_floor);
    let registered_core = residuals
        .iter()
        .enumerate()
        .filter(|(point, _)| graph[*point].len() == max_degree)
        .all(|(_, residual)| *residual <= core_tolerance);
    let mut ordered = residuals;
    ordered.sort_by(f64::total_cmp);
    let span = ordered.last()? - ordered.first()?;
    if !span.is_finite() {
        return None;
    }
    let mut gaps: Vec<f64> = ordered.windows(2).map(|pair| pair[1] - pair[0]).collect();
    gaps.sort_by(f64::total_cmp);
    let median_gap = gaps[gaps.len() / 2];
    let largest_gap = *gaps.last()?;
    // A lattice gap must exceed the spread expected if every adjacent residual
    // had the observed median gap; the point count supplies that expectation.
    // An exact crystal whose residuals differ only by floating-point roundoff
    // has no defect gap to separate; the half-significand floor bounds that
    // roundoff on the observed contact scale.
    let quantized = span <= residual_floor || largest_gap > gaps.len() as f64 * median_gap;
    let coherent = registered_core && (!require_quantized_residuals || quantized);
    if !coherent {
        return None;
    }
    candidate_from_basis(&basis, body_centre, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(11)
    }

    /// A compact fragment of the packing three mutually touching directions
    /// generate, slightly stretched: the input a stalled walk would see.
    fn packed(n: usize) -> Array1<f64> {
        let basis = [
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 6.0, (2.0_f64 / 3.0).sqrt()],
        ];
        candidate_from_basis(&basis, [0.1, -0.2, 0.15], n)
            .expect("a body of this many points")
            .mapv(|value| 1.12 * value)
    }

    fn reference_candidate_from_basis(
        basis: &[[f64; 3]; 3],
        centre: [f64; 3],
        n: usize,
    ) -> Option<Array1<f64>> {
        let mut sites = vec![[0; 3]];
        let mut occupied = HashSet::from([[0; 3]]);
        let mut frontier: HashSet<[i32; 3]> = lattice_neighbours([0; 3]).collect();
        let frames = four_point_frames(n);
        while sites.len() < frames
            && sites
                .iter()
                .filter(|&&site| lattice_neighbours(site).all(|other| occupied.contains(&other)))
                .count()
                < n
        {
            let next = frontier.iter().copied().min_by(|a, b| {
                distance2(&lattice_point(*a, basis), &centre)
                    .total_cmp(&distance2(&lattice_point(*b, basis), &centre))
                    .then_with(|| a.cmp(b))
            })?;
            frontier.remove(&next);
            occupied.insert(next);
            for neighbour in lattice_neighbours(next) {
                if !occupied.contains(&neighbour) {
                    frontier.insert(neighbour);
                }
            }
            sites.push(next);
        }
        if sites.len() < n {
            return None;
        }
        let radii: Vec<f64> = sites
            .iter()
            .map(|&site| distance2(&lattice_point(site, basis), &centre))
            .collect();
        let mut alive = occupied;
        while alive.len() > n {
            let worst = sites
                .iter()
                .enumerate()
                .filter(|(_, site)| alive.contains(*site))
                .min_by(|(i, a), (j, b)| {
                    let coordination = |site| {
                        lattice_neighbours(site)
                            .filter(|neighbour| alive.contains(neighbour))
                            .count()
                    };
                    coordination(**a)
                        .cmp(&coordination(**b))
                        .then_with(|| radii[*j].total_cmp(&radii[*i]))
                        .then_with(|| a.cmp(b))
                })
                .map(|(_, site)| *site)?;
            alive.remove(&worst);
        }
        let mut out = Array1::zeros(3 * n);
        for (i, site) in sites
            .into_iter()
            .filter(|site| alive.contains(site))
            .enumerate()
        {
            let point = lattice_point(site, basis);
            for axis in 0..3 {
                out[3 * i + axis] = point[axis];
            }
        }
        Some(out)
    }

    #[test]
    fn cached_lattice_cloud_matches_the_direct_construction() {
        let basis = [
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 6.0, (2.0_f64 / 3.0).sqrt()],
        ];
        for (centre, n) in [([0.0, 0.0, 0.0], 8), ([0.37, -0.21, 0.14], 13)] {
            assert_eq!(
                candidate_from_basis(&basis, centre, n),
                reference_candidate_from_basis(&basis, centre, n)
            );
        }
    }

    #[test]
    fn candidate_from_basis_matches_reference_at_cluster_sizes_and_ties() {
        let fcc = [
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 6.0, (2.0_f64 / 3.0).sqrt()],
        ];
        let cubic = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for basis in [fcc, cubic] {
            for (centre, n) in [
                ([0.0, 0.0, 0.0], 4),
                ([0.0, 0.0, 0.0], 13),
                ([0.0, 0.0, 0.0], 38),
                ([0.19, -0.27, 0.11], 26),
                ([0.19, -0.27, 0.11], 38),
            ] {
                let actual = candidate_from_basis(&basis, centre, n);
                let expected = reference_candidate_from_basis(&basis, centre, n);
                let actual_bits = actual
                    .as_ref()
                    .map(|arr| arr.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                let expected_bits = expected
                    .as_ref()
                    .map(|arr| arr.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                assert_eq!(actual_bits, expected_bits, "centre={centre:?} n={n}");
            }
        }
    }

    fn spectrum(c: ArrayView1<f64>, n: usize) -> Vec<f64> {
        let mut d = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                d.push(
                    (0..3)
                        .map(|k| {
                            let v = c[3 * i + k] - c[3 * j + k];
                            v * v
                        })
                        .sum::<f64>()
                        .sqrt(),
                );
            }
        }
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d
    }

    #[test]
    fn observed_growth_uses_only_the_structure_it_is_given() {
        let basis = [
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 6.0, (2.0_f64 / 3.0).sqrt()],
        ];
        // One frame, C(4, 4) = 1, cannot hold a second site.
        let tetra = Array1::from_iter(basis.into_iter().flatten().chain([0.0, 0.0, 0.0]));
        assert!(observed_basis_candidate(tetra.view(), true).is_none());

        let mut coords = Vec::new();
        for i in 0..2i32 {
            for j in 0..2i32 {
                for k in 0..2i32 {
                    let index = [i, j, k];
                    for axis in 0..3 {
                        coords.push(
                            (0..3)
                                .map(|vector| f64::from(index[vector]) * basis[vector][axis])
                                .sum(),
                        );
                    }
                }
            }
        }
        let x = Array1::from_vec(coords);
        let narrow = observed_basis_candidate(x.view(), true).expect("observed basis");
        let wide = observed_basis_candidate(x.mapv(|value| 2.0 * value).view(), true)
            .expect("scaled basis");
        assert_eq!(narrow.len(), x.len());
        assert!(narrow.iter().any(|value| *value != 0.0));
        assert!(narrow.iter().all(|value| value.is_finite()));
        for (narrow, wide) in narrow.iter().zip(wide) {
            assert_eq!(wide, 2.0 * narrow);
        }
    }

    #[test]
    fn four_point_frames_count_distinct_subsets() {
        assert_eq!(four_point_frames(3), 0);
        assert_eq!(four_point_frames(4), 1);
        assert_eq!(four_point_frames(5), 5);
        assert_eq!(four_point_frames(6), 15);
        assert_eq!(four_point_frames(8), 70);
        assert_eq!(four_point_frames(26), 14_950);
    }

    #[test]
    fn a_shallow_four_point_cluster_stops_at_the_frame_bound() {
        let square = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        for offset in [1.0e-3, 1.0e-4] {
            let mut lifted = square.clone();
            lifted[11] = offset;
            let started = std::time::Instant::now();
            let candidate = observed_basis_candidate(lifted.view(), false);
            let elapsed = started.elapsed();
            assert!(
                elapsed < std::time::Duration::from_secs(1),
                "offset {offset} took {elapsed:?}"
            );
            assert!(candidate.is_none(), "offset {offset} still grew a body");
        }
    }

    #[test]
    fn observed_growth_rejects_degenerate_order() {
        let line = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        ]);
        assert!(observed_basis_candidate(line.view(), true).is_none());
    }

    #[test]
    fn observed_growth_rejects_coplanar_and_ill_conditioned_cliques() {
        let square = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        assert!(observed_basis_candidate(square.view(), false).is_none());

        let mut nearly_coplanar = square;
        nearly_coplanar[11] = f64::EPSILON;
        assert!(observed_basis_candidate(nearly_coplanar.view(), false).is_none());
    }

    #[test]
    fn observed_contacts_match_a_brute_force_gabriel_graph() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.9, 0.1],
            [0.8, 0.7, 0.6],
            [-0.4, 0.3, 0.8],
        ];
        let x = Array1::from_iter(points.into_iter().flatten());
        let actual = observed_contact_graph(x.view());
        let mut expected = vec![Vec::new(); points.len()];
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let midpoint = [0, 1, 2].map(|axis| (points[i][axis] + points[j][axis]) / 2.0);
                let radius2 = distance2(&points[i], &points[j]) / 4.0;
                let blocked = points
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i && *k != j)
                    .any(|(_, point)| distance2(point, &midpoint) < radius2);
                if !blocked {
                    expected[i].push(j);
                    expected[j].push(i);
                }
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn frame_count_is_monotone_from_four_through_eight() {
        // Literal C(n, 4) for n = 4, 5, 6, 7, 8. Each is below 64(n+4),
        // so the bound is that count, and the counts rise with n.
        let expected = [1usize, 5, 15, 35, 70];
        let mut previous = 0usize;
        for (slot, n) in (4usize..=8).enumerate() {
            let frames = four_point_frames(n);
            let proportional = 64usize.saturating_mul(n.saturating_add(4));
            let bound = frames.min(proportional);
            assert_eq!(frames, expected[slot], "n={n}");
            assert_eq!(bound, expected[slot], "n={n}");
            assert!(proportional > expected[slot], "n={n}");
            assert!(bound >= previous, "n={n}");
            previous = bound;
        }
    }

    #[test]
    fn frame_cap_stops_indexed_growth_before_a_body() {
        let cubic = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        // C(5, 4) = 5 is above the one-frame early return, so the indexed
        // cloud is what stops. Five sites cannot enclose five interior sites.
        let small_cap = four_point_frames(5).min(64usize.saturating_mul(5 + 4));
        assert_eq!(small_cap, 5);
        assert!(candidate_from_basis(&cubic, [0.0; 3], 5).is_none());
        assert!(candidate_from_basis(&cubic, [0.0; 3], 8).is_some());

        // The long third vector is still well conditioned. C(13, 4) = 715
        // is tighter than 64(13+4) and is hit while the short plane is full
        // and the interior count is still zero.
        let anisotropic = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0e4]];
        assert!(basis_is_well_conditioned(&anisotropic));
        let flat_cap = four_point_frames(13).min(64usize.saturating_mul(13 + 4));
        assert_eq!(flat_cap, 715);
        assert!(candidate_from_basis(&anisotropic, [0.0; 3], 13).is_none());
    }

    #[test]
    fn observed_growth_accepts_an_exact_multishell_crystal() {
        // Integer face-centred cubic sites have identical contact lengths
        // (`sqrt(2)`), so `contact_spread` is zero while multi-shell
        // fractional projections carry floating-point roundoff.
        let mut sites = Vec::new();
        for x in -2i32..=2 {
            for y in -2i32..=2 {
                for z in -2i32..=2 {
                    if (x + y + z).rem_euclid(2) == 0 && x * x + y * y + z * z <= 6 {
                        sites.push([f64::from(x), f64::from(y), f64::from(z)]);
                    }
                }
            }
        }
        let n = sites.len();
        assert!(n >= 40);
        let exact = Array1::from_iter(sites.iter().copied().flatten());
        let rebuilt = observed_basis_candidate(exact.view(), true)
            .expect("an exact crystal with zero contact spread must reconstruct");
        assert_eq!(rebuilt.len(), 3 * n);
        assert!(rebuilt.iter().all(|value| value.is_finite()));

        let mut r = rng();
        let rot = random_rotation(&mut r);
        let rotated = Array1::from_iter(sites.iter().flat_map(|site| {
            [0, 1, 2].map(|axis| (0..3).map(|k| rot[axis][k] * site[k]).sum::<f64>() + 3.7)
        }));
        let rotated_rebuilt = observed_basis_candidate(rotated.view(), true)
            .expect("rigid motion roundoff must not disqualify an exact crystal");
        assert_eq!(rotated_rebuilt.len(), 3 * n);
        assert!((nearest_neighbour_scale(rotated_rebuilt.view()) - 2.0_f64.sqrt()).abs() < 1.0e-6);
    }

    #[test]
    fn observed_growth_seeds_near_an_outlier_shifted_centroid() {
        let basis = [
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 6.0, (2.0_f64 / 3.0).sqrt()],
        ];
        let local = candidate_from_basis(&basis, [0.1, -0.2, 0.15], 19).expect("local seed");
        let far_centre = [
            10_000.0 * basis[0][0] + 0.1,
            10_000.0 * basis[0][1] - 0.2,
            10_000.0 * basis[0][2] + 0.15,
        ];
        let shifted = candidate_from_basis(&basis, far_centre, 19).expect("shifted seed");
        assert_eq!(shifted.len(), local.len());
        let shifted_mean =
            [0, 1, 2].map(|axis| shifted.iter().skip(axis).step_by(3).sum::<f64>() / 19.0);
        assert!((shifted_mean[0] - far_centre[0]).abs() < 2.0);
        for (a, b) in spectrum(local.view(), 19)
            .into_iter()
            .zip(spectrum(shifted.view(), 19))
        {
            assert!((a - b).abs() < 1.0e-9);
        }
        assert!(
            candidate_from_basis(&basis, [1.0e20, 0.0, 0.0], 19).is_none(),
            "a shift that erases the basis spacing must be refused"
        );

        let mut with_outlier = packed(38);
        let tail = with_outlier.len() - 3;
        with_outlier[tail] = 50_000.0;
        let candidate = observed_basis_candidate(with_outlier.view(), false)
            .expect("registered core with an outlier shifts its seed without runaway growth");
        assert_eq!(candidate.len(), with_outlier.len());
        assert!(candidate.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn observed_growth_rejects_overflowing_scales_and_skewed_growth() {
        let base = packed(19);
        for scale in [1.0e200, 1.0e307] {
            let huge = base.mapv(|value| value * scale);
            assert!(
                huge.iter().all(|value| value.is_finite()),
                "fixture coordinates at {scale:e} must stay finite"
            );
            assert!(observed_basis_candidate(huge.view(), false).is_none());
            assert!(observed_basis_candidate(huge.view(), true).is_none());
        }

        // Orthogonal vectors with extreme aspect ratio have unit normalized
        // determinant, while Euclidean sphere growth floods the short plane
        // before reaching +/-basis[2]; the site cap stops that runaway growth.
        let anisotropic = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0e4]];
        assert!(basis_is_well_conditioned(&anisotropic));
        assert!(candidate_from_basis(&anisotropic, [0.0; 3], 13).is_none());
    }
}
