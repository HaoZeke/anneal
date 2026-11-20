//! Geometry proposals for an atomic binary cluster.
//!
//! A collective reassignment puts one species on the sites an unrelaxed
//! swap ranks lowest, then quenched exchanges finish the ordering. The
//! proposal is derived from the current coordinates and kept only when its
//! quenched energy is lower.

use ndarray::{Array1, ArrayView1};

use super::species_groups;

/// Places `group_a` on the sites that unrelaxed swaps rank as lowest for it.
///
/// `None` when that placement is the one already occupied, or when the
/// energy callback stops.
pub(crate) fn reassign_by_site_energy<F>(
    coords: ArrayView1<f64>,
    group_a: &[usize],
    group_b: &[usize],
    mut energy: F,
) -> Option<Array1<f64>>
where
    F: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let n = coords.len() / 3;
    if group_a.is_empty()
        || group_b.is_empty()
        || group_a.len() + group_b.len() != n
        || coords.len() != 3 * n
    {
        return None;
    }
    let base = energy(coords)?;
    if !base.is_finite() {
        return None;
    }
    let mut sum = vec![0.0; n];
    let mut count = vec![0u32; n];
    for &atom in group_a {
        sum[atom] += base;
        count[atom] = count[atom].saturating_add(1);
    }
    let mut trial = coords.to_owned();
    for &atom in group_a {
        for &other in group_b {
            for axis in 0..3 {
                trial.swap(3 * atom + axis, 3 * other + axis);
            }
            let value = energy(trial.view());
            for axis in 0..3 {
                trial.swap(3 * atom + axis, 3 * other + axis);
            }
            let value = value?;
            if value.is_finite() {
                sum[other] += value;
                count[other] = count[other].saturating_add(1);
            }
        }
    }
    let mut order: Vec<usize> = (0..n).filter(|&site| count[site] > 0).collect();
    order.sort_by(|&left, &right| {
        let left_score = sum[left] / f64::from(count[left]);
        let right_score = sum[right] / f64::from(count[right]);
        left_score.total_cmp(&right_score).then(left.cmp(&right))
    });
    if order.len() < group_a.len() {
        return None;
    }
    let chosen = &order[..group_a.len()];
    if chosen.iter().all(|site| group_a.contains(site)) {
        return None;
    }
    Some(assign_sites(coords, group_a, group_b, chosen))
}

fn assign_sites(
    coords: ArrayView1<f64>,
    group_a: &[usize],
    group_b: &[usize],
    chosen: &[usize],
) -> Array1<f64> {
    let n = coords.len() / 3;
    let position = |site: usize| [coords[3 * site], coords[3 * site + 1], coords[3 * site + 2]];
    let mut preferred = Vec::with_capacity(group_a.len());
    let mut rest = Vec::with_capacity(group_b.len());
    for site in 0..n {
        if chosen.contains(&site) {
            preferred.push(position(site));
        } else {
            rest.push(position(site));
        }
    }
    let mut placed = coords.to_owned();
    for (atom, pos) in group_a.iter().zip(preferred) {
        for axis in 0..3 {
            placed[3 * atom + axis] = pos[axis];
        }
    }
    for (atom, pos) in group_b.iter().zip(rest) {
        for axis in 0..3 {
            placed[3 * atom + axis] = pos[axis];
        }
    }
    placed
}

/// Where an fcc cut is centred.
#[derive(Clone, Copy)]
pub(crate) enum FccCentre {
    /// An occupied lattice site.
    Atom,
    /// The void with six neighbours.
    Octahedral,
    /// The void with four neighbours.
    Tetrahedral,
}

/// The `n` fcc sites nearest `centre`.
///
/// The final rescaling makes the shortest pair exactly `nearest`.  `n` only
/// truncates the distance ordering; there are no shell-size cases.
pub(crate) fn fcc_cut(n: usize, centre: FccCentre, nearest: f64) -> Array1<f64> {
    if !valid_request(n, nearest) {
        return Array1::zeros(0);
    }
    let origin = match centre {
        FccCentre::Atom => [0.0, 0.0, 0.0],
        FccCentre::Octahedral => [1.0, 0.0, 0.0],
        FccCentre::Tetrahedral => [0.5, 0.5, 0.5],
    };
    let mut limit = 1i32;
    let mut sites = Vec::new();
    while sites.len() < n {
        sites.clear();
        for i in -limit..=limit {
            for j in -limit..=limit {
                for k in -limit..=limit {
                    if (i + j + k).rem_euclid(2) == 0 {
                        sites.push([f64::from(i), f64::from(j), f64::from(k)]);
                    }
                }
            }
        }
        limit += 1;
    }
    sites.sort_by(|left, right| {
        dist2(left, &origin)
            .total_cmp(&dist2(right, &origin))
            .then_with(|| point_order(left, right))
    });
    sites.truncate(n);
    scale_to_nearest(&sites, nearest)
}

/// How successive icosahedra share sites.
#[derive(Clone, Copy)]
pub(crate) enum IcosahedronJoin {
    /// Grow around the site with the most neighbours already present.
    Compact,
    /// Grow from the most recently added centre.
    Linear,
}

/// The first `n` sites in a growth of interpenetrating icosahedra.
///
/// Every shell uses the same regular icosahedron.  Its centre is selected
/// from geometry already generated, not from a cluster-size table.
pub(crate) fn polyicosahedral_cut(n: usize, join: IcosahedronJoin, nearest: f64) -> Array1<f64> {
    if !valid_request(n, nearest) {
        return Array1::zeros(0);
    }
    let vertices = icosahedron_vertices();
    let radial = norm(vertices[0]);
    let unit: Vec<[f64; 3]> = vertices
        .iter()
        .map(|point| [point[0] / radial, point[1] / radial, point[2] / radial])
        .collect();
    let mut sites = vec![[0.0, 0.0, 0.0]];
    let mut centres = vec![0usize];
    while sites.len() < n {
        let host = match join {
            IcosahedronJoin::Linear => *centres.last().expect("one centre"),
            IcosahedronJoin::Compact => (0..sites.len())
                .filter(|site| !centres.contains(site))
                .max_by(|&left, &right| {
                    neighbour_count(&sites, left)
                        .cmp(&neighbour_count(&sites, right))
                        .then_with(|| right.cmp(&left))
                })
                .unwrap_or_else(|| sites.len() - 1),
        };
        let centre = sites[host];
        let before = sites.len();
        for direction in &unit {
            let point = [
                centre[0] + direction[0],
                centre[1] + direction[1],
                centre[2] + direction[2],
            ];
            push_distinct(&mut sites, point);
        }
        if sites.len() == before {
            let direction = unit[centres.len() % unit.len()];
            push_distinct(
                &mut sites,
                [
                    centre[0] + direction[0],
                    centre[1] + direction[1],
                    centre[2] + direction[2],
                ],
            );
        }
        match join {
            IcosahedronJoin::Compact => centres.push(host),
            IcosahedronJoin::Linear => centres.push(sites.len() - 1),
        }
    }
    sites.truncate(n);
    scale_to_nearest(&sites, nearest)
}

/// The first `n` sites of a Mackay icosahedron.
///
/// Shell `s` is the integer barycentric subdivision of every icosahedron
/// face with `i + j + k = s`.  This is the same construction at every shell,
/// so it continues past 55 atoms without a magic-size branch.
pub(crate) fn mackay_cut(n: usize, nearest: f64) -> Array1<f64> {
    if !valid_request(n, nearest) {
        return Array1::zeros(0);
    }
    let mut sites = vec![[0.0, 0.0, 0.0]];
    let mut shell = 1usize;
    while sites.len() < n {
        append_mackay_shell(&mut sites, shell);
        shell += 1;
    }
    sites.truncate(n);
    scale_to_nearest(&sites, nearest)
}

/// The `n` Mackay sites nearest a fivefold cap.
///
/// The underlying shell construction is [`mackay_cut`]'s.  Distance to an
/// outer vertex supplies one geometric ordering for every atom count.
pub(crate) fn mackay_cap(n: usize, nearest: f64) -> Array1<f64> {
    if !valid_request(n, nearest) {
        return Array1::zeros(0);
    }
    let mut sites = vec![[0.0, 0.0, 0.0]];
    let mut shell = 1usize;
    while sites.len() < n.saturating_mul(2) {
        append_mackay_shell(&mut sites, shell);
        shell += 1;
    }
    let vertex = icosahedron_vertices()[0];
    let focus = [
        vertex[0] * shell as f64,
        vertex[1] * shell as f64,
        vertex[2] * shell as f64,
    ];
    sites.sort_by(|left, right| {
        dist2(left, &focus)
            .total_cmp(&dist2(right, &focus))
            .then_with(|| point_order(left, right))
    });
    sites.truncate(n);
    scale_to_nearest(&sites, nearest)
}

/// The `n` sites nearest the axis of an hcp lattice.
///
/// This retains the axial packing role of the removed binary builder without
/// reading species, composition, or radii.  Atom count only truncates one
/// geometric ordering.
pub(crate) fn axial_spine_cut(n: usize, nearest: f64) -> Array1<f64> {
    if !valid_request(n, nearest) {
        return Array1::zeros(0);
    }
    let mut limit = 1i32;
    let mut sites = Vec::new();
    while sites.len() < n {
        sites.clear();
        for layer in -limit..=limit {
            let odd = layer.rem_euclid(2) != 0;
            let offset = if odd {
                [0.5, 3.0_f64.sqrt() / 6.0]
            } else {
                [0.0, 0.0]
            };
            for i in -limit..=limit {
                for j in -limit..=limit {
                    sites.push([
                        f64::from(i) + 0.5 * f64::from(j) + offset[0],
                        3.0_f64.sqrt() * 0.5 * f64::from(j) + offset[1],
                        f64::from(layer) * (2.0 / 3.0_f64).sqrt(),
                    ]);
                }
            }
        }
        limit += 1;
    }
    sites.sort_by(|left, right| {
        let left_radius = left[0] * left[0] + left[1] * left[1];
        let right_radius = right[0] * right[0] + right[1] * right[1];
        left_radius
            .total_cmp(&right_radius)
            .then_with(|| left[2].abs().total_cmp(&right[2].abs()))
            .then_with(|| point_order(left, right))
    });
    sites.truncate(n);
    scale_to_nearest(&sites, nearest)
}

/// All generic atomic packings offered by cluster search.
pub(crate) fn packing_candidates(n: usize, nearest: f64) -> Vec<Array1<f64>> {
    let mut candidates = Vec::new();
    for centre in [
        FccCentre::Atom,
        FccCentre::Octahedral,
        FccCentre::Tetrahedral,
    ] {
        candidates.push(fcc_cut(n, centre, nearest));
    }
    for join in [IcosahedronJoin::Compact, IcosahedronJoin::Linear] {
        candidates.push(polyicosahedral_cut(n, join, nearest));
    }
    candidates.push(mackay_cut(n, nearest));
    candidates.push(mackay_cap(n, nearest));
    candidates.push(axial_spine_cut(n, nearest));
    candidates
}

fn valid_request(n: usize, nearest: f64) -> bool {
    n > 0 && nearest.is_finite() && nearest > 0.0
}

fn append_mackay_shell(sites: &mut Vec<[f64; 3]>, shell: usize) {
    let vertices = icosahedron_vertices();
    for [a, b, c] in icosahedron_faces(&vertices) {
        for i in 0..=shell {
            for j in 0..=(shell - i) {
                let k = shell - i - j;
                let point = [
                    (i as f64 * a[0] + j as f64 * b[0] + k as f64 * c[0]),
                    (i as f64 * a[1] + j as f64 * b[1] + k as f64 * c[1]),
                    (i as f64 * a[2] + j as f64 * b[2] + k as f64 * c[2]),
                ];
                push_distinct(sites, point);
            }
        }
    }
}

fn icosahedron_vertices() -> [[f64; 3]; 12] {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    [
        [0.0, 1.0, phi],
        [0.0, 1.0, -phi],
        [0.0, -1.0, phi],
        [0.0, -1.0, -phi],
        [1.0, phi, 0.0],
        [1.0, -phi, 0.0],
        [-1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, 1.0],
        [phi, 0.0, -1.0],
        [-phi, 0.0, -1.0],
    ]
}

fn icosahedron_faces(vertices: &[[f64; 3]; 12]) -> Vec<[[f64; 3]; 3]> {
    let edge = (0..vertices.len())
        .flat_map(|i| ((i + 1)..vertices.len()).map(move |j| dist2(&vertices[i], &vertices[j])))
        .filter(|distance| *distance > 0.0)
        .min_by(f64::total_cmp)
        .expect("an icosahedron has edges");
    let tolerance = edge * f64::EPSILON.sqrt();
    let mut faces = Vec::new();
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            for k in (j + 1)..vertices.len() {
                if (dist2(&vertices[i], &vertices[j]) - edge).abs() <= tolerance
                    && (dist2(&vertices[i], &vertices[k]) - edge).abs() <= tolerance
                    && (dist2(&vertices[j], &vertices[k]) - edge).abs() <= tolerance
                {
                    faces.push([vertices[i], vertices[j], vertices[k]]);
                }
            }
        }
    }
    faces
}

fn neighbour_count(sites: &[[f64; 3]], site: usize) -> usize {
    let nearest = minimum_pair_distance(sites);
    let tolerance = nearest * f64::EPSILON.sqrt();
    sites
        .iter()
        .enumerate()
        .filter(|(other, point)| {
            *other != site && (dist2(&sites[site], point).sqrt() - nearest).abs() <= tolerance
        })
        .count()
}

fn push_distinct(sites: &mut Vec<[f64; 3]>, point: [f64; 3]) {
    let scale = norm(point).max(1.0);
    let tolerance = scale * f64::EPSILON.sqrt();
    if sites
        .iter()
        .all(|have| dist2(have, &point).sqrt() > tolerance)
    {
        sites.push(point);
    }
}

fn scale_to_nearest(sites: &[[f64; 3]], nearest: f64) -> Array1<f64> {
    let present = minimum_pair_distance(sites);
    let scale = if present.is_finite() && present > 0.0 {
        nearest / present
    } else {
        nearest
    };
    let mut out = Array1::zeros(3 * sites.len());
    for (index, point) in sites.iter().enumerate() {
        for axis in 0..3 {
            out[3 * index + axis] = scale * point[axis];
        }
    }
    out
}

fn minimum_pair_distance(sites: &[[f64; 3]]) -> f64 {
    let mut nearest = f64::INFINITY;
    for i in 0..sites.len() {
        for j in (i + 1)..sites.len() {
            nearest = nearest.min(dist2(&sites[i], &sites[j]).sqrt());
        }
    }
    nearest
}

fn norm(point: [f64; 3]) -> f64 {
    (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt()
}

fn dist2(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2)
}

fn point_order(left: &[f64; 3], right: &[f64; 3]) -> std::cmp::Ordering {
    left[0]
        .total_cmp(&right[0])
        .then_with(|| left[1].total_cmp(&right[1]))
        .then_with(|| left[2].total_cmp(&right[2]))
}

/// Coordination, then distance from the centre. Higher coordination and
/// a smaller radius sort first.
fn interior_order(coords: ArrayView1<f64>) -> Vec<usize> {
    let n = coords.len() / 3;
    let mut centre = [0.0; 3];
    for i in 0..n {
        for axis in 0..3 {
            centre[axis] += coords[3 * i + axis];
        }
    }
    for value in &mut centre {
        *value /= n.max(1) as f64;
    }
    let mut radius = vec![0.0; n];
    for i in 0..n {
        let mut r = 0.0;
        for axis in 0..3 {
            let delta = coords[3 * i + axis] - centre[axis];
            r += delta * delta;
        }
        radius[i] = r;
    }
    let mut nearest = vec![f64::INFINITY; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0;
            for axis in 0..3 {
                let delta = coords[3 * i + axis] - coords[3 * j + axis];
                d += delta * delta;
            }
            nearest[i] = nearest[i].min(d);
            nearest[j] = nearest[j].min(d);
        }
    }
    nearest.sort_by(|left, right| left.total_cmp(right));
    let bond = nearest.get(n / 2).copied().unwrap_or(1.0) * 1.25 * 1.25;
    let mut coordination = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0;
            for axis in 0..3 {
                let delta = coords[3 * i + axis] - coords[3 * j + axis];
                d += delta * delta;
            }
            if d < bond {
                coordination[i] += 1;
                coordination[j] += 1;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        coordination[right]
            .cmp(&coordination[left])
            .then(radius[left].total_cmp(&radius[right]))
            .then(left.cmp(&right))
    });
    order
}

/// Homonuclear distance of `label`: the caller's diameters when they
/// exist, otherwise the potential's like-pair `r0`.
pub(crate) fn label_radius(species: &[u32], sigma: Option<&[f64]>, label: u32) -> f64 {
    if let Some(sigma) = sigma {
        let mut sum = 0.0;
        let mut count = 0.0;
        for (index, species_label) in species.iter().enumerate() {
            if *species_label == label
                && let Some(value) = sigma.get(index).copied()
                && value.is_finite()
                && value > 0.0
            {
                sum += value;
                count += 1.0;
            }
        }
        if count > 0.0 {
            return sum / count;
        }
    }
    crate::potentials::Gupta::homonuclear_r0(label).unwrap_or(1.0)
}

/// Puts the smaller-radius species on the highest-coordination interior
/// sites. `radius` is a homonuclear distance for a label. Equal radii
/// leave the coordinates alone.
pub(crate) fn place_smaller_inside(
    coords: ArrayView1<f64>,
    species: &[u32],
    radius: impl Fn(u32) -> f64,
) -> Array1<f64> {
    let n = species.len();
    if coords.len() != 3 * n || n < 2 {
        return coords.to_owned();
    }
    if species.iter().all(|label| *label == species[0]) {
        return coords.to_owned();
    }
    let (group_a, group_b) = species_groups(species);
    if group_a.is_empty() || group_b.is_empty() {
        return coords.to_owned();
    }
    let radius_a = radius(species[group_a[0]]);
    let radius_b = radius(species[group_b[0]]);
    if !radius_a.is_finite() || !radius_b.is_finite() || (radius_a - radius_b).abs() < 1e-9 {
        return coords.to_owned();
    }
    let (small, large) = if radius_a < radius_b {
        (group_a.as_slice(), group_b.as_slice())
    } else {
        (group_b.as_slice(), group_a.as_slice())
    };
    let order = interior_order(coords);
    if order.len() < small.len() {
        return coords.to_owned();
    }
    assign_sites(coords, small, large, &order[..small.len()])
}

/// The geometry relaxed under one pair well at its own nearest-neighbour
/// distance.
///
/// A quenched alloy is shaped by its ordering: every site has relaxed
/// around the atom on it, so an exchange read on that geometry is ranked
/// against the ordering it already has. One well for every atom removes
/// that bias and keeps the packing.
pub(crate) fn label_blind(coords: ArrayView1<f64>) -> Array1<f64> {
    let nearest = crate::lattice::nearest_neighbour_scale(coords);
    if coords.len() < 6 || !(nearest > 0.0) || !nearest.is_finite() {
        return coords.to_owned();
    }
    let scale = 2f64.powf(1.0 / 6.0) / nearest;
    let reduced = coords.mapv(|value| value * scale);
    let mut solver = crate::methods::warm_lbfgs::WarmLbfgs::default();
    let (energy, relaxed, _) = solver.minimize(reduced.view(), 4 * coords.len(), |point| {
        let values = point.as_slice()?;
        let mut gradient = vec![0.0; values.len()];
        let energy = crate::methods::lattice_search::lj_value_gradient(values, &mut gradient);
        Some((energy, Array1::from_vec(gradient)))
    });
    if !energy.is_finite() || relaxed.iter().any(|value| !value.is_finite()) {
        return coords.to_owned();
    }
    relaxed.mapv(|value| value / scale)
}

pub(crate) trait QuenchProbe {
    fn energy(&mut self, point: ArrayView1<f64>) -> Option<f64>;
    fn quench(&mut self, point: ArrayView1<f64>) -> Option<(f64, Array1<f64>)>;
}

/// Quench a generated packing without reordering its sites.
///
/// The axial cut's geometric ordering is already complete. Site-energy or
/// coordination ordering would make it a different candidate.
pub(crate) fn quench_as_placed<P: QuenchProbe>(
    coords: Array1<f64>,
    probe: &mut P,
) -> Option<(f64, Array1<f64>)> {
    let (value, point) = probe.quench(coords.view())?;
    value.is_finite().then_some((value, point))
}

/// Site ranking, when two species are present, and one quench.
pub(crate) fn arrange<P: QuenchProbe>(
    mut coords: Array1<f64>,
    species: &[u32],
    probe: &mut P,
) -> Option<(f64, Array1<f64>)> {
    if species.len() >= 2 && species.iter().any(|label| *label != species[0]) {
        let (group_a, group_b) = species_groups(species);
        if let Some(moved) = reassign_by_site_energy(coords.view(), &group_a, &group_b, |point| {
            probe.energy(point)
        }) {
            coords = moved;
        }
    }
    let (value, point) = probe.quench(coords.view())?;
    value.is_finite().then_some((value, point))
}

/// Improving exchanges after [`arrange`]. One species is already arranged.
pub(crate) fn refine<P: QuenchProbe>(
    mut value: f64,
    mut point: Array1<f64>,
    species: &[u32],
    probe: &mut P,
) -> Option<(f64, Array1<f64>)> {
    if species.len() < 2 || species.iter().all(|label| *label == species[0]) {
        return Some((value, point));
    }
    let (group_a, group_b) = species_groups(species);
    for _ in 0..3 {
        let mut gained = false;
        let mut tried = 0usize;
        'pairs: for &i in &group_a {
            for &j in &group_b {
                if tried >= 800 {
                    break 'pairs;
                }
                tried += 1;
                let mut swapped = point.clone();
                for axis in 0..3 {
                    swapped.swap(3 * i + axis, 3 * j + axis);
                }
                let Some((next, relaxed)) = probe.quench(swapped.view()) else {
                    return Some((value, point));
                };
                if next < value - 1e-6 {
                    value = next;
                    point = relaxed;
                    gained = true;
                }
            }
        }
        if !gained {
            if let Some(moved) =
                reassign_by_site_energy(point.view(), &group_a, &group_b, |p| probe.energy(p))
                && let Some((next, relaxed)) = probe.quench(moved.view())
                && next < value - 1e-6
            {
                value = next;
                point = relaxed;
                continue;
            }
            break;
        }
    }
    Some((value, point))
}

/// One reassignment, a quench, then improving exchanges.
///
/// `None` when the quench is not finite. The search arranges every cut
/// before it refines, so this combined entry is the one the tests call.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn polish<P: QuenchProbe>(
    coords: Array1<f64>,
    species: &[u32],
    probe: &mut P,
) -> Option<(f64, Array1<f64>)> {
    let (value, point) = arrange(coords, species, probe)?;
    refine(value, point, species, probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shortest_pair(coordinates: ArrayView1<f64>) -> f64 {
        let n = coordinates.len() / 3;
        let mut shortest = f64::INFINITY;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut square = 0.0;
                for axis in 0..3 {
                    square += (coordinates[3 * i + axis] - coordinates[3 * j + axis]).powi(2);
                }
                shortest = shortest.min(square.sqrt());
            }
        }
        shortest
    }

    #[test]
    fn every_atom_count_uses_the_same_generic_packing_contract() {
        for n in [2usize, 3, 4, 5, 12, 13, 14, 37, 38, 39, 54, 55, 56, 79, 126] {
            let nearest = 0.73 + n as f64 / 100.0;
            let candidates = packing_candidates(n, nearest);
            assert_eq!(candidates.len(), 8);
            for (index, candidate) in candidates.iter().enumerate() {
                assert_eq!(candidate.len(), 3 * n, "generator {index}, n={n}");
                let measured = shortest_pair(candidate.view());
                assert!(
                    (measured - nearest).abs() <= nearest * 1.0e-10,
                    "generator {index}, n={n}: {measured} != {nearest}"
                );
            }
        }
    }

    #[test]
    fn axial_quench_keeps_the_generated_site_order() {
        struct Probe;

        impl QuenchProbe for Probe {
            fn energy(&mut self, _point: ArrayView1<f64>) -> Option<f64> {
                panic!("a no-reorder quench does not rank sites")
            }

            fn quench(&mut self, point: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
                Some((-1.0, point.to_owned()))
            }
        }

        let axial = axial_spine_cut(17, 1.7);
        let (energy, quenched) =
            quench_as_placed(axial.clone(), &mut Probe).expect("finite quench");
        assert_eq!(energy, -1.0);
        assert_eq!(quenched, axial);
    }

    #[test]
    fn a_label_blind_pair_keeps_its_own_distance() {
        let pair = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.7, 0.0, 0.0]);
        let blind = label_blind(pair.view());
        let distance = (blind[3] - blind[0])
            .hypot(blind[4] - blind[1])
            .hypot(blind[5] - blind[2]);
        assert!((distance - 2.7).abs() < 1e-6, "distance {distance}");
    }
}
