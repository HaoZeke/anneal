//! Geometry proposals for an atomic binary cluster.
//!
//! A collective reassignment puts one species on the sites an unrelaxed
//! swap ranks lowest, then quenched exchanges finish the ordering. The
//! cores are a close-packed fragment and interpenetrating icosahedra.
//! Neither the fragment nor the polish names a cluster size or a target
//! energy. A proposal is kept only when its quenched energy is lower.

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

/// Where an fcc cut is centred. The three holes are the generic ones:
/// a lattice site, an octahedral void, and a tetrahedral void.
#[derive(Clone, Copy)]
pub(crate) enum FccCentre {
    /// An occupied lattice site.
    Atom,
    /// The void with six neighbours.
    Octahedral,
    /// The void with four neighbours.
    Tetrahedral,
}

/// The `n` fcc sites nearest `centre`, scaled so the nearest-neighbour
/// distance is `nearest`.
///
/// Ties at a shell boundary break by coordinate order, which does not
/// depend on `n`. Every size gets a cut; nothing here is a fixed cluster.
pub(crate) fn fcc_cut(n: usize, centre: FccCentre, nearest: f64) -> Array1<f64> {
    if n == 0 || !nearest.is_finite() || nearest <= 0.0 {
        return Array1::zeros(0);
    }
    let origin = match centre {
        FccCentre::Atom => [0.0, 0.0, 0.0],
        FccCentre::Octahedral => [1.0, 0.0, 0.0],
        FccCentre::Tetrahedral => [0.5, 0.5, 0.5],
    };
    // Lattice constant chosen so a nearest neighbour is √2. The box has
    // to hold at least `n` sites; the cube root of the site density is
    // enough, with a margin so a shell is not clipped.
    let reach = (n as f64).cbrt() * 2.2 + 3.0;
    let lim = reach.ceil() as i32;
    let mut ranked: Vec<(f64, [f64; 3])> = Vec::new();
    for i in -lim..=lim {
        for j in -lim..=lim {
            for k in -lim..=lim {
                if (i + j + k).rem_euclid(2) != 0 {
                    continue;
                }
                let site = [i as f64, j as f64, k as f64];
                let delta = [
                    site[0] - origin[0],
                    site[1] - origin[1],
                    site[2] - origin[2],
                ];
                let distance = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                ranked.push((distance, site));
            }
        }
    }
    ranked.sort_by(|left, right| {
        left.0.total_cmp(&right.0).then_with(|| {
            left.1[0]
                .total_cmp(&right.1[0])
                .then(left.1[1].total_cmp(&right.1[1]))
                .then(left.1[2].total_cmp(&right.1[2]))
        })
    });
    ranked.truncate(n.min(ranked.len()));
    let scale = nearest / std::f64::consts::SQRT_2;
    let mut out = Array1::zeros(3 * ranked.len());
    for (index, (_, site)) in ranked.iter().enumerate() {
        for axis in 0..3 {
            out[3 * index + axis] = scale * site[axis];
        }
    }
    out
}

/// How a second icosahedron is attached to one already grown.
#[derive(Clone, Copy)]
pub(crate) enum IcosahedronJoin {
    /// The new centre is a site already bonded to as many centres as
    /// possible, so the added shell shares a face and stays compact.
    Compact,
    /// The new centre is the site furthest from the centres already
    /// placed, so the cluster grows as a chain of icosahedra.
    Linear,
}

/// The `n` sites of interpenetrating icosahedra, scaled so the
/// centre-to-vertex distance is `nearest`.
///
/// Growth starts from one icosahedron and keeps adding a shell about an
/// existing site. A size that falls in the middle of a shell keeps the
/// sites nearest the centres. Nothing here is a fixed cluster.
pub(crate) fn polyicosahedral_cut(n: usize, join: IcosahedronJoin, nearest: f64) -> Array1<f64> {
    if n == 0 || !nearest.is_finite() || nearest <= 0.0 {
        return Array1::zeros(0);
    }
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let vertices = [
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
    ];
    let radial = (1.0 + phi * phi).sqrt();
    let scale = nearest / radial;
    let mut sites = vec![[0.0, 0.0, 0.0]];
    for vertex in vertices {
        sites.push([vertex[0] * scale, vertex[1] * scale, vertex[2] * scale]);
    }
    let mut centres = vec![0usize];
    let bond = nearest * nearest * 1.35 * 1.35;
    let merge = (0.35 * nearest) * (0.35 * nearest);
    while sites.len() < n {
        let Some(host) = next_icosahedron_centre(&sites, &centres, bond, join) else {
            break;
        };
        let parent = nearest_centre(&sites, host, &centres);
        let added = shell_about(&sites, host, parent, &vertices, radial);
        centres.push(host);
        for point in added {
            if sites.iter().all(|have| dist2(have, &point) > merge) {
                sites.push(point);
            }
        }
    }
    if sites.len() > n {
        let origin = centroid(&sites, &centres);
        let mut order: Vec<usize> = (0..sites.len()).collect();
        order.sort_by(|&left, &right| {
            dist2(&sites[left], &origin)
                .total_cmp(&dist2(&sites[right], &origin))
                .then(left.cmp(&right))
        });
        order.truncate(n);
        order.sort_unstable();
        sites = order.into_iter().map(|index| sites[index]).collect();
    }
    let mut out = Array1::zeros(3 * sites.len());
    for (index, site) in sites.iter().enumerate() {
        for axis in 0..3 {
            out[3 * index + axis] = site[axis];
        }
    }
    out
}

/// Mackay icosahedron through two shells, then the sites nearest the centre.
///
/// Shell one is the twelve vertices. Shell two is those vertices again at
/// twice the radius and one atom on each of the thirty edges. `n` below 55
/// keeps the innermost sites. Larger `n` is that body; the growth does not
/// invent a third shell.
pub(crate) fn mackay_cut(n: usize, nearest: f64) -> Array1<f64> {
    scale_sites(&mackay_sites_centred(n), nearest)
}

/// The `n` sites of the two-shell Mackay icosahedron closest to one vertex.
///
/// At the complete two-shell body this is every site. Below that it is the
/// fivefold cap: the sites nearest one vertex. The construction does not
/// branch on a size.
pub(crate) fn mackay_cap(n: usize, nearest: f64) -> Array1<f64> {
    let mut sites = mackay_sites();
    if n == 0 || sites.len() < n {
        return Array1::zeros(0);
    }
    let vertex = sites[1];
    sites.sort_by(|left, right| dist2(left, &vertex).total_cmp(&dist2(right, &vertex)));
    sites.truncate(n);
    scale_sites(&sites, nearest)
}

fn mackay_sites_centred(n: usize) -> Vec<[f64; 3]> {
    let mut sites = mackay_sites();
    let origin = [0.0, 0.0, 0.0];
    sites.sort_by(|left, right| dist2(left, &origin).total_cmp(&dist2(right, &origin)));
    let count = n.min(sites.len());
    sites.truncate(count);
    sites
}

fn mackay_sites() -> Vec<[f64; 3]> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let vertices = [
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
    ];
    let mut sites = vec![[0.0, 0.0, 0.0]];
    sites.extend(vertices);
    for vertex in vertices {
        sites.push([2.0 * vertex[0], 2.0 * vertex[1], 2.0 * vertex[2]]);
    }
    let edge = {
        let mut best = f64::MAX;
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                best = best.min(dist2(&vertices[i], &vertices[j]));
            }
        }
        best * 1.21
    };
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            if dist2(&vertices[i], &vertices[j]) <= edge {
                let mid = [
                    vertices[i][0] + vertices[j][0],
                    vertices[i][1] + vertices[j][1],
                    vertices[i][2] + vertices[j][2],
                ];
                sites.push(mid);
            }
        }
    }
    sites
}

fn scale_sites(sites: &[[f64; 3]], nearest: f64) -> Array1<f64> {
    if sites.is_empty() || !nearest.is_finite() || nearest <= 0.0 {
        return Array1::zeros(0);
    }
    let scale = nearest / mean_nearest(sites).max(1e-6);
    let mut out = Array1::zeros(3 * sites.len());
    for (index, site) in sites.iter().enumerate() {
        for axis in 0..3 {
            out[3 * index + axis] = scale * site[axis];
        }
    }
    out
}

/// Homonuclear dimer through a ring of the minority species, then majority
/// rings along the axis.
///
/// The ring has one site per minority atom. The dimer and the caps are
/// the majority. Bond lengths are the homonuclear distances and their
/// mean, so the shape follows the composition rather than the cluster size.
/// Coordinates are already written onto the species indices.
pub(crate) fn axial_spine_cut(species: &[u32], radius: impl Fn(u32) -> f64) -> Option<Array1<f64>> {
    let n = species.len();
    if n < 5 {
        return None;
    }
    let mut labels = Vec::new();
    for label in species {
        if !labels.contains(label) {
            labels.push(*label);
        }
    }
    if labels.len() != 2 {
        return None;
    }
    let count = |label: u32| species.iter().filter(|item| **item == label).count();
    let (minority, majority) = {
        let (a, b) = (labels[0], labels[1]);
        let (na, nb) = (count(a), count(b));
        if na < nb {
            (a, b)
        } else if nb < na {
            (b, a)
        } else if radius(a) <= radius(b) {
            (a, b)
        } else {
            (b, a)
        }
    };
    let k = count(minority);
    let n_majority = count(majority);
    if k < 3 || n_majority < 2 {
        return None;
    }
    let ring = radius(minority);
    let cap = radius(majority);
    if !(ring.is_finite() && cap.is_finite() && ring > 0.0 && cap > 0.0) {
        return None;
    }
    let rho = ring / (2.0 * (std::f64::consts::PI / k as f64).sin());
    let spoke = 0.5 * (ring + cap);
    let half = if spoke > rho {
        (spoke * spoke - rho * rho).sqrt()
    } else {
        0.5 * cap
    };
    let mut spine = Vec::with_capacity(n);
    spine.push([0.0, 0.0, half]);
    spine.push([0.0, 0.0, -half]);
    push_ring(&mut spine, k, rho, 0.0, 0.0);
    let mut left = n_majority - 2;
    let outer = ring_radius_outside(rho, k, cap);
    let equatorial = left.min(k);
    push_ring_partial(
        &mut spine,
        equatorial,
        k,
        outer,
        0.0,
        std::f64::consts::PI / k as f64,
    );
    left -= equatorial;
    let layer = cap * (2.0 / 3.0_f64).sqrt();
    let mut shell = 1usize;
    while left > 0 && shell < 6 {
        let twist = if shell % 2 == 1 {
            0.0
        } else {
            std::f64::consts::PI / k as f64
        };
        let radius = if shell % 2 == 1 { outer } else { rho };
        for side in [-1.0, 1.0] {
            if left == 0 {
                break;
            }
            let take = left.min(k);
            push_ring_partial(
                &mut spine,
                take,
                k,
                radius,
                side * shell as f64 * layer,
                twist,
            );
            left -= take;
        }
        shell += 1;
    }
    if spine.len() != n {
        return None;
    }
    let mut minority_sites = Vec::new();
    let mut majority_sites = Vec::new();
    // The first two sites are the dimer, the next k are the ring, the rest caps.
    majority_sites.push(spine[0]);
    majority_sites.push(spine[1]);
    minority_sites.extend_from_slice(&spine[2..2 + k]);
    majority_sites.extend_from_slice(&spine[2 + k..]);
    let mut out = Array1::zeros(3 * n);
    let mut minor_at = 0usize;
    let mut major_at = 0usize;
    for (index, label) in species.iter().enumerate() {
        let site = if *label == minority {
            let site = minority_sites[minor_at];
            minor_at += 1;
            site
        } else {
            let site = majority_sites[major_at];
            major_at += 1;
            site
        };
        for axis in 0..3 {
            out[3 * index + axis] = site[axis];
        }
    }
    Some(out)
}

fn ring_radius_outside(inner: f64, k: usize, bond: f64) -> f64 {
    let cosine = (std::f64::consts::PI / k as f64).cos();
    let linear = -2.0 * inner * cosine;
    let constant = inner * inner - bond * bond;
    let disc = linear * linear - 4.0 * constant;
    if disc <= 0.0 {
        return inner + bond;
    }
    let root = (-linear + disc.sqrt()) * 0.5;
    if root > inner { root } else { inner + bond }
}

fn push_ring(sites: &mut Vec<[f64; 3]>, k: usize, radius: f64, height: f64, twist: f64) {
    push_ring_partial(sites, k, k, radius, height, twist);
}

fn push_ring_partial(
    sites: &mut Vec<[f64; 3]>,
    take: usize,
    k: usize,
    radius: f64,
    height: f64,
    twist: f64,
) {
    for index in 0..take {
        let angle = twist + 2.0 * std::f64::consts::PI * index as f64 / k as f64;
        sites.push([radius * angle.cos(), radius * angle.sin(), height]);
    }
}

fn mean_nearest(sites: &[[f64; 3]]) -> f64 {
    if sites.len() < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    for site in sites {
        let mut best = f64::MAX;
        for other in sites {
            let distance = dist2(site, other);
            if distance > 1e-12 {
                best = best.min(distance);
            }
        }
        total += best.sqrt();
    }
    total / sites.len() as f64
}

fn next_icosahedron_centre(
    sites: &[[f64; 3]],
    centres: &[usize],
    bond: f64,
    join: IcosahedronJoin,
) -> Option<usize> {
    let origin = centroid(sites, centres);
    let mut best: Option<(usize, usize, f64)> = None;
    for index in 0..sites.len() {
        if centres.contains(&index) {
            continue;
        }
        let parents = centres
            .iter()
            .filter(|centre| dist2(&sites[index], &sites[**centre]) <= bond)
            .count();
        if parents == 0 {
            continue;
        }
        let reach = dist2(&sites[index], &origin);
        let better = match join {
            IcosahedronJoin::Compact => best.is_none_or(|(held_parents, _, held_reach)| {
                parents > held_parents || (parents == held_parents && reach < held_reach - 1e-12)
            }),
            IcosahedronJoin::Linear => {
                best.is_none_or(|(_, _, held_reach)| reach > held_reach + 1e-12)
            }
        };
        if better {
            best = Some((parents, index, reach));
        }
    }
    best.map(|(_, index, _)| index)
}

fn nearest_centre(sites: &[[f64; 3]], host: usize, centres: &[usize]) -> usize {
    centres
        .iter()
        .copied()
        .filter(|centre| *centre != host)
        .min_by(|&left, &right| {
            dist2(&sites[host], &sites[left]).total_cmp(&dist2(&sites[host], &sites[right]))
        })
        .unwrap_or(0)
}

/// Twelve neighbours of `sites[host]`, one on `sites[parent]` and one
/// turned onto another site already bonded to the host.
fn shell_about(
    sites: &[[f64; 3]],
    host: usize,
    parent: usize,
    vertices: &[[f64; 3]; 12],
    radial: f64,
) -> Vec<[f64; 3]> {
    let centre = sites[host];
    let axis = [
        sites[parent][0] - centre[0],
        sites[parent][1] - centre[1],
        sites[parent][2] - centre[2],
    ];
    let align = rotation_from_to(vertices[0], axis);
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let scale = if length > 1e-12 { length / radial } else { 1.0 };
    let adjacent = (1..vertices.len())
        .min_by(|&left, &right| {
            dist2(&vertices[0], &vertices[left]).total_cmp(&dist2(&vertices[0], &vertices[right]))
        })
        .unwrap_or(1);
    let spoke = apply(align, vertices[adjacent]);
    let anchor = sites
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != host && *index != parent)
        .min_by(|(_, left), (_, right)| dist2(&centre, left).total_cmp(&dist2(&centre, right)))
        .map(|(_, point)| {
            [
                point[0] - centre[0],
                point[1] - centre[1],
                point[2] - centre[2],
            ]
        });
    let twist = anchor
        .map(|target| rotation_around(axis, spoke, target))
        .unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    vertices
        .iter()
        .map(|vertex| {
            let turned = apply(twist, apply(align, *vertex));
            [
                centre[0] + turned[0] * scale,
                centre[1] + turned[1] * scale,
                centre[2] + turned[2] * scale,
            ]
        })
        .collect()
}

fn rotation_around(axis: [f64; 3], from: [f64; 3], to: [f64; 3]) -> [[f64; 3]; 3] {
    let hat = unit(axis);
    let left = reject(from, hat);
    let right = reject(to, hat);
    if norm2(left) < 1e-16 || norm2(right) < 1e-16 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let u = unit(left);
    let v = unit(right);
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
    let sign = {
        let crossed = cross(u, v);
        let along = crossed[0] * hat[0] + crossed[1] * hat[1] + crossed[2] * hat[2];
        if along < 0.0 { -1.0 } else { 1.0 }
    };
    let angle = sign * dot.acos();
    let (sin, cos) = angle.sin_cos();
    let mut out = [[0.0; 3]; 3];
    for column in 0..3 {
        let basis = [
            if column == 0 { 1.0 } else { 0.0 },
            if column == 1 { 1.0 } else { 0.0 },
            if column == 2 { 1.0 } else { 0.0 },
        ];
        let crossed = cross(hat, basis);
        let along = hat[0] * basis[0] + hat[1] * basis[1] + hat[2] * basis[2];
        for row in 0..3 {
            out[row][column] =
                basis[row] * cos + crossed[row] * sin + hat[row] * along * (1.0 - cos);
        }
    }
    out
}

fn reject(point: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    let along = point[0] * axis[0] + point[1] * axis[1] + point[2] * axis[2];
    [
        point[0] - axis[0] * along,
        point[1] - axis[1] * along,
        point[2] - axis[2] * along,
    ]
}

fn norm2(point: [f64; 3]) -> f64 {
    point[0] * point[0] + point[1] * point[1] + point[2] * point[2]
}

fn rotation_from_to(from: [f64; 3], to: [f64; 3]) -> [[f64; 3]; 3] {
    let u = unit(from);
    let v = unit(to);
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
    if dot > 1.0 - 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let axis = if dot < -1.0 + 1e-12 {
        let mut ortho = [-u[1], u[0], 0.0];
        if ortho[0] * ortho[0] + ortho[1] * ortho[1] < 1e-12 {
            ortho = [0.0, -u[2], u[1]];
        }
        unit(ortho)
    } else {
        unit(cross(u, v))
    };
    let angle = dot.acos();
    let (sin, cos) = angle.sin_cos();
    let mut out = [[0.0; 3]; 3];
    for column in 0..3 {
        let basis = [
            if column == 0 { 1.0 } else { 0.0 },
            if column == 1 { 1.0 } else { 0.0 },
            if column == 2 { 1.0 } else { 0.0 },
        ];
        let crossed = cross(axis, basis);
        let along = axis[0] * basis[0] + axis[1] * basis[1] + axis[2] * basis[2];
        for row in 0..3 {
            out[row][column] =
                basis[row] * cos + crossed[row] * sin + axis[row] * along * (1.0 - cos);
        }
    }
    out
}

fn apply(matrix: [[f64; 3]; 3], point: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2],
    ]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn unit(point: [f64; 3]) -> [f64; 3] {
    let length = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
    if length < 1e-12 {
        return [1.0, 0.0, 0.0];
    }
    [point[0] / length, point[1] / length, point[2] / length]
}

fn dist2(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    dx * dx + dy * dy + dz * dz
}

fn centroid(sites: &[[f64; 3]], indices: &[usize]) -> [f64; 3] {
    if indices.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut sum = [0.0, 0.0, 0.0];
    for index in indices {
        for axis in 0..3 {
            sum[axis] += sites[*index][axis];
        }
    }
    let scale = indices.len() as f64;
    [sum[0] / scale, sum[1] / scale, sum[2] / scale]
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

pub(crate) trait QuenchProbe {
    fn energy(&mut self, point: ArrayView1<f64>) -> Option<f64>;
    fn quench(&mut self, point: ArrayView1<f64>) -> Option<(f64, Array1<f64>)>;
}

/// Site ranking, when two species are present, and one quench.
/// Quench without reordering. The spine cut is already written onto the
/// species indices, and a coordination sort would move the minority ring.
pub(crate) fn quench_as_placed<P: QuenchProbe>(
    coords: Array1<f64>,
    probe: &mut P,
) -> Option<(f64, Array1<f64>)> {
    let (value, point) = probe.quench(coords.view())?;
    value.is_finite().then_some((value, point))
}

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
        if !gained
            && let Some(moved) =
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
    use crate::methods::warm_lbfgs::WarmLbfgs;
    use crate::potentials::{COPPER_Z, GOLD_Z, Gupta};
    fn load(text: &str) -> (Vec<u32>, Array1<f64>) {
        let mut labels = Vec::new();
        let mut values = Vec::new();
        for line in text.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 4 {
                continue;
            }
            let Ok(kind) = fields[0].parse::<u32>() else {
                continue;
            };
            let Ok(row) = fields[1..]
                .iter()
                .map(|word| word.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
            else {
                continue;
            };
            labels.push(if kind == 0 { COPPER_Z } else { GOLD_Z });
            values.extend(row);
        }
        (labels, Array1::from_vec(values))
    }

    fn radius(label: u32) -> f64 {
        if label == COPPER_Z { 2.556 } else { 2.884 }
    }

    struct Direct<'a> {
        pot: &'a Gupta,
    }

    impl QuenchProbe for Direct<'_> {
        fn energy(&mut self, point: ArrayView1<f64>) -> Option<f64> {
            Some(self.pot.value_and_gradient(point).0)
        }

        fn quench(&mut self, point: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
            let mut solver = WarmLbfgs::default();
            let (energy, relaxed, _) =
                solver.minimize(point, 80, |v| Some(self.pot.value_and_gradient(v)));
            energy.is_finite().then_some((energy, relaxed))
        }
    }

    #[test]
    fn an_fcc_cut_has_one_site_per_atom_at_every_size() {
        for n in [2usize, 13, 19, 38, 55, 75] {
            for centre in [
                FccCentre::Atom,
                FccCentre::Octahedral,
                FccCentre::Tetrahedral,
            ] {
                let cut = fcc_cut(n, centre, 1.0);
                assert_eq!(cut.len(), 3 * n, "n {n}");
                for i in 0..n {
                    for j in (i + 1)..n {
                        let mut distance = 0.0;
                        for axis in 0..3 {
                            let delta = cut[3 * i + axis] - cut[3 * j + axis];
                            distance += delta * delta;
                        }
                        assert!(distance > 1e-8, "duplicate n {n} {i} {j}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_polyicosahedron_has_one_site_per_atom() {
        for n in [2usize, 13, 19, 23, 26, 38, 55] {
            for join in [IcosahedronJoin::Compact, IcosahedronJoin::Linear] {
                let cut = polyicosahedral_cut(n, join, 2.7);
                assert_eq!(cut.len(), 3 * n, "n {n}");
                for i in 0..n {
                    for j in (i + 1)..n {
                        let mut distance = 0.0;
                        for axis in 0..3 {
                            let delta = cut[3 * i + axis] - cut[3 * j + axis];
                            distance += delta * delta;
                        }
                        assert!(distance > 1e-6, "duplicate n {n} {i} {j}");
                    }
                }
            }
        }
    }

    #[test]
    fn an_axial_spine_orders_to_cu6au32() {
        let (labels, _) = load(include_str!(
            "../../../tests/fixtures/ccd/cuau38/Cu6Au32.txt"
        ));
        let pot = Gupta::elements(&labels);
        let spine = axial_spine_cut(&labels, radius).expect("spine");
        assert_eq!(spine.len(), 3 * labels.len());
        let (energy, point) = quench_as_placed(spine, &mut Direct { pot: &pot }).expect("quench");
        let best = refine(energy, point, &labels, &mut Direct { pot: &pot })
            .map(|(value, _)| value)
            .unwrap_or(energy);
        assert!(
            best < -130.79994836 + 1e-3,
            "axial spine {best:.6} from quench {energy:.6}"
        );
    }

    #[test]
    fn a_mackay_cut_orders_to_cu27au28() {
        let (labels, _) = load(include_str!(
            "../../../tests/fixtures/ccd/cuau55/Cu27Au28.txt"
        ));
        let pot = Gupta::elements(&labels);
        let nearest = (2.556 + 2.884) / 2.0;
        let grown = mackay_cut(labels.len(), nearest);
        assert_eq!(grown.len(), 3 * labels.len());
        let placed = place_smaller_inside(grown.view(), &labels, radius);
        let (energy, _) = polish(placed, &labels, &mut Direct { pot: &pot }).expect("polish");
        assert!(
            (energy + 182.93136675).abs() < 1e-3,
            "mackay cut {energy:.6}"
        );
    }

    #[test]
    fn a_compact_polyicosahedron_orders_to_the_ptcu_hits() {
        let cases = [
            (12usize, 11usize, -96.03443577),
            (11, 12, -94.01575979),
            (6, 17, -81.90409805),
        ];
        for (n_platinum, n_copper, reference) in cases {
            let labels = Gupta::platinum_copper_species(n_platinum, n_copper);
            let pot = Gupta::platinum_copper(n_platinum, n_copper);
            let nearest = (2.775 + 2.556) / 2.0;
            let mut best = f64::INFINITY;
            let mut note = String::new();
            for join in [IcosahedronJoin::Compact, IcosahedronJoin::Linear] {
                let grown = polyicosahedral_cut(labels.len(), join, nearest);
                let raw = crate::structure::cna(grown.view(), labels.len(), nearest * 1.2);
                let placed = place_smaller_inside(grown.view(), &labels, |label| {
                    Gupta::homonuclear_r0(label).unwrap_or(nearest)
                });
                let energy = polish(placed, &labels, &mut Direct { pot: &pot })
                    .map(|(value, _)| value)
                    .unwrap_or(f64::INFINITY);
                note.push_str(&format!(" {energy:.4}/555:{}", raw.get((5, 5, 5))));
                best = best.min(energy);
            }
            assert!(
                best < reference + 1e-3,
                "Pt{n_platinum}Cu{n_copper} {best:.6} against {reference:.6}{note}"
            );
        }
    }

    #[test]
    fn an_octahedral_fcc_cut_orders_to_cu19au19() {
        let (labels, _) = load(include_str!(
            "../../../tests/fixtures/ccd/cuau38/Cu19Au19.txt"
        ));
        let pot = Gupta::elements(&labels);
        let grown = fcc_cut(labels.len(), FccCentre::Octahedral, 2.7);
        let placed = place_smaller_inside(grown.view(), &labels, radius);
        let (energy, _) = polish(placed, &labels, &mut Direct { pot: &pot }).expect("polish");
        assert!(
            (energy + 122.99148391).abs() < 1e-3,
            "octahedral cut {energy:.6}"
        );
    }
}
