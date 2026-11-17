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
