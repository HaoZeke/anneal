//! Fingerprint taboo and the energy spread of the structures already kept.
//!
//! A proposal whose SOAP spectrum matches one already stored is refused
//! unless it is lower than that structure. The spread of the stored
//! energies is the temperature of a later hop. No packing is named.

use ndarray::{Array1, ArrayView1};

use crate::soap::{SoapSpec, local_spectra_z};

struct Seat {
    energy: f64,
    soap: Vec<f64>,
    coords: Array1<f64>,
}

/// Unlike-pair coordinate exchanges of one geometry.
///
/// The smaller label is paired with the larger one. Each candidate swaps
/// one inner atom of the smaller label with one outer atom of the larger
/// label. Species counts stay on the indices. More than two labels uses
/// the two most common.
pub fn homotop_candidates(x: ArrayView1<f64>, species: &[u32]) -> Vec<Array1<f64>> {
    let n = species.len().min(x.len() / 3);
    if n < 2 {
        return Vec::new();
    }
    let mut labels: Vec<u32> = species[..n].to_vec();
    labels.sort_unstable();
    labels.dedup();
    if labels.len() < 2 {
        return Vec::new();
    }
    let count = |label: u32| species[..n].iter().filter(|item| **item == label).count();
    labels.sort_by_key(|&label| std::cmp::Reverse(count(label)));
    let majority = labels[0];
    let minority = labels[1];
    let mut centre = [0.0_f64; 3];
    for index in 0..n {
        for axis in 0..3 {
            centre[axis] += x[3 * index + axis];
        }
    }
    for component in &mut centre {
        *component /= n as f64;
    }
    let radius = |index: usize| {
        (0..3)
            .map(|axis| {
                let delta = x[3 * index + axis] - centre[axis];
                delta * delta
            })
            .sum::<f64>()
    };
    let mut inner: Vec<usize> = (0..n).filter(|&index| species[index] == minority).collect();
    let mut outer: Vec<usize> = (0..n).filter(|&index| species[index] == majority).collect();
    inner.sort_by(|&left, &right| radius(left).total_cmp(&radius(right)));
    outer.sort_by(|&left, &right| radius(right).total_cmp(&radius(left)));
    inner.truncate(4);
    outer.truncate(3);
    let mut out = Vec::new();
    for &left in &inner {
        for &right in &outer {
            let mut y = x.to_owned();
            for axis in 0..3 {
                y.swap(3 * left + axis, 3 * right + axis);
            }
            out.push(y);
            if out.len() >= 8 {
                return out;
            }
        }
    }
    out
}

/// How often the collective plane family is forced onto the chain.
///
/// It starts on every eighth hop. A quenched cut that sets a new best
/// keeps that share. A cut that only beats the chain, and a cut that
/// beats neither, lower the rate, and the stride widens. The family
/// then spends fewer evaluations, which leaves the moves that were
/// already improving the chain in place. No size and no composition
/// enters the rate.
#[derive(Clone, Debug)]
pub struct CollectiveShare {
    ema: f64,
    stride: u32,
}

impl Default for CollectiveShare {
    fn default() -> Self {
        Self {
            ema: 1.0,
            stride: 8,
        }
    }
}

impl CollectiveShare {
    /// The hop interval on which the family is forced.
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// True on the hops where the family takes the proposal.
    pub fn due(&self, hops: u64) -> bool {
        self.stride > 0 && hops % u64::from(self.stride) == 2
    }

    /// One attempt. `accepted` beat the chain. `recorded` set a new best.
    pub fn observe(&mut self, accepted: bool, recorded: bool) {
        if recorded {
            self.ema = 1.0;
            self.stride = 8;
            return;
        }
        let reward = if accepted { 0.25 } else { 0.0 };
        self.ema = 0.5 * self.ema + 0.5 * reward;
        let share = self.ema.clamp(1.0 / 32.0, 1.0);
        let raw = (8.0 / share).round() as u32;
        self.stride = raw.clamp(8, 256).next_power_of_two();
    }

    /// A later family. The first attempt waits out a stretch of ordinary
    /// hops, and a miss widens the gap instead of taking the next one.
    pub fn patient() -> Self {
        Self {
            ema: 0.125,
            stride: 64,
        }
    }
}

/// Acceptance temperature of one family, as a multiple of the configured one.
///
/// A cold family that nothing accepts is heated. A family that accepts
/// almost every proposal is cooled. The multiple stays inside a fixed
/// factor of the configured temperature.
#[derive(Clone, Debug)]
pub struct MoveTemperature {
    attempts: u32,
    accepts: u32,
    scale: f64,
}

impl Default for MoveTemperature {
    fn default() -> Self {
        Self {
            attempts: 0,
            accepts: 0,
            scale: 1.0,
        }
    }
}

impl MoveTemperature {
    /// Temperature for the next decision.
    pub fn value(&self, base: f64) -> f64 {
        base.abs() * self.scale
    }

    /// Metropolis test, then a step of the scale toward one acceptance in five.
    pub fn accept(&mut self, uphill: f64, temperature: f64, draw: f64) -> bool {
        let ok = uphill <= 0.0 || (temperature > 0.0 && draw < (-uphill / temperature).exp());
        self.attempts = self.attempts.saturating_add(1);
        if ok {
            self.accepts = self.accepts.saturating_add(1);
        }
        if self.attempts >= 4 {
            let rate = f64::from(self.accepts) / f64::from(self.attempts);
            if rate < 0.2 {
                self.scale = (self.scale * 1.5).min(8.0);
            } else if rate > 0.5 {
                self.scale = (self.scale * 0.67).max(0.125);
            }
            self.attempts = 0;
            self.accepts = 0;
        }
        ok
    }
}

/// Grows a cluster back onto the holes of its own best-coordinated atoms.
///
/// The bond length is the median nearest-neighbour distance. Each later
/// atom is parked far from that core and then returned to the tetrahedral
/// hole, or to the seat it already had, that the energy ranks lowest.
/// The two apexes of a face are the generic hole; no packing is named.
pub fn regrow_by_energy<F>(coords: ArrayView1<f64>, mut energy: F) -> Option<Array1<f64>>
where
    F: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let n = coords.len() / 3;
    if n < 6 || coords.len() != 3 * n {
        return None;
    }
    let bond = median_bond(coords);
    if !(bond.is_finite() && bond > 1e-8) {
        return None;
    }
    let mut work = coords.to_owned();
    // A simplex at the walker's own bond length. Every later atom sits on a
    // hole of what has already been placed, so the body is whatever the
    // energy builds, not a named packing.
    let height = (6.0_f64).sqrt() / 3.0 * bond;
    let simplex = [
        [0.0, 0.0, 0.0],
        [bond, 0.0, 0.0],
        [0.5 * bond, 0.5 * (3.0_f64).sqrt() * bond, 0.0],
        [0.5 * bond, bond / (2.0 * (3.0_f64).sqrt()), height],
    ];
    for (atom, pos) in simplex.iter().enumerate() {
        set_point(&mut work, atom, *pos);
    }
    for atom in 4..n {
        set_point(&mut work, atom, [40.0 * bond + atom as f64, 0.0, 0.0]);
    }
    let mut placed: Vec<usize> = (0..4).collect();
    for atom in 4..n {
        let mut best_pos = point(&work, atom);
        let mut best_value = f64::INFINITY;
        let mut candidates = Vec::new();
        for face in open_faces(&work, &placed, bond) {
            let a = point(&work, face[0]);
            let b = point(&work, face[1]);
            let c = point(&work, face[2]);
            if let Some((up, down)) = tetrahedron_apexes(a, b, c) {
                let face = [
                    (a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0,
                    (a[2] + b[2] + c[2]) / 3.0,
                ];
                for (apex, scale) in [(up, 0.8), (up, 1.0), (up, 1.2), (down, 1.0)] {
                    candidates.push([
                        face[0] + (apex[0] - face[0]) * scale,
                        face[1] + (apex[1] - face[1]) * scale,
                        face[2] + (apex[2] - face[2]) * scale,
                    ]);
                }
            }
        }
        for pos in candidates {
            if occupied(&work, &placed, pos, 0.55 * bond) {
                continue;
            }
            set_point(&mut work, atom, pos);
            let value = energy(work.view())?;
            if value.is_finite() && value < best_value {
                best_value = value;
                best_pos = pos;
            }
        }
        if !best_value.is_finite() {
            return None;
        }
        set_point(&mut work, atom, best_pos);
        placed.push(atom);
    }
    Some(work)
}

fn point(coords: &Array1<f64>, index: usize) -> [f64; 3] {
    [
        coords[3 * index],
        coords[3 * index + 1],
        coords[3 * index + 2],
    ]
}

fn set_point(coords: &mut Array1<f64>, index: usize, pos: [f64; 3]) {
    for axis in 0..3 {
        coords[3 * index + axis] = pos[axis];
    }
}

fn median_bond(coords: ArrayView1<f64>) -> f64 {
    let n = coords.len() / 3;
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
    nearest.get(n / 2).copied().unwrap_or(1.0).sqrt()
}

fn open_faces(coords: &Array1<f64>, placed: &[usize], bond: f64) -> Vec<[usize; 3]> {
    let limit = (1.35 * bond) * (1.35 * bond);
    let close = |left: usize, right: usize| {
        let mut d = 0.0;
        for axis in 0..3 {
            let delta = coords[3 * left + axis] - coords[3 * right + axis];
            d += delta * delta;
        }
        d < limit && d > 1e-8
    };
    let mut centre = [0.0_f64; 3];
    for &index in placed {
        for axis in 0..3 {
            centre[axis] += coords[3 * index + axis];
        }
    }
    let scale = placed.len().max(1) as f64;
    for value in &mut centre {
        *value /= scale;
    }
    let mut faces = Vec::new();
    for (i, &a) in placed.iter().enumerate() {
        for (j, &b) in placed.iter().enumerate().skip(i + 1) {
            if !close(a, b) {
                continue;
            }
            for &c in placed.iter().skip(j + 1) {
                if close(a, c) && close(b, c) {
                    let g = [
                        (coords[3 * a] + coords[3 * b] + coords[3 * c]) / 3.0,
                        (coords[3 * a + 1] + coords[3 * b + 1] + coords[3 * c + 1]) / 3.0,
                        (coords[3 * a + 2] + coords[3 * b + 2] + coords[3 * c + 2]) / 3.0,
                    ];
                    let mut radius = 0.0;
                    for axis in 0..3 {
                        let delta = g[axis] - centre[axis];
                        radius += delta * delta;
                    }
                    faces.push((radius, [a, b, c]));
                }
            }
        }
    }
    faces.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut chosen = Vec::new();
    for (_, face) in faces.iter().take(12) {
        chosen.push(*face);
    }
    for (_, face) in faces.iter().rev().take(24) {
        if !chosen.contains(face) {
            chosen.push(*face);
        }
    }
    chosen
}

fn tetrahedron_apexes(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let nlen = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if nlen < 1e-8 {
        return None;
    }
    let edge = ((ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).sqrt()
        + (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]).sqrt()
        + ((c[0] - b[0]).powi(2) + (c[1] - b[1]).powi(2) + (c[2] - b[2]).powi(2)).sqrt())
        / 3.0;
    let height = (6.0_f64).sqrt() / 3.0 * edge;
    let g = [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ];
    let unit = [normal[0] / nlen, normal[1] / nlen, normal[2] / nlen];
    let up = [
        g[0] + unit[0] * height,
        g[1] + unit[1] * height,
        g[2] + unit[2] * height,
    ];
    let down = [
        g[0] - unit[0] * height,
        g[1] - unit[1] * height,
        g[2] - unit[2] * height,
    ];
    Some((up, down))
}

fn occupied(coords: &Array1<f64>, placed: &[usize], pos: [f64; 3], tol: f64) -> bool {
    let tol2 = tol * tol;
    placed.iter().any(|&index| {
        let mut d = 0.0;
        for axis in 0..3 {
            let delta = coords[3 * index + axis] - pos[axis];
            d += delta * delta;
        }
        d < tol2
    })
}

/// Structures already accepted, keyed by their SOAP spectrum.
#[derive(Default)]
pub struct FingerprintTaboo {
    seats: Vec<Seat>,
}

impl FingerprintTaboo {
    /// Standard deviation of the stored energies, once three exist.
    pub fn spread(&self) -> Option<f64> {
        if self.seats.len() < 3 {
            return None;
        }
        let mean = self.seats.iter().map(|seat| seat.energy).sum::<f64>() / self.seats.len() as f64;
        let var = self
            .seats
            .iter()
            .map(|seat| {
                let delta = seat.energy - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.seats.len() as f64;
        Some(var.sqrt())
    }

    /// True when `coords` matches a stored spectrum and is not lower.
    pub fn forbids(&self, energy: f64, coords: ArrayView1<f64>, species: &[u32]) -> bool {
        let Some(soap) = spectrum(coords, species) else {
            return false;
        };
        self.seats
            .iter()
            .any(|seat| soap_distance(&seat.soap, &soap) < 0.02 && energy >= seat.energy - 1e-6)
    }

    /// Records a structure that was accepted.
    pub fn remember(&mut self, energy: f64, coords: ArrayView1<f64>, species: &[u32]) {
        let Some(soap) = spectrum(coords, species) else {
            return;
        };
        if let Some(seat) = self
            .seats
            .iter_mut()
            .find(|seat| soap_distance(&seat.soap, &soap) < 0.02)
        {
            if energy < seat.energy {
                seat.energy = energy;
                seat.soap = soap;
                seat.coords = coords.to_owned();
            }
            return;
        }
        if self.seats.len() >= 24
            && let Some(worst) = self
                .seats
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.energy.total_cmp(&right.energy))
                .map(|(index, _)| index)
        {
            self.seats.remove(worst);
        }
        self.seats.push(Seat {
            energy,
            soap,
            coords: coords.to_owned(),
        });
    }

    /// Lowest stored structure, for a plateau return to the population.
    pub fn lowest(&self) -> Option<(f64, Array1<f64>)> {
        self.seats
            .iter()
            .min_by(|left, right| left.energy.total_cmp(&right.energy))
            .map(|seat| (seat.energy, seat.coords.clone()))
    }
}

fn spectrum(coords: ArrayView1<f64>, species: &[u32]) -> Option<Vec<f64>> {
    let n = coords.len() / 3;
    if n < 2 || species.len() != n {
        return None;
    }
    let spec = SoapSpec {
        n_max: 3,
        l_max: 3,
        rcut_nn: 6.0,
    };
    let local = local_spectra_z(coords, spec, Some(species));
    if local.nrows() == 0 || local.ncols() == 0 {
        return None;
    }
    let mut mean = vec![0.0; local.ncols()];
    for row in 0..local.nrows() {
        for col in 0..local.ncols() {
            mean[col] += local[[row, col]];
        }
    }
    let scale = local.nrows() as f64;
    for value in &mut mean {
        *value /= scale;
    }
    Some(mean)
}

fn soap_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn regrowth_pulls_a_loose_atom_back_to_the_bond_length() {
        let coords = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.866, 0.0, 0.5, 0.289, 0.816, 4.0, 0.2, 0.1, 0.2,
            3.5, 0.1,
        ]);
        let energy = |x: ArrayView1<f64>| -> Option<f64> {
            let n = x.len() / 3;
            let mut total = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut d = 0.0;
                    for axis in 0..3 {
                        let delta = x[3 * i + axis] - x[3 * j + axis];
                        d += delta * delta;
                    }
                    let r = d.sqrt();
                    let gap = r - 1.0;
                    total += gap * gap;
                }
            }
            Some(total)
        };
        let before = energy(coords.view()).unwrap();
        let grown = regrow_by_energy(coords.view(), energy).expect("regrow");
        assert_eq!(grown.len(), coords.len());
        let after = energy(grown.view()).unwrap();
        assert!(after < before, "regrowth left the loose pair where it was");
    }

    #[test]
    fn a_cold_family_heats_when_nothing_is_accepted() {
        let mut temperature = MoveTemperature::default();
        let start = temperature.value(1.0);
        for _ in 0..4 {
            assert!(!temperature.accept(1.0, start, 0.99));
        }
        assert!(temperature.value(1.0) > start);
    }

    #[test]
    fn a_fruitless_collective_family_gives_up_its_share() {
        let mut share = CollectiveShare::default();
        assert_eq!(share.stride(), 8);
        assert!(share.due(2));
        assert!(!share.due(3));
        for _ in 0..4 {
            share.observe(false, false);
        }
        assert!(share.stride() >= 64);
    }

    #[test]
    fn a_new_best_restores_the_collective_share() {
        let mut share = CollectiveShare::default();
        share.observe(false, false);
        share.observe(false, false);
        assert!(share.stride() > 8);
        share.observe(true, true);
        assert_eq!(share.stride(), 8);
        assert!(share.due(2));
    }

    #[test]
    fn an_accepted_cut_that_is_not_a_new_best_shrinks_the_share() {
        let mut share = CollectiveShare::default();
        share.observe(true, false);
        assert!(share.stride() > 8);
        assert!(share.stride() < 256);
    }

    #[test]
    fn a_repeated_fingerprint_is_refused_unless_it_is_lower() {
        let species = [29u32, 29, 79, 79];
        let coords = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 2.6, 0.0, 0.0, 1.3, 2.4, 0.1, 1.2, 0.4, 2.7,
        ]);
        let mut taboo = FingerprintTaboo::default();
        taboo.remember(-1.0, coords.view(), &species);
        assert!(taboo.forbids(-0.5, coords.view(), &species));
        assert!(!taboo.forbids(-1.5, coords.view(), &species));
    }

    #[test]
    fn a_plane_move_keeps_distances_inside_the_moved_set() {
        let x = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.2, 1.3, 0.1, 2.4, 0.4, 0.2, 1.1, 2.2, 0.3, 0.4, 0.5,
            1.6,
        ]);
        let n = 6;
        let mut moved = false;
        for seed in 0..16 {
            let mut rng = StdRng::seed_from_u64(seed);
            let y = super::super::moves::plane_rearrangement(x.view(), n, &mut rng);
            assert_eq!(y.len(), x.len());
            if x.iter().zip(y.iter()).any(|(a, b)| (a - b).abs() > 1e-6) {
                moved = true;
                break;
            }
        }
        assert!(moved, "the plane move left every atom where it was");
    }
}
