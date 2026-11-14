//! Contact-checked torsion proposals for a flexible molecule.
//!
//! A turn about a rotatable bond keeps every bond length and bond angle, so
//! the only thing it can break is a non-bonded contact. A proposal is checked
//! for one before anything else pays for it, and a contact is repaired in the
//! same torsion space, without the potential.

use super::moves::{RotatableBond, covalent_radius, rotatable_bonds, turn_bond};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::seq::SliceRandom;

/// Added to the covalent radii sum to give the closest non-bonded approach a
/// proposal may keep. 0.8 Å puts H···H at 1.42 Å, C···C at 2.32 Å and
/// O···H at 1.77 Å, inside a hydrogen bond's reach and well outside the
/// 0.90 Å overlap the hop refuses to hand the potential.
pub(crate) const CLASH_GAP: f64 = 0.8;

/// Atoms closer than this many bonds are held by the bonded geometry and do
/// not count as a contact: 1-2, 1-3 and 1-4 pairs.
const EXCLUDED_BONDS: usize = 3;

/// Torsion-space descent steps a clashing proposal gets before it is redrawn.
const REPAIR_STEPS: usize = 80;

/// Fresh draws of the angles before the least-clashing proposal is returned.
const REDRAWS: usize = 4;

/// Largest single repair rotation, in radians.
const REPAIR_STEP_CAP: f64 = 0.35;

/// Rotatable bonds and contact radii of one structure.
#[derive(Debug, Clone)]
pub(crate) struct Topology {
    n: usize,
    pub(crate) torsions: Vec<RotatableBond>,
    /// `near[i * n + j]`: within [`EXCLUDED_BONDS`] bonds of each other.
    near: Vec<bool>,
    radius: Vec<f64>,
    /// `moves[t * n + i]`: atom `i` turns with torsion `t`.
    moves: Vec<bool>,
}

fn distance(x: ArrayView1<f64>, i: usize, j: usize) -> f64 {
    let dx = x[3 * i] - x[3 * j];
    let dy = x[3 * i + 1] - x[3 * j + 1];
    let dz = x[3 * i + 2] - x[3 * j + 2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

impl Topology {
    /// The [`rotatable_bonds`] of `x` whose turning side lies inside
    /// `mobile`, and the bond graph that says which pairs are contacts.
    pub(crate) fn of(
        x: ArrayView1<f64>,
        species: &[u32],
        tolerance: f64,
        mobile: Option<&[usize]>,
    ) -> Self {
        let n = species.len().min(x.len() / 3);
        let torsions: Vec<RotatableBond> = rotatable_bonds(x, species, tolerance)
            .into_iter()
            .filter(|bond| mobile.is_none_or(|free| bond.side.iter().all(|a| free.contains(a))))
            .collect();
        let mut near = vec![false; n * n];
        if !torsions.is_empty() {
            let mut nb = vec![Vec::new(); n];
            for i in 0..n {
                let ri = covalent_radius(species[i]);
                for j in i + 1..n {
                    let cut = tolerance * (ri + covalent_radius(species[j]));
                    if cut > 0.0 && distance(x, i, j) < cut {
                        nb[i].push(j);
                        nb[j].push(i);
                    }
                }
            }
            for i in 0..n {
                let mut depth = vec![usize::MAX; n];
                depth[i] = 0;
                let mut frontier = vec![i];
                for d in 1..=EXCLUDED_BONDS {
                    let mut next = Vec::new();
                    for &p in &frontier {
                        for &q in &nb[p] {
                            if depth[q] == usize::MAX {
                                depth[q] = d;
                                next.push(q);
                            }
                        }
                    }
                    frontier = next;
                }
                for j in 0..n {
                    near[i * n + j] = depth[j] != usize::MAX;
                }
            }
        }
        let mut moves = vec![false; torsions.len() * n];
        for (t, torsion) in torsions.iter().enumerate() {
            for &i in &torsion.side {
                moves[t * n + i] = true;
            }
        }
        Self {
            n,
            torsions,
            near,
            radius: species[..n].iter().map(|&z| covalent_radius(z)).collect(),
            moves,
        }
    }

    fn contact(&self, i: usize, j: usize) -> f64 {
        self.radius[i] + self.radius[j] + CLASH_GAP
    }

    /// Non-bonded pairs closer than their contact distance, with the
    /// distance.
    pub(crate) fn clashes(&self, x: ArrayView1<f64>) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        for i in 0..self.n {
            for j in i + 1..self.n {
                if self.near[i * self.n + j] {
                    continue;
                }
                let d = distance(x, i, j);
                if d < self.contact(i, j) {
                    out.push((i, j, d));
                }
            }
        }
        out
    }

    fn penalty(&self, clashes: &[(usize, usize, f64)]) -> f64 {
        clashes
            .iter()
            .map(|&(i, j, d)| (self.contact(i, j) - d).powi(2))
            .sum()
    }

    /// Gradient of the clash penalty with respect to every torsion angle.
    fn penalty_gradient(&self, x: ArrayView1<f64>, clashes: &[(usize, usize, f64)]) -> Vec<f64> {
        let mut g = vec![0.0; self.torsions.len()];
        for (t, torsion) in self.torsions.iter().enumerate() {
            let (a, b) = (torsion.fixed, torsion.pivot);
            let mut u = [
                x[3 * b] - x[3 * a],
                x[3 * b + 1] - x[3 * a + 1],
                x[3 * b + 2] - x[3 * a + 2],
            ];
            let norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            if norm < 1e-12 {
                continue;
            }
            u.iter_mut().for_each(|c| *c /= norm);
            let row = &self.moves[t * self.n..(t + 1) * self.n];
            for &(i, j, d) in clashes {
                // Only a pair split by the bond changes its distance.
                let (m, f) = match (row[i], row[j]) {
                    (true, false) => (i, j),
                    (false, true) => (j, i),
                    _ => continue,
                };
                let v = [
                    x[3 * m] - x[3 * a],
                    x[3 * m + 1] - x[3 * a + 1],
                    x[3 * m + 2] - x[3 * a + 2],
                ];
                let velocity = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let r = [
                    x[3 * m] - x[3 * f],
                    x[3 * m + 1] - x[3 * f + 1],
                    x[3 * m + 2] - x[3 * f + 2],
                ];
                let dd =
                    (r[0] * velocity[0] + r[1] * velocity[1] + r[2] * velocity[2]) / d.max(1e-9);
                g[t] += -2.0 * (self.contact(i, j) - d) * dd;
            }
        }
        g
    }

    /// Descends the clash penalty in torsion space. True when no clash is
    /// left. Bond lengths and angles do not move.
    pub(crate) fn repair(&self, x: &mut Array1<f64>) -> bool {
        let mut clashes = self.clashes(x.view());
        let mut p = self.penalty(&clashes);
        let mut cap = REPAIR_STEP_CAP;
        for _ in 0..REPAIR_STEPS {
            if clashes.is_empty() {
                return true;
            }
            let g = self.penalty_gradient(x.view(), &clashes);
            let largest = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if largest < 1e-12 {
                return false;
            }
            let mut trial = x.clone();
            for (t, gt) in g.iter().enumerate() {
                if *gt != 0.0 {
                    turn_bond(&mut trial, &self.torsions[t], -cap * gt / largest);
                }
            }
            let next = self.clashes(trial.view());
            let q = self.penalty(&next);
            if q < p {
                *x = trial;
                clashes = next;
                p = q;
                cap = (cap * 1.2).min(REPAIR_STEP_CAP);
            } else {
                cap *= 0.5;
                if cap < 1e-3 {
                    return false;
                }
            }
        }
        clashes.is_empty()
    }

    /// Turns one to three random torsions through uniform angles, then
    /// repairs any contact it made. The least-clashing of a few draws is
    /// returned when none comes out clean. No torsion, no draw.
    pub(crate) fn propose<R: Rng + ?Sized>(&self, x: ArrayView1<f64>, rng: &mut R) -> Array1<f64> {
        let start = x.to_owned();
        if self.torsions.is_empty() {
            return start;
        }
        let mut best: Option<(f64, Array1<f64>)> = None;
        let mut order: Vec<usize> = (0..self.torsions.len()).collect();
        for _ in 0..REDRAWS {
            let mut count = 1;
            while count < 3.min(self.torsions.len()) && rng.random::<f64>() < 0.5 {
                count += 1;
            }
            order.shuffle(rng);
            let mut y = start.clone();
            for &t in &order[..count] {
                let angle = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
                turn_bond(&mut y, &self.torsions[t], angle);
            }
            if self.repair(&mut y) {
                return y;
            }
            let p = self.penalty(&self.clashes(y.view()));
            if best.as_ref().is_none_or(|(held, _)| p < *held) {
                best = Some((p, y));
            }
        }
        best.map(|(_, y)| y).unwrap_or(start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Two bent carbon arms on a central bond; turning it half way brings
    /// their tips together.
    fn hook() -> (Array1<f64>, Vec<u32>) {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, // C1
            1.5, 0.0, 0.0, // C2
            0.0, 1.5, 0.0, // arm on C1
            1.0, 2.6, 0.0, //
            1.5, -1.5, 0.0, // arm on C2
            1.0, -2.6, 0.0, //
        ]);
        (x, vec![6; 6])
    }

    fn central(top: &Topology) -> usize {
        top.torsions
            .iter()
            .position(|t| (t.fixed.min(t.pivot), t.fixed.max(t.pivot)) == (0, 1))
            .expect("the central bond turns")
    }

    #[test]
    fn the_penalty_gradient_matches_a_finite_difference() {
        let (x, z) = hook();
        let top = Topology::of(x.view(), &z, 1.2, None);
        assert!(top.clashes(x.view()).is_empty());
        let t = central(&top);
        let mut y = x.clone();
        turn_bond(&mut y, &top.torsions[t], 2.9);
        let clashes = top.clashes(y.view());
        assert!(!clashes.is_empty());
        let g = top.penalty_gradient(y.view(), &clashes);
        let h = 1e-6;
        let mut plus = y.clone();
        turn_bond(&mut plus, &top.torsions[t], h);
        let mut minus = y.clone();
        turn_bond(&mut minus, &top.torsions[t], -h);
        // The clash set is held fixed across the difference.
        let p = |v: &Array1<f64>| -> f64 {
            clashes
                .iter()
                .map(|&(i, j, _)| (top.contact(i, j) - distance(v.view(), i, j)).powi(2))
                .sum()
        };
        let fd = (p(&plus) - p(&minus)) / (2.0 * h);
        assert!(
            (fd - g[t]).abs() < 1e-5 * fd.abs().max(1.0),
            "{fd} {}",
            g[t]
        );
    }

    #[test]
    fn an_eclipsed_contact_is_repaired_in_torsion_space() {
        let (x, z) = hook();
        let top = Topology::of(x.view(), &z, 1.2, None);
        let mut y = x.clone();
        turn_bond(&mut y, &top.torsions[central(&top)], 2.9);
        assert!(!top.clashes(y.view()).is_empty());
        assert!(top.repair(&mut y));
        assert!(top.clashes(y.view()).is_empty());
        assert!((distance(x.view(), 0, 1) - distance(y.view(), 0, 1)).abs() < 1e-10);
    }

    #[test]
    fn a_proposal_moves_the_side_and_leaves_no_contact() {
        let (x, z) = hook();
        let top = Topology::of(x.view(), &z, 1.2, None);
        let mut rng = StdRng::seed_from_u64(7);
        let mut moved = 0;
        for _ in 0..40 {
            let y = top.propose(x.view(), &mut rng);
            assert!(top.clashes(y.view()).is_empty());
            moved += usize::from((distance(x.view(), 3, 5) - distance(y.view(), 3, 5)).abs() > 0.1);
            assert!((distance(x.view(), 0, 1) - distance(y.view(), 0, 1)).abs() < 1e-10);
        }
        assert!(moved > 30);
    }

    #[test]
    fn no_rotatable_bond_draws_nothing() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0]);
        let top = Topology::of(x.view(), &[8, 1, 1], 1.25, None);
        let mut rng = StdRng::seed_from_u64(1);
        let mut untouched = StdRng::seed_from_u64(1);
        assert_eq!(top.propose(x.view(), &mut rng), x);
        assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    }
}
