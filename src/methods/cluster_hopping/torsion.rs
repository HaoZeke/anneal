//! Contact-checked torsion proposals for a flexible molecule.
//!
//! A turn about a rotatable bond keeps every bond length and bond angle, so
//! the only thing it can break is a non-bonded contact. A proposal is checked
//! for one before anything else pays for it, and a contact is repaired in the
//! same torsion space, without the potential.

use super::moves::{RotatableBond, covalent_radius, ring_crankshafts, rotatable_bonds, turn_bond};
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

/// Angular displacement used to read local torsional curvature.
const BARRIER_PROBE_ANGLE: f64 = std::f64::consts::FRAC_PI_6;

/// Rotatable bonds and contact radii of one structure.
#[derive(Debug, Clone)]
pub(crate) struct Topology {
    n: usize,
    /// Rotatable bonds first, then ring crankshafts; the repair turns both.
    pub(crate) torsions: Vec<RotatableBond>,
    /// How many of `torsions` are rotatable bonds.
    bonds: usize,
    /// Ring torsions the crankshafts move, from [`ring_crankshafts`].
    ring_dof: usize,
    /// `near[i * n + j]`: within [`EXCLUDED_BONDS`] bonds of each other.
    near: Vec<bool>,
    radius: Vec<f64>,
    /// `moves[t * n + i]`: atom `i` turns with torsion `t`.
    moves: Vec<bool>,
    /// Four-bonded crankshaft ends that may be stereocentres, each with its
    /// four neighbours. A crankshaft moves one neighbour of its ends and can
    /// carry it through the plane of the other three; a turn about a
    /// rotatable bond cannot.
    hands: Vec<[usize; 5]>,
}

/// Signs of the volumes spanned by the bonds from `h[0]` to each three of
/// its four neighbours `h[1..]`. Whichever neighbour moves, one of them
/// turns over when it crosses the plane of the others.
fn hand(x: ArrayView1<f64>, h: &[usize; 5]) -> [bool; 4] {
    let bond = |k: usize| [0, 1, 2].map(|c| x[3 * h[k] + c] - x[3 * h[0] + c]);
    let volume = |i: usize, j: usize, k: usize| {
        let (u, v, w) = (bond(i), bond(j), bond(k));
        u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
    };
    [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)].map(|(i, j, k)| volume(i, j, k) > 0.0)
}

fn distance(x: ArrayView1<f64>, i: usize, j: usize) -> f64 {
    let dx = x[3 * i] - x[3 * j];
    let dy = x[3 * i + 1] - x[3 * j + 1];
    let dz = x[3 * i + 2] - x[3 * j + 2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

impl Topology {
    /// The [`rotatable_bonds`] of `x` whose turning side lies inside
    /// `mobile`, with `rings` also its [`ring_crankshafts`], and the bond
    /// graph that says which pairs are contacts.
    pub(crate) fn of(
        x: ArrayView1<f64>,
        species: &[u32],
        tolerance: f64,
        mobile: Option<&[usize]>,
        rings: bool,
    ) -> Self {
        let n = species.len().min(x.len() / 3);
        let inside = |bond: &RotatableBond| {
            mobile.is_none_or(|free| bond.side.iter().all(|a| free.contains(a)))
        };
        let mut torsions: Vec<RotatableBond> = rotatable_bonds(x, species, tolerance)
            .into_iter()
            .filter(&inside)
            .collect();
        let bonds = torsions.len();
        let mut ring_dof = 0;
        if rings {
            let (cranks, dof) = ring_crankshafts(x, species, tolerance);
            let before = torsions.len();
            torsions.extend(cranks.into_iter().filter(&inside));
            if torsions.len() > before {
                ring_dof = dof;
            }
        }
        let mut near = vec![false; n * n];
        let mut hands = Vec::new();
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
            // Two lone atoms of one element on a centre swap places under
            // an inversion and leave the molecule as it was, as a CH2 does.
            let mut ends: Vec<usize> = torsions[bonds..]
                .iter()
                .flat_map(|crank| [crank.fixed, crank.pivot])
                .collect();
            ends.sort_unstable();
            ends.dedup();
            for c in ends {
                let [a, b, d, e] = nb[c][..] else { continue };
                let four = [a, b, d, e];
                let lone = |q: usize| nb[q].len() == 1;
                let twins = four.iter().enumerate().any(|(k, &p)| {
                    lone(p)
                        && four[k + 1..]
                            .iter()
                            .any(|&q| lone(q) && species[q] == species[p])
                });
                if !twins {
                    hands.push([c, a, b, d, e]);
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
            bonds,
            ring_dof,
            near,
            radius: species[..n].iter().map(|&z| covalent_radius(z)).collect(),
            moves,
            hands,
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
            // Contact repair is allowed to rearrange free torsions and ring
            // arcs, not to ratchet a partial-double bridge through a sequence
            // of nominally small steps.
            if torsion.constrained {
                continue;
            }
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

    /// Whether `y` holds a crankshaft end in the other hand from `x`.
    fn inverts(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> bool {
        self.hands.iter().any(|h| hand(x, h) != hand(y, h))
    }

    fn propose_by<R, F>(&self, x: ArrayView1<f64>, rng: &mut R, mut angle: F) -> Array1<f64>
    where
        R: Rng + ?Sized,
        F: FnMut(usize, &RotatableBond, &mut R) -> f64,
    {
        let start = x.to_owned();
        if self.torsions.is_empty() {
            return start;
        }
        let cranks = self.torsions.len() - self.bonds;
        let ring_share = if cranks > 0 {
            self.ring_dof.max(1) as f64 / (self.ring_dof.max(1) + self.bonds) as f64
        } else {
            0.0
        };
        let mut best: Option<(f64, Array1<f64>)> = None;
        let mut order: Vec<usize> = (0..self.bonds).collect();
        for _ in 0..REDRAWS {
            let mut y = start.clone();
            if cranks > 0 && (self.bonds == 0 || rng.random::<f64>() < ring_share) {
                let crank = self.bonds + rng.random_range(0..cranks);
                let angle = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
                turn_bond(&mut y, &self.torsions[crank], angle);
            } else {
                let mut count = 1;
                while count < 3.min(self.bonds) && rng.random::<f64>() < 0.5 {
                    count += 1;
                }
                order.shuffle(rng);
                for &t in &order[..count] {
                    let theta = angle(t, &self.torsions[t], rng);
                    turn_bond(&mut y, &self.torsions[t], theta);
                }
            }
            let clean = self.repair(&mut y);
            if self.inverts(start.view(), y.view()) {
                continue;
            }
            if clean {
                return y;
            }
            let p = self.penalty(&self.clashes(y.view()));
            if best.as_ref().is_none_or(|(held, _)| p < *held) {
                best = Some((p, y));
            }
        }
        best.map(|(_, y)| y).unwrap_or(start)
    }

    /// Turns one to three random torsions through uniform angles, then
    /// repairs any contact it made. The least-clashing of a few draws is
    /// returned when none comes out clean. No torsion, no draw.
    ///
    /// With ring crankshafts a draw turns one of them instead, with the
    /// probability that ring torsions are among all torsions the molecule
    /// has, and a draw that inverts a crankshaft end is dropped. Without,
    /// the random stream is the torsion move's alone.
    pub(crate) fn propose<R: Rng + ?Sized>(&self, x: ArrayView1<f64>, rng: &mut R) -> Array1<f64> {
        // A constrained bridge is not a uniform rotor on this path either.
        // The measured proposal is what turns it, and only after the
        // potential has priced the barrier.
        self.propose_by(x, rng, |_, torsion, rng| {
            if torsion.constrained {
                0.0
            } else {
                rng.random_range(-std::f64::consts::PI..std::f64::consts::PI)
            }
        })
    }

    /// Proposes torsions after asking the potential how strongly each locally
    /// planar bridge is held.
    ///
    /// A constrained bridge is probed at small displacements, the midpoint of
    /// a half turn, and the half-turn endpoint. Only a small displacement or a
    /// half turn is offered, with Boltzmann weights whose scale is capped by
    /// the measured local barrier.
    /// The cap matters during the opening pilot, before the acceptance
    /// temperature has learned the potential's units. A half turn is
    /// preferred when the higher of the midpoint and the endpoint lies
    /// below the small-step energies. A low endpoint behind a high
    /// midpoint is not preferred on its own. Probe calls are supplied by
    /// the search so they are charged to the same evaluation ledger.
    pub(crate) fn propose_measured<R, F>(
        &self,
        x: ArrayView1<f64>,
        energy: f64,
        temperature: f64,
        rng: &mut R,
        mut evaluate: F,
    ) -> Array1<f64>
    where
        R: Rng + ?Sized,
        F: FnMut(ArrayView1<f64>) -> Option<f64>,
    {
        let start = x.to_owned();
        let mut measured = vec![false; self.bonds];
        let mut profiles = vec![None; self.bonds];
        self.propose_by(x, rng, |index, torsion, rng| {
            if !torsion.constrained {
                return rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
            }
            if !measured[index] {
                measured[index] = true;
                let probe = |theta: f64, evaluate: &mut F| {
                    let mut y = start.clone();
                    turn_bond(&mut y, torsion, theta);
                    evaluate(y.view())
                };
                profiles[index] = probe(BARRIER_PROBE_ANGLE, &mut evaluate)
                    .zip(probe(-BARRIER_PROBE_ANGLE, &mut evaluate))
                    .zip(probe(std::f64::consts::FRAC_PI_2, &mut evaluate))
                    .zip(probe(std::f64::consts::PI, &mut evaluate))
                    .map(|(((p, m), barrier), f)| (p, m, barrier, f));
            }
            let Some((plus, minus, barrier, flip)) = profiles[index] else {
                return rng.random_range(-BARRIER_PROBE_ANGLE..BARRIER_PROBE_ANGLE);
            };
            let local_rise = (0.5 * (plus + minus) - energy).max(0.0);
            let thermal = temperature
                .min(local_rise.max(f64::EPSILON))
                .max(f64::EPSILON);
            // The endpoint cannot measure a rotational barrier: a true
            // double bond can have two low E/Z endpoints separated by a very
            // high transition state. Price the half turn by the highest
            // sampled energy along its path, while retaining the endpoint so
            // a downhill flip is not mistaken for a barrierless one.
            let flip_barrier = barrier.max(flip) - energy;
            let changes = [plus - energy, minus - energy, flip_barrier];
            let floor = changes.iter().copied().fold(f64::INFINITY, f64::min);
            let weights = changes.map(|de| (-(de - floor) / thermal).max(-700.0).exp());
            let draw = rng.random::<f64>() * weights.iter().sum::<f64>();
            let mut cumulative = 0.0;
            let action = weights
                .iter()
                .position(|w| {
                    cumulative += *w;
                    draw <= cumulative
                })
                .unwrap_or(2);
            match action {
                0 => rng.random_range(0.0..BARRIER_PROBE_ANGLE),
                1 => rng.random_range(-BARRIER_PROBE_ANGLE..0.0),
                _ => std::f64::consts::PI,
            }
        })
    }
}

/// Four-bonded atoms of declared groups that may be stereocentres, each
/// with its four neighbours, and the hand each holds in one geometry.
///
/// Bonds are read once, inside each group of `x`, at `tolerance` times the
/// summed covalent radii. A geometry a quench is still distorting puts a
/// hydrogen within that reach of the next heavy atom as well, so the
/// centres are not reread from it. A centre with two lone atoms of one
/// element is no centre, as in [`Topology::of`].
#[derive(Debug, Clone, Default)]
pub(crate) struct HeldHands {
    centres: Vec<[usize; 5]>,
    signs: Vec<[bool; 4]>,
}

impl HeldHands {
    pub(crate) fn of(
        x: ArrayView1<f64>,
        species: &[u32],
        groups: &[Vec<usize>],
        tolerance: f64,
    ) -> Self {
        let n = species.len().min(x.len() / 3);
        let mut centres = Vec::new();
        for atoms in groups {
            let atoms: Vec<usize> = atoms.iter().copied().filter(|&a| a < n).collect();
            let nb: Vec<Vec<usize>> = atoms
                .iter()
                .map(|&a| {
                    let ra = covalent_radius(species[a]);
                    atoms
                        .iter()
                        .copied()
                        .filter(|&b| {
                            let cut = tolerance * (ra + covalent_radius(species[b]));
                            b != a && cut > 0.0 && distance(x, a, b) < cut
                        })
                        .collect()
                })
                .collect();
            let lone = |q: usize| {
                atoms
                    .iter()
                    .position(|&a| a == q)
                    .is_some_and(|i| nb[i].len() == 1)
            };
            for (i, &c) in atoms.iter().enumerate() {
                let [a, b, d, e] = nb[i][..] else { continue };
                let four = [a, b, d, e];
                let twins = four.iter().enumerate().any(|(k, &p)| {
                    lone(p)
                        && four[k + 1..]
                            .iter()
                            .any(|&q| lone(q) && species[q] == species[p])
                });
                if !twins {
                    centres.push([c, a, b, d, e]);
                }
            }
        }
        let signs = centres.iter().map(|h| hand(x, h)).collect();
        Self { centres, signs }
    }

    /// The same centres, holding the hands of `x`.
    pub(crate) fn at(&self, x: ArrayView1<f64>) -> Self {
        Self {
            centres: self.centres.clone(),
            signs: self.centres.iter().map(|h| hand(x, h)).collect(),
        }
    }

    /// Whether `y` holds a centre in the other hand.
    ///
    /// A centre pushed through the plane of its neighbours turns every
    /// volume over. A distorted one can turn a single volume whose three
    /// bonds are nearly coplanar, so three of the four must turn.
    pub(crate) fn turned_over(&self, y: ArrayView1<f64>) -> bool {
        self.centres.iter().zip(&self.signs).any(|(h, held)| {
            let now = hand(y, h);
            (0..4).filter(|&k| now[k] != held[k]).count() >= 3
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Two bent carbon arms on a central single bond; turning it half way
    /// brings their tips together. The central carbons carry their valence
    /// in hydrogens, so that bond is saturated.
    fn hook() -> (Array1<f64>, Vec<u32>) {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, // C1
            1.5, 0.0, 0.0, // C2
            0.0, 1.5, 0.0, // arm on C1
            1.0, 2.6, 0.0, //
            1.5, -1.5, 0.0, // arm on C2
            1.0, -2.6, 0.0, //
            -0.40217, -0.67028, 0.75966, // H on C1
            -0.40217, -0.67028, -0.75966, //
            1.84434, 0.44272, 0.93463, // H on C2
            1.84434, 0.44272, -0.93463, //
        ]);
        (x, vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1])
    }

    fn central(top: &Topology) -> usize {
        top.torsions
            .iter()
            .position(|t| (t.fixed.min(t.pivot), t.fixed.max(t.pivot)) == (0, 1))
            .expect("the central bond turns")
    }

    /// Chair cyclohexane with a fluorine and a hydrogen on carbon 0 and two
    /// hydrogens on every other carbon: carbon 0 is the one stereocentre.
    fn fluorocyclohexane() -> (Array1<f64>, Vec<u32>) {
        let (h, r) = (0.25_f64, (1.53_f64.powi(2) - 0.25).sqrt());
        let ring: Vec<[f64; 3]> = (0..6)
            .map(|i| {
                let a = i as f64 * std::f64::consts::FRAC_PI_3;
                [r * a.cos(), r * a.sin(), if i % 2 == 0 { h } else { -h }]
            })
            .collect();
        let unit = |v: [f64; 3]| {
            let n = v.iter().map(|c| c * c).sum::<f64>().sqrt();
            v.map(|c| c / n)
        };
        let mut x: Vec<f64> = ring.iter().flatten().copied().collect();
        let mut species = vec![6u32; 6];
        for i in 0..6 {
            let (c, p, q) = (ring[i], ring[(i + 1) % 6], ring[(i + 5) % 6]);
            let a = unit([0, 1, 2].map(|k| p[k] - c[k]));
            let b = unit([0, 1, 2].map(|k| q[k] - c[k]));
            let bis = unit([0, 1, 2].map(|k| -(a[k] + b[k])));
            let perp = unit([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]);
            let first = if i == 0 { (9, 1.35) } else { (1, 1.09) };
            for ((z, length), side) in [(first, 1.0), ((1, 1.09), -1.0)] {
                x.extend(
                    [0, 1, 2].map(|k| c[k] + length * (0.577 * bis[k] + side * 0.816 * perp[k])),
                );
                species.push(z);
            }
        }
        (Array1::from_vec(x), species)
    }

    #[test]
    fn held_hands_see_a_centre_turn_over_and_nothing_else() {
        let (x, z) = fluorocyclohexane();
        let groups = vec![(0..z.len()).collect::<Vec<usize>>()];
        let held = HeldHands::of(x.view(), &z, &groups, 1.25);
        let centres: Vec<usize> = held.centres.iter().map(|h| h[0]).collect();
        assert_eq!(centres, vec![0], "a CH2 is not a centre");
        let mut mirrored = x.clone();
        for k in 0..3 {
            mirrored.swap(3 * 6 + k, 3 * 7 + k);
        }
        assert!(held.turned_over(mirrored.view()));
        assert!(!held.at(mirrored.view()).turned_over(mirrored.view()));
        let h = held.centres[0];
        for &q in &h[1..] {
            let mut through = x.clone();
            for k in 0..3 {
                through[3 * q + k] = 2.0 * x[3 * h[0] + k] - x[3 * q + k];
            }
            assert!(
                held.turned_over(through.view()),
                "neighbour {q} crossed unseen"
            );
        }
        let turned = Array1::from_iter(
            x.exact_chunks(3)
                .into_iter()
                .flat_map(|p| [-p[1], p[0], p[2]]),
        );
        assert!(!held.turned_over(turned.view()));
        let mut nudged = x.clone();
        nudged[3 * h[1]] += 0.2;
        assert!(!held.turned_over(nudged.view()));
        assert!(HeldHands::of(x.view(), &z, &[], 1.25).centres.is_empty());
    }

    #[test]
    fn a_crankshaft_end_that_is_a_stereocentre_keeps_its_hand() {
        let (x, z) = fluorocyclohexane();
        let top = Topology::of(x.view(), &z, 1.25, None, true);
        let centres: Vec<usize> = top.hands.iter().map(|h| h[0]).collect();
        assert_eq!(centres, vec![0], "only the CHF carbon has a hand");
        let mut mirrored = x.clone();
        for k in 0..3 {
            mirrored.swap(3 * 6 + k, 3 * 7 + k);
        }
        assert!(top.inverts(x.view(), mirrored.view()));
        let h = top.hands[0];
        for &q in &h[1..] {
            let mut through = x.clone();
            for k in 0..3 {
                through[3 * q + k] = 2.0 * x[3 * h[0] + k] - x[3 * q + k];
            }
            assert!(
                top.inverts(x.view(), through.view()),
                "neighbour {q} crossed unseen"
            );
        }
        let turned = Array1::from_iter(
            x.exact_chunks(3)
                .into_iter()
                .flat_map(|p| [-p[1], p[0], p[2]]),
        );
        assert!(!top.inverts(x.view(), turned.view()));
        assert!(
            Topology::of(x.view(), &z, 1.25, None, false)
                .hands
                .is_empty()
        );
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..100 {
            assert!(!top.inverts(x.view(), top.propose(x.view(), &mut rng).view()));
        }
    }

    #[test]
    fn the_penalty_gradient_matches_a_finite_difference() {
        let (x, z) = hook();
        let top = Topology::of(x.view(), &z, 1.2, None, false);
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
        let top = Topology::of(x.view(), &z, 1.2, None, false);
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
        let top = Topology::of(x.view(), &z, 1.2, None, false);
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
    fn a_stiff_planar_torsion_uses_small_or_downhill_flip_moves() {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, // C1
            1.5, 0.0, 0.0, // C2
            0.0, 1.5, 0.0, // arm on C1
            1.5, -1.5, 0.0, // arm on C2
        ]);
        let mut top = Topology::of(x.view(), &[6; 4], 1.2, None, false);
        assert_eq!(top.bonds, 1);
        top.torsions[0].constrained = true;
        let cosine = |y: &Array1<f64>| -y[3 * 3 + 1] / 1.5;

        let mut calls = 0;
        let local = top.propose_measured(x.view(), 0.0, 0.1, &mut StdRng::seed_from_u64(4), |_| {
            calls += 1;
            Some(if calls < 3 { 2.0 } else { 10.0 })
        });
        assert_eq!(calls, 4, "one set of charged barrier probes");
        assert!(cosine(&local) > BARRIER_PROBE_ANGLE.cos() - 1e-12);

        let mut calls = 0;
        let held = top.propose_measured(x.view(), 0.0, 0.1, &mut StdRng::seed_from_u64(4), |_| {
            calls += 1;
            Some(match calls {
                1 | 2 => 2.0,
                3 => 10.0,
                _ => -1.0,
            })
        });
        assert_eq!(calls, 4);
        assert!(
            cosine(&held) > BARRIER_PROBE_ANGLE.cos() - 1e-12,
            "a low endpoint bypassed its rotational barrier"
        );

        let mut calls = 0;
        let flipped =
            top.propose_measured(x.view(), 0.0, 0.1, &mut StdRng::seed_from_u64(4), |_| {
                calls += 1;
                Some(if calls < 3 { 2.0 } else { -1.0 })
            });
        assert_eq!(calls, 4);
        assert!(
            cosine(&flipped) < -0.99,
            "downhill half turn was not preferred"
        );
    }

    fn flip_count(samples: &[(f64, f64, f64, f64)], temperature: f64, draws: u64) -> usize {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 1.5, -1.5, 0.0,
        ]);
        let mut top = Topology::of(x.view(), &[6; 4], 1.2, None, false);
        top.torsions[0].constrained = true;
        let cosine = |y: &Array1<f64>| -y[3 * 3 + 1] / 1.5;
        let mut flips = 0;
        for seed in 0..draws {
            let mut calls = 0usize;
            let y = top.propose_measured(
                x.view(),
                0.0,
                temperature,
                &mut StdRng::seed_from_u64(seed),
                |_| {
                    let sample = samples[calls.min(samples.len() - 1)];
                    calls += 1;
                    Some(match calls {
                        1 => sample.0,
                        2 => sample.1,
                        3 => sample.2,
                        _ => sample.3,
                    })
                },
            );
            if cosine(&y) < -0.99 {
                flips += 1;
            }
        }
        flips
    }

    #[test]
    fn a_hot_pilot_does_not_cross_a_barrier_above_the_local_rise() {
        let flips = flip_count(&[(2.0, 2.0, 100.0, 100.0)], 1_000.0, 40);
        assert_eq!(
            flips, 0,
            "the thermal cap let a hot pilot cross the barrier"
        );
    }

    #[test]
    fn a_low_endpoint_does_not_hide_the_barrier_above_it() {
        let flips = flip_count(&[(2.0, 2.0, 100.0, -5.0)], 1_000.0, 40);
        assert_eq!(flips, 0, "the half turn was priced by its endpoint alone");
    }

    #[test]
    fn a_downhill_half_turn_is_preferred_when_the_cap_binds() {
        let flips = flip_count(&[(2.0, 2.0, -5.0, -5.0)], 1_000.0, 40);
        assert!(
            flips > 30,
            "a downhill half turn was not preferred under the cap ({flips})"
        );
    }

    #[test]
    fn repair_does_not_ratchet_a_constrained_bridge() {
        // The tip on atom 5 sits four bonds from the left substituent, so a
        // half turn of the bridge is a real contact. Both bridges are planar,
        // and repair has no free torsion it can spend instead.
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 1.30, 0.0, 0.0, -0.60, 1.10, 0.0, -0.60, -1.10, 0.0, 1.90, 1.10, 0.0,
            1.20, -1.80, 0.0, -0.25, -2.75, 0.0,
        ]);
        let top = Topology::of(x.view(), &[6, 6, 7, 8, 7, 6, 8], 1.25, None, false);
        let t = top
            .torsions
            .iter()
            .position(|bond| (bond.fixed.min(bond.pivot), bond.fixed.max(bond.pivot)) == (0, 1))
            .expect("the bridge turns");
        assert!(top.torsions[t].constrained);
        let mut y = x.clone();
        // Short of a half turn, so the contact still has a slope. At exactly
        // pi the two atoms are at their closest and the torsion gradient
        // vanishes, which would hide a repair that is allowed to turn the bond.
        turn_bond(&mut y, &top.torsions[t], 2.6);
        assert!(
            !top.clashes(y.view()).is_empty(),
            "the half turn made no contact to repair"
        );
        let before = y.clone();
        assert!(!top.repair(&mut y));
        assert_eq!(before, y);
    }

    #[test]
    fn no_rotatable_bond_draws_nothing() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0]);
        let top = Topology::of(x.view(), &[8, 1, 1], 1.25, None, false);
        let mut rng = StdRng::seed_from_u64(1);
        let mut untouched = StdRng::seed_from_u64(1);
        assert_eq!(top.propose(x.view(), &mut rng), x);
        assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    }
}
