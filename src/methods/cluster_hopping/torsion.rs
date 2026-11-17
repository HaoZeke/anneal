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
use std::collections::BTreeMap;

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

/// A torsion whose side stands this far from where it was counts as turned:
/// half the spacing of three rotamers.
pub(crate) const TURNED: f64 = std::f64::consts::FRAC_PI_3;

fn wrap(angle: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    angle - tau * (angle / tau).round()
}

/// Dihedral of `a`, `b`, `c`, `d`, in radians on `(-pi, pi]`.
fn dihedral(x: ArrayView1<f64>, a: usize, b: usize, c: usize, d: usize) -> f64 {
    let v = |i: usize, j: usize| [0, 1, 2].map(|k| x[3 * j + k] - x[3 * i + k]);
    let cross = |u: [f64; 3], w: [f64; 3]| {
        [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ]
    };
    let dot = |u: [f64; 3], w: [f64; 3]| u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    let (b0, b1, b2) = (v(a, b), v(b, c), v(c, d));
    let n1 = cross(b0, b1);
    let n2 = cross(b1, b2);
    let axis = dot(b1, b1).sqrt().max(1e-12);
    let m1 = cross(n1, b1.map(|c| c / axis));
    dot(m1, n2).atan2(dot(n1, n2))
}

/// Hops of memory in the walk's yield table.
const MEMORY: f64 = 1000.0;

/// Proposals' worth of the class's own yield each torsion's estimate starts
/// from.
const PRIOR_TRIES: f64 = 3.0;

/// Share of proposals that ignore the table and draw the uniform way, so no
/// torsion is starved by an estimate that has stopped being true.
const EXPLORE: f64 = 0.2;

/// Power of a torsion's relative yield in its draw weight.
const SHARPNESS: f64 = 0.5;

/// A torsion as the walk remembers it: the atoms at the ends of its axis,
/// and whether it is a ring crank. The list of torsions a geometry offers
/// can change with the geometry, so an index would not name one torsion.
pub(crate) type TorsionKey = (usize, usize, bool);

#[derive(Debug, Clone, Copy, Default)]
struct Yield {
    hits: f64,
    calls: f64,
    tries: f64,
}

/// What each torsion has yielded per force call, as the walk has seen it.
///
/// A proposal that turns a torsion and is accepted with that torsion more
/// than [`TURNED`] from the walker's earns it a hit, and every proposal
/// pays its calls, split over the torsions it turned. A hindered rotor, or
/// a ring arc that quenches back where it was, pays and never earns. The
/// draw weights each torsion's share of the uniform draw by its yield
/// relative to the walk's, so the calls go where the conformer changes.
/// Counts fade over [`MEMORY`] hops, and a torsion's yield starts from its
/// class's, rotatable bond or ring crank. Until the first accepted change
/// there is nothing to weigh and the draw and the random stream are the
/// uniform one's.
#[derive(Debug, Clone, Default)]
pub(crate) struct TorsionWalk {
    seen: BTreeMap<TorsionKey, Yield>,
}

impl TorsionWalk {
    /// Adds one hop that turned `turned` and spent `calls`. `kept` are the
    /// turned torsions an accepted hop left more than [`TURNED`] from the
    /// walker's.
    pub(crate) fn observe(&mut self, turned: &[TorsionKey], kept: &[TorsionKey], calls: f64) {
        let fade = 1.0 - 1.0 / MEMORY;
        for v in self.seen.values_mut() {
            v.hits *= fade;
            v.calls *= fade;
            v.tries *= fade;
        }
        let share = 1.0 / turned.len().max(1) as f64;
        for key in turned {
            let entry = self.seen.entry(*key).or_default();
            entry.tries += share;
            entry.calls += calls * share;
            if kept.contains(key) {
                entry.hits += share;
            }
        }
    }

    /// Chance of drawing each of `keys` first: the uniform draw's, moved
    /// toward the torsions that have yielded. `ring_share` is the uniform
    /// draw's chance of a ring crank. `None` until a torsion has yielded.
    fn probabilities(&self, keys: &[TorsionKey], ring_share: f64) -> Option<Vec<f64>> {
        let n = keys.len();
        let sum = |pick: &dyn Fn(&TorsionKey) -> bool| {
            self.seen
                .iter()
                .filter(|(k, _)| pick(k))
                .fold(Yield::default(), |a, (_, v)| Yield {
                    hits: a.hits + v.hits,
                    calls: a.calls + v.calls,
                    tries: a.tries + v.tries,
                })
        };
        let all = sum(&|_| true);
        if n == 0 || !(all.hits > 0.0) || !(all.calls > 0.0) || !(all.tries > 0.0) {
            return None;
        }
        let per_call = all.calls / all.tries;
        let overall = all.hits / all.calls;
        let prior = PRIOR_TRIES * per_call;
        let class_yield = |crank: bool| {
            let c = sum(&|k| k.2 == crank);
            (c.hits + prior * overall) / (c.calls + prior)
        };
        let class_yield = [class_yield(false), class_yield(true)];
        let bonds = keys.iter().filter(|k| !k.2).count();
        let cranks = n - bonds;
        let uniform = |key: &TorsionKey| {
            if key.2 {
                ring_share / cranks as f64
            } else {
                (1.0 - ring_share) / bonds as f64
            }
        };
        let weights: Vec<f64> = keys
            .iter()
            .map(|key| {
                let own = self.seen.get(key).copied().unwrap_or_default();
                let y = (own.hits + prior * class_yield[usize::from(key.2)]) / (own.calls + prior);
                uniform(key) * (y / overall).powf(SHARPNESS)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) {
            return None;
        }
        Some(
            keys.iter()
                .zip(&weights)
                .map(|(key, w)| (1.0 - EXPLORE) * w / total + EXPLORE * uniform(key))
                .collect(),
        )
    }
}

/// Index drawn from `p` among `allowed`, with the weights renormalised.
fn pick<R: Rng + ?Sized>(p: &[f64], allowed: &[usize], rng: &mut R) -> usize {
    let total: f64 = allowed.iter().map(|&t| p[t]).sum();
    let mut at = rng.random::<f64>() * total;
    for &t in allowed {
        if at < p[t] {
            return t;
        }
        at -= p[t];
    }
    allowed[allowed.len() - 1]
}

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
    /// One atom off the turning side next to `fixed` and one on it next to
    /// `pivot`, per torsion. Their dihedral about the axis reads how far the
    /// side has turned between two geometries of one bond graph.
    frames: Vec<Option<[usize; 2]>>,
}

/// One torsion turned by a proposal, and the angle it was turned through.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Turn {
    pub(crate) torsion: usize,
    pub(crate) angle: f64,
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
        let mut frames = vec![None; torsions.len()];
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
            for (t, torsion) in torsions.iter().enumerate() {
                let off = nb[torsion.fixed]
                    .iter()
                    .copied()
                    .filter(|a| *a != torsion.pivot && !torsion.side.contains(a))
                    .min();
                let on = nb[torsion.pivot]
                    .iter()
                    .copied()
                    .filter(|a| *a != torsion.pivot && torsion.side.contains(a))
                    .min();
                frames[t] = off.zip(on).map(|(a, b)| [a, b]);
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
            frames,
        }
    }

    /// Each torsion as the walk names it.
    pub(crate) fn keys(&self) -> Vec<TorsionKey> {
        self.torsions
            .iter()
            .enumerate()
            .map(|(t, torsion)| (torsion.fixed, torsion.pivot, t >= self.bonds))
            .collect()
    }

    /// How far each torsion's side stands turned about its axis, in radians,
    /// as the dihedral of its frame. A torsion without a frame reads zero.
    pub(crate) fn angles(&self, x: ArrayView1<f64>) -> Vec<f64> {
        self.torsions
            .iter()
            .zip(&self.frames)
            .map(|(torsion, frame)| {
                let Some([a, b]) = *frame else { return 0.0 };
                dihedral(x, a, torsion.fixed, torsion.pivot, b)
            })
            .collect()
    }

    /// Torsions whose side stands more than [`TURNED`] from where `from`
    /// has it, in `to`.
    pub(crate) fn turned_between(&self, from: ArrayView1<f64>, to: ArrayView1<f64>) -> Vec<usize> {
        self.angles(from)
            .into_iter()
            .zip(self.angles(to))
            .enumerate()
            .filter(|(_, (before, after))| wrap(after - before).abs() > TURNED)
            .map(|(t, _)| t)
            .collect()
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

    /// Whether `y` holds a crankshaft end in the other hand from `x`.
    fn inverts(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> bool {
        self.hands.iter().any(|h| hand(x, h) != hand(y, h))
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
        self.propose_with(x, None, rng).0
    }

    /// [`Self::propose`], also saying which torsions it turned and through
    /// what angles. With a `walk` the further torsions of a multi-bond turn
    /// are drawn from it; without one, or while it has nothing to say, the
    /// draw and the random stream are those of [`Self::propose`].
    pub(crate) fn propose_with<R: Rng + ?Sized>(
        &self,
        x: ArrayView1<f64>,
        walk: Option<&TorsionWalk>,
        rng: &mut R,
    ) -> (Array1<f64>, Vec<Turn>) {
        let start = x.to_owned();
        if self.torsions.is_empty() {
            return (start, Vec::new());
        }
        let cranks = self.torsions.len() - self.bonds;
        let ring_share = if cranks > 0 {
            self.ring_dof.max(1) as f64 / (self.ring_dof.max(1) + self.bonds) as f64
        } else {
            0.0
        };
        let mut best: Option<(f64, Array1<f64>, Vec<Turn>)> = None;
        let mut order: Vec<usize> = (0..self.bonds).collect();
        let guide = walk.and_then(|walk| walk.probabilities(&self.keys(), ring_share));
        for _ in 0..REDRAWS {
            let mut y = start.clone();
            let mut turns = Vec::new();
            let first = guide.as_ref().map(|p| {
                let all: Vec<usize> = (0..self.torsions.len()).collect();
                pick(p, &all, rng)
            });
            let ring = match first {
                Some(t) => t >= self.bonds,
                None => cranks > 0 && (self.bonds == 0 || rng.random::<f64>() < ring_share),
            };
            if ring {
                let crank = match first {
                    Some(t) => t,
                    None => self.bonds + rng.random_range(0..cranks),
                };
                let angle = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
                turn_bond(&mut y, &self.torsions[crank], angle);
                turns.push(Turn {
                    torsion: crank,
                    angle,
                });
            } else {
                let mut count = 1;
                while count < 3.min(self.bonds) && rng.random::<f64>() < 0.5 {
                    count += 1;
                }
                let chosen = match (&guide, first) {
                    (Some(p), Some(t)) => {
                        let mut chosen = vec![t];
                        while chosen.len() < count {
                            let rest: Vec<usize> =
                                (0..self.bonds).filter(|u| !chosen.contains(u)).collect();
                            chosen.push(pick(p, &rest, rng));
                        }
                        chosen
                    }
                    _ => {
                        order.shuffle(rng);
                        order[..count].to_vec()
                    }
                };
                for t in chosen {
                    let angle = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
                    turn_bond(&mut y, &self.torsions[t], angle);
                    turns.push(Turn { torsion: t, angle });
                }
            }
            let clean = self.repair(&mut y);
            if self.inverts(start.view(), y.view()) {
                continue;
            }
            if clean {
                return (y, turns);
            }
            let p = self.penalty(&self.clashes(y.view()));
            if best.as_ref().is_none_or(|(held, _, _)| p < *held) {
                best = Some((p, y, turns));
            }
        }
        best.map(|(_, y, turns)| (y, turns))
            .unwrap_or((start, Vec::new()))
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
    fn no_rotatable_bond_draws_nothing() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0]);
        let top = Topology::of(x.view(), &[8, 1, 1], 1.25, None, false);
        let mut rng = StdRng::seed_from_u64(1);
        let mut untouched = StdRng::seed_from_u64(1);
        assert_eq!(top.propose(x.view(), &mut rng), x);
        assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    }

    /// A zig-zag chain of `n` carbons in one plane, anti throughout.
    fn zigzag(n: usize) -> Array1<f64> {
        let half = (0.5 * 112.0_f64).to_radians();
        let mut x = Array1::zeros(3 * n);
        for i in 1..n {
            x[3 * i] = x[3 * (i - 1)] + 1.52 * half.sin();
            x[3 * i + 1] = if i % 2 == 1 { 1.52 * half.cos() } else { 0.0 };
        }
        x
    }

    #[test]
    fn a_turn_reads_back_as_a_change_in_that_torsion_alone() {
        let x = zigzag(8);
        let top = Topology::of(x.view(), &[6; 8], 1.25, None, false);
        assert_eq!(top.bonds, 5);
        assert!(top.turned_between(x.view(), x.view()).is_empty());
        for t in 0..top.bonds {
            let mut y = x.clone();
            turn_bond(&mut y, &top.torsions[t], 2.0);
            let (before, after) = (top.angles(x.view()), top.angles(y.view()));
            assert!((wrap(after[t] - before[t]).abs() - 2.0).abs() < 1e-9);
            assert_eq!(top.turned_between(x.view(), y.view()), vec![t]);
            let mut small = x.clone();
            turn_bond(&mut small, &top.torsions[t], 0.5 * TURNED);
            assert!(top.turned_between(x.view(), small.view()).is_empty());
        }
    }

    fn bond(t: usize) -> TorsionKey {
        (t, t + 1, false)
    }

    fn crank(t: usize) -> TorsionKey {
        (t, t + 1, true)
    }

    #[test]
    fn the_walk_has_nothing_to_weigh_until_a_turn_is_kept() {
        let keys: Vec<_> = (0..5).map(bond).collect();
        let mut walk = TorsionWalk::default();
        assert!(walk.probabilities(&keys, 0.0).is_none());
        for _ in 0..30 {
            walk.observe(&[bond(1), bond(3)], &[], 90.0);
        }
        assert!(walk.probabilities(&keys, 0.0).is_none());
        walk.observe(&[bond(2)], &[bond(2)], 90.0);
        assert!(walk.probabilities(&keys, 0.0).is_some());
    }

    #[test]
    fn the_walk_moves_the_draw_toward_torsions_that_yield_and_starves_none() {
        let keys: Vec<_> = (0..5).map(bond).collect();
        let mut walk = TorsionWalk::default();
        for _ in 0..40 {
            walk.observe(&[bond(2)], &[bond(2)], 80.0);
            for t in [0, 1, 3, 4] {
                walk.observe(&[bond(t)], &[], 80.0);
            }
        }
        let p = walk
            .probabilities(&keys, 0.0)
            .expect("a torsion has yielded");
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[2] > 0.5, "{p:?}");
        let floor = EXPLORE / 5.0;
        assert!(p.iter().all(|&q| q >= floor - 1e-12), "{p:?}");
    }

    #[test]
    fn a_class_that_yields_takes_its_share_from_one_that_does_not() {
        let keys: Vec<_> = (0..4).map(bond).chain((10..12).map(crank)).collect();
        let mut walk = TorsionWalk::default();
        for _ in 0..40 {
            for t in 0..4 {
                walk.observe(&[bond(t)], &[], 80.0);
            }
            for t in 10..12 {
                walk.observe(&[crank(t)], &[crank(t)], 80.0);
            }
        }
        let p = walk.probabilities(&keys, 0.1).expect("cranks have yielded");
        assert!(p[4] + p[5] > 0.5, "{p:?}");
        let mut flat = TorsionWalk::default();
        for _ in 0..40 {
            for key in &keys {
                flat.observe(&[*key], &[*key], 80.0);
            }
        }
        let q = flat.probabilities(&keys, 0.1).expect("all have yielded");
        assert!((q[4] + q[5] - 0.1).abs() < 1e-9, "{q:?}");
    }

    #[test]
    fn a_torsion_the_geometry_stops_offering_keeps_its_record() {
        let mut walk = TorsionWalk::default();
        for _ in 0..20 {
            walk.observe(&[crank(7)], &[crank(7)], 80.0);
            walk.observe(&[bond(0)], &[], 80.0);
        }
        let without: Vec<_> = vec![bond(0), bond(1)];
        let with: Vec<_> = vec![bond(0), bond(1), crank(7)];
        assert!(walk.probabilities(&without, 0.2).is_some());
        let p = walk.probabilities(&with, 0.2).expect("seen");
        assert!(p[2] > 0.5, "{p:?}");
    }

    #[test]
    fn a_trained_walk_proposes_the_torsion_that_yields_more_often() {
        let x = zigzag(8);
        let top = Topology::of(x.view(), &[6; 8], 1.25, None, false);
        let keys = top.keys();
        let mut walk = TorsionWalk::default();
        for _ in 0..40 {
            walk.observe(&[keys[2]], &[keys[2]], 80.0);
            for t in [0, 1, 3, 4] {
                walk.observe(&[keys[t]], &[], 80.0);
            }
        }
        let share = |walk: Option<&TorsionWalk>| {
            let mut rng = StdRng::seed_from_u64(21);
            let hits = (0..400)
                .filter(|_| {
                    top.propose_with(x.view(), walk, &mut rng)
                        .1
                        .iter()
                        .any(|t| t.torsion == 2)
                })
                .count();
            hits as f64 / 400.0
        };
        let (plain, guided) = (share(None), share(Some(&walk)));
        assert!(guided > plain + 0.15, "{plain} against {guided}");
    }

    #[test]
    fn a_walk_with_nothing_to_say_proposes_what_the_plain_draw_does() {
        let x = zigzag(8);
        let top = Topology::of(x.view(), &[6; 8], 1.25, None, false);
        let walk = TorsionWalk::default();
        for seed in 0..8 {
            let mut plain = StdRng::seed_from_u64(seed);
            let mut guided = StdRng::seed_from_u64(seed);
            let (y, turns) = top.propose_with(x.view(), None, &mut plain);
            let (z, again) = top.propose_with(x.view(), Some(&walk), &mut guided);
            assert_eq!(y, z);
            assert_eq!(turns, again);
            assert_eq!(top.propose(x.view(), &mut StdRng::seed_from_u64(seed)), y);
            assert_eq!(plain.random::<u64>(), guided.random::<u64>());
        }
    }

    #[test]
    fn the_turns_a_proposal_reports_are_the_ones_it_made() {
        let x = zigzag(8);
        let top = Topology::of(x.view(), &[6; 8], 1.25, None, false);
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..20 {
            let (_, turns) = top.propose_with(x.view(), None, &mut rng);
            assert!((1..=3).contains(&turns.len()));
            let mut ids: Vec<usize> = turns.iter().map(|t| t.torsion).collect();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), turns.len());
            assert!(turns.iter().all(|t| t.torsion < top.bonds));
        }
    }
}
