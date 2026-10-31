//! Torsion proposals over the rotatable bonds of a molecule's own bond graph.
//!
//! The molecular library moves whole groups rigidly, which leaves a single
//! flexible molecule with no proposal that changes its conformation except
//! the SOAP pullback. A torsion move is the step conformer searches are built
//! on: Li and Scheraga's Monte Carlo minimisation on polypeptide dihedrals
//! (doi:10.1073/pnas.84.19.6611), the internal-coordinate Monte Carlo of
//! Chang, Guida and Still (doi:10.1021/ja00194a035), Tinker's MONTE, GMIN's
//! GROUPROTATION and the dihedral basin hopping of CREST 3.1. The survey
//! behind the choices here is `docs/orgmode/methods/torsion.org`.
//!
//! Bonds are read off the current geometry with the bond-matrix rule over
//! covalent radii, the same rule the per-hop molecular groups use. A bond is
//! rotatable when it is a bridge of that graph (so not in a ring), neither
//! end is terminal or linear, it is not a double bond, and it is not
//! excluded by [`RotorRules`]: amide-like linkages and methyl-like terminal
//! groups by default, after RDKit's strict rotor definition. The smaller side
//! rotates about the bond axis, so every bond length and bond angle is kept
//! exactly. Rejected draws never reach the objective, so the ledger charges
//! only the trial that is returned.

use std::f64::consts::{PI, TAU};

use ndarray::{Array1, ArrayView1};
use rand::Rng;

use super::covalent_radius;

/// Staggered sp3-sp3 dihedrals: gauche-, anti, gauche+.
const STAGGERED: [f64; 3] = [PI / 3.0, PI, 5.0 * PI / 3.0];
/// Six-fold sp2-sp3 dihedrals, eclipsed and staggered against the planar end.
const SIXFOLD: [f64; 6] = [
    0.0,
    PI / 3.0,
    2.0 * PI / 3.0,
    PI,
    4.0 * PI / 3.0,
    5.0 * PI / 3.0,
];
/// Planar sp2-sp2 dihedrals: syn and anti.
const PLANAR: [f64; 2] = [0.0, PI];

/// Separation below which a single bond between two trigonal or linear
/// atoms reads as a double bond, as a fraction of the covalent radii sum.
/// C=C is 0.88 of it and C=N 0.87; a conjugated C-C single bond is 0.96.
const DOUBLE_BOND_FRACTION: f64 = 0.93;

/// Bond angle above which a two-coordinate atom is linear.
const LINEAR_ANGLE: f64 = 165.0 * PI / 180.0;

/// Which bonds into a terminal group the proposal rotates.
///
/// A terminal group is one atom whose other neighbours are all terminal:
/// CH3, CF3, OH, NH2, NH3+, SH, COO-.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalRotors {
    /// No bond into a terminal group rotates, CF3 and COO- included. This is
    /// stricter than Confab's rule, [`TerminalRotors::HeavyAtom`].
    Exclude,
    /// Confab's rule: a bond rotates only when each end has two or more
    /// non-hydrogen neighbours. CH3, OH, NH2, NH3+ and SH do not turn; CF3
    /// and COO- do.
    HeavyAtom,
    /// Methyl-like groups, a carbon carrying only hydrogens and halogens, do
    /// not rotate; OH, NH2, NH3+, SH and COO- do. The quench restores a
    /// methyl's three-fold well, while a polar group's orientation decides
    /// which hydrogen bonds the minimum has.
    Polar,
    /// Every terminal group rotates.
    Include,
}

/// Which bonds of the bond graph count as rotatable.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct RotorRules {
    /// Two atoms bond when their separation is below this multiple of the
    /// sum of their covalent radii.
    pub bond_tolerance: f64,
    /// Bonds into terminal groups.
    pub terminal: TerminalRotors,
    /// Rotate amide C-N, ester and acid C-O, and thioester C-S bonds. Off is
    /// RDKit's strict definition: the partial double bond holds the linkage
    /// planar, and a flip is a cis-trans isomerisation, not a rotamer.
    pub amide_like: bool,
}

/// Angle and composition settings of the torsion proposal.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TorsionSettings {
    /// Bonds into terminal groups.
    pub terminal: TerminalRotors,
    /// Rotate amide-like linkages. See [`RotorRules::amide_like`].
    pub amide_like: bool,
    /// Probability that a rotor's new dihedral is a preferred value rather
    /// than a uniform draw.
    pub preferred_fraction: f64,
    /// Standard deviation of the Gaussian jitter on a preferred value, in
    /// radians.
    pub preferred_jitter: f64,
    /// Half-width, in radians, of the window around the current dihedral a
    /// uniform draw avoids. A smaller turn quenches back to where it began.
    pub uniform_exclusion: f64,
    /// Probability that a proposal turns several adjacent rotors at once.
    pub correlated_fraction: f64,
    /// Most rotors one correlated proposal turns.
    pub max_correlated: usize,
    /// Share of correlated proposals that are a counter-rotation of two
    /// parallel rotors one bond apart.
    pub counter_rotation_fraction: f64,
    /// Contact below this multiple of the van der Waals radii sum is a clash,
    /// for pairs three or more bonds apart that the draw brought closer.
    pub clash_factor: f64,
    /// Draws tried before the proposal gives up and returns the structure
    /// unchanged.
    pub max_attempts: usize,
}

impl TorsionSettings {
    /// Settings [`super::Config::recommended_molecular`] installs.
    ///
    /// The mixture and composition rates are design choices, not values
    /// fitted on a campaign.
    pub fn recommended() -> Self {
        Self {
            terminal: TerminalRotors::Polar,
            amide_like: false,
            preferred_fraction: 0.5,
            preferred_jitter: 15.0_f64.to_radians(),
            uniform_exclusion: 30.0_f64.to_radians(),
            correlated_fraction: 0.25,
            max_correlated: 3,
            counter_rotation_fraction: 0.5,
            clash_factor: 0.6,
            max_attempts: 32,
        }
    }

    /// Rotor rules at the given bond tolerance.
    pub fn rules(&self, bond_tolerance: f64) -> RotorRules {
        RotorRules {
            bond_tolerance,
            terminal: self.terminal,
            amide_like: self.amide_like,
        }
    }
}

impl Default for TorsionSettings {
    fn default() -> Self {
        Self::recommended()
    }
}

/// One rotatable bond and the side of it that turns.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotor {
    /// Axis atom on the side that stays put.
    pub fixed: usize,
    /// Axis atom on the side that turns.
    pub pivot: usize,
    /// Atoms that turn, the pivot excluded.
    pub moving: Vec<usize>,
    /// `[a, d]` of the dihedral `a-fixed-pivot-d` the preferred values refer to.
    pub reference: [usize; 2],
    /// Preferred values of that dihedral, radians in `[0, 2 pi)`.
    pub preferred: &'static [f64],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Centre {
    Terminal,
    Linear,
    Trigonal,
    Tetrahedral,
}

/// Bond graph of one geometry, with the coordination class of each atom.
#[derive(Debug, Clone)]
pub struct Topology {
    neighbours: Vec<Vec<usize>>,
    centre: Vec<Centre>,
    species: Vec<u32>,
}

impl Topology {
    /// Bonds every pair closer than `tolerance` times its covalent radii sum.
    pub fn new(x: ArrayView1<f64>, species: &[u32], tolerance: f64) -> Self {
        let n = species.len().min(x.len() / 3);
        let mut neighbours = vec![Vec::new(); n];
        for a in 0..n {
            let ra = covalent_radius(species[a]);
            for b in (a + 1)..n {
                let cut = tolerance * (ra + covalent_radius(species[b]));
                if cut > 0.0 && distance(x, a, b) < cut {
                    neighbours[a].push(b);
                    neighbours[b].push(a);
                }
            }
        }
        let degree_class = |a: usize| -> Centre {
            let nb = &neighbours[a];
            match nb.len() {
                0 | 1 => Centre::Terminal,
                2 if angle(x, nb[0], a, nb[1]) > LINEAR_ANGLE => Centre::Linear,
                2 | 3 => Centre::Trigonal,
                _ => Centre::Tetrahedral,
            }
        };
        let mut centre: Vec<Centre> = (0..n).map(degree_class).collect();
        // A carbon's class follows from its coordination. A three-coordinate
        // nitrogen or a two-coordinate oxygen or sulfur is planar only when
        // it is conjugated with an unsaturated carbon: an amide or aniline
        // nitrogen, an ester or phenol oxygen. Otherwise it is an sp3 centre
        // with lone pairs.
        let unsaturated_carbon =
            |b: usize, centre: &[Centre]| species[b] == 6 && centre[b] == Centre::Trigonal;
        let snapshot = centre.clone();
        for a in 0..n {
            let conjugated = neighbours[a]
                .iter()
                .any(|&b| unsaturated_carbon(b, &snapshot));
            match (species[a], neighbours[a].len()) {
                (7, 3) | (8, 2) | (16, 2) => {
                    centre[a] = if conjugated {
                        Centre::Trigonal
                    } else {
                        Centre::Tetrahedral
                    };
                }
                (6, _) | (7, _) | (8, _) | (16, _) => {}
                (_, 3) => centre[a] = Centre::Tetrahedral,
                _ => {}
            }
        }
        Self {
            neighbours,
            centre,
            species: species[..n].to_vec(),
        }
    }

    /// Atoms in the graph.
    pub fn len(&self) -> usize {
        self.neighbours.len()
    }

    /// Whether the graph has no atoms.
    pub fn is_empty(&self) -> bool {
        self.neighbours.is_empty()
    }

    /// Bonded neighbours of `atom`.
    pub fn neighbours(&self, atom: usize) -> &[usize] {
        &self.neighbours[atom]
    }

    /// Every bond as `(a, b)` with `a < b`.
    pub fn bonds(&self) -> Vec<(usize, usize)> {
        let mut bonds = Vec::new();
        for (a, nb) in self.neighbours.iter().enumerate() {
            for &b in nb {
                if a < b {
                    bonds.push((a, b));
                }
            }
        }
        bonds
    }

    /// Atoms reached from `start` without crossing the bond to `across`.
    pub fn side(&self, start: usize, across: usize) -> Vec<usize> {
        let mut seen = vec![false; self.len()];
        seen[start] = true;
        let mut queue = vec![start];
        let mut side = vec![start];
        while let Some(a) = queue.pop() {
            for &b in &self.neighbours[a] {
                if (a == start && b == across) || seen[b] {
                    continue;
                }
                seen[b] = true;
                side.push(b);
                queue.push(b);
            }
        }
        side.sort_unstable();
        side
    }

    fn is_terminal_group(&self, side: &[usize], root: usize) -> bool {
        side.iter()
            .all(|&a| a == root || self.neighbours[a].len() == 1)
    }

    fn is_methyl_like(&self, side: &[usize], root: usize) -> bool {
        self.species[root] == 6
            && side
                .iter()
                .all(|&a| a == root || matches!(self.species[a], 1 | 9 | 17 | 35 | 53))
    }

    fn carbonyl_carbon(&self, carbon: usize, other: usize) -> bool {
        self.species[carbon] == 6
            && self.neighbours[carbon].len() == 3
            && self.neighbours[carbon].iter().any(|&b| {
                b != other && matches!(self.species[b], 8 | 16) && self.neighbours[b].len() == 1
            })
    }

    /// Amide C-N, ester or acid C-O, thioester C-S.
    fn is_amide_like(&self, a: usize, b: usize) -> bool {
        let linked = |carbon: usize, hetero: usize| {
            self.carbonyl_carbon(carbon, hetero)
                && matches!(self.species[hetero], 7 | 8 | 16)
                && self.neighbours[hetero].len() >= 2
        };
        linked(a, b) || linked(b, a)
    }

    fn is_multiple_bond(&self, x: ArrayView1<f64>, a: usize, b: usize) -> bool {
        let planar = |c: Centre| matches!(c, Centre::Trigonal | Centre::Linear);
        planar(self.centre[a])
            && planar(self.centre[b])
            && distance(x, a, b)
                < DOUBLE_BOND_FRACTION
                    * (covalent_radius(self.species[a]) + covalent_radius(self.species[b]))
    }

    fn preferred(&self, a: usize, b: usize) -> &'static [f64] {
        match (self.centre[a], self.centre[b]) {
            (Centre::Tetrahedral, Centre::Tetrahedral) => &STAGGERED,
            (Centre::Trigonal, Centre::Trigonal) => &PLANAR,
            _ => &SIXFOLD,
        }
    }

    fn reference_neighbour(&self, atom: usize, skip: usize) -> usize {
        self.neighbours[atom]
            .iter()
            .copied()
            .filter(|&b| b != skip)
            .min_by_key(|&b| (self.species[b] == 1, b))
            .unwrap_or(skip)
    }

    /// The rotatable bonds under `rules`, each with the side that turns.
    ///
    /// The smaller side turns. A side holding a frozen atom never turns, and
    /// a bond with frozen atoms on both sides is dropped.
    pub fn rotors(
        &self,
        x: ArrayView1<f64>,
        rules: &RotorRules,
        frozen: Option<&[bool]>,
    ) -> Vec<Rotor> {
        let held = |side: &[usize]| {
            frozen.is_some_and(|mask| side.iter().any(|&a| mask.get(a).copied().unwrap_or(false)))
        };
        let mut rotors = Vec::new();
        for (a, b) in self.bonds() {
            if [a, b]
                .iter()
                .any(|&i| matches!(self.centre[i], Centre::Terminal | Centre::Linear))
            {
                continue;
            }
            let side_b = self.side(b, a);
            if side_b.binary_search(&a).is_ok() {
                continue;
            }
            let amide = self.is_amide_like(a, b);
            if amide && !rules.amide_like {
                continue;
            }
            if !amide && self.is_multiple_bond(x, a, b) {
                continue;
            }
            let side_a = self.side(a, b);
            let excluded = |side: &[usize], root: usize| match rules.terminal {
                TerminalRotors::Include => false,
                TerminalRotors::Exclude => self.is_terminal_group(side, root),
                TerminalRotors::HeavyAtom => {
                    self.neighbours[root]
                        .iter()
                        .filter(|&&b| self.species[b] != 1)
                        .count()
                        < 2
                }
                TerminalRotors::Polar => {
                    self.is_terminal_group(side, root) && self.is_methyl_like(side, root)
                }
            };
            if excluded(&side_a, a) || excluded(&side_b, b) {
                continue;
            }
            let turn_b = match (held(&side_a), held(&side_b)) {
                (true, true) => continue,
                (true, false) => true,
                (false, true) => false,
                (false, false) => side_b.len() <= side_a.len(),
            };
            let (fixed, pivot, side) = if turn_b {
                (a, b, side_b)
            } else {
                (b, a, side_a)
            };
            rotors.push(Rotor {
                fixed,
                pivot,
                moving: side.into_iter().filter(|&i| i != pivot).collect(),
                reference: [
                    self.reference_neighbour(fixed, pivot),
                    self.reference_neighbour(pivot, fixed),
                ],
                preferred: self.preferred(fixed, pivot),
            });
        }
        rotors
    }

    /// Pairs of atoms one or two bonds apart, as an `n x n` mask.
    fn within_two_bonds(&self) -> Vec<bool> {
        let n = self.len();
        let mut near = vec![false; n * n];
        for a in 0..n {
            near[a * n + a] = true;
            for &b in &self.neighbours[a] {
                near[a * n + b] = true;
                for &c in &self.neighbours[b] {
                    near[a * n + c] = true;
                }
            }
        }
        near
    }
}

/// Rotatable bonds of one geometry. See [`Topology::rotors`].
pub fn rotatable_bonds(x: ArrayView1<f64>, species: &[u32], rules: &RotorRules) -> Vec<Rotor> {
    Topology::new(x, species, rules.bond_tolerance).rotors(x, rules, None)
}

/// Dihedral `a-b-c-d` in `(-pi, pi]`.
///
/// Turning `d` about the axis from `b` to `c` by a right-handed angle raises
/// it by that angle.
pub fn dihedral(x: ArrayView1<f64>, a: usize, b: usize, c: usize, d: usize) -> f64 {
    let b1 = sub(point(x, b), point(x, a));
    let b2 = sub(point(x, c), point(x, b));
    let b3 = sub(point(x, d), point(x, c));
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let y = norm(b2) * dot(b1, n2);
    let x = dot(n1, n2);
    y.atan2(x)
}

/// Turns `atoms` about the axis from `fixed` to `pivot` by `angle` radians.
pub fn rotate_about_bond(
    y: &mut Array1<f64>,
    fixed: usize,
    pivot: usize,
    atoms: &[usize],
    angle: f64,
) {
    let origin = point(y.view(), pivot);
    let axis = sub(origin, point(y.view(), fixed));
    let length = norm(axis);
    if !(length > 0.0) {
        return;
    }
    let k = [axis[0] / length, axis[1] / length, axis[2] / length];
    let (s, c) = angle.sin_cos();
    for &atom in atoms {
        let v = sub(point(y.view(), atom), origin);
        let kv = cross(k, v);
        let along = dot(k, v) * (1.0 - c);
        for i in 0..3 {
            y[3 * atom + i] = origin[i] + v[i] * c + kv[i] * s + k[i] * along;
        }
    }
}

/// Rotor arm of the molecular library, [`super::ClusterMove::Rotor`].
#[derive(Debug, Clone, PartialEq)]
pub struct TorsionMove {
    /// Atomic number of each atom.
    pub species: Vec<u32>,
    /// Which bonds rotate.
    pub rules: RotorRules,
    /// Angle mixture, composition, and clash settings.
    pub settings: TorsionSettings,
    /// Radius about the centroid a draw may not newly exceed. `None` when a
    /// frozen frame, not the centroid, is the frame.
    pub container: Option<f64>,
    /// Atoms that never move.
    pub frozen: Option<Vec<bool>>,
}

impl TorsionMove {
    /// One proposal from `x`.
    ///
    /// A draw that brings a pair three or more bonds apart closer than the
    /// clash distance, or carries an atom past the container, is redrawn
    /// without an energy call. After [`TorsionSettings::max_attempts`] such
    /// draws, or with no rotatable bond, `x` comes back unchanged and the
    /// driver draws another arm instead of paying a quench for it.
    pub fn propose<R: Rng + ?Sized>(&self, x: ArrayView1<f64>, rng: &mut R) -> Array1<f64> {
        let topology = Topology::new(x, &self.species, self.rules.bond_tolerance);
        let n = topology.len();
        if n * 3 != x.len() {
            return x.to_owned();
        }
        let rotors = topology.rotors(x, &self.rules, self.frozen.as_deref());
        if rotors.is_empty() {
            return x.to_owned();
        }
        let near = topology.within_two_bonds();
        let radius_limit = self.container.map(|r| r.max(extent(x)));
        let pairs = counter_rotation_pairs(x, &topology, &rotors);
        for _ in 0..self.settings.max_attempts.max(1) {
            let mut y = x.to_owned();
            let correlated =
                rotors.len() >= 2 && rng.random::<f64>() < self.settings.correlated_fraction;
            let moved = if correlated
                && !pairs.is_empty()
                && rng.random::<f64>() < self.settings.counter_rotation_fraction
            {
                let pick = pairs[rng.random_range(0..pairs.len())];
                self.counter_rotate(&topology, &mut y, pick, rng)
            } else if correlated {
                for r in self.correlated_set(&topology, &rotors, rng) {
                    self.turn(&mut y, &rotors[r], rng);
                }
                true
            } else {
                let r = rng.random_range(0..rotors.len());
                self.turn(&mut y, &rotors[r], rng);
                true
            };
            if !moved {
                continue;
            }
            if radius_limit.is_some_and(|limit| extent(y.view()) > limit + 1e-9) {
                continue;
            }
            if !self.introduces_clash(x, y.view(), &near) {
                return y;
            }
        }
        x.to_owned()
    }

    fn turn<R: Rng + ?Sized>(&self, y: &mut Array1<f64>, rotor: &Rotor, rng: &mut R) {
        let [a, d] = rotor.reference;
        let current = dihedral(y.view(), a, rotor.fixed, rotor.pivot, d);
        let delta = self.draw_turn(current, rotor.preferred, rng);
        rotate_about_bond(y, rotor.fixed, rotor.pivot, &rotor.moving, delta);
    }

    /// Turn that takes `current` to a new dihedral.
    ///
    /// A preferred draw picks one of the preferred values other than the one
    /// nearest `current`, which is the well the structure already sits in.
    fn draw_turn<R: Rng + ?Sized>(&self, current: f64, preferred: &[f64], rng: &mut R) -> f64 {
        if preferred.len() >= 2 && rng.random::<f64>() < self.settings.preferred_fraction {
            let here = preferred
                .iter()
                .enumerate()
                .min_by(|(_, p), (_, q)| {
                    circular_gap(current, **p).total_cmp(&circular_gap(current, **q))
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
            let mut pick = rng.random_range(0..preferred.len() - 1);
            if pick >= here {
                pick += 1;
            }
            let target = preferred[pick] + self.settings.preferred_jitter * gaussian(rng);
            wrap(target - current)
        } else {
            let w = self.settings.uniform_exclusion.clamp(0.0, PI);
            w + rng.random::<f64>() * (TAU - 2.0 * w)
        }
    }

    /// Rotors turned together: a random rotor and up to two more, each
    /// adjacent to one already chosen. A rotor with no rotatable neighbour
    /// turns alone, so a draw never reaches into another molecule.
    fn correlated_set<R: Rng + ?Sized>(
        &self,
        topology: &Topology,
        rotors: &[Rotor],
        rng: &mut R,
    ) -> Vec<usize> {
        let cap = self.settings.max_correlated.clamp(2, rotors.len());
        let mut size = 2;
        while size < cap && rng.random::<f64>() < 0.5 {
            size += 1;
        }
        let mut chosen = vec![rng.random_range(0..rotors.len())];
        while chosen.len() < size {
            let adjacent: Vec<usize> = (0..rotors.len())
                .filter(|r| !chosen.contains(r))
                .filter(|&r| {
                    chosen
                        .iter()
                        .any(|&c| rotors_adjacent(topology, &rotors[c], &rotors[r]))
                })
                .collect();
            if adjacent.is_empty() {
                break;
            }
            chosen.push(adjacent[rng.random_range(0..adjacent.len())]);
        }
        chosen
    }

    /// Turns the first rotor of `[p0, p1, p2, p3]` by `delta` and the second
    /// by `-delta`, or by `delta` when the axes point opposite ways.
    ///
    /// The side behind `p0` stays put and the part beyond `p3` turns by at
    /// most twice the angle between the axes, which is at most
    /// [`PARALLEL_AXES`]; along an all-trans chain it is translated without
    /// turning. This is not an end-fixed crankshaft: the far end moves.
    fn counter_rotate<R: Rng + ?Sized>(
        &self,
        topology: &Topology,
        y: &mut Array1<f64>,
        [p0, p1, p2, p3]: [usize; 4],
        rng: &mut R,
    ) -> bool {
        let first: Vec<usize> = topology
            .side(p1, p0)
            .into_iter()
            .filter(|&a| a != p1)
            .collect();
        let second: Vec<usize> = topology
            .side(p3, p2)
            .into_iter()
            .filter(|&a| a != p3)
            .collect();
        if let Some(mask) = self.frozen.as_deref()
            && first.iter().any(|&a| mask.get(a).copied().unwrap_or(false))
        {
            return false;
        }
        let u = sub(point(y.view(), p1), point(y.view(), p0));
        let v = sub(point(y.view(), p3), point(y.view(), p2));
        let back = if dot(u, v) < 0.0 { 1.0 } else { -1.0 };
        let w = self.settings.uniform_exclusion.clamp(0.0, PI);
        let delta = w + rng.random::<f64>() * (TAU - 2.0 * w);
        rotate_about_bond(y, p0, p1, &first, delta);
        rotate_about_bond(y, p2, p3, &second, back * delta);
        true
    }

    fn introduces_clash(&self, x: ArrayView1<f64>, y: ArrayView1<f64>, near: &[bool]) -> bool {
        let n = self.species.len();
        let radii: Vec<f64> = self
            .species
            .iter()
            .map(|&z| van_der_waals_radius(z))
            .collect();
        let factor = self.settings.clash_factor;
        for i in 0..n {
            for j in (i + 1)..n {
                if near[i * n + j] {
                    continue;
                }
                let limit = factor * (radii[i] + radii[j]);
                let after = squared_distance(y, i, j);
                if after >= limit * limit {
                    continue;
                }
                if after < squared_distance(x, i, j) - 1e-12 {
                    return true;
                }
            }
        }
        false
    }
}

/// Whether two rotatable bonds share an atom or are joined by one bond.
fn rotors_adjacent(topology: &Topology, first: &Rotor, second: &Rotor) -> bool {
    let u = [first.fixed, first.pivot];
    let v = [second.fixed, second.pivot];
    u.iter().any(|&a| {
        v.iter()
            .any(|&b| a == b || topology.neighbours(a).contains(&b))
    })
}

/// Largest angle between the two axes of a counter-rotation pair.
const PARALLEL_AXES: f64 = 10.0 * PI / 180.0;

/// Rotor pairs on a path `p0-p1-p2-p3`, with `(p0, p1)` and `(p2, p3)`
/// rotatable, `p1-p2` bonded, and the two axes parallel or antiparallel
/// within [`PARALLEL_AXES`]. Each is oriented so the part beyond `p3` is no
/// larger than the part behind `p0`. Bonds two apart along an all-trans
/// chain qualify, as do peptide psi and the next phi across a trans amide.
fn counter_rotation_pairs(
    x: ArrayView1<f64>,
    topology: &Topology,
    rotors: &[Rotor],
) -> Vec<[usize; 4]> {
    let parallel = |p0: usize, p1: usize, p2: usize, p3: usize| {
        let u = sub(point(x, p1), point(x, p0));
        let v = sub(point(x, p3), point(x, p2));
        dot(u, v).abs() >= PARALLEL_AXES.cos() * norm(u) * norm(v)
    };
    let mut pairs = Vec::new();
    for (i, first) in rotors.iter().enumerate() {
        for second in rotors.iter().skip(i + 1) {
            let u = [first.fixed, first.pivot];
            let v = [second.fixed, second.pivot];
            if u.iter().any(|a| v.contains(a)) {
                continue;
            }
            for &p1 in &u {
                for &p2 in &v {
                    if !topology.neighbours(p1).contains(&p2) {
                        continue;
                    }
                    let p0 = if p1 == u[0] { u[1] } else { u[0] };
                    let p3 = if p2 == v[0] { v[1] } else { v[0] };
                    if !parallel(p0, p1, p2, p3) {
                        continue;
                    }
                    let behind = topology.side(p0, p1).len();
                    let beyond = topology.side(p3, p2).len();
                    pairs.push(if beyond <= behind {
                        [p0, p1, p2, p3]
                    } else {
                        [p3, p2, p1, p0]
                    });
                }
            }
        }
    }
    pairs
}

/// Van der Waals radius in Angstrom, Bondi (doi:10.1021/j100785a001).
/// Elements outside his table take the covalent radius plus 0.8.
pub fn van_der_waals_radius(z: u32) -> f64 {
    match z {
        1 => 1.20,
        2 => 1.40,
        6 => 1.70,
        7 => 1.55,
        8 => 1.52,
        9 => 1.47,
        10 => 1.54,
        14 => 2.10,
        15 => 1.80,
        16 => 1.80,
        17 => 1.75,
        18 => 1.88,
        33 => 1.85,
        34 => 1.90,
        35 => 1.85,
        36 => 2.02,
        53 => 1.98,
        54 => 2.16,
        _ => covalent_radius(z) + 0.8,
    }
}

fn gaussian<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

fn wrap(angle: f64) -> f64 {
    angle.rem_euclid(TAU)
}

fn circular_gap(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(TAU);
    d.min(TAU - d)
}

fn extent(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n == 0 {
        return 0.0;
    }
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for value in &mut c {
        *value /= n as f64;
    }
    (0..n)
        .map(|i| norm(sub(point(x, i), c)))
        .fold(0.0, f64::max)
}

fn point(x: ArrayView1<f64>, i: usize) -> [f64; 3] {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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

fn squared_distance(x: ArrayView1<f64>, i: usize, j: usize) -> f64 {
    let d = sub(point(x, i), point(x, j));
    dot(d, d)
}

fn distance(x: ArrayView1<f64>, i: usize, j: usize) -> f64 {
    squared_distance(x, i, j).sqrt()
}

fn angle(x: ArrayView1<f64>, a: usize, b: usize, c: usize) -> f64 {
    let u = sub(point(x, a), point(x, b));
    let v = sub(point(x, c), point(x, b));
    let denom = norm(u) * norm(v);
    if denom > 0.0 {
        (dot(u, v) / denom).clamp(-1.0, 1.0).acos()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type Row = (u32, isize, isize, isize, f64, f64, f64);

    /// Cartesian coordinates from a Z-matrix. A row `(z, c, b, a, r, theta,
    /// phi)` bonds the new atom to `c` at `r`, with angle `b-c-new` of
    /// `theta` degrees and dihedral `a-b-c-new` of `phi`. Indices -3 to -1
    /// are a dummy frame that is dropped from the result.
    fn build(rows: &[Row]) -> (Array1<f64>, Vec<u32>) {
        let mut points: Vec<[f64; 3]> = vec![[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let mut species = Vec::new();
        for &(z, c, b, a, r, theta, phi) in rows {
            let at = |i: isize| points[(i + 3) as usize];
            let placed = place(at(a), at(b), at(c), r, theta.to_radians(), phi.to_radians());
            points.push(placed);
            species.push(z);
        }
        let x = Array1::from_iter(points[3..].iter().flatten().copied());
        (x, species)
    }

    fn place(a: [f64; 3], b: [f64; 3], c: [f64; 3], r: f64, theta: f64, phi: f64) -> [f64; 3] {
        let unit = |v: [f64; 3]| {
            let l = norm(v);
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let bc = unit(sub(c, b));
        let n = unit(cross(sub(b, a), bc));
        let m = cross(n, bc);
        let d = [
            -r * theta.cos(),
            r * theta.sin() * phi.cos(),
            r * theta.sin() * phi.sin(),
        ];
        [
            c[0] + d[0] * bc[0] + d[1] * m[0] + d[2] * n[0],
            c[1] + d[0] * bc[1] + d[1] * m[1] + d[2] * n[1],
            c[2] + d[0] * bc[2] + d[1] * m[2] + d[2] * n[2],
        ]
    }

    /// A straight-chain alkane whose backbone dihedrals are `dihedrals`, in
    /// degrees. Carbons come first, in chain order.
    fn alkane(dihedrals: &[f64]) -> (Array1<f64>, Vec<u32>) {
        let k = dihedrals.len() + 3;
        let mut rows: Vec<Row> = vec![
            (6, -1, -2, -3, 1.0, 100.0, 0.0),
            (6, 0, -1, -2, 1.53, 109.5, 60.0),
            (6, 1, 0, -1, 1.53, 111.0, 180.0),
        ];
        for (j, &phi) in dihedrals.iter().enumerate() {
            let j = (j + 3) as isize;
            rows.push((6, j - 1, j - 2, j - 3, 1.53, 111.0, phi));
        }
        let last = (k - 1) as isize;
        for phi in [60.0, 180.0, 300.0] {
            rows.push((1, 0, 1, 2, 1.09, 109.5, phi));
            rows.push((1, last, last - 1, last - 2, 1.09, 109.5, phi));
        }
        for i in 1..k - 1 {
            let (b, a, along) = if i + 2 < k {
                (i + 1, i + 2, dihedrals[i - 1])
            } else {
                (i - 1, i - 2, dihedrals[i - 2])
            };
            for offset in [120.0, 240.0] {
                rows.push((
                    1,
                    i as isize,
                    b as isize,
                    a as isize,
                    1.09,
                    109.5,
                    along + offset,
                ));
            }
        }
        build(&rows)
    }

    fn butane(phi: f64) -> (Array1<f64>, Vec<u32>) {
        alkane(&[phi])
    }

    fn ethanol() -> (Array1<f64>, Vec<u32>) {
        let mut rows: Vec<Row> = vec![
            (6, -1, -2, -3, 1.0, 100.0, 0.0),
            (6, 0, -1, -2, 1.52, 109.5, 60.0),
            (8, 1, 0, -1, 1.43, 109.5, 180.0),
            (1, 2, 1, 0, 0.96, 108.5, 180.0),
        ];
        for phi in [60.0, 180.0, 300.0] {
            rows.push((1, 0, 1, 2, 1.09, 109.5, phi));
        }
        for phi in [60.0, 300.0] {
            rows.push((1, 1, 2, 3, 1.09, 109.5, phi));
        }
        build(&rows)
    }

    fn methylacetamide() -> (Array1<f64>, Vec<u32>) {
        let mut rows: Vec<Row> = vec![
            (6, -1, -2, -3, 1.0, 100.0, 0.0),
            (6, 0, -1, -2, 1.52, 109.5, 60.0),
            (8, 1, 0, -1, 1.23, 121.0, 0.0),
            (7, 1, 0, 2, 1.33, 116.0, 180.0),
            (6, 3, 1, 0, 1.45, 122.0, 180.0),
            (1, 3, 1, 0, 1.01, 119.0, 0.0),
        ];
        for phi in [0.0, 120.0, 240.0] {
            rows.push((1, 0, 1, 3, 1.09, 109.5, phi));
        }
        for phi in [60.0, 180.0, 300.0] {
            rows.push((1, 4, 3, 1, 1.09, 109.5, phi));
        }
        build(&rows)
    }

    fn butene() -> (Array1<f64>, Vec<u32>) {
        let mut rows: Vec<Row> = vec![
            (6, -1, -2, -3, 1.0, 100.0, 0.0),
            (6, 0, -1, -2, 1.50, 109.5, 60.0),
            (6, 1, 0, -1, 1.34, 123.0, 180.0),
            (6, 2, 1, 0, 1.50, 123.0, 180.0),
            (1, 1, 2, 3, 1.08, 120.0, 0.0),
            (1, 2, 1, 0, 1.08, 120.0, 0.0),
        ];
        for phi in [60.0, 180.0, 300.0] {
            rows.push((1, 0, 1, 2, 1.09, 109.5, phi));
            rows.push((1, 3, 2, 1, 1.09, 109.5, phi));
        }
        build(&rows)
    }

    fn cyclohexane_carbons() -> (Array1<f64>, Vec<u32>) {
        let radius = (1.53_f64.powi(2) - 0.25).sqrt();
        let x = Array1::from_iter((0..6).flat_map(|k| {
            let t = k as f64 * PI / 3.0;
            let z = if k % 2 == 0 { 0.25 } else { -0.25 };
            [radius * t.cos(), radius * t.sin(), z]
        }));
        (x, vec![6; 6])
    }

    fn rules(terminal: TerminalRotors, amide_like: bool) -> RotorRules {
        RotorRules {
            bond_tolerance: 1.25,
            terminal,
            amide_like,
        }
    }

    fn axes(rotors: &[Rotor]) -> Vec<[usize; 2]> {
        let mut axes: Vec<[usize; 2]> = rotors
            .iter()
            .map(|r| [r.fixed.min(r.pivot), r.fixed.max(r.pivot)])
            .collect();
        axes.sort_unstable();
        axes
    }

    fn mover(species: Vec<u32>, settings: TorsionSettings) -> TorsionMove {
        TorsionMove {
            species,
            rules: settings.rules(1.25),
            settings,
            container: None,
            frozen: None,
        }
    }

    #[test]
    fn the_z_matrix_and_the_dihedral_agree_on_sign() {
        for phi in [60.0_f64, 180.0, 300.0, 75.0] {
            let (x, _) = butane(phi);
            let measured = dihedral(x.view(), 0, 1, 2, 3);
            assert!(
                circular_gap(measured, phi.to_radians()) < 1e-9,
                "built {phi} degrees, measured {}",
                measured.to_degrees()
            );
        }
    }

    #[test]
    fn a_right_handed_turn_raises_the_dihedral_by_its_angle() {
        let (x, species) = butane(180.0);
        let topology = Topology::new(x.view(), &species, 1.25);
        let side: Vec<usize> = topology
            .side(2, 1)
            .into_iter()
            .filter(|&a| a != 2)
            .collect();
        let before = dihedral(x.view(), 0, 1, 2, 3);
        let mut y = x.clone();
        rotate_about_bond(&mut y, 1, 2, &side, 0.4);
        let after = dihedral(y.view(), 0, 1, 2, 3);
        assert!(circular_gap(after - before, 0.4) < 1e-12);
    }

    #[test]
    fn butane_rotates_its_central_bond_and_its_methyls_only_when_asked() {
        let (x, species) = butane(180.0);
        for (terminal, expected) in [
            (TerminalRotors::Exclude, vec![[1, 2]]),
            (TerminalRotors::Polar, vec![[1, 2]]),
            (TerminalRotors::Include, vec![[0, 1], [1, 2], [2, 3]]),
        ] {
            let found = rotatable_bonds(x.view(), &species, &rules(terminal, false));
            assert_eq!(axes(&found), expected, "{terminal:?}");
        }
        let central = &rotatable_bonds(x.view(), &species, &rules(TerminalRotors::Polar, false))[0];
        assert_eq!(central.preferred, &STAGGERED);
        assert_eq!(
            central.moving.len(),
            6,
            "the pivot's three hydrogens and its methyl"
        );
    }

    #[test]
    fn a_hydroxyl_is_polar_and_a_methyl_is_not() {
        let (x, species) = ethanol();
        let count = |terminal| rotatable_bonds(x.view(), &species, &rules(terminal, false));
        assert!(count(TerminalRotors::Exclude).is_empty());
        assert_eq!(axes(&count(TerminalRotors::Polar)), vec![[1, 2]]);
        assert_eq!(axes(&count(TerminalRotors::Include)), vec![[0, 1], [1, 2]]);
        let hydroxyl = &count(TerminalRotors::Polar)[0];
        assert_eq!((hydroxyl.fixed, hydroxyl.pivot), (1, 2));
        assert_eq!(hydroxyl.moving, vec![3]);
    }

    #[test]
    fn ring_bonds_never_rotate() {
        let (x, species) = cyclohexane_carbons();
        let topology = Topology::new(x.view(), &species, 1.25);
        assert_eq!(topology.bonds().len(), 6);
        assert!(
            rotatable_bonds(x.view(), &species, &rules(TerminalRotors::Include, true)).is_empty()
        );
    }

    #[test]
    fn the_amide_bond_rotates_only_when_asked() {
        let (x, species) = methylacetamide();
        let strict = rotatable_bonds(x.view(), &species, &rules(TerminalRotors::Include, false));
        assert_eq!(axes(&strict), vec![[0, 1], [3, 4]]);
        let loose = rotatable_bonds(x.view(), &species, &rules(TerminalRotors::Include, true));
        assert_eq!(axes(&loose), vec![[0, 1], [1, 3], [3, 4]]);
        let amide = loose
            .iter()
            .find(|r| axes(std::slice::from_ref(r)) == vec![[1, 3]])
            .unwrap();
        assert_eq!(amide.preferred, &PLANAR);
        assert!(
            rotatable_bonds(x.view(), &species, &rules(TerminalRotors::Polar, false)).is_empty()
        );
    }

    #[test]
    fn a_double_bond_is_not_a_rotor() {
        let (x, species) = butene();
        let found = rotatable_bonds(x.view(), &species, &rules(TerminalRotors::Include, true));
        assert_eq!(axes(&found), vec![[0, 1], [2, 3]]);
        assert!(found.iter().all(|r| r.preferred == SIXFOLD));
    }

    #[test]
    fn the_smaller_side_turns() {
        let (x, species) = alkane(&[180.0, 180.0, 180.0, 180.0, 180.0]);
        let n = species.len();
        for rotor in rotatable_bonds(x.view(), &species, &rules(TerminalRotors::Include, false)) {
            let turning = rotor.moving.len() + 1;
            assert!(turning <= n - turning, "{rotor:?} turns the larger side");
        }
    }

    fn bonds_and_angles(x: ArrayView1<f64>, topology: &Topology) -> (Vec<f64>, Vec<f64>) {
        let lengths = topology
            .bonds()
            .iter()
            .map(|&(a, b)| distance(x, a, b))
            .collect();
        let mut angles = Vec::new();
        for centre in 0..topology.len() {
            let nb = topology.neighbours(centre);
            for (i, &a) in nb.iter().enumerate() {
                for &c in &nb[i + 1..] {
                    angles.push(angle(x, a, centre, c));
                }
            }
        }
        (lengths, angles)
    }

    #[test]
    fn every_torsion_move_keeps_bond_lengths_and_angles() {
        let (x0, species) = alkane(&[180.0, 60.0, 180.0, 300.0, 180.0]);
        let topology = Topology::new(x0.view(), &species, 1.25);
        let (lengths, angles) = bonds_and_angles(x0.view(), &topology);
        let modes = [
            ("single", 0.0, 0.0),
            ("correlated", 1.0, 0.0),
            ("counter-rotation", 1.0, 1.0),
        ];
        for (label, correlated, counter) in modes {
            let settings = TorsionSettings {
                terminal: TerminalRotors::Include,
                correlated_fraction: correlated,
                counter_rotation_fraction: counter,
                ..TorsionSettings::recommended()
            };
            let kernel = mover(species.clone(), settings);
            let mut rng = StdRng::seed_from_u64(11);
            let mut x = x0.clone();
            let mut moved = 0;
            for _ in 0..300 {
                let y = kernel.propose(x.view(), &mut rng);
                if (&y - &x).iter().any(|d| d.abs() > 1e-9) {
                    moved += 1;
                }
                let after = Topology::new(y.view(), &species, 1.25);
                assert_eq!(
                    after.bonds(),
                    topology.bonds(),
                    "{label}: the move made or broke a bond"
                );
                let (l, a) = bonds_and_angles(y.view(), &topology);
                for (before, now) in lengths.iter().zip(&l) {
                    assert!(
                        (before - now).abs() < 1e-9,
                        "{label}: bond {before} became {now}"
                    );
                }
                for (before, now) in angles.iter().zip(&a) {
                    assert!(
                        (before - now).abs() < 1e-9,
                        "{label}: angle {before} became {now}"
                    );
                }
                x = y;
            }
            assert!(moved > 200, "{label}: only {moved} of 300 proposals moved");
            let backbone: Vec<f64> = (0..5)
                .map(|i| dihedral(x.view(), i, i + 1, i + 2, i + 3))
                .collect();
            let start: Vec<f64> = (0..5)
                .map(|i| dihedral(x0.view(), i, i + 1, i + 2, i + 3))
                .collect();
            assert!(
                backbone
                    .iter()
                    .zip(&start)
                    .any(|(a, b)| circular_gap(*a, *b) > 0.1),
                "{label}: no backbone dihedral changed"
            );
        }
    }

    #[test]
    fn a_preferred_draw_lands_in_another_staggered_well() {
        let settings = TorsionSettings {
            preferred_fraction: 1.0,
            preferred_jitter: 0.0,
            correlated_fraction: 0.0,
            clash_factor: 0.0,
            ..TorsionSettings::recommended()
        };
        let (x, species) = butane(180.0);
        let kernel = mover(species, settings);
        let mut rng = StdRng::seed_from_u64(3);
        let mut seen = [false; 2];
        for _ in 0..50 {
            let y = kernel.propose(x.view(), &mut rng);
            let phi = dihedral(y.view(), 0, 1, 2, 3);
            let gauche = [PI / 3.0, 5.0 * PI / 3.0].map(|w| circular_gap(phi, w) < 1e-9);
            assert!(
                gauche.iter().any(|&g| g),
                "anti went to {} degrees",
                phi.to_degrees()
            );
            seen[0] |= gauche[0];
            seen[1] |= gauche[1];
        }
        assert_eq!(seen, [true, true]);
    }

    #[test]
    fn a_uniform_draw_leaves_the_current_window() {
        let settings = TorsionSettings {
            preferred_fraction: 0.0,
            ..TorsionSettings::recommended()
        };
        let kernel = mover(vec![], settings.clone());
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..2_000 {
            let turn = kernel.draw_turn(1.0, &STAGGERED, &mut rng);
            assert!(circular_gap(turn, 0.0) >= settings.uniform_exclusion - 1e-12);
        }
    }

    #[test]
    fn an_accepted_draw_brings_no_pair_into_contact() {
        let (x0, species) = alkane(&[60.0, 60.0, 60.0, 60.0, 60.0]);
        let topology = Topology::new(x0.view(), &species, 1.25);
        let near = topology.within_two_bonds();
        let n = species.len();
        let settings = TorsionSettings {
            terminal: TerminalRotors::Include,
            ..TorsionSettings::recommended()
        };
        let factor = settings.clash_factor;
        let kernel = mover(species.clone(), settings);
        let mut rng = StdRng::seed_from_u64(19);
        let mut x = x0;
        for _ in 0..300 {
            let y = kernel.propose(x.view(), &mut rng);
            for i in 0..n {
                for j in (i + 1)..n {
                    if near[i * n + j] {
                        continue;
                    }
                    let limit = factor
                        * (van_der_waals_radius(species[i]) + van_der_waals_radius(species[j]));
                    let (before, after) = (distance(x.view(), i, j), distance(y.view(), i, j));
                    assert!(
                        after >= limit || after >= before - 1e-9,
                        "{i}-{j} moved in to {after}"
                    );
                }
            }
            x = y;
        }
    }

    #[test]
    fn a_draw_never_carries_an_atom_past_the_container() {
        let (x, species) = alkane(&[180.0, 180.0, 180.0, 180.0, 180.0]);
        let limit = extent(x.view());
        let mut kernel = mover(species, TorsionSettings::recommended());
        kernel.container = Some(limit);
        let mut rng = StdRng::seed_from_u64(23);
        let mut moved = 0;
        for _ in 0..200 {
            let y = kernel.propose(x.view(), &mut rng);
            assert!(extent(y.view()) <= limit + 1e-9);
            if y != x {
                moved += 1;
            }
        }
        assert!(moved > 0, "no draw fitted inside the starting extent");
    }

    #[test]
    fn frozen_atoms_stay_put() {
        let (x, species) = alkane(&[180.0, 60.0, 180.0, 300.0, 180.0]);
        // Carbon 0 and its three hydrogens.
        let held = [0, 8, 10, 12];
        let mut frozen = vec![false; species.len()];
        for atom in held {
            frozen[atom] = true;
        }
        let mut kernel = mover(
            species.clone(),
            TorsionSettings {
                terminal: TerminalRotors::Include,
                correlated_fraction: 0.5,
                counter_rotation_fraction: 0.5,
                ..TorsionSettings::recommended()
            },
        );
        kernel.frozen = Some(frozen.clone());
        let topology = Topology::new(x.view(), &species, 1.25);
        let rotors = topology.rotors(x.view(), &kernel.rules, Some(&frozen));
        assert!(!rotors.is_empty());
        for rotor in &rotors {
            assert!(rotor.moving.iter().all(|&a| !frozen[a]) && !frozen[rotor.pivot]);
        }
        let mut rng = StdRng::seed_from_u64(29);
        let mut y = x.clone();
        for _ in 0..200 {
            y = kernel.propose(y.view(), &mut rng);
            for atom in held {
                for k in 0..3 {
                    assert_eq!(y[3 * atom + k], x[3 * atom + k], "frozen atom {atom} moved");
                }
            }
        }
    }

    #[test]
    fn a_molecule_without_rotors_comes_back_unchanged() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0]);
        let kernel = mover(vec![8, 1, 1], TorsionSettings::recommended());
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(kernel.propose(x.view(), &mut rng), x);
    }

    fn joined(parts: &[(Array1<f64>, Vec<u32>, [f64; 3])]) -> (Array1<f64>, Vec<u32>) {
        let mut x = Vec::new();
        let mut species = Vec::new();
        for (coords, z, shift) in parts {
            x.extend(coords.iter().enumerate().map(|(i, v)| v + shift[i % 3]));
            species.extend_from_slice(z);
        }
        (Array1::from(x), species)
    }

    fn assert_same_geometry(x: ArrayView1<f64>, y: ArrayView1<f64>, topology: &Topology) {
        let (lengths, angles) = bonds_and_angles(x, topology);
        let (l, a) = bonds_and_angles(y, topology);
        for (before, now) in lengths.iter().zip(&l).chain(angles.iter().zip(&a)) {
            assert!((before - now).abs() < 1e-9, "{before} became {now}");
        }
    }

    fn geometry_changed(x: ArrayView1<f64>, y: ArrayView1<f64>, topology: &Topology) -> bool {
        let (lengths, angles) = bonds_and_angles(x, topology);
        let (l, a) = bonds_and_angles(y, topology);
        lengths
            .iter()
            .zip(&l)
            .chain(angles.iter().zip(&a))
            .any(|(before, now)| (before - now).abs() > 1e-6)
    }

    #[test]
    fn containment_keeps_the_bonds_of_a_rotor_trial() {
        use super::super::{Config, contain_trial, recentre};
        let settings = TorsionSettings::recommended();

        // A frozen frame and no active region: octane with one terminal
        // methyl held. The frame's clamp would bend the free end.
        let (x, species) = alkane(&[180.0, 180.0, 180.0, 180.0, 180.0]);
        let topology = Topology::new(x.view(), &species, 1.25);
        let all: Vec<usize> = (0..species.len()).collect();
        let mut cfg = Config::recommended_molecular(species.clone(), vec![all], 1.0);
        let mut frozen = vec![false; species.len()];
        for atom in [0, 8, 10, 12] {
            frozen[atom] = true;
        }
        cfg.frozen = Some(frozen.clone());
        let kernel = TorsionMove {
            species: species.clone(),
            rules: settings.rules(cfg.bond_tolerance),
            settings: settings.clone(),
            container: None,
            frozen: Some(frozen.clone()),
        };
        let mut rng = StdRng::seed_from_u64(37);
        let mut clamp_bends = 0;
        for _ in 0..100 {
            let trial = kernel.propose(x.view(), &mut rng);
            let mut held = trial.clone();
            contain_trial(&mut held, &cfg, Some(&frozen), true);
            assert_eq!(held, trial, "a rotor trial was clamped to the frozen frame");
            assert_same_geometry(x.view(), held.view(), &topology);
            let mut clamped = trial.clone();
            contain_trial(&mut clamped, &cfg, Some(&frozen), false);
            clamp_bends += usize::from(geometry_changed(x.view(), clamped.view(), &topology));
        }
        assert!(
            clamp_bends > 0,
            "the frame's clamp never reached the octane"
        );

        // Centred, with an ethanol beside the octane, in a container that
        // holds the pair: the arm redraws any turn that leaves the container,
        // so after recentring the point-wise clamp has nothing to pull in.
        // The preset's own container is narrower than an all-trans octane,
        // and the clamp bends a state wider than it on every arm.
        let (chain, chain_z) = alkane(&[180.0, 180.0, 180.0, 180.0, 180.0]);
        let (alcohol, alcohol_z) = ethanol();
        let (x, species) = joined(&[
            (chain, chain_z.clone(), [0.0; 3]),
            (alcohol, alcohol_z, [0.0, 7.0, 0.0]),
        ]);
        let topology = Topology::new(x.view(), &species, 1.25);
        let m = chain_z.len();
        let groups = vec![(0..m).collect(), (m..species.len()).collect()];
        let mut cfg = Config::recommended_molecular(species.clone(), groups, 1.0);
        assert!(cfg.frozen.is_none());
        cfg.container = cfg.container.max(extent(x.view()) + 0.5);
        let kernel = TorsionMove {
            species: species.clone(),
            rules: settings.rules(cfg.bond_tolerance),
            settings,
            container: Some(cfg.container),
            frozen: None,
        };
        let mut start = x.clone();
        recentre(&mut start, species.len());
        let mut rng = StdRng::seed_from_u64(41);
        let mut turned = 0;
        for _ in 0..100 {
            let mut trial = kernel.propose(x.view(), &mut rng);
            recentre(&mut trial, species.len());
            let mut held = trial.clone();
            contain_trial(&mut held, &cfg, None, true);
            assert_eq!(held, trial, "a centred rotor trial was clamped");
            assert_same_geometry(x.view(), held.view(), &topology);
            turned += usize::from(held != start);
        }
        assert!(turned > 0, "no trial turned a rotor");
    }

    #[test]
    fn a_counter_rotation_holds_one_end_and_translates_the_other() {
        let settings = TorsionSettings {
            terminal: TerminalRotors::Include,
            ..TorsionSettings::recommended()
        };
        let kernel = mover(vec![], settings.clone());
        let (x, species) = alkane(&[180.0, 180.0, 180.0, 180.0, 180.0]);
        let topology = Topology::new(x.view(), &species, 1.25);
        let rules = settings.rules(1.25);
        let pairs = counter_rotation_pairs(
            x.view(),
            &topology,
            &topology.rotors(x.view(), &rules, None),
        );
        assert_eq!(pairs.len(), 5, "{pairs:?}");
        let mut rng = StdRng::seed_from_u64(43);
        for pair in pairs {
            let [p0, p1, p2, p3] = pair;
            let behind = topology.side(p0, p1);
            let beyond = topology.side(p3, p2);
            assert!(beyond.len() <= behind.len());
            for _ in 0..30 {
                let mut y = x.clone();
                assert!(kernel.counter_rotate(&topology, &mut y, pair, &mut rng));
                assert_same_geometry(x.view(), y.view(), &topology);
                for &a in &behind {
                    for k in 0..3 {
                        assert_eq!(y[3 * a + k], x[3 * a + k], "atom {a} behind {pair:?} moved");
                    }
                }
                let shift: Vec<f64> = (0..3).map(|k| y[3 * p3 + k] - x[3 * p3 + k]).collect();
                for &a in &beyond {
                    for k in 0..3 {
                        let d = y[3 * a + k] - x[3 * a + k] - shift[k];
                        assert!(d.abs() < 1e-9, "atom {a} beyond {pair:?} turned by {d}");
                    }
                }
            }
        }

        // Along a gauche chain no two rotors one bond apart are parallel, so
        // there is no pair to counter-rotate.
        let (g, gz) = alkane(&[60.0, 60.0, 60.0, 60.0, 60.0]);
        let gauche = Topology::new(g.view(), &gz, 1.25);
        assert!(
            counter_rotation_pairs(g.view(), &gauche, &gauche.rotors(g.view(), &rules, None))
                .is_empty()
        );
    }

    #[test]
    fn a_correlated_draw_stays_within_one_molecule() {
        let (alcohol, z) = ethanol();
        let m = z.len();
        let (x, species) = joined(&[
            (alcohol.clone(), z.clone(), [0.0; 3]),
            (alcohol, z, [30.0, 0.0, 0.0]),
        ]);
        let settings = TorsionSettings {
            correlated_fraction: 1.0,
            ..TorsionSettings::recommended()
        };
        let kernel = mover(species.clone(), settings);
        let topology = Topology::new(x.view(), &species, 1.25);
        assert_eq!(
            axes(&topology.rotors(x.view(), &kernel.rules, None)),
            vec![[1, 2], [m + 1, m + 2]]
        );
        let hydroxyl = |y: ArrayView1<f64>, o: usize| dihedral(y, o, o + 1, o + 2, o + 3);
        let mut rng = StdRng::seed_from_u64(47);
        let mut moved = 0;
        for _ in 0..100 {
            let y = kernel.propose(x.view(), &mut rng);
            let turned = [0, m]
                .iter()
                .filter(|&&o| circular_gap(hydroxyl(y.view(), o), hydroxyl(x.view(), o)) > 1e-9)
                .count();
            assert!(turned <= 1, "one draw turned both ethanols");
            moved += turned;
        }
        assert!(moved > 90, "only {moved} of 100 draws turned a hydroxyl");
    }

    #[test]
    fn heavy_atom_rotors_follow_confab() {
        // 1,1,1-trifluorobutane: carbon 0's hydrogens become fluorines.
        let (x, mut species) = butane(180.0);
        for atom in [4, 6, 8] {
            species[atom] = 9;
        }
        let found = |terminal| {
            axes(&rotatable_bonds(
                x.view(),
                &species,
                &rules(terminal, false),
            ))
        };
        assert_eq!(found(TerminalRotors::HeavyAtom), vec![[0, 1], [1, 2]]);
        assert_eq!(found(TerminalRotors::Exclude), vec![[1, 2]]);
        assert_eq!(found(TerminalRotors::Polar), vec![[1, 2]]);
        assert_eq!(found(TerminalRotors::Include), vec![[0, 1], [1, 2], [2, 3]]);
        let (x, species) = ethanol();
        assert!(
            rotatable_bonds(x.view(), &species, &rules(TerminalRotors::HeavyAtom, false))
                .is_empty()
        );
    }

    #[test]
    fn recommended_molecular_adds_the_arm_to_flexible_molecules_only() {
        use super::super::{ClusterMove, Config};
        let (_, species) = alkane(&[180.0, 180.0]);
        let all: Vec<usize> = (0..species.len()).collect();
        let flexible = Config::recommended_molecular(species, vec![all], 1.0);
        assert_eq!(flexible.rotor, Some(TorsionSettings::recommended()));
        let names: Vec<String> = flexible
            .move_library
            .kernels(&flexible)
            .iter()
            .map(ClusterMove::name)
            .collect();
        assert!(names.iter().any(|n| n == "rotor"), "{names:?}");
        let json = flexible.resolved_json().unwrap();
        assert!(
            json.contains(r#""rotor":{"terminal":"polar","amide_like":false"#),
            "{json}"
        );

        let water = Config::recommended_molecular(
            vec![8, 1, 1, 8, 1, 1],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
            1.0,
        );
        assert!(water.rotor.is_none());
        assert!(
            water
                .move_library
                .kernels(&water)
                .iter()
                .all(|k| k.name() != "rotor")
        );
        assert!(!water.resolved_json().unwrap().contains(r#""rotor""#));

        let mut slab = flexible.clone();
        slab.active_region = Some((vec![0, 1], 1));
        assert!(
            slab.move_library
                .kernels(&slab)
                .iter()
                .all(|k| k.name() != "rotor")
        );
    }

    /// Harmonic springs on every pair one or two bonds apart, and a
    /// Lennard-Jones attraction between the rest, so a chain has folded
    /// minima for the rotor arm to find. Counts every evaluation.
    struct Chain {
        springs: Vec<(usize, usize, f64)>,
        contacts: Vec<(usize, usize)>,
        bounds: eindir_core::bounds::Bounds<f64>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Chain {
        fn new(x: ArrayView1<f64>, topology: &Topology) -> Self {
            let n = topology.len();
            let near = topology.within_two_bonds();
            let mut springs = Vec::new();
            let mut contacts = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    if near[i * n + j] {
                        springs.push((i, j, distance(x, i, j)));
                    } else {
                        contacts.push((i, j));
                    }
                }
            }
            Self {
                springs,
                contacts,
                bounds: eindir_core::bounds::Bounds::new(
                    Array1::from_elem(3 * n, -50.0),
                    Array1::from_elem(3 * n, 50.0),
                    0.0,
                ),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl eindir_core::Objective<f64> for Chain {
        fn dim(&self) -> usize {
            self.bounds.dims
        }
        fn bounds(&self) -> &eindir_core::bounds::Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).0
        }
    }

    impl eindir_core::gradient::Gradient<f64> for Chain {
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).1
        }
        fn dim(&self) -> usize {
            self.bounds.dims
        }
    }

    impl eindir_core::gradient::DifferentiableObjective<f64> for Chain {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut e = 0.0;
            let mut g = Array1::zeros(x.len());
            let mut pair = |i: usize, j: usize, value: f64, slope: f64, e: &mut f64| {
                let r = distance(x, i, j);
                *e += value;
                for k in 0..3 {
                    let u = (x[3 * i + k] - x[3 * j + k]) / r;
                    g[3 * i + k] += slope * u;
                    g[3 * j + k] -= slope * u;
                }
            };
            for &(i, j, r0) in &self.springs {
                let r = distance(x, i, j);
                pair(i, j, 50.0 * (r - r0).powi(2), 100.0 * (r - r0), &mut e);
            }
            for &(i, j) in &self.contacts {
                let r = distance(x, i, j);
                let s6 = (2.5 / r).powi(6);
                pair(
                    i,
                    j,
                    0.1 * (s6 * s6 - s6),
                    0.1 * (-12.0 * s6 * s6 + 6.0 * s6) / r,
                    &mut e,
                );
            }
            (e, g)
        }
    }

    #[test]
    fn every_energy_call_of_a_torsion_search_is_charged() {
        use super::super::{Config, Ledger};
        let (x, species) = alkane(&[180.0, 180.0, 180.0, 180.0, 180.0]);
        let topology = Topology::new(x.view(), &species, 1.25);
        let objective = Chain::new(x.view(), &topology);
        let all: Vec<usize> = (0..species.len()).collect();
        let cfg = Config::recommended_molecular(species, vec![all], 1.0);
        let budget = 1_200;
        let mut ledger = Ledger::new(budget);
        let (out, _) = crate::methods::cluster_search::search_from(
            &objective,
            &cfg,
            &mut ledger,
            x.view(),
            16_000,
        )
        .expect("search runs");
        let calls = objective.calls.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(calls, ledger.spent(), "an energy call went uncharged");
        assert!(calls <= budget);
        let torsion = out
            .arms
            .iter()
            .find(|arm| arm.0 == "rotor")
            .expect("rotor arm reported");
        assert!(
            torsion.1 > 0,
            "the allocator never drew the rotor arm: {:?}",
            out.arms
        );
    }
}
