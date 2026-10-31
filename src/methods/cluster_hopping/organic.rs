//! Bond-graph internal coordinates for any covalent molecule or cluster.
//!
//! A rotatable bond is a heavy-atom contact that is not a dead end. A ring
//! bond takes a small pucker. A fragment with no rotor is turned as a rigid
//! body about its own centre. Neither move follows a dihedral gradient, and
//! neither scores a proposal with an auxiliary potential. Proposals are
//! ranked by distance in the internal coordinates, or by a featomic SOAP
//! mean when that feature is built.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use rand::Rng;

use super::covalent_radius;

thread_local! {
    static PARTNER: std::cell::RefCell<Option<Vec<f64>>> = const { std::cell::RefCell::new(None) };
    static ARCHIVE: std::cell::RefCell<Vec<Vec<f64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct Internals {
    angles: Vec<f64>,
    distances: Vec<f64>,
}

/// Drop the partner geometry and the descriptor archive. One search, one memory.
pub fn reset_search_memory() {
    PARTNER.with(|slot| *slot.borrow_mut() = None);
    ARCHIVE.with(|slot| slot.borrow_mut().clear());
}

/// Geometry of another chain, copied in by the hop before a proposal.
pub fn offer_partner(coordinates: &[f64]) {
    PARTNER.with(|slot| *slot.borrow_mut() = Some(coordinates.to_vec()));
}

/// Forget the partner so a later single-chain proposal does not reuse it.
pub fn clear_partner() {
    PARTNER.with(|slot| *slot.borrow_mut() = None);
}

fn partner() -> Option<Vec<f64>> {
    PARTNER.with(|slot| slot.borrow().clone())
}

/// Sphere that holds a cluster of `n_fragments` rigid pieces.
///
/// The Lennard-Jones container is about one bond across for a handful of
/// waters, so a proposal is crushed through the hydrogen-bond network.
/// A diffuse linear chain is the other error: fragments drift apart and
/// the short budget is spent reassociating them. `2.5` bond lengths times
/// `n_frag^(1/3)` sits on the measured plateau for (H2O)4 and (H2O)6.
pub fn fragment_container(species: &[u32], n_fragments: usize) -> f64 {
    let bond = 2.0
        * species
            .iter()
            .copied()
            .map(covalent_radius)
            .fold(0.0_f64, f64::max);
    let n = (n_fragments.max(1) as f64).cbrt();
    2.5 * bond * n
}

/// Largest distance from the centre of mass.
pub fn com_radius(x: ArrayView1<f64>, n: usize) -> f64 {
    let count = n.max(1) as f64;
    let mut centre = [0.0; 3];
    for atom in 0..n {
        for axis in 0..3 {
            centre[axis] += x[3 * atom + axis];
        }
    }
    for value in &mut centre {
        *value /= count;
    }
    let mut radius = 0.0_f64;
    for atom in 0..n {
        let mut d2 = 0.0;
        for axis in 0..3 {
            let d = x[3 * atom + axis] - centre[axis];
            d2 += d * d;
        }
        radius = radius.max(d2.sqrt());
    }
    radius
}

#[derive(Clone, Copy)]
struct Rotor {
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    ring: bool,
}

fn neighbors(x: ArrayView1<f64>, species: &[u32], tolerance: f64) -> Vec<Vec<usize>> {
    let n = species.len().min(x.len() / 3);
    let mut nb = vec![Vec::new(); n];
    for a in 0..n {
        let ra = covalent_radius(species[a]);
        for b in (a + 1)..n {
            let cut = tolerance * (ra + covalent_radius(species[b]));
            if cut <= 0.0 {
                continue;
            }
            let mut d2 = 0.0;
            for axis in 0..3 {
                let d = x[3 * a + axis] - x[3 * b + axis];
                d2 += d * d;
            }
            if d2 < cut * cut {
                nb[a].push(b);
                nb[b].push(a);
            }
        }
    }
    nb
}

fn in_ring(nb: &[Vec<usize>], j: usize, k: usize) -> bool {
    let mut seen = vec![false; nb.len()];
    seen[j] = true;
    let mut layer = vec![j];
    for _ in 0..6 {
        let mut next = Vec::new();
        for &a in &layer {
            for &b in &nb[a] {
                if (a == j && b == k) || (a == k && b == j) {
                    continue;
                }
                if b == k {
                    return true;
                }
                if !seen[b] {
                    seen[b] = true;
                    next.push(b);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        layer = next;
    }
    false
}

fn anchor(nb: &[usize], banned: usize, species: &[u32]) -> Option<usize> {
    nb.iter()
        .copied()
        .find(|&atom| atom != banned && species.get(atom).copied().unwrap_or(0) > 1)
        .or_else(|| nb.iter().copied().find(|&atom| atom != banned))
}

fn rotors(nb: &[Vec<usize>], species: &[u32]) -> Vec<Rotor> {
    let mut found = Vec::new();
    for j in 0..nb.len() {
        if species.get(j).copied().unwrap_or(0) <= 1 {
            continue;
        }
        for &k in &nb[j] {
            if k < j || species.get(k).copied().unwrap_or(0) <= 1 {
                continue;
            }
            let Some(i) = anchor(&nb[j], k, species) else {
                continue;
            };
            let Some(l) = anchor(&nb[k], j, species) else {
                continue;
            };
            if species[i] <= 1 || species[l] <= 1 {
                continue;
            }
            found.push(Rotor {
                i,
                j,
                k,
                l,
                ring: in_ring(nb, j, k),
            });
        }
    }
    found
}

fn side_of(nb: &[Vec<usize>], j: usize, k: usize) -> Vec<usize> {
    let mut seen = vec![false; nb.len()];
    seen[j] = true;
    seen[k] = true;
    let mut side = vec![k];
    let mut layer = vec![k];
    while let Some(a) = layer.pop() {
        for &b in &nb[a] {
            if !seen[b] {
                seen[b] = true;
                side.push(b);
                layer.push(b);
            }
        }
    }
    side
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn point(x: &[f64], atom: usize) -> [f64; 3] {
    [x[3 * atom], x[3 * atom + 1], x[3 * atom + 2]]
}

fn dihedral(x: &[f64], rotor: Rotor) -> f64 {
    let b1 = sub(point(x, rotor.j), point(x, rotor.i));
    let b2 = sub(point(x, rotor.k), point(x, rotor.j));
    let b3 = sub(point(x, rotor.l), point(x, rotor.k));
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let n2len = dot(b2, b2).sqrt();
    if n2len < 1e-12 {
        return 0.0;
    }
    let b2u = [b2[0] / n2len, b2[1] / n2len, b2[2] / n2len];
    let m1 = cross(n1, b2u);
    dot(m1, n2).atan2(dot(n1, n2))
}

fn wrap(angle: f64) -> f64 {
    let mut a = angle;
    while a > std::f64::consts::PI {
        a -= std::f64::consts::TAU;
    }
    while a < -std::f64::consts::PI {
        a += std::f64::consts::TAU;
    }
    a
}

fn rotate_side(y: &mut [f64], j: usize, k: usize, side: &[usize], angle: f64) {
    let origin = point(y, j);
    let mut axis = sub(point(y, k), origin);
    let norm = dot(axis, axis).sqrt();
    if norm < 1e-8 {
        return;
    }
    axis = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let (s, c) = angle.sin_cos();
    for &atom in side {
        if atom == j || atom == k {
            continue;
        }
        let p = sub(point(y, atom), origin);
        let proj = dot(p, axis);
        let cr = cross(axis, p);
        for t in 0..3 {
            y[3 * atom + t] = origin[t] + p[t] * c + cr[t] * s + axis[t] * proj * (1.0 - c);
        }
    }
}

fn centroid(x: &[f64], atoms: &[usize]) -> [f64; 3] {
    let mut c = [0.0; 3];
    let mut n = 0usize;
    for &atom in atoms {
        if 3 * atom + 2 < x.len() {
            for t in 0..3 {
                c[t] += x[3 * atom + t];
            }
            n += 1;
        }
    }
    let scale = (n.max(1) as f64).recip();
    [c[0] * scale, c[1] * scale, c[2] * scale]
}

fn spin_lists<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    groups: &[Vec<usize>],
    rng: &mut R,
) -> Array1<f64> {
    let mut y = x.to_owned();
    if groups.is_empty() {
        return y;
    }
    let g = rng.random_range(0..groups.len());
    let atoms = &groups[g];
    let gc = centroid(x.as_slice().unwrap_or(&[]), atoms);
    let axis = loop {
        let v = [
            rng.random::<f64>() * 2.0 - 1.0,
            rng.random::<f64>() * 2.0 - 1.0,
            rng.random::<f64>() * 2.0 - 1.0,
        ];
        let n2 = dot(v, v);
        if n2 > 1e-6 && n2 <= 1.0 {
            let n = n2.sqrt();
            break [v[0] / n, v[1] / n, v[2] / n];
        }
    };
    let angle = rng.random_range(1.2..std::f64::consts::PI);
    let (s, c) = angle.sin_cos();
    let shift = [
        rng.random_range(-0.45..0.45),
        rng.random_range(-0.45..0.45),
        rng.random_range(-0.45..0.45),
    ];
    let slice = y.as_slice_mut().expect("coordinates are contiguous");
    for &atom in atoms {
        if 3 * atom + 2 >= slice.len() {
            continue;
        }
        let p = sub(point(slice, atom), gc);
        let proj = dot(p, axis);
        let cr = cross(axis, p);
        for t in 0..3 {
            slice[3 * atom + t] =
                gc[t] + shift[t] + p[t] * c + cr[t] * s + axis[t] * proj * (1.0 - c);
        }
    }
    y
}

fn copy_pose(dst: &mut [f64], src: &[f64], atoms: &[usize]) {
    let from = centroid(src, atoms);
    let to = centroid(dst, atoms);
    for &atom in atoms {
        if 3 * atom + 2 >= dst.len() || 3 * atom + 2 >= src.len() {
            continue;
        }
        for t in 0..3 {
            dst[3 * atom + t] = to[t] + (src[3 * atom + t] - from[t]);
        }
    }
}

fn components(nb: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; nb.len()];
    let mut groups = Vec::new();
    for start in 0..nb.len() {
        if seen[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(a) = stack.pop() {
            comp.push(a);
            for &b in &nb[a] {
                if !seen[b] {
                    seen[b] = true;
                    stack.push(b);
                }
            }
        }
        comp.sort_unstable();
        if comp.len() >= 3 {
            groups.push(comp);
        }
    }
    groups
}

fn fragment_lists(groups: Option<&[Vec<usize>]>, nb: &[Vec<usize>]) -> Vec<Vec<usize>> {
    if let Some(groups) = groups
        && groups.len() >= 2
    {
        return groups.to_vec();
    }
    components(nb)
}

fn one_internal<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
    groups: Option<&[Vec<usize>]>,
    rng: &mut R,
) -> Array1<f64> {
    let nb = neighbors(x, species, tolerance);
    let rotors = rotors(&nb, species);
    let frags = fragment_lists(groups, &nb);
    let multi = frags.len() >= 2;
    let use_torsion = !rotors.is_empty() && (!multi || rng.random::<f64>() < 0.7);
    if use_torsion {
        let rotor = rotors[rng.random_range(0..rotors.len())];
        let angle = if rotor.ring {
            let sign = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 };
            sign * rng.random_range(0.8..1.3)
        } else {
            const STEPS: [f64; 3] = [
                std::f64::consts::TAU / 3.0,
                std::f64::consts::PI,
                -std::f64::consts::TAU / 3.0,
            ];
            STEPS[rng.random_range(0..3)] + rng.random_range(-0.2..0.2)
        };
        let side = side_of(&nb, rotor.j, rotor.k);
        let mut y = x.to_owned();
        rotate_side(
            y.as_slice_mut().expect("coordinates are contiguous"),
            rotor.j,
            rotor.k,
            &side,
            angle,
        );
        return y;
    }
    if multi {
        return spin_lists(x, &frags, rng);
    }
    x.to_owned()
}

fn internals_of(
    x: &[f64],
    species: &[u32],
    tolerance: f64,
    groups: Option<&[Vec<usize>]>,
) -> Internals {
    let view = ArrayView1::from(x);
    let nb = neighbors(view, species, tolerance);
    let mut angles = Vec::new();
    for rotor in rotors(&nb, species) {
        angles.push(dihedral(x, rotor));
    }
    let frags = fragment_lists(groups, &nb);
    let mut distances = Vec::new();
    for a in 0..frags.len() {
        for b in (a + 1)..frags.len() {
            let ca = centroid(x, &frags[a]);
            let cb = centroid(x, &frags[b]);
            distances.push(dot(sub(ca, cb), sub(ca, cb)).sqrt());
        }
    }
    Internals { angles, distances }
}

fn internal_distance(left: &Internals, right: &Internals) -> f64 {
    let mut acc = 0.0;
    for i in 0..left.angles.len().min(right.angles.len()) {
        let d = wrap(left.angles[i] - right.angles[i]);
        acc += d * d;
    }
    for i in 0..left.distances.len().min(right.distances.len()) {
        let d = left.distances[i] - right.distances[i];
        acc += d * d;
    }
    acc.sqrt()
}

fn descriptor_distance(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
    groups: Option<&[Vec<usize>]>,
) -> f64 {
    #[cfg(feature = "featomic")]
    if let Some(distance) = crate::featomic_hop::conformer_soap_distance(x, y, species) {
        return distance;
    }
    let left = internals_of(x.as_slice().unwrap_or(&[]), species, tolerance, groups);
    let right = internals_of(y.as_slice().unwrap_or(&[]), species, tolerance, groups);
    internal_distance(&left, &right)
}

fn remember(coordinates: &[f64]) {
    ARCHIVE.with(|slot| {
        let mut archive = slot.borrow_mut();
        archive.push(coordinates.to_vec());
        if archive.len() > 24 {
            archive.remove(0);
        }
    });
}

fn farthest<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
    groups: Option<&[Vec<usize>]>,
    rng: &mut R,
    draw: impl Fn(&mut R) -> Array1<f64>,
) -> Array1<f64> {
    let mut best = x.to_owned();
    let mut best_score = -1.0;
    for _ in 0..4 {
        let cand = draw(rng);
        let mut score = descriptor_distance(x, cand.view(), species, tolerance, groups);
        ARCHIVE.with(|slot| {
            for prior in slot.borrow().iter() {
                let prior_view = ArrayView1::from(prior.as_slice());
                score = score.min(descriptor_distance(
                    cand.view(),
                    prior_view,
                    species,
                    tolerance,
                    groups,
                ));
            }
        });
        if score > best_score {
            best_score = score;
            best = cand;
        }
    }
    remember(best.as_slice().unwrap_or(&[]));
    best
}

/// One torsion or, when the graph has no rotor, one fragment turn.
pub fn propose_torsion<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
    groups: Option<&[Vec<usize>]>,
    rng: &mut R,
) -> Array1<f64> {
    farthest(x, species, tolerance, groups, rng, |rng| {
        one_internal(x, species, tolerance, groups, rng)
    })
}

/// Large rigid reorientation of one fragment.
pub fn propose_fragment<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
    groups: Option<&[Vec<usize>]>,
    rng: &mut R,
) -> Array1<f64> {
    farthest(x, species, tolerance, groups, rng, |rng| {
        let nb = neighbors(x, species, tolerance);
        let frags = fragment_lists(groups, &nb);
        if frags.len() >= 2 {
            spin_lists(x, &frags, rng)
        } else {
            one_internal(x, species, tolerance, groups, rng)
        }
    })
}

/// Copy a random subset of dihedrals and fragment orientations from the partner chain.
pub fn propose_crossover<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
    groups: Option<&[Vec<usize>]>,
    rng: &mut R,
) -> Array1<f64> {
    let Some(other) = partner() else {
        return propose_torsion(x, species, tolerance, groups, rng);
    };
    if other.len() != x.len() {
        return propose_torsion(x, species, tolerance, groups, rng);
    }
    let nb = neighbors(x, species, tolerance);
    let rotors = rotors(&nb, species);
    let mut y = x.to_owned();
    let mut changed = false;
    {
        let slice = y.as_slice_mut().expect("coordinates are contiguous");
        for rotor in rotors {
            if rng.random::<f64>() < 0.5 {
                let delta = wrap(dihedral(&other, rotor) - dihedral(slice, rotor));
                if delta.abs() > 1e-3 {
                    let side = side_of(&nb, rotor.j, rotor.k);
                    rotate_side(slice, rotor.j, rotor.k, &side, delta);
                    changed = true;
                }
            }
        }
        let frags = fragment_lists(groups, &nb);
        if frags.len() >= 2 {
            for frag in &frags {
                if rng.random::<f64>() < 0.5 {
                    copy_pose(slice, &other, frag);
                    changed = true;
                }
            }
        }
    }
    if !changed {
        return propose_fragment(x, species, tolerance, groups, rng);
    }
    y
}

/// A second chain starts from a bond-preserving kick of the quenched first chain.
pub fn kick<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
    groups: Option<&[Vec<usize>]>,
    rng: &mut R,
) -> Array1<f64> {
    one_internal(x, species, tolerance, groups, rng)
}

/// A bond the torsion kick turns, and the smaller of its two sides.
struct RotatableBond {
    j: usize,
    k: usize,
    side: Vec<usize>,
}

/// Two or more neighbours besides `other`, all hydrogen: a methyl or amino
/// top, whose turn is a symmetry of the molecule.
fn terminal_top(nb: &[Vec<usize>], species: &[u32], atom: usize, other: usize) -> bool {
    nb[atom].len() >= 3
        && nb[atom]
            .iter()
            .filter(|&&b| b != other)
            .all(|&b| species.get(b).copied().unwrap_or(0) <= 1)
}

/// Heavy-atom bonds outside rings whose ends each carry another atom and
/// neither of which is a terminal top.
fn rotatable_bonds(nb: &[Vec<usize>], species: &[u32]) -> Vec<RotatableBond> {
    let heavy = |atom: usize| species.get(atom).copied().unwrap_or(0) > 1;
    let mut found = Vec::new();
    for j in 0..nb.len() {
        if !heavy(j) || nb[j].len() < 2 {
            continue;
        }
        for &k in &nb[j] {
            if k < j || !heavy(k) || nb[k].len() < 2 {
                continue;
            }
            if terminal_top(nb, species, j, k)
                || terminal_top(nb, species, k, j)
                || in_ring(nb, j, k)
            {
                continue;
            }
            let far = side_of(nb, j, k);
            let near = side_of(nb, k, j);
            let side = if far.len() <= near.len() { far } else { near };
            found.push(RotatableBond { j, k, side });
        }
    }
    found
}

/// Rotatable bonds of `x`'s covalent graph, as [`propose_torsion_kick`]
/// counts them.
pub fn rotatable_bond_count(x: ArrayView1<f64>, species: &[u32], tolerance: f64) -> usize {
    rotatable_bonds(&neighbors(x, species, tolerance), species).len()
}

/// Draws before [`propose_torsion_kick`] keeps a kick that changed the graph.
const TORSION_KICK_DRAWS: usize = 8;

/// Turn between one and half of the rotatable bonds, each by a third of a
/// turn to a half turn either way, moving the smaller side of each bond.
///
/// A kick that brings two atoms within bonding distance changes the
/// covalent graph and is redrawn, up to [`TORSION_KICK_DRAWS`] times, after
/// which the last draw is kept. A graph with no rotatable bond is returned
/// unchanged.
pub fn propose_torsion_kick<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
    rng: &mut R,
) -> Array1<f64> {
    let nb = neighbors(x, species, tolerance);
    let bonds = rotatable_bonds(&nb, species);
    let mut kicked = x.to_owned();
    if bonds.is_empty() {
        return kicked;
    }
    let most = (bonds.len() / 2).max(1);
    let mut order: Vec<usize> = (0..bonds.len()).collect();
    for _ in 0..TORSION_KICK_DRAWS {
        let m = rng.random_range(1..=most);
        for i in 0..m {
            let pick = rng.random_range(i..order.len());
            order.swap(i, pick);
        }
        let mut y = x.to_owned();
        {
            let slice = y.as_slice_mut().expect("coordinates are contiguous");
            for &b in &order[..m] {
                let bond = &bonds[b];
                let magnitude =
                    rng.random_range(std::f64::consts::FRAC_PI_3..=std::f64::consts::PI);
                let angle = if rng.random_bool(0.5) {
                    magnitude
                } else {
                    -magnitude
                };
                rotate_side(slice, bond.j, bond.k, &bond.side, angle);
            }
        }
        let kept_graph = neighbors(y.view(), species, tolerance) == nb;
        kicked = y;
        if kept_graph {
            break;
        }
    }
    kicked
}

/// The covalent graph of `x` under the bond-matrix rule: for each atom, the
/// ascending indices of the atoms it bonds to. Two structures of the same
/// atoms have the same graph when these lists are equal; no isomorphism is
/// taken, so a hydrogen that moves to another molecule changes the graph.
pub(crate) fn bond_graph(x: ArrayView1<f64>, species: &[u32], tolerance: f64) -> Vec<Vec<usize>> {
    neighbors(x, species, tolerance)
}

/// The covalent graph a graph-keeping search must keep, taken from the
/// structure the caller started it from, and the quenches that did not.
#[derive(Debug)]
pub(crate) struct GraphReference {
    graph: Vec<Vec<usize>>,
    changes: AtomicUsize,
}

impl GraphReference {
    pub(crate) fn of(x: ArrayView1<f64>, species: &[u32], tolerance: f64) -> Self {
        Self {
            graph: bond_graph(x, species, tolerance),
            changes: AtomicUsize::new(0),
        }
    }

    /// Whether `x` has the reference graph. A structure that does not is
    /// counted.
    pub(crate) fn kept(&self, x: ArrayView1<f64>, species: &[u32], tolerance: f64) -> bool {
        let kept = bond_graph(x, species, tolerance) == self.graph;
        if !kept {
            self.changes.fetch_add(1, Ordering::Relaxed);
        }
        kept
    }

    /// Structures [`GraphReference::kept`] has turned away.
    pub(crate) fn changes(&self) -> usize {
        self.changes.load(Ordering::Relaxed)
    }
}

/// Ratio of successive factors in [`separate_start`].
const SEPARATION_RATIO: f64 = 1.05;
/// Factors tried before [`separate_start`] leaves a start as it is.
const SEPARATION_STEPS: i32 = 60;

/// What [`Config::separate_start_molecules`] did to a start whose declared
/// molecules were in covalent contact.
///
/// [`Config::separate_start_molecules`]: super::Config::separate_start_molecules
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StartSeparation {
    /// Every group moved by `(factor - 1)` times its centroid's offset from
    /// the mean centroid.
    Separated {
        /// The exponent k.
        steps: usize,
        /// 1.05^k.
        factor: f64,
    },
    /// No factor up to 1.05^60 removed every contact. The start is unchanged.
    Unresolved,
}

/// Each atom's group, its first declared group or a group of its own,
/// numbered from zero over the groups that hold an atom, and their count.
fn group_index(groups: &[Vec<usize>], n: usize) -> (Vec<usize>, usize) {
    let mut declared = vec![None; n];
    for (g, atoms) in groups.iter().enumerate() {
        for &atom in atoms {
            if atom < n && declared[atom].is_none() {
                declared[atom] = Some(g);
            }
        }
    }
    let mut compact: Vec<Option<usize>> = vec![None; groups.len()];
    let mut count = 0usize;
    let mut index = Vec::with_capacity(n);
    for d in declared {
        let known = d.and_then(|g| compact[g]);
        let id = known.unwrap_or(count);
        if known.is_none() {
            count += 1;
            if let Some(g) = d {
                compact[g] = Some(id);
            }
        }
        index.push(id);
    }
    (index, count)
}

fn groups_bonded(x: ArrayView1<f64>, species: &[u32], tolerance: f64, group: &[usize]) -> bool {
    neighbors(x, species, tolerance)
        .iter()
        .enumerate()
        .any(|(a, bonded)| bonded.iter().any(|&b| group[a] != group[b]))
}

/// The start [`Config::separate_start_molecules`] gives a search, and what it
/// did, when the setting acts and two atoms of different groups bond; `None`
/// otherwise. The state is `None` when no factor worked. Makes no call.
///
/// [`Config::separate_start_molecules`]: super::Config::separate_start_molecules
pub fn separate_start(
    cfg: &super::Config,
    start: ArrayView1<f64>,
) -> Option<(StartSeparation, Option<Array1<f64>>)> {
    let species = cfg.species.as_deref()?;
    let groups = cfg.move_library.declared_groups()?;
    let frozen = cfg.frozen.as_deref().is_some_and(|f| f.iter().any(|&v| v));
    if !cfg.separate_start_molecules
        || !cfg.keep_bond_graph
        || frozen
        || groups.len() < 2
        || start.len() != 3 * species.len()
    {
        return None;
    }
    let n = species.len();
    let (group, count) = group_index(groups, n);
    if !groups_bonded(start, species, cfg.bond_tolerance, &group) {
        return None;
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (a, &g) in group.iter().enumerate() {
        members[g].push(a);
    }
    let x = start.to_vec();
    let centroids: Vec<[f64; 3]> = members.iter().map(|atoms| centroid(&x, atoms)).collect();
    let mut mean = [0.0; 3];
    for c in &centroids {
        for t in 0..3 {
            mean[t] += c[t] / count as f64;
        }
    }
    let offsets: Vec<[f64; 3]> = centroids.iter().map(|&c| sub(c, mean)).collect();
    for steps in 1..=SEPARATION_STEPS {
        let factor = SEPARATION_RATIO.powi(steps);
        let moved = Array1::from_shape_fn(3 * n, |c| {
            x[c] + (factor - 1.0) * offsets[group[c / 3]][c % 3]
        });
        if !groups_bonded(moved.view(), species, cfg.bond_tolerance, &group) {
            let steps = steps as usize;
            return Some((StartSeparation::Separated { steps, factor }, Some(moved)));
        }
    }
    Some((StartSeparation::Unresolved, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn butane() -> (Vec<f64>, Vec<u32>) {
        // Anti carbon chain in the xy plane, with one hydrogen on the last carbon
        // so the dihedral has a reporter that leaves the plane.
        let coords = vec![
            0.0, 0.0, 0.0, // C0
            1.53, 0.0, 0.0, // C1
            2.54, 1.26, 0.0, // C2
            4.07, 1.26, 0.0, // C3
            4.60, 2.20, 0.30, // H on C3, not used as a rotor anchor
        ];
        let species = vec![6, 6, 6, 6, 1];
        (coords, species)
    }

    #[test]
    fn a_torsion_keeps_the_axis_length_and_moves_the_end() {
        let (coords, species) = butane();
        let x = Array1::from(coords);
        let nb = neighbors(x.view(), &species, 1.25);
        let found = rotors(&nb, &species);
        assert!(!found.is_empty(), "butane has a heavy-atom rotor");
        let rotor = found[0];
        let before = {
            let d = x[3 * rotor.j] - x[3 * rotor.k];
            let e = x[3 * rotor.j + 1] - x[3 * rotor.k + 1];
            let f = x[3 * rotor.j + 2] - x[3 * rotor.k + 2];
            (d * d + e * e + f * f).sqrt()
        };
        let side = side_of(&nb, rotor.j, rotor.k);
        let mut y = x.to_vec();
        let end_before = y[3 * rotor.l + 2];
        rotate_side(&mut y, rotor.j, rotor.k, &side, std::f64::consts::TAU / 3.0);
        let after = {
            let d = y[3 * rotor.j] - y[3 * rotor.k];
            let e = y[3 * rotor.j + 1] - y[3 * rotor.k + 1];
            let f = y[3 * rotor.j + 2] - y[3 * rotor.k + 2];
            (d * d + e * e + f * f).sqrt()
        };
        assert!((before - after).abs() < 1e-9);
        assert!(
            (y[3 * rotor.l + 2] - end_before).abs() > 0.2,
            "the far atom did not leave its old height"
        );
    }

    #[test]
    fn the_chain_container_holds_an_alkane_and_leaves_lj_alone() {
        let species = vec![6u32; 20];
        let groups = vec![(0..20).collect::<Vec<_>>()];
        let alkane = super::super::Config::recommended_molecular(species, groups, 1.0);
        let cluster = super::super::Config::recommended(20);
        assert!(!alkane.internal_moves);
        assert!(!cluster.internal_moves);
        // One covalent molecule keeps the preset sphere. The hop then
        // widens it to the geometry in hand, rather than to a chain of
        // every atom.
        let plain =
            super::super::Config::for_molecular(vec![6u32; 20], vec![(0..20).collect()], 1.0);
        assert!((alkane.container - plain.container).abs() < 1e-9);
        let waters = vec![8u32, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1];
        let water_groups = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9, 10, 11]];
        let water = super::super::Config::recommended_molecular(waters, water_groups, 1.0);
        assert!(water.container > cluster.container);
        assert!(water.container < 8.0, "water container {}", water.container);
        let names: Vec<String> = cluster
            .move_library
            .kernels(&cluster)
            .iter()
            .map(|kernel| kernel.name())
            .collect();
        assert!(
            names
                .iter()
                .all(|name| name != "torsion" && name != "frag" && name != "icross")
        );
    }

    /// A zigzag heavy-atom chain in the xy plane. Every carbon is filled to
    /// four bonds with hydrogens; a terminal oxygen carries one.
    fn saturated_chain(heavy: &[u32]) -> (Vec<f64>, Vec<u32>) {
        let mut coords = Vec::new();
        let mut species = Vec::new();
        let last = heavy.len() - 1;
        for (i, &z) in heavy.iter().enumerate() {
            let (x, y) = (1.27 * i as f64, if i % 2 == 1 { 0.84 } else { 0.0 });
            let out = if i % 2 == 1 { 1.0 } else { -1.0 };
            coords.extend([x, y, 0.0]);
            species.push(z);
            let mut hydrogens = Vec::new();
            if z == 6 {
                hydrogens.push([x, y + 0.51 * out, 0.89]);
                hydrogens.push([x, y + 0.51 * out, -0.89]);
                if i == 0 {
                    hydrogens.push([x - 1.03, y - 0.3 * out, 0.0]);
                }
                if i == last {
                    hydrogens.push([x + 1.03, y - 0.3 * out, 0.0]);
                }
            } else if i == last {
                hydrogens.push([x + 0.91, y - 0.3 * out, 0.0]);
            }
            for h in hydrogens {
                coords.extend(h);
                species.push(1);
            }
        }
        (coords, species)
    }

    fn water() -> (Vec<f64>, Vec<u32>) {
        (
            vec![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0],
            vec![8, 1, 1],
        )
    }

    #[test]
    fn methyl_tops_do_not_count_and_a_hydroxyl_turn_does() {
        let (hexane, hexane_species) = saturated_chain(&[6; 6]);
        let (ethanol, ethanol_species) = saturated_chain(&[6, 6, 8]);
        let (water, water_species) = water();
        let count =
            |x: &[f64], species: &[u32]| rotatable_bond_count(ArrayView1::from(x), species, 1.25);
        assert_eq!(hexane_species.len(), 20);
        assert_eq!(ethanol_species.len(), 9);
        assert_eq!(count(&hexane, &hexane_species), 3);
        assert_eq!(count(&ethanol, &ethanol_species), 1);
        assert_eq!(count(&water, &water_species), 0);
    }

    #[test]
    fn a_torsion_kick_keeps_the_bonds_and_turns_a_dihedral() {
        let (coords, species) = saturated_chain(&[6; 6]);
        let x = Array1::from(coords);
        let nb = neighbors(x.view(), &species, 1.25);
        let backbone: Vec<Rotor> = rotors(&nb, &species);
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..20 {
            let y = propose_torsion_kick(x.view(), &species, 1.25, &mut rng);
            assert_eq!(neighbors(y.view(), &species, 1.25), nb);
            for a in 0..species.len() {
                for &b in &nb[a] {
                    let length = |v: &[f64]| {
                        let d = sub(point(v, a), point(v, b));
                        dot(d, d).sqrt()
                    };
                    let before = length(x.as_slice().expect("contiguous"));
                    let after = length(y.as_slice().expect("contiguous"));
                    assert!((before - after).abs() < 1e-9, "bond {a}-{b}");
                }
            }
            let turned = backbone.iter().any(|&rotor| {
                let before = dihedral(x.as_slice().expect("contiguous"), rotor);
                let after = dihedral(y.as_slice().expect("contiguous"), rotor);
                wrap(after - before).abs() > std::f64::consts::FRAC_PI_3 - 1e-9
            });
            assert!(turned, "no backbone dihedral turned by a third of a turn");
        }
    }

    #[test]
    fn the_kick_arm_needs_a_rotatable_bond_and_the_flag() {
        let (hexane, hexane_species) = saturated_chain(&[6; 6]);
        let (waters, water_species) = water();
        let config = |species: &[u32], kicks: bool| {
            let n = species.len();
            let cfg = super::super::Config::recommended_molecular(
                species.to_vec(),
                vec![(0..n).collect()],
                1.0,
            );
            if kicks { cfg.with_torsion_kicks() } else { cfg }
        };
        let kicks = |cfg: &super::super::Config, x: &[f64]| {
            super::super::kicks_torsions(cfg, ArrayView1::from(x))
        };
        let kicked = config(&hexane_species, true);
        assert!(kicks(&kicked, &hexane));
        assert_eq!(
            super::super::torsion_kick_arm(&kicked)
                .map(|arm| arm.name())
                .as_deref(),
            Some("tkick")
        );
        assert!(!kicks(&config(&hexane_species, false), &hexane));
        assert!(!kicks(&config(&water_species, true), &waters));
        let plain = config(&hexane_species, false);
        let names: Vec<String> = plain
            .move_library
            .kernels(&plain)
            .iter()
            .map(|kernel| kernel.name())
            .collect();
        assert!(names.iter().all(|name| name != "tkick"));
    }

    /// Harmonic springs on the bonds of a reference geometry, and a soft
    /// wall between atoms closer than 2 Angstrom that are not bonded.
    struct Springs {
        bonds: Vec<(usize, usize, f64)>,
        bonded: Vec<Vec<bool>>,
        bounds: eindir_core::Bounds<f64>,
    }

    impl Springs {
        fn new(x: &[f64], species: &[u32]) -> Self {
            let n = species.len();
            let nb = neighbors(ArrayView1::from(x), species, 1.25);
            let mut bonded = vec![vec![false; n]; n];
            let mut bonds = Vec::new();
            for a in 0..n {
                for &b in &nb[a] {
                    bonded[a][b] = true;
                    if a < b {
                        let d = sub(point(x, a), point(x, b));
                        bonds.push((a, b, dot(d, d).sqrt()));
                    }
                }
            }
            let bounds = eindir_core::Bounds::new(
                Array1::from_elem(3 * n, -50.0),
                Array1::from_elem(3 * n, 50.0),
                0.0,
            );
            Self {
                bonds,
                bonded,
                bounds,
            }
        }

        fn pair(x: &[f64], g: &mut [f64], a: usize, b: usize, slope: f64, r: f64) {
            for t in 0..3 {
                let unit = (x[3 * a + t] - x[3 * b + t]) / r.max(1e-12);
                g[3 * a + t] += slope * unit;
                g[3 * b + t] -= slope * unit;
            }
        }
    }

    impl eindir_core::Objective<f64> for Springs {
        fn dim(&self) -> usize {
            3 * self.bonded.len()
        }
        fn bounds(&self) -> &eindir_core::Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).0
        }
    }

    impl eindir_core::Gradient<f64> for Springs {
        fn dim(&self) -> usize {
            3 * self.bonded.len()
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).1
        }
    }

    impl eindir_core::gradient::DifferentiableObjective<f64> for Springs {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let x = x.to_vec();
            let n = self.bonded.len();
            let mut g = vec![0.0; 3 * n];
            let mut e = 0.0;
            for &(a, b, r0) in &self.bonds {
                let d = sub(point(&x, a), point(&x, b));
                let r = dot(d, d).sqrt();
                e += (r - r0) * (r - r0);
                Self::pair(&x, &mut g, a, b, 2.0 * (r - r0), r);
            }
            for a in 0..n {
                for b in (a + 1)..n {
                    if self.bonded[a][b] {
                        continue;
                    }
                    let d = sub(point(&x, a), point(&x, b));
                    let r = dot(d, d).sqrt();
                    if r < 2.0 {
                        e += (2.0 - r) * (2.0 - r);
                        Self::pair(&x, &mut g, a, b, -2.0 * (2.0 - r), r);
                    }
                }
            }
            (e, Array1::from(g))
        }
    }

    /// The molecular library rebuilds its arms from the bond graph on every
    /// hop; the kick arm has to survive that, or the allocator draws an arm
    /// the list no longer holds.
    #[test]
    fn a_kicking_molecular_run_draws_the_arm_on_every_rebuild() {
        let (coords, species) = saturated_chain(&[6; 6]);
        let springs = Springs::new(&coords, &species);
        let n = species.len();
        let cfg = super::super::Config::recommended_molecular(
            species.clone(),
            vec![(0..n).collect()],
            1.0,
        )
        .with_torsion_kicks();
        let mut ledger = super::super::Ledger::new(600);
        let (out, _) = crate::methods::cluster_search::search_from(
            &springs,
            &cfg,
            &mut ledger,
            ArrayView1::from(&coords),
            1,
        )
        .expect("finite start");
        let draws = out
            .arms
            .iter()
            .find(|arm| arm.0 == "tkick")
            .map(|arm| arm.1)
            .expect("the run carries the kick arm");
        assert!(draws > 0, "the allocator never drew tkick");
        assert_eq!(out.charged, 600);
    }

    #[test]
    fn crossover_without_a_partner_still_moves_a_fragment() {
        reset_search_memory();
        let mut waters = Vec::new();
        let mut species = Vec::new();
        for t in 0..2 {
            let x = t as f64 * 3.0;
            waters.extend([x, 0.0, 0.0, x + 0.96, 0.0, 0.0, x - 0.24, 0.93, 0.0]);
            species.extend([8, 1, 1]);
        }
        let x = Array1::from(waters);
        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let mut rng = StdRng::seed_from_u64(7);
        let y = propose_crossover(x.view(), &species, 1.25, Some(&groups), &mut rng);
        let shift: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
        assert!(
            shift.sqrt() > 0.2,
            "a water dimer crossover was the identity"
        );
    }

    #[test]
    fn a_proton_moved_to_the_other_water_changes_the_graph() {
        let species = [8, 1, 1, 8, 1, 1];
        let intact = Array1::from(vec![
            0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 2.9, 0.0, 0.0, 3.2, 0.9, 0.0, 3.2,
            -0.9, 0.0,
        ]);
        let reference = GraphReference::of(intact.view(), &species, 1.25);
        assert_eq!(
            bond_graph(intact.view(), &species, 1.25),
            vec![vec![1, 2], vec![0], vec![0], vec![4, 5], vec![3], vec![3]]
        );

        let mut apart = intact.clone();
        for i in 9..18 {
            if i % 3 == 0 {
                apart[i] += 1.5;
            }
        }
        assert!(reference.kept(apart.view(), &species, 1.25));

        let mut transferred = intact.clone();
        transferred[3] = 1.95;
        assert!(!reference.kept(transferred.view(), &species, 1.25));

        // Each water is again intact, but the graph is over atom indices, so
        // the hydrogens trading molecules is still a change.
        let mut swapped = intact.clone();
        for axis in 0..3 {
            swapped.swap(3 + axis, 12 + axis);
        }
        assert!(!reference.kept(swapped.view(), &species, 1.25));
        assert_eq!(reference.changes(), 2);
    }

    /// Two waters with a hydrogen of the first 1.0 A from the second
    /// oxygen, inside the O-H bond cut of 1.2125 A. The centroids sit
    /// 2.12 A apart along x, so the contact opens by 2.12 (s - 1).
    fn crushed_dimer() -> (Array1<f64>, Vec<u32>, Vec<Vec<usize>>) {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 1.96, 0.0, 0.0, 2.2, 0.93, 0.0, 2.92,
            0.0, 0.0,
        ]);
        (
            x,
            vec![8, 1, 1, 8, 1, 1],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
        )
    }

    fn water_config(species: &[u32], groups: &[Vec<usize>]) -> super::super::Config {
        super::super::Config::recommended_molecular(species.to_vec(), groups.to_vec(), 1.0)
    }

    /// 1.05 leaves the contact at 1.106 A; 1.05^2 opens it to 1.2173 A.
    #[test]
    fn a_crushed_dimer_is_separated_at_the_smallest_factor() {
        let (x, species, groups) = crushed_dimer();
        let cfg = water_config(&species, &groups);
        let (report, moved) = separate_start(&cfg, x.view()).expect("a contact");
        let factor = 1.05_f64.powi(2);
        assert_eq!(report, StartSeparation::Separated { steps: 2, factor });
        let moved = moved.expect("a separated start");
        for atom in 0..6 {
            let sign = if atom < 3 { -1.0 } else { 1.0 };
            let want = x[3 * atom] + sign * (factor - 1.0) * 1.06;
            assert!((moved[3 * atom] - want).abs() < 1e-12);
            assert_eq!(moved[3 * atom + 1], x[3 * atom + 1]);
            assert_eq!(moved[3 * atom + 2], x[3 * atom + 2]);
        }
        let (group, _) = group_index(&groups, 6);
        assert!(!groups_bonded(
            moved.view(),
            &species,
            cfg.bond_tolerance,
            &group
        ));
        let once = Array1::from_shape_fn(18, |c| {
            let sign = if c < 9 { -1.0 } else { 1.0 };
            x[c] + if c % 3 == 0 { sign * 0.05 * 1.06 } else { 0.0 }
        });
        assert!(groups_bonded(
            once.view(),
            &species,
            cfg.bond_tolerance,
            &group
        ));
        assert_eq!(
            bond_graph(moved.view(), &species, 1.25),
            vec![vec![1, 2], vec![0], vec![0], vec![4, 5], vec![3], vec![3]]
        );
    }

    #[test]
    fn a_start_without_contact_between_groups_is_left_alone() {
        let (mut x, species, groups) = crushed_dimer();
        for atom in 3..6 {
            x[3 * atom] += 1.0;
        }
        assert!(separate_start(&water_config(&species, &groups), x.view()).is_none());
    }

    /// One group, a frozen point, the graph not kept, or the setting off.
    #[test]
    fn separation_needs_two_groups_no_frozen_point_and_the_graph_kept() {
        let (x, species, groups) = crushed_dimer();
        let one = water_config(&species, &[(0..6).collect()]);
        assert!(separate_start(&one, x.view()).is_none());
        let mut frozen = water_config(&species, &groups);
        frozen.frozen = Some(vec![true, false, false, false, false, false]);
        assert!(separate_start(&frozen, x.view()).is_none());
        let mut free = water_config(&species, &groups);
        free.keep_bond_graph = false;
        assert!(separate_start(&free, x.view()).is_none());
        let mut off = water_config(&species, &groups);
        off.separate_start_molecules = false;
        assert!(separate_start(&off, x.view()).is_none());
    }

    /// A search from the crushed start is the search from its separated
    /// structure, call for call.
    #[test]
    fn the_separation_charges_nothing() {
        let (x, species, groups) = crushed_dimer();
        let cfg = water_config(&species, &groups);
        let (_, moved) = separate_start(&cfg, x.view()).expect("a contact");
        let moved = moved.expect("a separated start");
        let springs = Springs::new(moved.as_slice().expect("contiguous"), &species);
        let run = |start: ArrayView1<f64>| {
            let mut ledger = super::super::Ledger::new(200);
            let (out, _) =
                crate::methods::cluster_search::search_from(&springs, &cfg, &mut ledger, start, 3)
                    .expect("finite start");
            assert_eq!(out.charged, ledger.spent());
            out
        };
        let crushed = run(x.view());
        let separated = run(moved.view());
        assert!(matches!(
            crushed.start_separation,
            Some(StartSeparation::Separated { steps: 2, .. })
        ));
        assert_eq!(separated.start_separation, None);
        assert!(crushed.charged <= 200);
        assert_eq!(crushed.charged, separated.charged);
        assert_eq!(crushed.best, separated.best);
        assert_eq!(crushed.best_state, separated.best_state);
        assert_eq!(crushed.improvements, separated.improvements);
    }

    /// The archive search separates its start as a search does. 1200 calls
    /// take its molecular branch past the 900-call first hop into the
    /// residual start of the second, so every start it derives is checked.
    #[cfg(feature = "graphkey")]
    #[test]
    fn the_archive_search_separates_its_start() {
        use crate::methods::archive_search::{Archive, ArchiveOutcome, archive_search};
        use eindir_core::gradient::DifferentiableObjective;
        let (x, species, groups) = crushed_dimer();
        let cfg = water_config(&species, &groups);
        let (_, moved) = separate_start(&cfg, x.view()).expect("a contact");
        let moved = moved.expect("a separated start");
        let springs = Springs::new(moved.as_slice().expect("contiguous"), &species);
        let run = |start: ArrayView1<f64>| -> ArchiveOutcome {
            let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
            opt.gtol = cfg.record_gradient;
            let mut relax = |led: &mut super::super::Ledger, x: ArrayView1<f64>, steps: usize| {
                opt.forget();
                let (f, y, _) = opt.minimize(x, steps, |v| {
                    if !led.charge() {
                        return None;
                    }
                    Some(springs.value_and_gradient(v))
                });
                (f, y)
            };
            let mut grad = |led: &mut super::super::Ledger, x: ArrayView1<f64>| {
                led.charge().then(|| springs.value_and_gradient(x).1)
            };
            let mut ledger = super::super::Ledger::new(1200);
            let mut archive = Archive::new();
            let mut rng = StdRng::seed_from_u64(3);
            let out = archive_search(
                &cfg,
                start,
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                &mut archive,
                &mut rng,
            );
            assert_eq!(out.charged, ledger.spent());
            out
        };
        let crushed = run(x.view());
        let separated = run(moved.view());
        assert_eq!(
            crushed.start_separation,
            Some(StartSeparation::Separated {
                steps: 2,
                factor: 1.05_f64.powi(2)
            })
        );
        assert_eq!(separated.start_separation, None);
        assert!(crushed.charged > 900, "the second hop never ran");
        assert!(crushed.best.is_finite());
        assert_eq!(crushed.charged, separated.charged);
        assert_eq!(crushed.best, separated.best);
        assert_eq!(crushed.best_state, separated.best_state);
        assert_eq!(crushed.best_at, separated.best_at);
        assert_eq!(crushed.full, separated.full);
        assert_eq!(crushed.screens, separated.screens);
    }
}
