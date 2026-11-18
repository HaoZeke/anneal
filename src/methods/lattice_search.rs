// Copyright (c) anneal developers

//! Counted quench and single-atom lattice exchange for pair-potential clusters.
//!
//! The exchange kernel is a local move: from a quenched structure it reads the
//! sites its own triangles define and moves the highest-energy atom to a
//! vacant one. It does not rebuild the structure. The whole-structure
//! constructions that used to sit beside it (successive placement,
//! occupation optimisation, reoccupation) are gone from this crate.
//!
//! Site energies are single-atom pair sums, one `N`th of a full evaluation,
//! and are charged to the ledger at that fraction so the reported cost is
//! comparable with every other method in the crate.

use rand::Rng;

use crate::methods::cluster_hopping::Ledger;
use crate::potentials::PairKind;
use std::collections::VecDeque;

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])
}

fn point(x: &[f64], i: usize) -> [f64; 3] {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

/// Median nearest-neighbour distance of `x`, if it has two points.
fn median_bond(x: &[f64]) -> Option<f64> {
    let n = x.len() / 3;
    let mut nearest = vec![f64::INFINITY; n];
    for a in 0..n {
        for b in (a + 1)..n {
            let d2 = dist2(point(x, a), point(x, b));
            nearest[a] = nearest[a].min(d2);
            nearest[b] = nearest[b].min(d2);
        }
    }
    let mut nn: Vec<f64> = nearest
        .iter()
        .filter(|d| d.is_finite())
        .map(|d| d.sqrt())
        .collect();
    if nn.is_empty() {
        return None;
    }
    nn.sort_by(|a, b| a.total_cmp(b));
    Some(nn[nn.len() / 2])
}

// A variant of dynamic lattice search on a Lennard-Jones cluster, with a
// counted quench.
//
// The search optimises the surface of a cluster on a lattice the cluster
// itself defines. Vacant sites sit over the cluster's own triangles, atoms
// move greedily from high-energy positions to low-energy vacant sites, and one
// quench settles the result. The lattice is rebuilt from the quenched
// structure and the descent repeats while the quenched energy falls.
//
// It is a variant of the dynamic lattice search of Shao, Cheng and Cai,
// J. Comput. Chem. 25, 1693 (2004), doi 10.1002/jcc.20096, and departs from
// it in four ways:
//
// - the sites are geometric, at the pair-well distance over every triangle of
//   atoms with edges under
//   [`Lattice::hollow_cutoff`](crate::methods::lattice_search::Lattice::hollow_cutoff),
//   and no site is relaxed;
// - every atom with fewer than twelve bonds is movable;
// - one deterministic greedy pass runs from the current structure, each move
//   chosen over the
//   [`Lattice::candidates`](crate::methods::lattice_search::Lattice::candidates)
//   highest-energy movable atoms against all sites;
// - one quench follows each search.
//
// Costs here are force calls. The local minimisations per hit that the paper
// reports count its own procedure and do not convert to them.
//
// Nothing about the answer enters: the sites are read off whatever structure
// the search stands on, so a decahedral core grows a decahedral surface and an
// icosahedral core an icosahedral one. The lattice needs a pair potential with
// one well distance, since every site sits at that distance from the three
// atoms of its triangle.
//
// Every value-and-gradient call is charged as one call; lattice pair terms
// are charged at their fraction of the n(n-1)/2 pairs of a full evaluation
// and settled in whole calls; under one call per chain can be outstanding at
// a first hit or at the end of a run. Each chain has its own
// [`Ledger`](crate::methods::cluster_hopping::Ledger), and the pair terms go
// through
// [`Ledger::charge_frac`](crate::methods::cluster_hopping::Ledger::charge_frac).
// A search pays for its atom and site energies and for each move before
// computing them, and for the scan that chooses a move just after it. A
// ledger that runs dry paying for a dearer scan can leave more than a call of
// it unpaid, so the bound needs k m to be at most n(n-1)/2, for k candidate
// atoms and m vacant sites. Building the lattice and choosing sites is
// geometry and calls no potential.

/// Pair-well minimum of the reduced Lennard-Jones potential, `2^(1/6)`.
pub const LJ_PAIR_MINIMUM: f64 = 1.122_462_048_309_373;

/// Reduced Lennard-Jones pair energy at squared distance `r2`.
#[inline]
pub fn lj_pair(r2: f64) -> f64 {
    let inv6 = 1.0 / (r2 * r2 * r2);
    4.0 * inv6 * (inv6 - 1.0)
}

/// Reduced Lennard-Jones energy of a flattened `3N` cluster, no cutoff.
pub fn lj_value(x: &[f64]) -> f64 {
    let n = x.len() / 3;
    let mut e = 0.0;
    for i in 0..n {
        let (xi, yi, zi) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        for j in (i + 1)..n {
            let dx = xi - x[3 * j];
            let dy = yi - x[3 * j + 1];
            let dz = zi - x[3 * j + 2];
            let inv2 = 1.0 / (dx * dx + dy * dy + dz * dz);
            let inv6 = inv2 * inv2 * inv2;
            e += inv6 * (inv6 - 1.0);
        }
    }
    4.0 * e
}

thread_local! {
    static ORACLE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Starts counting potential calls on this thread. Pair with [`oracle_end`].
pub fn oracle_begin() {
    ORACLE.with(|slot| slot.set(Some(0)));
}

/// Stops counting and returns the calls recorded since [`oracle_begin`].
pub fn oracle_end() -> usize {
    ORACLE.with(|slot| slot.replace(None).unwrap_or(0))
}

fn oracle_add(n: usize) {
    if n == 0 {
        return;
    }
    ORACLE.with(|slot| {
        if let Some(count) = slot.get() {
            slot.set(Some(count + n));
        }
    });
}

/// Reduced Lennard-Jones energy and gradient, writing the gradient into `g`.
pub fn lj_value_gradient(x: &[f64], g: &mut [f64]) -> f64 {
    oracle_add(1);
    let n = x.len() / 3;
    g.iter_mut().for_each(|v| *v = 0.0);
    let mut e = 0.0;
    for i in 0..n {
        let (xi, yi, zi) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        let (mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0);
        for j in (i + 1)..n {
            let dx = xi - x[3 * j];
            let dy = yi - x[3 * j + 1];
            let dz = zi - x[3 * j + 2];
            let inv2 = 1.0 / (dx * dx + dy * dy + dz * dz);
            let inv6 = inv2 * inv2 * inv2;
            e += inv6 * (inv6 - 1.0);
            let c = inv2 * inv6 * (2.0 * inv6 - 1.0);
            gx += c * dx;
            gy += c * dy;
            gz += c * dz;
            g[3 * j] -= c * dx;
            g[3 * j + 1] -= c * dy;
            g[3 * j + 2] -= c * dz;
        }
        g[3 * i] += gx;
        g[3 * i + 1] += gy;
        g[3 * i + 2] += gz;
    }
    for v in g.iter_mut() {
        *v *= -24.0;
    }
    4.0 * e
}

/// Pair terms in one full evaluation of an `n`-point cluster.
fn full_pairs(n: usize) -> f64 {
    (n * n.saturating_sub(1) / 2).max(1) as f64
}

/// Charges `frac` and, while [`oracle_begin`] is active, counts the whole
/// units that charge settled. The residue under one unit is not a call.
pub fn charge_oracle(ledger: &mut Ledger, frac: f64) -> bool {
    let before = ledger.spent();
    let ok = ledger.charge_frac(frac);
    oracle_add(ledger.spent().saturating_sub(before));
    ok
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(u, v)| u * v).sum()
}

fn rms(g: &[f64]) -> f64 {
    (dot(g, g) / g.len().max(1) as f64).sqrt()
}

/// L-BFGS quench (Liu and Nocedal, Math. Program. 45, 503 (1989)) in the
/// step-capped form of Wales's GMIN.
///
/// No line search: the step along `-H g` is capped in length and shortened
/// tenfold while the energy rises, so an iteration usually costs one
/// value-and-gradient call. A strong Wolfe search costs several per iteration
/// on the same relaxation, and a search that pays per call cannot afford it.
#[derive(Debug, Clone)]
pub struct Quench {
    /// Curvature pairs retained.
    pub memory: usize,
    /// Longest step, as the Euclidean norm of the whole `3N` displacement.
    pub max_step: f64,
    /// Root-mean-square gradient component below which the quench stops.
    pub rms_tolerance: f64,
    /// Iterations before the quench gives up.
    pub max_iterations: usize,
    /// Inverse-Hessian diagonal before any curvature pair is stored.
    pub initial_diagonal: f64,
    /// Pair potential evaluated at each step. The default is reduced
    /// Lennard-Jones, and that path is the historical gradient.
    pub kind: PairKind,
}

impl Default for Quench {
    fn default() -> Self {
        Self {
            memory: 8,
            max_step: 0.4,
            rms_tolerance: 1e-5,
            max_iterations: 4000,
            initial_diagonal: 0.1,
            kind: PairKind::LennardJones,
        }
    }
}

/// What one quench produced.
#[derive(Debug, Clone)]
pub struct Relaxed {
    /// Energy at [`Relaxed::state`].
    pub energy: f64,
    /// Final coordinates.
    pub state: Vec<f64>,
    /// Root-mean-square gradient component at the final coordinates.
    pub rms: f64,
    /// Value-and-gradient calls charged.
    pub calls: usize,
    /// Whether the gradient tolerance was met.
    pub converged: bool,
}

fn pair_value_gradient(kind: PairKind, x: &[f64], g: &mut [f64]) -> f64 {
    oracle_add(1);
    let n = x.len() / 3;
    g.fill(0.0);
    let mut energy = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let (value, coef) = kind.pair(r2);
            energy += value;
            for k in 0..3 {
                g[3 * i + k] += coef * d[k];
                g[3 * j + k] -= coef * d[k];
            }
        }
    }
    energy
}

impl Quench {
    fn value_gradient(&self, x: &[f64], g: &mut [f64]) -> f64 {
        match self.kind {
            PairKind::LennardJones => lj_value_gradient(x, g),
            kind => pair_value_gradient(kind, x, g),
        }
    }

    /// Quenches `start`, charging one unit per value-and-gradient call.
    ///
    /// Returns `None` when the ledger cannot pay for the first evaluation. A
    /// ledger that runs out later ends the quench where it stands, unconverged.
    pub fn relax(&self, ledger: &mut Ledger, start: &[f64]) -> Option<Relaxed> {
        let dim = start.len();
        if dim == 0 || !ledger.charge() {
            return None;
        }
        let memory = self.memory.max(1);
        let mut x = start.to_vec();
        let mut g = vec![0.0; dim];
        let mut e = self.value_gradient(&x, &mut g);
        let mut calls = 1usize;
        let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(memory);
        let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(memory);
        let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(memory);
        let mut alpha = vec![0.0; memory];
        let mut diagonal = self.initial_diagonal;
        let mut xn = vec![0.0; dim];
        let mut gn = vec![0.0; dim];
        let mut d = vec![0.0; dim];
        let mut grad_rms = rms(&g);
        let mut converged = false;
        for _ in 0..self.max_iterations {
            if !e.is_finite() {
                break;
            }
            if grad_rms < self.rms_tolerance {
                converged = true;
                break;
            }
            d.copy_from_slice(&g);
            let held = s_hist.len();
            for k in (0..held).rev() {
                alpha[k] = rho_hist[k] * dot(&s_hist[k], &d);
                for (di, yi) in d.iter_mut().zip(&y_hist[k]) {
                    *di -= alpha[k] * yi;
                }
            }
            for di in d.iter_mut() {
                *di *= diagonal;
            }
            for k in 0..held {
                let beta = rho_hist[k] * dot(&y_hist[k], &d);
                for (di, si) in d.iter_mut().zip(&s_hist[k]) {
                    *di += (alpha[k] - beta) * si;
                }
            }
            for di in d.iter_mut() {
                *di = -*di;
            }
            if dot(&d, &g) > 0.0 {
                for di in d.iter_mut() {
                    *di = -*di;
                }
            }
            let norm = dot(&d, &d).sqrt();
            if !(norm > 0.0) || !norm.is_finite() {
                break;
            }
            let mut step = if norm > self.max_step {
                self.max_step / norm
            } else {
                1.0
            };
            let mut accepted = None;
            for _ in 0..10 {
                for i in 0..dim {
                    xn[i] = x[i] + step * d[i];
                }
                if !ledger.charge() {
                    return Some(Relaxed {
                        energy: e,
                        state: x,
                        rms: grad_rms,
                        calls,
                        converged: false,
                    });
                }
                let en = self.value_gradient(&xn, &mut gn);
                calls += 1;
                if en.is_finite() && en - e <= 1e-10 * e.abs().max(1.0) {
                    accepted = Some(en);
                    break;
                }
                step *= 0.1;
            }
            let Some(en) = accepted else {
                if s_hist.is_empty() {
                    break;
                }
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                diagonal = self.initial_diagonal;
                continue;
            };
            let mut s = if s_hist.len() == memory {
                rho_hist.pop_front();
                y_hist.pop_front();
                s_hist.pop_front().unwrap_or_else(|| vec![0.0; dim])
            } else {
                vec![0.0; dim]
            };
            let mut y = vec![0.0; dim];
            for i in 0..dim {
                s[i] = xn[i] - x[i];
                y[i] = gn[i] - g[i];
            }
            let sy = dot(&s, &y);
            let yy = dot(&y, &y);
            if sy > 1e-16 && yy > 0.0 {
                diagonal = sy / yy;
                rho_hist.push_back(1.0 / sy);
                s_hist.push_back(s);
                y_hist.push_back(y);
            }
            std::mem::swap(&mut x, &mut xn);
            std::mem::swap(&mut g, &mut gn);
            e = en;
            grad_rms = rms(&g);
        }
        Some(Relaxed {
            energy: e,
            state: x,
            rms: grad_rms,
            calls,
            converged,
        })
    }
}

/// The dynamic lattice and the greedy search over it.
#[derive(Debug, Clone)]
pub struct Lattice {
    /// Pair distance below which two atoms are bonded, for coordination.
    pub bond_cutoff: f64,
    /// Longest triangle edge a hollow is built over. Above the square
    /// diagonal `sqrt(2)` times the pair minimum, so fourfold hollows are
    /// found as well as threefold ones.
    pub hollow_cutoff: f64,
    /// Distance from a site to each atom of its triangle.
    pub site_distance: f64,
    /// A candidate site nearer than this to any atom is occupied.
    pub clearance: f64,
    /// Sites nearer each other than this are one site.
    pub merge_distance: f64,
    /// Atoms with at least this many bonds are interior and do not move.
    pub interior_coordination: usize,
    /// Highest-energy movable atoms weighed per move, each against every
    /// vacant site.
    pub candidates: usize,
    /// Moves per search; zero means one per atom.
    pub max_moves: usize,
    /// Pair potential the site energies are read from.
    pub kind: PairKind,
}

impl Lattice {
    /// Lengths are fixed ratios of `kind.r_min()`.
    ///
    /// The ratios are the historical Lennard-Jones lattice: bond 1.2,
    /// hollow 1.47, site 1, clearance 0.85, merge 0.25.
    /// `interior_coordination` stays 12, a field of the lattice rather than
    /// a property of the pair.
    pub fn for_pair(kind: PairKind) -> Self {
        let r = kind.r_min();
        Self {
            bond_cutoff: 1.2 * r,
            hollow_cutoff: 1.47 * r,
            site_distance: r,
            clearance: 0.85 * r,
            merge_distance: 0.25 * r,
            interior_coordination: 12,
            candidates: 4,
            max_moves: 0,
            kind,
        }
    }

    fn pair_energy(&self, r2: f64) -> f64 {
        match self.kind {
            PairKind::LennardJones => lj_pair(r2),
            kind => kind.pair(r2).0,
        }
    }
}

impl Default for Lattice {
    fn default() -> Self {
        Self::for_pair(PairKind::LennardJones)
    }
}

/// One lattice search: the moved structure before it is quenched.
#[derive(Debug, Clone)]
pub struct LatticeMoves {
    /// Coordinates after the moves.
    pub state: Vec<f64>,
    /// Atoms moved.
    pub moves: usize,
    /// Vacant sites the lattice held.
    pub sites: usize,
    /// Unrelaxed energy change the moves made.
    pub drop: f64,
}

/// What a lattice descent did.
#[derive(Debug, Clone, Copy, Default)]
pub struct DescentStats {
    /// Lattice searches run.
    pub searches: usize,
    /// Quenches run after a search that moved something.
    pub quenches: usize,
    /// Atoms moved, summed over searches.
    pub moves: usize,
}

fn position(x: &[f64], i: usize) -> [f64; 3] {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

fn coord_cmp(a: [f64; 3], b: [f64; 3]) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0])
        .then(a[1].total_cmp(&b[1]))
        .then(a[2].total_cmp(&b[2]))
}

/// Points in coordinate order, so later sums do not follow atom labels.
fn label_free_points(x: &[f64]) -> Vec<[f64; 3]> {
    let n = x.len() / 3;
    let mut atoms: Vec<[f64; 3]> = (0..n).map(|i| position(x, i)).collect();
    atoms.sort_by(|a, b| coord_cmp(*a, *b));
    atoms
}

fn centroid_of(atoms: &[[f64; 3]]) -> [f64; 3] {
    let mut c = [0.0; 3];
    if atoms.is_empty() {
        return c;
    }
    for p in atoms {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    let scale = 1.0 / atoms.len() as f64;
    [c[0] * scale, c[1] * scale, c[2] * scale]
}

/// Collapses points closer than `tol` to the one nearer `centre`.
///
/// The order is distance to the cluster centroid, then the coordinates.
/// Which point arrives first does not matter, and neither do atom labels.
pub fn merge_toward_centroid(points: &[[f64; 3]], centre: [f64; 3], tol: f64) -> Vec<[f64; 3]> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&i, &j| {
        pair_dist2(points[i], centre)
            .total_cmp(&pair_dist2(points[j], centre))
            .then_with(|| coord_cmp(points[i], points[j]))
    });
    let mut kept: Vec<[f64; 3]> = Vec::new();
    for i in order {
        let site = points[i];
        if kept.iter().all(|q| pair_dist2(site, *q) >= tol * tol) {
            kept.push(site);
        }
    }
    kept
}

/// Highest-energy indices, ties broken by coordinates rather than labels.
fn highest_energy(energy: &[f64], atoms: &[[f64; 3]], pool: &[usize], k: usize) -> Vec<usize> {
    let mut ranked = pool.to_vec();
    ranked.sort_by(|&a, &b| {
        energy[b]
            .total_cmp(&energy[a])
            .then_with(|| coord_cmp(atoms[a], atoms[b]))
    });
    ranked.truncate(k.min(ranked.len()));
    ranked
}

fn pair_dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
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

fn norm2(a: [f64; 3]) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

fn neighbours_within(atoms: &[[f64; 3]], cut2: f64) -> Vec<Vec<usize>> {
    let n = atoms.len();
    let mut lists = vec![Vec::new(); n];
    if n < 2 || !(cut2 > 0.0) || !cut2.is_finite() {
        return lists;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if pair_dist2(atoms[i], atoms[j]) < cut2 {
                lists[i].push(j);
                lists[j].push(i);
            }
        }
    }
    lists
}

impl Lattice {
    /// Bonds per atom at [`Lattice::bond_cutoff`].
    pub fn coordination(&self, x: &[f64]) -> Vec<usize> {
        let n = x.len() / 3;
        let atoms: Vec<[f64; 3]> = (0..n).map(|i| position(x, i)).collect();
        let lists = neighbours_within(&atoms, self.bond_cutoff * self.bond_cutoff);
        lists.into_iter().map(|list| list.len()).collect()
    }

    /// Vacant sites: hollows over the cluster's own triangles, clear of every
    /// atom, with near-coincident hollows merged.
    ///
    /// A hollow is built with its vertices in coordinate order, so the
    /// construction does not follow atom labels. Hollows closer than
    /// [`Lattice::merge_distance`] collapse to the one nearer the cluster
    /// centroid.
    pub fn sites(&self, x: &[f64]) -> Vec<[f64; 3]> {
        let atoms = label_free_points(x);
        let n = atoms.len();
        let centre = centroid_of(&atoms);
        let hollow2 = self.hollow_cutoff * self.hollow_cutoff;
        let reach = self.hollow_cutoff + self.site_distance + self.clearance;
        let reach2 = reach * reach;
        let clear2 = self.clearance * self.clearance;
        let site2 = self.site_distance * self.site_distance;
        let near = neighbours_within(&atoms, reach2);
        let mut hollow: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for &j in &near[i] {
                if j > i && pair_dist2(atoms[i], atoms[j]) < hollow2 {
                    hollow[i].push(j);
                }
            }
        }
        let mut found: Vec<[f64; 3]> = Vec::new();
        for a in 0..n {
            let pa = atoms[a];
            for (bi, &b) in hollow[a].iter().enumerate() {
                let pb = atoms[b];
                for &c in &hollow[a][bi + 1..] {
                    let pc = atoms[c];
                    if pair_dist2(pb, pc) >= hollow2 {
                        continue;
                    }
                    let u = sub(pb, pa);
                    let v = sub(pc, pa);
                    let w = cross(u, v);
                    let ww = norm2(w);
                    if ww < 1e-12 {
                        continue;
                    }
                    let t1 = cross(w, u);
                    let t2 = cross(v, w);
                    let (uu, vv) = (norm2(u), norm2(v));
                    let offset = [
                        (uu * t2[0] + vv * t1[0]) / (2.0 * ww),
                        (uu * t2[1] + vv * t1[1]) / (2.0 * ww),
                        (uu * t2[2] + vv * t1[2]) / (2.0 * ww),
                    ];
                    let r2 = norm2(offset);
                    if r2 >= site2 {
                        continue;
                    }
                    let h = (site2 - r2).sqrt() / ww.sqrt();
                    for sign in [1.0, -1.0] {
                        let p = [
                            pa[0] + offset[0] + sign * h * w[0],
                            pa[1] + offset[1] + sign * h * w[1],
                            pa[2] + offset[2] + sign * h * w[2],
                        ];
                        let clear = near[a]
                            .iter()
                            .all(|&k| k == b || k == c || pair_dist2(p, atoms[k]) >= clear2);
                        if clear {
                            found.push(p);
                        }
                    }
                }
            }
        }
        merge_toward_centroid(&found, centre, self.merge_distance)
    }

    /// One greedy pass. Each move makes the exchange that lowers the unrelaxed
    /// energy most among the [`Lattice::candidates`] highest-energy movable
    /// atoms and every vacant site, and the position the atom leaves becomes a
    /// site. The pass ends when no such exchange lowers the energy or after
    /// [`Lattice::max_moves`] moves.
    ///
    /// Returns `None` when the ledger cannot pay for the site energies.
    pub fn search(&self, x: &[f64], ledger: &mut Ledger) -> Option<LatticeMoves> {
        let n = x.len() / 3;
        if n < 4 {
            return Some(LatticeMoves {
                state: x.to_vec(),
                moves: 0,
                sites: 0,
                drop: 0.0,
            });
        }
        let pairs = full_pairs(n);
        let mut vacant = self.sites(x);
        let coordination = self.coordination(x);
        let movable: Vec<usize> = (0..n)
            .filter(|&i| coordination[i] < self.interior_coordination)
            .collect();
        let mut atoms: Vec<[f64; 3]> = (0..n).map(|i| position(x, i)).collect();
        if movable.is_empty() || vacant.is_empty() {
            return Some(LatticeMoves {
                state: x.to_vec(),
                moves: 0,
                sites: vacant.len(),
                drop: 0.0,
            });
        }
        let m = vacant.len();
        if !charge_oracle(ledger, 1.0 + (m * n) as f64 / pairs) {
            return None;
        }
        let mut atom_energy = vec![0.0; n];
        for i in 0..n {
            let mut terms = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i == j {
                    continue;
                }
                terms.push(self.pair_energy(pair_dist2(atoms[i], atoms[j])));
            }
            terms.sort_by(|a, b| a.total_cmp(b));
            atom_energy[i] = terms.iter().sum();
        }
        let mut site_energy: Vec<f64> = vacant
            .iter()
            .map(|&p| {
                let mut terms: Vec<f64> = atoms
                    .iter()
                    .map(|&q| self.pair_energy(pair_dist2(p, q)))
                    .collect();
                terms.sort_by(|a, b| a.total_cmp(b));
                terms.iter().sum()
            })
            .collect();
        let limit = if self.max_moves == 0 {
            n
        } else {
            self.max_moves
        };
        let k = self.candidates.max(1);
        let mut moves = 0usize;
        let mut drop = 0.0;
        let mut worst: Vec<usize> = Vec::with_capacity(movable.len());
        while moves < limit {
            worst.clear();
            worst.extend(highest_energy(&atom_energy, &atoms, &movable, k));
            let ka = worst.len();
            // A site's energy includes its bond to the atom that would move
            // there, so the moving atom's own term comes off every site.
            let mut best: Option<(f64, usize, usize, f64)> = None;
            for &i in &worst[..ka] {
                for s in 0..m {
                    let r2 = pair_dist2(atoms[i], vacant[s]);
                    if r2 < 1e-12 {
                        continue;
                    }
                    let own = self.pair_energy(r2);
                    let delta = site_energy[s] - own - atom_energy[i];
                    if best.is_none_or(|(d, ..)| delta < d) {
                        best = Some((delta, i, s, own));
                    }
                }
            }
            if !charge_oracle(ledger, (ka * m) as f64 / pairs) {
                break;
            }
            let Some((delta, i, s, own)) = best else {
                break;
            };
            if delta > -1e-9 {
                break;
            }
            if !charge_oracle(ledger, (2 * n + 2 * m) as f64 / pairs) {
                break;
            }
            let old = atoms[i];
            let new = vacant[s];
            let old_energy = atom_energy[i];
            for j in 0..n {
                if j == i {
                    continue;
                }
                atom_energy[j] += self.pair_energy(pair_dist2(atoms[j], new))
                    - self.pair_energy(pair_dist2(atoms[j], old));
            }
            atom_energy[i] = site_energy[s] - own;
            for t in 0..m {
                if t == s {
                    continue;
                }
                site_energy[t] += self.pair_energy(pair_dist2(vacant[t], new))
                    - self.pair_energy(pair_dist2(vacant[t], old));
            }
            atoms[i] = new;
            vacant[s] = old;
            site_energy[s] = old_energy + own;
            moves += 1;
            drop += delta;
        }
        let mut state = Vec::with_capacity(3 * n);
        for p in &atoms {
            state.extend_from_slice(p);
        }
        Some(LatticeMoves {
            state,
            moves,
            sites: m,
            drop,
        })
    }

    /// Lattice search and quench, repeated while the quenched energy falls.
    ///
    /// `energy` must be the quenched energy of `x`. Returns the lowest
    /// quenched structure reached and its energy.
    pub fn descend(
        &self,
        quench: &Quench,
        ledger: &mut Ledger,
        energy: f64,
        x: &[f64],
    ) -> (f64, Vec<f64>, DescentStats) {
        let mut stats = DescentStats::default();
        let mut best_energy = energy;
        let mut best = x.to_vec();
        while let Some(moved) = self.search(&best, ledger) {
            stats.searches += 1;
            if moved.moves == 0 {
                break;
            }
            stats.moves += moved.moves;
            let Some(relaxed) = quench.relax(ledger, &moved.state) else {
                break;
            };
            stats.quenches += 1;
            if relaxed.converged && relaxed.energy < best_energy - 1e-7 {
                best_energy = relaxed.energy;
                best = relaxed.state;
            } else {
                break;
            }
        }
        (best_energy, best, stats)
    }

    /// Moves `count` movable atoms, drawn uniformly, onto as many vacant sites,
    /// drawn uniformly. Geometry only; nothing is charged.
    pub fn shuffle<R: Rng + ?Sized>(&self, x: &[f64], count: usize, rng: &mut R) -> Vec<f64> {
        let n = x.len() / 3;
        let mut out = x.to_vec();
        if n < 4 || count == 0 {
            return out;
        }
        let sites = self.sites(x);
        let coordination = self.coordination(x);
        let mut movable: Vec<usize> = (0..n)
            .filter(|&i| coordination[i] < self.interior_coordination)
            .collect();
        if movable.is_empty() || sites.is_empty() {
            return out;
        }
        let take = count.min(movable.len()).min(sites.len());
        let mut chosen_sites: Vec<usize> = (0..sites.len()).collect();
        for slot in 0..take {
            let a = rng.random_range(slot..movable.len());
            movable.swap(slot, a);
            let s = rng.random_range(slot..chosen_sites.len());
            chosen_sites.swap(slot, s);
            let atom = movable[slot];
            let site = sites[chosen_sites[slot]];
            out[3 * atom] = site[0];
            out[3 * atom + 1] = site[1];
            out[3 * atom + 2] = site[2];
        }
        out
    }

    /// DLS-style exchange kernel: one Boltzmann-weighted move.
    ///
    /// This is not the published dynamic lattice search. Shao, Cheng
    /// and Cai, J. Comput. Chem. 25, 1693 (2004), build a lattice by
    /// relaxing a probe atom and then repeat a greedy exchange. That
    /// procedure is the `lj_dls` example, which is not part of the crate. The draw here is
    /// one exchange among the smallest unrelaxed rises, weighted by
    /// `exp(-delta / temperature)`. The caller's quench decides whether
    /// it stays. Coordination only keeps a fully coordinated interior
    /// atom in place.
    ///
    /// The vacant sites are [`Lattice::sites`] of the current geometry.
    /// Their distance is the structure's own median nearest-neighbour
    /// length, so the lattice is not a template and does not know a
    /// preferred size. Pair sums are charged at their fraction of one
    /// full evaluation. Returns `None` when the ledger cannot pay or
    /// the geometry has no vacant site.
    pub fn propose_energy_exchange<R: Rng + ?Sized>(
        &self,
        x: &[f64],
        ledger: &mut Ledger,
        temperature: f64,
        rng: &mut R,
    ) -> Option<Vec<f64>> {
        let n = x.len() / 3;
        if n < 4 {
            return None;
        }
        if let Some(hit) = exchange_cache_hit(self, x, temperature) {
            if !charge_oracle(ledger, hit.charge) {
                return None;
            }
            return draw_exchange(x, &hit.offers, &hit.weights, hit.sum, &hit.vacant, rng);
        }
        let well = median_bond(x)
            .filter(|b| *b > 1e-6)
            .unwrap_or(self.site_distance);
        let mut lattice = self.clone();
        let ratio = well / self.site_distance.max(1e-6);
        lattice.bond_cutoff *= ratio;
        lattice.hollow_cutoff *= ratio;
        lattice.site_distance = well;
        lattice.clearance *= ratio;
        lattice.merge_distance *= ratio;
        let vacant = lattice.sites(x);
        if vacant.is_empty() {
            return None;
        }
        let coordination = lattice.coordination(x);
        let movable: Vec<usize> = (0..n)
            .filter(|&i| coordination[i] < lattice.interior_coordination)
            .collect();
        if movable.is_empty() {
            return None;
        }
        let pairs = full_pairs(n);
        let m = vacant.len();
        let k = lattice.candidates.max(1).min(movable.len());
        // Atom rows, site rows, and the own-pair term of each offer.
        if !charge_oracle(ledger, 1.0 + ((m * n) + (k * m)) as f64 / pairs) {
            return None;
        }
        let (offers, weights, _atoms, vacant) =
            self.boltzmann_offers(x, &vacant, well, temperature, &movable, k)?;
        let sum: f64 = weights.iter().sum();
        let charge = 1.0 + ((m * n) + (k * m)) as f64 / pairs;
        exchange_cache_store(
            self,
            x,
            temperature,
            charge,
            &offers,
            &weights,
            sum,
            &vacant,
        );
        draw_exchange(x, &offers, &weights, sum, &vacant, rng)
    }

    /// Boltzmann weights of one exchange, as `(atom, site, probability)`.
    ///
    /// The same offers [`Lattice::propose_energy_exchange`] draws. Nothing
    /// is charged. `None` means the geometry has no exchange to offer.
    #[cfg(test)]
    pub(crate) fn exchange_probabilities(
        &self,
        x: &[f64],
        temperature: f64,
    ) -> Option<Vec<([f64; 3], [f64; 3], f64)>> {
        let n = x.len() / 3;
        if n < 4 {
            return None;
        }
        let well = median_bond(x)
            .filter(|b| *b > 1e-6)
            .unwrap_or(self.site_distance);
        let mut lattice = self.clone();
        let ratio = well / self.site_distance.max(1e-6);
        lattice.bond_cutoff *= ratio;
        lattice.hollow_cutoff *= ratio;
        lattice.site_distance = well;
        lattice.clearance *= ratio;
        lattice.merge_distance *= ratio;
        let vacant = lattice.sites(x);
        if vacant.is_empty() {
            return None;
        }
        let coordination = lattice.coordination(x);
        let movable: Vec<usize> = (0..n)
            .filter(|&i| coordination[i] < lattice.interior_coordination)
            .collect();
        if movable.is_empty() {
            return None;
        }
        let k = lattice.candidates.max(1).min(movable.len());
        let (offers, weights, atoms, vacant) =
            self.boltzmann_offers(x, &vacant, well, temperature, &movable, k)?;
        let sum: f64 = weights.iter().sum();
        if !(sum > 0.0) {
            return None;
        }
        Some(
            offers
                .iter()
                .zip(weights)
                .map(|((_, atom, site), weight)| (atoms[*atom], vacant[*site], weight / sum))
                .collect(),
        )
    }

    fn boltzmann_offers(
        &self,
        x: &[f64],
        vacant: &[[f64; 3]],
        well: f64,
        temperature: f64,
        movable: &[usize],
        k: usize,
    ) -> Option<(
        Vec<(f64, usize, usize)>,
        Vec<f64>,
        Vec<[f64; 3]>,
        Vec<[f64; 3]>,
    )> {
        let n = x.len() / 3;
        let m = vacant.len();
        let atoms: Vec<[f64; 3]> = (0..n).map(|i| position(x, i)).collect();
        // `r_min` is a pure function of the pair. Computing the well scale
        // once keeps `r2 * scale * scale` bit for bit with the per-pair call.
        // The pair energy is symmetric. Filling each atom's row as the pairs
        // are visited writes the same label order as the `j != i` scan, and
        // each term is evaluated once. `total_cmp` puts equal bits in one
        // class, so an unstable sort writes the sequence a stable sort would.
        let scale = self.kind.r_min() / well.max(1e-6);
        let width = n.saturating_sub(1);
        let mut packed = vec![0_i64; n * width];
        let mut fill = vec![0usize; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let energy = total_key(ranked_pair(self, pair_dist2(atoms[i], atoms[j]), scale));
                packed[i * width + fill[i]] = energy;
                fill[i] += 1;
                packed[j * width + fill[j]] = energy;
                fill[j] += 1;
            }
        }
        let mut atom_energy = vec![0.0; n];
        for i in 0..n {
            let row = &mut packed[i * width..(i + 1) * width];
            row.sort_unstable();
            atom_energy[i] = row.iter().copied().map(energy_from_key).sum();
        }
        let mut terms = Vec::with_capacity(n);
        let site_energy: Vec<f64> = vacant
            .iter()
            .map(|&p| {
                terms.clear();
                for q in &atoms {
                    terms.push(total_key(ranked_pair(self, pair_dist2(p, *q), scale)));
                }
                terms.sort_unstable();
                terms.iter().copied().map(energy_from_key).sum()
            })
            .collect();
        let ranked = highest_energy(&atom_energy, &atoms, movable, k);
        let mut offers: Vec<(f64, usize, usize)> = Vec::new();
        for &i in &ranked {
            for s in 0..m {
                let r2 = pair_dist2(atoms[i], vacant[s]);
                if r2 < 1e-8 {
                    continue;
                }
                let own = ranked_pair(self, r2, scale);
                let delta = site_energy[s] - own - atom_energy[i];
                offers.push((delta, i, s));
            }
        }
        if offers.is_empty() {
            return None;
        }
        offers.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| coord_cmp(atoms[a.1], atoms[b.1]))
                .then_with(|| coord_cmp(vacant[a.2], vacant[b.2]))
        });
        let pool = ENERGY_EXCHANGE_POOL.min(offers.len());
        offers.truncate(pool);
        let temp = temperature.max(1e-8);
        let best = offers
            .iter()
            .fold(f64::INFINITY, |lo, offer| lo.min(offer.0));
        let weights: Vec<f64> = offers
            .iter()
            .map(|offer| libm::exp(-(offer.0 - best) / temp))
            .collect();
        let sum: f64 = weights.iter().sum();
        if !(sum > 0.0) {
            return None;
        }
        Some((offers, weights, atoms, vacant.to_vec()))
    }
}

/// Exchanges offered to one dynamic-lattice draw.
const ENERGY_EXCHANGE_POOL: usize = 8;

struct ExchangeMemo {
    bits: Vec<u64>,
    temperature: u64,
    kind: u64,
    candidates: usize,
    interior: usize,
    charge: f64,
    offers: Vec<(f64, usize, usize)>,
    weights: Vec<f64>,
    sum: f64,
    vacant: Vec<[f64; 3]>,
}

std::thread_local! {
    static EXCHANGE_MEMO: std::cell::RefCell<Option<ExchangeMemo>> =
        const { std::cell::RefCell::new(None) };
}

fn kind_tag(kind: PairKind) -> u64 {
    match kind {
        PairKind::LennardJones => 1,
        PairKind::Morse { rho } => rho.to_bits() ^ 0x8000_0000_0000_0000,
    }
}

fn exchange_cache_hit(lattice: &Lattice, x: &[f64], temperature: f64) -> Option<ExchangeMemo> {
    EXCHANGE_MEMO.with(|slot| {
        let slot = slot.borrow();
        let memo = slot.as_ref()?;
        if memo.temperature != temperature.to_bits()
            || memo.kind != kind_tag(lattice.kind)
            || memo.candidates != lattice.candidates
            || memo.interior != lattice.interior_coordination
            || memo.bits.len() != x.len()
            || memo
                .bits
                .iter()
                .zip(x.iter())
                .any(|(bit, value)| *bit != value.to_bits())
        {
            return None;
        }
        Some(ExchangeMemo {
            bits: Vec::new(),
            temperature: memo.temperature,
            kind: memo.kind,
            candidates: memo.candidates,
            interior: memo.interior,
            charge: memo.charge,
            offers: memo.offers.clone(),
            weights: memo.weights.clone(),
            sum: memo.sum,
            vacant: memo.vacant.clone(),
        })
    })
}

fn exchange_cache_store(
    lattice: &Lattice,
    x: &[f64],
    temperature: f64,
    charge: f64,
    offers: &[(f64, usize, usize)],
    weights: &[f64],
    sum: f64,
    vacant: &[[f64; 3]],
) {
    EXCHANGE_MEMO.with(|slot| {
        *slot.borrow_mut() = Some(ExchangeMemo {
            bits: x.iter().map(|value| value.to_bits()).collect(),
            temperature: temperature.to_bits(),
            kind: kind_tag(lattice.kind),
            candidates: lattice.candidates,
            interior: lattice.interior_coordination,
            charge,
            offers: offers.to_vec(),
            weights: weights.to_vec(),
            sum,
            vacant: vacant.to_vec(),
        });
    });
}

fn draw_exchange<R: Rng + ?Sized>(
    x: &[f64],
    offers: &[(f64, usize, usize)],
    weights: &[f64],
    sum: f64,
    vacant: &[[f64; 3]],
    rng: &mut R,
) -> Option<Vec<f64>> {
    if !(sum > 0.0) || offers.is_empty() {
        return None;
    }
    let mut draw = rng.random::<f64>() * sum;
    let mut pick = offers.len() - 1;
    for (index, weight) in weights.iter().enumerate() {
        draw -= weight;
        if draw <= 0.0 {
            pick = index;
            break;
        }
    }
    let (_, atom, site) = offers[pick];
    let mut state = x.to_vec();
    state[3 * atom] = vacant[site][0];
    state[3 * atom + 1] = vacant[site][1];
    state[3 * atom + 2] = vacant[site][2];
    Some(state)
}

/// Integer image of [`f64::total_cmp`]'s ordering, and its inverse.
///
/// Sorting the keys sorts the energies. The transform is an involution, so
/// the energy is recovered bit for bit and the sum matches a stable
/// `total_cmp` sort.
fn total_key(value: f64) -> i64 {
    let mut bits = value.to_bits() as i64;
    bits ^= (((bits >> 63) as u64) >> 1) as i64;
    bits
}

fn energy_from_key(key: i64) -> f64 {
    let mut bits = key;
    bits ^= (((bits >> 63) as u64) >> 1) as i64;
    f64::from_bits(bits as u64)
}

/// Pair energy with the well moved onto `well`.
///
/// [`PairKind::r_min`] is the well of the formula. Dividing distances by
/// `well` and multiplying by that minimum puts a contact at the
/// structure's own bond length on the bottom of the well, for whatever
/// pair `lattice` carries.
fn ranked_pair(lattice: &Lattice, r2: f64, scale: f64) -> f64 {
    lattice.pair_energy(r2 * scale * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::PairKind;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn fixture(name: &str) -> Array1<f64> {
        let path = format!("{}/tests/fixtures/{name}.xyz", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut coords = Vec::new();
        for line in text.lines().skip(2) {
            let f: Vec<f64> = line
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().expect("coordinate"))
                .collect();
            if f.len() == 3 {
                coords.extend(f);
            }
        }
        Array1::from(coords)
    }

    fn icosahedron13() -> Vec<f64> {
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, p],
            [0.0, 1.0, -p],
            [0.0, -1.0, p],
            [0.0, -1.0, -p],
            [1.0, p, 0.0],
            [1.0, -p, 0.0],
            [-1.0, p, 0.0],
            [-1.0, -p, 0.0],
            [p, 0.0, 1.0],
            [-p, 0.0, 1.0],
            [p, 0.0, -1.0],
            [-p, 0.0, -1.0],
        ];
        let s = 1.1 / (1.0 + p * p).sqrt();
        let mut x = vec![0.0; 39];
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * (i + 1) + k] = s * v[k];
            }
        }
        x
    }

    #[test]
    fn the_gradient_matches_central_differences() {
        let x = icosahedron13();
        let mut g = vec![0.0; x.len()];
        let e = lj_value_gradient(&x, &mut g);
        assert!((e - lj_value(&x)).abs() < 1e-12);
        let h = 1e-6;
        for k in [0, 5, 17, 38] {
            let mut up = x.clone();
            up[k] += h;
            let mut down = x.clone();
            down[k] -= h;
            let fd = (lj_value(&up) - lj_value(&down)) / (2.0 * h);
            assert!(
                (fd - g[k]).abs() < 1e-5,
                "component {k}: {fd} against {}",
                g[k]
            );
        }
    }

    #[test]
    fn the_quench_finds_the_thirteen_point_minimum_and_charges_every_call() {
        let mut x = icosahedron13();
        let mut rng = StdRng::seed_from_u64(3);
        for v in x.iter_mut() {
            *v += rng.random_range(-0.05..0.05);
        }
        let mut ledger = Ledger::new(10_000);
        let out = Quench::default().relax(&mut ledger, &x).unwrap();
        assert!(out.converged);
        assert!(
            (out.energy + 44.326801).abs() < 1e-5,
            "energy {}",
            out.energy
        );
        assert_eq!(out.calls, ledger.spent());
    }

    #[test]
    fn the_quench_stops_on_an_empty_ledger() {
        let x = icosahedron13();
        let mut ledger = Ledger::new(5);
        let out = Quench::default().relax(&mut ledger, &x).unwrap();
        assert!(ledger.spent() <= 5);
        assert_eq!(out.calls, ledger.spent());
        let mut empty = Ledger::new(0);
        assert!(Quench::default().relax(&mut empty, &x).is_none());
    }

    #[test]
    fn an_icosahedron_offers_hollows_over_its_twenty_faces() {
        let x = icosahedron13();
        let sites = Lattice::default().sites(&x);
        assert!(sites.len() >= 20, "{} sites", sites.len());
        for p in &sites {
            for i in 0..13 {
                assert!(pair_dist2(*p, position(&x, i)) >= (0.85 * LJ_PAIR_MINIMUM).powi(2) - 1e-9);
            }
        }
    }

    #[test]
    fn a_misplaced_atom_returns_to_a_hollow_and_the_search_is_charged() {
        let mut x = icosahedron13();
        x.extend_from_slice(&[0.0, 0.0, 3.2]);
        let lattice = Lattice::default();
        let mut ledger = Ledger::new(1_000);
        let quench = Quench::default();
        let start = quench.relax(&mut ledger, &x).unwrap();
        let spent = ledger.spent();
        let moved = lattice.search(&start.state, &mut ledger).unwrap();
        assert!(ledger.spent() > spent, "the site energies were not charged");
        assert!(moved.drop <= 0.0);
        let (e, _, stats) = lattice.descend(&quench, &mut ledger, start.energy, &start.state);
        assert!(e <= start.energy + 1e-9);
        assert!(stats.searches >= 1);
        assert!((e + 47.845157).abs() < 1e-4, "fourteen points ended at {e}");
    }

    #[test]
    fn shuffling_keeps_the_atom_count_and_charges_nothing() {
        let x = icosahedron13();
        let mut rng = StdRng::seed_from_u64(9);
        let y = Lattice::default().shuffle(&x, 3, &mut rng);
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
        assert_ne!(x, y);
    }

    #[test]
    fn morse_site_distance_is_the_pair_minimum() {
        for rho in [6.0, 10.0, 14.0] {
            let kind = PairKind::Morse { rho };
            let lattice = Lattice::for_pair(kind);
            assert_eq!(lattice.site_distance.to_bits(), kind.r_min().to_bits());
            assert_eq!(lattice.site_distance.to_bits(), 1.0_f64.to_bits());
            assert_eq!(
                lattice.bond_cutoff.to_bits(),
                (1.2 * kind.r_min()).to_bits()
            );
            assert_eq!(
                lattice.hollow_cutoff.to_bits(),
                (1.47 * kind.r_min()).to_bits()
            );
            assert_eq!(lattice.clearance.to_bits(), (0.85 * kind.r_min()).to_bits());
            assert_eq!(
                lattice.merge_distance.to_bits(),
                (0.25 * kind.r_min()).to_bits()
            );
            assert_eq!(lattice.interior_coordination, 12);
        }
    }

    #[test]
    fn lennard_jones_lattice_matches_the_historical_default() {
        let lattice = Lattice::for_pair(PairKind::LennardJones);
        assert_eq!(lattice.site_distance.to_bits(), LJ_PAIR_MINIMUM.to_bits());
        assert_eq!(
            lattice.bond_cutoff.to_bits(),
            (1.2 * LJ_PAIR_MINIMUM).to_bits()
        );
        assert_eq!(
            lattice.hollow_cutoff.to_bits(),
            (1.47 * LJ_PAIR_MINIMUM).to_bits()
        );
        assert_eq!(
            lattice.clearance.to_bits(),
            (0.85 * LJ_PAIR_MINIMUM).to_bits()
        );
        assert_eq!(
            lattice.merge_distance.to_bits(),
            (0.25 * LJ_PAIR_MINIMUM).to_bits()
        );
        assert_eq!(lattice.interior_coordination, 12);
        assert_eq!(lattice.candidates, 4);
        assert_eq!(lattice.max_moves, 0);
        let preset = Lattice::default();
        assert_eq!(
            preset.site_distance.to_bits(),
            lattice.site_distance.to_bits()
        );
        assert_eq!(preset.bond_cutoff.to_bits(), lattice.bond_cutoff.to_bits());
        assert_eq!(
            preset.hollow_cutoff.to_bits(),
            lattice.hollow_cutoff.to_bits()
        );
        assert_eq!(preset.clearance.to_bits(), lattice.clearance.to_bits());
        assert_eq!(
            preset.merge_distance.to_bits(),
            lattice.merge_distance.to_bits()
        );
        assert_eq!(preset.interior_coordination, lattice.interior_coordination);
        assert_eq!(preset.candidates, lattice.candidates);
        assert_eq!(preset.max_moves, lattice.max_moves);
        assert_eq!(preset.kind, PairKind::LennardJones);
    }

    #[test]
    fn lennard_jones_site_energy_matches_lj_pair_bits() {
        let lattice = Lattice::default();
        for r2 in [0.5_f64, 1.0, LJ_PAIR_MINIMUM * LJ_PAIR_MINIMUM, 4.0] {
            assert_eq!(lattice.pair_energy(r2).to_bits(), lj_pair(r2).to_bits());
        }
        let morse = Lattice::for_pair(PairKind::Morse { rho: 14.0 });
        let (energy, _) = PairKind::Morse { rho: 14.0 }.pair(1.0);
        assert_eq!(morse.pair_energy(1.0).to_bits(), energy.to_bits());
        assert!(morse.kind != PairKind::LennardJones);
    }

    #[test]
    fn morse_quench_of_a_displaced_pair_sits_at_the_well() {
        let quench = Quench {
            kind: PairKind::Morse { rho: 14.0 },
            max_iterations: 400,
            ..Quench::default()
        };
        let start = [0.0, 0.0, 0.0, 1.15, 0.0, 0.0];
        let mut ledger = Ledger::new(2_000);
        let relaxed = quench.relax(&mut ledger, &start).expect("budget");
        assert!(relaxed.converged, "rms {}", relaxed.rms);
        assert!(
            (relaxed.energy + 1.0).abs() < 1e-5,
            "energy {}",
            relaxed.energy
        );
        let d = [
            relaxed.state[3] - relaxed.state[0],
            relaxed.state[4] - relaxed.state[1],
            relaxed.state[5] - relaxed.state[2],
        ];
        let bond = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((bond - 1.0).abs() < 1e-3, "bond {bond}");
        assert_eq!(Quench::default().kind, PairKind::LennardJones);
    }

    /// The C2v shelf recorded by the 2M search, not a literature geometry.
    ///
    /// A walk of energy-ranked lattice exchanges has to leave that shelf.
    /// Copying this test onto the tree without `propose_energy_exchange`
    /// does not compile.
    #[test]
    fn energy_exchange_leaves_the_recorded_c2v_shelf() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let x = fixture("lj104-trap-c2v");
        assert_eq!(x.len(), 104 * 3);
        let lattice = Lattice::default();
        let quench = Quench {
            max_iterations: 160,
            ..Quench::default()
        };
        let mut ledger = Ledger::new(400_000);
        let start = quench.relax(&mut ledger, x.as_slice().unwrap()).unwrap();
        assert!(
            (start.energy + 580.42126628).abs() < 1e-4,
            "start {}",
            start.energy
        );
        let spent = ledger.spent();
        let mut rng = StdRng::seed_from_u64(1);
        let mut energy = start.energy;
        let mut state = start.state;
        let mut best = energy;
        let mut moved = false;
        for _ in 0..30 {
            let Some(proposed) =
                lattice.propose_energy_exchange(&state, &mut ledger, 0.8, &mut rng)
            else {
                break;
            };
            assert_ne!(proposed, state);
            moved = true;
            let Some(relaxed) = quench.relax(&mut ledger, &proposed) else {
                break;
            };
            let delta = relaxed.energy - energy;
            if delta < 1e-9 || rng.random::<f64>() < (-delta.max(0.0) / 0.8).exp() {
                energy = relaxed.energy;
                state = relaxed.state;
            }
            best = best.min(energy);
        }
        assert!(moved, "the exchange never proposed a different geometry");
        assert!(ledger.spent() > spent, "site energies were not charged");
        assert!(
            best < start.energy - 0.4,
            "best {best} did not leave {}",
            start.energy
        );
    }

    fn relabel(x: &[f64], order: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        for (slot, &atom) in order.iter().enumerate() {
            out[3 * slot..3 * slot + 3].copy_from_slice(&x[3 * atom..3 * atom + 3]);
        }
        out
    }

    fn same_points(left: &[[f64; 3]], right: &[[f64; 3]]) -> bool {
        if left.len() != right.len() {
            return false;
        }
        let mut used = vec![false; right.len()];
        for p in left {
            let hit = right.iter().enumerate().position(|(i, q)| {
                !used[i]
                    && (p[0] - q[0]).abs() <= 1e-10
                    && (p[1] - q[1]).abs() <= 1e-10
                    && (p[2] - q[2]).abs() <= 1e-10
            });
            let Some(i) = hit else {
                return false;
            };
            used[i] = true;
        }
        true
    }

    fn same_law(left: &[([f64; 3], [f64; 3], f64)], right: &[([f64; 3], [f64; 3], f64)]) -> bool {
        if left.len() != right.len() {
            return false;
        }
        let mut used = vec![false; right.len()];
        for (atom, site, weight) in left {
            let hit = right.iter().enumerate().position(|(i, (a, s, w))| {
                !used[i]
                    && (atom[0] - a[0]).abs() <= 1e-10
                    && (atom[1] - a[1]).abs() <= 1e-10
                    && (atom[2] - a[2]).abs() <= 1e-10
                    && (site[0] - s[0]).abs() <= 1e-10
                    && (site[1] - s[1]).abs() <= 1e-10
                    && (site[2] - s[2]).abs() <= 1e-10
                    && (weight - w).abs() <= 1e-10
            });
            let Some(i) = hit else {
                return false;
            };
            used[i] = true;
        }
        true
    }

    /// Relabelling must not move a vacant site.
    ///
    /// The base merge keeps the first hollow in atom-index order, so
    /// this assertion fails there. The centroid order passes.
    #[test]
    fn vacant_sites_ignore_atom_labels() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        let lattice = Lattice::default();
        let quench = Quench {
            max_iterations: 40,
            rms_tolerance: 1.0e-2,
            ..Quench::default()
        };
        for seed in [11_u64, 29, 47] {
            let n = 38;
            let cfg = crate::methods::cluster_hopping::Config::for_cluster(n);
            let mut rng = StdRng::seed_from_u64(seed);
            let start = crate::methods::cluster_hopping::random_cluster_in_radius(
                n,
                cfg.start_radius(),
                cfg.min_separation,
                &mut rng,
            );
            let mut ledger = Ledger::new(40_000);
            let relaxed = quench
                .relax(&mut ledger, start.as_slice().unwrap())
                .expect("quench");
            let state = relaxed.state;
            let sites = lattice.sites(&state);
            assert!(!sites.is_empty(), "a packed cluster has a vacant site");
            let mut orders = vec![(0..n).rev().collect::<Vec<_>>()];
            let mut rotated: Vec<usize> = (0..n).collect();
            rotated.rotate_left(n / 3);
            orders.push(rotated);
            let mut shuffled: Vec<usize> = (0..n).collect();
            shuffled.shuffle(&mut StdRng::seed_from_u64(5));
            orders.push(shuffled);
            for order in orders {
                let moved = relabel(&state, &order);
                let sites_b = lattice.sites(&moved);
                assert!(
                    same_points(&sites, &sites_b),
                    "vacant sites moved under a relabelling"
                );
            }
        }
    }

    /// Relabelling a minimum must not move a vacant site or a draw.
    #[test]
    fn vacant_sites_and_move_probabilities_ignore_labels() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        let lattice = Lattice::default();
        let quench = Quench {
            max_iterations: 80,
            rms_tolerance: 1.0e-3,
            ..Quench::default()
        };
        let mut minima: Vec<Vec<f64>> = Vec::new();
        for seed in [11_u64, 29, 47] {
            let n = 38;
            let cfg = crate::methods::cluster_hopping::Config::for_cluster(n);
            let mut rng = StdRng::seed_from_u64(seed);
            let start = crate::methods::cluster_hopping::random_cluster_in_radius(
                n,
                cfg.start_radius(),
                cfg.min_separation,
                &mut rng,
            );
            let mut ledger = Ledger::new(80_000);
            let relaxed = quench
                .relax(&mut ledger, start.as_slice().unwrap())
                .expect("quench");
            minima.push(relaxed.state);
        }
        minima.push(fixture("lj104-trap-c2v").to_vec());
        for state in &minima {
            let n = state.len() / 3;
            let sites = lattice.sites(state);
            let law = lattice
                .exchange_probabilities(state, 0.8)
                .expect("the minimum offers an exchange");
            assert!(!sites.is_empty(), "a packed minimum has a vacant site");
            let mut orders = vec![(0..n).rev().collect::<Vec<_>>()];
            let mut rotated: Vec<usize> = (0..n).collect();
            rotated.rotate_left(n / 3);
            orders.push(rotated);
            let mut shuffled: Vec<usize> = (0..n).collect();
            shuffled.shuffle(&mut StdRng::seed_from_u64(5));
            orders.push(shuffled);
            for order in orders {
                let moved = relabel(state, &order);
                let sites_b = lattice.sites(&moved);
                assert!(
                    same_points(&sites, &sites_b),
                    "vacant sites moved under a relabelling"
                );
                let law_b = lattice
                    .exchange_probabilities(&moved, 0.8)
                    .expect("relabelled minimum offers an exchange");
                assert!(
                    same_law(&law, &law_b),
                    "move probabilities moved under a relabelling"
                );
            }
        }
    }

    #[test]
    fn neighbour_lists_match_the_double_loop_and_sites_repeat() {
        let mut s = 0xA5A5_F00Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
        };
        for n in [16usize, 40, 70, 90] {
            let radius = 1.1 * (n as f64).cbrt();
            let atoms: Vec<[f64; 3]> = (0..n)
                .map(|_| {
                    [
                        2.0 * radius * next(),
                        2.0 * radius * next(),
                        2.0 * radius * next(),
                    ]
                })
                .collect();
            let cut2 = 1.6 * 1.6;
            let lists = neighbours_within(&atoms, cut2);
            for i in 0..n {
                let mut brute = Vec::new();
                for j in 0..n {
                    if i != j && pair_dist2(atoms[i], atoms[j]) < cut2 {
                        brute.push(j);
                    }
                }
                assert_eq!(lists[i], brute, "atom {i} at n={n}");
            }
            let mut flat = Vec::with_capacity(3 * n);
            for atom in &atoms {
                flat.extend_from_slice(atom);
            }
            let lattice = Lattice::default();
            let once = lattice.sites(&flat);
            let twice = lattice.sites(&flat);
            assert_eq!(once, twice);
        }
    }

    #[test]
    fn the_ranked_key_sums_like_total_cmp() {
        let mut bits = 0xC0FF_EE00_u64;
        let mut next = || {
            bits ^= bits << 13;
            bits ^= bits >> 7;
            bits ^= bits << 17;
            let unit = (bits >> 11) as f64 / ((1u64 << 53) as f64);
            (unit - 0.5) * 20.0
        };
        for n in [1usize, 2, 17, 79, 80] {
            let values: Vec<f64> = (0..n).map(|_| next()).collect();
            for value in &values {
                assert_eq!(
                    energy_from_key(total_key(*value)).to_bits(),
                    value.to_bits()
                );
            }
            let mut stable = values.clone();
            stable.sort_by(|a, b| a.total_cmp(b));
            let mut keys: Vec<i64> = values.iter().copied().map(total_key).collect();
            keys.sort_unstable();
            for (left, key) in stable.iter().zip(keys.iter()) {
                assert_eq!(left.to_bits(), energy_from_key(*key).to_bits());
            }
            let slow: f64 = stable.iter().sum();
            let fast: f64 = keys.iter().copied().map(energy_from_key).sum();
            assert_eq!(slow.to_bits(), fast.to_bits());
        }
    }

    #[test]
    fn the_hoisted_well_scale_matches_the_per_pair_product() {
        for kind in [PairKind::LennardJones, PairKind::Morse { rho: 6.0 }] {
            let well = 1.05_f64;
            let scale = kind.r_min() / well.max(1e-6);
            let mut bits = 0x1234_5678_9ABCu64;
            for _ in 0..64 {
                bits = bits.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                let r2 = (bits >> 11) as f64 / ((1u64 << 53) as f64) * 3.0 + 0.25;
                let per_pair = {
                    let scale_i = kind.r_min() / well.max(1e-6);
                    r2 * scale_i * scale_i
                };
                assert_eq!(
                    (r2 * scale * scale).to_bits(),
                    per_pair.to_bits(),
                    "hoisted scale drifted for {kind:?} at r2={r2}"
                );
            }
        }
    }
}
