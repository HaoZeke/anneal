//! Dynamic lattice search on a Lennard-Jones cluster, with a counted quench.
//!
//! Shao, Cheng and Cai (J. Comput. Chem. 25, 1693, 2004) optimise the surface
//! of a cluster on a lattice the cluster itself defines. The vacant sites are
//! the hollows over its own surface triangles, and atoms move from the
//! highest-energy occupied positions to the lowest-energy vacant sites before
//! one quench settles the result. The lattice is rebuilt from the quenched
//! structure and the descent repeats while it improves. Nothing about the
//! answer enters: the sites are read off whatever structure the search stands
//! on, so a decahedral core grows a decahedral surface and an icosahedral core
//! an icosahedral one.
//!
//! Every potential evaluation is charged to a [`Ledger`]. A quench step is one
//! value-and-gradient call. A lattice step evaluates pair terms only and is
//! charged the fraction of a full evaluation its pair count represents, the
//! convention [`Ledger::charge_frac`] documents. Building the lattice and
//! choosing sites is geometry and calls no potential.

use std::collections::VecDeque;

use rand::Rng;

use crate::methods::cluster_hopping::Ledger;

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

/// Reduced Lennard-Jones energy and gradient, writing the gradient into `g`.
pub fn lj_value_gradient(x: &[f64], g: &mut [f64]) -> f64 {
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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(u, v)| u * v).sum()
}

fn rms(g: &[f64]) -> f64 {
    (dot(g, g) / g.len().max(1) as f64).sqrt()
}

/// Limited-memory quasi-Newton quench in the form GMIN uses for clusters.
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
}

impl Default for Quench {
    fn default() -> Self {
        Self {
            memory: 8,
            max_step: 0.4,
            rms_tolerance: 1e-5,
            max_iterations: 4000,
            initial_diagonal: 0.1,
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

impl Quench {
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
        let mut e = lj_value_gradient(&x, &mut g);
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
                let en = lj_value_gradient(&xn, &mut gn);
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
    /// diagonal `sqrt(2) * 2^(1/6)`, so fourfold hollows are found as well
    /// as threefold ones.
    pub hollow_cutoff: f64,
    /// Distance from a site to each atom of its triangle.
    pub site_distance: f64,
    /// A candidate site nearer than this to any atom is occupied.
    pub clearance: f64,
    /// Sites nearer each other than this are one site.
    pub merge_distance: f64,
    /// Atoms with at least this many bonds are interior and do not move.
    pub interior_coordination: usize,
    /// Highest-energy atoms and lowest-energy sites paired per move.
    pub candidates: usize,
    /// Moves per search; zero means one per atom.
    pub max_moves: usize,
}

impl Default for Lattice {
    fn default() -> Self {
        Self {
            bond_cutoff: 1.2 * LJ_PAIR_MINIMUM,
            hollow_cutoff: 1.47 * LJ_PAIR_MINIMUM,
            site_distance: LJ_PAIR_MINIMUM,
            clearance: 0.85 * LJ_PAIR_MINIMUM,
            merge_distance: 0.25 * LJ_PAIR_MINIMUM,
            interior_coordination: 12,
            candidates: 4,
            max_moves: 0,
        }
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

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
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

impl Lattice {
    /// Bonds per atom at [`Lattice::bond_cutoff`].
    pub fn coordination(&self, x: &[f64]) -> Vec<usize> {
        let n = x.len() / 3;
        let cut2 = self.bond_cutoff * self.bond_cutoff;
        let mut count = vec![0usize; n];
        for i in 0..n {
            let pi = position(x, i);
            for j in (i + 1)..n {
                if dist2(pi, position(x, j)) < cut2 {
                    count[i] += 1;
                    count[j] += 1;
                }
            }
        }
        count
    }

    /// Vacant sites: hollows over the cluster's own triangles, clear of every
    /// atom, with near-coincident hollows merged.
    pub fn sites(&self, x: &[f64]) -> Vec<[f64; 3]> {
        let n = x.len() / 3;
        let hollow2 = self.hollow_cutoff * self.hollow_cutoff;
        let reach = self.hollow_cutoff + self.site_distance + self.clearance;
        let reach2 = reach * reach;
        let clear2 = self.clearance * self.clearance;
        let site2 = self.site_distance * self.site_distance;
        let mut near: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut hollow: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            let pi = position(x, i);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let r2 = dist2(pi, position(x, j));
                if r2 < reach2 {
                    near[i].push(j);
                }
                if j > i && r2 < hollow2 {
                    hollow[i].push(j);
                }
            }
        }
        let mut found: Vec<[f64; 3]> = Vec::new();
        for a in 0..n {
            let pa = position(x, a);
            for (bi, &b) in hollow[a].iter().enumerate() {
                let pb = position(x, b);
                for &c in &hollow[a][bi + 1..] {
                    let pc = position(x, c);
                    if dist2(pb, pc) >= hollow2 {
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
                            .all(|&k| k == b || k == c || dist2(p, position(x, k)) >= clear2);
                        if clear {
                            found.push(p);
                        }
                    }
                }
            }
        }
        let merge2 = self.merge_distance * self.merge_distance;
        let mut sites: Vec<[f64; 3]> = Vec::with_capacity(found.len() / 4);
        for p in found {
            if sites.iter().all(|q| dist2(p, *q) >= merge2) {
                sites.push(p);
            }
        }
        sites
    }

    /// One greedy search: the highest-energy movable atom goes to the
    /// lowest-energy vacant site while that lowers the unrelaxed energy.
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
        if !ledger.charge_frac(1.0 + (m * n) as f64 / pairs) {
            return None;
        }
        let mut atom_energy = vec![0.0; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let v = lj_pair(dist2(atoms[i], atoms[j]));
                atom_energy[i] += v;
                atom_energy[j] += v;
            }
        }
        let mut site_energy: Vec<f64> = vacant
            .iter()
            .map(|&p| atoms.iter().map(|&q| lj_pair(dist2(p, q))).sum())
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
            worst.extend_from_slice(&movable);
            let ka = k.min(worst.len());
            worst
                .select_nth_unstable_by(ka - 1, |&a, &b| atom_energy[b].total_cmp(&atom_energy[a]));
            // A site's energy includes its bond to the atom that would move
            // there, so the moving atom's own term comes off every site.
            let mut best: Option<(f64, usize, usize, f64)> = None;
            for &i in &worst[..ka] {
                for s in 0..m {
                    let r2 = dist2(atoms[i], vacant[s]);
                    if r2 < 1e-12 {
                        continue;
                    }
                    let own = lj_pair(r2);
                    let delta = site_energy[s] - own - atom_energy[i];
                    if best.is_none_or(|(d, ..)| delta < d) {
                        best = Some((delta, i, s, own));
                    }
                }
            }
            if !ledger.charge_frac((ka * m) as f64 / pairs) {
                break;
            }
            let Some((delta, i, s, own)) = best else {
                break;
            };
            if delta > -1e-9 {
                break;
            }
            if !ledger.charge_frac((2 * n + 2 * m) as f64 / pairs) {
                break;
            }
            let old = atoms[i];
            let new = vacant[s];
            let old_energy = atom_energy[i];
            for j in 0..n {
                if j == i {
                    continue;
                }
                atom_energy[j] += lj_pair(dist2(atoms[j], new)) - lj_pair(dist2(atoms[j], old));
            }
            atom_energy[i] = site_energy[s] - own;
            for t in 0..m {
                if t == s {
                    continue;
                }
                site_energy[t] += lj_pair(dist2(vacant[t], new)) - lj_pair(dist2(vacant[t], old));
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
        loop {
            let Some(moved) = self.search(&best, ledger) else {
                break;
            };
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
                assert!(dist2(*p, position(&x, i)) >= (0.85 * LJ_PAIR_MINIMUM).powi(2) - 1e-9);
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
}
