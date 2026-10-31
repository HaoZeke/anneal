//! Basin hopping with structural moves for short-range atomic clusters and
//! rigid water.
//!
//! The chain is the Monte Carlo minimisation of Li and Scheraga (1987) and
//! Wales and Doye (1997): every trial is relaxed to a local minimum, and the
//! Metropolis test runs on the relaxed energies at a fixed temperature. This
//! is the loop GMIN runs. What the other hoppers in this crate lack for the
//! problems this module targets is the moves.
//!
//! Short-range Morse clusters (`rho` of 10 and above) have narrow wells and
//! global minima that are decahedral or close packed. A random displacement
//! of every atom mostly lands in a disordered basin, and the chain then needs
//! many hops to anneal the surface. Doye, Wales and Berry (1995) and Doye and
//! Wales (1997) locate these minima with moves that take a poorly bonded
//! surface atom and put it in a surface vacancy, which is what
//! [`Moves::Atomic`] proposes alongside the uniform kick. The third atomic
//! move is the cut-and-splice crossover of Deaven and Ho (1995) between the
//! live structure and a member of a small pool of low distinct minima, which
//! recombines well-formed halves the chain found at different times.
//!
//! Rigid TIP4P has a landscape organised by the hydrogen-bond network, and
//! the moves that matter turn molecules rather than shift them. Wales and
//! Hodges (1998) located the TIP4P minima up to N = 21 with GMIN's rigid-body
//! steps, rotating every molecule by up to 1.4 rad and letting the quench
//! settle the centres. Kazachenko and Thakkar (2009, 2013) improved on that by
//! reorienting one molecule at a time and by moving the worst bonded molecule
//! to the surface. The surface move here takes a molecule with the fewest
//! intermolecular O to H contacts under 2.5 Å and puts it, in a uniformly
//! random orientation, on a random ray from the centroid, 2 Å beyond the
//! farthest projection of any other centre onto that ray. The surface
//! operator of Takeuchi (2008) inspired it, but that operator is a different
//! move: it takes the highest-energy molecule to the most stable surface
//! position. [`Moves::Water`] proposes all four, plus a molecular
//! cut-and-splice that keeps each molecule whole.
//!
//! On Morse 55 at `rho = 14` this chain does worse than
//! `cluster_hopping::Config::recommended(55)`: 0 of 32 hits against 2 of 32
//! on the registered seeds 9200 to 9231. It is not a Morse default.
//!
//! Every value-and-gradient call is charged to the [`Ledger`], and the run
//! ends when the ledger refuses a charge.

use crate::methods::cluster_hopping::{Ledger, Outcome};
use crate::methods::splice::cut_and_splice;
use crate::methods::warm_lbfgs::WarmLbfgs;
use crate::potentials::{
    compose_rotation, fold_rotation_vectors, gmin_tip4p_sites, mat_vec, random_tip4p_cluster,
    rotation_matrix, rotation_vector,
};
use eindir_core::gradient::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Which move set the chain proposes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Moves {
    /// Point particles, 3 coordinates each, lengths in units of the pair
    /// equilibrium distance.
    Atomic {
        /// Distance below which two atoms count as bonded.
        bond: f64,
        /// Probability of the surface vacancy move.
        surface: f64,
        /// Probability of a cut-and-splice with a pool member.
        splice: f64,
    },
    /// Rigid waters in the block chart: `3 n` centres in angstrom, then `3 n`
    /// rotation vectors.
    Water {
        /// Half width of the per-component orientation step, radians.
        orient_step: f64,
        /// Half width of the per-component centre step, angstrom.
        translate_step: f64,
        /// Probability of reorienting one molecule uniformly.
        single: f64,
        /// Probability of moving the worst bonded molecule to the surface.
        surface: f64,
        /// Probability of a molecular cut-and-splice with a pool member.
        splice: f64,
    },
}

/// Settings for [`hop`].
#[derive(Debug, Clone)]
pub struct Config {
    /// Particles: atoms for [`Moves::Atomic`], molecules for [`Moves::Water`].
    pub n: usize,
    /// Moves proposed.
    pub moves: Moves,
    /// Metropolis temperature on quenched energies, in energy units.
    pub temperature: f64,
    /// Starting half width of the uniform atomic kick.
    pub kick: f64,
    /// Acceptance the atomic kick width is steered towards.
    pub target_acceptance: f64,
    /// Gradient infinity norm at which a quench is converged.
    pub gtol: f64,
    /// Iteration cap on a quench.
    pub max_quench_iters: usize,
    /// Distinct low minima kept for splicing and restarts.
    pub pool_size: usize,
    /// Energies closer than this are one minimum in the pool.
    pub pool_tol: f64,
    /// Hops without lowering the chain's own best before it restarts from a
    /// splice of two pool members.
    pub patience: usize,
    /// Every this many stalls, the restart is a fresh random structure
    /// instead of a splice. Zero never reseeds.
    pub reseed_every: usize,
    /// Centres farther than this from the centroid are pulled back before a
    /// quench, and a quench ending with one outside is rejected.
    pub container: f64,
    /// Closest approach enforced on the structures the moves build.
    pub min_sep: f64,
}

impl Config {
    /// Short-range Morse clusters in reduced units.
    ///
    /// The kick starts at 0.4 pair distances and is steered to half
    /// acceptance; the temperature is 0.8 well depths, the Wales and Doye
    /// (1997) basin-hopping value. Nearly half the hops relocate a surface
    /// atom.
    pub fn morse(n: usize) -> Self {
        Self {
            n,
            moves: Moves::Atomic {
                bond: 1.25,
                surface: 0.45,
                splice: 0.1,
            },
            temperature: 0.8,
            kick: 0.4,
            target_acceptance: 0.5,
            gtol: 1e-4,
            max_quench_iters: 3000,
            pool_size: 8,
            pool_tol: 1e-4,
            patience: 400,
            reseed_every: 0,
            container: compact_radius(n) + 1.5,
            min_sep: 0.8,
        }
    }

    /// Rigid TIP4P in kJ/mol and angstrom.
    ///
    /// Orientation step 1.4 rad as in the Wales and Hodges runs, with a small
    /// centre step. The temperature follows GMIN's TIP4P inputs, 2.9 kJ/mol
    /// below 18 molecules and 2.2 above. Quenches stop at 250 iterations:
    /// most hops land in a basin the chain rejects, and finishing those
    /// quenches cost more hops than the unfinished ones lose. A chain that
    /// stalls restarts from a fresh random cluster, since at 20 molecules a
    /// splice of minima from one wide funnel stays in it.
    pub fn water(n: usize) -> Self {
        Self {
            n,
            moves: Moves::Water {
                orient_step: 1.4,
                translate_step: 0.2,
                single: 0.25,
                surface: 0.15,
                splice: 0.1,
            },
            temperature: if n < 18 { 2.9 } else { 2.2 },
            kick: 0.0,
            target_acceptance: 0.5,
            gtol: 1e-3,
            max_quench_iters: 250,
            pool_size: 8,
            pool_tol: 1e-3,
            patience: 300,
            reseed_every: 1,
            container: 1.45 * (n as f64 / 0.6).cbrt() + 4.0,
            min_sep: 2.4,
        }
    }
}

/// Radius of a close-packed sphere of `n` unit-diameter atoms.
fn compact_radius(n: usize) -> f64 {
    0.5 * (n as f64 / 0.74).cbrt()
}

/// Random atoms in a compact sphere, no two closer than 0.8.
pub fn random_compact_cluster<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Array1<f64> {
    let mut radius = 1.15 * compact_radius(n);
    let mut points: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut misses = 0;
    while points.len() < n {
        let p = [
            rng.random_range(-radius..radius),
            rng.random_range(-radius..radius),
            rng.random_range(-radius..radius),
        ];
        if norm(p) > radius {
            continue;
        }
        if points.iter().all(|q| dist(p, *q) >= 0.8) {
            points.push(p);
            misses = 0;
        } else {
            misses += 1;
            if misses > 2000 {
                radius *= 1.05;
                misses = 0;
            }
        }
    }
    Array1::from_iter(points.into_iter().flatten())
}

/// Moves that a hop can propose, for the acceptance statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Proposal {
    Kick,
    Surface,
    Splice,
    Single,
}

struct Pool {
    size: usize,
    tol: f64,
    members: Vec<(f64, Array1<f64>)>,
}

impl Pool {
    fn offer(&mut self, energy: f64, x: &Array1<f64>) {
        if let Some(same) = self
            .members
            .iter_mut()
            .find(|(e, _)| (*e - energy).abs() < self.tol)
        {
            if energy < same.0 {
                *same = (energy, x.clone());
            }
            return;
        }
        if self.members.len() < self.size {
            self.members.push((energy, x.clone()));
        } else if let Some(worst) = self
            .members
            .iter_mut()
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .filter(|worst| energy < worst.0)
        {
            *worst = (energy, x.clone());
        }
    }

    fn pick<'a, R: Rng + ?Sized>(&'a self, rng: &mut R, not: f64) -> Option<&'a Array1<f64>> {
        let others: Vec<&(f64, Array1<f64>)> = self
            .members
            .iter()
            .filter(|(e, _)| (*e - not).abs() >= self.tol)
            .collect();
        if others.is_empty() {
            return None;
        }
        Some(&others[rng.random_range(0..others.len())].1)
    }
}

/// Runs the chain from `start` until the ledger refuses a charge.
///
/// `start` is 3 coordinates per atom for [`Moves::Atomic`] and the block
/// rigid-body chart for [`Moves::Water`]. The outcome's `improvements` lists
/// each new best with the hop and the charged count it was reached at.
pub fn hop<O>(
    objective: &O,
    cfg: &Config,
    ledger: &mut Ledger,
    start: ArrayView1<f64>,
    seed: u64,
) -> Outcome
where
    O: DifferentiableObjective<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut lbfgs = WarmLbfgs::default();
    lbfgs.gtol = cfg.gtol;
    let mut basins: Vec<f64> = Vec::new();
    let mut out = Outcome {
        best: f64::INFINITY,
        ..Outcome::default()
    };
    let mut pool = Pool {
        size: cfg.pool_size,
        tol: cfg.pool_tol,
        members: Vec::new(),
    };

    let Some((mut energy, mut x)) = quench(objective, cfg, ledger, &mut lbfgs, start.to_owned())
    else {
        return out;
    };
    note(&mut out, ledger, &mut basins, cfg.pool_tol, energy, &x, 0);
    pool.offer(energy, &x);

    let mut kick = cfg.kick;
    let mut kick_window = (0usize, 0usize);
    let mut chain_best = energy;
    let mut stale = 0usize;
    let mut stalls = 0usize;
    let mut hops = 0usize;
    while ledger.remaining() > 0 {
        hops += 1;
        let (proposal, trial) = if stale >= cfg.patience && pool.members.len() >= 2 {
            stale = 0;
            stalls += 1;
            chain_best = f64::INFINITY;
            if cfg.reseed_every > 0 && stalls % cfg.reseed_every == 0 {
                (Proposal::Splice, fresh(cfg, &mut rng))
            } else {
                let a = pool.members[rng.random_range(0..pool.members.len())]
                    .1
                    .clone();
                let b = pool
                    .pick(&mut rng, f64::NAN)
                    .cloned()
                    .unwrap_or_else(|| x.clone());
                (Proposal::Splice, splice(cfg, a.view(), b.view(), &mut rng))
            }
        } else {
            propose(cfg, &x, energy, &pool, kick, &mut rng)
        };
        let Some((e_new, x_new)) = quench(objective, cfg, ledger, &mut lbfgs, trial) else {
            break;
        };
        let inside = max_radius(cfg, &x_new) <= cfg.container;
        if inside {
            note(
                &mut out,
                ledger,
                &mut basins,
                cfg.pool_tol,
                e_new,
                &x_new,
                hops,
            );
            pool.offer(e_new, &x_new);
        }
        let accept = inside
            && (e_new <= energy
                || rng.random::<f64>() < (-(e_new - energy) / cfg.temperature).exp());
        if proposal == Proposal::Kick {
            kick_window.0 += 1;
            kick_window.1 += usize::from(accept);
            if kick_window.0 == 50 {
                let rate = kick_window.1 as f64 / 50.0;
                kick *= if rate > cfg.target_acceptance {
                    1.05
                } else {
                    1.0 / 1.05
                };
                kick = kick.clamp(0.05, 1.2);
                kick_window = (0, 0);
            }
        }
        if accept {
            energy = e_new;
            x = x_new;
        }
        if energy < chain_best - cfg.pool_tol {
            chain_best = energy;
            stale = 0;
        } else {
            stale += 1;
        }
    }
    out.hops = hops;
    out.basins = basins.len();
    out.charged = ledger.spent();
    out.final_energy = energy;
    out.final_state = Some(x);
    out
}

fn fresh<R: Rng + ?Sized>(cfg: &Config, rng: &mut R) -> Array1<f64> {
    match cfg.moves {
        Moves::Atomic { .. } => random_compact_cluster(cfg.n, rng),
        Moves::Water { .. } => random_tip4p_cluster(cfg.n, rng),
    }
}

fn note(
    out: &mut Outcome,
    ledger: &mut Ledger,
    basins: &mut Vec<f64>,
    tol: f64,
    energy: f64,
    x: &Array1<f64>,
    hop: usize,
) {
    ledger.record(energy, x.view());
    if !basins.iter().any(|e| (e - energy).abs() < tol) && basins.len() < 1_000_000 {
        basins.push(energy);
    }
    if energy < out.best {
        out.best = energy;
        out.best_state = Some(x.clone());
        if out.improvements.len() < 10_000 {
            out.improvements
                .push((hop, ledger.spent(), basins.len(), energy));
        }
    }
}

fn quench<O>(
    objective: &O,
    cfg: &Config,
    ledger: &mut Ledger,
    lbfgs: &mut WarmLbfgs,
    mut x: Array1<f64>,
) -> Option<(f64, Array1<f64>)>
where
    O: DifferentiableObjective<f64>,
{
    contain(cfg, &mut x);
    lbfgs.forget();
    let mut refused = false;
    let (energy, mut x, _) = lbfgs.minimize(x.view(), cfg.max_quench_iters, |p| {
        if !ledger.charge() {
            refused = true;
            return None;
        }
        Some(objective.value_and_gradient(p))
    });
    if refused || !energy.is_finite() {
        return None;
    }
    recentre(cfg, &mut x);
    Some((energy, x))
}

fn centres(cfg: &Config) -> usize {
    cfg.n
}

fn centroid(cfg: &Config, x: &Array1<f64>) -> [f64; 3] {
    let n = centres(cfg);
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    c.map(|v| v / n as f64)
}

fn recentre(cfg: &Config, x: &mut Array1<f64>) {
    let c = centroid(cfg, x);
    for i in 0..centres(cfg) {
        for k in 0..3 {
            x[3 * i + k] -= c[k];
        }
    }
    if matches!(cfg.moves, Moves::Water { .. }) {
        fold_rotation_vectors(cfg.n, x.as_slice_mut().expect("contiguous state"));
    }
}

fn max_radius(cfg: &Config, x: &Array1<f64>) -> f64 {
    let c = centroid(cfg, x);
    (0..centres(cfg))
        .map(|i| dist(point(x, i), c))
        .fold(0.0, f64::max)
}

fn contain(cfg: &Config, x: &mut Array1<f64>) {
    let c = centroid(cfg, x);
    for i in 0..centres(cfg) {
        let p = point(x, i);
        let r = dist(p, c);
        if r > cfg.container {
            let s = 0.9 * cfg.container / r;
            for k in 0..3 {
                x[3 * i + k] = c[k] + s * (p[k] - c[k]);
            }
        }
    }
}

fn propose<R: Rng + ?Sized>(
    cfg: &Config,
    x: &Array1<f64>,
    energy: f64,
    pool: &Pool,
    kick: f64,
    rng: &mut R,
) -> (Proposal, Array1<f64>) {
    let u: f64 = rng.random();
    match cfg.moves {
        Moves::Atomic {
            bond,
            surface,
            splice: p_splice,
        } => {
            if u < p_splice
                && let Some(mate) = pool.pick(rng, energy)
            {
                return (Proposal::Splice, splice(cfg, x.view(), mate.view(), rng));
            }
            if u < p_splice + surface
                && let Some(moved) = surface_atom(x, bond, cfg.min_sep, rng)
            {
                return (Proposal::Surface, moved);
            }
            let mut trial = x.clone();
            for v in trial.iter_mut() {
                *v += rng.random_range(-kick..=kick);
            }
            (Proposal::Kick, trial)
        }
        Moves::Water {
            orient_step,
            translate_step,
            single,
            surface,
            splice: p_splice,
        } => {
            let n = cfg.n;
            if u < p_splice
                && let Some(mate) = pool.pick(rng, energy)
            {
                return (Proposal::Splice, splice(cfg, x.view(), mate.view(), rng));
            }
            let mut trial = x.clone();
            if u < p_splice + single {
                let i = rng.random_range(0..n);
                let w = rotation_vector(random_rotation(rng));
                trial
                    .slice_mut(ndarray::s![3 * n + 3 * i..3 * n + 3 * i + 3])
                    .assign(&ndarray::arr1(&w));
                return (Proposal::Single, trial);
            }
            if u < p_splice + single + surface {
                surface_molecule(cfg, &mut trial, rng);
                return (Proposal::Surface, trial);
            }
            for v in trial.iter_mut().take(3 * n) {
                *v += rng.random_range(-translate_step..=translate_step);
            }
            for v in trial.iter_mut().skip(3 * n) {
                *v += rng.random_range(-orient_step..=orient_step);
            }
            fold_rotation_vectors(n, trial.as_slice_mut().expect("contiguous state"));
            (Proposal::Kick, trial)
        }
    }
}

/// Moves the least coordinated atom into the best surface vacancy.
///
/// Candidate sites cap a triangle of mutually bonded atoms at unit distance
/// from all three. The site with most bonded neighbours wins, ties broken at
/// random. `None` when no site clears `min_sep`.
fn surface_atom<R: Rng + ?Sized>(
    x: &Array1<f64>,
    bond: f64,
    min_sep: f64,
    rng: &mut R,
) -> Option<Array1<f64>> {
    let n = x.len() / 3;
    let pts: Vec<[f64; 3]> = (0..n).map(|i| point(x, i)).collect();
    let neighbours: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i && dist(pts[i], pts[j]) < bond)
                .collect()
        })
        .collect();
    let lowest = neighbours.iter().map(Vec::len).min()?;
    let worst: Vec<usize> = (0..n)
        .filter(|&i| neighbours[i].len() <= lowest + 1)
        .collect();
    let mover = worst[rng.random_range(0..worst.len())];

    let mut best_score = 0usize;
    let mut best_sites: Vec<[f64; 3]> = Vec::new();
    for i in 0..n {
        if i == mover {
            continue;
        }
        for (a, &j) in neighbours[i].iter().enumerate() {
            if j <= i || j == mover {
                continue;
            }
            for &k in &neighbours[i][a + 1..] {
                if k <= j || k == mover || dist(pts[j], pts[k]) >= bond {
                    continue;
                }
                for site in caps(pts[i], pts[j], pts[k]) {
                    let mut score = 0usize;
                    let mut clear = true;
                    for (m, q) in pts.iter().enumerate() {
                        if m == mover {
                            continue;
                        }
                        let d = dist(site, *q);
                        if d < min_sep {
                            clear = false;
                            break;
                        }
                        if d < bond {
                            score += 1;
                        }
                    }
                    if !clear || score <= neighbours[mover].len() {
                        continue;
                    }
                    if score > best_score {
                        best_score = score;
                        best_sites.clear();
                    }
                    if score == best_score {
                        best_sites.push(site);
                    }
                }
            }
        }
    }
    if best_sites.is_empty() {
        return None;
    }
    let site = best_sites[rng.random_range(0..best_sites.len())];
    let mut moved = x.clone();
    for k in 0..3 {
        moved[3 * mover + k] = site[k];
    }
    Some(moved)
}

/// The two points at unit distance from `a`, `b` and `c`.
fn caps(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Vec<[f64; 3]> {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let normal = cross(ab, ac);
    let nn = dot(normal, normal);
    if nn < 1e-12 {
        return Vec::new();
    }
    // Circumcentre of the triangle, then out along the normal.
    let term = add(
        scale(cross(normal, ab), dot(ac, ac)),
        scale(cross(ac, normal), dot(ab, ab)),
    );
    let offset = scale(term, 0.5 / nn);
    let r2 = dot(offset, offset);
    if r2 >= 1.0 {
        return Vec::new();
    }
    let h = (1.0 - r2).sqrt();
    let centre = add(a, offset);
    let unit = scale(normal, 1.0 / nn.sqrt());
    vec![add(centre, scale(unit, h)), add(centre, scale(unit, -h))]
}

/// Moves the molecule with fewest hydrogen bonds to a random surface point,
/// in a uniformly random orientation.
fn surface_molecule<R: Rng + ?Sized>(cfg: &Config, x: &mut Array1<f64>, rng: &mut R) {
    let n = cfg.n;
    let bonds = hydrogen_bonds(n, x);
    let fewest = *bonds.iter().min().unwrap_or(&0);
    let worst: Vec<usize> = (0..n).filter(|&i| bonds[i] == fewest).collect();
    let mover = worst[rng.random_range(0..worst.len())];
    let c = centroid(cfg, x);
    let mut placed = None;
    for attempt in 0..30 {
        let u = random_unit(rng);
        let reach = (0..n)
            .filter(|&j| j != mover)
            .map(|j| dot(sub(point(x, j), c), u))
            .fold(f64::NEG_INFINITY, f64::max);
        let site = add(c, scale(u, reach + 2.0 + 0.1 * attempt as f64));
        if (0..n)
            .filter(|&j| j != mover)
            .all(|j| dist(point(x, j), site) >= cfg.min_sep)
        {
            placed = Some(site);
            break;
        }
    }
    let Some(site) = placed else {
        return;
    };
    let w = rotation_vector(random_rotation(rng));
    for k in 0..3 {
        x[3 * mover + k] = site[k];
        x[3 * n + 3 * mover + k] = w[k];
    }
}

/// Hydrogen bonds per molecule: an O to H distance under 2.5 angstrom
/// between different molecules counts once on each.
fn hydrogen_bonds(n: usize, x: &Array1<f64>) -> Vec<usize> {
    let body = gmin_tip4p_sites();
    let sites: Vec<[[f64; 3]; 3]> = (0..n)
        .map(|i| {
            let r = rotation_matrix(rotation_of(n, x, i));
            let c = point(x, i);
            [0, 1, 2].map(|s| add(c, mat_vec(r, body[s])))
        })
        .collect();
    let mut count = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            for h in 1..3 {
                if dist(sites[i][h], sites[j][0]) < 2.5 {
                    count[i] += 1;
                    count[j] += 1;
                }
            }
        }
    }
    count
}

fn splice<R: Rng + ?Sized>(
    cfg: &Config,
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    rng: &mut R,
) -> Array1<f64> {
    let rotation = random_rotation(rng);
    match cfg.moves {
        Moves::Atomic { .. } => {
            let mut turned = b.to_owned();
            for i in 0..cfg.n {
                let p = mat_vec(rotation, point(&turned, i));
                for k in 0..3 {
                    turned[3 * i + k] = p[k];
                }
            }
            cut_and_splice(a, turned.view(), None, cfg.min_sep, rng)
        }
        Moves::Water { .. } => {
            let n = cfg.n;
            let a = a.to_owned();
            let mut turned = b.to_owned();
            for i in 0..n {
                let p = mat_vec(rotation, point(&turned, i));
                let w = compose_rotation(rotation, rotation_of(n, &turned, i));
                for k in 0..3 {
                    turned[3 * i + k] = p[k];
                    turned[3 * n + 3 * i + k] = w[k];
                }
            }
            let take = rng.random_range(1..n);
            let normal = random_unit(rng);
            let order = |s: &Array1<f64>| {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&i, &j| dot(point(s, j), normal).total_cmp(&dot(point(s, i), normal)));
                idx
            };
            let from_a = order(&a);
            let from_b = order(&turned);
            let mut child = Array1::zeros(6 * n);
            let chosen = from_a[..take]
                .iter()
                .map(|&i| (&a, i))
                .chain(from_b[take..].iter().map(|&i| (&turned, i)));
            for (slot, (parent, i)) in chosen.enumerate() {
                for k in 0..3 {
                    child[3 * slot + k] = parent[3 * i + k];
                    child[3 * n + 3 * slot + k] = parent[3 * n + 3 * i + k];
                }
            }
            separate(n, &mut child, cfg.min_sep);
            child
        }
    }
}

/// Pushes the `b` side of the splice outwards until no two centres are closer
/// than `min_sep`, the gap repair of Deaven and Ho.
fn separate(n: usize, x: &mut Array1<f64>, min_sep: f64) {
    for _ in 0..50 {
        let mut worst = (f64::INFINITY, 0, 0);
        for i in 0..n {
            for j in i + 1..n {
                let d = dist(point(x, i), point(x, j));
                if d < worst.0 {
                    worst = (d, i, j);
                }
            }
        }
        if worst.0 >= min_sep {
            return;
        }
        let (d, i, j) = worst;
        let pi = point(x, i);
        let pj = point(x, j);
        let dir = if d > 1e-9 {
            scale(sub(pj, pi), 1.0 / d)
        } else {
            [0.0, 0.0, 1.0]
        };
        let push = 0.5 * (min_sep - d) + 1e-3;
        for k in 0..3 {
            x[3 * i + k] -= push * dir[k];
            x[3 * j + k] += push * dir[k];
        }
    }
}

/// Uniform random rotation from a unit quaternion (Shoemake 1992).
fn random_rotation<R: Rng + ?Sized>(rng: &mut R) -> [[f64; 3]; 3] {
    let (u1, u2, u3): (f64, f64, f64) = (rng.random(), rng.random(), rng.random());
    let tau = std::f64::consts::TAU;
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    let (w, x, y, z) = (
        b * (tau * u3).cos(),
        a * (tau * u2).sin(),
        a * (tau * u2).cos(),
        b * (tau * u3).sin(),
    );
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn random_unit<R: Rng + ?Sized>(rng: &mut R) -> [f64; 3] {
    loop {
        let v = [
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        ];
        let r = norm(v);
        if r > 1e-6 && r <= 1.0 {
            return scale(v, 1.0 / r);
        }
    }
}

fn rotation_of(n: usize, x: &Array1<f64>, i: usize) -> [f64; 3] {
    let b = 3 * n + 3 * i;
    [x[b], x[b + 1], x[b + 2]]
}

fn point(x: &Array1<f64>, i: usize) -> [f64; 3] {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
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

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potentials::{GminTip4p, PairPotential};

    #[test]
    fn caps_sit_at_unit_distance_from_the_triangle() {
        let s = 3f64.sqrt();
        let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5 * s, 0.0]);
        let sites = caps(a, b, c);
        assert_eq!(sites.len(), 2);
        for site in sites {
            for q in [a, b, c] {
                assert!((dist(site, q) - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn surface_move_raises_the_coordination_of_the_moved_atom() {
        // A tetrahedron with one atom hanging off a vertex: the dangling atom
        // has one neighbour and a capping site on a face gives it three.
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.866_025, 0.0, 0.5, 0.288_675, 0.816_497, -1.0,
            0.0, 0.0,
        ]);
        let mut rng = StdRng::seed_from_u64(3);
        let moved = surface_atom(&x, 1.25, 0.8, &mut rng).expect("a vacancy");
        let p = point(&moved, 4);
        let bonded = (0..4).filter(|&j| dist(p, point(&moved, j)) < 1.25).count();
        assert_eq!(bonded, 3);
    }

    #[test]
    fn random_rotation_is_orthonormal() {
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..20 {
            let r = random_rotation(&mut rng);
            for i in 0..3 {
                for j in 0..3 {
                    let d: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((d - want).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn water_splice_keeps_molecules_whole_and_apart() {
        let n = 8;
        let cfg = Config::water(n);
        let mut rng = StdRng::seed_from_u64(5);
        let a = random_tip4p_cluster(n, &mut rng);
        let b = random_tip4p_cluster(n, &mut rng);
        let child = splice(&cfg, a.view(), b.view(), &mut rng);
        assert_eq!(child.len(), 6 * n);
        for i in 0..n {
            for j in i + 1..n {
                assert!(dist(point(&child, i), point(&child, j)) >= cfg.min_sep - 1e-6);
            }
        }
    }

    #[test]
    fn every_call_is_charged_and_the_budget_holds() {
        let n = 13;
        let objective = PairPotential::morse(n, 6.0);
        let cfg = Config::morse(n);
        let mut rng = StdRng::seed_from_u64(1);
        let start = random_compact_cluster(n, &mut rng);
        let mut ledger = Ledger::new(20_000);
        let out = hop(&objective, &cfg, &mut ledger, start.view(), 1);
        assert_eq!(ledger.spent(), 20_000);
        assert_eq!(out.charged, 20_000);
        // Icosahedral Morse 13 at rho 6, Cambridge Cluster Database.
        assert!(out.best < -42.439 + 1e-3, "best {}", out.best);
    }

    #[test]
    fn water_chain_finds_the_hexamer_minimum() {
        let n = 6;
        let objective = GminTip4p::new(n);
        let cfg = Config::water(n);
        let mut rng = StdRng::seed_from_u64(2);
        let start = random_tip4p_cluster(n, &mut rng);
        let mut ledger = Ledger::new(60_000);
        let out = hop(&objective, &cfg, &mut ledger, start.view(), 2);
        assert!(out.best < -197.7805260034 + 1e-3, "best {}", out.best);
    }
}
