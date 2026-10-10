//! Persistent basin-hopping chains that trade cut-and-splice fragments.
//!
//! One process holds an ensemble of independently budgeted chains, each
//! walking the recommended cluster stack to the end of its own ledger. At
//! every checkpoint a chain publishes its live minimum to a shared board. In
//! the exchange arm, a chain periodically takes a partner's live structure,
//! cuts both by a random plane, quenches the spliced children off its own
//! oracle, judges the lowest child by the chain's Metropolis law, and adopts
//! it through [`CheckpointAction::ExternalAdopt`] with the construction cost
//! charged. The independent arm runs the same chains, seeds, and checkpoints
//! with the exchange disabled, so the two arms are paired at equal charged
//! work per chain.
//!
//! Usage:
//! `lj_ensemble_splice <n> <budget-per-chain> <chains> <ensembles> <indep|splice> [seed0]`
//!
//! Environment: `SPLICE_INTERVAL` charged evaluations between exchange
//! attempts (default 5000), `SPLICE_IMAGES` children per attempt (default 4),
//! `SPLICE_PARTNER` `random` or `best` (default `random`), `SPLICE_SOURCE`
//! `current` or `best` for the structures spliced (default `current`),
//! `CHECKPOINT` charged evaluations between board updates (default 500),
//! `COMPRESS_MU` two-phase quench: relax first on the compressed surface
//! `E + mu * sum |r_i - r_cm|^2`, then on the plain potential from there
//! (`mu` in energy per length squared, default 0, plain quench).
//! `DIAMETER_D` and `DIAMETER_BETA` add the Locatelli--Schoen diameter
//! penalty `beta * sum_{i<j} max(0, r_ij^2 - D^2)^2` to the same first
//! phase (`D` in units of the pair-well minimum distance, `beta` in energy
//! per length to the fourth, default 0 and 1); `DIAMETER_KAPPA` instead
//! sets the cutoff per quench
//! to `kappa` times the largest pair distance of the structure being
//! relaxed, a size-free rule that reads only the live structure. Every
//! evaluation of either phase is charged.

use std::sync::{Arc, Mutex};

use anneal_core::bias::BasinBias;
use anneal_core::coreclass::{CoreClassTable, CoreVerdict};
use anneal_core::corekey::motif_class;
use anneal_core::diversity::DiversityAnnealer;
use anneal_core::methods::bank::{Admission, Bank};
use anneal_core::methods::cluster_hopping::{
    AcceptedTransition, ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger,
    MoveLibrary, Outcome, random_cluster, run_with_bias_at_checkpoints,
};
use anneal_core::methods::cluster_search::{Encounter, median_encounter};
use anneal_core::methods::lattice_search::{LatticeSearchConfig, reoccupy};
use anneal_core::methods::csa_cluster::coordination_histogram_distance;
use anneal_core::methods::splice::cut_and_splice;
use anneal_core::methods::two_phase::{
    Cutoff, SharedSurfaceAllocator, SurfacePortfolio, TwoPhase, largest_pair_distance,
    penalty_axes, penalty_body, shared_surface_allocator,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Lennard-Jones value and gradient in reduced units, no cutoff.
fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            let inv12 = inv6 * inv6;
            e += 4.0 * (inv12 - inv6);
            let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= coef * d[k];
                g[3 * j + k] += coef * d[k];
            }
        }
    }
    (e, g)
}

/// Place the worst-bound atom just outside the current hull.
///
/// Copying the deepest population member parks every other chain on
/// that neighbour. A stall relocates one atom and the walk continues.
fn relocate_worst_atom(x: &[f64], count: usize, rng: &mut impl Rng) -> Array1<f64> {
    let n = x.len() / 3;
    let mut out = Array1::from(x.to_vec());
    if n < 2 || count == 0 {
        return out;
    }
    let mut bound = vec![0.0; n];
    let mut cm = [0.0; 3];
    for i in 0..n {
        cm[0] += x[3 * i];
        cm[1] += x[3 * i + 1];
        cm[2] += x[3 * i + 2];
    }
    let scale = n as f64;
    cm[0] /= scale;
    cm[1] /= scale;
    cm[2] /= scale;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < 1e-12 {
                continue;
            }
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            let vij = 4.0 * (inv6 * inv6 - inv6);
            bound[i] += vij;
            bound[j] += vij;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        bound[b]
            .partial_cmp(&bound[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let count = count.min(n.saturating_sub(1)).max(1);
    let mut radius: f64 = 0.0;
    for i in 0..n {
        let d0 = x[3 * i] - cm[0];
        let d1 = x[3 * i + 1] - cm[1];
        let d2 = x[3 * i + 2] - cm[2];
        radius = radius.max((d0 * d0 + d1 * d1 + d2 * d2).sqrt());
    }
    let bond = 2.0_f64.powf(1.0 / 6.0);
    for &atom in order.iter().take(count) {
        let z: f64 = rng.random_range(-1.0..1.0);
        let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
        let s = (1.0 - z * z).sqrt();
        let r = radius + bond;
        out[3 * atom] = cm[0] + r * s * phi.cos();
        out[3 * atom + 1] = cm[1] + r * s * phi.sin();
        out[3 * atom + 2] = cm[2] + r * z;
    }
    out
}

/// The pair potential the ensemble walks: reduced Lennard-Jones by default,
/// or Morse at the range parameter named by `POTENTIAL=morse:RHO`.
#[derive(Clone)]
enum Surface {
    LennardJones,
    Morse(PairPotential, f64),
}

impl Surface {
    fn from_environment(n: usize) -> Self {
        match std::env::var("POTENTIAL").ok().as_deref() {
            None | Some("lj") => Self::LennardJones,
            Some(spec) => {
                let rho: f64 = spec
                    .strip_prefix("morse:")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| panic!("POTENTIAL must be lj or morse:RHO, not {spec:?}"));
                Self::Morse(PairPotential::morse(n, rho), rho)
            }
        }
    }

    fn energy(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Self::LennardJones => lj(x),
            Self::Morse(pair, _) => pair.value_and_gradient(x),
        }
    }

    fn name(&self) -> String {
        match self {
            Self::LennardJones => "LJ".into(),
            Self::Morse(_, rho) => format!("Morse rho={rho}"),
        }
    }

    /// Published global minima, reporting only.
    fn reference(&self, n: usize) -> Option<f64> {
        match self {
            Self::LennardJones => reference(n),
            Self::Morse(_, rho) => match ((rho * 2.0).round() as i64, n) {
                (28, 38) => Some(-144.321054),
                (28, 55) => Some(-220.646208),
                (28, 75) => Some(-318.407330),
                (20, 38) => Some(-145.849817),
                (20, 55) => Some(-225.814286),
                (20, 75) => Some(-322.643558),
                (12, 38) => Some(-157.477108),
                (12, 55) => Some(-250.286609),
                (12, 75) => Some(-351.472365),
                _ => None,
            },
        }
    }
}

fn reference(n: usize) -> Option<f64> {
    Some(match n {
        13 => -44.326801,
        38 => -173.928427,
        55 => -279.248470,
        75 => -397.492331,
        98 => -543.665361,
        102 => -569.363652,
        104 => -582.086642,
        _ => return None,
    })
}

/// First-phase surface: the plain energy plus the library's two-phase penalty.
fn compressed(
    surface: &Surface,
    x: ArrayView1<f64>,
    mu: f64,
    diameter: f64,
    beta: f64,
    axes: [f64; 3],
    body: bool,
) -> (f64, Array1<f64>) {
    let (e, g) = surface.energy(x);
    let (pe, pg) = if body {
        penalty_body(x, diameter, beta, mu, axes)
    } else {
        penalty_axes(x, diameter, beta, mu, axes)
    };
    (e + pe, g + pg)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

/// What one chain publishes for the others to splice against.
#[derive(Clone, Default)]
struct Slot {
    energy: f64,
    state: Vec<f64>,
    best_energy: f64,
    best_state: Vec<f64>,
    /// First quenched structure this chain published. Lee, Lee and Scheraga
    /// keep that bank frozen and draw mix partners from it, so a later
    /// collapse of the live population still has something outside the funnel.
    first_energy: f64,
    first_state: Vec<f64>,
}

#[derive(Default, Clone, Copy)]
struct ExchangeTally {
    attempts: usize,
    adopted: usize,
    below_current: usize,
    external_calls: usize,
}

struct ChainReport {
    outcome: Outcome,
    charged: usize,
    tally: ExchangeTally,
    /// Charged evaluations at which the chain first reached the reference.
    first_hit: Option<usize>,
    /// Whether the transition that first reached the reference was a splice.
    hit_by_splice: bool,
}

#[derive(Clone)]
struct ExchangeConfig {
    enabled: bool,
    interval: usize,
    images: usize,
    partner_best: bool,
    source_best: bool,
    checkpoint: usize,
    /// Compression strength of the first quench phase; zero is a plain quench.
    /// Energy per length squared. Zero leaves the first phase unweighted.
    compress_mu: f64,
    /// Diameter penalty cutoff in sigma units; zero disables the penalty.
    diameter: f64,
    /// Energy per length to the fourth, on `(r^2 - D^2)^2`.
    diameter_beta: f64,
    /// Relative cutoff: `kappa` times the largest pair distance of the
    /// structure entering the quench; zero keeps the fixed cutoff.
    diameter_kappa: f64,
    /// Axis weights. In the laboratory frame the first entry is x and stays 1.
    /// In the inertia frame the first entry lies on the longest principal axis.
    diameter_wl: f64,
    diameter_wy: f64,
    diameter_wz: f64,
    /// Apply the axis weights in the structure's inertia frame.
    diameter_body: bool,
    /// Stretch factor applied to a spherical minimum before the kick. 0 disables it.
    sphere_stretch: f64,
    /// After a spherical incumbent stalls for 40 hops without improving,
    /// move this many worst-bound atoms onto the hull instead of the
    /// uniform kick. Zero disables the move.
    sphere_relocate: usize,
    /// How many atoms take the uniform kick. Zero moves every atom.
    kick_atoms: usize,
    /// Learned portfolio over surfaces (plain plus these), one arm held
    /// per block of hops; empty runs the fixed surface above.
    portfolio: Vec<TwoPhase>,
    /// Hops an arm is held for.
    portfolio_block: usize,
    /// Whether the chains of an ensemble share one portfolio posterior.
    portfolio_shared: bool,
    /// Fragment the surfaces across chains instead of learning inside one:
    /// chain `i` walks arm `i mod (1 + arms)` for its whole budget, the
    /// plain surface being arm zero.
    portfolio_split: bool,
    /// Quenched children are admitted by the bank. A child replaces the
    /// member it resembles, or the worst member when it resembles none.
    pbh: bool,
    /// `Dcut` starts at this multiple of the first bank's mean pairwise
    /// distance. Lee, Lee and Scheraga use one half (`PBH_DCUT`, default
    /// 0.5). The schedule then carries the cutoff to one fifth of that mean.
    pbh_dcut_scale: f64,
    /// Whether chains share a table of visited core keys and restart when
    /// the core they sit in has gone `core_patience` calls without any
    /// chain improving on it.
    core_tabu: bool,
    /// Calls a core is allowed without improvement before a chain in it
    /// restarts from a fresh random cluster.
    core_patience: usize,
    /// Calls a core class is allowed without any chain improving on it
    /// before chains in it that do not hold its best restart.
    core_tabu_calls: usize,
    /// Calls a chain spends in a fresh core before its best there is ranked
    /// against the trials of other chains in the same core class; below the
    /// median it continues, above it restarts. Zero disables the trial.
    core_trial: usize,
    /// Whether the chain rebuilds its surface from its interior on the
    /// lattice grown from that interior at every `reoccupy_interval` calls,
    /// quenches the rebuilt structure and adopts it when it is lower.
    reoccupy: bool,
    /// Calls between reoccupation attempts.
    reoccupy_interval: usize,
}

/// Ratio of the largest inertia eigenvalue to the smallest.
/// A value near 1 is a spherical cluster.
fn inertia_ratio(x: &[f64]) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return 1.0;
    }
    let mut cm = [0.0; 3];
    for i in 0..n {
        cm[0] += x[3 * i];
        cm[1] += x[3 * i + 1];
        cm[2] += x[3 * i + 2];
    }
    let scale = n as f64;
    for value in cm.iter_mut() {
        *value /= scale;
    }
    let mut tensor = ndarray::Array2::<f64>::zeros((3, 3));
    for i in 0..n {
        let v = [x[3 * i] - cm[0], x[3 * i + 1] - cm[1], x[3 * i + 2] - cm[2]];
        let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        for a in 0..3 {
            for b in 0..3 {
                tensor[[a, b]] += if a == b { r2 } else { 0.0 } - v[a] * v[b];
            }
        }
    }
    let (evals, _) = anneal_core::spectral::symmetric_eigen(tensor.view(), 8);
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    for value in evals.iter().copied() {
        if value.is_finite() {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    if !(lo > 1e-12 && hi.is_finite()) {
        return f64::INFINITY;
    }
    hi / lo
}

/// Stretch a spherical cluster along one principal axis before the kick.
fn stretch_if_spherical(x: &mut [f64], factor: f64) {
    if !(factor.is_finite() && factor > 1.0) || inertia_ratio(x) > 1.05 {
        return;
    }
    let n = x.len() / 3;
    let mut cm = [0.0; 3];
    for i in 0..n {
        cm[0] += x[3 * i];
        cm[1] += x[3 * i + 1];
        cm[2] += x[3 * i + 2];
    }
    let scale = n as f64;
    for value in cm.iter_mut() {
        *value /= scale;
    }
    let mut tensor = ndarray::Array2::<f64>::zeros((3, 3));
    for i in 0..n {
        let v = [x[3 * i] - cm[0], x[3 * i + 1] - cm[1], x[3 * i + 2] - cm[2]];
        let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        for a in 0..3 {
            for b in 0..3 {
                tensor[[a, b]] += if a == b { r2 } else { 0.0 } - v[a] * v[b];
            }
        }
    }
    let (_, vecs) = anneal_core::spectral::symmetric_eigen(tensor.view(), 8);
    // Column 0 is the smallest inertia eigenvalue, the longest axis.
    for i in 0..n {
        let mut along = 0.0;
        for k in 0..3 {
            along += vecs[[k, 0]] * (x[3 * i + k] - cm[k]);
        }
        let extra = (factor - 1.0) * along;
        for k in 0..3 {
            x[3 * i + k] += extra * vecs[[k, 0]];
        }
    }
}

fn radial_order(x: &[f64]) -> Vec<f64> {
    let n = x.len() / 3;
    let mut cm = [0.0; 3];
    for i in 0..n {
        cm[0] += x[3 * i];
        cm[1] += x[3 * i + 1];
        cm[2] += x[3 * i + 2];
    }
    let scale = (n as f64).max(1.0);
    for value in cm.iter_mut() {
        *value /= scale;
    }
    let mut radii = Vec::with_capacity(n);
    for i in 0..n {
        let d0 = x[3 * i] - cm[0];
        let d1 = x[3 * i + 1] - cm[1];
        let d2 = x[3 * i + 2] - cm[2];
        radii.push((d0 * d0 + d1 * d1 + d2 * d2).sqrt());
    }
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    radii
}

/// Ordered centre-of-mass distance with power 3.
fn ord_dissimilarity(a: &[f64], b: &[f64]) -> f64 {
    radial_order(a)
        .into_iter()
        .zip(radial_order(b))
        .map(|(u, v)| {
            let d = (u - v).abs();
            d * d * d
        })
        .sum()
}

fn dissimilarity_is_ord() -> bool {
    std::env::var("DISSIM").ok().as_deref() == Some("ord")
}

/// One bank slot per chain. Admission goes through [`anneal_core::methods::bank::Bank`],
/// so a child replaces the member it resembles, or the worst member when it
/// resembles none, and every other slot stays.
struct Population {
    bank: Bank,
    /// Chain that owns each bank slot, in bank order.
    owner: Vec<usize>,
    seeded: Vec<bool>,
    pending: Vec<Option<(f64, Vec<f64>)>>,
    cutoff_ready: bool,
    replacements_near: usize,
    replacements_far: usize,
    offers: usize,
    improved_at: Vec<u64>,
    tick: u64,
    /// Lee schedule: half the first-bank mean, down to one fifth of that mean.
    schedule: Option<DiversityAnnealer>,
}

fn population_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    if dissimilarity_is_ord() {
        let (Some(a), Some(b)) = (a.as_slice(), b.as_slice()) else {
            return f64::INFINITY;
        };
        return ord_dissimilarity(a, b);
    }
    let unit = 2f64.powf(1.0 / 6.0);
    coordination_histogram_distance(a, b, 1.25 * unit, 1.55 * unit)
}

impl Population {
    fn new(chains: usize) -> Self {
        Self {
            bank: Bank::new(chains.max(1), 1.0),
            owner: Vec::with_capacity(chains),
            seeded: vec![false; chains],
            pending: vec![None; chains],
            cutoff_ready: false,
            replacements_near: 0,
            replacements_far: 0,
            offers: 0,
            improved_at: vec![0; chains],
            tick: 0,
            schedule: None,
        }
    }

    fn cutoff(&self) -> Option<f64> {
        self.cutoff_ready.then_some(self.bank.dcut)
    }

    fn is_seeded(&self, chain: usize) -> bool {
        self.seeded.get(chain).copied().unwrap_or(false)
    }

    /// The chain's first minimum fills its slot. The cutoff is the caller
    /// factor times the mean pairwise distance of that full population.
    fn seed_chain(&mut self, p: usize, energy: f64, state: &[f64], factor: f64) {
        if self.seeded[p] || self.cutoff_ready {
            return;
        }
        let stored = Array1::from(state.to_vec());
        if !self.bank.seed(stored.view(), energy) {
            return;
        }
        self.owner.push(p);
        self.seeded[p] = true;
        self.tick = self.tick.saturating_add(1);
        self.improved_at[p] = self.tick;
        if self.seeded.iter().all(|&done| done) {
            self.calibrate(factor);
        }
    }

    fn calibrate(&mut self, factor: f64) {
        let scale = if factor.is_finite() && factor > 0.0 {
            factor
        } else {
            0.5
        };
        let slots: Vec<usize> = (0..self.bank.len()).collect();
        let schedule = DiversityAnnealer::scaled_from_population(
            &slots,
            |i, j| {
                population_distance(
                    self.bank.members()[i].state.view(),
                    self.bank.members()[j].state.view(),
                )
            },
            scale,
        );
        if let Some(schedule) = schedule {
            // Initial threshold is `factor * mean`. One fifth of the mean is
            // `0.2 / factor` of that threshold. At the Lee factor 0.5 this
            // floor fraction is 0.4.
            let floor = (0.2 / scale).clamp(1e-6, 1.0);
            let schedule = schedule.with_final_fraction(floor);
            self.bank.dcut = schedule.initial();
            self.schedule = Some(schedule);
        }
        self.cutoff_ready = true;
    }

    /// Move `Dcut` along the Lee schedule. `progress` is the fraction of
    /// this chain's charged budget already spent, in `[0, 1]`.
    fn set_progress(&mut self, progress: f64) {
        if let Some(schedule) = self.schedule.as_mut() {
            self.bank.dcut = schedule.threshold(progress.clamp(0.0, 1.0));
        }
    }

    /// Offer one quenched child. Returns whether some chain was told to move.
    ///
    /// The child is not written into slot `p` first. [`Bank::offer`] replaces
    /// the nearest member when the child is inside the cutoff and strictly
    /// better, and otherwise the worst member when the child resembles none
    /// and is strictly better.
    fn offer(&mut self, p: usize, energy: f64, state: &[f64], dcut_scale: f64) -> bool {
        if !self.is_seeded(p) {
            self.seed_chain(p, energy, state, dcut_scale);
            return false;
        }
        if !self.cutoff_ready {
            return false;
        }
        if std::env::var("PBH_HOLD").ok().as_deref() == Some("1") {
            return false;
        }
        let stored = Array1::from(state.to_vec());
        if std::env::var("PBH_NEAR_ONLY").ok().as_deref() == Some("1") {
            let nearest = self
                .bank
                .members()
                .iter()
                .map(|member| population_distance(stored.view(), member.state.view()))
                .fold(f64::INFINITY, f64::min);
            if nearest > self.bank.dcut {
                return false;
            }
        }
        if std::env::var("SAME_PACKING").ok().as_deref() == Some("1")
            && self.nearest_is_other_family(stored.view())
        {
            return false;
        }
        self.tick = self.tick.saturating_add(1);
        self.offers = self.offers.saturating_add(1);
        let admission = self.bank.offer(stored.view(), energy, population_distance);
        let replaced = match admission {
            Admission::Improved(slot) => {
                self.replacements_near += 1;
                Some(slot)
            }
            Admission::Displaced(slot) => {
                self.replacements_far += 1;
                Some(slot)
            }
            Admission::Duplicate(_) | Admission::Added(_) | Admission::Rejected => None,
        };
        if let Some(slot) = replaced {
            let chain = self.owner[slot];
            self.pending[chain] = Some((energy, state.to_vec()));
            self.improved_at[chain] = self.tick;
        }
        replaced.is_some()
    }

    fn nearest_is_other_family(&self, state: ArrayView1<f64>) -> bool {
        let mut nearest: Option<(f64, usize)> = None;
        for (slot, member) in self.bank.members().iter().enumerate() {
            let distance = population_distance(state, member.state.view());
            if nearest.is_none_or(|(best, _)| distance < best) {
                nearest = Some((distance, slot));
            }
        }
        let Some((distance, slot)) = nearest else {
            return false;
        };
        let Some(coords) = self.bank.members()[slot].state.as_slice() else {
            return false;
        };
        distance < self.bank.dcut
            && state
                .as_slice()
                .is_some_and(|child| anneal_core::catalog::different_decaf_family(child, coords))
    }

    /// A stalled chain takes the member that improved most recently.
    /// That member is further along the same descent. Filtering it out
    /// for being the same packing leaves only a different trap.
    fn pull_improving(&self, chain: usize, mine: f64) -> Option<Vec<f64>> {
        let my_tick = self.improved_at.get(chain).copied().unwrap_or(0);
        let mut chosen: Option<(u64, Vec<f64>)> = None;
        for (slot, member) in self.bank.members().iter().enumerate() {
            let Some(&owner) = self.owner.get(slot) else {
                continue;
            };
            if owner == chain {
                continue;
            }
            // A shallow leader that improved by a wiggle glues the ensemble
            // on one shelf. The donor has to be deeper.
            if member.energy + 0.5 >= mine {
                continue;
            }
            // A minimum just above the reference is deep enough to pass
            // the 0.5 test and then collects every stall. Two occupants
            // is enough. Later stalls keep their own walk.
            let occupants = self
                .bank
                .members()
                .iter()
                .filter(|other| (other.energy - member.energy).abs() < 1e-4)
                .count();
            if occupants >= 2 {
                continue;
            }
            let tick = self.improved_at.get(owner).copied().unwrap_or(0);
            if tick <= my_tick {
                continue;
            }
            if chosen.as_ref().is_none_or(|(best, _)| tick > *best) {
                chosen = Some((tick, member.state.to_vec()));
            }
        }
        chosen.map(|(_, state)| state)
    }

    fn take_pending(&mut self, chain: usize) -> Option<(f64, Vec<f64>)> {
        self.pending[chain].take()
    }
}

fn km_median_first_hit(records: &[(Option<usize>, usize)]) -> Option<usize> {
    let encounters = records
        .iter()
        .map(|(first_hit, charged)| match first_hit {
            Some(charged) => Encounter::Found {
                charged: *charged,
                hops: 0,
            },
            None => Encounter::Censored { charged: *charged },
        })
        .collect::<Vec<_>>();
    median_encounter(&encounters)
}

/// `SURFACES` items `mu:5`, `d:3.5` (pair-well units), `kappa:0.7`, with an
/// optional `:beta` suffix on the diameter forms.
fn parse_surfaces(spec: &str) -> Vec<TwoPhase> {
    let unit = 2f64.powf(1.0 / 6.0);
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|item| {
            let parts: Vec<&str> = item.split(':').collect();
            let value: f64 = parts
                .get(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("SURFACES item {item:?} needs a number"));
            let beta: f64 = parts.get(2).and_then(|v| v.parse().ok()).unwrap_or(1.0);
            match parts[0] {
                "mu" => TwoPhase {
                    cutoff: Cutoff::Fixed(0.0),
                    beta: 0.0,
                    mu: value,
                    anisotropic: false,
                },
                "d" => TwoPhase::diameter(value * unit, beta),
                "kappa" => TwoPhase::relative(value, beta),
                other => panic!("SURFACES item {item:?}: unknown kind {other:?}"),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_chain(
    n: usize,
    budget: usize,
    seed: u64,
    chain: usize,
    board: &Arc<Mutex<Vec<Slot>>>,
    exchange: ExchangeConfig,
    target: Option<f64>,
    resume: Option<Array1<f64>>,
    shared_surfaces: Option<SharedSurfaceAllocator>,
    population: Option<Arc<Mutex<Population>>>,
    cores: Option<Arc<Mutex<CoreClassTable>>>,
) -> ChainReport {
    let mut cfg = Config::recommended(n);
    if std::env::var("MOVE").ok().as_deref() == Some("wales") {
        cfg.move_library = MoveLibrary::WalesDoye;
        cfg.allocate_moves = false;
        cfg.depth_reward = false;
        cfg.tabu_on_stall = false;
    }
    if let Ok(temperature) = std::env::var("TEMP") {
        cfg.temperature = temperature
            .parse()
            .unwrap_or_else(|_| panic!("TEMP must be a finite temperature"));
    }
    let surface_kind = Surface::from_environment(n);
    let child_surface = surface_kind.clone();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut exchange_rng = StdRng::seed_from_u64(seed ^ 0x0057_11ce);
    let start = resume.unwrap_or_else(|| random_cluster(n, 0.7, cfg.min_separation, &mut rng));
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let compress_mu = exchange.compress_mu;
    let diameter = exchange.diameter;
    let beta = exchange.diameter_beta;
    let kappa = exchange.diameter_kappa;
    let axes = if exchange.diameter_body {
        [
            exchange.diameter_wl,
            exchange.diameter_wy,
            exchange.diameter_wz,
        ]
    } else {
        [1.0, exchange.diameter_wy, exchange.diameter_wz]
    };
    let body = exchange.diameter_body;
    let two_phase = compress_mu > 0.0 || ((diameter > 0.0 || kappa > 0.0) && beta > 0.0);
    let screen_steps = cfg.screen_steps;
    let split_surface = (exchange.portfolio_split && !exchange.portfolio.is_empty()).then(|| {
        let arms = 1 + exchange.portfolio.len();
        match chain % arms {
            0 => None,
            k => Some(exchange.portfolio[k - 1]),
        }
    });
    let mut portfolio = (!exchange.portfolio.is_empty() && !exchange.portfolio_split).then(|| {
        let mut portfolio =
            SurfacePortfolio::with_block(&exchange.portfolio, seed, exchange.portfolio_block);
        if let Some(shared) = shared_surfaces.clone() {
            // One declared source for the ensemble. Block rewards are
            // finite and carry the block's charged work, so the book
            // can credit them. The Cambridge energy is not this key.
            let source = anneal_core::surface_evidence::SourceTransferKey {
                descriptor_schema: "lj".into(),
                descriptor_version: 1,
                region: 1,
                proposal: "hop".into(),
                quench_schema: "lbfgs".into(),
                block: exchange.portfolio_block.max(1),
            };
            portfolio = portfolio
                .sharing(shared, source)
                .expect("shared surface source matches the portfolio block");
        }
        portfolio
    });
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        let before = led.spent();
        let screening = iters <= screen_steps;
        let mut start = x.to_owned();
        // The learned portfolio names the surface when present; otherwise
        // the fixed transform from the environment applies to every quench.
        let surface = match portfolio.as_mut() {
            Some(portfolio) => portfolio
                .begin(screening)
                .map(|two| (two.mu, two.cutoff_for(x), two.beta)),
            None if split_surface.is_some() => split_surface
                .flatten()
                .map(|two| (two.mu, two.cutoff_for(x), two.beta)),
            None => two_phase.then(|| {
                let cutoff = if kappa > 0.0 {
                    kappa * largest_pair_distance(x)
                } else {
                    diameter
                };
                (compress_mu, cutoff, beta)
            }),
        };
        if let Some((mu, cutoff, beta)) = surface {
            opt.forget();
            let (_, compressed, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                Some(compressed(&surface_kind, v, mu, cutoff, beta, axes, body))
            });
            start = compressed;
        }
        opt.forget();
        let (f, xr, _) = opt.minimize(start.view(), iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(surface_kind.energy(v))
        });
        if let Some(portfolio) = portfolio.as_mut() {
            portfolio.observe(screening, f, led.best);
        }
        led.record_quench_boundary(before, f, xr.clone(), None);
        (f, xr)
    };
    let mut bias = BasinBias::new(
        ClusterFingerprint::for_keying(n, cfg.shape_keyed),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let mut tally = ExchangeTally::default();
    let mut next_attempt = exchange.interval;
    let mut next_reoccupy = exchange.reoccupy_interval;
    let lattice_cfg = match &surface_kind {
        Surface::LennardJones => LatticeSearchConfig::lennard_jones(n),
        Surface::Morse(_, rho) => LatticeSearchConfig::morse(n, *rho),
    };
    let relax_steps = cfg.relax_steps;
    let temperature = cfg.temperature;
    let min_separation = cfg.min_separation;
    let mut child_opt = WarmLbfgs::default();
    let stall_restart = env_usize("STALL_RESTART", 0);
    let stall_adopt = env_usize("STALL_ADOPT", 0);
    let stall_handoff = env_usize("STALL_HANDOFF", 0);
    let mut best_mark = f64::INFINITY;
    let mut mark_hop = 0usize;
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        if snapshot.best_energy() + 1e-6 < best_mark {
            best_mark = snapshot.best_energy();
            mark_hop = snapshot.hops();
        }
        {
            let mut slots = board.lock().expect("ensemble board");
            let slot = &mut slots[chain];
            slot.energy = snapshot.current_energy();
            slot.state = snapshot.current_state().to_vec();
            slot.best_energy = snapshot.best_energy();
            if let Some(best) = snapshot.best_state() {
                slot.best_state = best.to_vec();
            }
            if slot.first_state.is_empty() {
                if !slot.best_state.is_empty() {
                    slot.first_state = slot.best_state.clone();
                    slot.first_energy = slot.best_energy;
                } else if !slot.state.is_empty() {
                    slot.first_state = slot.state.clone();
                    slot.first_energy = slot.energy;
                }
            }
        }
        if exchange.reoccupy && snapshot.charged() >= next_reoccupy {
            next_reoccupy = snapshot.charged() + exchange.reoccupy_interval;
            let mut private = Ledger::new(usize::MAX / 2);
            let rebuilt = reoccupy(&lattice_cfg, &mut private, snapshot.current_state());
            let mut external_calls = private.spent();
            child_opt.forget();
            let (energy, relaxed, _) = child_opt.minimize(rebuilt.view(), relax_steps, |v| {
                external_calls += 1;
                Some(child_surface.energy(v))
            });
            tally.attempts += 1;
            tally.external_calls += external_calls;
            if energy.is_finite() && energy < snapshot.current_energy() - 1e-6 {
                tally.adopted += 1;
                tally.below_current += 1;
                return CheckpointAction::ExternalAdopt {
                    state: relaxed,
                    action: "reoccupy".to_owned(),
                    external_calls,
                };
            }
            return CheckpointAction::ExternalWork { external_calls };
        }
        if let Some(cores) = cores.as_ref() {
            let class = motif_class(snapshot.current_state()).index();
            let mut table = cores.lock().expect("core table");
            let verdict = table.report(chain, class, snapshot.current_energy(), snapshot.charged());
            if verdict == CoreVerdict::Continue {
                return CheckpointAction::Continue;
            }
            drop(table);
            tally.adopted += 1;
            let fresh = random_cluster(n, 0.7, min_separation, &mut exchange_rng);
            return CheckpointAction::ExternalAdopt {
                state: fresh,
                action: "coretabu".to_owned(),
                external_calls: 0,
            };
        }
        if let Some(population) = population.as_ref() {
            let mut population = population.lock().expect("population");
            let spent = snapshot.charged();
            let left = snapshot.remaining();
            let progress = spent as f64 / (spent.saturating_add(left).max(1)) as f64;
            population.set_progress(progress);
            if !population.is_seeded(chain) {
                if let Some(current) = snapshot.current_state().as_slice() {
                    population.offer(
                        chain,
                        snapshot.current_energy(),
                        current,
                        exchange.pbh_dcut_scale,
                    );
                }
            } else {
                for boundary in snapshot.quench_boundaries() {
                    let quenched = boundary.state();
                    let Some(quenched) = quenched.as_slice() else {
                        continue;
                    };
                    tally.attempts += 1;
                    population.offer(chain, boundary.energy(), quenched, exchange.pbh_dcut_scale);
                }
            }
            if let Some((_, state)) = population.take_pending(chain) {
                tally.adopted += 1;
                return CheckpointAction::BoundaryProposal {
                    state: Array1::from(state),
                    action: "pbh".to_owned(),
                };
            }
            if stall_handoff > 0 && snapshot.hops().saturating_sub(mark_hop) >= stall_handoff {
                if let Some(state) = population.pull_improving(chain, snapshot.best_energy()) {
                    mark_hop = snapshot.hops();
                    tally.adopted += 1;
                    return CheckpointAction::BoundaryProposal {
                        state: Array1::from(state),
                        action: "handoff".to_owned(),
                    };
                }
            }
            if stall_adopt > 0 && snapshot.hops().saturating_sub(mark_hop) >= stall_adopt {
                // A kick of the whole cluster and a packing hop both quenched
                // back onto the same neighbour. Copying that neighbour parks
                // the ensemble there. Move the worst-bound atom instead.
                if let Some(origin) = snapshot
                    .best_state()
                    .as_ref()
                    .and_then(|mine| mine.as_slice())
                {
                    mark_hop = snapshot.hops();
                    tally.adopted += 1;
                    return CheckpointAction::BoundaryProposal {
                        state: relocate_worst_atom(origin, 1, &mut exchange_rng),
                        action: "exit".to_owned(),
                    };
                }
            }
            if stall_restart > 0 && snapshot.hops().saturating_sub(mark_hop) >= stall_restart {
                mark_hop = snapshot.hops();
                return CheckpointAction::BoundaryProposal {
                    state: random_cluster(n, 0.7, min_separation, &mut exchange_rng),
                    action: "restart".to_owned(),
                };
            }
            return CheckpointAction::Continue;
        }
        if !exchange.enabled || snapshot.charged() < next_attempt {
            return CheckpointAction::Continue;
        }
        next_attempt = snapshot.charged() + exchange.interval;
        let (mine, my_energy) = if exchange.source_best {
            match snapshot.best_state() {
                Some(best) => (best.to_owned(), snapshot.best_energy()),
                None => return CheckpointAction::Continue,
            }
        } else {
            (
                snapshot.current_state().to_owned(),
                snapshot.current_energy(),
            )
        };
        let from_first = exchange_rng.random::<bool>();
        let partner = {
            let slots = board.lock().expect("ensemble board");
            let candidates: Vec<(f64, &Vec<f64>)> = slots
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != chain)
                .map(|(_, slot)| {
                    if from_first && !slot.first_state.is_empty() {
                        (slot.first_energy, &slot.first_state)
                    } else if exchange.partner_best {
                        (slot.best_energy, &slot.best_state)
                    } else {
                        (slot.energy, &slot.state)
                    }
                })
                .filter(|(energy, state)| {
                    state.len() == mine.len()
                        && energy.is_finite()
                        && (energy - my_energy).abs() > 1e-6
                })
                .collect();
            if candidates.is_empty() {
                None
            } else if exchange.partner_best {
                candidates
                    .iter()
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, state)| Array1::from((*state).clone()))
            } else {
                let pick = exchange_rng.random_range(0..candidates.len());
                Some(Array1::from(candidates[pick].1.clone()))
            }
        };
        let Some(partner) = partner else {
            return CheckpointAction::Continue;
        };
        tally.attempts += 1;
        let mut external_calls = 0usize;
        let mut lowest: Option<(f64, Array1<f64>)> = None;
        for _ in 0..exchange.images.max(1) {
            let child = cut_and_splice(
                mine.view(),
                partner.view(),
                None,
                min_separation,
                &mut exchange_rng,
            );
            child_opt.forget();
            let (energy, relaxed, _) = child_opt.minimize(child.view(), relax_steps, |v| {
                external_calls += 1;
                Some(child_surface.energy(v))
            });
            if !energy.is_finite() {
                continue;
            }
            if lowest.as_ref().is_none_or(|(best, _)| energy < *best) {
                lowest = Some((energy, relaxed));
            }
        }
        tally.external_calls += external_calls;
        let Some((child_energy, child_state)) = lowest else {
            return CheckpointAction::ExternalWork { external_calls };
        };
        let current = snapshot.current_energy();
        if child_energy < current {
            tally.below_current += 1;
        }
        let accept = child_energy < current
            || exchange_rng.random::<f64>() < ((current - child_energy) / temperature).exp();
        if accept {
            tally.adopted += 1;
            CheckpointAction::ExternalAdopt {
                state: child_state,
                action: "splice".to_owned(),
                external_calls,
            }
        } else {
            CheckpointAction::ExternalWork { external_calls }
        }
    };
    let box_half = std::env::var("BOX").ok().and_then(|value| {
        if value == "1" {
            Some(0.25)
        } else {
            value.parse::<f64>().ok().filter(|half| *half > 0.0)
        }
    });
    let outcome = if let Some(half) = box_half {
        // One uniform kick and one two-phase local search per hop. The bank
        // decides which slot receives the quenched child. This chain kicks
        // from its own slot, which stays put when the child replaces another.
        let (mut energy, mut state) = relax(&mut ledger, start.view(), relax_steps);
        if let Some(population) = population.as_ref() {
            let mut population = population.lock().expect("population");
            if let Some(slice) = state.as_slice() {
                population.offer(chain, energy, slice, exchange.pbh_dcut_scale);
            }
        }
        let mut best = energy;
        let mut best_state = state.clone();
        let mut improvements = Vec::new();
        if energy.is_finite() {
            improvements.push((0usize, ledger.spent(), 0usize, energy));
        }
        let mut accepted_transitions = Vec::new();
        let mut hops = 0usize;
        // Hop index of the last incumbent improvement; relocation waits for a stall.
        let mut mark_hop = 0usize;
        const SPHERE_RELOCATE_STALL: usize = 40;
        while ledger.remaining() > 0 {
            if let Some(population) = population.as_ref() {
                let mut population = population.lock().expect("population");
                if let Some((offered, offered_state)) = population.take_pending(chain) {
                    if offered < energy - 1e-9 && offered_state.len() == state.len() {
                        accepted_transitions.push(AcceptedTransition {
                            hop: hops,
                            action: "pbh".to_owned(),
                            from_energy: energy,
                            to_energy: offered,
                            from_state: state.clone(),
                            from_gradient: None,
                            to_state: Array1::from(offered_state.clone()),
                            to_gradient: None,
                            validated: true,
                            adopted: true,
                        });
                        energy = offered;
                        state = Array1::from(offered_state);
                        if energy < best {
                            best = energy;
                            best_state = state.clone();
                            mark_hop = hops;
                            if improvements.len() < 512 {
                                improvements.push((hops, ledger.spent(), 0, energy));
                            }
                        }
                    }
                }
            }
            if ledger.remaining() == 0 {
                break;
            }
            hops += 1;
            let mut trial = state.clone();
            let spherical = state
                .as_slice()
                .is_some_and(|slice| inertia_ratio(slice) < 1.05);
            let stalled = hops.saturating_sub(mark_hop) >= SPHERE_RELOCATE_STALL;
            if exchange.sphere_relocate > 0 && spherical && stalled {
                if let Some(slice) = state.as_slice() {
                    trial = relocate_worst_atom(slice, exchange.sphere_relocate, &mut rng);
                }
            } else {
                if let Some(slice) = trial.as_slice_mut() {
                    stretch_if_spherical(slice, exchange.sphere_stretch);
                }
                let width = 2.0 * half;
                let n_atoms = trial.len() / 3;
                if exchange.kick_atoms > 0 && exchange.kick_atoms < n_atoms {
                    let mut order: Vec<usize> = (0..n_atoms).collect();
                    for i in 0..exchange.kick_atoms {
                        let j = rng.random_range(i..n_atoms);
                        order.swap(i, j);
                    }
                    for &atom in order.iter().take(exchange.kick_atoms) {
                        for k in 0..3 {
                            trial[3 * atom + k] += (rng.random::<f64>() - 0.5) * width;
                        }
                    }
                } else {
                    for coord in trial.iter_mut() {
                        *coord += (rng.random::<f64>() - 0.5) * width;
                    }
                }
            }
            let (child, child_state) = relax(&mut ledger, trial.view(), relax_steps);
            if let Some(population) = population.as_ref() {
                let mut population = population.lock().expect("population");
                if let Some(slice) = child_state.as_slice() {
                    population.offer(chain, child, slice, exchange.pbh_dcut_scale);
                }
                if let Some((offered, offered_state)) = population.take_pending(chain) {
                    if offered < energy - 1e-9 && offered_state.len() == state.len() {
                        accepted_transitions.push(AcceptedTransition {
                            hop: hops,
                            action: "pbh".to_owned(),
                            from_energy: energy,
                            to_energy: offered,
                            from_state: state.clone(),
                            from_gradient: None,
                            to_state: Array1::from(offered_state.clone()),
                            to_gradient: None,
                            validated: true,
                            adopted: true,
                        });
                        energy = offered;
                        state = Array1::from(offered_state);
                        if energy < best {
                            best = energy;
                            best_state = state.clone();
                            mark_hop = hops;
                            if improvements.len() < 512 {
                                improvements.push((hops, ledger.spent(), 0, energy));
                            }
                        }
                    }
                }
            } else if child < energy {
                accepted_transitions.push(AcceptedTransition {
                    hop: hops,
                    action: "box".to_owned(),
                    from_energy: energy,
                    to_energy: child,
                    from_state: state.clone(),
                    from_gradient: None,
                    to_state: child_state.clone(),
                    to_gradient: None,
                    validated: true,
                    adopted: true,
                });
                energy = child;
                state = child_state;
                if energy < best {
                    best = energy;
                    best_state = state.clone();
                    mark_hop = hops;
                    if improvements.len() < 512 {
                        improvements.push((hops, ledger.spent(), 0, energy));
                    }
                }
            }
        }
        Outcome {
            best,
            best_state: Some(best_state),
            final_state: Some(state),
            final_energy: energy,
            accepted_transitions,
            hops,
            charged: ledger.spent(),
            improvements,
            ..Outcome::default()
        }
    } else {
        run_with_bias_at_checkpoints(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            &mut bias,
            &mut rng,
            exchange.checkpoint,
            &mut checkpoint,
        )
    };
    let first_hit = target.and_then(|reference| {
        outcome
            .improvements
            .iter()
            .find(|&&(_, _, _, energy)| energy < reference + 1e-4)
            .map(|&(_, charged, _, _)| charged)
    });
    let hit_by_splice = target.is_some_and(|reference| {
        outcome
            .accepted_transitions
            .iter()
            .filter(|t| t.to_energy < reference + 1e-4)
            .min_by_key(|t| t.hop)
            .is_some_and(|t| t.action == "splice" || t.action == "pbh" || t.action == "reoccupy")
    });
    ChainReport {
        charged: ledger.spent(),
        outcome,
        tally,
        first_hit,
        hit_by_splice,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(38);
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let chains: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
    let ensembles: u64 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(4);
    let mode = args.get(5).cloned().unwrap_or_else(|| "indep".to_owned());
    let seed0: u64 = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(0);
    let enabled = match mode.as_str() {
        "indep" | "halving" | "shared" | "pbh" | "coretabu" => false,
        "splice" => true,
        other => {
            eprintln!(
                "unknown mode {other:?}: expected indep, splice, halving, shared, pbh or coretabu"
            );
            std::process::exit(2);
        }
    };
    let exchange = ExchangeConfig {
        enabled,
        interval: env_usize("SPLICE_INTERVAL", 5_000),
        images: env_usize("SPLICE_IMAGES", 4),
        partner_best: env_string("SPLICE_PARTNER", "random") == "best",
        source_best: env_string("SPLICE_SOURCE", "current") == "best",
        checkpoint: env_usize("CHECKPOINT", 500),
        compress_mu: env_f64("COMPRESS_MU", 0.0),
        // The published cutoff is quoted in pair-well minimum units; the
        // objective here is in sigma units, so the cutoff scales by 2^(1/6).
        diameter: env_f64("DIAMETER_D", 0.0) * 2f64.powf(1.0 / 6.0),
        diameter_beta: env_f64("DIAMETER_BETA", 1.0),
        diameter_kappa: env_f64("DIAMETER_KAPPA", 0.0),
        diameter_wl: env_f64("DIAMETER_WL", 1.0),
        diameter_wy: env_f64("DIAMETER_WY", 1.0),
        diameter_wz: env_f64("DIAMETER_WZ", 1.0),
        diameter_body: env_usize("DIAMETER_BODY", 0) == 1,
        sphere_stretch: env_f64("SPHERE_STRETCH", 0.0),
        sphere_relocate: env_usize("SPHERE_RELOCATE", 0),
        kick_atoms: env_usize("KICK_ATOMS", 0),
        portfolio: std::env::var("SURFACES")
            .map(|spec| parse_surfaces(&spec))
            .unwrap_or_default(),
        portfolio_block: env_usize("SURFACE_BLOCK", 100),
        portfolio_shared: mode == "shared",
        portfolio_split: env_usize("SURFACES_SPLIT", 0) == 1,
        pbh: mode == "pbh",
        pbh_dcut_scale: env_f64("PBH_DCUT", 0.5),
        core_tabu: mode == "coretabu",
        core_patience: env_usize("CORE_PATIENCE", 20_000),
        core_tabu_calls: env_usize("CORE_TABU", 50_000),
        core_trial: env_usize("CORE_TRIAL", 0),
        reoccupy: env_usize("REOCCUPY", 0) == 1,
        reoccupy_interval: env_usize("REOCCUPY_INTERVAL", 5_000),
    };
    let surface = Surface::from_environment(n);
    let target = surface.reference(n);
    println!(
        "{} N={n}, {chains} chains x {budget} charged, {ensembles} ensembles, mode {mode}, \
         interval {} images {} partner {} source {} checkpoint {} compress {} diameter {:.3} kappa {} beta {} portfolio {:?} block {} shared {}, reference {}",
        surface.name(),
        exchange.interval,
        exchange.images,
        if exchange.partner_best {
            "best"
        } else {
            "random"
        },
        if exchange.source_best {
            "best"
        } else {
            "current"
        },
        exchange.checkpoint,
        exchange.compress_mu,
        exchange.diameter,
        exchange.diameter_kappa,
        exchange.diameter_beta,
        exchange.portfolio,
        exchange.portfolio_block,
        exchange.portfolio_shared,
        target
            .map(|r| format!("{r:.6}"))
            .unwrap_or_else(|| "none".into())
    );

    if mode == "halving" {
        run_halving(n, budget, chains, ensembles, seed0, exchange, target);
        return;
    }
    let mut ensembles_solved = 0usize;
    let mut chains_solved = 0usize;
    let mut splice_hits = 0usize;
    let mut first_hits: Vec<usize> = Vec::new();
    let mut chain_encounters: Vec<(Option<usize>, usize)> = Vec::new();
    let mut tally = ExchangeTally::default();
    let mut total_charged = 0usize;
    for ensemble in seed0..(seed0 + ensembles) {
        let board = Arc::new(Mutex::new(vec![
            Slot {
                energy: f64::INFINITY,
                best_energy: f64::INFINITY,
                ..Slot::default()
            };
            chains
        ]));
        let shared = exchange
            .portfolio_shared
            .then(|| shared_surface_allocator(&exchange.portfolio));
        let population = exchange
            .pbh
            .then(|| Arc::new(Mutex::new(Population::new(chains))));
        let cores = exchange.core_tabu.then(|| {
            Arc::new(Mutex::new(
                CoreClassTable::new(exchange.core_patience, exchange.core_trial)
                    .with_class_tabu(exchange.core_tabu_calls)
                    .with_visit_charge(exchange.checkpoint),
            ))
        });
        let reports: Vec<ChainReport> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..chains)
                .map(|chain| {
                    let board = Arc::clone(&board);
                    let shared = shared.clone();
                    let population = population.clone();
                    let cores = cores.clone();
                    let exchange = exchange.clone();
                    let seed = ensemble
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(chain as u64)
                        .wrapping_add(7);
                    scope.spawn(move || {
                        run_chain(
                            n, budget, seed, chain, &board, exchange, target, None, shared,
                            population, cores,
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("chain thread"))
                .collect()
        });
        let deepest = reports
            .iter()
            .map(|r| r.outcome.best)
            .fold(f64::INFINITY, f64::min);
        let solved: Vec<usize> = reports
            .iter()
            .enumerate()
            .filter(|(_, r)| target.is_some_and(|t| r.outcome.best < t + 1e-4))
            .map(|(i, _)| i)
            .collect();
        let earliest = reports.iter().filter_map(|r| r.first_hit).min();
        let hops: usize = reports.iter().map(|r| r.outcome.hops).sum();
        let charged: usize = reports.iter().map(|r| r.charged).sum();
        total_charged += charged;
        for r in &reports {
            chain_encounters.push((r.first_hit, r.charged));
            tally.attempts += r.tally.attempts;
            tally.adopted += r.tally.adopted;
            tally.below_current += r.tally.below_current;
            tally.external_calls += r.tally.external_calls;
            if r.hit_by_splice {
                splice_hits += 1;
            }
        }
        if let Some(cores) = cores.as_ref() {
            let table = cores.lock().expect("core table");
            let mut deepest: Vec<(f64, usize)> = table
                .stats()
                .map(|(_, stat)| (stat.best, stat.visits))
                .collect();
            deepest.sort_by(|a, b| a.0.total_cmp(&b.0));
            println!(
                "      coretabu: {} cores seen, {} restarts, deepest cores {:?}",
                table.class_count(),
                table.restarts(),
                deepest
                    .iter()
                    .take(5)
                    .map(|(e, v)| format!("{e:.3}x{v}"))
                    .collect::<Vec<_>>()
            );
        }
        if let Some(population) = population.as_ref() {
            let population = population.lock().expect("population");
            println!(
                "      pbh: dcut {:?}, {} near replacements, {} far replacements",
                population.cutoff(),
                population.replacements_near,
                population.replacements_far
            );
        }
        chains_solved += solved.len();
        if !solved.is_empty() {
            ensembles_solved += 1;
        }
        if let Some(first) = earliest {
            first_hits.push(first);
        }
        let earliest_hops = reports
            .iter()
            .filter_map(|report| {
                target.and_then(|reference| {
                    report
                        .outcome
                        .improvements
                        .iter()
                        .find(|&&(_, _, _, energy)| energy < reference + 1e-4)
                        .map(|&(hop, _, _, _)| hop)
                })
            })
            .min();
        println!(
            "  ensemble {ensemble}: deepest {deepest:.6}  solved chains {:?}  first hit {}  first hop {}  hops {hops}  charged {charged}  splice attempts {} adopted {} below {} calls {}",
            solved,
            earliest
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".into()),
            earliest_hops
                .map(|hop| hop.to_string())
                .unwrap_or_else(|| "-".into()),
            reports.iter().map(|r| r.tally.attempts).sum::<usize>(),
            reports.iter().map(|r| r.tally.adopted).sum::<usize>(),
            reports.iter().map(|r| r.tally.below_current).sum::<usize>(),
            reports
                .iter()
                .map(|r| r.tally.external_calls)
                .sum::<usize>(),
        );
    }
    first_hits.sort_unstable();
    let conditional_parallel_latency = first_hits.get(first_hits.len() / 2).copied();
    let chain_km_median = km_median_first_hit(&chain_encounters);
    println!(
        "{ensembles_solved}/{ensembles} ensembles solved, {chains_solved}/{} chains solved, {splice_hits} first hits by splice, conditional median earliest-chain latency {}, chain KM median first-hit cost {}, splice attempts {} adopted {} below-current {} external calls {} ({:.2}% of charged)",
        chains * ensembles as usize,
        conditional_parallel_latency
            .map(|m| m.to_string())
            .unwrap_or_else(|| "-".into()),
        chain_km_median
            .map(|m| m.to_string())
            .unwrap_or_else(|| "-".into()),
        tally.attempts,
        tally.adopted,
        tally.below_current,
        tally.external_calls,
        100.0 * tally.external_calls as f64 / total_charged.max(1) as f64,
    );
}

/// Successive halving over chains at the independent arm's total charged
/// work.
///
/// One ensemble owns a pool of `chains * budget` charged evaluations. A
/// bracket launches `chains` fresh starts at the first rung `r0`, ranks them
/// by best energy, keeps the top `1/eta`, and continues the survivors from
/// their live states to the next rung `eta` times longer, until one rung
/// reaches the per-chain budget of the independent arm. Brackets repeat with
/// fresh starts until the pool is spent, so every retired walk hands its
/// unspent share to a new start rather than idling. `HALVING_ETA` (3) and
/// `HALVING_R0` (budget / eta^2) size the schedule.
#[allow(clippy::too_many_arguments)]
fn run_halving(
    n: usize,
    budget: usize,
    chains: usize,
    ensembles: u64,
    seed0: u64,
    exchange: ExchangeConfig,
    target: Option<f64>,
) {
    let eta = env_usize("HALVING_ETA", 3).max(2);
    let r0 = env_usize("HALVING_R0", (budget / (eta * eta)).max(1000));
    let mut rungs = Vec::new();
    let mut r = r0;
    while r < budget {
        rungs.push(r);
        r *= eta;
    }
    rungs.push(budget);
    println!(
        "  halving: eta {eta}, rungs {rungs:?}, pool {} per ensemble",
        chains * budget
    );
    let mut ensembles_solved = 0usize;
    let mut first_hits: Vec<usize> = Vec::new();
    let mut brackets_total = 0usize;
    let mut launches_total = 0usize;
    for ensemble in seed0..(seed0 + ensembles) {
        let mut pool = chains * budget;
        let mut spent = 0usize;
        let mut first_hit: Option<usize> = None;
        let mut deepest = f64::INFINITY;
        let mut launches = 0usize;
        let mut brackets = 0usize;
        let mut next_seed = ensemble.wrapping_mul(0x9E37_79B9).wrapping_add(7);
        // A bracket needs at least one first-rung launch per chain; below
        // that the remainder is not worth a start and the ensemble is done.
        while pool >= chains {
            brackets += 1;
            let pool_before = pool;
            // Live walks of this bracket: (state, best so far, seed).
            let mut live: Vec<(Option<Array1<f64>>, f64, u64)> = (0..chains)
                .map(|_| {
                    next_seed = next_seed.wrapping_add(1);
                    (None, f64::INFINITY, next_seed)
                })
                .collect();
            let mut cumulative = 0usize;
            for (rung_index, &rung) in rungs.iter().enumerate() {
                let slice = rung - cumulative;
                let count = live.len();
                if count == 0 || pool == 0 {
                    break;
                }
                // The pool caps the last rung of the last bracket.
                let per_chain = slice.min(pool / count.max(1));
                if per_chain == 0 {
                    break;
                }
                let board = Arc::new(Mutex::new(vec![
                    Slot {
                        energy: f64::INFINITY,
                        best_energy: f64::INFINITY,
                        ..Slot::default()
                    };
                    count
                ]));
                let reports: Vec<ChainReport> = std::thread::scope(|scope| {
                    let handles: Vec<_> = live
                        .iter()
                        .enumerate()
                        .map(|(chain, (state, _, seed))| {
                            let board = Arc::clone(&board);
                            let resume = state.clone();
                            let exchange = exchange.clone();
                            let seed = seed.wrapping_add(rung_index as u64 * 0x1000);
                            scope.spawn(move || {
                                run_chain(
                                    n, per_chain, seed, chain, &board, exchange, target, resume,
                                    None, None, None,
                                )
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().expect("chain thread"))
                        .collect()
                });
                launches += count;
                let rung_charged: usize = reports.iter().map(|r| r.charged).sum();
                if first_hit.is_none() {
                    // Rung walks run side by side, so the pool cost of the
                    // earliest hit is what every walk had spent by then.
                    if let Some(hit) = reports.iter().filter_map(|r| r.first_hit).min() {
                        first_hit = Some(spent + hit * count);
                    }
                }
                spent += rung_charged;
                pool = pool.saturating_sub(rung_charged);
                cumulative += per_chain;
                let mut ranked: Vec<(usize, f64)> = reports
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (i, r.outcome.best.min(live[i].1)))
                    .collect();
                ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
                deepest = deepest.min(ranked.first().map_or(f64::INFINITY, |r| r.1));
                let keep = if rung_index + 1 < rungs.len() {
                    count.div_ceil(eta)
                } else {
                    0
                };
                let mut survivors = Vec::with_capacity(keep);
                for &(i, best) in ranked.iter().take(keep) {
                    let state = reports[i]
                        .outcome
                        .final_state
                        .clone()
                        .or_else(|| reports[i].outcome.best_state.clone());
                    survivors.push((state, best, live[i].2));
                }
                live = survivors;
                if per_chain < slice {
                    break;
                }
            }
            if pool == pool_before {
                break;
            }
        }
        let solved = target.is_some_and(|t| deepest < t + 1e-4);
        if solved {
            ensembles_solved += 1;
        }
        if let Some(hit) = first_hit {
            first_hits.push(hit);
        }
        brackets_total += brackets;
        launches_total += launches;
        println!(
            "  ensemble {ensemble}: deepest {deepest:.6}  solved {solved}  first hit pool {}  spent {spent}  brackets {brackets}  launches {launches}",
            first_hit
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".into())
        );
    }
    first_hits.sort_unstable();
    let median = first_hits.get(first_hits.len() / 2).copied();
    println!(
        "{ensembles_solved}/{ensembles} ensembles solved (halving), median first hit pool {}, brackets {brackets_total}, launches {launches_total}",
        median.map(|m| m.to_string()).unwrap_or_else(|| "-".into())
    );
}

#[cfg(test)]
mod tests {
    use super::km_median_first_hit;

    #[test]
    fn chain_median_retains_budget_censoring() {
        let records = [(Some(10), 100), (None, 100), (None, 100)];

        assert_eq!(km_median_first_hit(&records), None);
    }
}
