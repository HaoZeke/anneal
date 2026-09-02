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
//! (default 0, plain quench), `DIAMETER_D` and `DIAMETER_BETA` add the
//! Locatelli--Schoen diameter penalty `beta * sum_{i<j} max(0, r_ij^2 - D^2)^2`
//! to the same first phase (`D` in units of the pair-well minimum distance,
//! default 0 and 1); `DIAMETER_KAPPA` instead sets the cutoff per quench
//! to `kappa` times the largest pair distance of the structure being
//! relaxed, a size-free rule that reads only the live structure. Every
//! evaluation of either phase is charged.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anneal_core::bias::BasinBias;
use anneal_core::corekey::{CoreRule, core_key_nn, motif_class};
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, Outcome, random_cluster,
    run_with_bias_at_checkpoints,
};
use anneal_core::methods::cluster_search::{Encounter, median_encounter};
use anneal_core::methods::lattice_search::{LatticeSearchConfig, reoccupy};
use anneal_core::methods::splice::cut_and_splice;
use anneal_core::methods::two_phase::{
    Cutoff, SharedSurfaceAllocator, SurfacePortfolio, TwoPhase, largest_pair_distance, penalty,
    shared_surface_allocator,
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
) -> (f64, Array1<f64>) {
    let (e, g) = surface.energy(x);
    let (pe, pg) = penalty(x, diameter, beta, mu);
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
    compress_mu: f64,
    /// Diameter penalty cutoff in sigma units; zero disables the penalty.
    diameter: f64,
    /// Diameter penalty strength.
    diameter_beta: f64,
    /// Relative cutoff: `kappa` times the largest pair distance of the
    /// structure entering the quench; zero keeps the fixed cutoff.
    diameter_kappa: f64,
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
    /// Population basin hopping: at every checkpoint a chain's live minimum
    /// is offered to the ensemble under the Grosso--Locatelli--Schoen
    /// replacement rule, and a chain told to move adopts the offered
    /// structure at its next checkpoint.
    pbh: bool,
    /// Replacement radius as a multiple of the ensemble's mean pairwise
    /// dissimilarity at the first exchange.
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

/// One core superbasin shared by the chains of an ensemble.
#[derive(Debug, Clone)]
struct CoreStat {
    /// Lowest energy any chain has seen with this core.
    best: f64,
    /// Calls spent in this core since it last improved.
    calls_since_improvement: usize,
    /// Checkpoints at which some chain sat in this core.
    visits: usize,
    /// Best energies of chains at the end of their trial in this core.
    trials: Vec<f64>,
}

/// Cores visited by an ensemble, keyed by the coloured ring-graph hash.
#[derive(Debug, Default)]
struct CoreTable {
    stats: HashMap<u64, CoreStat>,
    restarts: usize,
}

/// Coordination-shell histogram dissimilarity of Grosso, Locatelli and
/// Schoen: `H1[n]` counts atoms with exactly `n` neighbours inside the
/// first shell, `H2[n]` those with exactly `n` in the second shell, and the
/// distance is `sum_n n (2 |dH1| + |dH2|)`. Shell radii are the published
/// 1.25 and 1.55 pair-well units in sigma units.
fn shell_histograms(x: &[f64]) -> ([u32; 32], [u32; 32]) {
    let n = x.len() / 3;
    let unit = 2f64.powf(1.0 / 6.0);
    let (r1, r2) = (1.25 * unit, 1.55 * unit);
    let (r1sq, r2sq) = (r1 * r1, r2 * r2);
    let mut first = vec![0usize; n];
    let mut second = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                d2 += d * d;
            }
            if d2 < r1sq {
                first[i] += 1;
                first[j] += 1;
            } else if d2 < r2sq {
                second[i] += 1;
                second[j] += 1;
            }
        }
    }
    let mut h1 = [0u32; 32];
    let mut h2 = [0u32; 32];
    for i in 0..n {
        h1[first[i].min(31)] += 1;
        h2[second[i].min(31)] += 1;
    }
    (h1, h2)
}

fn shell_dissimilarity(a: &([u32; 32], [u32; 32]), b: &([u32; 32], [u32; 32])) -> f64 {
    (0..32)
        .map(|n| {
            n as f64
                * (2.0 * (a.0[n] as f64 - b.0[n] as f64).abs()
                    + (a.1[n] as f64 - b.1[n] as f64).abs())
        })
        .sum()
}

type Member = (f64, Vec<f64>, ([u32; 32], [u32; 32]));

/// The ensemble's population under the replacement rule: one member per
/// chain, a pending relocation per chain, and the cutoff once set.
#[derive(Default)]
struct Population {
    members: Vec<Option<Member>>,
    pending: Vec<Option<(f64, Vec<f64>)>>,
    dcut: Option<f64>,
    replacements_near: usize,
    replacements_far: usize,
}

impl Population {
    fn new(chains: usize) -> Self {
        Self {
            members: vec![None; chains],
            pending: vec![None; chains],
            ..Self::default()
        }
    }

    /// Offer chain `p`'s live minimum. Returns whether some chain was told
    /// to move.
    fn offer(&mut self, p: usize, energy: f64, state: &[f64], dcut_scale: f64) -> bool {
        let hist = shell_histograms(state);
        self.members[p] = Some((energy, state.to_vec(), hist));
        let filled: Vec<usize> = (0..self.members.len())
            .filter(|&i| self.members[i].is_some())
            .collect();
        if self.dcut.is_none() {
            if filled.len() < self.members.len() {
                return false;
            }
            // Every chain has reported once: the cutoff is a multiple of
            // the mean pairwise dissimilarity of that first population.
            let mut total = 0.0;
            let mut pairs = 0usize;
            for (a, &i) in filled.iter().enumerate() {
                for &j in &filled[a + 1..] {
                    let (hi, hj) = (
                        &self.members[i].as_ref().unwrap().2,
                        &self.members[j].as_ref().unwrap().2,
                    );
                    total += shell_dissimilarity(hi, hj);
                    pairs += 1;
                }
            }
            self.dcut = Some(dcut_scale * total / pairs.max(1) as f64);
            return false;
        }
        let dcut = self.dcut.unwrap();
        let mut nearest: Option<(usize, f64)> = None;
        for &q in &filled {
            if q == p {
                continue;
            }
            let d = shell_dissimilarity(&hist, &self.members[q].as_ref().unwrap().2);
            if nearest.is_none_or(|(_, best)| d < best) {
                nearest = Some((q, d));
            }
        }
        let Some((q, d)) = nearest else {
            return false;
        };
        if d < dcut {
            // Same region as q: only a better child displaces q.
            let eq = self.members[q].as_ref().unwrap().0;
            if energy < eq - 1e-9 {
                self.pending[q] = Some((energy, state.to_vec()));
                self.replacements_near += 1;
                return true;
            }
            return false;
        }
        // A new region: the worst member moves there if the child beats it.
        let worst = filled.iter().copied().filter(|&i| i != p).max_by(|&a, &b| {
            self.members[a]
                .as_ref()
                .unwrap()
                .0
                .total_cmp(&self.members[b].as_ref().unwrap().0)
        });
        if let Some(w) = worst
            && energy < self.members[w].as_ref().unwrap().0 - 1e-9
        {
            self.pending[w] = Some((energy, state.to_vec()));
            self.replacements_far += 1;
            return true;
        }
        false
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
    cores: Option<Arc<Mutex<CoreTable>>>,
) -> ChainReport {
    let cfg = Config::recommended(n);
    let surface_kind = Surface::from_environment(n);
    let child_surface = surface_kind.clone();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut exchange_rng = StdRng::seed_from_u64(seed ^ 0x5711_ce);
    let start = resume.unwrap_or_else(|| random_cluster(n, 0.7, cfg.min_separation, &mut rng));
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let compress_mu = exchange.compress_mu;
    let diameter = exchange.diameter;
    let beta = exchange.diameter_beta;
    let kappa = exchange.diameter_kappa;
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
            portfolio = portfolio.sharing(shared);
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
                Some(compressed(&surface_kind, v, mu, cutoff, beta))
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
    let species = vec![1u32; n];
    // CORE_KEY=motif keys the core table on the coarse five-fold class
    // instead of the per-minimum ring-graph hash.
    let motif_key = env_string("CORE_KEY", "ring") == "motif";
    // The chain's own progress inside its current core: the key it sits in,
    // its best energy there and the charged calls at that best.
    let mut own_key: Option<u64> = None;
    let mut own_best = f64::INFINITY;
    let mut own_best_at = 0usize;
    let mut own_entered_at = 0usize;
    let mut own_tried = false;
    let mut child_opt = WarmLbfgs::default();
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        {
            let mut slots = board.lock().expect("ensemble board");
            let slot = &mut slots[chain];
            slot.energy = snapshot.current_energy();
            slot.state = snapshot.current_state().to_vec();
            slot.best_energy = snapshot.best_energy();
            if let Some(best) = snapshot.best_state() {
                slot.best_state = best.to_vec();
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
            let key = if motif_key {
                u64::from(motif_class(snapshot.current_state()).index())
            } else {
                core_key_nn(
                    snapshot.current_state(),
                    &species,
                    CoreRule::NearMaximum { slack: 1 },
                )
                .key
            };
            let energy = snapshot.current_energy();
            let mut table = cores.lock().expect("core table");
            let stat = table.stats.entry(key).or_insert(CoreStat {
                best: f64::INFINITY,
                calls_since_improvement: 0,
                visits: 0,
                trials: Vec::new(),
            });
            stat.visits += 1;
            stat.calls_since_improvement += exchange.checkpoint;
            if energy < stat.best - 1e-6 {
                stat.best = energy;
                stat.calls_since_improvement = 0;
            }
            if own_key != Some(key) {
                own_key = Some(key);
                own_best = f64::INFINITY;
                own_best_at = snapshot.charged();
                own_entered_at = snapshot.charged();
                own_tried = false;
            }
            if energy < own_best - 1e-6 {
                own_best = energy;
                own_best_at = snapshot.charged();
            }
            let own_stalled =
                snapshot.charged().saturating_sub(own_best_at) >= exchange.core_patience;
            let class_tabu = stat.calls_since_improvement >= exchange.core_tabu_calls
                && energy > stat.best + 1e-6;
            let mut trial_lost = false;
            if exchange.core_trial > 0
                && !own_tried
                && snapshot.charged().saturating_sub(own_entered_at) >= exchange.core_trial
            {
                own_tried = true;
                stat.trials.push(own_best);
                let mut sorted = stat.trials.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let median = sorted[sorted.len() / 2];
                trial_lost = sorted.len() >= 4 && own_best > median + 1e-6;
            }
            if !own_stalled && !class_tabu && !trial_lost {
                return CheckpointAction::Continue;
            }
            own_key = None;
            table.restarts += 1;
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
            if let Some((_, state)) = population.take_pending(chain) {
                tally.adopted += 1;
                return CheckpointAction::BoundaryProposal {
                    state: Array1::from(state),
                    action: "pbh".to_owned(),
                };
            }
            if let Some(current) = snapshot.current_state().as_slice() {
                tally.attempts += 1;
                population.offer(
                    chain,
                    snapshot.current_energy(),
                    current,
                    exchange.pbh_dcut_scale,
                );
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
        let partner = {
            let slots = board.lock().expect("ensemble board");
            let candidates: Vec<(f64, &Vec<f64>)> = slots
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != chain)
                .map(|(_, slot)| {
                    if exchange.partner_best {
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
    let outcome = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        None,
        &mut bias,
        &mut rng,
        exchange.checkpoint,
        &mut checkpoint,
    );
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
        portfolio: std::env::var("SURFACES")
            .map(|spec| parse_surfaces(&spec))
            .unwrap_or_default(),
        portfolio_block: env_usize("SURFACE_BLOCK", 100),
        portfolio_shared: mode == "shared",
        portfolio_split: env_usize("SURFACES_SPLIT", 0) == 1,
        pbh: mode == "pbh",
        pbh_dcut_scale: env_f64("PBH_DCUT", 1.5),
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
        let cores = exchange
            .core_tabu
            .then(|| Arc::new(Mutex::new(CoreTable::default())));
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
                .stats
                .values()
                .map(|stat| (stat.best, stat.visits))
                .collect();
            deepest.sort_by(|a, b| a.0.total_cmp(&b.0));
            println!(
                "      coretabu: {} cores seen, {} restarts, deepest cores {:?}",
                table.stats.len(),
                table.restarts,
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
                population.dcut, population.replacements_near, population.replacements_far
            );
        }
        chains_solved += solved.len();
        if !solved.is_empty() {
            ensembles_solved += 1;
        }
        if let Some(first) = earliest {
            first_hits.push(first);
        }
        println!(
            "  ensemble {ensemble}: deepest {deepest:.6}  solved chains {:?}  first hit {}  hops {hops}  charged {charged}  splice attempts {} adopted {} below {} calls {}",
            solved,
            earliest
                .map(|c| c.to_string())
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
