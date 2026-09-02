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

use std::sync::{Arc, Mutex};

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, Outcome,
    random_cluster, run_with_bias_at_checkpoints,
};
use anneal_core::methods::splice::cut_and_splice;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
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

/// First-phase surface: the plain energy plus Doye's centroid compression
/// `mu * sum_i |r_i - r_cm|^2` and the Locatelli--Schoen diameter penalty
/// `beta * sum_{i<j} max(0, r_ij^2 - D^2)^2`. The centroid term contributes
/// no gradient of its own because the displacements from it sum to zero.
fn lj_compressed(x: ArrayView1<f64>, mu: f64, diameter: f64, beta: f64) -> (f64, Array1<f64>) {
    let (mut e, mut g) = lj(x);
    let n = x.len() / 3;
    if diameter > 0.0 && beta > 0.0 {
        let d2 = diameter * diameter;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let excess = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - d2;
                if excess > 0.0 {
                    e += beta * excess * excess;
                    let coef = 4.0 * beta * excess;
                    for k in 0..3 {
                        g[3 * i + k] += coef * d[k];
                        g[3 * j + k] -= coef * d[k];
                    }
                }
            }
        }
    }
    let mut cm = [0.0_f64; 3];
    for i in 0..n {
        for k in 0..3 {
            cm[k] += x[3 * i + k];
        }
    }
    for value in cm.iter_mut() {
        *value /= n.max(1) as f64;
    }
    for i in 0..n {
        for k in 0..3 {
            let d = x[3 * i + k] - cm[k];
            e += mu * d * d;
            g[3 * i + k] += 2.0 * mu * d;
        }
    }
    (e, g)
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

#[derive(Clone, Copy)]
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
}

/// Largest pair distance in a 3N coordinate vector.
fn largest_pair_distance(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut best = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                r2 += d * d;
            }
            best = best.max(r2);
        }
    }
    best.sqrt()
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
) -> ChainReport {
    let cfg = Config::recommended(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut exchange_rng = StdRng::seed_from_u64(seed ^ 0x5711_ce);
    let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let compress_mu = exchange.compress_mu;
    let diameter = exchange.diameter;
    let beta = exchange.diameter_beta;
    let kappa = exchange.diameter_kappa;
    let two_phase = compress_mu > 0.0 || ((diameter > 0.0 || kappa > 0.0) && beta > 0.0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        let before = led.spent();
        let mut start = x.to_owned();
        if two_phase {
            let cutoff = if kappa > 0.0 {
                kappa * largest_pair_distance(x)
            } else {
                diameter
            };
            opt.forget();
            let (_, compressed, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                Some(lj_compressed(v, compress_mu, cutoff, beta))
            });
            start = compressed;
        }
        opt.forget();
        let (f, xr, _) = opt.minimize(start.view(), iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(lj(v))
        });
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
    let relax_steps = cfg.relax_steps;
    let temperature = cfg.temperature;
    let min_separation = cfg.min_separation;
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
                Some(lj(v))
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
            .is_some_and(|t| t.action == "splice")
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
        "indep" => false,
        "splice" => true,
        other => {
            eprintln!("unknown mode {other:?}: expected indep or splice");
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
    };
    let target = reference(n);
    println!(
        "LJ{n}, {chains} chains x {budget} charged, {ensembles} ensembles, mode {mode}, \
         interval {} images {} partner {} source {} checkpoint {} compress {} diameter {:.3} kappa {} beta {}, reference {}",
        exchange.interval,
        exchange.images,
        if exchange.partner_best { "best" } else { "random" },
        if exchange.source_best { "best" } else { "current" },
        exchange.checkpoint,
        exchange.compress_mu,
        exchange.diameter,
        exchange.diameter_kappa,
        exchange.diameter_beta,
        target.map(|r| format!("{r:.6}")).unwrap_or_else(|| "none".into())
    );

    let mut ensembles_solved = 0usize;
    let mut chains_solved = 0usize;
    let mut splice_hits = 0usize;
    let mut first_hits: Vec<usize> = Vec::new();
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
        let reports: Vec<ChainReport> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..chains)
                .map(|chain| {
                    let board = Arc::clone(&board);
                    let seed = ensemble
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(chain as u64)
                        .wrapping_add(7);
                    scope.spawn(move || run_chain(n, budget, seed, chain, &board, exchange, target))
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
            tally.attempts += r.tally.attempts;
            tally.adopted += r.tally.adopted;
            tally.below_current += r.tally.below_current;
            tally.external_calls += r.tally.external_calls;
            if r.hit_by_splice {
                splice_hits += 1;
            }
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
            reports.iter().map(|r| r.tally.external_calls).sum::<usize>(),
        );
    }
    first_hits.sort_unstable();
    let median = first_hits.get(first_hits.len() / 2).copied();
    println!(
        "{ensembles_solved}/{ensembles} ensembles solved, {chains_solved}/{} chains solved, {splice_hits} first hits by splice, median first hit {}, splice attempts {} adopted {} below-current {} external calls {} ({:.2}% of charged)",
        chains * ensembles as usize,
        median.map(|m| m.to_string()).unwrap_or_else(|| "-".into()),
        tally.attempts,
        tally.adopted,
        tally.below_current,
        tally.external_calls,
        100.0 * tally.external_calls as f64 / total_charged.max(1) as f64,
    );
}
