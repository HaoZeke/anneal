//! Communicating lattice chains against the recommended chain at matched
//! ensemble force totals.
//!
//! Usage: `lj_lattice_ensemble <n> <total> <ensembles> <arm> [seed0] [threads]`
//!
//! Every arm spends `<total>` force calls per ensemble on per-chain ledgers
//! that sum to it:
//!
//! - `serial`: one recommended chain on the whole total.
//! - `indep`: `CHAINS` recommended chains (default 40) on `total / CHAINS`
//!   each, with no exchange.
//! - `shared`: the lattice ensemble of
//!   [`anneal_core::methods::lattice_ensemble`] with one bank.
//! - `private`: the same ensemble with one bank per chain.
//!
//! The recommended chain is the code path of `lj_ensemble_splice` in its
//! `indep` mode: [`Config::recommended`], a random start at density 0.7, the
//! warm L-BFGS quench charged per call, and a checkpoint every 500 calls that
//! changes nothing. Ensemble `s` seeds chain `c` with `s * 0x9E3779B9 + c + 7`
//! in every arm, so the arms are paired by ensemble.
//!
//! First-hit cost is in force calls of the whole ensemble. `serial` reports
//! its chain's ledger at the first hit; `indep` reports `CHAINS` times the
//! earliest chain's ledger at its hit, since the chains run side by side; a
//! lattice arm reports the calls of every earlier generation and of the
//! hitting generation's trials up to and including the hitting one, in chain
//! order, which any other order within that generation moves by less than one
//! generation's cost. Every value-and-gradient call is charged as one call;
//! lattice pair terms are charged at their fraction of the n(n-1)/2 pairs of a
//! full evaluation and settled in whole calls; under one call per chain can be
//! outstanding at a first hit or at the end of a run, while choosing a move
//! costs at most one call ([`anneal_core::methods::lattice_search`]).
//!
//! The lattice arms read `CHAINS`, `SLOTS`, `FRESH`, `SPLICE`, `MOVED_MIN`,
//! `MOVED_MAX`, `MERGE_START`, `MERGE_END`, `RETIRE`, `DENSITY`, `QUENCH_STEP`,
//! `QUENCH_TOL`, `QUENCH_MEMORY` and `LATTICE_CANDIDATES`.
//!
//! The Cambridge reference energy and the common-neighbour 555 fraction of
//! the end point are read after an ensemble finishes, to score and describe
//! it. Neither enters a search. Wall times go to standard error so that the
//! standard output of a replay is byte-identical.

use std::io::Write;
use std::time::Instant;

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, random_cluster,
    run_with_bias_at_checkpoints,
};
use anneal_core::methods::cluster_search::{Encounter, median_encounter};
use anneal_core::methods::lattice_ensemble::{self, Plan, Sharing};
use anneal_core::methods::lattice_search::{Lattice, Quench};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::structure::cna_descriptor;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Hit tolerance above the reference energy.
const HIT: f64 = 1e-4;

/// Lennard-Jones value and gradient in reduced units, no cutoff.
///
/// The summation order is that of `lj_ensemble_splice`, so the recommended
/// arms replay its chains bit for bit.
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
        _ => return None,
    })
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

fn f555(state: &[f64], n: usize) -> f64 {
    if state.len() != 3 * n {
        return f64::NAN;
    }
    cna_descriptor(ArrayView1::from(state), n, 1.39)[0]
}

struct ChainResult {
    best: f64,
    best_state: Vec<f64>,
    first_hit: Option<usize>,
    charged: usize,
    hops: usize,
}

/// One recommended chain on `budget` calls, scored after it ends.
fn recommended_chain(n: usize, budget: usize, seed: u64, target: Option<f64>) -> ChainResult {
    let cfg = Config::recommended(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        let before = led.spent();
        opt.forget();
        let (f, xr, _) = opt.minimize(x, iters, |v| {
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
    let mut checkpoint = |_: ChainCheckpoint<'_>| CheckpointAction::Continue;
    let outcome = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        None,
        &mut bias,
        &mut rng,
        500,
        &mut checkpoint,
    );
    let first_hit = target.and_then(|reference| {
        outcome
            .improvements
            .iter()
            .find(|&&(_, _, _, energy)| energy < reference + HIT)
            .map(|&(_, charged, _, _)| charged)
    });
    ChainResult {
        best: outcome.best,
        best_state: outcome.best_state.map(|s| s.to_vec()).unwrap_or_default(),
        first_hit,
        charged: ledger.spent(),
        hops: outcome.hops,
    }
}

/// What the summary needs from one ensemble.
struct Scored {
    deepest: f64,
    solved: bool,
    first_hit: Option<usize>,
    charged: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(38);
    let total: usize = args
        .get(2)
        .and_then(|v| v.parse::<f64>().ok())
        .map_or(400_000, |v| v as usize);
    let ensembles: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(10);
    let arm = args.get(4).cloned().unwrap_or_else(|| "shared".to_owned());
    let seed0: u64 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(900);
    let threads: usize = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(4);
    if !matches!(arm.as_str(), "serial" | "indep" | "shared" | "private") {
        eprintln!("unknown arm {arm:?}: expected serial, indep, shared or private");
        std::process::exit(2);
    }
    let chains = if arm == "serial" {
        1
    } else {
        env_usize("CHAINS", 40).max(1)
    };
    let target = reference(n);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    let seeds: Vec<u64> = (seed0..seed0 + ensembles).collect();
    let lattice_arm = matches!(arm.as_str(), "shared" | "private");
    let plan = Plan {
        chains,
        slots: env_usize("SLOTS", Plan::default().slots),
        fresh: env_f64("FRESH", Plan::default().fresh),
        splice: env_f64("SPLICE", Plan::default().splice),
        moved: (
            env_usize("MOVED_MIN", Plan::default().moved.0),
            env_usize("MOVED_MAX", Plan::default().moved.1),
        ),
        merge: (
            env_f64("MERGE_START", Plan::default().merge.0),
            env_f64("MERGE_END", Plan::default().merge.1),
        ),
        retire: env_usize("RETIRE", Plan::default().retire),
        density: env_f64("DENSITY", Plan::default().density),
        sharing: if arm == "private" {
            Sharing::Private
        } else {
            Sharing::Shared
        },
        ..Plan::default()
    };
    let quench = Quench {
        max_step: env_f64("QUENCH_STEP", Quench::default().max_step),
        rms_tolerance: env_f64("QUENCH_TOL", Quench::default().rms_tolerance),
        memory: env_usize("QUENCH_MEMORY", Quench::default().memory),
        ..Quench::default()
    };
    let lattice = Lattice {
        candidates: env_usize("LATTICE_CANDIDATES", Lattice::default().candidates),
        ..Lattice::default()
    };
    println!(
        "LJ N={n}, arm {arm}, total {total} per ensemble, {chains} chains x {} charged, \
         {ensembles} ensembles from seed {seed0}, threads {threads}, reference {}",
        total / chains,
        target.map_or_else(|| "none".to_owned(), |r| format!("{r:.6}")),
    );
    if lattice_arm {
        println!(
            "  plan: slots {} fresh {} splice {} moved {}..{} merge {}->{} retire {} density {} min_separation {}; \
             quench step {} tol {} memory {} iterations {}; lattice candidates {} bond {:.4} hollow {:.4} site {:.4} clearance {:.4}",
            plan.slots,
            plan.fresh,
            plan.splice,
            plan.moved.0,
            plan.moved.1,
            plan.merge.0,
            plan.merge.1,
            plan.retire,
            plan.density,
            plan.min_separation,
            quench.max_step,
            quench.rms_tolerance,
            quench.memory,
            quench.max_iterations,
            lattice.candidates,
            lattice.bond_cutoff,
            lattice.hollow_cutoff,
            lattice.site_distance,
            lattice.clearance,
        );
    } else {
        println!("  recommended chain: Config::recommended({n}), checkpoint 500, density 0.7");
    }
    let show_bank = env_usize("SHOW_BANK", 0) == 1;
    let chunk = if arm == "indep" { 1 } else { threads.max(1) };
    let mut scored: Vec<Scored> = Vec::new();
    for group in seeds.chunks(chunk) {
        let clock = Instant::now();
        let lines: Vec<(String, Scored)> = if lattice_arm {
            pool.install(|| {
                group
                    .par_iter()
                    .map(|&seed| {
                        let run = lattice_ensemble::run(n, total, seed, &plan, &quench, &lattice);
                        let solved = target.is_some_and(|r| run.best < r + HIT);
                        let hit = target.and_then(|r| {
                            run.trace.iter().find(|record| record.energy < r + HIT)
                        });
                        let low = run.bank.iter().map(|m| m.energy).fold(f64::INFINITY, f64::min);
                        let high = run
                            .bank
                            .iter()
                            .map(|m| m.energy)
                            .fold(f64::NEG_INFINITY, f64::max);
                        let line = format!(
                            "  ensemble {seed}: deepest {:.6}  solved {solved}  first hit {}  charged {}  \
                             f555 {:.3}  hit by {}  generations {}  trials {}/{}/{}  calls {}/{}/{}  \
                             admitted {}/{}/{}  bank {} in [{:.6}, {:.6}]  trace {}",
                            run.best,
                            hit.map_or_else(|| "-".to_owned(), |h| h.charged.to_string()),
                            run.charged,
                            f555(&run.best_state, n),
                            hit.map_or("-", |h| h.origin.label()),
                            run.generations,
                            run.trials[0],
                            run.trials[1],
                            run.trials[2],
                            run.calls[0],
                            run.calls[1],
                            run.calls[2],
                            run.admitted[0],
                            run.admitted[1],
                            run.admitted[2],
                            run.bank.len(),
                            low,
                            high,
                            run.trace.len(),
                        );
                        let mut line = line;
                        if show_bank {
                            let mut members: Vec<_> = run.bank.iter().collect();
                            members.sort_by(|a, b| a.energy.total_cmp(&b.energy));
                            for m in members {
                                line.push_str(&format!(
                                    "\n      member {:.6}  f555 {:.3}  draws {}  from {}",
                                    m.energy,
                                    f555(&m.state, n),
                                    m.draws,
                                    m.origin.label()
                                ));
                            }
                        }
                        (
                            line,
                            Scored {
                                deepest: run.best,
                                solved,
                                first_hit: hit.map(|h| h.charged),
                                charged: run.charged,
                            },
                        )
                    })
                    .collect()
            })
        } else {
            let share = total / chains;
            let jobs: Vec<(u64, usize)> = group
                .iter()
                .flat_map(|&seed| (0..chains).map(move |c| (seed, c)))
                .collect();
            let results: Vec<ChainResult> = pool.install(|| {
                jobs.par_iter()
                    .map(|&(seed, c)| {
                        recommended_chain(n, share, lattice_ensemble::chain_seed(seed, c), target)
                    })
                    .collect()
            });
            group
                .iter()
                .enumerate()
                .map(|(k, &seed)| {
                    let reports = &results[k * chains..(k + 1) * chains];
                    let (deepest_chain, deepest) = reports
                        .iter()
                        .enumerate()
                        .map(|(c, r)| (c, r.best))
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .expect("at least one chain");
                    let solved_chains: Vec<usize> = reports
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| target.is_some_and(|t| r.best < t + HIT))
                        .map(|(c, _)| c)
                        .collect();
                    // Chains run side by side, so the ensemble has spent
                    // `chains` times the earliest chain's cost at its hit.
                    let first_hit = reports
                        .iter()
                        .filter_map(|r| r.first_hit)
                        .min()
                        .map(|h| h * chains);
                    let charged: usize = reports.iter().map(|r| r.charged).sum();
                    let hops: usize = reports.iter().map(|r| r.hops).sum();
                    let line = format!(
                        "  ensemble {seed}: deepest {deepest:.6}  solved {}  first hit {}  charged {charged}  \
                         f555 {:.3}  solved chains {:?}  hops {hops}",
                        !solved_chains.is_empty(),
                        first_hit.map_or_else(|| "-".to_owned(), |h| h.to_string()),
                        f555(&reports[deepest_chain].best_state, n),
                        solved_chains,
                    );
                    (
                        line,
                        Scored {
                            deepest,
                            solved: !solved_chains.is_empty(),
                            first_hit,
                            charged,
                        },
                    )
                })
                .collect()
        };
        for (line, score) in lines {
            println!("{line}");
            scored.push(score);
        }
        std::io::stdout().flush().ok();
        eprintln!(
            "  [{} ensembles in {:.1} s]",
            group.len(),
            clock.elapsed().as_secs_f64()
        );
    }
    let solved = scored.iter().filter(|s| s.solved).count();
    let encounters: Vec<Encounter> = scored
        .iter()
        .map(|s| match s.first_hit {
            Some(charged) => Encounter::Found { charged, hops: 0 },
            None => Encounter::Censored { charged: s.charged },
        })
        .collect();
    let mut failures: Vec<f64> = scored
        .iter()
        .filter(|s| !s.solved)
        .map(|s| s.deepest)
        .collect();
    failures.sort_by(f64::total_cmp);
    let charged: usize = scored.iter().map(|s| s.charged).sum();
    println!(
        "{solved}/{} ensembles solved, KM median first-hit cost {}, charged {charged}, failures [{}]",
        scored.len(),
        median_encounter(&encounters).map_or_else(|| "-".to_owned(), |m| m.to_string()),
        failures
            .iter()
            .map(|e| format!("{e:.6}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
}
