#![allow(dead_code, unused_imports)]
//! One lattice-ensemble arm per difference from the published methods.
//!
//! Usage: `lattice_ensemble_ablation [lj38|lj75|lj98|morse80|all]`
//!
//! The full arm is [`Plan::default`]: a shared bank, one slot, splice 0.2,
//! retirement after 20 unimproved draws, the radial key, and a merge
//! distance held at D_ave/2. Each other arm changes one of those and leaves
//! the rest. `private` also uses two slots, because a one-slot private bank
//! has no donor and never splices. `splice_off` is the arm next to it.
//! There is no arm for the force ledger.
//!
//! `PAPER_SMOKE=1` runs two registered seeds, four chains and a short
//! budget, and writes a file whose name contains `SMOKE`. That output is
//! not a paper table. A run without it uses the registered seed block and
//! the budgets named below, and is still not a tagged v0.12.0 result.
//!
//! Columns, per arm: hits as n/N with a Wilson 95% interval, median force
//! calls to the hit with PAR-2 on a miss and the interquartile range, the
//! same for wall time, force calls per second, and the best energy when no
//! run hits. Each arm after the full one also reports the paired difference
//! of the hit indicators against the full arm on the same seeds, with a Wald
//! 95% interval, and the median paired PAR-2 force difference.

use std::io::Write;
use std::time::Instant;

use anneal_core::methods::lattice_ensemble::{self, CutoffSchedule, Plan, Resemblance, Sharing};
use anneal_core::methods::lattice_search::Quench;
use anneal_core::potentials::PairKind;

/// Hit window above the reference energy.
const HIT: f64 = 1e-4;
/// Smallest private bank that has a donor. One slot never splices.
const PRIVATE_SLOTS: usize = 2;
/// Normal quantile used for both intervals.
const Z95: f64 = 1.96;

struct Landscape {
    name: &'static str,
    n: usize,
    kind: PairKind,
    /// Reference energy. Scoring only. The search never reads it.
    reference: f64,
    seed0: u64,
    seeds: u64,
    budget: usize,
    seeds_from: &'static str,
}

struct Arm {
    name: &'static str,
    plan: Plan,
}

#[derive(Clone, Copy)]
struct Trial {
    hit: bool,
    /// Force calls to the hit, or twice the budget on a miss.
    par2_force: f64,
    /// Wall seconds to the hit, or twice the run's wall time on a miss.
    par2_wall: f64,
    force_calls: f64,
    wall_seconds: f64,
    best: f64,
    spliced: usize,
}

fn landscapes() -> Vec<Landscape> {
    vec![
        Landscape {
            name: "lj38",
            n: 38,
            kind: PairKind::LennardJones,
            // Cambridge Cluster Database value already scored by
            // `lj_lattice_ensemble`.
            reference: -173.928427,
            seed0: 3100,
            seeds: 100,
            // Default total of `lj_lattice_ensemble`, and the LJ38 campaign budget.
            budget: 400_000,
            seeds_from: "paper/seeds.toml r4-lattice-lj38, 3100-3199",
        },
        Landscape {
            name: "lj75",
            n: 75,
            kind: PairKind::LennardJones,
            reference: -397.492331,
            seed0: 3000,
            seeds: 30,
            budget: 4_000_000,
            seeds_from: "paper/seeds.toml r4-lattice-lj75-lj98, 3000-3029",
        },
        Landscape {
            name: "lj98",
            n: 98,
            kind: PairKind::LennardJones,
            reference: -543.665361,
            seed0: 3000,
            seeds: 30,
            budget: 4_000_000,
            seeds_from: "paper/seeds.toml r4-lattice-lj75-lj98, 3000-3029",
        },
        Landscape {
            name: "morse80",
            n: 80,
            kind: PairKind::Morse { rho: 14.0 },
            // 80G, the rho = 14 global minimum, fourth energy column of
            // Doye and Wales' Morse table (the columns are rho = 3, 6, 10, 14).
            // https://doye.chem.ox.ac.uk/jon/structures/Morse/tables.html
            reference: -340.811371,
            seed0: 2200,
            seeds: 48,
            // No Morse lattice budget is registered. This is the harness
            // default total of `lj_lattice_ensemble`.
            budget: 400_000,
            seeds_from: "paper/seeds.toml r2-held-out, 2200-2247",
        },
    ]
}

fn arms(chains: usize) -> Vec<Arm> {
    let full = Plan {
        chains,
        ..Plan::default()
    };
    vec![
        Arm {
            name: "full",
            plan: full.clone(),
        },
        Arm {
            name: "private",
            plan: Plan {
                sharing: Sharing::Private,
                slots: PRIVATE_SLOTS,
                ..full.clone()
            },
        },
        Arm {
            name: "splice_off",
            plan: Plan {
                splice: 0.0,
                ..full.clone()
            },
        },
        Arm {
            name: "retire_off",
            plan: Plan {
                retire: 0,
                ..full.clone()
            },
        },
        Arm {
            name: "lee_distance",
            plan: Plan {
                resemblance: Resemblance::LeeCsa,
                ..full.clone()
            },
        },
        Arm {
            name: "lee_anneal",
            plan: Plan {
                cutoff: CutoffSchedule::LeeFixedRatio,
                ..full.clone()
            },
        },
        Arm {
            name: "shao_lattice",
            plan: Plan {
                banked: false,
                ..full
            },
        },
    ]
}

fn score(run: &lattice_ensemble::Run, reference: f64, budget: usize) -> Trial {
    let hit_at = run
        .trace
        .iter()
        .find(|record| record.energy < reference + HIT);
    let hit = hit_at.is_some();
    Trial {
        hit,
        par2_force: hit_at.map_or(2.0 * budget as f64, |record| record.force_calls as f64),
        par2_wall: hit_at.map_or(2.0 * run.wall_seconds, |record| record.wall_seconds),
        force_calls: run.force_calls as f64,
        wall_seconds: run.wall_seconds,
        best: run.best,
        spliced: run.trials[2],
    }
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let weight = rank - lo as f64;
    sorted[lo] * (1.0 - weight) + sorted[hi] * weight
}

fn median_iqr(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = quantile(&sorted, 0.5);
    let iqr = quantile(&sorted, 0.75) - quantile(&sorted, 0.25);
    (median, iqr)
}

/// Wilson score interval for `hits` successes in `n` trials.
fn wilson(hits: usize, n: usize) -> (f64, f64, f64) {
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let nn = n as f64;
    let phat = hits as f64 / nn;
    let z2 = Z95 * Z95;
    let denom = 1.0 + z2 / nn;
    let centre = (phat + z2 / (2.0 * nn)) / denom;
    let margin = Z95 * ((phat * (1.0 - phat) / nn) + z2 / (4.0 * nn * nn)).sqrt() / denom;
    (phat, centre - margin, centre + margin)
}

/// Paired difference of hit indicators, arm minus full, with a Wald interval.
///
/// `n10` is full hit and arm miss. `n01` is full miss and arm hit.
/// `diff = (n01 - n10) / n`, and the standard error is
/// `sqrt((n10 + n01) - (n10 - n01)^2 / n) / n`.
fn paired_hits(full: &[Trial], arm: &[Trial]) -> (f64, f64, f64) {
    let n = full.len().min(arm.len());
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut n10 = 0.0;
    let mut n01 = 0.0;
    for i in 0..n {
        if full[i].hit && !arm[i].hit {
            n10 += 1.0;
        }
        if !full[i].hit && arm[i].hit {
            n01 += 1.0;
        }
    }
    let nn = n as f64;
    let diff = (n01 - n10) / nn;
    let se = ((n10 + n01) - (n10 - n01).powi(2) / nn).max(0.0).sqrt() / nn;
    (diff, diff - Z95 * se, diff + Z95 * se)
}

fn median_paired_par2(full: &[Trial], arm: &[Trial]) -> f64 {
    let values: Vec<f64> = full
        .iter()
        .zip(arm)
        .map(|(f, a)| a.par2_force - f.par2_force)
        .collect();
    median_iqr(&values).0
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn main() {}
