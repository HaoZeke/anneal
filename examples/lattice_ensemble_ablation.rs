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

fn command_output(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn fmt_opt(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        "-".to_owned()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(String::as_str).unwrap_or("all");
    let smoke = std::env::var("PAPER_SMOKE").ok().as_deref() == Some("1");
    let threads = args.get(2).and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let mut selected: Vec<Landscape> = landscapes()
        .into_iter()
        .filter(|landscape| which == "all" || landscape.name == which)
        .collect();
    if selected.is_empty() {
        eprintln!("unknown landscape {which:?}: expected lj38, lj75, lj98, morse80 or all");
        std::process::exit(2);
    }
    let chains = if smoke { 4 } else { Plan::default().chains };
    if smoke {
        for landscape in &mut selected {
            landscape.seeds = 2;
            landscape.budget = 8_000;
        }
    }
    let label = if smoke { "SMOKE" } else { "UNTAGGED" };
    let arm_list = arms(chains);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    let mut report = String::new();
    let header = format!(
        "label {label}\n\
         hardware {cpu}\n\
         cores {cores}\n\
         threads {threads}\n\
         rustc {rustc}\n\
         commit {commit}\n\
         start PresetSphere\n\
         lengths Lattice::for_pair of the landscape pair, in units of r_min\n\
         chains {chains}\n\
         private_slots {PRIVATE_SLOTS}\n\
         hit_window {HIT}\n\
         z95 {Z95}\n\
         note force ledger is accounting and has no arm\n",
        cpu = cpu_model(),
        cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        rustc = command_output("rustc", &["--version"]),
        commit = command_output("git", &["rev-parse", "HEAD"]),
    );
    report.push_str(&header);
    println!("{header}");

    for landscape in &selected {
        let seeds: Vec<u64> = (landscape.seed0..landscape.seed0 + landscape.seeds).collect();
        let started = Instant::now();
        let per_seed: Vec<Vec<Trial>> = pool.install(|| {
            use rayon::prelude::*;
            seeds
                .par_iter()
                .map(|&seed| {
                    arm_list
                        .iter()
                        .map(|arm| {
                            let run = lattice_ensemble::run(
                                landscape.n,
                                landscape.budget,
                                seed,
                                &arm.plan,
                                &Quench::default(),
                                landscape.kind,
                            );
                            score(&run, landscape.reference, landscape.budget)
                        })
                        .collect()
                })
                .collect()
        });
        let block = format!(
            "landscape {name} n {n} kind {kind} reference {reference:.6} seeds {seed0}+{count} ({from}) budget {budget} wall_s {wall:.3}\n",
            name = landscape.name,
            n = landscape.n,
            kind = kind_label(landscape.kind),
            reference = landscape.reference,
            seed0 = landscape.seed0,
            count = landscape.seeds,
            from = landscape.seeds_from,
            budget = landscape.budget,
            wall = started.elapsed().as_secs_f64(),
        );
        print!("{block}");
        report.push_str(&block);
        let columns: Vec<Vec<Trial>> = (0..arm_list.len())
            .map(|arm| per_seed.iter().map(|row| row[arm]).collect())
            .collect();
        for (index, arm) in arm_list.iter().enumerate() {
            let trials = &columns[index];
            let hits = trials.iter().filter(|trial| trial.hit).count();
            let (rate, wlo, whi) = wilson(hits, trials.len());
            let (force_med, force_iqr) = median_iqr(
                &trials
                    .iter()
                    .map(|trial| trial.par2_force)
                    .collect::<Vec<_>>(),
            );
            let (wall_med, wall_iqr) = median_iqr(
                &trials
                    .iter()
                    .map(|trial| trial.par2_wall)
                    .collect::<Vec<_>>(),
            );
            let force_sum: f64 = trials.iter().map(|trial| trial.force_calls).sum();
            let wall_sum: f64 = trials.iter().map(|trial| trial.wall_seconds).sum();
            let per_second = if wall_sum > 0.0 {
                force_sum / wall_sum
            } else {
                f64::NAN
            };
            let deepest = trials
                .iter()
                .map(|trial| trial.best)
                .fold(f64::INFINITY, f64::min);
            let best_if_none = if hits == 0 {
                fmt_opt(deepest)
            } else {
                "-".to_owned()
            };
            let spliced: usize = trials.iter().map(|trial| trial.spliced).sum();
            let mut line = format!(
                "arm {name} hits {hits}/{n} rate {rate:.6} wilson [{wlo:.6}, {whi:.6}] median_par2_force {fmed} iqr_force {fiqr} median_par2_wall {wmed} iqr_wall {wiqr} force_per_s {rate_s} best_if_none {best_if_none} spliced_trials {spliced}",
                name = arm.name,
                n = trials.len(),
                fmed = fmt_opt(force_med),
                fiqr = fmt_opt(force_iqr),
                wmed = fmt_opt(wall_med),
                wiqr = fmt_opt(wall_iqr),
                rate_s = fmt_opt(per_second),
            );
            if index > 0 {
                let (diff, lo, hi) = paired_hits(&columns[0], trials);
                let paired = median_paired_par2(&columns[0], trials);
                line.push_str(&format!(
                    " paired_diff {diff:.6} interval [{lo:.6}, {hi:.6}] median_paired_par2_force {paired}",
                    paired = fmt_opt(paired),
                ));
            }
            println!("{line}");
            report.push_str(&line);
            report.push('\n');
        }
        let _ = std::io::stdout().flush();
    }

    if let Ok(dir) = std::env::var("PAPER_OUT") {
        let name = if smoke {
            "lattice-ablation-SMOKE.txt"
        } else {
            "lattice-ablation-UNTAGGED.txt"
        };
        let path = std::path::Path::new(&dir).join(name);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(err) = std::fs::write(&path, &report) {
            eprintln!("could not write {}: {err}", path.display());
        }
    }
}

fn kind_label(kind: PairKind) -> String {
    match kind {
        PairKind::LennardJones => "LennardJones".to_owned(),
        PairKind::Morse { rho } => format!("Morse rho {rho}"),
    }
}
