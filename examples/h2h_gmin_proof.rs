// Copyright (c) anneal developers

//! One seed of `Config::recommended` on a paper Lennard-Jones size.
//!
//! ```text
//! h2h_gmin_proof run <system> <seed>
//! h2h_gmin_proof coords <system> <seed> <path>
//! h2h_gmin_proof energy <system> <path>
//! h2h_gmin_proof summarize <rows.csv> <summary.csv>
//! ```
//!
//! `run` charges the ledger the way `paper_controls` does: one charge for
//! each energy-and-gradient pair inside a quench, and one charge for each
//! gradient the climb reads. The search stops when the quenched energy is
//! at or below the published minimum plus 0.001, or when the budget is
//! spent. A miss is reported as the full budget. The configuration is
//! `Config::recommended` with no further change.
//!
//! `coords` writes the start that `run` quenches, one Cartesian line per
//! atom, so another code can open the same geometry.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "coords" => cmd_coords(&args),
        "energy" => cmd_energy(&args),
        "budget" => cmd_budget(&args),
        "summarize" => cmd_summarize(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("run needs a system");
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let budget = env_budget().unwrap_or(spec.budget);
    let cfg = Config::recommended(spec.n);
    let start = start_of(&cfg, spec.n, seed);
    if let Some(path) = args.get(3) {
        if let Err(err) = write_coords(path, &start) {
            eprintln!("coords: {err}");
            return 1;
        }
    }
    let (outcome, wall) = search(&cfg, spec.n, &start, seed, budget, spec.reference);
    let level = spec.reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = evals_to_hit(hit, at_hit.or(hit.then_some(outcome.charged)), budget);
    println!("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins");
    println!(
        "{system},anneal,{seed},{},{hit},{evals},{},{wall:.6},{:.8},{},{},{},{}",
        budget,
        outcome.charged,
        outcome.best,
        git(&["rev-parse", "HEAD"]),
        git(&["rev-parse", "HEAD^{tree}"]),
        cpu_model(),
        cpu_pins(),
    );
    0
}

fn cmd_coords(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("coords needs a system");
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("coords needs a seed");
        return 2;
    };
    let Some(path) = args.get(3) else {
        eprintln!("coords needs a path");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let cfg = Config::recommended(spec.n);
    let start = start_of(&cfg, spec.n, seed);
    match write_coords(path, &start) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("coords: {err}");
            1
        }
    }
}

fn cmd_budget(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("budget needs a system");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    println!("{}", spec.budget);
    println!("{}", spec.reference);
    0
}

fn env_budget() -> Option<usize> {
    let text = std::env::var("H2H_BUDGET").ok()?;
    let budget = text.parse::<usize>().ok()?;
    (budget > 0).then_some(budget)
}

fn cmd_energy(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("energy needs a system");
        return 2;
    };
    let Some(path) = args.get(2) else {
        eprintln!("energy needs a coordinate file");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let coords = match read_coords(path) {
        Ok(coords) => coords,
        Err(err) => {
            eprintln!("energy: {err}");
            return 1;
        }
    };
    if coords.len() != 3 * spec.n {
        eprintln!("energy: expected {} numbers", 3 * spec.n);
        return 2;
    }
    let pot = PairPotential::lennard_jones(spec.n);
    let (energy, _) = pot.value_and_gradient(coords.view());
    println!("{energy:.10}");
    0
}

fn cmd_summarize(args: &[String]) -> i32 {
    let Some(rows_path) = args.get(1) else {
        eprintln!("summarize needs a rows file");
        return 2;
    };
    let Some(out_path) = args.get(2) else {
        eprintln!("summarize needs an output path");
        return 2;
    };
    let text = match std::fs::read_to_string(rows_path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("summarize: {err}");
            return 1;
        }
    };
    let rows = match parse_rows(&text) {
        Ok(rows) => rows,
        Err(err) => {
            eprintln!("summarize: {err}");
            return 2;
        }
    };
    let summary = summarize(&rows);
    let mut out = String::from(
        "system,code,n,hits,hits_over_n,budget,evals_median,evals_iqr,wall_median,wall_iqr,head\n",
    );
    for row in summary {
        out.push_str(&format!(
            "{},{},{},{},{}/{},{},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.system,
            row.code,
            row.n,
            row.hits,
            row.hits,
            row.n,
            row.budget,
            row.evals_median,
            row.evals_iqr,
            row.wall_median,
            row.wall_iqr,
            row.head,
        ));
    }
    if let Err(err) = std::fs::write(out_path, out) {
        eprintln!("summarize: {err}");
        return 1;
    }
    0
}

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    let (n, reference) = match name {
        "lj38" => (38, -173.928427),
        "lj55" => (55, -279.248470),
        "lj75" => (75, -397.492331),
        "lj98" => (98, -543.665361),
        "lj104" => (104, -582.086642),
        _ => return None,
    };
    Some(Spec {
        n,
        budget: paper_budget(n)?,
        reference,
    })
}

/// Budgets from `scripts/elja_submit_jcc_lj.sh`.
fn paper_budget(n: usize) -> Option<usize> {
    match n {
        38 => Some(400_000),
        55 => Some(1_000_000),
        75 | 98 | 102 | 104 => Some(4_000_000),
        _ => None,
    }
}

fn start_of(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn search(
    cfg: &Config,
    n: usize,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
    reference: f64,
) -> (Outcome, f64) {
    let potential = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(v))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(v).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    (outcome, started.elapsed().as_secs_f64())
}

/// A hit keeps the charged count at the first quenched record at or below
/// the level. A miss is the whole budget.
fn evals_to_hit(hit: bool, calls_at_hit: Option<usize>, budget: usize) -> usize {
    if hit {
        calls_at_hit.unwrap_or(budget)
    } else {
        budget
    }
}

/// Hyndman-Fan type 7 quantile. `sorted` is ascending. `p` is in `[0, 1]`.
fn type7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    assert!(n >= 1, "type 7 needs a sample");
    if n == 1 {
        return sorted[0];
    }
    let h = 1.0 + (n as f64 - 1.0) * p;
    let lower = h.floor();
    let frac = h - lower;
    let index = (lower as usize).saturating_sub(1);
    if frac == 0.0 {
        sorted[index]
    } else {
        sorted[index] * (1.0 - frac) + sorted[index + 1] * frac
    }
}

#[cfg(test)]
mod coverage_checks {
    use super::{evals_to_hit, paper_budget, type7};

    #[test]
    fn paper_budget_names_the_published_sizes() {
        assert_eq!(paper_budget(55), Some(1_000_000));
        assert_eq!(paper_budget(104), Some(4_000_000));
        assert!(paper_budget(1).is_none());
    }

    #[test]
    fn a_hit_keeps_the_charged_count() {
        assert_eq!(evals_to_hit(true, Some(4), 30), 4);
        assert_eq!(evals_to_hit(true, None, 30), 30);
        assert_eq!(evals_to_hit(false, Some(4), 30), 30);
    }

    #[test]
    fn type7_lands_on_a_recorded_sample() {
        assert_eq!(type7(&[1.0, 3.0, 5.0], 0.0), 1.0);
        assert_eq!(type7(&[1.0, 3.0, 5.0], 1.0), 5.0);
        assert_eq!(type7(&[5.0], 0.5), 5.0);
    }
}

fn median_iqr(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = type7(&sorted, 0.25);
    let q3 = type7(&sorted, 0.75);
    (type7(&sorted, 0.5), q3 - q1)
}

/// Anneal, then GMIN, for each seed, in seed order.
#[cfg(test)]
fn schedule(seeds: &[u64]) -> Vec<(u64, &'static str)> {
    let mut order = Vec::with_capacity(seeds.len() * 2);
    for seed in seeds {
        order.push((*seed, "anneal"));
        order.push((*seed, "gmin"));
    }
    order
}

#[derive(Clone)]
struct Row {
    system: String,
    code: String,
    budget: usize,
    hit: bool,
    evals: f64,
    wall: f64,
    head: String,
}

struct Summary {
    system: String,
    code: String,
    n: usize,
    hits: usize,
    budget: usize,
    evals_median: f64,
    evals_iqr: f64,
    wall_median: f64,
    wall_iqr: f64,
    head: String,
}

fn parse_rows(text: &str) -> Result<Vec<Row>, String> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().ok_or("rows file is empty")?;
    let columns: Vec<&str> = header.split(',').collect();
    let index = |name: &str| {
        columns
            .iter()
            .position(|column| *column == name)
            .ok_or_else(|| format!("missing column {name}"))
    };
    let system_i = index("system")?;
    let code_i = index("code")?;
    let budget_i = index("budget")?;
    let hit_i = index("hit")?;
    let evals_i = index("evals_to_hit")?;
    let wall_i = index("wall_s")?;
    let head_i = index("head")?;
    let mut rows = Vec::new();
    for (number, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != columns.len() {
            return Err(format!("row {} has {} fields", number + 2, fields.len()));
        }
        let hit = match fields[hit_i] {
            "1" | "true" => true,
            "0" | "false" => false,
            other => return Err(format!("row {} hit {other}", number + 2)),
        };
        rows.push(Row {
            system: fields[system_i].to_string(),
            code: fields[code_i].to_string(),
            budget: fields[budget_i]
                .parse()
                .map_err(|_| format!("row {} budget", number + 2))?,
            hit,
            evals: fields[evals_i]
                .parse()
                .map_err(|_| format!("row {} evals", number + 2))?,
            wall: fields[wall_i]
                .parse()
                .map_err(|_| format!("row {} wall", number + 2))?,
            head: fields[head_i].to_string(),
        });
    }
    Ok(rows)
}

fn summarize(rows: &[Row]) -> Vec<Summary> {
    let mut groups: Vec<(String, String)> = Vec::new();
    for row in rows {
        let key = (row.system.clone(), row.code.clone());
        if !groups.contains(&key) {
            groups.push(key);
        }
    }
    groups
        .into_iter()
        .map(|(system, code)| {
            let group: Vec<&Row> = rows
                .iter()
                .filter(|row| row.system == system && row.code == code)
                .collect();
            let (evals_median, evals_iqr) =
                median_iqr(&group.iter().map(|row| row.evals).collect::<Vec<_>>());
            let (wall_median, wall_iqr) =
                median_iqr(&group.iter().map(|row| row.wall).collect::<Vec<_>>());
            Summary {
                system,
                code,
                n: group.len(),
                hits: group.iter().filter(|row| row.hit).count(),
                budget: group[0].budget,
                evals_median,
                evals_iqr,
                wall_median,
                wall_iqr,
                head: group[0].head.clone(),
            }
        })
        .collect()
}

fn write_coords(path: &str, coords: &Array1<f64>) -> std::io::Result<()> {
    let mut text = String::new();
    for atom in coords.exact_chunks(3) {
        text.push_str(&format!(
            "{:.10} {:.10} {:.10}\n",
            atom[0], atom[1], atom[2]
        ));
    }
    std::fs::write(path, text)
}

fn read_coords(path: &str) -> std::io::Result<Array1<f64>> {
    let text = std::fs::read_to_string(path)?;
    let mut values = Vec::new();
    for line in text.lines() {
        for word in line.split_whitespace() {
            if let Ok(value) = word.parse::<f64>() {
                values.push(value);
            }
        }
    }
    Ok(Array1::from_vec(values))
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().replace(',', " "))
        })
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_pins() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("Cpus_allowed_list"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::{
        HIT_TOL, evals_to_hit, median_iqr, paper_budget, parse_rows, schedule, summarize,
        system_spec, type7,
    };
    use anneal_core::methods::cluster_hopping::{Config, MoveLibrary};

    #[test]
    fn paper_budgets_match_the_elja_map() {
        assert_eq!(paper_budget(38), Some(400_000));
        assert_eq!(paper_budget(55), Some(1_000_000));
        assert_eq!(paper_budget(75), Some(4_000_000));
        assert_eq!(paper_budget(98), Some(4_000_000));
        assert_eq!(paper_budget(102), Some(4_000_000));
        assert_eq!(paper_budget(104), Some(4_000_000));
        assert_eq!(paper_budget(31), None);
    }

    #[test]
    fn the_board_uses_those_budgets_and_the_published_minima() {
        let lj38 = system_spec("lj38").unwrap();
        assert_eq!((lj38.n, lj38.budget), (38, 400_000));
        assert!((lj38.reference + 173.928427).abs() < 1e-12);
        assert_eq!(system_spec("lj55").unwrap().budget, 1_000_000);
        assert_eq!(system_spec("lj75").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("lj98").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("lj104").unwrap().budget, 4_000_000);
        assert!((system_spec("lj104").unwrap().reference + 582.086642).abs() < 1e-12);
        assert!((HIT_TOL - 0.001).abs() < 1e-15);
        assert!(system_spec("lj31").is_none());
    }

    #[test]
    fn recommended_is_the_stock_constructor() {
        let cfg = Config::recommended(75);
        assert_eq!(cfg.move_library, MoveLibrary::LeanBurst);
        assert!(cfg.escape_on_stall);
        assert!(!cfg.restart_on_stall);
        assert!(!cfg.surfaces.is_empty());
        let large = Config::recommended(98);
        assert!(large.surfaces.is_empty());
        let small = Config::recommended(38);
        assert_eq!(small.move_library, MoveLibrary::LeanBurst);
        assert!(!small.surfaces.is_empty());
    }

    #[test]
    fn a_miss_is_charged_the_full_budget() {
        assert_eq!(evals_to_hit(false, None, 400_000), 400_000);
        assert_eq!(evals_to_hit(true, Some(1200), 400_000), 1200);
        assert_eq!(evals_to_hit(true, None, 4_000_000), 4_000_000);
    }

    #[test]
    fn type7_quartiles_on_six_points() {
        let sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!((type7(&sample, 0.25) - 2.25).abs() < 1e-12);
        assert!((type7(&sample, 0.5) - 3.5).abs() < 1e-12);
        assert!((type7(&sample, 0.75) - 4.75).abs() < 1e-12);
        let (median, iqr) = median_iqr(&sample);
        assert!((median - 3.5).abs() < 1e-12);
        assert!((iqr - 2.5).abs() < 1e-12);
    }

    #[test]
    fn seeds_alternate_anneal_then_gmin() {
        let order = schedule(&[2000, 2001]);
        assert_eq!(
            order,
            vec![
                (2000, "anneal"),
                (2000, "gmin"),
                (2001, "anneal"),
                (2001, "gmin"),
            ]
        );
    }

    #[test]
    fn summary_uses_the_charged_column_and_one_commit() {
        let text = "\
system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins
lj38,anneal,2000,400000,1,100,100,1.0,-173.928427,abc,tree,cpu,0
lj38,anneal,2001,400000,0,400000,400000,9.0,-170.0,abc,tree,cpu,0
lj38,gmin,2000,400000,1,300,300,2.0,-173.928427,abc,tree,cpu,0
lj38,gmin,2001,400000,0,400000,400000,8.0,-160.0,abc,tree,cpu,0
";
        let rows = parse_rows(text).unwrap();
        let summary = summarize(&rows);
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0].code, "anneal");
        assert_eq!((summary[0].hits, summary[0].n), (1, 2));
        assert!((summary[0].evals_median - 200_050.0).abs() < 1e-6);
        assert_eq!(summary[0].head, "abc");
        assert_eq!(summary[1].code, "gmin");
        assert_eq!(summary[1].budget, 400_000);
    }
}
