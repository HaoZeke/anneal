// Copyright (c) anneal developers
//! Single-knockout ablation driver for `Config::recommended` (G1).
//!
//! ```text
//! g1_knockout arms
//! g1_knockout systems
//! g1_knockout config <system> <arm>
//! g1_knockout config-hash <system> <arm>
//! g1_knockout run <system> <arm> <seed> [budget]
//! ```
//!
//! Runs through `cluster_search::search_from` with `rgpot` kernels installed so
//! the preset's relaxation surfaces (`Config::surfaces`) and record gradient
//! tolerance are exercised. Never reads `H2H_*` environment variables.

use anneal_core::methods::cluster_hopping::{
    Config, ContinuousSymmetry, Ledger, Outcome, random_cluster_in_radius,
};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::potentials::{Gupta, PairPotential};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

const ARMS: &[&str] = &[
    "recommended",
    "a_restart_on_stall",
    "b_no_orbit",
    "c_no_symmetry",
    "d_no_screens",
    "e_no_surfaces",
    "f_no_energy_lattice",
    "g_no_priors",
    "h_no_basin_bias",
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum PotentialKind {
    LennardJones,
    Morse { rho: f64 },
    GuptaAu,
}

#[derive(Clone, Copy, Debug)]
struct SystemSpec {
    name: &'static str,
    kind: PotentialKind,
    n: usize,
    budget: usize,
    reference: f64,
}

const SYSTEMS: &[SystemSpec] = &[
    SystemSpec {
        name: "lj38",
        kind: PotentialKind::LennardJones,
        n: 38,
        budget: 400_000,
        reference: -173.928427,
    },
    SystemSpec {
        name: "lj55",
        kind: PotentialKind::LennardJones,
        n: 55,
        budget: 1_000_000,
        reference: -279.248470,
    },
    SystemSpec {
        name: "lj75",
        kind: PotentialKind::LennardJones,
        n: 75,
        budget: 4_000_000,
        reference: -397.492331,
    },
    SystemSpec {
        name: "lj98",
        kind: PotentialKind::LennardJones,
        n: 98,
        budget: 4_000_000,
        reference: -543.665361,
    },
    SystemSpec {
        name: "lj150",
        kind: PotentialKind::LennardJones,
        n: 150,
        budget: 4_000_000,
        reference: -893.310258,
    },
    SystemSpec {
        name: "m55r14",
        kind: PotentialKind::Morse { rho: 14.0 },
        n: 55,
        budget: 1_000_000,
        reference: -220.646208,
    },
    SystemSpec {
        name: "m80r6",
        kind: PotentialKind::Morse { rho: 6.0 },
        n: 80,
        budget: 4_000_000,
        reference: -378.333471,
    },
    SystemSpec {
        name: "m80r14",
        kind: PotentialKind::Morse { rho: 14.0 },
        n: 80,
        budget: 4_000_000,
        reference: -340.811371,
    },
    SystemSpec {
        name: "au55",
        kind: PotentialKind::GuptaAu,
        n: 55,
        budget: 1_000_000,
        reference: -191.03085280,
    },
];

fn system_spec(name: &str) -> Option<SystemSpec> {
    SYSTEMS.iter().copied().find(|spec| spec.name == name)
}

fn arm_config(n: usize, arm: &str) -> Result<Config, String> {
    let mut cfg = Config::recommended(n);
    match arm {
        "recommended" => {}
        "a_restart_on_stall" => {
            cfg.restart_on_stall = true;
            cfg.escape_on_stall = false;
            cfg.trail_on_stall = false;
            cfg.tabu_on_stall = false;
        }
        "b_no_orbit" => {
            cfg.orbit_complete_on_new = false;
        }
        "c_no_symmetry" => {
            cfg.continuous_symmetry = ContinuousSymmetry::Off;
            cfg.symmetrise_on_stall = false;
        }
        "d_no_screens" => {
            cfg.return_screen = false;
            cfg.bayes_screen = false;
        }
        "e_no_surfaces" => {
            cfg.surfaces.clear();
        }
        "f_no_energy_lattice" => {
            cfg.energy_lattice = false;
        }
        "g_no_priors" => {
            cfg.prior_soap = false;
        }
        "h_no_basin_bias" => {
            cfg.bias_height = 0.0;
        }
        other => return Err(format!("unknown arm {other}")),
    }
    Ok(cfg)
}

enum PotentialModel {
    Pair(PairPotential),
    Metal(Gupta),
}

struct TimedObjective {
    model: PotentialModel,
    pot_ns: AtomicU64,
}

impl TimedObjective {
    fn for_spec(spec: SystemSpec) -> Self {
        let model = match spec.kind {
            PotentialKind::LennardJones => PotentialModel::Pair(PairPotential::lennard_jones(spec.n)),
            PotentialKind::Morse { rho } => PotentialModel::Pair(PairPotential::morse(spec.n, rho)),
            PotentialKind::GuptaAu => PotentialModel::Metal(Gupta::copper_gold(0, spec.n)),
        };
        Self {
            model,
            pot_ns: AtomicU64::new(0),
        }
    }

    fn pot_seconds(&self) -> f64 {
        self.pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9
    }
}

impl Objective<f64> for TimedObjective {
    fn dim(&self) -> usize {
        match &self.model {
            PotentialModel::Pair(p) => Objective::dim(p),
            PotentialModel::Metal(g) => Objective::dim(g),
        }
    }

    fn bounds(&self) -> &Bounds<f64> {
        match &self.model {
            PotentialModel::Pair(p) => p.bounds(),
            PotentialModel::Metal(g) => g.bounds(),
        }
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let mark = Instant::now();
        let val = match &self.model {
            PotentialModel::Pair(p) => p.eval(x),
            PotentialModel::Metal(g) => g.eval(x),
        };
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        val
    }
}

impl Gradient<f64> for TimedObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let mark = Instant::now();
        let g = match &self.model {
            PotentialModel::Pair(p) => p.grad(x),
            PotentialModel::Metal(m) => m.grad(x),
        };
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        g
    }

    fn dim(&self) -> usize {
        Objective::dim(self)
    }
}

impl DifferentiableObjective<f64> for TimedObjective {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mark = Instant::now();
        let pair = match &self.model {
            PotentialModel::Pair(p) => p.value_and_gradient(x),
            PotentialModel::Metal(g) => g.value_and_gradient(x),
        };
        self.pot_ns
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        pair
    }
}

fn start_of(n: usize, seed: u64) -> Array1<f64> {
    let base = Config::recommended(n);
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, base.start_radius(), base.min_separation, &mut rng)
}

fn run_search(
    spec: SystemSpec,
    cfg: &Config,
    seed: u64,
    budget: usize,
) -> Result<(Outcome, f64, f64), String> {
    let start = start_of(spec.n, seed);
    let objective = TimedObjective::for_spec(spec);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(spec.reference + HIT_TOL);
    let started = Instant::now();
    let (outcome, _stats) = search_from(&objective, cfg, &mut ledger, start.view(), seed)
        .map_err(|err| format!("search failed: {err}"))?;
    let wall_s = started.elapsed().as_secs_f64();
    let pot_s = objective.pot_seconds();
    Ok((outcome, wall_s, pot_s))
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

fn cmd_systems() -> i32 {
    let items: Vec<serde_json::Value> = SYSTEMS
        .iter()
        .map(|spec| {
            let (kind, rho) = match spec.kind {
                PotentialKind::LennardJones => ("lj", 0.0),
                PotentialKind::Morse { rho } => ("morse", rho),
                PotentialKind::GuptaAu => ("gupta", 0.0),
            };
            serde_json::json!({
                "name": spec.name,
                "kind": kind,
                "n": spec.n,
                "rho": rho,
                "budget": spec.budget,
                "reference": spec.reference,
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string(&items).expect("serialize systems")
    );
    0
}

fn cmd_config(args: &[String], hash_only: bool) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("config needs <system> <arm>");
        return 2;
    };
    let Some(arm) = args.get(2).map(String::as_str) else {
        eprintln!("config needs <system> <arm>");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let cfg = match arm_config(spec.n, arm) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{err}");
            return 2;
        }
    };
    if hash_only {
        println!("{}", cfg.resolved_sha256().expect("config sha256"));
    } else {
        println!("{}", cfg.resolved_json().expect("config json"));
    }
    0
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(system) = args.get(1).map(String::as_str) else {
        eprintln!("run needs <system> <arm> <seed> [budget]");
        return 2;
    };
    let Some(arm) = args.get(2).map(String::as_str) else {
        eprintln!("run needs <system> <arm> <seed> [budget]");
        return 2;
    };
    let Some(seed) = args.get(3).and_then(|v| v.parse::<u64>().ok()) else {
        eprintln!("run needs a u64 seed");
        return 2;
    };
    let Some(spec) = system_spec(system) else {
        eprintln!("unknown system {system}");
        return 2;
    };
    let budget = match args.get(4) {
        Some(text) => match text.parse::<usize>().ok().filter(|v| *v > 0) {
            Some(v) => v,
            None => {
                eprintln!("invalid budget {text}");
                return 2;
            }
        },
        None => spec.budget,
    };
    let cfg = match arm_config(spec.n, arm) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{err}");
            return 2;
        }
    };
    let config_hash = cfg.resolved_sha256().expect("config sha256");
    let (outcome, wall_s, pot_s) = match run_search(spec, &cfg, seed, budget) {
        Ok(res) => res,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    let hit = anneal_core::energy_hits(outcome.best, spec.reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, spec.reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let calls = if hit {
        match at_hit {
            Some(spent) if spent <= budget => spent,
            _ => outcome.charged.min(budget),
        }
    } else {
        budget
    };
    let outside_s = (wall_s - pot_s).max(0.0);
    let best_str = if outcome.best.is_finite() {
        format!("{:.10}", outcome.best)
    } else {
        format!("{}", outcome.best)
    };
    let best_bits = format!("0x{:016x}", outcome.best.to_bits());
    let row = serde_json::json!({
        "system": spec.name,
        "arm": arm,
        "seed": seed,
        "core": cpu_pins(),
        "hit": u8::from(hit),
        "calls": calls,
        "charged": outcome.charged,
        "wall_s": format!("{wall_s:.6}"),
        "pot_s": format!("{pot_s:.6}"),
        "outside_s": format!("{outside_s:.6}"),
        "best": best_str,
        "best_bits": best_bits,
        "budget": budget,
        "config_hash": config_hash,
        "hops": outcome.hops,
        "basins": outcome.basins,
    });
    println!("{}", serde_json::to_string(&row).expect("serialize row"));
    0
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let status = match cmd {
        "arms" => {
            for arm in ARMS {
                println!("{arm}");
            }
            0
        }
        "systems" => cmd_systems(),
        "config" => cmd_config(&args, false),
        "config-hash" => cmd_config(&args, true),
        "run" => cmd_run(&args),
        _ => {
            eprintln!("unknown command {cmd:?}; expected arms|systems|config|config-hash|run");
            2
        }
    };
    std::process::exit(status);
}

#[cfg(test)]
mod tests {
    use super::{ARMS, SYSTEMS, arm_config, run_search, system_spec};
    use anneal_core::methods::cluster_hopping::ClusterMove;
    use std::collections::BTreeSet;

    fn diff_keys(n: usize, arm: &str) -> BTreeSet<String> {
        let base: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&arm_config(n, "recommended").unwrap().resolved_json().unwrap())
                .unwrap();
        let other: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&arm_config(n, arm).unwrap().resolved_json().unwrap()).unwrap();
        let all_keys: BTreeSet<String> = base.keys().chain(other.keys()).cloned().collect();
        all_keys
            .into_iter()
            .filter(|k| base.get(k) != other.get(k))
            .collect()
    }

    #[test]
    fn panel_systems_match_paper_budgets_and_refs() {
        assert_eq!(SYSTEMS.len(), 9);
        assert_eq!(ARMS.len(), 9);
        assert_eq!(system_spec("lj38").unwrap().budget, 400_000);
        assert_eq!(system_spec("lj55").unwrap().budget, 1_000_000);
        assert_eq!(system_spec("lj75").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("lj98").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("lj150").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("m55r14").unwrap().budget, 1_000_000);
        assert_eq!(system_spec("m80r6").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("m80r14").unwrap().budget, 4_000_000);
        assert_eq!(system_spec("au55").unwrap().budget, 1_000_000);
    }

    #[test]
    fn arm_configs_differ_only_in_declared_fields() {
        let expected = |keys: &[&str]| -> BTreeSet<String> {
            keys.iter().map(|s| (*s).to_string()).collect()
        };
        for &n in &[38usize, 55, 75, 80, 98, 150] {
            assert_eq!(diff_keys(n, "recommended"), expected(&[]));
            assert_eq!(
                diff_keys(n, "a_restart_on_stall"),
                expected(&[
                    "restart_on_stall",
                    "escape_on_stall",
                    "trail_on_stall",
                    "tabu_on_stall",
                ])
            );
            assert_eq!(diff_keys(n, "b_no_orbit"), expected(&["orbit_complete_on_new"]));
            assert_eq!(
                diff_keys(n, "c_no_symmetry"),
                expected(&["continuous_symmetry", "symmetrise_on_stall"])
            );
            assert_eq!(
                diff_keys(n, "d_no_screens"),
                expected(&["return_screen", "bayes_screen"])
            );
            if n < 98 {
                assert_eq!(diff_keys(n, "e_no_surfaces"), expected(&["surfaces"]));
            } else {
                assert_eq!(diff_keys(n, "e_no_surfaces"), expected(&[]));
            }
            assert_eq!(
                diff_keys(n, "f_no_energy_lattice"),
                expected(&["energy_lattice"])
            );
            assert_eq!(diff_keys(n, "g_no_priors"), expected(&["prior_soap"]));
            assert_eq!(diff_keys(n, "h_no_basin_bias"), expected(&["bias_height"]));
        }
    }

    #[test]
    fn no_priors_removes_soap_kernel() {
        let rec = arm_config(55, "recommended").unwrap();
        let no_priors = arm_config(55, "g_no_priors").unwrap();
        assert!(
            rec.move_library
                .kernels(&rec)
                .iter()
                .any(|k| matches!(k, ClusterMove::Soap { .. }))
        );
        assert!(
            no_priors
                .move_library
                .kernels(&no_priors)
                .iter()
                .all(|k| !matches!(k, ClusterMove::Soap { .. }))
        );
    }

    #[test]
    fn no_surfaces_changes_lj38_trajectory() {
        anneal_core::install_rgpot_kernels!();
        let spec = system_spec("lj38").unwrap();
        let rec = arm_config(spec.n, "recommended").unwrap();
        let no_surf = arm_config(spec.n, "e_no_surfaces").unwrap();
        let (out_rec, _, _) = run_search(spec, &rec, 2000, 1500).unwrap();
        let (out_plain, _, _) = run_search(spec, &no_surf, 2000, 1500).unwrap();
        assert_ne!(out_rec.best.to_bits(), out_plain.best.to_bits());
    }

    #[test]
    fn all_potentials_and_bias_zero_smoke_cleanly() {
        anneal_core::install_rgpot_kernels!();
        for sys in ["lj38", "m55r14", "au55"] {
            let spec = system_spec(sys).unwrap();
            for arm in ["recommended", "a_restart_on_stall", "h_no_basin_bias"] {
                let cfg = arm_config(spec.n, arm).unwrap();
                let (out, wall_s, pot_s) = run_search(spec, &cfg, 2000, 2500).unwrap();
                assert!(out.best.is_finite(), "{sys} {arm}: best={}", out.best);
                assert!(out.charged > 0 && out.charged <= 2500);
                assert!(wall_s >= 0.0 && pot_s >= 0.0);
            }
        }
    }
}
