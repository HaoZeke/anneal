//! GFN2 conformer search on one xyz file.
//!
//! Usage: `organic_bench <xyz> <budget> <seed> <arm>`
//!
//! The objective is the xtb binary through `examples/ase_objective.py`
//! (`ASE_ENGINE=xtb-cli`). `XTB_ACC` selects `--acc`. `preset` is
//! [`Config::recommended_molecular`] as it stands. `baseline` is the
//! molecular preset without internal-coordinate moves. `fixed` adds
//! [`Config::with_bonded_container`]. `loose`, `chains` and `tkick` add
//! [`Config::with_loose_quench`], [`Config::with_chain_exchange`] and
//! [`Config::with_torsion_kicks`] to the preset, and `combo` adds all
//! three. `chains4` is the four exchanging chains without the group gate.
//! `polish` runs only the preset's final polish on the xyz structure, with
//! the whole budget. `crush` is the preset with atom-by-atom containment,
//! [`Config::rigid_containment`] off. `molecular` is
//! [`Config::for_molecular`] at the preset's gate, and `communicating` and
//! `derived` overlay those water arms on the preset. `ORGANIC_LOCAL_STEP`
//! (`lbfgs` or `anc`), `ORGANIC_RECORD_GRADIENT` and `ORGANIC_RELAX_STEPS`
//! override any arm's local optimiser, gate and full-relaxation cap.
//!
//! The xyz comment may be `groups i,j;k,l`. Without it, every atom is
//! one group.
//!
//! A call the bridge answers `FAIL` scores `+inf` with a NaN gradient, so
//! the relaxation's guard rejects it. It is charged like any other call
//! and counted in `FAILED_CALLS`. A bridge that stops answering ends the
//! run.

#[path = "common/pipe_engine.rs"]
mod pipe_engine;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Mutex;

use anneal_core::catalog::molecular::{WaterSearchArm, configure_water_arm};
use anneal_core::methods::anc_opt::LocalStep;
use anneal_core::methods::cluster_hopping::{
    Config, GraphCause, GraphCauses, GraphRefusal, Ledger, SoapProposalMode, StartSeparation,
    polish_structure, rotatable_bond_count,
};
use anneal_core::methods::cluster_search::search_from;
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

struct Engine {
    pipe: pipe_engine::PipeEngine,
    calls: usize,
    nonfinite_requests: usize,
}

struct XtbObjective {
    engine: Mutex<Engine>,
    dim: usize,
    bounds: Bounds<f64>,
}

impl Objective<f64> for XtbObjective {
    fn dim(&self) -> usize {
        self.dim
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for XtbObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

impl DifferentiableObjective<f64> for XtbObjective {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mut engine = self.engine.lock().expect("xtb pipe");
        engine.calls += 1;
        if x.iter().any(|v| !v.is_finite()) {
            engine.nonfinite_requests += 1;
        }
        let failed = engine.pipe.failures;
        match engine.pipe.eval(x) {
            Some(pair) => pair,
            // A zero gradient would certify the point as a converged minimum.
            None if engine.pipe.failures > failed => {
                (f64::INFINITY, Array1::from_elem(x.len(), f64::NAN))
            }
            None => panic!("the xtb bridge stopped answering at call {}", engine.calls),
        }
    }
}

fn atomic_number(symbol: &str) -> u32 {
    match symbol {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "Br" => 35,
        other => panic!("unsupported element {other}"),
    }
}

/// Symbols, atomic numbers, coordinates, declared groups and the comment line.
type Xyz = (Vec<String>, Vec<u32>, Array1<f64>, Vec<Vec<usize>>, String);

fn load_xyz(path: &str) -> Xyz {
    let file = File::open(path).unwrap_or_else(|err| panic!("open {path}: {err}"));
    let mut lines = BufReader::new(file).lines();
    let n: usize = lines
        .next()
        .expect("atom count")
        .expect("atom count")
        .trim()
        .parse()
        .expect("atom count");
    let comment = lines.next().expect("comment").expect("comment");
    let mut symbols = Vec::with_capacity(n);
    let mut species = Vec::with_capacity(n);
    let mut coords = Vec::with_capacity(3 * n);
    for _ in 0..n {
        let line = lines.next().expect("atom").expect("atom");
        let mut parts = line.split_whitespace();
        let symbol = parts.next().expect("symbol").to_string();
        species.push(atomic_number(&symbol));
        symbols.push(symbol);
        for _ in 0..3 {
            coords.push(
                parts
                    .next()
                    .expect("coordinate")
                    .parse::<f64>()
                    .expect("coordinate"),
            );
        }
    }
    let groups = if let Some(rest) = comment.trim().strip_prefix("groups ") {
        rest.split(';')
            .filter(|piece| !piece.is_empty())
            .map(|piece| {
                piece
                    .split(',')
                    .map(|index| index.parse::<usize>().expect("group index"))
                    .collect()
            })
            .collect()
    } else {
        vec![(0..n).collect()]
    };
    (symbols, species, Array1::from(coords), groups, comment)
}

fn baseline(species: Vec<u32>, groups: Vec<Vec<usize>>) -> Config {
    let mut cfg = Config::for_molecular(species, groups, 1.0);
    cfg.allocate_moves = true;
    cfg.depth_reward = true;
    cfg.tabu_on_stall = true;
    cfg.soap_class_residual = false;
    cfg.soap_mode = SoapProposalMode::Flexible;
    cfg.adaptive_height = true;
    cfg.height_revisits = 20.0;
    cfg.record_gradient = 5.0e-2;
    cfg
}

/// The preset and the opt-in arms layered on it.
fn gap_arm(arm: &str, species: &[u32], groups: &[Vec<usize>]) -> Option<Config> {
    let cfg = Config::recommended_molecular(species.to_vec(), groups.to_vec(), 1.0);
    Some(match arm {
        "preset" => cfg,
        "crush" => {
            let mut cfg = cfg;
            cfg.rigid_containment = false;
            cfg
        }
        "loose" => cfg.with_loose_quench(),
        "chains" => cfg.with_chain_exchange(),
        "tkick" => cfg.with_torsion_kicks(),
        "combo" => cfg
            .with_loose_quench()
            .with_chain_exchange()
            .with_torsion_kicks(),
        "chains4" => {
            let mut cfg = cfg;
            cfg.replicas = Config::EXCHANGE_CHAINS;
            cfg.swap_period = 1;
            cfg.chain_exchange = true;
            cfg
        }
        "molecular" => {
            let mut cfg = Config::for_molecular(species.to_vec(), groups.to_vec(), 1.0);
            cfg.record_gradient = 5.0e-2;
            cfg
        }
        "communicating" => {
            let mut cfg = cfg;
            configure_water_arm(&mut cfg, WaterSearchArm::Communicating);
            cfg
        }
        "derived" => {
            let mut cfg = cfg;
            configure_water_arm(&mut cfg, WaterSearchArm::Derived);
            cfg
        }
        _ => return None,
    })
}

/// `ORGANIC_LOCAL_STEP`, `ORGANIC_RECORD_GRADIENT` and `ORGANIC_RELAX_STEPS`
/// override the arm's local optimiser, gate and full-relaxation cap.
fn apply_overrides(cfg: &mut Config) {
    match std::env::var("ORGANIC_LOCAL_STEP").as_deref() {
        Ok("anc") => cfg.local_step = LocalStep::ANC_EV_ANGSTROM,
        Ok("lbfgs") | Err(_) => cfg.local_step = LocalStep::Lbfgs,
        Ok(other) => panic!("ORGANIC_LOCAL_STEP {other}: expected anc or lbfgs"),
    }
    if let Ok(value) = std::env::var("ORGANIC_RECORD_GRADIENT") {
        cfg.record_gradient = value.parse().expect("ORGANIC_RECORD_GRADIENT");
    }
    if let Ok(value) = std::env::var("ORGANIC_RELAX_STEPS") {
        cfg.relax_steps = value.parse().expect("ORGANIC_RELAX_STEPS");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("mol.xyz");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(700);
    let seed: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let arm = args.get(4).map(String::as_str).unwrap_or("fixed");
    let (symbols, species, start, groups, comment) = load_xyz(path);
    let n = species.len();
    assert_eq!(start.len(), 3 * n);
    if std::env::var("ASE_ENGINE").is_err() {
        // SAFETY: the process has not spawned threads yet.
        unsafe { std::env::set_var("ASE_ENGINE", "xtb-cli") };
    }
    let pipe = pipe_engine::PipeEngine::start("xtb-cli", symbols, None);
    let obj = XtbObjective {
        engine: Mutex::new(Engine {
            pipe,
            calls: 0,
            nonfinite_requests: 0,
        }),
        dim: 3 * n,
        bounds: Bounds::new(
            Array1::from_elem(3 * n, -1.0e3),
            Array1::from_elem(3 * n, 1.0e3),
            0.0,
        ),
    };
    if arm == "polish" {
        polish_only(&obj, path, species, groups, &start, budget, seed);
        return;
    }
    let mut cfg = if let Some(cfg) = gap_arm(arm, &species, &groups) {
        cfg
    } else if arm == "baseline" || arm == "wide" {
        let mut cfg = baseline(species.clone(), groups.clone());
        if arm == "wide" {
            let span =
                anneal_core::methods::cluster_hopping::fragment_container(&species, groups.len());
            if span > cfg.container {
                cfg.container = span;
            }
        }
        cfg
    } else {
        let mut cfg = Config::recommended_molecular(species, groups, 1.0).with_bonded_container();
        let replicas = std::env::var("ORGANIC_REPLICAS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        cfg.replicas = replicas;
        if let Ok(value) = std::env::var("ORGANIC_CONTAINER") {
            cfg.container = value.parse().expect("ORGANIC_CONTAINER");
        }
        cfg
    };
    cfg.replicas = cfg.replicas.max(1);
    apply_overrides(&mut cfg);
    let rotors = rotatable_bond_count(
        start.view(),
        cfg.species.as_deref().unwrap_or(&[]),
        cfg.bond_tolerance,
    );
    println!("arm {arm}");
    println!("seed {seed}");
    println!("budget {budget}");
    println!("comment {comment}");
    println!("container {:.6}", cfg.container);
    println!("internal_moves {}", cfg.internal_moves);
    println!("internal_relax {}", cfg.internal_relax);
    println!("replicas {}", cfg.replicas);
    println!("keep_bond_graph {}", cfg.keep_bond_graph);
    println!("record_polish {}", cfg.record_polish);
    println!("separate_start_molecules {}", cfg.separate_start_molecules);
    println!("rigid_containment {}", cfg.rigid_containment);
    println!("local_step {:?}", cfg.local_step);
    println!("record_gradient {:e}", cfg.record_gradient);
    println!("relax_steps {}", cfg.relax_steps);
    let mut ledger = Ledger::new(budget);
    let started = std::time::Instant::now();
    let (out, stats) = search_from(&obj, &cfg, &mut ledger, start.view(), seed)
        .unwrap_or_else(|err| panic!("search failed: {err}"));
    let wall = started.elapsed().as_secs_f64();
    let (calls, failed, nonfinite) = {
        let engine = obj.engine.lock().expect("xtb pipe");
        (
            engine.calls,
            engine.pipe.failures,
            engine.nonfinite_requests,
        )
    };
    println!("FINAL {:.8}", out.final_energy);
    if let Some(state) = &out.final_state {
        println!("GMAX {:.6}", largest_component(&obj.grad(state.view())));
    }
    if let Some(state) = &out.best_state {
        println!(
            "BEST_GMAX {:.6}",
            largest_component(&obj.grad(state.view()))
        );
    }
    println!("BEST {:.8}", out.best);
    println!("CHARGED {}", out.charged);
    println!("HOPS {}", out.hops);
    println!("BASINS {}", out.basins);
    println!("UNCONVERGED {}", out.unconverged_records);
    println!("SCREENED {}", out.screened_out);
    println!(
        "QUENCH screen {} full {} check {}",
        stats.screen_charged, stats.full_charged, stats.check_charged
    );
    println!("REFUSED {}", stats.refused);
    println!("REFUSED_STARTS {}", stats.refused_starts);
    println!("ENGINE_CALLS {calls}");
    println!("FAILED_CALLS {failed}");
    println!("NONFINITE_REQUESTS {nonfinite}");
    println!("SOAP_FALLBACKS {}", out.soap_fallbacks);
    println!("REJECTED_TRIALS {}", out.rejected_trials);
    println!(
        "GRAPH_REJECTIONS {} start {}",
        out.graph_rejections, out.start_graph_rejected
    );
    let symbols = load_symbols(path);
    println!("{}", causes_text(out.graph_causes));
    println!("{}", first_text(out.graph_first.as_ref(), &symbols));
    println!("START_CONTINUATIONS {}", out.start_continuations);
    println!("START_SEPARATION {}", separation_text(out.start_separation));
    println!("REFINES {} charged {}", stats.refines, stats.refine_charged);
    println!(
        "RELAX converged {} capped {} screens {}",
        stats.converged, stats.capped, stats.screens
    );
    match &out.polish {
        Some(polish) => {
            println!(
                "POLISH reserve {} search_charged {} calls {} records {} unspent {}",
                polish.reserve,
                polish.search_charged,
                polish.calls,
                polish.records.len(),
                budget.saturating_sub(out.charged)
            );
            for (k, record) in polish.records.iter().enumerate() {
                println!(
                    "POLISH_RECORD {k} from {:.8} to {:.8} calls {} converged {} gnorm {:.6} recorded {}",
                    record.from,
                    record.to,
                    record.calls,
                    record.converged,
                    record.gradient_norm,
                    record.recorded
                );
            }
        }
        None => println!("POLISH none"),
    }
    println!("ROTORS {rotors}");
    println!(
        "CHAINS {} swaps {} accepted {}",
        cfg.replicas, out.swaps_tried, out.swaps_accepted
    );
    println!("WALL {wall:.3}");
    for (name, draws, accepts, reward) in &out.arms {
        println!("ARM {name} draws {draws} accepts {accepts} reward {reward:.4}");
    }
    for (_, charged, _, energy) in &out.improvements {
        println!("IMPROVE {charged} {energy:.8}");
    }
    if let Some(state) = &out.best_state {
        write_best(path, &format!("best_{arm}_s{seed}.xyz"), out.best, state);
    }
}

fn write_best(path: &str, xyz: &str, energy: f64, state: &Array1<f64>) {
    let symbols = load_symbols(path);
    let n = symbols.len();
    let mut body = format!("{n}\nbest {energy:.8} eV\n");
    for i in 0..n {
        body.push_str(&format!(
            "{} {:.8} {:.8} {:.8}\n",
            symbols[i],
            state[3 * i],
            state[3 * i + 1],
            state[3 * i + 2]
        ));
    }
    std::fs::write(xyz, body).expect("write best xyz");
    println!("WROTE {xyz}");
}

fn causes_text(causes: GraphCauses) -> String {
    format!(
        "GRAPH_CAUSES kernel {} crush {} contained {} reaction {}",
        causes.kernel, causes.crush, causes.crush_contained, causes.reaction
    )
}

/// The first refused trial: its cause and the bonds its quench broke and
/// formed, as `C15-O17`, by element and zero-based index.
fn first_text(first: Option<&GraphRefusal>, symbols: &[String]) -> String {
    let Some(first) = first else {
        return "GRAPH_FIRST none".to_owned();
    };
    let bonds = |pairs: &[(usize, usize)]| {
        if pairs.is_empty() {
            return "-".to_owned();
        }
        pairs
            .iter()
            .map(|&(a, b)| format!("{}{a}-{}{b}", symbols[a], symbols[b]))
            .collect::<Vec<_>>()
            .join(",")
    };
    let cause = match first.cause {
        GraphCause::Kernel => "kernel",
        GraphCause::Crush => "crush",
        GraphCause::Reaction => "reaction",
    };
    format!(
        "GRAPH_FIRST {cause} contained {} broken {} formed {}",
        first.contained_changed,
        bonds(&first.broken),
        bonds(&first.formed)
    )
}

fn separation_text(separation: Option<StartSeparation>) -> String {
    match separation {
        Some(StartSeparation::Separated { steps, factor }) => format!("k {steps} s {factor:.6}"),
        Some(StartSeparation::Unresolved) => "unresolved".to_owned(),
        None => "none".to_owned(),
    }
}

/// Largest absolute component, NaN when any is not finite.
fn largest_component(gradient: &Array1<f64>) -> f64 {
    if gradient.iter().all(|v| v.is_finite()) {
        gradient.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
    } else {
        f64::NAN
    }
}

/// The `polish` arm: the final polish alone on the xyz structure, with the
/// whole budget, under the preset.
fn polish_only(
    obj: &XtbObjective,
    path: &str,
    species: Vec<u32>,
    groups: Vec<Vec<usize>>,
    start: &Array1<f64>,
    budget: usize,
    seed: u64,
) {
    let cfg = Config::recommended_molecular(species, groups, 1.0);
    println!("arm polish");
    println!("seed {seed}");
    println!("budget {budget}");
    println!("keep_bond_graph {}", cfg.keep_bond_graph);
    let mut ledger = Ledger::new(budget);
    let mut energy_grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<(f64, Array1<f64>)> {
        if !led.charge() {
            return None;
        }
        Some(obj.value_and_gradient(x))
    };
    let started = std::time::Instant::now();
    let (record, best) = polish_structure(&cfg, &mut ledger, &mut energy_grad, start.view())
        .expect("the preset has species");
    let wall = started.elapsed().as_secs_f64();
    let calls = obj.engine.lock().expect("xtb pipe").calls;
    println!(
        "POLISH_SMOKE from {:.8} to {:.8} calls {} converged {} gnorm {:.6}",
        record.from, record.to, record.calls, record.converged, record.gradient_norm
    );
    println!("CHARGED {}", ledger.spent());
    println!("ENGINE_CALLS {calls}");
    println!("WALL {wall:.3}");
    if let Some((energy, state)) = best {
        println!("BEST {energy:.8}");
        println!(
            "BEST_GMAX {:.6}",
            largest_component(&obj.grad(state.view()))
        );
        write_best(path, &format!("best_polish_s{seed}.xyz"), energy, &state);
    }
}

fn load_symbols(path: &str) -> Vec<String> {
    let file = File::open(path).expect("xyz");
    let mut lines = BufReader::new(file).lines();
    let n: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();
    let _ = lines.next();
    let mut symbols = Vec::new();
    for _ in 0..n {
        let line = lines.next().unwrap().unwrap();
        symbols.push(line.split_whitespace().next().unwrap().to_string());
    }
    symbols
}
