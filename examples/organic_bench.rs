//! GFN2 conformer search on one xyz file.
//!
//! Usage: `organic_bench <xyz> <budget> <seed> <preset|baseline|wide|fixed>`
//!
//! The objective is the xtb binary through `examples/ase_objective.py`
//! (`ASE_ENGINE=xtb-cli`). `XTB_ACC` selects `--acc`. `preset` is
//! [`Config::recommended_molecular`] as it stands. `baseline` is the
//! molecular preset without internal-coordinate moves. `fixed` adds
//! [`Config::with_bonded_container`].
//!
//! The xyz comment may be `groups i,j;k,l`. Without it, every atom is
//! one group.

#[path = "common/pipe_engine.rs"]
mod pipe_engine;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Mutex;

use anneal_core::methods::cluster_hopping::{Config, Ledger, SoapProposalMode};
use anneal_core::methods::cluster_search::search_from;
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

struct XtbObjective {
    pipe: Mutex<pipe_engine::PipeEngine>,
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
        let mut pipe = self.pipe.lock().expect("xtb pipe");
        match pipe.eval(x) {
            Some(pair) => pair,
            None => (1.0e6, Array1::zeros(x.len())),
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
        pipe: Mutex::new(pipe),
        dim: 3 * n,
        bounds: Bounds::new(
            Array1::from_elem(3 * n, -1.0e3),
            Array1::from_elem(3 * n, 1.0e3),
            0.0,
        ),
    };
    let mut cfg = if arm == "preset" {
        Config::recommended_molecular(species.clone(), groups.clone(), 1.0)
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
    println!("arm {arm}");
    println!("seed {seed}");
    println!("budget {budget}");
    println!("comment {comment}");
    println!("container {:.6}", cfg.container);
    println!("internal_moves {}", cfg.internal_moves);
    println!("internal_relax {}", cfg.internal_relax);
    println!("replicas {}", cfg.replicas);
    let mut ledger = Ledger::new(budget);
    let started = std::time::Instant::now();
    let (out, stats) = search_from(&obj, &cfg, &mut ledger, start.view(), seed)
        .unwrap_or_else(|err| panic!("search failed: {err}"));
    let wall = started.elapsed().as_secs_f64();
    println!("FINAL {:.8}", out.final_energy);
    if let Some(state) = &out.final_state {
        let gradient = obj.grad(state.view());
        let gmax = gradient.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        println!("GMAX {gmax:.6}");
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
    println!("SOAP_FALLBACKS {}", out.soap_fallbacks);
    println!("WALL {wall:.3}");
    for (name, draws, accepts, reward) in &out.arms {
        println!("ARM {name} draws {draws} accepts {accepts} reward {reward:.4}");
    }
    for (_, charged, _, energy) in &out.improvements {
        println!("IMPROVE {charged} {energy:.8}");
    }
    if let Some(state) = &out.best_state {
        let xyz = format!("best_{arm}_s{seed}.xyz");
        let mut body = format!("{n}\nbest {:.8} eV\n", out.best);
        let symbols = load_symbols(path);
        for i in 0..n {
            body.push_str(&format!(
                "{} {:.8} {:.8} {:.8}\n",
                symbols[i],
                state[3 * i],
                state[3 * i + 1],
                state[3 * i + 2]
            ));
        }
        std::fs::write(&xyz, body).expect("write best xyz");
        println!("WROTE {xyz}");
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
