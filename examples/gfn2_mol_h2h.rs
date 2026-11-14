//! One GFN2-xTB molecular seed for the CREST head-to-head.
//!
//! Usage: gfn2_mol_h2h <system> <budget> <seed> <outdir>
//!
//! Systems: `nh3-8`, `h2o-8`, `h2o-12`, `hf-6`, `meoh-4`, `nh3-2-h2o-4`.
//! The search is `Config::recommended_molecular`. There is no wall and no
//! constant set for one molecule. Monomers are GFN2-xTB 6.7.1 vtight
//! geometries with the first atom at the origin. `H2H_START_XYZ` shares a
//! start. `pot_s` is time inside `value_and_gradient`.

#[path = "common/rgpot_eindir.rs"]
mod rgpot_eindir;

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anneal_core::catalog::molecular::component_gradient_tolerance;
use anneal_core::methods::cluster_hopping::Outcome;
use anneal_core::methods::cluster_hopping::{Config, Ledger, repack_rigid_groups};
use anneal_core::methods::cluster_search::{RelaxStats, search_from};
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rgpot_eindir::RgpotObjective;

const FORCE_BAR_EV_PER_A: f64 = 0.0514221;
const INTACT_H_A: f64 = 1.2;
const MIN_INTERMOLECULAR_A: f64 = 1.5;

/// GFN2-xTB 6.7.1 vtight ammonia, nitrogen at the origin.
const AMMONIA: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.9438867749, 0.0, 0.3668373921],
    [-0.4719433871, 0.8174299251, 0.3668373931],
    [-0.4719433871, -0.8174299251, 0.3668373931],
];
/// GFN2-xTB 6.7.1 vtight water, oxygen at the origin.
const WATER: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.0],
    [0.7720869669, 0.5691152394, 0.0],
    [-0.7720869668, 0.5691152394, 0.0],
];
/// GFN2-xTB 6.7.1 vtight hydrogen fluoride, fluorine at the origin.
const HF: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [-0.9324912314, 0.0, 0.0]];
/// GFN2-xTB 6.7.1 vtight methanol, carbon at the origin.
const METHANOL: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.3866970290, 0.0876435957, -0.2171225794],
    [-0.4686275014, 0.9896629176, 0.0320423015],
    [-0.2320956881, -0.5343031153, 0.9277487583],
    [-0.4101383355, -0.5573999395, -0.8400740969],
    [1.7880786425, 0.5765628308, 0.5087847454],
];

struct Piece {
    z: &'static [u32],
    xyz: &'static [[f64; 3]],
    count: usize,
}

struct System {
    name: &'static str,
    pieces: &'static [Piece],
}

fn systems() -> &'static [System] {
    &[
        System {
            name: "nh3-8",
            pieces: &[Piece {
                z: &[7, 1, 1, 1],
                xyz: &AMMONIA,
                count: 8,
            }],
        },
        System {
            name: "h2o-8",
            pieces: &[Piece {
                z: &[8, 1, 1],
                xyz: &WATER,
                count: 8,
            }],
        },
        System {
            name: "h2o-12",
            pieces: &[Piece {
                z: &[8, 1, 1],
                xyz: &WATER,
                count: 12,
            }],
        },
        System {
            name: "hf-6",
            pieces: &[Piece {
                z: &[9, 1],
                xyz: &HF,
                count: 6,
            }],
        },
        System {
            name: "meoh-4",
            pieces: &[Piece {
                z: &[6, 8, 1, 1, 1, 1],
                xyz: &METHANOL,
                count: 4,
            }],
        },
        System {
            name: "nh3-2-h2o-4",
            pieces: &[
                Piece {
                    z: &[7, 1, 1, 1],
                    xyz: &AMMONIA,
                    count: 2,
                },
                Piece {
                    z: &[8, 1, 1],
                    xyz: &WATER,
                    count: 4,
                },
            ],
        },
    ]
}

struct Built {
    name: String,
    symbols: Vec<String>,
    species: Vec<u32>,
    groups: Vec<Vec<usize>>,
    template: Array1<f64>,
    /// Hydrogen count within 1.2 Å of each heavy atom, in atom order.
    heavy_h: Vec<(usize, usize)>,
}

fn symbol(z: u32) -> &'static str {
    match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        _ => "X",
    }
}

fn build(system: &System) -> Built {
    let mut symbols = Vec::new();
    let mut species = Vec::new();
    let mut groups = Vec::new();
    let mut coords = Vec::new();
    let mut heavy_h = Vec::new();
    for piece in system.pieces {
        let h_of: Vec<usize> = piece
            .z
            .iter()
            .enumerate()
            .map(|(atom, &z)| {
                if z == 1 {
                    return 0;
                }
                piece
                    .z
                    .iter()
                    .enumerate()
                    .filter(|(other, oz)| **oz == 1 && *other != atom)
                    .filter(|(other, _)| {
                        let mut square = 0.0;
                        for axis in 0..3 {
                            let delta = piece.xyz[atom][axis] - piece.xyz[*other][axis];
                            square += delta * delta;
                        }
                        square.sqrt() <= INTACT_H_A
                    })
                    .count()
            })
            .collect();
        for _copy in 0..piece.count {
            let offset = species.len();
            groups.push((offset..offset + piece.z.len()).collect());
            for (atom, &z) in piece.z.iter().enumerate() {
                if z != 1 {
                    heavy_h.push((offset + atom, h_of[atom]));
                }
                symbols.push(symbol(z).to_string());
                species.push(z);
                coords.extend_from_slice(&piece.xyz[atom]);
            }
        }
    }
    Built {
        name: system.name.to_string(),
        symbols,
        species,
        groups,
        template: Array1::from_vec(coords),
        heavy_h,
    }
}

fn source_stamp(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".to_string())
}

type ForceEval = Box<dyn FnMut(&[f64]) -> (f64, Vec<f64>) + Send>;

struct ForceObjective {
    bounds: Bounds<f64>,
    dim: usize,
    eval: Mutex<ForceEval>,
}

impl ForceObjective {
    fn new(dim: usize, eval: impl FnMut(&[f64]) -> (f64, Vec<f64>) + Send + 'static) -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_elem(dim, -80.0),
                Array1::from_elem(dim, 80.0),
                0.0,
            ),
            dim,
            eval: Mutex::new(Box::new(eval)),
        }
    }

    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let owned;
        let coordinates: &[f64] = if let Some(slice) = x.as_slice() {
            slice
        } else {
            owned = x.iter().copied().collect::<Vec<f64>>();
            &owned
        };
        let (energy, gradient) = (self.eval.lock().expect("engine lock").as_mut())(coordinates);
        if gradient.len() != self.dim || !energy.is_finite() {
            return (f64::NAN, Array1::from_elem(self.dim, f64::NAN));
        }
        (energy, Array1::from_vec(gradient))
    }
}

impl Objective<f64> for ForceObjective {
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

impl Gradient<f64> for ForceObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

impl DifferentiableObjective<f64> for ForceObjective {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        ForceObjective::value_and_gradient(self, x)
    }
}

fn monomers_intact(coordinates: &[f64], built: &Built) -> bool {
    if coordinates.len() != built.symbols.len() * 3 {
        return false;
    }
    let hydrogens: Vec<usize> = built
        .symbols
        .iter()
        .enumerate()
        .filter(|(_, symbol)| symbol.as_str() == "H")
        .map(|(index, _)| index)
        .collect();
    built.heavy_h.iter().all(|&(heavy, expect)| {
        hydrogens
            .iter()
            .filter(|&&hydrogen| {
                let mut square = 0.0;
                for axis in 0..3 {
                    let delta = coordinates[3 * hydrogen + axis] - coordinates[3 * heavy + axis];
                    square += delta * delta;
                }
                square.sqrt() <= INTACT_H_A
            })
            .count()
            == expect
    })
}

fn min_intermolecular(coordinates: &[f64], groups: &[Vec<usize>]) -> f64 {
    let mut best = f64::INFINITY;
    for a in 0..groups.len() {
        for b in (a + 1)..groups.len() {
            for &left in &groups[a] {
                for &right in &groups[b] {
                    let mut square = 0.0;
                    for axis in 0..3 {
                        let delta = coordinates[3 * left + axis] - coordinates[3 * right + axis];
                        square += delta * delta;
                    }
                    best = best.min(square.sqrt());
                }
            }
        }
    }
    best
}

fn separate_overlapped_start(
    mut start: Array1<f64>,
    groups: &[Vec<usize>],
) -> (Array1<f64>, usize) {
    let mut steps = 0usize;
    for _ in 0..256 {
        let slice = start.as_slice().unwrap_or(&[]);
        if min_intermolecular(slice, groups) >= MIN_INTERMOLECULAR_A {
            return (start, steps);
        }
        steps += 1;
        let mut coms = vec![[0.0; 3]; groups.len()];
        for (group_index, atoms) in groups.iter().enumerate() {
            if atoms.is_empty() {
                continue;
            }
            for &atom in atoms {
                for axis in 0..3 {
                    coms[group_index][axis] += start[3 * atom + axis];
                }
            }
            let count = atoms.len() as f64;
            for value in &mut coms[group_index] {
                *value /= count;
            }
        }
        let mut global = [0.0; 3];
        for com in &coms {
            for axis in 0..3 {
                global[axis] += com[axis];
            }
        }
        let count = groups.len().max(1) as f64;
        for value in &mut global {
            *value /= count;
        }
        for (group_index, atoms) in groups.iter().enumerate() {
            let mut direction = [0.0; 3];
            for axis in 0..3 {
                direction[axis] = coms[group_index][axis] - global[axis];
            }
            let mut norm = direction.iter().map(|value| value * value).sum::<f64>();
            if norm < 1e-8 {
                direction = [
                    (group_index % 3) as f64 - 1.0,
                    (group_index / 3) as f64 - 1.0,
                    0.4,
                ];
                norm = direction.iter().map(|value| value * value).sum::<f64>();
            }
            let scale = 0.25 / norm.sqrt();
            for &atom in atoms {
                for axis in 0..3 {
                    start[3 * atom + axis] += direction[axis] * scale;
                }
            }
        }
    }
    (start, steps)
}

fn write_xyz(
    path: &std::path::Path,
    coordinates: &[f64],
    symbols: &[String],
    comment: &str,
) -> std::io::Result<()> {
    let n = coordinates.len() / 3;
    let mut file = fs::File::create(path)?;
    writeln!(file, "{n}")?;
    writeln!(file, "{comment}")?;
    for i in 0..n {
        writeln!(
            file,
            "{} {:.10} {:.10} {:.10}",
            symbols[i],
            coordinates[3 * i],
            coordinates[3 * i + 1],
            coordinates[3 * i + 2]
        )?;
    }
    Ok(())
}

fn read_xyz(path: &std::path::Path) -> Vec<f64> {
    let text =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let mut coords = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index < 2 || line.split_whitespace().count() < 4 {
            continue;
        }
        let mut parts = line.split_whitespace();
        let _symbol = parts.next();
        for _axis in 0..3 {
            coords.push(
                parts
                    .next()
                    .unwrap_or_else(|| panic!("short xyz line in {}", path.display()))
                    .parse()
                    .unwrap_or_else(|error| panic!("xyz coordinate: {error}")),
            );
        }
    }
    coords
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().expect("system");
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).expect("seed");
    let outdir = PathBuf::from(args.get(3).expect("outdir"));
    let system = systems()
        .iter()
        .find(|system| system.name == name)
        .unwrap_or_else(|| panic!("unknown system {name}"));
    let built = build(system);
    fs::create_dir_all(&outdir).expect("outdir");
    assert!(budget > 0, "budget must be positive");

    let cfg = Config::recommended_molecular(built.species.clone(), built.groups.clone(), 1.0);
    let mut rng = StdRng::seed_from_u64(seed);
    let packed = repack_rigid_groups(
        built.template.view(),
        &built.groups,
        cfg.length_scale,
        &mut rng,
    );
    let (start, spread_steps) = match std::env::var("H2H_START_XYZ") {
        Ok(xyz) => {
            let coords = read_xyz(std::path::Path::new(&xyz));
            assert_eq!(coords.len(), built.symbols.len() * 3, "start atom count");
            (Array1::from_vec(coords), 0)
        }
        Err(_) => separate_overlapped_start(packed, &built.groups),
    };
    let start_slice = start.as_slice().expect("contiguous start");
    write_xyz(
        &outdir.join("start.xyz"),
        start_slice,
        &built.symbols,
        &format!(
            "{} seed {seed} rigid repack spread_steps {spread_steps}",
            built.name
        ),
    )
    .expect("start xyz");
    let intact = monomers_intact(start_slice, &built);
    let mut meta = fs::File::create(outdir.join("run.txt")).expect("run.txt");
    writeln!(meta, "code: anneal").ok();
    writeln!(meta, "arm: recommended_molecular").ok();
    writeln!(meta, "system: {}", built.name).ok();
    writeln!(meta, "wall: none").ok();
    if cfg.orbit_complete_on_new {
        eprintln!("orbit completion is on; recommended_molecular was expected to leave it off");
        std::process::exit(2);
    }
    writeln!(meta, "orbit_complete_on_new: {}", cfg.orbit_complete_on_new).ok();
    writeln!(
        meta,
        "source_commit: {}",
        source_stamp("ANNEAL_SOURCE_COMMIT")
    )
    .ok();
    writeln!(meta, "source_tree: {}", source_stamp("ANNEAL_SOURCE_TREE")).ok();
    writeln!(
        meta,
        "config_sha256: {}",
        cfg.resolved_sha256()
            .unwrap_or_else(|_| "unavailable".into())
    )
    .ok();
    writeln!(meta, "molecules: {}", built.groups.len()).ok();
    writeln!(meta, "seed: {seed}").ok();
    writeln!(meta, "budget: {budget}").ok();
    writeln!(meta, "spread_steps: {spread_steps}").ok();
    writeln!(meta, "start_intact: {intact}").ok();
    writeln!(meta, "crate: {}", env!("CARGO_PKG_VERSION")).ok();
    if std::env::var("H2H_START_ONLY").ok().as_deref() == Some("1") {
        writeln!(meta, "start_only: true").ok();
        return;
    }

    let atmnrs: Vec<i32> = built.species.iter().map(|z| *z as i32).collect();
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let mut pot = RgpotObjective::xtb(&atmnrs, box_);
    pot.restart();
    let pot_ns = Arc::new(AtomicU64::new(0));
    let timed_calls = Arc::new(AtomicU64::new(0));
    let pot_ns_eval = Arc::clone(&pot_ns);
    let timed_calls_eval = Arc::clone(&timed_calls);
    let objective = ForceObjective::new(3 * atmnrs.len(), move |coordinates| {
        let started = Instant::now();
        let (energy, gradient) = pot
            .wrapper()
            .value_and_gradient(ArrayView1::from(coordinates));
        pot_ns_eval.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        timed_calls_eval.fetch_add(1, Ordering::Relaxed);
        let values = if gradient.is_standard_layout() {
            gradient.to_vec()
        } else {
            gradient.iter().copied().collect()
        };
        (energy, values)
    });

    let tolerance = component_gradient_tolerance(3 * atmnrs.len()).expect("polish tolerance");
    let mut ledger = Ledger::new(budget);
    let wall = Instant::now();
    let searched = search_from(&objective, &cfg, &mut ledger, start.view(), seed);
    let polished_from = ledger
        .answer_seats()
        .first()
        .map(|(energy, _)| *energy)
        .unwrap_or(f64::NAN);
    let polish_calls = ledger.answer_polish_calls();
    let wall_s = wall.elapsed().as_secs_f64();
    let spent = ledger.spent();
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let calls_timed = timed_calls.load(Ordering::Relaxed);
    let outside_s = (wall_s - pot_s).max(0.0);
    writeln!(meta, "wall_s: {wall_s:.6}").ok();
    writeln!(meta, "energy_calls: {spent}").ok();
    writeln!(meta, "timed_calls: {calls_timed}").ok();
    writeln!(meta, "pot_s: {pot_s:.6}").ok();
    writeln!(meta, "outside_s: {outside_s:.6}").ok();
    if calls_timed > 0 {
        writeln!(meta, "pot_per_call_s: {:.9}", pot_s / calls_timed as f64).ok();
        writeln!(
            meta,
            "outside_per_call_s: {:.9}",
            outside_s / calls_timed as f64
        )
        .ok();
    }
    writeln!(meta, "search_calls: {}", spent.saturating_sub(polish_calls)).ok();
    writeln!(meta, "polish_calls: {polish_calls}").ok();
    writeln!(meta, "polish_component_tolerance_eV_per_A: {tolerance:.6e}").ok();
    writeln!(meta, "polished_from_eV: {polished_from:.10}").ok();
    let (outcome, stats): (Outcome, RelaxStats) = match searched {
        Ok(pair) => pair,
        Err(error) => {
            writeln!(meta, "search_error: {error}").ok();
            writeln!(meta, "best_intact: false").ok();
            write_xyz(
                &outdir.join("best.xyz"),
                start_slice,
                &built.symbols,
                "search failed",
            )
            .ok();
            return;
        }
    };
    writeln!(meta, "quench: {:?}", stats.local_step).ok();
    if let Some((per_call, per_step)) = stats.quench_seconds {
        writeln!(meta, "quench_call_s: {per_call:.6e}").ok();
        writeln!(meta, "quench_step_s: {per_step:.6e}").ok();
    }
    writeln!(meta, "hops: {}", outcome.hops).ok();
    writeln!(meta, "basins: {}", outcome.basins).ok();
    writeln!(meta, "quenches_converged: {}", stats.converged).ok();
    writeln!(meta, "quenches_capped: {}", stats.capped).ok();
    writeln!(meta, "objective_best_eV: {:.10}", outcome.best).ok();
    let fallback = outcome.final_state.clone().unwrap_or_else(|| start.clone());
    let reported = outcome.best_state.as_ref().unwrap_or(&fallback);
    let coords = reported.as_slice().expect("best contiguous");
    let (engine_e, gradient) = objective.value_and_gradient(reported.view());
    let gnorm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    let best_intact = monomers_intact(coords, &built);
    writeln!(meta, "engine_best_eV: {engine_e:.10}").ok();
    writeln!(meta, "gradient_norm_eV_per_A: {gnorm:.6e}").ok();
    let max_force = gradient
        .iter()
        .fold(0.0_f64, |best, value| best.max(value.abs()));
    let verdict = if max_force.is_finite() && max_force <= FORCE_BAR_EV_PER_A {
        "PASS"
    } else {
        "FAIL"
    };
    let line = format!(
        "FORCE_AUDIT seed {seed} quench_gradient {max_force:.6e} bar {FORCE_BAR_EV_PER_A} {verdict}"
    );
    writeln!(meta, "{line}").ok();
    println!("{line}");
    writeln!(meta, "best_intact: {best_intact}").ok();
    write_xyz(
        &outdir.join("best.xyz"),
        coords,
        &built.symbols,
        &format!("{} seed {seed} best", built.name),
    )
    .expect("best xyz");
}
