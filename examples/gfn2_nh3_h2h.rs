//! One (NH3)m GFN2-xTB seed.
//!
//! Usage: gfn2_nh3_h2h <m_molecules> <budget> <seed> <outdir>
//!
//! The start is a rigid repack of the GFN2 ammonia monomer from
//! `StdRng::seed_from_u64(seed)`. `H2H_START_ONLY=1` writes that start
//! and exits before any energy call. `H2H_START_XYZ=<file>` starts from
//! that geometry, which is how both arms share one start.
//!
//! The objective is `Config::recommended_molecular` on the rgpot GFN2
//! handle. There is no added wall and no constant set for ammonia.
//! `pot_s` is time inside `value_and_gradient`. `outside_s` is the rest
//! of the wall.

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
use anneal_core::methods::cluster_search::{RelaxStats, polish_answer, search_from};
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rgpot_eindir::RgpotObjective;

/// Quenched maximum force bar. 0.001 Hartree per Bohr is 0.0514221 eV per angstrom.
const FORCE_BAR_EV_PER_A: f64 = 0.0514221;
/// Same neighbour cutoff the water scorer uses for a covalent hydrogen.
const INTACT_H_A: f64 = 1.2;
const MIN_INTERMOLECULAR_A: f64 = 1.5;

/// GFN2-xTB 6.7.1 vtight ammonia, nitrogen shifted to the origin.
const AMMONIA: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.9438867749, 0.0000000000, 0.3668373921],
    [-0.4719433871, 0.8174299251, 0.3668373931],
    [-0.4719433871, -0.8174299251, 0.3668373931],
];

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

fn species_and_groups(m: usize) -> (Vec<u32>, Vec<Vec<usize>>) {
    let species = (0..m).flat_map(|_| [7u32, 1, 1, 1]).collect();
    let groups = (0..m)
        .map(|molecule| (4 * molecule..4 * molecule + 4).collect())
        .collect();
    (species, groups)
}

fn template(m: usize) -> Array1<f64> {
    let mut coordinates = Array1::zeros(12 * m);
    for molecule in 0..m {
        for (atom, xyz) in AMMONIA.iter().enumerate() {
            for axis in 0..3 {
                coordinates[12 * molecule + 3 * atom + axis] = xyz[axis];
            }
        }
    }
    coordinates
}

/// Every nitrogen has exactly three hydrogens within 1.2 Å.
fn monomers_intact(coordinates: &[f64], m: usize) -> bool {
    if coordinates.len() != 12 * m {
        return false;
    }
    let mut hydrogens = Vec::new();
    let mut nitrogens = Vec::new();
    for atom in 0..4 * m {
        let xyz = [
            coordinates[3 * atom],
            coordinates[3 * atom + 1],
            coordinates[3 * atom + 2],
        ];
        if atom % 4 == 0 {
            nitrogens.push(xyz);
        } else {
            hydrogens.push(xyz);
        }
    }
    nitrogens.iter().all(|nitrogen| {
        hydrogens
            .iter()
            .filter(|hydrogen| {
                let mut square = 0.0;
                for axis in 0..3 {
                    let delta = hydrogen[axis] - nitrogen[axis];
                    square += delta * delta;
                }
                square.sqrt() <= INTACT_H_A
            })
            .count()
            == 3
    })
}

fn min_intermolecular(coordinates: &[f64], m: usize) -> f64 {
    let mut best = f64::INFINITY;
    for a in 0..m {
        for b in (a + 1)..m {
            for ia in 0..4 {
                for ib in 0..4 {
                    let left = 4 * a + ia;
                    let right = 4 * b + ib;
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
    m: usize,
) -> (Array1<f64>, usize) {
    let mut steps = 0usize;
    for _ in 0..256 {
        let slice = start.as_slice().unwrap_or(&[]);
        if monomers_intact(slice, m) && min_intermolecular(slice, m) >= MIN_INTERMOLECULAR_A {
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

fn write_xyz(path: &std::path::Path, coordinates: &[f64], comment: &str) -> std::io::Result<()> {
    let n = coordinates.len() / 3;
    let mut file = fs::File::create(path)?;
    writeln!(file, "{n}")?;
    writeln!(file, "{comment}")?;
    for i in 0..n {
        let symbol = if i % 4 == 0 { "N" } else { "H" };
        writeln!(
            file,
            "{symbol} {:.10} {:.10} {:.10}",
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
            let value: f64 = parts
                .next()
                .unwrap_or_else(|| panic!("short xyz line in {}", path.display()))
                .parse()
                .unwrap_or_else(|error| panic!("xyz coordinate: {error}"));
            coords.push(value);
        }
    }
    coords
}

fn max_abs_force(gradient: &[f64]) -> f64 {
    gradient
        .iter()
        .fold(0.0_f64, |best, value| best.max(value.abs()))
}

fn force_audit_line(seed: u64, max_force: f64) -> String {
    let verdict = if max_force.is_finite() && max_force <= FORCE_BAR_EV_PER_A {
        "PASS"
    } else {
        "FAIL"
    };
    format!(
        "FORCE_AUDIT seed {seed} quench_gradient {max_force:.6e} bar {FORCE_BAR_EV_PER_A} {verdict}"
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let m: usize = args
        .first()
        .and_then(|v| v.parse().ok())
        .expect("m_molecules");
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).expect("seed");
    let outdir = PathBuf::from(args.get(3).expect("outdir"));
    fs::create_dir_all(&outdir).expect("outdir");
    assert!(budget > 0 && m > 0, "budget and m must be positive");

    let (species, groups) = species_and_groups(m);
    let cfg = Config::recommended_molecular(species, groups.clone(), 1.0);
    let mut rng = StdRng::seed_from_u64(seed);
    let packed = repack_rigid_groups(template(m).view(), &groups, cfg.length_scale, &mut rng);
    let (start, spread_steps) = match std::env::var("H2H_START_XYZ") {
        Ok(xyz) => {
            let coords = read_xyz(std::path::Path::new(&xyz));
            assert_eq!(coords.len(), 12 * m, "start geometry atom count");
            (Array1::from_vec(coords), 0)
        }
        Err(_) => separate_overlapped_start(packed, &groups, m),
    };
    let start_slice = start.as_slice().expect("contiguous start");
    write_xyz(
        &outdir.join("start.xyz"),
        start_slice,
        &format!("(NH3){m} seed {seed} rigid repack spread_steps {spread_steps}"),
    )
    .expect("start xyz");
    let intact = monomers_intact(start_slice, m);
    let mut meta = fs::File::create(outdir.join("run.txt")).expect("run.txt");
    writeln!(meta, "code: anneal").ok();
    writeln!(meta, "arm: recommended_molecular").ok();
    writeln!(meta, "system: (NH3){m}").ok();
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
    writeln!(meta, "molecules: {m}").ok();
    writeln!(meta, "seed: {seed}").ok();
    writeln!(meta, "budget: {budget}").ok();
    writeln!(meta, "spread_steps: {spread_steps}").ok();
    writeln!(meta, "start_intact: {intact}").ok();
    writeln!(meta, "crate: {}", env!("CARGO_PKG_VERSION")).ok();
    if std::env::var("H2H_START_ONLY").ok().as_deref() == Some("1") {
        writeln!(meta, "start_only: true").ok();
        writeln!(meta, "search: not started").ok();
        return;
    }

    let atmnrs: Vec<i32> = (0..m).flat_map(|_| [7, 1, 1, 1]).collect();
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

    let search_budget = budget.saturating_sub(cfg.relax_steps).max(budget / 2);
    let tolerance = component_gradient_tolerance(3 * atmnrs.len()).expect("polish tolerance");
    let mut ledger = Ledger::new(search_budget);
    let wall = Instant::now();
    let mut searched = search_from(&objective, &cfg, &mut ledger, start.view(), seed);
    let mut polish = Ledger::new(budget - ledger.spent());
    let mut polished_from = f64::NAN;
    if let Ok((outcome, _)) = searched.as_mut()
        && let Some(state) = outcome.best_state.as_ref()
        && let Some((energy, x)) = polish_answer(
            &objective,
            &mut polish,
            state.view(),
            tolerance,
            cfg.relax_steps,
        )
        && energy < outcome.best
        && x.as_slice()
            .is_some_and(|coords| monomers_intact(coords, m))
    {
        polished_from = outcome.best;
        outcome.best = energy;
        outcome.best_state = Some(x);
    }
    let wall_s = wall.elapsed().as_secs_f64();
    let spent = ledger.spent() + polish.spent();
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
    writeln!(meta, "search_calls: {}", ledger.spent()).ok();
    writeln!(meta, "polish_calls: {}", polish.spent()).ok();
    writeln!(meta, "polish_component_tolerance_eV_per_A: {tolerance:.6e}").ok();
    writeln!(meta, "polished_from_eV: {polished_from:.10}").ok();
    let (outcome, stats): (Outcome, RelaxStats) = match searched {
        Ok(pair) => pair,
        Err(error) => {
            writeln!(meta, "search_error: {error}").ok();
            writeln!(meta, "hops: 0").ok();
            writeln!(meta, "objective_best_eV: nan").ok();
            writeln!(meta, "best_intact: false").ok();
            write_xyz(
                &outdir.join("best.xyz"),
                start_slice,
                "search failed; start geometry kept",
            )
            .expect("best xyz");
            return;
        }
    };
    writeln!(meta, "hops: {}", outcome.hops).ok();
    writeln!(meta, "basins: {}", outcome.basins).ok();
    writeln!(meta, "quenches_converged: {}", stats.converged).ok();
    writeln!(meta, "quenches_capped: {}", stats.capped).ok();
    writeln!(meta, "screens: {}", stats.screens).ok();
    writeln!(meta, "objective_best_eV: {:.10}", outcome.best).ok();
    let mut trace = fs::File::create(outdir.join("improvements.txt")).expect("trace");
    writeln!(trace, "hops charged basins e_eV").ok();
    for &(hops, charged, basins, energy) in &outcome.improvements {
        writeln!(trace, "{hops} {charged} {basins} {energy:.10}").ok();
    }
    let fallback = outcome
        .final_state
        .as_ref()
        .map(|state| state.to_owned())
        .unwrap_or_else(|| start.clone());
    let reported = outcome.best_state.as_ref().unwrap_or(&fallback);
    let coords = reported.as_slice().expect("best contiguous");
    let (engine_e, gradient) = objective.value_and_gradient(reported.view());
    let gnorm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    let best_intact = monomers_intact(coords, m);
    writeln!(meta, "engine_best_eV: {engine_e:.10}").ok();
    writeln!(meta, "gradient_norm_eV_per_A: {gnorm:.6e}").ok();
    let max_force = max_abs_force(gradient.as_slice().unwrap_or(&[]));
    let line = force_audit_line(seed, max_force);
    writeln!(meta, "{line}").ok();
    println!("{line}");
    writeln!(meta, "best_intact: {best_intact}").ok();
    write_xyz(
        &outdir.join("best.xyz"),
        coords,
        &format!("objective {engine_e:.10} eV"),
    )
    .expect("best xyz");
}
