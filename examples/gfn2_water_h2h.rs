//! One (H2O)m GFN2-xTB seed for the arm-3 head-to-head.
//!
//! Usage: gfn2_water_h2h <m_molecules> <budget> <seed> <outdir> [target_energy]
//!
//! `target_energy`, or `H2O_TARGET_EV` when the argument is omitted, is an
//! optional stop in eV. Unset, the search runs out the budget. The binary
//! has no built-in reference.
//!
//! The start is a rigid repack of the isolated-water template from
//! `StdRng::seed_from_u64(seed)`, then the hexamer driver's overlap
//! separation. `H2H_START_ONLY=1` writes that start and exits before
//! any energy call, so CREST can be given the same geometry.
//! `H2H_START_XYZ=<file>` starts from that geometry instead, which is how
//! a run is seeded from the CREST minimum.
//!
//! The objective is the rgpot GFN2 handle (eV, angstrom) plus the
//! hexamer O-H wall, which is zero inside 1.15 angstrom. The reported
//! `engine_best_eV` is the handle energy at the best geometry.

#[path = "common/rgpot_eindir.rs"]
mod rgpot_eindir;

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Nanoseconds spent inside the potential, for the wall split.
static POTENTIAL_NS: AtomicU64 = AtomicU64::new(0);

use anneal_core::catalog::molecular::{
    WaterSearchArm, component_gradient_tolerance, configure_water_arm, monomers_intact,
    reference_coordinates, water_groups, water_species,
};
use anneal_core::methods::cluster_hopping::Outcome;
use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, repack_rigid_groups};
use anneal_core::methods::cluster_search::{RelaxStats, polish_answer, search_from};
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rgpot_eindir::RgpotObjective;

const OH_WALL_ONSET_A: f64 = 1.15;
const OH_WALL_EV_PER_A2: f64 = 200.0;

/// Quenched maximum force bar. 0.001 Hartree per Bohr is 0.0514221 eV per angstrom.
const FORCE_BAR_EV_PER_A: f64 = 0.0514221;

fn source_stamp(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".to_string())
}

type ForceEval = Box<dyn FnMut(&[f64]) -> (f64, Vec<f64>) + Send>;

struct ForceObjective {
    bounds: Bounds<f64>,
    dim: usize,
    n_molecules: usize,
    eval: Mutex<ForceEval>,
}

impl ForceObjective {
    fn new(
        dim: usize,
        n_molecules: usize,
        eval: impl FnMut(&[f64]) -> (f64, Vec<f64>) + Send + 'static,
    ) -> Self {
        Self {
            bounds: Bounds::new(
                Array1::from_elem(dim, -80.0),
                Array1::from_elem(dim, 80.0),
                0.0,
            ),
            dim,
            n_molecules,
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
        let (energy, mut gradient) = (self.eval.lock().expect("engine lock").as_mut())(coordinates);
        if gradient.len() != self.dim || !energy.is_finite() {
            return (f64::NAN, Array1::from_elem(self.dim, f64::NAN));
        }
        let (wall, wall_gradient) = oh_wall(coordinates, self.n_molecules);
        if wall != 0.0 {
            for (component, extra) in gradient.iter_mut().zip(wall_gradient) {
                *component += extra;
            }
        }
        (energy + wall, Array1::from_vec(gradient))
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

fn oh_wall(coordinates: &[f64], n_molecules: usize) -> (f64, Vec<f64>) {
    let mut energy = 0.0;
    let mut gradient = vec![0.0; coordinates.len()];
    if coordinates.len() != 9 * n_molecules {
        return (0.0, gradient);
    }
    for molecule in 0..n_molecules {
        let oxygen = 9 * molecule;
        for hydrogen in [oxygen + 3, oxygen + 6] {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = coordinates[hydrogen + axis] - coordinates[oxygen + axis];
                square += delta * delta;
            }
            let distance = square.sqrt();
            if !distance.is_finite() || distance <= OH_WALL_ONSET_A || distance < 1e-12 {
                continue;
            }
            let excess = distance - OH_WALL_ONSET_A;
            energy += OH_WALL_EV_PER_A2 * excess * excess;
            let slope = 2.0 * OH_WALL_EV_PER_A2 * excess / distance;
            for axis in 0..3 {
                let delta = coordinates[hydrogen + axis] - coordinates[oxygen + axis];
                gradient[oxygen + axis] -= slope * delta;
                gradient[hydrogen + axis] += slope * delta;
            }
        }
    }
    (energy, gradient)
}

fn separate_overlapped_start(mut start: Array1<f64>, n_molecules: usize) -> (Array1<f64>, usize) {
    let Ok(groups) = water_groups(n_molecules) else {
        return (start, 0);
    };
    let mut steps = 0usize;
    for _ in 0..256 {
        let slice = start.as_slice().unwrap_or(&[]);
        let intact = monomers_intact(slice, n_molecules).unwrap_or(false);
        if intact && min_intermolecular(slice, n_molecules) >= 1.5 {
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

fn min_intermolecular(coordinates: &[f64], n_molecules: usize) -> f64 {
    if coordinates.len() != 9 * n_molecules {
        return 0.0;
    }
    let mut best = f64::INFINITY;
    for a in 0..n_molecules {
        for b in (a + 1)..n_molecules {
            for ia in 0..3 {
                for ib in 0..3 {
                    let ia = 3 * a + ia;
                    let ib = 3 * b + ib;
                    let mut square = 0.0;
                    for axis in 0..3 {
                        let delta = coordinates[3 * ia + axis] - coordinates[3 * ib + axis];
                        square += delta * delta;
                    }
                    best = best.min(square.sqrt());
                }
            }
        }
    }
    best
}

fn write_xyz(path: &std::path::Path, coordinates: &[f64], comment: &str) -> std::io::Result<()> {
    let n = coordinates.len() / 3;
    let mut file = fs::File::create(path)?;
    writeln!(file, "{n}")?;
    writeln!(file, "{comment}")?;
    for i in 0..n {
        let symbol = if i % 3 == 0 { "O" } else { "H" };
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

fn overlap_self_check() {
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let pot = RgpotObjective::xtb(&[1, 1], box_);
    // 0.05 Å is the separation that made seed 15014's trial step
    // abort xtb with "Error in Broyden matrix inversion".
    let coordinates = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0];
    let before = rgpot_eindir::geometry_rejected_count();
    let answer = pot.fresh_evaluation(&coordinates);
    let rejected = rgpot_eindir::geometry_rejected_count();
    assert!(
        answer.is_err(),
        "overlapped geometry was accepted as a finite evaluation"
    );
    assert!(rejected > before, "overlapped geometry was handed to xtb");
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

fn audit_geometry(path: &std::path::Path, molecules: usize, seed: u64, outdir: &std::path::Path) {
    let coords = read_xyz(path);
    assert_eq!(coords.len(), 9 * molecules, "audit geometry atom count");
    let atmnrs: Vec<i32> = (0..molecules).flat_map(|_| [8, 1, 1]).collect();
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let pot = RgpotObjective::xtb(&atmnrs, box_);
    let objective = ForceObjective::new(coords.len(), molecules, move |coordinates| {
        let (energy, gradient) = pot
            .wrapper()
            .value_and_gradient(ArrayView1::from(coordinates));
        let values = if gradient.is_standard_layout() {
            gradient.to_vec()
        } else {
            gradient.iter().copied().collect()
        };
        (energy, values)
    });
    let (energy, gradient) = objective.value_and_gradient(ArrayView1::from(&coords));
    let max_force = max_abs_force(gradient.as_slice().unwrap_or(&[]));
    let line = force_audit_line(seed, max_force);
    let mut meta = fs::File::create(outdir.join("force-audit.txt")).expect("force audit");
    writeln!(meta, "code: anneal").ok();
    writeln!(meta, "audit: quenched maximum force").ok();
    writeln!(meta, "seed: {seed}").ok();
    writeln!(meta, "molecules: {molecules}").ok();
    writeln!(meta, "objective_eV: {energy:.10}").ok();
    writeln!(meta, "{line}").ok();
    println!("{line}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("overlap") {
        overlap_self_check();
        return;
    }
    let m: usize = args
        .first()
        .and_then(|v| v.parse().ok())
        .expect("m_molecules");
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).expect("budget");
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).expect("seed");
    let outdir = PathBuf::from(args.get(3).expect("outdir"));
    let target_energy = args
        .get(4)
        .map(|v| v.parse::<f64>().expect("target_energy"))
        .or_else(|| {
            std::env::var("H2O_TARGET_EV")
                .ok()
                .map(|v| v.parse::<f64>().expect("H2O_TARGET_EV"))
        });
    fs::create_dir_all(&outdir).expect("outdir");
    assert!(budget > 0 && m > 0, "budget and m must be positive");
    if let Ok(xyz) = std::env::var("H2H_AUDIT_XYZ") {
        audit_geometry(std::path::Path::new(&xyz), m, seed, &outdir);
        return;
    }

    let species = water_species(m).expect("species");
    let groups = water_groups(m).expect("groups");
    let mut cfg = Config::recommended_molecular(species, groups.clone(), 1.0);
    configure_water_arm(&mut cfg, WaterSearchArm::Recommended);
    cfg.target_energy = target_energy;
    cfg.move_library = MoveLibrary::Molecular {
        groups: groups.clone(),
        reactive: false,
        reverse_cycles: true,
    };
    let template = reference_coordinates(m).expect("template");
    let mut rng = StdRng::seed_from_u64(seed);
    let packed = repack_rigid_groups(
        ArrayView1::from(&template),
        &groups,
        cfg.length_scale,
        &mut rng,
    );
    let (start, spread_steps) = match std::env::var("H2H_START_XYZ") {
        Ok(xyz) => {
            let coords = read_xyz(std::path::Path::new(&xyz));
            assert_eq!(coords.len(), 9 * m, "start geometry atom count");
            (Array1::from_vec(coords), 0)
        }
        Err(_) => separate_overlapped_start(packed, m),
    };
    let start_slice = start.as_slice().expect("contiguous start");
    let start_path = outdir.join("start.xyz");
    write_xyz(
        &start_path,
        start_slice,
        &format!("(H2O){m} seed {seed} rigid repack spread_steps {spread_steps}"),
    )
    .expect("start xyz");
    let intact = monomers_intact(start_slice, m).unwrap_or(false);
    let mut meta = fs::File::create(outdir.join("run.txt")).expect("run.txt");
    writeln!(meta, "code: anneal").ok();
    writeln!(meta, "arm: recommended_molecular").ok();
    writeln!(meta, "water_arm: recommended").ok();
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
    match cfg.target_energy {
        Some(energy) => writeln!(meta, "target_energy_eV: {energy:.6}").ok(),
        None => writeln!(meta, "target_energy_eV: off").ok(),
    };
    writeln!(meta, "spread_steps: {spread_steps}").ok();
    writeln!(meta, "start_intact: {intact}").ok();
    writeln!(meta, "crate: {}", env!("CARGO_PKG_VERSION")).ok();
    writeln!(meta, "oh_wall_onset_A: {OH_WALL_ONSET_A}").ok();
    writeln!(meta, "oh_wall_eV_per_A2: {OH_WALL_EV_PER_A2}").ok();
    if std::env::var("H2H_START_ONLY").ok().as_deref() == Some("1") {
        writeln!(meta, "start_only: true").ok();
        writeln!(meta, "search: not started").ok();
        return;
    }

    let atmnrs: Vec<i32> = (0..m).flat_map(|_| [8, 1, 1]).collect();
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let mut pot = RgpotObjective::xtb(&atmnrs, box_);
    // One pot for the seed. A search starts with the engine's own guess.
    pot.restart();
    let n_molecules = m;
    let objective = ForceObjective::new(3 * atmnrs.len(), n_molecules, move |coordinates| {
        let pot_at = Instant::now();
        let (energy, gradient) = pot
            .wrapper()
            .value_and_gradient(ArrayView1::from(coordinates));
        POTENTIAL_NS.fetch_add(pot_at.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let values = if gradient.is_standard_layout() {
            gradient.to_vec()
        } else {
            gradient.iter().copied().collect()
        };
        (energy, values)
    });

    // The answer polish is paid from the same budget as the search, so the
    // total stays at `budget`. One quench's iteration cap is set aside.
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
            .is_some_and(|coords| monomers_intact(coords, m).unwrap_or(false))
    {
        polished_from = outcome.best;
        outcome.best = energy;
        outcome.best_state = Some(x);
    }
    let wall_s = wall.elapsed().as_secs_f64();
    let spent = ledger.spent() + polish.spent();
    let potential_s = POTENTIAL_NS.load(Ordering::Relaxed) as f64 / 1e9;
    let outside_s = (wall_s - potential_s).max(0.0);
    writeln!(meta, "wall_s: {wall_s:.6}").ok();
    writeln!(meta, "potential_s: {potential_s:.6}").ok();
    writeln!(meta, "outside_s: {outside_s:.6}").ok();
    writeln!(meta, "energy_calls: {spent}").ok();
    writeln!(meta, "search_calls: {}", ledger.spent()).ok();
    writeln!(meta, "polish_calls: {}", polish.spent()).ok();
    writeln!(meta, "polish_component_tolerance_eV_per_A: {tolerance:.6e}").ok();
    writeln!(meta, "polished_from_eV: {polished_from:.10}").ok();
    writeln!(meta, "scc_rejected: {}", rgpot_eindir::scc_rejected_count()).ok();
    writeln!(
        meta,
        "geometry_rejected: {}",
        rgpot_eindir::geometry_rejected_count()
    )
    .ok();
    let (outcome, stats): (Outcome, RelaxStats) = match searched {
        Ok(pair) => pair,
        Err(error) => {
            writeln!(meta, "search_error: {error}").ok();
            writeln!(meta, "hops: 0").ok();
            writeln!(meta, "objective_best_eV: nan").ok();
            writeln!(meta, "engine_best_eV: nan").ok();
            writeln!(meta, "best_intact: false").ok();
            write_xyz(
                &outdir.join("best.xyz"),
                start_slice,
                "search failed; start geometry kept",
            )
            .expect("best xyz");
            let line = force_audit_line(seed, f64::INFINITY);
            writeln!(meta, "{line}").ok();
            println!("{line}");
            return;
        }
    };
    writeln!(meta, "hops: {}", outcome.hops).ok();
    writeln!(meta, "basins: {}", outcome.basins).ok();
    let overlap_rejected = stats.overlap_rejected + outcome.overlap_rejected;
    writeln!(meta, "overlap_rejected: {overlap_rejected}").ok();
    writeln!(
        meta,
        "proposal_overlap_rejected: {}",
        outcome.overlap_rejected
    )
    .ok();
    writeln!(
        meta,
        "linesearch_overlap_rejected: {}",
        stats.overlap_rejected
    )
    .ok();
    writeln!(
        meta,
        "overlap_note: a geometry closer than 0.35 angstrom is refused before the energy call and is not charged"
    )
    .ok();
    writeln!(meta, "quenches_converged: {}", stats.converged).ok();
    writeln!(meta, "quenches_capped: {}", stats.capped).ok();
    writeln!(
        meta,
        "best_record: converged quench at the preset record gradient"
    )
    .ok();
    writeln!(meta, "screens: {}", stats.screens).ok();
    writeln!(meta, "objective_best_eV: {:.10}", outcome.best).ok();
    let mut trace = fs::File::create(outdir.join("improvements.txt")).expect("trace");
    writeln!(trace, "hops charged basins e_eV").ok();
    for &(hops, charged, basins, energy) in &outcome.improvements {
        writeln!(trace, "{hops} {charged} {basins} {energy:.10}").ok();
    }
    let mut arms = fs::File::create(outdir.join("arms.txt")).expect("arms");
    writeln!(arms, "move proposed accepted improved").ok();
    for row in &outcome.proposal_counts {
        writeln!(
            arms,
            "{} {} {} {}",
            row.move_type, row.proposed, row.accepted, row.improved
        )
        .ok();
    }
    let fallback = outcome
        .final_state
        .as_ref()
        .map(|state| state.to_owned())
        .unwrap_or_else(|| start.clone());
    let reported = outcome.best_state.as_ref().unwrap_or(&fallback);
    let coords = reported.as_slice().expect("best contiguous");
    let (wall_e, _) = oh_wall(coords, m);
    let (engine_e, gradient) = objective.value_and_gradient(reported.view());
    // The objective includes the wall. The engine energy is that total
    // minus the wall, which is zero when every covalent O-H is inside
    // the onset.
    let engine_best = if engine_e.is_finite() {
        engine_e - wall_e
    } else {
        f64::NAN
    };
    let gnorm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    let best_intact = monomers_intact(coords, m).unwrap_or(false);
    writeln!(meta, "wall_at_best_eV: {wall_e:.10}").ok();
    writeln!(meta, "engine_best_eV: {engine_best:.10}").ok();
    writeln!(meta, "gradient_norm_eV_per_A: {gnorm:.6e}").ok();
    let max_force = max_abs_force(gradient.as_slice().unwrap_or(&[]));
    let line = force_audit_line(seed, max_force);
    writeln!(meta, "{line}").ok();
    println!("{line}");
    writeln!(meta, "best_intact: {best_intact}").ok();
    write_xyz(
        &outdir.join("best.xyz"),
        coords,
        &format!("objective {engine_e:.10} eV wall {wall_e:.10} eV"),
    )
    .expect("best xyz");
}
