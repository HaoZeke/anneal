//! (H2O)6 under GFN2-xTB, NWChem PBE/6-31G*, and PET-MAD.
//!
//! Usage: `water_hexamer [gfn2|pbe|petmad|all] [recommended|communicating|derived|good_turing|all]`
//!
//! `PAPER_SMOKE=1` runs seed 0 at a short budget and writes a file
//! whose name contains `SMOKE`. That file is not a paper table. A run
//! without it uses seeds 0-31 and 4000 force evaluations on every
//! engine, unless `WATER_FORCE_BUDGET`, `WATER_SEED_FIRST` and
//! `WATER_SEED_COUNT` select a block. Full campaign numbers still come
//! from the v0.12.0 tag.
//!
//! Hit columns that need a same-level reference energy stay unscored.
//! Isomer names are assigned only when `WATER_ISOMER_DIR` holds
//! same-level geometries in O, H, H order.

#[path = "common/pipe_engine.rs"]
mod pipe_engine;
#[path = "common/rgpot_eindir.rs"]
mod rgpot_eindir;

use std::ffi::{CString, c_char, c_void};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use anneal_core::catalog::census::{
    BasinCensus, PRODUCTION_MAX_UNSEEN_MASS, PRODUCTION_MINIMUM_VISITS,
};
use anneal_core::catalog::molecular::{
    EINDIR_REV, EINDIR_VERSION, ENGINE_KIND, GFN2_ACCURACY, GFN2_HEXAMER_BUDGET,
    GFN2_MAX_ITERATIONS, GFN2_METHOD, GOOD_TURING_CHECKPOINT_INTERVAL, MolecularCatalogPresetError,
    OXYGEN_DISTANCE_CENSUS_RADIUS_A, PBE_BASIS, PBE_CHARGE, PBE_DISPERSION, PBE_ENGINE_KIND,
    PBE_HEXAMER_BUDGET, PBE_MULTIPLICITY, PBE_TASK, PBE_THEORY, PBE_XC, PETMAD_DEVICE,
    PETMAD_ENGINE_KIND, PETMAD_HEXAMER_BUDGET, PETMAD_MODEL_NAME, RGPOT_REV, RGPOT_VERSION,
    VACUUM_BOX, WATER_HEXAMER_MOLECULES, WATER_HEXAMER_SEED_COUNT, WATER_HEXAMER_SEED_START,
    WATER_ISOMER_NAMES, WaterSearchArm, assign_isomer, classify_geometry, configure_water_arm,
    good_turing_target_stop, isomer_reference_gate, isomer_witness, monomers_intact,
    reference_coordinates, sorted_oxygen_distances, water_groups, water_species,
};
use anneal_core::methods::cluster_hopping::{
    CheckpointAction, Config, Ledger, MoveLibrary, QuenchStatus, repack_rigid_groups,
};
use anneal_core::methods::cluster_search::{search_from, search_from_with_checkpoint};
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use eindir_core::{Bounds, Objective};
use libloading::{Library, Symbol};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rgpot_eindir::{RgpotObjective, emit_engine_manifest};

const SMOKE_BUDGET: usize = 12;

/// Onset of the template O-H wall. Intact GFN2 minima sit near 0.97 A, so the
/// wall is zero there. A cartesian quench from an overlapped start can otherwise
/// walk a covalent bond past the 1.2 A integrity check.
const OH_WALL_ONSET_A: f64 = 1.15;

/// Stiffness of that wall, in eV per square angstrom.
const OH_WALL_EV_PER_A2: f64 = 200.0;

#[repr(C)]
struct MtaConfig {
    model_path: *const c_char,
    device: *const c_char,
    length_unit: *const c_char,
    extensions_directory: *const c_char,
    check_consistency: i32,
    uncertainty_threshold: f64,
    dtype_override: *const c_char,
    random_rotation: i32,
    n_symmetry_rotations: i64,
    so3_probe_scatter: i32,
    torch_determinism_strict: i32,
}

type MtaCreate = unsafe extern "C" fn(*const MtaConfig, *mut u8, usize) -> *mut c_void;
type MtaDestroy = unsafe extern "C" fn(*mut c_void);
type MtaForce = unsafe extern "C" fn(
    *mut c_void,
    i64,
    *const f64,
    *const i32,
    *mut f64,
    *mut f64,
    *mut f64,
    *const f64,
) -> i32;

type ForceEval = Box<dyn FnMut(&[f64]) -> (f64, Vec<f64>) + Send>;

struct ForceObjective {
    bounds: Bounds<f64>,
    dim: usize,
    eval: Mutex<ForceEval>,
}

impl ForceObjective {
    fn new(dim: usize, eval: impl FnMut(&[f64]) -> (f64, Vec<f64>) + Send + 'static) -> Self {
        let extent = 80.0;
        Self {
            bounds: Bounds::new(
                Array1::from_elem(dim, -extent),
                Array1::from_elem(dim, extent),
                0.0,
            ),
            dim,
            eval: Mutex::new(Box::new(eval)),
        }
    }

    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let coordinates: Vec<f64> = x.iter().copied().collect();
        let (energy, mut gradient) =
            (self.eval.lock().expect("engine lock").as_mut())(&coordinates);
        if gradient.len() != self.dim || !energy.is_finite() {
            return (f64::NAN, Array1::from_elem(self.dim, f64::NAN));
        }
        let (wall, wall_gradient) = oh_wall(&coordinates);
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

fn main() {}

/// Push overlapped molecules apart without drawing more random numbers.
///
/// `repack_rigid_groups` keeps each water's internal geometry and can still
/// place a hydrogen of one molecule inside 1.2 A of another oxygen. The
/// integrity check then calls the start dissociated, and the cartesian quench
/// can walk a covalent bond apart. Each step is a rigid translation.
fn separate_overlapped_start(mut start: Array1<f64>, n_molecules: usize) -> (Array1<f64>, usize) {
    let Ok(groups) = water_groups(n_molecules) else {
        return (start, 0);
    };
    let mut steps = 0usize;
    for _ in 0..64 {
        let intact = start
            .as_slice()
            .and_then(|coordinates| monomers_intact(coordinates, n_molecules).ok())
            .unwrap_or(false);
        if intact {
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
            for axis in 0..3 {
                coms[group_index][axis] /= count;
            }
        }
        let mut global = [0.0; 3];
        for com in &coms {
            for axis in 0..3 {
                global[axis] += com[axis];
            }
        }
        let count = groups.len().max(1) as f64;
        for axis in 0..3 {
            global[axis] /= count;
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

fn covalent_oh_span(coordinates: &[f64]) -> (f64, f64) {
    if coordinates.len() != 9 * WATER_HEXAMER_MOLECULES {
        return (f64::NAN, f64::NAN);
    }
    let mut min = f64::INFINITY;
    let mut max = 0.0_f64;
    for molecule in 0..WATER_HEXAMER_MOLECULES {
        let oxygen = 9 * molecule;
        for hydrogen in [oxygen + 3, oxygen + 6] {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = coordinates[hydrogen + axis] - coordinates[oxygen + axis];
                square += delta * delta;
            }
            let distance = square.sqrt();
            min = min.min(distance);
            max = max.max(distance);
        }
    }
    (min, max)
}

/// Harmonic wall outside [`OH_WALL_ONSET_A`]. Zero, with zero gradient, inside it.
fn oh_wall(coordinates: &[f64]) -> (f64, Vec<f64>) {
    let mut energy = 0.0;
    let mut gradient = vec![0.0; coordinates.len()];
    if coordinates.len() != 9 * WATER_HEXAMER_MOLECULES {
        return (0.0, gradient);
    }
    for molecule in 0..WATER_HEXAMER_MOLECULES {
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
                let component = (coordinates[hydrogen + axis] - coordinates[oxygen + axis]) * slope;
                gradient[hydrogen + axis] += component;
                gradient[oxygen + axis] -= component;
            }
        }
    }
    (energy, gradient)
}

fn write_geometry(seed: u64, coordinates: &[f64], energy: f64) {
    let Ok(dir) = std::env::var("WATER_GEOM_DIR") else {
        return;
    };
    if coordinates.len() != 9 * WATER_HEXAMER_MOLECULES {
        return;
    }
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let mut text = format!("18\nseed {seed} energy {energy:.10}\n");
    for molecule in 0..WATER_HEXAMER_MOLECULES {
        for (offset, symbol) in [(0, "O"), (3, "H"), (6, "H")] {
            let base = 9 * molecule + offset;
            text.push_str(&format!(
                "{symbol} {:.10} {:.10} {:.10}\n",
                coordinates[base],
                coordinates[base + 1],
                coordinates[base + 2]
            ));
        }
    }
    let path = PathBuf::from(dir).join(format!("seed-{seed}.xyz"));
    let _ = std::fs::write(path, text);
}

