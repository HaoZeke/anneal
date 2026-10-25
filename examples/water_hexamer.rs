//! (H2O)6 under GFN2-xTB, NWChem PBE/6-31G*, and PET-MAD.
//!
//! Usage: `water_hexamer [gfn2|pbe|petmad|all] [recommended|communicating|derived|good_turing|all]`
//!
//! `PAPER_SMOKE=1` runs seed 0 at a short budget and writes a file
//! whose name contains `SMOKE`. That file is not a paper table. A run
//! without it uses seeds 0-31 and 4000 force evaluations on every
//! engine. Full campaign numbers still come from the v0.12.0 tag.
//!
//! Hit columns that need a same-level reference energy stay unscored.
//! Isomer names are assigned only when `WATER_ISOMER_DIR` holds
//! same-level geometries in O, H, H order.

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
    GFN2_MAX_ITERATIONS, GFN2_METHOD, GOOD_TURING_CHECKPOINT_INTERVAL,
    OXYGEN_DISTANCE_CENSUS_RADIUS_A, PBE_BASIS, PBE_CHARGE, PBE_DISPERSION, PBE_ENGINE_KIND,
    PBE_HEXAMER_BUDGET, PBE_MULTIPLICITY, PBE_TASK, PBE_THEORY, PBE_XC, PETMAD_DEVICE,
    PETMAD_ENGINE_KIND, PETMAD_HEXAMER_BUDGET, PETMAD_MODEL_NAME, RGPOT_REV, RGPOT_VERSION,
    VACUUM_BOX, WATER_HEXAMER_MOLECULES, WATER_HEXAMER_SEED_COUNT, WATER_HEXAMER_SEED_START,
    WATER_ISOMER_NAMES, WaterSearchArm, assign_isomer, configure_water_arm,
    good_turing_stop_reason, monomers_intact, reference_coordinates, sorted_oxygen_distances,
    water_groups, water_species,
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
        let (energy, gradient) = (self.eval.lock().expect("engine lock").as_mut())(&coordinates);
        if gradient.len() == self.dim {
            (energy, Array1::from_vec(gradient))
        } else {
            (f64::NAN, Array1::from_elem(self.dim, f64::NAN))
        }
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

struct PetKernel {
    _lib: Library,
    pot: *mut c_void,
    force: Symbol<'static, MtaForce>,
    destroy: Symbol<'static, MtaDestroy>,
    _model: CString,
    _device: CString,
    _unit: CString,
    atomic_numbers: Vec<i32>,
    box_: [f64; 9],
}

unsafe impl Send for PetKernel {}

impl Drop for PetKernel {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.pot) };
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let engines = selection(
        args.first().map(String::as_str).unwrap_or("all"),
        &[("gfn2", "gfn2"), ("pbe", "pbe"), ("petmad", "petmad")],
    );
    let arms = selection(
        args.get(1).map(String::as_str).unwrap_or("all"),
        &[
            ("recommended", "recommended"),
            ("communicating", "communicating"),
            ("derived", "derived"),
            ("good_turing", "good_turing"),
        ],
    );
    let smoke = std::env::var("PAPER_SMOKE").ok().as_deref() == Some("1");
    let mut report = String::new();
    report.push_str(&format!(
        "label: {}\ndriver: water_hexamer\nfull_runs: v0.12.0 tag\nrgpot: {RGPOT_VERSION} {RGPOT_REV}\neindir: {EINDIR_VERSION} {EINDIR_REV}\n",
        if smoke { "smoke" } else { "campaign-shape" }
    ));
    for engine in &engines {
        for arm in &arms {
            report.push_str(&run_arm(engine, arm, smoke));
        }
    }
    print!("{report}");
    if let Ok(path) = std::env::var("PAPER_OUT") {
        if smoke && !path.contains("SMOKE") {
            panic!("PAPER_SMOKE output path must contain SMOKE");
        }
        std::fs::write(&path, &report).unwrap_or_else(|error| panic!("write {path}: {error}"));
    } else if smoke {
        std::fs::write("water-hexamer-SMOKE.txt", &report).expect("smoke file");
    }
}

fn selection(arg: &str, choices: &[(&str, &str)]) -> Vec<String> {
    if arg == "all" {
        return choices.iter().map(|(_, name)| (*name).to_owned()).collect();
    }
    if choices.iter().any(|(name, _)| *name == arg) {
        return vec![arg.to_owned()];
    }
    panic!(
        "unknown selector {arg}; expected all or {}",
        choices
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join("|")
    );
}
