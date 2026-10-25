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
    GFN2_MAX_ITERATIONS, GFN2_METHOD, GOOD_TURING_CHECKPOINT_INTERVAL, MolecularCatalogPresetError,
    OXYGEN_DISTANCE_CENSUS_RADIUS_A, PBE_BASIS, PBE_CHARGE, PBE_DISPERSION, PBE_ENGINE_KIND,
    PBE_HEXAMER_BUDGET, PBE_MULTIPLICITY, PBE_TASK, PBE_THEORY, PBE_XC, PETMAD_DEVICE,
    PETMAD_ENGINE_KIND, PETMAD_HEXAMER_BUDGET, PETMAD_MODEL_NAME, RGPOT_REV, RGPOT_VERSION,
    VACUUM_BOX, WATER_HEXAMER_MOLECULES, WATER_HEXAMER_SEED_COUNT, WATER_HEXAMER_SEED_START,
    WATER_ISOMER_NAMES, WaterSearchArm, assign_isomer, classify_geometry, configure_water_arm,
    good_turing_stop_reason, isomer_reference_gate, monomers_intact, reference_coordinates,
    sorted_oxygen_distances, water_groups, water_species,
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

fn run_arm(engine: &str, arm_name: &str, smoke: bool) -> String {
    let arm = match arm_name {
        "recommended" => WaterSearchArm::Recommended,
        "communicating" => WaterSearchArm::Communicating,
        "derived" => WaterSearchArm::Derived,
        "good_turing" => WaterSearchArm::GoodTuring,
        other => panic!("unknown arm {other}"),
    };
    let budget = if smoke {
        SMOKE_BUDGET
    } else {
        match engine {
            "gfn2" => GFN2_HEXAMER_BUDGET,
            "pbe" => PBE_HEXAMER_BUDGET,
            "petmad" => PETMAD_HEXAMER_BUDGET,
            other => panic!("unknown engine {other}"),
        }
    };
    let seed0 = WATER_HEXAMER_SEED_START;
    let seeds = if smoke { 1 } else { WATER_HEXAMER_SEED_COUNT };
    let mut text = String::new();
    text.push_str(&format!(
        "\nengine: {engine}\narm: {arm_name}\nbudget: {budget}\nseeds: {seed0}..{}\n",
        seed0 + seeds - 1
    ));
    text.push_str(&engine_settings(engine));
    let n_atoms = 3 * WATER_HEXAMER_MOLECULES;
    let loaded = load_engine(engine, n_atoms);
    let Ok(objective) = loaded else {
        text.push_str(&format!(
            "engine_loaded: false\nsearch: not started\nload_error: {}\n",
            loaded.err().unwrap_or_default()
        ));
        return text;
    };
    text.push_str("engine_loaded: true\n");
    let species = water_species(WATER_HEXAMER_MOLECULES).expect("hexamer species");
    let groups = water_groups(WATER_HEXAMER_MOLECULES).expect("hexamer groups");
    let mut cfg = Config::recommended_molecular(species, groups.clone(), 1.0);
    configure_water_arm(&mut cfg, arm);
    cfg.move_library = MoveLibrary::Molecular {
        groups: groups.clone(),
        reactive: false,
    };
    let references = load_references();
    let reference_views: Vec<(&str, &[f64])> = references
        .iter()
        .map(|(name, coordinates)| (name.as_str(), coordinates.as_slice()))
        .collect();
    let gate_result = isomer_reference_gate(&reference_views, WATER_HEXAMER_MOLECULES);
    let (gate, gate_text, gate_reason) = match &gate_result {
        Ok(Some(value)) => (
            Some(*value),
            format!("{value:.6}"),
            "half the minimum inter-reference IRA RMSD",
        ),
        Ok(None) => (
            None,
            "unset".to_owned(),
            "fewer than two deposited references",
        ),
        Err(MolecularCatalogPresetError::AssignmentNeedsIra) => {
            (None, "unset".to_owned(), "ira feature not built")
        }
        Err(MolecularCatalogPresetError::IsolatedTemplate) => {
            (None, "unset".to_owned(), "isolated-water template")
        }
        Err(_) => (None, "unset".to_owned(), "reference gate failed"),
    };
    text.push_str(&format!(
        "isomer_gate_A: {gate_text}\nisomer_gate_when: before seeds\nisomer_gate_reason: {gate_reason}\n"
    ));
    text.push_str(&classify_gofee(&reference_views, gate));
    let template = reference_coordinates(WATER_HEXAMER_MOLECULES).expect("template");
    for seed in seed0..seed0 + seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let start = repack_rigid_groups(
            ArrayView1::from(&template),
            &groups,
            cfg.length_scale,
            &mut rng,
        );
        let start_intact =
            monomers_intact(start.as_slice().unwrap_or(&[]), WATER_HEXAMER_MOLECULES)
                .unwrap_or(false);
        if seed == seed0 {
            let started = Instant::now();
            let (energy, gradient) = objective.value_and_gradient(start.view());
            let eval_us = started.elapsed().as_micros();
            let finite = energy.is_finite() && gradient.iter().all(|value| value.is_finite());
            text.push_str(&format!(
                "first_eval_us: {eval_us}\nfirst_energy: {energy}\nfirst_gradient_finite: {finite}\n"
            ));
            if eval_us > 0 {
                let full_calls = match engine {
                    "gfn2" => GFN2_HEXAMER_BUDGET,
                    "pbe" => PBE_HEXAMER_BUDGET,
                    _ => PETMAD_HEXAMER_BUDGET,
                } as u128
                    * WATER_HEXAMER_SEED_COUNT as u128;
                let estimate_s = eval_us * full_calls / 1_000_000;
                text.push_str(&format!(
                    "one_geometry_full_run_estimate_s: {estimate_s}\nestimate_note: one geometry, quench overhead not included, smoke\n"
                ));
            }
        }
        let mut ledger = Ledger::new(budget);
        let wall = Instant::now();
        let (outcome, stats) = if arm == WaterSearchArm::GoodTuring {
            let mut census = BasinCensus::new(
                sorted_oxygen_distances(&template, WATER_HEXAMER_MOLECULES)
                    .expect("descriptor")
                    .len(),
                OXYGEN_DISTANCE_CENSUS_RADIUS_A,
            )
            .expect("census");
            let mut trace = String::new();
            let mut checkpoint =
                |snapshot: anneal_core::methods::cluster_hopping::ChainCheckpoint<'_>| {
                    for boundary in snapshot.quench_boundaries() {
                        if boundary.status() != QuenchStatus::Validated {
                            continue;
                        }
                        let state = boundary.state();
                        let Some(state) = state.as_slice() else {
                            continue;
                        };
                        let Ok(descriptor) =
                            sorted_oxygen_distances(state, WATER_HEXAMER_MOLECULES)
                        else {
                            continue;
                        };
                        if census.observe(&descriptor).is_err() {
                            continue;
                        }
                        let mass = census
                            .unseen_mass()
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "none".to_owned());
                        trace.push_str(&format!(
                        "good_turing_trace: force_calls={} visits={} unseen_mass={mass} minimum_visits={PRODUCTION_MINIMUM_VISITS} max_unseen_mass={PRODUCTION_MAX_UNSEEN_MASS}\n",
                        snapshot.charged(),
                        census.total_visits()
                    ));
                    }
                    if let Some(reason) = good_turing_stop_reason(&census) {
                        CheckpointAction::Retire {
                            reason: reason.to_owned(),
                        }
                    } else {
                        CheckpointAction::Continue
                    }
                };
            let result = search_from_with_checkpoint(
                &objective,
                &cfg,
                &mut ledger,
                start.view(),
                seed,
                GOOD_TURING_CHECKPOINT_INTERVAL,
                &mut checkpoint,
            )
            .expect("finite start");
            text.push_str(&trace);
            text.push_str(&format!(
                "good_turing_saturated: {}\n",
                good_turing_stop_reason(&census).is_some()
            ));
            result
        } else {
            search_from(&objective, &cfg, &mut ledger, start.view(), seed).expect("finite start")
        };
        let wall_s = wall.elapsed().as_secs_f64();
        let best = outcome.best;
        let intact = outcome
            .best_state
            .as_ref()
            .and_then(|state| state.as_slice())
            .and_then(|state| monomers_intact(state, WATER_HEXAMER_MOLECULES).ok())
            .unwrap_or(false);
        let (isomer_name, isomer_rmsd, isomer_reason) = if !intact {
            (
                "unassigned".to_owned(),
                "none".to_owned(),
                "dissociated monomer",
            )
        } else if reference_views.is_empty() {
            (
                "unassigned".to_owned(),
                "none".to_owned(),
                "no deposited references",
            )
        } else {
            match assign_isomer(
                outcome
                    .best_state
                    .as_ref()
                    .and_then(|state| state.as_slice())
                    .unwrap_or(&[]),
                WATER_HEXAMER_MOLECULES,
                &reference_views,
                gate,
            ) {
                Ok(matched) => (
                    matched.name.unwrap_or_else(|| "unassigned".to_owned()),
                    matched
                        .rmsd
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "none".to_owned()),
                    if matched.name.is_some() {
                        "inside gate"
                    } else if gate.is_none() {
                        "gate unset"
                    } else {
                        "outside gate"
                    },
                ),
                Err(MolecularCatalogPresetError::AssignmentNeedsIra) => (
                    "unassigned".to_owned(),
                    "none".to_owned(),
                    "ira feature not built",
                ),
                Err(_) => (
                    "unassigned".to_owned(),
                    "none".to_owned(),
                    "assignment failed",
                ),
            }
        };
        let per_second = if wall_s > 0.0 {
            ledger.spent() as f64 / wall_s
        } else {
            0.0
        };
        text.push_str(&format!(
            "seed: {seed}\nstart_monomers_intact: {start_intact}\nbest_energy: {best}\nbest_monomers_intact: {intact}\nisomer: {isomer_name}\nisomer_rmsd_A: {isomer_rmsd}\nisomer_gate_A: {gate_text}\nisomer_reason: {isomer_reason}\nforce_calls: {}\nwall_seconds: {wall_s:.6}\nforce_calls_per_second: {per_second:.6}\nhops: {}\nbasins: {}\nconverged: {}\nquenches: {}\nhits: unscored\nwilson_95: unscored\nmedian_par2_force: unscored\nmedian_par2_wall: unscored\nunscored_reason: no same-level reference energy in this tree\n",
            ledger.spent(),
            outcome.hops,
            outcome.basins,
            stats.converged,
            stats.total()
        ));
    }
    text
}

fn engine_settings(engine: &str) -> String {
    match engine {
        "gfn2" => format!(
            "engine_kind: {ENGINE_KIND}\nmethod: {GFN2_METHOD}\naccuracy: {GFN2_ACCURACY}\netemp_K: 300\nmaxiter: {GFN2_MAX_ITERATIONS}\ncharge: 0\nuhf: 0\nvacuum_box_A: {VACUUM_BOX}\nfull_budget: {GFN2_HEXAMER_BUDGET}\n"
        ),
        "pbe" => format!(
            "engine_kind: {PBE_ENGINE_KIND}\ncode: NWChem\nbasis: {PBE_BASIS}\ntheory: {PBE_THEORY}\nxc: {PBE_XC}\ndispersion: {PBE_DISPERSION}\ncharge: {PBE_CHARGE}\nmultiplicity: {PBE_MULTIPLICITY}\ntask: {PBE_TASK}\ncell: none\nfull_budget: {PBE_HEXAMER_BUDGET}\n"
        ),
        "petmad" => format!(
            "engine_kind: {PETMAD_ENGINE_KIND}\nmodel_name: {PETMAD_MODEL_NAME}\ndevice: {PETMAD_DEVICE}\nlength_unit: angstrom\nenergy_unit: eV\nvacuum_box_A: {VACUUM_BOX}\nmodel_cutoff: unnamed, inside the model file\ncheck_consistency: 0\nuncertainty_threshold: 0\nrandom_rotation: 0\nfull_budget: {PETMAD_HEXAMER_BUDGET}\n"
        ),
        other => format!("engine_kind: {other}\n"),
    }
}

fn load_engine(engine: &str, n_atoms: usize) -> Result<ForceObjective, String> {
    let atomic_numbers: Vec<i32> = (0..WATER_HEXAMER_MOLECULES)
        .flat_map(|_| [8, 1, 1])
        .collect();
    match engine {
        "gfn2" => load_gfn2(&atomic_numbers, n_atoms),
        "pbe" => load_pbe(atomic_numbers, n_atoms),
        "petmad" => load_petmad(atomic_numbers, n_atoms),
        other => Err(format!("unknown engine {other}")),
    }
}

fn load_gfn2(atomic_numbers: &[i32], n_atoms: usize) -> Result<ForceObjective, String> {
    let Some(path) = engine_file("RGPOT_XTB_ENGINE", "libxtb_engine.so") else {
        return Err("libxtb_engine.so not found (RGPOT_XTB_ENGINE or ./libxtb_engine.so)".into());
    };
    let box_ = vacuum_box();
    let numbers = atomic_numbers.to_vec();
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        RgpotObjective::xtb(&numbers, box_)
    }));
    let Ok(pot) = loaded else {
        return Err(format!("gfn2 dlopen failed for {}", path.display()));
    };
    emit_engine_manifest("xtb");
    let dim = 3 * n_atoms;
    Ok(ForceObjective::new(dim, move |coordinates: &[f64]| {
        let objective = pot.wrapper();
        let (energy, gradient) = objective.value_and_gradient(ArrayView1::from(coordinates));
        (energy, gradient.to_vec())
    }))
}

fn load_pbe(_atomic_numbers: Vec<i32>, _n_atoms: usize) -> Result<ForceObjective, String> {
    let library = engine_file("POTENTIAL_LIBRARY", "libnwchemc.so")
        .or_else(|| engine_file("RGPOT_NWCHEMC_ENGINE", "libnwchemc.so"));
    let config = env_file("POTENTIAL_CONFIG").or_else(|| env_file("NWCHEMC_CONFIG"));
    let library_state = match &library {
        Some(path) => format!("library present at {}", path.display()),
        None => "libnwchemc.so not found".to_owned(),
    };
    let config_state = match &config {
        Some(path) => format!("config present at {}", path.display()),
        None => "PotentialConfig bytes not found".to_owned(),
    };
    Err(format!(
        "NWChem PBE/6-31G* was not evaluated. {library_state}. {config_state}. This build of rgpot-core does not enable the profile client, and the NWChem driver binary is not the embed SDK. Settings remain basis {PBE_BASIS}, theory {PBE_THEORY}, xc {PBE_XC}, charge {PBE_CHARGE}, multiplicity {PBE_MULTIPLICITY}, task {PBE_TASK}, dispersion {PBE_DISPERSION}."
    ))
}

fn load_petmad(atomic_numbers: Vec<i32>, n_atoms: usize) -> Result<ForceObjective, String> {
    let Some(library) = engine_file("RGPOT_METATOMIC_ENGINE", "libmetatomic_engine.so") else {
        return Err(
            "libmetatomic_engine.so not found (RGPOT_METATOMIC_ENGINE or ./libmetatomic_engine.so)"
                .into(),
        );
    };
    let Some(model) = env_file("RGPOT_METATOMIC_MODEL") else {
        return Err(format!(
            "PET-MAD model not found (RGPOT_METATOMIC_MODEL). Cited file name is {PETMAD_MODEL_NAME}."
        ));
    };
    let model_name = model
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unnamed");
    if model_name != PETMAD_MODEL_NAME {
        eprintln!("petmad model file is {model_name}; the cited name is {PETMAD_MODEL_NAME}");
    }
    let lib = unsafe { Library::new(&library) }.map_err(|error| error.to_string())?;
    let create: Symbol<MtaCreate> =
        unsafe { lib.get(b"rgpot_mta_create\0") }.map_err(|error| error.to_string())?;
    let destroy: Symbol<MtaDestroy> =
        unsafe { lib.get(b"rgpot_mta_destroy\0") }.map_err(|error| error.to_string())?;
    let force: Symbol<MtaForce> =
        unsafe { lib.get(b"rgpot_mta_force\0") }.map_err(|error| error.to_string())?;
    let model_c =
        CString::new(model.to_string_lossy().as_bytes()).map_err(|error| error.to_string())?;
    let device = CString::new(PETMAD_DEVICE).map_err(|error| error.to_string())?;
    let unit = CString::new("angstrom").map_err(|error| error.to_string())?;
    let cfg = MtaConfig {
        model_path: model_c.as_ptr(),
        device: device.as_ptr(),
        length_unit: unit.as_ptr(),
        extensions_directory: std::ptr::null(),
        check_consistency: 0,
        uncertainty_threshold: 0.0,
        dtype_override: std::ptr::null(),
        random_rotation: 0,
        n_symmetry_rotations: 0,
        so3_probe_scatter: 0,
        torch_determinism_strict: 0,
    };
    let mut err = [0u8; 512];
    let pot = unsafe { create(&cfg, err.as_mut_ptr(), err.len()) };
    if pot.is_null() {
        return Err(format!(
            "rgpot_mta_create failed: {}",
            String::from_utf8_lossy(&err)
        ));
    }
    let force =
        unsafe { std::mem::transmute::<Symbol<MtaForce>, Symbol<'static, MtaForce>>(force) };
    let destroy =
        unsafe { std::mem::transmute::<Symbol<MtaDestroy>, Symbol<'static, MtaDestroy>>(destroy) };
    let kernel = PetKernel {
        _lib: lib,
        pot,
        force,
        destroy,
        _model: model_c,
        _device: device,
        _unit: unit,
        atomic_numbers,
        box_: vacuum_box(),
    };
    let dim = 3 * n_atoms;
    Ok(ForceObjective::new(dim, move |coordinates: &[f64]| {
        let mut forces = vec![0.0; coordinates.len()];
        let mut energy = 0.0;
        let mut variance = 0.0;
        let rc = unsafe {
            (kernel.force)(
                kernel.pot,
                (coordinates.len() / 3) as i64,
                coordinates.as_ptr(),
                kernel.atomic_numbers.as_ptr(),
                forces.as_mut_ptr(),
                &mut energy,
                &mut variance,
                kernel.box_.as_ptr(),
            )
        };
        if rc != 0 || !energy.is_finite() {
            return (f64::NAN, vec![f64::NAN; coordinates.len()]);
        }
        // The metatomic ABI returns forces. The search gradient is the
        // negative of the force, matching the NWChem profile adapter.
        let gradient = forces.iter().map(|force| -force).collect();
        let _ = variance;
        (energy, gradient)
    }))
}

fn vacuum_box() -> [f64; 9] {
    [
        VACUUM_BOX, 0.0, 0.0, 0.0, VACUUM_BOX, 0.0, 0.0, 0.0, VACUUM_BOX,
    ]
}

fn engine_file(var: &str, filename: &str) -> Option<PathBuf> {
    env_file(var).or_else(|| {
        let local = PathBuf::from(filename);
        local.is_file().then_some(local)
    })
}

fn env_file(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(var)?);
    path.is_file().then_some(path)
}

fn classify_gofee(references: &[(&str, &[f64])], gate: Option<f64>) -> String {
    let Ok(dir) = std::env::var("WATER_GOFEE_DIR") else {
        return "gofee: unscored\ngofee_reason: no deposited geometries\n".to_owned();
    };
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) => return format!("gofee: unscored\ngofee_reason: {error}\n"),
    };
    let mut rows = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("xyz") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Some(coordinates) = parse_xyz(&text) else {
            continue;
        };
        rows.push((path, coordinates));
    }
    if rows.is_empty() {
        return "gofee: unscored\ngofee_reason: no deposited geometries\n".to_owned();
    }
    let mut assigned = 0usize;
    let mut unassigned = 0usize;
    let mut lines = String::from("gofee: classified\n");
    for (path, coordinates) in &rows {
        match classify_geometry(coordinates, WATER_HEXAMER_MOLECULES, references, gate) {
            Ok(found) => {
                if found.name.is_some() {
                    assigned += 1;
                } else {
                    unassigned += 1;
                }
                let name = found
                    .name
                    .clone()
                    .unwrap_or_else(|| "unassigned".to_owned());
                let rmsd = found
                    .rmsd
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "none".to_owned());
                lines.push_str(&format!(
                    "gofee_geometry: {} intact: {} isomer: {name} isomer_rmsd_A: {rmsd}\n",
                    path.display(),
                    found.intact
                ));
            }
            Err(error) => {
                unassigned += 1;
                lines.push_str(&format!(
                    "gofee_geometry: {} intact: unknown isomer: unassigned isomer_reason: {error}\n",
                    path.display()
                ));
            }
        }
    }
    lines.push_str(&format!(
        "gofee_assigned: {assigned}\ngofee_unassigned: {unassigned}\n"
    ));
    lines
}

fn load_references() -> Vec<(String, Vec<f64>)> {
    let Ok(dir) = std::env::var("WATER_ISOMER_DIR") else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for name in WATER_ISOMER_NAMES {
        let path = PathBuf::from(&dir).join(format!("{name}.xyz"));
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if let Some(coordinates) = parse_xyz(&text) {
            if monomers_intact(&coordinates, WATER_HEXAMER_MOLECULES).unwrap_or(false) {
                found.push((name.to_owned(), coordinates));
            } else {
                eprintln!("isomer reference {name} failed the monomer check");
            }
        }
    }
    found
}

fn parse_xyz(text: &str) -> Option<Vec<f64>> {
    let mut lines = text.lines();
    let count: usize = lines.next()?.trim().parse().ok()?;
    if count != 3 * WATER_HEXAMER_MOLECULES {
        return None;
    }
    let _comment = lines.next()?;
    let mut coordinates = Vec::with_capacity(3 * count);
    for line in lines.take(count) {
        let mut fields = line.split_whitespace();
        let element = fields.next()?;
        if element != "O" && element != "H" {
            return None;
        }
        for _ in 0..3 {
            coordinates.push(fields.next()?.parse().ok()?);
        }
    }
    (coordinates.len() == 9 * WATER_HEXAMER_MOLECULES).then_some(coordinates)
}
