//! One GFN2-xTB calculator kept for a sequence of geometries.
//!
//! libxtb builds the environment, the calculator, and the molecule once.
//! Each later geometry is an `xtb_updateMolecule` plus one single point.
//! The single point keeps the previous SCC wavefunction in its results
//! object and uses it as the guess. A quench step moves atoms by a fraction
//! of an angstrom, and that guess converges to the same energy as a fresh
//! one. A new proposal can move an atom by more than [`WARM_MOVE_A`]. The
//! old wavefunction then spends the iteration cap and still fails, which is
//! slower than xtb's own guess, so the results object is dropped and the
//! next single point starts cold. A failed SCC drops it as well.

use std::ffi::{CStr, c_void};
use std::time::Instant;

use libloading::Library;

/// Largest atom move, in angstrom, that still reuses the SCC wavefunction.
///
/// Measured on GFN2 (H2O)20 at accuracy 0.01 with one thread: a cold single
/// point is 44–49 ms, a step of 0.05–0.8 Å on the stored wavefunction is
/// 33–36 ms, and a 1.2 Å step is already slower than a cold start. A 2.5 Å
/// step does not converge and burns the 500-iteration cap, about 2.5 s.
/// CREST's native xtb spends 44 ms in the SCC and 52 ms of wall on one cold
/// `--gfn 2 --acc 0.01` single point of the same cluster; 5 ms of that SCC
/// is integrals, 22 ms the iterations, and 16 ms the gradient.
pub const WARM_MOVE_A: f64 = 1.0;

const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_903;
const HARTREE_TO_EV: f64 = 27.211_386_245_988;
const NEG_GRAD_TO_FORCE: f64 = -HARTREE_TO_EV * ANGSTROM_TO_BOHR;
/// Search accuracy. An accuracy of 1 reproduces an accuracy of 0.01 to
/// a few nanohartree on (H2O)20, including a minimum and a strained
/// geometry, and the SCC iterations are what get shorter. The reported
/// geometry is still scored at accuracy 0.01.
pub const SEARCH_ACCURACY: f64 = 1.0;
const ELECTRONIC_TEMPERATURE_K: f64 = 300.0;
const MAX_ITERATIONS: i32 = 500;

type NewEnv = unsafe extern "C" fn() -> *mut c_void;
type DelEnv = unsafe extern "C" fn(*mut *mut c_void);
type SetVerbosity = unsafe extern "C" fn(*mut c_void, i32);
type CheckEnv = unsafe extern "C" fn(*mut c_void) -> i32;
type GetError = unsafe extern "C" fn(*mut c_void, *mut u8, *const i32);
type NewCalc = unsafe extern "C" fn() -> *mut c_void;
type DelCalc = unsafe extern "C" fn(*mut *mut c_void);
type NewRes = unsafe extern "C" fn() -> *mut c_void;
type DelRes = unsafe extern "C" fn(*mut *mut c_void);
type NewMol = unsafe extern "C" fn(
    *mut c_void,
    *const i32,
    *const i32,
    *const f64,
    *const f64,
    *const i32,
    *const f64,
    *const u8,
) -> *mut c_void;
type DelMol = unsafe extern "C" fn(*mut *mut c_void);
type UpdateMol = unsafe extern "C" fn(*mut c_void, *mut c_void, *const f64, *const f64);
type LoadGfn2 = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *const u8);
type SetAccuracy = unsafe extern "C" fn(*mut c_void, *mut c_void, f64);
type SetTemp = unsafe extern "C" fn(*mut c_void, *mut c_void, f64);
type SetMaxIter = unsafe extern "C" fn(*mut c_void, *mut c_void, i32);
type Singlepoint = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
type GetEnergy = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut f64);
type GetGradient = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut f64);

struct Api {
    new_env: NewEnv,
    del_env: DelEnv,
    set_verbosity: SetVerbosity,
    check_env: CheckEnv,
    get_error: GetError,
    new_calc: NewCalc,
    del_calc: DelCalc,
    new_res: NewRes,
    del_res: DelRes,
    new_mol: NewMol,
    del_mol: DelMol,
    update_mol: UpdateMol,
    load_gfn2: LoadGfn2,
    set_accuracy: SetAccuracy,
    set_temp: SetTemp,
    set_max_iter: SetMaxIter,
    singlepoint: Singlepoint,
    get_energy: GetEnergy,
    get_gradient: GetGradient,
}

/// Why a single point was refused.
#[derive(Debug)]
pub struct XtbError {
    message: String,
    /// Time already spent in `xtb_singlepoint` when the call failed.
    pub singlepoint_ns: u64,
}

impl std::fmt::Display for XtbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for XtbError {}

/// Energy, forces, and whether the SCC guess was the previous wavefunction.
#[derive(Debug)]
pub struct XtbPoint {
    /// Total energy in Hartree.
    pub energy_hartree: f64,
    /// Total energy in eV. Same conversion the rgpot engine uses.
    pub energy_ev: f64,
    /// Wall time inside `xtb_singlepoint`.
    pub singlepoint_ns: u64,
    /// The results object still held the previous SCC wavefunction.
    pub reused_wavefunction: bool,
}

/// GFN2 calculator for one composition.
pub struct XtbSession {
    api: Api,
    _lib: Library,
    env: *mut c_void,
    calc: *mut c_void,
    mol: *mut c_void,
    res: *mut c_void,
    loaded: bool,
    accuracy: f64,
    numbers: Vec<i32>,
    previous: Vec<f64>,
    bohr: Vec<f64>,
    box_bohr: [f64; 9],
    gradient: Vec<f64>,
}

// The handles are owned by this session and evaluated from one worker.
unsafe impl Send for XtbSession {}

fn symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, XtbError> {
    unsafe { lib.get(name) }
        .map(|sym: libloading::Symbol<T>| *sym)
        .map_err(|err| XtbError {
            message: format!(
                "libxtb is missing {}: {err}",
                String::from_utf8_lossy(name).trim_end_matches('\0')
            ),
            singlepoint_ns: 0,
        })
}

fn library_path() -> String {
    std::env::var("XTB_LIB").unwrap_or_else(|_| "libxtb.so.6".to_string())
}

impl XtbSession {
    /// Open libxtb at [`SEARCH_ACCURACY`].
    ///
    /// `XTB_LIB` overrides the library name. Otherwise `libxtb.so.6` is
    /// loaded from the process search path.
    pub fn open() -> Result<Self, XtbError> {
        Self::open_with_accuracy(SEARCH_ACCURACY)
    }

    /// Open libxtb at `accuracy`. The tight value is 0.01.
    pub fn open_with_accuracy(accuracy: f64) -> Result<Self, XtbError> {
        let path = library_path();
        let lib = unsafe { Library::new(&path) }.map_err(|err| XtbError {
            message: format!("dlopen {path}: {err}"),
            singlepoint_ns: 0,
        })?;
        let api = Api {
            new_env: symbol(&lib, b"xtb_newEnvironment\0")?,
            del_env: symbol(&lib, b"xtb_delEnvironment\0")?,
            set_verbosity: symbol(&lib, b"xtb_setVerbosity\0")?,
            check_env: symbol(&lib, b"xtb_checkEnvironment\0")?,
            get_error: symbol(&lib, b"xtb_getError\0")?,
            new_calc: symbol(&lib, b"xtb_newCalculator\0")?,
            del_calc: symbol(&lib, b"xtb_delCalculator\0")?,
            new_res: symbol(&lib, b"xtb_newResults\0")?,
            del_res: symbol(&lib, b"xtb_delResults\0")?,
            new_mol: symbol(&lib, b"xtb_newMolecule\0")?,
            del_mol: symbol(&lib, b"xtb_delMolecule\0")?,
            update_mol: symbol(&lib, b"xtb_updateMolecule\0")?,
            load_gfn2: symbol(&lib, b"xtb_loadGFN2xTB\0")?,
            set_accuracy: symbol(&lib, b"xtb_setAccuracy\0")?,
            set_temp: symbol(&lib, b"xtb_setElectronicTemp\0")?,
            set_max_iter: symbol(&lib, b"xtb_setMaxIter\0")?,
            singlepoint: symbol(&lib, b"xtb_singlepoint\0")?,
            get_energy: symbol(&lib, b"xtb_getEnergy\0")?,
            get_gradient: symbol(&lib, b"xtb_getGradient\0")?,
        };
        let env = unsafe { (api.new_env)() };
        if env.is_null() {
            return Err(XtbError {
                message: "xtb_newEnvironment failed".into(),
                singlepoint_ns: 0,
            });
        }
        unsafe { (api.set_verbosity)(env, 0) };
        let calc = unsafe { (api.new_calc)() };
        if calc.is_null() {
            let mut env = env;
            unsafe { (api.del_env)(&mut env) };
            return Err(XtbError {
                message: "xtb_newCalculator failed".into(),
                singlepoint_ns: 0,
            });
        }
        Ok(Self {
            api,
            _lib: lib,
            env,
            calc,
            mol: std::ptr::null_mut(),
            res: std::ptr::null_mut(),
            loaded: false,
            accuracy,
            numbers: Vec::new(),
            previous: Vec::new(),
            bohr: Vec::new(),
            box_bohr: [0.0; 9],
            gradient: Vec::new(),
        })
    }

    /// Energy and forces for `positions` in angstrom.
    ///
    /// Forces are the negative gradient in eV/Å, matching the rgpot engine.
    /// `cell` is the box in angstrom; periodicity stays off, as the engine's
    /// cluster potential does.
    pub fn evaluate(
        &mut self,
        positions: &[f64],
        atomic_numbers: &[i32],
        cell: &[f64],
        forces: &mut [f64],
    ) -> Result<XtbPoint, XtbError> {
        let n = atomic_numbers.len();
        if positions.len() != 3 * n || forces.len() != 3 * n || cell.len() != 9 || n == 0 {
            return Err(XtbError {
                message: "xtb geometry shape is wrong".into(),
                singlepoint_ns: 0,
            });
        }
        let composition_changed = self.numbers.as_slice() != atomic_numbers;
        let moved = max_atom_move(&self.previous, positions);
        let reuse =
            self.loaded && !composition_changed && moved <= WARM_MOVE_A && !self.res.is_null();
        if composition_changed || self.mol.is_null() {
            self.rebuild_molecule(atomic_numbers, positions, cell)?;
        } else {
            self.write_bohr(positions, cell);
            let coordinates = self.bohr.as_ptr();
            let lattice = self.box_bohr.as_ptr();
            unsafe { (self.api.update_mol)(self.env, self.mol, coordinates, lattice) };
        }
        if !reuse {
            self.fresh_results();
        }
        let started = Instant::now();
        unsafe { (self.api.singlepoint)(self.env, self.mol, self.calc, self.res) };
        let singlepoint_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if unsafe { (self.api.check_env)(self.env) } != 0 {
            let message = self.error_text();
            self.discard_after_failure();
            return Err(XtbError {
                message,
                singlepoint_ns,
            });
        }
        let mut hartree = 0.0;
        unsafe { (self.api.get_energy)(self.env, self.res, &mut hartree) };
        self.gradient.resize(3 * n, 0.0);
        unsafe { (self.api.get_gradient)(self.env, self.res, self.gradient.as_mut_ptr()) };
        if !hartree.is_finite() || self.gradient.iter().any(|value| !value.is_finite()) {
            self.discard_after_failure();
            return Err(XtbError {
                message: "xtb returned a non-finite energy or gradient".into(),
                singlepoint_ns,
            });
        }
        for (force, gradient) in forces.iter_mut().zip(&self.gradient) {
            *force = gradient * NEG_GRAD_TO_FORCE;
        }
        self.previous.clear();
        self.previous.extend_from_slice(positions);
        Ok(XtbPoint {
            energy_hartree: hartree,
            energy_ev: hartree * HARTREE_TO_EV,
            singlepoint_ns,
            reused_wavefunction: reuse,
        })
    }

    fn rebuild_molecule(
        &mut self,
        atomic_numbers: &[i32],
        positions: &[f64],
        cell: &[f64],
    ) -> Result<(), XtbError> {
        self.drop_molecule();
        self.fresh_results();
        self.write_bohr(positions, cell);
        let n = i32::try_from(atomic_numbers.len()).unwrap_or(0);
        let charge = 0.0;
        let uhf = 0;
        let periodic = [0u8; 3];
        let coordinates = self.bohr.as_ptr();
        let lattice = self.box_bohr.as_ptr();
        let mol = unsafe {
            (self.api.new_mol)(
                self.env,
                &n,
                atomic_numbers.as_ptr(),
                coordinates,
                &charge,
                &uhf,
                lattice,
                periodic.as_ptr(),
            )
        };
        if mol.is_null() {
            return Err(XtbError {
                message: "xtb_newMolecule failed".into(),
                singlepoint_ns: 0,
            });
        }
        self.mol = mol;
        if !self.loaded {
            unsafe { (self.api.load_gfn2)(self.env, self.mol, self.calc, std::ptr::null()) };
            unsafe { (self.api.set_accuracy)(self.env, self.calc, self.accuracy) };
            unsafe { (self.api.set_temp)(self.env, self.calc, ELECTRONIC_TEMPERATURE_K) };
            unsafe { (self.api.set_max_iter)(self.env, self.calc, MAX_ITERATIONS) };
            self.loaded = true;
        }
        self.numbers.clear();
        self.numbers.extend_from_slice(atomic_numbers);
        Ok(())
    }

    fn write_bohr(&mut self, positions: &[f64], cell: &[f64]) {
        self.bohr.resize(positions.len(), 0.0);
        for (dst, src) in self.bohr.iter_mut().zip(positions) {
            *dst = src * ANGSTROM_TO_BOHR;
        }
        for (dst, src) in self.box_bohr.iter_mut().zip(cell) {
            *dst = src * ANGSTROM_TO_BOHR;
        }
    }

    fn fresh_results(&mut self) {
        if !self.res.is_null() {
            unsafe { (self.api.del_res)(&mut self.res) };
            self.res = std::ptr::null_mut();
        }
        self.res = unsafe { (self.api.new_res)() };
    }

    fn discard_after_failure(&mut self) {
        // The failed wavefunction is not a guess, and the environment keeps
        // the error until it is replaced.
        self.drop_molecule();
        if !self.res.is_null() {
            unsafe { (self.api.del_res)(&mut self.res) };
            self.res = std::ptr::null_mut();
        }
        self.previous.clear();
        self.loaded = false;
        if !self.calc.is_null() {
            unsafe { (self.api.del_calc)(&mut self.calc) };
        }
        self.calc = unsafe { (self.api.new_calc)() };
        if !self.env.is_null() {
            unsafe { (self.api.del_env)(&mut self.env) };
        }
        self.env = unsafe { (self.api.new_env)() };
        if !self.env.is_null() {
            unsafe { (self.api.set_verbosity)(self.env, 0) };
        }
    }

    fn error_text(&self) -> String {
        let mut buf = [0u8; 512];
        let len = i32::try_from(buf.len()).unwrap_or(512);
        unsafe { (self.api.get_error)(self.env, buf.as_mut_ptr(), &len) };
        let text = CStr::from_bytes_until_nul(&buf)
            .map(|text| text.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "xtb single point failed".into());
        if text.is_empty() {
            "xtb single point failed".into()
        } else {
            text
        }
    }

    fn drop_molecule(&mut self) {
        if !self.mol.is_null() {
            unsafe { (self.api.del_mol)(&mut self.mol) };
            self.mol = std::ptr::null_mut();
        }
        self.numbers.clear();
    }
}

impl Drop for XtbSession {
    fn drop(&mut self) {
        if !self.res.is_null() {
            unsafe { (self.api.del_res)(&mut self.res) };
        }
        if !self.calc.is_null() {
            unsafe { (self.api.del_calc)(&mut self.calc) };
        }
        self.drop_molecule();
        if !self.env.is_null() {
            unsafe { (self.api.del_env)(&mut self.env) };
        }
    }
}

fn max_atom_move(previous: &[f64], next: &[f64]) -> f64 {
    if previous.len() != next.len() || previous.is_empty() {
        return f64::INFINITY;
    }
    let mut furthest = 0.0f64;
    for (start, end) in previous.chunks_exact(3).zip(next.chunks_exact(3)) {
        let dx = start[0] - end[0];
        let dy = start[1] - end[1];
        let dz = start[2] - end[2];
        furthest = furthest.max((dx * dx + dy * dy + dz * dz).sqrt());
    }
    furthest
}
