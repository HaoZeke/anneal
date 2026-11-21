//! C ABI of anneal-core.
//!
//! The box ensemble is exported for callers that own an objective behind
//! two C callbacks, value and gradient, over DLPack tensors. The layout of
//! the callbacks is the one rgmin's `rgmin_minimize` takes, so one pair of
//! callbacks serves a local rgmin fit and this global stage.
//!
//! The driver is deterministic for a seed and runs its replicas in one
//! thread, in one fixed order, so several MPI ranks that run it with the
//! same seed and identical callback results evaluate the same points in the
//! same order; a collective objective is safe behind the callbacks.

// The C names are the ABI: snake-case types and upper-case variants, as
// cbindgen emits them.
#![allow(non_camel_case_types)]

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned, DLPackVersion,
    DLTensor,
};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEnsembleResult, BoxEscape, GleEscapeConfig,
    box_ensemble_optimize_with_coverage,
};
use crate::methods::ensemble::HistoryMode;
use crate::methods::gle_langevin::GleNoise;
use crate::methods::minima_hopping::{HistoryMembership, SharedVisitPolicy};

/// Status of a C ABI call.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum anneal_status_t {
    /// Completed.
    ANNEAL_SUCCESS = 0,
    /// Null pointer, inconsistent length, or an invalid configuration.
    ANNEAL_INVALID_PARAMETER = 1,
    /// Panic or internal failure behind the C boundary.
    ANNEAL_INTERNAL_ERROR = 2,
    /// The callback asked the search to stop. The best point found is kept.
    ANNEAL_STOPPED = 3,
}

/// `f(x)` callback over a borrowed CPU float64 DLPack tensor.
pub type anneal_eval_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> anneal_status_t;

/// `∇f(x)` callback. Writes into the pre-allocated `grad_out` tensor.
pub type anneal_grad_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> anneal_status_t;

/// Escape mechanism of one replica between polishes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum anneal_box_escape_t {
    /// Reflected Gaussian kick without gradient work in the proposal.
    ANNEAL_BOX_ESCAPE_GAUSSIAN = 0,
    /// Persistent Langevin segments driven by the gradient callback.
    ANNEAL_BOX_ESCAPE_LANGEVIN = 1,
}

/// Configuration of one communicating-chain box ensemble.
///
/// Coverage is well-tempered metadynamics on the box: every evaluated
/// region receives a Gaussian deposit of `coverage_height` objective units
/// and `coverage_radius` normalized RMS box distance, tempered by
/// `well_tempering`, and the deposits are shared between the replicas.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct anneal_box_config_t {
    /// Chains that divide the aggregate budget.
    pub replicas: usize,
    /// Combined objective and gradient work units across all chains.
    pub budget: usize,
    /// Seed of the deterministic driver.
    pub seed: u64,
    /// Coverage deposit radius in normalized RMS box distance.
    pub coverage_radius: f64,
    /// Initial deposit height in objective units; zero disables coverage.
    pub coverage_height: f64,
    /// Well-tempering factor, finite and greater than one.
    pub well_tempering: f64,
    /// Escape mechanism of every replica.
    pub escape: anneal_box_escape_t,
    /// Langevin steps per escape segment (Langevin escape only).
    pub langevin_steps: usize,
    /// Langevin harmonic frequency (Langevin escape only).
    pub langevin_omega0: f64,
    /// Langevin time step (Langevin escape only).
    pub langevin_dt: f64,
}

/// Outcome of one box ensemble.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct anneal_box_report_t {
    /// Objective at the returned point.
    pub value: f64,
    /// Objective evaluations charged, polish included.
    pub n_evals: usize,
    /// Gradient evaluations charged, polish included.
    pub n_grads: usize,
    /// Accepted plus rejected hops across replicas.
    pub hops: usize,
}

/// Minimum history the replicas keep.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum anneal_box_history_t {
    /// No history: the control with no lookup cost.
    ANNEAL_BOX_HISTORY_NONE = 0,
    /// One history per replica.
    ANNEAL_BOX_HISTORY_PRIVATE = 1,
    /// One history for the ensemble.
    ANNEAL_BOX_HISTORY_SHARED = 2,
}

/// Which visits count as known under the history.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum anneal_box_membership_t {
    /// Minima some chain has adopted count.
    ANNEAL_BOX_MEMBERSHIP_ACCEPTED = 0,
    /// Every certified observation counts, adopted or not.
    ANNEAL_BOX_MEMBERSHIP_OBSERVED = 1,
}

/// How shared visit counts steer escape and acceptance.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum anneal_box_visit_policy_t {
    /// Peer visits mark a basin known and inflate escape.
    ANNEAL_BOX_VISIT_TABU = 0,
    /// Peer visits only identify the basin.
    ANNEAL_BOX_VISIT_RECOGNITION = 1,
}

/// How coverage is shared between replicas.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum anneal_box_coverage_t {
    /// Follow the history: shared when the history is shared, there is more
    /// than one replica and `shared_deposits` is positive.
    ANNEAL_BOX_COVERAGE_AUTO = -1,
    /// Each replica keeps its own coverage.
    ANNEAL_BOX_COVERAGE_PRIVATE = 0,
    /// Replicas share coverage, even without a minimum history.
    ANNEAL_BOX_COVERAGE_SHARED = 1,
}

/// Every setting of a box ensemble.
///
/// A superset of `anneal_box_config_t`: it adds the history, the identity
/// tolerance, the shared deposits, the visit policy, the Langevin noise and
/// the rest of the coverage field. Call `anneal_box_options_default`, then
/// change fields. The enum-valued fields are plain `int32_t` so that a value
/// this library does not know is refused, not undefined. `struct_size` lets
/// the structure grow: the library reads only the bytes the caller's
/// structure has, and takes the default for the fields after them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct anneal_box_options_t {
    /// `sizeof(anneal_box_options_t)` of the caller's header.
    pub struct_size: usize,
    /// Chains that divide the aggregate budget.
    pub replicas: usize,
    /// Combined objective and gradient work units across all chains.
    pub budget: usize,
    /// Seed of the deterministic driver.
    pub seed: u64,
    /// An `anneal_box_history_t`.
    pub history: i32,
    /// An `anneal_box_membership_t`.
    pub membership: i32,
    /// Scaled max-norm that identifies two quenched points as one minimum.
    pub identity_tol: f64,
    /// Foreign coverage visits paid into each receiving region per hop
    /// boundary. Zero disables incoming deposits.
    pub shared_deposits: usize,
    /// An `anneal_box_visit_policy_t`.
    pub visit_policy: i32,
    /// An `anneal_box_escape_t`.
    pub escape: i32,
    /// Langevin steps per escape segment (Langevin escape only).
    pub langevin_steps: usize,
    /// Langevin harmonic frequency (Langevin escape only).
    pub langevin_omega0: f64,
    /// Langevin time step (Langevin escape only).
    pub langevin_dt: f64,
    /// Damping rate of scalar white noise; zero selects the colored noise
    /// (Langevin escape only).
    pub langevin_friction: f64,
    /// An `anneal_box_coverage_t`.
    pub coverage_shared: i32,
    /// Reserved alignment bytes. Callers must leave this zero.
    pub _padding: i32,
    /// Coverage deposit radius in normalized RMS box distance.
    pub coverage_radius: f64,
    /// Initial deposit height in objective units; zero disables coverage.
    pub coverage_height: f64,
    /// Well-tempering factor, finite and greater than one.
    pub coverage_well_tempering: f64,
    /// Nonnegative multiplier for foreign deposits.
    pub coverage_peer_weight: f64,
    /// Direct cyclic neighbourhood radius of the coverage exchange; zero is
    /// all-to-all.
    pub coverage_neighbors: usize,
}

/// Outcome of one box ensemble, with the history and coverage counters.
///
/// `struct_size` is the number of bytes this library wrote; the caller sets
/// it to its `sizeof(anneal_box_stats_t)` before the call.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct anneal_box_stats_t {
    /// `sizeof(anneal_box_stats_t)` of the caller's header.
    pub struct_size: usize,
    /// Objective at the returned point.
    pub value: f64,
    /// Objective evaluations charged, polish included.
    pub n_evals: usize,
    /// Gradient evaluations charged, polish included.
    pub n_grads: usize,
    /// Accepted plus rejected hops across replicas.
    pub hops: usize,
    /// Certified observations reported to a history.
    pub history_observations: usize,
    /// Distinct minima in the shared history, or the largest private one.
    pub history_minima: usize,
    /// History attempts the history refused.
    pub history_refusals: usize,
    /// Summed seconds the history operations took across replicas.
    pub history_seconds: f64,
    /// Bias deposits made on behalf of other chains' visits.
    pub shared_deposits: usize,
    /// Finite feasible boundaries each chain observed.
    pub coverage_observations: u64,
    /// Local visits placed on the exchange.
    pub coverage_published: u64,
    /// Foreign visits applied to receiving biases.
    pub coverage_applied_foreign: u64,
    /// Foreign visits excluded by the per-region cap.
    pub coverage_capped_foreign: u64,
    /// Uncertified quenches returning into a known region.
    pub coverage_recrossings: u64,
    /// Arrivals in a region no chain had held.
    pub coverage_novel_arrivals: u64,
    /// Proposals pushed away from an evaluated region.
    pub coverage_repelled_proposals: u64,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: &str) {
    let text = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(text));
}

/// Message of the last failed call on this thread, or null.
#[unsafe(no_mangle)]
pub extern "C" fn anneal_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr())
    })
}

/// Package version as a NUL-terminated C string.
///
/// The pointer is valid for the entire process lifetime; the string lives in
/// the binary's read-only data segment and is never freed.
#[unsafe(no_mangle)]
pub extern "C" fn anneal_core_version() -> *const c_char {
    crate::version::ANNEAL_VERSION_NUL.as_ptr() as *const c_char
}

/// Write the default box configuration: four replicas, a budget of 400
/// work units, the default coverage field, and Gaussian escapes.
///
/// # Safety
///
/// `out` must be null, or aligned and writable for one `anneal_box_config_t`.
/// A null pointer returns `ANNEAL_INVALID_PARAMETER` and writes nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anneal_box_config_default(
    out: *mut anneal_box_config_t,
) -> anneal_status_t {
    if out.is_null() {
        set_last_error("anneal_box_config_default: null argument");
        return anneal_status_t::ANNEAL_INVALID_PARAMETER;
    }
    let coverage = BoxCoverageConfig::default();
    let langevin = GleEscapeConfig::default();
    unsafe {
        *out = anneal_box_config_t {
            replicas: 4,
            budget: 400,
            seed: 0,
            coverage_radius: coverage.radius,
            coverage_height: coverage.height,
            well_tempering: coverage.well_tempering,
            escape: anneal_box_escape_t::ANNEAL_BOX_ESCAPE_GAUSSIAN,
            langevin_steps: langevin.steps,
            langevin_omega0: langevin.omega0,
            langevin_dt: langevin.dt,
        };
    }
    anneal_status_t::ANNEAL_SUCCESS
}

/// A one-dimensional CPU float64 DLPack shell over borrowed memory.
struct StandingShell(Box<ShellInner>);

struct ShellInner {
    managed: DLManagedTensorVersioned,
    shape: [i64; 1],
    strides: [i64; 1],
}

unsafe impl Send for StandingShell {}

impl StandingShell {
    fn new() -> Self {
        let mut inner = Box::new(ShellInner {
            managed: DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
                flags: 0,
                dl_tensor: DLTensor {
                    data: std::ptr::null_mut(),
                    device: DLDevice {
                        device_type: DLDeviceType::kDLCPU,
                        device_id: 0,
                    },
                    ndim: 1,
                    dtype: DLDataType {
                        code: DLDataTypeCode::kDLFloat,
                        bits: 64,
                        lanes: 1,
                    },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
            },
            shape: [0],
            strides: [1],
        });
        inner.managed.dl_tensor.shape = inner.shape.as_mut_ptr();
        inner.managed.dl_tensor.strides = inner.strides.as_mut_ptr();
        StandingShell(inner)
    }

    fn point_at(&mut self, data: *mut f64, len: usize) -> *mut DLManagedTensorVersioned {
        self.0.shape[0] = len as i64;
        self.0.managed.dl_tensor.data = data.cast();
        &mut self.0.managed
    }
}

struct Scratch {
    x: StandingShell,
    out: StandingShell,
    xbuf: Vec<f64>,
}

/// The caller's callbacks as an eindir objective with a gradient.
///
/// Every call is serialized behind one mutex: the callbacks share whatever
/// state `user` points at, and the box driver is single-threaded anyway.
struct CallbackObjective {
    name: String,
    eval: anneal_eval_fn,
    grad: anneal_grad_fn,
    user: usize,
    bounds: Bounds<f64>,
    scratch: Mutex<Scratch>,
    failed: Mutex<bool>,
    stop: Arc<AtomicBool>,
}

unsafe impl Send for CallbackObjective {}
unsafe impl Sync for CallbackObjective {}

impl CallbackObjective {
    fn with_x<R>(
        &self,
        x: ArrayView1<f64>,
        f: impl FnOnce(&mut Scratch, *mut DLManagedTensorVersioned) -> R,
    ) -> R {
        let mut scratch = self.scratch.lock().expect("anneal ffi scratch");
        scratch.xbuf.clear();
        scratch.xbuf.extend(x.iter());
        let ptr = scratch.xbuf.as_mut_ptr();
        let len = scratch.xbuf.len();
        let xt = scratch.x.point_at(ptr, len);
        f(&mut scratch, xt)
    }

    fn fail(&self, what: &str) {
        *self.failed.lock().expect("anneal ffi failure flag") = true;
        set_last_error(what);
    }
}

impl Objective<f64> for CallbackObjective {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if self.stop.load(Ordering::Relaxed) {
            return f64::INFINITY;
        }
        let user = self.user as *mut c_void;
        let eval = self.eval;
        let mut value = f64::INFINITY;
        let status = self.with_x(x, |_, xt| unsafe { eval(user, xt, &mut value) });
        if status == anneal_status_t::ANNEAL_STOPPED {
            self.stop.store(true, Ordering::Relaxed);
            return if value.is_nan() { f64::INFINITY } else { value };
        }
        if status != anneal_status_t::ANNEAL_SUCCESS {
            self.fail(&format!("{}: objective callback failed", self.name));
            return f64::INFINITY;
        }
        if value.is_nan() { f64::INFINITY } else { value }
    }
}

impl Gradient<f64> for CallbackObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let n = x.len();
        if self.stop.load(Ordering::Relaxed) {
            return Array1::zeros(n);
        }
        let user = self.user as *mut c_void;
        let grad = self.grad;
        let mut gradient = Array1::<f64>::zeros(n);
        let status = self.with_x(x, |scratch, xt| {
            let gt = scratch.out.point_at(gradient.as_mut_ptr(), n);
            unsafe { grad(user, xt, gt) }
        });
        if status == anneal_status_t::ANNEAL_STOPPED {
            self.stop.store(true, Ordering::Relaxed);
            return gradient;
        }
        if status != anneal_status_t::ANNEAL_SUCCESS {
            self.fail(&format!("{}: gradient callback failed", self.name));
            return Array1::from_elem(n, f64::NAN);
        }
        gradient
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

unsafe fn cpu_f64_slice_mut<'a>(
    tensor: *mut DLManagedTensorVersioned,
    name: &str,
    what: &str,
) -> Result<&'a mut [f64], anneal_status_t> {
    if tensor.is_null() {
        set_last_error(&format!("{name}: {what} is null"));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    }
    let t = unsafe { &(*tensor).dl_tensor };
    let cpu = t.device.device_type == DLDeviceType::kDLCPU;
    let f64_dtype =
        t.dtype.code == DLDataTypeCode::kDLFloat && t.dtype.bits == 64 && t.dtype.lanes == 1;
    if !cpu || !f64_dtype || t.ndim != 1 || t.shape.is_null() || t.data.is_null() {
        set_last_error(&format!(
            "{name}: {what} must be a one-dimensional CPU float64 tensor"
        ));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    }
    let shape = unsafe { *t.shape };
    let fits = usize::try_from(shape).ok().filter(|&len| {
        len.checked_mul(std::mem::size_of::<f64>())
            .is_some_and(|bytes| bytes <= isize::MAX as usize)
    });
    let Some(len) = fits else {
        set_last_error(&format!("{name}: {what} has shape {shape}, not a length"));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    };
    if !t.strides.is_null() && unsafe { *t.strides } != 1 && len > 1 {
        set_last_error(&format!("{name}: {what} must be contiguous"));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    }
    let start = usize::try_from(t.byte_offset)
        .ok()
        .filter(|&offset| offset <= isize::MAX as usize)
        .and_then(|offset| {
            (t.data as usize)
                .checked_add(offset)
                .map(|address| (offset, address))
        });
    let Some((offset, address)) = start else {
        set_last_error(&format!("{name}: {what} byte_offset is out of range"));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    };
    if address % std::mem::align_of::<f64>() != 0 {
        set_last_error(&format!(
            "{name}: {what} data plus byte_offset must be aligned for float64"
        ));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    }
    let data = unsafe { t.data.cast::<u8>().add(offset) }.cast::<f64>();
    Ok(unsafe { std::slice::from_raw_parts_mut(data, len) })
}

/// Copy a caller's `anneal_box_config_t` field by field. `escape` is read
/// as the `int32_t` a C enum is, since a value outside
/// `anneal_box_escape_t` must not become a Rust enum.
const _: () = assert!(std::mem::size_of::<anneal_box_escape_t>() == std::mem::size_of::<i32>());

unsafe fn read_config(cfg: *const anneal_box_config_t) -> Result<anneal_box_config_t, String> {
    let escape = unsafe { std::ptr::addr_of!((*cfg).escape).cast::<i32>().read() };
    let escape = match escape {
        0 => anneal_box_escape_t::ANNEAL_BOX_ESCAPE_GAUSSIAN,
        1 => anneal_box_escape_t::ANNEAL_BOX_ESCAPE_LANGEVIN,
        other => return Err(format!("escape {other} is not an anneal_box_escape_t")),
    };
    unsafe {
        Ok(anneal_box_config_t {
            replicas: std::ptr::addr_of!((*cfg).replicas).read(),
            budget: std::ptr::addr_of!((*cfg).budget).read(),
            seed: std::ptr::addr_of!((*cfg).seed).read(),
            coverage_radius: std::ptr::addr_of!((*cfg).coverage_radius).read(),
            coverage_height: std::ptr::addr_of!((*cfg).coverage_height).read(),
            well_tempering: std::ptr::addr_of!((*cfg).well_tempering).read(),
            escape,
            langevin_steps: std::ptr::addr_of!((*cfg).langevin_steps).read(),
            langevin_omega0: std::ptr::addr_of!((*cfg).langevin_omega0).read(),
            langevin_dt: std::ptr::addr_of!((*cfg).langevin_dt).read(),
        })
    }
}

fn ensemble_config(
    cfg: &anneal_box_config_t,
) -> Result<(BoxEnsembleConfig, BoxCoverageConfig), String> {
    if cfg.replicas == 0 || cfg.budget == 0 {
        return Err("replicas and budget must be positive".into());
    }
    if !(cfg.coverage_radius.is_finite() && cfg.coverage_radius > 0.0) {
        return Err("coverage_radius must be finite and positive".into());
    }
    if !(cfg.coverage_height.is_finite() && cfg.coverage_height >= 0.0) {
        return Err("coverage_height must be finite and nonnegative".into());
    }
    if !(cfg.well_tempering.is_finite() && cfg.well_tempering > 1.0) {
        return Err("well_tempering must be finite and greater than one".into());
    }
    let escape = match cfg.escape {
        anneal_box_escape_t::ANNEAL_BOX_ESCAPE_GAUSSIAN => BoxEscape::Gaussian,
        anneal_box_escape_t::ANNEAL_BOX_ESCAPE_LANGEVIN => {
            if cfg.langevin_steps == 0
                || !(cfg.langevin_omega0.is_finite() && cfg.langevin_omega0 > 0.0)
                || !(cfg.langevin_dt.is_finite() && cfg.langevin_dt > 0.0)
            {
                return Err("Langevin escape needs positive steps, omega0 and dt".into());
            }
            BoxEscape::Langevin(GleEscapeConfig {
                steps: cfg.langevin_steps,
                omega0: cfg.langevin_omega0,
                dt: cfg.langevin_dt,
                noise: GleNoise::Colored,
            })
        }
    };
    let ensemble = BoxEnsembleConfig {
        replicas: cfg.replicas,
        budget: cfg.budget,
        history: HistoryMode::Shared,
        escape,
        ..BoxEnsembleConfig::default()
    };
    let coverage = BoxCoverageConfig {
        radius: cfg.coverage_radius,
        height: cfg.coverage_height,
        well_tempering: cfg.well_tempering,
        shared: cfg.replicas > 1 && ensemble.shared_deposits > 0,
        ..BoxCoverageConfig::default()
    };
    Ok((ensemble, coverage))
}

/// What a box search runs with, once the C structs are checked.
struct Resolved {
    ensemble: BoxEnsembleConfig,
    coverage: BoxCoverageConfig,
    seed: u64,
}

/// Every axis finite and ordered, with a reflection period `2 (h - l)`
/// and a sum of widths that stay finite: the rule the Python drivers
/// apply, so an overflowing width is refused, not folded into NaN.
fn check_box(low: &[f64], high: &[f64]) -> Result<(), String> {
    let mut total_width = 0.0;
    for (i, (&lo, &hi)) in low.iter().zip(high).enumerate() {
        if !(lo.is_finite() && hi.is_finite() && (2.0 * (hi - lo)).is_finite()) {
            return Err(format!(
                "bounds must be finite, with a finite width, at dimension {i}"
            ));
        }
        if lo > hi {
            return Err(format!("bounds must have low <= high at dimension {i}"));
        }
        total_width += hi - lo;
    }
    if total_width.is_finite() {
        Ok(())
    } else {
        Err("the box is too wide: the sum of its widths is not finite".into())
    }
}

/// The box search behind both entry points.
///
/// Returns the status and, for `ANNEAL_SUCCESS` and for `ANNEAL_STOPPED`
/// after a finite point, the result. `x` holds the best point on exit then.
unsafe fn minimize_core(
    eval: anneal_eval_fn,
    grad: anneal_grad_fn,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    low: *const f64,
    high: *const f64,
    name: &str,
    resolve: impl FnOnce() -> Result<Resolved, String>,
) -> (anneal_status_t, Option<BoxEnsembleResult>) {
    let start = match unsafe { cpu_f64_slice_mut(x, name, "x") } {
        Ok(s) => s.to_vec(),
        Err(st) => return (st, None),
    };
    let dim = start.len();
    if dim == 0 {
        set_last_error(&format!("{name}: x is empty"));
        return (anneal_status_t::ANNEAL_INVALID_PARAMETER, None);
    }
    let low = unsafe { std::slice::from_raw_parts(low, dim) };
    let high = unsafe { std::slice::from_raw_parts(high, dim) };
    if let Err(message) = check_box(low, high) {
        set_last_error(&format!("{name}: {message}"));
        return (anneal_status_t::ANNEAL_INVALID_PARAMETER, None);
    }
    if let Some(i) = start.iter().position(|value| !value.is_finite()) {
        set_last_error(&format!("{name}: x[{i}] is not finite"));
        return (anneal_status_t::ANNEAL_INVALID_PARAMETER, None);
    }
    let Resolved {
        ensemble,
        coverage,
        seed,
    } = match resolve() {
        Ok(resolved) => resolved,
        Err(message) => {
            set_last_error(&format!("{name}: {message}"));
            return (anneal_status_t::ANNEAL_INVALID_PARAMETER, None);
        }
    };
    let replica_states = ensemble.replicas.checked_mul(dim);
    if !replica_states.is_some_and(|n| Vec::<f64>::new().try_reserve_exact(n).is_ok()) {
        set_last_error(&format!(
            "{name}: {} replicas of dimension {dim} are too many to allocate",
            ensemble.replicas
        ));
        return (anneal_status_t::ANNEAL_INVALID_PARAMETER, None);
    }
    let stop = Arc::new(AtomicBool::new(false));
    let _caller_stop = crate::caller_stop::Bind::new(Arc::clone(&stop));
    let objective = CallbackObjective {
        name: name.to_owned(),
        eval,
        grad,
        user: user as usize,
        bounds: Bounds::new(
            Array1::from_vec(low.to_vec()),
            Array1::from_vec(high.to_vec()),
            0.0,
        ),
        scratch: Mutex::new(Scratch {
            x: StandingShell::new(),
            out: StandingShell::new(),
            xbuf: Vec::with_capacity(dim),
        }),
        failed: Mutex::new(false),
        stop: Arc::clone(&stop),
    };
    let x0 = Array1::from_vec(start);
    let result = box_ensemble_optimize_with_coverage(
        &objective,
        &objective,
        seed,
        Some(x0.view()),
        &ensemble,
        &coverage,
    );
    if *objective.failed.lock().expect("anneal ffi failure flag") {
        return (anneal_status_t::ANNEAL_INTERNAL_ERROR, None);
    }
    let stopped = stop.load(Ordering::Relaxed);
    if stopped && !(result.best_pos.len() == dim && result.best_val.is_finite()) {
        set_last_error(&format!("{name}: stopped before a finite point"));
        return (anneal_status_t::ANNEAL_STOPPED, None);
    }
    if !stopped && (!result.best_val.is_finite() || result.best_pos.len() != dim) {
        set_last_error(&format!("{name}: no finite point was evaluated"));
        return (anneal_status_t::ANNEAL_INTERNAL_ERROR, None);
    }
    let dest = match unsafe { cpu_f64_slice_mut(x, name, "x") } {
        Ok(s) => s,
        Err(st) => return (st, None),
    };
    dest.copy_from_slice(result.best_pos.as_slice().expect("contiguous"));
    let status = if stopped {
        anneal_status_t::ANNEAL_STOPPED
    } else {
        anneal_status_t::ANNEAL_SUCCESS
    };
    (status, Some(result))
}

fn panic_status(name: &str, payload: Box<dyn std::any::Any + Send>) -> anneal_status_t {
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string());
    set_last_error(&format!("{name}: {message}"));
    anneal_status_t::ANNEAL_INTERNAL_ERROR
}

fn default_options() -> anneal_box_options_t {
    let ensemble = BoxEnsembleConfig::default();
    let coverage = BoxCoverageConfig::default();
    let langevin = GleEscapeConfig::default();
    anneal_box_options_t {
        struct_size: std::mem::size_of::<anneal_box_options_t>(),
        replicas: ensemble.replicas,
        budget: 400,
        seed: 0,
        history: anneal_box_history_t::ANNEAL_BOX_HISTORY_SHARED as i32,
        membership: anneal_box_membership_t::ANNEAL_BOX_MEMBERSHIP_ACCEPTED as i32,
        identity_tol: ensemble.identity_tol,
        shared_deposits: ensemble.shared_deposits,
        visit_policy: anneal_box_visit_policy_t::ANNEAL_BOX_VISIT_TABU as i32,
        escape: anneal_box_escape_t::ANNEAL_BOX_ESCAPE_GAUSSIAN as i32,
        langevin_steps: langevin.steps,
        langevin_omega0: langevin.omega0,
        langevin_dt: langevin.dt,
        langevin_friction: 0.0,
        coverage_shared: anneal_box_coverage_t::ANNEAL_BOX_COVERAGE_AUTO as i32,
        _padding: 0,
        coverage_radius: coverage.radius,
        coverage_height: coverage.height,
        coverage_well_tempering: coverage.well_tempering,
        coverage_peer_weight: coverage.peer_weight,
        coverage_neighbors: coverage.neighbors,
    }
}

fn options_size_is_field_boundary(size: usize) -> bool {
    size > std::mem::size_of::<anneal_box_options_t>()
        || [
            std::mem::offset_of!(anneal_box_options_t, history),
            std::mem::offset_of!(anneal_box_options_t, membership),
            std::mem::offset_of!(anneal_box_options_t, identity_tol),
            std::mem::offset_of!(anneal_box_options_t, shared_deposits),
            std::mem::offset_of!(anneal_box_options_t, visit_policy),
            std::mem::offset_of!(anneal_box_options_t, escape),
            std::mem::offset_of!(anneal_box_options_t, langevin_steps),
            std::mem::offset_of!(anneal_box_options_t, langevin_omega0),
            std::mem::offset_of!(anneal_box_options_t, langevin_dt),
            std::mem::offset_of!(anneal_box_options_t, langevin_friction),
            std::mem::offset_of!(anneal_box_options_t, coverage_shared),
            std::mem::offset_of!(anneal_box_options_t, _padding),
            std::mem::offset_of!(anneal_box_options_t, coverage_radius),
            std::mem::offset_of!(anneal_box_options_t, coverage_height),
            std::mem::offset_of!(anneal_box_options_t, coverage_well_tempering),
            std::mem::offset_of!(anneal_box_options_t, coverage_peer_weight),
            std::mem::offset_of!(anneal_box_options_t, coverage_neighbors),
            std::mem::size_of::<anneal_box_options_t>(),
        ]
        .contains(&size)
}

fn stats_size_is_field_boundary(size: usize) -> bool {
    size > std::mem::size_of::<anneal_box_stats_t>()
        || [
            std::mem::offset_of!(anneal_box_stats_t, n_evals),
            std::mem::offset_of!(anneal_box_stats_t, n_grads),
            std::mem::offset_of!(anneal_box_stats_t, hops),
            std::mem::offset_of!(anneal_box_stats_t, history_observations),
            std::mem::offset_of!(anneal_box_stats_t, history_minima),
            std::mem::offset_of!(anneal_box_stats_t, history_refusals),
            std::mem::offset_of!(anneal_box_stats_t, history_seconds),
            std::mem::offset_of!(anneal_box_stats_t, shared_deposits),
            std::mem::offset_of!(anneal_box_stats_t, coverage_observations),
            std::mem::offset_of!(anneal_box_stats_t, coverage_published),
            std::mem::offset_of!(anneal_box_stats_t, coverage_applied_foreign),
            std::mem::offset_of!(anneal_box_stats_t, coverage_capped_foreign),
            std::mem::offset_of!(anneal_box_stats_t, coverage_recrossings),
            std::mem::offset_of!(anneal_box_stats_t, coverage_novel_arrivals),
            std::mem::offset_of!(anneal_box_stats_t, coverage_repelled_proposals),
            std::mem::size_of::<anneal_box_stats_t>(),
        ]
        .contains(&size)
}

fn invalid_options_size(size: usize) -> String {
    let least = std::mem::offset_of!(anneal_box_options_t, history);
    if size < least {
        format!("options.struct_size is {size}, below the {least} bytes of the oldest layout")
    } else {
        format!("options.struct_size {size} ends within a field")
    }
}

/// The caller's options over the defaults: the first `struct_size` bytes
/// come from the caller, the fields after them keep the default.
unsafe fn read_options(given: *const anneal_box_options_t) -> Result<anneal_box_options_t, String> {
    let size = unsafe { (*given).struct_size };
    if !options_size_is_field_boundary(size) {
        return Err(invalid_options_size(size));
    }
    let mut merged = default_options();
    let take = size.min(std::mem::size_of::<anneal_box_options_t>());
    unsafe {
        std::ptr::copy_nonoverlapping(
            given.cast::<u8>(),
            (&mut merged as *mut anneal_box_options_t).cast::<u8>(),
            take,
        );
    }
    merged.struct_size = size;
    Ok(merged)
}

fn positive(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be finite and positive"))
    }
}

fn resolve_options(options: &anneal_box_options_t) -> Result<Resolved, String> {
    if options.replicas == 0 || options.budget == 0 {
        return Err("replicas and budget must be positive".into());
    }
    let history = match options.history {
        0 => HistoryMode::None,
        1 => HistoryMode::Private,
        2 => HistoryMode::Shared,
        other => return Err(format!("history {other} is not an anneal_box_history_t")),
    };
    let membership = match options.membership {
        0 => HistoryMembership::Accepted,
        1 => HistoryMembership::Observed,
        other => {
            return Err(format!(
                "membership {other} is not an anneal_box_membership_t"
            ));
        }
    };
    positive("identity_tol", options.identity_tol)?;
    let visit_policy = match options.visit_policy {
        0 => SharedVisitPolicy::Tabu,
        1 => SharedVisitPolicy::Recognition,
        other => {
            return Err(format!(
                "visit_policy {other} is not an anneal_box_visit_policy_t"
            ));
        }
    };
    let escape = match options.escape {
        0 => BoxEscape::Gaussian,
        1 => {
            if options.langevin_steps == 0 {
                return Err("langevin_steps must be positive".into());
            }
            positive("langevin_omega0", options.langevin_omega0)?;
            positive("langevin_dt", options.langevin_dt)?;
            let noise = if options.langevin_friction == 0.0 {
                GleNoise::Colored
            } else {
                positive("langevin_friction", options.langevin_friction)?;
                GleNoise::White {
                    friction: options.langevin_friction,
                }
            };
            BoxEscape::Langevin(GleEscapeConfig {
                steps: options.langevin_steps,
                omega0: options.langevin_omega0,
                dt: options.langevin_dt,
                noise,
            })
        }
        other => return Err(format!("escape {other} is not an anneal_box_escape_t")),
    };
    positive("coverage_radius", options.coverage_radius)?;
    if !(options.coverage_height.is_finite() && options.coverage_height >= 0.0) {
        return Err("coverage_height must be finite and nonnegative".into());
    }
    if !(options.coverage_well_tempering.is_finite() && options.coverage_well_tempering > 1.0) {
        return Err("coverage_well_tempering must be finite and greater than one".into());
    }
    if !(options.coverage_peer_weight.is_finite() && options.coverage_peer_weight >= 0.0) {
        return Err("coverage_peer_weight must be finite and nonnegative".into());
    }
    let ensemble = BoxEnsembleConfig {
        replicas: options.replicas,
        budget: options.budget,
        history,
        membership,
        identity_tol: options.identity_tol,
        shared_deposits: options.shared_deposits,
        shared_visit_policy: visit_policy,
        escape,
    };
    let shared = match options.coverage_shared {
        -1 => {
            matches!(history, HistoryMode::Shared)
                && options.replicas > 1
                && options.shared_deposits > 0
        }
        0 => false,
        1 => true,
        other => {
            return Err(format!(
                "coverage_shared {other} is not an anneal_box_coverage_t"
            ));
        }
    };
    let coverage = BoxCoverageConfig {
        radius: options.coverage_radius,
        height: options.coverage_height,
        well_tempering: options.coverage_well_tempering,
        peer_weight: options.coverage_peer_weight,
        shared,
        neighbors: options.coverage_neighbors,
    };
    Ok(Resolved {
        ensemble,
        coverage,
        seed: options.seed,
    })
}

fn stats_of(result: &BoxEnsembleResult) -> anneal_box_stats_t {
    anneal_box_stats_t {
        struct_size: std::mem::size_of::<anneal_box_stats_t>(),
        value: result.best_val,
        n_evals: result.n_evals,
        n_grads: result.n_grads,
        hops: result.hops,
        history_observations: result.history_observations,
        history_minima: result.history_minima,
        history_refusals: result.history_cost.1,
        history_seconds: result.history_cost.2,
        shared_deposits: result.shared_deposits,
        coverage_observations: result.coverage.local_observations as u64,
        coverage_published: result.coverage.published_visits,
        coverage_applied_foreign: result.coverage.applied_foreign_visits as u64,
        coverage_capped_foreign: result.coverage.capped_foreign_visits,
        coverage_recrossings: result.coverage.recrossings as u64,
        coverage_novel_arrivals: result.coverage.novel_arrivals as u64,
        coverage_repelled_proposals: result.coverage.repelled_proposals as u64,
    }
}

/// Write the default box options: `anneal_box_config_default`'s values for
/// the fields both have, a shared history, the default identity tolerance
/// and deposits, the tabu visit policy, colored Langevin noise, and coverage
/// that follows the history.
///
/// Set `out->struct_size` to `sizeof(anneal_box_options_t)` first. The
/// library writes the defaults into the first `struct_size` bytes it knows,
/// sets `struct_size` to the number of bytes written, and leaves the bytes of
/// a newer, larger structure alone. A size below the oldest layout or ending
/// within a field returns `ANNEAL_INVALID_PARAMETER` and writes nothing.
///
/// # Safety
///
/// `out` must be null, or aligned, with `struct_size` readable, and
/// writable for its own `struct_size` bytes. A null pointer returns
/// `ANNEAL_INVALID_PARAMETER`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anneal_box_options_default(
    out: *mut anneal_box_options_t,
) -> anneal_status_t {
    if out.is_null() {
        set_last_error("anneal_box_options_default: null argument");
        return anneal_status_t::ANNEAL_INVALID_PARAMETER;
    }
    let size = unsafe { std::ptr::addr_of!((*out).struct_size).read() };
    if !options_size_is_field_boundary(size) {
        set_last_error(&format!(
            "anneal_box_options_default: {}; set it to sizeof(anneal_box_options_t)",
            invalid_options_size(size)
        ));
        return anneal_status_t::ANNEAL_INVALID_PARAMETER;
    }
    let take = size.min(std::mem::size_of::<anneal_box_options_t>());
    let mut defaults = default_options();
    defaults.struct_size = take;
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&defaults as *const anneal_box_options_t).cast::<u8>(),
            out.cast::<u8>(),
            take,
        );
    }
    anneal_status_t::ANNEAL_SUCCESS
}

/// Minimize `f` over the box `[low, high]` with every setting of the box
/// ensemble.
///
/// Like `anneal_box_minimize`, with `anneal_box_options_t` for the settings
/// and `anneal_box_stats_t` for the report. `stats` may be null. When it is
/// not, its `struct_size` must be set to `sizeof(anneal_box_stats_t)`, and
/// it is written for `ANNEAL_SUCCESS` and for `ANNEAL_STOPPED` after a finite
/// point. A value outside the range of a setting returns
/// `ANNEAL_INVALID_PARAMETER`, with the reason in `anneal_last_error`,
/// before any callback runs.
///
/// # Safety
///
/// As `anneal_box_minimize`. `options` must be null or readable for its own
/// `struct_size` bytes, and `stats` null or writable for its own
/// `struct_size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anneal_box_minimize_ex(
    eval: anneal_eval_fn,
    grad: anneal_grad_fn,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    low: *const f64,
    high: *const f64,
    options: *const anneal_box_options_t,
    stats: *mut anneal_box_stats_t,
) -> anneal_status_t {
    const NAME: &str = "anneal_box_minimize_ex";
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if low.is_null() || high.is_null() || options.is_null() {
            set_last_error("anneal_box_minimize_ex: null argument");
            return anneal_status_t::ANNEAL_INVALID_PARAMETER;
        }
        let stats_size = if stats.is_null() {
            0
        } else {
            let size = unsafe { (*stats).struct_size };
            if !stats_size_is_field_boundary(size) {
                set_last_error(&format!(
                    "anneal_box_minimize_ex: stats.struct_size {size} is not a field boundary"
                ));
                return anneal_status_t::ANNEAL_INVALID_PARAMETER;
            }
            size
        };
        let read = match unsafe { read_options(options) } {
            Ok(read) => read,
            Err(message) => {
                set_last_error(&format!("{NAME}: {message}"));
                return anneal_status_t::ANNEAL_INVALID_PARAMETER;
            }
        };
        let (status, result) = unsafe {
            minimize_core(eval, grad, user, x, low, high, NAME, || {
                resolve_options(&read)
            })
        };
        if let (Some(result), false) = (result, stats.is_null()) {
            let take = stats_size.min(std::mem::size_of::<anneal_box_stats_t>());
            let mut full = stats_of(&result);
            full.struct_size = take;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&full as *const anneal_box_stats_t).cast::<u8>(),
                    stats.cast::<u8>(),
                    take,
                );
            }
        }
        status
    })) {
        Ok(status) => status,
        Err(payload) => panic_status(NAME, payload),
    }
}

/// Minimize `f` over the box `[low, high]` with a communicating-chain box
/// ensemble.
///
/// `eval` and `grad` must be non-null. `x` holds the start of the first
/// replica on entry and the best evaluated point on exit; the other replicas start at uniform draws in the box. A
/// callback that fails ends the search with `ANNEAL_INTERNAL_ERROR`, and
/// `x` is left at its start. A callback that returns `ANNEAL_STOPPED`
/// ends the search at once: the best point found is written to `x` and
/// the call returns `ANNEAL_STOPPED`. `low` and `high` have `dim` entries
/// each and must be finite with `low <= high`, a finite `2 (high - low)`
/// and a finite sum of widths, and `x` must be finite. An `escape` outside
/// `anneal_box_escape_t` is refused. The settings that
/// `anneal_box_config_t` lacks are in `anneal_box_minimize_ex`.
///
/// # Safety
///
/// `eval` and `grad` must be non-null and safe to call for the whole search.
/// `x` must be null or a writable one-dimensional CPU float64 DLPack tensor.
/// `low`, `high`, `cfg`, and `out` must each be null or valid for the reads
/// and writes this function performs. A null argument returns
/// `ANNEAL_INVALID_PARAMETER`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anneal_box_minimize(
    eval: anneal_eval_fn,
    grad: anneal_grad_fn,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    low: *const f64,
    high: *const f64,
    cfg: *const anneal_box_config_t,
    out: *mut anneal_box_report_t,
) -> anneal_status_t {
    const NAME: &str = "anneal_box_minimize";
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if low.is_null() || high.is_null() || cfg.is_null() || out.is_null() {
            set_last_error("anneal_box_minimize: null argument");
            return anneal_status_t::ANNEAL_INVALID_PARAMETER;
        }
        let cfg = match unsafe { read_config(cfg) } {
            Ok(cfg) => cfg,
            Err(message) => {
                set_last_error(&format!("{NAME}: {message}"));
                return anneal_status_t::ANNEAL_INVALID_PARAMETER;
            }
        };
        let cfg = &cfg;
        let (status, result) = unsafe {
            minimize_core(eval, grad, user, x, low, high, NAME, || {
                ensemble_config(cfg).map(|(ensemble, coverage)| Resolved {
                    ensemble,
                    coverage,
                    seed: cfg.seed,
                })
            })
        };
        if let Some(result) = result {
            unsafe {
                *out = anneal_box_report_t {
                    value: result.best_val,
                    n_evals: result.n_evals,
                    n_grads: result.n_grads,
                    hops: result.hops,
                };
            }
        }
        status
    })) {
        Ok(status) => status,
        Err(payload) => panic_status(NAME, payload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter {
        evals: usize,
        grads: usize,
    }

    fn rastrigin_value(xs: &[f64]) -> f64 {
        xs.iter()
            .map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos() + 10.0)
            .sum::<f64>()
    }

    // Rastrigin in two dimensions on [-5.12, 5.12]^2: the global minimum is
    // zero at the origin among roughly one hundred local minima.
    unsafe extern "C" fn rastrigin(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value_out: *mut f64,
    ) -> anneal_status_t {
        let counter = unsafe { &mut *(user as *mut Counter) };
        counter.evals += 1;
        let xs = unsafe { cpu_f64_slice_mut(x as *mut _, "test", "x") }.unwrap();
        unsafe { *value_out = rastrigin_value(xs) };
        anneal_status_t::ANNEAL_SUCCESS
    }

    unsafe extern "C" fn rastrigin_grad(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        grad_out: *mut DLManagedTensorVersioned,
    ) -> anneal_status_t {
        let counter = unsafe { &mut *(user as *mut Counter) };
        counter.grads += 1;
        let xs = unsafe { cpu_f64_slice_mut(x as *mut _, "test", "x") }.unwrap();
        let gs = unsafe { cpu_f64_slice_mut(grad_out, "test", "grad") }.unwrap();
        for (g, v) in gs.iter_mut().zip(xs.iter()) {
            *g = 2.0 * v + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * v).sin();
        }
        anneal_status_t::ANNEAL_SUCCESS
    }

    fn default_config() -> anneal_box_config_t {
        let mut cfg = anneal_box_config_t {
            replicas: 0,
            budget: 0,
            seed: 0,
            coverage_radius: 0.0,
            coverage_height: 0.0,
            well_tempering: 0.0,
            escape: anneal_box_escape_t::ANNEAL_BOX_ESCAPE_GAUSSIAN,
            langevin_steps: 0,
            langevin_omega0: 0.0,
            langevin_dt: 0.0,
        };
        assert_eq!(
            unsafe { anneal_box_config_default(&mut cfg) },
            anneal_status_t::ANNEAL_SUCCESS
        );
        cfg
    }

    #[test]
    fn box_minimize_leaves_a_local_basin_of_rastrigin() {
        let mut counter = Counter { evals: 0, grads: 0 };
        let mut x = vec![3.0, -2.0];
        let mut shell = StandingShell::new();
        let low = [-5.12, -5.12];
        let high = [5.12, 5.12];
        let mut cfg = default_config();
        cfg.seed = 7;
        cfg.budget = 600;
        let mut report = anneal_box_report_t::default();
        let status = unsafe {
            anneal_box_minimize(
                rastrigin,
                rastrigin_grad,
                (&mut counter as *mut Counter).cast(),
                shell.point_at(x.as_mut_ptr(), 2),
                low.as_ptr(),
                high.as_ptr(),
                &cfg,
                &mut report,
            )
        };
        assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
        // The start sits in the basin at (3, -2) of value 13; the ensemble
        // must reach a strictly lower basin within the budget.
        assert!(report.value < 13.0 - 1e-6, "value {}", report.value);
        assert!(report.n_evals > 0 && report.n_grads > 0);
        assert!(counter.evals >= report.n_evals);
        let value = rastrigin_value(&x);
        assert!((value - report.value).abs() <= 1e-9 * (1.0 + value.abs()));
    }

    #[test]
    fn box_minimize_is_deterministic_for_a_seed() {
        let mut values = Vec::new();
        for _ in 0..2 {
            let mut counter = Counter { evals: 0, grads: 0 };
            let mut x = vec![3.0, -2.0];
            let mut shell = StandingShell::new();
            let low = [-5.12, -5.12];
            let high = [5.12, 5.12];
            let mut cfg = default_config();
            cfg.seed = 11;
            cfg.budget = 300;
            let mut report = anneal_box_report_t::default();
            let status = unsafe {
                anneal_box_minimize(
                    rastrigin,
                    rastrigin_grad,
                    (&mut counter as *mut Counter).cast(),
                    shell.point_at(x.as_mut_ptr(), 2),
                    low.as_ptr(),
                    high.as_ptr(),
                    &cfg,
                    &mut report,
                )
            };
            assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
            values.push((x.clone(), report.value, counter.evals, counter.grads));
        }
        assert_eq!(values[0], values[1]);
    }

    #[test]
    fn box_minimize_rejects_bad_arguments() {
        let cfg = default_config();
        let mut counter = Counter { evals: 0, grads: 0 };
        let mut x = vec![0.0, 0.0];
        let mut shell = StandingShell::new();
        let low = [0.0, 0.0];
        let high = [-1.0, 1.0];
        let mut report = anneal_box_report_t::default();
        let status = unsafe {
            anneal_box_minimize(
                rastrigin,
                rastrigin_grad,
                (&mut counter as *mut Counter).cast(),
                shell.point_at(x.as_mut_ptr(), 2),
                low.as_ptr(),
                high.as_ptr(),
                &cfg,
                &mut report,
            )
        };
        assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER);
        assert!(!anneal_last_error().is_null());
        let mut zero_budget = cfg;
        zero_budget.budget = 0;
        let status = unsafe {
            anneal_box_minimize(
                rastrigin,
                rastrigin_grad,
                (&mut counter as *mut Counter).cast(),
                shell.point_at(x.as_mut_ptr(), 2),
                low.as_ptr(),
                low.as_ptr(),
                &zero_budget,
                &mut report,
            )
        };
        assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER);
        assert_eq!(counter.evals, 0);
    }

    struct StopAfter {
        evals: usize,
        grads: usize,
        after: usize,
    }

    unsafe extern "C" fn stop_value(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value_out: *mut f64,
    ) -> anneal_status_t {
        let counter = unsafe { &mut *(user as *mut StopAfter) };
        counter.evals += 1;
        let xs = unsafe { cpu_f64_slice_mut(x as *mut _, "test", "x") }.unwrap();
        unsafe { *value_out = rastrigin_value(xs) };
        if counter.evals >= counter.after {
            anneal_status_t::ANNEAL_STOPPED
        } else {
            anneal_status_t::ANNEAL_SUCCESS
        }
    }

    unsafe extern "C" fn stop_grad(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        grad_out: *mut DLManagedTensorVersioned,
    ) -> anneal_status_t {
        let counter = unsafe { &mut *(user as *mut StopAfter) };
        counter.grads += 1;
        let xs = unsafe { cpu_f64_slice_mut(x as *mut _, "test", "x") }.unwrap();
        let gs = unsafe { cpu_f64_slice_mut(grad_out, "test", "grad") }.unwrap();
        for (g, v) in gs.iter_mut().zip(xs.iter()) {
            *g = 2.0 * v + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * v).sin();
        }
        if counter.evals >= counter.after {
            anneal_status_t::ANNEAL_STOPPED
        } else {
            anneal_status_t::ANNEAL_SUCCESS
        }
    }

    #[test]
    fn box_minimize_stops_when_the_callback_says_so() {
        use std::time::{Duration, Instant};

        const AFTER: usize = 24;
        let mut counter = StopAfter {
            evals: 0,
            grads: 0,
            after: AFTER,
        };
        let mut x = vec![3.0, -2.0];
        let mut shell = StandingShell::new();
        let low = [-5.12, -5.12];
        let high = [5.12, 5.12];
        let mut cfg = default_config();
        cfg.seed = 7;
        cfg.budget = 20_000;
        let mut report = anneal_box_report_t::default();
        let started = Instant::now();
        let status = unsafe {
            anneal_box_minimize(
                stop_value,
                stop_grad,
                (&mut counter as *mut StopAfter).cast(),
                shell.point_at(x.as_mut_ptr(), 2),
                low.as_ptr(),
                high.as_ptr(),
                &cfg,
                &mut report,
            )
        };
        assert_eq!(status, anneal_status_t::ANNEAL_STOPPED);
        assert!(counter.evals <= AFTER + 4, "evals {}", counter.evals);
        assert!(counter.evals >= AFTER, "evals {}", counter.evals);
        assert!(counter.grads <= AFTER + 4, "grads {}", counter.grads);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(report.value.is_finite());
        assert!((rastrigin_value(&x) - report.value).abs() <= 1e-8);
    }

    struct Wells {
        evals: usize,
    }

    unsafe extern "C" fn wells_value(
        user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value_out: *mut f64,
    ) -> anneal_status_t {
        let wells = unsafe { &mut *(user as *mut Wells) };
        wells.evals += 1;
        let xs = unsafe { cpu_f64_slice_mut(x as *mut _, "test", "x") }.unwrap();
        unsafe {
            *value_out = xs
                .iter()
                .map(|v| (3.0 * v).cos() + 0.05 * v * v)
                .sum::<f64>();
        }
        anneal_status_t::ANNEAL_SUCCESS
    }

    unsafe extern "C" fn wells_grad(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        grad_out: *mut DLManagedTensorVersioned,
    ) -> anneal_status_t {
        let xs = unsafe { cpu_f64_slice_mut(x as *mut _, "test", "x") }.unwrap();
        let gs = unsafe { cpu_f64_slice_mut(grad_out, "test", "grad") }.unwrap();
        for (g, v) in gs.iter_mut().zip(xs.iter()) {
            *g = -3.0 * (3.0 * v).sin() + 0.1 * v;
        }
        anneal_status_t::ANNEAL_SUCCESS
    }

    fn wells_options() -> anneal_box_options_t {
        let mut options = default_options();
        options.seed = 4;
        options.budget = 3000;
        options
    }

    fn run_wells(
        options: &anneal_box_options_t,
    ) -> (anneal_status_t, Vec<f64>, anneal_box_stats_t, usize) {
        let mut wells = Wells { evals: 0 };
        let mut x = vec![3.0, -2.0];
        let mut shell = StandingShell::new();
        let low = [-5.0, -5.0];
        let high = [5.0, 5.0];
        let mut stats = anneal_box_stats_t {
            struct_size: std::mem::size_of::<anneal_box_stats_t>(),
            ..Default::default()
        };
        let status = unsafe {
            anneal_box_minimize_ex(
                wells_value,
                wells_grad,
                (&mut wells as *mut Wells).cast(),
                shell.point_at(x.as_mut_ptr(), 2),
                low.as_ptr(),
                high.as_ptr(),
                options,
                &mut stats,
            )
        };
        (status, x, stats, wells.evals)
    }

    fn last_error() -> String {
        let pointer = anneal_last_error();
        assert!(!pointer.is_null());
        unsafe { std::ffi::CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn the_default_options_are_the_rust_defaults() {
        let mut options = wells_options();
        options.replicas = 0;
        assert_eq!(
            unsafe { anneal_box_options_default(&mut options) },
            anneal_status_t::ANNEAL_SUCCESS
        );
        let ensemble = BoxEnsembleConfig::default();
        let coverage = BoxCoverageConfig::default();
        let langevin = GleEscapeConfig::default();
        assert_eq!(
            options.struct_size,
            std::mem::size_of::<anneal_box_options_t>()
        );
        assert_eq!(options.replicas, ensemble.replicas);
        assert_eq!(options.identity_tol, ensemble.identity_tol);
        assert_eq!(options.shared_deposits, ensemble.shared_deposits);
        assert_eq!(options.langevin_steps, langevin.steps);
        assert_eq!(options.langevin_omega0, langevin.omega0);
        assert_eq!(options.langevin_dt, langevin.dt);
        assert_eq!(options.coverage_radius, coverage.radius);
        assert_eq!(options.coverage_height, coverage.height);
        assert_eq!(options.coverage_well_tempering, coverage.well_tempering);
        assert_eq!(options.coverage_peer_weight, coverage.peer_weight);
        assert_eq!(options.coverage_neighbors, coverage.neighbors);
        assert_eq!(
            options.history,
            anneal_box_history_t::ANNEAL_BOX_HISTORY_SHARED as i32
        );
        assert_eq!(
            options.coverage_shared,
            anneal_box_coverage_t::ANNEAL_BOX_COVERAGE_AUTO as i32
        );
        assert_eq!(
            unsafe { anneal_box_options_default(std::ptr::null_mut()) },
            anneal_status_t::ANNEAL_INVALID_PARAMETER
        );
    }

    #[test]
    fn the_options_call_reproduces_the_config_call() {
        let mut counter = Counter { evals: 0, grads: 0 };
        let mut x = vec![3.0, -2.0];
        let mut shell = StandingShell::new();
        let low = [-5.12, -5.12];
        let high = [5.12, 5.12];
        let mut cfg = default_config();
        cfg.seed = 7;
        cfg.budget = 600;
        let mut report = anneal_box_report_t::default();
        let status = unsafe {
            anneal_box_minimize(
                rastrigin,
                rastrigin_grad,
                (&mut counter as *mut Counter).cast(),
                shell.point_at(x.as_mut_ptr(), 2),
                low.as_ptr(),
                high.as_ptr(),
                &cfg,
                &mut report,
            )
        };
        assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);

        let mut options = default_options();
        options.seed = 7;
        options.budget = 600;
        let mut counter_ex = Counter { evals: 0, grads: 0 };
        let mut x_ex = vec![3.0, -2.0];
        let mut stats = anneal_box_stats_t {
            struct_size: std::mem::size_of::<anneal_box_stats_t>(),
            ..Default::default()
        };
        let status = unsafe {
            anneal_box_minimize_ex(
                rastrigin,
                rastrigin_grad,
                (&mut counter_ex as *mut Counter).cast(),
                shell.point_at(x_ex.as_mut_ptr(), 2),
                low.as_ptr(),
                high.as_ptr(),
                &options,
                &mut stats,
            )
        };
        assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
        assert_eq!(x, x_ex);
        assert_eq!(
            (report.value, report.n_evals, report.n_grads, report.hops),
            (stats.value, stats.n_evals, stats.n_grads, stats.hops)
        );
        assert_eq!(
            (counter.evals, counter.grads),
            (counter_ex.evals, counter_ex.grads)
        );
    }

    #[test]
    fn each_setting_reaches_the_ensemble() {
        let (status, _, base, _) = run_wells(&wells_options());
        assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
        assert!(base.history_minima > 1);
        assert!(base.shared_deposits > 0 && base.coverage_applied_foreign > 0);
        assert!(base.coverage_repelled_proposals > 0);
        assert!(base.coverage_published > 0 && base.coverage_observations > 0);
        assert!(base.history_observations > 0);

        let run = |edit: &dyn Fn(&mut anneal_box_options_t)| {
            let mut options = wells_options();
            edit(&mut options);
            let (status, _, stats, _) = run_wells(&options);
            assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
            stats
        };
        assert_eq!(run(&|o| o.identity_tol = 1e6).history_minima, 1);
        assert!(run(&|o| o.identity_tol = 1e-12).history_minima > base.history_minima);
        let off = run(&|o| o.shared_deposits = 0);
        assert_eq!((off.shared_deposits, off.coverage_applied_foreign), (0, 0));
        let private =
            run(&|o| o.coverage_shared = anneal_box_coverage_t::ANNEAL_BOX_COVERAGE_PRIVATE as i32);
        assert_eq!(private.coverage_applied_foreign, 0);
        assert_eq!(
            run(&|o| o.coverage_height = 0.0).coverage_repelled_proposals,
            0
        );
        assert_eq!(
            run(&|o| o.coverage_peer_weight = 0.0).coverage_applied_foreign,
            0
        );
        let none = run(&|o| o.history = anneal_box_history_t::ANNEAL_BOX_HISTORY_NONE as i32);
        assert_eq!(none.history_minima, 0);
        let neighbours = run(&|o| o.coverage_neighbors = 1);
        assert!(neighbours.shared_deposits < base.shared_deposits);
        let key = |s: &anneal_box_stats_t| (s.n_evals, s.hops, s.history_minima);
        let recognition = run(&|o| {
            o.visit_policy = anneal_box_visit_policy_t::ANNEAL_BOX_VISIT_RECOGNITION as i32
        });
        assert_ne!(key(&recognition), key(&base));
        let observed =
            run(&|o| o.membership = anneal_box_membership_t::ANNEAL_BOX_MEMBERSHIP_OBSERVED as i32);
        assert_ne!(key(&observed), key(&base));
        let langevin = |friction: f64| {
            run(&|o| {
                o.escape = anneal_box_escape_t::ANNEAL_BOX_ESCAPE_LANGEVIN as i32;
                o.langevin_friction = friction;
            })
        };
        let colored = langevin(0.0);
        let white = langevin(2.0);
        assert_ne!(key(&colored), key(&base));
        assert_ne!(
            (
                colored.n_evals,
                colored.hops,
                colored.coverage_repelled_proposals
            ),
            (white.n_evals, white.hops, white.coverage_repelled_proposals)
        );
    }

    #[test]
    fn a_bad_option_is_refused_before_any_callback() {
        type Edit = fn(&mut anneal_box_options_t);
        let langevin = |o: &mut anneal_box_options_t| {
            o.escape = anneal_box_escape_t::ANNEAL_BOX_ESCAPE_LANGEVIN as i32;
        };
        let cases: Vec<(&str, Edit)> = vec![
            ("replicas", |o| o.replicas = 0),
            ("budget", |o| o.budget = 0),
            ("history", |o| o.history = 7),
            ("membership", |o| o.membership = 9),
            ("identity_tol", |o| o.identity_tol = 0.0),
            ("identity_tol", |o| o.identity_tol = f64::NAN),
            ("visit_policy", |o| o.visit_policy = 5),
            ("escape", |o| o.escape = 3),
            ("coverage_shared", |o| o.coverage_shared = 4),
            ("coverage_radius", |o| o.coverage_radius = 0.0),
            ("coverage_height", |o| o.coverage_height = -1.0),
            ("coverage_well_tempering", |o| {
                o.coverage_well_tempering = 1.0
            }),
            ("coverage_peer_weight", |o| o.coverage_peer_weight = -1.0),
        ];
        for (name, edit) in cases {
            let mut options = wells_options();
            edit(&mut options);
            let (status, _, _, evals) = run_wells(&options);
            assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER, "{name}");
            assert_eq!(evals, 0, "{name}");
            assert!(last_error().contains(name), "{name}: {}", last_error());
        }
        let cases: Vec<(&str, Edit)> = vec![
            ("langevin_steps", |o| o.langevin_steps = 0),
            ("langevin_omega0", |o| o.langevin_omega0 = 0.0),
            ("langevin_dt", |o| o.langevin_dt = -1.0),
            ("langevin_friction", |o| o.langevin_friction = -1.0),
            ("langevin_friction", |o| o.langevin_friction = f64::NAN),
        ];
        for (name, edit) in cases {
            let mut options = wells_options();
            langevin(&mut options);
            edit(&mut options);
            let (status, _, _, evals) = run_wells(&options);
            assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER, "{name}");
            assert_eq!(evals, 0, "{name}");
            assert!(last_error().contains(name), "{name}: {}", last_error());
        }
    }

    #[test]
    fn struct_size_lets_the_options_grow() {
        let (_, full_x, full, _) = run_wells(&wells_options());

        // An older caller whose structure ends before `history`: whatever
        // lies past its end must not be read.
        let mut older = wells_options();
        older.struct_size = std::mem::offset_of!(anneal_box_options_t, history);
        older.identity_tol = -1.0;
        older.coverage_radius = f64::NAN;
        older.shared_deposits = 0;
        let (status, x, stats, _) = run_wells(&older);
        assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
        assert_eq!(x, full_x);
        assert_eq!(stats.n_evals, full.n_evals);

        // A newer caller whose structure is larger than this library's.
        #[repr(C)]
        struct Newer {
            options: anneal_box_options_t,
            future: [u64; 4],
        }
        let mut newer = Newer {
            options: wells_options(),
            future: [u64::MAX; 4],
        };
        newer.options.struct_size = std::mem::size_of::<Newer>();
        let (status, x, _, _) = run_wells(&newer.options);
        assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
        assert_eq!(x, full_x);

        let mut tiny = wells_options();
        tiny.struct_size = 8;
        let (status, _, _, evals) = run_wells(&tiny);
        assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER);
        assert_eq!(evals, 0);
        assert!(last_error().contains("struct_size"));
    }

    #[test]
    fn the_stats_fit_the_callers_struct_size() {
        let mut wells = Wells { evals: 0 };
        let low = [-5.0, -5.0];
        let high = [5.0, 5.0];
        let options = wells_options();
        let mut shell = StandingShell::new();
        let mut call = |stats: *mut anneal_box_stats_t| {
            let mut x = vec![3.0, -2.0];
            unsafe {
                anneal_box_minimize_ex(
                    wells_value,
                    wells_grad,
                    (&mut wells as *mut Wells).cast(),
                    shell.point_at(x.as_mut_ptr(), 2),
                    low.as_ptr(),
                    high.as_ptr(),
                    &options,
                    stats,
                )
            }
        };
        let mut short = anneal_box_stats_t {
            struct_size: std::mem::offset_of!(anneal_box_stats_t, history_observations),
            history_minima: 777,
            ..Default::default()
        };
        assert_eq!(call(&mut short), anneal_status_t::ANNEAL_SUCCESS);
        assert!(short.n_evals > 0 && short.hops > 0);
        assert_eq!(short.history_minima, 777);
        assert_eq!(
            short.struct_size,
            std::mem::offset_of!(anneal_box_stats_t, history_observations)
        );
        #[repr(C)]
        struct NewerStats {
            stats: anneal_box_stats_t,
            future: [u64; 2],
        }
        let mut newer = NewerStats {
            stats: anneal_box_stats_t {
                struct_size: std::mem::size_of::<NewerStats>(),
                ..Default::default()
            },
            future: [u64::MAX; 2],
        };
        assert_eq!(call(&mut newer.stats), anneal_status_t::ANNEAL_SUCCESS);
        assert_eq!(
            newer.stats.struct_size,
            std::mem::size_of::<anneal_box_stats_t>()
        );
        assert_eq!(newer.future, [u64::MAX; 2]);
        assert_eq!(call(std::ptr::null_mut()), anneal_status_t::ANNEAL_SUCCESS);
        let mut unset = anneal_box_stats_t::default();
        assert_eq!(call(&mut unset), anneal_status_t::ANNEAL_INVALID_PARAMETER);
        assert!(last_error().contains("struct_size"));
    }

    #[test]
    fn the_options_call_stops_when_the_callback_says_so() {
        let mut counter = StopAfter {
            evals: 0,
            grads: 0,
            after: 24,
        };
        let mut x = vec![3.0, -2.0];
        let mut shell = StandingShell::new();
        let low = [-5.12, -5.12];
        let high = [5.12, 5.12];
        let mut options = default_options();
        options.seed = 7;
        options.budget = 20_000;
        let mut stats = anneal_box_stats_t {
            struct_size: std::mem::size_of::<anneal_box_stats_t>(),
            ..Default::default()
        };
        let status = unsafe {
            anneal_box_minimize_ex(
                stop_value,
                stop_grad,
                (&mut counter as *mut StopAfter).cast(),
                shell.point_at(x.as_mut_ptr(), 2),
                low.as_ptr(),
                high.as_ptr(),
                &options,
                &mut stats,
            )
        };
        assert_eq!(status, anneal_status_t::ANNEAL_STOPPED);
        assert!(stats.value.is_finite() && stats.n_evals > 0);
        assert!((rastrigin_value(&x) - stats.value).abs() <= 1e-8);
    }

    #[test]
    fn the_options_call_refuses_null_arguments() {
        let options = wells_options();
        let mut wells = Wells { evals: 0 };
        let mut x = vec![0.0, 0.0];
        let mut shell = StandingShell::new();
        let low = [-1.0, -1.0];
        let high = [1.0, 1.0];
        for (null_low, null_options) in [(true, false), (false, true)] {
            let status = unsafe {
                anneal_box_minimize_ex(
                    wells_value,
                    wells_grad,
                    (&mut wells as *mut Wells).cast(),
                    shell.point_at(x.as_mut_ptr(), 2),
                    if null_low {
                        std::ptr::null()
                    } else {
                        low.as_ptr()
                    },
                    high.as_ptr(),
                    if null_options {
                        std::ptr::null()
                    } else {
                        &options
                    },
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER);
        }
        assert_eq!(wells.evals, 0);
    }

    fn run_on(
        x: &mut [f64],
        low: &[f64],
        high: &[f64],
        options: &anneal_box_options_t,
        tamper: impl FnOnce(*mut DLManagedTensorVersioned),
    ) -> (anneal_status_t, usize) {
        let mut wells = Wells { evals: 0 };
        let mut shell = StandingShell::new();
        let tensor = shell.point_at(x.as_mut_ptr(), x.len());
        tamper(tensor);
        let status = unsafe {
            anneal_box_minimize_ex(
                wells_value,
                wells_grad,
                (&mut wells as *mut Wells).cast(),
                tensor,
                low.as_ptr(),
                high.as_ptr(),
                options,
                std::ptr::null_mut(),
            )
        };
        (status, wells.evals)
    }

    #[test]
    fn the_default_options_fill_only_the_callers_struct_size() {
        let full = std::mem::size_of::<anneal_box_options_t>();
        let least = std::mem::offset_of!(anneal_box_options_t, history);
        let fill = |size: usize| -> (anneal_status_t, Vec<u8>) {
            let mut bytes = vec![0xA5_u8; full + 16];
            let out = bytes.as_mut_ptr().cast::<anneal_box_options_t>();
            assert_eq!(
                out as usize % std::mem::align_of::<anneal_box_options_t>(),
                0
            );
            unsafe { std::ptr::addr_of_mut!((*out).struct_size).write(size) };
            let status = unsafe { anneal_box_options_default(out) };
            (status, bytes)
        };

        let (status, bytes) = fill(least);
        assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
        assert!(
            bytes[least..].iter().all(|&b| b == 0xA5),
            "wrote past struct_size"
        );
        let mut older = default_options();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (&mut older as *mut anneal_box_options_t).cast::<u8>(),
                least,
            );
        }
        assert_eq!(older.struct_size, least);
        assert_eq!(older.replicas, default_options().replicas);
        assert_eq!(older.budget, default_options().budget);

        let (status, bytes) = fill(full + 16);
        assert_eq!(status, anneal_status_t::ANNEAL_SUCCESS);
        assert!(
            bytes[full..].iter().all(|&b| b == 0xA5),
            "wrote a newer caller's tail"
        );
        let mut newer = default_options();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (&mut newer as *mut anneal_box_options_t).cast::<u8>(),
                full,
            );
        }
        assert_eq!(newer.struct_size, full);
        assert_eq!(
            anneal_box_options_t {
                struct_size: full,
                ..newer
            },
            default_options()
        );

        for size in [0, 8, least - 1] {
            let (status, bytes) = fill(size);
            assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER, "{size}");
            assert!(
                bytes[std::mem::size_of::<usize>()..]
                    .iter()
                    .all(|&b| b == 0xA5),
                "{size}: wrote a refused structure"
            );
            assert!(last_error().contains("struct_size"));
        }
    }

    #[test]
    fn a_box_the_python_drivers_refuse_is_refused_here_too() {
        let options = wells_options();
        for (low, high, words) in [
            ([-1e308, -1.0], [1e308, 1.0], "finite width"),
            ([-5e307, -5e307], [5e307, 5e307], "finite width"),
            ([f64::NEG_INFINITY, -1.0], [1.0, 1.0], "finite"),
            ([0.0, 1.0], [1.0, f64::NAN], "finite"),
            ([1.0, -1.0], [-1.0, 1.0], "low <= high"),
        ] {
            let mut x = [0.0, 0.0];
            let (status, evals) = run_on(&mut x, &low, &high, &options, |_| {});
            assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER, "{low:?}");
            assert_eq!(evals, 0);
            assert!(last_error().contains(words), "{low:?}: {}", last_error());
        }
    }

    #[test]
    fn a_start_that_is_not_finite_is_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut x = [0.0, bad];
            let (status, evals) =
                run_on(&mut x, &[-5.0, -5.0], &[5.0, 5.0], &wells_options(), |_| {});
            assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER);
            assert_eq!(evals, 0);
            assert!(
                last_error().contains("x[1] is not finite"),
                "{}",
                last_error()
            );
        }
    }

    #[test]
    fn a_tensor_with_a_bad_shape_or_alignment_is_refused() {
        let options = wells_options();
        for shape in [-1_i64, -2, i64::MIN, i64::MAX] {
            let mut x = [0.0, 0.0];
            let (status, evals) =
                run_on(&mut x, &[-5.0, -5.0], &[5.0, 5.0], &options, |t| unsafe {
                    *(*t).dl_tensor.shape = shape;
                });
            assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER, "{shape}");
            assert_eq!(evals, 0);
            assert!(
                last_error().starts_with("anneal_box_minimize_ex:"),
                "{}",
                last_error()
            );
            assert!(last_error().contains("shape"), "{}", last_error());
        }
        let mut x = [0.0, 0.0, 0.0];
        let (status, evals) = run_on(&mut x, &[-5.0, -5.0], &[5.0, 5.0], &options, |t| unsafe {
            *(*t).dl_tensor.shape = 2;
            (*t).dl_tensor.byte_offset = 4;
        });
        assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER);
        assert_eq!(evals, 0);
        assert!(last_error().contains("aligned"), "{}", last_error());
    }

    #[test]
    fn replicas_past_memory_are_refused_before_a_callback() {
        for replicas in [usize::MAX, usize::MAX / 2, 1 << 62] {
            let mut options = wells_options();
            options.replicas = replicas;
            let mut x = [0.0, 0.0];
            let (status, evals) = run_on(&mut x, &[-5.0, -5.0], &[5.0, 5.0], &options, |_| {});
            assert_eq!(
                status,
                anneal_status_t::ANNEAL_INVALID_PARAMETER,
                "{replicas}"
            );
            assert_eq!(evals, 0);
            assert!(last_error().contains("replicas"), "{}", last_error());
        }
    }

    #[test]
    fn an_escape_outside_the_enum_is_refused_not_undefined() {
        for escape in [2_i32, -1, i32::MAX] {
            let mut cfg = default_config();
            unsafe {
                std::ptr::addr_of_mut!(cfg.escape)
                    .cast::<i32>()
                    .write(escape)
            };
            let mut counter = Counter { evals: 0, grads: 0 };
            let mut x = vec![0.0, 0.0];
            let mut shell = StandingShell::new();
            let mut report = anneal_box_report_t::default();
            let status = unsafe {
                anneal_box_minimize(
                    rastrigin,
                    rastrigin_grad,
                    (&mut counter as *mut Counter).cast(),
                    shell.point_at(x.as_mut_ptr(), 2),
                    [-1.0, -1.0].as_ptr(),
                    [1.0, 1.0].as_ptr(),
                    std::ptr::addr_of!(cfg),
                    &mut report,
                )
            };
            assert_eq!(
                status,
                anneal_status_t::ANNEAL_INVALID_PARAMETER,
                "{escape}"
            );
            assert_eq!(counter.evals, 0);
            assert!(last_error().contains("escape"), "{}", last_error());
        }
    }
}
