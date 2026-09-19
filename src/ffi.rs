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

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::sync::Mutex;

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned, DLPackVersion,
    DLTensor,
};
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::methods::box_hopping::{
    BoxCoverageConfig, BoxEnsembleConfig, BoxEscape, GleEscapeConfig,
    box_ensemble_optimize_with_coverage,
};
use crate::methods::ensemble::HistoryMode;
use crate::methods::gle_langevin::GleNoise;

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
    eval: anneal_eval_fn,
    grad: anneal_grad_fn,
    user: usize,
    bounds: Bounds<f64>,
    scratch: Mutex<Scratch>,
    failed: Mutex<bool>,
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
        let user = self.user as *mut c_void;
        let eval = self.eval;
        let mut value = f64::INFINITY;
        let status = self.with_x(x, |_, xt| unsafe { eval(user, xt, &mut value) });
        if status != anneal_status_t::ANNEAL_SUCCESS {
            self.fail("anneal_box_minimize: objective callback failed");
            return f64::INFINITY;
        }
        if value.is_nan() { f64::INFINITY } else { value }
    }
}

impl Gradient<f64> for CallbackObjective {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let user = self.user as *mut c_void;
        let grad = self.grad;
        let n = x.len();
        let mut gradient = Array1::<f64>::zeros(n);
        let status = self.with_x(x, |scratch, xt| {
            let gt = scratch.out.point_at(gradient.as_mut_ptr(), n);
            unsafe { grad(user, xt, gt) }
        });
        if status != anneal_status_t::ANNEAL_SUCCESS {
            self.fail("anneal_box_minimize: gradient callback failed");
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
    what: &str,
) -> Result<&'a mut [f64], anneal_status_t> {
    if tensor.is_null() {
        set_last_error(&format!("anneal_box_minimize: {what} is null"));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    }
    let t = unsafe { &(*tensor).dl_tensor };
    let cpu = t.device.device_type == DLDeviceType::kDLCPU;
    let f64_dtype =
        t.dtype.code == DLDataTypeCode::kDLFloat && t.dtype.bits == 64 && t.dtype.lanes == 1;
    if !cpu || !f64_dtype || t.ndim != 1 || t.shape.is_null() || t.data.is_null() {
        set_last_error(&format!(
            "anneal_box_minimize: {what} must be a one-dimensional CPU float64 tensor"
        ));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    }
    let len = unsafe { *t.shape } as usize;
    if !t.strides.is_null() && unsafe { *t.strides } != 1 && len > 1 {
        set_last_error(&format!("anneal_box_minimize: {what} must be contiguous"));
        return Err(anneal_status_t::ANNEAL_INVALID_PARAMETER);
    }
    let data = unsafe { t.data.cast::<u8>().add(t.byte_offset as usize) }.cast::<f64>();
    Ok(unsafe { std::slice::from_raw_parts_mut(data, len) })
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

/// Minimize `f` over the box `[low, high]` with a communicating-chain box
/// ensemble.
///
/// `x` holds the start of the first replica on entry and the best evaluated
/// point on exit; the other replicas start at uniform draws in the box. A
/// callback that fails ends the search with `ANNEAL_INTERNAL_ERROR`, and
/// `x` is left at its start. `low` and `high` have `dim` entries each and
/// must be finite with `low <= high`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn anneal_box_minimize(
    eval: Option<anneal_eval_fn>,
    grad: Option<anneal_grad_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    low: *const f64,
    high: *const f64,
    cfg: *const anneal_box_config_t,
    out: *mut anneal_box_report_t,
) -> anneal_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (Some(eval), Some(grad)) = (eval, grad) else {
            set_last_error("anneal_box_minimize: eval and grad callbacks are required");
            return anneal_status_t::ANNEAL_INVALID_PARAMETER;
        };
        if low.is_null() || high.is_null() || cfg.is_null() || out.is_null() {
            set_last_error("anneal_box_minimize: null argument");
            return anneal_status_t::ANNEAL_INVALID_PARAMETER;
        }
        let start = match unsafe { cpu_f64_slice_mut(x, "x") } {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let dim = start.len();
        if dim == 0 {
            set_last_error("anneal_box_minimize: x is empty");
            return anneal_status_t::ANNEAL_INVALID_PARAMETER;
        }
        let low = unsafe { std::slice::from_raw_parts(low, dim) };
        let high = unsafe { std::slice::from_raw_parts(high, dim) };
        for (l, h) in low.iter().zip(high) {
            if !(l.is_finite() && h.is_finite() && l <= h) {
                set_last_error(
                    "anneal_box_minimize: bounds must be finite with low <= high",
                );
                return anneal_status_t::ANNEAL_INVALID_PARAMETER;
            }
        }
        let (ensemble, coverage) = match ensemble_config(unsafe { &*cfg }) {
            Ok(c) => c,
            Err(message) => {
                set_last_error(&format!("anneal_box_minimize: {message}"));
                return anneal_status_t::ANNEAL_INVALID_PARAMETER;
            }
        };
        let objective = CallbackObjective {
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
        };
        let x0 = Array1::from_vec(start);
        let result = box_ensemble_optimize_with_coverage(
            &objective,
            &objective,
            unsafe { (*cfg).seed },
            Some(x0.view()),
            &ensemble,
            &coverage,
        );
        if *objective.failed.lock().expect("anneal ffi failure flag") {
            return anneal_status_t::ANNEAL_INTERNAL_ERROR;
        }
        if !result.best_val.is_finite() || result.best_pos.len() != dim {
            set_last_error("anneal_box_minimize: no finite point was evaluated");
            return anneal_status_t::ANNEAL_INTERNAL_ERROR;
        }
        let dest = match unsafe { cpu_f64_slice_mut(x, "x") } {
            Ok(s) => s,
            Err(st) => return st,
        };
        dest.copy_from_slice(result.best_pos.as_slice().expect("contiguous"));
        unsafe {
            *out = anneal_box_report_t {
                value: result.best_val,
                n_evals: result.n_evals,
                n_grads: result.n_grads,
                hops: result.hops,
            };
        }
        anneal_status_t::ANNEAL_SUCCESS
    })) {
        Ok(status) => status,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            set_last_error(&format!("anneal_box_minimize: {message}"));
            anneal_status_t::ANNEAL_INTERNAL_ERROR
        }
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
        let xs = unsafe { cpu_f64_slice_mut(x as *mut _, "x") }.unwrap();
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
        let xs = unsafe { cpu_f64_slice_mut(x as *mut _, "x") }.unwrap();
        let gs = unsafe { cpu_f64_slice_mut(grad_out, "grad") }.unwrap();
        for (g, v) in gs.iter_mut().zip(xs.iter()) {
            *g = 2.0 * v
                + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * v).sin();
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
                Some(rastrigin),
                Some(rastrigin_grad),
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
                    Some(rastrigin),
                    Some(rastrigin_grad),
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
                Some(rastrigin),
                Some(rastrigin_grad),
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
        let status = unsafe {
            anneal_box_minimize(
                None,
                Some(rastrigin_grad),
                (&mut counter as *mut Counter).cast(),
                shell.point_at(x.as_mut_ptr(), 2),
                low.as_ptr(),
                low.as_ptr(),
                &cfg,
                &mut report,
            )
        };
        assert_eq!(status, anneal_status_t::ANNEAL_INVALID_PARAMETER);
        assert_eq!(counter.evals, 0);
    }
}
