// MIT License
// Copyright 2023--present rgpot developers

//! rgpot as a strict superset of eindir: first-member struct embedding.
//!
//! `rgpot_potential_t` embeds `eindir_objective_t` as its FIRST member, so a
//! `rgpot_potential_t*` IS-A `eindir_objective_t*` at the C ABI with zero cost:
//!
//! ```c
//! rgpot_potential_t *pot = rgpot_potential_new_eindir(...);
//! eindir_objective_t *obj = (eindir_objective_t*)pot;  // zero-cost cast
//! double val;
//! eindir_objective_eval(obj, x, &val);                 // works
//! ```
//!
//! The eval_fn / grad_fn stored in the embedded base call through to the
//! rgpot callback, so evaluation logic lives once -- there is no duplicate
//! eval path and no conversion method.

use std::os::raw::{c_char, c_void};
use std::sync::{Arc, Mutex};

use dlpk::sys::DLManagedTensorVersioned;

use crate::fused::{
    lock, rgpot_fused_error_kind_t, FusedCache, FusedError, FusedSettings, FusedValue,
};
use crate::hvp::HvpState;
use crate::potential::PotentialCallback;
use crate::status::{last_error_message, rgpot_status_t, set_last_error};
use crate::tensor::{
    checked_f64_data, copy_f64_tensor, rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3,
    rgpot_tensor_cpu_i32_1d, rgpot_tensor_free,
};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

// ---------------------------------------------------------------------------
// eindir C ABI types -- the real ones, from the shared eindir-core crate
//
// rgpot does NOT mirror eindir's #[repr(C)] types. It re-uses eindir-core's own
// definitions, so `rgpot_potential_t` embeds the exact same `eindir_objective_t`
// and the eval/grad entry points resolve through one compilation of eindir-core.
// That is what lets a downstream Rust consumer (e.g. anneal-core) link both
// crates without the duplicate-symbol / two-Rust-runtime conflict that a
// separate prebuilt `libeindir_core.a` would cause.
// ---------------------------------------------------------------------------

pub use eindir_core::ffi::eindir_status_t;
pub use eindir_core::ffi::{
    eindir_abi_stamp_t, eindir_objective_eval, eindir_objective_grad, eindir_objective_has_grad,
    eindir_objective_t, EindirEvalFn, EindirFreeFn, EindirGradFn,
};

/// The eindir ABI stamp of the `eindir_objective_t` base that
/// `rgpot_potential_t` embeds as its first member.
///
/// A consumer that takes the potential as an eindir objective (rgmin's
/// `rgmin_minimize_eindir`, for one) passes this stamp to
/// `eindir_core_abi_compatible`, so a layout from a different eindir-core
/// revision is refused instead of read as the wrong struct.
#[no_mangle]
pub extern "C" fn rgpot_eindir_abi_stamp() -> eindir_abi_stamp_t {
    eindir_core::ffi::eindir_core_abi_stamp()
}

/// Return nonzero when the eindir-core compiled into this library can
/// consume an objective described by `stamp`.
///
/// This is `eindir_core_abi_compatible` reached through rgpot, so a caller
/// that links only rgpot checks the stamp against the same eindir-core that
/// built the `eindir_objective_t` base. A null `stamp` is not compatible.
///
/// # Safety
///
/// `stamp` must be null or point to a readable `eindir_abi_stamp_t`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_eindir_abi_compatible(stamp: *const eindir_abi_stamp_t) -> i32 {
    unsafe { eindir_core::ffi::eindir_core_abi_compatible(stamp) }
}

// ---------------------------------------------------------------------------
// rgpot_potential_t: rgpot_potential_t* IS-A eindir_objective_t*
// ---------------------------------------------------------------------------

/// Superset potential: embeds `eindir_objective_t` as its first member.
///
/// Because `base` is first and both structs are `#[repr(C)]`, casting a
/// `*mut rgpot_potential_t` to `*mut eindir_objective_t` is defined behaviour
/// at the C ABI.  No conversion method is needed.
#[repr(C)]
pub struct rgpot_potential_t {
    /// Embedded eindir base: the IS-A relationship lives here.
    /// MUST remain the first field.
    pub base: eindir_objective_t,
    /// The underlying rgpot force/energy callback.
    pub callback: PotentialCallback,
    /// Opaque context for `callback` (distinct from `base.user_data`).
    pub pot_user_data: *mut c_void,
    /// Optional destructor for `pot_user_data`.
    pub pot_free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
    /// Atom count; set by `rgpot_potential_new_eindir`.
    pub n_atoms: usize,
    /// Atomic numbers, heap-allocated (len = n_atoms), owned by this struct.
    pub atomic_numbers: *mut i32,
    /// Box matrix copied at construction. Nine doubles, lattice vectors as
    /// rows: ax ay az bx by bz cx cy cz. Same layout as `ForceInput.box`.
    pub box_matrix: [f64; 9],
    /// Outcome of the last fused evaluation, shared by every request with
    /// the same key. Held behind a pointer so the C view of the struct stays
    /// plain C.
    fused_cache: Box<rgpot_fused_cache_t>,
}

impl rgpot_potential_t {
    /// The recorded peer refusal, if capabilities were installed and refused.
    pub(crate) fn fused_cache_peer_refusal(&self) -> Option<String> {
        self.fused_cache.peer_refusal()
    }

    /// Hessian-vector state: the analytic callback and its counters.
    pub(crate) fn hvp_state(&self) -> &HvpState {
        &self.fused_cache.hvp
    }

    /// The cache every fused request of this handle goes through.
    pub(crate) fn fused(&self) -> &FusedCache<Arc<FusedValue>> {
        &self.fused_cache.fused
    }

    /// Run the callback once at `input` and classify the outcome.
    ///
    /// The thread's error message is cleared first, so a failure carries
    /// the message the callback set and never a stale one.
    ///
    /// # Safety
    ///
    /// The tensors of `input` must be valid for the call, and the callback
    /// safe to call with its `user_data`.
    pub(crate) unsafe fn run_engine(
        &self,
        input: &rgpot_force_input_t,
        n_atoms: usize,
    ) -> Result<Arc<FusedValue>, FusedError> {
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        set_last_error("");
        let status = unsafe { (self.callback)(self.pot_user_data, input, &mut output) };
        let outcome = if status == rgpot_status_t::RGPOT_SUCCESS {
            validated_result(&output, 3 * n_atoms)
                .map(|forces| {
                    Arc::new(FusedValue {
                        energy: output.energy,
                        variance: output.variance,
                        forces,
                    })
                })
                .map_err(|message| FusedError::Invalid { message })
        } else {
            let message = last_error_message();
            let message = if !message.is_empty() {
                message
            } else if status == rgpot_status_t::RGPOT_SCC_NOT_CONVERGED {
                "SCC did not converge".to_owned()
            } else {
                format!("callback returned {status:?}")
            };
            Err(FusedError::from_status(status, message))
        };
        unsafe { rgpot_tensor_free(output.forces) };
        outcome
    }

    /// The fused outcome at `x` for the species and cell of this handle.
    ///
    /// # Safety
    ///
    /// The callback must be safe to call with its `user_data`, and `x` must
    /// hold `3 * n_atoms` values.
    unsafe fn fused_at(&self, x: &[f64]) -> Result<Arc<FusedValue>, FusedError> {
        let mut pos = x.to_vec();
        let mut atmnrs =
            unsafe { std::slice::from_raw_parts(self.atomic_numbers, self.n_atoms) }.to_vec();
        let mut box_ = self.box_matrix;
        let key = self.fused().key(x, Some(&atmnrs), Some(&box_));
        let (outcome, _) = self.fused().evaluate(key, || {
            let input = rgpot_force_input_t {
                positions: unsafe {
                    rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), self.n_atoms as i64, 3)
                },
                atomic_numbers: unsafe {
                    rgpot_tensor_cpu_i32_1d(atmnrs.as_mut_ptr(), self.n_atoms as i64)
                },
                box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(box_.as_mut_ptr()) },
            };
            let outcome = unsafe { self.run_engine(&input, self.n_atoms) };
            unsafe {
                rgpot_tensor_free(input.positions);
                rgpot_tensor_free(input.atomic_numbers);
                rgpot_tensor_free(input.box_matrix);
            }
            outcome
        });
        outcome
    }
}

/// Opaque cache of the last fused energy and gradient evaluation.
pub struct rgpot_fused_cache_t {
    fused: FusedCache<Arc<FusedValue>>,
    /// Why the peer behind this handle was refused, once capabilities were
    /// installed and found incompatible. Evaluations fail while it is set.
    peer_refusal: Mutex<Option<String>>,
    /// Lives here, behind the pointer, so `rgpot_potential_t` keeps its C
    /// layout.
    hvp: HvpState,
}

impl rgpot_fused_cache_t {
    fn new() -> Self {
        Self {
            fused: FusedCache::new(),
            peer_refusal: Mutex::new(None),
            hvp: HvpState::default(),
        }
    }

    /// The recorded refusal, if any.
    pub(crate) fn peer_refusal(&self) -> Option<String> {
        lock(&self.peer_refusal).clone()
    }

    #[cfg(feature = "schema")]
    fn set_peer_refusal(&self, why: Option<String>) {
        *lock(&self.peer_refusal) = why;
        self.fused.invalidate();
    }
}

/// Install the `Capabilities` message of the peer behind `pot`, checked
/// against what this build speaks.
///
/// A compatible peer is accepted. An incompatible one makes
/// `rgpot_potential_calculate`, `eindir_objective_eval` and
/// `eindir_objective_grad` on this handle fail before the callback runs: the
/// first returns `RGPOT_INVALID_PARAMETER`, the others
/// `EINDIR_INVALID_PARAMETER`, all with a `rgpot_last_error()` message of the
/// form `peer refused: <first mismatch>`. A handle that never receives a
/// message keeps the plain callback contract.
///
/// `caps` points to `size` bytes of a flat Cap'n Proto `Capabilities`
/// message.
///
/// # Safety
///
/// `pot` must be NULL or a live handle, and `caps` NULL or `size` readable
/// bytes.
#[cfg(feature = "schema")]
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_peer_capabilities(
    pot: *const rgpot_potential_t,
    caps: *const u8,
    size: usize,
) -> rgpot_status_t {
    if pot.is_null() || caps.is_null() || size == 0 || size % 8 != 0 {
        set_last_error("rgpot_potential_set_peer_capabilities: invalid argument");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let mut words = vec![0u64; size / 8];
    unsafe { std::ptr::copy_nonoverlapping(caps, words.as_mut_ptr().cast::<u8>(), size) };
    let verdict =
        crate::compat::check_capabilities_words(&words, &crate::compat::Expectation::default());
    let cache = unsafe { &*(*pot).fused_cache };
    match verdict {
        Ok(()) => {
            cache.set_peer_refusal(None);
            rgpot_status_t::RGPOT_SUCCESS
        }
        Err(why) => {
            let message = format!("peer refused: {why}");
            set_last_error(&message);
            cache.set_peer_refusal(Some(message));
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        }
    }
}

/// Evaluations a potential handle has answered, split by origin.
///
/// `computed` counts callback invocations made through the fused cache;
/// `served` counts requests, energy or gradient, success or failure, answered
/// from the stored outcome without invoking the callback. A caller that
/// charges evaluations chooses which field to bill.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct rgpot_eval_counts_t {
    pub computed: u64,
    pub served: u64,
}

/// Read the computed and served counters of `pot` into `out`.
///
/// # Safety
///
/// `pot` must be NULL or a live pointer from [`rgpot_potential_new_eindir`],
/// and `out` NULL or writable. NULL for either yields
/// `RGPOT_INVALID_PARAMETER`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_eval_counts(
    pot: *const rgpot_potential_t,
    out: *mut rgpot_eval_counts_t,
) -> rgpot_status_t {
    if pot.is_null() || out.is_null() {
        set_last_error("rgpot_potential_eval_counts: NULL argument");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let cache = unsafe { &*pot }.fused();
    unsafe {
        *out = rgpot_eval_counts_t {
            computed: cache.computed(),
            served: cache.served(),
        };
    }
    rgpot_status_t::RGPOT_SUCCESS
}

/// Zero the counters of `pot`.
///
/// # Safety
///
/// `pot` must be NULL or a live pointer from [`rgpot_potential_new_eindir`].
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_reset_eval_counts(pot: *const rgpot_potential_t) {
    if pot.is_null() {
        return;
    }
    unsafe { &*pot }.fused().reset_counts();
}

/// Engine settings that change the answer of a fused evaluation, for the C
/// ABI.
///
/// Fill it with [`rgpot_fused_settings_default`], set the fields that
/// differ, and install it with [`rgpot_potential_set_fused_settings`]. Two
/// requests at the same positions share an outcome only when these fields,
/// the species and the cell are equal, so changing one is how a failed
/// geometry is evaluated again. `struct_size` is the size of the caller's
/// struct; a library that grows the struct reads only the fields the caller's
/// size covers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct rgpot_fused_settings_t {
    /// `sizeof` of the struct the caller compiled against.
    pub struct_size: usize,
    /// Bit `i` is set when the cell is periodic along vector `i`.
    pub periodic: u32,
    /// Number of unpaired electrons.
    pub spin: i32,
    /// SCC iteration budget.
    pub scc_max_iterations: u32,
    /// Total charge, in elementary charges.
    pub charge: f64,
    /// Numerical accuracy level of the engine.
    pub accuracy: f64,
    /// Electronic temperature, in kelvin.
    pub electronic_temperature: f64,
    /// NUL-terminated name of the SCC mixer, or NULL.
    pub scc_mixer: *const c_char,
    /// NUL-terminated build or version string of the engine, or NULL.
    pub engine_build: *const c_char,
}

impl Default for rgpot_fused_settings_t {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            periodic: 0,
            spin: 0,
            scc_max_iterations: 0,
            charge: 0.0,
            accuracy: 0.0,
            electronic_temperature: 0.0,
            scc_mixer: std::ptr::null(),
            engine_build: std::ptr::null(),
        }
    }
}

impl rgpot_fused_settings_t {
    #[must_use]
    pub fn with_periodic(mut self, bits: u32) -> Self {
        self.periodic = bits;
        self
    }

    #[must_use]
    pub fn with_spin(mut self, spin: i32) -> Self {
        self.spin = spin;
        self
    }

    #[must_use]
    pub fn with_scc_max_iterations(mut self, iterations: u32) -> Self {
        self.scc_max_iterations = iterations;
        self
    }

    #[must_use]
    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    #[must_use]
    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = accuracy;
        self
    }

    #[must_use]
    pub fn with_electronic_temperature(mut self, kelvin: f64) -> Self {
        self.electronic_temperature = kelvin;
        self
    }

    #[must_use]
    pub fn with_scc_mixer(mut self, mixer: *const c_char) -> Self {
        self.scc_mixer = mixer;
        self
    }

    #[must_use]
    pub fn with_engine_build(mut self, build: *const c_char) -> Self {
        self.engine_build = build;
        self
    }
}

/// Fill `out` with the default settings and the size of this library's
/// struct.
///
/// # Safety
///
/// `out` must be NULL or writable. NULL is ignored.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fused_settings_default(out: *mut rgpot_fused_settings_t) {
    if !out.is_null() {
        unsafe { *out = rgpot_fused_settings_t::default() };
    }
}

unsafe fn read_text(text: *const c_char, what: &str) -> Result<String, String> {
    if text.is_null() {
        return Ok(String::new());
    }
    unsafe { std::ffi::CStr::from_ptr(text) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| format!("rgpot_potential_set_fused_settings: {what} is not UTF-8"))
}

/// Install the engine settings that key the fused cache of `pot`.
///
/// The stored outcome stays, and a request made under other settings has
/// another key, so it runs the engine even at the stored positions. A request
/// under the same settings returns the stored outcome, a failure included.
///
/// Returns `RGPOT_INVALID_PARAMETER` for a NULL argument, a `struct_size`
/// smaller than this library's struct, or a string that is not UTF-8.
///
/// # Safety
///
/// `pot` must be NULL or a live handle, `settings` NULL or readable, and its
/// strings NULL or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_fused_settings(
    pot: *const rgpot_potential_t,
    settings: *const rgpot_fused_settings_t,
) -> rgpot_status_t {
    if pot.is_null() || settings.is_null() {
        set_last_error("rgpot_potential_set_fused_settings: NULL argument");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let settings = unsafe { &*settings };
    if settings.struct_size < std::mem::size_of::<rgpot_fused_settings_t>() {
        set_last_error("rgpot_potential_set_fused_settings: struct_size is too small");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let mixer = match unsafe { read_text(settings.scc_mixer, "scc_mixer") } {
        Ok(text) => text,
        Err(why) => {
            set_last_error(&why);
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
    };
    let build = match unsafe { read_text(settings.engine_build, "engine_build") } {
        Ok(text) => text,
        Err(why) => {
            set_last_error(&why);
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
    };
    let converted = FusedSettings::default()
        .with_periodic([
            settings.periodic & 1 != 0,
            settings.periodic & 2 != 0,
            settings.periodic & 4 != 0,
        ])
        .with_charge(settings.charge)
        .with_spin(settings.spin)
        .with_accuracy(settings.accuracy)
        .with_electronic_temperature(settings.electronic_temperature)
        .with_scc_max_iterations(settings.scc_max_iterations)
        .with_scc_mixer(mixer)
        .with_engine_build(build);
    unsafe { &*pot }.fused().set_settings(converted);
    rgpot_status_t::RGPOT_SUCCESS
}

/// The kind of failure stored for the last geometry `pot` evaluated through
/// its fused cache: `RGPOT_FUSED_SCC_NOT_CONVERGED` for an SCC that did not
/// converge, `RGPOT_FUSED_ENGINE_FAILED` or `RGPOT_FUSED_INVALID_RESULT` for
/// another failure, and `RGPOT_FUSED_OK` for a success, an empty cache or a
/// NULL `pot`.
///
/// Both the energy and the gradient request that find the failure return the
/// same status, and `rgpot_last_error` carries the message of the first
/// failure. Query the kind after either to tell the failures apart.
///
/// # Safety
///
/// `pot` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_fused_error_kind(
    pot: *const rgpot_potential_t,
) -> rgpot_fused_error_kind_t {
    if pot.is_null() {
        return rgpot_fused_error_kind_t::RGPOT_FUSED_OK;
    }
    match unsafe { &*pot }.fused().stored_error() {
        Some(error) => error.kind(),
        None => rgpot_fused_error_kind_t::RGPOT_FUSED_OK,
    }
}

// Safety: user pointers are opaque; caller guarantees thread safety.
unsafe impl Send for rgpot_potential_t {}

// ---------------------------------------------------------------------------
// Eval / grad callbacks wired into the embedded base
//
// base.user_data = ptr to the rgpot_potential_t itself (set in the constructor
// after Box::into_raw so the address is stable).  The callbacks cast it back
// to access molecular data and the rgpot callback.
// ---------------------------------------------------------------------------

/// Record `msg` for `rgpot_last_error` and report a malformed argument or
/// callback result, which stays distinguishable from a failed callback
/// (`EINDIR_INTERNAL_ERROR`).
fn reject(msg: &str) -> eindir_status_t {
    set_last_error(msg);
    eindir_status_t::EINDIR_INVALID_PARAMETER
}

/// Forces returned by a callback that reported success: shape, dtype and
/// finiteness are checked before any element is read, and the energy must be
/// finite.
pub(crate) fn validated_result(output: &rgpot_force_out_t, n: usize) -> Result<Vec<f64>, String> {
    let forces = copy_f64_tensor(output.forces, "callback forces", n, true)?;
    if !output.energy.is_finite() {
        return Err("callback energy is not finite".into());
    }
    Ok(forces)
}

/// Record the message of a failed fused evaluation for `rgpot_last_error`
/// and map it to the eindir status. A rejected result stays distinguishable
/// from a failed callback (`EINDIR_INTERNAL_ERROR`).
fn report(error: &FusedError) -> eindir_status_t {
    set_last_error(error.message());
    match error {
        FusedError::Invalid { .. } => eindir_status_t::EINDIR_INVALID_PARAMETER,
        _ => eindir_status_t::EINDIR_INTERNAL_ERROR,
    }
}

unsafe extern "C" fn rgpot_eval_cb(
    user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t {
    let pot = unsafe { &*(user_data as *const rgpot_potential_t) };
    if let Some(why) = pot.fused_cache.peer_refusal() {
        return reject(&why);
    }
    let Some(n) = pot.n_atoms.checked_mul(3) else {
        return reject("n_atoms * 3 overflows");
    };
    let x_data = match copy_f64_tensor(x, "eindir x", n, false) {
        Ok(v) => v,
        Err(msg) => return reject(&msg),
    };
    match unsafe { pot.fused_at(&x_data) } {
        Ok(value) => {
            unsafe { *value_out = value.energy };
            eindir_status_t::EINDIR_SUCCESS
        }
        Err(error) => report(&error),
    }
}

unsafe extern "C" fn rgpot_grad_cb(
    user_data: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    let pot = unsafe { &*(user_data as *const rgpot_potential_t) };
    if let Some(why) = pot.fused_cache.peer_refusal() {
        return reject(&why);
    }
    let Some(n) = pot.n_atoms.checked_mul(3) else {
        return reject("n_atoms * 3 overflows");
    };
    let x_data = match copy_f64_tensor(x, "eindir x", n, false) {
        Ok(v) => v,
        Err(msg) => return reject(&msg),
    };
    let dst_ptr = match checked_f64_data(grad_out, "eindir grad_out", n) {
        Ok(p) => p,
        Err(msg) => return reject(&msg),
    };
    match unsafe { pot.fused_at(&x_data) } {
        Ok(value) => {
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, n) };
            for (d, force) in dst.iter_mut().zip(&value.forces) {
                *d = -*force;
            }
            eindir_status_t::EINDIR_SUCCESS
        }
        Err(error) => report(&error),
    }
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

/// Create a potential that is ALSO a valid eindir objective.
///
/// The returned pointer is simultaneously a `rgpot_potential_t*` and (via
/// zero-cost cast to the first member) an `eindir_objective_t*`.
///
/// - `callback`: the rgpot force/energy callback.
/// - `user_data` / `free_fn`: opaque context for the rgpot callback.
/// - `n_atoms`, `atomic_numbers`, `box_matrix`: molecular context.
/// - `bounds_low`, `bounds_high`: f64 arrays of length `n_atoms * 3`
///   (may be NULL; defaults to [-50, 50] per dimension).
///
/// The caller must eventually pass the returned pointer to
/// [`rgpot_potential_free`].
///
/// # Safety
///
/// `atomic_numbers` must be NULL or point to `n_atoms` readable `i32`
/// values, `box_matrix` NULL or 9 readable `f64` values, and each bound NULL
/// or `n_atoms * 3` readable `f64` values. `callback` and `free_fn` must be
/// safe to call with `user_data`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_new_eindir(
    callback: PotentialCallback,
    user_data: *mut c_void,
    free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
    n_atoms: usize,
    atomic_numbers: *const i32,
    box_matrix: *const f64,
    bounds_low: *const f64,
    bounds_high: *const f64,
) -> *mut rgpot_potential_t {
    let Some(dim) = n_atoms.checked_mul(3) else {
        set_last_error("n_atoms * 3 overflows");
        return std::ptr::null_mut();
    };

    let alloc_f64 = |src: *const f64, default: f64| -> *mut f64 {
        let mut v = vec![default; dim];
        if !src.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(src, v.as_mut_ptr(), dim) };
        }
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        ptr
    };

    let low = alloc_f64(bounds_low, -50.0);
    let high = alloc_f64(bounds_high, 50.0);

    let atmnrs = {
        let mut v = vec![0i32; n_atoms];
        if !atomic_numbers.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(atomic_numbers, v.as_mut_ptr(), n_atoms) };
        }
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        ptr
    };

    let mut box_arr = [0.0f64; 9];
    if !box_matrix.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(box_matrix, box_arr.as_mut_ptr(), 9) };
    }

    let pot = Box::new(rgpot_potential_t {
        base: eindir_objective_t {
            dim,
            low,
            high,
            eval_fn: rgpot_eval_cb,
            grad_fn: Some(rgpot_grad_cb),
            // user_data points to self; set below after Box::into_raw.
            user_data: std::ptr::null_mut(),
            // No free_fn in the embedded base: rgpot_potential_free handles all cleanup.
            free_fn: None,
        },
        callback,
        pot_user_data: user_data,
        pot_free_fn: free_fn,
        n_atoms,
        atomic_numbers: atmnrs,
        box_matrix: box_arr,
        fused_cache: Box::new(rgpot_fused_cache_t::new()),
    });
    let ptr = Box::into_raw(pot);
    // Self-referential: the eindir base's user_data points to the owning struct
    // so the callbacks can access molecular data via a back-cast.
    unsafe { (*ptr).base.user_data = ptr as *mut c_void };
    ptr
}

/// Free a potential created by [`rgpot_potential_new_eindir`].
///
/// Calls `pot_free_fn(pot_user_data)` if provided, frees all owned arrays,
/// then frees the struct.  Do NOT call `eindir_objective_free` on the embedded
/// base; call this function.
///
/// # Safety
///
/// `pot` must be NULL or a pointer returned by
/// [`rgpot_potential_new_eindir`] that has not been freed yet. The pointer is
/// invalid after the call.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_free_eindir(pot: *mut rgpot_potential_t) {
    if pot.is_null() {
        return;
    }
    let p = unsafe { &*pot };
    if let Some(ff) = p.pot_free_fn {
        if !p.pot_user_data.is_null() {
            unsafe { ff(p.pot_user_data) };
        }
    }
    // Free embedded base's bounds arrays.
    let dim = p.base.dim;
    if !p.base.low.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.base.low, dim, dim)) };
    }
    if !p.base.high.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.base.high, dim, dim)) };
    }
    // Free molecular arrays.
    if !p.atomic_numbers.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.atomic_numbers, p.n_atoms, p.n_atoms)) };
    }
    unsafe { drop(Box::from_raw(pot)) };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::create_owned_f64_tensor;
    use eindir_core::ffi::EindirObjectiveWrapper;
    use eindir_core::gradient::DifferentiableObjective;
    use ndarray::Array1;

    struct Ctx1d {
        shape: [i64; 1],
        strides: [i64; 1],
    }

    unsafe extern "C" fn del1d(ptr: *mut DLManagedTensorVersioned) {
        if ptr.is_null() {
            return;
        }
        let ctx = unsafe { (*ptr).manager_ctx.cast::<Ctx1d>() };
        if !ctx.is_null() {
            drop(unsafe { Box::from_raw(ctx) });
        }
        drop(unsafe { Box::from_raw(ptr) });
    }

    fn make_1d(data: *mut f64, n: usize) -> *mut DLManagedTensorVersioned {
        use dlpk::sys::{
            DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLPackVersion, DLTensor,
        };
        let mut ctx = Box::new(Ctx1d {
            shape: [n as i64],
            strides: [1],
        });
        let t = DLTensor {
            data: data.cast(),
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
            shape: ctx.shape.as_mut_ptr(),
            strides: ctx.strides.as_mut_ptr(),
            byte_offset: 0,
        };
        let managed = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(del1d),
            flags: 0,
            dl_tensor: t,
        });
        Box::into_raw(managed)
    }

    // Mock rgpot callback: energy = sum(positions), forces = -1 everywhere.
    unsafe extern "C" fn mock_energy_callback(
        _user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        let n = unsafe { inp.n_atoms() }.unwrap_or(0);
        let pt = unsafe { &(*inp.positions).dl_tensor };
        let pos = unsafe { std::slice::from_raw_parts(pt.data as *const f64, n * 3) };
        out.energy = pos.iter().sum();
        out.variance = 0.0;
        out.forces = create_owned_f64_tensor(vec![-1.0f64; n * 3], vec![n as i64, 3]);
        rgpot_status_t::RGPOT_SUCCESS
    }

    unsafe extern "C" fn counting_energy_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let calls = unsafe { &mut *(user_data as *mut usize) };
        *calls += 1;
        unsafe { mock_energy_callback(std::ptr::null_mut(), input, output) }
    }

    #[test]
    fn fused_value_and_gradient_executes_one_force_request() {
        let n_atoms = 2usize;
        let atmnrs = [1i32, 1];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let mut calls = 0usize;
        let pot = unsafe {
            rgpot_potential_new_eindir(
                counting_energy_callback,
                (&mut calls as *mut usize).cast(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert!(!pot.is_null());
        let objective = unsafe { EindirObjectiveWrapper::new(&(*pot).base) };
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let (energy, gradient) = objective.value_and_gradient(x.view());

        assert_eq!(energy, 21.0);
        assert!(gradient.iter().all(|&value| value == 1.0));
        assert_eq!(calls, 1, "a fused request must execute the engine once");
        unsafe { rgpot_potential_free_eindir(pot) };
    }

    /// The IS-A cast: a rgpot_potential_t* is passed to eindir functions by
    /// casting to eindir_objective_t* -- no conversion method, no duplicate
    /// eval path.  Energy = sum(positions) = 1+2+3+4+5+6 = 21.
    #[test]
    fn rgpot_is_eindir_objective() {
        let n_atoms = 2usize;
        let dim = n_atoms * 3;
        let atmnrs = [1i32, 1];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let low_data = vec![-50.0f64; dim];
        let high_data = vec![50.0f64; dim];

        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                low_data.as_ptr(),
                high_data.as_ptr(),
            )
        };
        assert!(!pot.is_null());

        // THE IS-A CAST: zero cost, no conversion method.
        let obj = pot as *mut eindir_objective_t;

        let mut x_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_t = make_1d(x_data.as_mut_ptr(), dim);

        let mut value = 0.0f64;
        let s = unsafe { eindir_objective_eval(obj, x_t, &mut value) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        assert_eq!(value, 21.0); // 1+2+3+4+5+6

        assert_eq!(unsafe { eindir_objective_has_grad(obj) }, 1);

        // Gradient = -forces = 1.0 everywhere.
        let mut g_data = vec![0.0f64; dim];
        let g_t = make_1d(g_data.as_mut_ptr(), dim);
        let s = unsafe { eindir_objective_grad(obj, x_t, g_t) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        assert!(g_data.iter().all(|&v| v == 1.0));

        unsafe {
            del1d(x_t);
            del1d(g_t);
            rgpot_potential_free_eindir(pot);
        }
    }

    /// Verify struct field values survive a round-trip through the IS-A cast
    /// without calling into the shared library.
    #[test]
    fn base_fields_via_cast() {
        let n_atoms = 2usize;
        let atmnrs = [1i32, 1];
        let box_ = [10.0f64; 9];
        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert!(!pot.is_null());
        let obj = pot as *const eindir_objective_t;
        let dim = unsafe { (*obj).dim };
        let user_data = unsafe { (*obj).user_data };
        assert_eq!(dim, 6); // n_atoms * 3
        assert_eq!(user_data, pot as *mut c_void);
        unsafe { rgpot_potential_free_eindir(pot) };
    }

    /// Verify that the embedded base is the first member at offset 0.
    #[test]
    fn base_is_first_member() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(rgpot_potential_t, base), 0);
    }

    /// Existing rgpot calculations still work through the direct callback path.
    #[test]
    fn direct_callback_path_still_works() {
        let n_atoms = 3usize;
        let dim = n_atoms * 3;
        let atmnrs = [1i32; 3];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        // Evaluate through the eindir path to check correctness.
        let obj = pot as *mut eindir_objective_t;
        let mut positions = vec![1.0f64; dim];
        let x_t = make_1d(positions.as_mut_ptr(), dim);
        let mut value = 0.0f64;
        let s = unsafe { eindir_objective_eval(obj, x_t, &mut value) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        assert_eq!(value, dim as f64); // sum of 9 ones

        unsafe {
            del1d(x_t);
            rgpot_potential_free_eindir(pot);
        }
    }

    /// The gradient is the negative of the forces (sign convention).
    #[test]
    fn bridge_grad_returns_neg_forces() {
        let n_atoms = 2usize;
        let dim = n_atoms * 3;
        let atmnrs = [1i32; 2];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                n_atoms,
                atmnrs.as_ptr(),
                box_.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        let obj = pot as *mut eindir_objective_t;
        let mut x_data = vec![0.5f64; dim];
        let x_t = make_1d(x_data.as_mut_ptr(), dim);
        let mut g_data = vec![0.0f64; dim];
        let g_t = make_1d(g_data.as_mut_ptr(), dim);

        let s = unsafe { eindir_objective_grad(obj, x_t, g_t) };
        assert_eq!(s, eindir_status_t::EINDIR_SUCCESS);
        // mock_energy_callback sets forces = -1.0; gradient = -force = 1.0.
        assert!(g_data.iter().all(|&v| v == 1.0), "grad = -(-1) = 1");

        unsafe {
            del1d(x_t);
            del1d(g_t);
            rgpot_potential_free_eindir(pot);
        }
    }

    /// What a hostile or buggy callback hands back with RGPOT_SUCCESS.
    #[derive(Clone, Copy)]
    enum Bad {
        NullForces,
        WrongLength,
        WrongRank,
        NotF64,
        Nan,
        Inf,
        NullData,
        NonFiniteEnergy,
    }

    unsafe extern "C" fn bad_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let mode = unsafe { *(user_data as *const Bad) };
        let n = unsafe { (*input).n_atoms() }.unwrap_or(0);
        let out = unsafe { &mut *output };
        out.energy = 1.0;
        out.forces = match mode {
            Bad::NullForces => std::ptr::null_mut(),
            Bad::WrongLength => {
                create_owned_f64_tensor(vec![0.0; n * 3 - 1], vec![(n * 3 - 1) as i64])
            }
            Bad::WrongRank => create_owned_f64_tensor(vec![0.0; n * 3], vec![1, n as i64, 3]),
            Bad::NotF64 => {
                let t = create_owned_f64_tensor(vec![0.0; n * 3], vec![n as i64, 3]);
                unsafe { (*t).dl_tensor.dtype.bits = 32 };
                t
            }
            Bad::Nan => {
                let mut v = vec![0.0; n * 3];
                v[1] = f64::NAN;
                create_owned_f64_tensor(v, vec![n as i64, 3])
            }
            Bad::Inf => {
                let mut v = vec![0.0; n * 3];
                v[2] = f64::INFINITY;
                create_owned_f64_tensor(v, vec![n as i64, 3])
            }
            Bad::NullData => {
                let t = create_owned_f64_tensor(vec![0.0; n * 3], vec![n as i64, 3]);
                unsafe { (*t).dl_tensor.data = std::ptr::null_mut() };
                t
            }
            Bad::NonFiniteEnergy => {
                out.energy = f64::NAN;
                create_owned_f64_tensor(vec![0.0; n * 3], vec![n as i64, 3])
            }
        };
        rgpot_status_t::RGPOT_SUCCESS
    }

    #[test]
    fn malformed_successful_results_are_rejected() {
        let atmnrs = [1i32, 1];
        let box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        for mode in [
            Bad::NullForces,
            Bad::WrongLength,
            Bad::WrongRank,
            Bad::NotF64,
            Bad::Nan,
            Bad::Inf,
            Bad::NullData,
            Bad::NonFiniteEnergy,
        ] {
            let mut mode = mode;
            let pot = unsafe {
                rgpot_potential_new_eindir(
                    bad_callback,
                    (&mut mode as *mut Bad).cast(),
                    None,
                    2,
                    atmnrs.as_ptr(),
                    box_.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                )
            };
            let obj = pot as *mut eindir_objective_t;
            let mut x_data = [0.5f64; 6];
            let x_t = make_1d(x_data.as_mut_ptr(), 6);
            let mut g_data = [7.0f64; 6];
            let g_t = make_1d(g_data.as_mut_ptr(), 6);

            let mut value = 0.0f64;
            let s = unsafe { eindir_objective_eval(obj, x_t, &mut value) };
            assert_eq!(s, eindir_status_t::EINDIR_INVALID_PARAMETER);
            let s = unsafe { eindir_objective_grad(obj, x_t, g_t) };
            assert_eq!(s, eindir_status_t::EINDIR_INVALID_PARAMETER);
            assert!(
                g_data.iter().all(|&v| v == 7.0),
                "gradient must stay untouched"
            );

            unsafe {
                del1d(x_t);
                del1d(g_t);
                rgpot_potential_free_eindir(pot);
            }
        }
    }

    #[test]
    fn malformed_input_tensors_are_rejected() {
        let atmnrs = [1i32, 1];
        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                2,
                atmnrs.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        let obj = pot as *mut eindir_objective_t;
        let mut value = 0.0f64;
        let mut short = [0.5f64; 5];
        let short_t = make_1d(short.as_mut_ptr(), 5);
        let mut x_data = [0.5f64; 6];
        let x_t = make_1d(x_data.as_mut_ptr(), 6);
        let mut g_data = [0.0f64; 5];
        let g_short = make_1d(g_data.as_mut_ptr(), 5);

        assert_eq!(
            unsafe { eindir_objective_eval(obj, std::ptr::null(), &mut value) },
            eindir_status_t::EINDIR_INVALID_PARAMETER
        );
        assert_eq!(
            unsafe { eindir_objective_eval(obj, short_t, &mut value) },
            eindir_status_t::EINDIR_INVALID_PARAMETER
        );
        assert_eq!(
            unsafe { eindir_objective_grad(obj, x_t, g_short) },
            eindir_status_t::EINDIR_INVALID_PARAMETER
        );
        unsafe {
            del1d(short_t);
            del1d(x_t);
            del1d(g_short);
            rgpot_potential_free_eindir(pot);
        }
    }

    #[test]
    fn oversized_atom_count_is_refused() {
        let pot = unsafe {
            rgpot_potential_new_eindir(
                mock_energy_callback,
                std::ptr::null_mut(),
                None,
                usize::MAX,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert!(pot.is_null());
    }

    #[test]
    fn eval_counts_split_computed_from_served() {
        let atmnrs = [1i32, 1];
        let mut calls = 0usize;
        let pot = unsafe {
            rgpot_potential_new_eindir(
                counting_energy_callback,
                (&mut calls as *mut usize).cast(),
                None,
                2,
                atmnrs.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        let obj = pot as *mut eindir_objective_t;
        let mut counts = rgpot_eval_counts_t::default();
        assert_eq!(
            unsafe { rgpot_potential_eval_counts(pot, &mut counts) },
            rgpot_status_t::RGPOT_SUCCESS
        );
        assert_eq!(
            counts,
            rgpot_eval_counts_t {
                computed: 0,
                served: 0
            }
        );

        let mut x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_t = make_1d(x.as_mut_ptr(), 6);
        let mut g = [0.0f64; 6];
        let g_t = make_1d(g.as_mut_ptr(), 6);
        let mut value = 0.0f64;

        // Energy then gradient at the same point: one computed, one served.
        unsafe { eindir_objective_eval(obj, x_t, &mut value) };
        unsafe { eindir_objective_grad(obj, x_t, g_t) };
        unsafe { rgpot_potential_eval_counts(pot, &mut counts) };
        assert_eq!(
            counts,
            rgpot_eval_counts_t {
                computed: 1,
                served: 1
            }
        );
        assert_eq!(calls, 1);

        // The same geometry again, as a gradient, is served too.
        unsafe { eindir_objective_grad(obj, x_t, g_t) };
        unsafe { rgpot_potential_eval_counts(pot, &mut counts) };
        assert_eq!(
            counts,
            rgpot_eval_counts_t {
                computed: 1,
                served: 2
            }
        );
        assert_eq!(calls, 1);

        // A gradient at another geometry is computed.
        let mut x2 = [1.0f64, 2.0, 3.0, 4.0, 5.0, 7.0];
        let x2_t = make_1d(x2.as_mut_ptr(), 6);
        unsafe { eindir_objective_grad(obj, x2_t, g_t) };
        unsafe { del1d(x2_t) };
        unsafe { rgpot_potential_eval_counts(pot, &mut counts) };
        assert_eq!(
            counts,
            rgpot_eval_counts_t {
                computed: 2,
                served: 2
            }
        );
        assert_eq!(calls, 2);

        unsafe { rgpot_potential_reset_eval_counts(pot) };
        unsafe { rgpot_potential_eval_counts(pot, &mut counts) };
        assert_eq!(counts, rgpot_eval_counts_t::default());

        assert_eq!(
            unsafe { rgpot_potential_eval_counts(std::ptr::null(), &mut counts) },
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert_eq!(
            unsafe { rgpot_potential_eval_counts(pot, std::ptr::null_mut()) },
            rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        unsafe {
            del1d(x_t);
            del1d(g_t);
            rgpot_potential_free_eindir(pot);
        }
    }

    #[cfg(feature = "schema")]
    mod peer {
        use super::*;
        use crate::c_api::potential::rgpot_potential_calculate;
        use crate::Potentials_capnp::capabilities::{Builder as Caps, Operation};
        use capnp::message::Builder as Message;

        fn caps_bytes(edit: impl FnOnce(&mut Caps<'_>)) -> Vec<u8> {
            let mut msg = Message::new_default();
            {
                let mut caps = msg.init_root::<Caps<'_>>();
                crate::compat::fill_compatibility(caps.reborrow());
                let mut ops = caps.reborrow().init_operations(2);
                ops.set(0, Operation::Energy);
                ops.set(1, Operation::Forces);
                edit(&mut caps);
            }
            capnp::serialize::write_message_to_words(&msg)
        }

        struct Handle {
            pot: *mut rgpot_potential_t,
            calls: Box<usize>,
        }

        impl Handle {
            fn new() -> Self {
                let mut calls = Box::new(0usize);
                let atmnrs = [1i32, 1];
                let pot = unsafe {
                    rgpot_potential_new_eindir(
                        counting_energy_callback,
                        (&mut *calls as *mut usize).cast(),
                        None,
                        2,
                        atmnrs.as_ptr(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    )
                };
                Self { pot, calls }
            }

            fn install(&self, bytes: &[u8]) -> rgpot_status_t {
                unsafe {
                    rgpot_potential_set_peer_capabilities(self.pot, bytes.as_ptr(), bytes.len())
                }
            }

            fn eval(&self) -> eindir_status_t {
                let mut x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x_t = make_1d(x.as_mut_ptr(), 6);
                let mut value = 0.0f64;
                let s = unsafe {
                    eindir_objective_eval(self.pot as *mut eindir_objective_t, x_t, &mut value)
                };
                unsafe { del1d(x_t) };
                s
            }

            fn grad(&self) -> eindir_status_t {
                let mut x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x_t = make_1d(x.as_mut_ptr(), 6);
                let mut g = [0.0f64; 6];
                let g_t = make_1d(g.as_mut_ptr(), 6);
                let s =
                    unsafe { eindir_objective_grad(self.pot as *mut eindir_objective_t, x_t, g_t) };
                unsafe {
                    del1d(x_t);
                    del1d(g_t);
                }
                s
            }

            fn calculate(&self) -> rgpot_status_t {
                let mut pos = [0.0f64; 6];
                let mut atm = [1i32, 1];
                let mut box_ = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
                let input = rgpot_force_input_t {
                    positions: unsafe { rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), 2, 3) },
                    atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(atm.as_mut_ptr(), 2) },
                    box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(box_.as_mut_ptr()) },
                };
                let mut out = rgpot_force_out_t {
                    forces: std::ptr::null_mut(),
                    energy: 0.0,
                    variance: 0.0,
                };
                let s = unsafe { rgpot_potential_calculate(self.pot, &input, &mut out) };
                unsafe {
                    rgpot_tensor_free(out.forces);
                    rgpot_tensor_free(input.positions);
                    rgpot_tensor_free(input.atomic_numbers);
                    rgpot_tensor_free(input.box_matrix);
                }
                s
            }
        }

        impl Drop for Handle {
            fn drop(&mut self) {
                unsafe { rgpot_potential_free_eindir(self.pot) };
                let _ = &self.calls;
            }
        }

        fn last_error() -> String {
            unsafe { std::ffi::CStr::from_ptr(crate::status::rgpot_last_error()) }
                .to_string_lossy()
                .into_owned()
        }

        #[test]
        fn handle_without_a_message_keeps_the_callback_contract() {
            let h = Handle::new();
            assert_eq!(h.eval(), eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(h.calculate(), rgpot_status_t::RGPOT_SUCCESS);
        }

        #[test]
        fn compatible_peer_is_accepted() {
            let h = Handle::new();
            assert_eq!(
                h.install(&caps_bytes(|_| {})),
                rgpot_status_t::RGPOT_SUCCESS
            );
            assert_eq!(h.eval(), eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(h.grad(), eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(h.calculate(), rgpot_status_t::RGPOT_SUCCESS);
        }

        #[test]
        fn incompatible_peer_is_refused_before_the_callback_runs() {
            let h = Handle::new();
            let bytes = caps_bytes(|c| c.set_protocol_family("other.family"));
            assert_eq!(h.install(&bytes), rgpot_status_t::RGPOT_INVALID_PARAMETER);
            assert!(
                last_error().starts_with("peer refused: protocol family"),
                "{}",
                last_error()
            );

            assert_eq!(h.eval(), eindir_status_t::EINDIR_INVALID_PARAMETER);
            assert!(
                last_error().starts_with("peer refused:"),
                "{}",
                last_error()
            );
            assert_eq!(h.grad(), eindir_status_t::EINDIR_INVALID_PARAMETER);
            assert_eq!(h.calculate(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
            assert!(
                last_error().starts_with("peer refused:"),
                "{}",
                last_error()
            );
            assert_eq!(*h.calls, 0, "no callback may run for a refused peer");
        }

        #[test]
        fn a_compatible_message_lifts_an_earlier_refusal() {
            let h = Handle::new();
            h.install(&caps_bytes(|c| c.set_dlpack_major(2)));
            assert_eq!(h.eval(), eindir_status_t::EINDIR_INVALID_PARAMETER);
            assert_eq!(
                h.install(&caps_bytes(|_| {})),
                rgpot_status_t::RGPOT_SUCCESS
            );
            assert_eq!(h.eval(), eindir_status_t::EINDIR_SUCCESS);
        }

        #[test]
        fn invalid_arguments_are_rejected_and_leave_the_handle_usable() {
            let h = Handle::new();
            assert_eq!(
                h.install(&[0u8; 3]),
                rgpot_status_t::RGPOT_INVALID_PARAMETER
            );
            assert_eq!(
                unsafe {
                    rgpot_potential_set_peer_capabilities(std::ptr::null(), [0u8; 8].as_ptr(), 8)
                },
                rgpot_status_t::RGPOT_INVALID_PARAMETER
            );
            assert_eq!(
                unsafe { rgpot_potential_set_peer_capabilities(h.pot, std::ptr::null(), 8) },
                rgpot_status_t::RGPOT_INVALID_PARAMETER
            );
            assert_eq!(h.eval(), eindir_status_t::EINDIR_SUCCESS);
        }

        #[test]
        fn an_unreadable_message_is_refused() {
            let h = Handle::new();
            assert_eq!(
                h.install(&[0u8; 8]),
                rgpot_status_t::RGPOT_INVALID_PARAMETER
            );
            assert!(
                last_error().starts_with("peer refused: capabilities unreadable"),
                "{}",
                last_error()
            );
            assert_eq!(h.eval(), eindir_status_t::EINDIR_INVALID_PARAMETER);
        }

        #[test]
        fn a_message_that_serves_no_operation_is_refused() {
            let h = Handle::new();
            let bytes = caps_bytes(|c| {
                c.reborrow().init_operations(0);
            });
            assert_eq!(h.install(&bytes), rgpot_status_t::RGPOT_INVALID_PARAMETER);
            assert!(
                last_error().starts_with("peer refused: operation"),
                "{}",
                last_error()
            );
            assert_eq!(h.calculate(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
        }
    }

    #[test]
    fn eindir_abi_stamp_is_accepted_by_eindir_core() {
        let stamp = rgpot_eindir_abi_stamp();
        assert_eq!(
            stamp.objective_size,
            std::mem::size_of::<eindir_objective_t>()
        );
        assert_eq!(
            unsafe { eindir_core::ffi::eindir_core_abi_compatible(&stamp) },
            1
        );
    }

    #[test]
    fn eindir_abi_compatible_matches_eindir_core() {
        let stamp = rgpot_eindir_abi_stamp();
        assert_eq!(stamp.abi_major, 1);
        assert_eq!(stamp.objective_layout, 1);
        assert_eq!(unsafe { rgpot_eindir_abi_compatible(&stamp) }, 1);
        assert_eq!(unsafe { rgpot_eindir_abi_compatible(std::ptr::null()) }, 0);

        let edits: [(&str, fn(&mut eindir_abi_stamp_t)); 6] = [
            ("major", |s| s.abi_major += 1),
            ("minor", |s| s.abi_minor += 1),
            ("layout", |s| s.objective_layout += 1),
            ("size", |s| s.objective_size += 8),
            ("dlpack major", |s| s.dlpack_major += 1),
            ("features", |s| s.features |= 1 << 40),
        ];
        for (name, edit) in edits {
            let mut other = stamp;
            edit(&mut other);
            let ours = unsafe { rgpot_eindir_abi_compatible(&other) };
            let core = unsafe { eindir_core::ffi::eindir_core_abi_compatible(&other) };
            assert_eq!(ours, 0, "{name}");
            assert_eq!(ours, core, "{name}");
        }
    }

    /// Energy and forces that depend on every coordinate and on the cell
    /// diagonal, so a gradient answered for other positions or another cell
    /// has other bits.
    unsafe extern "C" fn cell_energy_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let calls = unsafe { &mut *(user_data as *mut usize) };
        *calls += 1;
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        let n = unsafe { inp.n_atoms() }.unwrap_or(0);
        let pt = unsafe { &(*inp.positions).dl_tensor };
        let pos = unsafe { std::slice::from_raw_parts(pt.data as *const f64, n * 3) };
        let bt = unsafe { &(*inp.box_matrix).dl_tensor };
        let cell = unsafe { std::slice::from_raw_parts(bt.data as *const f64, 9) };
        let mut forces = vec![0.0f64; n * 3];
        out.energy = 0.0;
        for (i, (&x, f)) in pos.iter().zip(forces.iter_mut()).enumerate() {
            let k = std::f64::consts::TAU / cell[(i % 3) * 4];
            out.energy += 0.5 * x * x + (k * x).sin() / k;
            *f = -(x + (k * x).cos());
        }
        out.variance = 0.0;
        out.forces = create_owned_f64_tensor(forces, vec![n as i64, 3]);
        rgpot_status_t::RGPOT_SUCCESS
    }

    fn cell_handle(edge: f64, calls: &mut usize) -> *mut rgpot_potential_t {
        handle_with(cell_energy_callback, edge, calls)
    }

    fn handle_with(
        callback: PotentialCallback,
        edge: f64,
        calls: &mut usize,
    ) -> *mut rgpot_potential_t {
        let atmnrs = [1i32, 1];
        let cell = [edge, 0.0, 0.0, 0.0, edge, 0.0, 0.0, 0.0, edge];
        let pot = unsafe {
            rgpot_potential_new_eindir(
                callback,
                (calls as *mut usize).cast(),
                None,
                2,
                atmnrs.as_ptr(),
                cell.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert!(!pot.is_null());
        pot
    }

    fn eval_at(pot: *mut rgpot_potential_t, x: &[f64]) -> f64 {
        let mut xv = x.to_vec();
        let x_t = make_1d(xv.as_mut_ptr(), xv.len());
        let mut value = 0.0f64;
        let status = unsafe { eindir_objective_eval(pot.cast(), x_t, &mut value) };
        unsafe { del1d(x_t) };
        assert_eq!(status, eindir_status_t::EINDIR_SUCCESS);
        value
    }

    fn grad_at(pot: *mut rgpot_potential_t, x: &[f64]) -> Vec<f64> {
        let mut xv = x.to_vec();
        let mut g = vec![0.0f64; x.len()];
        let x_t = make_1d(xv.as_mut_ptr(), xv.len());
        let g_t = make_1d(g.as_mut_ptr(), g.len());
        let status = unsafe { eindir_objective_grad(pot.cast(), x_t, g_t) };
        unsafe {
            del1d(x_t);
            del1d(g_t);
        }
        assert_eq!(status, eindir_status_t::EINDIR_SUCCESS);
        g
    }

    fn counts(pot: *mut rgpot_potential_t) -> rgpot_eval_counts_t {
        let mut c = rgpot_eval_counts_t::default();
        assert_eq!(
            unsafe { rgpot_potential_eval_counts(pot, &mut c) },
            rgpot_status_t::RGPOT_SUCCESS
        );
        c
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    const X: [f64; 6] = [0.3, -1.7, 2.9, 4.1, -0.05, 7.3];

    #[test]
    fn served_gradient_has_the_bits_of_a_separate_call() {
        let (mut fused_calls, mut split_calls) = (0usize, 0usize);
        let fused = cell_handle(10.0, &mut fused_calls);
        let split = cell_handle(10.0, &mut split_calls);

        let e_fused = eval_at(fused, &X);
        let g_fused = grad_at(fused, &X);
        assert_eq!(
            counts(fused),
            rgpot_eval_counts_t {
                computed: 1,
                served: 1
            }
        );

        let g_split = grad_at(split, &X);
        let e_split = eval_at(split, &X);
        assert_eq!(
            counts(split),
            rgpot_eval_counts_t {
                computed: 1,
                served: 1
            }
        );

        assert_eq!(e_fused.to_bits(), e_split.to_bits());
        assert_eq!(bits(&g_fused), bits(&g_split));

        let objective = unsafe { EindirObjectiveWrapper::new(&(*fused).base) };
        let (e_pair, g_pair) = objective.value_and_gradient(Array1::from(X.to_vec()).view());
        assert_eq!(e_pair.to_bits(), e_split.to_bits());
        assert_eq!(bits(g_pair.as_slice().unwrap()), bits(&g_split));
        assert_eq!(
            counts(fused),
            rgpot_eval_counts_t {
                computed: 1,
                served: 3
            }
        );
        assert_eq!((fused_calls, split_calls), (1, 1));

        unsafe {
            rgpot_potential_free_eindir(fused);
            rgpot_potential_free_eindir(split);
        }
    }

    #[test]
    fn a_moved_geometry_is_computed_not_served() {
        let (mut calls, mut ref_calls) = (0usize, 0usize);
        let pot = cell_handle(10.0, &mut calls);
        let reference = cell_handle(10.0, &mut ref_calls);
        let g_x = grad_at(reference, &X);

        let mut far = X;
        far[0] += 0.25;
        eval_at(pot, &X);
        let g_far = grad_at(pot, &far);
        assert_eq!(
            counts(pot),
            rgpot_eval_counts_t {
                computed: 2,
                served: 0
            }
        );
        assert_eq!(bits(&g_far), bits(&grad_at(reference, &far)));
        assert_ne!(bits(&g_far), bits(&g_x));

        let mut ulp = X;
        ulp[4] = f64::from_bits(ulp[4].to_bits() + 1);
        eval_at(pot, &X);
        let g_ulp = grad_at(pot, &ulp);
        assert_eq!(
            counts(pot),
            rgpot_eval_counts_t {
                computed: 4,
                served: 0
            }
        );
        assert_eq!(bits(&g_ulp), bits(&grad_at(reference, &ulp)));

        // The ulp geometry replaced the entry, so X is computed again.
        let g_again = grad_at(pot, &X);
        assert_eq!(
            counts(pot),
            rgpot_eval_counts_t {
                computed: 5,
                served: 0
            }
        );
        assert_eq!(bits(&g_again), bits(&g_x));

        unsafe {
            rgpot_potential_free_eindir(pot);
            rgpot_potential_free_eindir(reference);
        }
    }

    /// The cell is copied when the handle is built and has no setter, so
    /// another cell is another handle, with its own cache.
    #[test]
    fn handles_with_other_cells_never_share_a_result() {
        let mut calls = [0usize; 4];
        let [ca, cb, cra, crb] = &mut calls;
        let a = cell_handle(10.0, ca);
        let b = cell_handle(12.5, cb);
        let ref_a = cell_handle(10.0, cra);
        let ref_b = cell_handle(12.5, crb);

        let e_a = eval_at(a, &X);
        let e_b = eval_at(b, &X);
        let g_a = grad_at(a, &X);
        let g_b = grad_at(b, &X);
        assert_eq!(
            counts(a),
            rgpot_eval_counts_t {
                computed: 1,
                served: 1
            }
        );
        assert_eq!(
            counts(b),
            rgpot_eval_counts_t {
                computed: 1,
                served: 1
            }
        );

        assert_ne!(bits(&g_a), bits(&g_b));
        assert_eq!(bits(&g_a), bits(&grad_at(ref_a, &X)));
        assert_eq!(bits(&g_b), bits(&grad_at(ref_b, &X)));
        assert_eq!(e_a.to_bits(), eval_at(ref_a, &X).to_bits());
        assert_eq!(e_b.to_bits(), eval_at(ref_b, &X).to_bits());
        assert_eq!(
            counts(ref_a),
            rgpot_eval_counts_t {
                computed: 1,
                served: 1
            }
        );

        unsafe {
            for pot in [a, b, ref_a, ref_b] {
                rgpot_potential_free_eindir(pot);
            }
        }
    }

    /// Forces that read the sign bit of each coordinate, so `+0.0` and
    /// `-0.0` have other gradients although they compare equal.
    unsafe extern "C" fn sign_bit_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let calls = unsafe { &mut *(user_data as *mut usize) };
        *calls += 1;
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        let n = unsafe { inp.n_atoms() }.unwrap_or(0);
        let pt = unsafe { &(*inp.positions).dl_tensor };
        let pos = unsafe { std::slice::from_raw_parts(pt.data as *const f64, n * 3) };
        let forces: Vec<f64> = pos
            .iter()
            .map(|x| if x.is_sign_negative() { -1.0 } else { 1.0 })
            .collect();
        out.energy = pos.iter().sum();
        out.variance = 0.0;
        out.forces = create_owned_f64_tensor(forces, vec![n as i64, 3]);
        rgpot_status_t::RGPOT_SUCCESS
    }

    #[test]
    fn a_negative_zero_is_computed_not_served() {
        let (mut calls, mut ref_calls) = (0usize, 0usize);
        let pot = handle_with(sign_bit_callback, 10.0, &mut calls);
        let reference = handle_with(sign_bit_callback, 10.0, &mut ref_calls);

        let plus = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut minus = plus;
        minus[0] = -0.0;
        assert!(plus == minus);

        eval_at(pot, &plus);
        let g_minus = grad_at(pot, &minus);
        assert_eq!(
            counts(pot),
            rgpot_eval_counts_t {
                computed: 2,
                served: 0
            }
        );
        assert_eq!(g_minus[0], 1.0);
        assert_eq!(bits(&g_minus), bits(&grad_at(reference, &minus)));

        eval_at(pot, &minus);
        let g_same = grad_at(pot, &minus);
        assert_eq!(
            counts(pot),
            rgpot_eval_counts_t {
                computed: 2,
                served: 2
            }
        );
        assert_eq!(bits(&g_same), bits(&g_minus));

        unsafe {
            rgpot_potential_free_eindir(pot);
            rgpot_potential_free_eindir(reference);
        }
    }

    /// Behaviour of the fused cache against a counting engine.
    mod fused_cache {
        use super::*;
        use crate::hvp::HvpError;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const MESSAGE_SCC: &str = "SCC did not converge in 250 iterations";
        const MESSAGE_OTHER: &str = "engine lost its license";

        #[derive(Clone, Copy, PartialEq)]
        enum Mode {
            Succeed,
            SccFails,
            OtherFails,
            NonFiniteEnergy,
        }

        /// The engine behind the handle: it counts the calls it receives and
        /// fails as `mode` says.
        struct Engine {
            calls: AtomicUsize,
            mode: std::sync::Mutex<Mode>,
        }

        impl Engine {
            fn mode(&self) -> Mode {
                *self.mode.lock().unwrap()
            }
        }

        unsafe extern "C" fn engine_callback(
            user_data: *mut c_void,
            input: *const rgpot_force_input_t,
            output: *mut rgpot_force_out_t,
        ) -> rgpot_status_t {
            let engine = unsafe { &*(user_data as *const Engine) };
            engine.calls.fetch_add(1, Ordering::SeqCst);
            match engine.mode() {
                Mode::SccFails => {
                    set_last_error(MESSAGE_SCC);
                    return rgpot_status_t::RGPOT_SCC_NOT_CONVERGED;
                }
                Mode::OtherFails => {
                    set_last_error(MESSAGE_OTHER);
                    return rgpot_status_t::RGPOT_INTERNAL_ERROR;
                }
                Mode::Succeed | Mode::NonFiniteEnergy => {}
            }
            let status = unsafe { mock_energy_callback(std::ptr::null_mut(), input, output) };
            if engine.mode() == Mode::NonFiniteEnergy {
                unsafe { (*output).energy = f64::NAN };
            }
            status
        }

        struct Rig {
            pot: *mut rgpot_potential_t,
            engine: Box<Engine>,
        }

        impl Rig {
            fn new(mode: Mode) -> Self {
                let engine = Box::new(Engine {
                    calls: AtomicUsize::new(0),
                    mode: std::sync::Mutex::new(mode),
                });
                let atmnrs = [1i32, 1];
                let cell = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
                let pot = unsafe {
                    rgpot_potential_new_eindir(
                        engine_callback,
                        (&*engine as *const Engine as *mut Engine).cast(),
                        None,
                        2,
                        atmnrs.as_ptr(),
                        cell.as_ptr(),
                        std::ptr::null(),
                        std::ptr::null(),
                    )
                };
                assert!(!pot.is_null());
                Self { pot, engine }
            }

            fn set_mode(&self, mode: Mode) {
                *self.engine.mode.lock().unwrap() = mode;
            }

            fn calls(&self) -> usize {
                self.engine.calls.load(Ordering::SeqCst)
            }

            /// The status of the base energy function, and the energy.
            fn energy(&self, x: &[f64]) -> (eindir_status_t, f64) {
                let mut xv = x.to_vec();
                let x_t = make_1d(xv.as_mut_ptr(), xv.len());
                let mut value = f64::NAN;
                let base = unsafe { &(*self.pot).base };
                let status = unsafe { (base.eval_fn)(base.user_data, x_t, &mut value) };
                unsafe { del1d(x_t) };
                (status, value)
            }

            /// The status of the base gradient function itself, not of a
            /// wrapper that turns a failure into NaN, and the gradient.
            fn gradient(&self, x: &[f64]) -> (eindir_status_t, Vec<f64>) {
                let mut xv = x.to_vec();
                let mut g = vec![f64::NAN; x.len()];
                let x_t = make_1d(xv.as_mut_ptr(), xv.len());
                let g_t = make_1d(g.as_mut_ptr(), g.len());
                let base = unsafe { &(*self.pot).base };
                let grad_fn = base.grad_fn.expect("a potential handle has a gradient");
                let status = unsafe { grad_fn(base.user_data, x_t, g_t) };
                unsafe {
                    del1d(x_t);
                    del1d(g_t);
                }
                (status, g)
            }

            fn counts(&self) -> (u64, u64) {
                let c = counts(self.pot);
                (c.computed, c.served)
            }

            fn kind(&self) -> rgpot_fused_error_kind_t {
                unsafe { rgpot_potential_fused_error_kind(self.pot) }
            }

            fn set_settings(&self, edit: impl FnOnce(FusedSettings) -> FusedSettings) {
                let mut settings = rgpot_fused_settings_t::default();
                let current = self.fused_settings();
                let edited = edit(current);
                settings.periodic = edited
                    .periodic
                    .iter()
                    .enumerate()
                    .fold(0u32, |bits, (i, on)| bits | (u32::from(*on) << i));
                settings.charge = edited.charge;
                settings.spin = edited.spin;
                settings.accuracy = edited.accuracy;
                settings.electronic_temperature = edited.electronic_temperature;
                settings.scc_max_iterations = edited.scc_max_iterations;
                let mixer = std::ffi::CString::new(edited.scc_mixer).unwrap();
                let build = std::ffi::CString::new(edited.engine_build).unwrap();
                settings.scc_mixer = mixer.as_ptr();
                settings.engine_build = build.as_ptr();
                let status = unsafe { rgpot_potential_set_fused_settings(self.pot, &settings) };
                assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
            }

            fn fused_settings(&self) -> FusedSettings {
                unsafe { &*self.pot }.fused().settings()
            }
        }

        impl Drop for Rig {
            fn drop(&mut self) {
                unsafe { rgpot_potential_free_eindir(self.pot) };
                let _ = &self.engine;
            }
        }

        fn last_error() -> String {
            last_error_message()
        }

        #[test]
        fn a_failed_energy_then_gradient_makes_one_engine_call() {
            for (mode, message, status, kind) in [
                (
                    Mode::OtherFails,
                    MESSAGE_OTHER,
                    eindir_status_t::EINDIR_INTERNAL_ERROR,
                    rgpot_fused_error_kind_t::RGPOT_FUSED_ENGINE_FAILED,
                ),
                (
                    Mode::SccFails,
                    MESSAGE_SCC,
                    eindir_status_t::EINDIR_INTERNAL_ERROR,
                    rgpot_fused_error_kind_t::RGPOT_FUSED_SCC_NOT_CONVERGED,
                ),
                (
                    Mode::NonFiniteEnergy,
                    "callback energy is not finite",
                    eindir_status_t::EINDIR_INVALID_PARAMETER,
                    rgpot_fused_error_kind_t::RGPOT_FUSED_INVALID_RESULT,
                ),
            ] {
                let rig = Rig::new(mode);
                let (first_status, _) = rig.energy(&X);
                assert_eq!(first_status, status);
                assert_eq!(last_error(), message);
                assert_eq!(rig.calls(), 1);

                set_last_error("stale message from elsewhere");
                let (second_status, gradient) = rig.gradient(&X);
                assert_eq!(rig.calls(), 1, "the gradient must not run the engine again");
                assert_eq!(second_status, first_status, "the stored failure status");
                assert_ne!(second_status, eindir_status_t::EINDIR_SUCCESS);
                assert_eq!(last_error(), message, "the stored failure message");
                assert!(
                    gradient.iter().all(|g| g.is_nan()),
                    "no gradient was written"
                );
                assert_eq!(rig.kind(), kind);
                assert_eq!(rig.counts(), (1, 1));
            }
        }

        #[test]
        fn a_failed_gradient_then_energy_makes_one_engine_call() {
            let rig = Rig::new(Mode::SccFails);
            let (first_status, _) = rig.gradient(&X);
            assert_eq!(first_status, eindir_status_t::EINDIR_INTERNAL_ERROR);
            set_last_error("stale message from elsewhere");
            let (second_status, value) = rig.energy(&X);
            assert_eq!(rig.calls(), 1);
            assert_eq!(second_status, first_status);
            assert_eq!(last_error(), MESSAGE_SCC);
            assert!(value.is_nan());
        }

        #[test]
        fn a_successful_energy_then_gradient_makes_one_engine_call() {
            let rig = Rig::new(Mode::Succeed);
            let (status, energy) = rig.energy(&X);
            assert_eq!(status, eindir_status_t::EINDIR_SUCCESS);
            let (status, gradient) = rig.gradient(&X);
            assert_eq!(status, eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(rig.calls(), 1);
            assert_eq!(gradient, vec![1.0; 6]);

            let (_, energy_again) = rig.energy(&X);
            let (_, gradient_again) = rig.gradient(&X);
            assert_eq!(rig.calls(), 1, "the same bits are answered from the entry");
            assert_eq!(energy.to_bits(), energy_again.to_bits());
            assert_eq!(gradient, gradient_again);
            assert_eq!(rig.kind(), rgpot_fused_error_kind_t::RGPOT_FUSED_OK);
            assert_eq!(rig.counts(), (1, 3));
        }

        #[test]
        fn a_gradient_then_energy_makes_one_engine_call() {
            let rig = Rig::new(Mode::Succeed);
            assert_eq!(rig.gradient(&X).0, eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(rig.energy(&X).0, eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(rig.calls(), 1);
        }

        #[test]
        fn one_ulp_of_any_position_is_a_new_engine_call() {
            let rig = Rig::new(Mode::Succeed);
            rig.energy(&X);
            let mut calls = 1;
            for i in 0..X.len() {
                let mut moved = X;
                moved[i] = f64::from_bits(moved[i].to_bits() + 1);
                assert_ne!(moved[i], X[i]);
                rig.energy(&moved);
                calls += 1;
                assert_eq!(rig.calls(), calls, "coordinate {i}");
                rig.gradient(&moved);
                assert_eq!(rig.calls(), calls, "the moved geometry is stored");
                // The entry now holds the moved geometry, so X is new again.
                rig.energy(&X);
                calls += 1;
                assert_eq!(rig.calls(), calls, "coordinate {i} back to X");
            }
        }

        #[test]
        fn any_one_changed_setting_is_a_new_engine_call() {
            type Edit = fn(FusedSettings) -> FusedSettings;
            let edits: [(&str, Edit); 8] = [
                ("periodic", |s| s.with_periodic([true, false, false])),
                ("charge", |s| s.with_charge(-1.0)),
                ("spin", |s| s.with_spin(2)),
                ("accuracy", |s| s.with_accuracy(0.5)),
                ("electronic temperature", |s| {
                    s.with_electronic_temperature(900.0)
                }),
                ("scc max iterations", |s| s.with_scc_max_iterations(500)),
                ("scc mixer", |s| s.with_scc_mixer("broyden")),
                ("engine build", |s| s.with_engine_build("xtb 6.7.1")),
            ];
            for (name, edit) in edits {
                let rig = Rig::new(Mode::Succeed);
                rig.energy(&X);
                rig.set_settings(|s| s);
                rig.gradient(&X);
                assert_eq!(rig.calls(), 1, "{name}: unchanged settings keep the entry");

                rig.set_settings(edit);
                rig.energy(&X);
                assert_eq!(rig.calls(), 2, "{name}: the changed setting is a new key");
                rig.gradient(&X);
                assert_eq!(rig.calls(), 2, "{name}: the new key is stored");
            }
        }

        #[test]
        fn a_plain_retry_cannot_rerun_a_failed_geometry() {
            let rig = Rig::new(Mode::SccFails);
            let (first, _) = rig.energy(&X);
            for _ in 0..3 {
                assert_eq!(rig.energy(&X).0, first);
                assert_eq!(rig.gradient(&X).0, first);
            }
            // The engine would succeed now, but a plain retry never asks.
            rig.set_mode(Mode::Succeed);
            assert_eq!(rig.energy(&X).0, first);
            assert_eq!(rig.calls(), 1);

            rig.set_settings(|s| s.with_electronic_temperature(1000.0));
            assert_eq!(rig.energy(&X).0, eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(rig.calls(), 2);
            assert_eq!(rig.gradient(&X).0, eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(rig.calls(), 2);
            assert_eq!(rig.kind(), rgpot_fused_error_kind_t::RGPOT_FUSED_OK);
        }

        #[test]
        fn scc_non_convergence_is_told_apart_from_other_failures() {
            let scc = Rig::new(Mode::SccFails);
            let other = Rig::new(Mode::OtherFails);
            scc.energy(&X);
            other.energy(&X);
            assert_eq!(
                scc.kind(),
                rgpot_fused_error_kind_t::RGPOT_FUSED_SCC_NOT_CONVERGED
            );
            assert_eq!(
                other.kind(),
                rgpot_fused_error_kind_t::RGPOT_FUSED_ENGINE_FAILED
            );

            let scc_error = unsafe { &*scc.pot }.fused().stored_error().unwrap();
            let other_error = unsafe { &*other.pot }.fused().stored_error().unwrap();
            assert_ne!(scc_error, other_error);
            assert_eq!(scc_error.status(), rgpot_status_t::RGPOT_SCC_NOT_CONVERGED);
            assert_eq!(other_error.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
            assert_eq!(scc_error.message(), MESSAGE_SCC);
            assert_eq!(other_error.message(), MESSAGE_OTHER);
        }

        #[test]
        fn both_requests_after_a_failure_return_the_same_error_value() {
            let rig = Rig::new(Mode::SccFails);
            rig.energy(&X);
            let first = unsafe { &*rig.pot }.fused().stored_error().unwrap();
            rig.gradient(&X);
            let second = unsafe { &*rig.pot }.fused().stored_error().unwrap();
            assert_eq!(first, second);
            let key = unsafe { &*rig.pot }.fused().key(
                &X,
                Some(&[1, 1]),
                Some(&[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]),
            );
            let (stored, source) = unsafe { &*rig.pot }
                .fused()
                .evaluate(key, || panic!("the engine must not run"));
            assert_eq!(source, crate::fused::FusedSource::Stored);
            assert_eq!(stored.unwrap_err(), first);
        }

        #[test]
        fn the_hessian_centre_shares_the_entry_and_its_failure() {
            let rig = Rig::new(Mode::SccFails);
            let mut pos = X;
            let mut atm = [1i32, 1];
            let mut cell = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
            let input = rgpot_force_input_t {
                positions: unsafe { rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), 2, 3) },
                atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(atm.as_mut_ptr(), 2) },
                box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
            };
            let pot = unsafe { &*rig.pot };
            let first = unsafe { pot.hvp_centre(&input) }.unwrap_err();
            assert_eq!(
                first,
                HvpError::Callback(rgpot_status_t::RGPOT_SCC_NOT_CONVERGED)
            );
            assert_eq!(unsafe { pot.hvp_centre(&input) }.unwrap_err(), first);
            assert_eq!(rig.energy(&X).0, eindir_status_t::EINDIR_INTERNAL_ERROR);
            assert_eq!(rig.gradient(&X).0, eindir_status_t::EINDIR_INTERNAL_ERROR);
            assert_eq!(
                rig.calls(),
                1,
                "centre, energy and gradient are one geometry"
            );
            unsafe {
                rgpot_tensor_free(input.positions);
                rgpot_tensor_free(input.atomic_numbers);
                rgpot_tensor_free(input.box_matrix);
            }
        }

        #[test]
        fn settings_with_a_short_struct_or_a_bad_string_are_refused() {
            let rig = Rig::new(Mode::Succeed);
            let settings = rgpot_fused_settings_t {
                struct_size: 8,
                ..rgpot_fused_settings_t::default()
            };
            assert_eq!(
                unsafe { rgpot_potential_set_fused_settings(rig.pot, &settings) },
                rgpot_status_t::RGPOT_INVALID_PARAMETER
            );
            let bad = [0xffu8, 0x00];
            let settings = rgpot_fused_settings_t::default().with_scc_mixer(bad.as_ptr().cast());
            assert_eq!(
                unsafe { rgpot_potential_set_fused_settings(rig.pot, &settings) },
                rgpot_status_t::RGPOT_INVALID_PARAMETER
            );
            assert_eq!(
                unsafe { rgpot_potential_set_fused_settings(std::ptr::null(), &settings) },
                rgpot_status_t::RGPOT_INVALID_PARAMETER
            );
            assert_eq!(
                unsafe { rgpot_potential_fused_error_kind(std::ptr::null()) },
                rgpot_fused_error_kind_t::RGPOT_FUSED_OK
            );
        }
    }
}
