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

use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use dlpk::sys::DLManagedTensorVersioned;

use crate::hvp::HvpState;
use crate::potential::PotentialCallback;
use crate::status::{rgpot_status_t, set_last_error};
use crate::tensor::{
    checked_f64_data, copy_f64_tensor,
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3,
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
    eindir_abi_stamp_t, eindir_objective_eval, eindir_objective_grad,
    eindir_objective_has_grad, eindir_objective_t, EindirEvalFn, EindirFreeFn, EindirGradFn,
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
    /// Energy/gradient result shared by a matching eval-then-grad request.
    /// Held behind a pointer so the C view of the struct stays plain C.
    fused_cache: Box<rgpot_fused_cache_t>,
}

impl rgpot_potential_t {
    /// The recorded peer refusal, if capabilities were installed and refused.
    pub(crate) fn fused_cache_peer_refusal(&self) -> Option<String> {
        self.fused_cache.peer_refusal()
    }

    /// Hessian-vector state: the analytic callback and the cached centre.
    pub(crate) fn hvp_state(&self) -> &HvpState {
        &self.fused_cache.hvp
    }
}

/// Opaque cache of the last fused energy and gradient evaluation.
pub struct rgpot_fused_cache_t {
    last: Mutex<Option<FusedEvaluation>>,
    computed: AtomicU64,
    served: AtomicU64,
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
            last: Mutex::new(None),
            computed: AtomicU64::new(0),
            served: AtomicU64::new(0),
            peer_refusal: Mutex::new(None),
            hvp: HvpState::default(),
        }
    }

    /// The recorded refusal, if any.
    pub(crate) fn peer_refusal(&self) -> Option<String> {
        self.peer_refusal
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn set_peer_refusal(&self, why: Option<String>) {
        *self
            .peer_refusal
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = why;
        *self
            .last
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        self.hvp.invalidate();
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
    let verdict = crate::compat::check_capabilities_words(&words, &crate::compat::Expectation::default());
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
/// `computed` counts callback invocations; `served` counts gradients
/// answered from the fused energy-and-gradient result without invoking the
/// callback. A caller that charges evaluations chooses which field to bill.
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
    let cache = unsafe { &*(*pot).fused_cache };
    unsafe {
        *out = rgpot_eval_counts_t {
            computed: cache.computed.load(Ordering::Relaxed),
            served: cache.served.load(Ordering::Relaxed),
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
    let cache = unsafe { &*(*pot).fused_cache };
    cache.computed.store(0, Ordering::Relaxed);
    cache.served.store(0, Ordering::Relaxed);
}

struct FusedEvaluation {
    /// `f64::to_bits` of each coordinate. Float equality would let `-0.0`
    /// be served the result computed at `+0.0`.
    positions: Vec<u64>,
    gradient: Vec<f64>,
}

fn position_bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
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
    let mut pos = x_data.clone();
    let mut atmnrs =
        unsafe { std::slice::from_raw_parts(pot.atomic_numbers, pot.n_atoms) }.to_vec();
    let mut box_ = pot.box_matrix;
    let input = rgpot_force_input_t {
        positions: unsafe {
            rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), pot.n_atoms as i64, 3)
        },
        atomic_numbers: unsafe {
            rgpot_tensor_cpu_i32_1d(atmnrs.as_mut_ptr(), pot.n_atoms as i64)
        },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(box_.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    pot.fused_cache.computed.fetch_add(1, Ordering::Relaxed);
    let status =
        unsafe { (pot.callback)(pot.pot_user_data, &input, &mut output) };
    let forces = if status == rgpot_status_t::RGPOT_SUCCESS {
        validated_result(&output, n)
    } else {
        Err(String::new())
    };
    let fused = forces.as_ref().ok().map(|forces| FusedEvaluation {
        positions: position_bits(&x_data),
        gradient: forces.iter().map(|force| -*force).collect(),
    });
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    if status != rgpot_status_t::RGPOT_SUCCESS {
        *pot.fused_cache
            .last
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        return eindir_status_t::EINDIR_INTERNAL_ERROR;
    }
    if let Err(msg) = forces {
        *pot.fused_cache
            .last
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        return reject(&msg);
    }
    *pot.fused_cache
        .last
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = fused;
    unsafe { *value_out = output.energy };
    eindir_status_t::EINDIR_SUCCESS
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
    let cached = pot
        .fused_cache
        .last
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    if let Some(cached) = cached {
        if cached.positions == position_bits(&x_data) {
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, n) };
            dst.copy_from_slice(&cached.gradient);
            pot.fused_cache.served.fetch_add(1, Ordering::Relaxed);
            return eindir_status_t::EINDIR_SUCCESS;
        }
    }
    let mut pos = x_data.to_vec();
    let mut atmnrs =
        unsafe { std::slice::from_raw_parts(pot.atomic_numbers, pot.n_atoms) }.to_vec();
    let mut box_ = pot.box_matrix;
    let input = rgpot_force_input_t {
        positions: unsafe {
            rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), pot.n_atoms as i64, 3)
        },
        atomic_numbers: unsafe {
            rgpot_tensor_cpu_i32_1d(atmnrs.as_mut_ptr(), pot.n_atoms as i64)
        },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(box_.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    pot.fused_cache.computed.fetch_add(1, Ordering::Relaxed);
    let status =
        unsafe { (pot.callback)(pot.pot_user_data, &input, &mut output) };
    let forces = if status == rgpot_status_t::RGPOT_SUCCESS {
        validated_result(&output, n)
    } else {
        Err(String::new())
    };
    if let Ok(src) = &forces {
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, n) };
        for (d, s) in dst.iter_mut().zip(src) {
            *d = -*s; // gradient = -force
        }
    }
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    if status != rgpot_status_t::RGPOT_SUCCESS {
        return eindir_status_t::EINDIR_INTERNAL_ERROR;
    }
    if let Err(msg) = forces {
        return reject(&msg);
    }
    eindir_status_t::EINDIR_SUCCESS
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
            Bad::WrongLength => create_owned_f64_tensor(vec![0.0; n * 3 - 1], vec![(n * 3 - 1) as i64]),
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
            assert!(g_data.iter().all(|&v| v == 7.0), "gradient must stay untouched");

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
        assert_eq!(counts, rgpot_eval_counts_t { computed: 0, served: 0 });

        let mut x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_t = make_1d(x.as_mut_ptr(), 6);
        let mut g = [0.0f64; 6];
        let g_t = make_1d(g.as_mut_ptr(), 6);
        let mut value = 0.0f64;

        // Energy then gradient at the same point: one computed, one served.
        unsafe { eindir_objective_eval(obj, x_t, &mut value) };
        unsafe { eindir_objective_grad(obj, x_t, g_t) };
        unsafe { rgpot_potential_eval_counts(pot, &mut counts) };
        assert_eq!(counts, rgpot_eval_counts_t { computed: 1, served: 1 });
        assert_eq!(calls, 1);

        // A gradient with no preceding energy is computed.
        unsafe { eindir_objective_grad(obj, x_t, g_t) };
        unsafe { rgpot_potential_eval_counts(pot, &mut counts) };
        assert_eq!(counts, rgpot_eval_counts_t { computed: 2, served: 1 });
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
                unsafe { rgpot_potential_set_peer_capabilities(self.pot, bytes.as_ptr(), bytes.len()) }
            }

            fn eval(&self) -> eindir_status_t {
                let mut x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x_t = make_1d(x.as_mut_ptr(), 6);
                let mut value = 0.0f64;
                let s = unsafe { eindir_objective_eval(self.pot as *mut eindir_objective_t, x_t, &mut value) };
                unsafe { del1d(x_t) };
                s
            }

            fn grad(&self) -> eindir_status_t {
                let mut x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
                let x_t = make_1d(x.as_mut_ptr(), 6);
                let mut g = [0.0f64; 6];
                let g_t = make_1d(g.as_mut_ptr(), 6);
                let s = unsafe { eindir_objective_grad(self.pot as *mut eindir_objective_t, x_t, g_t) };
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
            assert_eq!(h.install(&caps_bytes(|_| {})), rgpot_status_t::RGPOT_SUCCESS);
            assert_eq!(h.eval(), eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(h.grad(), eindir_status_t::EINDIR_SUCCESS);
            assert_eq!(h.calculate(), rgpot_status_t::RGPOT_SUCCESS);
        }

        #[test]
        fn incompatible_peer_is_refused_before_the_callback_runs() {
            let h = Handle::new();
            let bytes = caps_bytes(|c| c.set_protocol_family("other.family"));
            assert_eq!(h.install(&bytes), rgpot_status_t::RGPOT_INVALID_PARAMETER);
            assert!(last_error().starts_with("peer refused: protocol family"), "{}", last_error());

            assert_eq!(h.eval(), eindir_status_t::EINDIR_INVALID_PARAMETER);
            assert!(last_error().starts_with("peer refused:"), "{}", last_error());
            assert_eq!(h.grad(), eindir_status_t::EINDIR_INVALID_PARAMETER);
            assert_eq!(h.calculate(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
            assert!(last_error().starts_with("peer refused:"), "{}", last_error());
            assert_eq!(*h.calls, 0, "no callback may run for a refused peer");
        }

        #[test]
        fn a_compatible_message_lifts_an_earlier_refusal() {
            let h = Handle::new();
            h.install(&caps_bytes(|c| c.set_dlpack_major(2)));
            assert_eq!(h.eval(), eindir_status_t::EINDIR_INVALID_PARAMETER);
            assert_eq!(h.install(&caps_bytes(|_| {})), rgpot_status_t::RGPOT_SUCCESS);
            assert_eq!(h.eval(), eindir_status_t::EINDIR_SUCCESS);
        }

        #[test]
        fn invalid_arguments_are_rejected_and_leave_the_handle_usable() {
            let h = Handle::new();
            assert_eq!(h.install(&[0u8; 3]), rgpot_status_t::RGPOT_INVALID_PARAMETER);
            assert_eq!(
                unsafe { rgpot_potential_set_peer_capabilities(std::ptr::null(), [0u8; 8].as_ptr(), 8) },
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
            assert_eq!(h.install(&[0u8; 8]), rgpot_status_t::RGPOT_INVALID_PARAMETER);
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
            assert!(last_error().starts_with("peer refused: operation"), "{}", last_error());
            assert_eq!(h.calculate(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
        }
    }

    #[test]
    fn eindir_abi_stamp_is_accepted_by_eindir_core() {
        let stamp = rgpot_eindir_abi_stamp();
        assert_eq!(stamp.objective_size, std::mem::size_of::<eindir_objective_t>());
        assert_eq!(unsafe { eindir_core::ffi::eindir_core_abi_compatible(&stamp) }, 1);
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
        assert_eq!(counts(fused), rgpot_eval_counts_t { computed: 1, served: 1 });

        let g_split = grad_at(split, &X);
        let e_split = eval_at(split, &X);
        assert_eq!(counts(split), rgpot_eval_counts_t { computed: 2, served: 0 });

        assert_eq!(e_fused.to_bits(), e_split.to_bits());
        assert_eq!(bits(&g_fused), bits(&g_split));

        let objective = unsafe { EindirObjectiveWrapper::new(&(*fused).base) };
        let (e_pair, g_pair) = objective.value_and_gradient(Array1::from(X.to_vec()).view());
        assert_eq!(e_pair.to_bits(), e_split.to_bits());
        assert_eq!(bits(g_pair.as_slice().unwrap()), bits(&g_split));
        assert_eq!(counts(fused), rgpot_eval_counts_t { computed: 2, served: 2 });
        assert_eq!((fused_calls, split_calls), (2, 2));

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
        assert_eq!(counts(pot), rgpot_eval_counts_t { computed: 2, served: 0 });
        assert_eq!(bits(&g_far), bits(&grad_at(reference, &far)));
        assert_ne!(bits(&g_far), bits(&g_x));

        let mut ulp = X;
        ulp[4] = f64::from_bits(ulp[4].to_bits() + 1);
        eval_at(pot, &X);
        let g_ulp = grad_at(pot, &ulp);
        assert_eq!(counts(pot), rgpot_eval_counts_t { computed: 4, served: 0 });
        assert_eq!(bits(&g_ulp), bits(&grad_at(reference, &ulp)));

        // A mismatched gradient drops the stored result, so the energy at X
        // no longer answers a later gradient at X.
        let g_again = grad_at(pot, &X);
        assert_eq!(counts(pot), rgpot_eval_counts_t { computed: 5, served: 0 });
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
        assert_eq!(counts(a), rgpot_eval_counts_t { computed: 1, served: 1 });
        assert_eq!(counts(b), rgpot_eval_counts_t { computed: 1, served: 1 });

        assert_ne!(bits(&g_a), bits(&g_b));
        assert_eq!(bits(&g_a), bits(&grad_at(ref_a, &X)));
        assert_eq!(bits(&g_b), bits(&grad_at(ref_b, &X)));
        assert_eq!(e_a.to_bits(), eval_at(ref_a, &X).to_bits());
        assert_eq!(e_b.to_bits(), eval_at(ref_b, &X).to_bits());
        assert_eq!(counts(ref_a), rgpot_eval_counts_t { computed: 2, served: 0 });

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
        assert_eq!(counts(pot), rgpot_eval_counts_t { computed: 2, served: 0 });
        assert_eq!(g_minus[0], 1.0);
        assert_eq!(bits(&g_minus), bits(&grad_at(reference, &minus)));

        eval_at(pot, &minus);
        let g_same = grad_at(pot, &minus);
        assert_eq!(counts(pot), rgpot_eval_counts_t { computed: 3, served: 1 });
        assert_eq!(bits(&g_same), bits(&g_minus));

        unsafe {
            rgpot_potential_free_eindir(pot);
            rgpot_potential_free_eindir(reference);
        }
    }
}
