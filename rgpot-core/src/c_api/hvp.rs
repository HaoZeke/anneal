// MIT License
// Copyright 2023--present rgpot developers

//! C API for Hessian-vector products of a potential handle.
//!
//! ```c
//! // A Lanczos loop at one centre: k products, k + 1 force calls.
//! rgpot_force_out_t centre = rgpot_force_out_create();
//! rgpot_potential_hvp_centre(pot, &input, &centre);  // 1 force call
//! for (int i = 0; i < k; ++i) {
//!   rgpot_potential_hvp(pot, &input, q[i], 0.0, RGPOT_HVP_AUTO, hq);  // 1 each
//!   // ... Lanczos recurrence on hq ...
//! }
//! rgpot_tensor_free(centre.forces);
//! ```
//!
//! See [`crate::hvp`] for the schemes and the centre cache.

use std::os::raw::c_void;

use crate::eindir::rgpot_potential_t;
use crate::hvp::{rgpot_hvp_counts_t, rgpot_hvp_scheme_t, HessianVectorCallback, HvpError};
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::tensor::create_owned_f64_tensor;
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

fn fail(err: HvpError) -> rgpot_status_t {
    if let HvpError::Invalid(why) = &err {
        set_last_error(why);
    }
    err.status()
}

fn refuse(msg: &str) -> rgpot_status_t {
    set_last_error(msg);
    rgpot_status_t::RGPOT_INVALID_PARAMETER
}

/// Register `callback` as the analytic Hessian-vector product of `pot`, or
/// clear it with NULL.
///
/// The callback receives the `user_data` the handle was created with, so a
/// potential object supplies its forces and its Hessian through one
/// pointer. With a callback registered, `RGPOT_HVP_AUTO` products call it
/// and make no force call.
///
/// # Safety
///
/// `pot` must be NULL or a live handle. A NULL `pot` returns
/// `RGPOT_INVALID_PARAMETER`. `callback` must be safe to call with the
/// handle's `user_data` for as long as it stays registered.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_hvp_callback(
    pot: *const rgpot_potential_t,
    callback: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            input: *const rgpot_force_input_t,
            direction: *const f64,
            hv_out: *mut f64,
        ) -> rgpot_status_t,
    >,
) -> rgpot_status_t {
    let callback: Option<HessianVectorCallback> = callback;
    if pot.is_null() {
        return refuse("rgpot_potential_set_hvp_callback: pot is NULL");
    }
    unsafe { &*pot }.set_hessian_vector_callback(callback);
    rgpot_status_t::RGPOT_SUCCESS
}

/// Write the Hessian-vector product `H direction` at the geometry of
/// `input` to `hv_out`.
///
/// `H` is the Hessian of the energy with respect to the `3 * n_atoms`
/// positions. `direction` and `hv_out` hold `3 * n_atoms` doubles laid out
/// like the positions and must not overlap. `scheme` is a
/// `rgpot_hvp_scheme_t` value:
///
/// - `RGPOT_HVP_AUTO`: the analytic callback when one is registered,
///   otherwise `RGPOT_HVP_FORWARD`.
/// - `RGPOT_HVP_FORWARD`: `(F(x) - F(x + h u)) |v| / h` with `u = v / |v|`.
///   `F(x)` comes from the handle's cached centre when the positions,
///   atomic numbers and cell carry the same bits as the last centre, so
///   `k` products at one geometry make `k + 1` force calls. Error `O(h)`.
/// - `RGPOT_HVP_CENTRAL`: `(F(x - h u) - F(x + h u)) |v| / (2 h)`, two
///   force calls, error `O(h^2)`.
/// - `RGPOT_HVP_ANALYTIC`: the registered callback; `RGPOT_INVALID_PARAMETER`
///   when none is registered.
///
/// `step` is `h`, a length in the units of the positions; 0 selects
/// `RGPOT_HVP_DEFAULT_STEP`. A zero direction writes zeros without a call.
/// A failing force callback's status is returned with its own
/// `rgpot_last_error` message.
///
/// # Safety
///
/// NULL pointers, an unknown scheme, a non-positive step and overlapping
/// buffers return `RGPOT_INVALID_PARAMETER`. Otherwise `pot` must be a live
/// handle, `input` must point to a valid `rgpot_force_input_t` whose tensors
/// stay alive for the call, and `direction` and `hv_out` must point to
/// `3 * n_atoms` readable and writable doubles.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_hvp(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    step: f64,
    scheme: i32,
    hv_out: *mut f64,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() {
            return refuse("rgpot_potential_hvp: pot is NULL");
        }
        if input.is_null() {
            return refuse("rgpot_potential_hvp: input is NULL");
        }
        if direction.is_null() || hv_out.is_null() {
            return refuse("rgpot_potential_hvp: direction or hv_out is NULL");
        }
        let Some(scheme) = rgpot_hvp_scheme_t::from_raw(scheme) else {
            return refuse(&format!("rgpot_potential_hvp: unknown scheme {scheme}"));
        };
        let input = unsafe { &*input };
        let Some(dim) = unsafe { input.n_atoms() }.and_then(|n| n.checked_mul(3)) else {
            return refuse("rgpot_potential_hvp: positions tensor is NULL or too large");
        };
        let bytes = dim * std::mem::size_of::<f64>();
        let (d, o) = (direction as usize, hv_out as usize);
        if dim > 0 && d < o.saturating_add(bytes) && o < d.saturating_add(bytes) {
            return refuse("rgpot_potential_hvp: direction and hv_out overlap");
        }
        let direction = unsafe { std::slice::from_raw_parts(direction, dim) };
        let hv_out = unsafe { std::slice::from_raw_parts_mut(hv_out, dim) };
        match unsafe { (*pot).hessian_vector(input, direction, step, scheme, hv_out) } {
            Ok(()) => rgpot_status_t::RGPOT_SUCCESS,
            Err(err) => fail(err),
        }
    }))
}

/// Energy and forces at the geometry of `input`, from the cached centre of
/// `pot` when it matches, otherwise from one force call that becomes the
/// cached centre.
///
/// `output->forces` is set to a new owning `[n_atoms, 3]` tensor; free it
/// with `rgpot_tensor_free`. A search that needs the centre forces calls
/// this instead of `rgpot_potential_calculate`, so the forward products at
/// that centre share the evaluation.
///
/// # Safety
///
/// A NULL `pot`, `input` or `output` returns `RGPOT_INVALID_PARAMETER`.
/// Otherwise the pointers follow `rgpot_potential_calculate`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_hvp_centre(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() || input.is_null() || output.is_null() {
            return refuse("rgpot_potential_hvp_centre: NULL argument");
        }
        match unsafe { (*pot).hvp_centre(&*input) } {
            Ok(centre) => {
                let n_atoms = (centre.forces.len() / 3) as i64;
                let out = unsafe { &mut *output };
                out.energy = centre.energy;
                out.variance = centre.variance;
                out.forces = create_owned_f64_tensor(centre.forces, vec![n_atoms, 3]);
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(err) => fail(err),
        }
    }))
}

/// Drop the cached centre of `pot`.
///
/// A centre is keyed on positions, atomic numbers and cell, not on the
/// state behind the callback's `user_data`. Call this after changing that
/// state at a fixed geometry. NULL is a no-op.
///
/// # Safety
///
/// `pot` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_hvp_invalidate(pot: *const rgpot_potential_t) {
    if !pot.is_null() {
        unsafe { &*pot }.invalidate_hvp_centre();
    }
}

/// Read the Hessian-vector counters of `pot` into `out`.
///
/// # Safety
///
/// `pot` must be NULL or a live handle, and `out` NULL or writable. NULL
/// for either returns `RGPOT_INVALID_PARAMETER`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_hvp_counts(
    pot: *const rgpot_potential_t,
    out: *mut rgpot_hvp_counts_t,
) -> rgpot_status_t {
    if pot.is_null() || out.is_null() {
        return refuse("rgpot_potential_hvp_counts: NULL argument");
    }
    unsafe { *out = (*pot).hvp_counts() };
    rgpot_status_t::RGPOT_SUCCESS
}

/// Zero the Hessian-vector counters of `pot`. The cached centre stays.
///
/// # Safety
///
/// `pot` must be NULL or a live handle. NULL is a no-op.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_reset_hvp_counts(pot: *const rgpot_potential_t) {
    if !pot.is_null() {
        unsafe { &*pot }.reset_hvp_counts();
    }
}
