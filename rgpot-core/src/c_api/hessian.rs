// MIT License
// Copyright 2023--present rgpot developers

//! C API for dense Hessians of a potential handle, and for the capabilities
//! and counts of its derivative entry points.
//!
//! ```c
//! rgpot_derivative_caps_t caps;
//! rgpot_potential_derivative_caps(pot, &caps);
//! int scheme = caps.hessian_analytic ? RGPOT_HESSIAN_ANALYTIC
//!                                    : RGPOT_HESSIAN_CENTRAL;
//! rgpot_hessian_scheme_t used;
//! rgpot_potential_hessian(pot, &input, 0.0, scheme, h, &used);
//! ```
//!
//! See [`crate::hessian`] for the schemes.

use std::os::raw::c_void;

use crate::c_api::hvp::{fail, refuse};
use crate::eindir::rgpot_potential_t;
use crate::hessian::{
    refuse_overlap, rgpot_derivative_caps_t, rgpot_derivative_counts_t, rgpot_hessian_scheme_t,
    HessianCallback,
};
use crate::status::{catch_unwind, rgpot_status_t};
use crate::types::rgpot_force_input_t;

/// Register `callback` as the analytic Hessian of `pot`, or clear it with
/// NULL.
///
/// The callback receives the `user_data` the handle was created with and
/// writes `(3 * n_atoms)^2` doubles, row-major. `RGPOT_HESSIAN_ANALYTIC`
/// calls it, and so does `RGPOT_HESSIAN_EXACT`.
///
/// # Safety
///
/// `pot` must be NULL or a live handle. A NULL `pot` returns
/// `RGPOT_INVALID_PARAMETER`. `callback` must be safe to call with the
/// handle's `user_data` for as long as it stays registered.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_hessian_callback(
    pot: *const rgpot_potential_t,
    callback: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            input: *const rgpot_force_input_t,
            hessian_out: *mut f64,
        ) -> rgpot_status_t,
    >,
) -> rgpot_status_t {
    let callback: Option<HessianCallback> = callback;
    if pot.is_null() {
        return refuse("rgpot_potential_set_hessian_callback: pot is NULL");
    }
    unsafe { &*pot }.set_hessian_callback(callback);
    rgpot_status_t::RGPOT_SUCCESS
}

/// Register `callback` as the dual-number Hessian of `pot`, or clear it
/// with NULL.
///
/// It has the signature of the analytic Hessian callback.
/// `RGPOT_HESSIAN_AD` calls it, and `RGPOT_HESSIAN_EXACT` when no analytic
/// Hessian callback is registered.
///
/// # Safety
///
/// As for `rgpot_potential_set_hessian_callback`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_ad_hessian_callback(
    pot: *const rgpot_potential_t,
    callback: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            input: *const rgpot_force_input_t,
            hessian_out: *mut f64,
        ) -> rgpot_status_t,
    >,
) -> rgpot_status_t {
    let callback: Option<HessianCallback> = callback;
    if pot.is_null() {
        return refuse("rgpot_potential_set_ad_hessian_callback: pot is NULL");
    }
    unsafe { &*pot }.set_ad_hessian_callback(callback);
    rgpot_status_t::RGPOT_SUCCESS
}

/// Write the dense Hessian of the energy at the geometry of `input` to
/// `hessian_out` and the scheme that computed it to `scheme_used`.
///
/// `hessian_out` holds `(3 * n_atoms)^2` doubles, row-major, coordinate
/// `3 i + k` for axis `k` of atom `i`. `scheme` is a
/// `rgpot_hessian_scheme_t` value, and there is no automatic one:
///
/// - `RGPOT_HESSIAN_ANALYTIC`: the registered Hessian callback;
///   `RGPOT_INVALID_PARAMETER` when none is registered.
/// - `RGPOT_HESSIAN_AD`: the registered dual-number Hessian callback;
///   `RGPOT_INVALID_PARAMETER` when none is registered.
/// - `RGPOT_HESSIAN_EXACT`: the analytic callback, else the dual-number
///   one; `RGPOT_INVALID_PARAMETER` when neither is registered.
/// - `RGPOT_HESSIAN_CENTRAL`: column `c` is
///   `(F(x - h e_c) - F(x + h e_c)) / (2 h)`, from `6 * n_atoms` force
///   calls that bypass the cached centre, and the matrix written is the
///   symmetric part of those columns. Error `O(h^2)`.
///
/// `step` is `h`, a length in the units of the positions; 0 selects
/// `RGPOT_HVP_DEFAULT_STEP`. `RGPOT_HESSIAN_EXACT` reports
/// `RGPOT_HESSIAN_ANALYTIC` or `RGPOT_HESSIAN_AD`, and any other scheme
/// reports itself. `scheme_used` is written only on success; on a failure
/// the content of `hessian_out` is unspecified. A failing callback's status
/// is returned with the message it set, or `<callback> returned <status>`
/// when it set none. A non-finite entry from any scheme returns
/// `RGPOT_INVALID_PARAMETER`.
///
/// # Safety
///
/// NULL pointers, an unknown scheme, a negative or non-finite step and a
/// `hessian_out` that overlaps the positions, atomic numbers or box
/// return `RGPOT_INVALID_PARAMETER` without a call. Otherwise `pot` must be a live
/// handle, `input` must point to a valid `rgpot_force_input_t` whose tensors
/// stay alive for the call, `hessian_out` must point to `(3 * n_atoms)^2`
/// writable doubles and `scheme_used` must be writable.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_hessian(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    step: f64,
    scheme: i32,
    hessian_out: *mut f64,
    scheme_used: *mut rgpot_hessian_scheme_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() || input.is_null() {
            return refuse("rgpot_potential_hessian: pot or input is NULL");
        }
        if hessian_out.is_null() || scheme_used.is_null() {
            return refuse("rgpot_potential_hessian: hessian_out or scheme_used is NULL");
        }
        let Some(scheme) = rgpot_hessian_scheme_t::from_raw(scheme) else {
            return refuse(&format!("rgpot_potential_hessian: unknown scheme {scheme}"));
        };
        let input = unsafe { &*input };
        let Some(entries) = unsafe { input.n_atoms() }
            .and_then(|n| n.checked_mul(3))
            .and_then(|dim| dim.checked_mul(dim))
            .filter(|&entries| entries <= isize::MAX as usize / std::mem::size_of::<f64>())
        else {
            return refuse("rgpot_potential_hessian: positions tensor is NULL or too large");
        };
        if let Err(err) = unsafe { refuse_overlap(input, hessian_out, entries) } {
            return fail(err);
        }
        let hessian_out = unsafe { std::slice::from_raw_parts_mut(hessian_out, entries) };
        match unsafe { (*pot).hessian(input, step, scheme, hessian_out) } {
            Ok(ran) => {
                unsafe { scheme_used.write(ran) };
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(err) => fail(err),
        }
    }))
}

/// Read which derivative schemes `pot` has into `out`.
///
/// The analytic and dual-number flags follow the callbacks registered at
/// the time of the call; the finite-difference flags are set for every
/// handle, which always has its force callback.
///
/// # Safety
///
/// `pot` must be NULL or a live handle, and `out` NULL or writable. NULL
/// for either returns `RGPOT_INVALID_PARAMETER`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_derivative_caps(
    pot: *const rgpot_potential_t,
    out: *mut rgpot_derivative_caps_t,
) -> rgpot_status_t {
    if pot.is_null() || out.is_null() {
        return refuse("rgpot_potential_derivative_caps: NULL argument");
    }
    unsafe { out.write((*pot).derivative_caps()) };
    rgpot_status_t::RGPOT_SUCCESS
}

/// Read the counts of requests each derivative scheme of `pot` answered
/// into `out`. `rgpot_potential_reset_hvp_counts` zeros them.
///
/// # Safety
///
/// `pot` must be NULL or a live handle, and `out` NULL or writable. NULL
/// for either returns `RGPOT_INVALID_PARAMETER`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_derivative_counts(
    pot: *const rgpot_potential_t,
    out: *mut rgpot_derivative_counts_t,
) -> rgpot_status_t {
    if pot.is_null() || out.is_null() {
        return refuse("rgpot_potential_derivative_counts: NULL argument");
    }
    unsafe { out.write((*pot).derivative_counts()) };
    rgpot_status_t::RGPOT_SUCCESS
}
