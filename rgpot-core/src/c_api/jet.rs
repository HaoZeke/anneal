// MIT License
// Copyright 2023--present rgpot developers

//! C API for validated Taylor jets of a potential handle.
//!
//! ```c
//! double lo[ORDER + 1], hi[ORDER + 1];
//! rgpot_status_t s = rgpot_potential_taylor_jet(pot, &input, v, ORDER, lo, hi);
//! if (s == RGPOT_NO_JET) {
//!   // the potential has no exact jet; there is no fallback
//! }
//! ```
//!
//! See [`crate::jet`] for the meaning of the enclosures.

use std::os::raw::c_void;

use crate::eindir::rgpot_potential_t;
use crate::jet::{disjoint_outputs, rgpot_taylor_jet_fn};
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::types::rgpot_force_input_t;

fn refuse(msg: &str) -> rgpot_status_t {
    set_last_error(msg);
    rgpot_status_t::RGPOT_INVALID_PARAMETER
}

/// Register `jet` as the Taylor jet of `pot`, or clear it with NULL.
///
/// The callback receives the `user_data` the handle was created with.
///
/// # Safety
///
/// `pot` must be NULL or a live handle. A NULL `pot` returns
/// `RGPOT_INVALID_PARAMETER`. `jet` must be safe to call with the handle's
/// `user_data` for as long as it stays registered.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_set_taylor_jet(
    pot: *const rgpot_potential_t,
    jet: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            input: *const rgpot_force_input_t,
            direction: *const f64,
            order: usize,
            lower: *mut f64,
            upper: *mut f64,
        ) -> rgpot_status_t,
    >,
) -> rgpot_status_t {
    let jet: Option<rgpot_taylor_jet_fn> = jet;
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() {
            return refuse("rgpot_potential_set_taylor_jet: pot is NULL");
        }
        unsafe { &*pot }.set_taylor_jet(jet);
        rgpot_status_t::RGPOT_SUCCESS
    }))
}

/// Write the enclosures of the Taylor coefficients `0..=order` of the
/// energy along `x + t direction` to `lower` and `upper`.
///
/// `direction` holds `3 * n_atoms` doubles laid out like the positions;
/// `lower` and `upper` hold `order + 1` doubles each, must not overlap, and
/// are written only on success. Without a registered jet the status is
/// `RGPOT_NO_JET`; a branch or domain failure of the jet is
/// `RGPOT_JET_BRANCH` or `RGPOT_JET_DOMAIN`, with the test or function and
/// its enclosure in `rgpot_last_error`. An `order` whose `order + 1`
/// doubles overflow in bytes is `RGPOT_BUFFER_SIZE_ERROR`.
///
/// # Safety
///
/// NULL pointers and overlapping `lower` and `upper` return
/// `RGPOT_INVALID_PARAMETER` before the registered callback is called.
/// Otherwise `pot` must be a live handle, `input` must point to a valid
/// `rgpot_force_input_t` whose tensors stay alive for the call, `direction`
/// to `3 * n_atoms` readable doubles, and `lower` and `upper` to
/// `order + 1` writable doubles each.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_taylor_jet(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    order: usize,
    lower: *mut f64,
    upper: *mut f64,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() || input.is_null() {
            return refuse("rgpot_potential_taylor_jet: pot or input is NULL");
        }
        if direction.is_null() || lower.is_null() || upper.is_null() {
            return refuse("rgpot_potential_taylor_jet: direction, lower or upper is NULL");
        }
        let count = match disjoint_outputs(lower, upper, order) {
            Ok(count) => count,
            Err(failure) => {
                set_last_error(&format!("rgpot_potential_taylor_jet: {}", failure.message));
                return failure.status;
            }
        };
        let input = unsafe { &*input };
        let Some(dim) = unsafe { input.n_atoms() }.and_then(|n| n.checked_mul(3)) else {
            return refuse("rgpot_potential_taylor_jet: positions tensor is NULL or too large");
        };
        let direction = unsafe { std::slice::from_raw_parts(direction, dim) };
        match unsafe { (*pot).taylor_jet(input, direction, order) } {
            Ok(coefficients) => {
                let lower = unsafe { std::slice::from_raw_parts_mut(lower, count) };
                let upper = unsafe { std::slice::from_raw_parts_mut(upper, count) };
                for (k, c) in coefficients.iter().enumerate() {
                    lower[k] = c.lo;
                    upper[k] = c.hi;
                }
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(failure) => {
                set_last_error(&failure.message);
                failure.status
            }
        }
    }))
}
