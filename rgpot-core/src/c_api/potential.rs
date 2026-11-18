// MIT License
// Copyright 2023--present rgpot developers

//! C API for the potential handle lifecycle: create, calculate, free.
//!
//! `rgpot_potential_t` is the superset struct defined in `crate::eindir`:
//! it embeds `eindir_objective_t` as its first member so every
//! `rgpot_potential_t*` IS-A `eindir_objective_t*` at the C ABI.
//!
//! The three lifecycle functions here use the direct callback path
//! (`pot.callback`); the eindir evaluation path goes through
//! `eindir_objective_eval` with the zero-cost cast.
//!
//! ```c
//! // Direct rgpot path:
//! rgpot_potential_t *pot = rgpot_potential_new(my_callback, my_data, NULL);
//! rgpot_status_t s = rgpot_potential_calculate(pot, &input, &output);
//! rgpot_potential_free(pot);
//!
//! // eindir path (IS-A cast, no conversion method):
//! eindir_objective_t *obj = (eindir_objective_t*)pot;
//! double val;
//! eindir_objective_eval(obj, x_tensor, &val);
//! ```

use std::os::raw::c_void;

use crate::eindir::rgpot_potential_t;
use crate::potential::PotentialCallback;
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Create a new potential handle from a callback function pointer.
///
/// Creates a `rgpot_potential_t` with no molecular context (n_atoms = 0) and
/// no eindir bounds.  The potential evaluates through `callback` directly.
/// Use `rgpot_potential_new_eindir` when the IS-A eindir embedding is needed.
///
/// Returns a heap-allocated `rgpot_potential_t*`, or `NULL` on failure.
/// The caller must eventually pass the returned pointer to `rgpot_potential_free`.
///
/// # Safety
///
/// `callback` and `free_fn` must be safe to call with `user_data` for as
/// long as the handle lives.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_new(
    callback: PotentialCallback,
    user_data: *mut c_void,
    free_fn: Option<unsafe extern "C" fn(*mut c_void)>,
) -> *mut rgpot_potential_t {
    use crate::eindir::rgpot_potential_new_eindir;
    // Create a superset struct with no molecular context.
    unsafe {
        rgpot_potential_new_eindir(
            callback,
            user_data,
            free_fn,
            0,                // n_atoms = 0
            std::ptr::null(), // atomic_numbers
            std::ptr::null(), // box_matrix
            std::ptr::null(), // bounds_low
            std::ptr::null(), // bounds_high
        )
    }
}

/// Perform a force/energy calculation using the potential handle.
///
/// Routes through the direct rgpot callback (`pot.callback`), bypassing the
/// eindir evaluation path.
///
/// # Safety
///
/// A NULL `pot`, `input` or `output` returns `RGPOT_INVALID_PARAMETER`.
/// A missing positions tensor, shape or data, a non-positive atom count,
/// a missing species tensor or data, or a missing cell tensor or data
/// also returns `RGPOT_INVALID_PARAMETER` and does not call the callback.
/// A non-positive atom count is reported on its own, before any force
/// buffer is considered. A successful call does not clear `rgpot_last_error`.
/// Otherwise `pot` must be a live handle from `rgpot_potential_new` or
/// `rgpot_potential_new_eindir`, `input` must point to a valid
/// `rgpot_force_input_t` whose tensors stay alive for the call, and `output`
/// must point to writable `rgpot_force_out_t` storage.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_calculate(
    pot: *const rgpot_potential_t,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if pot.is_null() {
            set_last_error("rgpot_potential_calculate: pot is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if input.is_null() {
            set_last_error("rgpot_potential_calculate: input is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if output.is_null() {
            set_last_error("rgpot_potential_calculate: output is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let p = unsafe { &*pot };
        if let Some(why) = p.fused_cache_peer_refusal() {
            set_last_error(&why);
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if let Some(why) = unsafe { force_input_defect(input) } {
            set_last_error(why);
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        unsafe { (p.callback)(p.pot_user_data, input, output) }
    }))
}

/// Missing buffer in a force input, or `None` when the callback may run.
///
/// A non-positive row count is decided before the species, the cell or
/// the output forces. An empty call has no force allocation, and that
/// null must not become the reported error. A usable input returns
/// `None` and leaves `rgpot_last_error` unchanged.
///
/// # Safety
///
/// `input` must be a non-null `rgpot_force_input_t`.
unsafe fn force_input_defect(input: *const rgpot_force_input_t) -> Option<&'static str> {
    let input = unsafe { &*input };
    if input.positions.is_null() {
        return Some("invalid parameter: positions is null");
    }
    let positions = unsafe { &(*input.positions).dl_tensor };
    if positions.ndim < 1 || positions.shape.is_null() {
        return Some("invalid parameter: positions is null");
    }
    let rows = unsafe { *positions.shape };
    if rows <= 0 {
        return Some("invalid parameter: nAtoms must be positive");
    }
    if positions.data.is_null() {
        return Some("invalid parameter: positions is null");
    }
    if input.atomic_numbers.is_null() {
        return Some("invalid parameter: atomic numbers are null");
    }
    let species = unsafe { &(*input.atomic_numbers).dl_tensor };
    if species.data.is_null() {
        return Some("invalid parameter: atomic numbers are null");
    }
    if input.box_matrix.is_null() {
        return Some("invalid parameter: box is null");
    }
    let cell = unsafe { &(*input.box_matrix).dl_tensor };
    if cell.data.is_null() {
        return Some("invalid parameter: box is null");
    }
    None
}

/// Free a potential handle previously obtained from `rgpot_potential_new` or
/// `rgpot_potential_new_eindir`.
///
/// Calls `pot_free_fn(pot_user_data)` if provided, frees all owned arrays,
/// then frees the struct.
///
/// # Safety
///
/// `pot` must be NULL or a pointer returned by one of those constructors
/// that has not been freed yet. The pointer is invalid after the call.
#[no_mangle]
pub unsafe extern "C" fn rgpot_potential_free(pot: *mut rgpot_potential_t) {
    if pot.is_null() {
        return;
    }
    let p = unsafe { &*pot };
    if let Some(ff) = p.pot_free_fn {
        if !p.pot_user_data.is_null() {
            unsafe { ff(p.pot_user_data) };
        }
    }
    let dim = p.base.dim;
    if !p.base.low.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.base.low, dim, dim)) };
    }
    if !p.base.high.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.base.high, dim, dim)) };
    }
    if !p.atomic_numbers.is_null() {
        unsafe { drop(Vec::from_raw_parts(p.atomic_numbers, p.n_atoms, p.n_atoms)) };
    }
    unsafe { drop(Box::from_raw(pot)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{create_owned_f64_tensor, rgpot_tensor_free};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    unsafe extern "C" fn sum_callback(
        _ud: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        let n = unsafe { inp.n_atoms() }.unwrap_or(0);
        out.energy = n as f64;
        out.variance = 0.0;
        out.forces = create_owned_f64_tensor(vec![0.0; n * 3], vec![n as i64, 3]);
        rgpot_status_t::RGPOT_SUCCESS
    }

    unsafe extern "C" fn failing_callback(
        _ud: *mut c_void,
        _input: *const rgpot_force_input_t,
        _output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        crate::status::set_last_error("deliberately failed");
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    }

    unsafe extern "C" fn counting_callback(
        ud: *mut c_void,
        _input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let counter = unsafe { &*(ud as *const AtomicU32) };
        counter.fetch_add(1, Ordering::SeqCst);
        let out = unsafe { &mut *output };
        out.energy = counter.load(Ordering::SeqCst) as f64;
        rgpot_status_t::RGPOT_SUCCESS
    }

    /// The input borrows `pos`, `atmnrs` and `box_`, which are returned with
    /// it. They live on the heap so moving them out of this function keeps
    /// the pointers in the input valid; fixed-size arrays would be copied
    /// and leave the input pointing at this frame.
    fn make_test_input() -> (Vec<f64>, Vec<i32>, Vec<f64>, rgpot_force_input_t) {
        let mut pos = vec![0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut atmnrs = vec![1_i32, 1];
        let mut box_ = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let input = unsafe {
            crate::c_api::types::rgpot_force_input_create(
                2,
                pos.as_mut_ptr(),
                atmnrs.as_mut_ptr(),
                box_.as_mut_ptr(),
            )
        };
        (pos, atmnrs, box_, input)
    }

    unsafe fn cleanup(input: &mut rgpot_force_input_t, output: &mut rgpot_force_out_t) {
        unsafe {
            rgpot_tensor_free(output.forces);
            crate::c_api::types::rgpot_force_input_free(input);
        }
        output.forces = std::ptr::null_mut();
    }

    #[test]
    fn new_returns_non_null() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        assert!(!pot.is_null());
        unsafe { rgpot_potential_free(pot) };
    }

    #[test]
    fn free_null_is_noop() {
        unsafe { rgpot_potential_free(std::ptr::null_mut()) };
    }

    #[test]
    fn calculate_null_pot_returns_invalid_parameter() {
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(std::ptr::null(), &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe { crate::c_api::types::rgpot_force_input_free(&mut input) };
    }

    #[test]
    fn calculate_null_input_returns_invalid_parameter() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(pot, std::ptr::null(), &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe { rgpot_potential_free(pot as *mut _) };
    }

    #[test]
    fn calculate_null_output_returns_invalid_parameter() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let status = unsafe { rgpot_potential_calculate(pot, &input, std::ptr::null_mut()) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot as *mut _);
        }
    }

    #[test]
    fn full_lifecycle_new_calculate_free() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(output.energy, 2.0);
        assert!(!output.forces.is_null());
        unsafe {
            cleanup(&mut input, &mut output);
            rgpot_potential_free(pot as *mut _);
        }
    }

    #[test]
    fn callback_error_propagates() {
        let pot = unsafe { rgpot_potential_new(failing_callback, std::ptr::null_mut(), None) };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR);
        let msg = unsafe { std::ffi::CStr::from_ptr(crate::status::rgpot_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "deliberately failed");
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot as *mut _);
        }
    }

    #[test]
    fn user_data_is_forwarded_to_callback() {
        let counter = AtomicU32::new(0);
        let pot = unsafe {
            rgpot_potential_new(counting_callback, &counter as *const _ as *mut c_void, None)
        };
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(output.energy, 1.0);
        unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(output.energy, 2.0);
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot as *mut _);
        }
    }

    static FREE_CALLED: AtomicBool = AtomicBool::new(false);

    unsafe extern "C" fn track_free(_ptr: *mut c_void) {
        FREE_CALLED.store(true, Ordering::SeqCst);
    }

    #[test]
    fn free_fn_is_invoked_on_drop() {
        FREE_CALLED.store(false, Ordering::SeqCst);
        let mut dummy: u8 = 42;
        let pot = unsafe {
            rgpot_potential_new(
                sum_callback,
                &mut dummy as *mut u8 as *mut c_void,
                Some(track_free),
            )
        };
        assert!(!FREE_CALLED.load(Ordering::SeqCst));
        unsafe { rgpot_potential_free(pot) };
        assert!(FREE_CALLED.load(Ordering::SeqCst));
    }

    #[test]
    fn multiple_calculations_same_handle() {
        let pot = unsafe { rgpot_potential_new(sum_callback, std::ptr::null_mut(), None) };
        for n in 1..=5_usize {
            let mut pos = vec![0.0_f64; n * 3];
            let mut atmnrs = vec![1_i32; n];
            let mut box_ = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
            let mut input = unsafe {
                crate::c_api::types::rgpot_force_input_create(
                    n,
                    pos.as_mut_ptr(),
                    atmnrs.as_mut_ptr(),
                    box_.as_mut_ptr(),
                )
            };
            let mut output = rgpot_force_out_t {
                forces: std::ptr::null_mut(),
                energy: 0.0,
                variance: 0.0,
            };
            let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
            assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
            assert_eq!(output.energy, n as f64);
            unsafe { cleanup(&mut input, &mut output) };
        }
        unsafe { rgpot_potential_free(pot as *mut _) };
    }

    fn last_error() -> String {
        unsafe {
            std::ffi::CStr::from_ptr(crate::status::rgpot_last_error())
                .to_string_lossy()
                .into_owned()
        }
    }

    fn counting_pot(counter: &AtomicU32) -> *mut rgpot_potential_t {
        unsafe {
            rgpot_potential_new(
                counting_callback,
                counter as *const AtomicU32 as *mut c_void,
                None,
            )
        }
    }

    #[test]
    fn calculate_null_position_data_skips_the_callback() {
        let counter = AtomicU32::new(0);
        let pot = counting_pot(&counter);
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        unsafe {
            (*input.positions).dl_tensor.data = std::ptr::null_mut();
        }
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        assert_eq!(last_error(), "invalid parameter: positions is null");
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot);
        }
    }

    #[test]
    fn calculate_zero_atoms_is_not_a_null_force_buffer() {
        let counter = AtomicU32::new(0);
        let pot = counting_pot(&counter);
        let mut box_ = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let mut input = unsafe {
            crate::c_api::types::rgpot_force_input_create(
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                box_.as_mut_ptr(),
            )
        };
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        assert_eq!(last_error(), "invalid parameter: nAtoms must be positive");
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot);
        }
    }

    #[test]
    fn calculate_null_box_data_skips_the_callback() {
        let counter = AtomicU32::new(0);
        let pot = counting_pot(&counter);
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        unsafe {
            (*input.box_matrix).dl_tensor.data = std::ptr::null_mut();
        }
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        assert_eq!(last_error(), "invalid parameter: box is null");
        unsafe {
            crate::c_api::types::rgpot_force_input_free(&mut input);
            rgpot_potential_free(pot);
        }
    }

    #[test]
    fn calculate_success_keeps_the_previous_error() {
        let counter = AtomicU32::new(0);
        let pot = counting_pot(&counter);
        crate::status::set_last_error("kept across success");
        let (_pos, _atmnrs, _box_, mut input) = make_test_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { rgpot_potential_calculate(pot, &input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(last_error(), "kept across success");
        unsafe {
            cleanup(&mut input, &mut output);
            rgpot_potential_free(pot);
        }
    }
}
