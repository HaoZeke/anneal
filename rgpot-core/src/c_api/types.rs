// MIT License
// Copyright 2023--present rgpot developers

//! Convenience constructors for the core C types.
//!
//! The struct definitions themselves live in [`crate::types`]. This module
//! provides `extern "C"` factory functions so that C callers can construct the
//! structs without needing to create DLPack tensors manually.
//!
//! `rgpot_force_input_create` creates non-owning DLPack tensors internally.
//! The caller must call `rgpot_force_input_free` when done.

use crate::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_free,
};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};
use std::os::raw::c_int;

/// Create a `rgpot_force_input_t` from raw CPU arrays.
///
/// Internally creates non-owning DLPack tensors wrapping the raw pointers.
/// The caller must call `rgpot_force_input_free` to free the tensor metadata
/// when done (this does NOT free the underlying data arrays).
///
/// # Safety
/// All pointers must be valid for the lifetime of the returned struct.
/// `pos` must point to at least `n_atoms * 3` doubles.
/// `atmnrs` must point to at least `n_atoms` ints.
/// `box_` must point to at least 9 doubles.
#[no_mangle]
pub unsafe extern "C" fn rgpot_force_input_create(
    n_atoms: usize,
    pos: *mut f64,
    atmnrs: *mut c_int,
    box_: *mut f64,
) -> rgpot_force_input_t {
    rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(pos, n_atoms as i64, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(atmnrs, n_atoms as i64) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(box_) },
    }
}

/// Free the DLPack tensor metadata in a `rgpot_force_input_t`.
///
/// This frees the `DLManagedTensorVersioned` wrappers created by
/// `rgpot_force_input_create`, but does NOT free the underlying data arrays
/// (which are borrowed).
///
/// After this call, all tensor pointers in `input` are set to NULL.
///
/// If `input` is NULL, this is a no-op.
///
/// # Safety
/// `input` must be NULL or point to a valid `rgpot_force_input_t` whose
/// tensors were created by `rgpot_force_input_create`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_force_input_free(input: *mut rgpot_force_input_t) {
    if input.is_null() {
        return;
    }
    let inp = unsafe { &mut *input };
    unsafe {
        rgpot_tensor_free(inp.positions);
        rgpot_tensor_free(inp.atomic_numbers);
        rgpot_tensor_free(inp.box_matrix);
    }
    inp.positions = std::ptr::null_mut();
    inp.atomic_numbers = std::ptr::null_mut();
    inp.box_matrix = std::ptr::null_mut();
}

/// Create a `rgpot_force_out_t` with null forces and zeroed scalars.
///
/// The `forces` field starts as NULL — the potential callback is responsible
/// for setting it to a valid DLPack tensor.
///
/// # Safety
///
/// The function reads no memory. It stays `unsafe` because the C ABI
/// declares it so; the caller owns the `forces` tensor once the callback
/// sets it.
#[no_mangle]
pub unsafe extern "C" fn rgpot_force_out_create() -> rgpot_force_out_t {
    rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlpk::sys::DLDeviceType;

    #[test]
    fn force_input_create_produces_valid_tensors() {
        let mut pos = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut atmnrs = [6_i32, 1];
        let mut box_ = [10.0_f64; 9];

        let input = unsafe {
            rgpot_force_input_create(2, pos.as_mut_ptr(), atmnrs.as_mut_ptr(), box_.as_mut_ptr())
        };

        assert!(!input.positions.is_null());
        assert!(!input.atomic_numbers.is_null());
        assert!(!input.box_matrix.is_null());

        // Verify positions tensor metadata
        let pt = unsafe { &(*input.positions).dl_tensor };
        assert_eq!(pt.ndim, 2);
        assert_eq!(pt.device.device_type, DLDeviceType::kDLCPU);
        let shape = unsafe { std::slice::from_raw_parts(pt.shape, 2) };
        assert_eq!(shape, &[2, 3]);

        // Verify data roundtrips
        let data = unsafe { std::slice::from_raw_parts(pt.data as *const f64, 6) };
        assert_eq!(data, &pos);

        let mut input = input;
        unsafe { rgpot_force_input_free(&mut input) };
        assert!(input.positions.is_null());
    }

    #[test]
    fn force_input_free_null_is_noop() {
        unsafe { rgpot_force_input_free(std::ptr::null_mut()) };
    }

    #[test]
    fn force_out_create_starts_null() {
        let out = unsafe { rgpot_force_out_create() };
        assert!(out.forces.is_null());
        assert_eq!(out.energy, 0.0);
        assert_eq!(out.variance, 0.0);
    }

    #[test]
    fn force_input_n_atoms_accessor() {
        let mut pos = [0.0_f64; 9];
        let mut atmnrs = [1_i32; 3];
        let mut box_ = [0.0_f64; 9];

        let input = unsafe {
            rgpot_force_input_create(3, pos.as_mut_ptr(), atmnrs.as_mut_ptr(), box_.as_mut_ptr())
        };

        assert_eq!(unsafe { input.n_atoms() }, Some(3));

        let mut input = input;
        unsafe { rgpot_force_input_free(&mut input) };
    }
}
