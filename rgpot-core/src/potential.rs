// MIT License
// Copyright 2023--present rgpot developers

//! Callback-based potential dispatch.
//!
//! This module defines [`PotentialImpl`], an opaque handle that wraps a C
//! function pointer callback together with a `void* user_data` and an optional
//! destructor. This is the core abstraction that lets existing C++ potentials
//! (LJ, CuH2, or any future implementation) plug into the Rust infrastructure
//! without the Rust side knowing the concrete type.
//!
//! ## How it Works
//!
//! 1. The C++ side creates a potential object (e.g., `LJPot`).
//! 2. A trampoline function with the [`PotentialCallback`] signature is
//!    registered, casting `user_data` back to the concrete type and calling
//!    `forceImpl`.
//! 3. The Rust core dispatches through the function pointer, receiving results
//!    via the [`rgpot_force_out_t`] output struct.
//!
//! ## Lifetime Contract
//!
//! - The `user_data` pointer is borrowed by `PotentialImpl`. The caller must
//!   keep the underlying object alive for the lifetime of the handle.
//! - If a `free_fn` is provided, it is called on drop when `user_data` is
//!   non-null, transferring ownership to `PotentialImpl`.
//! - The handle is exposed to C as `rgpot_potential_t` — an opaque pointer
//!   managed via `rgpot_potential_new` / `rgpot_potential_free`.

use std::os::raw::c_void;

use crate::status::rgpot_status_t;
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Function pointer type for a potential energy calculation.
///
/// The callback receives:
/// - `user_data`: opaque pointer to the C++ object (e.g. `LJPot*`)
/// - `input`: the atomic configuration (DLPack tensors)
/// - `output`: the buffer for results (callback sets `forces` tensor)
///
/// Returns `RGPOT_SUCCESS` on success, or an error status code.
pub type PotentialCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t;

/// Destructor for the user_data pointer.
pub type FreeFn = unsafe extern "C" fn(*mut c_void);

/// Opaque potential handle wrapping a callback + user data.
pub struct PotentialImpl {
    pub(crate) callback: PotentialCallback,
    pub(crate) user_data: *mut c_void,
    pub(crate) free_fn: Option<FreeFn>,
}

// PotentialImpl stores a raw pointer but we guarantee exclusive access
// through the opaque handle pattern.
unsafe impl Send for PotentialImpl {}

impl PotentialImpl {
    /// Create a new potential from a callback, user data, and optional destructor.
    pub fn new(
        callback: PotentialCallback,
        user_data: *mut c_void,
        free_fn: Option<FreeFn>,
    ) -> Self {
        Self {
            callback,
            user_data,
            free_fn,
        }
    }

    /// Invoke the underlying callback.
    ///
    /// # Safety
    /// The caller must ensure `input` and `output` point to valid, properly
    /// sized structures.
    pub unsafe fn calculate(
        &self,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        (self.callback)(self.user_data, input, output)
    }
}

impl Drop for PotentialImpl {
    fn drop(&mut self) {
        if let Some(free) = self.free_fn {
            if !self.user_data.is_null() {
                unsafe { free(self.user_data) };
            }
        }
    }
}

// rgpot_potential_t is defined in the `eindir` module as a #[repr(C)] struct
// with eindir_objective_t embedded as its first member.  Re-exported here so
// existing import paths (e.g. rpc/server.rs) continue to work unchanged.
pub use crate::eindir::rgpot_potential_t;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{
        create_owned_f64_tensor, rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3,
        rgpot_tensor_cpu_i32_1d, rgpot_tensor_free,
    };
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// Helper: create input tensors from stack arrays.  Returns (input, cleanup_fn).
    struct TestIO {
        pos: Vec<f64>,
        atmnrs: Vec<i32>,
        box_: [f64; 9],
    }

    impl TestIO {
        fn new(n: usize) -> Self {
            Self {
                pos: vec![0.0; n * 3],
                atmnrs: vec![1; n],
                box_: [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            }
        }

        fn make_input(&mut self) -> rgpot_force_input_t {
            let n = self.atmnrs.len();
            rgpot_force_input_t {
                positions: unsafe { rgpot_tensor_cpu_f64_2d(self.pos.as_mut_ptr(), n as i64, 3) },
                atomic_numbers: unsafe {
                    rgpot_tensor_cpu_i32_1d(self.atmnrs.as_mut_ptr(), n as i64)
                },
                box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(self.box_.as_mut_ptr()) },
            }
        }

        unsafe fn free_input(&self, input: &rgpot_force_input_t) {
            unsafe {
                rgpot_tensor_free(input.positions);
                rgpot_tensor_free(input.atomic_numbers);
                rgpot_tensor_free(input.box_matrix);
            }
        }
    }

    /// Mock callback: sets energy = n_atoms, creates owning forces tensor.
    unsafe extern "C" fn mock_callback(
        _user_data: *mut c_void,
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

    unsafe extern "C" fn error_callback(
        _ud: *mut c_void,
        _input: *const rgpot_force_input_t,
        _output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        rgpot_status_t::RGPOT_INVALID_PARAMETER
    }

    unsafe extern "C" fn writing_callback(
        _ud: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        let n = unsafe { inp.n_atoms() }.unwrap_or(0);
        let forces: Vec<f64> = (1..=(n * 3)).map(|i| i as f64).collect();
        out.forces = create_owned_f64_tensor(forces, vec![n as i64, 3]);
        out.energy = -42.0;
        out.variance = 0.5;
        rgpot_status_t::RGPOT_SUCCESS
    }

    #[test]
    fn test_potential_callback() {
        let pot = PotentialImpl::new(mock_callback, std::ptr::null_mut(), None);

        let mut io = TestIO::new(3);
        let input = io.make_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        let status = unsafe { pot.calculate(&input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(output.energy, 3.0);

        unsafe {
            rgpot_tensor_free(output.forces);
            io.free_input(&input);
        }
    }

    #[test]
    fn callback_error_status_is_returned() {
        let pot = PotentialImpl::new(error_callback, std::ptr::null_mut(), None);
        let mut io = TestIO::new(1);
        let input = io.make_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let status = unsafe { pot.calculate(&input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
        unsafe { io.free_input(&input) };
    }

    #[test]
    fn callback_writes_forces_energy_variance() {
        let pot = PotentialImpl::new(writing_callback, std::ptr::null_mut(), None);
        let mut io = TestIO::new(2);
        let input = io.make_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        let status = unsafe { pot.calculate(&input, &mut output) };
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(output.energy, -42.0);
        assert_eq!(output.variance, 0.5);

        // Read forces from the DLPack tensor
        assert!(!output.forces.is_null());
        let ft = unsafe { &(*output.forces).dl_tensor };
        let forces = unsafe { std::slice::from_raw_parts(ft.data as *const f64, 6) };
        assert_eq!(forces, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        unsafe {
            rgpot_tensor_free(output.forces);
            io.free_input(&input);
        }
    }

    static DROP_CALLED: AtomicBool = AtomicBool::new(false);

    unsafe extern "C" fn track_drop(ptr: *mut c_void) {
        DROP_CALLED.store(true, Ordering::SeqCst);
        let val = unsafe { *(ptr as *const u64) };
        assert_eq!(val, 0xDEAD_BEEF);
    }

    #[test]
    fn drop_calls_free_fn_with_user_data() {
        DROP_CALLED.store(false, Ordering::SeqCst);
        let mut sentinel: u64 = 0xDEAD_BEEF;
        {
            let _pot = PotentialImpl::new(
                mock_callback,
                &mut sentinel as *mut u64 as *mut c_void,
                Some(track_drop),
            );
            assert!(!DROP_CALLED.load(Ordering::SeqCst));
        }
        assert!(DROP_CALLED.load(Ordering::SeqCst));
    }

    #[test]
    fn drop_skips_free_fn_when_user_data_is_null() {
        let _pot = PotentialImpl::new(mock_callback, std::ptr::null_mut(), Some(track_drop));
    }

    #[test]
    fn drop_without_free_fn_is_safe() {
        let _pot = PotentialImpl::new(mock_callback, std::ptr::null_mut(), None);
    }

    #[test]
    fn user_data_is_passed_through() {
        static CALL_COUNT: AtomicU32 = AtomicU32::new(0);

        unsafe extern "C" fn count_cb(
            ud: *mut c_void,
            _input: *const rgpot_force_input_t,
            output: *mut rgpot_force_out_t,
        ) -> rgpot_status_t {
            let ctr = unsafe { &*(ud as *const AtomicU32) };
            ctr.fetch_add(1, Ordering::SeqCst);
            unsafe { (*output).energy = ctr.load(Ordering::SeqCst) as f64 };
            rgpot_status_t::RGPOT_SUCCESS
        }

        CALL_COUNT.store(0, Ordering::SeqCst);
        let pot = PotentialImpl::new(count_cb, &CALL_COUNT as *const _ as *mut c_void, None);

        let mut io = TestIO::new(1);
        let input = io.make_input();
        let mut output = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };

        unsafe { pot.calculate(&input, &mut output) };
        assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(output.energy, 1.0);

        unsafe { pot.calculate(&input, &mut output) };
        assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 2);
        assert_eq!(output.energy, 2.0);

        unsafe { io.free_input(&input) };
    }

    #[test]
    fn potential_impl_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PotentialImpl>();
    }
}
