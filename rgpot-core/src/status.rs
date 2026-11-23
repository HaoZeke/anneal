// MIT License
// Copyright 2023--present rgpot developers

//! Error handling following the
//! [metatensor](https://docs.metatensor.org/) pattern.
//!
//! This module provides three components that work together to give C/C++
//! callers safe, informative error reporting from Rust:
//!
//! 1. **[`rgpot_status_t`]** — An integer-valued enum returned from every
//!    `extern "C"` function. `RGPOT_SUCCESS` (0) means the call succeeded;
//!    any other value indicates a specific error category.
//!
//! 2. **Thread-local error message** — On failure, a human-readable
//!    description is stored in a thread-local `CString`. The C caller
//!    retrieves it with [`rgpot_last_error()`]. The pointer is valid until
//!    the next `rgpot_*` call on the same thread.
//!
//! 3. **`catch_unwind`** — A wrapper used inside every `extern "C"`
//!    function to catch Rust panics before they unwind across the FFI
//!    boundary (which is undefined behaviour). Caught panics become
//!    `RGPOT_INTERNAL_ERROR` with the panic message stored for retrieval.
//!
//! **Usage from C**
//!
//! ```c
//! rgpot_status_t s = rgpot_potential_calculate(pot, &input, &output);
//! if (s != RGPOT_SUCCESS) {
//!     fprintf(stderr, "rgpot error: %s\n", rgpot_last_error());
//! }
//! ```

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

/// Status codes returned by all C API functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum rgpot_status_t {
    /// Operation completed successfully.
    RGPOT_SUCCESS = 0,
    /// An invalid parameter was passed (null pointer, wrong size, etc.).
    RGPOT_INVALID_PARAMETER = 1,
    /// An internal error occurred (e.g. a Rust panic was caught).
    RGPOT_INTERNAL_ERROR = 2,
    /// An RPC communication error occurred.
    RGPOT_RPC_ERROR = 3,
    /// A buffer was too small for the requested operation.
    RGPOT_BUFFER_SIZE_ERROR = 4,
    /// The self-consistent field of an electronic-structure engine did not
    /// converge. A callback returns it so that the core can tell this
    /// failure from any other; see `rgpot_potential_fused_error_kind`.
    RGPOT_SCC_NOT_CONVERGED = 5,
    /// The potential has no exact Taylor jet.
    RGPOT_NO_JET = 6,
    /// A branch test of a Taylor jet has an enclosure containing its threshold.
    RGPOT_JET_BRANCH = 7,
    /// A Taylor jet left the analytic domain of a function or a division.
    RGPOT_JET_DOMAIN = 8,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Store an error message in the thread-local slot.
pub(crate) fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        let c = match CString::new(msg) {
            Ok(c) => c,
            Err(_) => CString::new("(error message contained interior NUL)").unwrap(),
        };
        *cell.borrow_mut() = c;
    });
}

/// The last error message of the current thread, as text.
pub(crate) fn last_error_message() -> String {
    LAST_ERROR.with(|cell| cell.borrow().to_string_lossy().into_owned())
}

/// Set the error message of the current thread from C or C++.
///
/// A potential callback calls this before it returns a failure status, so
/// that the message travels with the failure. A NULL `msg` clears the
/// message.
///
/// # Safety
///
/// `msg` must be NULL or point to a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn rgpot_set_last_error(msg: *const c_char) {
    if msg.is_null() {
        set_last_error("");
        return;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
    set_last_error(&text);
}

/// Retrieve a pointer to the last error message for the current thread.
///
/// The pointer is valid until the next call to any `rgpot_*` function
/// on the same thread.
///
/// # Safety
/// This is intended to be called from C. The returned pointer must not
/// be freed by the caller.
#[no_mangle]
pub unsafe extern "C" fn rgpot_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

/// Execute a closure, catching any panics and converting them to status codes.
///
/// On success, returns `RGPOT_SUCCESS`. On panic, stores the panic message
/// in the thread-local error slot and returns `RGPOT_INTERNAL_ERROR`.
pub(crate) fn catch_unwind<F>(f: F) -> rgpot_status_t
where
    F: FnOnce() -> rgpot_status_t + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(status) => status,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            set_last_error(&msg);
            rgpot_status_t::RGPOT_INTERNAL_ERROR
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get_last_error() {
        set_last_error("test error");
        let ptr = unsafe { rgpot_last_error() };
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "test error");
    }

    #[test]
    fn test_catch_unwind_success() {
        let status = catch_unwind(|| rgpot_status_t::RGPOT_SUCCESS);
        assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    }

    #[test]
    fn test_catch_unwind_panic() {
        let status = catch_unwind(|| panic!("boom"));
        assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR);
        let ptr = unsafe { rgpot_last_error() };
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "boom");
    }

    #[test]
    fn test_error_overwrite() {
        set_last_error("first");
        set_last_error("second");
        let ptr = unsafe { rgpot_last_error() };
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "second");
    }

    #[test]
    fn test_interior_nul_is_handled() {
        set_last_error("has\0interior nul");
        let ptr = unsafe { rgpot_last_error() };
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        // CString::new fails on interior NUL, so the fallback message is stored.
        assert_eq!(
            msg.to_str().unwrap(),
            "(error message contained interior NUL)"
        );
    }

    #[test]
    fn test_catch_unwind_returns_callback_status() {
        // A closure that returns a non-success status without panicking.
        let status = catch_unwind(|| rgpot_status_t::RGPOT_RPC_ERROR);
        assert_eq!(status, rgpot_status_t::RGPOT_RPC_ERROR);
    }

    #[test]
    fn test_catch_unwind_string_panic() {
        let status = catch_unwind(|| panic!("{}", "formatted panic"));
        assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR);
        let ptr = unsafe { rgpot_last_error() };
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "formatted panic");
    }
}
