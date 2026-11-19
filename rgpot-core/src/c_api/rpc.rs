// MIT License
// Copyright 2023--present rgpot developers

//! C API for the RPC client.
//!
//! Provides `rgpot_rpc_client_t` — an opaque handle wrapping the async
//! Cap'n Proto RPC client — and functions to connect, calculate, and
//! disconnect.

#[cfg(feature = "rpc")]
use std::os::raw::c_char;

#[cfg(feature = "rpc")]
use crate::rpc::client::RpcClient;
#[cfg(feature = "rpc")]
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
#[cfg(feature = "rpc")]
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Opaque RPC client handle.
#[cfg(feature = "rpc")]
pub type rgpot_rpc_client_t = RpcClient;

/// Create a new RPC client connected to `host:port`.
///
/// Returns a heap-allocated handle, or `NULL` on failure.
/// The caller must eventually call `rgpot_rpc_client_free`.
///
/// # Safety
///
/// `host` must be null or a valid NUL-terminated string. A non-null return
/// is owned by the caller and freed with `rgpot_rpc_client_free`.
#[cfg(feature = "rpc")]
#[no_mangle]
pub unsafe extern "C" fn rgpot_rpc_client_new(
    host: *const c_char,
    port: u16,
) -> *mut rgpot_rpc_client_t {
    if host.is_null() {
        set_last_error("rgpot_rpc_client_new: host is NULL");
        return std::ptr::null_mut();
    }

    let host_str = match unsafe { std::ffi::CStr::from_ptr(host) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(&format!("rgpot_rpc_client_new: invalid host string: {e}"));
            return std::ptr::null_mut();
        }
    };

    match RpcClient::new(&host_str, port) {
        Ok(client) => Box::into_raw(Box::new(client)),
        Err(e) => {
            set_last_error(&format!("rgpot_rpc_client_new: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Perform a remote force/energy calculation.
///
/// Returns `RGPOT_SUCCESS` on success, or an error status code.
///
/// # Safety
///
/// `client` must be null or a pointer from `rgpot_rpc_client_new`. `input`
/// and `output` must be null or point at live structs whose tensors, when
/// non-null, stay valid for the call.
#[cfg(feature = "rpc")]
#[no_mangle]
pub unsafe extern "C" fn rgpot_rpc_calculate(
    client: *mut rgpot_rpc_client_t,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if client.is_null() {
            set_last_error("rgpot_rpc_calculate: client is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if input.is_null() {
            set_last_error("rgpot_rpc_calculate: input is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        if output.is_null() {
            set_last_error("rgpot_rpc_calculate: output is NULL");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }

        let client_ref = unsafe { &mut *client };
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };

        match client_ref.calculate(inp, out) {
            Ok(()) => rgpot_status_t::RGPOT_SUCCESS,
            Err(e) => {
                set_last_error(&format!("rgpot_rpc_calculate: {e}"));
                rgpot_status_t::RGPOT_RPC_ERROR
            }
        }
    }))
}

/// Compute an analytic Hessian-vector product through the remote potential.
///
/// `direction` and `product` each hold `3 * n_atoms` doubles.
///
/// # Safety
///
/// `client` must be a live pointer from `rgpot_rpc_client_new`.
/// `input` must point to a valid `rgpot_force_input_t`.
/// `direction` and `product` must each point to arrays of at least
/// `3 * n_atoms` doubles, where `n_atoms` comes from `input`.
#[cfg(feature = "rpc")]
#[no_mangle]
pub unsafe extern "C" fn rgpot_rpc_calculate_hvp(
    client: *mut rgpot_rpc_client_t,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    product: *mut f64,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if client.is_null() || input.is_null() || direction.is_null() || product.is_null() {
            set_last_error("rgpot_rpc_calculate_hvp: NULL argument");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let inp = unsafe { &*input };
        let Some(dimension) = unsafe { inp.n_atoms() }.and_then(|n| n.checked_mul(3)) else {
            set_last_error("rgpot_rpc_calculate_hvp: invalid positions tensor");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let direction = unsafe { std::slice::from_raw_parts(direction, dimension) };
        match unsafe { &mut *client }.calculate_hvp(inp, direction) {
            Ok(values) => {
                unsafe {
                    std::ptr::copy_nonoverlapping(values.as_ptr(), product, dimension);
                }
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(error) => {
                set_last_error(&format!("rgpot_rpc_calculate_hvp: {error}"));
                rgpot_status_t::RGPOT_RPC_ERROR
            }
        }
    }))
}

/// Free an RPC client handle.
///
/// If `client` is `NULL`, this is a no-op.
///
/// # Safety
///
/// `client` must be null or a pointer from `rgpot_rpc_client_new` that has
/// not already been freed.
#[cfg(feature = "rpc")]
#[no_mangle]
pub unsafe extern "C" fn rgpot_rpc_client_free(client: *mut rgpot_rpc_client_t) {
    if !client.is_null() {
        drop(unsafe { Box::from_raw(client) });
    }
}
