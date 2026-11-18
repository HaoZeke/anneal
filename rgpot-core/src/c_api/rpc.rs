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

#[cfg(all(test, feature = "rpc"))]
mod tests {
    use super::*;
    use crate::tensor::{rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_free};
    use std::ffi::CString;

    fn last_error() -> String {
        let ptr = unsafe { crate::status::rgpot_last_error() };
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn new_rejects_null_host() {
        let client = unsafe { rgpot_rpc_client_new(std::ptr::null(), 9090) };
        assert!(client.is_null());
        assert!(last_error().contains("host is NULL"), "{}", last_error());
    }

    #[test]
    fn new_rejects_invalid_utf8_host() {
        // A byte sequence that is a valid C string (no interior NUL) but not
        // valid UTF-8: a lone continuation byte.
        let bytes = [0x68u8, 0x80, 0x00];
        let host = CString::from_vec_with_nul(bytes.to_vec()).unwrap();
        let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), 9090) };
        assert!(client.is_null());
        assert!(
            last_error().contains("invalid host string"),
            "{}",
            last_error()
        );
    }

    #[test]
    fn new_succeeds_for_a_well_formed_host() {
        // Connection happens lazily: construction never touches the network.
        let host = CString::new("127.0.0.1").unwrap();
        let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), 9090) };
        assert!(!client.is_null());
        unsafe { rgpot_rpc_client_free(client) };
    }

    #[test]
    fn client_free_null_is_noop() {
        unsafe { rgpot_rpc_client_free(std::ptr::null_mut()) };
    }

    fn make_output() -> rgpot_force_out_t {
        rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        }
    }

    #[test]
    fn calculate_rejects_null_client() {
        let input = rgpot_force_input_t {
            positions: std::ptr::null_mut(),
            atomic_numbers: std::ptr::null_mut(),
            box_matrix: std::ptr::null_mut(),
        };
        let mut output = make_output();
        let status = unsafe { rgpot_rpc_calculate(std::ptr::null_mut(), &input, &mut output) };
        assert_eq!(
            status,
            crate::status::rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert!(last_error().contains("client is NULL"), "{}", last_error());
    }

    #[test]
    fn calculate_rejects_null_input() {
        let host = CString::new("127.0.0.1").unwrap();
        let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), 9090) };
        let mut output = make_output();
        let status = unsafe { rgpot_rpc_calculate(client, std::ptr::null(), &mut output) };
        assert_eq!(
            status,
            crate::status::rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert!(last_error().contains("input is NULL"), "{}", last_error());
        unsafe { rgpot_rpc_client_free(client) };
    }

    #[test]
    fn calculate_rejects_null_output() {
        let host = CString::new("127.0.0.1").unwrap();
        let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), 9090) };
        let input = rgpot_force_input_t {
            positions: std::ptr::null_mut(),
            atomic_numbers: std::ptr::null_mut(),
            box_matrix: std::ptr::null_mut(),
        };
        let status = unsafe { rgpot_rpc_calculate(client, &input, std::ptr::null_mut()) };
        assert_eq!(
            status,
            crate::status::rgpot_status_t::RGPOT_INVALID_PARAMETER
        );
        assert!(last_error().contains("output is NULL"), "{}", last_error());
        unsafe { rgpot_rpc_client_free(client) };
    }

    #[test]
    fn calculate_rejects_missing_positions_tensor_before_connecting() {
        // No listener is bound on this port: a positions-tensor failure
        // must be detected (and reported) before any socket is touched,
        // or this test would hang or flake on the connection instead.
        let host = CString::new("127.0.0.1").unwrap();
        let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), 1) };
        assert!(!client.is_null());
        let input = rgpot_force_input_t {
            positions: std::ptr::null_mut(),
            atomic_numbers: std::ptr::null_mut(),
            box_matrix: std::ptr::null_mut(),
        };
        let mut output = make_output();
        let status = unsafe { rgpot_rpc_calculate(client, &input, &mut output) };
        assert_eq!(status, crate::status::rgpot_status_t::RGPOT_RPC_ERROR);
        assert!(
            last_error().contains("cannot determine n_atoms"),
            "{}",
            last_error()
        );
        unsafe { rgpot_rpc_client_free(client) };
    }

    #[test]
    fn calculate_rejects_missing_atomic_numbers_tensor_before_connecting() {
        let host = CString::new("127.0.0.1").unwrap();
        let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), 1) };
        let mut pos = [0.0_f64; 3];
        let pos_t = unsafe { rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), 1, 3) };
        let input = rgpot_force_input_t {
            positions: pos_t,
            atomic_numbers: std::ptr::null_mut(),
            box_matrix: std::ptr::null_mut(),
        };
        let mut output = make_output();
        let status = unsafe { rgpot_rpc_calculate(client, &input, &mut output) };
        assert_eq!(status, crate::status::rgpot_status_t::RGPOT_RPC_ERROR);
        assert!(
            last_error().contains("atomic_numbers tensor is NULL"),
            "{}",
            last_error()
        );
        unsafe {
            rgpot_tensor_free(pos_t);
            rgpot_rpc_client_free(client);
        }
    }

    #[test]
    fn calculate_rejects_missing_box_tensor_before_connecting() {
        let host = CString::new("127.0.0.1").unwrap();
        let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), 1) };
        let mut pos = [0.0_f64; 3];
        let mut atm = [1_i32];
        let pos_t = unsafe { rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), 1, 3) };
        let atm_t = unsafe { crate::tensor::rgpot_tensor_cpu_i32_1d(atm.as_mut_ptr(), 1) };
        let input = rgpot_force_input_t {
            positions: pos_t,
            atomic_numbers: atm_t,
            box_matrix: std::ptr::null_mut(),
        };
        let mut output = make_output();
        let status = unsafe { rgpot_rpc_calculate(client, &input, &mut output) };
        assert_eq!(status, crate::status::rgpot_status_t::RGPOT_RPC_ERROR);
        assert!(
            last_error().contains("box_matrix tensor is NULL"),
            "{}",
            last_error()
        );
        unsafe {
            rgpot_tensor_free(pos_t);
            rgpot_tensor_free(atm_t);
            rgpot_rpc_client_free(client);
        }
    }

    #[test]
    fn calculate_fails_when_nothing_listens_on_the_port() {
        // Port 1 is a privileged, normally-unbound port: connecting there
        // refuses immediately instead of timing out, so this exercises the
        // real `TcpStream::connect` failure path end to end with
        // well-formed tensors.
        let host = CString::new("127.0.0.1").unwrap();
        let client = unsafe { rgpot_rpc_client_new(host.as_ptr(), 1) };
        let mut pos = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut atm = [1_i32, 1];
        let mut cell = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let pos_t = unsafe { rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), 2, 3) };
        let atm_t = unsafe { crate::tensor::rgpot_tensor_cpu_i32_1d(atm.as_mut_ptr(), 2) };
        let box_t = unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) };
        let input = rgpot_force_input_t {
            positions: pos_t,
            atomic_numbers: atm_t,
            box_matrix: box_t,
        };
        let mut output = make_output();
        let status = unsafe { rgpot_rpc_calculate(client, &input, &mut output) };
        assert_eq!(status, crate::status::rgpot_status_t::RGPOT_RPC_ERROR);
        assert!(
            last_error().contains("connection failed"),
            "{}",
            last_error()
        );
        unsafe {
            rgpot_tensor_free(pos_t);
            rgpot_tensor_free(atm_t);
            rgpot_tensor_free(box_t);
            rgpot_rpc_client_free(client);
        }
    }
}
