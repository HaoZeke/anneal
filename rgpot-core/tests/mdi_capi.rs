// MIT License
// Copyright 2023--present rgpot developers

//! `rgpot_mdi_server_start` as a C host reaches it. ABI minor 9 promises the
//! symbol, so it links in every build; a build without the `mdi` feature
//! answers with a status and a message that names the feature.

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

use rgpot_core::c_api::mdi::rgpot_mdi_server_start;
use rgpot_core::c_api::potential::{rgpot_potential_free, rgpot_potential_new};
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

unsafe extern "C" fn unused_potential(
    _user_data: *mut c_void,
    _input: *const rgpot_force_input_t,
    _output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    rgpot_status_t::RGPOT_SUCCESS
}

fn last_error() -> String {
    let message = unsafe { rgpot_last_error() };
    assert!(!message.is_null(), "no error message was stored");
    unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(not(feature = "mdi"))]
#[test]
fn a_build_without_mdi_refuses_with_a_message_naming_the_feature() {
    let options = c"-role ENGINE -name RGPOT -method TCP -hostname localhost -port 8021";
    let pot = unsafe { rgpot_potential_new(unused_potential, ptr::null_mut(), None) };
    assert!(!pot.is_null());
    let status = unsafe { rgpot_mdi_server_start(pot, options.as_ptr(), ptr::null()) };
    assert_eq!(status, rgpot_status_t::RGPOT_RPC_ERROR);
    let message = last_error();
    assert!(
        message.starts_with("rgpot_mdi_server_start:")
            && message.contains("built without the `mdi` feature"),
        "unexpected message: {message}"
    );
    unsafe { rgpot_potential_free(pot) };
}

#[cfg(not(feature = "mdi"))]
#[test]
fn a_build_without_mdi_refuses_null_arguments_the_same_way() {
    let status = unsafe { rgpot_mdi_server_start(ptr::null(), ptr::null::<c_char>(), ptr::null()) };
    assert_eq!(status, rgpot_status_t::RGPOT_RPC_ERROR);
    assert!(last_error().contains("built without the `mdi` feature"));
}

#[cfg(feature = "mdi")]
#[test]
fn a_null_handle_or_options_string_is_an_invalid_parameter() {
    let options = c"-role ENGINE -name RGPOT -method TCP -hostname localhost -port 8021";
    let status = unsafe { rgpot_mdi_server_start(ptr::null(), options.as_ptr(), ptr::null()) };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert!(last_error().starts_with("rgpot_mdi_server_start:"));

    let pot = unsafe { rgpot_potential_new(unused_potential, ptr::null_mut(), None) };
    assert!(!pot.is_null());
    let status = unsafe { rgpot_mdi_server_start(pot, ptr::null::<c_char>(), ptr::null()) };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert!(last_error().contains("options is NULL"));
    unsafe { rgpot_potential_free(pot) };
}

#[cfg(feature = "mdi")]
#[test]
fn a_library_that_cannot_load_is_an_rpc_error() {
    let options = c"-role ENGINE -name RGPOT -method MPI";
    let library = c"/nonexistent/rgpot/libmdi.so";
    let pot = unsafe { rgpot_potential_new(unused_potential, ptr::null_mut(), None) };
    assert!(!pot.is_null());
    let status = unsafe { rgpot_mdi_server_start(pot, options.as_ptr(), library.as_ptr()) };
    assert_eq!(status, rgpot_status_t::RGPOT_RPC_ERROR);
    let message = last_error();
    assert!(
        message.starts_with("rgpot_mdi_server_start:")
            && message.contains("could not load the official MDI Library"),
        "unexpected message: {message}"
    );
    unsafe { rgpot_potential_free(pot) };
}
