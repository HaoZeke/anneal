// MIT License
// Copyright 2023--present rgpot developers

//! C entry point for the optional MDI engine frontend.

use std::ffi::c_char;
#[cfg(feature = "mdi")]
use std::ffi::CStr;
#[cfg(feature = "mdi")]
use std::path::PathBuf;

use crate::eindir::rgpot_potential_t;
#[cfg(feature = "mdi")]
use crate::mdi::{serve_mdi, MdiEngineConfig};
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};

/// Message stored by a build compiled without the `mdi` feature.
#[cfg(not(feature = "mdi"))]
const NOT_BUILT: &str = "rgpot_mdi_server_start: this rgpot-core was built without the `mdi` feature (Cargo feature `mdi`, Meson `-Dwith_mdi=true`, CMake `-DRGPOT_WITH_MDI=ON`)";

/// Start an MDI engine for `pot` and block until the driver sends `EXIT`.
///
/// `options` is passed to the official MDI Library, for example
/// `-role ENGINE -name RGPOT -method TCP -hostname localhost -port 8021`.
/// `library_path` may be NULL to use `RGPOT_MDI_LIBRARY` or the platform
/// loader's normal `libmdi` search.
///
/// The symbol exists in every build so that the ABI minor that introduced it
/// is true of every library. A build without the `mdi` feature returns
/// `RGPOT_RPC_ERROR` and stores a message that names the missing feature.
///
/// # Safety
///
/// `pot` must remain valid until this function returns. `options` must point
/// to a NUL-terminated UTF-8 string. `library_path` must be NULL or point to
/// a NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn rgpot_mdi_server_start(
    pot: *const rgpot_potential_t,
    options: *const c_char,
    library_path: *const c_char,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        start(pot, options, library_path)
    }))
}

#[cfg(not(feature = "mdi"))]
unsafe fn start(
    _pot: *const rgpot_potential_t,
    _options: *const c_char,
    _library_path: *const c_char,
) -> rgpot_status_t {
    set_last_error(NOT_BUILT);
    rgpot_status_t::RGPOT_RPC_ERROR
}

#[cfg(feature = "mdi")]
unsafe fn start(
    pot: *const rgpot_potential_t,
    options: *const c_char,
    library_path: *const c_char,
) -> rgpot_status_t {
    if pot.is_null() {
        set_last_error("rgpot_mdi_server_start: pot is NULL");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    if options.is_null() {
        set_last_error("rgpot_mdi_server_start: options is NULL");
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let options = match unsafe { CStr::from_ptr(options) }.to_str() {
        Ok(value) => value,
        Err(error) => {
            set_last_error(&format!("rgpot_mdi_server_start: invalid options: {error}"));
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
    };
    let mut config = MdiEngineConfig::default().with_options(options);
    if !library_path.is_null() {
        let path = match unsafe { CStr::from_ptr(library_path) }.to_str() {
            Ok(value) => value,
            Err(error) => {
                set_last_error(&format!(
                    "rgpot_mdi_server_start: invalid library path: {error}"
                ));
                return rgpot_status_t::RGPOT_INVALID_PARAMETER;
            }
        };
        config = config.with_library_path(PathBuf::from(path));
    }
    match unsafe { serve_mdi(pot, &config) } {
        Ok(_) => rgpot_status_t::RGPOT_SUCCESS,
        Err(error) => {
            set_last_error(&format!("rgpot_mdi_server_start: {error}"));
            rgpot_status_t::RGPOT_RPC_ERROR
        }
    }
}
