//! C ABI surface. Filled in once the typed algebra lands; the only export
//! today is the package version string for downstream sanity checks.

use std::os::raw::c_char;

/// Returns the anneal-core package version as a NUL-terminated ASCII string.
///
/// The pointer is valid for the entire process lifetime; the string lives in
/// the binary's read-only data segment and is never freed.
#[unsafe(no_mangle)]
pub extern "C" fn anneal_core_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}
