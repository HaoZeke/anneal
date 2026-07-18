//! Checked-in package version metadata.

/// Public package version shared by Rust, Python, and C ABI surfaces.
pub const ANNEAL_VERSION: &str = "0.6.0";

/// NUL-terminated package version for C ABI callers.
pub const ANNEAL_VERSION_NUL: &[u8] = b"0.6.0\0";
