//! Package version read from the workspace Cargo.toml at compile time.

/// Public package version shared by Rust, Python, and the C ABI.
pub const ANNEAL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// NUL-terminated package version for C ABI callers.
pub const ANNEAL_VERSION_NUL: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
