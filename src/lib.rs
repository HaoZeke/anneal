#![warn(missing_docs)]
//! anneal-core: simulated-annealing components on the eindir typed primitives.
//!
//! This crate ships the typed component algebra (Cool / Neigh / Move / Accept
//! traits, Boltzmann / Fast / Tsallis points, and the SaVariant tuple)
//! consumed by the Python `anneal` package via pyo3. The current revision is
//! a structural scaffold; the typed component algebra lands in v0.3.0.

/// Error variants returned by `anneal-core`.
pub mod error;
/// Reserved for the typed component algebra (Spec 2, v0.3.0).
pub mod types;
/// C ABI surface (cargo-c builds).
#[cfg(feature = "capi")]
pub mod ffi;
/// pyo3 module entry point exposed as `anneal._core`.
#[cfg(feature = "python")]
pub mod python;

pub use error::Error;
