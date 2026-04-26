#![warn(missing_docs)]
//! anneal-core: simulated-annealing components on the eindir typed primitives.
//!
//! This crate ships the typed component algebra (Cool / Neigh / Move / Accept
//! traits, Boltzmann / Fast / Tsallis points, and the SaVariant tuple)
//! consumed by the Python `anneal` package via pyo3. v0.3.0 introduces
//! the four trait signatures and the SaVariant scaffold; concrete impls
//! and the run loop land in subsequent commits.

/// The acceptance-rule trait: `(delta_e, T) -> p`.
pub mod accept;
/// Stan-style windowed adaptation: `trait Adapter<T, S>`.
pub mod adapter;
/// The cooling-schedule trait: `epoch -> temperature`.
pub mod cool;
/// Error variants returned by `anneal-core`.
pub mod error;
/// Per-epoch run history returned by `run_rs`.
pub mod history;
/// Law-witness helpers and the `LawViolation` diagnostic type.
pub mod laws;
/// The move-kernel trait: temperature-indexed proposal sampling.
pub mod movekernel;
/// The neighborhood trait: `state -> set-of-states`.
pub mod neigh;
/// Annealing-method extensions built on the typed algebra:
/// MCMC-style multi-chain SA with Gelman-Rubin termination.
pub mod methods;
/// The pure-Rust SA driver loop.
pub mod runner;
/// Stan-style single-step sampler trait: `trait Sampler<T>`.
pub mod sampler;
/// Reserved for the typed component algebra (Spec 2, v0.3.0).
pub mod types;
/// The fully-typed `SaVariant` tuple satisfying L1-L4.
pub mod variant;

/// C ABI surface (cargo-c builds).
#[cfg(feature = "capi")]
pub mod ffi;
/// pyo3 module entry point exposed as `anneal._core`.
#[cfg(feature = "python")]
pub mod python;

pub use accept::AcceptRule;
pub use adapter::{Adapter, IdentityAdapter};
pub use cool::Cooling;
pub use error::Error;
pub use history::{EpochLine, History, State};
pub use laws::LawViolation;
pub use movekernel::MoveKernel;
pub use neigh::Neighborhood;
pub use methods::{GelmanRubin, MultiChainResult, MultiChainSampler, MultiChainState};
pub use runner::{run_rs, run_rs_variant};
pub use sampler::Sampler;
pub use variant::SaVariant;
