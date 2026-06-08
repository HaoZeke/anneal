//! Gradient support for anneal drivers (HMC/Method B, projected-gradient
//! polish, GLE-Langevin, ...).
//!
//! The canonical `Gradient`, `DifferentiableObjective`, `AnalyticGradient`,
//! and `FiniteDiffGradient` live in `eindir_core`, the typed ND objective
//! primitives that anneal uses. This module re-exports them so
//! `anneal_core::grad::*` imports share one gradient contract.
//!
//! Use `eindir_core::gradient` directly when building objectives outside
//! `anneal_core`.

// Re-export the shared typed primitives so anneal drivers and downstream
// callers use the same derivative traits.
pub use eindir_core::{
    gradient::{AnalyticGradient, FiniteDiffGradient},
    DifferentiableObjective, Gradient,
};

// Objective is part of the same public surface as Gradient.
pub use eindir_core::Objective;
