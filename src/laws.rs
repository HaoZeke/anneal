//! Runtime witness helpers and the `LawViolation` error returned by
//! `SaVariant::checked` when a tuple of components fails one of the four
//! IISE-manuscript composition laws.

use thiserror::Error;

/// A law-violation diagnostic surfaced by `SaVariant::checked`.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LawViolation {
    /// L1: `is_symmetric()` returned false on the supplied `Neighborhood`.
    #[error("L1 violation: neighborhood is not symmetric")]
    Symmetry,
    /// L2: `MoveKernel::supports_in(neigh)` returned false.
    #[error("L2 violation: move support escapes the neighborhood")]
    SupportEscape,
    /// L4: `Cooling::is_monotone()` returned false on the supplied `Cooling`.
    #[error("L4 violation: cooling schedule is not non-increasing")]
    NonMonotoneCooling,
}
