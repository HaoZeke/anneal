use thiserror::Error;

/// Error type returned by all fallible operations in `anneal-core`.
#[derive(Debug, Error)]
pub enum Error {
    /// Returned when an SA component pairing fails the algebra's
    /// composition laws (L1-L4 of the IISE manuscript).
    #[error("law violation: {0}")]
    LawViolation(String),
    /// Returned when an SA configuration is internally inconsistent
    /// (e.g. epoch counter out of range, temperature non-positive).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}
