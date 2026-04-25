//! The acceptance-rule trait of the IISE manuscript: `Accept : R x R_>0 -> [0, 1]`.

use num_traits::Float;

/// A `(delta_e, T) -> p` acceptance rule.
///
/// IISE manuscript laws:
///   L3: `accept_prob(delta_e, T) = 1` when `delta_e <= 0` (downhill always accepts).
///   L4: `T -> accept_prob(delta_e, T)` is non-decreasing for fixed `delta_e > 0`.
///
/// Implementors are responsible for satisfying these contracts; proptest
/// sweeps in tests/laws_proptest.rs witness them at runtime.
pub trait AcceptRule<T: Float>: Send + Sync {
    /// Returns `p in [0, 1]`, the acceptance probability for an uphill move
    /// of size `delta_e` at temperature `temp`.
    fn accept_prob(&self, delta_e: T, temp: T) -> T;
}
