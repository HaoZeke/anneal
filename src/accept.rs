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

/// Metropolis acceptance: `p = 1` if `delta_e <= 0`, else `exp(-delta_e / T)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Metropolis;

impl<T: Float + Send + Sync> AcceptRule<T> for Metropolis {
    fn accept_prob(&self, delta_e: T, temp: T) -> T {
        if delta_e <= T::zero() {
            T::one()
        } else {
            (-delta_e / temp).exp()
        }
    }
}

/// Tsallis (GSA) acceptance with index `q_a`.
///
/// `p = max(0, 1 - (q_a - 1) * delta_e / T)^(1/(q_a - 1))` for uphill moves.
/// The case `q_a == 1` is the Metropolis limit and is dispatched explicitly
/// to avoid the `0^0`/log indeterminate form.
#[derive(Clone, Copy, Debug)]
pub struct TsallisAccept<T: Float> {
    /// Tsallis acceptance index. `q_a == 1` is the Metropolis limit;
    /// `q_a < 1` rejects fewer uphill moves than Metropolis at fixed `T`.
    pub q_a: T,
}

impl<T: Float> TsallisAccept<T> {
    /// Constructs a Tsallis acceptance rule.
    pub fn new(q_a: T) -> Self {
        Self { q_a }
    }
}

impl<T: Float + Send + Sync> AcceptRule<T> for TsallisAccept<T> {
    fn accept_prob(&self, delta_e: T, temp: T) -> T {
        if delta_e <= T::zero() {
            return T::one();
        }
        if (self.q_a - T::one()).abs() < T::epsilon() {
            return (-delta_e / temp).exp();
        }
        let base = T::one() - (self.q_a - T::one()) * delta_e / temp;
        if base <= T::zero() {
            T::zero()
        } else {
            base.powf(T::one() / (self.q_a - T::one()))
        }
    }
}
