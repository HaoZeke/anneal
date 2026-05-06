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

/// Tsallis-Stariolo 1996 generalised acceptance with index `q_a`
/// (doi:10.1016/S0378-4371(96)00271-3).
///
/// `p = [1 + (q_a - 1) * delta_e / T]^(1 / (1 - q_a))` for uphill moves.
/// Equivalently, `p = exp_q(-delta_e / T)` where `exp_q` is the Tsallis
/// q-exponential. The case `q_a == 1` is the Metropolis limit
/// (`exp(-delta_e / T)`) and is dispatched explicitly.
///
/// For `q_a > 1` the acceptance is heavy-tailed: at large `delta_e / T`
/// it decays as a power law instead of exponentially, which is why GSA
/// outperforms classical SA on multimodal landscapes -- more uphill
/// acceptance at high `T` enables basin escape. Xiang/Sun/Fan/Gong 1997
/// use the default `q_a = 2.7` (doi:10.1016/S0375-9601(97)00474-X).
/// At fixed `T` and `delta_e > 0`, larger `q_a` gives larger `p`.
///
/// For `q_a < 1` the base can go negative when
/// `delta_e / T > 1 / (1 - q_a)`; this is the compact-support regime
/// of the Tsallis q-exponential and is clamped to zero acceptance,
/// matching Tsallis 1988 Eq.(7) (doi:10.1007/BF01016429).
#[derive(Clone, Copy, Debug)]
pub struct TsallisAccept<T: Float> {
    /// Tsallis acceptance index. `q_a == 1` is the Metropolis limit;
    /// `q_a > 1` is heavy-tailed (accepts more uphill than Metropolis
    /// at fixed `T`); `q_a < 1` is compact-support.
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
        let base = T::one() + (self.q_a - T::one()) * delta_e / temp;
        if base <= T::zero() {
            T::zero()
        } else {
            base.powf(T::one() / (T::one() - self.q_a))
        }
    }
}
