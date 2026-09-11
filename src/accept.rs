//! The acceptance-rule trait of the IISE manuscript: `Accept : R x R_>0 -> [0, 1]`.

use num_traits::Float;

/// Elementwise arithmetic for one acceptance rule on scalars or device arrays.
///
/// Backends preserve their value shape and storage location. Only scalar policy
/// parameters cross this interface; an array need not be copied to host memory.
pub trait ProbabilityArithmetic<T: Float> {
    /// Scalar or array value owned by the backend.
    type Value;
    /// Failure reported by a backend operation.
    type Error;

    /// A scalar broadcast to the backend's value shape.
    fn constant(&self, value: T) -> Result<Self::Value, Self::Error>;
    /// Multiply every entry by a scalar.
    fn scale(&self, value: &Self::Value, factor: T) -> Result<Self::Value, Self::Error>;
    /// Divide every entry by a scalar.
    fn divide(&self, value: &Self::Value, divisor: T) -> Result<Self::Value, Self::Error>;
    /// Add a scalar to every entry.
    fn offset(&self, value: &Self::Value, addend: T) -> Result<Self::Value, Self::Error>;
    /// Elementwise exponential.
    fn exp(&self, value: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Elementwise power with a scalar exponent.
    fn powf(&self, value: &Self::Value, exponent: T) -> Result<Self::Value, Self::Error>;
    /// Select the first value where the condition is nonpositive, else the second.
    fn select_nonpositive(
        &self,
        condition: &Self::Value,
        nonpositive: &Self::Value,
        positive: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// Map positive entries and return a scalar for nonpositive entries.
    ///
    /// Array backends evaluate the map on safe positive replacements for masked
    /// entries, avoiding invalid powers in compact-support acceptance. NaNs are
    /// not classified as nonpositive and retain the rule's nonfinite result.
    fn positive_map<F>(
        &self,
        value: &Self::Value,
        nonpositive: T,
        positive: F,
    ) -> Result<Self::Value, Self::Error>
    where
        F: FnOnce(&Self::Value) -> Result<Self::Value, Self::Error>,
    {
        let one = self.constant(T::one())?;
        let safe = self.select_nonpositive(value, &one, value)?;
        let mapped = positive(&safe)?;
        let other = self.constant(nonpositive)?;
        self.select_nonpositive(value, &other, &mapped)
    }
}

struct ScalarArithmetic<T>(std::marker::PhantomData<T>);

impl<T: Float> ProbabilityArithmetic<T> for ScalarArithmetic<T> {
    type Value = T;
    type Error = std::convert::Infallible;

    fn constant(&self, value: T) -> Result<T, Self::Error> {
        Ok(value)
    }
    fn scale(&self, value: &T, factor: T) -> Result<T, Self::Error> {
        Ok(*value * factor)
    }
    fn divide(&self, value: &T, divisor: T) -> Result<T, Self::Error> {
        Ok(*value / divisor)
    }
    fn offset(&self, value: &T, addend: T) -> Result<T, Self::Error> {
        Ok(*value + addend)
    }
    fn exp(&self, value: &T) -> Result<T, Self::Error> {
        Ok(value.exp())
    }
    fn powf(&self, value: &T, exponent: T) -> Result<T, Self::Error> {
        Ok(value.powf(exponent))
    }
    fn select_nonpositive(
        &self,
        condition: &T,
        nonpositive: &T,
        positive: &T,
    ) -> Result<T, Self::Error> {
        Ok(if *condition <= T::zero() {
            *nonpositive
        } else {
            *positive
        })
    }
    fn positive_map<F>(&self, value: &T, nonpositive: T, positive: F) -> Result<T, Self::Error>
    where
        F: FnOnce(&T) -> Result<T, Self::Error>,
    {
        if *value <= T::zero() {
            Ok(nonpositive)
        } else {
            positive(value)
        }
    }
}

/// A `(delta_e, T) -> p` acceptance rule.
///
/// IISE manuscript laws:
///
/// - L3: downhill moves are accepted with probability one.
/// - L4: for a fixed uphill move, the acceptance probability is non-decreasing
///   in temperature.
///
/// Implementors are responsible for satisfying these contracts; proptest
/// sweeps in tests/laws_proptest.rs witness them at runtime.
pub trait AcceptRule<T: Float>: Send + Sync {
    /// Returns `p in [0, 1]`, the acceptance probability for an uphill move
    /// of size `delta_e` at temperature `temp`.
    fn accept_prob(&self, delta_e: T, temp: T) -> T;
}

/// Metropolis acceptance: probability one for downhill moves, otherwise
/// exp(-delta_e / T).
#[derive(Clone, Copy, Debug, Default)]
pub struct Metropolis;

impl Metropolis {
    /// Evaluate the native rule with backend-owned scalar or array arithmetic.
    pub fn probabilities_with<T: Float, A: ProbabilityArithmetic<T>>(
        &self,
        delta_e: &A::Value,
        temp: T,
        arithmetic: &A,
    ) -> Result<A::Value, A::Error> {
        arithmetic.positive_map(delta_e, T::one(), |delta| {
            let negative = arithmetic.scale(delta, -T::one())?;
            let exponent = arithmetic.divide(&negative, temp)?;
            arithmetic.exp(&exponent)
        })
    }
}

impl<T: Float + Send + Sync> AcceptRule<T> for Metropolis {
    fn accept_prob(&self, delta_e: T, temp: T) -> T {
        let arithmetic = ScalarArithmetic(std::marker::PhantomData);
        match self.probabilities_with(&delta_e, temp, &arithmetic) {
            Ok(value) => value,
            Err(never) => match never {},
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
/// For q_a greater than one the acceptance is heavy-tailed: at large
/// `delta_e / T`
/// it decays as a power law instead of exponentially, which is why GSA
/// outperforms classical SA on multimodal landscapes -- more uphill
/// acceptance at high `T` enables basin escape. Xiang/Sun/Fan/Gong 1997
/// use the default `q_a = 2.7` (doi:10.1016/S0375-9601(97)00474-X).
/// At fixed `T` and positive `delta_e`, larger `q_a` gives larger `p`.
///
/// For q_a less than one the base can go negative when
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

    /// Evaluate the native rule with backend-owned scalar or array arithmetic.
    pub fn probabilities_with<A: ProbabilityArithmetic<T>>(
        &self,
        delta_e: &A::Value,
        temp: T,
        arithmetic: &A,
    ) -> Result<A::Value, A::Error> {
        if (self.q_a - T::one()).abs() < T::epsilon() {
            return Metropolis.probabilities_with(delta_e, temp, arithmetic);
        }
        arithmetic.positive_map(delta_e, T::one(), |delta| {
            let scaled = arithmetic.scale(delta, self.q_a - T::one())?;
            let ratio = arithmetic.divide(&scaled, temp)?;
            let base = arithmetic.offset(&ratio, T::one())?;
            arithmetic.positive_map(&base, T::zero(), |base| {
                arithmetic.powf(base, T::one() / (T::one() - self.q_a))
            })
        })
    }
}

impl<T: Float + Send + Sync> AcceptRule<T> for TsallisAccept<T> {
    fn accept_prob(&self, delta_e: T, temp: T) -> T {
        let arithmetic = ScalarArithmetic(std::marker::PhantomData);
        match self.probabilities_with(&delta_e, temp, &arithmetic) {
            Ok(value) => value,
            Err(never) => match never {},
        }
    }
}
