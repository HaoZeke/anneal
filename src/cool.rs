//! The cooling-schedule trait of the IISE manuscript: `Cool : N -> R_>0`,
//! non-increasing.

use num_traits::Float;

/// A cooling schedule: maps an epoch index to a positive temperature.
///
/// The IISE manuscript law L4 requires the schedule to be non-increasing in
/// the epoch counter. Implementors override `is_monotone` to advertise this
/// property; the conservative default is `false`.
pub trait Cooling<T: Float>: Send + Sync {
    /// Returns the temperature at the given epoch.
    fn temperature(&self, epoch: usize) -> T;

    /// Witnesses L4: returns `true` iff `temperature` is non-increasing in
    /// `epoch`. Default `true`.
    fn is_monotone(&self) -> bool {
        false
    }
}

/// Boltzmann (logarithmic) cooling: `T(k) = T_0 * log(k0) / log(k + k0)`.
///
/// Strictly decreasing for positive T_0 and k0 greater than one because
/// log(k0) is positive and the argument to the outer log strictly grows.
#[derive(Clone, Debug)]
pub struct LogCool<T: Float> {
    /// Initial temperature `T_0`.
    pub t_init: T,
    /// Logarithmic offset `k0 > 1` controlling the decay rate.
    pub k0: T,
}

impl<T: Float> LogCool<T> {
    /// Constructs a `LogCool` schedule. Requires t_init > 0 and k0 > 1.
    pub fn new(t_init: T, k0: T) -> Self {
        assert!(t_init > T::zero(), "t_init must be positive");
        assert!(k0 > T::one(), "k0 must be > 1");
        Self { t_init, k0 }
    }
}

impl<T: Float + Send + Sync> Cooling<T> for LogCool<T> {
    fn temperature(&self, epoch: usize) -> T {
        let k = T::from(epoch).unwrap_or(T::zero());
        self.t_init * self.k0.ln() / (k + self.k0).ln()
    }

    fn is_monotone(&self) -> bool {
        true
    }
}

/// Reciprocal cooling: `T(k) = T_0 / (k + 1)`.
///
/// The `+1` shift avoids a divide-by-zero at `k = 0` and keeps the schedule
/// well-defined on N. Strictly decreasing for positive T_0.
#[derive(Clone, Debug)]
pub struct ReciprocalCool<T: Float> {
    /// Initial temperature `T_0 = T(0)`.
    pub t_init: T,
}

impl<T: Float> ReciprocalCool<T> {
    /// Constructs a `ReciprocalCool` schedule. Requires t_init > 0.
    pub fn new(t_init: T) -> Self {
        assert!(t_init > T::zero(), "t_init must be positive");
        Self { t_init }
    }
}

impl<T: Float + Send + Sync> Cooling<T> for ReciprocalCool<T> {
    fn temperature(&self, epoch: usize) -> T {
        let k = T::from(epoch).unwrap_or(T::zero());
        self.t_init / (k + T::one())
    }

    fn is_monotone(&self) -> bool {
        true
    }
}

/// Tsallis (GSA) cooling: `T(k) = T_0 * (2^(q_v-1) - 1) / ((1+k)^(q_v-1) - 1)`.
///
/// The case q_v == 1 is a zero-over-zero indeterminate form; an explicit branch
/// returns the L'Hopital limit `T_0 * ln 2 / ln(1+k)` so the schedule remains
/// continuous and finite. Monotone-decreasing for q_v in the open interval
/// (1, 3).
#[derive(Clone, Debug)]
pub struct TsallisCool<T: Float> {
    /// Initial temperature `T_0`.
    pub t_init: T,
    /// Tsallis visiting index. Practical range: `(1, 3)`.
    pub q_v: T,
}

impl<T: Float> TsallisCool<T> {
    /// Constructs a `TsallisCool` schedule. Requires t_init > 0.
    pub fn new(t_init: T, q_v: T) -> Self {
        assert!(t_init > T::zero(), "t_init must be positive");
        assert!(q_v > T::one() && q_v < T::from(3.0).unwrap(), "q_v must lie in (1, 3)");
        Self { t_init, q_v }
    }
}

impl<T: Float + Send + Sync> Cooling<T> for TsallisCool<T> {
    fn temperature(&self, epoch: usize) -> T {
        let k = T::from(epoch).unwrap_or(T::zero());
        let one_plus_k = T::one() + k;
        let two = T::one() + T::one();
        if (self.q_v - T::one()).abs() < T::epsilon() {
            // L'Hopital branch: q_v -> 1 limit is T_0 * ln 2 / ln(1+k).
            // At k=0 this is T_0 * ln 2 / ln 1 = +inf; clamp by returning
            // t_init for k=0 as the natural extension.
            if epoch == 0 {
                return self.t_init;
            }
            self.t_init * two.ln() / one_plus_k.ln()
        } else {
            let exp = self.q_v - T::one();
            let num = two.powf(exp) - T::one();
            let den = one_plus_k.powf(exp) - T::one();
            if epoch == 0 {
                // (1+0)^exp - 1 = 0; natural extension is T_0.
                return self.t_init;
            }
            self.t_init * num / den
        }
    }

    fn is_monotone(&self) -> bool {
        // Strictly decreasing iff (1+k)^(q_v-1) is strictly increasing in k,
        // which holds for q_v > 1. q_v < 1 inverts the schedule.
        self.q_v > T::one()
    }
}
