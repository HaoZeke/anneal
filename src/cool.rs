//! The cooling-schedule trait of the IISE manuscript: `Cool : N -> R_>0`,
//! non-increasing.

use num_traits::Float;

/// A cooling schedule: maps an epoch index to a positive temperature.
///
/// The IISE manuscript law L4 requires the schedule to be non-increasing in
/// the epoch counter. Implementors should override `is_monotone` to
/// advertise this property; the default is `true` because every shipped
/// schedule satisfies L4 by construction.
pub trait Cooling<T: Float>: Send + Sync {
    /// Returns the temperature at the given epoch.
    fn temperature(&self, epoch: usize) -> T;

    /// Witnesses L4: returns `true` iff `temperature` is non-increasing in
    /// `epoch`. Default `true`.
    fn is_monotone(&self) -> bool {
        true
    }
}
