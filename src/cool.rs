//! The cooling-schedule trait of the IISE manuscript: `Cool : N -> R_>0`,
//! non-increasing.

use std::any::TypeId;
use std::cell::Cell;

use num_traits::Float;

/// One remembered `f64` temperature.
///
/// An epoch holds the temperature fixed for every proposal, and `pow` / `ln`
/// are not pure to LLVM, so it will not lift them out of that loop. The key
/// is the schedule parameters themselves: a different temperature, a different
/// offset, or a mutated schedule misses and recomputes.
#[derive(Clone, Copy)]
struct F64EpochMemo {
    a: u64,
    b: u64,
    epoch: usize,
    bits: u64,
    valid: bool,
}

impl F64EpochMemo {
    const EMPTY: Self = Self {
        a: 0,
        b: 0,
        epoch: 0,
        bits: 0,
        valid: false,
    };
}

thread_local! {
    static LOG_COOL_MEMO: Cell<F64EpochMemo> = const { Cell::new(F64EpochMemo::EMPTY) };
    static TSALLIS_COOL_MEMO: Cell<F64EpochMemo> = const { Cell::new(F64EpochMemo::EMPTY) };
}

fn read_f64<T: 'static>(value: &T) -> f64 {
    debug_assert_eq!(TypeId::of::<T>(), TypeId::of::<f64>());
    // SAFETY: the caller has established `T` is `f64`.
    unsafe { *(value as *const T as *const f64) }
}

fn write_f64<T: Copy + 'static>(value: f64) -> T {
    debug_assert_eq!(TypeId::of::<T>(), TypeId::of::<f64>());
    // SAFETY: the caller has established `T` is `f64`, which is `Copy`.
    unsafe { *(&value as *const f64 as *const T) }
}

/// Cached `f64` temperature for `(a, b, epoch)`, or `None` when `T` is not
/// `f64` or this key has not been evaluated yet.
fn memoized_f64_epoch<T: Float + 'static>(
    slot: &'static std::thread::LocalKey<Cell<F64EpochMemo>>,
    a: &T,
    b: &T,
    epoch: usize,
) -> Option<T> {
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        return None;
    }
    let a_bits = read_f64(a).to_bits();
    let b_bits = read_f64(b).to_bits();
    slot.with(|cell| {
        let hit = cell.get();
        if hit.valid && hit.a == a_bits && hit.b == b_bits && hit.epoch == epoch {
            Some(write_f64(f64::from_bits(hit.bits)))
        } else {
            None
        }
    })
}

fn store_f64_epoch<T: Float + 'static>(
    slot: &'static std::thread::LocalKey<Cell<F64EpochMemo>>,
    a: &T,
    b: &T,
    epoch: usize,
    value: &T,
) {
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        return;
    }
    slot.with(|cell| {
        cell.set(F64EpochMemo {
            a: read_f64(a).to_bits(),
            b: read_f64(b).to_bits(),
            epoch,
            bits: read_f64(value).to_bits(),
            valid: true,
        });
    });
}

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

fn log_cool_value<T: Float>(t_init: T, k0: T, epoch: usize) -> T {
    let k = T::from(epoch).unwrap_or(T::zero());
    t_init * k0.ln() / (k + k0).ln()
}

impl<T: Float + Send + Sync + 'static> Cooling<T> for LogCool<T> {
    #[inline(always)]
    fn temperature(&self, epoch: usize) -> T {
        if let Some(cached) = memoized_f64_epoch(&LOG_COOL_MEMO, &self.t_init, &self.k0, epoch) {
            return cached;
        }
        let value = log_cool_value(self.t_init, self.k0, epoch);
        store_f64_epoch(&LOG_COOL_MEMO, &self.t_init, &self.k0, epoch, &value);
        value
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
    #[inline(always)]
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
        assert!(
            q_v > T::one() && q_v < T::from(3.0).unwrap(),
            "q_v must lie in (1, 3)"
        );
        Self { t_init, q_v }
    }
}

fn tsallis_cool_value<T: Float>(t_init: T, q_v: T, epoch: usize) -> T {
    let k = T::from(epoch).unwrap_or(T::zero());
    let one_plus_k = T::one() + k;
    let two = T::one() + T::one();
    if (q_v - T::one()).abs() < T::epsilon() {
        // L'Hopital branch: q_v -> 1 limit is T_0 * ln 2 / ln(1+k).
        // At k=0 this is T_0 * ln 2 / ln 1 = +inf; clamp by returning
        // t_init for k=0 as the natural extension.
        if epoch == 0 {
            return t_init;
        }
        t_init * two.ln() / one_plus_k.ln()
    } else {
        let exp = q_v - T::one();
        let num = two.powf(exp) - T::one();
        let den = one_plus_k.powf(exp) - T::one();
        if epoch == 0 {
            // (1+0)^exp - 1 = 0; natural extension is T_0.
            return t_init;
        }
        t_init * num / den
    }
}

impl<T: Float + Send + Sync + 'static> Cooling<T> for TsallisCool<T> {
    #[inline(always)]
    fn temperature(&self, epoch: usize) -> T {
        if let Some(cached) = memoized_f64_epoch(&TSALLIS_COOL_MEMO, &self.t_init, &self.q_v, epoch)
        {
            return cached;
        }
        let value = tsallis_cool_value(self.t_init, self.q_v, epoch);
        store_f64_epoch(&TSALLIS_COOL_MEMO, &self.t_init, &self.q_v, epoch, &value);
        value
    }

    fn is_monotone(&self) -> bool {
        // Strictly decreasing iff (1+k)^(q_v-1) is strictly increasing in k,
        // which holds for q_v > 1. q_v < 1 inverts the schedule.
        self.q_v > T::one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_tsallis(t_init: f64, q_v: f64, epoch: usize) -> f64 {
        tsallis_cool_value(t_init, q_v, epoch)
    }

    fn fresh_log(t_init: f64, k0: f64, epoch: usize) -> f64 {
        log_cool_value(t_init, k0, epoch)
    }

    #[test]
    fn memoized_temperatures_match_the_formula_and_notice_a_new_schedule() {
        let tsallis = TsallisCool::new(5.0_f64, 2.62);
        for epoch in [0usize, 1, 2, 7, 49] {
            let first: f64 = tsallis.temperature(epoch);
            let second: f64 = tsallis.temperature(epoch);
            assert_eq!(first.to_bits(), fresh_tsallis(5.0, 2.62, epoch).to_bits());
            assert_eq!(first.to_bits(), second.to_bits());
        }
        let mut edited = TsallisCool::new(5.0_f64, 2.62);
        let before: f64 = edited.temperature(4);
        edited.t_init = 8.0;
        let after: f64 = edited.temperature(4);
        assert_ne!(before.to_bits(), after.to_bits());
        assert_eq!(after.to_bits(), fresh_tsallis(8.0, 2.62, 4).to_bits());

        let log = LogCool::new(5.0_f64, 2.0);
        for epoch in [0usize, 1, 11, 40] {
            let value: f64 = log.temperature(epoch);
            assert_eq!(value.to_bits(), fresh_log(5.0, 2.0, epoch).to_bits());
        }
    }
}
