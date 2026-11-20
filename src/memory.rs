//! Sizes of buffers whose length the caller chooses.
//!
//! A search over `n` points holds vectors of `3n` or `6n` numbers. The
//! allocator aborts the process when it cannot give one, and a length that
//! overflows `usize` panics in a debug build and wraps in a release build.
//! These functions turn both into a value the caller can report. They set no
//! largest `n`: a request is refused only when the arithmetic overflows or
//! the allocator says no.

use std::collections::TryReserveError;
use std::mem::size_of;

/// State-sized vectors a search holds at once, counted generously.
///
/// The hop loop keeps the current, trial, best and gradient states, the
/// bounds, the start and the probe, and the history grows by whole states.
pub const WORKING_VECTORS: usize = 32;

/// Pair-distance descriptors a search holds at once.
///
/// A basin is recognised by the sorted distances of its `n (n - 1) / 2`
/// pairs, and every basin the bias remembers keeps one. The current, trial
/// and best states, the one being compared and the scratch that sorts it
/// make up this many. The history keeps one more per basin it records, so a
/// long run grows past the working set; the allocator, not this count,
/// decides that.
pub const PAIR_DESCRIPTORS: usize = 8;

/// Why a size was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizeError {
    /// The length does not fit in `usize`.
    Overflow,
    /// The allocator would not give this many bytes.
    Unavailable {
        /// Bytes asked for.
        bytes: usize,
    },
}

impl std::fmt::Display for SizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => f.write_str("the length does not fit in a machine word"),
            Self::Unavailable { bytes } => {
                write!(f, "the allocator would not give {bytes} bytes")
            }
        }
    }
}

impl std::error::Error for SizeError {}

/// Length of a state with `per_point` numbers for each of `n_points`.
pub fn state_len(n_points: usize, per_point: usize) -> Result<usize, SizeError> {
    n_points.checked_mul(per_point).ok_or(SizeError::Overflow)
}

/// Number of pairs among `n_points`.
pub fn pair_count(n_points: usize) -> Result<usize, SizeError> {
    let ordered = n_points
        .checked_mul(n_points.saturating_sub(1))
        .ok_or(SizeError::Overflow)?;
    Ok(ordered / 2)
}

/// Ask the allocator for `count` vectors of `len` numbers at once, then give
/// them back. Nothing is written.
pub fn reserve_vectors(len: usize, count: usize) -> Result<(), SizeError> {
    let bytes = len
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| bytes.checked_mul(count))
        .ok_or(SizeError::Overflow)?;
    let mut probe: Vec<u8> = Vec::new();
    probe
        .try_reserve_exact(bytes)
        .map_err(|_: TryReserveError| SizeError::Unavailable { bytes })
}

/// Ask the allocator for the working set of a search over `n_points` with a
/// state of `len` numbers, then give it back.
///
/// The working set is [`WORKING_VECTORS`] states and [`PAIR_DESCRIPTORS`]
/// descriptors of one number per pair. Nothing is written, so the call costs
/// no more than the allocator takes to answer.
pub fn reserve_working_set(n_points: usize, len: usize) -> Result<(), SizeError> {
    let word = size_of::<f64>();
    let states = len
        .checked_mul(word * WORKING_VECTORS)
        .ok_or(SizeError::Overflow)?;
    let descriptors = pair_count(n_points)?
        .checked_mul(word * PAIR_DESCRIPTORS)
        .ok_or(SizeError::Overflow)?;
    let bytes = states.checked_add(descriptors).ok_or(SizeError::Overflow)?;
    let mut probe: Vec<u8> = Vec::new();
    probe
        .try_reserve_exact(bytes)
        .map_err(|_: TryReserveError| SizeError::Unavailable { bytes })
}

/// A vector of `len` zeros, or the reason it cannot exist.
pub fn try_zeros(len: usize) -> Result<Vec<f64>, SizeError> {
    let mut out: Vec<f64> = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| SizeError::Unavailable {
            bytes: len.saturating_mul(size_of::<f64>()),
        })?;
    out.resize(len, 0.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_length_that_overflows_is_named() {
        assert_eq!(state_len(usize::MAX / 2, 3), Err(SizeError::Overflow));
        assert_eq!(state_len(usize::MAX, 6), Err(SizeError::Overflow));
        assert_eq!(state_len(usize::MAX / 6, 6), Ok(usize::MAX / 6 * 6));
        assert_eq!(state_len(10, 3), Ok(30));
        assert_eq!(state_len(0, 3), Ok(0));
    }

    #[test]
    fn pairs_are_counted_without_overflow() {
        assert_eq!(pair_count(0), Ok(0));
        assert_eq!(pair_count(1), Ok(0));
        assert_eq!(pair_count(5), Ok(10));
        assert_eq!(pair_count(usize::MAX), Err(SizeError::Overflow));
    }

    #[test]
    fn a_working_set_the_allocator_refuses_returns() {
        assert_eq!(reserve_working_set(2, usize::MAX), Err(SizeError::Overflow));
        assert_eq!(reserve_working_set(usize::MAX, 6), Err(SizeError::Overflow));
        assert!(matches!(
            reserve_working_set(1 << 28, 3 << 28),
            Err(SizeError::Unavailable { .. })
        ));
        assert!(matches!(
            reserve_working_set(2, 1 << 52),
            Err(SizeError::Unavailable { .. })
        ));
    }

    #[test]
    fn the_pairs_alone_can_refuse_a_state_that_fits() {
        // The same state is 50 MB of working set for two points. For
        // 2^20 points it would come with 2^39 pairs, which is 32 TiB.
        let len = 3usize << 16;
        reserve_working_set(2, len).unwrap();
        match reserve_working_set(1 << 20, len) {
            Err(SizeError::Unavailable { bytes }) => assert!(bytes > 1 << 44),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_working_set_that_fits_is_given_and_returned() {
        reserve_working_set(0, 0).unwrap();
        reserve_working_set(1000, 3 * 1000).unwrap();
        reserve_working_set(2000, 6 * 2000).unwrap();
    }

    #[test]
    fn vectors_are_reserved_or_refused() {
        reserve_vectors(1000, 2).unwrap();
        reserve_vectors(0, 2).unwrap();
        assert_eq!(reserve_vectors(usize::MAX, 2), Err(SizeError::Overflow));
        assert_eq!(reserve_vectors(1 << 62, 1), Err(SizeError::Overflow));
        assert_eq!(
            reserve_vectors(1 << 52, 2),
            Err(SizeError::Unavailable { bytes: 1 << 56 })
        );
    }

    #[test]
    fn zeros_are_zero_and_an_impossible_length_returns() {
        let zeros = try_zeros(12).unwrap();
        assert_eq!(zeros, vec![0.0; 12]);
        assert!(matches!(
            try_zeros(usize::MAX / 8),
            Err(SizeError::Unavailable { .. })
        ));
        assert!(matches!(
            try_zeros(1 << 52),
            Err(SizeError::Unavailable { .. })
        ));
    }
}
