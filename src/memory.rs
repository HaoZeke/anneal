//! Sizes of buffers whose length the caller chooses.
//!
//! A search over `n` points holds vectors of `3n` or `6n` numbers. The
//! allocator aborts the process when it cannot give one, and a length that
//! overflows `usize` panics in a debug build and wraps in a release build.
//! These functions turn both into a value the caller can report. They set no
//! largest `n`: a request is refused only when the arithmetic overflows or
//! the allocator says no. What a search needs is computed from what it
//! allocates ([`first_hop_bytes`]), not from a round number.

use std::collections::TryReserveError;
use std::mem::size_of;

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

/// The vectors the first hop of a cluster search holds, by owner.
///
/// Every count is a number of separate vectors, one entry for each place in
/// the search that keeps one alive at the same time, and the sum is what
/// [`first_hop_bytes`] multiplies by a state or a pair descriptor. The
/// first hop is the opening quench and one perturb, quench and compare, so
/// it is also where the quasi-Newton memory is full, two minima are
/// registered and the pair-distance table has been asked about three
/// states. `tests/first_hop_allocation.rs` runs that hop under a counting
/// allocator and fails when the bytes below are fewer than the bytes held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FirstHop {
    /// Vectors of one state length (`3n`, or `6n` for rigid bodies).
    pub states: usize,
    /// Vectors of one number per pair of points.
    pub descriptors: usize,
}

/// Minima registered by the end of the first hop: the one the opening quench
/// ends in and the one the first trial ends in.
const MINIMA: usize = 2;
/// States the pair-distance table is asked about by then: the opening
/// minimum, the screened trial and the fully relaxed trial.
const DESCRIBED: usize = 3;
/// Vectors a quench works on beside its curvature pairs: the iterate, its
/// gradient, the direction, the line-search probe, the gradient at the probe
/// and the pair `s`, `y` being formed.
const QUENCH_WORKING: usize = 7;
/// Vectors the hop loop and the search wrapper keep: the occupied state, the
/// trial, the screened trial, the canonical reference, the ledger's best,
/// the bits of the held state, the start, the two sides of the box, the
/// validation gradient, the paid-for anchor and the gradient behind it.
const CHAIN_HELD: usize = 13;
/// States of an accepted transition: source and destination, each with the
/// gradient retained for validation.
const TRANSITION_STATES: usize = 4;
/// Quench boundaries recorded by the first hop, each with a state and a
/// gradient.
const BOUNDARIES: usize = 2;
/// Pair vectors one description makes: the unsorted distances, the sorted
/// copy and the array it returns.
const DESCRIBE_SCRATCH: usize = 3;
/// Pair vectors the hop loop holds beside the table: the descriptor of the
/// occupied state, of the trial and of the state the contextual model kept.
const DESCRIPTORS_HELD: usize = 3;

/// Bytes the search holds whatever `n` is: the arm statistics, the
/// contextual models of the moves, the move kernels, the copies of the
/// `Config` (1.3 kB each) and the thread that runs the chain. The first hop
/// of every preset at three and four points holds 8 to 21 kB in all, and
/// most of that is this.
pub const BOOKKEEPING_BYTES: usize = 32 * 1024;

impl FirstHop {
    /// The counts, from the constants of the code that holds the vectors.
    ///
    /// The curvature memory is the `max_pairs` of the quench the search
    /// builds, two vectors a pair. The pair-distance table has
    /// [`crate::bias::PAIR_SLOTS`] slots and at most [`DESCRIBED`] of them
    /// are filled by the first hop, each owning the unsorted and the sorted
    /// distances and the bits of the state.
    pub fn counted() -> Self {
        let memory = crate::methods::warm_lbfgs::WarmLbfgs::default().max_pairs;
        let slots = DESCRIBED.min(crate::bias::PAIR_SLOTS);
        Self {
            states: 2 * memory
                + QUENCH_WORKING
                + CHAIN_HELD
                + TRANSITION_STATES
                + 2 * BOUNDARIES
                + slots,
            descriptors: 2 * slots + DESCRIBE_SCRATCH + DESCRIPTORS_HELD + MINIMA,
        }
    }
}

/// Bytes the first hop of a search over `n_points` holds, with a state of
/// `len` numbers.
///
/// The sum of what [`FirstHop`] counts, each vector at its length, and of
/// the structures that grow with `n_points` or `len` in their own right:
/// the headers of the table's slots, the curvature pairs, the records of the
/// first hop, and the table the symmetry move sorts, which has an entry for
/// every atom and site pair and a scratch buffer as large. The bookkeeping
/// that does not depend on `n` is [`BOOKKEEPING_BYTES`].
pub fn first_hop_bytes(n_points: usize, len: usize) -> Result<usize, SizeError> {
    use crate::methods::cluster_hopping::{AcceptedTransition, QuenchBoundary};
    use std::mem::size_of;

    let word = size_of::<f64>();
    let held = FirstHop::counted();
    let state = len.checked_mul(word).ok_or(SizeError::Overflow)?;
    let descriptor = pair_count(n_points)?
        .checked_mul(word)
        .ok_or(SizeError::Overflow)?;
    let max_pairs = crate::methods::warm_lbfgs::WarmLbfgs::default().max_pairs;
    let curvature_pair = 2 * size_of::<ndarray::Array1<f64>>() + word;
    let structures = (max_pairs + 1).next_power_of_two() * curvature_pair
        + DESCRIBED.min(crate::bias::PAIR_SLOTS) * crate::bias::PAIR_SLOT_BYTES
        + 4 * size_of::<AcceptedTransition>()
        + 4 * size_of::<QuenchBoundary>();
    let assignments = n_points
        .checked_mul(n_points)
        .and_then(|table| table.checked_mul(crate::symmetrise::ASSIGNMENT_BYTES))
        .and_then(|table| table.checked_mul(2))
        .ok_or(SizeError::Overflow)?;
    held.states
        .checked_mul(state)
        .and_then(|bytes| bytes.checked_add(held.descriptors.checked_mul(descriptor)?))
        .and_then(|bytes| bytes.checked_add(assignments))
        .and_then(|bytes| bytes.checked_add(structures))
        .and_then(|bytes| bytes.checked_add(BOOKKEEPING_BYTES))
        .ok_or(SizeError::Overflow)
}

/// Ask the allocator for the first hop of a search over `n_points` with a
/// state of `len` numbers, then give it back.
///
/// The size is [`first_hop_bytes`]. Nothing is written, so the call costs
/// no more than the allocator takes to answer.
pub fn reserve_working_set(n_points: usize, len: usize) -> Result<(), SizeError> {
    let bytes = first_hop_bytes(n_points, len)?;
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
    fn the_first_hop_counts_come_from_the_constants_of_the_code() {
        let held = FirstHop::counted();
        let memory = crate::methods::warm_lbfgs::WarmLbfgs::default().max_pairs;
        let slots = 3.min(crate::bias::PAIR_SLOTS);
        assert!(held.states >= 2 * memory + slots);
        assert!(held.descriptors >= 2 * slots);
    }

    #[test]
    fn the_first_hop_grows_with_the_square_of_the_points() {
        let small = first_hop_bytes(10_000, 30_000).unwrap() - BOOKKEEPING_BYTES;
        let large = first_hop_bytes(20_000, 60_000).unwrap() - BOOKKEEPING_BYTES;
        let ratio = large as f64 / small as f64;
        assert!((3.9..4.1).contains(&ratio), "{ratio}");
    }

    #[test]
    fn the_first_hop_of_a_size_that_overflows_is_named() {
        assert_eq!(first_hop_bytes(usize::MAX, 6), Err(SizeError::Overflow));
        assert_eq!(first_hop_bytes(2, usize::MAX), Err(SizeError::Overflow));
        assert_eq!(first_hop_bytes(1 << 32, 3 << 32), Err(SizeError::Overflow));
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
