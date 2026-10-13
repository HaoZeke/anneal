//! Run history: per-epoch summary and full trajectory record returned by
//! the `run_rs` driver loop.

use eindir_core::FPair;

/// One row of the run history: summary statistics for an epoch.
#[derive(Clone, Debug)]
pub struct EpochLine {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Temperature at this epoch (the cooling-schedule output).
    pub temp: f64,
    /// Number of proposals accepted in this epoch.
    pub accepted: usize,
    /// Number of proposals rejected in this epoch.
    pub rejected: usize,
    /// Best objective value seen up to and including this epoch.
    pub best_val: f64,
}

impl EpochLine {
    /// Acceptance ratio in this epoch (`accepted / (accepted + rejected)`),
    /// or 0 if the epoch had no proposals.
    pub fn accept_ratio(&self) -> f64 {
        let total = self.accepted + self.rejected;
        if total == 0 {
            0.0
        } else {
            self.accepted as f64 / total as f64
        }
    }
}

/// Whether the SA driver accepted or rejected the most recent proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcceptKind {
    /// Downhill move (`delta_e <= 0`); accepted with probability 1 (L3).
    AcceptedDownhill,
    /// Uphill move accepted by the Boltzmann/Tsallis criterion.
    AcceptedUphill,
    /// Uphill move rejected.
    Rejected,
}

/// Internal driver state: current position and best-seen position.
#[derive(Clone, Debug)]
pub struct State {
    /// Current position and its objective value.
    pub cur: FPair<f64>,
    /// Best position seen so far and its objective value.
    pub best: FPair<f64>,
}

/// Run history returned by `run_rs`: per-epoch summary lines plus the
/// best-seen `(pos, val)` pair across the full run.
#[derive(Clone, Debug)]
pub struct History {
    /// One `EpochLine` per epoch, in epoch order.
    pub epochs: Vec<EpochLine>,
    /// Best `(position, value)` pair seen across the entire run.
    pub best: FPair<f64>,
    /// Robust rolling mean-shift diagnostic, one flag per epoch.
    pub stationarity_flags: Vec<bool>,
}

impl History {
    /// Constructs an empty `History` with the given epoch capacity.
    pub fn with_capacity(n_epochs: usize, best: FPair<f64>) -> Self {
        Self {
            epochs: Vec::with_capacity(n_epochs),
            best,
            stationarity_flags: Vec::with_capacity(n_epochs),
        }
    }

    /// Whether the allocator grants together the vectors
    /// [`History::with_capacity`] reserves for each count of epochs in
    /// `histories`, as for that many histories held at once: each
    /// reservation is held while the next is tried, all are freed on
    /// return, and a refusal returns `false` instead of aborting.
    pub fn can_reserve(histories: &[usize]) -> bool {
        let Some((&n_epochs, rest)) = histories.split_first() else {
            return true;
        };
        let mut lines = Vec::<EpochLine>::new();
        let mut flags = Vec::<bool>::new();
        if lines.try_reserve_exact(n_epochs).is_err() || flags.try_reserve_exact(n_epochs).is_err()
        {
            return false;
        }
        // An allocation nothing reads may be dropped by the optimizer, its
        // success taken for granted; this keeps each one asked of the allocator.
        std::hint::black_box((&lines, &flags));
        Self::can_reserve(rest)
    }

    /// Recomputes the per-epoch non-stationarity flags from epoch summaries,
    /// in the room the flags hold: no copy of the values is made, and the
    /// flags take no new room while it holds one per epoch.
    pub fn refresh_stationarity_flags(&mut self) {
        stationarity_flags_into(
            &self.epochs,
            |epoch| epoch.best_val,
            &mut self.stationarity_flags,
        );
    }

    /// Total proposals accepted across all epochs.
    pub fn total_accepted(&self) -> usize {
        self.epochs.iter().map(|e| e.accepted).sum()
    }

    /// Total proposals rejected across all epochs.
    pub fn total_rejected(&self) -> usize {
        self.epochs.iter().map(|e| e.rejected).sum()
    }
}

/// Flags sustained local mean shifts in a scalar trajectory.
///
/// The diagnostic compares equal-sized windows on either side of each point.
/// A three-standard-deviation threshold suppresses ordinary stationary noise,
/// while a nonzero shift in constant data remains detectable.
pub fn trajectory_stationarity_flags(values: &[f64]) -> Vec<bool> {
    let mut flags = Vec::new();
    stationarity_flags_into(values, |value| *value, &mut flags);
    flags
}

/// [`trajectory_stationarity_flags`] of the trajectory `value` reads from
/// `items`, written over `flags`, which grows only past its capacity.
fn stationarity_flags_into<T>(items: &[T], value: impl Fn(&T) -> f64, flags: &mut Vec<bool>) {
    let n = items.len();
    flags.clear();
    flags.resize(n, false);
    if n < 6 {
        return;
    }
    let window = (n / 10).max(3).min(n / 2);
    for center in window..(n - window) {
        let left = &items[center - window..center];
        let right = &items[center..center + window];
        if left
            .iter()
            .chain(right.iter())
            .any(|item| !value(item).is_finite())
        {
            continue;
        }
        let left_mean = left.iter().map(&value).sum::<f64>() / window as f64;
        let right_mean = right.iter().map(&value).sum::<f64>() / window as f64;
        let left_variance = left
            .iter()
            .map(|item| (value(item) - left_mean).powi(2))
            .sum::<f64>()
            / window as f64;
        let right_variance = right
            .iter()
            .map(|item| (value(item) - right_mean).powi(2))
            .sum::<f64>()
            / window as f64;
        let scale = left_variance.max(right_variance).sqrt();
        flags[center] = (right_mean - left_mean).abs() > 3.0 * scale.max(f64::EPSILON);
    }
}

#[cfg(test)]
mod tests {
    use super::{EpochLine, History, trajectory_stationarity_flags};
    use eindir_core::FPair;

    #[test]
    fn room_past_the_address_space_cannot_be_reserved() {
        assert!(History::can_reserve(&[]));
        assert!(History::can_reserve(&[0]));
        assert!(History::can_reserve(&[100, 100]));
        assert!(!History::can_reserve(&[usize::MAX / 8]));
        assert!(!History::can_reserve(&[100, usize::MAX / 8]));
    }

    #[test]
    fn refreshed_flags_fill_the_room_reserved_for_them() {
        let values: Vec<f64> = (0..60)
            .map(|k| match k {
                7 => f64::NAN,
                0..30 => 1.0 + 0.1 * f64::from(k % 3),
                _ => 4.0 + 0.1 * f64::from(k % 4),
            })
            .collect();
        let start = FPair {
            pos: ndarray::Array1::zeros(1),
            val: values[0],
        };
        let mut history = History::with_capacity(values.len(), start);
        for (epoch, &best_val) in values.iter().enumerate() {
            history.epochs.push(EpochLine {
                epoch,
                temp: 1.0,
                accepted: 0,
                rejected: 0,
                best_val,
            });
        }
        let room = history.stationarity_flags.as_ptr();
        history.refresh_stationarity_flags();
        assert_eq!(history.stationarity_flags.as_ptr(), room);
        let flags = trajectory_stationarity_flags(&values);
        assert_eq!(history.stationarity_flags, flags);
        assert!(flags[30] && flags.iter().any(|&flag| !flag));
    }

    #[test]
    fn flags_a_sustained_mean_shift_at_the_change_point() {
        let mut values = vec![0.0; 20];
        values.extend(vec![10.0; 20]);
        let flags = trajectory_stationarity_flags(&values);
        assert!(flags[20]);
        assert!(
            flags
                .iter()
                .enumerate()
                .any(|(index, flag)| { *flag && (18..=22).contains(&index) })
        );
    }

    #[test]
    fn stationary_constant_trajectory_has_no_flags() {
        assert!(
            !trajectory_stationarity_flags(&vec![2.0; 40])
                .into_iter()
                .any(|flag| flag)
        );
    }
}
