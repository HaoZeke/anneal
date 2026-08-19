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

    /// Recomputes the per-epoch non-stationarity flags from epoch summaries.
    pub fn refresh_stationarity_flags(&mut self) {
        let values: Vec<f64> = self.epochs.iter().map(|epoch| epoch.best_val).collect();
        self.stationarity_flags = trajectory_stationarity_flags(&values);
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
    let n = values.len();
    let mut flags = vec![false; n];
    if n < 6 {
        return flags;
    }
    let window = (n / 10).max(3).min(n / 2);
    for center in window..(n - window) {
        let left = &values[center - window..center];
        let right = &values[center..center + window];
        if left.iter().chain(right.iter()).any(|value| !value.is_finite()) {
            continue;
        }
        let left_mean = left.iter().sum::<f64>() / window as f64;
        let right_mean = right.iter().sum::<f64>() / window as f64;
        let variance = left
            .iter()
            .chain(right.iter())
            .map(|value| {
                let centered = value - (left_mean + right_mean) / 2.0;
                centered * centered
            })
            .sum::<f64>()
            / (2 * window) as f64;
        let scale = variance.sqrt();
        flags[center] = (right_mean - left_mean).abs() > 3.0 * scale.max(f64::EPSILON);
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::trajectory_stationarity_flags;

    #[test]
    fn flags_a_sustained_mean_shift_at_the_change_point() {
        let mut values = vec![0.0; 20];
        values.extend(vec![10.0; 20]);
        let flags = trajectory_stationarity_flags(&values);
        assert!(flags[20]);
        assert!(flags.iter().enumerate().any(|(index, flag)| {
            *flag && (18..=22).contains(&index)
        }));
    }

    #[test]
    fn stationary_constant_trajectory_has_no_flags() {
        assert!(!trajectory_stationarity_flags(&vec![2.0; 40])
            .into_iter()
            .any(|flag| flag));
    }
}
