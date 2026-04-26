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
}

impl History {
    /// Constructs an empty `History` with the given epoch capacity.
    pub fn with_capacity(n_epochs: usize, best: FPair<f64>) -> Self {
        Self {
            epochs: Vec::with_capacity(n_epochs),
            best,
        }
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
