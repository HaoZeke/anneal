//! Shared core-class table for cooperative cluster chains.
//!
//! Chains report the coarse five-fold class of their live state
//! ([`crate::corekey::motif_class`]) together with the energy and charged
//! work at that checkpoint. The table keeps one [`CoreClassStat`] per
//! class and one private record per chain. A chain restarts when its own
//! best inside its class has not improved for `core_patience` charged
//! calls, or when, once it has spent `core_trial` calls in a fresh class,
//! its best there sits above the median of that class's recorded trial
//! energies and at least four trials are on file. A chain that holds the
//! class best never restarts on the class rule.

use std::collections::HashMap;

/// Verdict returned by [`CoreClassTable::report`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreVerdict {
    /// Stay in the live class.
    Continue,
    /// Leave the class from a fresh random cluster.
    Restart,
}

/// Shared catalog of class statistics and per-chain core-class state.
pub trait CooperativeCatalog {
    /// Record one checkpoint and decide whether the chain should restart.
    fn report_core_class(
        &mut self,
        chain: usize,
        class: u8,
        energy: f64,
        charged: usize,
    ) -> CoreVerdict;
}

/// One core class shared by the chains of an ensemble.
#[derive(Debug, Clone)]
pub struct CoreClassStat {
    /// Lowest energy any chain has seen in this class.
    pub best: f64,
    /// Charged-work increment accumulated since the class last improved.
    pub calls_since_improvement: usize,
    /// Checkpoints at which some chain sat in this class.
    pub visits: usize,
    /// Best energies of chains at the end of their trial in this class.
    pub trials: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ChainState {
    class: Option<u8>,
    best: f64,
    best_at: usize,
    entered_at: usize,
    tried: bool,
}

impl Default for ChainState {
    fn default() -> Self {
        Self {
            class: None,
            best: f64::INFINITY,
            best_at: 0,
            entered_at: 0,
            tried: false,
        }
    }
}

/// Cores visited by an ensemble, keyed by [`crate::corekey::MotifClass::index`].
#[derive(Debug, Clone)]
pub struct CoreClassTable {
    patience: usize,
    trial: usize,
    class_tabu: usize,
    visit_charge: usize,
    stats: HashMap<u8, CoreClassStat>,
    chains: HashMap<usize, ChainState>,
    restarts: usize,
}

impl CoreClassTable {
    /// Construct a table with the stall patience and the trial budget.
    ///
    /// A zero trial budget disables the within-class rank rule.
    pub fn new(patience: usize, trial: usize) -> Self {
        Self {
            patience,
            trial,
            class_tabu: 0,
            visit_charge: 1,
            stats: HashMap::new(),
            chains: HashMap::new(),
            restarts: 0,
        }
    }

    /// Calls a class may go without improvement before chains that do
    /// not hold its best restart. Zero disables the class tabu.
    pub fn with_class_tabu(mut self, calls: usize) -> Self {
        self.class_tabu = calls;
        self
    }

    /// Charged-work increment added to [`CoreClassStat::calls_since_improvement`]
    /// on each report. The ensemble splice driver charges one checkpoint
    /// interval per visit so the class-tabu budget stays in the same units.
    pub fn with_visit_charge(mut self, charge: usize) -> Self {
        self.visit_charge = charge.max(1);
        self
    }

    /// Record one checkpoint and return whether the chain should restart.
    pub fn report(&mut self, chain: usize, class: u8, energy: f64, charged: usize) -> CoreVerdict {
        let visit_charge = self.visit_charge;
        {
            let stat = self.stats.entry(class).or_insert(CoreClassStat {
                best: f64::INFINITY,
                calls_since_improvement: 0,
                visits: 0,
                trials: Vec::new(),
            });
            stat.visits += 1;
            stat.calls_since_improvement =
                stat.calls_since_improvement.saturating_add(visit_charge);
            if energy < stat.best - 1e-6 {
                stat.best = energy;
                stat.calls_since_improvement = 0;
            }
        }
        let class_best = self.stats[&class].best;
        let class_calls = self.stats[&class].calls_since_improvement;

        let own = self.chains.entry(chain).or_default();
        if own.class != Some(class) {
            own.class = Some(class);
            own.best = f64::INFINITY;
            own.best_at = charged;
            own.entered_at = charged;
            own.tried = false;
        }
        if energy < own.best - 1e-6 {
            own.best = energy;
            own.best_at = charged;
        }
        let own_best = own.best;
        let own_stalled = charged.saturating_sub(own.best_at) >= self.patience;
        let holds_best = energy <= class_best + 1e-6 || own_best <= class_best + 1e-6;
        let class_tabu = self.class_tabu > 0 && class_calls >= self.class_tabu && !holds_best;
        let finish_trial =
            self.trial > 0 && !own.tried && charged.saturating_sub(own.entered_at) >= self.trial;
        if finish_trial {
            own.tried = true;
        }
        if own_stalled || class_tabu {
            own.class = None;
            self.restarts += 1;
            return CoreVerdict::Restart;
        }
        if !finish_trial {
            return CoreVerdict::Continue;
        }
        let trials = {
            let stat = self.stats.get_mut(&class).expect("class just visited");
            stat.trials.push(own_best);
            stat.trials.clone()
        };
        let mut sorted = trials;
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[sorted.len() / 2];
        let trial_lost = !holds_best && sorted.len() >= 4 && own_best > median + 1e-6;
        if !trial_lost {
            return CoreVerdict::Continue;
        }
        if let Some(own) = self.chains.get_mut(&chain) {
            own.class = None;
        }
        self.restarts += 1;
        CoreVerdict::Restart
    }

    /// Restarts the table has issued.
    pub fn restarts(&self) -> usize {
        self.restarts
    }

    /// Number of distinct classes that have been reported.
    pub fn class_count(&self) -> usize {
        self.stats.len()
    }

    /// Statistics for one class, if any chain has sat there.
    pub fn class_stat(&self, class: u8) -> Option<&CoreClassStat> {
        self.stats.get(&class)
    }

    /// Class statistics in class-index order.
    pub fn stats(&self) -> impl Iterator<Item = (u8, &CoreClassStat)> {
        let mut keys: Vec<u8> = self.stats.keys().copied().collect();
        keys.sort_unstable();
        keys.into_iter()
            .map(|class| (class, self.stats.get(&class).expect("key from stats")))
    }
}

impl Default for CoreClassTable {
    fn default() -> Self {
        Self::new(10_000, 2_000)
    }
}

impl CooperativeCatalog for CoreClassTable {
    fn report_core_class(
        &mut self,
        chain: usize,
        class: u8,
        energy: f64,
        charged: usize,
    ) -> CoreVerdict {
        self.report(chain, class, energy, charged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trial_energy_above_the_class_median_restarts() {
        let mut table = CoreClassTable::new(10_000, 10);
        assert_eq!(table.report(0, 0, 1.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(1, 0, 2.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(2, 0, 3.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(3, 0, 4.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(0, 0, 1.0, 10), CoreVerdict::Continue);
        assert_eq!(table.report(1, 0, 2.0, 10), CoreVerdict::Continue);
        assert_eq!(table.report(2, 0, 3.0, 10), CoreVerdict::Continue);
        assert_eq!(table.report(3, 0, 4.0, 10), CoreVerdict::Restart);
    }

    #[test]
    fn a_trial_energy_below_the_class_median_continues() {
        let mut table = CoreClassTable::new(10_000, 10);
        assert_eq!(table.report(0, 0, 1.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(1, 0, 2.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(2, 0, 3.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(3, 0, 1.5, 0), CoreVerdict::Continue);
        assert_eq!(table.report(0, 0, 1.0, 10), CoreVerdict::Continue);
        assert_eq!(table.report(1, 0, 2.0, 10), CoreVerdict::Continue);
        assert_eq!(table.report(2, 0, 3.0, 10), CoreVerdict::Continue);
        assert_eq!(table.report(3, 0, 1.5, 10), CoreVerdict::Continue);
    }

    #[test]
    fn a_stalled_chain_restarts() {
        let mut table = CoreClassTable::new(10, 0);
        assert_eq!(table.report(0, 0, 1.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(0, 0, 1.0, 9), CoreVerdict::Continue);
        assert_eq!(table.report(0, 0, 1.0, 10), CoreVerdict::Restart);
    }

    #[test]
    fn a_chain_holding_the_class_best_never_restarts_on_the_class_rule() {
        let mut table = CoreClassTable::new(10_000, 0).with_class_tabu(3);
        assert_eq!(table.report(0, 0, 1.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(1, 0, 2.0, 0), CoreVerdict::Continue);
        assert_eq!(table.report(1, 0, 2.0, 1), CoreVerdict::Continue);
        assert_eq!(table.report(1, 0, 2.0, 2), CoreVerdict::Restart);
        assert_eq!(table.report(0, 0, 1.0, 3), CoreVerdict::Continue);
    }
}
