//! Action-conditioned transition counts over target-blind structural microstates.
//!
//! Structural descriptors determine the microstate labels supplied to this
//! module. They do not determine attraction-region membership. Region evidence
//! comes from repeated dynamics under a named proposal action, with a fixed
//! probe action kept separate from adaptive search and transport actions.

use std::collections::BTreeMap;

use ndarray::Array2;

/// Destination observed for one perturb--quench attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionOutcome {
    /// The quench reached a classified structural microstate.
    Resolved(usize),
    /// The result was invalid, unclassified, or outside the represented set.
    Unresolved,
}

/// Error that leaves the requested posterior undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TransitionGraphError {
    /// The symmetric Dirichlet concentration must be finite and positive.
    #[error("Dirichlet concentration must be finite and positive")]
    InvalidConcentration,
    /// A transition count cannot be represented by `u64`.
    #[error("transition count overflow")]
    CountOverflow,
}

#[derive(Debug, Clone, Default)]
struct ActionCounts {
    resolved: Vec<Vec<u64>>,
    unresolved: Vec<u64>,
}

impl ActionCounts {
    fn resize(&mut self, nodes: usize) {
        for row in &mut self.resolved {
            row.resize(nodes, 0);
        }
        self.resolved.resize_with(nodes, || vec![0; nodes]);
        self.unresolved.resize(nodes, 0);
    }
}

/// Append-only transition evidence separated by proposal action.
#[derive(Debug, Clone, Default)]
pub struct TransitionGraph {
    nodes: usize,
    actions: BTreeMap<String, ActionCounts>,
}

impl TransitionGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of structural microstates represented by any observation.
    pub fn node_count(&self) -> usize {
        self.nodes
    }

    /// Record one source, action, and resolved or unresolved destination.
    pub fn observe(
        &mut self,
        action: impl Into<String>,
        from: usize,
        outcome: TransitionOutcome,
    ) -> Result<(), TransitionGraphError> {
        let destination_nodes = match outcome {
            TransitionOutcome::Resolved(to) => to.saturating_add(1),
            TransitionOutcome::Unresolved => 0,
        };
        let required = from.saturating_add(1).max(destination_nodes);
        if required > self.nodes {
            self.nodes = required;
            for counts in self.actions.values_mut() {
                counts.resize(self.nodes);
            }
        }

        let counts = self.actions.entry(action.into()).or_default();
        counts.resize(self.nodes);
        let slot = match outcome {
            TransitionOutcome::Resolved(to) => &mut counts.resolved[from][to],
            TransitionOutcome::Unresolved => &mut counts.unresolved[from],
        };
        *slot = slot
            .checked_add(1)
            .ok_or(TransitionGraphError::CountOverflow)?;
        Ok(())
    }

    /// Exact count for one action-conditioned transition outcome.
    pub fn count(&self, action: &str, from: usize, outcome: TransitionOutcome) -> u64 {
        let Some(counts) = self.actions.get(action) else {
            return 0;
        };
        match outcome {
            TransitionOutcome::Resolved(to) => counts
                .resolved
                .get(from)
                .and_then(|row| row.get(to))
                .copied()
                .unwrap_or(0),
            TransitionOutcome::Unresolved => counts.unresolved.get(from).copied().unwrap_or(0),
        }
    }

    /// Total observations leaving `from` under exactly one action.
    pub fn observations(&self, action: &str, from: usize) -> u64 {
        let Some(counts) = self.actions.get(action) else {
            return 0;
        };
        let resolved = counts
            .resolved
            .get(from)
            .map(|row| row.iter().copied().sum::<u64>())
            .unwrap_or(0);
        resolved.saturating_add(counts.unresolved.get(from).copied().unwrap_or(0))
    }

    /// Dirichlet-posterior mean transition matrix for exactly one action.
    ///
    /// Columns `0..node_count` are resolved destinations. The final column is
    /// the unresolved outcome. A symmetric pseudocount keeps every row proper
    /// before probes have reached all destinations.
    pub fn posterior_matrix(
        &self,
        action: &str,
        concentration: f64,
    ) -> Result<Array2<f64>, TransitionGraphError> {
        if !concentration.is_finite() || concentration <= 0.0 {
            return Err(TransitionGraphError::InvalidConcentration);
        }
        let columns = self.nodes.saturating_add(1);
        let mut matrix = Array2::zeros((self.nodes, columns));
        for from in 0..self.nodes {
            let total = self.observations(action, from) as f64;
            let denominator = total + concentration * columns as f64;
            for to in 0..self.nodes {
                matrix[[from, to]] = (self.count(action, from, TransitionOutcome::Resolved(to))
                    as f64
                    + concentration)
                    / denominator;
            }
            matrix[[from, self.nodes]] =
                (self.count(action, from, TransitionOutcome::Unresolved) as f64 + concentration)
                    / denominator;
        }
        Ok(matrix)
    }

    /// Inverse posterior concentration for one action-conditioned source row.
    pub fn uncertainty(&self, action: &str, from: usize, concentration: f64) -> Option<f64> {
        if from >= self.nodes || !concentration.is_finite() || concentration <= 0.0 {
            return None;
        }
        let categories = self.nodes.saturating_add(1) as f64;
        Some(1.0 / (self.observations(action, from) as f64 + concentration * categories))
    }
}
