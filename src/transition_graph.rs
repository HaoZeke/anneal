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
    /// A structural microstate index cannot be extended to a node count.
    #[error("structural microstate index overflow")]
    NodeIndexOverflow,
    /// Diffusion time must contain at least one transition step.
    #[error("diffusion steps must be positive")]
    ZeroDiffusionSteps,
    /// Complete-linkage distance must be finite and nonnegative.
    #[error("maximum attraction-region distance must be finite and nonnegative")]
    InvalidMaximumDistance,
    /// A resolved region requires at least one fixed-probe observation.
    #[error("minimum fixed-probe observations must be positive")]
    ZeroMinimumProbes,
}

/// Posterior squared-error risk for one categorical transition row.
///
/// If the row probabilities have a Dirichlet posterior with total
/// concentration `A`, `covariance_trace` is the Bayes risk of the posterior
/// mean under squared Euclidean loss. Predictive observations reduce that
/// risk without an empirical acquisition coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletInformation {
    total_concentration: f64,
    covariance_trace: f64,
}

impl DirichletInformation {
    /// Sum of the posterior Dirichlet parameters.
    pub fn total_concentration(self) -> f64 {
        self.total_concentration
    }

    /// Trace of the posterior covariance matrix.
    pub fn covariance_trace(self) -> f64 {
        self.covariance_trace
    }

    /// Expected covariance trace after `additional_observations` draws.
    pub fn expected_covariance_trace(self, additional_observations: usize) -> f64 {
        let additional = additional_observations as f64;
        self.covariance_trace * self.total_concentration / (self.total_concentration + additional)
    }

    /// Expected reduction in covariance trace from the next allocated draw.
    pub fn marginal_risk_reduction(self, observations_already_allocated: usize) -> f64 {
        let allocated = observations_already_allocated as f64;
        self.covariance_trace * self.total_concentration
            / ((self.total_concentration + allocated)
                * (self.total_concentration + allocated + 1.0))
    }
}

/// Target-blind coarse-graining parameters for attraction regions.
#[derive(Debug, Clone, PartialEq)]
pub struct AttractionRegionConfig {
    /// Action whose repeated dynamics define comparable return behaviour.
    pub probe_action: String,
    /// Symmetric Dirichlet concentration for all resolved states and `U`.
    pub concentration: f64,
    /// Number of probe-operator steps used in diffusion distance.
    pub diffusion_steps: usize,
    /// Maximum complete-linkage diffusion distance within one region.
    pub maximum_distance: f64,
    /// Probe observations required before a microstate can merge.
    pub minimum_probes: u64,
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

    /// Ensure that a structural microstate exists before it has outgoing data.
    pub fn register_node(&mut self, node: usize) -> Result<(), TransitionGraphError> {
        let required = node
            .checked_add(1)
            .ok_or(TransitionGraphError::NodeIndexOverflow)?;
        if required > self.nodes {
            self.nodes = required;
            for counts in self.actions.values_mut() {
                counts.resize(self.nodes);
            }
        }
        Ok(())
    }

    /// Record one source, action, and resolved or unresolved destination.
    pub fn observe(
        &mut self,
        action: impl Into<String>,
        from: usize,
        outcome: TransitionOutcome,
    ) -> Result<(), TransitionGraphError> {
        let destination_nodes = match outcome {
            TransitionOutcome::Resolved(to) => to
                .checked_add(1)
                .ok_or(TransitionGraphError::NodeIndexOverflow)?,
            TransitionOutcome::Unresolved => 0,
        };
        self.register_node(from.max(destination_nodes.saturating_sub(1)))?;

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

    /// Dirichlet Bayes risk and predictive value of information for one row.
    ///
    /// The outcome space contains every resolved node plus the unresolved
    /// outcome. Returns `None` when `from` is not a registered node.
    pub fn dirichlet_information(
        &self,
        action: &str,
        from: usize,
        concentration: f64,
    ) -> Result<Option<DirichletInformation>, TransitionGraphError> {
        if !concentration.is_finite() || concentration <= 0.0 {
            return Err(TransitionGraphError::InvalidConcentration);
        }
        if from >= self.nodes {
            return Ok(None);
        }
        let categories = self.nodes.saturating_add(1);
        let mut total_concentration = 0.0;
        let mut squared_concentration = 0.0;
        for to in 0..self.nodes {
            let alpha =
                self.count(action, from, TransitionOutcome::Resolved(to)) as f64 + concentration;
            total_concentration += alpha;
            squared_concentration += alpha * alpha;
        }
        let unresolved =
            self.count(action, from, TransitionOutcome::Unresolved) as f64 + concentration;
        total_concentration += unresolved;
        squared_concentration += unresolved * unresolved;
        debug_assert_eq!(categories, self.nodes + 1);
        let squared_mean = squared_concentration / (total_concentration * total_concentration);
        let covariance_trace = ((1.0 - squared_mean) / (total_concentration + 1.0)).max(0.0);
        Ok(Some(DirichletInformation {
            total_concentration,
            covariance_trace,
        }))
    }

    /// Deterministic complete-linkage attraction regions from fixed-probe dynamics.
    ///
    /// The unresolved posterior column is retained as an evidence diagnostic.
    /// Diffusion propagates on the resolved conditional operator; nodes below
    /// `minimum_probes` remain singleton unresolved regions and cannot merge.
    pub fn attraction_regions(
        &self,
        config: &AttractionRegionConfig,
    ) -> Result<Vec<Vec<usize>>, TransitionGraphError> {
        if config.diffusion_steps == 0 {
            return Err(TransitionGraphError::ZeroDiffusionSteps);
        }
        if !config.maximum_distance.is_finite() || config.maximum_distance < 0.0 {
            return Err(TransitionGraphError::InvalidMaximumDistance);
        }
        if config.minimum_probes == 0 {
            return Err(TransitionGraphError::ZeroMinimumProbes);
        }
        let posterior = self.posterior_matrix(&config.probe_action, config.concentration)?;
        let n = self.nodes;
        let mut resolved = Array2::zeros((n, n));
        for from in 0..n {
            let mass = (0..n).map(|to| posterior[[from, to]]).sum::<f64>();
            for to in 0..n {
                resolved[[from, to]] = posterior[[from, to]] / mass;
            }
        }
        let propagated = matrix_power(&resolved, config.diffusion_steps);
        let mut reference = vec![0.0; n];
        for to in 0..n {
            reference[to] =
                (0..n).map(|from| propagated[[from, to]]).sum::<f64>() / n.max(1) as f64;
            reference[to] = reference[to].max(f64::EPSILON);
        }
        let mut distances = Array2::zeros((n, n));
        for left in 0..n {
            for right in (left + 1)..n {
                let squared = (0..n)
                    .map(|to| {
                        let delta = propagated[[left, to]] - propagated[[right, to]];
                        delta * delta / reference[to]
                    })
                    .sum::<f64>();
                let distance = squared.sqrt();
                distances[[left, right]] = distance;
                distances[[right, left]] = distance;
            }
        }
        let eligible = (0..n)
            .map(|node| self.observations(&config.probe_action, node) >= config.minimum_probes)
            .collect::<Vec<_>>();
        Ok(complete_linkage_regions(
            &distances,
            &eligible,
            config.maximum_distance,
        ))
    }
}

fn matrix_power(matrix: &Array2<f64>, exponent: usize) -> Array2<f64> {
    let n = matrix.nrows();
    let mut result = Array2::eye(n);
    for _ in 0..exponent {
        let mut product = Array2::zeros((n, n));
        for row in 0..n {
            for column in 0..n {
                product[[row, column]] = (0..n)
                    .map(|inner| result[[row, inner]] * matrix[[inner, column]])
                    .sum();
            }
        }
        result = product;
    }
    result
}

fn complete_linkage_regions(
    distances: &Array2<f64>,
    eligible: &[bool],
    maximum_distance: f64,
) -> Vec<Vec<usize>> {
    let mut regions = (0..eligible.len())
        .map(|node| vec![node])
        .collect::<Vec<_>>();
    loop {
        let mut selected = None;
        for left in 0..regions.len() {
            if regions[left].iter().any(|node| !eligible[*node]) {
                continue;
            }
            for right in (left + 1)..regions.len() {
                if regions[right].iter().any(|node| !eligible[*node]) {
                    continue;
                }
                let linkage = regions[left]
                    .iter()
                    .flat_map(|a| regions[right].iter().map(move |b| distances[[*a, *b]]))
                    .fold(0.0_f64, f64::max);
                if linkage > maximum_distance {
                    continue;
                }
                let candidate = (linkage, left, right);
                if selected.is_none_or(|best: (f64, usize, usize)| candidate < best) {
                    selected = Some(candidate);
                }
            }
        }
        let Some((_, left, right)) = selected else {
            break;
        };
        let merged = regions.remove(right);
        regions[left].extend(merged);
        regions[left].sort_unstable();
    }
    regions.sort();
    regions
}
