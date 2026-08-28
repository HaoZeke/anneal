//! Same-system scheduling and evidence for minimum-mode transition searches.
//!
//! A ledger belongs to one PES/system-signature coordinator. It never compares
//! energies, basin identifiers, local-environment classes, or outcomes across
//! systems. Replicas communicate by claiming concrete transition experiments
//! and reporting their charged cost and certified result. Failed and duplicate
//! searches remain evidence, preventing the ensemble from blindly repeating
//! work that another live chain has performed.

use std::collections::{BTreeMap, BTreeSet};

use ndarray::ArrayView2;

use crate::pes_exploration::RideMethod;

/// One sign of a localized initial mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RideDirection {
    /// Displace opposite to the generated mode.
    Negative,
    /// Displace along the generated mode.
    Positive,
}

/// One representative atom for a local-environment class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnvironmentTarget {
    /// Coordinator-local environment class.
    pub class: u32,
    /// Representative atom in the source structure.
    pub atom: u32,
}

/// Leader-clustered invariant local environments for one system signature.
#[derive(Debug, Clone)]
pub struct EnvironmentBook {
    radius_squared: f64,
    feature_dim: Option<usize>,
    leaders: Vec<Vec<f64>>,
}

impl EnvironmentBook {
    /// Construct a codebook with a fixed within-system feature radius.
    pub fn new(radius: f64) -> Result<Self, RideLedgerError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(RideLedgerError::InvalidEnvironmentRadius);
        }
        Ok(Self {
            radius_squared: radius * radius,
            feature_dim: None,
            leaders: Vec::new(),
        })
    }

    /// Assign every atom and return one representative for each present class.
    ///
    /// Rows must be rotation/permutation-invariant local features produced by
    /// the caller's descriptor contract. Classes are coordinator-local and
    /// therefore cannot identify or compare environments from another PES.
    pub fn observe(
        &mut self,
        features: ArrayView2<'_, f64>,
    ) -> Result<Vec<EnvironmentTarget>, RideLedgerError> {
        if features.nrows() == 0 || features.ncols() == 0 {
            return Err(RideLedgerError::EmptyEnvironmentFeatures);
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(RideLedgerError::NonfiniteEnvironmentFeature);
        }
        match self.feature_dim {
            Some(expected) if expected != features.ncols() => {
                return Err(RideLedgerError::EnvironmentFeatureDimension {
                    expected,
                    actual: features.ncols(),
                });
            }
            None => self.feature_dim = Some(features.ncols()),
            Some(_) => {}
        }

        let mut representatives = BTreeMap::<u32, u32>::new();
        for (atom, row) in features.rows().into_iter().enumerate() {
            let nearest = self
                .leaders
                .iter()
                .enumerate()
                .map(|(class, leader)| {
                    let squared = row
                        .iter()
                        .zip(leader)
                        .map(|(left, right)| {
                            let delta = left - right;
                            delta * delta
                        })
                        .sum::<f64>();
                    (class, squared)
                })
                .filter(|(_, squared)| *squared <= self.radius_squared)
                .min_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| left.0.cmp(&right.0))
                })
                .map(|(class, _)| class);
            let class = match nearest {
                Some(class) => class,
                None => {
                    self.leaders.push(row.to_vec());
                    self.leaders.len() - 1
                }
            };
            let class = u32::try_from(class).map_err(|_| RideLedgerError::CounterOverflow)?;
            let atom = u32::try_from(atom).map_err(|_| RideLedgerError::CounterOverflow)?;
            representatives.entry(class).or_insert(atom);
        }
        Ok(representatives
            .into_iter()
            .map(|(class, atom)| EnvironmentTarget { class, atom })
            .collect())
    }

    /// Number of invariant local-environment classes observed in this system.
    pub fn class_count(&self) -> usize {
        self.leaders.len()
    }
}

/// A quenched minimum eligible for transition exploration.
#[derive(Debug, Clone, PartialEq)]
pub struct RideSource {
    /// Exact basin identifier within one system catalogue.
    pub basin: u64,
    /// PES energy used only to order sources within this system.
    pub energy: f64,
    /// Distinct local environments and representative atoms.
    pub environments: Vec<EnvironmentTarget>,
}

/// Finite initial portfolio crossed with every source environment and sign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RidePortfolio {
    mode_ranks: u16,
    methods: Vec<RideMethod>,
}

impl RidePortfolio {
    /// Construct a portfolio with mode ranks `0..mode_ranks`.
    pub fn new(mode_ranks: u16, mut methods: Vec<RideMethod>) -> Result<Self, RideLedgerError> {
        if mode_ranks == 0 {
            return Err(RideLedgerError::EmptyModePortfolio);
        }
        methods.sort_unstable();
        methods.dedup();
        if methods.is_empty() {
            return Err(RideLedgerError::EmptyMethodPortfolio);
        }
        Ok(Self {
            mode_ranks,
            methods,
        })
    }

    /// Number of initial mode ranks explored for each environment and method.
    pub fn mode_ranks(&self) -> u16 {
        self.mode_ranks
    }

    /// Minimum-mode solvers in deterministic scheduling order.
    pub fn methods(&self) -> &[RideMethod] {
        &self.methods
    }
}

/// Scientific identity of one transition-search arm.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RideArm {
    /// Exact source minimum.
    pub source_basin: u64,
    /// Source-local environment class selected for perturbation.
    pub environment_class: u32,
    /// Ranked localized mode seed.
    pub mode_rank: u16,
    /// Sign of the initial displacement.
    pub direction: RideDirection,
    /// Minimum-mode solver.
    pub method: RideMethod,
}

/// One exclusive assignment to a live replica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RideWorkOrder {
    /// Monotonic coordinator-local experiment identifier.
    pub id: u64,
    /// Replica holding the exclusive claim.
    pub replica: u32,
    /// Scientific arm being evaluated.
    pub arm: RideArm,
    /// Atom representing the arm's local-environment class.
    pub representative_atom: u32,
    /// One-based attempt number for this arm.
    pub attempt: u64,
    /// Caller-controlled deterministic random seed.
    pub seed: u64,
}

/// Explicit unsuccessful outcome shared with every replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RideFailure {
    /// The source quench did not reach its force condition.
    QuenchNotConverged,
    /// The minimum-mode search did not reach its force condition.
    SaddleNotConverged,
    /// Receiving-side certification found no unstable mode.
    NoNegativeMode,
    /// Receiving-side certification found more than one unstable mode.
    HigherIndex,
    /// An IRC branch did not settle in a minimum.
    IrcNotConverged,
    /// Both IRC branches reached the same exact minimum.
    CollapsedConnection,
    /// The PES evaluator failed or returned invalid values.
    Surface,
    /// The charged-work budget ended during the experiment.
    BudgetExhausted,
}

/// Receiving-side result of a claimed experiment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RideOutcome {
    /// An index-one saddle with two exact, distinct IRC endpoints.
    Certified {
        /// Exact saddle identifier within the same system catalogue.
        saddle: u64,
        /// Exact minima connected by the saddle.
        endpoints: [u64; 2],
    },
    /// Classified failure, retained as negative search evidence.
    Failed(RideFailure),
}

/// Credit assigned by the coordinator rather than trusted from a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RideCredit {
    /// Whether this report introduced a previously unseen endpoint pair.
    pub novel_edge: bool,
}

/// Invalid source, claim, or report.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RideLedgerError {
    /// At least one ranked mode is required.
    #[error("ride portfolio needs at least one mode rank")]
    EmptyModePortfolio,
    /// At least one minimum-mode solver is required.
    #[error("ride portfolio needs at least one method")]
    EmptyMethodPortfolio,
    /// Source energies must be finite.
    #[error("ride source energy must be finite")]
    NonfiniteSourceEnergy,
    /// A source needs at least one local target.
    #[error("ride source has no local-environment targets")]
    EmptyEnvironmentSet,
    /// Environment clustering needs a finite positive radius.
    #[error("local-environment radius must be finite and positive")]
    InvalidEnvironmentRadius,
    /// A local descriptor must contain at least one atom and one feature.
    #[error("local-environment feature matrix must be nonempty")]
    EmptyEnvironmentFeatures,
    /// All local descriptors in one codebook use one feature schema.
    #[error("local-environment feature dimension is {actual}, expected {expected}")]
    EnvironmentFeatureDimension {
        /// Dimension established by the first observation.
        expected: usize,
        /// Dimension supplied by this observation.
        actual: usize,
    },
    /// Local descriptor rows must contain only finite values.
    #[error("local-environment feature matrix contains a nonfinite value")]
    NonfiniteEnvironmentFeature,
    /// A work identifier is unknown to this ledger.
    #[error("unknown ride work order {0}")]
    UnknownWork(u64),
    /// Only the replica holding a claim may report it.
    #[error("replica {replica} does not own ride work order {work}")]
    WrongReplica {
        /// Reporting replica.
        replica: u32,
        /// Claimed work identifier.
        work: u64,
    },
    /// A repeated delivery must carry the same scientific result and cost.
    #[error("ride work order {0} was reported with conflicting content")]
    ConflictingReport(u64),
    /// Certified IRC endpoints must be distinct.
    #[error("certified ride endpoints collapse to one basin")]
    CollapsedCertifiedConnection,
    /// A one-sided ride must reconnect to its exact source basin.
    #[error("certified ride does not contain source basin {0}")]
    DisconnectedSource(u64),
    /// A monotonic counter cannot be represented.
    #[error("ride-ledger counter overflow")]
    CounterOverflow,
}

#[derive(Debug, Clone, Default)]
struct ArmEvidence {
    completed: u64,
    novel_edges: u64,
    certified: u64,
    charged_evaluations: u64,
    failures: BTreeMap<RideFailure, u64>,
}

#[derive(Debug, Clone)]
struct SourceRecord {
    energy: f64,
    representatives: BTreeMap<u32, u32>,
}

#[derive(Debug, Clone)]
struct CompletedReport {
    replica: u32,
    charged_evaluations: u64,
    outcome: RideOutcome,
    credit: RideCredit,
}

/// Append-only shared evidence and exclusive live claims for one PES.
#[derive(Debug, Clone)]
pub struct RideLedger {
    portfolio: RidePortfolio,
    sources: BTreeMap<u64, SourceRecord>,
    evidence: BTreeMap<RideArm, ArmEvidence>,
    active: BTreeMap<u64, RideWorkOrder>,
    active_arms: BTreeSet<RideArm>,
    replica_work: BTreeMap<u32, u64>,
    completed_reports: BTreeMap<u64, CompletedReport>,
    edges: BTreeSet<[u64; 2]>,
    next_work: u64,
    completed_attempts: u64,
    certified_connections: u64,
    charged_evaluations: u64,
}

impl RideLedger {
    /// Create an empty ledger for one coordinator-owned system signature.
    pub fn new(portfolio: RidePortfolio) -> Self {
        Self {
            portfolio,
            sources: BTreeMap::new(),
            evidence: BTreeMap::new(),
            active: BTreeMap::new(),
            active_arms: BTreeSet::new(),
            replica_work: BTreeMap::new(),
            completed_reports: BTreeMap::new(),
            edges: BTreeSet::new(),
            next_work: 0,
            completed_attempts: 0,
            certified_connections: 0,
            charged_evaluations: 0,
        }
    }

    /// Add or enrich an exact source minimum.
    ///
    /// Re-observation is idempotent. One representative, the smallest atom
    /// index, is kept for each environment class.
    pub fn register_source(&mut self, source: RideSource) -> Result<(), RideLedgerError> {
        if !source.energy.is_finite() {
            return Err(RideLedgerError::NonfiniteSourceEnergy);
        }
        if source.environments.is_empty() {
            return Err(RideLedgerError::EmptyEnvironmentSet);
        }
        let record = self.sources.entry(source.basin).or_insert(SourceRecord {
            energy: source.energy,
            representatives: BTreeMap::new(),
        });
        record.energy = record.energy.min(source.energy);
        for target in source.environments {
            record
                .representatives
                .entry(target.class)
                .and_modify(|atom| *atom = (*atom).min(target.atom))
                .or_insert(target.atom);
        }
        Ok(())
    }

    /// Claim the highest-acquisition experiment not held by another replica.
    ///
    /// A retry by the same replica returns its existing order. Every arm is
    /// attempted once before Bayesian upper-confidence scheduling repeats an
    /// arm. Raw energy only breaks acquisition ties inside this ledger.
    pub fn claim(&mut self, replica: u32, seed: u64) -> Option<RideWorkOrder> {
        if let Some(id) = self.replica_work.get(&replica) {
            return self.active.get(id).cloned();
        }

        let total = self.completed_attempts;
        let selected = self
            .arms()
            .filter(|(arm, _)| !self.active_arms.contains(arm))
            .max_by(|(left, left_atom), (right, right_atom)| {
                self.acquisition(left, total)
                    .total_cmp(&self.acquisition(right, total))
                    .then_with(|| {
                        self.source_energy(right.source_basin)
                            .total_cmp(&self.source_energy(left.source_basin))
                    })
                    .then_with(|| right.cmp(left))
                    .then_with(|| right_atom.cmp(left_atom))
            })?;
        let arm = selected.0;
        let representative_atom = selected.1;
        let attempt = self
            .evidence
            .get(&arm)
            .map_or(1, |row| row.completed.saturating_add(1));
        let id = self.next_work;
        self.next_work = self.next_work.checked_add(1)?;
        let order = RideWorkOrder {
            id,
            replica,
            arm: arm.clone(),
            representative_atom,
            attempt,
            seed,
        };
        self.active_arms.insert(arm);
        self.replica_work.insert(replica, id);
        self.active.insert(id, order.clone());
        Some(order)
    }

    /// Report charged cost and a certified or classified result.
    ///
    /// Repeated delivery of the same completed report is idempotent. Novelty
    /// is computed from the canonical endpoint pair held by the coordinator.
    pub fn report(
        &mut self,
        replica: u32,
        work: u64,
        charged_evaluations: u64,
        outcome: RideOutcome,
    ) -> Result<RideCredit, RideLedgerError> {
        if let Some(completed) = self.completed_reports.get(&work) {
            if completed.replica != replica {
                return Err(RideLedgerError::WrongReplica { replica, work });
            }
            return if completed.charged_evaluations == charged_evaluations
                && completed.outcome == outcome
            {
                Ok(completed.credit)
            } else {
                Err(RideLedgerError::ConflictingReport(work))
            };
        }
        let order = self
            .active
            .get(&work)
            .ok_or(RideLedgerError::UnknownWork(work))?;
        if order.replica != replica {
            return Err(RideLedgerError::WrongReplica { replica, work });
        }

        let canonical_edge = match outcome {
            RideOutcome::Certified { endpoints, .. } => {
                if endpoints[0] == endpoints[1] {
                    return Err(RideLedgerError::CollapsedCertifiedConnection);
                }
                if !endpoints.contains(&order.arm.source_basin) {
                    return Err(RideLedgerError::DisconnectedSource(order.arm.source_basin));
                }
                Some(if endpoints[0] < endpoints[1] {
                    endpoints
                } else {
                    [endpoints[1], endpoints[0]]
                })
            }
            RideOutcome::Failed(_) => None,
        };

        let order = self
            .active
            .remove(&work)
            .expect("validated work remains active");
        self.active_arms.remove(&order.arm);
        self.replica_work.remove(&replica);

        let novel_edge = canonical_edge.is_some_and(|edge| self.edges.insert(edge));
        let evidence = self.evidence.entry(order.arm).or_default();
        evidence.completed = evidence
            .completed
            .checked_add(1)
            .ok_or(RideLedgerError::CounterOverflow)?;
        evidence.charged_evaluations = evidence
            .charged_evaluations
            .checked_add(charged_evaluations)
            .ok_or(RideLedgerError::CounterOverflow)?;
        match &outcome {
            RideOutcome::Certified { .. } => {
                evidence.certified = evidence
                    .certified
                    .checked_add(1)
                    .ok_or(RideLedgerError::CounterOverflow)?;
                if novel_edge {
                    evidence.novel_edges = evidence
                        .novel_edges
                        .checked_add(1)
                        .ok_or(RideLedgerError::CounterOverflow)?;
                }
                self.certified_connections = self
                    .certified_connections
                    .checked_add(1)
                    .ok_or(RideLedgerError::CounterOverflow)?;
            }
            RideOutcome::Failed(failure) => {
                let count = evidence.failures.entry(*failure).or_default();
                *count = count
                    .checked_add(1)
                    .ok_or(RideLedgerError::CounterOverflow)?;
            }
        }
        self.completed_attempts = self
            .completed_attempts
            .checked_add(1)
            .ok_or(RideLedgerError::CounterOverflow)?;
        self.charged_evaluations = self
            .charged_evaluations
            .checked_add(charged_evaluations)
            .ok_or(RideLedgerError::CounterOverflow)?;
        let credit = RideCredit { novel_edge };
        self.completed_reports.insert(
            work,
            CompletedReport {
                replica,
                charged_evaluations,
                outcome,
                credit,
            },
        );
        Ok(credit)
    }

    /// Release a replica's unfinished claim without fabricating an outcome.
    pub fn release(&mut self, replica: u32) -> Option<RideWorkOrder> {
        let id = self.replica_work.remove(&replica)?;
        let order = self.active.remove(&id)?;
        self.active_arms.remove(&order.arm);
        Some(order)
    }

    /// Number of completed transition experiments.
    pub fn completed_attempts(&self) -> u64 {
        self.completed_attempts
    }

    /// Number of reports containing a certified index-one connection.
    pub fn certified_connections(&self) -> u64 {
        self.certified_connections
    }

    /// Number of unique exact endpoint pairs.
    pub fn unique_edges(&self) -> usize {
        self.edges.len()
    }

    /// PES evaluations charged to completed experiments.
    pub fn charged_evaluations(&self) -> u64 {
        self.charged_evaluations
    }

    /// Number of exclusive claims held by live replicas.
    pub fn active_attempts(&self) -> usize {
        self.active.len()
    }

    fn source_energy(&self, basin: u64) -> f64 {
        self.sources
            .get(&basin)
            .map_or(f64::INFINITY, |source| source.energy)
    }

    fn acquisition(&self, arm: &RideArm, total: u64) -> f64 {
        let Some(row) = self.evidence.get(arm) else {
            return f64::INFINITY;
        };
        if row.completed == 0 {
            return f64::INFINITY;
        }
        let completed = row.completed as f64;
        let posterior_mean = (row.novel_edges as f64 + 0.5) / (completed + 1.0);
        let exploration = (2.0 * ((total as f64) + 2.0).ln() / (completed + 1.0)).sqrt();
        posterior_mean + exploration
    }

    fn arms(&self) -> impl Iterator<Item = (RideArm, u32)> + '_ {
        self.sources
            .iter()
            .flat_map(move |(&source_basin, source)| {
                source
                    .representatives
                    .iter()
                    .flat_map(move |(&environment_class, &atom)| {
                        (0..self.portfolio.mode_ranks).flat_map(move |mode_rank| {
                            [RideDirection::Negative, RideDirection::Positive]
                                .into_iter()
                                .flat_map(move |direction| {
                                    self.portfolio.methods.iter().copied().map(move |method| {
                                        (
                                            RideArm {
                                                source_basin,
                                                environment_class,
                                                mode_rank,
                                                direction,
                                                method,
                                            },
                                            atom,
                                        )
                                    })
                                })
                        })
                    })
            })
    }
}
