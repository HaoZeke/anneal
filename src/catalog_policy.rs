//! Pure cooperative-catalog decisions and target-free proposal primitives.

pub mod proposal;

use crate::catalog::{BasinCensus, BasinId, MixingEvidence, in_interface_ensemble};

/// Exact local visit count that classifies one occupied basin as exhausted.
pub const LOCAL_CENSUS_LEAVE: u64 = 8;
/// Consecutive nondeepening slices that classify one occupied basin as stalled.
pub const LOCAL_STALL_LEAVE: u32 = 8;

/// Relation between a replica's local state and the active catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveCatalogRelation {
    /// No active representative is available.
    Empty,
    /// The replica occupies the catalog incumbent basin.
    Incumbent,
    /// The replica occupies a nonincumbent active basin.
    SameBasin,
    /// Active representatives belong to other basins.
    Unrelated {
        /// A validated lower-energy representative is admissible for exploitation.
        lower_energy_anchor: bool,
    },
}

/// Validation state of the catalog evidence visible to the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationState {
    /// Signature, geometry, quench, and fresh-engine checks succeeded.
    Validated,
    /// At least one required validation check failed.
    Rejected,
}

/// Good--Turing and local-basin counts copied only from an exact census.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensusEvidence {
    total_visits: u64,
    singleton_basins: u64,
    local_basin_visits: u64,
    globally_saturated: bool,
}

impl CensusEvidence {
    /// Read policy evidence from the append-only census and an optional local basin.
    pub fn from_census(census: &BasinCensus, local_basin: Option<BasinId>) -> Self {
        Self {
            total_visits: census.total_visits(),
            singleton_basins: census.singleton_count(),
            local_basin_visits: local_basin
                .and_then(|basin_id| census.entry(basin_id))
                .map_or(0, |entry| entry.visits()),
            globally_saturated: census.is_saturated(),
        }
    }

    /// Construct checked evidence from exact coordinator counts.
    pub fn from_exact_counts(
        total_visits: u64,
        singleton_basins: u64,
        local_basin_visits: u64,
        globally_saturated: bool,
    ) -> Result<Self, PolicyInputError> {
        if singleton_basins > total_visits || local_basin_visits > total_visits {
            return Err(PolicyInputError::ImpossibleCensusCounts {
                total_visits,
                singleton_basins,
                local_basin_visits,
            });
        }
        Ok(Self {
            total_visits,
            singleton_basins,
            local_basin_visits,
            globally_saturated,
        })
    }

    /// Exact number of validated census observations.
    pub fn total_visits(self) -> u64 {
        self.total_visits
    }

    /// Exact number of singleton census basins.
    pub fn singleton_basins(self) -> u64 {
        self.singleton_basins
    }

    /// Exact visits assigned to the replica's local census basin.
    pub fn local_basin_visits(self) -> u64 {
        self.local_basin_visits
    }

    /// Whether the fixed-census production saturation rule is satisfied.
    pub fn globally_saturated(self) -> bool {
        self.globally_saturated
    }
}

/// Aggregate charged work relative to the declared ensemble budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggregateProgress {
    charged: u64,
    budget: u64,
}

impl AggregateProgress {
    /// Construct checked aggregate progress.
    pub fn new(charged: u64, budget: u64) -> Result<Self, PolicyInputError> {
        if budget == 0 {
            return Err(PolicyInputError::ZeroAggregateBudget);
        }
        if charged > budget {
            return Err(PolicyInputError::ChargedExceedsBudget { charged, budget });
        }
        Ok(Self { charged, budget })
    }

    /// Exact aggregate charged work.
    pub fn charged(self) -> u64 {
        self.charged
    }

    /// Whether at least half of aggregate charged work has been consumed.
    pub fn win_only(self) -> bool {
        self.charged.saturating_mul(2) >= self.budget
    }
}

/// Invalid aggregate-work input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PolicyInputError {
    /// An ensemble must declare a positive charged-work budget.
    #[error("aggregate budget must be positive")]
    ZeroAggregateBudget,
    /// Charged work cannot exceed its declared budget.
    #[error("aggregate charged work {charged} exceeds budget {budget}")]
    ChargedExceedsBudget {
        /// Observed aggregate charged work.
        charged: u64,
        /// Declared aggregate budget.
        budget: u64,
    },
    /// Exact census components cannot exceed the global visit total.
    #[error(
        "impossible census counts: total={total_visits}, singleton={singleton_basins}, local={local_basin_visits}"
    )]
    ImpossibleCensusCounts {
        /// Exact global visit total.
        total_visits: u64,
        /// Reported singleton-basin count.
        singleton_basins: u64,
        /// Reported local-basin visit count.
        local_basin_visits: u64,
    },
}

/// Complete pure input for one cooperative policy decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogPolicyInput {
    /// Validation state of the catalog evidence.
    pub validation: ValidationState,
    /// Relation of the local state to active catalog entries.
    pub relation: ActiveCatalogRelation,
    /// Exact fixed-census evidence.
    pub census: CensusEvidence,
    /// Aggregate charged-work progress.
    pub progress: AggregateProgress,
    /// Consecutive slices without a deeper local quench.
    pub local_stall_slices: u32,
    /// Whether the replica deepened its own basin in this slice.
    pub local_deepened: bool,
    /// Inverted Gelman--Rubin evidence for the live ensemble.
    pub mixing: MixingEvidence,
    /// Leftover-SOAP \(\lambda\) posted through the catalog RPC.
    pub leftover_lambda: f64,
    /// TIS interface rank. `u32::MAX` is the occupied-packing champion.
    pub interface_rank: u32,
    /// Threshold \(\lambda_i\) this extra must reach on Leave.
    pub interface_threshold: f64,
    /// Occupied DECAF families on the book. Not leftover-SOAP basins.
    pub occupied_family_count: usize,
    /// DECAF-family Good--Turing: unseen packing mass is small.
    pub packing_saturated: bool,
    /// Consecutive leftover-SOAP occupancy_gt records under the ceiling.
    pub leftover_dwell: bool,
    /// Published-energy score: this replica sits on a known hurdle GM.
    pub on_published_prize: bool,
}

/// Selectable action for one cooperative search slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyAction {
    /// Preserve the replica's independent local trajectory.
    ContinueLocal,
    /// Adopt a validated active representative.
    Exploit {
        /// Require an energy improvement over the local quench.
        win_only: bool,
    },
    /// Request a target-free descriptor-space exploration proposal.
    Explore,
    /// Leave the related exhausted basin through a descriptor-space hole.
    Leave,
}

/// Stable reason attached to every policy action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyReason {
    /// Catalog evidence failed validation.
    ValidationRejected,
    /// No active entry is available.
    EmptyCatalog,
    /// The incumbent replica retains local search responsibility.
    IncumbentLocalSearch,
    /// A deeper local quench is not replaced in the same slice.
    LocalDescent,
    /// Exact visits exhaust the related census basin.
    LocalCensusExhausted,
    /// The related basin has not deepened for the declared slice count.
    LocalStall,
    /// A remote lower-energy anchor is available during open progress.
    RemoteAnchorOpen,
    /// A remote lower-energy anchor is available during closed progress.
    RemoteAnchorClosed,
    /// Globally saturated evidence requests exploration, not unrelated leave.
    GlobalCensusSaturatedExplore,
    /// An unrelated catalog without an improving anchor requests exploration.
    UnrelatedCatalogExplore,
    /// A related but unexhausted basin requests exploration.
    SameBasinExplore,
    /// A TIS extra reached its leftover-SOAP interface. Shoot from the
    /// path that got there rather than keep walking the packing.
    InterfaceCrossed,
    /// A FamilyExtra of a crowded packing Leaves it. The champion stays.
    OccupiedPackingLeave,
    /// Explore-role chains have mixed; they must leave rather than keep exploring.
    ExploreCollapsed,
    /// The packing champion walks isomers of the occupied family.
    IsomerWalk,
    /// The incumbent attractor won the occupancy contest and stays occupied.
    CertifiedAttractor,
    /// Successive halving discarded a crowded-family walk at a rung.
    HyperbandPruned,
}

impl PolicyReason {
    /// Stable event-stream code.
    pub const fn code(self) -> &'static str {
        match self {
            Self::ValidationRejected => "validation_rejected",
            Self::EmptyCatalog => "empty_catalog",
            Self::IncumbentLocalSearch => "incumbent_local_search",
            Self::LocalDescent => "local_descent",
            Self::LocalCensusExhausted => "local_census_exhausted",
            Self::LocalStall => "local_stall",
            Self::RemoteAnchorOpen => "remote_anchor_open",
            Self::RemoteAnchorClosed => "remote_anchor_closed",
            Self::GlobalCensusSaturatedExplore => "global_census_saturated_explore",
            Self::UnrelatedCatalogExplore => "unrelated_catalog_explore",
            Self::SameBasinExplore => "same_basin_explore",
            Self::InterfaceCrossed => "interface_crossed",
            Self::OccupiedPackingLeave => "occupied_packing_leave",
            Self::IsomerWalk => "isomer_walk",
            Self::ExploreCollapsed => "explore_collapsed",
            Self::CertifiedAttractor => "certified_attractor",
            Self::HyperbandPruned => "hyperband_pruned",
        }
    }
}

/// Action and stable trace reason from the pure policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyDecision {
    /// Selected cooperative action.
    pub action: PolicyAction,
    /// Stable explanation recorded in the event stream.
    pub reason: PolicyReason,
}

/// Stateless cooperative catalog policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct CatalogPolicy;

impl CatalogPolicy {
    /// Decide one action without consulting energy-height or mutable global state.
    ///
    /// A lower catalog energy is taken before a same-packing leave. The
    /// incumbent of a known well leaves once it stalls, so the ensemble
    /// does not keep relaxing one funnel for the rest of the budget.
    ///
    /// A FamilyExtra of a crowded packing Leaves it. The champion
    /// stays and walks isomers. Collapse also forces extras of the
    /// occupied packing to Leave. A walk rematched to a different
    /// DECAF family is not collapse: yanking it would abandon the
    /// second funnel. Surplus extras are pruned at occupancy rungs
    /// and reseed. A better isomer of the occupied packing may be
    /// taken when the ensemble is not collapsed. A certified
    /// incumbent stays.
    pub fn decide(input: CatalogPolicyInput) -> PolicyDecision {
        if input.validation == ValidationState::Rejected {
            return decision(
                PolicyAction::ContinueLocal,
                PolicyReason::ValidationRejected,
            );
        }
        match input.relation {
            ActiveCatalogRelation::Empty => {
                decision(PolicyAction::ContinueLocal, PolicyReason::EmptyCatalog)
            }
            _ if input.mixing.certified_attractor
                && matches!(input.relation, ActiveCatalogRelation::Incumbent) =>
            {
                decision(
                    PolicyAction::ContinueLocal,
                    PolicyReason::CertifiedAttractor,
                )
            }
            _ if input.mixing.pruned => {
                decision(PolicyAction::Leave, PolicyReason::HyperbandPruned)
            }
            _ if input.local_deepened => {
                decision(PolicyAction::ContinueLocal, PolicyReason::LocalDescent)
            }
            _ if input.interface_rank != u32::MAX
                && input.on_published_prize
                && matches!(input.relation, ActiveCatalogRelation::Incumbent) =>
            {
                decision(
                    PolicyAction::ContinueLocal,
                    PolicyReason::IncumbentLocalSearch,
                )
            }
            _ if input.mixing.explore_collapsed
                && !matches!(input.relation, ActiveCatalogRelation::Incumbent)
                && !matches!(
                    input.relation,
                    ActiveCatalogRelation::Unrelated {
                        lower_energy_anchor: false
                    }
                ) =>
            {
                decision(PolicyAction::Leave, PolicyReason::ExploreCollapsed)
            }
            ActiveCatalogRelation::Unrelated {
                lower_energy_anchor: true,
            } => {
                let win_only = input.progress.win_only();
                decision(
                    PolicyAction::Exploit { win_only },
                    if win_only {
                        PolicyReason::RemoteAnchorClosed
                    } else {
                        PolicyReason::RemoteAnchorOpen
                    },
                )
            }
            ActiveCatalogRelation::Incumbent if input.local_stall_slices >= LOCAL_STALL_LEAVE => {
                decision(PolicyAction::Leave, PolicyReason::LocalStall)
            }
            ActiveCatalogRelation::Incumbent => {
                decision(PolicyAction::ContinueLocal, PolicyReason::IsomerWalk)
            }
            ActiveCatalogRelation::SameBasin
                if input.census.local_basin_visits() >= LOCAL_CENSUS_LEAVE =>
            {
                decision(PolicyAction::Leave, PolicyReason::LocalCensusExhausted)
            }
            ActiveCatalogRelation::SameBasin if input.local_stall_slices >= LOCAL_STALL_LEAVE => {
                decision(PolicyAction::Leave, PolicyReason::LocalStall)
            }
            ActiveCatalogRelation::SameBasin
                if input.interface_rank != u32::MAX
                    && in_interface_ensemble(input.leftover_lambda, input.interface_threshold) =>
            {
                decision(PolicyAction::Leave, PolicyReason::InterfaceCrossed)
            }
            ActiveCatalogRelation::SameBasin => {
                decision(PolicyAction::Leave, PolicyReason::OccupiedPackingLeave)
            }
            ActiveCatalogRelation::Unrelated {
                lower_energy_anchor: false,
            } => decision(
                PolicyAction::Explore,
                if input.census.globally_saturated() {
                    PolicyReason::GlobalCensusSaturatedExplore
                } else {
                    PolicyReason::UnrelatedCatalogExplore
                },
            ),
        }
    }
}

const fn decision(action: PolicyAction, reason: PolicyReason) -> PolicyDecision {
    PolicyDecision { action, reason }
}
