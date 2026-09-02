//! Replay-safe aggregate accounting for charged cooperative search work.

use std::collections::{BTreeMap, btree_map::Entry};

/// Boundary at which one replica records work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChargeKind {
    /// Construction of a local Cartesian proposal; no potential call.
    LocalProposal,
    /// Construction of a catalog-derived proposal; no potential call.
    RemoteProposal,
    /// SOAP or ACE evaluation; no potential call.
    DescriptorEvaluation,
    /// Potential calls consumed by a quench that produced a valid candidate.
    AcceptedQuench,
    /// Potential calls consumed by a quench that did not produce a candidate.
    RejectedQuench,
    /// Receiving-side fresh potential evaluation.
    FreshValidation,
    /// Potential calls consumed by a linked deterministic retry.
    Retry,
    /// Communication failure followed by independent local fallback.
    RpcFallback,
    /// Potential or gradient calls used by proposal machinery outside a quench.
    AuxiliaryEvaluation,
    /// Potential calls consumed by one basin-escape attempt.
    BasinEscape,
    /// Potential calls consumed by one minimum-mode saddle-ride attempt.
    SaddleRide,
}

impl ChargeKind {
    /// Stable protocol discriminant.
    pub const fn wire_code(self) -> u16 {
        match self {
            Self::LocalProposal => 0,
            Self::RemoteProposal => 1,
            Self::DescriptorEvaluation => 2,
            Self::AcceptedQuench => 3,
            Self::RejectedQuench => 4,
            Self::FreshValidation => 5,
            Self::Retry => 6,
            Self::RpcFallback => 7,
            Self::AuxiliaryEvaluation => 8,
            Self::BasinEscape => 9,
            Self::SaddleRide => 10,
        }
    }

    /// Decode one stable protocol discriminant.
    pub const fn from_wire_code(code: u16) -> Option<Self> {
        match code {
            0 => Some(Self::LocalProposal),
            1 => Some(Self::RemoteProposal),
            2 => Some(Self::DescriptorEvaluation),
            3 => Some(Self::AcceptedQuench),
            4 => Some(Self::RejectedQuench),
            5 => Some(Self::FreshValidation),
            6 => Some(Self::Retry),
            7 => Some(Self::RpcFallback),
            8 => Some(Self::AuxiliaryEvaluation),
            9 => Some(Self::BasinEscape),
            10 => Some(Self::SaddleRide),
            _ => None,
        }
    }

    const fn carries_potential_calls(self) -> bool {
        matches!(
            self,
            Self::AcceptedQuench
                | Self::RejectedQuench
                | Self::FreshValidation
                | Self::Retry
                | Self::AuxiliaryEvaluation
                | Self::BasinEscape
                | Self::SaddleRide
        )
    }
}

/// Replay-safe work attributed to one charged mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChargeSummary {
    /// Number of unique operation-boundary events.
    pub events: u64,
    /// Potential calls retained by those events.
    pub charged_calls: u64,
}

/// One immutable replica event with its monotone charged counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicaLedgerEvent {
    /// Replica identity within one isolated ensemble.
    pub replica: u32,
    /// Monotone event sequence, starting at one.
    pub sequence: u64,
    /// Work boundary represented by this event.
    pub kind: ChargeKind,
    /// Potential calls charged at this boundary.
    pub charged_calls: u64,
    /// Replica charged counter including this event.
    pub cumulative_charged: u64,
}

/// Result of recording a valid event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerUpdate {
    /// A new event entered the ledger.
    Recorded,
    /// An identical replay was already present.
    Duplicate,
}

/// Invalid event or ledger configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LedgerError {
    /// At least one replica is required.
    #[error("cooperative ledger requires at least one replica")]
    EmptyReplicaSet,
    /// Replica identities must be unique.
    #[error("duplicate replica identity {replica}")]
    DuplicateReplica {
        /// Repeated replica identity.
        replica: u32,
    },
    /// Each replica requires a positive charged budget.
    #[error("per-replica charged budget must be positive")]
    ZeroBudget,
    /// The aggregate budget must fit the ledger counter representation.
    #[error("aggregate charged budget cannot be represented")]
    AggregateBudgetOverflow,
    /// A request names a replica outside the ensemble.
    #[error("unknown replica identity {replica}")]
    UnknownReplica {
        /// Foreign replica identity.
        replica: u32,
    },
    /// Sequence zero is reserved.
    #[error("replica event sequence must start at one")]
    ZeroSequence,
    /// A replay at one identity differs from the stored event.
    #[error("conflicting replay for replica {replica} sequence {sequence}")]
    ConflictingReplay {
        /// Replica identity.
        replica: u32,
        /// Conflicting sequence.
        sequence: u64,
    },
    /// An uncharged boundary cannot carry potential calls.
    #[error("uncharged work kind {kind:?} carries {charged_calls} potential calls")]
    UnchargedKindHasCalls {
        /// Uncharged work boundary.
        kind: ChargeKind,
        /// Invalid positive charge.
        charged_calls: u64,
    },
    /// A charged engine boundary must retain its consumed calls.
    #[error("charged work kind {kind:?} has no potential calls")]
    ChargedKindHasNoCalls {
        /// Charged work boundary.
        kind: ChargeKind,
    },
    /// The event exceeds its replica budget.
    #[error("replica {replica} charged {charged} calls beyond budget {budget}")]
    BudgetExceeded {
        /// Replica identity.
        replica: u32,
        /// Proposed cumulative charge.
        charged: u64,
        /// Per-replica budget.
        budget: u64,
    },
    /// A cumulative counter moves backward or disagrees with adjacent events.
    #[error("counter regression for replica {replica} sequence {sequence}")]
    CounterRegression {
        /// Replica identity.
        replica: u32,
        /// Invalid sequence.
        sequence: u64,
    },
}

#[derive(Debug, Clone, Default)]
struct ReplicaLedger {
    events: BTreeMap<u64, ReplicaLedgerEvent>,
}

impl ReplicaLedger {
    fn total(&self) -> u64 {
        self.events
            .last_key_value()
            .map_or(0, |(_, event)| event.cumulative_charged)
    }
}

/// Counter vector frozen at the first validated target encounter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstEncounter {
    replica_totals: Vec<(u32, u64)>,
    ensemble_total: u64,
}

impl FirstEncounter {
    /// Replica counters in ascending identity order.
    pub fn replica_totals(&self) -> &[(u32, u64)] {
        &self.replica_totals
    }

    /// Sum of all replica counters at the encounter.
    pub fn ensemble_total(&self) -> u64 {
        self.ensemble_total
    }
}

/// Isolated ensemble ledger with idempotent event ingestion.
#[derive(Debug, Clone)]
pub struct CooperativeLedger {
    replicas: BTreeMap<u32, ReplicaLedger>,
    per_replica_budget: u64,
    first_encounter: Option<FirstEncounter>,
}

impl CooperativeLedger {
    /// Construct an empty ledger for one explicit ensemble replica set.
    pub fn new(
        replicas: impl IntoIterator<Item = u32>,
        per_replica_budget: u64,
    ) -> Result<Self, LedgerError> {
        if per_replica_budget == 0 {
            return Err(LedgerError::ZeroBudget);
        }
        let mut states = BTreeMap::new();
        for replica in replicas {
            if states.insert(replica, ReplicaLedger::default()).is_some() {
                return Err(LedgerError::DuplicateReplica { replica });
            }
        }
        if states.is_empty() {
            return Err(LedgerError::EmptyReplicaSet);
        }
        let replica_count =
            u64::try_from(states.len()).map_err(|_| LedgerError::AggregateBudgetOverflow)?;
        per_replica_budget
            .checked_mul(replica_count)
            .ok_or(LedgerError::AggregateBudgetOverflow)?;
        Ok(Self {
            replicas: states,
            per_replica_budget,
            first_encounter: None,
        })
    }

    /// Record a new event or classify an identical replay without double charging.
    pub fn record(&mut self, event: ReplicaLedgerEvent) -> Result<LedgerUpdate, LedgerError> {
        let ledger = self
            .replicas
            .get_mut(&event.replica)
            .ok_or(LedgerError::UnknownReplica {
                replica: event.replica,
            })?;
        if event.sequence == 0 {
            return Err(LedgerError::ZeroSequence);
        }
        if let Some(stored) = ledger.events.get(&event.sequence) {
            return if *stored == event {
                Ok(LedgerUpdate::Duplicate)
            } else {
                Err(LedgerError::ConflictingReplay {
                    replica: event.replica,
                    sequence: event.sequence,
                })
            };
        }
        if event.kind.carries_potential_calls() && event.charged_calls == 0 {
            return Err(LedgerError::ChargedKindHasNoCalls { kind: event.kind });
        }
        if !event.kind.carries_potential_calls() && event.charged_calls != 0 {
            return Err(LedgerError::UnchargedKindHasCalls {
                kind: event.kind,
                charged_calls: event.charged_calls,
            });
        }
        if event.cumulative_charged > self.per_replica_budget {
            return Err(LedgerError::BudgetExceeded {
                replica: event.replica,
                charged: event.cumulative_charged,
                budget: self.per_replica_budget,
            });
        }
        if !counter_fits(&ledger.events, event) {
            return Err(LedgerError::CounterRegression {
                replica: event.replica,
                sequence: event.sequence,
            });
        }
        match ledger.events.entry(event.sequence) {
            Entry::Vacant(slot) => {
                slot.insert(event);
                Ok(LedgerUpdate::Recorded)
            }
            Entry::Occupied(_) => unreachable!("replay handled before validation"),
        }
    }

    /// Admit a replica that is not yet in the ledger.
    ///
    /// A replica already present is accepted again so a journaled attach
    /// replays without becoming a duplicate error.
    pub fn attach(&mut self, replica: u32) -> Result<(), LedgerError> {
        if self.replicas.contains_key(&replica) {
            return Ok(());
        }
        let replica_count = u64::try_from(self.replicas.len().saturating_add(1))
            .map_err(|_| LedgerError::AggregateBudgetOverflow)?;
        self.per_replica_budget
            .checked_mul(replica_count)
            .ok_or(LedgerError::AggregateBudgetOverflow)?;
        self.replicas.insert(replica, ReplicaLedger::default());
        Ok(())
    }

    /// Latest charged counter for one ensemble replica.
    pub fn replica_total(&self, replica: u32) -> Option<u64> {
        self.replicas.get(&replica).map(ReplicaLedger::total)
    }

    /// Sum of the latest counter from every replica.
    pub fn ensemble_total(&self) -> u64 {
        self.replicas.values().map(ReplicaLedger::total).sum()
    }

    /// Declared aggregate charged-work budget.
    pub fn aggregate_budget(&self) -> u64 {
        let replicas = u64::try_from(self.replicas.len())
            .expect("replica count is checked when the ledger is constructed");
        self.per_replica_budget * replicas
    }

    /// Number of unique recorded events across replicas.
    pub fn event_count(&self) -> usize {
        self.replicas
            .values()
            .map(|ledger| ledger.events.len())
            .sum()
    }

    /// Aggregate unique events and calls for one work mechanism.
    pub fn charge_summary(&self, kind: ChargeKind) -> ChargeSummary {
        self.replicas
            .values()
            .flat_map(|ledger| ledger.events.values())
            .filter(|event| event.kind == kind)
            .fold(ChargeSummary::default(), |summary, event| ChargeSummary {
                events: summary
                    .events
                    .checked_add(1)
                    .expect("ledger event count must fit u64"),
                charged_calls: summary
                    .charged_calls
                    .checked_add(event.charged_calls)
                    .expect("charged mechanism total is bounded by the aggregate budget"),
            })
    }

    /// Freeze and return the complete counter vector at first target encounter.
    pub fn record_first_encounter(&mut self) -> &FirstEncounter {
        if self.first_encounter.is_none() {
            let replica_totals = self
                .replicas
                .iter()
                .map(|(&replica, ledger)| (replica, ledger.total()))
                .collect::<Vec<_>>();
            let ensemble_total = replica_totals.iter().map(|(_, total)| total).sum();
            self.first_encounter = Some(FirstEncounter {
                replica_totals,
                ensemble_total,
            });
        }
        self.first_encounter
            .as_ref()
            .expect("first encounter is initialized")
    }
}

fn counter_fits(events: &BTreeMap<u64, ReplicaLedgerEvent>, event: ReplicaLedgerEvent) -> bool {
    if event.sequence == 1 && event.cumulative_charged != event.charged_calls {
        return false;
    }
    if let Some((&previous_sequence, previous)) = events.range(..event.sequence).next_back() {
        if event.cumulative_charged < previous.cumulative_charged {
            return false;
        }
        if previous_sequence.checked_add(1) == Some(event.sequence)
            && previous.cumulative_charged.checked_add(event.charged_calls)
                != Some(event.cumulative_charged)
        {
            return false;
        }
    }
    let next = event
        .sequence
        .checked_add(1)
        .and_then(|lower_bound| events.range(lower_bound..).next());
    if let Some((&next_sequence, next)) = next {
        if event.cumulative_charged > next.cumulative_charged {
            return false;
        }
        if event.sequence.checked_add(1) == Some(next_sequence)
            && event.cumulative_charged.checked_add(next.charged_calls)
                != Some(next.cumulative_charged)
        {
            return false;
        }
    }
    true
}
