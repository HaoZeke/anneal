//! Target-free Feynman--Kac reconfiguration for cooperative search chains.
//!
//! A search epoch supplies one validated representative per chain. The
//! selection potential combines within-epoch energy rank, descriptor novelty,
//! and census scarcity. Systematic resampling keeps the chain population fixed,
//! while a family cap prevents a single observed funnel from occupying every
//! slot. Transition-network inference is diagnostic output rather than
//! population evidence. This is a population-management operator, not a
//! Green-function approximation and not an electronic-structure convergence
//! claim.

use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// Invalid evidence or reconfiguration parameters.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ReconfigurationError {
    /// An evidence rank is not finite or lies outside the unit interval.
    #[error("{field} rank must be finite and lie in [0, 1], got {value}")]
    InvalidRank {
        /// Name of the invalid evidence component.
        field: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// A scalar parameter is outside its admissible domain.
    #[error("invalid reconfiguration parameter {field}: {value}")]
    InvalidParameter {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// Reconfiguration requires at least one chain.
    #[error("reconfiguration requires a nonempty population")]
    EmptyPopulation,
    /// One synchronization epoch contains the same replica more than once.
    #[error("duplicate population evidence for replica {replica}")]
    DuplicateReplica {
        /// Replica repeated in the submitted population.
        replica: u32,
    },
    /// A submission names a replica outside the configured population.
    #[error("unknown population replica {replica}")]
    UnknownReplica {
        /// Replica absent from the configured ensemble.
        replica: u32,
    },
    /// A submission does not belong to the collector's open epoch.
    #[error("population epoch {received} does not match open epoch {expected}")]
    EpochMismatch {
        /// Open synchronization epoch.
        expected: u64,
        /// Submitted synchronization epoch.
        received: u64,
    },
    /// A replica changed its evidence within one immutable epoch.
    #[error("replica {replica} submitted conflicting evidence for epoch {epoch}")]
    ConflictingSubmission {
        /// Synchronization epoch containing the conflict.
        epoch: u64,
        /// Replica that changed its submitted evidence.
        replica: u32,
    },
}

/// Target-free evidence attached to one chain at a synchronization epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasinEvidence {
    energy_rank: f64,
    novelty_rank: f64,
    scarcity_rank: f64,
}

impl BasinEvidence {
    /// Construct evidence from ranks in `[0, 1]`.
    ///
    /// Lower energy rank is better. Higher novelty and scarcity ranks are
    /// better. Ranks keep selection pressure independent of energy units and
    /// cluster-size-dependent energy scale.
    pub fn new(
        energy_rank: f64,
        novelty_rank: f64,
        scarcity_rank: f64,
    ) -> Result<Self, ReconfigurationError> {
        validate_rank("energy", energy_rank)?;
        validate_rank("novelty", novelty_rank)?;
        validate_rank("scarcity", scarcity_rank)?;
        Ok(Self {
            energy_rank,
            novelty_rank,
            scarcity_rank,
        })
    }

    /// Within-epoch energy rank, where zero is best.
    pub fn energy_rank(self) -> f64 {
        self.energy_rank
    }

    /// Descriptor novelty rank, where one is most novel.
    pub fn novelty_rank(self) -> f64 {
        self.novelty_rank
    }

    /// Census scarcity rank, where one is least sampled.
    pub fn scarcity_rank(self) -> f64 {
        self.scarcity_rank
    }
}

/// Raw coordinator evidence for one replica at a synchronization epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationMember {
    replica: u32,
    energy: f64,
    novelty: f64,
    basin_visits: f64,
}

impl PopulationMember {
    /// Construct one member from fresh energy, descriptor novelty, and the
    /// exact visit count of its immutable census basin.
    pub fn new(
        replica: u32,
        energy: f64,
        novelty: f64,
        basin_visits: f64,
    ) -> Result<Self, ReconfigurationError> {
        if !energy.is_finite() {
            return Err(ReconfigurationError::InvalidParameter {
                field: "member_energy",
                value: energy,
            });
        }
        if !novelty.is_finite() || novelty < 0.0 {
            return Err(ReconfigurationError::InvalidParameter {
                field: "member_novelty",
                value: novelty,
            });
        }
        if !basin_visits.is_finite() || basin_visits <= 0.0 {
            return Err(ReconfigurationError::InvalidParameter {
                field: "member_basin_visits",
                value: basin_visits,
            });
        }
        Ok(Self {
            replica,
            energy,
            novelty,
            basin_visits,
        })
    }

    /// Replica identity within the isolated ensemble.
    pub fn replica(self) -> u32 {
        self.replica
    }
}

/// One replica and its coordinator-derived rank evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankedPopulationMember {
    replica: u32,
    evidence: BasinEvidence,
}

impl RankedPopulationMember {
    /// Replica identity retained from the raw epoch evidence.
    pub fn replica(self) -> u32 {
        self.replica
    }

    /// Target-free rank evidence derived across the complete population.
    pub fn evidence(self) -> BasinEvidence {
        self.evidence
    }
}

/// Rank a complete synchronization population in stable input order.
///
/// Novelty is ranked directly. Scarcity is ranked through inverse exact basin
/// visits, so a less-visited census basin receives a higher scarcity rank.
pub fn rank_population(
    members: &[PopulationMember],
) -> Result<Vec<RankedPopulationMember>, ReconfigurationError> {
    if members.is_empty() {
        return Err(ReconfigurationError::EmptyPopulation);
    }
    let mut replicas = BTreeSet::new();
    for member in members {
        if !replicas.insert(member.replica) {
            return Err(ReconfigurationError::DuplicateReplica {
                replica: member.replica,
            });
        }
    }
    let energy = members
        .iter()
        .map(|member| member.energy)
        .collect::<Vec<_>>();
    let novelty = members
        .iter()
        .map(|member| member.novelty)
        .collect::<Vec<_>>();
    let scarcity = members
        .iter()
        .map(|member| 1.0 / member.basin_visits)
        .collect::<Vec<_>>();
    let energy_ranks = ascending_fractional_ranks(&energy)?;
    let novelty_ranks = ascending_fractional_ranks(&novelty)?;
    let scarcity_ranks = ascending_fractional_ranks(&scarcity)?;

    members
        .iter()
        .enumerate()
        .map(|(index, member)| {
            Ok(RankedPopulationMember {
                replica: member.replica,
                evidence: BasinEvidence::new(
                    energy_ranks[index],
                    novelty_ranks[index],
                    scarcity_ranks[index],
                )?,
            })
        })
        .collect()
}

fn validate_rank(field: &'static str, value: f64) -> Result<(), ReconfigurationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ReconfigurationError::InvalidRank { field, value })
    }
}

/// Coefficients of the bounded logarithmic selection potential.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionCoefficients {
    /// Pressure toward lower within-epoch energy rank.
    pub energy: f64,
    /// Pressure toward descriptor novelty.
    pub novelty: f64,
    /// Pressure toward census scarcity.
    pub scarcity: f64,
    /// Maximum log-weight difference retained before exponentiation.
    pub log_weight_clip: f64,
}

impl Default for SelectionCoefficients {
    fn default() -> Self {
        Self {
            energy: 1.0,
            novelty: 0.8,
            scarcity: 0.6,
            log_weight_clip: 4.0,
        }
    }
}

impl SelectionCoefficients {
    fn validate(self) -> Result<(), ReconfigurationError> {
        for (field, value) in [
            ("energy", self.energy),
            ("novelty", self.novelty),
            ("scarcity", self.scarcity),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ReconfigurationError::InvalidParameter { field, value });
            }
        }
        if !self.log_weight_clip.is_finite() || self.log_weight_clip <= 0.0 {
            return Err(ReconfigurationError::InvalidParameter {
                field: "log_weight_clip",
                value: self.log_weight_clip,
            });
        }
        Ok(())
    }
}

/// Population and genealogy diagnostics at one synchronization epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenealogyDiagnostics {
    /// Kish effective sample size of the normalized selection weights.
    pub effective_sample_size: f64,
    /// Number of source chains represented among the offspring.
    pub unique_parents: usize,
    /// Largest number of offspring assigned to one source chain.
    pub max_family_size: usize,
    /// Population variance of source-chain offspring counts.
    pub offspring_variance: f64,
}

/// Replayable fixed-population assignment for one synchronization epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct ReconfigurationPlan {
    weights: Vec<f64>,
    parents: Vec<usize>,
    diagnostics: GenealogyDiagnostics,
}

/// Replica-addressed reconfiguration result for one complete epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationEpochPlan {
    epoch: u64,
    destinations: Vec<u32>,
    parents: Vec<u32>,
    weights: Vec<f64>,
    diagnostics: GenealogyDiagnostics,
}

/// Stable position of one destination within its realized parent family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PopulationFamilyPosition {
    parent: u32,
    ordinal: usize,
    family_size: usize,
}

impl PopulationFamilyPosition {
    /// Source replica assigned to this destination.
    pub fn parent(self) -> u32 {
        self.parent
    }

    /// Zero-based position among destinations sharing the same parent.
    pub fn ordinal(self) -> usize {
        self.ordinal
    }

    /// Number of destinations assigned to this parent.
    pub fn family_size(self) -> usize {
        self.family_size
    }
}

/// Locate a destination within an immutable replica-addressed genealogy.
///
/// Malformed vector lengths, absent destinations, and duplicate destination
/// identities return `None` so a caller cannot silently adopt the wrong
/// parent.
pub fn population_family_position(
    destinations: &[u32],
    parents: &[u32],
    destination: u32,
) -> Option<PopulationFamilyPosition> {
    if destinations.len() != parents.len() {
        return None;
    }
    let matches = destinations
        .iter()
        .enumerate()
        .filter(|(_, candidate)| **candidate == destination)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return None;
    }
    let index = matches[0].0;
    let parent = parents[index];
    let ordinal = parents[..index]
        .iter()
        .filter(|candidate| **candidate == parent)
        .count();
    let family_size = parents
        .iter()
        .filter(|candidate| **candidate == parent)
        .count();
    Some(PopulationFamilyPosition {
        parent,
        ordinal,
        family_size,
    })
}

/// Deterministic descriptor-space rejuvenation draw for one offspring.
///
/// Destination identity and family ordinal remain explicit even when several
/// offspring share the same parent, preventing cloned chains from requesting
/// the same catalog-space perturbation.
pub fn population_rejuvenation_draw(
    seed: u64,
    epoch: u64,
    destination: u32,
    family_ordinal: usize,
) -> u64 {
    let mut value = splitmix64(seed ^ epoch.rotate_left(17));
    value = splitmix64(value ^ u64::from(destination).rotate_left(31));
    splitmix64(value ^ (family_ordinal as u64).rotate_left(47))
}

impl PopulationEpochPlan {
    /// Synchronization epoch represented by this immutable plan.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Destination replicas in stable ascending order.
    pub fn destinations(&self) -> &[u32] {
        &self.destinations
    }

    /// Parent replica for each destination at the same index.
    pub fn parents(&self) -> &[u32] {
        &self.parents
    }

    /// Normalized source weights in destination/source replica order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Weight and realized-family diagnostics.
    pub fn diagnostics(&self) -> GenealogyDiagnostics {
        self.diagnostics
    }
}

/// Result of submitting one replica's evidence to a synchronous epoch.
#[derive(Clone, Debug, PartialEq)]
pub enum EpochSubmissionOutcome {
    /// The epoch remains open until every replica the barrier still requires
    /// has submitted.
    Pending {
        /// Open synchronization epoch.
        epoch: u64,
        /// Unique replica submissions received.
        submitted: usize,
        /// Replicas the barrier still waits on.
        required: usize,
    },
    /// All replicas submitted and the immutable parent plan is available.
    Ready(PopulationEpochPlan),
}

#[derive(Clone, Debug)]
struct CompletedEpoch {
    submissions: BTreeMap<u32, PopulationMember>,
    plan: PopulationEpochPlan,
}

/// Coordinator-owned barrier and replay state for fixed-population epochs.
#[derive(Clone, Debug)]
pub struct SynchronousPopulation {
    replicas: Vec<u32>,
    coefficients: SelectionCoefficients,
    max_offspring: usize,
    seed: u64,
    open_epoch: u64,
    submissions: BTreeMap<u32, PopulationMember>,
    completed: BTreeMap<u64, CompletedEpoch>,
    abstained: BTreeSet<u32>,
    live: BTreeSet<u32>,
    /// When set, [`Self::required`] counts [`Self::live`] even if retirement
    /// has emptied that set, instead of falling back to the configured roster.
    live_roster: bool,
    /// Fraction of the required roster whose submissions close an epoch
    /// once the deadline has passed; one waits for everyone.
    quorum: f64,
    /// Coordinator ticks after the first submission of an epoch before the
    /// quorum may close it; `u64::MAX` never closes short of everyone.
    deadline_ticks: u64,
    /// Ticks the coordinator has delivered so far.
    ticks: u64,
    /// The epoch and tick at which the open epoch received its first
    /// submission.
    first_submission: Option<(u64, u64)>,
}

impl SynchronousPopulation {
    /// Construct a collector for one isolated ensemble.
    ///
    /// The live roster starts empty, so the barrier waits on every configured
    /// replica until a submit sees a nonempty live set or a replica retires.
    pub fn new(
        replicas: impl IntoIterator<Item = u32>,
        coefficients: SelectionCoefficients,
        max_offspring: usize,
        seed: u64,
    ) -> Result<Self, ReconfigurationError> {
        coefficients.validate()?;
        if max_offspring == 0 {
            return Err(ReconfigurationError::InvalidParameter {
                field: "max_offspring",
                value: 0.0,
            });
        }
        let mut unique = BTreeSet::new();
        for replica in replicas {
            if !unique.insert(replica) {
                return Err(ReconfigurationError::DuplicateReplica { replica });
            }
        }
        if unique.is_empty() {
            return Err(ReconfigurationError::EmptyPopulation);
        }
        Ok(Self {
            replicas: unique.into_iter().collect(),
            coefficients,
            max_offspring,
            seed,
            open_epoch: 0,
            submissions: BTreeMap::new(),
            completed: BTreeMap::new(),
            abstained: BTreeSet::new(),
            live: BTreeSet::new(),
            live_roster: false,
            quorum: 1.0,
            deadline_ticks: u64::MAX,
            ticks: 0,
            first_submission: None,
        })
    }

    /// Close epochs on `quorum` of the required roster once `deadline_ticks`
    /// coordinator ticks have passed since the epoch's first submission.
    pub fn with_quorum(
        mut self,
        quorum: f64,
        deadline_ticks: u64,
    ) -> Result<Self, ReconfigurationError> {
        if !(quorum.is_finite() && quorum > 0.0 && quorum <= 1.0) {
            return Err(ReconfigurationError::InvalidParameter {
                field: "quorum",
                value: quorum,
            });
        }
        self.quorum = quorum;
        self.deadline_ticks = deadline_ticks;
        Ok(self)
    }

    /// Add a replica to the roster at runtime, live from now on.
    pub fn attach(&mut self, replica: u32) -> Result<(), ReconfigurationError> {
        if self.replicas.contains(&replica) {
            return Err(ReconfigurationError::DuplicateReplica { replica });
        }
        self.replicas.push(replica);
        self.live.insert(replica);
        self.live_roster = true;
        Ok(())
    }

    /// Replicas currently counted live, in configured order.
    pub fn live_replicas(&self) -> Vec<u32> {
        self.replicas
            .iter()
            .copied()
            .filter(|replica| self.live.contains(replica))
            .collect()
    }

    /// Advance the coordinator clock by one tick, closing the open epoch
    /// when its quorum stands past the deadline.
    pub fn tick(&mut self) -> Result<Option<EpochSubmissionOutcome>, ReconfigurationError> {
        self.ticks = self.ticks.saturating_add(1);
        if !self.quorum_deadline_passed() || !self.quorum_met() {
            return Ok(None);
        }
        let epoch = self.open_epoch;
        Ok(Some(self.complete_open_epoch(epoch)?))
    }

    /// Submissions that satisfy the quorum for the open epoch.
    fn quorum_required(&self) -> usize {
        let required = self.required();
        ((self.quorum * required as f64).ceil() as usize).clamp(1, required.max(1))
    }

    fn quorum_met(&self) -> bool {
        !self.submissions.is_empty() && self.submissions.len() >= self.quorum_required()
    }

    fn quorum_deadline_passed(&self) -> bool {
        match self.first_submission {
            Some((epoch, tick)) if epoch == self.open_epoch => {
                self.ticks.saturating_sub(tick) >= self.deadline_ticks
            }
            _ => false,
        }
    }

    /// Epoch currently accepting one submission from every replica.
    pub fn open_epoch(&self) -> u64 {
        self.open_epoch
    }

    /// Replicas the open epoch still waits on.
    ///
    /// Without a declared live roster the count is the configured population
    /// minus abstentions. A declared live roster counts live replicas that
    /// have not abstained, including zero when every live replica has
    /// retired.
    fn required(&self) -> usize {
        if self.live_roster {
            self.live
                .iter()
                .filter(|replica| !self.abstained.contains(replica))
                .count()
        } else {
            self.replicas.len().saturating_sub(self.abstained.len())
        }
    }

    /// Declare that `replica` is an active walker the barrier waits on.
    ///
    /// Unknown identities, including replicas outside the configured roster,
    /// return [`ReconfigurationError::UnknownReplica`]. A replica already in
    /// the live set is accepted again.
    pub fn mark_live(&mut self, replica: u32) -> Result<(), ReconfigurationError> {
        if !self.replicas.contains(&replica) {
            return Err(ReconfigurationError::UnknownReplica { replica });
        }
        self.live.insert(replica);
        self.abstained.remove(&replica);
        Ok(())
    }

    /// Remove `replica` from the live roster so the barrier no longer waits
    /// for it.
    ///
    /// Retirement is an abstention from the open epoch: the replica is not a
    /// required destination and any submission it already made is dropped. A
    /// replica that is already retired is accepted again. Unknown identities
    /// return [`ReconfigurationError::UnknownReplica`].
    ///
    /// If every live replica has retired and nobody remains to submit, the
    /// open epoch closes vacantly so the barrier does not stay open.
    pub fn retire(&mut self, replica: u32) -> Result<(), ReconfigurationError> {
        if !self.replicas.contains(&replica) {
            return Err(ReconfigurationError::UnknownReplica { replica });
        }
        self.live.remove(&replica);
        self.live_roster = true;
        self.abstained.insert(replica);
        self.submissions.remove(&replica);
        let epoch = self.open_epoch;
        self.close_if_ready(epoch)?;
        Ok(())
    }

    /// Replicas whose submission forms the open epoch, in configured order.
    fn participants(&self) -> Vec<u32> {
        self.replicas
            .iter()
            .copied()
            .filter(|replica| self.submissions.contains_key(replica))
            .collect()
    }

    /// Replicas the open epoch still requires, after abstentions and
    /// retirement from the live roster.
    pub fn open_requirement(&self) -> usize {
        self.required()
    }

    /// Whether every replica that can still submit has done so.
    fn epoch_is_complete(&self) -> bool {
        !self.submissions.is_empty()
            && (self.submissions.len() >= self.required()
                || (self.quorum_deadline_passed() && self.quorum_met()))
    }

    /// Close the open epoch when the barrier is met or nobody remains.
    fn close_if_ready(
        &mut self,
        epoch: u64,
    ) -> Result<EpochSubmissionOutcome, ReconfigurationError> {
        if self.submissions.is_empty() && self.required() == 0 {
            // Every replica the barrier still counted declined, so there is
            // no population to select from, yet the epoch must still close:
            // leaving it open wedges every replica's epoch counter on a
            // barrier nobody can meet. A vacant close is reported as zero
            // submitted of zero required, which no genuinely pending epoch
            // can produce, since a pending epoch always has at least one
            // replica still expected.
            self.abstained.clear();
            self.open_epoch =
                self.open_epoch
                    .checked_add(1)
                    .ok_or(ReconfigurationError::InvalidParameter {
                        field: "epoch_overflow",
                        value: self.open_epoch as f64,
                    })?;
            return Ok(EpochSubmissionOutcome::Pending {
                epoch,
                submitted: 0,
                required: 0,
            });
        }
        if !self.epoch_is_complete() {
            return Ok(EpochSubmissionOutcome::Pending {
                epoch,
                submitted: self.submissions.len(),
                required: self.required(),
            });
        }
        self.complete_open_epoch(epoch)
    }

    /// Record that a replica will not submit to the open epoch.
    ///
    /// A replica reaches this when the barrier arrives and its own state
    /// yields no validated representative. Announcing it releases the
    /// replicas already waiting, which otherwise poll until their budgets
    /// drain, because the barrier requires everyone and a replica that
    /// cannot submit never arrives.
    pub fn abstain(
        &mut self,
        epoch: u64,
        replica: u32,
    ) -> Result<EpochSubmissionOutcome, ReconfigurationError> {
        if !self.replicas.contains(&replica) {
            return Err(ReconfigurationError::UnknownReplica { replica });
        }
        if epoch < self.open_epoch {
            return match self.completed.get(&epoch) {
                Some(completed) => Ok(EpochSubmissionOutcome::Ready(completed.plan.clone())),
                // Closed with no plan: every replica abstained from it.
                None => Ok(EpochSubmissionOutcome::Pending {
                    epoch,
                    submitted: 0,
                    required: 0,
                }),
            };
        }
        if epoch > self.open_epoch {
            return Err(ReconfigurationError::EpochMismatch {
                expected: self.open_epoch,
                received: epoch,
            });
        }
        self.abstained.insert(replica);
        self.submissions.remove(&replica);
        self.close_if_ready(epoch)
    }

    /// Submit one replica's member to the open epoch.
    ///
    /// Completes the epoch when every replica the barrier still requires has
    /// submitted; a repeat of an identical submission is answered from the
    /// completed record.
    pub fn submit(
        &mut self,
        epoch: u64,
        member: PopulationMember,
    ) -> Result<EpochSubmissionOutcome, ReconfigurationError> {
        if !self.replicas.contains(&member.replica) {
            return Err(ReconfigurationError::UnknownReplica {
                replica: member.replica,
            });
        }
        if epoch < self.open_epoch {
            let Some(completed) = self.completed.get(&epoch) else {
                return Err(ReconfigurationError::EpochMismatch {
                    expected: self.open_epoch,
                    received: epoch,
                });
            };
            return if completed.submissions.get(&member.replica) == Some(&member) {
                Ok(EpochSubmissionOutcome::Ready(completed.plan.clone()))
            } else {
                Err(ReconfigurationError::ConflictingSubmission {
                    epoch,
                    replica: member.replica,
                })
            };
        }
        if epoch > self.open_epoch {
            return Err(ReconfigurationError::EpochMismatch {
                expected: self.open_epoch,
                received: epoch,
            });
        }
        if let Some(stored) = self.submissions.get(&member.replica) {
            return if stored == &member {
                Ok(EpochSubmissionOutcome::Pending {
                    epoch,
                    submitted: self.submissions.len(),
                    required: self.required(),
                })
            } else {
                Err(ReconfigurationError::ConflictingSubmission {
                    epoch,
                    replica: member.replica,
                })
            };
        }
        if self.live.len() >= 2 {
            self.live_roster = true;
        }
        if self.submissions.is_empty() {
            self.first_submission = Some((self.open_epoch, self.ticks));
        }
        self.submissions.insert(member.replica, member);
        self.close_if_ready(epoch)
    }

    fn complete_open_epoch(
        &mut self,
        epoch: u64,
    ) -> Result<EpochSubmissionOutcome, ReconfigurationError> {
        let participants = self.participants();
        let members = participants
            .iter()
            .map(|replica| {
                *self
                    .submissions
                    .get(replica)
                    .expect("a participant has submitted by construction")
            })
            .collect::<Vec<_>>();
        let ranked = rank_population(&members)?;
        let evidence = ranked
            .iter()
            .map(|member| member.evidence())
            .collect::<Vec<_>>();
        let plan = reconfiguration_plan(
            &evidence,
            self.coefficients,
            epoch_systematic_offset(self.seed, epoch),
            self.max_offspring,
        )?;
        let epoch_plan = PopulationEpochPlan {
            epoch,
            destinations: participants.clone(),
            parents: plan
                .parents()
                .iter()
                .map(|index| participants[*index])
                .collect(),
            weights: plan.weights().to_vec(),
            diagnostics: plan.diagnostics(),
        };
        let submissions = std::mem::take(&mut self.submissions);
        self.abstained.clear();
        self.completed.insert(
            epoch,
            CompletedEpoch {
                submissions,
                plan: epoch_plan.clone(),
            },
        );
        self.open_epoch =
            self.open_epoch
                .checked_add(1)
                .ok_or(ReconfigurationError::InvalidParameter {
                    field: "epoch_overflow",
                    value: self.open_epoch as f64,
                })?;
        Ok(EpochSubmissionOutcome::Ready(epoch_plan))
    }
}

fn epoch_systematic_offset(seed: u64, epoch: u64) -> f64 {
    let value = splitmix64(seed ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    (value >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl ReconfigurationPlan {
    /// Normalized selection weights in source-chain order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Parent source index for every destination chain.
    pub fn parents(&self) -> &[usize] {
        &self.parents
    }

    /// Selection and realized-genealogy diagnostics.
    pub fn diagnostics(&self) -> GenealogyDiagnostics {
        self.diagnostics
    }
}

/// Fractional ascending ranks with average ranks for ties.
///
/// The minimum has rank zero and the maximum has rank one when at least two
/// distinct positions are present. Positive affine changes of units preserve
/// the result.
pub fn ascending_fractional_ranks(values: &[f64]) -> Result<Vec<f64>, ReconfigurationError> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    for &value in values {
        if !value.is_finite() {
            return Err(ReconfigurationError::InvalidParameter {
                field: "rank_input",
                value,
            });
        }
    }
    if values.len() == 1 {
        return Ok(vec![0.0]);
    }

    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let scale = (values.len() - 1) as f64;
    let mut ranks = vec![0.0; values.len()];
    let mut begin = 0;
    while begin < order.len() {
        let mut end = begin + 1;
        while end < order.len() && values[order[end]] == values[order[begin]] {
            end += 1;
        }
        let average_position = 0.5 * (begin + end - 1) as f64;
        let rank = average_position / scale;
        for &index in &order[begin..end] {
            ranks[index] = rank;
        }
        begin = end;
    }
    Ok(ranks)
}

/// Build a fixed-size, family-capped reconfiguration plan.
///
/// `systematic_offset` lies in `[0, 1)` and is stored by the caller as part of
/// the coordinator event. Replaying the same snapshot and offset produces the
/// same parent assignment.
pub fn reconfiguration_plan(
    evidence: &[BasinEvidence],
    coefficients: SelectionCoefficients,
    systematic_offset: f64,
    max_offspring: usize,
) -> Result<ReconfigurationPlan, ReconfigurationError> {
    if evidence.is_empty() {
        return Err(ReconfigurationError::EmptyPopulation);
    }
    coefficients.validate()?;
    if !systematic_offset.is_finite() || !(0.0..1.0).contains(&systematic_offset) {
        return Err(ReconfigurationError::InvalidParameter {
            field: "systematic_offset",
            value: systematic_offset,
        });
    }
    if max_offspring == 0 {
        return Err(ReconfigurationError::InvalidParameter {
            field: "max_offspring",
            value: 0.0,
        });
    }

    let mut log_weights: Vec<f64> = evidence
        .iter()
        .map(|item| {
            -coefficients.energy * item.energy_rank
                + coefficients.novelty * item.novelty_rank
                + coefficients.scarcity * item.scarcity_rank
        })
        .collect();
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    for value in &mut log_weights {
        *value = (*value - maximum).max(-coefficients.log_weight_clip);
    }
    let normalizer: f64 = log_weights.iter().map(|value| value.exp()).sum();
    let weights: Vec<f64> = log_weights
        .iter()
        .map(|value| value.exp() / normalizer)
        .collect();

    let mut parents = systematic_parents(&weights, systematic_offset);
    cap_families(&mut parents, &weights, max_offspring);
    let diagnostics = genealogy_diagnostics(&weights, &parents);
    Ok(ReconfigurationPlan {
        weights,
        parents,
        diagnostics,
    })
}

fn systematic_parents(weights: &[f64], offset: f64) -> Vec<usize> {
    let population = weights.len();
    let mut parents = Vec::with_capacity(population);
    let mut parent = 0;
    let mut cumulative = weights[0];
    for destination in 0..population {
        let threshold = (offset + destination as f64) / population as f64;
        while parent + 1 < population && threshold >= cumulative {
            parent += 1;
            cumulative += weights[parent];
        }
        parents.push(parent);
    }
    parents
}

fn cap_families(parents: &mut [usize], weights: &[f64], max_offspring: usize) {
    let mut counts = vec![0usize; weights.len()];
    for &parent in parents.iter() {
        counts[parent] += 1;
    }
    while let Some(donor) = counts.iter().position(|&count| count > max_offspring) {
        let receiver = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count < max_offspring)
            .max_by(|(left, left_count), (right, right_count)| {
                let left_deficit = weights[*left] * parents.len() as f64 - **left_count as f64;
                let right_deficit = weights[*right] * parents.len() as f64 - **right_count as f64;
                left_deficit
                    .total_cmp(&right_deficit)
                    .then_with(|| right.cmp(left))
            })
            .map(|(index, _)| index)
            .expect("a positive family cap has enough total capacity");
        let slot = parents
            .iter()
            .rposition(|&parent| parent == donor)
            .expect("the donor family has an offspring slot");
        parents[slot] = receiver;
        counts[donor] -= 1;
        counts[receiver] += 1;
    }
}

fn genealogy_diagnostics(weights: &[f64], parents: &[usize]) -> GenealogyDiagnostics {
    let effective_sample_size = 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>();
    let mut counts = vec![0usize; weights.len()];
    for &parent in parents {
        counts[parent] += 1;
    }
    let unique_parents = counts.iter().filter(|&&count| count > 0).count();
    let max_family_size = counts.iter().copied().max().unwrap_or(0);
    let mean = parents.len() as f64 / counts.len() as f64;
    let offspring_variance = counts
        .iter()
        .map(|&count| {
            let residual = count as f64 - mean;
            residual * residual
        })
        .sum::<f64>()
        / counts.len() as f64;
    GenealogyDiagnostics {
        effective_sample_size,
        unique_parents,
        max_family_size,
        offspring_variance,
    }
}

/// Destination replica with packing-family identity and energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackingOccupant {
    /// Replica identity.
    pub replica: u32,
    /// Packing-family index, or none before DECAF assigns one.
    pub family: Option<usize>,
    /// Occupant energy; lower is deeper.
    pub energy: f64,
}

/// Assign parents from the shared Keep/Reseed occupancy rule.
///
/// [`crate::catalog::keep_ids`] decides who may stay on a packing.
/// The family champion is its own parent. Kept extras adopt the
/// champion (better isomer). Everyone else is an extra beyond the
/// cap: they stay self-parented so the client reseeds instead of
/// cloning the well. A destination never receives a parent from a
/// different packing.
pub fn assign_parents_by_packing(occupants: &[PackingOccupant], max_offspring: usize) -> Vec<u32> {
    let _ = max_offspring;
    if occupants.is_empty() {
        return Vec::new();
    }
    let walks: Vec<crate::catalog::WalkRecord> = occupants
        .iter()
        .map(|occupant| crate::catalog::WalkRecord {
            id: occupant.replica,
            resource: crate::catalog::DEFAULT_MAX_RESOURCE,
            energy: occupant.energy,
            family: occupant.family,
        })
        .collect();
    let keep = crate::catalog::keep_ids(&walks, crate::catalog::DEFAULT_MAX_RESOURCE);
    let mut champion = BTreeMap::<usize, u32>::new();
    for occupant in occupants {
        let Some(family) = occupant.family else {
            continue;
        };
        match champion.get(&family) {
            None => {
                champion.insert(family, occupant.replica);
            }
            Some(&current) => {
                let current_energy = occupants
                    .iter()
                    .find(|other| other.replica == current)
                    .map(|other| other.energy)
                    .unwrap_or(f64::INFINITY);
                if occupant.energy < current_energy - 1e-12 {
                    champion.insert(family, occupant.replica);
                }
            }
        }
    }
    occupants
        .iter()
        .map(|occupant| {
            if !keep.contains(&occupant.replica) {
                return occupant.replica;
            }
            occupant
                .family
                .and_then(|family| champion.get(&family).copied())
                .filter(|donor| keep.contains(donor))
                .unwrap_or(occupant.replica)
        })
        .collect()
}
