//! Batch assignment of same-PES global-minimum search operators.
//!
//! Every replica contributes one basin-escape opportunity and may contribute a
//! ridge opportunity. The coordinator maximizes summed joint-minimum
//! information per charged PES evaluation under the exclusive ride-arm
//! capacity. An unmeasured consumer mechanism can impose a minimum ride count
//! for identifiability. This is a two-choice cardinality-constrained
//! assignment: start with every basin action, sort the ride-minus-basin gains,
//! and take the largest gains within the cardinality bounds. Stable replica
//! ordering resolves interchangeable maximizers.

use std::collections::{BTreeMap, BTreeSet};

use crate::minimum_information::{
    MinimumInformationError, MinimumInformationSearch, SearchActionCandidate, SearchMechanism,
};

/// One same-system global-minimum search role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryRole {
    /// Perturb and quench to sample another terminal minimum.
    BasinEscape,
    /// Follow a minimum mode and quench its downhill branch or branches.
    SaddleRide,
}

/// Information-rate alternatives available to one replica.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscoveryOpportunity {
    /// Replica identifier from the isolated ensemble roster.
    pub replica: u32,
    /// Joint-minimum information per charged evaluation for basin escape.
    pub basin_information_rate: f64,
    /// Best unclaimed ride-arm information rate, if a ride is feasible.
    pub ride_information_rate: Option<f64>,
}

impl DiscoveryOpportunity {
    /// Validate one pair of same-PES action values.
    pub fn new(
        replica: u32,
        basin_information_rate: f64,
        ride_information_rate: Option<f64>,
    ) -> Result<Self, DiscoveryRosterError> {
        if !valid_rate(basin_information_rate)
            || ride_information_rate.is_some_and(|rate| !valid_rate(rate))
        {
            return Err(DiscoveryRosterError::InvalidInformationRate);
        }
        Ok(Self {
            replica,
            basin_information_rate,
            ride_information_rate,
        })
    }
}

/// One replica's deterministic role inside an evidence epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveryAssignment {
    /// Replica identifier from the isolated ensemble roster.
    pub replica: u32,
    /// Search operator assigned to this replica.
    pub role: DiscoveryRole,
    /// Minimum-information model version behind the assignment.
    pub epoch: u64,
}

/// One live-chain seat in a redundancy-aware joint-information batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveryBatchAssignment {
    /// Replica receiving exactly one action family.
    pub replica: u32,
    /// Search operator assigned to the replica.
    pub role: DiscoveryRole,
    /// Index in the supplied shared ride-action slice, when assigned.
    pub ride_action: Option<usize>,
    /// Minimum-information evidence version behind the assignment.
    pub epoch: u64,
}

/// Invalid roster or action values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiscoveryRosterError {
    /// A discovery roster requires at least one replica.
    #[error("discovery roster is empty")]
    EmptyRoster,
    /// Each replica may occupy only one roster seat.
    #[error("replica {0} occurs more than once in the discovery roster")]
    DuplicateReplica(u32),
    /// Information per charged evaluation must be finite and nonnegative.
    #[error("discovery opportunity contains an invalid information rate")]
    InvalidInformationRate,
}

/// Invalid candidates or incomplete batch allocation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DiscoveryBatchError {
    /// At least one live chain must offer a basin action.
    #[error("discovery batch has no live-chain basin actions")]
    EmptyRoster,
    /// Every live replica owns exactly one basin alternative.
    #[error("replica {0} occurs more than once in the discovery batch")]
    DuplicateReplica(u32),
    /// Basin and ridge slices must carry their declared mechanisms.
    #[error("discovery batch contains an action in the wrong mechanism slice")]
    WrongMechanism,
    /// The conditional joint-information model rejected an action.
    #[error(transparent)]
    MinimumInformation(#[from] MinimumInformationError),
    /// Feasible candidates did not fill every live-chain seat.
    #[error("joint-information assignment did not fill the live-chain batch")]
    IncompleteBatch,
}

/// Allocate a simultaneous same-PES action batch across live chains.
///
/// Each replica contributes its own basin action while ridge actions form a
/// shared exclusive pool. Unique family capacities prevent duplicate actions;
/// [`MinimumInformationSearch::assign_batch`] discounts correlated outcomes by
/// the predictive log determinant and prices every marginal bit by PES calls.
/// A selected basin stays with its owner. Selected rides occupy the replicas
/// whose redundant basin actions were omitted, in stable replica order.
pub fn assign_discovery_batch(
    search: &mut MinimumInformationSearch,
    basin_actions: &[(u32, SearchActionCandidate)],
    ride_actions: &[SearchActionCandidate],
    minimum_samples: usize,
) -> Result<Vec<DiscoveryBatchAssignment>, DiscoveryBatchError> {
    if basin_actions.is_empty() {
        return Err(DiscoveryBatchError::EmptyRoster);
    }
    let mut basins = basin_actions.to_vec();
    basins.sort_by_key(|(replica, _)| *replica);
    let mut replicas = BTreeSet::new();
    for (replica, candidate) in &basins {
        if !replicas.insert(*replica) {
            return Err(DiscoveryBatchError::DuplicateReplica(*replica));
        }
        if candidate.mechanism != SearchMechanism::BasinEscape {
            return Err(DiscoveryBatchError::WrongMechanism);
        }
    }
    if ride_actions
        .iter()
        .any(|candidate| candidate.mechanism != SearchMechanism::SaddleRide)
    {
        return Err(DiscoveryBatchError::WrongMechanism);
    }

    let mut candidates = basins
        .iter()
        .map(|(_, candidate)| candidate.clone())
        .collect::<Vec<_>>();
    candidates.extend_from_slice(ride_actions);
    let families = (0..candidates.len()).collect::<Vec<_>>();
    let selected = search.assign_batch(&candidates, &families, basins.len(), 1, minimum_samples)?;
    if selected.len() != basins.len() {
        return Err(DiscoveryBatchError::IncompleteBatch);
    }
    let selected = selected.into_iter().collect::<BTreeSet<_>>();
    let selected_rides = selected
        .iter()
        .filter_map(|index| index.checked_sub(basins.len()))
        .collect::<Vec<_>>();
    let unselected_replicas = basins
        .iter()
        .enumerate()
        .filter_map(|(index, (replica, _))| (!selected.contains(&index)).then_some(*replica))
        .collect::<Vec<_>>();
    if selected_rides.len() != unselected_replicas.len() {
        return Err(DiscoveryBatchError::IncompleteBatch);
    }
    let rides_by_replica = unselected_replicas
        .into_iter()
        .zip(selected_rides)
        .collect::<BTreeMap<_, _>>();
    let epoch = search.version();
    Ok(basins
        .into_iter()
        .map(|(replica, _)| DiscoveryBatchAssignment {
            replica,
            role: if rides_by_replica.contains_key(&replica) {
                DiscoveryRole::SaddleRide
            } else {
                DiscoveryRole::BasinEscape
            },
            ride_action: rides_by_replica.get(&replica).copied(),
            epoch,
        })
        .collect())
}

/// Maximize total information rate under an exclusive ride-arm capacity.
pub fn assign_discovery_roles(
    opportunities: &[DiscoveryOpportunity],
    ride_capacity: usize,
    epoch: u64,
) -> Result<Vec<DiscoveryAssignment>, DiscoveryRosterError> {
    assign_discovery_roles_with_minimum(opportunities, ride_capacity, 0, epoch)
}

/// Maximize information subject to a minimum available ride-consumer count.
///
/// `minimum_rides` is clipped by both the exclusive arm capacity and the number
/// of replicas that actually have a ride alternative. This supports a bounded
/// initialization barrier without changing steady-state information-per-cost
/// allocation.
pub fn assign_discovery_roles_with_minimum(
    opportunities: &[DiscoveryOpportunity],
    ride_capacity: usize,
    minimum_rides: usize,
    epoch: u64,
) -> Result<Vec<DiscoveryAssignment>, DiscoveryRosterError> {
    if opportunities.is_empty() {
        return Err(DiscoveryRosterError::EmptyRoster);
    }
    let mut replicas = BTreeSet::new();
    for opportunity in opportunities {
        if !replicas.insert(opportunity.replica) {
            return Err(DiscoveryRosterError::DuplicateReplica(opportunity.replica));
        }
        if !valid_rate(opportunity.basin_information_rate)
            || opportunity
                .ride_information_rate
                .is_some_and(|rate| !valid_rate(rate))
        {
            return Err(DiscoveryRosterError::InvalidInformationRate);
        }
    }

    let mut ride_gains = opportunities
        .iter()
        .filter_map(|opportunity| {
            Some((
                opportunity.ride_information_rate? - opportunity.basin_information_rate,
                opportunity.replica,
            ))
        })
        .collect::<Vec<_>>();
    ride_gains.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    let positive = ride_gains
        .iter()
        .take_while(|(gain, _)| *gain > 0.0)
        .count();
    let zero = ride_gains
        .iter()
        .skip(positive)
        .take_while(|(gain, _)| *gain == 0.0)
        .count();
    let positive_selected = positive.min(ride_capacity);
    let tied_rides = ride_capacity
        .saturating_sub(positive_selected)
        .min(zero.div_ceil(2));
    let selected_count = positive_selected
        .saturating_add(tied_rides)
        .max(minimum_rides.min(ride_capacity).min(ride_gains.len()));
    let selected = ride_gains
        .iter()
        .take(selected_count)
        .map(|(_, replica)| *replica)
        .collect::<BTreeSet<_>>();

    let mut assignments = opportunities
        .iter()
        .map(|opportunity| DiscoveryAssignment {
            replica: opportunity.replica,
            role: if selected.contains(&opportunity.replica) {
                DiscoveryRole::SaddleRide
            } else {
                DiscoveryRole::BasinEscape
            },
            epoch,
        })
        .collect::<Vec<_>>();
    assignments.sort_by_key(|assignment| assignment.replica);
    Ok(assignments)
}

fn valid_rate(rate: f64) -> bool {
    rate.is_finite() && rate >= 0.0
}
