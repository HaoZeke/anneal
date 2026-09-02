//! Batch assignment of same-PES global-minimum search operators.
//!
//! Every replica contributes one basin-escape opportunity and may contribute a
//! ridge opportunity. The coordinator maximizes summed minimum-value
//! information per charged PES evaluation under the exact number of unclaimed
//! ridge arms. This is a two-choice cardinality-constrained assignment: start
//! with every basin action, sort the ride-minus-basin gains, and take the
//! positive gains up to capacity. Exact ties are interchangeable maximizers and
//! are split deterministically so neither unobserved operator is hidden.

use std::collections::BTreeSet;

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
    /// GIBBON information per charged evaluation for basin escape.
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

/// Maximize total information rate under an exclusive ride-arm capacity.
pub fn assign_discovery_roles(
    opportunities: &[DiscoveryOpportunity],
    ride_capacity: usize,
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
    let selected = ride_gains
        .iter()
        .take(positive_selected)
        .chain(ride_gains.iter().skip(positive).take(tied_rides))
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
