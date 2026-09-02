//! Coverage-proportional division of same-PES discovery work.
//!
//! A roster receives only evidence owned by one system-signature coordinator.
//! It allocates live replicas between minimum discovery and index-one saddle
//! discovery, retaining at least one seat for each available mechanism. Stable
//! roster rotation changes which replica owns each role without introducing a
//! separate random stream into matched campaigns.

use std::collections::BTreeSet;

const MINIMUM_ROLE_WEIGHT: f64 = 0.02;
const GOLDEN_EPOCH_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

/// One same-system stationary-object discovery role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryRole {
    /// Perturb and quench to expose an exact minimum.
    BasinEscape,
    /// Follow a minimum mode to expose an exact index-one saddle.
    SaddleRide,
}

/// Global coverage evidence used to divide one system's replicas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscoveryCoverage {
    /// One-sided upper bound on unseen exact-basin probability.
    pub basin_unseen_mass_upper: f64,
    /// One-sided upper bound on unseen exact-saddle probability.
    pub saddle_unseen_mass_upper: f64,
    /// Whether the coordinator has a source/mode arm that is not claimed.
    pub ride_available: bool,
}

/// One replica's deterministic role inside a coverage epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveryAssignment {
    /// Replica identifier from the isolated ensemble roster.
    pub replica: u32,
    /// Stationary-object mechanism assigned to this replica.
    pub role: DiscoveryRole,
    /// Coordinator evidence epoch from which the assignment was derived.
    pub epoch: u64,
}

/// Invalid roster or coverage evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiscoveryRosterError {
    /// A discovery roster requires at least one replica.
    #[error("discovery roster is empty")]
    EmptyRoster,
    /// Each replica may occupy only one roster seat.
    #[error("replica {0} occurs more than once in the discovery roster")]
    DuplicateReplica(u32),
    /// Missing-mass upper bounds must be finite and nonnegative.
    #[error("discovery coverage contains an invalid missing-mass bound")]
    InvalidCoverage,
}

/// Assign one isolated ensemble between basin escapes and saddle rides.
///
/// Esty upper bounds act as unresolved-coverage weights. With two or more
/// replicas, both available mechanisms retain a seat and the remaining seats
/// are proportional to their bounds. A single replica follows the same ratio
/// through a deterministic low-discrepancy epoch sequence.
pub fn assign_discovery_roles(
    members: &[u32],
    coverage: DiscoveryCoverage,
    epoch: u64,
) -> Result<Vec<DiscoveryAssignment>, DiscoveryRosterError> {
    if members.is_empty() {
        return Err(DiscoveryRosterError::EmptyRoster);
    }
    if !coverage.basin_unseen_mass_upper.is_finite()
        || coverage.basin_unseen_mass_upper < 0.0
        || !coverage.saddle_unseen_mass_upper.is_finite()
        || coverage.saddle_unseen_mass_upper < 0.0
    {
        return Err(DiscoveryRosterError::InvalidCoverage);
    }

    let mut unique = BTreeSet::new();
    for &replica in members {
        if !unique.insert(replica) {
            return Err(DiscoveryRosterError::DuplicateReplica(replica));
        }
    }
    let mut members = unique.into_iter().collect::<Vec<_>>();
    let count = members.len();
    let basin_weight = coverage
        .basin_unseen_mass_upper
        .clamp(MINIMUM_ROLE_WEIGHT, 1.0);
    let saddle_weight = coverage
        .saddle_unseen_mass_upper
        .clamp(MINIMUM_ROLE_WEIGHT, 1.0);

    let basin_seats = if !coverage.ride_available {
        count
    } else if count == 1 {
        let basin_fraction = basin_weight / (basin_weight + saddle_weight);
        usize::from(epoch_phase(epoch) < basin_fraction)
    } else {
        (((count as f64) * basin_weight / (basin_weight + saddle_weight)).round() as usize)
            .clamp(1, count - 1)
    };
    let rotation = usize::try_from(epoch % count as u64)
        .expect("discovery-roster rotation is bounded by membership");
    members.rotate_left(rotation);

    Ok(members
        .into_iter()
        .enumerate()
        .map(|(seat, replica)| DiscoveryAssignment {
            replica,
            role: if seat < basin_seats {
                DiscoveryRole::BasinEscape
            } else {
                DiscoveryRole::SaddleRide
            },
            epoch,
        })
        .collect())
}

fn epoch_phase(epoch: u64) -> f64 {
    let scrambled = epoch.wrapping_mul(GOLDEN_EPOCH_MULTIPLIER);
    (scrambled as f64) / ((u64::MAX as f64) + 1.0)
}
