//! Good-UCB division of same-PES stationary-object discovery work.
//!
//! A roster receives only evidence owned by one system-signature coordinator.
//! Exact minima and exact index-one saddles are distinct species, so basin
//! escape and saddle riding form the non-intersecting expert supports required
//! by the Good-UCB analysis. Stable roster rotation changes which replica owns
//! each role without introducing a separate random stream into matched
//! campaigns.

use std::collections::BTreeSet;

const GOOD_UCB_CONFIDENCE_FACTOR: f64 = 2.414_213_562_373_095;

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
    /// Exact basin observations made by this system coordinator.
    pub basin_observations: u64,
    /// Exact basin identities occurring once in the basin sample.
    pub basin_singletons: u64,
    /// Exact saddle observations made by this system coordinator.
    pub saddle_observations: u64,
    /// Exact saddle identities occurring once in the saddle sample.
    pub saddle_singletons: u64,
    /// Whether the coordinator has a source/mode arm that is not claimed.
    pub ride_available: bool,
}

/// Good-UCB index for one exact-species discovery mechanism.
///
/// The Good--Turing term `singletons / observations` estimates the probability
/// that the next draw exposes an unseen interesting species. The second term
/// is the distribution-free upper-confidence correction from Proposition 1
/// and Algorithm 1 of Bubeck, Ernst, and Garivier (JMLR 14, 2013), evaluated
/// with `delta = 1 / total_observations`.
pub fn good_ucb_missing_mass_index(
    singletons: u64,
    observations: u64,
    total_observations: u64,
) -> f64 {
    if observations == 0 {
        return f64::INFINITY;
    }
    if singletons > observations || total_observations < observations {
        return f64::NAN;
    }
    let observations = observations as f64;
    let total = total_observations.max(1) as f64;
    singletons as f64 / observations
        + GOOD_UCB_CONFIDENCE_FACTOR * ((4.0 * total).ln() / observations).sqrt()
}

fn batched_good_ucb_index(
    singletons: u64,
    observations: u64,
    provisional_observations: u64,
    total_observations: u64,
) -> f64 {
    if observations == 0 && provisional_observations == 0 {
        return f64::INFINITY;
    }
    let effective_observations = observations.saturating_add(provisional_observations);
    let estimate = if observations == 0 {
        0.0
    } else {
        singletons as f64 / observations as f64
    };
    let total = total_observations.max(1) as f64;
    estimate
        + GOOD_UCB_CONFIDENCE_FACTOR * ((4.0 * total).ln() / effective_observations as f64).sqrt()
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
    /// Singleton counts cannot exceed their exact observation counts.
    #[error("discovery coverage contains impossible occupancy counts")]
    InvalidCoverage,
}

/// Conservative missing-mass weight used before role allocation.
///
/// An asymptotic estimator cannot retire a discovery mechanism before its
/// declared observation floor. Missing or invalid upper bounds also retain
/// full uncertainty.
pub fn coverage_allocation_weight(
    observations: u64,
    minimum_observations: u64,
    unseen_mass_upper: Option<f64>,
) -> f64 {
    if observations < minimum_observations {
        return 1.0;
    }
    unseen_mass_upper
        .filter(|upper| upper.is_finite() && *upper >= 0.0)
        .map_or(1.0, |upper| upper.clamp(0.0, 1.0))
}

/// Assign one isolated ensemble between basin escapes and saddle rides.
///
/// Each parallel seat is one delayed Good-UCB decision. Provisional pulls
/// reduce only the confidence term, preventing a batch from cloning one stale
/// decision while leaving the Good--Turing estimate unchanged until results
/// arrive. No minimum seat, proportional split, or random exploration floor is
/// imposed.
pub fn assign_discovery_roles(
    members: &[u32],
    coverage: DiscoveryCoverage,
    epoch: u64,
) -> Result<Vec<DiscoveryAssignment>, DiscoveryRosterError> {
    if members.is_empty() {
        return Err(DiscoveryRosterError::EmptyRoster);
    }
    if coverage.basin_singletons > coverage.basin_observations
        || coverage.saddle_singletons > coverage.saddle_observations
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
    let base_total = coverage
        .basin_observations
        .saturating_add(coverage.saddle_observations);
    let mut provisional = [0_u64; 2];
    let mut basin_seats = 0_usize;
    for seat in 0..count {
        if !coverage.ride_available {
            basin_seats += 1;
            provisional[0] = provisional[0].saturating_add(1);
            continue;
        }
        let total = base_total
            .saturating_add(provisional[0])
            .saturating_add(provisional[1]);
        let basin_index = batched_good_ucb_index(
            coverage.basin_singletons,
            coverage.basin_observations,
            provisional[0],
            total,
        );
        let saddle_index = batched_good_ucb_index(
            coverage.saddle_singletons,
            coverage.saddle_observations,
            provisional[1],
            total,
        );
        let choose_basin = match basin_index.total_cmp(&saddle_index) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => epoch.saturating_add(seat as u64).is_multiple_of(2),
        };
        let arm = usize::from(!choose_basin);
        provisional[arm] = provisional[arm].saturating_add(1);
        basin_seats += usize::from(choose_basin);
    }
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
